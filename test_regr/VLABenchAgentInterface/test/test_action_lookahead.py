import itertools
import json

import pytest
import torch

from domiknows.generation.applications.action_lookahead import ActionAutomaton, ActionLookahead, TokenOperators

from ..native_lookahead import DFAActionLookahead, PLAN_CLOSE, PLAN_OPEN, canonical_body, operation_texts, plan_tokens, tokenization_gaps
from ..native_planning import NativeConstraintProcessor, NativePlanDFA, validate_native_plan
from ..object_grounding import ObjectConstraints

VOCABULARY = 134
EOS = 133


def random_hmm(states, vocabulary, seed):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.softmax(torch.randn(states, generator=generator, dtype=torch.double), -1)
    transition = torch.softmax(torch.randn(states, states, generator=generator, dtype=torch.double), -1)
    emission = torch.softmax(torch.randn(states, vocabulary, generator=generator, dtype=torch.double) * 2, -1)
    return initial, transition, emission


def sequence_mass(belief, transition, emission, tokens):
    """Unnormalised P(tokens) from a belief over the state of the next emission."""
    for token in tokens:
        belief = (belief * emission[:, token]) @ transition
    return float(belief.sum())


def test_action_masses_equal_exhaustive_enumeration_of_plans():
    initial, transition, emission = random_hmm(4, 10, seed=3)
    # Three actions in two groups; the plan must be group 0 then group 1, then optionally repeat (0, 1).
    groups = [0, 0, 1]
    cont = [[0, 1], [0, 2], [3, 4, 1]]
    stop = [[5, 1], [5, 2], [5, 4, 4]]
    opening, closing = [7], [6, 8]
    automaton = ActionAutomaton(0, (False, False, True), ({0: 1}, {1: 2}, {0: 1}))
    limit = 5
    lookahead = ActionLookahead(TokenOperators(transition, emission), groups=groups, cont=cont, stop=stop, open=opening,
                                close=closing, automaton=automaton, max_actions=limit)
    belief = torch.tensor([.1, .2, .3, .4], dtype=torch.double)

    plans = []  # every complete plan within the budget
    def extend(actions, state):
        if automaton.accepting[state] and actions:
            plans.append(list(actions))
        if len(actions) == limit:
            return
        for action, group in enumerate(groups):
            if group in automaton.successors[state]:
                extend(actions + [action], automaton.successors[state][group])
    extend([], 0)

    def tokens_of(actions):
        result = list(opening)
        for position, action in enumerate(actions):
            result += stop[action] if position == len(actions) - 1 else cont[action]
        return result + closing

    span = lookahead.span(belief, 0, 0)
    for action in range(3):
        starting = [plan for plan in plans if plan[0] == action]
        expected = sum(sequence_mass(belief, transition, emission, tokens_of(plan)) for plan in starting)
        assert span.action_mass(action) == pytest.approx(expected, rel=1e-9, abs=0)
        expected_stop = sum(sequence_mass(belief, transition, emission, tokens_of(plan)) for plan in starting if len(plan) == 1)
        assert span.action_mass(action, "stop") == pytest.approx(expected_stop, rel=1e-9, abs=0)
        assert span.action_mass(action, "continue") == pytest.approx(expected - expected_stop, rel=1e-9)
    assert span.mass_of(range(3)) == pytest.approx(sum(sequence_mass(belief, transition, emission, tokens_of(p)) for p in plans), rel=1e-9)

    # Ctrl-G style conditional: valid mass over the HMM mass of the same action text followed by anything.
    for action in range(3):
        seen_cont = sequence_mass(belief, transition, emission, opening + cont[action])
        seen_stop = sequence_mass(belief, transition, emission, opening + stop[action])
        starting = [plan for plan in plans if plan[0] == action]
        valid = sum(sequence_mass(belief, transition, emission, tokens_of(plan)) for plan in starting)
        assert span.conditional([action]) == pytest.approx(valid / (seen_cont + seen_stop), rel=1e-9)
        assert span.conditional([action], "continue") == pytest.approx(
            (valid - sum(sequence_mass(belief, transition, emission, tokens_of(p)) for p in starting if len(p) == 1)) / seen_cont, rel=1e-9)
        assert 0 <= span.conditional([action], "stop") <= 1
    assert span.conditional([]) == 0.0

    # After the first action (group 0 -> state 1) the next span sees only the plans that extend it.
    after = belief
    for token in opening + cont[0]:
        after = (after * emission[:, token]) @ transition
    follow = [plan for plan in plans if plan[0] == 0 and len(plan) >= 2]
    second = lookahead.span(after, 1, 1)
    expected = sum(sequence_mass(belief, transition, emission, tokens_of(plan)) for plan in follow)
    assert second.mass_of(range(3)) == pytest.approx(expected, rel=1e-9)
    assert second.action_mass(0) == 0.0  # group 0 has no successor in state 1


def test_budget_is_enforced_by_the_reachability_table():
    initial, transition, emission = random_hmm(3, 9, seed=1)
    automaton = ActionAutomaton(0, (False, True), ({0: 1}, {0: 1}))
    lookahead = ActionLookahead(TokenOperators(transition, emission), groups=[0], cont=[[0]], stop=[[1]], open=[3],
                                close=[2], automaton=automaton, max_actions=2)
    span = lookahead.span(initial, 0, 0)
    one = sequence_mass(initial, transition, emission, [3, 1, 2])
    two = sequence_mass(initial, transition, emission, [3, 0, 1, 2])
    assert span.action_mass(0, "stop") == pytest.approx(one)
    assert span.action_mass(0, "continue") == pytest.approx(two)
    assert lookahead.span(initial, 1, 2).mass_of([0]) == 0.0  # no action may start once the budget is spent


def character_dfa(**kwargs):
    tokens = {byte: bytes([byte]) for byte in [9, 10, 13, *range(32, 127)]}
    kwargs.setdefault("entity_count", 2)
    kwargs.setdefault("max_operations", 6)
    return NativePlanDFA(tokens, EOS, **kwargs)


class CharacterTokenizer:
    eos_token_id = EOS

    def __call__(self, text, add_special_tokens=False):
        return type("Encoded", (), {"input_ids": [ord(character) for character in text]})()


def table_for(dfa, hmm):
    return DFAActionLookahead(dfa, CharacterTokenizer(), TokenOperators(hmm[1], hmm[2]))


def canonical_tokens(plan):
    return [ord(character) for character in json.dumps(plan, indent=4)] + [EOS]


def chain_hmm(tokens, smoothing=1e-4):
    """A deterministic left-to-right HMM that emits ``tokens`` (state i emits token i)."""
    size = len(tokens) + 1
    initial = torch.zeros(size, dtype=torch.double)
    initial[0] = 1
    transition = torch.full((size, size), 1e-9, dtype=torch.double)
    for index in range(size - 1):
        transition[index, index + 1] = 1
    transition[size - 1, size - 1] = 1
    transition = transition / transition.sum(-1, keepdim=True)
    emission = torch.full((size, VOCABULARY), smoothing / VOCABULARY, dtype=torch.double)
    for index, token in enumerate(tokens):
        emission[index, token] += 1 - smoothing
    emission[size - 1] = 1 / VOCABULARY
    return initial, transition, emission / emission.sum(-1, keepdim=True)


def teacher_forced(processor, prefix, base=None):
    scores = torch.zeros(1, VOCABULARY) if base is None else base
    return processor(torch.tensor([[0] + list(prefix)]), scores)[0]


def test_canonical_serialisation_matches_json_dumps_layout():
    plan = [{"name": "pick", "params": {"target_entity_name": 1}}, {"name": "place", "params": {"target_container_name": 0}}]
    carry, _ = operation_texts("pick", ("target_entity_name",), 1)
    _, last = operation_texts("place", ("target_container_name",), 0)
    assert PLAN_OPEN + carry + last + PLAN_CLOSE == json.dumps(plan, indent=4)
    assert canonical_body("pull", (), None).endswith('"params": {}\n    }')
    arguments = {"pick": ("target_entity_name",), "place": ("target_container_name",)}
    assert tokenization_gaps(CharacterTokenizer(), [plan], arguments) == []
    assert plan_tokens(CharacterTokenizer(), plan, arguments) == [ord(c) for c in json.dumps(plan, indent=4)]


def test_segment_tokens_equal_the_whole_plan_tokens_for_the_real_tokenizer():
    """BPE merges characters across operation boundaries; the segments must absorb them."""
    import os
    path = os.environ.get("VLABENCH_TOKENIZER")
    if not path:
        pytest.skip("set VLABENCH_TOKENIZER to a Qwen3-VL tokenizer folder")
    from transformers import AutoTokenizer
    from ..native_planning import BENCHMARK_ARGUMENTS
    from ..planning_metrics import BENCHMARK_PATTERNS
    import random
    tokenizer = AutoTokenizer.from_pretrained(path)
    rng = random.Random(0)
    plans = []
    for _ in range(300):
        plan = []
        for _ in range(rng.randint(1, 4)):
            for skill in rng.choice(BENCHMARK_PATTERNS):
                roles = BENCHMARK_ARGUMENTS[skill]
                plan.append({"name": skill, "params": {role: rng.choice([0, 1, 7, 9, 10, 12, 33, 63]) for role in roles[:1]}})
        plans.append(plan)
    assert tokenization_gaps(tokenizer, plans, BENCHMARK_ARGUMENTS) == []


def test_dfa_exposes_the_action_inventory_the_lookahead_uses():
    dfa = character_dfa()
    assert len(dfa.actions) == len(dfa.action_index) == 2 * 7 + 2  # seven skills with one id role, pull and lift without
    pick_one = dfa.action_index[("pick", 1)]
    spelled = b'{"name":"pick","params":{"target_entity_name":1}}'
    state = dfa.start_state
    for byte in b"[" + spelled[:-1]:
        state = dfa.byte_step(state, byte)
    assert pick_one in dfa.node_actions[state[3]]
    assert dfa.completed_action(state, ord("}")) == (pick_one, b"")
    index, successors, accepting = dfa.semantic_automaton()
    assert accepting[0] and index[((),)] == 0
    assert len(successors) == len(accepting)


def test_decision_groups_never_include_tokens_the_dfa_forbids():
    dfa = character_dfa()
    state = dfa.start_state
    pending = [state]
    seen = 0
    for byte in b'[{"name":"pick","params":{"target_entity_name":0}}':
        state = dfa.byte_step(state, byte)
        pending.append(state)
    for state in pending:
        allowed = dfa.allowed_tokens(state)
        for labels in dfa.effect_groups(state).values():
            assert set(labels) <= allowed
            seen += len(labels)
    assert seen > 0


def test_lookahead_ranks_the_hmm_preferred_object_id_and_stop_decision():
    plan = [{"name": "pick", "params": {"target_entity_name": 1}}, {"name": "place", "params": {"target_container_name": 0}}]
    tokens = canonical_tokens(plan)
    dfa = character_dfa()
    hmm = chain_hmm(tokens)
    table = table_for(dfa, hmm)
    text = json.dumps(plan, indent=4)

    def processor():
        return NativeConstraintProcessor(dfa, 1, 400, hmm=hmm, weight=1.0, action_lookahead=table)

    # Just before the first object id the HMM says 1, not 0.
    prefix = [ord(c) for c in text[:text.index("1")]]
    scored = teacher_forced(processor(), prefix)
    assert scored[ord("1")] > scored[ord("0")] + 3
    assert torch.isfinite(scored[ord("1")]) and torch.isfinite(scored[ord("0")])
    plain = teacher_forced(NativeConstraintProcessor(dfa, 1, 400), prefix)
    assert plain[ord("1")] == plain[ord("0")] == 0  # without the HMM the DFA treats them alike

    # After the place operation the HMM stops, although the DFA also allows another subtask.
    prefix = [ord(c) for c in text[:text.rindex("}") + 1]]
    scored = teacher_forced(processor(), prefix)
    assert scored[ord("]")] > scored[ord(",")] + 3
    assert scored[ord(" ")] == scored[ord("\n")] == 0  # formatting tokens are neutral


def test_lookahead_is_neutral_for_formatting_tokens_and_for_a_zero_weight():
    dfa = character_dfa()
    hmm = random_hmm(6, VOCABULARY, seed=11)
    table = table_for(dfa, hmm)
    prefix = [ord(c) for c in '[\n    {\n        "name": "pi']
    base = torch.randn(1, VOCABULARY, generator=torch.Generator().manual_seed(2))
    plain = teacher_forced(NativeConstraintProcessor(dfa, 1, 400), prefix, base.clone())
    steered = teacher_forced(NativeConstraintProcessor(dfa, 1, 400, hmm=hmm, weight=1.0, action_lookahead=table), prefix, base.clone())
    zero = teacher_forced(NativeConstraintProcessor(dfa, 1, 400, hmm=hmm, weight=0.0, action_lookahead=table), prefix, base.clone())
    assert torch.equal(plain, zero)
    assert torch.equal(torch.isinf(plain), torch.isinf(steered))  # the legal token set is identical
    assert (steered <= plain + 1e-9).all()  # the lookahead only demotes relative to the best group


def decomposes(names, patterns):
    """Whether ``names`` is a concatenation of complete subtask patterns."""
    reachable = {0}
    for position in range(len(names) + 1):
        if position in reachable:
            reachable |= {position + len(pattern) for pattern in patterns if tuple(names[position:position + len(pattern)]) == tuple(pattern)}
    return len(names) in reachable


def decode(dfa, table, hmm, weight, seed, max_tokens=1200, score="joint"):
    generator = torch.Generator().manual_seed(seed)
    processor = NativeConstraintProcessor(dfa, 1, max_tokens, hmm=hmm, weight=weight, action_lookahead=table,
                                          lookahead_score=score)
    ids = [0]
    for _ in range(max_tokens):
        scores = torch.randn(1, VOCABULARY, generator=generator) * 2
        scores[0, [9, 10, 13, 32, 92]] -= 10  # a random model would otherwise spend the budget on whitespace and escapes
        masked = processor(torch.tensor([ids]), scores)
        allowed = dfa.allowed_tokens(processor.state, remaining_steps=max_tokens - len(ids))
        legal = torch.zeros(VOCABULARY, dtype=torch.bool)
        legal[list(allowed)] = True
        assert torch.equal(torch.isfinite(masked[0]), legal), "the lookahead changed the legal token set"
        token = int(masked[0].argmax())
        ids.append(token)
        if token == EOS:
            return ids[1:], processor
    raise AssertionError("generation did not stop")


@pytest.mark.parametrize("weight", [0.5, 5.0, 1000.0])
@pytest.mark.parametrize("kwargs", [
    {},
    {"max_operations": 4},
    {"patterns": (("pick", "place"), ("press",))},
    {"patterns": (("press",),), "max_operations": 1},
    {"object_constraints": ObjectConstraints(restrictions=(("pick", "target_entity_name", (1,)),))},
])
def test_plans_decoded_with_the_lookahead_satisfy_every_grammar_constraint(kwargs, weight):
    dfa = character_dfa(**kwargs)
    hmm = random_hmm(8, VOCABULARY, seed=5)
    table = table_for(dfa, hmm)
    for seed in range(4):
        labels, processor = decode(dfa, table, hmm, weight, seed)
        assert dfa.accepts(labels)
        text = bytes(label for label in labels if label != EOS).decode()
        plan = json.loads(text)
        ok, errors = validate_native_plan(plan, ["a", "b"], max_operations=dfa.max_operations)
        assert ok, errors
        assert len(plan) <= dfa.max_operations
        assert decomposes([operation["name"] for operation in plan], dfa.patterns)
        for operation in plan:
            if operation["name"] == "pick" and "object_constraints" in kwargs:
                assert operation["params"]["target_entity_name"] == 1


def test_the_lookahead_is_built_from_the_dfa_it_steers():
    restricted = character_dfa(object_constraints=ObjectConstraints(restrictions=(("pick", "target_entity_name", (1,)),)))
    full = character_dfa()
    assert len(restricted.actions) == len(full.actions) - 1
    table = table_for(restricted, random_hmm(4, VOCABULARY, seed=0))
    assert table.core.group_of.numel() == len(restricted.actions)


def test_a_strong_lookahead_changes_decisions_and_counts_them():
    dfa = character_dfa()
    hmm = random_hmm(8, VOCABULARY, seed=5)
    table = table_for(dfa, hmm)
    changed = 0
    differs = 0
    for seed in range(6):
        steered, processor = decode(dfa, table, hmm, 1000.0, seed)
        plain, _ = decode(dfa, table, hmm, 1e-9, seed)
        changed += processor.session.stats["changed"]
        differs += steered != plain
        assert processor.session.stats["scored_steps"] <= processor.session.stats["steps"]
    assert changed > 0 and differs > 0


@pytest.mark.parametrize("score", ["joint", "conditional"])
def test_both_scores_keep_the_legal_token_set(score):
    dfa = character_dfa()
    hmm = random_hmm(8, VOCABULARY, seed=7)
    table = table_for(dfa, hmm)
    for seed in range(3):
        labels, _ = decode(dfa, table, hmm, 50.0, seed, score=score)
        assert dfa.accepts(labels)
