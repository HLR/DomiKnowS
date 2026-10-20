"""Action-level HMM lookahead bound to the DomiKnowS plan-language DFA.

The token-level product of an HMM with this DFA has millions of states (see ``official_hmm``).
``DFAActionLookahead`` instead reads the *actions* of the DFA (skill, object id), gives each one a
canonical token serialisation, and ranks the DFA's legal tokens by the HMM mass of the plans they
leave open (``domiknows.generation.applications.action_lookahead``).

Constraint contract: the DFA is the only authority on which tokens are legal. It already encodes
the JSON syntax, the skill/argument roles, the complete-subtask patterns, the id range or object
restrictions and the operation bound. The lookahead is built from that same DFA object (its action
inventory, patterns and budget), only adds a finite bonus to tokens the DFA allows, and leaves every
token outside an action's span (whitespace, brackets) unscored. A token the DFA forbids stays at
``-inf`` whatever the weight or the HMM.

Canonical serialisation: the HMM is fitted on ``json.dumps(plan, indent=4)`` tokenised as one
string, and a BPE tokenizer merges characters across the operation boundary (``' },\\n'`` before
another operation, ``' }\\n'`` before the closing ``']'``). The plan is therefore
``open + C(a_0) + .. + C(a_{n-1}) + S(a_n) + close`` where ``C``/``S`` tokenise the operation
together with the separator that follows it; ``tokenization_gaps`` checks that identity.
"""
from __future__ import annotations

import json
import math

import torch

from domiknows.generation.applications.action_lookahead import ActionAutomaton, ActionLookahead, TokenOperators

PLAN_OPEN, PLAN_CLOSE = "[\n", "]"  # json.dumps(plan, indent=4): '[\n' + operations + '\n]'
INDENT = "    "
MASS_FLOOR = 1e-300
MAX_PENALTY = 60.0


def canonical_body(skill, roles, pointer):
    """One operation as ``json.dumps(plan, indent=4)`` prints it inside the array, without its indent."""
    text = json.dumps([{"name": skill, "params": {role: pointer for role in roles}}], indent=4)
    prefix, suffix = PLAN_OPEN + INDENT, "\n" + PLAN_CLOSE
    assert text.startswith(prefix) and text.endswith(suffix)
    return text[len(prefix):-len(suffix)]


def operation_texts(skill, roles, pointer):
    """``(continue, stop)`` texts of an operation: followed by another one, or by the closing bracket."""
    body = INDENT + canonical_body(skill, roles, pointer)
    return body + ",\n", body + "\n"


def plan_tokens(tokenizer, plan, arguments):
    """Canonical tokens of ``plan`` assembled from the operation segments (open, C.., S, close)."""
    encode = lambda text: list(tokenizer(text, add_special_tokens=False).input_ids)
    texts = [operation_texts(op["name"], arguments[op["name"]], next(iter(op["params"].values()), None)) for op in plan]
    tokens = encode(PLAN_OPEN)
    for index, (carry, stop) in enumerate(texts):
        tokens += encode(stop if index == len(texts) - 1 else carry)
    return tokens + encode(PLAN_CLOSE)


def tokenization_gaps(tokenizer, plans, arguments):
    """Plans whose segment-wise tokens differ from tokenising ``json.dumps(plan, indent=4)`` whole."""
    return [plan for plan in plans
            if plan_tokens(tokenizer, plan, arguments)
            != list(tokenizer(json.dumps(plan, indent=4), add_special_tokens=False).input_ids)]


class DFAActionLookahead:
    """Reachability table for one plan-language DFA and one HMM (shared ``TokenOperators``)."""

    def __init__(self, dfa, tokenizer, operators: TokenOperators):
        skills = list(dfa.skill_arguments)

        def encode(text):
            return list(tokenizer(text, add_special_tokens=False).input_ids)

        groups, cont, stop = [], [], []
        for skill, pointer in dfa.actions:
            carry, last = operation_texts(skill, dfa.skill_arguments[skill], pointer)
            groups.append(skills.index(skill))
            cont.append(encode(carry))
            stop.append(encode(last))
        index, successors, accepting = dfa.semantic_automaton()
        self.dfa, self.states = dfa, index
        self.core = ActionLookahead(operators, groups=groups, cont=cont, stop=stop, open=encode(PLAN_OPEN),
                                    close=encode(PLAN_CLOSE) + [dfa.eos_label],
                                    automaton=ActionAutomaton(0, accepting, successors), max_actions=dfa.max_operations)
        self._tensors = {}

    def session(self, belief: torch.Tensor) -> "LookaheadSession":
        return LookaheadSession(self, belief)

    def label_tensor(self, state, key, groups, device):
        slot = (state, key, str(device))
        if slot not in self._tensors:
            self._tensors[slot] = torch.tensor(groups[key], dtype=torch.long, device=device)
        return self._tensors[slot]


class LookaheadSession:
    """Per-generation state: the HMM belief at the start of the current action and its scores."""

    def __init__(self, table: DFAActionLookahead, belief: torch.Tensor):
        self.table = table
        self.span = table.core.span(belief, 0, 0)
        self.pending = None  # (span, action) of an operation closed by a bare brace, awaiting its separator
        self.stats = {"steps": 0, "scored_steps": 0, "changed": 0}
        self._node_mass: dict[int, float] = {}

    def observe(self, previous, label, state, belief: torch.Tensor) -> None:
        """Called after each generated token; a finished operation starts a new span at the new belief."""
        if state[2] != "separator":
            self.pending = None
        if self.span is not None and state[0] == self.span.count:
            return
        finished = self.table.dfa.completed_action(previous, label) if state[0] == previous[0] + 1 else None
        self.pending = (self.span, finished[0]) if finished is not None and state[2] == "separator" and self.span is not None else None
        index = self.table.states.get(state[1])
        self.span = None if index is None else self.table.core.span(belief, index, state[0])
        self._node_mass = {}

    def _mass(self, key) -> float:
        kind = key[0]
        if kind == "action":
            if key[1] not in self._node_mass:
                self._node_mass[key[1]] = self.span.mass_of(self.table.dfa.node_actions[key[1]])
            return self._node_mass[key[1]]
        if kind == "done":
            return self.span.action_mass(key[1], key[2])
        if self.pending is None:
            return 0.0
        span, action = self.pending
        return span.action_mass(action, kind)  # "continue" or "stop"

    def scores(self, state) -> dict:
        """``log`` HMM mass of the plans left open by each decision group of ``state``."""
        if self.span is None or state[0] != self.span.count:
            return {}
        groups = self.table.dfa.effect_groups(state)
        if self.pending is None:
            groups = {key: labels for key, labels in groups.items() if key[0] not in ("continue", "stop")}
        return {key: math.log(max(self._mass(key), MASS_FLOOR)) for key in groups}

    def apply(self, result: torch.Tensor, state, weight: float) -> None:
        """Add ``weight * (log mass - best log mass)`` to the tokens of each decision group, in place.

        ``result`` already holds ``-inf`` for every token the DFA forbids, so those stay forbidden.
        Tokens outside every group are unchanged, and the best-ranked group is unchanged too.
        """
        self.stats["steps"] += 1
        scores = self.scores(state)
        if len(scores) < 2:
            return
        best = max(scores.values())
        if best <= math.log(MASS_FLOOR) + 1:
            return
        groups = self.table.dfa.effect_groups(state)
        before = int(result[0].argmax())
        for key, score in scores.items():
            bonus = weight * max(score - best, -MAX_PENALTY)
            if bonus:
                result[0, self.table.label_tensor(state, key, groups, result.device)] += bonus
        self.stats["scored_steps"] += 1
        self.stats["changed"] += int(int(result[0].argmax()) != before)
