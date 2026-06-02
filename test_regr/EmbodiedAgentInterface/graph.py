from dataclasses import dataclass

from dataset import ACTION_VOCAB, EOS_TOKEN
from domiknows.generation.constraints import GenerationConstraint


@dataclass(frozen=True)
class ActionFollowedByObjectConstraint(GenerationConstraint):
    action_tokens: tuple[str, ...]
    object_tokens: tuple[str, ...]
    name: str = "actions requiring objects are followed by object tokens"
    supports_dfa = True
    supports_domiknows = False

    def to_dfa(self, vocabulary):
        from domiknows.generation.automata import DFA

        action_labels = frozenset(
            vocabulary.label_for_token(token)
            for token in self.action_tokens
            if token in vocabulary.tokens
        )
        object_labels = frozenset(
            vocabulary.label_for_token(token)
            for token in self.object_tokens
            if token in vocabulary.tokens
        )
        alphabet = frozenset(vocabulary.alphabet)
        states = frozenset({"ok", "need_object", "dead"})

        def step(state, symbol):
            if state == "dead":
                return "dead"
            if state == "need_object":
                return "ok" if symbol in object_labels else "dead"
            return "need_object" if symbol in action_labels else "ok"

        return DFA(
            states=states,
            alphabet=alphabet,
            transitions={(state, symbol): step(state, symbol) for state in states for symbol in alphabet},
            start_state="ok",
            accepting_states=frozenset({"ok"}),
            dead_states=frozenset({"dead"}),
        )


def _token_concept(bundle, token):
    return getattr(bundle.generated_token, str(bundle.vocabulary.label_for_token(token)))


def _disjunction(calls, orL):
    return calls[0] if len(calls) == 1 else orL(*calls)


def _add_action_object_logical_constraint(graph, bundle, action_tokens, object_tokens):
    from domiknows.generation import mark_for_dfa
    from domiknows.graph.logicalConstrain import ifL, orL

    valid_actions = tuple(token for token in action_tokens if token in bundle.vocabulary.tokens)
    valid_objects = tuple(token for token in object_tokens if token in bundle.vocabulary.tokens)
    if not valid_actions or not valid_objects:
        return

    with graph:
        action_calls = [
            _token_concept(bundle, action)("x", path=("before", bundle.first_token))
            for action in valid_actions
        ]
        object_calls = [
            _token_concept(bundle, obj)("y", path=("before", bundle.second_token))
            for obj in valid_objects
        ]
        lc = ifL(
            bundle.is_before_rel("before"),
            ifL(_disjunction(action_calls, orL), _disjunction(object_calls, orL)),
        )
        # To enforce the state in DFA
        mark_for_dfa(lc, ActionFollowedByObjectConstraint(valid_actions, valid_objects))


def create_generation_graph(
    max_steps=8,
    required_tokens=None,
    forbidden_tokens=None,
    vocab=None,
    object_tokens=None,
    action_tokens=None,
    enforce_action_object=True,
):
    from domiknows.generation import GenerationEncoder, default_generation_constraints

    vocab = tuple(vocab or ACTION_VOCAB)
    object_tokens = tuple(object_tokens or ())
    action_tokens = tuple(action_tokens or ())
    constraints = default_generation_constraints(
        max_non_eos_count=max_steps - 1,
        required_tokens=required_tokens or {},
        forbidden_tokens=forbidden_tokens or [],
    )

    encoder = GenerationEncoder(
        vocab,
        eos_token=EOS_TOKEN,
        graph_name="eai_generation_graph",
    )
    graph, bundle = encoder.build_graph(constraints)
    if enforce_action_object:
        _add_action_object_logical_constraint(graph, bundle, action_tokens, object_tokens)
    return graph, bundle


# Backward-compatible alias for older imports in this folder.
def create_graph(max_steps=8):
    graph, bundle = create_generation_graph(max_steps=max_steps)
    return graph, bundle
