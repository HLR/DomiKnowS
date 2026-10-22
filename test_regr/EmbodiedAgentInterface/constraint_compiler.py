"""EAI action adapter for the shared predicate compiler and grounded state.

Only planning_facts and exposed entity metadata initialize decoding. The
reference-derived PreparedEAIGoal state is deliberately not an input.
"""
from dataclasses import dataclass, replace

from domiknows.generation.grounded import PlanningFacts, Operation, observations, report, FactState
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding, PredicateConstrainedDFA, PredicateEvent, SemanticTransition,
    compile_predicate_constraints, find_predicate_monitor, PredicateMonitorDFA,
)
from .world_graph import ACTION_SPECS, _precondition_status


def compile_world_constraints(world):
    from domiknows.generation.grounded import predicate_bindings
    bindings = [PredicateBinding(world.action, "operation", "operation"),
                PredicateBinding(world.trajectory, "episode", "episode")]
    bindings += [PredicateBinding(c, "action:"+name, "operation") for name,c in world.actions.items()]
    bindings += [PredicateBinding(c, "precondition:"+name, "operation") for name,c in world.precondition_concepts.items()]
    bindings += predicate_bindings(world.grounded_constraints)
    return compile_predicate_constraints(world.graph, bindings)


@dataclass(frozen=True)
class CompiledEAIPlannerDFA(PredicateConstrainedDFA):
    world_constraint_bundle: object = None
    generation_graph: object = None


_UNSET = object()
_FINISHED_LIMIT = 1 << 16


@dataclass(frozen=True)
class EAIState:
    grounded: object
    pending: str | None = None
    args: tuple = ()
    ended: bool = False


class EAIExecutor:
    def __init__(self, world, context, entities):
        self.world = world
        self.context = dict(context or {})
        self.context["generation_entity_types"] = tuple(self.context.get("generation_entity_types") or ())
        self.planning = PlanningFacts.parse(self.context.get("planning_facts"), entities=entities,
                                             default_destination=self.context.get("task_target_container"),
                                             constants=("left", "right", "character"))
        initial = self.planning.initial
        holdings = dict(initial.holdings)
        for f in initial.facts.positive:
            if len(f) == 3 and f[0] in ("holds_lh", "holds_rh"):
                hand = "left" if f[0] == "holds_lh" else "right"
                if hand in holdings and holdings[hand] != f[2]:
                    raise ValueError("contradictory hand observations")
                holdings[hand] = f[2]
        def signed(facts):
            yes, no = set(), set(facts.negative)
            for f in facts.positive:
                if f[0].startswith("not_"):
                    no.add((f[0][4:], *f[1:]))
                else:
                    yes.add(f)
            for f in tuple(yes):
                if f[0] == "closed": no.add(("open", *f[1:]))
                if f[0] == "open": no.add(("closed", *f[1:]))
            return FactState(frozenset(yes), frozenset(no))
        self.signed = signed
        observed = signed(initial.facts)
        for hand, obj in holdings.items():
            predicate = "holds_lh" if hand == "left" else "holds_rh"
            if obj is not None and observed.truth((predicate, "character", obj)) is False:
                raise ValueError("contradictory held-object observations")
            if hand in dict(initial.holdings) and any(
                len(f) == 3 and f[0] == predicate and f[2] != obj for f in observed.positive
            ):
                raise ValueError("contradictory held-object observations")
        observed = observed.update((("holds_lh" if hand == "left" else "holds_rh", "character", obj)
                                    for hand, obj in holdings.items() if obj is not None))
        self.planning = replace(self.planning, initial=replace(initial, facts=observed, holdings=tuple(sorted(holdings.items()))),
                                goals=signed(self.planning.goals), preserve=signed(self.planning.preserve))

    def advance(self, state, name, args):
        from .reward import _ActionEvent, apply_observed_action, _placement_relation, _side_for_action
        hand = _side_for_action(name)
        entity = args[0] if args else None
        holdings = dict(state.holdings)
        grasp = any(word in name for word in ("grasp", "grab", "pickup", "take"))
        if hand is None:
            hand = ("right" if holdings.get("right") is None else "left") if grasp else next(
                (h for h in ("right", "left") if entity is not None and holdings.get(h) == entity), next(
                    (h for h in ("right", "left") if holdings.get(h) is not None), "right"))
        entity = args[0] if args else None
        relation = _placement_relation(name)
        destination = entity if relation or name == "pour" else None
        subject = holdings.get(hand) if destination is not None else entity
        release = bool(relation) or any(word in name for word in ("release", "realease", "drop", "putobjback"))
        already = bool(relation and destination and any(
            len(f) == 3 and f[0] == relation and f[2] == destination for f in state.facts.positive))
        if release and not relation and entity is not None:
            already = any(len(f) == 3 and f[0] in {"inside", "nextto", "ontop", "onfloor", "under"}
                          and f[1] == entity for f in state.facts.positive)
        capabilities = []
        if grasp: capabilities.append((subject, "graspable"))
        if name in ("open", "close"): capabilities.append((subject, "openable"))
        if name == "pour": capabilities.append((subject, "pourable"))
        observational = name in {"find", "lookat", "watch", "turnto", "read", "pointat"}
        op = Operation(name, subject, destination, hand, grasp=grasp,
                       requires_held=bool(relation) or release or name == "pour", release=release,
                       affordances=tuple(capabilities), writes=() if observational else tuple(x for x in (subject,destination) if x),
                       relation=relation, idempotent=already, container_required=relation == "inside",
                       preconditions=(("open", destination),) if relation == "inside" else ())
        source = set(state.facts.positive)
        for h, obj in state.holdings:
            if obj is not None:
                source.add(("holds_lh" if h == "left" else "holds_rh", "character", obj))
        predicted = apply_observed_action(_ActionEvent(name, args), source, world_bundle=self.world)
        predicted_signed = self.signed(FactState(frozenset(predicted)))
        next_state, checks = self.planning.evaluate(
            state, op, positive=predicted_signed.positive-source,
            negative=(source-predicted_signed.positive) | predicted_signed.negative,
            holding_fact=lambda hand, obj: ("holds_lh" if hand == "left" else "holds_rh", "character", obj),
            clear_relations=tuple((r, subject) for r in ("inside","ontop","under","onfloor") if subject and (grasp or relation)))
        values = {"operation": True, **observations(checks)}
        values.update({"action:"+action: name == action for action in self.world.actions})
        for kind in self.world.precondition_concepts:
            _applicable, satisfied = _precondition_status(kind, name, args, source,
                                                          (self.context.get("generation_entity_types") or ()))
            if kind in {"placement_source_ready", "release_source_ready", "pour_source_ready"} and hand not in holdings:
                satisfied = True  # Unknown source state is separately reported, not fabricated as empty.
            values["precondition:"+kind] = bool(satisfied)
        return next_state, checks, values

    def audit(self, events):
        state = self.planning.initial
        values, checks = [], []
        for event in events:
            state, result, facts = self.advance(state, event.name, event.args)
            values.append(facts)
            checks.extend(result)
        final = self.planning.completion(state)
        checks.append(final)
        return values, {"episode": True, **observations((final,))}, report(checks)


class EAIAdapter:
    def __init__(self, world, vocabulary, context):
        self.vocabulary = vocabulary
        self.world = world
        entities = context.get("entities") or context.get("entity_table")
        if isinstance(entities, dict): entities = entities.values()
        # Vocabulary identities are a syntactic universe, never a source of state/goal truth.
        if not entities:
            entities = set(context.get("generation_entity_types") or ())
            entities.update(x for x in vocabulary.tokens if x not in world.actions and x != vocabulary.eos_token)
        self.executor = EAIExecutor(world, context, entities)
        self.start_state = EAIState(self.executor.planning.initial)
        self._finished = {}

    def finish(self, state):
        if state.pending is None:
            return SemanticTransition(state)
        # Closing the pending action depends on the state alone, yet every candidate next token of that state
        # closes it again (an action, the end token, ...): compute it once per state.
        cached = self._finished.get(state, _UNSET)
        if cached is _UNSET:
            cached = self._finish(state)
            if len(self._finished) >= _FINISHED_LIMIT:
                self._finished.clear()
            self._finished[state] = cached
        return cached

    def _finish(self, state):
        if len(state.args) < ACTION_SPECS[state.pending].min_args:
            return None
        successor, _checks, values = self.executor.advance(state.grounded, state.pending, state.args)
        return SemanticTransition(EAIState(successor), (PredicateEvent("operation", values),))

    def step(self, state, symbol):
        from .reward import _normalize_name
        token = self.vocabulary.token_for_label(int(symbol))
        if state.ended:
            return SemanticTransition(state) if token == self.vocabulary.eos_token else None
        if token == self.vocabulary.eos_token:
            result = self.finish(state)
            if result is None: return None
            ended = replace(result.state, ended=True)
            return SemanticTransition(ended, result.events + self.acceptance_events(ended))
        token = _normalize_name(token)
        if token in self.world.actions:
            previous = self.finish(state)
            if previous is None: return None
            successor = EAIState(previous.state.grounded, token)
            if ACTION_SPECS[token].max_args == 0:
                result = self.finish(successor)
                return SemanticTransition(result.state, previous.events + result.events)
            return SemanticTransition(successor, previous.events)
        if state.pending is None: return None
        successor = replace(state, args=state.args+(token,))
        if len(successor.args) >= ACTION_SPECS[state.pending].max_args:
            return self.finish(successor)
        return SemanticTransition(successor)

    def is_accepting(self, state):
        return state.ended and state.pending is None

    def acceptance_events(self, state):
        return (PredicateEvent("episode", {"episode": True,
                **observations((self.executor.planning.completion(state.grounded),))}, "completion"),)


def bind_eai_policy(dfa, vocabulary, context):
    if find_predicate_monitor(dfa) is not None: return dfa
    if not isinstance(dfa, CompiledEAIPlannerDFA): return dfa
    from domiknows.generation import bind_contextual_dfa
    context = context or {}
    adapter = EAIAdapter(dfa.world_constraint_bundle, vocabulary, context)
    completion = adapter.executor.planning.completion(adapter.start_state.grounded)
    from domiknows.generation.dfa.core import with_initial_completion
    allow_noop = bool(completion.applicable and completion.satisfied is True
                      and not completion.missing and not context.get("task_skill_sequence"))
    grammar = with_initial_completion(dfa, vocabulary.label_for_token(vocabulary.eos_token), enabled=allow_noop)
    grammar = bind_contextual_dfa(grammar, dfa.generation_graph, context)
    return PredicateMonitorDFA(grammar, dfa.constraint_program, adapter)
