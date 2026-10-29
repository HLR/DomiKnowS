"""Action-budget search using the same compiled policy as plan verification.

Queries return True (a witness exists), False (exhaustively impossible), or
None (required observations are missing). Computational exhaustion raises
SemanticSearchLimitExceeded and is never cached as impossibility.
"""
from itertools import combinations, product
from operator import index

from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
from .official.knowledge_dfa import KnowledgePlanDFA


class WorldActionReachability:
    def __init__(self, policy, *, strict=True, max_state_expansions=10000):
        self.policy = policy
        self.adapter = policy.adapter
        self.vocabulary = self.adapter.vocabulary
        self.strict = strict
        self.max_state_expansions = index(max_state_expansions)
        if self.max_state_expansions < 1:
            raise ValueError("max_state_expansions must be positive")
        self.max_actions = policy.base_dfa.max_operations
        self.active_families = {rule.name.removeprefix("grounded__") for rule in policy.program.rules
                                if rule.name.startswith("grounded__")}
        self._moves, self._terminal, self._answers = {}, {}, {}

    def _semantic(self, state):
        value = state.semantic
        # Optional arguments defer completion until the next skill or EOS.
        return self.adapter._finish(value).state.semantic if value.skill is not None else value.semantic

    def terminal(self, state):
        if state not in self._terminal:
            target = self.policy.step(state, self.vocabulary.eos_label)
            valid = target is not None and self.policy.is_accepting(target)
            if valid and self.strict and "goal_completion" in self.active_families:
                check = self.adapter.requirements.planning.completion(self._semantic(state).grounded)
                if check.applicable and (check.satisfied is None or check.missing):
                    valid = None
            self._terminal[state] = valid
        return self._terminal[state]

    def advance(self, state, skill, params):
        """Return (successor, grounded); None successor means an LC/grammar rejection."""
        from .plan_codec import plan_to_tokens
        key = (state, skill, tuple(sorted(params.items())))
        if key not in self._moves:
            tokens = plan_to_tokens([{"name": skill, "params": params}], self.adapter.entities,
                                    world=self.adapter.world)[:-1]
            target = state
            for token in tokens:
                target = self.policy.step(target, self.vocabulary.label_for_token(token))
                if target is None:
                    break
            known = True
            if target is not None and self.strict:
                _, checks = self.adapter.semantics.grounded_transition(self._semantic(state), skill, params)
                known = all(c.family not in self.active_families or not c.applicable
                            or (c.satisfied is not None and not c.missing) for c in checks)
            self._moves[key] = target, known
        return self._moves[key]

    def actions(self):
        world = self.adapter.world
        ids = tuple(self.adapter.entities)
        for skill, required in world.skill_arguments.items():
            optional = world.optional_skill_arguments.get(skill, ())
            for size in range(len(optional) + 1):
                for selected in combinations(optional, size):
                    roles = (*required, *selected)
                    for values in product(ids, repeat=len(roles)):
                        yield skill, dict(zip(roles, values))

    def can_reach(self, state=None, remaining_actions=None):
        state = self.policy.start_state if state is None else state
        if remaining_actions is None:
            semantic = state.semantic
            used = (semantic.semantic.completed_operations
                    - len(self.adapter.requirements.planning.completed_skill_prefix)
                    + int(semantic.skill is not None))
            remaining_actions = self.max_actions - used
        budget = index(remaining_actions)
        work = [0]

        def search(current, remaining):
            key = current, remaining
            if key in self._answers:
                return self._answers[key]
            if remaining < 0:
                return False
            work[0] += 1
            if work[0] > self.max_state_expansions:
                raise SemanticSearchLimitExceeded("action reachability expansion limit exceeded; feasibility is undetermined")
            terminal = self.terminal(current)
            if terminal is True:
                self._answers[key] = True
                return True
            unknown = terminal is None
            if remaining:
                for skill, params in self.actions():
                    successor, known = self.advance(current, skill, params)
                    if successor is None:
                        continue
                    result = search(successor, remaining - 1)
                    if result is True and known:
                        self._answers[key] = True
                        return True
                    unknown |= result is None or (result is True and not known)
            self._answers[key] = None if unknown else False
            return self._answers[key]

        return search(state, budget)


class ReachablePlanDFA(KnowledgePlanDFA):
    """Native JSON decoding intersected with a request-bound world LC policy.

World-policy states are committed once per completed action. Native pattern,
causal and instruction restrictions participate in the same bounded search.
The policy and this DFA must describe the same entity ids and action schemas.
"""

    def __init__(self, *args, policy, strict=True, max_state_expansions=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_reachability = WorldActionReachability(
            policy, strict=strict, max_state_expansions=max_state_expansions)
        adapter = policy.adapter
        if set(adapter.entities) != set(range(self.entity_count)):
            raise ValueError("native entity ids must match the bound world policy")
        if self.max_operations > self.world_reachability.max_actions:
            raise ValueError("native action budget exceeds the world policy budget")
        for skill, roles in self.skill_arguments.items():
            required = adapter.world.skill_arguments.get(skill)
            optional = adapter.world.optional_skill_arguments.get(skill, ())
            if required is None or not set(required) <= set(roles) or not set(roles) <= set(required) | set(optional):
                raise ValueError(f"native/world action schemas differ for {skill!r}")
        self.start_state += (policy.start_state,)
        self._action_moves, self._action_answers = {}, {}

    def _successor(self, action, count, extras):
        key = action, count, extras
        if key not in self._action_moves:
            if not super().operation_admitted(action, count, extras[:4]):
                result = None, True
            else:
                skill, pointer = self.actions[action]
                params = {role: pointer for role in self.skill_arguments[skill]}
                world, known = self.world_reachability.advance(extras[4], skill, params)
                result = (None if world is None else
                          super().advance_state(action, count, extras[:4]) + (world,)), known
            self._action_moves[key] = result
        return self._action_moves[key]

    def _complete(self, prefixes, extras):
        if () not in prefixes or not super().plan_complete(extras[:4]):
            return False
        return self.world_reachability.terminal(extras[4])

    def _query(self, node, prefixes, count, extras, work):
        key = node, prefixes, count, extras
        if key in self._action_answers:
            return self._action_answers[key]
        work[0] += 1
        if work[0] > self.world_reachability.max_state_expansions:
            raise SemanticSearchLimitExceeded("native action reachability expansion limit exceeded; feasibility is undetermined")
        # node=None means an action boundary where stopping is also possible.
        terminal = self._complete(prefixes, extras) if node is None else False
        if terminal is True:
            self._action_answers[key] = True
            return True
        unknown = terminal is None
        if count < self.max_operations:
            skills = self.allowed_skills(prefixes, count)
            for action in self.node_actions[0 if node is None else node]:
                skill = self.actions[action][0]
                if skill not in skills:
                    continue
                successor, known = self._successor(action, count, extras)
                if successor is None:
                    continue
                result = self._query(None, self.next_semantics(prefixes, skill), count + 1, successor, work)
                if result is True and known:
                    self._action_answers[key] = True
                    return True
                unknown |= result is None or (result is True and not known)
        self._action_answers[key] = None if unknown else False
        return self._action_answers[key]

    def can_reach_goal(self, state=None):
        state = self.start_state if state is None else state
        count, prefixes, phase, node, _ = state[:5]
        action_node = node if phase == "operation" and not (count == 0 and node == 0) else None
        return self._query(action_node, prefixes, count, state[5:], [0])

    def allowed_tokens(self, state, remaining_steps=None):
        # Do not leave whitespace loops available when no complete plan is
        # proven reachable. This also gives the processor a grounding failure
        # instead of silently generating until its token horizon expires.
        if state is None or self.can_reach_goal(state) is not True:
            return set()
        return super().allowed_tokens(state, remaining_steps)

    def reachable(self, node, prefixes, count, extras):
        return self._query(node, prefixes, count, extras, [0]) is True

    def operation_admitted(self, action, count, extras):
        successor, known = self._successor(action, count, extras)
        return successor is not None and known

    def advance_state(self, action, count, extras):
        return self._successor(action, count, extras)[0]

    def plan_complete(self, extras):
        return super().plan_complete(extras[:4]) and self.world_reachability.terminal(extras[4]) is True

    def byte_step(self, state, byte):
        # The base native language requires a nonempty plan; a bound world
        # policy may explicitly allow a no-op for an already satisfied goal.
        if state[2] == "operation" and state[3] == 0 and state[0] == 0 and byte == ord("]"):
            return ((0, state[1], "closed", 0, "") + state[5:]
                    if self._complete(state[1], state[5:]) is True else None)
        return super().byte_step(state, byte)
