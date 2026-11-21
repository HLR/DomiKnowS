"""Bind portable executor declarations to the existing Shared DomiKnowS graph."""
from dataclasses import dataclass

from .shared_constraints import SHARED_RULES, bind_shared, shared_observations, shared_profile


CAPABILITIES = 'shared-verified-capabilities'
EXECUTOR = 'shared-verified-executor'
POLICIES = ('none', 'shared-verified', CAPABILITIES, EXECUTOR)
GROUNDED_RULES = frozenset('grounded__' + name for name in
                           ('holding', 'capacity', 'affordance', 'compatibility'))
SUBJECT = 'target_entity_name'
DESTINATION = 'target_container_name'


@dataclass(frozen=True)
class Action:
    name: str
    arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...] = ()
    hand: str = 'right'
    grasp: bool = False
    requires_held: bool = False
    release: bool = False
    affordances: tuple[tuple[str, str], ...] = ()
    idempotent: bool = False

    def bind(self, subject, destination):
        from domiknows.generation.grounded_facts import Operation

        values = {SUBJECT: subject, DESTINATION: destination}
        return Operation(self.name, subject, destination, self.hand, grasp=self.grasp,
                         requires_held=self.requires_held, release=self.release,
                         affordances=tuple((values[role], prop) for role, prop in self.affordances),
                         idempotent=self.idempotent)


@dataclass(frozen=True)
class ExecutorProfile:
    actions: tuple[Action, ...]
    capacities: tuple[tuple[str, int], ...] = ()
    initial_holdings: tuple[tuple[str, int | None], ...] = ()
    effects: tuple = ()  # PlanningFacts action/add/delete/released-hands declarations.

    def operation(self, skill, subject, destination):
        from domiknows.generation.grounded_facts import Operation

        for action in self.actions:
            if action.name == skill:
                return action.bind(subject, destination)
        # A complete DFA may still visit a rejecting sink during direct replay.
        # Only the compiled capability rule rejects unsupported operations.
        return Operation(skill, subject, destination)


def pick_place_profile(hand='right'):
    """Contract for a single gripper verified empty by the reset protocol."""
    return ExecutorProfile((
        Action('pick', (SUBJECT,), hand=hand, grasp=True, affordances=((SUBJECT, 'graspable'),)),
        Action('place', (DESTINATION,), hand=hand, requires_held=True, release=True),
    ), capacities=((hand, 1),), initial_holdings=((hand, None),))


def compiled_rule_names(policy, profile=None, vocabulary=None):
    if policy == 'none':
        return []
    names = set(SHARED_RULES)
    if profile is not None and policy in (CAPABILITIES, EXECUTOR):
        if set(vocabulary.skills) - {action.name for action in profile.actions}:
            names.add('executor_unsupported_skill')
        if policy == EXECUTOR:
            names.update(GROUNDED_RULES)
    return sorted(names)


def bind_plan_policy(policy, vocabulary, maximum, context, profile=None):
    """Compile the action inventory and bind observations without task facts."""
    if policy not in POLICIES:
        raise ValueError(f'unknown plan policy {policy}')
    if policy == 'none':
        return None
    # Strict source evaluation has no independently observed executor profile.
    if policy == 'shared-verified' or profile is None:
        return bind_shared(vocabulary, maximum, context)

    from domiknows.generation.grounded_facts import GroundedState, PlanningFacts
    from test_regr.VLABenchAgentInterface.constraint_compiler import WorldConstraintDFA
    from test_regr.VLABenchAgentInterface.task_policy import TaskRequirements

    actions = {action.name: action for action in profile.actions}
    required = dict(vocabulary.skill_arguments)
    optional = dict(vocabulary.optional_skill_arguments)
    if not actions or len(actions) != len(profile.actions) or any(
        skill not in vocabulary.skills or action.arguments != required[skill]
        or action.optional_arguments != optional.get(skill, ())
        for skill, action in actions.items()
    ):
        raise ValueError('executor action roles must match the canonical decoder API')
    grounded = GROUNDED_RULES if policy == EXECUTOR else frozenset()
    template = shared_profile(vocabulary, maximum, frozenset(actions), grounded)
    entities = shared_observations(context)
    if not grounded:
        return WorldConstraintDFA(template, vocabulary, template.world_constraint_bundle,
                                  template.constraint_program, entities)

    # Only explicit affordance/compatibility observations can enter this view.
    # Missing properties remain absent from PlanningFacts.
    properties = tuple(sorted((entities[index], prop, truth)
                              for index, prop, truth in context.get('executor_properties', ())))
    if any(type(value) is not bool for _, _, value in properties):
        raise ValueError('verified affordance properties must be Boolean')
    if len({(entity, prop) for entity, prop, _ in properties}) != len(properties):
        raise ValueError('duplicate affordance observation')
    # Use the shared fact parser to validate the compatibility representation.
    compatibility = PlanningFacts.parse({
        'compatibility': context.get('executor_compatibility', ()),
    }, entities=entities.values(), resolve=entities.__getitem__).compatibility
    holding = tuple((hand, None if pointer is None else entities[pointer])
                    for hand, pointer in profile.initial_holdings)
    facts = PlanningFacts(initial=GroundedState(holdings=holding), capacities=profile.capacities,
                          properties=properties, effects=profile.effects, compatibility=compatibility,
                          supplied=True)
    return WorldConstraintDFA(template, vocabulary, template.world_constraint_bundle,
                              template.constraint_program, entities, requirements=TaskRequirements(planning=facts),
                              operation_binding=profile.operation)
