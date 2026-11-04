"""Image-only, uncertainty-aware grounding; candidate preferences never forbid tokens."""
from dataclasses import replace
import math

try:
    from ..object_grounding import ObjectConstraints, perceived_object_constraints
    from .perception_observations import merged_constraints
    from .selection_observations import selection_constraints
except ImportError:  # flat-module entry point
    from object_grounding import ObjectConstraints, perceived_object_constraints
    from official.perception_observations import merged_constraints
    from official.selection_observations import selection_constraints


def soft_observation(observed):
    """Retain positive candidates for scoring, never claim an exhaustive inventory."""
    supported = {tag for tag, _ in (*observed.categories, *observed.cards)}
    preferred = tuple((skill, role, tuple(i for i in ids if i in supported))
                      for skill, role, ids in observed.candidates.restrictions
                      if any(i in supported for i in ids))
    request = observed.audit.get('selection_request') or {}
    if request.get('kind') in ('value', 'spatial') and observed.audit.get('uncertain_inventory', True):
        preferred = ()  # ranks require the whole group, not just positive detections
    return replace(observed, visible=None, grounding=None, candidates=ObjectConstraints(),
                   soft_candidates=ObjectConstraints(preferred, observed.candidates.evidence),
                   audit={**observed.audit, 'constraint_policy': 'soft',
                          'absence_semantics': 'unknown; not a hard exclusion'})


def verified_observation(primary, verifier, instruction, selection):
    """Agree on positive identities; box unions prevent false precision in ordinal order.

    Both inputs must come from observe_episode (images/instruction only). Agreement
    is still a prediction, not calibrated truth, and feeds finite preferences only.
    """
    visible = tuple(sorted(set(primary.visible or ()) & set(verifier.visible or ())))
    categories = {tag: kind for tag, kind in primary.categories
                  if tag in visible and dict(verifier.categories).get(tag) == kind}
    cards = {tag: face for tag, face in primary.cards
             if tag in visible and dict(verifier.cards).get(tag) == face}
    def boxes(observed):
        return {int(k): tuple(v) for k, v in observed.audit.get('boxes', {}).items()}
    a, b = boxes(primary), boxes(verifier)
    union = {tag: (min(a[tag][0], b[tag][0]), min(a[tag][1], b[tag][1]),
                   max(a[tag][2], b[tag][2]), max(a[tag][3], b[tag][3]))
             for tag in visible if tag in a and tag in b}
    uncertain = (set(primary.visible or ()) != set(verifier.visible or ())
                 or primary.audit.get('uncertain_inventory', True)
                 or verifier.audit.get('uncertain_inventory', True))
    candidates = merged_constraints(
        perceived_object_constraints(instruction, categories, tuple(t for t in visible if t not in categories)),
        selection_constraints(selection, instruction, visible, union, categories, cards, uncertain))
    audit = {**primary.audit, 'verification': {
        'status': 'agreement' if visible else 'abstained', 'agreed_tags': list(visible),
        'agreed_category_tags': sorted(categories), 'agreed_card_tags': sorted(cards),
        'verifier_fingerprint': verifier.audit.get('fingerprint'),
        'calibrated': False}, 'boxes': {str(k): list(v) for k, v in union.items()}}
    return replace(primary, visible=visible or None, grounding=None, candidates=candidates,
                   categories=tuple(sorted(categories.items())), cards=tuple(sorted(cards.items())), audit=audit)


class SoftCandidatePreference:
    """Bounded potential differences over a native action trie.

    Only object decisions within a single skill are scored. Potential differences
    telescope over multi-token IDs, avoiding a bonus for longer token spellings.
    The native DFA remains the only source of hard legality.
    """
    def __init__(self, dfa, candidates, weight=0.5):
        if not math.isfinite(weight) or not 0 <= weight <= 2:
            raise ValueError('perception preference weight must be finite and in [0, 2]')
        self.dfa, self.weight = dfa, float(weight)
        self.candidates = {(skill, role): frozenset(ids) for skill, role, ids in candidates.restrictions}
        self.stats = {'scored_steps': 0, 'changed': 0}

    def apply(self, result, state):
        if not self.weight or state[2] != 'operation':
            return
        dfa = self.dfa
        current = dfa.node_actions[state[3]]
        skills = {dfa.actions[a][0] for a in current}
        if len(skills) != 1:
            return  # never reward choosing an easier skill instead of the requested one
        skill = next(iter(skills))
        roles = dfa.skill_arguments[skill]
        targets = self.candidates.get((skill, roles[0])) if len(roles) == 1 else None
        if not targets:
            return
        def potential(actions):
            # Mean support supplies a positive increment to a supported branch,
            # rather than only penalizing alternatives and favoring whitespace.
            return sum(dfa.actions[a][1] in targets for a in actions) / len(actions)
        initial = potential(current)
        before = int(result[0].argmax())
        scored = False
        for key, labels in dfa.effect_groups(state).items():
            actions = (dfa.node_actions[key[1]] if key[0] == 'action' else
                       (key[1],) if key[0] == 'done' else ())
            if not actions:
                continue
            delta = self.weight * (potential(actions) - initial)
            if delta:
                # Adding a finite value preserves every -inf hard mask.
                result[0, list(labels)] += delta
                scored = True
        self.stats['scored_steps'] += int(scored)
        self.stats['changed'] += int(int(result[0].argmax()) != before)
