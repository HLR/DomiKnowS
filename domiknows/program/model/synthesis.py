"""R4 Phase A — model synthesis from the declared graph.

Two pieces, deliberately split by how much they change:

* :func:`synthesize_model` **builds** a shared trunk plus one joint head per
  concept group, so a constraint gradient on any concept reaches parameters
  every other concept reads. For an ``EnumConcept`` group the head is a single
  K-way softmax, which *cannot* violate the group's exclusivity — the constraint
  becomes architecture instead of a penalty.

* :func:`analyze_exclusivity` is **advisory only**. It reports which binary
  sibling groups are provably mutually exclusive and prints the exact
  ``EnumConcept`` declaration to switch to. It never rewrites the graph:
  silently turning declared constraints into architecture would change what a
  graph *means*, so the user makes the edit.

The payoff of the advice — exclusivity constraints becoming impossible to
violate, leaving the loss and needing no dual — is real; keeping it advisory is
what makes it safe.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional

import torch
from torch import nn

from domiknows.graph.concept import Concept, EnumConcept


# --------------------------------------------------------------------------- #
# Graph walking helpers
# --------------------------------------------------------------------------- #

def _all_concepts(graph) -> List[Concept]:
    """Every concept in ``graph`` and its subgraphs, in declaration order."""
    seen = {}
    stack = [graph]
    while stack:
        g = stack.pop()
        for c in getattr(g, 'concepts', {}).values():
            seen.setdefault(id(c), c)
        stack.extend(getattr(g, 'subgraphs', {}).values())
    return list(seen.values())


def _all_constraints(graph):
    """``(name, lc)`` for every head logical constraint, recursively."""
    rec = getattr(graph, 'logicalConstrainsRecursive', None)
    if rec is not None:
        yield from rec
    else:  # pragma: no cover - defensive
        yield from getattr(graph, 'logicalConstrains', {}).items()


def _is_a_parent(concept):
    """The concept this one ``is_a``, or None."""
    rels = getattr(concept, '_out', {}).get('is_a')
    if rels:
        return rels[0].dst
    return None


def _concept_operands(lc):
    """Concept tuples ``(Concept, name, class_index, card)`` referenced by ``lc``."""
    out = []
    for el in getattr(lc, 'e', []):
        if isinstance(el, tuple) and el and isinstance(el[0], Concept):
            out.append(el)
    return out


# --------------------------------------------------------------------------- #
# Advisory exclusivity analysis
# --------------------------------------------------------------------------- #

@dataclass
class GroupAdvice:
    parent: str
    members: List[str]
    complete: bool                       # every pair among members is excluded
    redundant_constraints: List[str]     # constraints subsumed by a joint head
    already_enum: bool = False

    def suggestion(self) -> str:
        values = ', '.join(f"'{m}'" for m in self.members)
        return (f"{self.parent}(name='{self.parent}_label', "
                f"ConceptClass=EnumConcept, values=[{values}])")


@dataclass
class ExclusivityReport:
    groups: List[GroupAdvice] = field(default_factory=list)

    def render(self) -> str:
        if not self.groups:
            return 'analyze_exclusivity: no mutually-exclusive binary groups found.'
        lines = ['analyze_exclusivity — advisory (no graph is modified):']
        for g in self.groups:
            if g.already_enum:
                lines.append(f'  [{g.parent}] already an EnumConcept joint head '
                             f'({len(g.members)} values) — exclusivity is architectural.')
                continue
            status = 'complete' if g.complete else 'PARTIAL (not every pair excluded)'
            lines.append(f'  [{g.parent}] {len(g.members)} mutually-exclusive '
                         f'binary siblings — {status}:')
            lines.append(f'      members: {", ".join(g.members)}')
            if g.redundant_constraints:
                lines.append('      constraints made redundant by a joint softmax head: '
                             + ', '.join(g.redundant_constraints))
            lines.append('      switch to a single joint head with:')
            lines.append(f'          {g.suggestion()}')
        return '\n'.join(lines)


def analyze_exclusivity(graph) -> ExclusivityReport:
    """Report binary sibling groups that a joint head would make exclusive.

    Detects two exclusivity encodings:

    * pairwise ``nandL`` over binary siblings, and
    * ``atMostL(..., 1)`` / ``exactL(..., 1)`` over a sibling set,

    grouped by the shared ``is_a`` parent. Already-``EnumConcept`` groups are
    reported too, as confirmation that their exclusivity is already structural.
    """
    from domiknows.graph.logicalConstrain import nandL, atMostL, exactL

    # parent id -> {'parent': concept, 'pairs': {frozenset: [lcName]},
    #               'covered': {frozenset(members): lcName}}
    parents: Dict[int, dict] = {}

    def bucket(parent):
        return parents.setdefault(id(parent), {
            'parent': parent, 'pairs': {}, 'covers': []})

    for name, lc in _all_constraints(graph):
        operands = _concept_operands(lc)
        # Binary siblings only: class_index is None and there is an is_a parent.
        # NB: Concept is a mapping subclass, so an empty concept is *falsy* —
        # every parent test must be ``is not None``, never a truthiness check.
        binary = [op for op in operands
                  if op[2] is None and _is_a_parent(op[0]) is not None]
        if len(binary) < 2:
            continue
        # All operands must share one parent to be one exclusivity group.
        parent = _is_a_parent(binary[0][0])
        if any(_is_a_parent(op[0]) is not parent for op in binary):
            continue

        lcName = getattr(lc, 'lcName', name)
        if type(lc) is nandL and len(binary) == 2:
            key = frozenset(op[0].name for op in binary)
            bucket(parent)['pairs'].setdefault(key, []).append(lcName)
        elif type(lc) in (atMostL, exactL) and _limit_is_one(lc):
            members = frozenset(op[0].name for op in binary)
            bucket(parent)['covers'].append((members, lcName))

    report = ExclusivityReport()

    # EnumConcept groups: already joint.
    for c in _all_concepts(graph):
        if isinstance(c, EnumConcept):
            enum_parent = _is_a_parent(c)
            report.groups.append(GroupAdvice(
                parent=(enum_parent.name if enum_parent is not None else c.name),
                members=list(c.enum), complete=True,
                redundant_constraints=[], already_enum=True))

    for info in parents.values():
        parent = info['parent']
        members: List[str] = []
        redundant: List[str] = []

        for key, lcNames in info['pairs'].items():
            for m in key:
                if m not in members:
                    members.append(m)
            redundant.extend(lcNames)
        for cover_members, lcName in info['covers']:
            for m in cover_members:
                if m not in members:
                    members.append(m)
            redundant.append(lcName)

        if len(members) < 2:
            continue

        # Complete iff every pair among the members is excluded (a clique) — the
        # condition under which a single joint head loses nothing.
        needed = set(frozenset(p) for p in combinations(members, 2))
        have = set(info['pairs'].keys())
        covered_by_atmost = any(set(members) <= set(cm) for cm, _ in info['covers'])
        complete = covered_by_atmost or needed <= have

        report.groups.append(GroupAdvice(
            parent=parent.name, members=members, complete=complete,
            redundant_constraints=sorted(set(redundant))))

    return report


def _limit_is_one(lc) -> bool:
    # Counting constraints (atMostL/exactL) store the limit as an int argument in
    # ``e``; ``lc.p`` is the constraint *priority*, not a limit, so it must not be
    # consulted here. Read the last int element of ``e``.
    for el in reversed(getattr(lc, 'e', [])):
        if isinstance(el, bool):
            continue
        if isinstance(el, int):
            return el == 1
    return False


# --------------------------------------------------------------------------- #
# Model synthesis
# --------------------------------------------------------------------------- #

class _GroupModule(nn.Module):
    """Trunk → one group's head → softmax, sharing the trunk's parameters.

    Usable directly as a ``ModuleLearner(module=...)`` so the synthesized
    architecture drops into the existing sensor wiring.
    """

    def __init__(self, trunk: nn.Module, head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x):
        return torch.softmax(self.head(self.trunk(x)), dim=-1)


class SynthesizedModel(nn.Module):
    """Shared trunk + one joint softmax head per concept group.

    :param groups: ``{group_name: num_classes}`` — one head each. Built from the
        graph's ``EnumConcept`` groups by :func:`synthesize_model`.
    """

    def __init__(self, input_dim: int, groups: Dict[str, int],
                 hidden_dim: Optional[int] = None, trunk: Optional[nn.Module] = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.trunk = trunk if trunk is not None else nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.group_k = dict(groups)
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, k) for name, k in groups.items()})

    def forward(self, x) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {name: torch.softmax(head(h), dim=-1)
                for name, head in self.heads.items()}

    def group_module(self, name: str) -> nn.Module:
        """A standalone module for group ``name`` sharing the trunk.

        Wire it in with, e.g.::

            object_node[material] = ModuleLearner('emb', module=model.group_module('material'))
        """
        return _GroupModule(self.trunk, self.heads[name])


def synthesize_model(graph, input_dim: int, hidden_dim: Optional[int] = None,
                     trunk: Optional[nn.Module] = None,
                     groups: Optional[Dict[str, int]] = None) -> SynthesizedModel:
    """Build a shared-trunk model with one joint head per ``EnumConcept`` group.

    Only ``EnumConcept`` groups are synthesized — those are the groups the graph
    *already* declares as joint, so no constraint semantics are invented. Use
    :func:`analyze_exclusivity` first to learn which binary sibling groups to
    convert to ``EnumConcept`` before calling this.
    """
    if groups is None:
        groups = {}
        for c in _all_concepts(graph):
            if isinstance(c, EnumConcept):
                groups[c.name] = len(c.enum)
    if not groups:
        raise ValueError(
            'synthesize_model found no EnumConcept groups to build heads for; '
            'pass groups=... explicitly, or convert exclusive binary siblings to '
            'an EnumConcept first (see analyze_exclusivity).')
    return SynthesizedModel(input_dim, groups, hidden_dim=hidden_dim, trunk=trunk)
