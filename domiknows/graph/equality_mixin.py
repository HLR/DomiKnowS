# domiknows/graph/equality_mixin.py

"""
Equality mixin and an opt-in applier for Concept-like classes.

Usage (no auto-apply to avoid circular imports):
    from domiknows.graph.equality_mixin import apply_equality_mixin
    apply_equality_mixin(Concept)
    apply_equality_mixin(EnumConcept)  # optional
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING


class EqualityMixin:
    def get_equal_concepts(self, transitive: bool = False) -> List['Concept']:
        from .relation import Equal

        equal_concepts = []
        if 'equal' in self._out:
            for rel in self._out['equal']:
                if isinstance(rel, Equal):
                    equal_concepts.append(rel.dst)
        if 'equal' in self._in:
            for rel in self._in['equal']:
                if isinstance(rel, Equal):
                    equal_concepts.append(rel.src)

        equal_concepts = list(set(equal_concepts))
        if transitive:
            return self._get_equal_transitive_closure(equal_concepts)
        
        if not equal_concepts:
            return []
        
        return equal_concepts

    def _get_equal_transitive_closure(self, direct_equals: List['Concept']) -> List['Concept']:
        visited = {self}
        to_visit = set(direct_equals)
        equivalence_class = []

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            equivalence_class.append(current)
            for eq in current.get_equal_concepts(transitive=False):
                if eq not in visited:
                    to_visit.add(eq)
        return equivalence_class

    def is_equal_to(self, other_concept: 'Concept') -> bool:
        from .relation import Equal
        equal_concepts = self.get_equal_concepts(transitive=False)
        if other_concept in equal_concepts:
            return True
        relations = self.relate_to(other_concept, Equal)
        return len(relations) > 0

    def is_equal_to_transitive(self, other_concept: 'Concept') -> bool:
        equal_concepts = self.get_equal_concepts(transitive=True)
        return other_concept in equal_concepts

    def get_equivalence_class(self) -> List['Concept']:
        transitive_equals = self.get_equal_concepts(transitive=True)
        return [self] + transitive_equals

    def get_canonical_concept(self) -> 'Concept':
        equivalence_class = self.get_equivalence_class()
        return sorted(equivalence_class, key=lambda c: c.name)[0]

    def merge_equal_concepts(self, property_merge_strategy: str = 'first') -> Dict[str, Any]:
        equal_concepts = self.get_equal_concepts(transitive=True)
        merged_properties = {}
        for concept in [self] + equal_concepts:
            for prop_name, prop_values in concept.items():
                merged_properties.setdefault(prop_name, []).extend(prop_values)

        result = {}
        for prop_name, values in merged_properties.items():
            if not values:
                continue
            if property_merge_strategy == 'first':
                result[prop_name] = values[0]
            elif property_merge_strategy == 'last':
                result[prop_name] = values[-1]
            elif property_merge_strategy == 'all':
                result[prop_name] = values
            else:
                result[prop_name] = values[0]
        return result

    def get_equal_relations(self) -> List['Equal']:
        from .relation import Equal
        relations = []
        if 'equal' in self._out:
            relations.extend([rel for rel in self._out['equal'] if isinstance(rel, Equal)])
        if 'equal' in self._in:
            relations.extend([rel for rel in self._in['equal'] if isinstance(rel, Equal)])
        return relations


def apply_equality_mixin(cls: type) -> None:
    """Apply EqualityMixin methods to the given class (e.g., Concept)."""
    if hasattr(cls, "get_equal_concepts"):
        return

    for name in dir(EqualityMixin):
        if name.startswith("__") and name != "__doc__":
            continue
        if name == "_get_equal_transitive_closure" or not name.startswith("_"):
            attr = getattr(EqualityMixin, name)
            if callable(attr):
                setattr(cls, name, attr)


__all__ = ["EqualityMixin", "apply_equality_mixin"]