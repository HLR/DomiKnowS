"""Content snapshots: hints cannot hide in-place or untracked graph edits."""
from dataclasses import dataclass
import hashlib
import re

import torch

from domiknows.graph.concept import Concept, EnumConcept
from domiknows.graph.logicalConstrain import LcElement
from domiknows.graph import CandidateSelection
from domiknows.solver.explanation.session import _nodes, _graphs


def signature(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        content = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return ('tensor', tuple(tensor.shape), str(tensor.dtype), str(value.device),
                hashlib.sha256(content).hexdigest(),
                hashlib.sha256(torch.isfinite(tensor).numpy().tobytes()).hexdigest())
    if isinstance(value, Concept):
        return ('concept', value.fullname, id(value))
    if isinstance(value, CandidateSelection):
        return ('opaque', type(value).__name__, id(value), signature(value.e))
    if isinstance(value, LcElement):
        return (type(value).__name__, signature(value.e))
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, repr(value))
    if isinstance(value, dict):
        return tuple((signature(k), signature(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(signature(v) for v in value))
    if isinstance(value, (set, frozenset)):
        return ('set', tuple(sorted((signature(v) for v in value), key=repr)))
    if hasattr(value, 'ontologyNode'):
        return ('node', value.id, id(value), value.ontologyNode.fullname, signature(value.instanceID))
    if hasattr(value, 'tobytes') and hasattr(value, 'shape'):
        return ('array', tuple(value.shape), str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest())
    return ('opaque', type(value).__name__, id(value))


def _opaque(value):
    if isinstance(value, tuple):
        return bool(value and value[0] == 'opaque') or any(_opaque(v) for v in value)
    return False


def output_key(key):
    return ('/ILP' in key or '/argmax' in key or key.endswith('/answer')
            or key.endswith('/probability') or key in ('rootDataNode',))


@dataclass
class Snapshot:
    entities: tuple
    edges: tuple
    schema: tuple
    rules: dict
    predictions: dict
    attributes: dict
    opaque: bool

    @classmethod
    def read(cls, root, concepts=()):
        nodes, graphs = _nodes(root), _graphs(root, concepts)
        entities = tuple((n.id, id(n), n.ontologyNode.fullname, signature(n.instanceID),
                          str(n.current_device), str(getattr(n, 'current_dtype', None))) for n in nodes)
        edges = tuple((n.id, direction, name, tuple(target.id for target in targets))
            for n in nodes for direction, links in (('out', n.relationLinks), ('in', n.impactLinks))
            for name, targets in links.items())
        schema = tuple((id(g), signature(g.ontology), tuple((id(c), c.fullname,
            tuple(c.enum) if isinstance(c, EnumConcept) else None,
            tuple((kind, tuple((rel.name, rel.dst.fullname) for rel in rels))
                  for kind, rels in c._out.items() if rels)) for c in g.concepts.values())) for g in graphs)
        rules = {}
        for graph in graphs:
            for name, lc in graph.logicalConstrains.items():
                rules[(graph.fullname, name)] = (id(lc), type(lc).__name__, signature(lc.e),
                    lc.headLC, lc.active, lc.p, getattr(lc, '_compile_revision', 0))
            for name, lc in graph.executableLCs.items():
                rules[(graph.fullname, name)] = (id(lc), signature(lc.innerLC.e), lc.active,
                                               getattr(lc.innerLC, '_compile_revision', 0))
        predictions, attributes = {}, {}
        for node in nodes:
            for key, value in node.attributes.items():
                if key == 'variableSet' or key == 'propertySet':
                    values = value.items()
                else:
                    values = [(key, value)]
                for name, item in values:
                    if not isinstance(name, str) or output_key(name):
                        continue
                    match = re.search(r'<([^>]+)>', name)
                    target = predictions if match and '/label' not in name else attributes
                    target[(node.id, name)] = signature(item)
        opaque = any(_opaque(v) for v in (*rules.values(), *predictions.values(), *attributes.values()))
        return cls(entities, edges, schema, rules, predictions, attributes, opaque)

    def diff(self, old, hints=None):
        if hints is not None and (not isinstance(hints, dict) or
                set(hints) - {'nodes', 'concepts', 'relations', 'rules'}):
            raise ValueError('changes accepts nodes, concepts, relations, and rules')
        if old is None:
            return {'initial': True, 'entities': True, 'relations': [], 'schema': True,
                    'rules': list(self.rules), 'concepts': [], 'attributes': True, 'opaque': self.opaque}
        prediction_keys = {key for key in self.predictions.keys() | old.predictions.keys()
                           if self.predictions.get(key) != old.predictions.get(key)}
        concepts = {re.search(r'<([^>]+)>', key[1]).group(1) for key in prediction_keys}
        def layout(value):
            return (value[:4], value[5]) if value and value[0] == 'tensor' else value
        input_layout = any(layout(self.predictions.get(key)) != layout(old.predictions.get(key))
                           for key in prediction_keys
                           if self.entities == old.entities or
                           key in self.predictions and key in old.predictions)
        changed_edges = set(self.edges) ^ set(old.edges)
        changed = {'initial': False, 'entities': self.entities != old.entities,
            'relations': sorted({edge[2] for edge in changed_edges}), 'schema': self.schema != old.schema,
            'rules': [key for key in self.rules.keys() | old.rules.keys()
                      if self.rules.get(key) != old.rules.get(key)],
            'concepts': sorted(concepts), 'attributes': self.attributes != old.attributes,
            'opaque': self.opaque}
        changed['input_layout'] = input_layout
        if hints:
            if not isinstance(hints, dict) or set(hints) - {'nodes', 'concepts', 'relations', 'rules'}:
                raise ValueError('changes accepts nodes, concepts, relations, and rules')
            changed['concepts'] = sorted(set(changed['concepts']) |
                {getattr(c, 'name', c) for c in hints.get('concepts', ())})
            changed['relations'] = sorted(set(changed['relations']) | set(hints.get('relations', ())))
            if hints.get('nodes'):
                changed['attributes'] = True
            changed['hinted_rules'] = [getattr(r, 'lcName', r) for r in hints.get('rules', ())]
        return changed


def has_changes(change):
    return any(change.get(key) for key in ('initial', 'entities', 'relations', 'schema', 'rules',
                                         'concepts', 'attributes', 'opaque', 'hinted_rules'))


def legacy_reuse_signature(root, concepts, extra=(), ignore_pins=False, key=('local', 'softmax')):
    """Never reuse a model solely because another sample has the same size."""
    snapshot = Snapshot.read(root, concepts)
    availability = []
    for spec in concepts:
        for node in root.findDatanodes(select=root.findRootConceptOrRelation(spec[0])):
            value = node.getAttribute(spec[0], *key)
            availability.append((spec[0].fullname, node.id,
                (tuple(value.shape), str(value.dtype), str(value.device),
                 signature(torch.isnan(value))) if torch.is_tensor(value) else signature(value)))
    return (id(root), snapshot.entities, snapshot.edges, snapshot.schema,
            tuple(snapshot.rules.items()), tuple(snapshot.attributes.items()),
            tuple((c[0].fullname, c[1:]) for c in concepts), signature(extra), ignore_pins,
            tuple(availability), object() if snapshot.opaque else None)
