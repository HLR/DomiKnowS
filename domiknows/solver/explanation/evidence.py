"""Opt-in grounding evidence shared by the interpreter and compiled evaluator."""
from contextvars import ContextVar
from functools import wraps
import math
from numbers import Integral, Real

import torch


collector = ContextVar('domiknows_explanation_collector', default=None)


def native(value):
    """Convert evidence to strict JSON values, never arbitrary object reprs."""
    if torch.is_tensor(value):
        return native(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    if hasattr(value, 'VarName'):
        return {'solver_variable': value.VarName}
    if hasattr(value, 'variable_key'):
        return {'leaf': native(value.variable_key), 'value_index': value.value_index,
                'probability': native(value.probability)}
    if hasattr(value, 'node_id'):
        return {'circuit_node': value.node_id}
    if hasattr(value, 'is_literal') and hasattr(value, 'id'):
        return {'circuit_node': value.id}
    return {'unavailable_type': type(value).__name__}


def graph_path(concept):
    graph = concept.getOntologyGraph()
    names = []
    while graph is not None:
        names.append(graph.name)
        graph = getattr(graph, 'sup', None)
    return '/'.join(reversed(names)) + '/' + concept.name


class EvidenceCollector:
    def __init__(self, sample, limit, observer=None):
        self.sample = sample
        self.limit = limit
        self.records = []
        self.omitted = 0
        self.phase = 'inference'
        self.row_sources = {}
        self.observer = observer
        self.total = 0

    def entity(self, dn):
        return {'sample': self.sample, 'path': graph_path(dn.ontologyNode),
                'instance_id': native(dn.instanceID)}

    def add(self, **record):
        record = native(record)
        record.update(id=f's{self.sample}:e{self.total}', phase=self.phase)
        self.total += 1
        if self.observer is not None:
            self.observer(record)
        if len(self.records) >= self.limit:
            self.omitted += 1
            return None
        self.records.append(record)
        return record['id']

    def evaluation(self, lc, bindings, operands, output, *, head, compiled, grounding=None):
        if len(self.records) >= self.limit and self.observer is None:
            self.omitted += 1
            return []
        bound = {}
        for name, groups in (bindings or {}).items():
            bound[name] = [[self.entity(dn) if hasattr(dn, 'ontologyNode') else None
                            for dn in group] for group in groups]
        common = dict(kind='evaluation', rule=lc.lcName, expression=lc.strEs(),
                      rule_path=lc.graph.fullname + '/' + lc.lcName,
                      concept_names=sorted(lc.getLcConcepts()),
                      operator=type(lc).__name__, head=head, compiled=compiled,
                      priority=getattr(lc, 'p', None))
        count = 0
        if grounding is not None:
            names, grid = grounding
            grid = grid.detach().cpu().tolist() if torch.is_tensor(grid) else list(grid)
            count = len(grid)
        def output_rows(value):
            if torch.is_tensor(value):
                return value.numel()
            if isinstance(value, (list, tuple)):
                if len(value) == 1 and isinstance(value[0], (list, tuple)) and value[0]:
                    if torch.is_tensor(value[0][0]):
                        return value[0][0].numel()
                return len(value)
            return 1
        def row(value, index):
            if torch.is_tensor(value):
                return value.reshape(-1)[index] if value.numel() == count else value
            if isinstance(value, dict):
                return {k: row(v, index) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                if len(value) == count and count > 1:
                    return value[index]
                return [row(v, index) for v in value]
            return value
        # Aggregators retain the complete domain; per-row results use the
        # executor's actual joined index grid, never an invented Cartesian join.
        if count and output_rows(output) == count:
            identifiers = []
            for index, coordinates in enumerate(grid):
                if len(self.records) >= self.limit and self.observer is None:
                    self.omitted += count - index
                    break
                selected = {}
                for name, coordinate in zip(names, coordinates):
                    candidates = bound.get(name)
                    if candidates is not None and 0 <= int(coordinate) < len(candidates):
                        selected[name] = [candidates[int(coordinate)]]
                identifier = self.add(**common, bindings=selected,
                    binding_kind='grounding', grounding_index=index,
                    inputs=row(operands, index), result=row(output, index))
                if identifier is not None:
                    identifiers.append(identifier)
            return identifiers
        identifier = self.add(**common, bindings=bound, binding_kind='candidate_domain',
                              inputs=operands, result=output)
        return [identifier] if identifier is not None else []


def begin_operator(model):
    active = collector.get()
    if active is None or model is None:
        return None
    model.update()
    return model.NumConstrs, model.NumGenConstrs


def record_evaluation(lc, bindings, operands, output, *, head=False, compiled=False,
                      model=None, row_start=None, grounding=None):
    active = collector.get()
    if active is not None:
        identifiers = active.evaluation(lc, bindings, operands, output, head=head,
                                        compiled=compiled, grounding=grounding)
        if model is not None and row_start is not None:
            model.update()
            for kind, rows, offset in (('linear', model.getConstrs(), row_start[0]),
                                       ('general', model.getGenConstrs(), row_start[1])):
                for index in range(offset, len(rows)):
                    source = active.row_sources.setdefault((id(model), kind, index), {})
                    if identifiers:
                        source.setdefault('operator_evidence_ids', []).extend(identifiers)


def trace_leaf(function):
    """Record the actual interpreter input, including fixedL substitutions."""
    @wraps(function)
    def traced(self, dn, xPkey, e, p, *args, **kwargs):
        result = function(self, dn, xPkey, e, p, *args, **kwargs)
        active = collector.get()
        if active is not None:
            concept = kwargs.get('concept')
            active.add(kind='leaf', entity=active.entity(dn),
                       concept=graph_path(concept) if concept is not None else e[0],
                       class_index=e[1], input_key=xPkey, value=result)
        return result
    return traced


def trace_gather(function):
    """Unpack batched fuzzy inputs without changing the compiled gather."""
    @wraps(function)
    def traced(self, groups, element):
        result = function(self, groups, element)
        active = collector.get()
        if active is None:
            return result
        concept = element[0]
        indices = (list(range(len(concept.enum))) if hasattr(concept, 'enum') and element[2] is None
                   else [element[2] if hasattr(concept, 'enum') else 1])
        packed = (len(result) == 1 and result[0] and
                  all(torch.is_tensor(column) for column in result[0]))
        position = 0
        for group_index, group in enumerate(groups):
            for node_index, dn in enumerate(group):
                if dn is None:
                    continue
                for column_index, class_index in enumerate(indices):
                    try:
                        value = (result[0][column_index].reshape(-1)[position] if packed else
                                 result[group_index][node_index * len(indices) + column_index])
                    except (IndexError, TypeError):
                        value = None
                    active.add(kind='leaf', entity=active.entity(dn), concept=graph_path(concept),
                               class_index=class_index, input_key=self.key, value=value)
                position += 1
        return result
    return traced
