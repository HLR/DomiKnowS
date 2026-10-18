"""Private sample/solver sessions for explanation and counterfactual execution."""
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
import copy
import json
import threading
from time import perf_counter

import torch

from domiknows.graph.concept import EnumConcept
from domiknows.graph.dataNode import DataNode
from domiknows.solver.answerModule import AnswerSolver
from domiknows.solver.executableInference import AdHocExecutableQueries, ExecutableInference
from domiknows.solver.gurobiILPOntSolver import gurobiILPOntSolver
from domiknows.solver.ilpConfig import ilpConfig
from .evidence import EvidenceCollector, collector, graph_path, native
from .report import ExplanationReport
from domiknows.solver.bdd import CircuitSizeLimitExceeded


_lock = threading.RLock()


def _containers(value, memo=None):
    """Copy containers and tensor storage while preserving ontology identity."""
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if hasattr(value, 'fullname') or hasattr(value, 'lcName'):
        return value
    if torch.is_tensor(value):
        result = value.detach().clone()
        memo[id(value)] = result
        return result
    if isinstance(value, dict):
        result = value.copy()
        result.clear()
        memo[id(value)] = result
        result.update((k, _containers(v, memo)) for k, v in value.items())
        if hasattr(value, 'revision'):
            result.revision = value.revision
        return result
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_containers(v, memo) for v in value)
        return result
    if isinstance(value, tuple):
        return tuple(_containers(v, memo) for v in value)
    if isinstance(value, set):
        return {_containers(v, memo) for v in value}
    return value


def _nodes(root):
    seen, result = set(), []
    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        result.append(node)
        for links in (node.relationLinks, node.impactLinks):
            for targets in links.values():
                for target in targets:
                    if isinstance(target, DataNode):
                        visit(target)
    visit(root)
    return result


def _clone(root):
    nodes = _nodes(root)
    for node in list(nodes):
        builder = node.myBuilder
        if builder is not None:
            for extra in builder.get('allDns', ()):
                if isinstance(extra, DataNode) and not any(extra is n for n in nodes):
                    nodes.extend(n for n in _nodes(extra) if not any(n is old for old in nodes))
    memo = {id(node): copy.copy(node) for node in nodes}
    builders = {id(node.myBuilder): node.myBuilder for node in nodes if node.myBuilder is not None}
    for identifier, builder in builders.items():
        memo[identifier] = dict.__new__(type(builder))
    for identifier, builder in builders.items():
        duplicate = memo[identifier]
        duplicate.__dict__ = _containers(builder.__dict__, memo)
        for k, v in builder.items():
            dict.__setitem__(duplicate, k, _containers(v, memo))
    for node in nodes:
        duplicate = memo[id(node)]
        duplicate.__dict__ = _containers(node.__dict__, memo)
        duplicate.gurobiModel = None
        duplicate.myBuilder = memo.get(id(node.myBuilder))
        duplicate.attributes = {k: v for k, v in duplicate.attributes.items()
                                if '/ILP/x' not in k and '/ILP/notx' not in k}
    return memo[id(root)]


def _graphs(root, concepts=()):
    result, seen = [], set()
    def visit(graph):
        if graph is None:
            return
        if id(graph) in seen:
            return
        seen.add(id(graph))
        result.append(graph)
        for subgraph in getattr(graph, 'subgraphs', {}).values():
            visit(subgraph)
    visit(root.graph)
    for node in _nodes(root):
        visit(node.ontologyNode.getOntologyGraph())
    for specification in concepts:
        concept = specification[0] if isinstance(specification, tuple) else specification
        if isinstance(concept, str):
            concept = root.graph.findConcept(concept)
        if concept is not None:
            visit(concept.getOntologyGraph())
    return result


@contextmanager
def _ontology_transaction(graphs):
    """Restore registries and flags that the existing DSL temporarily changes."""
    objects, seen, containers, container_seen = [], set(), [], set()
    def save_container(value):
        if id(value) in container_seen:
            return
        if hasattr(value, 'fullname') or hasattr(value, 'lcName'):
            return
        container_seen.add(id(value))
        if isinstance(value, dict):
            containers.append((value, list(value.items()), getattr(value, 'revision', None)))
            for child in value.values():
                save_container(child)
        elif isinstance(value, list):
            containers.append((value, list(value), None))
            for child in value:
                save_container(child)
        elif isinstance(value, tuple):
            for child in value:
                save_container(child)
        elif isinstance(value, set):
            containers.append((value, set(value), None))
    def add(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        objects.append((obj, dict(obj.__dict__)))
        for value in obj.__dict__.values():
            save_container(value)
        for element in obj.__dict__.get('e', ()):
            if hasattr(element, 'lcName'):
                add(element)
        inner = obj.__dict__.get('innerLC')
        if inner is not None:
            add(inner)
    for graph in graphs:
        add(graph)
        save_container(graph.varNameReversedMap)
        for concept in graph.concepts.values():
            add(concept)
        for lc in list(graph.logicalConstrains.values()) + list(graph.executableLCs.values()):
            add(lc)
    old_collected = DataNode.collectedConceptsAndRelations
    old_find_cache = DataNode._findCache
    old_concepts_map = DataNode.conceptsMap
    DataNode._findCache = None
    DataNode.conceptsMap = dict(old_concepts_map)
    try:
        yield
    finally:
        for value, state, revision in reversed(containers):
            if isinstance(value, dict):
                base = OrderedDict if isinstance(value, OrderedDict) else dict
                base.clear(value)
                for k, v in state:
                    base.__setitem__(value, k, v)
                if revision is not None:
                    value.revision = revision
            elif isinstance(value, list):
                value[:] = state
            else:
                value.clear()
                value.update(state)
        for obj, state in reversed(objects):
            obj.__dict__.clear()
            obj.__dict__.update(state)
        DataNode.collectedConceptsAndRelations = old_collected
        DataNode._findCache = old_find_cache
        DataNode.conceptsMap = old_concepts_map


class UndefinedEvidence(ValueError):
    def __init__(self, impossible):
        self.impossible = impossible
        super().__init__('impossible evidence' if impossible else 'zero-probability evidence')


class DiagnosticAnswerSolver(AnswerSolver):
    """Label candidate models without changing ordinary hypothesis selection."""
    def _hypothesis_spec(self, executable, dn, concepts_relations):
        values, build = super()._hypothesis_spec(executable, dn, concepts_relations)
        name = next(name for name, candidate in dn.graph.executableLCs.items()
                    if candidate is executable)
        def annotated(value):
            constraint = build(value)
            constraint._explanation_executable_name = name
            constraint._explanation_hypothesis_value = value
            return constraint
        return values, annotated


class DiagnosticSession:
    def __init__(self, root, concepts, mode, queries, options, sample):
        self.root = _clone(root)
        self.concepts = concepts
        self.mode = mode
        self.queries = queries
        self.options = dict(options)
        self.sample = sample
        self.graphs = _graphs(root, concepts)
        self.fingerprint = self._fingerprint()

    def _fingerprint(self):
        return tuple((id(g), getattr(g, 'constraint_revision', None),
                      tuple((n, id(lc), lc.active, lc.headLC, lc.p, lc.strEs())
                            for n, lc in g.logicalConstrains.items()),
                      tuple((n, id(lc), lc.active, lc.innerLC.strEs())
                            for n, lc in g.executableLCs.items()),
                      tuple((n, tuple(c.enum) if isinstance(c, EnumConcept) else None,
                             tuple((kind, tuple((rel.name, id(rel.dst)) for rel in relations))
                                   for kind, relations in c._out.items()))
                            for n, c in g.concepts.items())) for g in self.graphs)

    def counterfactual(self, report, assignments):
        if not isinstance(assignments, Mapping):
            raise TypeError('assignments must map prediction IDs to labels')
        combined = dict(report._payload.get('assignments', {}))
        combined.update(assignments)
        with _lock:
            if self._fingerprint() != self.fingerprint:
                raise ValueError('Graph changed since this report; create a new explanation')
            result = self.run(combined)
        baseline, changed = report.to_dict(), result.to_dict()
        comparison = {'semantics': {'ilp': 'forced-label re-solve',
                                   'circuit': 'conditioning',
                                   'tnorm': 'input replacement'}[self.mode]}
        old = {p['id']: p for p in baseline['predictions']}
        comparison['changed_labels'] = [p['id'] for p in changed['predictions']
            if p.get('inferred_label', p['local_label']) !=
               old[p['id']].get('inferred_label', old[p['id']]['local_label'])]
        if self.mode == 'ilp' and baseline['status'] == changed['status'] == 'optimal':
            comparison['objective_difference'] = changed['objective']['value'] - baseline['objective']['value']
        if self.mode != 'ilp':
            field = 'probability' if self.mode == 'circuit' else 'score'
            comparison[f'{field}_differences'] = {
                name: answer[field] - baseline['answers'][name][field]
                for name, answer in changed['answers'].items()
                if name in baseline['answers'] and isinstance(answer.get(field), (int, float))
                and isinstance(baseline['answers'][name].get(field), (int, float))}
            comparison['changed_answers'] = {
                name: {'before': baseline['answers'].get(name), 'after': answer}
                for name, answer in changed['answers'].items()
                if answer != baseline['answers'].get(name)}
        if self.mode == 'circuit':
            comparison.pop('changed_labels')
            comparison['conditioned_inputs'] = combined
        result._payload['comparison'] = comparison
        return result

    def _predictions(self, dn, concepts, key):
        entries, private = [], {}
        seen = set()
        for concept_spec in concepts:
            concept = concept_spec[0] if isinstance(concept_spec, tuple) else concept_spec
            if id(concept) in seen:
                continue
            seen.add(id(concept))
            root = dn.findRootConceptOrRelation(concept)
            for node in dn.findDatanodes(select=root):
                raw = node.getAttribute(concept, *key)
                if raw is None:
                    continue
                values = raw.detach().reshape(-1)
                if values.numel() < 2:
                    continue
                identity = {'sample': self.sample, 'path': graph_path(concept),
                            'instance_id': native(node.instanceID)}
                identifier = json.dumps(identity, sort_keys=True, separators=(',', ':'))
                if identifier in private:
                    raise ValueError('Non-unique grounded prediction identifier')
                enum = isinstance(concept, EnumConcept)
                label = concept.enum[int(values.argmax())] if enum else bool(values.argmax().item())
                entry = {'id': identifier, **identity, 'class_names': list(concept.enum) if enum else None,
                         'entity_path': graph_path(node.ontologyNode),
                         'probabilities': native(values), 'local_label': label}
                entries.append(entry)
                private[identifier] = (node, concept, entry)
        return entries, private

    def run(self, assignments=None):
        assignments = dict(assignments or {})
        started = perf_counter()
        evidence = EvidenceCollector(self.sample, self.options.get('evidence_limit', 1000),
                                     self.options.get('_evidence_observer'))
        payload = {'schema_version': 1, 'sample': self.sample,
                   'requested_backend': self.mode, 'actual_backend': self.mode,
                   'settings': native({k: v for k, v in self.options.items() if k not in ('fun', 'queryNamespace') and not k.startswith('_')}),
                   'status': 'complete', 'predictions': [], 'answers': {}, 'evidence': [],
                   'conflicts': [], 'runs': [], 'objective': None, 'unsupported': [],
                   'assignments': assignments, 'counterfactuals': []}
        namespace = self.options.get('queryNamespace')
        if isinstance(namespace, Mapping):
            payload['settings']['query_namespace_symbols'] = sorted(str(name) for name in namespace)
        with _lock, _ontology_transaction(self.graphs):
            dn = _clone(self.root)
            if dn.myBuilder is not None:
                builder = dn.myBuilder
                skeleton = builder.skeletonDataNode
                try:
                    # Materialize all entity rows on the private builder. The
                    # ordinary skeleton shortcut otherwise keeps only its root.
                    if builder.skeletonDataNodeFull:
                        builder.skeletonDataNode = False
                    builder.createFullDataNode(dn)
                finally:
                    builder.skeletonDataNode = skeleton
                if skeleton:
                    for node in _nodes(dn):
                        if node is not dn:
                            node.attributes['rootDataNode'] = dn
            ontologies = tuple(dict.fromkeys(g.ontology for g in self.graphs
                if g.ontology is not None and g.ontology.iri is not None))
            solver = gurobiILPOntSolver(self.graphs, ontologies, ilpConfig, reuse_model=False)
            solver.reuse_model = False
            solver._explanation_collector = evidence
            solver._hypothesisBatchSupported = lambda *args: False
            owned_models = []
            key = tuple(self.options.get('key', ('local', 'softmax')))
            if len(key) != 2:
                raise ValueError('key must be an inference key pair, e.g. (local, softmax)')
            concepts = dn.collectConceptsAndRelations(self.concepts)
            if not concepts:
                raise ValueError('No concepts with predictions found for explanation')
            if 'local' in key:
                retained = [(node.attributes, name, value) for node in _nodes(dn)
                            for name, value in node.attributes.items()
                            if name.endswith('/' + '/'.join(key))] if self.options.get('_preserve_inputs') else []
                if self.options.get('_preserve_inputs'):
                    for node in _nodes(dn):
                        for container_name in ('variableSet', 'propertySet'):
                            container = node.attributes.get(container_name, {})
                            retained.extend((container, name, value) for name, value in container.items()
                                            if isinstance(name, str) and name.endswith('/' + '/'.join(key)))
                dn.inferLocal(keys=(key[1],))
                for container, name, value in retained:
                    container[name] = value
            predictions, private = self._predictions(dn, concepts, key)
            payload['predictions'] = predictions
            for identifier, value in assignments.items():
                if identifier not in private:
                    raise ValueError(f'Unknown grounded prediction ID: {identifier}')
                node, concept, entry = private[identifier]
                if isinstance(concept, EnumConcept):
                    if not isinstance(value, str) or value not in concept.enum:
                        raise ValueError('Enum assignment must be a class name from class_names')
                elif type(value) is not bool:
                    raise ValueError('Binary assignment must be bool')
                if self.mode == 'tnorm':
                    raw = node.getAttribute(concept, *key)
                    replacement = torch.zeros_like(raw)
                    replacement.reshape(-1)[concept.enum.index(value) if isinstance(concept, EnumConcept) else int(value)] = 1
                    node.attributes[f'<{concept.name}>/' + '/'.join(key)] = replacement
                    entry['local_label'] = value
                    entry['probabilities'] = native(replacement)
            context = (AdHocExecutableQueries(dn.graph, [dn], solver, self.queries,
                       query_namespace=self.options.get('queryNamespace'))
                       if self.queries is not None else nullcontext())
            token = collector.set(evidence)
            try:
                with context as adhoc:
                    if adhoc is not None and self.mode == 'ilp':
                        adhoc.prepare_ilp()
                    dn.setActiveExecutableLCs()
                    names = (list(adhoc.public_to_internal.values()) if adhoc is not None
                             else list(dn.getActiveExecutableConstraintNames()))
                    requested = self.options.get('constraints')
                    if requested is not None:
                        requested = {requested} if isinstance(requested, str) else set(requested)
                        if self.mode == 'ilp' and requested != set(names):
                            raise ValueError('ILP constraints must exactly match active executable names')
                        unknown = requested.difference(dn.graph.executableLCs)
                        if unknown:
                            raise ValueError('Unknown executable constraints: ' + ', '.join(sorted(unknown)))
                        names = [name for name in dn.graph.executableLCs if name in requested]
                    for entry in predictions:
                        evidence.add(kind='prediction', prediction=entry)
                    if self.mode == 'ilp':
                        self._ilp(dn, solver, concepts, key, names, assignments, private, payload, evidence, owned_models)
                    else:
                        self._soft(dn, solver, concepts, key, names, assignments, private, payload, evidence)
                    if adhoc is not None:
                        payload['answers'] = {public: payload['answers'][internal]
                            for public, internal in adhoc.public_to_internal.items()
                            if internal in payload['answers']}
            except UndefinedEvidence as error:
                payload['status'] = 'impossible_evidence' if error.impossible else 'undefined_evidence'
                payload['answers'] = {}
            except CircuitSizeLimitExceeded:
                payload['status'] = 'size_limit'
                payload['unsupported'].append('Circuit size limit reached; no backend fallback was used.')
                payload['answers'] = {}
            except ValueError as error:
                if 'Unsupported executable constraint type' not in str(error):
                    raise
                payload['status'] = 'unsupported'
                payload['unsupported'].append(str(error))
            except NotImplementedError as error:
                # Invalid public arguments are validated outside this block.
                payload['status'] = 'unsupported'
                payload['unsupported'].append(str(error))
            finally:
                collector.reset(token)
                for model in owned_models:
                    model.dispose()
        payload['evidence'] = evidence.records
        payload['truncation'] = {'limit': evidence.limit, 'omitted': evidence.omitted}
        payload['elapsed_ms'] = (perf_counter() - started) * 1000
        return ExplanationReport(payload, self)

    def _soft(self, dn, solver, concepts, key, names, assignments, private, payload, evidence):
        options = {k: self.options[k] for k in ('tnorm', 'counting_tnorm', 'threshold',
            'circuit_backend', 'circuit_max_nodes', 'circuit_aggregation', 'compiled') if k in self.options}
        engine = ExecutableInference(solver, mode=self.mode, circuit_size_limit_action='raise', **options)
        engine._prepare_processor(dn)
        payload['actual_backend'] = (solver.myCircuitBooleanMethods.backend_name
                                     if self.mode == 'circuit' else 'tnorm:' + engine.tnorm)
        solver.circuitLossCalculator.compiled_executor.grounding_features = lambda *args: None
        # Amortized-training feature matrices are not needed for diagnostics;
        # aggregating operators can legitimately have no row-aligned matrix.
        for executor in (solver.circuitLossCalculator.compiled_executor, engine.loss_calculator):
            evaluator = getattr(executor, '_evaluator', None)
            if evaluator is not None:
                evaluator.grounding_features = lambda *args: None
        if self.mode == 'circuit' and assignments:
            processor = solver.myCircuitBooleanMethods
            original_wmc = processor.wmc
            def conditional(node):
                literals = []
                for identifier, value in assignments.items():
                    target, concept, _ = private[identifier]
                    index = concept.enum.index(value) if isinstance(concept, EnumConcept) else 1
                    leaf = solver.constraintConstructor._circuit_leaf(target,
                        f'<{concept.name}>/' + '/'.join(key),
                        (concept.name, index, index if isinstance(concept, EnumConcept) else 0), concept)
                    literal = processor._node(leaf)
                    if not isinstance(concept, EnumConcept) and not value:
                        literal = processor.manager.negate(literal)
                    literals.append(literal)
                event = processor.manager.and_all(literals)
                denominator = original_wmc(event)
                if float(denominator.detach()) <= 0:
                    raise UndefinedEvidence(event == processor.manager.false)
                payload['evidence_probability'] = native(denominator)
                return original_wmc(processor.manager.conjunction(processor._node(node), event)) / denominator
            processor.wmc = conditional
            # Validate even when there are no rules or executable queries.
            processor.begin_evaluation()
            conditional(processor.manager.true)
        for graph in solver.myGraph:
            for name, lc in graph.logicalConstrains.items():
                if lc.headLC and lc.active and type(lc).__name__ != 'fixedL':
                    answer = engine._infer_one(name, lc, dn, key, concepts)
                    payload['answers'][graph.name + '/' + name] = self._answer(answer)
        for name, answer in engine.infer(dn, names, concepts, key=key).items():
            payload['answers'][name] = self._answer(answer)
        if self.mode == 'circuit':
            payload['actual_backend'] = solver.myCircuitBooleanMethods.backend_name
        else:
            payload['actual_backend'] = 'tnorm:' + engine.tnorm
        payload['result_semantics'] = 'query probabilities' if self.mode == 'circuit' else 'fuzzy scores'

    def _answer(self, answer):
        result = native(answer)
        if result.get('type') == 'boolean':
            result['truth_probability' if self.mode == 'circuit' else 'truth_score'] = result['distribution'][1]
        if self.mode == 'tnorm':
            for old, new in (('probability', 'score'), ('positionProbabilities', 'position_scores')):
                if old in result:
                    result[new] = result.pop(old)
            result['semantics'] = 'fuzzy score'
        else:
            result['semantics'] = 'query probability'
        return result

    def _ilp(self, dn, solver, concepts, key, names, assignments, private, payload, evidence, owned_models):
        from gurobipy import GRB
        payload['actual_backend'] = 'gurobi'
        forced = []
        for identifier, value in assignments.items():
            node, concept, entry = private[identifier]
            forced.append((concept, node.instanceID, value, identifier))

        def before(model, variables):
            if not any(model is owned for owned in owned_models):
                owned_models.append(model)
            for concept, instance, value, identifier in forced:
                selected = [(k, var) for k, var in variables.items()
                    if k[0] is concept and k[2] == instance and not str(k[1]).startswith('Not') and var is not None]
                if not selected:
                    raise ValueError(f'No ILP variable for {identifier}')
                for variable_key, var in selected:
                    truth = int(variable_key[3] == concept.enum.index(value)) if isinstance(concept, EnumConcept) else int(value)
                    row = model.addConstr(var == truth, name='explanation_forced_' + str(len(model.getConstrs())))
                    model.update()
                    evidence.row_sources[(id(model), 'linear', row.index)] = {
                        'assignment': identifier, 'value': value, 'kind': 'counterfactual'}

        def after(model, variables, lcs, priority):
            run = {'status_code': model.Status, 'priority': priority,
                   'status': 'optimal' if model.Status == GRB.OPTIMAL else
                             'infeasible' if model.Status == GRB.INFEASIBLE else 'interrupted_or_ambiguous',
                   'objective': model.ObjVal if model.Status == GRB.OPTIMAL else None,
                   'rules': [lc.lcName for lc in lcs],
                   'hypotheses': {lc._explanation_executable_name: native(lc._explanation_hypothesis_value)
                       for lc in lcs if hasattr(lc, '_explanation_executable_name')},
                   'selected': False}
            payload['runs'].append(run)
            variable_info = {var.VarName: {
                'concept': graph_path(k[0]), 'instance_id': native(k[2]),
                'class_index': k[3], 'label': k[1]}
                for k, var in variables.items() if var is not None}
            def expression_vars(expression):
                return [expression.getVar(i).VarName for i in range(expression.size())]
            def row_variables(kind, row):
                if kind == 'linear':
                    return expression_vars(model.getRow(row))
                if row.GenConstrType == GRB.GENCONSTR_INDICATOR:
                    indicator, _, expression, _, _ = model.getGenConstrIndicator(row)
                    return [indicator.VarName] + expression_vars(expression)
                if row.GenConstrType in (GRB.GENCONSTR_AND, GRB.GENCONSTR_OR):
                    target, inputs = (model.getGenConstrAnd(row) if row.GenConstrType == GRB.GENCONSTR_AND
                                      else model.getGenConstrOr(row))
                    return [target.VarName] + [v.VarName for v in inputs]
                return []
            rows = []
            for kind, constraints in (('linear', model.getConstrs()), ('general', model.getGenConstrs())):
                for index, row in enumerate(constraints):
                    source = evidence.row_sources.get((id(model), kind, index),
                        {'kind': 'implicit_ontology', 'solver_name': row.ConstrName if kind == 'linear' else row.GenConstrName})
                    source = dict(source)
                    source['variables'] = [variable_info.get(name, {'auxiliary_variable': name})
                                           for name in row_variables(kind, row)]
                    rows.append((kind, row, source))
                    evidence.add(kind='solver_row', source=source,
                                 solver_name=row.ConstrName if kind == 'linear' else row.GenConstrName,
                                 priority=priority)
            if model.Status == GRB.INFEASIBLE:
                model.computeIIS()
                conflict = {'priority': priority, 'irreducible': bool(model.IISMinimal),
                            'minimum_certified': False, 'run_index': len(payload['runs']) - 1,
                            'hypotheses': run['hypotheses'], 'rows': [], 'bounds': []}
                for kind, row, source in rows:
                    if row.IISConstr if kind == 'linear' else row.IISGenConstr:
                        conflict['rows'].append(source)
                for var in model.getVars():
                    if var.IISLB or var.IISUB:
                        conflict['bounds'].append({
                            **variable_info.get(var.VarName, {'auxiliary_variable': var.VarName}),
                            'lower': bool(var.IISLB), 'upper': bool(var.IISUB),
                            'lower_value': var.LB, 'upper_value': var.UB})
                payload['conflicts'].append(conflict)
                evidence.add(kind='conflict', conflict=conflict)

        solver._explanation_before_optimize = before
        solver._explanation_after_optimize = after
        options = {'key': key, 'fun': self.options.get('fun'),
                   'epsilon': self.options.get('epsilon', 0.00001),
                   'compiled': self.options.get('compiled', True)}
        if names:
            result = DiagnosticAnswerSolver(dn.graph, solver=solver, compiled=options['compiled']).solve_active_constraints(
                dn, names, concepts, key=key, fun=options['fun'], epsilon=options['epsilon'],
                minimize_objective=self.options.get('minimizeObjective', False),
                ignore_pin_lcs=self.options.get('ignorePinLCs', False), populate=False, raise_on_infeasible=False)
        else:
            result = solver._calculateILPSelection(dn, *concepts, **options,
                minimizeObjective=self.options.get('minimizeObjective', False),
                ignorePinLCs=self.options.get('ignorePinLCs', False),
                populate=False, forceFreshModel=True, raiseOnInfeasible=False, computeIIS=False)
        if result is None:
            payload['status'] = ('infeasible' if payload['runs'] and
                all(run['status'] == 'infeasible' for run in payload['runs']) else 'interrupted_or_ambiguous')
            return
        payload['status'] = ('optimal' if all(run['status'] in ('optimal', 'infeasible')
                               for run in payload['runs']) else 'interrupted_or_ambiguous')
        payload['objective'] = {'value': result['objective'],
            'mode': result.get('objective_mode', 'custom' if options['fun'] is not None else 'raw_probability'),
            'direction': 'minimize' if self.options.get('minimizeObjective', False) else 'maximize'}
        if options['fun'] is not None:
            payload['settings']['objective_transform'] = (
                getattr(options['fun'], '__module__', '') + '.' +
                getattr(options['fun'], '__qualname__', type(options['fun']).__name__))
        payload['answers'] = native(result.get('hypotheses', {}))
        candidates = payload['runs'] if names else reversed(payload['runs'])
        for run in candidates:
            if (run['status'] == 'optimal' and run['objective'] == result['objective'] and
                    all(payload['answers'].get(name) == value for name, value in run['hypotheses'].items())):
                run['selected'] = True
                payload['selected_priority'] = run['priority']
                break
        solver.populateILPSelection(dn, concepts, result['values'])
        for entry in payload['predictions']:
            node, concept, _ = private[entry['id']]
            value = node.getAttribute(concept, 'ILP')
            if value is not None:
                entry['inferred_label'] = (concept.enum[int(value.argmax())]
                    if isinstance(concept, EnumConcept) else bool(value.reshape(-1)[0].item()))
                entry['changed'] = entry['inferred_label'] != entry['local_label']
        # Verification supplies discrete violations before/after independently
        # of the objective and records the grounded operator evaluations.
        from domiknows.solver.logicalConstraintVerifier import LogicalConstraintVerifier
        verifier = LogicalConstraintVerifier(solver)
        solver.booleanMethodsCalculator.current_device = dn.current_device
        executable_ids = {id(wrapper.innerLC) for graph in solver.myGraph
                          for wrapper in graph.executableLCs.values()}
        for phase, prediction_key in (('local_verification', '/local/argmax'), ('ilp_verification', '/ILP')):
            evidence.phase = phase
            dn.inferLocal(keys=('argmax',)) if phase == 'local_verification' else None
            if phase == 'local_verification' and self.options.get('_preserve_inputs'):
                for node, concept, entry in private.values():
                    values = torch.as_tensor(entry['probabilities'])
                    one_hot = torch.zeros_like(values)
                    one_hot[values.argmax()] = 1
                    node.attributes[f'<{concept.name}>/local/argmax'] = one_hot
            verified = {}
            for graph in solver.myGraph:
                for name, constraint in graph.logicalConstrains.items():
                    if (not constraint.headLC or not constraint.active or
                            type(constraint).__name__ == 'fixedL' or id(constraint) in executable_ids):
                        continue
                    result = verifier.verifySingleConstraint(constraint,
                        solver.booleanMethodsCalculator, dn, key=prediction_key,
                        compiled=options['compiled'])
                    verified[graph.fullname + '/' + name] = {k: v for k, v in result.items()
                        if k not in ('lc', 'elapsedInMsLC')}
            # An executable answer can legitimately be false or a nonzero
            # count; activation labels are not hard gold goals to verify.
            payload[phase] = native(verified)
        evidence.phase = 'inference'
        for entry in payload['predictions']:
            if entry.get('changed'):
                path = entry['path']
                entry['related_evidence_ids'] = [r['id'] for r in evidence.records
                    if r['kind'] == 'evaluation' and any(
                        entity and entity['instance_id'] == entry['instance_id']
                        and entity['path'] == entry['entity_path']
                        for groups in r.get('bindings', {}).values() for group in groups for entity in group)
                    and path.rsplit('/', 1)[-1] in r.get('concept_names', [])]
                entry['relationship'] = 'rule involvement; not a causal attribution'


def explain(root, concepts, *, mode='ilp', queries=None, **options):
    if mode not in ('ilp', 'circuit', 'tnorm'):
        raise ValueError('mode must be ilp, circuit, or tnorm')
    for alias, canonical in (('circuitBackend', 'circuit_backend'),
                             ('circuitMaxNodes', 'circuit_max_nodes'),
                             ('circuitAggregation', 'circuit_aggregation'),
                             ('circuitSizeLimitAction', 'circuit_size_limit_action')):
        if alias in options:
            if canonical in options:
                raise TypeError(f'Use either {alias} or {canonical}, not both')
            options[canonical] = options.pop(alias)
    if options.get('circuit_size_limit_action', 'raise') not in ('raise', None):
        raise ValueError('Explanations require circuitSizeLimitAction="raise"; fallback is disabled')
    if queries is not None and options.get('constraints') is not None:
        raise ValueError('queries and constraints are mutually exclusive')
    allowed = {'key', 'compiled', 'tnorm', 'counting_tnorm', 'threshold', 'circuit_backend',
               'circuit_max_nodes', 'circuit_aggregation', 'fun', 'epsilon',
               'minimizeObjective', 'ignorePinLCs', 'evidence_limit', 'queryNamespace',
               'constraints', 'circuit_size_limit_action', '_evidence_observer', '_preserve_inputs'}
    unknown = set(options) - allowed
    if unknown:
        raise TypeError('Unknown explanation options: ' + ', '.join(sorted(unknown)))
    limit = options.get('evidence_limit', 1000)
    if type(limit) is not int or limit < 0:
        raise ValueError('evidence_limit must be a non-negative integer')
    options.setdefault('evidence_limit', limit)
    options.setdefault('compiled', True)
    options.setdefault('key', ('local', 'softmax'))
    if mode != 'ilp':
        options.setdefault('tnorm', 'P')
        options.setdefault('threshold', 0.5)
    if mode == 'circuit':
        options.setdefault('circuit_aggregation', 'joint')
        options['circuit_size_limit_action'] = 'raise'
    with _lock:
        if root.graph.batch is not None and root.ontologyNode == root.graph.batch:
            return [DiagnosticSession(sample, concepts, mode, queries, options, index).run()
                    for index, sample in enumerate(root.relationLinks.get('contains', []))]
        return DiagnosticSession(root, concepts, mode, queries, options, 0).run()
