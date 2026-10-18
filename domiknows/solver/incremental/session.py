"""Transactional persistent inference, with explicit measured reuse."""
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
import copy
from time import perf_counter

import torch
from gurobipy import GRB

from domiknows.graph.dataNode import DataNode
from domiknows.solver.answerModule import AnswerSolver
from domiknows.solver.executableInference import ExecutableInference, AdHocExecutableQueries
from domiknows.solver.ilpConfig import ilpConfig
from domiknows.solver.explanation.evidence import native
from domiknows.solver.explanation.session import _clone, _containers, _nodes, _graphs, _ontology_transaction, _lock
from .ilp import IncrementalILPSolver
from .result import IncrementalInferenceResult, new_stats
from .state import Snapshot, signature, has_changes


def _materialize(root):
    if root.myBuilder is None:
        return root
    builder = root.myBuilder
    skeleton = builder.skeletonDataNode
    try:
        if builder.skeletonDataNodeFull:
            builder.skeletonDataNode = False
        builder.createFullDataNode(root)
    finally:
        builder.skeletonDataNode = skeleton
    if skeleton:
        for node in _nodes(root):
            if node is not root:
                node.attributes['rootDataNode'] = root
    return root


class IncrementalInferenceSession:
    def __init__(self, root, concepts=(), *, mode='ilp', queries=None, **options):
        if mode not in ('ilp', 'circuit', 'tnorm'):
            raise ValueError('mode must be ilp, circuit, or tnorm')
        aliases = {'circuitBackend': 'circuit_backend', 'circuitMaxNodes': 'circuit_max_nodes',
                   'circuitAggregation': 'circuit_aggregation', 'circuitSizeLimitAction': 'circuit_size_limit_action'}
        for old, new in aliases.items():
            if old in options:
                if new in options:
                    raise TypeError(f'Use either {old} or {new}')
                options[new] = options.pop(old)
        allowed = {'key', 'compiled', 'tnorm', 'counting_tnorm', 'threshold', 'circuit_backend',
                   'circuit_max_nodes', 'circuit_aggregation', 'circuit_size_limit_action',
                   'queryNamespace', 'constraints', 'fun', 'epsilon', 'minimizeObjective', 'ignorePinLCs'}
        unknown = set(options) - allowed
        if unknown:
            raise TypeError('Unknown incremental options: ' + ', '.join(sorted(unknown)))
        if queries is not None and options.get('constraints') is not None:
            raise ValueError('queries and constraints are mutually exclusive')
        self.root, self.concepts, self.mode, self.queries = root, tuple(concepts), mode, queries
        self.options = dict(options)
        self.options.setdefault('key', ('local', 'softmax'))
        self.options.setdefault('compiled', True)
        self.key = tuple(self.options['key'])
        if len(self.key) != 2:
            raise ValueError('key must contain the inference key pair')
        self.closed = False
        self.snapshot = None
        self.last_result = None
        self._private = None
        self._node_map = {}
        self._solver = None
        self._soft_results = {}
        self._soft_signatures = {}
        self._prepared_queries = None
        self._values = {}
        self._selected_concepts = ()
        self._published = []
        self._samples = {}
        self._is_batch = root.graph.batch is not None and root.ontologyNode == root.graph.batch
        self._had_circuit_fallback = False

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Session is closed')
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with _lock:
            if self.closed:
                return
            if self._solver is not None:
                self._solver.close()
            for child in self._samples.values():
                child.close()
            self._samples.clear()
            self._private = self._solver = None
            self._node_map.clear()
            self._soft_results.clear()
            self._soft_signatures.clear()
            self._prepared_queries = self.last_result = self.snapshot = None
            self._values.clear()
            self._published.clear()
            self.closed = True

    def _sync(self):
        fresh = _materialize(_clone(self.root))
        public_ids = {node.id for node in _nodes(self.root)}
        nodes = _nodes(fresh)
        def key(node):
            return ('public', node.id) if node.id in public_ids else (
                'materialized', node.ontologyNode.fullname, signature(node.instanceID))
        mapping = {}
        live = {}
        for node in nodes:
            identity = key(node)
            target = self._node_map.get(identity)
            if target is None:
                target = copy.copy(node)
            mapping[id(node)] = target
            live[identity] = target
        builders = {id(n.myBuilder): n.myBuilder for n in nodes if n.myBuilder is not None}
        for identifier, builder in builders.items():
            mapping[identifier] = dict.__new__(type(builder))
        for identifier, builder in builders.items():
            duplicate = mapping[identifier]
            duplicate.__dict__ = _containers(builder.__dict__, mapping)
            for name, value in builder.items():
                dict.__setitem__(duplicate, name, _containers(value, mapping))
        for node in nodes:
            target = mapping[id(node)]
            numerical_id = target.id
            target.__dict__ = _containers(node.__dict__, mapping)
            target.id = numerical_id
            target.collectedConceptsAndRelations = None
            target.__dict__.pop('_relationIdCache', None)
            target.__dict__.pop('_impactIdCache', None)
        self._private = mapping[id(fresh)]
        self._node_map = live

    def _clear_outputs(self):
        for container, key, value in self._published:
            if container.get(key) is value:
                container.pop(key, None)
        self._published = []

    def _publish(self, result):
        self._clear_outputs()
        root = self._private
        if self.mode == 'ilp':
            self._solver.populateILPSelection(root, self._selected_concepts, self._values)
            public = {node.id: node for node in _nodes(self.root)}
            for node in _nodes(root):
                target = public.get(node.id)
                if target is None:
                    continue
                for key, value in node.attributes.items():
                    if key.endswith('/ILP'):
                        published = _containers(value)
                        target.attributes[key] = published
                        self._published.append((target.attributes, key, published))
                for key, value in node.attributes.get('variableSet', {}).items():
                    if key.endswith('/ILP'):
                        target.attributes.setdefault('variableSet', {})[key] = _containers(value)
                        container = target.attributes['variableSet']
                        self._published.append((container, key, container[key]))
        constraint = self.root._getExecutableConstraintDataNode()
        if constraint is not None:
            for name, answer in result.answers.items():
                if name not in self.root.graph.executableLCs:
                    continue
                values = (answer.get('answer'), answer.get('probability')) if isinstance(answer, dict) else (answer, None)
                for suffix, value in zip(('answer', 'probability'), values):
                    key = name + '/' + suffix
                    constraint.attributes[key] = value
                    self._published.append((constraint.attributes, key, value))

    def _refresh_bindings(self, change):
        solver = self._solver
        executors = [solver.compiledModeExecutor, solver.circuitLossCalculator.compiled_executor,
                     getattr(solver, '_compiled_loss_calculator', None)]
        for executor in executors:
            if executor is None:
                continue
            store = getattr(executor, '_store', None) or getattr(executor, '_prob_store', None)
            if store is not None:
                store.incremental_changes = change
                store.rebind(self._private, graphs=solver.myGraph)
                evaluator = getattr(executor, '_evaluator', None)
                if evaluator is not None:
                    evaluator.rebind(store)

    def _dependencies(self, lc):
        names = set(lc.getLcConcepts())
        # A sibling predicate can supply another weight of a categorical leaf.
        for name in list(names):
            concept = self._private.graph.findConcept(name)
            if concept is not None:
                for relation in concept._out.get('is_a', ()):
                    parent = relation.dst
                    if parent._out.get('is_a'):
                        names.update(r.src.name for r in parent._in.get('is_a', ()))
        return names

    def _adjacency_stats(self):
        executors = (self._solver.compiledModeExecutor,
                     self._solver.circuitLossCalculator.compiled_executor,
                     getattr(self._solver, '_compiled_loss_calculator', None))
        resolvers = {id(e._evaluator.candidateResolver): e._evaluator.candidateResolver
                     for e in executors if e is not None and getattr(e, '_evaluator', None) is not None}
        return (sum(r.adjacency_hits for r in resolvers.values()),
                sum(r.adjacency_misses for r in resolvers.values()))

    def _reset_circuit(self):
        from domiknows.solver.circuitBooleanMethods import circuitBooleanMethods
        solver = self._solver
        processor = solver.myCircuitBooleanMethods
        solver.myCircuitBooleanMethods = circuitBooleanMethods(
            backend=processor.requested_backend, max_nodes=processor.max_nodes,
            size_limit_action=processor.size_limit_action)
        solver.myCircuitBooleanMethods.current_device = processor.current_device
        solver.myCircuitBooleanMethods.current_dtype = processor.current_dtype
        solver.circuitLossCalculator._compile_cache.clear()
        solver.circuitLossCalculator.__dict__.pop('_incremental_construct_cache', None)

    def _soft(self, names, concepts, change, stats, force_fresh=False):
        solver, root = self._solver, self._private
        arguments = {k: v for k, v in self.options.items() if k in
            ('compiled', 'tnorm', 'counting_tnorm', 'threshold', 'circuit_backend',
             'circuit_max_nodes', 'circuit_aggregation', 'circuit_size_limit_action')}
        engine = ExecutableInference(solver, mode=self.mode, **arguments)
        engine._prepare_processor(root)
        self._refresh_bindings(change)
        # These sessions perform inference, not amortized-dual training.
        solver.circuitLossCalculator.compiled_executor.grounding_features = lambda *a: None
        evaluator = getattr(engine.loss_calculator, '_evaluator', None)
        if evaluator is not None:
            evaluator.grounding_features = lambda *a: None
        calculator = solver.circuitLossCalculator
        if self.mode == 'circuit' and not force_fresh:
            processor = solver.myCircuitBooleanMethods
            compact = processor.max_nodes is not None and processor.node_count > processor.max_nodes * .8
            if self._had_circuit_fallback or compact:
                self._reset_circuit()
                force_fresh = True
                stats['full_rebuild_reasons'].append('circuit manager compaction or previous fallback')
        if not hasattr(calculator, '_incremental_construct_cache'):
            calculator._incremental_construct_cache = {}
        calculator._incremental_stats = stats
        if self.mode == 'circuit':
            processor = solver.myCircuitBooleanMethods
            if not hasattr(processor, '_incremental_native_wmc'):
                processor._incremental_native_wmc = processor.wmc
            original_wmc = processor._incremental_native_wmc
            def measured_wmc(node):
                started = perf_counter()
                try:
                    return original_wmc(node)
                finally:
                    stats['evaluation_ms'] += (perf_counter() - started) * 1000
            processor.wmc = measured_wmc
        output, fingerprints = {}, {}
        targets = OrderedDict()
        for graph in solver.myGraph:
            for name, lc in graph.logicalConstrains.items():
                if lc.headLC and lc.active and type(lc).__name__ != 'fixedL':
                    targets[graph.fullname + '/' + name] = lc
        for name in names:
            wrapper = root.graph.executableLCs[name]
            if wrapper.innerLC.active:
                targets[name] = wrapper.innerLC
        live_ids = {id(lc) for lc in targets.values()}
        for key in tuple(calculator._incremental_construct_cache):
            if key[0] not in live_ids:
                del calculator._incremental_construct_cache[key]
        broad = force_fresh or any(change.get(k) for k in ('initial', 'entities', 'relations', 'schema', 'input_layout', 'attributes', 'opaque'))
        if change.get('rules') or change.get('hinted_rules'):
            # Pins affect every read of their predicate, including queries
            # whose own expression did not change. Include deleted pins.
            broad |= any(type(lc).__name__ == 'fixedL' for g in solver.myGraph
                         for lc in g.logicalConstrains.values())
            broad |= self.snapshot is not None and any(
                len(value) > 1 and value[1] == 'fixedL' for value in self.snapshot.rules.values())
        for name, lc in targets.items():
            calculator.__dict__.setdefault('_incremental_construct_cache', {})
            fingerprints[name] = signature(lc)
            structural = broad or self._soft_signatures.get(name) != fingerprints[name]
            if structural:
                for key in tuple(calculator._incremental_construct_cache):
                    if key[0] == id(lc):
                        del calculator._incremental_construct_cache[key]
            dirty = (structural or name not in self._soft_results or
                     bool(self._dependencies(lc) & set(change.get('concepts', ()))) or
                     lc.lcName in change.get('hinted_rules', ()))
            if not dirty:
                output[name] = copy.deepcopy(self._soft_results[name])
                stats['constraints_reused'] += 1
                continue
            started = perf_counter()
            evaluating_before = stats['evaluation_ms']
            output[name] = native(engine._infer_one(name, lc, root, self.key, concepts))
            elapsed = (perf_counter() - started) * 1000
            if self.mode == 'circuit':
                stats['grounding_ms'] += max(0., elapsed - (stats['evaluation_ms'] - evaluating_before))
            else:
                stats['evaluation_ms'] += elapsed
            stats['constraints_rebuilt'] += 1
        if self.mode == 'circuit' and any(value.get('fallback') for value in output.values()):
            stats['backend_fallbacks'] = [name for name, value in output.items() if value.get('fallback')]
            if not force_fresh and not change.get('initial'):
                # A limit depends on manager history. Repeat with a compact,
                # fresh manager so the fallback sequence matches a cold call.
                self._reset_circuit()
                stats['full_rebuild_reasons'].append('circuit size-limit fallback; repeated cold')
                return self._soft(names, concepts, change, stats, force_fresh=True)
        self._staged_soft = output, fingerprints
        return output

    def _ilp(self, names, concepts, stats):
        solver, root = self._solver, self._private
        args = dict(key=self.key, fun=self.options.get('fun'), epsilon=self.options.get('epsilon', .00001))
        if names:
            selected = AnswerSolver(root.graph, solver=solver, compiled=self.options['compiled']).solve_active_constraints(
                root, names, concepts, **args, populate=False, raise_on_infeasible=False,
                minimize_objective=self.options.get('minimizeObjective', False),
                ignore_pin_lcs=self.options.get('ignorePinLCs', False))
        else:
            selected = solver._calculateILPSelection(root, *concepts, **args,
                compiled=self.options['compiled'], populate=False, forceFreshModel=True,
                raiseOnInfeasible=False, computeIIS=False,
                minimizeObjective=self.options.get('minimizeObjective', False),
                ignorePinLCs=self.options.get('ignorePinLCs', False))
        if selected is None:
            return ('infeasible' if solver.statuses and all(s == GRB.INFEASIBLE for s in solver.statuses)
                    else 'interrupted_or_ambiguous'), {}, {}, None
        status = 'optimal' if all(s in (GRB.OPTIMAL, GRB.INFEASIBLE) for s in solver.statuses) else 'interrupted_or_ambiguous'
        return status, native(selected.get('hypotheses', {})), selected['values'], selected['objective']

    def infer(self, changes=None, populate=True):
        with _lock:
            if self.closed:
                raise RuntimeError('Session is closed')
            if self._is_batch:
                live = list(self.root.relationLinks.get('contains', ()))
                for identity in list(self._samples):
                    if not any(id(root) == identity for root in live):
                        removed = self._samples.pop(identity)
                        removed._clear_outputs()
                        removed.close()
                result = []
                for root in live:
                    child = self._samples.setdefault(id(root), None)
                    if child is None:
                        child = IncrementalInferenceSession(root, self.concepts, mode=self.mode,
                                                            queries=self.queries, **self.options)
                        self._samples[id(root)] = child
                    result.append(child.infer(changes, populate=populate))
                return result
            return self._infer(changes, populate)

    def _infer(self, hints, populate):
        started = perf_counter()
        stats = new_stats()
        source = Snapshot.read(self.root, self.concepts)
        change = source.diff(self.snapshot, hints)
        if self.options.get('fun') is not None:
            # A custom objective may read external mutable state. Re-evaluate
            # conservatively rather than caching an unverifiable callback.
            change['opaque'] = True
        stats['detection_ms'] = (perf_counter() - started) * 1000
        if self.last_result is not None and not has_changes(change):
            result = copy.deepcopy(self.last_result)
            result.changes, result.stats = change, stats
            stats['constraints_reused'] = len(self._soft_results) if self.mode != 'ilp' else sum(
                len(run.blocks) for runs in self._solver.committed.values() for run in runs)
            if populate:
                publishing = perf_counter()
                self._publish(result)
                stats['population_ms'] = (perf_counter() - publishing) * 1000
            stats['total_ms'] = (perf_counter() - started) * 1000
            return result
        old_soft, old_signatures = self._soft_results, self._soft_signatures
        previous_solver, previous_queries = self._solver, self._prepared_queries
        graphs = _graphs(self.root, self.concepts)
        try:
            with _ontology_transaction(graphs), torch.no_grad():
                binding = perf_counter()
                self._sync()
                if self._solver is None or change.get('schema'):
                    ontologies = tuple(dict.fromkeys(g.ontology for g in graphs
                        if g.ontology is not None and g.ontology.iri is not None))
                    self._solver = IncrementalILPSolver(graphs, ontologies, ilpConfig)
                    stats['full_rebuild_reasons'].append('schema changed' if previous_solver is not None else 'initial session')
                    self._prepared_queries = None
                solver, root = self._solver, self._private
                solver.myGraph = graphs
                solver.logical_constraints = {name: lc for graph in graphs for name, lc in graph.logicalConstrains.items()}
                solver.begin(change, stats)
                plan_before = solver.compiledModeExecutor.plan_cache.info()
                adjacency_before = self._adjacency_stats()
                concepts = root.collectConceptsAndRelations(self.concepts)
                if not concepts:
                    raise ValueError('No concepts with predictions found')
                if 'local' in self.key:
                    # inferLocal historically recomputes existing values because
                    # DataNode.hasAttribute is a stub. Explicit session inputs
                    # are authoritative; compute only absent derived inputs.
                    suffix = '/' + '/'.join(self.key)
                    retained = [(n.attributes, k, v) for n in _nodes(root)
                                for k, v in n.attributes.items() if k.endswith(suffix)]
                    for n in _nodes(root):
                        for container_name in ('variableSet', 'propertySet'):
                            container = n.attributes.get(container_name, {})
                            retained.extend((container, k, v) for k, v in container.items()
                                            if isinstance(k, str) and k.endswith(suffix))
                    root.inferLocal(keys=(self.key[1],))
                    for container, name, value in retained:
                        container[name] = value
                self._refresh_bindings(change)
                stats['grounding_ms'] = (perf_counter() - binding) * 1000
                context = (AdHocExecutableQueries(root.graph, [root], solver,
                    self._prepared_queries or self.queries,
                    query_namespace=self.options.get('queryNamespace')) if self.queries is not None else nullcontext())
                with context as adhoc:
                    if adhoc is not None:
                        if self.mode == 'ilp':
                            adhoc.prepare_ilp()
                        self._prepared_queries = OrderedDict((public, root.graph.executableLCs[internal])
                            for public, internal in adhoc.public_to_internal.items())
                    root.setActiveExecutableLCs()
                    names = (list(adhoc.public_to_internal.values()) if adhoc is not None else
                             list(root.getActiveExecutableConstraintNames()))
                    if self.options.get('constraints') is not None:
                        requested = self.options['constraints']
                        requested = {requested} if isinstance(requested, str) else set(requested)
                        if self.mode == 'ilp' and requested != set(names):
                            raise ValueError('ILP constraints must exactly match active names')
                        if requested - set(root.graph.executableLCs):
                            raise ValueError('Unknown executable constraint')
                        names = [n for n in root.graph.executableLCs if n in requested]
                    if self.mode == 'ilp':
                        status, answers, values, objective = self._ilp(names, concepts, stats)
                    else:
                        answers = self._soft(names, concepts, change, stats)
                        status, values, objective = 'complete', {}, None
                    if adhoc is not None:
                        answers = {public: answers[internal] for public, internal in adhoc.public_to_internal.items()
                                   if internal in answers}
                assignments = [{'concept': concept.fullname, 'instance_id': native(instance),
                                'class_index': index, 'value': float(value)}
                    for (concept, label, instance, index), value in values.items()
                    if value is not None and not str(label).startswith('Not_')]
                assignments.sort(key=lambda item: (item['concept'], repr(item['instance_id']), item['class_index']))
                result = IncrementalInferenceResult(status, self.mode, copy.deepcopy(answers), assignments,
                                                     native(change), stats, objective)
                plan_after = solver.compiledModeExecutor.plan_cache.info()
                stats['formula_plan_hits'] = plan_after['hits'] - plan_before['hits']
                stats['formula_plan_misses'] = plan_after['misses'] - plan_before['misses']
                stats['compilation_ms'] = plan_after['compilation_ms'] - plan_before['compilation_ms']
                adjacency_after = self._adjacency_stats()
                stats['adjacency_reused'] = adjacency_after[0] - adjacency_before[0]
                stats['adjacency_rebuilt'] = adjacency_after[1] - adjacency_before[1]
            if Snapshot.read(self.root, self.concepts) != source:
                raise RuntimeError('Concurrent graph mutation detected; results were not published')
            if status not in ('complete', 'optimal'):
                solver.rollback()
                if solver is not previous_solver:
                    solver.close()
                    self._solver, self._prepared_queries = previous_solver, previous_queries
                self._clear_outputs()
                stats['total_ms'] = (perf_counter() - started) * 1000
                return result
            old_values, old_concepts = self._values, self._selected_concepts
            self._values, self._selected_concepts = values, concepts
            if populate:
                publishing = perf_counter()
                try:
                    self._publish(result)
                except Exception:
                    self._values, self._selected_concepts = old_values, old_concepts
                    raise
                stats['population_ms'] = (perf_counter() - publishing) * 1000
            solver.commit()
            if previous_solver is not None and solver is not previous_solver:
                previous_solver.close()
            if self.mode != 'ilp':
                self._soft_results, self._soft_signatures = self._staged_soft
                self._had_circuit_fallback = any(v.get('fallback') for v in self._soft_results.values())
            stats['total_ms'] = (perf_counter() - started) * 1000
            self.snapshot, self.last_result = source, copy.deepcopy(result)
            return result
        except Exception as error:
            if self._solver is not None:
                self._solver.rollback()
                self._reset_circuit()
                if self._solver is not previous_solver:
                    self._solver.close()
                    self._solver, self._prepared_queries = previous_solver, previous_queries
            self._soft_results, self._soft_signatures = old_soft, old_signatures
            self._clear_outputs()
            stats['total_ms'] = (perf_counter() - started) * 1000
            return IncrementalInferenceResult('error', self.mode, {}, [], native(change), stats,
                                              error=f'{type(error).__name__}: {error}')
