"""Normalized acquisition signals with full-stream grounding and bounded display."""
from collections import defaultdict
from collections.abc import Mapping
import copy
import json
import math
from numbers import Real, Integral
import random
from time import perf_counter

from domiknows.solver.explanation.evidence import native
from domiknows.solver.incremental.state import Snapshot, signature

SIGNALS = ('uncertainty', 'constraint_conflict', 'inference_disagreement')
DEFAULT_WEIGHTS = dict(zip(SIGNALS, (.35, .35, .30)))


def _key(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError('sample_id must return a string or integer')
    return json.dumps(native(value), ensure_ascii=True)


def _flatten(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten(item)
    elif isinstance(value, (Real, bool)) and math.isfinite(value):
        yield float(value)


def _entropy(values):
    if (not isinstance(values, list) or len(values) < 2 or
            any(not isinstance(v, Real) or not math.isfinite(v) or v < 0 or v > 1 for v in values)
            or not math.isclose(sum(values), 1., abs_tol=1e-5)):
        return None
    return -sum(p * math.log(p) for p in values if p > 0) / math.log(len(values))


def _target_id(sample, prediction):
    return json.dumps({'sample': native(sample), 'path': prediction['path'],
                       'instance_id': prediction['instance_id']}, sort_keys=True, separators=(',', ':'))


class _GroundingStream:
    """Retain domain associations and scalar sums, never tensors or raw inputs."""
    def __init__(self):
        self.entities = defaultdict(set)
        self.concepts = defaultdict(set)
        self.conflicts = defaultdict(lambda: [0., 0])
        self.local_groundings = defaultdict(lambda: [0., 0])
        self.pending_leaves = set()
        self.phase = None

    def __call__(self, record):
        if self.phase != record['phase']:
            self.pending_leaves.clear()
            self.phase = record['phase']
        if record.get('kind') == 'leaf':
            entity = record.get('entity')
            if entity:
                self.pending_leaves.add((entity['path'], json.dumps(entity['instance_id'], sort_keys=True)))
            return
        if record.get('kind') != 'evaluation' or not record.get('head'):
            return
        rule = record['rule_path']
        entities = set()
        for groups in record.get('bindings', {}).values():
            for group in groups:
                for entity in group:
                    if entity:
                        entities.add((entity['path'], json.dumps(entity['instance_id'], sort_keys=True)))
        self.entities[rule].update(entities | self.pending_leaves)
        self.pending_leaves.clear()
        self.concepts[rule].update(record.get('concept_names', ()))
        if record['phase'] == 'local_verification':
            values = list(_flatten(record['result']))
            if values:
                conflict = sum(1. - v for v in values) / len(values)
                self.conflicts[rule][0] += conflict
                self.conflicts[rule][1] += 1
                # Per-grounding attribution only when the executor actually
                # reports a joined grounding. Aggregates use the whole domain.
                for entity in entities:
                    entry = self.local_groundings[(rule, entity)]
                    entry[0] += conflict
                    entry[1] += 1


class ConstraintActiveLearner:
    def __init__(self, *concepts, mode='ilp', weights=None, strategy='weighted',
                 top_targets=5, evidence_limit=1000, diversity_weight=.25,
                 seed=0, **inference_options):
        if mode not in ('ilp', 'circuit', 'tnorm'):
            raise ValueError('mode must be ilp, circuit, or tnorm')
        if strategy not in ('weighted', 'random', *SIGNALS):
            raise ValueError('Unknown acquisition strategy')
        weights = DEFAULT_WEIGHTS if weights is None else weights
        if not isinstance(weights, Mapping) or set(weights) - set(SIGNALS):
            raise ValueError('Unknown acquisition weight')
        self.weights = {name: float(weights.get(name, 0.)) for name in SIGNALS}
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError('weights must be finite, nonnegative, and have positive sum')
        if strategy in SIGNALS:
            self.weights = {name: float(name == strategy) for name in SIGNALS}
        if isinstance(top_targets, bool) or not isinstance(top_targets, int) or top_targets < 1:
            raise ValueError('top_targets must be a positive integer')
        if isinstance(evidence_limit, bool) or not isinstance(evidence_limit, int) or evidence_limit < 0:
            raise ValueError('evidence_limit must be a nonnegative integer')
        if not math.isfinite(diversity_weight) or not 0 <= diversity_weight <= 1:
            raise ValueError('diversity_weight must be between zero and one')
        self.concepts, self.mode, self.strategy = concepts, mode, strategy
        self.top_targets, self.evidence_limit = top_targets, evidence_limit
        self.diversity_weight, self.seed = diversity_weight, seed
        self.options = dict(inference_options)
        allowed = {'key', 'compiled', 'tnorm', 'counting_tnorm', 'threshold', 'circuit_backend',
                   'circuit_max_nodes', 'circuit_aggregation', 'circuit_size_limit_action',
                   'fun', 'epsilon', 'minimizeObjective', 'ignorePinLCs', 'queryNamespace',
                   'constraints', 'queries', 'circuitBackend', 'circuitMaxNodes',
                   'circuitAggregation', 'circuitSizeLimitAction'}
        if set(self.options) - allowed:
            raise ValueError('Unknown or reserved inference option')
        action = self.options.get('circuit_size_limit_action', self.options.get('circuitSizeLimitAction', 'raise'))
        if action not in ('raise', None):
            raise ValueError('Acquisition requires explicit circuit semantics; size limit action must be raise')

    def session(self):
        return ActiveLearningSession(self)

    def select(self, pool, *, budget, sample_id=None, annotations=None, embeddings=None):
        return self._select(pool, budget, sample_id, annotations, embeddings, None)

    def _evaluate(self, root):
        stream = _GroundingStream()
        report = root.explainInference(*self.concepts, mode=self.mode,
            evidence_limit=self.evidence_limit, _evidence_observer=stream,
            _preserve_inputs=True, **self.options)
        if isinstance(report, list):
            raise ValueError('Pass individual sample roots, rather than a batch wrapper')
        payload = report.to_dict()
        # Detach the observer and private counterfactual session immediately.
        del report
        return payload, stream

    def _score(self, sample, payload, stream, mask):
        predictions = payload['predictions']
        evidence = copy.deepcopy(payload['evidence'])
        def namespace_references(value):
            if isinstance(value, dict):
                for name, item in value.items():
                    if name.endswith('evidence_ids') and isinstance(item, list):
                        value[name] = [_key(sample) + ':' + ref for ref in item]
                    else:
                        namespace_references(item)
            elif isinstance(value, list):
                for item in value:
                    namespace_references(item)
        for record in evidence:
            record['id'] = _key(sample) + ':' + record['id']
            record['sample_id'] = native(sample)
            namespace_references(record)
        ids = {_target_id(sample, prediction) for prediction in predictions}
        if mask - ids:
            raise ValueError(f'Annotation mask contains unknown targets for sample {sample}')
        targets = []
        unsupported = list(payload['unsupported'])
        if self.mode != 'ilp':
            unsupported.append('Target-aligned constrained-label disagreement is unavailable for this backend; its weight is omitted.')
        for p in predictions:
            identifier = _target_id(sample, p)
            if identifier in mask:
                continue
            entity = (p['entity_path'], json.dumps(p['instance_id'], sort_keys=True))
            concept = p['path'].rsplit('/', 1)[-1]
            related = [rule for rule, entities in stream.entities.items()
                       if entity in entities and concept in stream.concepts[rule]]
            conflicts = []
            if self.mode == 'ilp':
                conflicts = [total / count for (rule, bound), (total, count) in stream.local_groundings.items()
                             if entity == bound and concept in stream.concepts[rule]]
                if not conflicts:
                    conflicts = [stream.conflicts[rule][0] / stream.conflicts[rule][1]
                                 for rule in related if stream.conflicts[rule][1]]
            else:
                for rule in related:
                    graph_path, name = rule.rsplit('/', 1)
                    answer = payload['answers'].get(rule)
                    if answer is None:
                        answer = payload['answers'].get(graph_path.rsplit('/', 1)[-1] + '/' + name)
                    if isinstance(answer, dict):
                        truth = answer.get('truth_probability' if self.mode == 'circuit' else 'truth_score')
                        if isinstance(truth, Real) and math.isfinite(truth) and 0 <= truth <= 1:
                            conflicts.append(1. - truth)
            components = {'uncertainty': _entropy(p['probabilities']),
                          'constraint_conflict': sum(conflicts) / len(conflicts) if conflicts else None,
                          'inference_disagreement': float(p['inferred_label'] != p['local_label'])
                              if 'inferred_label' in p and self.mode == 'ilp' else None}
            effective = {name: weight for name, weight in self.weights.items()
                         if weight > 0 and components[name] is not None}
            total = sum(effective.values())
            effective = {name: value / total for name, value in effective.items()} if total else {}
            score = sum(components[name] * weight for name, weight in effective.items())
            references = [r['id'] for r in evidence
                          if r.get('rule_path') in related and r.get('kind') == 'evaluation']
            labels = p['class_names'] or [False, True]
            label_candidates = [{'label': label, 'classifier_probability': probability}
                                for label, probability in zip(labels, p['probabilities'])]
            label_candidates.sort(key=lambda label: -(label['classifier_probability']
                if isinstance(label['classifier_probability'], Real) else -1))
            targets.append(dict(id=identifier, path=p['path'], entity_path=p['entity_path'],
                instance_id=p['instance_id'], class_names=p['class_names'],
                label_candidates=label_candidates,
                suggested_label=p.get('inferred_label', p['local_label']),
                components=components, effective_weights=effective, score=score,
                eligible=bool(effective) or self.strategy == 'random',
                unavailable_signals=[name for name, value in components.items() if value is None],
                related_rules=sorted(related), evidence_ids=references,
                relationship='rule involvement; not a causal attribution'))
        targets.sort(key=lambda t: (-t['score'], t['id']))
        scored = [t for t in targets if t['eligible']]
        best = scored[:self.top_targets]
        score = sum(t['score'] for t in best) / len(best) if best else 0.
        return dict(sample_id=native(sample), status=payload['status'], actual_backend=payload['actual_backend'],
                    score=score, targets=targets, eligible=bool(scored), evidence=evidence,
                    truncation=payload['truncation'], unsupported=unsupported,
                    inference_settings=payload['settings'], review_required=payload['status'] == 'infeasible')

    def _select(self, pool, budget, sample_id, annotations, embeddings, session):
        from .report import AnnotationSelection
        started = perf_counter()
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError('budget must be a nonnegative integer')
        roots = list(pool)
        identify = sample_id or (lambda root: root.instanceID)
        samples = [identify(root) for root in roots]
        keys = [_key(sample) for sample in samples]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate sample identifiers')
        if annotations is not None and not isinstance(annotations, Mapping):
            raise ValueError('annotations must map sample IDs to target-ID collections')
        annotations = annotations or {}
        if set(_key(sample) for sample in annotations) - set(keys):
            raise ValueError('Annotation mask references an unknown sample')
        masks = {}
        for sample, values in annotations.items():
            if not isinstance(values, (list, tuple, set, frozenset)) or any(not isinstance(v, str) for v in values):
                raise ValueError('Annotation masks contain target IDs, not labels')
            masks[_key(sample)] = set(values)
        vectors = self._vectors(samples, embeddings)
        if session is not None:
            session._evict(keys)
        candidates, excluded = [], []
        reused, incremental_stats = 0, {}
        for sample, key, root in zip(samples, keys, roots):
            try:
                if session is None:
                    payload, stream = self._evaluate(root)
                else:
                    payload, stream, hit, stats = session._evaluate(key, root)
                    reused += int(hit)
                    incremental_stats[key] = stats
            except Exception as error:
                excluded.append(dict(sample_id=native(sample), reason='evaluation_failed',
                                     error=f'{type(error).__name__}: {error}'))
                continue
            # Validate masks even when backend evaluation cannot finish.
            candidate = self._score(sample, payload, stream, masks.get(key, set()))
            if payload['status'] not in ('complete', 'optimal'):
                excluded.append(dict(sample_id=native(sample), reason=payload['status'],
                                     review_required=payload['status'] == 'infeasible',
                                     unsupported=payload['unsupported'], conflicts=payload['conflicts'],
                                     evidence=candidate['evidence'], truncation=payload['truncation']))
            elif not candidate['targets']:
                excluded.append(dict(sample_id=native(sample), reason='fully_annotated_or_no_targets'))
            elif not candidate['eligible']:
                excluded.append(dict(sample_id=native(sample), reason='no_available_weighted_signal'))
            else:
                candidates.append(candidate)
        ranked = sorted(candidates, key=lambda c: (-c['score'], _key(c['sample_id'])))
        if self.strategy == 'random':
            ranked = sorted(candidates, key=lambda c: _key(c['sample_id']))
            random.Random(self.seed).shuffle(ranked)
        selected = self._choose(ranked, budget, vectors)
        return AnnotationSelection(dict(schema_version=1, mode=self.mode, strategy=self.strategy,
            requested_budget=budget, weights=self.weights, top_targets=self.top_targets,
            diversity=dict(enabled=vectors is not None, weight=self.diversity_weight),
            selected=selected, ranked=ranked, excluded=excluded,
            timing=dict(total_ms=(perf_counter() - started) * 1000),
            stats=dict(explanations_reused=reused, examples_evaluated=len(roots)-reused,
                       incremental=incremental_stats),
            annotation_semantics='whole examples; suggestions are model predictions'))

    def _vectors(self, samples, embeddings):
        if embeddings is None:
            return None
        if not isinstance(embeddings, Mapping) or set(_key(s) for s in embeddings) != set(_key(s) for s in samples):
            raise ValueError('embeddings must cover exactly the pool sample IDs')
        result, dimension = {}, None
        for sample in samples:
            raw = native(embeddings[sample])
            if not isinstance(raw, list) or not raw or any(not isinstance(v, Real) or not math.isfinite(v) for v in raw):
                raise ValueError('Embeddings must be finite nonempty one-dimensional vectors')
            if dimension is not None and len(raw) != dimension:
                raise ValueError('Embedding dimensions must match')
            dimension = len(raw)
            norm = math.sqrt(sum(v*v for v in raw))
            if norm == 0 or not math.isfinite(norm):
                raise ValueError('Embedding norms must be finite and nonzero')
            result[_key(sample)] = [v / norm for v in raw]
        return result

    def _choose(self, ranked, budget, vectors):
        remaining, selected = list(ranked), []
        while remaining and len(selected) < budget:
            if vectors is None or not selected:
                chosen, diversity, acquisition = remaining[0], None, remaining[0]['score']
            else:
                def utility(candidate):
                    vector = vectors[_key(candidate['sample_id'])]
                    distance = min((1. - sum(a*b for a, b in zip(vector, vectors[_key(s['sample_id'])]))) / 2.
                                   for s in selected)
                    distance = min(1., max(0., distance))
                    score = (1-self.diversity_weight)*candidate['score'] + self.diversity_weight*distance
                    return score, distance
                chosen = min(remaining, key=lambda c: (-utility(c)[0], _key(c['sample_id'])))
                acquisition, diversity = utility(chosen)
            remaining.remove(chosen)
            chosen = copy.deepcopy(chosen)
            chosen['selection_score'], chosen['diversity_score'] = acquisition, diversity
            selected.append(chosen)
        return selected


class ActiveLearningSession:
    """Owned incremental engines plus reusable detached explanation evidence.

    Changed samples refresh evidence with a cold diagnostic pass. Unchanged
    samples reuse both results and evidence; annotations only affect ranking.
    """
    def __init__(self, learner):
        self.learner, self.closed = learner, False
        self._entries = {}

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Session is closed')
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for entry in self._entries.values():
            entry['engine'].close()
        self._entries.clear()
        self.closed = True

    def _evict(self, keys):
        for key in list(self._entries):
            if key not in keys:
                self._entries.pop(key)['engine'].close()

    def _evaluate(self, key, root):
        learner = self.learner
        snapshot = Snapshot.read(root, learner.concepts)
        config = signature((learner.mode, learner.concepts, learner.options, learner.evidence_limit))
        previous = self._entries.get(key)
        fresh = previous is None or previous['root'] is not root or previous['config'] != config
        entry = previous
        if fresh:
            options = dict(learner.options)
            options.pop('circuit_size_limit_action', None)
            entry = dict(root=root, engine=root.incrementalInference(*learner.concepts,
                         mode=learner.mode, **options), snapshot=None, config=config)
        if not snapshot.opaque and learner.options.get('fun') is None and entry['snapshot'] == snapshot:
            return entry['payload'], entry['stream'], True, {}
        try:
            result = entry['engine'].infer(populate=False)
            payload, stream = learner._evaluate(root)
            if Snapshot.read(root, learner.concepts) != snapshot:
                raise RuntimeError('Pool sample changed during acquisition')
        except Exception:
            if fresh:
                entry['engine'].close()
            raise
        if payload['status'] in ('complete', 'optimal') and result.status in ('complete', 'optimal'):
            entry.update(snapshot=snapshot, payload=payload, stream=stream)
            self._entries[key] = entry
            if fresh and previous is not None:
                previous['engine'].close()
        elif fresh:
            entry['engine'].close()
        return payload, stream, False, result.stats

    def select(self, pool, *, budget, sample_id=None, annotations=None, embeddings=None):
        if self.closed:
            raise RuntimeError('Session is closed')
        return self.learner._select(pool, budget, sample_id, annotations, embeddings, self)
