"""Bounded template mining with example-clustered validation."""
from collections.abc import Mapping
import itertools
import math
import random
from statistics import mean, pstdev

from domiknows.solver.explanation.evidence import graph_path, native
from domiknows.solver.explanation.session import _lock, _ontology_transaction, _graphs
from .specs import Vocabulary, root_concept, digest
from .data import extract, evaluate, sample_key, sample_nodes

TEMPLATES = ('implication', 'disjointness', 'relation_typing', 'counting')


def metrics(samples, spec, vocabulary, source, *, seed, rounds, evidence_limit):
    counts = dict(applicable_groundings=0, unknown_groundings=0, vacuous_groundings=0,
                  violations=0, supporting_examples=0, violating_examples=0, covered_examples=0)
    confidences, evidence = [], []
    omitted = 0
    for sample in samples:
        observed = list(evaluate(sample, spec, vocabulary, source))
        applicable = [row for row in observed if row[0] is True]
        violations = sum(row[1] for row in applicable)
        counts['applicable_groundings'] += len(applicable)
        counts['violations'] += violations
        counts['unknown_groundings'] += sum(row[0] is None for row in observed)
        counts['vacuous_groundings'] += sum(row[0] is False for row in observed)
        counts['covered_examples'] += int(bool(applicable))
        counts['supporting_examples'] += int(any(not row[1] for row in applicable))
        counts['violating_examples'] += int(violations > 0)
        if applicable:
            confidences.append(1 - violations / len(applicable))
        for applicable_flag, violated, grounding in observed:
            if applicable_flag is True:
                record = dict(sample_id=sample['sample_id'], source=source,
                              kind='contradiction' if violated else 'support', grounding=grounding)
                if len(evidence) < evidence_limit:
                    record['id'] = 'observation-' + digest(record)[:16]
                    evidence.append(record)
                else:
                    omitted += 1
    randomizer = random.Random(seed)
    bootstrap = [mean(randomizer.choices(confidences, k=len(confidences))) for _ in range(rounds)] if confidences else []
    bootstrap.sort()
    confidence = mean(confidences) if confidences else None
    return dict(**counts, example_count=len(samples),
        coverage=counts['covered_examples']/len(samples) if samples else 0.,
        confidence=confidence, violation_rate=1-confidence if confidence is not None else None,
        micro_confidence=1-counts['violations']/counts['applicable_groundings'] if counts['applicable_groundings'] else None,
        stability=dict(unit='example', bootstrap_rounds=rounds, std=pstdev(bootstrap) if bootstrap else None,
                       interval=[bootstrap[int(.025*(len(bootstrap)-1))], bootstrap[int(.975*(len(bootstrap)-1))]] if bootstrap else None),
        evidence=evidence, truncation=dict(limit=evidence_limit, omitted=omitted))


class ConstraintInducer:
    def __init__(self, *concepts, templates=TEMPLATES, min_support_examples=2,
                 min_coverage=.1, max_violation_rate=.05, max_stability_std=.2,
                 bootstrap_rounds=50, evidence_limit=100, max_candidates=256,
                 allow_prediction_only=False, seed=0, key=('local', 'softmax')):
        if not concepts:
            raise ValueError('Provide the allowed concept vocabulary')
        self.vocabulary = Vocabulary(concepts)
        if not templates or set(templates) - set(TEMPLATES):
            raise ValueError('Unknown or empty template selection')
        for name, value, lower in (('min_support_examples', min_support_examples, 1),
            ('bootstrap_rounds', bootstrap_rounds, 2), ('evidence_limit', evidence_limit, 0),
            ('max_candidates', max_candidates, 1)):
            if type(value) is not int or value < lower:
                raise ValueError(f'{name} must be an integer at least {lower}')
        for value in (min_coverage, max_violation_rate, max_stability_std):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Validation thresholds must be between zero and one')
        if len(key) != 2:
            raise ValueError('key must be an inference key pair')
        self.templates, self.seed, self.key = tuple(templates), seed, tuple(key)
        self.rounds, self.evidence_limit, self.max_candidates = bootstrap_rounds, evidence_limit, max_candidates
        self.policy = dict(min_support_examples=min_support_examples, min_coverage=min_coverage,
            max_violation_rate=max_violation_rate, max_stability_std=max_stability_std,
            allow_prediction_only=bool(allow_prediction_only))

    def _automatic(self, samples):
        vocabulary = self.vocabulary
        literals = vocabulary.literals()
        entities = [literal for literal in literals if not root_concept(vocabulary.concepts[literal['concept']])._out.get('has_a')]
        for a, b in itertools.permutations(entities, 2):
            if root_concept(vocabulary.concepts[a['concept']]) is not root_concept(vocabulary.concepts[b['concept']]):
                continue
            if a['concept'] == b['concept']:
                continue  # categorical exclusivity is already an ontology rule
            if 'implication' in self.templates:
                yield dict(template='implication', antecedent=a, consequent=b)
            if 'disjointness' in self.templates and digest(a) < digest(b):
                yield dict(template='disjointness', left=a, right=b)
        for literal in literals:
            c = vocabulary.concepts[literal['concept']]
            roles = root_concept(c)._out.get('has_a', ())
            if roles and len(roles) <= 4 and 'relation_typing' in self.templates:
                choices = [[v for v in entities if root_concept(vocabulary.concepts[v['concept']]) is root_concept(r.dst)] for r in roles]
                for selected in itertools.product(*choices):
                    yield dict(template='relation_typing', relation=literal,
                               roles={r.name: v for r, v in zip(roles, selected)})
            if 'counting' in self.templates:
                for role in [r.name for r in roles] if roles else [None]:
                    dummy = vocabulary.validate(dict(template='counting', predicate=literal, operator='at_most', limit=0, group_by=role))
                    for source in ('annotation', 'prediction'):
                        counts = [row[2]['count'] for sample in samples for row in evaluate(sample, dummy, vocabulary, source)
                                  if row[0] is True]
                        if counts:
                            yield {**dummy, 'limit': max(counts), 'operator': 'at_most', '_generation_source': source}
                            yield {**dummy, 'limit': min(counts), 'operator': 'at_least', '_generation_source': source}

    def propose(self, discovery_examples, *, guidance=(), validation_examples=(),
                guidance_adapter=None, sample_id=None, complete_domains=None):
        from .proposals import ConstraintProposals
        discovery, validation = list(discovery_examples), list(validation_examples)
        identify = sample_id or (lambda root: root.instanceID)
        discovery_ids, validation_ids = [identify(r) for r in discovery], [identify(r) for r in validation]
        keys = [sample_key(value) for value in discovery_ids + validation_ids]
        if len(keys) != len(set(keys)) or {id(n) for r in discovery for n in sample_nodes(r)} & {id(n) for r in validation for n in sample_nodes(r)}:
            raise ValueError('Discovery and validation require disjoint, unique sample identities')
        if complete_domains is not None and not isinstance(complete_domains, Mapping):
            raise ValueError('complete_domains maps sample IDs to concept-path collections')
        complete = {}
        for sample, paths in (complete_domains or {}).items():
            if sample_key(sample) not in keys or not isinstance(paths, (list, tuple, set, frozenset)):
                raise ValueError('Invalid complete-domain declaration')
            complete[sample] = [self.vocabulary.resolve(path) for path in paths]
        graphs = list(dict.fromkeys(g for root in discovery + validation for g in _graphs(root, tuple(self.vocabulary.concepts.values()))))
        with _lock, _ontology_transaction(graphs):
            discovered = extract(discovery, self.vocabulary, identify, complete, self.key)
            heldout = extract(validation, self.vocabulary, identify, complete, self.key)
        if isinstance(guidance, str):
            guidance = [guidance]
        guidance = list(guidance)
        if any(not isinstance(text, str) or not text.strip() for text in guidance):
            raise ValueError('Guidance must contain nonempty text')
        candidates, unresolved, omitted = {}, [], 0
        def add(spec, origin):
            nonlocal omitted
            spec = self.vocabulary.validate(spec)
            if spec['template'] not in self.templates:
                raise ValueError('Template is not enabled')
            identifier = 'c-' + digest(spec)[:16]
            if identifier not in candidates and len(candidates) + len(unresolved) >= self.max_candidates:
                omitted += 1
                return
            candidate = candidates.setdefault(identifier, dict(id=identifier, specification=spec, origins=[]))
            if origin not in candidate['origins']:
                candidate['origins'].append(origin)
        # Guidance is considered first, so a large search does not bury explicit
        # domain requirements behind statistical candidates.
        for index, text in enumerate(guidance):
            if len(candidates) + len(unresolved) >= self.max_candidates:
                omitted += len(guidance)-index
                break
            origin = dict(method='guidance', guidance_id=f'g{index}', text=text)
            try:
                if guidance_adapter is None:
                    add(self.vocabulary.parse(text), origin)
                else:
                    response = guidance_adapter(dict(text=text, guidance_id=f'g{index}',
                        vocabulary=[dict(path=p, classes=getattr(c, 'enum', None),
                            roles=[r.name for r in root_concept(c)._out.get('has_a', ())])
                            for p, c in self.vocabulary.concepts.items()], templates=list(self.templates)))
                    if not isinstance(response, dict) or set(response) != {'candidates'} or not isinstance(response['candidates'], list) or not response['candidates']:
                        raise ValueError('Adapter must return a nonempty candidates list')
                    if len(response['candidates']) > self.max_candidates:
                        raise ValueError('Adapter candidate count exceeds the configured bound')
                    normalized = [self.vocabulary.validate(spec) for spec in response['candidates']]
                    if any(spec['template'] not in self.templates for spec in normalized):
                        raise ValueError('Adapter requested a disabled template')
                    for spec in normalized:
                        add(spec, {**origin, 'adapter': True})
            except Exception as error:
                unresolved.append(dict(id='u-' + digest(origin)[:16], specification=None,
                    origins=[origin], error=f'{type(error).__name__}: {error}'))
        for spec in self._automatic(discovered):
            origin = spec.pop('_generation_source', 'compatible template enumeration')
            add(spec, dict(method='statistical', split='discovery', generation_source=origin,
                           observations='annotations and predictions separately'))
            if len(candidates) + len(unresolved) >= self.max_candidates:
                omitted += 1
                break
        return ConstraintProposals(self, list(candidates.values()) + unresolved,
            discovery, validation, identify, complete, discovered, heldout,
            dict(limit=self.max_candidates, omitted_at_least=omitted))
