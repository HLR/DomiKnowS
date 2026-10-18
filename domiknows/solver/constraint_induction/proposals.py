"""Auditable human review, guarded registration, and isolated comparisons."""
import copy
import json
from time import perf_counter

from domiknows.solver.explanation.evidence import native
from domiknows.solver.explanation.session import _lock, _graphs, _ontology_transaction
from domiknows.solver.incremental.state import Snapshot, signature
from .data import extract, sample_key
from .specs import digest
from .inducer import metrics


class InductionPreview:
    def __init__(self, payload):
        self._payload = copy.deepcopy(payload)

    def to_dict(self):
        return copy.deepcopy(self._payload)

    def to_json(self, **options):
        return json.dumps(self._payload, allow_nan=False, **options)

    def render_text(self):
        lines = ['Constraint impact preview: ' + ', '.join(self._payload['candidate_ids'])]
        for item in self._payload['results']:
            lines.append(f"Sample {item['sample_id']} ({item['mode']}): "
                         f"{item['baseline']['status']} -> {item['proposed']['status']}; "
                         f"{len(item['changed_labels'])} changed labels")
            if item['objective_difference'] is not None:
                lines.append(f"  Objective difference: {item['objective_difference']}")
            lines.extend('  ' + message for message in item['unsupported'])
        return '\n'.join(lines)


class ConstraintProposals:
    def __init__(self, inducer, candidates, discovery, validation, identify, complete,
                 discovered, heldout, truncation):
        self.inducer, self.identify = inducer, identify
        self.discovery, self.validation, self.complete = discovery, validation, copy.deepcopy(complete)
        self._candidates, self._events, self._decisions, self._activations = {}, [], {}, {}
        self._activation_fingerprints = {}
        self._truncation = truncation
        self._data_revision = 0
        self._signature = self._snapshot()
        for raw in candidates:
            candidate = dict(id=raw['id'], version=1, origins=raw['origins'], versions=[], status='proposed',
                             specification=raw['specification'], unresolved=raw.get('error'), approval=None)
            self._candidates[candidate['id']] = candidate
            self._validate(candidate, discovered, heldout)

    def _snapshot(self):
        concepts = tuple(self.inducer.vocabulary.concepts.values())
        return dict(samples=[(id(root), sample_key(self.identify(root)), Snapshot.read(root, concepts))
                             for root in self.discovery + self.validation],
                    configuration=signature((self.inducer.policy, self.inducer.key, self.inducer.templates,
                                             self.inducer.seed, self.inducer.rounds)))

    def _candidate(self, identifier):
        if identifier not in self._candidates:
            raise ValueError('Unknown candidate identifier')
        return self._candidates[identifier]

    def _validate(self, candidate, discovered, heldout):
        if candidate['specification'] is None:
            candidate.update(validation_status='unresolved', metrics=None, rank_score=None)
            return
        all_metrics = {}
        for split, samples in (('discovery', discovered), ('validation', heldout)):
            all_metrics[split] = {source: metrics(samples, candidate['specification'], self.inducer.vocabulary,
                source, seed=self.inducer.seed, rounds=self.inducer.rounds,
                evidence_limit=self.inducer.evidence_limit) for source in ('annotation', 'prediction')}
        policy = self.inducer.policy
        reference = all_metrics['validation']['annotation']
        evidence_source = 'annotation'
        if reference['covered_examples'] == 0 and policy['allow_prediction_only']:
            reference, evidence_source = all_metrics['validation']['prediction'], 'prediction'
        reasons = []
        if candidate['specification']['template'] not in self.inducer.templates:
            reasons.append('template is no longer enabled')
        if not heldout:
            reasons.append('missing held-out examples')
        if reference['covered_examples'] < policy['min_support_examples']:
            reasons.append('insufficient applicable examples')
        if reference['coverage'] < policy['min_coverage']:
            reasons.append('insufficient coverage')
        if reference['violation_rate'] is not None and reference['violation_rate'] > policy['max_violation_rate']:
            reasons.append('violations exceed tolerance')
        std = reference['stability']['std']
        if std is not None and std > policy['max_stability_std']:
            reasons.append('unstable across example resamples')
        candidate.update(metrics=all_metrics, validation_status='passed' if not reasons else 'needs_review',
            validation_reasons=reasons, validation_source=evidence_source,
            rank_score=(reference['confidence'] * reference['coverage'] * (1-(std or 0)))
                       if reference['confidence'] is not None else None,
            data_revision=self._data_revision)

    def to_dict(self):
        candidates = list(self._candidates.values())
        candidates.sort(key=lambda c: (-(c['rank_score'] if c['rank_score'] is not None else -1), c['id']))
        return copy.deepcopy(dict(schema_version=1, candidates=candidates, events=self._events,
            policy=self.inducer.policy, truncation=self._truncation,
            datasets=dict(discovery_sample_ids=[native(self.identify(r)) for r in self.discovery],
                          validation_sample_ids=[native(self.identify(r)) for r in self.validation],
                          complete_domains=native(self.complete)),
            settings=dict(key=list(self.inducer.key), seed=self.inducer.seed,
                          templates=list(self.inducer.templates), bootstrap_rounds=self.inducer.rounds),
            data_revision=self._data_revision,
            review_semantics='Proposals are empirical patterns or domain requirements; reviewer identity is caller-provided metadata.'))

    def to_json(self, **options):
        return json.dumps(self.to_dict(), allow_nan=False, **options)

    def render_text(self):
        lines = ['Constraint proposals: human review required before activation']
        for c in self.to_dict()['candidates']:
            lines.append(f"{c['id']} v{c['version']}: {c['status']}; validation={c['validation_status']}")
            lines.append('  ' + (json.dumps(c['specification'], sort_keys=True) if c['specification'] else c['unresolved']))
            if c.get('metrics'):
                for source, values in c['metrics']['validation'].items():
                    lines.append(f"  Held-out {source}: applicable examples={values['covered_examples']}, "
                                 f"confidence={values['confidence']}, unknown groundings={values['unknown_groundings']}")
        return '\n'.join(lines)

    def _event(self, action, candidate, reviewer, rationale, **fields):
        if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(rationale, str) or not rationale.strip():
            raise ValueError('A reviewer identity and nonempty rationale are required')
        event = dict(id=f'decision-{len(self._events)}', action=action, candidate_id=candidate['id'],
                     version=candidate['version'], reviewer=reviewer, rationale=rationale,
                     specification_digest=digest(candidate['specification']), data_revision=self._data_revision, **fields)
        self._events.append(copy.deepcopy(event))
        return event

    def approve(self, identifier, *, reviewer, rationale, override=False):
        c = self._candidate(identifier)
        if c['status'] == 'active':
            raise ValueError('Candidate is already active')
        if c['specification'] is None:
            raise ValueError('Resolve the structured specification before approval')
        if c['specification']['template'] not in self.inducer.templates:
            raise ValueError('This template is disabled')
        if self._snapshot() != self._signature:
            raise ValueError('Examples or rules changed; revalidate before approval')
        if not self.validation or c.get('data_revision') != self._data_revision:
            raise ValueError('Held-out validation is required before approval')
        if c['validation_status'] != 'passed' and not override:
            raise ValueError('Validation needs review; an explicit override with rationale is required')
        decision = self._event('approve', c, reviewer, rationale, override=bool(override),
            validation_status=c['validation_status'], validation_source=c['validation_source'],
            validation_metrics={split: {source: {key: copy.deepcopy(value) for key, value in values.items()
                                if key != 'evidence'} for source, values in sources.items()}
                                for split, sources in c['metrics'].items()},
            validation_policy=copy.deepcopy(self.inducer.policy),
            dataset_ids=dict(discovery=[native(self.identify(r)) for r in self.discovery],
                             validation=[native(self.identify(r)) for r in self.validation]))
        self._decisions[decision['id']] = copy.deepcopy(decision)
        c.update(status='approved', approval=decision['id'])
        return copy.deepcopy(decision)

    def reject(self, identifier, *, reviewer, rationale):
        c = self._candidate(identifier)
        if c['status'] == 'active':
            raise ValueError('Deactivate the registered rule before rejecting it')
        event = self._event('reject', c, reviewer, rationale)
        c.update(status='rejected', approval=None)
        return copy.deepcopy(event)

    def edit(self, identifier, specification, *, reviewer, rationale):
        c = self._candidate(identifier)
        if c['status'] == 'active':
            raise ValueError('Deactivate before editing an active version')
        spec = self.inducer.vocabulary.validate(specification)
        if spec['template'] not in self.inducer.templates:
            raise ValueError('Template is not enabled')
        self._event('edit', c, reviewer, rationale, next_version=c['version']+1)
        old = {key: copy.deepcopy(value) for key, value in c.items() if key != 'versions'}
        c['versions'].append(old)
        c.update(version=c['version']+1, specification=spec, status='proposed', approval=None,
                 unresolved=None, validation_status='unvalidated', metrics=None, rank_score=None,
                 data_revision=None)
        return copy.deepcopy(c)

    def revalidate(self):
        vocabulary = self.inducer.vocabulary
        graphs = list(dict.fromkeys(g for r in self.discovery + self.validation for g in _graphs(r, tuple(vocabulary.concepts.values()))))
        with _lock, _ontology_transaction(graphs):
            discovered = extract(self.discovery, vocabulary, self.identify, self.complete, self.inducer.key)
            heldout = extract(self.validation, vocabulary, self.identify, self.complete, self.inducer.key)
        self._data_revision += 1
        self._signature = self._snapshot()
        for c in self._candidates.values():
            if c['status'] == 'active':
                continue  # Activated provenance is a frozen reviewed version.
            c['approval'] = None
            if c['status'] == 'approved':
                c['status'] = 'proposed'
            self._validate(c, discovered, heldout)
        return self.to_dict()

    def activate(self, decision, graph):
        if not isinstance(decision, dict) or decision != self._decisions.get(decision.get('id')):
            raise ValueError('Use an approval decision issued by this proposal set')
        c = self._candidate(decision['candidate_id'])
        if c['status'] != 'approved' or c['approval'] != decision['id'] or c['version'] != decision['version']:
            raise ValueError('Approval is obsolete or invalidated')
        if digest(c['specification']) != decision['specification_digest'] or self._snapshot() != self._signature:
            raise ValueError('Specification or validation data changed; revalidate and review again')
        vocabulary = self.inducer.vocabulary
        graphs = list(dict.fromkeys(g for r in self.discovery + self.validation for g in _graphs(r, tuple(vocabulary.concepts.values()))))
        with _lock, _ontology_transaction(graphs):
            retained = self._registered_version(c)
            head, nodes = retained if retained is not None else vocabulary.compile(c['specification'], c['id'], c['version'])
            context = head.graph
            while context is not None and context is not graph:
                context = getattr(context, 'sup', None)
            if context is None:
                raise ValueError('Requested graph does not contain the candidate concepts')
        with _lock:
            for node in nodes:
                if node.lcName in node.graph.logicalConstrains and node.graph.logicalConstrains[node.lcName] is not node:
                    raise ValueError('Candidate version is already registered')
            provenance = dict(candidate_id=c['id'], version=c['version'], decision=copy.deepcopy(decision),
                specification=copy.deepcopy(c['specification']), origins=copy.deepcopy(c['origins']),
                metrics=copy.deepcopy(c['metrics']))
            for node in nodes:
                node.active = True
                node.induction_provenance = copy.deepcopy(provenance)
                node.graph.logicalConstrains[node.lcName] = node
            c['status'] = 'active'
            activation = self._event('activate', c, decision['reviewer'], decision['rationale'],
                                   approval_id=decision['id'], rule_path=head.graph.fullname+'/'+head.lcName)
            self._activations[activation['id']] = (head, nodes, c['id'], c['version'])
            self._activation_fingerprints[activation['id']] = [signature((n.e, n.p, n.headLC)) for n in nodes]
            c['active_activation'] = activation['id']
            # Our own reviewed graph edit is expected. It does not invalidate
            # other decisions validated against the same observed examples.
            self._signature = self._snapshot()
            return copy.deepcopy(activation)

    def _registered_version(self, c):
        for activation_id, (head, nodes, identifier, version) in reversed(list(self._activations.items())):
            if identifier != c['id'] or version != c['version']:
                continue
            if (any(n.graph.logicalConstrains.get(n.lcName) is not n for n in nodes) or
                    [signature((n.e, n.p, n.headLC)) for n in nodes] != self._activation_fingerprints[activation_id]):
                raise ValueError('Registered version was modified externally; edit a new version')
            return head, nodes
        return None

    def deactivate(self, activation, *, reviewer, rationale, _action='deactivate'):
        if not isinstance(activation, dict) or activation.get('id') not in self._activations:
            raise ValueError('Unknown activation')
        recorded = next(e for e in self._events if e['id'] == activation['id'])
        if activation != recorded:
            raise ValueError('Activation record was modified')
        head, nodes, identifier, version = self._activations[activation['id']]
        c = self._candidate(identifier)
        if c['version'] != version or c['status'] != 'active' or c.get('active_activation') != activation['id']:
            raise ValueError('This activation is no longer active')
        if head.graph.logicalConstrains.get(head.lcName) is not head:
            raise ValueError('Registered rule was replaced externally')
        was_current = self._snapshot() == self._signature
        event = self._event(_action, c, reviewer, rationale, activation_id=activation['id'])
        for node in nodes:
            node.active = False
        c.update(status='deactivated', approval=None, active_activation=None)
        if was_current:
            self._signature = self._snapshot()
        return copy.deepcopy(event)

    def rollback(self, activation, *, reviewer, rationale):
        return self.deactivate(activation, reviewer=reviewer, rationale=rationale, _action='rollback')

    def preview(self, candidate_ids, examples, *, modes=('ilp', 'circuit', 'tnorm'), **options):
        identifiers = [candidate_ids] if isinstance(candidate_ids, str) else list(candidate_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError('Preview requires unique candidate IDs')
        candidates = [self._candidate(identifier) for identifier in identifiers]
        if any(c['specification'] is None for c in candidates):
            raise ValueError('Unresolved candidates cannot be compiled for preview')
        if any(mode not in ('ilp', 'circuit', 'tnorm') for mode in modes) or not modes:
            raise ValueError('Unknown or empty preview backend list')
        roots = list(examples)
        if len({sample_key(self.identify(r)) for r in roots}) != len(roots):
            raise ValueError('Preview sample IDs must be unique')
        started, results = perf_counter(), []
        vocabulary = self.inducer.vocabulary
        concepts = tuple(vocabulary.concepts.values())
        graphs = list(dict.fromkeys(g for r in roots + self.discovery + self.validation for g in _graphs(r, concepts)))
        with _lock, _ontology_transaction(graphs):
            baseline = {(id(root), mode): root.explainInference(*concepts, mode=mode,
                        _preserve_inputs=True, **options).to_dict() for root in roots for mode in modes}
            for c in candidates:
                if c['status'] != 'active':
                    retained = self._registered_version(c)
                    if retained is not None:
                        for node in retained[1]:
                            node.active = True
                    else:
                        vocabulary.compile(c['specification'], c['id'], c['version'])
            for root in roots:
                for mode in modes:
                    after = root.explainInference(*concepts, mode=mode, _preserve_inputs=True, **options).to_dict()
                    before = baseline[(id(root), mode)]
                    old = {p['id']: p for p in before['predictions']}
                    changed = [dict(id=p['id'], before=old[p['id']].get('inferred_label'), after=p['inferred_label'])
                               for p in after['predictions'] if p['id'] in old and 'inferred_label' in p
                               and old[p['id']].get('inferred_label') != p['inferred_label']]
                    delta, unsupported = None, []
                    if mode == 'ilp':
                        a, b = before['objective'], after['objective']
                        if (before['status'] == after['status'] == 'optimal' and a and b and
                                a['mode'] == b['mode'] and a['direction'] == b['direction'] and options.get('fun') is None):
                            delta = b['value']-a['value']
                        else:
                            unsupported.append('Objective differences require matching completed optimal solves and a verifiable objective configuration.')
                    elif mode == 'circuit':
                        unsupported.append('Circuit rules are evaluated under existing query semantics; adding a rule does not globally condition other queries.')
                    answers = {name: dict(before=before['answers'][name], after=after['answers'][name])
                               for name in before['answers'].keys() & after['answers'].keys()
                               if before['answers'][name] != after['answers'][name]}
                    results.append(dict(sample_id=native(self.identify(root)), mode=mode,
                        baseline=before, proposed=after, changed_labels=changed,
                        changed_answers=answers, objective_difference=delta, unsupported=unsupported,
                        semantics='logical/computational comparison; not a real-world causal claim'))
        return InductionPreview(dict(schema_version=1, candidate_ids=identifiers,
            already_active=[c['id'] for c in candidates if c['status']=='active'],
            versions={c['id']: c['version'] for c in candidates}, results=results,
            timing=dict(total_ms=(perf_counter()-started)*1000)))
