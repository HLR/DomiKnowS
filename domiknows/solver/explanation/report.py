"""Detached reports; live diagnostic sessions are private and never serialized."""
import copy
import json


class ExplanationReport:
    def __init__(self, payload, session=None):
        self._payload = copy.deepcopy(payload)
        self._session = session

    def to_dict(self):
        return copy.deepcopy(self._payload)

    def to_json(self, **kwargs):
        return json.dumps(self._payload, allow_nan=False, **kwargs)

    def render_text(self):
        p = self._payload
        lines = [f"{p['requested_backend']} explanation: {p['status']}"]
        lines.append(f"Actual backend: {p['actual_backend']}")
        for prediction in p['predictions']:
            label = f"{prediction['id']}: local={prediction['local_label']}"
            if 'inferred_label' in prediction:
                label += f", constrained={prediction['inferred_label']}"
            lines.append(label)
        if p['objective'] is not None:
            lines.append(f"Objective: {p['objective']['value']} ({p['objective']['mode']}, "
                         f"{p['objective']['direction']})")
        for name, answer in p['answers'].items():
            lines.append(f"{name}: {answer}")
        for conflict in p['conflicts']:
            lines.append(f"Candidate run {conflict.get('run_index')}, hypotheses "
                         f"{conflict.get('hypotheses', {})}: irreducible conflict "
                         f"(not necessarily smallest): {conflict}")
        evaluations = [r for r in p['evidence'] if r['kind'] == 'evaluation']
        for record in evaluations[:50]:
            bindings = json.dumps(record['bindings'], sort_keys=True)
            result = json.dumps(record['result'], sort_keys=True)
            lines.append(f"[{record['id']}] {record['phase']} {record['rule']} "
                         f"{record['operator']}: bindings={bindings[:400]}, result={result[:400]}")
        if len(evaluations) > 50:
            lines.append('More operator evidence is available in to_dict()/to_json().')
        if p.get('comparison'):
            lines.append(f"Counterfactual comparison: {p['comparison']}")
        lines.extend(p['unsupported'])
        if p['truncation']['omitted']:
            lines.append(f"Evidence truncated: {p['truncation']['omitted']} records omitted")
        return '\n'.join(lines)

    def counterfactual(self, assignments):
        """Map prediction IDs to bool (binary) or class-name (enum) values."""
        if self._session is None:
            raise RuntimeError('This detached report has no diagnostic session')
        result = self._session.counterfactual(self, assignments)
        self._payload['counterfactuals'].append(result.to_dict())
        return result

    def summarize(self, summarizer):
        """Adapter receives a detached report and returns summary/evidence_ids."""
        payload = self.to_dict()
        payload.pop('counterfactuals', None)
        valid = {r['id'] for r in self._payload['evidence']}
        try:
            result = summarizer(payload)
            if not isinstance(result, dict) or not isinstance(result.get('summary'), str):
                raise ValueError('Adapter must return summary and evidence_ids')
            references = result.get('evidence_ids')
            if (not isinstance(references, list) or not references
                    or any(not isinstance(r, str) or r not in valid for r in references)):
                raise ValueError('Summary must cite known evidence IDs')
            return {'status': 'complete', 'interpretation': 'LLM-generated',
                    'summary': result['summary'], 'evidence_ids': list(references)}
        except Exception as error:
            return {'status': 'failed', 'error_type': type(error).__name__,
                    'summary': None, 'evidence_ids': []}
