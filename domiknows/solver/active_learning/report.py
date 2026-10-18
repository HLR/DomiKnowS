"""Detached, strictly JSON-safe annotation queues."""
import copy
import json


class AnnotationSelection:
    def __init__(self, payload):
        self._payload = copy.deepcopy(payload)

    def to_dict(self):
        return copy.deepcopy(self._payload)

    def to_json(self, **options):
        return json.dumps(self._payload, allow_nan=False, **options)

    def render_text(self):
        p = self._payload
        lines = [f"Annotation queue ({p['mode']}): {len(p['selected'])} examples"]
        for example in p['selected']:
            lines.append(f"Example {example['sample_id']}: score={example['score']:.6f}")
            for target in example['targets']:
                lines.append(f"  {target['path']} instance={target['instance_id']}: "
                             f"score={target['score']:.6f}, suggested={target['suggested_label']}")
            lines.extend('  Unavailable: ' + message for message in example['unsupported'])
        for excluded in p['excluded']:
            lines.append(f"Excluded {excluded['sample_id']}: {excluded['reason']}")
        lines.append('Suggestions are model predictions; rule links indicate involvement, not causation.')
        return '\n'.join(lines)
