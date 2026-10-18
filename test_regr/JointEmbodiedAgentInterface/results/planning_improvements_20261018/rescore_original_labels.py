"""Run on gpu2 in /workspace; rescore saved predictions without changing data."""
from pathlib import Path
import hashlib
import json
import math

from test_regr.VLABenchAgentInterface.dataset import load_planning_examples, deterministic_split
from test_regr.VLABenchAgentInterface.planning_metrics import PlanningMetrics, score_planning_metrics

RUN = Path('/home/auszok/evaluation_runs/planning_improvements_20261018T174007Z')
BACKUP = Path('/home/auszok/density_label_repairs/20261018T150922Z')
DATA = Path('/workspace/test_regr/VLABenchAgentInterface/data/planning')
manifest = json.loads((BACKUP / 'manifest.json').read_text())
originals = {}
for entries in manifest.values():
    if not isinstance(entries, list):
        continue
    for item in entries:
        if not isinstance(item, dict) or not item.get('path', '').endswith('/output/operation_sequence.json'):
            continue
        raw = (BACKUP / item['backup']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['before_sha256']
        episode = Path(item['path']).parent.parent.relative_to(DATA).as_posix()
        originals[episode] = json.loads(raw)

examples = deterministic_split(load_planning_examples(DATA), seed=42)['test']
by_id = {e.episode_id: e for e in examples}
source = RUN / 'grounded_visual.predictions.jsonl'
rows = [json.loads(line) for line in source.read_text().splitlines()]
assert len(rows) == len(by_id) == 450
assert len({row['episode_id'] for row in rows}) == 450
assert {row['episode_id'] for row in rows} == set(by_id)
corrected, original = PlanningMetrics(), PlanningMetrics()
changes = []
density = []
for row in rows:
    e = by_id[row['episode_id']]
    old_reference = originals.get(e.episode_id, e.operation_sequence)
    before = score_planning_metrics(row['prediction'], e.operation_sequence, e.dependency, entity_table=e.entities)
    after = score_planning_metrics(row['prediction'], old_reference, e.dependency, entity_table=e.entities)
    corrected.update(before, task=e.task, dimension=e.dimension)
    original.update(after, task=e.task, dimension=e.dimension)
    if e.episode_id in originals:
        changes.append({'episode_id': e.episode_id, 'prediction': row['prediction'], 'original_reference': old_reference,
                        'corrected_reference': e.operation_sequence, 'original_scores': after, 'corrected_scores': before})
    if e.task == 'density_qa':
        density.append({'episode_id': e.episode_id, 'original_scores': after, 'corrected_scores': before})

saved = json.loads((RUN / 'grounded_visual.json').read_text())
for key, value in corrected.result()['overall'].items():
    assert math.isclose(value, saved['overall'][key], abs_tol=1e-9), key
assert math.isclose(corrected.result()['Macro'], saved['macro_score'], abs_tol=1e-9)
result = {'protocol': 'Original unmodified density labels from hash-verified pre-repair backups; all other labels unchanged.',
          'condition': 'retained champion, grounded Complex planning, visual QA',
          'generation_rerun': False, 'examples': 450, 'generation_failures': saved['generation_failures'],
          'semantic_search_limit_failures': saved['semantic_search_limit_failures'],
          'predictions_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'repair_manifest_sha256': hashlib.sha256((BACKUP/'manifest.json').read_bytes()).hexdigest(),
          'changed_test_references': len(changes), 'corrected_label_score_reproduced': True,
          'original_labels': original.result(), 'corrected_labels': corrected.result(),
          'affected_episodes': changes, 'density_episodes': density}
(RUN / 'grounded_visual.original_labels.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('affected_episodes','density_episodes')}, indent=2))
