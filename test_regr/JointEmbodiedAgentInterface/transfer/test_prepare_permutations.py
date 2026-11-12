from pathlib import Path
from PIL import Image
import pytest
from .annotations import render, verify
from .prepare_permutations import prepare
from .protocol import freeze, read, sha
from .permutation import verified_episode_permutation
from .protocol import public_input


def test_materialization_freezes_references_without_parsing_or_transforming_gold(tmp_path):
    raw = tmp_path/'raw.png'; Image.new('RGB', (96, 48), (30, 70, 110)).save(raw)
    numbered = tmp_path/'numbered.png'; centers = [(0, 10, 10), (1, 40, 10), (2, 70, 10)]
    render(Image.open(raw), centers, style='rlbench-v33').save(numbered)
    private = tmp_path/'private'; private.write_bytes(b'OPAQUE SCORING BYTES - MUST NOT BE PARSED')
    public_path = tmp_path/'public.json'
    freeze(public_path, {'instruction': 'put the object in the bowl',
        'entities': ['entity_000', 'entity_001', 'entity_002'], 'movable': ['entity_000'],
        'positions': {'entity_000': [0, 0, 0], 'entity_001': [1, 0, 0], 'entity_002': [2, 0, 0]},
        'images': [str(raw), str(numbered)], 'visible_entity_ids': [0, 1, 2]})
    files = {str(p): sha(p) for p in (raw, numbered, private, public_path)}
    records = [{'episode_id': f'case/{i}', 'split': 'evaluation', 'generation_input': str(public_path),
                'scoring_private': str(private), 'files': files} for i in range(100)]
    manifest = tmp_path/'source_manifest.json'; freeze(manifest, {'episodes': records})
    proof = verify(raw, numbered, centers, style='rlbench-v33', entity_count=3)
    binding = tmp_path/'bindings.json'
    freeze(binding, {'verified': True, 'episodes': [{'episode_id': r['episode_id'], **proof} for r in records]})
    config = {'seed': 42, 'datasets': {'rlbench': {'manifest': str(manifest)}}}
    prepare(config, 'rlbench', binding, tmp_path/'output')
    assert private.read_bytes() == b'OPAQUE SCORING BYTES - MUST NOT BE PARSED'
    seen = set()
    for trial in range(5):
        current = read(tmp_path/'output'/f'trial_{trial:03d}'/'manifest.json')['episodes']
        assert len(current) == len({r['episode_id'] for r in current}) == 100
        assert all(r['scoring_private'] == str(private) and r['original_reference_sha256'] == sha(private) for r in current)
        for record in current:
            permutation = verified_episode_permutation(record, public_input(record))
            assert list(permutation.order) == record['new_to_old']
        bad = {**current[0], 'old_to_new': list(reversed(current[0]['old_to_new']))}
        with pytest.raises(ValueError, match='inverse pointer'):
            verified_episode_permutation(bad, public_input(bad))
        seen.add(tuple(current[0]['new_to_old']))
    assert len(seen) == 5
    prepare(config, 'rlbench', binding, tmp_path/'output')  # exact resume
    private.write_bytes(b'changed hidden gold')
    with pytest.raises(ValueError, match='frozen episode or reference changed'):
        prepare(config, 'rlbench', binding, tmp_path/'output')
