import numpy as np
from PIL import Image, ImageDraw
import pytest
from .annotations import render, verify, write_counterfactual
from .permutation import EntityPermutation


@pytest.mark.parametrize('style', ['libero-v3', 'rlbench-v33'])
def test_counterfactual_uses_verified_physical_locations_and_three_cycle_labels(tmp_path, style):
    raw = Image.new('RGB', (128, 48), (30, 90, 120))
    centers = [(0, 10, 10), (1, 50, 10), (2, 90, 10)]
    rp = tmp_path/'raw.png'; raw.save(rp)
    npth = tmp_path/'numbered.png'; render(raw, centers, style=style).save(npth)
    proof = write_counterfactual(rp, npth, centers, style=style,
        permutation=EntityPermutation((2, 0, 1)), output=tmp_path/'permuted.png')
    expected = raw.copy(); draw = ImageDraw.Draw(expected)
    # Explicit physical ground truth: old A→new 1, B→new 2, C→new 0.
    for (_, x, y), label in zip(centers, ['1', '2', '0']):
        bounds = (x-3, y-3, x+14, y+13) if style == 'libero-v3' else (x-2, y-2, x+16, y+14)
        draw.rectangle(bounds, fill='white', outline='black'); draw.text((x, y), label, fill='black')
    assert np.array_equal(np.asarray(Image.open(tmp_path/'permuted.png')), np.asarray(expected))
    assert proof['new_to_old'] == [2, 0, 1] and proof['old_to_new'] == [1, 2, 0]
    assert np.array_equal(np.asarray(Image.open(rp)), np.asarray(raw))


def test_unverified_binding_stops_before_creating_perturbation(tmp_path):
    raw = Image.new('RGB', (80, 40), 'gray'); rp = tmp_path/'raw.png'; raw.save(rp)
    numbered = render(raw, [(0, 10, 10)], style='libero-v3')
    numbered.putpixel((40, 20), (1, 2, 3)); path = tmp_path/'numbered.png'; numbered.save(path)
    output = tmp_path/'must-not-exist.png'
    with pytest.raises(ValueError, match='reconstruction differs'):
        write_counterfactual(rp, path, [(0, 10, 10)], style='libero-v3',
            permutation=EntityPermutation((1, 0)), output=output)
    assert not output.exists()


def test_no_silent_marker_guessing_or_out_of_frame_placement():
    raw = Image.new('RGB', (32, 32))
    with pytest.raises(ValueError, match='unsupported annotation'):
        render(raw, [], style='guessed-source-style')
    with pytest.raises(ValueError, match='outside'):
        render(raw, [(0, -1, 0)], style='libero-v3')


def test_sensor_projection_drift_recovered_only_by_exact_public_pixel_match(tmp_path):
    from .annotations import recover_centers
    raw = Image.new('RGB', (128, 48), (35, 70, 110))
    centers = [(0, 10, 10), (1, 50, 10), (2, 90, 10)]
    rp = tmp_path/'raw.png'; raw.save(rp)
    npth = tmp_path/'numbered.png'; render(raw, centers, style='libero-v3').save(npth)
    projected = [(0, 11, 10), (1, 50, 11), (2, 89, 9)]
    proof = recover_centers(rp, npth, projected, style='libero-v3', entity_count=3)
    assert sorted(proof['centers']) == centers and proof['passed']
    # A broken public stamp cannot be accepted by raising the pixel tolerance.
    broken = Image.open(npth); broken.putpixel((10, 10), (1, 2, 3)); broken.save(npth)
    with pytest.raises(ValueError, match='exact locations'):
        recover_centers(rp, npth, projected, style='libero-v3', entity_count=3)


def test_public_marker_occlusion_order_is_recovered_instead_of_using_native_iteration(tmp_path):
    from .annotations import recover_centers
    raw = Image.new('RGB', (64, 40), (35, 70, 110))
    original = [(0, 10, 10), (1, 18, 10)]
    rp = tmp_path/'raw.png'; raw.save(rp)
    npth = tmp_path/'numbered.png'; render(raw, original, style='libero-v3').save(npth)
    proof = recover_centers(rp, npth, list(reversed(original)), style='libero-v3', entity_count=2)
    assert proof['passed'] and proof['centers'] == original


def test_fully_occluded_marker_remains_occluded_for_every_permuted_id(tmp_path):
    from .annotations import recover_centers
    raw = Image.new('RGB', (64, 40), (35, 70, 110))
    original = [(0, 10, 10), (1, 10, 10)]
    rp = tmp_path/'raw.png'; raw.save(rp)
    npth = tmp_path/'numbered.png'; render(raw, original, style='libero-v3').save(npth)
    proof = recover_centers(rp, npth, list(reversed(original)), style='libero-v3', entity_count=2)
    assert proof['fully_occluded_markers'] == [0]
    for perm in [EntityPermutation((0, 1)), EntityPermutation((1, 0))]:
        assert np.array_equal(np.asarray(render(raw, proof['centers'], style='libero-v3', permutation=perm)),
                              np.asarray(render(raw, original, style='libero-v3', permutation=perm)))


def test_unique_complete_public_stamp_can_resolve_large_projection_offset(tmp_path):
    from .annotations import recover_centers
    raw = Image.new('RGB', (128, 64), (35, 70, 110))
    rp = tmp_path/'raw.png'; raw.save(rp)
    npth = tmp_path/'numbered.png'
    render(raw, [(0, 20, 20)], style='libero-v3').save(npth)
    proof = recover_centers(rp, npth, [(0, 80, 20)], style='libero-v3', entity_count=1)
    assert proof['centers'] == [(0, 20, 20)]
    assert proof['global_full_stamp_markers'] == [0]
    # Two equally exact public IDs cannot be resolved by the global fallback.
    render(raw, [(0, 20, 20), (0, 45, 20)], style='libero-v3').save(npth)
    with pytest.raises(ValueError, match='exact locations'):
        recover_centers(rp, npth, [(0, 100, 20)], style='libero-v3', entity_count=1)
