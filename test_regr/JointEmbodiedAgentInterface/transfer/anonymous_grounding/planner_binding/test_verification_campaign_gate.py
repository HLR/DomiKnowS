import pytest
from ...protocol import freeze
from .verification_campaign import wait_reproduction


def certificates(root):
    for policy in ('none','shared-verified','full-metadata'):
        freeze(root/'control_reproduction'/(policy+'.json'),{'passed':True,'policy':policy})


def test_admission_requires_reproduction_but_not_finished_independent_controls(tmp_path):
    certificates(tmp_path)
    freeze(tmp_path/'control_status.json',{'stage':'matched-controls-and-epoch1-reproduction'})
    wait_reproduction(tmp_path)
    assert not (tmp_path/'control_complete.json').exists()


def test_failed_correctness_or_mislabeled_certificate_still_blocks_admission(tmp_path):
    certificates(tmp_path)
    freeze(tmp_path/'control_status.json',{'stage':'failed','error':'input mismatch'})
    with pytest.raises(RuntimeError,match='input mismatch'):wait_reproduction(tmp_path)
    (tmp_path/'control_status.json').unlink()
    (tmp_path/'control_reproduction/none.json').unlink()
    freeze(tmp_path/'control_reproduction/none.json',{'passed':True,'policy':'another policy'})
    with pytest.raises(ValueError,match='certificate differs'):wait_reproduction(tmp_path)
