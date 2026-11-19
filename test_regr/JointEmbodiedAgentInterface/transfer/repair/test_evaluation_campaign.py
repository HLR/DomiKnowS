import json
import pytest
from .evaluation_campaign import require_unopened_training, freeze_parent_selection


def test_handoff_refuses_scored_or_changed_campaign():
    require_unopened_training({'stage':'two-fresh-four-epoch-source-arms'}, [])
    with pytest.raises(ValueError, match='unopened'):
        require_unopened_training({'stage':'two-fresh-four-epoch-source-arms'}, ['predictions.jsonl'])
    with pytest.raises(ValueError, match='active source training'):
        require_unopened_training({'stage':'fixed-eight-condition-confirmation'}, [])


def test_selection_reuses_source_choices_and_failed_status(tmp_path):
    parent=tmp_path/'parent';root=tmp_path/'strict'
    for name in ('native_base/initialization.pt','arms/joint/initialization.pt',
                 'arms/native/epoch_002.pt','arms/joint/epoch_003.pt'):
        p=parent/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
    for arm,epoch,qualified in [('native',2,True),('joint',3,False)]:
        chosen={'checkpoint':str(parent/'arms'/arm/f'epoch_{epoch:03d}.pt'),
            'epoch':epoch,'progress_score':45.5}
        p=parent/'arms'/arm/'selection.json'
        p.write_text(json.dumps({'selected':chosen if qualified else None,
            'diagnostic_best':chosen,'qualified':qualified}))
    systems=freeze_parent_selection(root,parent)
    assert systems['native-trained']['source_epoch']==2
    assert systems['native-trained']['qualified']
    assert not systems['joint-trained']['qualified']
    assert systems['joint-trained']['source_epoch']==3
    changed=parent/'arms/native/selection.json'
    value=json.loads(changed.read_text());value['qualified']=False;changed.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='resume fingerprint differs'):
        freeze_parent_selection(root,parent)
