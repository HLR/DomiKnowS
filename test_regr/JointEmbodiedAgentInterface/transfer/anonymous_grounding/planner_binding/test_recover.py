from pathlib import Path
from .recover import retry_command,redirected_path


def test_retry_changes_only_allocation_name_and_output(tmp_path):
    original=['docker','run','--name','old','--gpus','device=1','-e','CUDA_VISIBLE_DEVICES=old',
        '-e','BINDING_GPU_UUID=old','image','-m','module','--config','frozen-full.json','--output','old-output']
    command,output=retry_command(original,tmp_path,'training-full-retry001',2,'new-uuid')
    assert command[command.index('--config')+1]=='frozen-full.json'
    assert command[command.index('--gpus')+1]=='device=2' and 'CUDA_VISIBLE_DEVICES=new-uuid' in command
    assert output==tmp_path/'arms/training-full-retry001/checkpoints' and original[-1]=='old-output'
    expected=original.copy()
    for flag in ('--name','--gpus','--output'):expected[expected.index(flag)+1]=command[command.index(flag)+1]
    expected=[value.replace('=old','=new-uuid') if value.startswith(('CUDA_VISIBLE_DEVICES=','BINDING_GPU_UUID=')) else value for value in expected]
    assert command==expected


def test_recovery_aliases_preserve_masked_and_failed_original_artifacts(tmp_path):
    root=tmp_path/'run';retry='training-full-retry001'
    assert redirected_path(root/'arms/full/checkpoints/best.json',root,retry)==root/'arms'/retry/'checkpoints/best.json'
    assert redirected_path(root/'jobs/training-full.json',root,retry)==root/'jobs'/f'{retry}.json'
    for path in [root/'arms/masked/checkpoints/best.json',root/'full.json',root/'jobs/training-full-failed.json']:
        assert redirected_path(path,root,retry)==path
