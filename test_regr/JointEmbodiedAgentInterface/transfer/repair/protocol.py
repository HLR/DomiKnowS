"""Frozen experiment specification, selection, and target reservation contracts."""
from pathlib import Path
from ..protocol import read,freeze,sha,digest

VERSION='fresh-source-decoder-repair-v1'
ARMS=('native','joint')
RECIPE={'epochs':4,'seed':42,'batch_size':8,'lora_lr':5e-6,'graph_lr':1e-4,
        'weight_decay':.01,'warmup_fraction':.1,'preservation_kl_weight':.1,
        'eai_weight':1.,'primary_role_hints':False,'grounding_residual':False,
        'training_dfa':False,'allowed_mass_weight':0.,'source_only':True}
RECIPE['quant_matmul']='bitsandbytes-default-nf4-bf16-v1'


def anonymous_entities(count):return [f'entity_{i:03d}' for i in range(count)]


def normalize_anonymous_prediction(plan,entity_count,pointer_keys):
    """Serialization only: private canonical names cannot enter this API.

    Unknown names must not reach source metrics, whose entity grounding can
    otherwise resolve a private name as if it were an observed integer ID.
    Integer choices are never replaced or filtered by semantic/task facts.
    """
    from ..matched_baseline.protocol import normalize_ids
    result,changes=normalize_ids(plan,anonymous_entities(entity_count))
    for operation in result:
        for role,value in operation.get('params',{}).items():
            if role in pointer_keys and type(value) is not int:
                raise ValueError('unknown public entity ID alias for '+role)
    return result,changes


def verify(c):
    if c['recipe']!=RECIPE:raise ValueError('frozen repair recipe differs')
    if sha(c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('source manifest changed')
    for path,expected in c['input_hashes'].items():
        if sha(path)!=expected:raise ValueError('immutable input changed: '+path)
    for path,expected in read(c['source_manifest'])['files'].items():
        if sha(Path(c['source'])/path)!=expected:raise ValueError('source changed: '+path)


def select(rows,base_progress):
    if len(rows)!=4 or {r['epoch'] for r in rows}!={1,2,3,4}:raise ValueError('four complete source validations required')
    eligible=[r for r in rows if r['eai_successes']>=47 and r['progress_score']>=base_progress-1]
    ranked=sorted(eligible or rows,key=lambda r:(-r['progress_score'],-r['source_pointer_accuracy'],r['epoch']))
    return {'qualified':bool(eligible),'selected':ranked[0] if eligible else None,
            'diagnostic_best':ranked[0],'native_base_progress':base_progress,
            'selection':'source validation only; greedy generation; fixed gates'}


def reject_reserved(manifest,reservation):
    """Apply before any target training or validation, including older manifests."""
    reserved={(r['suite'],r['task'],r['state_index']) for r in reservation['episodes']}
    for episode in manifest['episodes']:
        if not all(k in episode for k in ('suite','task','state_index')):continue
        if episode['split'] in ('train','training','validation','valid') and (
            episode['suite'],episode['task'],episode['state_index']) in reserved:
            raise ValueError('confirmation state is reserved; frozen old manifest is not a training authorization')


def checkpoint_kind(payload,arm):
    if payload.get('format')!=VERSION or payload.get('arm')!=arm:
        raise ValueError('incompatible native-JSON / Joint-label checkpoint')
