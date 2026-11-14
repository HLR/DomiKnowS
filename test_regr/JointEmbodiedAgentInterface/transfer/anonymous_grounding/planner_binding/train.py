"""Reuse the audited source trainer with explicit operation-binding components."""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys
import torch
from ...protocol import read,sha,freeze
from ..agent import load_head
from .agent import primary
from .binding import BindingHead
from .objective import planning_loss


class Prepared(Exception):pass


def main():
    from .. import train as trainer
    from .... import checkpoint as checkpoints,main as joint
    from ....models import JointQwenVLPlanner
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-only',action='store_true');p.add_argument('--resume',type=Path)
    options=p.parse_args();c=read(options.config);output=options.output
    if c['arm'] not in ('masked','full'):raise ValueError('matched CE arm required')
    for name in ('observations','features','text_manifest','external_manifest'):
        if sha(c[name])!=c[name+'_sha256']:raise ValueError('frozen campaign input changed: '+name)
    text_proof=read(c['text_manifest'])
    if text_proof['references_read'] or sha(text_proof['path'])!=text_proof['sha256']:raise ValueError('invalid public text priors')
    if any(sha(f)!=h for f,h in text_proof['public_instruction_files'].items()):raise ValueError('public source instruction changed')
    text_cache=torch.load(text_proof['path'],map_location='cpu',weights_only=True)
    state=SimpleNamespace(config=c,head=None,validation={r:Counter() for r in ('source','destination')})
    original_freeze=trainer.freeze
    def training_record(path,value):
        if Path(path)==output/'settings.json':
            value={**value,'method':'per-operation-binding-source-only','ce_arm':c['arm'],
                'initialization':'selected anonymous-ID grounding continuation epoch 2',
                'auxiliary_weight':1.,'allowed_mass_weight':.5,'full_ce_scope':'VLABench graph labels; EAI objective matched',
                'grounding_supervision':'each causal operation pointer, not an episode-level role union',
                'primary_hints':'frozen paired SigLIP, instruction phrases only',
                'inference':'planner_binding.agent required; finite learned residual, no forced pointer',
                'fresh_external_manifest_sha256':c['external_manifest_sha256']}
        return original_freeze(path,value)
    trainer.freeze=training_record
    original_load=checkpoints.load_joint_checkpoint
    def load(path,**kwargs):
        kwargs['planner']._grounding_extension_active=True
        payload=original_load(path,**kwargs)
        if Path(path)==Path(c['checkpoint']):
            state.initialization=payload;state.runtime=kwargs['runtime'];state.controller=kwargs['controller'];state.planner=kwargs['planner']
        return payload
    checkpoints.load_joint_checkpoint=load
    def head_factory(query_dim,visual_dim,hidden_dim=256):
        legacy=load_head(state.initialization,c['checkpoint'],'cpu')
        if (legacy.query_dim,legacy.visual_dim,legacy.hidden_dim)!=(query_dim,visual_dim,hidden_dim):raise ValueError('initial head dimensions differ')
        head=BindingHead(query_dim,visual_dim,hidden_dim);head.role_head.load_state_dict(legacy.state_dict(),strict=True)
        state.head=head;return head
    trainer.RoleGrounder=head_factory
    def predicted(planner,head,context,features,visible):
        effective,predictions=primary(context,text_cache,features,visible)
        state.features=features;state.visible=visible
        return effective,predictions,head.gain.expand(len(visible)+1,2)
    trainer.predicted_context=predicted
    def no_episode_union_loss(logits,visible,targets):
        return logits.sum()*0,{r:{'targets':len(ids),'observable_targets':sum(bool(visible[i]) for i in ids)} for r,ids in targets.items()}
    trainer.grounding_loss=no_episode_union_loss
    def objective(planner,runtime,original,effective,labels,visible):
        loss,terms=planning_loss(planner,runtime,original,effective,labels,visible,
            head=state.head,features=state.features,arm=c['arm'])
        if not state.head.training:
            for role,counts in terms['operation_coverage'].items():state.validation[role].update(counts)
        return loss,terms
    trainer.planning_loss=objective
    original_audits=trainer.SourceAudits
    def audits(root,fingerprint,workers=8):
        # Proofs are model-independent. Native compiler/auditor hashes are checked
        # before reuse; SourceAudits still verifies every current per-path input hash.
        source=Path(c['audit_cache']);root=Path(root);root.mkdir(parents=True,exist_ok=True)
        for name in c['audit_dependency_hashes']:
            if sha(Path(c['source'])/name)!=c['audit_dependency_hashes'][name]:raise ValueError('native audit implementation changed')
        for folder in source.iterdir():
            if folder.is_dir():shutil.copytree(folder,root/folder.name,dirs_exist_ok=True)
        return original_audits(root,fingerprint,workers=workers)
    trainer.SourceAudits=audits
    args=joint.build_parser().parse_args(read(c['original_configuration'])['arguments'])
    args.output=str(output);args.transfer_initialization_sha256=c['checkpoint_sha256']
    def save_snapshot(path,*,runtime,planner,head,controller,optimizer,epoch,next_batch,complete,best,settings_sha,controller_sha,initialization,args,initial=False):
        if path.exists():raise ValueError('preserve existing operation-binding snapshot')
        sidecar=path.with_name(path.stem+'_grounding.pt')
        torch.save({'format':'operation-binding-v1','state_dict':head.state_dict(),
            'dimensions':{'query_dim':head.query_dim,'visual_dim':head.visual_dim,'hidden_dim':head.hidden_dim,'max_gain':head.max_gain}},sidecar)
        metrics={'grounding_required':True,'operation_binding_required':True,'transfer_method':'per-operation-binding-source-only',
            'grounding':{'head_file':sidecar.name,'sha256':sha(sidecar),'visual_encoder_sha256':head.visual_encoder_sha256,'weight':1.},
            'transfer_settings_sha256':settings_sha,'campaign_configuration_sha256':sha(options.config),
            'next_batch':next_batch,'epoch_completed':complete,'best_validation_loss':best,
            'controller_sha256':controller_sha,'initialization_sha256':c['checkpoint_sha256'],
            'training_data_protocol':joint.training_data_protocol(args,initialization),
            'source_only':True,'target_training':False,'ce_arm':c['arm'],'allowed_mass_weight':.5,
            'operation_grounding_weight':1.,'training_mode':c['arm']+' graph-label CE + mass 0.5 + per-operation grounding 1.0',
            'planner_input_protocol_extension':'operation-binding-v1','initialization_before_updates':initial,
            'primary_hints':'frozen paired SigLIP, public instruction phrases only; never gold roles',
            'validation_operation_grounding':{r:dict(v) for r,v in state.validation.items()}}
        checkpoints.save_joint_checkpoint(path,runtime=runtime,planner=planner,controller=controller,
            planner_optimizer=optimizer,controller_optimizer=None,stage='stage1',epoch=epoch,round_robin_cursor=0,metrics=metrics)
        if complete and not initial:
            freeze(output/f'operation_validation_epoch_{epoch:03d}.json',metrics['validation_operation_grounding'])
            state.validation={r:Counter() for r in ('source','destination')}
        return metrics
    trainer.save_snapshot=save_snapshot
    original_manager=trainer.BatchGradientManager
    class Manager(original_manager):
        def __init__(self,combined,optimizer):
            super().__init__(combined,optimizer)
            if options.resume:return
            save_snapshot(output/'joint_initialization.pt',runtime=state.runtime,planner=state.planner,
                head=state.head,controller=state.controller,optimizer=optimizer,epoch=-1,next_batch=0,
                complete=True,best=float('inf'),settings_sha=sha(output/'settings.json'),
                controller_sha=trainer.tensor_digest(state.controller),initialization=state.initialization,args=args,initial=True)
            freeze(output/'initialization_parameters.json',{'planner_trainable_sha256':trainer.tensor_digest(state.planner,trainable=True),
                'grounder_sha256':trainer.tensor_digest(state.head),'controller_sha256':trainer.tensor_digest(state.controller),
                'gain':float(state.head.gain.detach()),'checkpoint_sha256':c['checkpoint_sha256']})
    trainer.BatchGradientManager=Manager
    if options.audit_only:
        def stop(**kwargs):raise Prepared()
        JointQwenVLPlanner.from_pretrained=stop
    else:
        original_model=JointQwenVLPlanner.from_pretrained
        def checked_model(**kwargs):
            import os,subprocess
            uuid=os.environ['BINDING_GPU_UUID']
            row=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip().split(',')
            if int(row[0])>=100 or int(row[1])!=0:raise RuntimeError('allocated GPU was claimed before model loading; no optimizer update')
            return original_model(**kwargs)
        JointQwenVLPlanner.from_pretrained=checked_model
    sys.argv=[sys.argv[0],'--config',str(options.config),'--manifest',c['observations'],
              '--features',str(Path(c['features']).parent),'--output',str(output)]
    if options.resume:sys.argv+=['--resume',str(options.resume)]
    try:trainer.main()
    except Prepared:
        freeze(output/'prepared.json',{'passed':True,'configuration_sha256':sha(options.config),'initialization_loaded':False})
        print('[binding-train] source reference, permutation and feature gates passed; no model loaded',flush=True)


if __name__=='__main__':main()
