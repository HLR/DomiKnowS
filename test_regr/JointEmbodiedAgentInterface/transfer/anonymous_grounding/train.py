"""Four source-only epochs: anonymous IDs, observed pointers, supervised roles."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch import nn

from ..protocol import read,freeze,sha,digest
from ..train import encode,tensor_digest
from ..source_transform import transform
from ..batch_gradients import BatchGradientManager
from ..campaign import save,utc
from .features import prepare_features
from .grounder import RoleGrounder,role_targets,grounding_loss
from .agent import predicted_context
from .objective import planning_loss
from .provenance import validate_source
from .augmentation import identity_epoch
from .audits import SourceAudits


def records_for(runtime,examples,bindings,feature_records,epoch,augmented=True,use_training_schedule=True):
    keys={key for values in runtime.world.vlabench.skill_arguments.values() for key in values}
    keys.update(key for values in runtime.world.vlabench.optional_skill_arguments.values() for key in values)
    transformed=[];orders=[]
    for example in examples:
        record=bindings[example.episode_id]
        changed=augmented and not (use_training_schedule and identity_epoch(example.episode_id,len(example.entities),epoch))
        value=transform(example,record,epoch=epoch,augmented=changed,pointer_keys=keys)
        selected=record['augmentations'][epoch]['image'] if changed else record['canonical_image']
        transformed.append(replace(value,image_paths=(record['raw_image'],),
                                   segmented_image_paths=(record['raw_image'],selected)))
        orders.append(record['augmentations'][epoch]['new_to_old'] if changed else list(range(len(value.entities))))
    items=encode(runtime,transformed);output=[]
    for example,item,order in zip(transformed,items,orders):
        output.append({'item':item,'episode_id':example.episode_id,'order':order,
                       'features':feature_records[example.episode_id],
                       'targets':role_targets(example.operation_sequence,example.entities)})
    return items,orders,output


def visual(row,cache,device):
    value=cache[row['episode_id']];order=torch.tensor(row['order'],dtype=torch.long)
    return value['features'][order].to(device).float(),value['visible'][order].to(device)


def public_record(planner,domain,context):
    images=context.get('images',()) if domain=='vlabench' else ()
    return {'prompt':planner._chat(planner._prompt_base(domain,context),len(images)),
            'images':[{'path':str(p),'sha256':sha(p)} for p in images]}


def save_snapshot(path,*,runtime,planner,head,controller,optimizer,epoch,next_batch,complete,best,settings_sha,controller_sha,initialization,args):
    from ...checkpoint import save_joint_checkpoint
    from ...main import training_data_protocol
    if path.exists():raise ValueError('preserve existing snapshot')
    sidecar=path.with_name(path.stem+'_grounding.pt');temporary=sidecar.with_suffix('.tmp')
    torch.save({'state_dict':head.state_dict(),'dimensions':{'query_dim':head.query_dim,
        'visual_dim':head.visual_dim,'hidden_dim':head.hidden_dim}},temporary)
    temporary.replace(sidecar)
    metrics={'grounding_required':True,'transfer_method':'anonymous-id-grounding-source-only',
        'grounding':{'head_file':sidecar.name,'sha256':sha(sidecar),'weight':1.,
            'visual_encoder_sha256':head.visual_encoder_sha256,
            'inputs':'original RGB anchor crops, image coordinates, instruction only; predicted roles only'},
        'transfer_settings_sha256':settings_sha,'next_batch':next_batch,'epoch_completed':complete,
        'best_validation_loss':best,'controller_sha256':controller_sha,
        'initialization_sha256':args.transfer_initialization_sha256,'source_only':True,
        'training_data_protocol':training_data_protocol(args,initialization),
        'training_mode':'masked CE + 0.5 allowed mass + 1.0 supervised grounding; unknown pointers masked',
        'no_dfa_official':False,'planner_input_protocol_extension':'anonymous-grounding-v1'}
    save_joint_checkpoint(path,runtime=runtime,planner=planner,controller=controller,
        planner_optimizer=optimizer,controller_optimizer=None,stage='stage1',epoch=epoch,
        round_robin_cursor=0,metrics=metrics)
    return metrics


def main():
    from ... import main as joint
    from ...models import JointQwenVLPlanner
    from ...checkpoint import load_joint_checkpoint
    from ...program import JointSolverPOIProgram
    from ...utils.planner_context import planner_context
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    from test_regr.VLABenchAgentInterface.dataset_protocol import guard_training_dataset
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',type=Path)
    options=parser.parse_args();config=read(options.config);output=options.output;device=torch.device('cuda:0')
    validate_source(config)
    if (config['seed'],config['epochs'],config['batch_size'])!=(42,4,8) or not config['source_only']:
        raise ValueError('source-only four-epoch configuration required')
    source=read(options.manifest)
    if not source['complete'] or source['test_used'] or source['references_read'] or len(source['episodes'])!=4050:
        raise ValueError('complete observation-only training/validation manifest required')
    if source['split_sha256']!=sha(config['split_manifest']):raise ValueError('source split differs')
    bindings={r['episode_id']:r for r in source['episodes']}
    # Audit in CPU processes before CUDA is initialized, even on a fresh run.
    feature_records={identity:None for identity in bindings}
    output.mkdir(parents=True,exist_ok=True)
    settings={'method':'anonymous-id-grounding-source-only','config_sha256':sha(options.config),
        'source_manifest_sha256':sha(options.manifest),
        'initialization_sha256':sha(config['checkpoint']),'epochs':4,'batch_size':8,'seed':42,
        'auxiliary_weight':1.,'planner_learning_rate':2e-5,'grounding_learning_rate':1e-3,
        'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.json')},
        'initialization':'full-metadata epoch_003; internal official-corpus exposure',
        'controller':'frozen','target_training_used':False,'model_selection':'source validation only',
        'validation_conditions':['canonical','fixed_id_permutation_0'],
        'augmentation_schedule':'v1: balanced identity/swap for 2 entities; identity once per four epochs otherwise',
        'unknown_bindings':'mask pointer loss and unobservable positive role labels; retain every episode',
        'inference':'predicted roles only; anonymous_grounding.agent and sidecar required'}
    freeze(output/'split_identities.json',read(config['split_manifest']))
    args=joint.build_parser().parse_args(read(config['original_configuration'])['arguments'])
    args.output=str(output);args.save_initialization=False;args.initialize_from=config['checkpoint'];args.resume=None
    args.stage1_eai_warmup_steps=0;args.device='cuda:0'
    args.transfer_initialization_sha256=config['checkpoint_sha256']
    if args.planner_learning_rate!=2e-5 or args.ce_dfa_mass_weight!=.5:raise ValueError('source optimizer/loss recipe differs')
    guard_training_dataset(args.vlabench_planning_dir,output,resume=config['checkpoint'],allow_internal_evaluation_training=True)
    apply_research_arguments(args);apply_knowledge_argument(args)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    save(output/'status.json',{'stage':'building-source-runtime-and-auditing-permutations','updated_utc':utc()})
    audits=SourceAudits(output/'audits',{
        'source_manifest_sha256':config['source_manifest_sha256'],
        'initialization_sha256':config['checkpoint_sha256'],
        'split_sha256':sha(config['split_manifest']),
        'observations_sha256':sha(options.manifest),
        'configuration_sha256':sha(config['original_configuration'])},
        workers=config.get('audit_workers',8))
    runtime,eai_train,eai_valid,splits,original_items=joint._prepare(args,device,path_auditor=audits.references)
    if {key:[e.episode_id for e in value] for key,value in splits.items()}!=read(config['split_manifest'])['splits']:
        raise ValueError('frozen source identities changed')
    for name in ('training_reference_policy.json','training_reference_hashes.json'):
        if sha(output/name)!=sha(Path(config['checkpoint']).parent/name):raise ValueError('reference policy changed')
    epochs=[]
    for epoch in range(4):
        items,orders,rows=records_for(runtime,splits['train'],bindings,feature_records,epoch)
        audits.equivalence(runtime,original_items['train'],items,orders,f'permutations_epoch_{epoch:03d}')
        epochs.append(rows)
    _,_,validation=records_for(runtime,splits['validation'],bindings,feature_records,0,augmented=False)
    for row in validation:row['validation_condition']='canonical'
    valid_items,valid_orders,permuted_validation=records_for(runtime,splits['validation'],bindings,feature_records,0,use_training_schedule=False)
    audits.equivalence(runtime,original_items['validation'],valid_items,valid_orders,'permutations_validation')
    for row in permuted_validation:row['validation_condition']='fixed_id_permutation_0'
    validation+=permuted_validation
    freeze(output/'permutation_audit.json',{'passed':True,'epochs':4,'episodes_per_epoch':3600,
        'whole_allowed_sets_compared':True,'image_bindings':'training-only atlas; original RGB; unknowns masked'})
    if not (options.features/'manifest.json').exists():
        save(output/'status.json',{'stage':'extracting-frozen-visual-features','updated_utc':utc()})
        prepare_features(config,options.manifest,options.features,device)
    feature_proof=read(options.features/'manifest.json')
    if not feature_proof['complete'] or feature_proof['source_manifest_sha256']!=sha(options.manifest):
        raise ValueError('visual feature provenance differs')
    feature_records={r['episode_id']:r for r in feature_proof['episodes']}
    if set(feature_records)!=set(bindings):raise ValueError('feature identity coverage differs')
    cache={}
    for identity,record in feature_records.items():
        if sha(record['path'])!=record['sha256']:raise ValueError('frozen entity features changed')
        cache[identity]=torch.load(record['path'],map_location='cpu',weights_only=True)
    settings['features_manifest_sha256']=sha(options.features/'manifest.json')
    freeze(output/'settings.json',settings);settings_sha=sha(output/'settings.json')
    coverage={role:{'targets':0,'observable_targets':0} for role in ('source','destination')}
    for row in epochs[0]:
        _,visible=visual(row,cache,'cpu')
        for role in coverage:
            coverage[role]['targets']+=len(row['targets'][role])
            coverage[role]['observable_targets']+=sum(bool(visible[i]) for i in row['targets'][role])
    if any(not value['observable_targets'] for value in coverage.values()):raise ValueError('no observable supervision for a required role')
    freeze(output/'grounding_coverage.json',coverage)
    save(output/'status.json',{'stage':'loading-planner-initialization','updated_utc':utc()})
    planner=JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,model_id=args.planner_model,load_in_4bit=True,
        decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller=joint._controller(args,device)
    initialization=load_joint_checkpoint(config['checkpoint'],runtime=runtime,planner=planner,controller=controller,map_location=device)
    for parameter in controller.parameters():parameter.requires_grad_(False)
    controller.eval();controller_sha=tensor_digest(controller)
    if tensor_digest(controller.image_encoder)!=feature_proof['encoder_sha256']:
        raise ValueError('frozen feature encoder differs from the checkpoint visual encoder')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    head=RoleGrounder(planner.backbone_hidden_size,feature_proof['visual_dim']).to(device)
    head.visual_encoder_sha256=feature_proof['encoder_sha256']
    _ANONYMOUS_OBJECTS.set(True);planner._grounding_extension_active=True;planner.free_decoding=False
    planner_parameters=[p for p in planner.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':planner_parameters,'lr':2e-5},{'params':list(head.parameters()),'lr':1e-3}])
    program=JointSolverPOIProgram(runtime,planner,planner_optimizer=optimizer,controller=None,device=device,
        eai_loss_weight=args.stage1_eai_loss_weight,dfa_loss=joint._dfa_loss_config(args))
    combined=nn.ModuleDict({'planner':planner,'grounder':head});gradients=BatchGradientManager(combined,optimizer)
    start_epoch=0;start_batch=0;best=float('inf')
    if options.resume:
        payload=load_joint_checkpoint(options.resume,runtime=runtime,planner=planner,controller=controller,
            planner_optimizer=optimizer,map_location=device)
        metrics=payload['metrics']
        if metrics['transfer_settings_sha256']!=settings_sha:raise ValueError('resume source, features or settings differ')
        spec=metrics['grounding'];path=options.resume.parent/spec['head_file']
        if path.name!=spec['head_file'] or sha(path)!=spec['sha256']:raise ValueError('grounding sidecar changed')
        saved=torch.load(path,map_location=device,weights_only=True);head.load_state_dict(saved['state_dict'],strict=True)
        start_epoch=int(payload['epoch'])+int(metrics['epoch_completed'])
        start_batch=0 if metrics['epoch_completed'] else int(metrics['next_batch']);best=metrics['best_validation_loss']
    audit_dir=output/'input_audits';audit_dir.mkdir(exist_ok=True)
    for epoch in range(start_epoch,4):
        planner.train();head.train();rows=epochs[epoch]
        rng=random.Random(42+epoch);order=list(range(3600));rng.shuffle(order)
        paired=[eai_train[i%len(eai_train)] for i in range(3600)];rng.shuffle(paired)
        mode='a' if epoch==start_epoch and start_batch else 'x'
        with (audit_dir/f'epoch_{epoch:03d}.jsonl').open(mode) as audit:
            for batch in range(start_batch if epoch==start_epoch else 0,450):
                indices=order[batch*8:batch*8+8];gradients.begin_step()
                before_shared=tensor_digest(planner,trainable=True,shared=True) if epoch==0 and batch==0 else None
                before_head=tensor_digest(head) if before_shared else None
                total=0.;unknown=0
                for offset,index in enumerate(indices):
                    row=rows[index];item=row['item'];context=planner_context('vlabench',item)
                    features,visible=visual(row,cache,device)
                    with runtime.domain_scope('vlabench',parameter_policy='freeze_inactive'):
                        effective,predictions,logits=predicted_context(planner,head,context,features,visible)
                        auxiliary,role_coverage=grounding_loss(logits,visible,row['targets'])
                        plan,terms=planning_loss(planner,runtime,context,effective,item['target_plan_labels'],visible)
                        loss=plan+auxiliary;gradients.capture('vlabench',loss/8)
                    total+=float(loss.detach());unknown+=terms['unknown_pointer_tokens']
                    rendered=public_record(planner,'vlabench',effective)
                    audit.write(json.dumps({'domain':'vlabench','epoch':epoch,'batch':batch,
                        'episode_id':row['episode_id'],'new_to_old':row['order'],'input':rendered,
                        'input_sha256':digest(rendered),'predicted_roles':predictions,
                        'observable_ids':visible.nonzero().flatten().tolist(),'loss_coverage':terms,
                        'grounding_coverage':role_coverage})+'\n')
                    eai=paired[batch*8+offset];eai_context=planner_context('eai',eai)
                    with runtime.domain_scope('eai',parameter_policy='freeze_inactive'):
                        eai_loss,_=program._domain_loss('eai',eai)
                        gradients.capture('eai',eai_loss*args.stage1_eai_loss_weight/8)
                    total+=float(eai_loss.detach())*args.stage1_eai_loss_weight
                    audit.write(json.dumps({'domain':'eai','epoch':epoch,'batch':batch,
                                            'input':public_record(planner,'eai',eai_context)})+'\n')
                resolved=gradients.resolve_gradients('pcgrad')
                if not all(g is None or torch.isfinite(g).all() for g in resolved):raise ValueError('nonfinite training gradient')
                if before_shared:
                    ids={id(p) for p in planner.shared_parameters() if p.requires_grad}
                    if not any(id(p) in ids and g is not None and torch.any(g!=0) for p,g in zip(gradients.parameters,resolved)):
                        raise ValueError('shared backbone received no gradient')
                    ids={id(p) for p in head.parameters()}
                    if not any(id(p) in ids and g is not None and torch.any(g!=0) for p,g in zip(gradients.parameters,resolved)):
                        raise ValueError('grounding head received no supervised gradient')
                gradients.step(optimizer,method='pcgrad',clip_grad_norm=1.)
                if before_shared:
                    after_shared=tensor_digest(planner,trainable=True,shared=True);after_head=tensor_digest(head)
                    if before_shared==after_shared or before_head==after_head:raise ValueError('shared backbone or grounding head did not update')
                    freeze(output/'first_update.json',{'shared_gradient_nonzero':True,
                        'supervised_grounding_gradient_nonzero':True,
                        'shared_before':before_shared,'shared_after':after_shared,'head_before':before_head,'head_after':after_head})
                audit.flush()
                if batch==0 or (batch+1)%10==0:
                    save(output/'status.json',{'stage':'training','epoch':epoch+1,'epochs':4,'batch':batch+1,
                        'batches_per_epoch':450,'weighted_loss':total/8,'unknown_pointer_tokens':unknown,'updated_utc':utc()})
                    print(f'[anonymous-grounding] epoch {epoch+1}/4 batch {batch+1}/450 loss={total/8:.4f}',flush=True)
                if batch==0 or (batch+1)%100==0:
                    save_snapshot(output/f'progress_epoch_{epoch:03d}_batch_{batch+1:03d}.pt',runtime=runtime,
                        planner=planner,head=head,controller=controller,optimizer=optimizer,epoch=epoch,
                        next_batch=batch+1,complete=False,best=best,settings_sha=settings_sha,
                        controller_sha=controller_sha,initialization=initialization,args=args)
        planner.eval();head.eval();validation_total=0.;role_metrics={role:{'correct':0,'cases':0,'targets':0,'observable_targets':0} for role in ('source','destination')}
        save(output/'status.json',{'stage':'validation','epoch':epoch+1,'updated_utc':utc()})
        with torch.no_grad(),(audit_dir/f'validation_epoch_{epoch:03d}.jsonl').open('x') as audit:
            for index,row in enumerate(validation):
                context=planner_context('vlabench',row['item']);features,visible=visual(row,cache,device)
                effective,predictions,logits=predicted_context(planner,head,context,features,visible)
                auxiliary,coverage=grounding_loss(logits,visible,row['targets'])
                plan,terms=planning_loss(planner,runtime,context,effective,row['item']['target_plan_labels'],visible)
                validation_total+=float(plan+auxiliary)
                for role in role_metrics:
                    if row['targets'][role]:
                        role_metrics[role]['cases']+=1;role_metrics[role]['correct']+=predictions[role] in row['targets'][role]
                    role_metrics[role]['targets']+=coverage[role]['targets']
                    role_metrics[role]['observable_targets']+=coverage[role]['observable_targets']
                audit.write(json.dumps({'episode_id':row['episode_id'],'input':public_record(planner,'vlabench',effective),
                    'condition':row['validation_condition'],'new_to_old':row['order'],
                    'predicted_roles':predictions,'loss_coverage':terms})+'\n')
                if (index+1)%50==0:print(f'[anonymous-grounding] validation {index+1}/{len(validation)}',flush=True)
            eai_total=0.
            for item in eai_valid:
                with runtime.domain_scope('eai'):loss,_=program._domain_loss('eai',item)
                eai_total+=float(loss)
        score=validation_total/len(validation)+args.stage1_eai_loss_weight*eai_total/len(eai_valid)
        if tensor_digest(controller)!=controller_sha:raise ValueError('fixed physical controller changed')
        improved=score<best;best=min(best,score)
        path=output/f'joint_stage1_epoch_{epoch:03d}.pt'
        save_snapshot(path,runtime=runtime,planner=planner,head=head,controller=controller,optimizer=optimizer,
            epoch=epoch,next_batch=0,complete=True,best=best,settings_sha=settings_sha,
            controller_sha=controller_sha,initialization=initialization,args=args)
        freeze(output/f'epoch_{epoch:03d}.json',{'validation_loss':score,'grounding':role_metrics,
            'validation_physical_episodes':450,'validation_conditions':2,'full_plan_success_measured':False,
            'checkpoint':str(path),'selected':improved,'selection':'source validation only'})
        if improved:save(output/'best.json',{'checkpoint':str(path),'epoch':epoch,'validation_loss':score})
        print(f'[anonymous-grounding] epoch {epoch+1}/4 saved; validation={score:.4f}',flush=True)
        start_batch=0
    freeze(output/'complete.json',{'epochs':4,'best':read(output/'best.json'),'controller_sha256':controller_sha,
        'training':'source-only; anonymous IDs + observed image permutations + supervised predicted grounding'})
    save(output/'status.json',{'stage':'complete','updated_utc':utc()})


if __name__=='__main__':main()
