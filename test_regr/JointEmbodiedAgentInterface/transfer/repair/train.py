"""Two fresh, matched source-only recipes; no training-time constraint masks."""
import argparse
import math
import json
from pathlib import Path
import random
import torch
from ..protocol import read,freeze,sha,digest
from ..campaign import save,utc
from ..anonymous_grounding.train import records_for,visual,public_record
from ..train import tensor_digest
from ..matched_baseline.protocol import common_prompt,normalize_ids
from ...utils.planner_context import planner_context
from .protocol import ARMS,verify,select,normalize_anonymous_prediction
from .common import build_runtime,fresh_planner,save_checkpoint,load_checkpoint,source_context,native_decode,graph_decode,seed
from .objective import planning_loss,eai_loss as eai_objective


def validation(planner,runtime,rows,examples,eai_valid,arm,output):
    from ...main import _evaluate_eai
    from ...utils.paired_references import PairedReferenceAudit
    output=Path(output);output.mkdir(parents=True,exist_ok=True);planner.eval()
    if (output/'result.json').exists():raise ValueError('validation already exists; do not overwrite')
    source_correct=source_cases=0
    with common_prompt(planner),(output/'predictions.jsonl').open('x') as stream,(output/'raw_predictions.jsonl').open('x') as raw_stream:
        audit=PairedReferenceAudit(stream,corrector=lambda values:values)
        for index,(row,example) in enumerate(zip(rows,examples)):
            bound=source_context(planner,row);prediction=[];error=None;raw_prediction=[];raw_text='';labels=[];changes=[]
            try:
                result=(native_decode(planner,bound) if arm=='native' else graph_decode(planner,runtime,bound))
                raw_prediction,labels,raw_text=result
                prediction,changes=normalize_anonymous_prediction(raw_prediction,len(bound['entity_table']),runtime.vlabench_vocabulary.argument_keys)
                from test_regr.VLABenchAgentInterface.plan_codec import plan_to_tokens
                tokens=plan_to_tokens(prediction,bound['_public_entity_aliases'],world=runtime.world.vlabench)
                for token in tokens:runtime.vlabench_vocabulary.label_for_token(token)
            except (ValueError,RuntimeError,KeyError) as exc:
                if isinstance(exc,torch.cuda.OutOfMemoryError):raise
                error={'type':type(exc).__name__,'message':str(exc)}
                labels=getattr(exc,'labels',labels);raw_text=getattr(exc,'raw_text',raw_text)
                prediction=[]
            raw_stream.write(json.dumps({'episode_id':row['episode_id'],'raw_prediction':raw_prediction,
                'labels':labels,'raw_text':raw_text,'prediction':prediction,'normalizations':changes,
                'public_entity_aliases':bound['_public_entity_aliases'],'generation_error':error})+'\n');raw_stream.flush()
            audit.observe(example,prediction,generation_error=error)
            targets=row['targets']['source']
            if len(targets)==1:
                source_cases+=1
                chosen={op['params']['target_entity_name'] for op in prediction if 'target_entity_name' in op.get('params',{})}
                source_correct+=chosen==set(targets) and error is None
            freeze(output/'inputs'/f'{index:04d}.json',{'episode_id':row['episode_id'],
                **public_record(planner,'vlabench',bound)})
            if (index+1)%25==0:print(f'[repair] greedy source validation {index+1}/450',flush=True)
        report=audit.report()
    planner.free_decoding=False
    # Retain the canonical EAI evaluation protocol, including its compiled rules.
    from ..anonymous_grounding.planner_binding.epoch_recovery import retry_eai_search
    retries=[]
    with torch.no_grad(),retry_eai_search(runtime,retries):
        eai=_evaluate_eai(planner,runtime,eai_valid,limit=None)
    freeze(output/'eai.json',eai);freeze(output/'eai_search_retries.json',retries)
    freeze(output/'vlabench.json',report)
    result={'eai_successes':round(eai['goal_success']*88),
        'progress_score':report['corrected']['planning_metrics']['overall']['score'],
        'source_pointer_accuracy':source_correct/source_cases if source_cases else 0.,
        'source_pointer_cases':source_cases,'examples':450,'eai_examples':88}
    freeze(output/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='Only an epoch boundary of this same arm may resume')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--preflight',action='store_true',help='Real source forward/backward checks; zero optimizer updates')
    a=p.parse_args();c=read(a.config);verify(c)
    if a.seed!=42 or sha(a.manifest)!=c['input_hashes'][str(a.manifest)]:raise ValueError('source manifest/seed changed')
    if a.baseline_only and a.arm!='native':raise ValueError('frozen native-Base reference required')
    output=a.output;output.mkdir(parents=True,exist_ok=True);seed()
    args,(runtime,eai_train,eai_valid,splits,_)=build_runtime(c,output)
    source=read(a.manifest)
    if not source['complete'] or source['test_used'] or source['references_read'] or len(source['episodes'])!=4050:
        raise ValueError('source observation gate failed')
    bindings={r['episode_id']:r for r in source['episodes']}
    for record in bindings.values():
        if any(sha(path)!=expected for path,expected in record['files'].items()):
            raise ValueError('original RGB, verified tags or permutation input changed')
    feature_proof=read(c['features']);feature_records={r['episode_id']:r for r in feature_proof['episodes']}
    if set(feature_records)!=set(bindings):raise ValueError('visibility coverage differs')
    cache={}
    for identity,record in feature_records.items():
        if sha(record['path'])!=record['sha256']:raise ValueError('verified visibility features changed')
        cache[identity]=torch.load(record['path'],map_location='cpu',weights_only=True)
    epochs=[records_for(runtime,splits['train'],bindings,feature_records,epoch)[2] for epoch in range(4)]
    valid=records_for(runtime,splits['validation'],bindings,feature_records,0,augmented=False)[2]
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
    planner=fresh_planner(c,runtime,a.arm)
    settings={'config_sha256':sha(a.config),'source_manifest_sha256':sha(c['source_manifest']),'arm':a.arm,'recipe':c['recipe'],
        'source_manifest_sha256':sha(a.manifest),'features_manifest_sha256':sha(c['features']),
        'provenance':'internal source experiment; historical official-corpus split; no target training',
        'public_prefix':'anonymous observed images and instruction; no oracle task or role hints',
        'loss_groups':'equal means of present syntax, source-pointer, destination-pointer groups',
        'unknown_pointers':'all overlapping native BPE tokens or graph pointer labels excluded',
        'teacher':'frozen pretrained native Qwen; adapters disabled; exact mode restoration',
        'controllers':'never updated; qualified external executor unchanged'}
    freeze(output/'settings.json',settings);settings_sha=sha(output/'settings.json')
    lora=[];heads=[]
    for name,param in planner.named_parameters():
        if not param.requires_grad:continue
        if name.startswith('label_heads.'):heads.append(param)
        elif 'lora_' in name:lora.append(param)
        else:raise ValueError('pretrained parameter would be trained: '+name)
    if not lora or not heads:raise ValueError('missing trainable LoRA or graph head')
    optimizer=torch.optim.AdamW([{'params':lora,'lr':5e-6},{'params':heads,'lr':1e-4}],weight_decay=.01)
    def rate(step):
        if step<180:return step/180
        return .5*(1+math.cos(math.pi*min(1.,(step-180)/1620)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    start=0
    if a.checkpoint:
        payload=load_checkpoint(a.checkpoint,planner,a.arm,settings_sha)
        start=payload['epoch'];optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
    elif not (output/'initialization.pt').exists():
        save_checkpoint(output/'initialization.pt',planner,runtime,a.arm,0,settings_sha,optimizer,scheduler)
    else:raise ValueError('existing experiment needs an explicit epoch checkpoint to resume')
    if a.preflight:
        row=epochs[0][0];_,visible=visual(row,cache,'cpu')
        with common_prompt(planner):
            bound=source_context(planner,row)
            plan=labels_to_plan(row['item']['target_plan_labels'].tolist(),runtime.vlabench_vocabulary,world=runtime.world.vlabench)
            plan=normalize_anonymous_prediction(plan,len(bound['entity_table']),runtime.vlabench_vocabulary.argument_keys)[0]
            planner.train()
            loss,terms=planning_loss(planner,bound,plan,row['item']['target_plan_labels'],visible,a.arm)
            loss.backward()
            if not torch.isfinite(loss) or not any(p.grad is not None and p.grad.abs().sum()>0 for p in lora):
                raise ValueError('real-model objective or LoRA gradient failed')
            optimizer.zero_grad(set_to_none=True)
            item=eai_train[0]
            eai_loss,eai_terms=eai_objective(planner,planner_context('eai',item),item['target_action_labels'])
            eai_loss.backward()
            if not torch.isfinite(eai_loss):raise ValueError('EAI loss nonfinite')
            optimizer.zero_grad(set_to_none=True);planner.eval()
            from .alignment import verify_alignment
            alignment=verify_alignment(planner,bound,plan,row['item']['target_plan_labels'],visible)
            try:
                decoded=native_decode(planner,bound) if a.arm=='native' else graph_decode(planner,runtime,bound)
                generation_error=None
            except ValueError as exc:
                # Fresh graph heads can emit an invalid plan. Record that
                # baseline behavior; it does not invalidate a finite loss.
                decoded=([],[],'');generation_error=str(exc)
        freeze(output/'preflight.json',{'passed':True,'arm':a.arm,'optimizer_updates':0,
            'loss':terms,'eai_loss':float(eai_loss.detach()),'eai_terms':eai_terms,'generation_labels':decoded[1],
            'generation_error':generation_error})
        freeze(output/'alignment.json',alignment)
        return
    if a.baseline_only:
        result=validation(planner,runtime,valid,splits['validation'],eai_valid,'native',output/'source_validation')
        freeze(output/'complete.json',{'base_reference':True,**result});return
    history=[read(output/f'epoch_{e:03d}/epoch.json') for e in range(1,start+1)]
    for epoch in range(start,4):
        seed();planner.train();rng=random.Random(42+epoch)
        order=list(range(3600));rng.shuffle(order)
        paired=[eai_train[i%len(eai_train)] for i in range(3600)];rng.shuffle(paired)
        folder=output/f'epoch_{epoch+1:03d}';folder.mkdir(parents=True,exist_ok=True)
        with common_prompt(planner),(folder/'training_inputs.jsonl').open('x') as stream:
            for batch in range(450):
                optimizer.zero_grad(set_to_none=True);total=0.
                for offset,index in enumerate(order[batch*8:batch*8+8]):
                    row=epochs[epoch][index];bound=source_context(planner,row)
                    _,visible=visual(row,cache,'cpu')
                    plan=labels_to_plan(row['item']['target_plan_labels'].tolist(),runtime.vlabench_vocabulary,world=runtime.world.vlabench)
                    plan=normalize_anonymous_prediction(plan,len(bound['entity_table']),runtime.vlabench_vocabulary.argument_keys)[0]
                    loss,terms=planning_loss(planner,bound,plan,row['item']['target_plan_labels'],visible,a.arm)
                    (loss/8).backward();total+=float(loss.detach())
                    del loss
                    item=paired[batch*8+offset];eai_context=planner_context('eai',item)
                    eai_loss,eai_terms=eai_objective(planner,eai_context,item['target_action_labels'])
                    (eai_loss/8).backward();total+=float(eai_loss.detach());del eai_loss
                    rendered=public_record(planner,'vlabench',bound)
                    stream.write(json.dumps({'episode_id':row['episode_id'],'epoch':epoch+1,'batch':batch+1,
                        'new_to_old':row['order'],'input':rendered,'input_sha256':digest(rendered),'loss':terms})+'\n')
                    eai_rendered=public_record(planner,'eai',eai_context)
                    stream.write(json.dumps({'domain':'eai','epoch':epoch+1,'batch':batch+1,
                        'input':eai_rendered,'input_sha256':digest(eai_rendered),'loss':eai_terms})+'\n')
                if not any(p.grad is not None and torch.any(p.grad!=0) for p in lora):raise ValueError('LoRA received no gradient')
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in lora+heads):raise ValueError('nonfinite gradient')
                before=tensor_digest(planner,trainable=True,shared=True) if epoch==0 and batch==1 else None
                torch.nn.utils.clip_grad_norm_(lora+heads,1.);optimizer.step();scheduler.step();stream.flush()
                if before:
                    after=tensor_digest(planner,trainable=True,shared=True)
                    if before==after:raise ValueError('shared LoRA weights did not update after warm-up began')
                    freeze(output/'first_update.json',{'passed':True,'batch':2,'shared_lora_before':before,
                        'shared_lora_after':after,'initialization_saved_before_updates':True})
                if batch==0 or (batch+1)%10==0:
                    save(output/'status.json',{'stage':'training','arm':a.arm,'epoch':epoch+1,'epochs':4,
                        'batch':batch+1,'batches_per_epoch':450,'loss':total/8,'updated_utc':utc()})
                    print(f'[repair] {a.arm} epoch {epoch+1}/4 batch {batch+1}/450 loss={total/8:.4f}',flush=True)
        checkpoint=output/f'epoch_{epoch+1:03d}.pt'
        save_checkpoint(checkpoint,planner,runtime,a.arm,epoch+1,settings_sha,optimizer,scheduler)
        result=validation(planner,runtime,valid,splits['validation'],eai_valid,a.arm,folder)
        result.update(epoch=epoch+1,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        # result.json is immutable validation; epoch metadata lives separately.
        freeze(folder/'epoch.json',result);history.append(result)
    baseline=read(Path(c['output'])/'native_base/source_validation/result.json')
    selection=select(history,baseline['progress_score']);freeze(output/'selection.json',selection)
    freeze(output/'complete.json',{'epochs':4,'selection':selection,'recipe':c['recipe']})
    save(output/'status.json',{'stage':'complete','qualified':selection['qualified'],'updated_utc':utc()})


if __name__=='__main__':main()
