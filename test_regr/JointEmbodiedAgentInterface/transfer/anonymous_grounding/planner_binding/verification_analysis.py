"""Validation gates, counterfactual traces and preregistered paired reporting."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

if not __package__:
    root = Path(sys.argv[sys.argv.index('--root')+1])
    sys.path.insert(0,str(root/'source'))
    __package__ = 'test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding'
from ...protocol import read, freeze, sha, episodes, public_input, context
from .verification_protocol import exact_policy_search, paired_interval

POLICIES=('none','shared-verified','executor')


def indexed(path, expected):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
    result={r['episode_id']:r for r in rows}
    if len(rows)!=len(expected) or len(result)!=len(expected) or set(result)!=set(expected):
        raise ValueError('incomplete or duplicated episode coverage: '+str(path))
    return result


def make_policy(runtime, public, name):
    from ....models import FreeDecoding
    from ....generalization.shared_constraints import bind_shared
    from ..object_fixes.contract import ExecutorContract
    if name=='none':return FreeDecoding(runtime.vlabench_vocabulary.label_count,runtime.vlabench_vocabulary.eos_label)
    base=bind_shared(runtime.vlabench_vocabulary,runtime.max_vlabench_operations,context(public,'libero'))
    if name=='shared-verified':return base
    if name!='executor':raise ValueError('unknown policy')
    return ExecutorContract(base,runtime.vlabench_vocabulary,public,
        {'pick':'target_entity_name','place':'target_container_name'},runtime.max_vlabench_operations)


def plan_labels(plan, vocabulary):
    labels=[]
    for operation in plan:
        name=operation['name'];key='target_entity_name' if name=='pick' else 'target_container_name'
        if name not in ('pick','place') or set(operation['params'])!={key}:raise ValueError('not an executor-interface plan')
        pointer=operation['params'][key]
        if type(pointer) is not int:raise ValueError('noncanonical pointer')
        labels += [vocabulary.label_for_token('skill:'+name), vocabulary.label_for_token('arg:'+key),
                   vocabulary.label_for_token('obj:'+str(pointer))]
    return labels+[vocabulary.eos_label]


def accepts(policy, labels, budget):
    state=policy.start_state
    with exact_policy_search(policy,[]):
        for i,label in enumerate(labels):
            if label not in policy.allowed_tokens(state,remaining_steps=budget-i-1):return False
            state=policy.step(state,label)
            if state is None:return False
    return policy.is_accepting(state)


def shadow_trace(policy, records, vocabulary, budget):
    """Only replay prefixes still belonging to this policy's language."""
    state=policy.start_state;prefix=[];steps=[]
    with exact_policy_search(policy,[]) as _:
        for index,row in enumerate(records):
            if row['prefix']!=prefix:raise ValueError('trace prefix does not match emitted trajectory')
            allowed=policy.allowed_tokens(state,remaining_steps=budget-index-1)
            raw=vocabulary.label_for_token(row['raw_argmax']);chosen=vocabulary.label_for_token(row['chosen'])
            blocked=raw not in allowed
            reason=policy.reason(state,raw) if hasattr(policy,'reason') else None
            steps.append({'step':index,'raw_top':row['raw_argmax'],'raw_top_rejected':blocked,
                          'reason':reason or ('shared_rule_or_completion' if blocked else None),
                          'allowed_count':len(allowed),'raw_top_logit_margin':row.get('raw_top_logit_margin'),
                          'prefix':list(prefix)})
            if chosen not in allowed:
                return {'first_divergence':index,'stopped_at_rejected_prefix':True,'steps':steps}
            state=policy.step(state,chosen)
            if state is None:raise ValueError('policy allowed a rejected transition')
            prefix.append(row['chosen'])
    return {'first_divergence':None,'stopped_at_rejected_prefix':False,'steps':steps}


def runtime_for(c):
    import torch
    from .... import eval_planning
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    parser=eval_planning.build_eval_parser()
    args=parser.parse_args(['--checkpoint',c['snapshots']['1']['path'],'--planning-dir',c['planning_dir'],
                           '--device','cpu','--seed','42'])
    args.knowledge='metadata';args.observation_mode='metadata-assisted';args.vlabench_planning_dir=str(args.planning_dir)
    apply_research_arguments(args);apply_knowledge_argument(args)
    return eval_planning._prepare_eval(args,torch.device('cpu'))[0]


def analyze(root, c, cohort, snapshot, runtime=None):
    records=episodes(read(c['external_manifest']));ids=[e['episode_id'] for e in records]
    base=root/cohort/snapshot;data={p:indexed(base/p/'libero/predictions.jsonl',ids) for p in POLICIES}
    common=read(base/'none/libero/report.json')['input_hashes'];coverage={};counterfactual={};failures={}
    for policy in POLICIES:
        report=read(base/policy/'libero/report.json')
        if report['input_hashes']!=common:raise ValueError('policy changed rendered neural inputs')
        actual=Counter();decisions=defaultdict(list);rules=Counter();applications=Counter();executor=Counter()
        for line in (base/policy/'libero/decisions.jsonl').open():
            row=json.loads(line)
            if 'raw_argmax_allowed' in row:
                decisions[row['episode_id']].append(row)
                actual['tokens']+=1;actual['raw_top_rejected']+=not row['raw_argmax_allowed']
                actual['chosen_changed']+=bool(row['chosen_differs_from_raw_argmax'])
            else:
                trace=row.get('rule_trace',{});rules.update(trace.get('triggered_rejections',{}));applications.update(trace.get('applicable_events',{}));executor.update(trace.get('executor_rejections',{}))
        coverage[policy]={'decisions':dict(actual),'rule_rejected_alternatives':dict(rules),
                          'rule_applicable_events':dict(applications),'executor_rejected_alternatives':dict(executor)}
        counts=Counter()
        for e in records:
            row=data[policy][e['episode_id']]
            if row['generation_error']:counts['generation_error']+=1
            elif not row['valid_plan']:counts['syntax_capability_or_precondition']+=1
            elif not row['goal_success']:counts['legal_wrong_goal_binding']+=1
            else:counts['abstract_success']+=1
        failures[policy]=dict(counts)
        if runtime is not None:
            free=defaultdict(list)
            for line in (base/'none/libero/decisions.jsonl').open():
                row=json.loads(line)
                if 'raw_argmax_allowed' in row:free[row['episode_id']].append(row)
            traces={}
            for e in records:
                public=public_input(e);dfa=make_policy(runtime,public,policy)
                traces[e['episode_id']]=shadow_trace(dfa,free[e['episode_id']],runtime.vlabench_vocabulary,runtime.max_vlabench_operations*5+1)
            counterfactual[policy]=traces
    result={'coverage':coverage,'failure_partition':failures,'counterfactual':counterfactual,
        'plan_changes_vs_none':{p:sum(data[p][i]['prediction']!=data['none'][i]['prediction'] for i in ids) for p in POLICIES}}
    freeze(root/'analysis'/f'{cohort}-{snapshot}.json',result)
    return result,data


def validate(root, c):
    from ....generalization.libero_protocol import reference_plan,abstract_score
    runtime=runtime_for(c);analysis,data=analyze(root,c,'validation','epoch-1',runtime)
    budget=runtime.max_vlabench_operations*5+1;references=alternatives=wrong_objects=preserved=0
    for e in episodes(read(c['external_manifest'])):
        identity=e['episode_id'];public=public_input(e)
        if sha(e['scoring_private'])!=e['files'][e['scoring_private']]:raise ValueError('validation reference changed')
        spec=read(e['scoring_private']);plan=reference_plan(spec);dfa=make_policy(runtime,public,'executor')
        if not abstract_score(plan,spec)['goal_success'] or not accepts(dfa,plan_labels(plan,runtime.vlabench_vocabulary),budget):
            raise ValueError('independently valid reference excluded by capability DFA')
        references+=1
        if not accepts(dfa,plan_labels(plan+plan,runtime.vlabench_vocabulary),budget):raise ValueError('valid repeated relocation excluded')
        alternatives+=1
        source=plan[0]['params']['target_entity_name'];dest=plan[1]['params']['target_container_name']
        other=next((i for i,n in enumerate(public['entities']) if n in public['movable'] and i not in (source,dest)),None)
        if other is not None:
            wrong=[{'name':'pick','params':{'target_entity_name':other}}, {'name':'place','params':{'target_container_name':dest}}]
            score=abstract_score(wrong,spec)
            if score['valid_plan'] and not score['goal_success']:
                if not accepts(dfa,plan_labels(wrong,runtime.vlabench_vocabulary),budget):raise ValueError('DFA used hidden goal to exclude legal wrong object')
                wrong_objects+=1
        free=data['none'][identity]
        if free['valid_plan']:
            if free['prediction']!=data['executor'][identity]['prediction']:raise ValueError('capability-valid greedy plan changed')
            preserved+=1
    if wrong_objects==0:raise ValueError('no legal wrong-object semantic counterexample verified')
    if any(not r['valid_plan'] or r['generation_error'] for r in data['executor'].values()):raise ValueError('executor policy did not produce 100 conforming plans')
    freeze(root/'validation_gate.json',{'passed':True,'configuration_sha256':sha(root/'validation_config.json'),
        'analysis_source_sha256':sha(__file__),
        'model_source_sha256':c['evaluation_source_sha256'],'preregistration_sha256':sha(root/'preregistration.json'),
        'valid_reference_plans_preserved':references,'valid_alternative_plans_preserved':alternatives,
        'legal_wrong_object_plans_preserved':wrong_objects,'already_valid_greedy_plans_unchanged':preserved,
        'capability_conforming_predictions':100,'selection_or_parameter_tuning':False})


def report(root, c):
    ids=[e['episode_id'] for e in episodes(read(c['external_manifest']))];native={};rows=[];coverage={};runtime=runtime_for(c)
    for snapshot in ('initialization','epoch-1'):
        analysis,data=analyze(root,c,'confirmation',snapshot,runtime);coverage[snapshot]=analysis
        for policy in POLICIES:
            folder=root/'confirmation'/snapshot/policy/'libero';outcomes=indexed(folder/'execution/episodes.jsonl',ids)
            if any(data[policy][i]['prediction']!=outcomes[i]['prediction'] for i in ids):raise ValueError('native executor used another plan')
            if any(r['steps']>1000 for r in outcomes.values()):raise ValueError('physical horizon changed')
            summary=read(folder/'execution/summary.json');planning=read(folder/'report.json')
            if sum(r['success'] for r in outcomes.values())!=summary['successes']:raise ValueError('native score arithmetic differs')
            native[snapshot,policy]=outcomes
            rows.append({'snapshot':snapshot,'policy':policy,'episodes':100,'abstract_percent':planning['abstract_goal_success']*100,
                'native_percent':summary['success_rate']*100,'native_ci95':summary['success_ci95'],
                'completed_success_percent':sum(r['success'] and r['termination_reason']=='completed' for r in outcomes.values()),
                'valid_percent':planning['valid_plan_rate']*100,'generation_errors':planning['generation_errors'],
                'raw_top_rejections':analysis['coverage'][policy]['decisions'].get('raw_top_rejected',0),
                'plan_changes_vs_none':analysis['plan_changes_vs_none'][policy]})
    contrasts={
        'training':paired_interval(native['initialization','none'],native['epoch-1','none'],ids,confidence=.975),
        'constraints':paired_interval(native['epoch-1','none'],native['epoch-1','executor'],ids,confidence=.975)}
    secondary={f'{snapshot}:{policy}-vs-none':paired_interval(native[snapshot,'none'],native[snapshot,policy],ids)
        for snapshot in ('initialization','epoch-1') for policy in ('shared-verified','executor')}
    secondary.update({f'training:{policy}':paired_interval(native['initialization',policy],native['epoch-1',policy],ids)
        for policy in ('shared-verified','executor')})
    control=[];archive=Path(read(root/'setup.json')['archive'])
    for snapshot in ('initialization','epoch-1'):
        base=root/'control'/('initialization' if snapshot=='initialization' else 'reproduction')
        eai=read(base/'none/eai.json')
        for policy in ('none','shared-verified','full-metadata'):
            out=base/policy;vla=read(out/'vlabench/report.json');r={'snapshot':snapshot,'policy':policy,'eai_goal_percent':eai['goal_success']*100 if policy=='none' else None,'vla_macro':vla['Macro'],'progress_score':vla['progress_score'],'libero_abstract_percent':None,'libero_physical_percent':None}
            if policy!='full-metadata':
                native_folder=out/'libero/execution' if snapshot=='initialization' else archive/'evaluation/epoch_01'/policy/'libero/execution'
                plans=indexed(out/'libero/predictions.jsonl',[e['episode_id'] for e in episodes(read(read(root/'control_config.json')['external_manifest']))])
                actual=indexed(native_folder/'episodes.jsonl',list(plans))
                if any(plans[i]['prediction']!=actual[i]['prediction'] for i in plans):raise ValueError('reused control execution differs')
                r.update(libero_abstract_percent=read(out/'libero/report.json')['abstract_goal_success']*100,
                         libero_physical_percent=read(native_folder/'summary.json')['success_rate']*100,
                         execution_reused=snapshot=='epoch-1')
            control.append(r)
    result={'control_rows':control,'confirmation_rows':rows,'primary_contrasts':contrasts,'secondary_contrasts':secondary,
        'validation_gate':read(root/'validation_gate.json'),'preregistration_sha256':sha(root/'preregistration.json'),
        'confirmation_states':[45,46,47,48,49],'original_selection_unchanged':True,
        'training_benefit_supported':contrasts['training']['task_cluster_interval'][0]>0,
        'executor_dfa_native_benefit_supported':contrasts['constraints']['task_cluster_interval'][0]>0,
        'scope':'one-seed planner transfer through a fixed executor, conditional on the frozen source-only initialization'}
    freeze(root/'results.json',result)
    for name,data in [('matched_control',control),('confirmation',rows)]:
        with (root/(name+'.csv')).open('w',newline='') as stream:
            columns=list(dict.fromkeys(k for r in data for k in r));writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(data)
    (root/'RESULTS.md').write_text('# Matched transfer verification\n\n'+json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','report']);p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();c=read(a.config)
    from .epoch_evaluation import verify_inputs
    verify_inputs(c)
    if a.mode=='validation':validate(a.root,c)
    else:report(a.root,c)
