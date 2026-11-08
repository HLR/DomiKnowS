"""Frozen-snapshot VLABench/LIBERO planning transfer and canonical EAI evaluation."""
import json
from pathlib import Path
import random
import subprocess
from collections import Counter
import torch

from . import eval_planning
from .checkpoint import load_joint_checkpoint
from .models import JointQwenVLPlanner, FreeDecoding
from .main import _controller, _evaluate_eai, _runtime_adapter
from .utils.planner_context import planner_context
from .utils.generalization_protocol import file_hash, freeze_json
from .utils.paired_references import PairedReferenceAudit
from .generalization.shared_constraints import bind_shared, rule_trace, SHARED_RULES, EXCLUDED_NONUNIVERSAL
from .generalization.libero_protocol import digest, abstract_score
from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
from test_regr.VLABenchAgentInterface.reward import score_vlabench_plan


def input_record(planner, context):
    images = [str(p) for p in context.get('images', ())]
    prompt = planner._chat(planner._prompt_base('vlabench', context), len(images))
    return {'prompt': prompt, 'images': [{'path': p, 'sha256': file_hash(p)} for p in images]}


def generate(planner, runtime, context, policy):
    planner.free_decoding = policy == 'none'
    if policy == 'none':
        dfa = FreeDecoding(runtime.vlabench_vocabulary.label_count, runtime.vlabench_vocabulary.eos_label)
    elif policy == 'shared-verified':
        dfa = bind_shared(runtime.vlabench_vocabulary, runtime.max_vlabench_operations, context)
    else:
        dfa = runtime.dfa_for('vlabench', context)
    labels = []; error = None
    expected_input = input_record(planner, context)
    original_prepare = planner._prepare_inputs_and_boundaries
    verified = False
    def checked_prepare(domain, effective_context, active_tokens=()):
        nonlocal verified
        if not verified:
            if domain != 'vlabench' or input_record(planner, effective_context) != expected_input:
                raise AssertionError('bound constraint policy changed the actual generation inputs')
            verified = True
        return original_prepare(domain, effective_context, active_tokens)
    planner._prepare_inputs_and_boundaries = checked_prepare
    try:
        with torch.no_grad(), runtime.domain_scope('vlabench'):
            labels, _ = planner.sample_labels('vlabench', context, dfa,
                max_steps=_runtime_adapter(runtime, 'vlabench').max_tokens, deterministic=True)
        prediction = labels_to_plan(labels, runtime.vlabench_vocabulary, world=runtime.world.vlabench)
    except (ValueError, RuntimeError, KeyError) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        prediction = []
    finally:
        planner._prepare_inputs_and_boundaries = original_prepare
    trace = rule_trace(dfa, labels) if policy != 'none' else {'declared': [], 'applicable_events': {}, 'triggered_rejections': {}, 'events': []}
    return prediction, error, trace, dfa


def evaluate_vlabench(planner, runtime, examples, policy, output, settings):
    root = output/'vlabench'/policy; root.mkdir(parents=True, exist_ok=True)
    freeze_json(root/'settings.json', settings)
    if (root/'report.json').exists():
        return json.loads((root/'report.json').read_text())
    valid = 0; errors = 0; traces = Counter(); applicable = Counter(); inputs = {}
    with (root/'predictions.jsonl').open('w') as stream, (root/'decisions.jsonl').open('w') as decisions:
        audit = PairedReferenceAudit(stream)
        for index, example in enumerate(examples):
            context = planner_context('vlabench', example)
            context['images'] = example.segmented_image_paths or example.image_paths
            record = input_record(planner, context); inputs[example.episode_id] = digest(record)
            freeze_json(root.parent.parent/'vlabench_inputs'/f'{index:04d}.json',
                        {'episode_id': example.episode_id, **record})
            planner.decision_observer = lambda row: decisions.write(json.dumps(
                {'episode_id': example.episode_id, **row}, default=str)+'\n')
            prediction, error, trace, dfa = generate(planner, runtime, context, policy)
            errors += int(error is not None)
            audit.observe(example, prediction, generation_error=error)
            result = score_vlabench_plan(prediction, example.operation_sequence, example.dependency,
                entity_table=example.entities, world_bundle=runtime.world.vlabench,
                task_name=example.task, task_context=context, policy=dfa)
            valid += int(result.valid and error is None)
            traces.update(trace['triggered_rejections'])
            applicable.update(trace['applicable_events'])
            decisions.write(json.dumps({'episode_id': example.episode_id, 'rule_trace': trace})+'\n')
            if index % 25 == 0:
                print(f'[generalization] VLABench {policy} {index+1}/{len(examples)}', flush=True)
        report = audit.report()
    metric = report['corrected']['planning_metrics']
    dimensions = metric['official_mapping']['legacy_total_score_by_dimension']
    report.update(dataset='vlabench_internal_test', constraint_policy=policy, examples=len(examples),
        exact_match=report['corrected']['strict_plan_match'], valid_plan_rate=valid/len(examples),
        dimensions=dimensions, Macro=sum(dimensions.values())/len(dimensions),
        progress_score=metric['overall']['score'], generation_errors=errors,
        input_hashes=inputs, triggered_rejections=dict(traces), applicable_events=dict(applicable), settings=settings)
    if policy == 'shared-verified':
        report['groundable_rule_episodes'] = {name: (0 if name == 'pick_excludes_static_fixtures' else len(examples))
                                               for name in SHARED_RULES}
        report['fixture_evidence'] = 'not provided to the VLABench shared adapter; unknown permits candidates'
    freeze_json(root/'report.json', report)
    return report


def evaluate_libero(planner, runtime, manifest_path, policy, output, settings):
    if policy == 'full-metadata':
        raise ValueError('VLABench task metadata cannot be aliased to LIBERO task identities')
    manifest = json.loads(manifest_path.read_text())
    episodes = [e for e in manifest['episodes'] if e['split'] == 'evaluation']
    if len(episodes) != 100 or len({e['episode_id'] for e in episodes}) != 100:
        raise ValueError('require exactly 100 unique external evaluation episodes')
    root = output/'libero'/policy; root.mkdir(parents=True, exist_ok=True)
    freeze_json(root/'settings.json', settings)
    if (root/'report.json').exists():
        return json.loads((root/'report.json').read_text())
    rows = []; inputs = {}; totals = Counter(); applicable = Counter()
    with (root/'predictions.jsonl').open('w') as stream, (root/'decisions.jsonl').open('w') as decisions:
        for index, episode in enumerate(episodes):
            if any(file_hash(p) != h for p, h in episode['files'].items()):
                raise ValueError('frozen external input/reference changed')
            public = json.loads(Path(episode['generation_input']).read_text())
            context = {'instruction': public['instruction']+'\nInitial scene relations: '+json.dumps(public['initial_relations']),
                'entity_table': public['entities'], 'images': tuple(public['images']),
                'generation_entity_types': public['entity_types'],
                'generation_static_fixture_ids': [i for i, name in enumerate(public['entities'])
                                                  if name not in public['movable']]}
            record = input_record(planner, context); inputs[episode['episode_id']] = digest(record)
            freeze_json(output/'libero_inputs'/f'{index:04d}.json', {'episode_id': episode['episode_id'], **record})
            planner.decision_observer = lambda row: decisions.write(json.dumps(
                {'episode_id': episode['episode_id'], **row}, default=str)+'\n')
            prediction, error, trace, _ = generate(planner, runtime, context, policy)
            # Scoring-private data is opened only after generation completed.
            spec = json.loads(Path(episode['scoring_private']).read_text())
            scores = abstract_score(prediction, spec)
            row = {'episode_id': episode['episode_id'], 'suite': episode['suite'], 'prediction': prediction,
                'generation_error': error, 'input_hash': inputs[episode['episode_id']], **scores,
                'rule_trace': trace}
            stream.write(json.dumps(row)+'\n'); stream.flush(); rows.append(row)
            decisions.write(json.dumps({'episode_id': episode['episode_id'], 'rule_trace': trace})+'\n')
            totals.update(trace['triggered_rejections'])
            applicable.update(trace['applicable_events'])
            print(f'[generalization] LIBERO {policy} {index+1}/100', flush=True)
    violations = Counter(v['reason'] for row in rows for v in row['violations'])
    report = {'dataset': 'libero_metadata_assisted_transfer_subset', 'constraint_policy': policy,
        'examples': 100, 'abstract_goal_success': sum(r['goal_success'] and not r['generation_error'] for r in rows)/100,
        'valid_plan_rate': sum(r['valid_plan'] and not r['generation_error'] for r in rows)/100,
        'generation_errors': sum(bool(r['generation_error']) for r in rows), 'input_hashes': inputs,
        'abstract_rule_violations': dict(violations), 'abstract_rule_violation_episode_denominator': 100,
        'triggered_rejections': dict(totals), 'applicable_events': dict(applicable), 'settings': settings,
        'execution': 'separate fixed-executor evaluation gated on calibration; not learned-controller transfer'}
    if policy == 'shared-verified':
        report['groundable_rule_episodes'] = {name:100 for name in SHARED_RULES}
    freeze_json(root/'report.json', report)
    return report


def main():
    parser = eval_planning.build_eval_parser()
    parser.add_argument('--episode-manifest', type=Path, required=True)
    parser.add_argument('--experiment-output', type=Path, required=True)
    parser.add_argument('--constraint-policy', choices=['full-metadata', 'shared-verified', 'none'])
    parser.add_argument('--dataset', choices=['all', 'vlabench', 'libero'], default='all')
    args = parser.parse_args()
    args.knowledge = 'metadata'; args.observation_mode = 'metadata-assisted'; args.split = 'test'
    args.vlabench_planning_dir = str(args.planning_dir)
    apply_research_arguments(args); apply_knowledge_argument(args)
    random.seed(args.seed); torch.manual_seed(args.seed)
    runtime, splits = eval_planning._prepare_eval(args, torch.device(args.device))
    if len(splits['test']) != 450:
        raise ValueError('internal test identities must cover 450 examples')
    split_manifest = args.checkpoint.parent/'split_identities.json'
    frozen_splits = json.loads(split_manifest.read_text())
    if frozen_splits['seed'] != args.seed or frozen_splits['splits']['test'] != [e.episode_id for e in splits['test']]:
        raise ValueError('test identities differ from the frozen training split')
    planner = JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary, model_id=args.planner_model,
        load_in_4bit=args.load_in_4bit, decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller = _controller(args, torch.device(args.device))
    payload = load_joint_checkpoint(args.checkpoint, runtime=runtime, planner=planner, controller=controller,
                                    map_location=args.device)
    for name, expected in payload.get('campaign_provenance', {}).get('manifests', {}).items():
        if file_hash(args.checkpoint.parent/name) != expected:
            raise ValueError(f'checkpoint campaign manifest changed: {name}')
    planner.eval()
    source_commit = subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], text=True).strip()
    settings = {'checkpoint_sha256': file_hash(args.checkpoint), 'manifest_sha256': file_hash(args.episode_manifest),
        'split_manifest_sha256': file_hash(split_manifest), 'source_commit': source_commit,
        'dataset': args.dataset, 'requested_constraint_policy': args.constraint_policy,
        'decoding': 'greedy', 'observation_mode': args.observation_mode,
        'stage': payload['stage'], 'epoch': payload['epoch'], 'seed': args.seed,
        'shared_rules': sorted(SHARED_RULES), 'excluded_nonuniversal': EXCLUDED_NONUNIVERSAL,
        'knowledge_manifest': runtime.knowledge_manifest}
    output = args.experiment_output; output.mkdir(parents=True, exist_ok=True)
    freeze_json(output/'settings.json', settings)
    reports = {}
    if args.dataset in ('all', 'vlabench'):
        policies = [args.constraint_policy] if args.constraint_policy else ['full-metadata', 'shared-verified', 'none']
        for policy in policies:
            reports['vlabench:'+policy] = evaluate_vlabench(planner, runtime, splits['test'], policy, output, settings)
    if args.dataset in ('all', 'libero'):
        policies = [args.constraint_policy] if args.constraint_policy else ['shared-verified', 'none']
        for policy in policies:
            reports['libero:'+policy] = evaluate_libero(planner, runtime, args.episode_manifest, policy, output, settings)
    for dataset in ('vlabench', 'libero'):
        values = [r['input_hashes'] for k, r in reports.items() if k.startswith(dataset+':')]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError('constraint ablation changed generation inputs')
    planner.free_decoding = False
    planner.decision_observer = None
    from test_regr.EmbodiedAgentInterface.dataset import load_eai_dataset, split_train_dev
    eai = load_eai_dataset(args.eai_dataset, split=args.eai_split, data_path=args.eai_data_path,
                           device='cpu', max_steps=args.eai_max_steps)
    _, validation = split_train_dev(eai, dev_fraction=.2)
    from test_regr.EmbodiedAgentInterface.reward import make_eai_reward_function
    for item in validation:
        item['reward_function'] = make_eai_reward_function(item, vocabulary=runtime.eai_vocabulary,
            mode='dense', world_bundle=runtime.world.eai)
    if len(validation) != 88:
        raise ValueError('canonical EAI validation must contain 88 examples')
    with (output/'eai_predictions.jsonl').open('w') as stream:
        def observe(item, labels, result):
            trace = rule_trace(runtime.dfa_for('eai', item), labels)
            stream.write(json.dumps({'instruction': planner_context('eai', item)['instruction'],
                'labels': labels, 'goal': result, 'rule_trace': trace}, default=str)+'\n')
        eai_report = _evaluate_eai(planner, runtime, validation, limit=None, prediction_observer=observe)
    freeze_json(output/'eai.json', eai_report)
    from .generalization.overlap_report import overlap_report
    freeze_json(output/'constraint_overlap.json', overlap_report(runtime.knowledge_manifest, reports))
    freeze_json(output/'summary.json', {'settings': settings,
        'reports': {k: {n: v for n, v in r.items() if n not in ('input_hashes', 'original', 'corrected')}
                    for k, r in reports.items()}, 'eai': eai_report})


if __name__ == '__main__':
    main()
