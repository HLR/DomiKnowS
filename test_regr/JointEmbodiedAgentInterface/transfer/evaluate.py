"""Greedy external planning with frozen inputs and an archived-baseline gate."""
from collections import Counter
from pathlib import Path
import random

from .protocol import (POLICIES, SEMANTIC_VERSION, adapt, compare_baseline, context,
                       digest, episodes, freeze, indexed, jsonl, public_input, read, sha)
from .permutation import verified_episode_permutation


def evaluate(planner, runtime, *, dataset, manifest_path, method, output, settings, archive):
    from ..eval_generalization import generate, input_record
    from ..generalization.rlbench_protocol import plan_validity
    from ..generalization.libero_protocol import abstract_score
    import json

    selected = episodes(read(manifest_path))
    expected = [e['episode_id'] for e in selected]
    output = Path(output)
    freeze(output / 'settings.json', {**settings, 'method': method, 'dataset': dataset,
                                     'manifest_sha256': sha(manifest_path), 'semantic_version': SEMANTIC_VERSION})
    reports = {}
    for policy in POLICIES:
        root = output / policy
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'predictions.jsonl'
        if (root / 'report.json').exists():
            reports[policy] = read(root / 'report.json')
            indexed(jsonl(path), expected)
            continue
        if path.exists():
            raise ValueError('incomplete attempt preserved; use a fresh output directory')
        rows, inputs = [], {}
        with path.open('w', encoding='utf-8') as stream, (root / 'decisions.jsonl').open('w', encoding='utf-8') as decisions:
            for index, episode in enumerate(selected):
                public = public_input(episode)
                permutation = (verified_episode_permutation(episode, public)
                               if method == 'id-sensitivity' else None)
                bound = adapt(context(public, dataset), 'baseline' if permutation else method, public['instruction'])
                rendered = input_record(planner, bound)
                inputs[episode['episode_id']] = digest(rendered)
                freeze(output / 'rendered_inputs' / f'{index:03d}.json',
                       {'episode_id': episode['episode_id'], **rendered})
                planner.decision_observer = lambda row: decisions.write(json.dumps(
                    {'episode_id': episode['episode_id'], **row}, default=str) + '\n')
                prediction, error, trace, _ = generate(planner, runtime, bound, policy)
                raw_prediction = prediction
                if permutation is not None:
                    prediction = permutation.plan(prediction, inverse=True, strict=False)
                    scoring_public = public_input({**episode, 'generation_input': episode['canonical_generation_input']})
                else:
                    scoring_public = public
                # Scoring-private data is opened only after generation, never adapted into inputs.
                score = (abstract_score(prediction, read(episode['scoring_private'])) if dataset == 'libero'
                         else plan_validity(prediction, scoring_public))
                row = {'episode_id': episode['episode_id'], 'prediction': prediction,
                       'generation_error': error, 'input_hash': inputs[episode['episode_id']],
                       **score, 'rule_trace': trace}
                if permutation is not None:
                    row.update(permuted_prediction=raw_prediction, new_to_old=list(permutation.order),
                               trial=episode['trial'], reference_sha256=episode['original_reference_sha256'])
                rows.append(row)
                stream.write(json.dumps(row) + '\n'); stream.flush()
                print(f'[transfer] {dataset} {method} {policy} {index+1}/100', flush=True)
        planner.decision_observer = None
        report = {'episodes': 100, 'valid_plan_rate': sum(r['valid_plan'] and r['generation_error'] is None for r in rows)/100,
                  'generation_errors': sum(r['generation_error'] is not None for r in rows), 'input_hashes': inputs,
                  'violations': dict(Counter(v['reason'] for r in rows for v in r['violations'])),
                  'triggered_rejections': dict(sum((Counter(r['rule_trace']['triggered_rejections']) for r in rows), Counter()))}
        if dataset == 'libero':
            report['abstract_goal_success'] = sum(r['goal_success'] and r['generation_error'] is None for r in rows)/100
        if method == 'baseline':
            comparison = compare_baseline(rows, jsonl(Path(archive)/policy/'predictions.jsonl'), expected)
            freeze(root / 'baseline_reproduction.json', comparison)
            if not comparison['passed']:
                raise ValueError('archived baseline reproduction failed; stop campaign before method scoring')
        freeze(root / 'report.json', report)
        reports[policy] = report
    if reports['none']['input_hashes'] != reports['shared-verified']['input_hashes']:
        raise ValueError('constraint policy changed rendered inputs')
    freeze(output / 'summary.json', {'method': method, 'dataset': dataset, 'reports': reports, 'settings': settings})
    return reports


def main():
    import torch
    from .. import eval_planning
    from ..checkpoint import load_joint_checkpoint
    from ..main import _controller
    from ..models import JointQwenVLPlanner
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument

    parser = eval_planning.build_eval_parser()
    parser.set_defaults(split='test')
    parser.add_argument('--dataset', choices=['libero', 'rlbench'], required=True)
    parser.add_argument('--method', choices=['baseline', 'semantic-prompt', 'id-sensitivity'], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--permutation-root', type=Path)
    parser.add_argument('--baseline-output', type=Path)
    # --output is already supplied by the inherited planning parser.
    parser.set_defaults(output=None)
    args = parser.parse_args()
    if args.output is None:
        parser.error('--output is required')
    if args.method == 'id-sensitivity' and (args.permutation_root is None or args.baseline_output is None):
        parser.error('ID sensitivity requires --permutation-root and --baseline-output')
    if args.method != 'id-sensitivity' and (args.permutation_root is not None or args.baseline_output is not None):
        parser.error('permutation settings apply only to ID sensitivity')
    if args.no_dfa or args.no_dfa_official or args.input_probe != 'original' or args.limit or args.num_shards != 1:
        parser.error('external transfer uses its frozen manifest and both explicit constraint conditions')
    args.knowledge = 'metadata'; args.observation_mode = 'metadata-assisted'; args.split = 'test'
    args.vlabench_planning_dir = str(args.planning_dir)
    apply_research_arguments(args); apply_knowledge_argument(args)
    random.seed(args.seed); torch.manual_seed(args.seed)
    runtime, _ = eval_planning._prepare_eval(args, torch.device(args.device))
    planner = JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary, model_id=args.planner_model,
        load_in_4bit=args.load_in_4bit, decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller = _controller(args, torch.device(args.device))
    load_joint_checkpoint(args.checkpoint, runtime=runtime, planner=planner, controller=controller, map_location=args.device)
    planner.eval()
    source = Path(__file__).parents[1]
    files = [*Path(__file__).parent.glob('*.py'), source/'models.py', source/'eval_generalization.py',
             source/'generalization/shared_constraints.py', source/'generalization/rlbench_protocol.py',
             source/'generalization/libero_protocol.py']
    settings = {'checkpoint_sha256': sha(args.checkpoint), 'seed': args.seed, 'decoding': 'greedy',
                'load_in_4bit': args.load_in_4bit, 'archive_predictions_sha256': {
                    p: sha(args.archive/p/'predictions.jsonl') for p in POLICIES},
                'source_sha256': {str(p.relative_to(source)): sha(p) for p in files}}
    if args.method == 'id-sensitivity':
        baseline_settings = read(args.baseline_output/'settings.json')
        if (baseline_settings['checkpoint_sha256'] != settings['checkpoint_sha256']
                or baseline_settings['manifest_sha256'] != sha(args.manifest)):
            raise ValueError('sensitivity baseline checkpoint or original manifest differs')
        for name in ('models.py', 'eval_generalization.py', 'generalization/shared_constraints.py',
                     'generalization/rlbench_protocol.py', 'generalization/libero_protocol.py'):
            if settings['source_sha256'][name] != baseline_settings['source_sha256'][name]:
                raise ValueError('baseline generation or scoring source differs: '+name)
        for policy in POLICIES:
            if not read(args.baseline_output/policy/'baseline_reproduction.json')['passed']:
                raise ValueError('baseline reproduction has not passed')
            compare = compare_baseline(jsonl(args.baseline_output/policy/'predictions.jsonl'),
                jsonl(args.archive/policy/'predictions.jsonl'),
                [e['episode_id'] for e in episodes(read(args.manifest))])
            if not compare['passed']:
                raise ValueError('baseline predictions no longer reproduce archive')
        index = read(args.permutation_root/'manifest.json')
        if index['trials'] != 5 or index['episodes_per_trial'] != 100 or index['physical_episodes'] != 100:
            raise ValueError('sensitivity requires five transformations of the same 100 physical episodes')
        for trial in range(5):
            manifest = args.permutation_root/f'trial_{trial:03d}'/'manifest.json'
            if sha(manifest) != index['manifest_sha256'][str(trial)]:
                raise ValueError('permutation manifest changed')
            if read(manifest)['original_manifest_sha256'] != sha(args.manifest):
                raise ValueError('permutation original dataset differs')
            evaluate(planner, runtime, dataset=args.dataset, manifest_path=manifest, method=args.method,
                output=args.output/f'trial_{trial:03d}', settings={**settings, 'trial': trial,
                    'original_manifest_sha256': sha(args.manifest),
                    'baseline_settings_sha256': sha(args.baseline_output/'settings.json')}, archive=args.archive)
        freeze(args.output/'sensitivity_complete.json', {'trials': 5, 'episodes_per_trial': 100,
               'physical_episodes': 100, 'settings': settings})
        return
    if args.method != 'baseline':
        for policy in POLICIES:
            gate = read(args.output.parent/'baseline'/policy/'baseline_reproduction.json')
            if not gate['passed']:
                raise ValueError('baseline reproduction gate has not passed')
    evaluate(planner, runtime, dataset=args.dataset, manifest_path=args.manifest, method=args.method,
             output=args.output, settings=settings, archive=args.archive)


if __name__ == '__main__':
    main()
