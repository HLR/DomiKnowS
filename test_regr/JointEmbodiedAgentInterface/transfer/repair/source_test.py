"""Strict-input source test scoring for frozen decoder-repair snapshots."""
import argparse
import json
from pathlib import Path

from ..protocol import read, freeze, sha, digest
from ..campaign import save, utc
from ..matched_baseline.protocol import common_prompt
from ..matched_baseline.evaluate import input_record
from ...generalization.executor_policy import CAPABILITIES, EXECUTOR, POLICIES, bind_plan_policy
from ...utils.paired_references import PairedReferenceAudit
from .protocol import verify, anonymous_entities, normalize_anonymous_prediction
from .common import build_runtime, fresh_planner, load_checkpoint, native_decode, graph_decode
from .input_protocol import IMAGE_INSTRUCTION, generation_context, contract_record
from .surface import surface_decoder
from .evaluate import SYSTEMS


def ordered_observations(manifest, identities):
    rows = manifest['vlabench']; indexed = {r['episode_id']: r for r in rows}
    if not manifest['complete'] or manifest['references_read'] or manifest['test_training_used']:
        raise ValueError('observation-only test manifest required')
    if len(rows) != 450 or len(indexed) != 450 or set(indexed) != set(identities):
        raise ValueError('require exact frozen 450 test identities')
    return [indexed[i] for i in identities]


def observation_inputs(example, observation, vocabulary):
    if tuple(observation['canonical_entities']) != tuple(example.entities):
        raise ValueError('canonical entity pointers differ')
    if any(sha(p) != expected for p, expected in observation['files'].items()):
        raise ValueError('frozen original-image observation changed')
    public = {'instruction': example.instruction, 'entities': anonymous_entities(len(example.entities)),
        'images': [observation['raw_image'], observation['canonical_image']]}
    # No source fixture evidence has been independently verified. Unknown
    # objects stay eligible; no substring/answer-based fixture guesses.
    constraint = {'entity_table': public['entities']}
    return public, generation_context(public, vocabulary), constraint


def metric_columns(report):
    metrics = report['corrected']['planning_metrics']
    # Match the report's original six-dimension Macro on the 0–80 scale.
    values = metrics['official_mapping']['legacy_total_score_by_dimension']
    if len(values) != 6:
        raise ValueError('all six official dimensions required')
    return {'Macro': sum(values.values())/6, 'progress_score': metrics['overall']['score'],
        'dimension_scores': values, 'Macro_definition': 'mean of six official-mapping legacy scores, 0–80',
        'coverage_Macro_0_100': metrics['Macro']}


def collect(c):
    root=Path(c['output']); comparison=Path(c['comparison_root']); parent=Path(c['training_parent'])
    selection=read(comparison/'frozen_selection.json')['systems']
    external=read(comparison/'results.json'); rows=[];common=None;files={}
    for system in SYSTEMS:
        spec=selection[system]
        if sha(spec['path'])!=spec['sha256']:raise ValueError('selected checkpoint changed')
        if system in ('base','joint-initialization'):
            eai_path=parent/'native_base/source_validation/eai.json'
            proof=read(parent/'initialization_match.json')
            if not proof['passed'] or not proof['identical_saved_planner_weights']:
                raise ValueError('initialization EAI component equality not verified')
            for path,expected in proof['files'].items():
                if sha(path)!=expected:raise ValueError('initialization equality proof changed')
        else:
            arm='native' if system=='native-trained' else 'joint'
            eai_path=parent/'arms'/arm/f"epoch_{spec['source_epoch']:03d}"/'eai.json'
        eai=read(eai_path)
        if eai['examples']!=88:raise ValueError('canonical EAI coverage differs')
        files[str(eai_path)]=sha(eai_path)
        for policy in ('none','shared-verified'):
            path=root/'evaluation'/system/policy/'report.json';report=read(path)
            if report['examples']!=450 or report['settings']['split']!='test':raise ValueError('test coverage differs')
            if report['settings']['checkpoint']!=spec:raise ValueError('checkpoint-specific report differs')
            if common is None:common=report['input_hashes']
            if common!=report['input_hashes']:raise ValueError('neural inputs differ across systems/policies')
            libero=next(r for r in external['rows'] if r['system']==system and r['policy']==policy)
            rows.append({'system':system,'source_epoch':spec.get('source_epoch',0),
                'source_qualified':spec['qualified'],'policy':policy,
                'eai_goal_percent':100*eai['goal_success'],'vla_Macro_0_80':report['Macro'],
                'progress_score_0_100':report['progress_score'],
                'libero_abstract_percent':libero['abstract_percent'],
                'libero_physical_percent':libero['native_percent']})
            files[str(path)]=sha(path)
    result={'complete':True,'rows':rows,'input_protocol':IMAGE_INSTRUCTION,'vlabench_split':'test',
        'vlabench_examples':450,'eai_examples':88,'libero_examples':100,
        'Macro_definition':'mean of six official-mapping legacy dimension scores, 0–80',
        'EAI_protocol':'unchanged canonical compiled-rule evaluation; not a no-DFA EAI claim',
        'initialization_EAI_reuse':'identical saved LoRA/heads, verified initialization hashes',
        'reused_files':files,'libero_results_sha256':sha(comparison/'results.json')}
    freeze(root/'table_results.json',result)
    return result


def main():
    import torch
    from test_regr.VLABenchAgentInterface.native_planning import byte_token_map
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', choices=SYSTEMS)
    parser.add_argument('--policy', choices=POLICIES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args(); c = read(args.config); verify(c)
    if args.collect:
        collect(c);return
    if not args.arm or not args.policy or not args.output:
        parser.error('--arm, --policy and --output are required for inference')
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    selection = read(Path(c['comparison_root'])/'frozen_selection.json')['systems'][args.arm]
    if sha(selection['path']) != selection['sha256']:
        raise ValueError('frozen checkpoint changed')
    settings = {**contract_record(IMAGE_INSTRUCTION), 'config_sha256': sha(args.config),
        'checkpoint': selection, 'system': args.arm, 'policy': args.policy, 'split': 'test',
        'observations_sha256': sha(c['source_test_observations']),
        'source_manifest_sha256': sha(c['source_manifest']), 'native_budget': 768, 'graph_budget': 61,
        'fixture_evidence': 'unknown; no fixture exclusions guessed', 'max_operations': 12}
    if args.policy in (CAPABILITIES, EXECUTOR):
        settings['executor_evidence'] = 'unknown; effective policy is shared-verified'
    freeze(output/'settings.json', settings)
    if (output/'report.json').exists():
        return
    if (output/'raw_predictions.jsonl').exists():
        raise ValueError('partial evaluation preserved; use a new output')
    _, (runtime, _, _, splits, _) = build_runtime(c, output/'runtime')
    test = splits['test']; identities = read(c['split_manifest'])['splits']['test']
    if [e.episode_id for e in test] != identities:
        raise ValueError('runtime test split differs')
    observations = ordered_observations(read(c['source_test_observations']), identities)
    freeze(output/'references.json', {e.episode_id: {'reference': e.operation_sequence,
        'dependency': e.dependency, 'reference_policy': 'frozen scene-verified density targets'} for e in test})
    arm = SYSTEMS[args.arm]; planner = fresh_planner(c, runtime, arm)
    load_checkpoint(selection['path'], planner, arm); planner.eval(); vocabulary = runtime.vlabench_vocabulary
    tokens = byte_token_map(planner.processor.tokenizer) if arm == 'native' else None
    cache = {}; errors = 0; hashes = {}
    with common_prompt(planner), (output/'predictions.jsonl').open('x') as scored, (output/'raw_predictions.jsonl').open('x') as raw_stream:
        audit = PairedReferenceAudit(scored, corrector=lambda values: values)
        for index, (example, observation) in enumerate(zip(test, observations)):
            public, bound, constraint = observation_inputs(example, observation, vocabulary)
            inputs, _ = planner._prepare_inputs_and_boundaries('vlabench', bound, ())
            rendered = input_record(planner.processor, bound['_matched_core'], public, None, inputs)
            hashes[example.episode_id] = digest(rendered)
            freeze(output/'inputs'/f'{index:04d}.json', {'episode_id': example.episode_id, **rendered})
            prediction = []; raw_prediction = []; labels = []; raw = ''; changes = []; error = None
            try:
                policy = None
                if args.policy != 'none':
                    if arm == 'native':
                        key = len(public['entities'])
                        if key not in cache:
                            cache[key] = surface_decoder(tokens, planner.processor.tokenizer.eos_token_id,
                                {'entities': public['entities'], 'movable': public['entities']}, vocabulary, 12,
                                policy_name=args.policy)
                        policy = cache[key]
                    else:
                        policy = bind_plan_policy(args.policy, vocabulary, 12, constraint)
                raw_prediction, labels, raw = (native_decode(planner, bound, policy) if arm == 'native'
                    else graph_decode(planner, runtime, bound, policy))
                prediction, changes = normalize_anonymous_prediction(raw_prediction, len(public['entities']), vocabulary.argument_keys)
                # Validate only the wire codec before independent scoring. In
                # particular, no named answer can be grounded by the scorer.
                from test_regr.VLABenchAgentInterface.plan_codec import plan_to_tokens
                for token in plan_to_tokens(prediction, public['entities'], world=runtime.world.vlabench):
                    vocabulary.label_for_token(token)
            except (ValueError, RuntimeError, KeyError) as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                error = {'type': type(exc).__name__, 'message': str(exc)}; prediction = []
                labels = getattr(exc, 'labels', labels); raw = getattr(exc, 'raw_text', raw)
            errors += bool(error)
            raw_stream.write(json.dumps({'episode_id': example.episode_id, 'prediction': prediction,
                'raw_prediction': raw_prediction, 'labels': labels, 'raw_text': raw, 'normalizations': changes,
                'generation_error': error, 'input_hash': hashes[example.episode_id]})+'\n'); raw_stream.flush()
            audit.observe(example, prediction, generation_error=error)
            save(output/'status.json', {'stage': 'test-evaluation', 'completed': index+1,
                'total': 450, 'system': args.arm, 'policy': args.policy, 'updated_utc': utc()})
            if (index+1) % 25 == 0:
                print(f'[source-test] {args.arm} {args.policy} {index+1}/450', flush=True)
        report = audit.report()
    report.update(metric_columns(report)); report.update(settings=settings, generation_errors=errors, input_hashes=hashes)
    freeze(output/'report.json', report)
    save(output/'status.json', {'stage': 'complete', 'examples': 450, 'updated_utc': utc()})


if __name__ == '__main__':
    main()
