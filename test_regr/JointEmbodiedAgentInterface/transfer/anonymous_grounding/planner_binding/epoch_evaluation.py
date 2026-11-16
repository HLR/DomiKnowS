"""Evaluate every saved binding epoch without changing training or selection."""
import argparse
import json
from collections import Counter
from pathlib import Path
import random
import sys

from ...protocol import read, freeze, sha, digest, episodes, public_input, context
from ...campaign import save, utc

from .verification_protocol import POLICIES, snapshot_key, selected_datasets, exact_policy_search, enrich_decision


def verify_inputs(config):
    for path, expected in config['input_hashes'].items():
        if sha(path) != expected:
            raise ValueError('frozen epoch-evaluation input changed: ' + path)
    if 'source_manifest' in config:
        for name, expected in read(config['source_manifest'])['files'].items():
            if sha(Path(config['source'])/name) != expected:
                raise ValueError('frozen evaluation source changed: ' + name)


def episode_order(records, expected, count):
    indexed = {r['episode_id']: r for r in records}
    if len(records) != count or len(indexed) != count or set(indexed) != set(expected):
        raise ValueError('evaluation identities or coverage differ')
    return [indexed[i] for i in expected]


def prepare(config_path):
    """Reuse immutable visual features; encode missing public instructions only."""
    import torch
    from transformers import AutoTokenizer
    from test_regr.VLABenchAgentInterface.dataset import _read_instruction
    from test_regr.VLABenchAgentInterface.simulation.controller import FrozenSigLIPEncoder
    from ...train import tensor_digest
    from ..object_fixes.grounding import phrases
    c = read(config_path); verify_inputs(c); out = Path(c['output'])
    observations = read(c['vla_observations'])['vlabench']
    identities = read(c['split_manifest'])['splits']['test']
    observations = episode_order(observations, identities, 450)
    proof = read(c['vla_features'])
    cached = {r['episode_id']: r for r in proof['episodes']}
    selected = []
    for r in observations:
        for f, expected in r['files'].items():
            if sha(f) != expected: raise ValueError('VLABench observation changed')
        f = cached[r['episode_id']]
        if sha(f['path']) != f['sha256']: raise ValueError('reused visual feature changed')
        selected.append(f)
    prior = read(c['text_manifest'])
    if prior['references_read'] or sha(prior['path']) != prior['sha256']:
        raise ValueError('public instruction cache differs')
    if prior['encoder_sha256'] != proof['encoder_sha256']:
        raise ValueError('visual/text encoders differ')
    text = torch.load(prior['path'], map_location='cpu', weights_only=True)
    instruction_files = {}; instructions = set()
    for r in observations:
        folder = Path(r['raw_image']).parent
        instructions.add(_read_instruction(folder))
        for f in folder.iterdir():
            if f.is_file() and 'instruction' in f.name.lower(): instruction_files[str(f)] = sha(f)
    missing = sorted(instructions - set(text))
    if missing:
        encoder = FrozenSigLIPEncoder(c['vision_model'], local_files_only=True, image_size=384).eval()
        payload = torch.load(c['initialization'], map_location='cpu', weights_only=False)
        encoder.load_state_dict({k[len('image_encoder.'):]: v for k, v in payload['controller'].items()
                                 if k.startswith('image_encoder.')}, strict=True)
        del payload
        if tensor_digest(encoder) != proof['encoder_sha256']: raise ValueError('text encoder changed')
        tokenizer = AutoTokenizer.from_pretrained(c['vision_model'], local_files_only=True)
        for start in range(0, len(missing), 16):
            batch = missing[start:start+16]; roles = {s: phrases(s) for s in batch}
            queries = ['a photo of ' + (roles[s][r] or 'an unidentified object') + '.'
                       for s in batch for r in ('source', 'destination')]
            tokens = tokenizer(queries, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
            with torch.no_grad(): vectors = encoder.model.get_text_features(**tokens)
            if not torch.is_tensor(vectors):
                vectors = vectors.text_embeds if getattr(vectors, 'text_embeds', None) is not None else vectors.pooler_output
            for i, s in enumerate(batch): text[s] = {'phrases': roles[s], 'features': vectors[i*2:i*2+2].cpu().half()}
    folder = out/'prepared'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/'text_features.pt'
    if not path.exists(): torch.save(text, path)
    freeze(folder/'manifest.json', {
        'config_sha256': sha(config_path), 'vla_ids': identities,
        'visual_encoder_sha256': proof['encoder_sha256'], 'features': selected,
        'reused_feature_manifest_sha256': sha(c['vla_features']),
        'feature_reuse': 'same frozen visual encoder; no dependency on trained planner or binding-head weights',
        'text_path': str(path), 'text_sha256': sha(path), 'missing_public_instructions_encoded': len(missing),
        'public_instruction_files': instruction_files, 'references_read_for_priors': False,
        'external_ids': [e['episode_id'] for e in episodes(read(c['external_manifest']))]})
    print('[epoch-eval] preparation complete; 450 VLABench and 100 fresh LIBERO identities frozen', flush=True)


def decode(planner, head, runtime, bound, values, roles, policy, public=None):
    import torch
    from ....models import FreeDecoding
    from ....generalization.shared_constraints import bind_shared, rule_trace
    from ....eval_generalization import input_record
    from test_regr.VLABenchAgentInterface.plan_codec import labels_to_plan
    from ..grounder import condition
    from .binding import operation_bindings
    if policy not in POLICIES: raise ValueError('unknown evaluation policy')
    effective = condition(bound, roles); planner.free_decoding = policy == 'none'
    if policy == 'none':
        dfa = FreeDecoding(runtime.vlabench_vocabulary.label_count, runtime.vlabench_vocabulary.eos_label)
    elif policy == 'shared-verified':
        dfa = bind_shared(runtime.vlabench_vocabulary, runtime.max_vlabench_operations, bound)
    elif policy == 'executor':
        from ..object_fixes.contract import ExecutorContract
        if public is None: raise ValueError('executor policy requires public LIBERO metadata')
        dfa = ExecutorContract(bind_shared(runtime.vlabench_vocabulary, runtime.max_vlabench_operations, bound),
            runtime.vlabench_vocabulary, public, {'pick':'target_entity_name','place':'target_container_name'}, runtime.max_vlabench_operations)
    else: dfa = runtime.dfa_for('vlabench', bound)
    rendered = input_record(planner, effective); original = planner._prepare_inputs_and_boundaries
    verified = False
    def checked(domain, actual, active_tokens=()):
        nonlocal verified
        if not verified:
            if domain != 'vlabench' or input_record(planner, actual) != rendered:
                raise AssertionError('constraint policy changed neural inputs')
            verified = True
        return original(domain, actual, active_tokens)
    planner._prepare_inputs_and_boundaries = checked
    labels = []; events = []; error = None; retries = []
    try:
        with torch.no_grad(), exact_policy_search(dfa, retries), operation_bindings(planner, head, values['features'], values['visible']) as events, runtime.domain_scope('vlabench'):
            labels, _ = planner.sample_labels('vlabench', effective, dfa,
                max_steps=runtime.max_vlabench_operations*5+1, deterministic=True)
        prediction = labels_to_plan(labels, runtime.vlabench_vocabulary, world=runtime.world.vlabench)
    except Exception as exc:
        from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
        if isinstance(exc, SemanticSearchLimitExceeded): raise
        if not isinstance(exc, (ValueError, RuntimeError, KeyError)): raise
        prediction = []; error = {'type': type(exc).__name__, 'message': str(exc)}
    finally: planner._prepare_inputs_and_boundaries = original
    trace = (dfa.trace(labels) if policy == 'executor' else rule_trace(dfa, labels)) if policy != 'none' else {'triggered_rejections': {}}
    trace['search_retries'] = retries
    bindings = [{'step': e['step'], 'role': e['role'],
                 'probabilities': torch.softmax(e['logits'][:, e['column']], 0).cpu().tolist()} for e in events]
    return prediction, error, rendered, trace, bindings


def evaluate():
    import torch
    from .... import eval_planning
    from ....main import _controller, _evaluate_eai
    from ....models import JointQwenVLPlanner
    from ....utils.planner_context import planner_context
    from ....utils.paired_references import PairedReferenceAudit
    from ....generalization.libero_protocol import abstract_score
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    from test_regr.VLABenchAgentInterface.repair_density_labels import updated_density_examples
    from .agent import load_binding, primary
    p = eval_planning.build_eval_parser()
    p.add_argument('--epoch-config', type=Path, required=True)
    p.add_argument('--epoch', type=int, choices=range(1, 5))
    p.add_argument('--snapshot', choices=['initialization', *[f'epoch-{n}' for n in range(1, 5)]])
    p.add_argument('--datasets', nargs='+', choices=['vlabench', 'libero'])
    p.add_argument('--skip-eai', action='store_true')
    p.add_argument('--constraint-policy', choices=POLICIES, required=True)
    p.add_argument('--experiment-output', type=Path, required=True)
    a = p.parse_args(); c = read(a.epoch_config); verify_inputs(c); out = a.experiment_output
    key, label = snapshot_key(a.epoch, a.snapshot); datasets = selected_datasets(a.constraint_policy, a.datasets)
    spec = c['snapshots'][key]
    if str(a.checkpoint) != spec['path'] or sha(a.checkpoint) != spec['sha256']:
        raise ValueError('evaluation checkpoint changed')
    proof = read(Path(c['output'])/'prepared/manifest.json')
    if proof['config_sha256'] != sha(a.epoch_config) or sha(proof['text_path']) != proof['text_sha256']:
        raise ValueError('prepared evaluation fingerprint changed')
    a.knowledge = 'metadata'; a.observation_mode = 'metadata-assisted'; a.split = 'test'
    a.vlabench_planning_dir = str(a.planning_dir)
    apply_research_arguments(a); apply_knowledge_argument(a)
    random.seed(42); torch.manual_seed(42)
    runtime, splits = eval_planning._prepare_eval(a, torch.device(a.device))
    if [e.episode_id for e in splits['test']] != proof['vla_ids']:
        raise ValueError('frozen VLABench test split changed')
    planner = JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary, model_id=a.planner_model,
        load_in_4bit=True, decoder_hidden_size=a.planner_decoder_hidden_dim)
    controller = _controller(a, torch.device(a.device))
    head, payload = load_binding(a.checkpoint, runtime=runtime, planner=planner, controller=controller, device=a.device)
    if sha(Path(a.checkpoint).with_name(payload['metrics']['grounding']['head_file'])) != spec.get('head_sha256', payload['metrics']['grounding']['sha256']):
        raise ValueError('checkpoint-specific grounding head changed')
    if key == 'initialization' and not payload['metrics'].get('initialization_before_updates'):
        raise ValueError('initialization must precede every continuation update')
    if head.visual_encoder_sha256 != proof['visual_encoder_sha256']: raise ValueError('reused encoder differs')
    planner.eval(); head.eval(); _ANONYMOUS_OBJECTS.set(True)
    settings = {'config_sha256': sha(a.epoch_config), 'epoch': a.epoch, 'snapshot': label, 'datasets': datasets, 'checkpoint_sha256': sha(a.checkpoint),
        'grounding_sha256': payload['metrics']['grounding']['sha256'], 'prepared_sha256': sha(Path(c['output'])/'prepared/manifest.json'),
        'policy': a.constraint_policy, 'seed': 42, 'selection': 'all four epochs; diagnostic only; original best unchanged',
        'source_manifest_sha256': c['evaluation_source_sha256']}
    freeze(out/'settings.json', settings)
    if (out/'complete.json').exists(): return
    text = torch.load(proof['text_path'], map_location='cpu', weights_only=True)
    if a.constraint_policy == 'none' and not a.skip_eai:
        from test_regr.EmbodiedAgentInterface.dataset import load_eai_dataset, split_train_dev
        from test_regr.EmbodiedAgentInterface.reward import make_eai_reward_function
        data = load_eai_dataset(a.eai_dataset, split=a.eai_split, data_path=a.eai_data_path, device='cpu', max_steps=a.eai_max_steps)
        _, valid = split_train_dev(data, dev_fraction=.2)
        if len(valid) != 88: raise ValueError('canonical EAI count changed')
        for item in valid:
            item['reward_function'] = make_eai_reward_function(item, vocabulary=runtime.eai_vocabulary, mode='dense', world_bundle=runtime.world.eai)
        planner.free_decoding = False
        with (out/'eai_predictions.jsonl').open('x') as stream:
            def observer(item, labels, result):
                stream.write(json.dumps({'instruction': planner_context('eai', item)['instruction'], 'labels': labels, 'goal': result}, default=str)+'\n'); stream.flush()
            from .epoch_recovery import retry_eai_search
            retries = []
            with retry_eai_search(runtime, retries):
                eai_report = _evaluate_eai(planner, runtime, valid, limit=None, prediction_observer=observer)
            freeze(out/'eai_search_retries.json', retries)
            freeze(out/'eai.json', eai_report)
    observations = {r['episode_id']: r for r in read(c['vla_observations'])['vlabench']}
    features = {r['episode_id']: r for r in proof['features']}
    fresh_features = {r['episode_id']: r for r in read(c['fresh_features'])['episodes']}
    fresh_priors = {r['episode_id']: r for r in read(c['fresh_priors'])['episodes']}
    # Freeze all scoring references before any VLABench input is decoded.
    corrected = {e.episode_id: e for e in updated_density_examples(splits['test'])}
    freeze(out/'references.json', {e.episode_id: {'original': e.operation_sequence,
           'scored': corrected[e.episode_id].operation_sequence} for e in splits['test']})
    for dataset in datasets:
        folder = out/dataset; folder.mkdir(parents=True, exist_ok=True)
        selected = splits['test'] if dataset == 'vlabench' else episodes(read(c['external_manifest']))
        expected = 450 if dataset == 'vlabench' else 100
        if len(selected) != expected: raise ValueError('incomplete evaluation dataset')
        hashes = {}; rows = []; triggers = Counter(); errors = 0
        with (folder/'predictions.jsonl').open('x') as stream, (folder/'decisions.jsonl').open('x') as decisions:
            audit = PairedReferenceAudit(stream, corrector=lambda values: [corrected[e.episode_id] for e in values]) if dataset == 'vlabench' else None
            for i, e in enumerate(selected):
                if dataset == 'vlabench':
                    identity = e.episode_id; observation = observations[identity]; bound = planner_context('vlabench', e)
                    if tuple(observation['canonical_entities']) != tuple(e.entities): raise ValueError('canonical pointers differ')
                    if any(sha(f) != h for f, h in observation['files'].items()): raise ValueError('observation changed')
                    bound['images'] = (observation['raw_image'], observation['canonical_image']); record = features[identity]; public = None
                else:
                    identity = e['episode_id']; public = public_input(e); bound = context(public, 'libero'); record = fresh_features[identity]
                if sha(record['path']) != record['sha256']: raise ValueError('visual feature changed')
                raw = torch.load(record['path'], map_location=a.device, weights_only=True)
                values = {'features': raw['features'].float(), 'visible': raw['visible']}
                roles = primary(bound, text, values['features'], values['visible'])[1] if dataset == 'vlabench' else fresh_priors[identity]['predictions']
                planner.decision_observer = lambda row: decisions.write(json.dumps({'episode_id': identity, **enrich_decision(row)}, default=str)+'\n')
                prediction, error, rendered, trace, bindings = decode(planner, head, runtime, bound, values, roles, a.constraint_policy, public)
                hashes[identity] = digest(rendered); errors += int(error is not None); triggers.update(trace['triggered_rejections'])
                freeze(folder/'rendered'/f'{i:04d}.json', {'episode_id': identity, 'input': rendered, 'predicted_roles': roles})
                decisions.write(json.dumps({'episode_id': identity, 'operation_grounding': bindings, 'rule_trace': trace})+'\n'); decisions.flush()
                if audit: audit.observe(e, prediction, generation_error=error)
                else:
                    if sha(e['scoring_private']) != e['files'][e['scoring_private']]: raise ValueError('held-out goal changed')
                    score = abstract_score(prediction, read(e['scoring_private']))
                    row = {'episode_id': identity, 'prediction': prediction, 'generation_error': error,
                           'input_hash': hashes[identity], 'suite': e['suite'], **score}
                    rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
                save(out/'status.json', {'dataset': dataset, 'completed': i+1, 'total': expected, 'updated_utc': utc()})
                if (i+1) % 10 == 0: print(f'[epoch-eval] {label} {a.constraint_policy} {dataset} {i+1}/{expected}', flush=True)
            if audit:
                report = audit.report(); metric = report['corrected']['planning_metrics']
                dimensions = metric['official_mapping']['legacy_total_score_by_dimension']
                report.update(dimensions=dimensions, Macro=sum(dimensions.values())/len(dimensions),
                              progress_score=metric['overall']['score'], exact_match=report['corrected']['strict_plan_match'])
            else:
                report = {'abstract_goal_success': sum(r['goal_success'] and not r['generation_error'] for r in rows)/expected,
                          'valid_plan_rate': sum(r['valid_plan'] and not r['generation_error'] for r in rows)/expected}
        report.update(examples=expected, generation_errors=errors, input_hashes=hashes, triggered_rejections=dict(triggers), settings=settings)
        freeze(folder/'report.json', report)
    planner.decision_observer = None
    freeze(out/'complete.json', settings)
    save(out/'status.json', {'stage': 'complete', 'updated_utc': utc()})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        parser = argparse.ArgumentParser(); parser.add_argument('mode'); parser.add_argument('--config', type=Path, required=True)
        prepare(parser.parse_args().config)
    else: evaluate()
