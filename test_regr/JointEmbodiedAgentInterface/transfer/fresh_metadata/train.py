"""Train a fresh Joint planner and two independent supervised grounding heads."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random

import torch

from domiknows.generation.learners.common.losses import allowed_label_sets
from ...main import _evaluate_eai
from ...program import DFALossConfig, JointSolverPOIProgram
from ...utils.paired_references import PairedReferenceAudit
from ...utils.planner_context import planner_context
from ..anonymous_grounding.audits import SourceAudits
from ..anonymous_grounding.grounder import RoleGrounder, known_pointer_steps
from ..anonymous_grounding.planner_binding.epoch_recovery import retry_eai_search
from ..anonymous_grounding.planner_binding.objective import loss_terms
from ..anonymous_grounding.train import records_for, visual, public_record
from ..batch_gradients import BatchGradientManager
from ..campaign import save, utc
from ..matched_baseline.protocol import common_prompt
from ..permutation import EntityPermutation
from ..protocol import digest, freeze, read, sha
from ..repair.common import build_runtime, fresh_planner, graph_decode, seed
from ..repair.input_protocol import generation_context
from ..repair.protocol import normalize_anonymous_prediction, verify
from ..train import tensor_digest
from .objectives import grounding_losses, permutation_loss, target_prefix


RECIPE = 'metadata-pmd-pcgrad-grounding-preference'
FORMAT = 'fresh-joint-metadata-preference-v1'
BATCH_SIZE = 8
EPOCHS = 4
MASS_WEIGHT = .5
PREFERENCE_WEIGHT = .5
CONSISTENCY_WEIGHT = .1


def status(root, stage, **details):
    save(root/'status.json', {'stage': stage, 'updated_utc': utc(), **details})


def neural_context(row, vocabulary):
    context = planner_context('vlabench', row['item'])
    return generation_context({key: context[key] for key in ('instruction', 'images')} |
                              {'entities': context['entity_table']}, vocabulary)


def source_data(config, runtime, splits):
    manifest = read(config['source_observations'])
    features = read(config['features'])
    if (not manifest['complete'] or manifest['test_used'] or manifest['references_read']
            or not features['complete'] or features['source_manifest_sha256'] != sha(config['source_observations'])):
        raise ValueError('source manifests must be complete and bound to observational inputs')
    bindings = {r['episode_id']: r for r in manifest['episodes']}
    feature_records = {r['episode_id']: r for r in features['episodes']}
    ids = {e.episode_id for split in ('train', 'validation') for e in splits[split]}
    if len(ids) != 4050 or len(bindings) != len(manifest['episodes']) or len(feature_records) != len(features['episodes']):
        raise ValueError('duplicate or incomplete source coverage')
    if ids != set(bindings) or ids != set(feature_records):
        raise ValueError('unexpected non-source identities in features or observations')
    cache = {}
    for identity, record in bindings.items():
        if any(sha(path) != expected for path, expected in record['files'].items()):
            raise ValueError(f'changed original or relabeled input for {identity}')
        feature = feature_records[identity]
        if sha(feature['path']) != feature['sha256']:
            raise ValueError(f'changed frozen visual features for {identity}')
        cache[identity] = torch.load(feature['path'], map_location='cpu', weights_only=True)
        if len(cache[identity]['visible']) != len(record['canonical_entities']):
            raise ValueError('frozen feature/pointer count changed')

    def views(split, epoch, augment):
        return records_for(runtime, splits[split], bindings, feature_records, epoch,
                           augmented=augment, use_training_schedule=False)

    return views, cache, features


def audit_permutations(root, config, runtime, splits, canonical_items):
    """Check entire allowed sets on a frozen task-stratified sample before CUDA."""
    auditor = SourceAudits(root/'audits', {'config_sha256': sha(root/'config.json')})
    bindings = {r['episode_id']: r for r in read(config['source_observations'])['episodes']}
    report = {}
    for split, epochs in (('train', range(4)), ('validation', range(1))):
        tasks = {}
        for index, example in enumerate(splits[split]):
            task = example.episode_id.rsplit('/', 1)[0]
            group = tasks.setdefault(task, [])
            if len(group) < 2:
                group.append(index)
        indices = sorted(i for group in tasks.values() for i in group)
        for epoch in epochs:
            examples = [splits[split][i] for i in indices]
            selected, orders, _ = records_for(runtime, examples, bindings, dict.fromkeys(bindings),
                                              epoch, augmented=True, use_training_schedule=False)
            key = f'{split}_permutation_{epoch}'
            auditor.equivalence(runtime, [canonical_items[split][i] for i in indices], selected, orders, key)
            report[key] = len(indices)
    freeze(root/'permutation_audit.json', {'passed': True, 'complete_allowed_sets': report,
                                          'training_checks_each_reference_path': True})


@contextmanager
def capture_queries(planner):
    captured = []
    handle = planner.label_heads['vlabench'].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    try:
        yield captured
    finally:
        handle.remove()


def reference_policy(runtime, row, labels, vocabulary):
    context = planner_context('vlabench', row['item'])
    dfa = runtime.dfa_for('vlabench', context)
    allowed = allowed_label_sets(dfa, labels, vocabulary.label_count,
                                 lookahead=True, max_steps=runtime.max_vlabench_operations*5+1)
    state = dfa.start_state
    for step, label in enumerate(labels.tolist()):
        if label not in allowed.get(step, ()):
            raise ValueError(f'metadata DFA excludes the reference for {row["episode_id"]} at step {step}')
        state = dfa.step(state, label)
        if state is None:
            raise ValueError('reference walked into a nonaccepting DFA state')
    if not dfa.is_accepting(state):
        raise ValueError('reference must end in an accepting DFA state')
    return allowed


def accumulated_step(planner, runtime, program, heads, row, alternative, cache):
    vocabulary = runtime.vlabench_vocabulary
    context = neural_context(row, vocabulary)
    labels = row['item']['target_plan_labels'].to(planner.device)
    targets = target_prefix(labels, vocabulary)
    features, visible = visual(row, cache, planner.device)
    observed = known_pointer_steps(targets, vocabulary, visible)
    allowed = reference_policy(runtime, row, targets, vocabulary)
    pmd_model = program.pmd_programs['vlabench']
    with capture_queries(planner) as captured:
        if bool(observed.all()):
            _, _, _, builder = pmd_model.model({'planner_context': context,
                'token_positions': torch.arange(labels.numel()), 'target_plan_labels': labels})
            pmd, *_ = pmd_model.cmodel(builder)
            logits = planner.last_logits[:len(targets)]
        else:
            # Keep all observed CE tokens. Do not expose an unknown target to primal-dual supervision.
            logits = planner.sequence_logits('vlabench', context, planner.shift_right('vlabench', targets)).squeeze(0)
            pmd = logits.new_zeros(())
    if not captured or len(captured[-1]) < len(targets):
        raise ValueError('teacher-forced causal boundary queries missing')
    role, rank, n_role, n_rank = grounding_losses(
        heads, captured[-1], features, visible, targets, vocabulary, allowed, row['order'])
    loss, terms = loss_terms(logits, targets, allowed, observed, 'masked')
    if terms['masked_ce_misses']:
        raise ValueError('masked CE fallback is forbidden in the audited recipe')
    consistency = logits.new_zeros(())
    if alternative is not None and alternative['order'] != row['order']:
        alt_targets = target_prefix(alternative['item']['target_plan_labels'].to(planner.device), vocabulary)
        alt_allowed = reference_policy(runtime, alternative, alt_targets, vocabulary)
        _, alt_visible = visual(alternative, cache, planner.device)
        with torch.no_grad():
            alt_logits = planner.sequence_logits('vlabench', neural_context(alternative, vocabulary),
                planner.shift_right('vlabench', alt_targets)).squeeze(0)
        consistency = permutation_loss(logits, targets, row['order'], allowed, visible, alt_logits, alt_targets,
                                       alternative['order'], alt_allowed, alt_visible, vocabulary)
    program._dual_grads('vlabench', pmd / BATCH_SIZE)
    terms.update(pmd=float(pmd.detach()), role=float(role.detach()), preference=float(rank.detach()),
                 consistency=float(consistency.detach()), role_cases=n_role, preference_cases=n_rank,
                 pmd_examples=int(bool(observed.all())))
    return loss + pmd + CONSISTENCY_WEIGHT * consistency, role + PREFERENCE_WEIGHT * rank, terms


def save_snapshot(path, planner, heads, optimizer, head_optimizer, program, root, epoch):
    if path.exists():
        raise ValueError('refusing to overwrite an existing snapshot')
    state = {n: p.detach().cpu() for n, p in planner.named_parameters()
             if 'lora_' in n or n.startswith('label_heads.')}
    tmp = path.with_suffix('.tmp')
    torch.save({'format': FORMAT, 'epoch': epoch, 'settings_sha256': sha(root/'settings.json'),
                'planner': state, 'grounding': heads[0].state_dict(), 'preference': heads[1].state_dict(),
                'planner_optimizer': optimizer.state_dict(), 'head_optimizer': head_optimizer.state_dict(),
                'pmd': program.pmd_programs['vlabench'].cmodel.state_dict(),
                'dual_optimizer': program.dual_optimizers['vlabench'].state_dict(),
                'dual_rounds': program.dfa_loss_rounds}, tmp)
    tmp.replace(path)
    freeze(path.with_suffix('.json'), {'epoch': epoch, 'sha256': sha(path),
        'initialization_before_updates': epoch == 0, 'pretrained_base': True, 'format': FORMAT})


def validate(planner, runtime, base, permuted, source, eai_valid, root, epoch):
    vocabulary = runtime.vlabench_vocabulary
    out = root/f'validation_{epoch:03d}'
    out.mkdir(exist_ok=True)
    planner.eval()
    metrics = []
    for name, rows in (('canonical', base), ('permuted', permuted)):
        pointer_correct, pointer_cases, errors = 0, 0, 0
        with (out/f'{name}.jsonl').open('x', encoding='utf-8') as stream:
            audit = PairedReferenceAudit(stream, corrector=lambda values: values)
            for index, (row, example) in enumerate(zip(rows, source)):
                context = neural_context(row, vocabulary)
                error = None
                prediction = []
                try:
                    plan, _, _ = graph_decode(planner, runtime, context)
                    plan, _ = normalize_anonymous_prediction(plan, len(row['order']), vocabulary.argument_keys)
                    prediction = EntityPermutation(row['order']).plan(plan, inverse=True)
                    for role, ids in row['targets'].items():
                        if len(ids) != 1:
                            continue
                        key = 'target_entity_name' if role == 'source' else 'target_container_name'
                        pointer_correct += {op['params'][key] for op in plan if key in op['params']} == set(ids)
                except (ValueError, RuntimeError, KeyError) as exc:
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                    error = {'type': type(exc).__name__, 'message': str(exc)}
                    errors += 1
                pointer_cases += sum(len(ids) == 1 for ids in row['targets'].values())
                audit.observe(example, prediction, generation_error=error)
                if (index+1) % 25 == 0:
                    status(root, 'source-validation', epoch=epoch, condition=name, cases=index+1, total=len(rows))
            report = audit.report()
        freeze(out/f'{name}.json', report)
        metrics.append({'condition': name, 'examples': len(rows), 'generation_errors': errors,
                        'progress': report['corrected']['planning_metrics']['overall']['score'],
                        'pointer_accuracy': pointer_correct/max(1, pointer_cases)})
    retries = []
    planner.free_decoding = False
    with torch.no_grad(), retry_eai_search(runtime, retries):
        eai = _evaluate_eai(planner, runtime, eai_valid, limit=None)
    freeze(out/'eai.json', {'result': eai, 'retries': retries})
    return {'epoch': epoch, 'vla': metrics, 'progress': sum(x['progress'] for x in metrics)/2,
            'pointer_accuracy': sum(x['pointer_accuracy'] for x in metrics)/2,
            'eai_successes': round(eai['goal_success']*88)}


def training(args):
    root = args.root
    config = read(root/'config.json')
    verify(config)
    if config['fresh_recipe'] != RECIPE or not config['internal_training_approved']:
        raise ValueError('explicit internal-corpus recipe authorization required')
    from test_regr.VLABenchAgentInterface.dataset_protocol import guard_training_dataset
    guard_training_dataset(config['planning_dir'], root, allow_internal_evaluation_training=True)
    seed()
    status(root, 'preparing-source-and-auditing')
    device = 'cpu' if args.audit_only else 'cuda:0'
    runtime_args, (runtime, eai_train, eai_valid, splits, prepared) = build_runtime(config, root, device)
    if runtime_args.knowledge != 'metadata' or runtime_args.observation_mode != 'metadata-assisted':
        raise ValueError('metadata DFA source policy changed')
    views, cache, features = source_data(config, runtime, splits)
    if args.audit_only:
        audit_permutations(root, config, runtime, splits, prepared)
        status(root, 'source-preflight-passed')
        return
    if not read(root/'permutation_audit.json')['passed']:
        raise ValueError('source preflight has not passed')

    validation = views('validation', 0, False)[2], views('validation', 0, True)[2]
    canonical = views('train', 0, False)[2]
    alternate_first = views('train', 0, True)[2]
    pair_ids = set(sorted((row['episode_id'] for row in canonical),
                          key=lambda value: digest([42, 'id-consistency', value]))[:900])
    planner = fresh_planner(config, runtime, 'joint')
    planner.free_decoding = False
    heads = torch.nn.ModuleList([RoleGrounder(planner.backbone_hidden_size, features['visual_dim'])
                                for _ in range(2)]).to(planner.device)
    heads[1].load_state_dict(heads[0].state_dict())
    lora = [p for name, p in planner.named_parameters() if p.requires_grad and 'lora_' in name]
    labels = [p for p in planner.label_heads.parameters() if p.requires_grad]
    if not lora or not labels or {id(p) for p in lora + labels} != {id(p) for p in planner.parameters() if p.requires_grad}:
        raise ValueError('fresh pre-trained base weights must remain frozen')
    optimizer = torch.optim.AdamW([{'params': lora, 'lr': 5e-6}, {'params': labels, 'lr': 1e-4}], weight_decay=.01)
    head_optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-4, weight_decay=.01)
    program = JointSolverPOIProgram(runtime, planner, planner_optimizer=optimizer, device=torch.device(device),
        controller=None, dfa_loss=DFALossConfig(domains=('vlabench',), pmd_domains=('vlabench',), mass_weight=MASS_WEIGHT))
    if set(program.dual_optimizers) != {'vlabench'}:
        raise ValueError('expected VLABench-only primal-dual optimizer')
    gradients = BatchGradientManager(planner, optimizer)
    freeze(root/'settings.json', {'recipe': RECIPE, 'config_sha256': sha(root/'config.json'),
        'source_manifest_sha256': sha(config['source_manifest']), 'encoder_sha256': features['encoder_sha256'],
        'fresh_qwen': config['model'], 'seed': 42, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
        'pmd_weight': 1.0, 'dual_lr': .05, 'dual_warmup_rounds': 10, 'dual_every': 10,
        'allowed_mass_weight': MASS_WEIGHT, 'preference_weight': PREFERENCE_WEIGHT,
        'consistency_weight': CONSISTENCY_WEIGHT, 'consistency_ids': sorted(pair_ids),
        'controllers': 'fixed and unused in training', 'prompt': 'strict image/instruction + public API',
        'metadata': 'DFA bindings only', 'target_training': False})
    save_snapshot(root/'joint_initialization.pt', planner, heads, optimizer, head_optimizer, program, root, 0)
    scorecards = []
    dual_before = tensor_digest(program.pmd_programs['vlabench'].cmodel)
    with common_prompt(planner):
        for epoch in range(EPOCHS):
            rows = canonical if epoch == 0 else views('train', epoch, True)[2]
            planner.train()
            planner.free_decoding = False
            heads.train()
            rng = random.Random(42+epoch)
            order = list(range(len(rows)))
            rng.shuffle(order)
            paired = [eai_train[i % len(eai_train)] for i in range(len(rows))]
            rng.shuffle(paired)
            with (root/f'training_{epoch+1:03d}.jsonl').open('x', encoding='utf-8') as trace:
                pending_role_cases = [0, 0]
                for batch, start in enumerate(range(0, len(rows), BATCH_SIZE), 1):
                    gradients.begin_step()
                    head_optimizer.zero_grad(set_to_none=True)
                    before = tensor_digest(planner, trainable=True, shared=True) if epoch == 0 and batch == 1 else None
                    indices = order[start:start+BATCH_SIZE]
                    for offset, index in enumerate(indices):
                        row = rows[index]
                        alt = (alternate_first if epoch == 0 else canonical)[index] if row['episode_id'] in pair_ids else None
                        with runtime.domain_scope('vlabench', parameter_policy='freeze_inactive'):
                            loss, aux, terms = accumulated_step(planner, runtime, program, heads, row, alt, cache)
                            if not torch.isfinite(loss) or not torch.isfinite(aux):
                                raise ValueError('nonfinite VLABench or grounding loss')
                            gradients.capture('vlabench', loss/BATCH_SIZE)
                            if aux.requires_grad:
                                (aux/BATCH_SIZE).backward()
                        with runtime.domain_scope('eai', parameter_policy='freeze_inactive'):
                            eai, _ = program._domain_loss('eai', paired[start+offset])
                            if not torch.isfinite(eai):
                                raise ValueError('nonfinite EAI loss')
                            gradients.capture('eai', eai/BATCH_SIZE)
                        pending_role_cases[0] += terms['role_cases']
                        pending_role_cases[1] += terms['preference_cases']
                        observation = neural_context(row, runtime.vlabench_vocabulary)
                        trace.write(json.dumps({'episode_id': row['episode_id'], 'batch': batch,
                            'new_to_old': row['order'], 'vla_loss': float(loss.detach()),
                            'eai_ce': float(eai.detach()),
                            'input_sha256': digest(public_record(planner, 'vlabench', observation)), **terms})+'\n')
                    trace.flush()
                    if epoch == 0 and batch <= 20:
                        if any(p.grad is not None for p in planner.parameters()):
                            raise ValueError('a detached grounding loss changed planner gradients')
                    if before is not None:
                        shared = {id(p) for p in lora}
                        for domain in ('vlabench', 'eai'):
                            if not any(id(p) in shared and g is not None and bool(g.any()) for p, g in
                                       zip(gradients.parameters, gradients._recorded_gradients[domain])):
                                raise ValueError(f'{domain} produced no shared LoRA gradient')
                    gradients.step(optimizer, method='pcgrad', clip_grad_norm=1.)
                    for head in heads:
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    head_optimizer.step()
                    program._dual_step()
                    if before is not None:
                        after = tensor_digest(planner, trainable=True, shared=True)
                        if before == after:
                            raise ValueError('first optimizer step did not update shared LoRA')
                        freeze(root/'first_update.json', {'pcgrad_domains': ['eai','vlabench'],
                            'epoch': 1, 'batch': 1, 'shared_before': before, 'shared_after': after})
                    if epoch == 0 and batch == 20:
                        after = tensor_digest(program.pmd_programs['vlabench'].cmodel)
                        if after == dual_before:
                            raise ValueError('first scheduled VLABench dual step did not update the multipliers')
                        if not all(pending_role_cases):
                            raise ValueError('grounding or latent ranking had no supervised cases in 20 batches')
                        freeze(root/'first_dual_update.json', {'batch': batch, 'domain': 'vlabench',
                            'before': dual_before, 'after': after, 'grounding_cases': pending_role_cases[0],
                            'preference_cases': pending_role_cases[1]})
                    if batch == 1 or batch % 10 == 0:
                        status(root, 'training', epoch=epoch+1, epochs=EPOCHS, batch=batch, batches_per_epoch=450)
                        print(f'[fresh-metadata] epoch {epoch+1}/{EPOCHS} batch {batch}/450', flush=True)
            snapshot = root/f'joint_epoch_{epoch+1:03d}.pt'
            save_snapshot(snapshot, planner, heads, optimizer, head_optimizer, program, root, epoch+1)
            result = validate(planner, runtime, *validation, splits['validation'], eai_valid, root, epoch+1)
            result['checkpoint'] = str(snapshot)
            freeze(root/f'epoch_{epoch+1:03d}.json', result)
            scorecards.append(result)
            eligible = [row for row in scorecards if row['eai_successes'] >= 47]
            best = max(eligible or scorecards, key=lambda x: (x['progress'], x['pointer_accuracy'], -x['epoch']))
            save(root/'selection.json', {'qualified': bool(eligible), 'checkpoint': best, 'source_validation_only': True})
    status(root, 'training-complete', epochs=EPOCHS, held_out_target_evaluation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    try:
        training(args)
    except Exception as exc:
        status(args.root, 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
