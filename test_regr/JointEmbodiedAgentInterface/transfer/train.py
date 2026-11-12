"""Matched source continuations: four epochs, fixed controller, fresh optimizers."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from .protocol import freeze, read, sha, digest
from .source_transform import transform
from .batch_gradients import BatchGradientManager


def tensor_digest(module, *, trainable=False, shared=False):
    import hashlib
    h = hashlib.sha256()
    values = dict(module.named_parameters()) if trainable else module.state_dict()
    shared_ids = {id(p) for p in module.shared_parameters()} if shared else None
    for name, value in sorted(values.items()):
        if trainable and not value.requires_grad: continue
        if shared_ids is not None and id(value) not in shared_ids: continue
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def encode(runtime, examples):
    from ..main import _runtime_adapter
    from test_regr.VLABenchAgentInterface.training import prepare_planner_program_examples
    return prepare_planner_program_examples(examples, _runtime_adapter(runtime, 'vlabench'))


def check_equivalence(runtime, originals, changed, orders):
    """Compare whole allowed sets, including forbidden counterexamples, along targets."""
    from ..utils.planner_context import planner_context
    from .permutation import EntityPermutation
    vocab = runtime.vlabench_vocabulary
    for index, (before, after, order) in enumerate(zip(originals, changed, orders)):
        permutation = EntityPermutation(order)
        def inverse(label):
            token = vocab.token_for_label(int(label))
            if token.startswith('obj:') and int(token[4:]) < len(order):
                return vocab.label_for_token('obj:'+str(permutation.decode_pointer(int(token[4:]))))
            return int(label)
        a = runtime.dfa_for('vlabench', planner_context('vlabench', before))
        b = runtime.dfa_for('vlabench', planner_context('vlabench', after))
        sa, sb = a.start_state, b.start_state
        for la, lb in zip(before['target_plan_labels'].tolist(), after['target_plan_labels'].tolist()):
            if {int(t) for t in a.allowed_tokens(sa)} != {inverse(t) for t in b.allowed_tokens(sb)}:
                raise ValueError('permutation changed constraint semantics')
            if inverse(lb) != la: raise ValueError('permutation did not preserve target pointers')
            sa, sb = a.step(sa, la), b.step(sb, lb)
            if sa is None or sb is None: raise ValueError('source target conflicts with its constraint policy')
            if la == vocab.eos_label: break
        if not a.is_accepting(sa) or not b.is_accepting(sb): raise ValueError('source target is incomplete')
        if (index+1)%100 == 0: print(f'[transfer-train] equivalence {index+1}/{len(originals)}', flush=True)


def main():
    from .. import main as joint
    from ..checkpoint import load_joint_checkpoint, save_joint_checkpoint
    from ..models import JointQwenVLPlanner
    from ..program import JointSolverPOIProgram
    from test_regr.VLABenchAgentInterface.knowledge import apply_knowledge_argument
    from test_regr.VLABenchAgentInterface.research.config import apply_research_arguments
    from test_regr.VLABenchAgentInterface.prompt_protocol import _ANONYMOUS_OBJECTS
    from test_regr.VLABenchAgentInterface.dataset_protocol import guard_training_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--arm', choices=['control','augmented'], required=True)
    parser.add_argument('--resume', type=Path)
    options = parser.parse_args(); config = read(options.config); output = options.output
    if config['epochs'] != 4 or config['batch_size'] != 8 or config['seed'] != 42:
        raise ValueError('approved continuation settings differ')
    proof = read(options.manifest)
    if not proof['complete'] or proof['train'] != 3600 or proof['validation'] != 450 or proof['test_used']:
        raise ValueError('complete source observation integrity gate required')
    if proof['split_sha256'] != sha(config['split_manifest']): raise ValueError('source split changed')
    rows = {r['episode_id']: r for r in proof['episodes']}
    if len(rows) != 4050: raise ValueError('source observation identities overlap')
    original_args = read(config['original_configuration'])['arguments']
    args = joint.build_parser().parse_args(original_args)
    args.output = str(output); args.save_initialization = False; args.initialize_from = config['checkpoint']
    args.resume = None; args.stage1_eai_warmup_steps = 0; args.device = 'cuda:0'
    output.mkdir(parents=True, exist_ok=True)
    settings = {'checkpoint_sha256': sha(config['checkpoint']), 'manifest_sha256': sha(options.manifest),
        'config_sha256': sha(options.config), 'arm': options.arm, 'epochs': 4, 'seed': 42, 'batch_size': 8,
        'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'controller': 'frozen', 'initialization': 'same independent full-metadata epoch4 checkpoint',
        'optimizer': 'AdamW; accumulated batch-8 domain means; existing two-domain PCGrad',
        'internal_evaluation_corpus_exposure': True, 'common_observation_format': True,
        'selection': 'minimum full canonical source validation objective, no external test selection'}
    freeze(output/'settings.json', settings)
    freeze(output/'split_identities.json', read(config['split_manifest']))
    guard_training_dataset(args.vlabench_planning_dir, output, resume=config['checkpoint'],
                           allow_internal_evaluation_training=True)
    apply_research_arguments(args); apply_knowledge_argument(args)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    runtime, eai_train, eai_valid, splits, original_items = joint._prepare(args, torch.device('cuda:0'))
    expected_splits = read(config['split_manifest'])['splits']
    if {key:[e.episode_id for e in value] for key,value in splits.items()} != expected_splits:
        raise ValueError('source train/validation/test identities differ')
    for name in ('training_reference_policy.json','training_reference_hashes.json'):
        if sha(output/name) != sha(Path(config['checkpoint']).parent/name):
            raise ValueError('frozen scene-verified reference policy differs')
    keys = {key for values in runtime.world.vlabench.skill_arguments.values() for key in values}
    keys.update(key for values in runtime.world.vlabench.optional_skill_arguments.values() for key in values)
    valid_examples = [transform(e, rows[e.episode_id], epoch=0, augmented=False, pointer_keys=keys) for e in splits['validation']]
    valid_items = encode(runtime, valid_examples)
    epochs = []
    for epoch in range(4):
        changed = [transform(e, rows[e.episode_id], epoch=epoch, augmented=options.arm=='augmented', pointer_keys=keys)
                   for e in splits['train']]
        items = encode(runtime, changed)
        orders = [rows[e.episode_id]['augmentations'][epoch]['new_to_old'] if options.arm=='augmented'
                  else list(range(len(e.entities))) for e in splits['train']]
        check_equivalence(runtime, original_items['train'], items, orders)
        epochs.append(items)
    freeze(output/'constraint_equivalence.json', {'passed': True, 'epochs': 4, 'episodes_per_epoch': 3600,
        'whole_allowed_sets_compared': True, 'reference_labels_used_only_for_audit_paths': True})
    print('[transfer-train] loading shared initialization', flush=True)
    planner = JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary, model_id=args.planner_model,
        load_in_4bit=True, decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller = joint._controller(args, torch.device('cuda:0'))
    initialization = load_joint_checkpoint(config['checkpoint'], runtime=runtime, planner=planner, controller=controller, map_location='cuda:0')
    for p in controller.parameters(): p.requires_grad_(False)
    controller.eval(); controller_hash = tensor_digest(controller)
    _ANONYMOUS_OBJECTS.set(True)
    from ..utils.planner_context import planner_context
    from ..eval_generalization import input_record
    prompt_groups = [('validation', valid_items), *[(f'train_epoch_{epoch:03d}', items) for epoch,items in enumerate(epochs)]]
    for group, items in prompt_groups:
        path = output/'rendered_inputs'/f'{group}.json'
        rendered = []
        for item in items:
            bound = planner_context('vlabench',item)
            record = input_record(planner,bound)
            rendered.append({'episode_id':bound['episode_id'], **record, 'input_sha256':digest(record),
                             'canonical_entity_table':list(bound['entity_table'])})
        freeze(path,{'inputs':rendered,'planner_input_protocol':'anonymous_objects'})
    parameters = [p for p in planner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.planner_learning_rate)
    program = JointSolverPOIProgram(runtime, planner, planner_optimizer=optimizer, controller=None,
        device='cuda:0', eai_loss_weight=args.stage1_eai_loss_weight, dfa_loss=joint._dfa_loss_config(args))
    gradients = BatchGradientManager(planner, optimizer)
    start = 0; best = float('inf')
    if options.resume:
        payload = load_joint_checkpoint(options.resume, runtime=runtime, planner=planner, controller=controller,
            planner_optimizer=optimizer, map_location='cuda:0')
        if payload['metrics']['transfer_settings_sha256'] != sha(output/'settings.json'):
            raise ValueError('resumed continuation fingerprint differs')
        start = int(payload['epoch'])+1; best = payload['metrics']['best_validation_loss']
    else:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
    for epoch in range(start, 4):
        planner.train(); train = epochs[epoch]
        rng = random.Random(42+epoch); order = list(range(len(train))); rng.shuffle(order)
        eai_order = [eai_train[i%len(eai_train)] for i in range(len(train))]; rng.shuffle(eai_order)
        total = 0.; before = tensor_digest(planner, trainable=True, shared=True) if epoch==0 else None
        for batch in range(0,len(order),8):
            gradients.begin_step(); indices = order[batch:batch+8]
            for offset,index in enumerate(indices):
                for domain, item, weight in (('vlabench',train[index],1.),('eai',eai_order[batch+offset],args.stage1_eai_loss_weight)):
                    with runtime.domain_scope(domain, parameter_policy='freeze_inactive'):
                        loss,_ = program._domain_loss(domain,item)
                        if not torch.isfinite(loss): raise ValueError('nonfinite continuation objective')
                        gradients.capture(domain,loss*weight/len(indices))
                        total += float(loss.detach())*weight
            if batch==0 and epoch==0:
                ids = {id(p) for p in planner.shared_parameters() if p.requires_grad}
                resolved = gradients.resolve_gradients(method='pcgrad')
                if not any(id(p) in ids and g is not None and torch.any(g != 0)
                           for p,g in zip(gradients.parameters,resolved)):
                    raise ValueError('shared planner received no gradient')
            gradients.step(optimizer,method='pcgrad',clip_grad_norm=1.)
            if batch==0 and epoch==0:
                after = tensor_digest(planner,trainable=True,shared=True)
                if before==after: raise ValueError('first optimizer update changed no planner parameters')
                freeze(output/'first_update.json', {'shared_gradient_nonzero':True,'before':before,'after':after})
            if batch%80==0 or batch+8>=len(order):
                print(f'[transfer-train] {options.arm} epoch={epoch+1}/4 source={min(batch+8,len(order))}/3600 loss={total/max(1,batch+len(indices)):.5f}',flush=True)
        planner.eval(); validation = {}
        with torch.no_grad():
            for domain, values in (('vlabench',valid_items),('eai',eai_valid)):
                summed = 0.
                for index,item in enumerate(values):
                    with runtime.domain_scope(domain): loss,_ = program._domain_loss(domain,item)
                    summed += float(loss)
                    if (index+1)%100==0: print(f'[transfer-train] validation {domain} {index+1}/{len(values)}',flush=True)
                validation[domain] = summed/len(values)
        value = validation['vlabench']+args.stage1_eai_loss_weight*validation['eai']; improved=value<best
        best=min(best,value)
        if tensor_digest(controller)!=controller_hash: raise ValueError('fixed controller changed')
        metrics={'transfer_settings_sha256':sha(output/'settings.json'),'transfer_method':'id-augmentation',
            'arm':options.arm,'validation':validation,'validation_loss':value,'best_validation_loss':best,
            'controller_sha256':controller_hash,'initialization_sha256':settings['checkpoint_sha256'],
            'internal_experiment':True,'common_source_observation_format':True}
        metrics['training_data_protocol'] = joint.training_data_protocol(args,initialization)
        path=output/f'joint_stage1_epoch_{epoch:03d}.pt'
        if path.exists(): raise ValueError('preserve existing epoch snapshot')
        save_joint_checkpoint(path,runtime=runtime,planner=planner,controller=controller,planner_optimizer=optimizer,
            controller_optimizer=None,stage='stage1',epoch=epoch,round_robin_cursor=0,metrics=metrics)
        if improved:
            from .campaign import save
            save(output/'best.json',{'checkpoint':str(path),'epoch':epoch,'validation_loss':value,'selection':'source validation only'})
        freeze(output/f'epoch_{epoch:03d}.json',metrics)
        print(f'[transfer-train] saved epoch={epoch+1}/4 validation={value:.5f} selected={improved}',flush=True)
    freeze(output/'complete.json',{'epochs':4,'arm':options.arm,'best':read(output/'best.json'),'controller_sha256':controller_hash})


if __name__ == '__main__': main()
