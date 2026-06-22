import os

os.environ.setdefault("HF_HOME", "/egr/research-hlr2/premsrit/transformer_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/egr/research-hlr2/premsrit/transformer_cache")
import argparse
import sys
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parents[1]))

from dataset import ACTION_VOCAB, EOS_TOKEN, dummy_dataset, load_eai_dataset, split_train_dev
from modules import (
    AutoregressiveActionObjectGenerator,
    CausalLMActionObjectGenerator,
    SmallLLMPlanGenerator,
    TinyTransformerActionObjectGenerator,
)


RUN_DIR = Path(__file__).parent.resolve()
RESULTS_PATHS = {
    "solver": RUN_DIR / "results_solver.txt",
    "primal-dual": RUN_DIR / "results_pmd.txt",
}
# ``MODEL_DIR`` honours ``EAI_MODEL_DIR`` first, then falls back to the
# original Linux path, then to a per-user temp dir so the script
# can run on any machine without manual setup.
_default_model_dir = "/egr/research-hlr2/premsrit/model_EAI"
MODEL_DIR = Path(os.environ.get("EAI_MODEL_DIR", _default_model_dir))
try:
    MODEL_DIR.mkdir(exist_ok=True)
except (FileNotFoundError, OSError):
    MODEL_DIR = Path(tempfile.gettempdir()) / "model_EAI"
    MODEL_DIR.mkdir(exist_ok=True)


def build_program(
    device="cpu",
    feature_dim=None,
    hidden_dim=128,
    encoder_model_path="bert-base-uncased",
    encoder_max_length=256,
    freeze_encoder=True,
    max_steps=8,
    use_llm=False,
    llm_model_path=None,
    max_new_tokens=128,
    vocab=None,
    object_tokens=None,
    action_tokens=None,
    openable_object_tokens=None,
    action_object_constraint_tokens=None,
    program_type="solver",
    baseline_model="tiny-transformer",
    llm_backbone_path="Qwen/Qwen2.5-0.5B-Instruct",
    transformer_layers=2,
    transformer_heads=4,
    use_lora=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=None,
    llm_device_map=None,
    gradient_checkpointing=False,
):
    from domiknows import setProductionLogMode
    from domiknows.program import SolverPOIProgram
    from domiknows.program.loss import NBCrossEntropyLoss
    from domiknows.program.lossprogram import PrimalDualProgram
    from domiknows.program.metric import MacroAverageTracker
    from domiknows.program.model.pytorch import SolverModel
    from domiknows.sensor.pytorch.sensors import ModuleSensor, ReaderSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor

    from graph import create_generation_graph

    setProductionLogMode(True)
    graph, bundle = create_generation_graph(
        max_steps=max_steps,
        vocab=vocab,
        object_tokens=object_tokens,
        action_tokens=action_tokens,
        openable_object_tokens=openable_object_tokens,
        action_object_constraint_tokens=action_object_constraint_tokens,
    )
    graph.detach()
    if program_type not in {"solver", "primal-dual"}:
        raise ValueError(f"Unsupported program_type={program_type!r}")
    program_args = (graph,) if program_type == "solver" else (graph, SolverModel)
    supervised_loss = MacroAverageTracker(NBCrossEntropyLoss())

    # Match the generation examples: text owns prompt-level inputs, token owns
    # per-step features/labels, and token[generated_token] owns predictions.
    text = bundle.text
    token = bundle.token
    generated_token = bundle.generated_token

    text["instruction_text"] = ReaderSensor(keyword="text")
    text["tl_goal"] = ReaderSensor(keyword="tl_goal")
    token["position"] = ReaderSensor(keyword="token_positions")
    token[bundle.contains] = EdgeSensor(
        text["instruction_text"],
        token["position"],
        relation=bundle.contains,
        forward=lambda _text, positions: torch.ones_like(positions).unsqueeze(-1).float(),
    )
    token["target_action_label"] = ReaderSensor(keyword="target_action_labels")

    if use_llm:
        llm_generator = SmallLLMPlanGenerator(
            model_path=llm_model_path or "Qwen/Qwen2.5-0.5B-Instruct",
            device=device,
            max_new_tokens=max_new_tokens,
            max_steps=max_steps,
            vocabulary=bundle.vocabulary,
        )
        token[generated_token] = ModuleLearner(
            bundle.contains,
            text["instruction_text"],
            text["tl_goal"],
            module=llm_generator,
            device=device,
        )
        program = SolverPOIProgram(
            graph,
            poi=[text, token, generated_token, token[bundle.contains], token[generated_token]],
            inferTypes=['local/argmax'],
        )
        return program, bundle

    if baseline_model == "bert-gru":
        autoregressive_head = AutoregressiveActionObjectGenerator(
            model_path=encoder_model_path,
            label_count=bundle.vocabulary.label_count,
            eos_label=bundle.vocabulary.eos_label,
            device=device,
            max_length=encoder_max_length,
            freeze=freeze_encoder,
            hidden_dim=hidden_dim,
        ).to(device)
    elif baseline_model == "tiny-transformer":
        autoregressive_head = TinyTransformerActionObjectGenerator(
            label_count=bundle.vocabulary.label_count,
            eos_label=bundle.vocabulary.eos_label,
            device=device,
            max_length=encoder_max_length,
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
        ).to(device)
    elif baseline_model == "causal-lm":
        autoregressive_head = CausalLMActionObjectGenerator(
            model_path=llm_backbone_path,
            label_count=bundle.vocabulary.label_count,
            eos_label=bundle.vocabulary.eos_label,
            device=device,
            max_length=encoder_max_length,
            freeze=freeze_encoder,
            hidden_dim=hidden_dim,
            vocabulary=bundle.vocabulary,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=llm_device_map,
            gradient_checkpointing=gradient_checkpointing,
        )
        if not (llm_device_map and str(llm_device_map).lower() != "none"):
            autoregressive_head = autoregressive_head.to(device)
    else:
        raise ValueError(f"Unsupported baseline_model={baseline_model!r}")
    token[generated_token] = ModuleLearner(
        bundle.contains,
        text["instruction_text"],
        "target_action_label",
        module=autoregressive_head,
        device=device,
    )
    token[generated_token] = ReaderSensor(keyword="target_action_labels", label=True)

    if program_type == "solver":
        program = SolverPOIProgram(
            *program_args,
            poi=[text, token, generated_token, token[bundle.contains], token[generated_token]],
            inferTypes=['local/argmax'],
            loss=supervised_loss,
            device=device,
            metric={},
        )
    else:
         program = PrimalDualProgram(
            *program_args,
            poi=[text, token, generated_token, token[bundle.contains], token[generated_token]],
            inferTypes=['local/argmax'],
            loss=supervised_loss,
            device=device,
            metric={},
        )
    program.autoregressive_head = autoregressive_head
    return program, bundle


def load_examples(args, device):
    if args.dummy:
        return dummy_dataset(device=device, max_steps=args.max_steps)
    return load_eai_dataset(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        data_path=args.data_path,
        device=device,
        max_steps=args.max_steps,
    )


def labels_to_actions(labels, vocabulary=None):
    actions = []
    for label in labels:
        idx = int(label.item() if torch.is_tensor(label) else label)
        if vocabulary is not None and 0 <= idx < vocabulary.label_count:
            token = vocabulary.token_for_label(idx)
        else:
            token = ACTION_VOCAB[idx] if 0 <= idx < len(ACTION_VOCAB) else "other"
        if token == getattr(vocabulary, "other_token", None):
            token = "other"
        actions.append(token)
        eos_token = vocabulary.eos_token if vocabulary is not None else EOS_TOKEN
        if token == eos_token:
            break
    return actions


def prediction_to_labels(value):
    if not torch.is_tensor(value):
        return []
    value = value.detach()
    if value.dim() == 0:
        return value.reshape(1).long()
    if value.dim() == 1:
        return value.long()
    if value.dim() == 2:
        return torch.argmax(value, dim=-1)
    if value.dim() == 3:
        return torch.argmax(value[0], dim=-1)
    return []


def get_text_attribute(datanode, name):
    names = (name, f"<{name}>")
    prediction_keys = ("local/argmax", "local/softmax", "local/logits", "local")
    for attr_name in names:
        for pred_key in prediction_keys:
            try:
                value = datanode.getAttribute(attr_name, pred_key)
            except Exception:
                value = None
            if value is not None:
                return value

    for attr_name in names:
        try:
            value = datanode.getAttribute(attr_name)
        except Exception:
            value = None
        if value is not None:
            return value

    attrs = datanode.getAttributes() or {}
    for key, value in attrs.items():
        if str(key).strip("<>") == name:
            if isinstance(value, dict):
                for pred_key in prediction_keys:
                    if pred_key in value:
                        return value[pred_key]
            return value
    return None


def scalar_int(value, default=0):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return int(value.detach().reshape(-1)[0].item())
    try:
        return int(value)
    except Exception:
        return default


def prediction_value_to_label(value):
    if not torch.is_tensor(value):
        return None
    value = value.detach()
    if value.dim() == 0 or value.numel() == 1:
        return int(value.reshape(-1)[0].item())
    return int(torch.argmax(value.reshape(-1), dim=-1).item())


def generated_token_sequence(datanode, bundle):
    try:
        token_nodes = list(datanode.getChildDataNodes(conceptName=bundle.token))
    except Exception:
        token_nodes = []
    if not token_nodes:
        try:
            token_nodes = [node for node in datanode.getChildDataNodes() if node.ontologyNode.name == "token"]
        except Exception:
            token_nodes = []

    labels = []
    for node in token_nodes:
        value = get_text_attribute(node, bundle.generated_token.name)
        label = prediction_value_to_label(value)
        if label is None:
            continue
        position = scalar_int(get_text_attribute(node, "position"), default=len(labels))
        labels.append((position, label))

    return [label for _position, label in sorted(labels)]


def generation_vocab_from_examples(examples):
    if not examples:
        return ACTION_VOCAB
    return examples[0].get("generation_vocab", ACTION_VOCAB)


def action_tokens_from_examples(examples):
    if not examples:
        return ()
    vocab = generation_vocab_from_examples(examples)
    return tuple(token for token in vocab if any(token in sample.get("action_tokens", ()) for sample in examples))


def action_tokens_requiring_object_from_examples(examples):
    if not examples:
        return ()
    vocab = generation_vocab_from_examples(examples)
    return tuple(
        token
        for token in vocab
        if any(token in sample.get("action_requires_object_tokens", ()) for sample in examples)
    )


def object_tokens_from_examples(examples):
    if not examples:
        return ()
    vocab = generation_vocab_from_examples(examples)
    return tuple(token for token in vocab if any(token in sample.get("object_tokens", ()) for sample in examples))


def openable_object_tokens_from_examples(examples):
    if not examples:
        return ()
    vocab = generation_vocab_from_examples(examples)
    return tuple(
        token
        for token in vocab
        if any(token in sample.get("openable_object_tokens", ()) for sample in examples)
    )


def action_object_constraint_tokens_from_examples(examples):
    if not examples:
        return {}
    vocab = generation_vocab_from_examples(examples)
    vocab_set = set(vocab)
    action_to_objects = {}
    for sample in examples:
        for action, obj in sample.get("action_object_constraint_pairs", ()):
            if action in vocab_set and obj in vocab_set:
                action_to_objects.setdefault(action, set()).add(obj)
    return {
        action: tuple(token for token in vocab if token in objects)
        for action, objects in sorted(action_to_objects.items())
    }


def write_vocab_info_log(examples):
    log_dir = RUN_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    vocab = generation_vocab_from_examples(examples)
    action_tokens = action_tokens_from_examples(examples)
    object_tokens = object_tokens_from_examples(examples)
    action_requires_object_tokens = action_tokens_requiring_object_from_examples(examples)
    openable_object_tokens = openable_object_tokens_from_examples(examples)
    action_object_constraint_tokens = action_object_constraint_tokens_from_examples(examples)
    info_path = log_dir / "info.log"
    with info_path.open("w") as log_file:
        log_file.write("EmbodiedAgentInterface vocabulary info\n")
        log_file.write(f"example_count: {len(examples)}\n")
        log_file.write(f"vocab_size: {len(vocab)}\n")
        log_file.write(f"action_count: {len(action_tokens)}\n")
        log_file.write(f"object_count: {len(object_tokens)}\n")
        log_file.write(
            f"action_requires_object_count: {len(action_requires_object_tokens)}\n"
        )
        log_file.write(f"openable_object_count: {len(openable_object_tokens)}\n")
        log_file.write(
            f"action_object_constraint_count: {len(action_object_constraint_tokens)}\n"
        )
        log_file.write("\n[vocabulary]\n")
        for index, token in enumerate(vocab):
            log_file.write(f"{index}: {token}\n")
        log_file.write("\n[action_tokens]\n")
        for token in action_tokens:
            log_file.write(f"{token}\n")
        log_file.write("\n[action_requires_object_tokens]\n")
        for token in action_requires_object_tokens:
            log_file.write(f"{token}\n")
        log_file.write("\n[object_tokens]\n")
        for token in object_tokens:
            log_file.write(f"{token}\n")
        log_file.write("\n[openable_object_tokens]\n")
        for token in openable_object_tokens:
            log_file.write(f"{token}\n")
        log_file.write("\n[action_object_constraints]\n")
        for action, objects in action_object_constraint_tokens.items():
            log_file.write(f"{action}: {', '.join(objects)}\n")
    print(f"Vocabulary info log: {info_path}")


def dfa_constrained_sequence(program, bundle, sample, max_steps):
    from domiknows.generation import constrained_label_greedy_decode, constraints_to_dfa_from_graph

    dfa = constraints_to_dfa_from_graph(program.graph, bundle)
    result = constrained_label_greedy_decode(
        program.autoregressive_head,
        [bundle.vocabulary.eos_label],
        bundle.vocabulary,
        dfa,
        max_new_tokens=max_steps,
        next_label_kwargs={"text": sample.get("text", "")},
    )
    return result.labels


def greedy_sequence(program, bundle, sample, max_steps):
    labels = []
    prefix = [bundle.vocabulary.eos_label]
    for _step in range(max_steps):
        logits = program.autoregressive_head.next_label_logits(
            prefix,
            text=sample.get("text", ""),
        )
        label = int(torch.argmax(logits.detach(), dim=-1).item())
        labels.append(label)
        prefix.append(label)
        if label == bundle.vocabulary.eos_label:
            break
    return labels


def sequence_score(program, bundle, examples, max_steps, device="cpu", use_dfa=False, limit=None, show=False):
    from domiknows.generation import constraints_to_dfa_from_graph

    eval_examples = examples if limit is None else examples[:limit]
    if not eval_examples:
        return {"examples": 0, "exact_sequence": 0.0, "token_accuracy": 0.0, "dfa_valid": 0.0}

    dfa = constraints_to_dfa_from_graph(program.graph, bundle)
    exact = 0
    token_correct = 0
    token_total = 0
    dfa_valid = 0
    for idx, sample in enumerate(eval_examples):
        program.populate_one(sample, device=device)
        labels = dfa_constrained_sequence(program, bundle, sample, max_steps) if use_dfa else greedy_sequence(program, bundle, sample, max_steps)
        gold = [int(x.item() if torch.is_tensor(x) else x) for x in sample["target_action_labels"][:max_steps]]
        pred = [int(x.item() if torch.is_tensor(x) else x) for x in labels[:max_steps]]
        pred_padded = pred + [bundle.vocabulary.eos_label] * max(0, len(gold) - len(pred))
        pred_padded = pred_padded[:len(gold)]
        exact += int(pred_padded == gold)
        token_correct += sum(int(p == g) for p, g in zip(pred_padded, gold))
        token_total += len(gold)
        dfa_valid += int(dfa.accepts(pred_padded))
        if show:
            print()
            print(f"## Example {idx}: {sample.get('task_id', 'task')}")
            print(f"Instruction: {sample.get('natural_language_description') or sample.get('text')}")
            print(f"Gold sequence:      {labels_to_actions(gold, bundle.vocabulary)}")
            print(f"Predicted sequence: {labels_to_actions(pred_padded, bundle.vocabulary)}")
    return {
        "examples": len(eval_examples),
        "exact_sequence": exact / len(eval_examples),
        "token_accuracy": token_correct / token_total if token_total else 0.0,
        "dfa_valid": dfa_valid / len(eval_examples),
    }


def results_path_for_program(program_type):
    return RESULTS_PATHS.get(program_type, RUN_DIR / "results.txt")


def print_score(title, score, program_type=None):
    line = (
        f"{title}: examples={score['examples']} "
        f"exact_sequence={score['exact_sequence']:.3f} "
        f"token_accuracy={score['token_accuracy']:.3f} "
        f"dfa_valid={score['dfa_valid']:.3f}"
    )
    print(line)
    with results_path_for_program(program_type).open("a") as results_file:
        results_file.write(line + "\n")


def build_trainable_program(args, examples, device):
    return build_program(
        device=device,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        encoder_model_path=args.encoder_model_path,
        encoder_max_length=args.encoder_max_length,
        freeze_encoder=not args.finetune_encoder,
        max_steps=args.max_steps,
        vocab=generation_vocab_from_examples(examples),
        object_tokens=object_tokens_from_examples(examples),
        action_tokens=action_tokens_requiring_object_from_examples(examples),
        openable_object_tokens=openable_object_tokens_from_examples(examples),
        action_object_constraint_tokens=action_object_constraint_tokens_from_examples(examples),
        program_type=args.program,
        baseline_model=args.baseline_model,
        llm_backbone_path=args.llm_backbone_path,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        llm_device_map=args.llm_device_map,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def _train_kwargs(args, train, device, epochs):
    train_kwargs = {
        "device": device,
        "train_epoch_num": epochs,
        "Optim": lambda params: torch.optim.Adam(params, lr=args.lr),
        "test_every_epoch": False,
    }
    if args.program == "primal-dual":
        train_kwargs.update(
            c_warmup_iters=args.constraint_warmup_iters,
            batch_size=args.batch_size,
            dataset_size=len(train),
        )
    return train_kwargs


def _epoch_accuracy_title(args, split_name, epoch):
    dfa_text = "with DFA" if args.use_dfa else "without DFA"
    return f"epoch {epoch} {args.dataset} {args.program} {split_name} {dfa_text}"


def report_epoch_accuracy(args, program, bundle, train, dev, device, epoch):
    limit = args.epoch_eval_limit if args.epoch_eval_limit > 0 else None
    for split_name, split_examples in (("train", train), ("dev", dev)):
        if not split_examples:
            continue
        score = sequence_score(
            program,
            bundle,
            split_examples,
            args.max_steps,
            device=device,
            use_dfa=args.use_dfa,
            limit=limit,
            show=False,
        )
        print_score(_epoch_accuracy_title(args, split_name, epoch), score, args.program)


def train_program(args, train, dev, examples, device):
    program, bundle = build_trainable_program(args, examples, device)
    print(f"Starting training and will save at {args.model}")
    if args.eval_every_epoch:
        for epoch in range(1, args.epochs + 1):
            print(f"Training epoch {epoch}/{args.epochs}")
            program.train(train, valid_set=dev, test_set=None, **_train_kwargs(args, train, device, 1))
            report_epoch_accuracy(args, program, bundle, train, dev, device, epoch)
    else:
        program.train(train, valid_set=dev, test_set=None, **_train_kwargs(args, train, device, args.epochs))
        # report_epoch_accuracy(args, program, bundle, train, dev, device, epoch)

    if args.model:
        model_path = Path(args.model)
        model_path.parent.mkdir(exist_ok=True, parents=True)
        program.save(model_path)
        print(f"Saved model: {model_path}")
    return program, bundle


def load_trained_program(args, examples, device):
    program, bundle = build_trainable_program(args, examples, device)
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            program.load(model_path, map_location=device)
            print(f"Loaded model: {model_path}")
        else:
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return program, bundle


def run_train_or_evaluate(args, examples, device):
    train, dev = split_train_dev(examples, args.dev_fraction)
    eval_examples = dev or train
    program = bundle = None
    if args.train:
        program, bundle = train_program(args, train, dev, examples, device)
    if args.evaluate or args.eval_only:
        if program is None or bundle is None:
            program, bundle = load_trained_program(args, examples, device)
        title = f"{args.dataset} {args.program} {'with DFA' if args.use_dfa else 'without DFA'}"
        score = sequence_score(
            program,
            bundle,
            eval_examples,
            args.max_steps,
            device=device,
            use_dfa=args.use_dfa,
            limit=args.num_generations if args.num_generations > 0 else None,
            show=args.show_predictions,
        )
        print_score(title, score, args.program)
    return 0


def generate_baseline_sequences(args, examples, device):
    program, bundle = build_program(
        device=device,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        encoder_model_path=args.encoder_model_path,
        encoder_max_length=args.encoder_max_length,
        freeze_encoder=not args.finetune_encoder,
        max_steps=args.max_steps,
        vocab=generation_vocab_from_examples(examples),
        object_tokens=object_tokens_from_examples(examples),
        action_tokens=action_tokens_requiring_object_from_examples(examples),
        openable_object_tokens=openable_object_tokens_from_examples(examples),
        action_object_constraint_tokens=action_object_constraint_tokens_from_examples(examples),
        program_type=args.program,
        baseline_model=args.baseline_model,
        llm_backbone_path=args.llm_backbone_path,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        llm_device_map=args.llm_device_map,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    correct = 0
    total = 0
    for idx, sample in enumerate(examples[:args.num_generations]):
        program.populate_one(sample, device=device)
        labels = dfa_constrained_sequence(program, bundle, sample, args.max_steps) if args.use_dfa else greedy_sequence(program, bundle, sample, args.max_steps)
        pred = labels_to_actions(labels, bundle.vocabulary)
        gold = sample["target_action_tokens"]
        correct += int(bool(pred) and pred == gold[: len(pred)])
        total += 1
        print()
        print(f"## Example {idx}: {sample.get('task_id', 'task')}")
        print(f"Instruction: {sample.get('natural_language_description') or sample.get('text')}")
        print(f"Gold sequence:      {gold}")
        print(f"Baseline sequence:  {pred}")
    if total:
        print(f"\nSequence exact-prefix accuracy on shown examples: {correct / total:.3f}")


def generate_llm_sequences(args, examples, device):
    program, bundle = build_program(
        device=device,
        max_steps=args.max_steps,
        use_llm=True,
        llm_model_path=args.llm_model_path,
        max_new_tokens=args.max_new_tokens,
        vocab=generation_vocab_from_examples(examples),
        object_tokens=object_tokens_from_examples(examples),
        action_tokens=action_tokens_requiring_object_from_examples(examples),
        openable_object_tokens=openable_object_tokens_from_examples(examples),
        action_object_constraint_tokens=action_object_constraint_tokens_from_examples(examples),
    )
    for idx, sample in enumerate(examples[:args.num_generations]):
        datanode = program.populate_one(sample, device=device)
        sequence = labels_to_actions(generated_token_sequence(datanode, bundle), bundle.vocabulary)
        print()
        print(f"## Example {idx}: {sample.get('task_id', 'task')}")
        print(f"Instruction: {sample.get('natural_language_description') or sample.get('text')}")
        print(f"Gold sequence: {sample['target_action_tokens']}")
        print("LLM sequence:")
        print(sequence if sequence else "<no generated_token prediction found>")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DomiKnowS EAI action-sequence generation baseline."
    )
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--split", default=None, help="Optional HF split override.")
    parser.add_argument("--data-path", default=None, help="Local parquet/csv/json/jsonl copy of EAI.")
    parser.add_argument("--dummy", action="store_true", help="Use a tiny local smoke-test dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit loaded rows.")
    parser.add_argument("--max-steps", type=int, default=60, help="Padded action-token sequence length including EOS.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--program", choices=["solver", "primal-dual"], default="solver", help="DomiKnowS training program to use for the autoregressive baseline.")
    parser.add_argument("--baseline-model", choices=["tiny-transformer", "bert-gru", "causal-lm"], default="tiny-transformer", help="Autoregressive baseline architecture. tiny-transformer is small and fully trainable; causal-lm uses a frozen small LLM backbone.")
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-0.5B-Instruct", help="Causal LM backbone for --baseline-model causal-lm.")
    parser.add_argument("--use-lora", action="store_true", help="Train LoRA adapters on the causal LM backbone.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank for --baseline-model causal-lm --use-lora.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha for --baseline-model causal-lm --use-lora.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout for --baseline-model causal-lm --use-lora.")
    parser.add_argument("--lora-target-modules", nargs="+", default=None, help="Optional LoRA target module names. Defaults to Qwen attention/MLP projections.")
    parser.add_argument("--llm-device-map", default=None, help="Optional Hugging Face device_map for causal LM loading, e.g. auto for multi-GPU sharding.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing for causal LM LoRA training to reduce activation memory.")
    parser.add_argument("--transformer-layers", type=int, default=2, help="Layers for --baseline-model tiny-transformer.")
    parser.add_argument("--transformer-heads", type=int, default=4, help="Attention heads for --baseline-model tiny-transformer.")
    parser.add_argument("--constraint-warmup-iters", type=int, default=5, help="Model-only warmup iterations before primal-dual constraint updates.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--feature-dim", type=int, default=None, help="Override encoder hidden size for the sequence head input.")
    parser.add_argument("--encoder-model-path", default="bert-base-uncased", help="Hugging Face/local BERT-style encoder for task text.")
    parser.add_argument("--encoder-max-length", type=int, default=256, help="Max tokens for the BERT text encoder.")
    parser.add_argument("--finetune-encoder", action="store_true", help="Allow gradients through the BERT encoder. Default freezes it.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train", action="store_true", help="Train the selected --program and save --model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the selected --program, loading --model unless training in the same run.")
    parser.add_argument("--eval-every-epoch", action="store_true", help="Report train/dev sequence accuracy after each training epoch.")
    parser.add_argument("--epoch-eval-limit", type=int, default=0, help="Limit examples used for --eval-every-epoch scoring; 0 evaluates the full split.")
    parser.add_argument("--use-dfa", action="store_true", help="Use DFA-constrained decoding during evaluation/generation.")
    parser.add_argument("--show-predictions", action="store_true", help="Print decoded examples during evaluation.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", default=str(MODEL_DIR / "eai_action_sequence_baseline.pth"))
    parser.add_argument("--use-llm", action="store_true", help="Attach a small text LLM as a DomiKnowS ModuleSensor for action-sequence generation.")
    parser.add_argument("--llm-model-path", default="Qwen/Qwen2.5-0.5B-Instruct", help="Small Hugging Face causal LM used by the DomiKnowS generated action sequence sensor.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated plan tokens for --use-llm.")
    parser.add_argument("--num-generations", type=int, default=300, help="Number of examples to decode/show.")
    parser.add_argument("--single-run", action="store_true", help="Train only --program on --dataset instead of the BEHAVIOR/VirtualHome normal+PMD suite.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    examples = load_examples(args, device)
    write_vocab_info_log(examples)

    if args.use_llm:
        generate_llm_sequences(args, examples, device)
        return 0

    if args.train or args.evaluate or args.eval_only:
        return run_train_or_evaluate(args, examples, device)

    train, dev = split_train_dev(examples, args.dev_fraction)
    shown = dev or train
    generate_baseline_sequences(args, shown, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
