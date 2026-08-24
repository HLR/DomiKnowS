import argparse
import gc
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_default_hf_home = "/egr/research-hlr2/premsrit/transformer_cache"
if os.path.exists(_default_hf_home):
    os.environ.setdefault("HF_HOME", _default_hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", _default_hf_home)

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parents[1]))

from dataset import (
    ACTION_VOCAB,
    EOS_TOKEN,
    VLABENCH_AUX_DATA_DIR,
    dummy_dataset,
    ensure_vlabench_auxiliary_planning_data,
    load_eai_dataset,
    load_vlabench_auxiliary_planning_examples,
    split_train_dev,
    split_vlabench_auxiliary_examples,
)
from modules import (
    AutoregressiveActionObjectGenerator,
    CAUSAL_PROMPT_FORMAT,
    CausalLMActionObjectGenerator,
    EOSMaskedCrossEntropyLoss,
    SmallLLMPlanGenerator,
    TinyTransformerActionObjectGenerator,
    _prepare_transformers_imports,
)
from rl_sequence_program import AutoregressiveSequenceReinforcementProgram
from reward import (
    TokenVocabulary,
    abstract_state_from_tokens,
    eai_action_decoder,
    eai_goal_reward_function,
    evaluate_goal_satisfaction,
    make_eai_reward_function,
)
from world_graph import (
    EAIWorldGraphBundle,
    build_eai_world_graph,
)


@dataclass(frozen=True)
class EAIProgramBundle:
    """Generation and world schemas used by one EAI program lifecycle."""
    generation: object
    world: EAIWorldGraphBundle
    reward_mode: str = "dense"
    constraint_weight: float = 0.25
    constraint_aggregate: str = "mean"
    policy_dfa: object | None = None
    generation_graph: object | None = None
    generation_constraints: str = "always"
    model_metadata: dict | None = None
    _policy_cache: dict = field(
        default_factory=dict, compare=False, repr=False
    )

    def __getattr__(self, name):
        return getattr(self.generation, name)

    def policy_for(self, context):
        """Bind graph-declared contextual policies to one task example."""
        if self.policy_dfa is None or self.generation_graph is None:
            return self.policy_dfa
        from domiknows.generation import bind_contextual_dfa

        cache_key = id(context)
        cached = self._policy_cache.get(cache_key)
        if cached is not None:
            return cached
        bound = bind_contextual_dfa(
            self.policy_dfa, self.generation_graph, context or {}
        )
        self._policy_cache[cache_key] = bound
        return bound

RUN_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = RUN_DIR / "results"
RESULTS_PATHS = {
    "solver": RESULTS_DIR / "results_solver.txt",
    "primal-dual": RESULTS_DIR / "results_pmd.txt",
    "reinforcement": RESULTS_DIR / "results_reinforcement.txt",
}

_default_model_dir = SCRIPT_DIR / "models"
MODEL_DIR = Path(os.environ.get("EAI_MODEL_DIR", str(_default_model_dir)))
try:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    MODEL_DIR = Path(tempfile.gettempdir()) / "model_EAI"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
    action_sequence_tokens=None,
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
    shared_llm_model=None,
    shared_llm_tokenizer=None,
    enforce_action_object=True,
    enforce_action_object_constraints=True,
    rl_estimator="reinforce",
    rl_num_samples=8,
    rl_rescore_microbatch=1,
    rl_reward_mode="dense",
    rl_supervised_weight=0.5,
    rl_constraint_weight=0.25,
    rl_constraint_aggregate="mean",
    world_constraint_builders=None,
    shared_autoregressive_head=None,
    generation_constraints="always",
    causal_label_head="pretrained-adapter",
    label_adapter_rank=64,
):
    if not 0.0 <= float(rl_constraint_weight) <= 1.0:
        raise ValueError("rl_constraint_weight must be between 0 and 1")
    if rl_constraint_aggregate not in {"mean", "min", "prod"}:
        raise ValueError(f"Unsupported rl_constraint_aggregate={rl_constraint_aggregate!r}")
    if generation_constraints not in {"always", "eval", "off"}:
        raise ValueError(f"Unsupported generation_constraints={generation_constraints!r}")

    from domiknows import setProductionLogMode
    from domiknows.program import SolverPOIProgram
    from domiknows.program.lossprogram import PrimalDualProgram
    from domiknows.program.metric import MacroAverageTracker
    from domiknows.program.model.pytorch import SolverModel
    from domiknows.sensor.pytorch.sensors import ModuleSensor, ReaderSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor

    from graph import create_generation_graph

    setProductionLogMode(True)
    graph, generation_bundle = create_generation_graph(
        max_steps=max_steps,
        vocab=vocab,
        object_tokens=object_tokens,
        action_tokens=action_tokens,
        action_sequence_tokens=action_sequence_tokens,
        openable_object_tokens=openable_object_tokens,
        action_object_constraint_tokens=action_object_constraint_tokens,
        enforce_action_object=enforce_action_object,
        enforce_action_object_constraints=enforce_action_object_constraints,
    )
    from domiknows.generation import constraints_to_dfa_from_graph
    policy_dfa = constraints_to_dfa_from_graph(
        graph, generation_bundle, on_unsupported="raise", minimize=False
    )
    graph.detach()
    include_default_constraints = world_constraint_builders is None
    if world_constraint_builders is None:
        world_constraint_builders = ()
    world_bundle = build_eai_world_graph(
        graph_name="eai_world",
        constraint_builders=world_constraint_builders,
        include_default_constraints=include_default_constraints,
    )
    bundle = EAIProgramBundle(
        generation=generation_bundle,
        world=world_bundle,
        reward_mode=rl_reward_mode,
        constraint_weight=float(rl_constraint_weight),
        constraint_aggregate=rl_constraint_aggregate,
        policy_dfa=policy_dfa,
        generation_graph=graph,
        generation_constraints=generation_constraints,
        model_metadata={
            "backbone": llm_backbone_path if baseline_model == "causal-lm" else baseline_model,
            "vocabulary": list(generation_bundle.vocabulary.labels),
            "label_head": causal_label_head if baseline_model == "causal-lm" else "native",
            "label_adapter_rank": int(label_adapter_rank) if baseline_model == "causal-lm" else 0,
            "prompt_format": CAUSAL_PROMPT_FORMAT if baseline_model == "causal-lm" else "native",
        },
    )
    if program_type not in {"solver", "primal-dual", "reinforcement"}:
        raise ValueError(f"Unsupported program_type={program_type!r}")
    program_args = (graph,) if program_type in {"solver", "reinforcement"} else (graph, SolverModel)
    supervised_loss = MacroAverageTracker(
        EOSMaskedCrossEntropyLoss(bundle.vocabulary.eos_label)
    )

    # Match the generation examples: text owns prompt-level inputs, token owns
    # per-step features/labels, and token[generated_token] owns predictions.
    text = bundle.text
    token = bundle.token
    generated_token = bundle.generated_token

    text["instruction_text"] = ReaderSensor(keyword="text")
    text["causal_prompt_text"] = ReaderSensor(keyword="causal_prompt_text")
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
            policy_dfa=policy_dfa if generation_constraints == "always" else None,
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

    if shared_autoregressive_head is not None:
        autoregressive_head = shared_autoregressive_head
        if getattr(autoregressive_head, "label_count", bundle.vocabulary.label_count) != bundle.vocabulary.label_count:
            raise ValueError("The shared autoregressive head uses a different vocabulary size")
        if getattr(autoregressive_head, "eos_label", bundle.vocabulary.eos_label) != bundle.vocabulary.eos_label:
            raise ValueError("The shared autoregressive head uses a different EOS label")
    elif baseline_model == "bert-gru":
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
            shared_model=shared_llm_model,
            shared_tokenizer=shared_llm_tokenizer,
            label_head=causal_label_head,
            label_adapter_rank=label_adapter_rank,
        )
        if not (llm_device_map and str(llm_device_map).lower() != "none"):
            autoregressive_head = autoregressive_head.to(device)
    else:
        raise ValueError(f"Unsupported baseline_model={baseline_model!r}")
    prompt_sensor = (
        text["causal_prompt_text"]
        if baseline_model == "causal-lm"
        else text["instruction_text"]
    )
    token[generated_token] = ModuleLearner(
        bundle.contains,
        prompt_sensor,
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
    elif program_type == "primal-dual":
        program = PrimalDualProgram(
            *program_args,
            poi=[text, token, generated_token, token[bundle.contains], token[generated_token]],
            inferTypes=['local/argmax'],
            loss=supervised_loss,
            device=device,
            metric={},
        )
    elif program_type == "reinforcement":
        program = AutoregressiveSequenceReinforcementProgram(
            graph,
            targets=[generated_token],
            autoregressive_head=autoregressive_head,
            eos_label=bundle.vocabulary.eos_label,
            max_steps=max_steps,
            supervised_weight=rl_supervised_weight,
            rescore_microbatch_size=rl_rescore_microbatch,
            reward_key="reward_function",
            decoder=eai_action_decoder,
            num_samples=rl_num_samples,
            estimator=rl_estimator,
            policy_dfa=policy_dfa if generation_constraints == "always" else None,
            policy_dfa_factory=(
                bundle.policy_for if generation_constraints == "always" else None
            ),
            poi=[text, token, generated_token, token[bundle.contains], token[generated_token]],
            device=device,
        )
    program.autoregressive_head = autoregressive_head
    return program, bundle


def load_examples(args, device):
    if args.dummy:
        examples = dummy_dataset(device=device, max_steps=args.max_steps)
    else:
        examples = load_eai_dataset(
            dataset_name=args.dataset,
            split=args.split,
            limit=args.limit,
            data_path=args.data_path,
            device=device,
            max_steps=args.max_steps,
        )
    if examples:
        vocab = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
        mode = getattr(args, "rl_reward_mode", "dense")
        for ex in examples:
            ex["reward_function"] = make_eai_reward_function(ex, vocabulary=vocab, mode=mode)
    return examples


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
    from domiknows.generation import constrained_label_greedy_decode

    dfa = bundle.policy_for(sample)
    result = constrained_label_greedy_decode(
        program.autoregressive_head,
        [bundle.vocabulary.eos_label],
        bundle.vocabulary,
        dfa,
        max_new_tokens=max_steps,
        next_label_kwargs={"text": prompt_text_for_head(program.autoregressive_head, sample)},
    )
    return result.labels


def greedy_sequence(program, bundle, sample, max_steps):
    if bundle.generation_constraints in {"always", "eval"}:
        return dfa_constrained_sequence(program, bundle, sample, max_steps)
    labels = []
    prefix = [bundle.vocabulary.eos_label]
    for _step in range(max_steps):
        logits = program.autoregressive_head.next_label_logits(
            prefix,
            text=prompt_text_for_head(program.autoregressive_head, sample),
        )
        label = int(torch.argmax(logits.detach(), dim=-1).item())
        labels.append(label)
        prefix.append(label)
        if label == bundle.vocabulary.eos_label:
            break
    return labels


def prompt_text_for_head(head, sample):
    """Select the input field expected by an autoregressive generation head."""
    key = getattr(head, "prompt_key", "text")
    return sample.get(key, sample.get("text", ""))


def labels_through_first_eos(labels, eos_label):
    """Return meaningful sequence labels, including the first EOS marker."""
    effective = []
    for label in labels:
        value = int(label.item() if torch.is_tensor(label) else label)
        effective.append(value)
        if value == eos_label:
            break
    return effective


def sequence_score(program, bundle, examples, max_steps, device="cpu", use_dfa=False, limit=None, show=False):
    eval_examples = examples if limit is None else examples[:limit]
    if not eval_examples:
        return {
            "examples": 0,
            "exact_sequence": 0.0,
            "token_accuracy": 0.0,
            "dfa_valid": 0.0,
            "dfa_checked": bool(use_dfa),
            "gt_state_success": 0.0,
            "gt_state_recall": 0.0,
            "temporal_progress": 0.0,
            "world_constraint_score": None,
            "world_constraint_applicable": 0.0,
            "world_constraint_declared": 0.0,
            "rl_reward_score": 0.0,
            "nonempty_plan_rate": 0.0,
            "average_predicted_length": 0.0,
            "positive_reward_rate": 0.0,
        }

    dfa = (
        bundle.policy_dfa
        if use_dfa or bundle.generation_constraints in {"always", "eval"}
        else None
    )
    exact = 0
    token_correct = 0
    token_total = 0
    dfa_valid = 0
    gt_state_success = 0
    gt_state_recall_total = 0.0
    temporal_progress_total = 0.0
    world_constraint_total = 0.0
    world_constraint_count = 0
    world_constraint_applicable_total = 0
    world_constraint_declared_total = 0
    rl_reward_total = 0.0
    nonempty_plan_total = 0
    predicted_length_total = 0
    positive_reward_total = 0
    autoregressive_head = getattr(program, "autoregressive_head", None)
    head_was_training = (
        autoregressive_head.training if autoregressive_head is not None else None
    )
    if autoregressive_head is not None:
        autoregressive_head.eval()

    for idx, sample in enumerate(eval_examples):
        program.populate_one(sample, device=device)
        labels = dfa_constrained_sequence(program, bundle, sample, max_steps) if use_dfa else greedy_sequence(program, bundle, sample, max_steps)
        gold = [int(x.item() if torch.is_tensor(x) else x) for x in sample["target_action_labels"][:max_steps]]
        pred = [int(x.item() if torch.is_tensor(x) else x) for x in labels[:max_steps]]
        effective_pred = labels_through_first_eos(pred, bundle.vocabulary.eos_label)
        predicted_actions = [label for label in effective_pred if label != bundle.vocabulary.eos_label]
        nonempty_plan_total += int(bool(predicted_actions))
        predicted_length_total += len(predicted_actions)
        pred_padded = pred + [bundle.vocabulary.eos_label] * max(0, len(gold) - len(pred))
        pred_padded = pred_padded[:len(gold)]
        exact += int(pred_padded == gold)
        effective_gold = labels_through_first_eos(gold, bundle.vocabulary.eos_label)
        token_correct += sum(int(p == g) for p, g in zip(pred_padded, effective_gold))
        token_total += len(effective_gold)
        if dfa is not None:
            dfa_valid += int(bundle.policy_for(sample).accepts(pred_padded))
        else:
            dfa_valid += 1

        eval_res = evaluate_goal_satisfaction(
            pred_padded,
            sample,
            bundle.vocabulary,
            world_bundle=getattr(bundle, "world", None),
            reward_mode=getattr(bundle, "reward_mode", "binary"),
            constraint_weight=getattr(bundle, "constraint_weight", 0.25),
            constraint_aggregate=getattr(bundle, "constraint_aggregate", "mean"),
        )
        gt_state_success += int(eval_res["is_success"] == 1.0)
        gt_state_recall_total += eval_res["recall"]
        temporal_progress_total += eval_res.get("temporal_progress", 1.0)
        if eval_res["world_constraint_score"] is not None:
            world_constraint_total += eval_res["world_constraint_score"]
            world_constraint_count += 1
        world_constraint_applicable_total += eval_res["world_constraint_applicable_count"]
        world_constraint_declared_total += eval_res["world_constraint_declared_count"]
        rl_reward_total += eval_res["rl_reward_score"]
        positive_reward_total += int(eval_res.get("task_reward_score", eval_res["rl_reward_score"]) > 0.0)

        if show:
            print()
            print(f"## Example {idx}: {sample.get('task_id', 'task')}")
            print(f"Instruction: {sample.get('natural_language_description') or sample.get('text')}")
            print(f"Gold sequence:      {labels_to_actions(gold, bundle.vocabulary)}")
            print(f"Predicted sequence: {labels_to_actions(pred_padded, bundle.vocabulary)}")
            print(f"Gold State:         {sorted(eval_res['gold_state'])}")
            print(f"Predicted State:    {sorted(eval_res['predicted_state'])}")
            print(
                f"Goal Success (0/1): {eval_res['is_success']} "
                f"(recall={eval_res['recall']:.2f}, "
                f"temporal_progress={eval_res.get('temporal_progress', 1.0):.2f})"
            )

    if autoregressive_head is not None:
        autoregressive_head.train(head_was_training)
    return {
        "examples": len(eval_examples),
        "exact_sequence": exact / len(eval_examples),
        "token_accuracy": token_correct / token_total if token_total else 0.0,
        "dfa_valid": dfa_valid / len(eval_examples),
        "dfa_checked": dfa is not None,
        "gt_state_success": gt_state_success / len(eval_examples),
        "gt_state_recall": gt_state_recall_total / len(eval_examples),
        "temporal_progress": temporal_progress_total / len(eval_examples),
        "world_constraint_score": (
            world_constraint_total / world_constraint_count
            if world_constraint_count else None
        ),
        "world_constraint_applicable": (
            world_constraint_applicable_total / len(eval_examples)
        ),
        "world_constraint_declared": (
            world_constraint_declared_total / len(eval_examples)
        ),
        "rl_reward_score": rl_reward_total / len(eval_examples),
        "nonempty_plan_rate": nonempty_plan_total / len(eval_examples),
        "average_predicted_length": predicted_length_total / len(eval_examples),
        "positive_reward_rate": positive_reward_total / len(eval_examples),
    }


def results_path_for_program(program_type):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_PATHS.get(program_type, RESULTS_DIR / "results.txt")


def _checkpoint_metadata(args, bundle, stage, epoch):
    head = bundle.vocabulary
    baseline_model = getattr(args, "baseline_model", "tiny-transformer")
    label_head = getattr(args, "causal_label_head", "pretrained-adapter")
    adapter_rank = int(getattr(args, "label_adapter_rank", 64))
    metadata = {
        "backbone": getattr(args, "llm_backbone_path", None) if baseline_model == "causal-lm" else baseline_model,
        "vocabulary": list(head.labels),
        "label_head": label_head if baseline_model == "causal-lm" else "native",
        "label_adapter_rank": adapter_rank if baseline_model == "causal-lm" else 0,
        "prompt_format": CAUSAL_PROMPT_FORMAT if baseline_model == "causal-lm" else "native",
        "stage": str(stage),
        "epoch": int(epoch),
    }
    auxiliary = (getattr(bundle, "model_metadata", None) or {}).get(
        "vlabench_auxiliary"
    )
    if auxiliary is not None:
        # Optional provenance keeps version-1 checkpoints backward compatible:
        # existing loaders validate their original required keys only.
        metadata["vlabench_auxiliary"] = dict(auxiliary)
    return metadata


def save_eai_checkpoint(program, bundle, args, path, stage, epoch):
    metadata = _checkpoint_metadata(args, bundle, stage, epoch)
    torch.save(
        {
            "eai_checkpoint_version": 1,
            "metadata": metadata,
            "model": program.model.state_dict(),
        },
        path,
    )


def load_eai_checkpoint(program, bundle, args, path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not (isinstance(checkpoint, dict) and checkpoint.get("eai_checkpoint_version") == 1):
        if (
            getattr(args, "baseline_model", "tiny-transformer") == "causal-lm"
            and getattr(args, "causal_label_head", "pretrained-adapter") != "linear"
        ):
            raise ValueError(
                "Legacy causal-LM checkpoints use the random linear label head; "
                "reload with --causal-label-head linear."
            )
        if isinstance(checkpoint, dict) and "model" in checkpoint and "cmodel" in checkpoint:
            program.model.load_state_dict(checkpoint["model"])
            if getattr(program, "cmodel", None) is not None:
                program.cmodel.load_state_dict(checkpoint["cmodel"])
        else:
            program.model.load_state_dict(checkpoint)
        return None
    expected = _checkpoint_metadata(args, bundle, stage="load", epoch=0)
    actual = checkpoint.get("metadata", {})
    for key in (
        "backbone",
        "vocabulary",
        "label_head",
        "label_adapter_rank",
        "prompt_format",
    ):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Incompatible EAI checkpoint {key}: saved={actual.get(key)!r}, "
                f"requested={expected.get(key)!r}"
            )
    program.model.load_state_dict(checkpoint["model"])
    return actual


def print_score(title, score, program_type=None):
    dfa_value = f"{score['dfa_valid']:.3f}" if score.get("dfa_checked", True) else "n/a"
    constraint_score = score.get("world_constraint_score")
    constraint_value = f"{constraint_score:.3f}" if constraint_score is not None else "n/a"
    line = (
        f"{title}: examples={score['examples']} "
        f"exact_sequence={score['exact_sequence']:.3f} "
        f"token_accuracy={score['token_accuracy']:.3f} "
        f"nonempty_plan_rate={score.get('nonempty_plan_rate', 0.0):.3f} "
        f"average_predicted_length={score.get('average_predicted_length', 0.0):.2f} "
        f"positive_reward_rate={score.get('positive_reward_rate', 0.0):.3f} "
        f"dfa_valid={dfa_value} "
        f"gt_state_success={score.get('gt_state_success', 0.0):.3f} "
        f"gt_state_recall={score.get('gt_state_recall', 0.0):.3f} "
        f"temporal_progress={score.get('temporal_progress', 1.0):.3f} "
        f"world_constraint_score_applicable={constraint_value} "
        f"world_constraints_applicable_per_example="
        f"{score.get('world_constraint_applicable', 0.0):.1f} "
        f"world_constraints_declared="
        f"{score.get('world_constraint_declared', 0.0):.1f} "
        f"rl_reward_score={score.get('rl_reward_score', 0.0):.3f}"
    )
    print(line)
    with results_path_for_program(program_type).open("a") as results_file:
        results_file.write(line + "\n")


def build_trainable_program(args, examples, device, shared_autoregressive_head=None):
    world_constraint_builders = getattr(args, "world_constraint_builders", None)
    if getattr(args, "no_world_constraints", False):
        world_constraint_builders = ()
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
        action_sequence_tokens=action_tokens_from_examples(examples),
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
        shared_llm_model=getattr(args, "_shared_llm_model", None),
        shared_llm_tokenizer=getattr(args, "_shared_llm_tokenizer", None),
        enforce_action_object=getattr(args, "_enforce_action_object", True),
        enforce_action_object_constraints=getattr(args, "_enforce_action_object_constraints", True),
        rl_estimator=getattr(args, "rl_estimator", "reinforce"),
        rl_num_samples=getattr(args, "rl_num_samples", 8),
        rl_rescore_microbatch=getattr(args, "rl_rescore_microbatch", 1),
        rl_reward_mode=getattr(args, "rl_reward_mode", "dense"),
        rl_supervised_weight=getattr(args, "rl_supervised_weight", 0.5),
        rl_constraint_weight=getattr(args, "rl_constraint_weight", 0.25),
        rl_constraint_aggregate=getattr(args, "rl_constraint_aggregate", "mean"),
        world_constraint_builders=world_constraint_builders,
        shared_autoregressive_head=shared_autoregressive_head,
        generation_constraints=getattr(args, "generation_constraints", "always"),
        causal_label_head=getattr(args, "causal_label_head", "pretrained-adapter"),
        label_adapter_rank=getattr(args, "label_adapter_rank", 64),
    )
    mode = getattr(args, "rl_reward_mode", "dense")
    weight = getattr(args, "rl_constraint_weight", 0.25)
    aggregate = getattr(args, "rl_constraint_aggregate", "mean")
    for example in examples:
        example["reward_function"] = make_eai_reward_function(
            example,
            vocabulary=bundle.vocabulary,
            mode=mode,
            world_bundle=bundle.world,
            constraint_weight=weight,
            constraint_aggregate=aggregate,
        )
    return program, bundle


def build_stage2_program(args, solver_program, bundle, examples, device):
    """Create the RL program on the exact Stage 1 graph, head, world, and DFA."""
    program = AutoregressiveSequenceReinforcementProgram(
        solver_program.graph,
        targets=[bundle.generated_token],
        autoregressive_head=solver_program.autoregressive_head,
        eos_label=bundle.vocabulary.eos_label,
        max_steps=args.max_steps,
        supervised_weight=args.rl_supervised_weight,
        rescore_microbatch_size=getattr(args, "rl_rescore_microbatch", 1),
        reward_key="reward_function",
        decoder=eai_action_decoder,
        num_samples=args.rl_num_samples,
        estimator=args.rl_estimator,
        policy_dfa=(
            bundle.policy_dfa if args.generation_constraints == "always" else None
        ),
        policy_dfa_factory=(
            bundle.policy_for if args.generation_constraints == "always" else None
        ),
        poi=[
            bundle.text,
            bundle.token,
            bundle.generated_token,
            bundle.token[bundle.contains],
            bundle.token[bundle.generated_token],
        ],
        device=device,
    )
    program.autoregressive_head = solver_program.autoregressive_head
    for example in examples:
        example["reward_function"] = make_eai_reward_function(
            example,
            vocabulary=bundle.vocabulary,
            mode=args.rl_reward_mode,
            world_bundle=bundle.world,
            constraint_weight=args.rl_constraint_weight,
            constraint_aggregate=args.rl_constraint_aggregate,
        )
    return program, bundle


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
    dfa_text = (
        "with graph DFA"
        if getattr(args, "generation_constraints", "always") in {"always", "eval"}
        else "without graph DFA"
    )
    return f"epoch {epoch} {args.dataset} {args.program} {split_name} {dfa_text}"


def _mean_loss_value(value):
    if isinstance(value, dict):
        values = [_mean_loss_value(item) for item in value.values()]
        values = [item for item in values if item is not None]
        return sum(values) / len(values) if values else None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def stage1_selection_key(score, supervised_loss=None):
    """Rank checkpoints by task semantics, using loss only as a final tie-breaker."""
    loss = float(supervised_loss) if supervised_loss is not None else float("inf")
    return (
        float(score.get("gt_state_recall", 0.0)),
        float(score.get("gt_state_success", 0.0)),
        float(score.get("positive_reward_rate", 0.0)),
        float(score.get("exact_sequence", 0.0)),
        float(score.get("token_accuracy", 0.0)),
        -loss,
    )


def stage2_selection_key(score):
    """Rank RL checkpoints by task completion, then plan quality."""
    return (
        float(score.get("gt_state_success", 0.0)),
        float(score.get("gt_state_recall", 0.0)),
        float(score.get("temporal_progress", 0.0)),
        float(score.get("positive_reward_rate", 0.0)),
        -float(score.get("average_predicted_length", float("inf"))),
        float(score.get("exact_sequence", 0.0)),
        float(score.get("token_accuracy", 0.0)),
        float(score.get("rl_reward_score", 0.0)),
    )


def _capture_trainable_parameters(program):
    """Keep only LoRA/adapter/trainable tensors, never a second frozen Qwen copy."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in program.model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_parameters(program, snapshot):
    parameters = dict(program.model.named_parameters())
    missing = sorted(set(snapshot) - set(parameters))
    if missing:
        raise ValueError(f"Stage 1 snapshot parameters are missing from the model: {missing[:3]}")
    with torch.no_grad():
        for name, saved in snapshot.items():
            parameters[name].copy_(saved.to(parameters[name].device, dtype=parameters[name].dtype))


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
    _maybe_train_vlabench_auxiliary(args, program, bundle, device)
    print(f"Starting training and will save at {args.model}")
    if args.eval_every_epoch:
        for epoch in range(1, args.epochs + 1):
            print(f"Training epoch {epoch}/{args.epochs}")
            program.train(train, valid_set=dev, test_set=None, **_train_kwargs(args, train, device, 1))
            report_epoch_accuracy(args, program, bundle, train, dev, device, epoch)
    else:
        program.train(train, valid_set=dev, test_set=None, **_train_kwargs(args, train, device, args.epochs))

    if args.model:
        model_path = Path(args.model)
        model_path.parent.mkdir(exist_ok=True, parents=True)
        save_eai_checkpoint(program, bundle, args, model_path, stage="supervised", epoch=args.epochs)
        print(f"Saved model: {model_path}")
    return program, bundle


def _maybe_train_vlabench_auxiliary(args, program, bundle, device):
    """Run the optional domain-separated text warm-up on the shared LoRA."""
    enabled = bool(
        getattr(args, "vlabench_aux_planning_dir", None)
        and int(getattr(args, "vlabench_aux_epochs", 0)) > 0
    )
    if not enabled:
        return None

    from test_regr.VLABenchAgentInterface.training import build_constraint_runtime
    from vlabench_auxiliary import train_vlabench_text_auxiliary

    planning_dir = ensure_vlabench_auxiliary_planning_data(
        args.vlabench_aux_planning_dir
    )
    auxiliary_examples = load_vlabench_auxiliary_planning_examples(
        planning_dir,
        limit=args.vlabench_aux_limit,
    )
    auxiliary_split = split_vlabench_auxiliary_examples(auxiliary_examples)
    max_entities = max(64, max(len(item.entities) for item in auxiliary_examples))
    max_operations = max(
        8, max(len(item.operation_sequence) for item in auxiliary_examples)
    )
    auxiliary_runtime = build_constraint_runtime(
        max_entities=max_entities,
        max_operations=max_operations,
        name_prefix="eai_vlabench_aux",
    )
    final_model_path = (
        Path(args.model)
        if args.model
        else MODEL_DIR / "eai_action_sequence_baseline.pth"
    )
    auxiliary_checkpoint_path = final_model_path.with_suffix(
        ".vlabench_aux.pth"
    )
    auxiliary_lr = args.lr if args.vlabench_aux_lr is None else args.vlabench_aux_lr
    auxiliary_result = train_vlabench_text_auxiliary(
        program.autoregressive_head,
        auxiliary_split,
        auxiliary_runtime,
        epochs=args.vlabench_aux_epochs,
        lr=auxiliary_lr,
        device=device,
        max_length=args.encoder_max_length,
        label_head=args.causal_label_head,
        label_adapter_rank=args.label_adapter_rank,
        checkpoint_path=auxiliary_checkpoint_path,
        resume_path=args.vlabench_aux_resume,
    )
    bundle.model_metadata["vlabench_auxiliary"] = {
        "planning_dir": str(planning_dir),
        "selected_epoch": auxiliary_result.selected_epoch,
        "vocabulary_checksum": auxiliary_result.vocabulary_checksum,
        "domain_checksum": auxiliary_result.domain_checksum,
        "checkpoint": str(auxiliary_checkpoint_path),
    }
    metadata = dict(bundle.model_metadata["vlabench_auxiliary"])
    # The result contains a CPU snapshot for programmatic callers. EAI has
    # already restored that state, so release the duplicate before Stage 1.
    del auxiliary_result, auxiliary_runtime, auxiliary_split, auxiliary_examples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def train_two_stage(args, train, dev, examples, device):
    """Two-stage training: Exact Match pretraining -> DomiKnowS Reinforcement Learning fine-tuning."""
    orig_program = args.program
    args.program = "solver"

    auxiliary_enabled = bool(
        getattr(args, "vlabench_aux_planning_dir", None)
        and int(getattr(args, "vlabench_aux_epochs", 0)) > 0
    )
    if auxiliary_enabled:
        print("\n" + "=" * 65)
        print("AUXILIARY: Text-Only VLABench Planning Warm-Up")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print("STAGE 1: Supervised Exact Match Pretraining (SolverPOI)")
        print("=" * 65)
    solver_program, bundle = build_trainable_program(args, examples, device)
    if auxiliary_enabled:
        _maybe_train_vlabench_auxiliary(args, solver_program, bundle, device)
        print("\n" + "=" * 65)
        print("STAGE 1: Supervised Exact Match Pretraining (SolverPOI)")
        print("=" * 65)
    stage1_optimizer = torch.optim.Adam(solver_program.model.parameters(), lr=args.lr)
    score_stage1 = None
    best_score = None
    best_key = None
    best_epoch = 0
    best_trainable_parameters = None
    for epoch in range(1, args.epochs + 1):
        solver_program.global_epoch = epoch
        stage_kwargs = _train_kwargs(args, train, device, 1)
        stage_kwargs["Optim"] = lambda _params, optimizer=stage1_optimizer: optimizer
        solver_program.train(train, valid_set=dev, test_set=None, **stage_kwargs)
        validation_loss = solver_program.model.loss.value() if solver_program.model.loss else None
        validation_loss = _mean_loss_value(validation_loss)
        score_stage1 = sequence_score(
            solver_program, bundle, dev or train, args.max_steps,
            device=device, use_dfa=args.use_dfa,
        )
        print(f"Stage 1 epoch {epoch} supervised_loss={validation_loss if validation_loss is not None else 'n/a'}")
        print_score(f"Stage 1 Epoch {epoch} Eval", score_stage1, "solver")
        selection_key = stage1_selection_key(score_stage1, validation_loss)
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            best_score = dict(score_stage1)
            best_trainable_parameters = _capture_trainable_parameters(solver_program)
            print(
                f"Stage 1 epoch {epoch} is the new best semantic checkpoint "
                f"(recall={score_stage1['gt_state_recall']:.3f}, "
                f"success={score_stage1['gt_state_success']:.3f}, "
                f"positive_reward_rate={score_stage1['positive_reward_rate']:.3f})."
            )
        if args.epoch_predictions > 0:
            sequence_score(
                solver_program, bundle, dev or train, args.max_steps,
                device=device, use_dfa=args.use_dfa,
                limit=args.epoch_predictions, show=True,
            )
    solver_program.global_epoch = None

    if score_stage1 is None:
        score_stage1 = sequence_score(
            solver_program, bundle, dev or train, args.max_steps,
            device=device, use_dfa=args.use_dfa,
        )
        best_score = dict(score_stage1)
        best_epoch = 0
        best_trainable_parameters = _capture_trainable_parameters(solver_program)
    if best_trainable_parameters is None or best_score is None:
        raise RuntimeError("Stage 1 did not produce a restorable checkpoint")
    _restore_trainable_parameters(solver_program, best_trainable_parameters)
    score_stage1 = sequence_score(
        solver_program, bundle, dev or train, args.max_steps,
        device=device, use_dfa=args.use_dfa,
    )
    print(f"Restored best Stage 1 epoch {best_epoch} before checkpointing and RL.")
    print_score("Stage 1 Best Checkpoint Eval", score_stage1, "solver")

    final_model_path = Path(args.model) if args.model else MODEL_DIR / "eai_action_sequence_baseline.pth"
    stage1_path = (
        Path(args.stage1_checkpoint)
        if args.stage1_checkpoint
        else final_model_path.with_suffix(".stage1.pth")
    )
    stage1_path.parent.mkdir(exist_ok=True, parents=True)
    save_eai_checkpoint(
        solver_program, bundle, args, stage1_path, stage="stage1", epoch=best_epoch
    )
    print(f"Saved Stage 1 checkpoint: {stage1_path}")
    if not stage1_allows_rl(
        score_stage1,
        min_positive_reward_rate=args.stage1_min_positive_reward_rate,
        min_goal_recall=args.stage1_min_goal_recall,
        min_goal_success_rate=args.stage1_min_goal_success_rate,
    ):
        print(
            "Stage 2 skipped: best Stage 1 validation metrics did not meet the "
            "exploration gate: "
            f"positive_reward_rate={score_stage1['positive_reward_rate']:.3f} "
            f"(required {args.stage1_min_positive_reward_rate:.3f}), "
            f"gt_state_recall={score_stage1['gt_state_recall']:.3f} "
            f"(required {args.stage1_min_goal_recall:.3f}), "
            f"gt_state_success={score_stage1['gt_state_success']:.3f} "
            f"(required {args.stage1_min_goal_success_rate:.3f})."
        )
        args.program = orig_program
        return solver_program, bundle, False

    print("\n" + "=" * 65)
    print("STAGE 2: Reinforcement Learning Fine-Tuning (Dense Goal + Constraint-Modulated Reward)")
    print("=" * 65)
    # Stage 2 constructs its own optimizer over the shared trainable weights.
    # Drop Stage 1's optimizer state before allocating any Qwen rollout graphs.
    solver_program.opt = None
    stage_kwargs = None
    del stage1_optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args.program = "reinforcement"
    rl_program, rl_bundle = build_stage2_program(
        args, solver_program, bundle, examples, device
    )

    rl_epochs = getattr(args, "rl_epochs", args.epochs)
    rl_lr = getattr(args, "rl_lr", None)
    if rl_lr is None:
        rl_lr = 1e-5 if args.baseline_model == "causal-lm" else 1e-4
    stage2_optimizer = torch.optim.Adam(rl_program.model.parameters(), lr=rl_lr)
    best_stage2_key = None
    best_stage2_epoch = 0
    best_stage2_parameters = None
    for epoch in range(1, rl_epochs + 1):
        rl_program.global_epoch = epoch
        rl_kwargs = {
            "device": device,
            "train_epoch_num": 1,
            "Optim": lambda _params, optimizer=stage2_optimizer: optimizer,
            "test_every_epoch": False,
        }
        rl_program.train(train, valid_set=dev, test_set=None, **rl_kwargs)
        epoch_score = sequence_score(
            rl_program,
            rl_bundle,
            dev or train,
            args.max_steps,
            device=device,
            use_dfa=args.use_dfa,
        )
        print_score(
            f"Stage 2 Epoch {epoch} Eval", epoch_score, "reinforcement"
        )
        selection_key = stage2_selection_key(epoch_score)
        if best_stage2_key is None or selection_key > best_stage2_key:
            best_stage2_key = selection_key
            best_stage2_epoch = epoch
            best_stage2_parameters = _capture_trainable_parameters(rl_program)
            print(
                f"Stage 2 epoch {epoch} is the new best semantic checkpoint "
                f"(success={epoch_score['gt_state_success']:.3f}, "
                f"recall={epoch_score['gt_state_recall']:.3f}, "
                f"temporal_progress={epoch_score['temporal_progress']:.3f}, "
                f"positive_reward_rate={epoch_score['positive_reward_rate']:.3f})."
            )
    rl_program.global_epoch = None

    if best_stage2_parameters is None:
        # Preserve a well-defined checkpoint for an explicitly requested
        # zero-epoch ablation, even though normal CLI runs use positive epochs.
        best_stage2_parameters = _capture_trainable_parameters(rl_program)
    _restore_trainable_parameters(rl_program, best_stage2_parameters)
    score_stage2 = sequence_score(
        rl_program,
        rl_bundle,
        dev or train,
        args.max_steps,
        device=device,
        use_dfa=args.use_dfa,
    )
    print(f"Restored best Stage 2 epoch {best_stage2_epoch} before checkpointing.")
    print_score("Stage 2 (Reinforcement Learning) Eval", score_stage2, "reinforcement")

    if args.model:
        model_path = Path(args.model)
        model_path.parent.mkdir(exist_ok=True, parents=True)
        save_eai_checkpoint(
            rl_program,
            rl_bundle,
            args,
            model_path,
            stage="stage2",
            epoch=best_stage2_epoch,
        )
        print(f"Saved two-stage model: {model_path}")
    args.program = orig_program
    return rl_program, rl_bundle, True


def stage1_allows_rl(
    score,
    min_positive_reward_rate=0.25,
    min_goal_recall=0.10,
    min_goal_success_rate=0.05,
):
    """Require broad task progress, not one coincidentally rewarded trajectory."""
    thresholds = {
        "min_positive_reward_rate": min_positive_reward_rate,
        "min_goal_recall": min_goal_recall,
        "min_goal_success_rate": min_goal_success_rate,
    }
    for name, value in thresholds.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    return (
        float(score.get("positive_reward_rate", 0.0)) >= float(min_positive_reward_rate)
        and float(score.get("gt_state_recall", 0.0)) >= float(min_goal_recall)
        and float(score.get("gt_state_success", 0.0)) >= float(min_goal_success_rate)
    )


def load_trained_program(args, examples, device):
    program, bundle = build_trainable_program(args, examples, device)
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            metadata = load_eai_checkpoint(
                program, bundle, args, model_path, map_location=device
            )
            print(f"Loaded model: {model_path}")
            if metadata is not None:
                print(
                    f"Checkpoint metadata: stage={metadata['stage']} "
                    f"epoch={metadata['epoch']} label_head={metadata['label_head']}"
                )
        else:
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return program, bundle


def run_train_or_evaluate(args, examples, device):
    train, dev = split_train_dev(examples, args.dev_fraction)
    eval_examples = dev or train
    program = bundle = None
    stage2_completed = True
    if args.two_stage:
        program, bundle, stage2_completed = train_two_stage(args, train, dev, examples, device)
        if not stage2_completed:
            return 2
    elif args.train:
        program, bundle = train_program(args, train, dev, examples, device)
    if args.evaluate or args.eval_only:
        if program is None or bundle is None:
            program, bundle = load_trained_program(args, examples, device)
        evaluated_program_type = "reinforcement" if args.two_stage else args.program
        dfa_text = (
            "with graph DFA"
            if args.generation_constraints in {"always", "eval"}
            else "without graph DFA"
        )
        title = f"{args.dataset} {evaluated_program_type} {dfa_text}"
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
        print_score(title, score, evaluated_program_type)
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
        action_sequence_tokens=action_tokens_from_examples(examples),
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
        rl_estimator=args.rl_estimator,
        rl_num_samples=args.rl_num_samples,
        rl_reward_mode=args.rl_reward_mode,
        rl_supervised_weight=args.rl_supervised_weight,
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
    parser.add_argument("--lr", type=float, default=None, help="Stage 1 learning rate. Defaults to 1e-4 for causal-lm and 1e-3 for smaller baselines.")
    parser.add_argument("--program", choices=["solver", "primal-dual", "reinforcement"], default="solver", help="DomiKnowS training program to use for the autoregressive baseline.")
    parser.add_argument("--two-stage", action="store_true", help="Two-stage training: Exact Match pretraining -> DomiKnowS Reinforcement Learning fine-tuning.")
    parser.add_argument("--rl-estimator", choices=["importance_weighted", "reinforce"], default="reinforce", help="Stage 2 policy-gradient estimator. REINFORCE is the on-policy default; importance_weighted uses recorded detached proposal probabilities.")
    parser.add_argument("--rl-num-samples", type=int, default=8, help="Number of decodings sampled per Stage 2 step (minimum 4; default 8).")
    parser.add_argument("--rl-rescore-microbatch", type=int, default=1, help="Differentiable rollout rescoring microbatch. Keep 1 for Qwen-8B memory safety; the optimizer still uses all rollouts in one estimate.")
    parser.add_argument("--rl-reward-mode", choices=["binary", "dense"], default="dense", help="Task reward used by RL; dense final-state recall is multiplied by ordered temporal-prefix progress.")
    parser.add_argument("--rl-supervised-weight", type=float, default=0.5, help="Teacher-forced Stage 1 anchor retained during RL; the stronger default limits policy drift. Set 0 to disable.")
    parser.add_argument("--rl-constraint-weight", type=float, default=0.25, help="Maximum task-reward discount for world-constraint violations; constraints cannot reward a zero-task plan.")
    parser.add_argument("--rl-constraint-aggregate", choices=["mean", "min", "prod"], default="mean", help="Aggregation across declared world constraints.")
    parser.add_argument("--no-world-constraints", action="store_true", help="Disable the default world and transition constraints.")
    parser.add_argument("--rl-epochs", type=int, default=3, help="Epochs for Stage 2 RL fine-tuning.")
    parser.add_argument("--rl-lr", type=float, default=None, help="Stage 2 learning rate. Defaults to 1e-5 for causal-lm/LoRA and 1e-4 for smaller baselines.")
    parser.add_argument("--baseline-model", choices=["tiny-transformer", "bert-gru", "causal-lm"], default="tiny-transformer", help="Autoregressive baseline architecture. tiny-transformer is small and fully trainable; causal-lm uses a frozen small LLM backbone.")
    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct", help="Causal LM backbone for --baseline-model causal-lm.")
    parser.add_argument("--causal-label-head", choices=["pretrained-adapter", "linear"], default="pretrained-adapter", help="Use Qwen's native output embeddings or the legacy random linear label classifier.")
    parser.add_argument("--label-adapter-rank", type=int, default=64, help="Rank of the trainable residual in the pretrained causal label adapter.")
    parser.add_argument("--use-lora", action="store_true", help="Train LoRA adapters on the causal LM backbone.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank for --baseline-model causal-lm --use-lora.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha for --baseline-model causal-lm --use-lora.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout for --baseline-model causal-lm --use-lora.")
    parser.add_argument("--lora-target-modules", nargs="+", default=None, help="Optional LoRA target module names. Defaults to Qwen attention/MLP projections.")
    parser.add_argument("--llm-device-map", default=None, help="Optional Hugging Face device_map for causal LM loading, e.g. auto for multi-GPU sharding.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing for causal LM LoRA training to reduce activation memory.")
    parser.add_argument("--vlabench-aux-planning-dir", default=str(VLABENCH_AUX_DATA_DIR), help="Local VLABench planning snapshot for text-only LoRA warm-up; defaults inside the EAI data directory.")
    parser.add_argument("--vlabench-aux-epochs", type=int, default=0, help="Text-only VLABench auxiliary epochs before EAI Stage 1; 0 disables the phase.")
    parser.add_argument("--vlabench-aux-limit", type=int, default=None, help="Optional planning-episode limit for the VLABench auxiliary phase.")
    parser.add_argument("--vlabench-aux-lr", type=float, default=None, help="VLABench auxiliary learning rate; defaults to the EAI Stage 1 learning rate.")
    parser.add_argument("--vlabench-aux-resume", default=None, help="Optional compatible .vlabench_aux.pth checkpoint used to initialize the warm-up.")
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
    parser.add_argument("--generation-constraints", choices=["always", "eval", "off"], default="always", help="Apply the DFA compiled from graph policies during RL and evaluation, evaluation only, or not at runtime.")
    parser.add_argument("--stage1-checkpoint", default=None, help="Stage 1 checkpoint path; defaults to <model-stem>.stage1.pth.")
    parser.add_argument("--epoch-predictions", type=int, default=3, help="Decoded validation examples printed after each Stage 1 epoch; 0 disables them.")
    parser.add_argument("--stage1-min-positive-reward-rate", type=float, default=0.25, help="Minimum fraction of validation trajectories with positive task reward required before RL.")
    parser.add_argument("--stage1-min-goal-recall", type=float, default=0.10, help="Minimum validation goal-fact recall required before RL.")
    parser.add_argument("--stage1-min-goal-success-rate", type=float, default=0.05, help="Minimum validation goal-success rate required before RL.")
    parser.add_argument("--show-predictions", action="store_true", help="Print decoded examples during evaluation.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", default=str(MODEL_DIR / "eai_action_sequence_baseline.pth"))
    parser.add_argument("--use-llm", action="store_true", help="Attach a small text LLM as a DomiKnowS ModuleSensor for action-sequence generation.")
    parser.add_argument("--llm-model-path", default="Qwen/Qwen2.5-0.5B-Instruct", help="Small Hugging Face causal LM used by the DomiKnowS generated action sequence sensor.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated plan tokens for --use-llm.")
    parser.add_argument("--num-generations", type=int, default=300, help="Number of examples to decode/show.")
    parser.add_argument("--single-run", action="store_true", help="Train only --program on --dataset instead of the BEHAVIOR/VirtualHome normal+PMD suite.")
    args = parser.parse_args()
    if (args.two_stage or args.program == "reinforcement") and args.rl_num_samples < 4:
        parser.error(
            "--rl-num-samples must be at least 4 for Stage 2; two samples do "
            "not provide a stable within-task policy-gradient estimate"
        )
    auxiliary_enabled = bool(
        args.vlabench_aux_planning_dir and args.vlabench_aux_epochs > 0
    )
    if args.vlabench_aux_epochs < 0:
        parser.error("--vlabench-aux-epochs cannot be negative")
    if args.vlabench_aux_limit is not None and args.vlabench_aux_limit <= 0:
        parser.error("--vlabench-aux-limit must be positive")
    if args.vlabench_aux_lr is not None and args.vlabench_aux_lr <= 0:
        parser.error("--vlabench-aux-lr must be positive")
    if auxiliary_enabled and (
        args.baseline_model != "causal-lm" or not args.use_lora
    ):
        parser.error(
            "VLABench auxiliary warm-up requires --baseline-model causal-lm --use-lora"
        )
    if auxiliary_enabled and not args.two_stage and args.program != "solver":
        parser.error(
            "VLABench auxiliary warm-up must precede supervised EAI solver training"
        )
    if args.vlabench_aux_resume and not auxiliary_enabled:
        parser.error(
            "--vlabench-aux-resume requires an enabled VLABench auxiliary phase"
        )
    return args


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 1e-4 if args.baseline_model == "causal-lm" else 1e-3
        print(f"Using architecture-aware Stage 1 learning rate: {args.lr:g}")
    if args.rl_lr is None:
        args.rl_lr = 1e-5 if args.baseline_model == "causal-lm" else 1e-4
        print(f"Using architecture-aware Stage 2 learning rate: {args.rl_lr:g}")
    if args.use_dfa:
        print("Warning: --use-dfa is deprecated; using --generation-constraints always.")
        args.generation_constraints = "always"
    device = args.device
    examples = load_examples(args, device)
    write_vocab_info_log(examples)

    if args.use_llm:
        generate_llm_sequences(args, examples, device)
        return 0

    if args.train or args.two_stage or args.evaluate or args.eval_only:
        return run_train_or_evaluate(args, examples, device)

    train, dev = split_train_dev(examples, args.dev_fraction)
    shown = dev or train
    generate_baseline_sequences(args, shown, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
