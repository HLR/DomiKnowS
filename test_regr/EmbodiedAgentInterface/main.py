import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parents[1]))

from dataset import ACTION_VOCAB, EOS_TOKEN, dummy_dataset, load_eai_dataset, split_train_dev
from modules import SmallLLMPlanGenerator, TextBERTTokenEncoder, TokenActionClassifier


RUN_DIR = Path(__file__).parent.resolve()
MODEL_DIR = RUN_DIR / "models"
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
):
    from domiknows import setProductionLogMode
    from domiknows.program import SolverPOIProgram
    from domiknows.sensor.pytorch.sensors import ModuleSensor, ReaderSensor
    from domiknows.sensor.pytorch.learners import ModuleLearner
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor

    from graph import create_generation_graph

    setProductionLogMode(True)
    graph, bundle = create_generation_graph(max_steps=max_steps)
    graph.detach()

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
    token["target_action_label"] = ReaderSensor(keyword="target_action_labels", label=True)

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

    token_encoder = TextBERTTokenEncoder(
        model_path=encoder_model_path,
        device=device,
        max_length=encoder_max_length,
        freeze=freeze_encoder,
        max_steps=max_steps,
    )
    feature_dim = feature_dim or token_encoder.hidden_size
    token["features"] = ModuleSensor(
        bundle.contains,
        text["instruction_text"],
        "position",
        module=token_encoder,
        device=device,
    )
    token_classifier = TokenActionClassifier(
        feature_dim=feature_dim,
        label_count=bundle.vocabulary.label_count,
        hidden_dim=hidden_dim,
        device=device,
    )
    token[generated_token] = ModuleLearner(
        "features",
        module=token_classifier,
        device=device,
    )
    program = SolverPOIProgram(
        graph,
        poi=[text, token, generated_token, token[bundle.contains], token[generated_token]],
        inferTypes=['local/argmax']
    )
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


def generate_baseline_sequences(args, examples, device):
    program, bundle = build_program(
        device=device,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        encoder_model_path=args.encoder_model_path,
        encoder_max_length=args.encoder_max_length,
        freeze_encoder=not args.finetune_encoder,
        max_steps=args.max_steps,
    )
    correct = 0
    total = 0
    for idx, sample in enumerate(examples[:args.num_generations]):
        datanode = program.populate_one(sample, device=device)
        pred = labels_to_actions(generated_token_sequence(datanode, bundle), bundle.vocabulary)
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
    parser.add_argument("--max-steps", type=int, default=8, help="Padded action-token sequence length including EOS.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--feature-dim", type=int, default=None, help="Override encoder hidden size for the sequence head input.")
    parser.add_argument("--encoder-model-path", default="bert-base-uncased", help="Hugging Face/local BERT-style encoder for task text.")
    parser.add_argument("--encoder-max-length", type=int, default=256, help="Max tokens for the BERT text encoder.")
    parser.add_argument("--finetune-encoder", action="store_true", help="Allow gradients through the BERT encoder. Default freezes it.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", default=str(MODEL_DIR / "eai_action_sequence_baseline.pth"))
    parser.add_argument("--use-llm", action="store_true", help="Attach a small text LLM as a DomiKnowS ModuleSensor for action-sequence generation.")
    parser.add_argument("--llm-model-path", default="Qwen/Qwen2.5-0.5B-Instruct", help="Small Hugging Face causal LM used by the DomiKnowS generated action sequence sensor.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated plan tokens for --use-llm.")
    parser.add_argument("--num-generations", type=int, default=5, help="Number of examples to decode/show.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    examples = load_examples(args, device)

    if args.use_llm:
        generate_llm_sequences(args, examples, device)
        return 0

    train, dev = split_train_dev(examples, args.dev_fraction)
    shown = dev or train
    generate_baseline_sequences(args, shown, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
