import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parents[1]))

from dataset import EOS_TOKEN, split_train_dev
from graph import create_generation_graph
from main import (
    action_object_constraint_tokens_from_examples,
    action_tokens_from_examples,
    action_tokens_requiring_object_from_examples,
    build_trainable_program,
    dfa_constrained_sequence,
    generation_vocab_from_examples,
    greedy_sequence,
    labels_to_actions,
    load_examples,
    load_trained_program,
    object_tokens_from_examples,
    openable_object_tokens_from_examples,
)
from train_qwen_hmm import (
    build_prompt,
    generate_text,
    labels_to_ids,
    load_qwen,
    parse_generated_text,
)


def progress_bar(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate raw Qwen, no-training DomiKnowS DFA, trained DomiKnowS Qwen, DFA, and Ctrl-G-style HMM+DFA on EAI."
    )
    parser.add_argument("--dataset", choices=["all", "behavior", "virtualhome"], default="all")
    parser.add_argument("--split", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples loaded from the dataset.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit examples scored after selecting the eval split.")
    parser.add_argument("--eval-split", choices=["dev", "train", "full"], default="dev")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=135)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results_eval_settings.txt"))
    parser.add_argument("--show", type=int, default=0)

    parser.add_argument("--llm-backbone-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--qwen-label-batch-size", type=int, default=4, help="Batch size for scoring EAI compact labels with raw Qwen during constrained decoding.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--skip-raw-qwen", action="store_true")
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=[
            "0", "1", "2", "3", "4",
            "raw_dfa", "raw", "domiknows", "dfa", "hmm",
            "nt_domiknows", "nt_dfa", "nt_hmm_dfa", "nt_suite",
        ],
        default=["1", "2", "3", "4"],
        help=(
            "Evaluation settings to run. Use nt_suite for no-training original-Qwen "
            "DomiKnowS, DomiKnowS+DFA, and HMM+DFA under both constraint modes. "
            "Use numbers 1-4 for the trained checkpoint settings."
        ),
    )

    parser.add_argument("--model", default=None, help="Saved DomiKnowS model checkpoint to evaluate. Required for settings 2, 3, and 4.")
    parser.add_argument("--program", choices=["solver", "primal-dual"], default="solver")
    parser.add_argument("--baseline-model", choices=["bert-gru", "tiny-transformer", "causal-lm"], default="causal-lm")
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-model-path", default="bert-base-uncased")
    parser.add_argument("--encoder-max-length", type=int, default=256)
    parser.add_argument("--finetune-encoder", action="store_true")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="*", default=None)
    parser.add_argument("--llm-device-map", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--constraint-modes", nargs="+", choices=["general", "specific"], default=["general", "specific"], help="Constraint modes for no-training settings. general uses only action-followed-by-object; specific also uses action-specific object compatibility.")
    parser.add_argument("--hmm", default=None, help="Ctrl-G-style HMM .npz file for setting 4 or nt_hmm_dfa.")
    parser.add_argument("--hmm-alpha", type=float, default=1.0)
    parser.add_argument("--hmm-soft-mask", action="store_true", help="Legacy wrapper option; strict product HMM+DFA ignores this.")
    parser.add_argument("--hmm-search", choices=["greedy", "beam", "sample"], default="greedy")
    parser.add_argument("--hmm-beam-size", type=int, default=4)
    parser.add_argument("--hmm-weight", type=float, default=1.0)
    parser.add_argument("--hmm-hf-weight", type=float, default=0.0)
    parser.add_argument("--hmm-lookahead-weight", type=float, default=0.0)
    parser.add_argument("--hmm-lookahead-max-steps", type=int, default=8)
    parser.add_argument("--hmm-keep-rejected", action="store_true")
    parser.add_argument("--allow-empty-plan", action="store_true", help="Allow HMM+DFA to emit <eos> as the first generated token. Disabled by default for EAI success evaluation.")
    return parser.parse_args()


def select_eval_examples(examples, eval_split, dev_fraction, limit):
    if eval_split == "full":
        selected = examples
    else:
        train, dev = split_train_dev(examples, dev_fraction)
        selected = dev if eval_split == "dev" else train
        if eval_split == "dev" and not selected:
            selected = train
    return selected if limit is None else selected[:limit]


def gold_labels(sample, max_steps):
    return [int(x.item() if torch.is_tensor(x) else x) for x in sample["target_action_labels"][:max_steps]]


def pad_prediction(pred, gold, eos_label):
    pred = [int(x.item() if torch.is_tensor(x) else x) for x in pred]
    pred = pred[: len(gold)]
    if len(pred) < len(gold):
        pred = pred + [eos_label] * (len(gold) - len(pred))
    return pred


def trim_at_eos(labels, eos_label):
    trimmed = []
    for label in labels:
        label = int(label.item() if torch.is_tensor(label) else label)
        if label == eos_label:
            break
        trimmed.append(label)
    return trimmed


def token_sequence(labels, vocabulary):
    return labels_to_actions(labels, vocabulary)


def _side_for_action(action):
    if action.startswith("left_"):
        return "left"
    if action.startswith("right_"):
        return "right"
    return None


def _placement_relation(action):
    if "inside" in action:
        return "inside"
    if "ontop" in action or "on_top" in action:
        return "ontop"
    if "nextto" in action:
        return "nextto"
    if "under" in action:
        return "under"
    return None


def _consume_action_object(tokens, index):
    action = tokens[index]
    obj = tokens[index + 1] if index + 1 < len(tokens) else None
    return action, obj, index + (2 if obj is not None else 1)


def abstract_state_from_tokens(labels, vocabulary):
    """Build a light symbolic final state from EAI action/object tokens.

    This is not trajectory matching.  It extracts state-changing effects so a
    different action order can still succeed if it reaches the same gold facts.
    """
    tokens = token_sequence(labels, vocabulary)
    facts = set()
    held = {"left": None, "right": None}
    index = 0
    while index < len(tokens):
        action, obj, index = _consume_action_object(tokens, index)
        if not action or action == EOS_TOKEN:
            continue

        side = _side_for_action(action)
        if "grasp" in action and obj:
            held[side or "right"] = obj
            continue

        relation = _placement_relation(action)
        if relation and obj:
            held_obj = held.get(side or "right") or held.get("right") or held.get("left")
            if held_obj:
                facts = {fact for fact in facts if not (len(fact) >= 2 and fact[1] == held_obj and fact[0] in {"inside", "ontop", "nextto", "under", "onfloor"})}
                facts.add((relation, held_obj, obj))
                if side:
                    held[side] = None
                elif held.get("right") == held_obj:
                    held["right"] = None
                elif held.get("left") == held_obj:
                    held["left"] = None
            continue

        if action == "put" and obj:
            held_obj = held.get("right") or held.get("left")
            if held_obj:
                facts.add(("inside", held_obj, obj))
            continue

        if action == "clean" and obj:
            facts.add(("not_dusty", obj))
        elif action == "open" and obj:
            facts.add(("open", obj))
            facts.discard(("closed", obj))
        elif action == "close" and obj:
            facts.add(("closed", obj))
            facts.discard(("open", obj))
        elif action in {"toggle_on", "switch_on", "turn_on"} and obj:
            facts.add(("on", obj))
            facts.discard(("off", obj))
        elif action in {"toggle_off", "switch_off", "turn_off"} and obj:
            facts.add(("off", obj))
            facts.discard(("on", obj))
        elif action == "slice" and obj:
            facts.add(("sliced", obj))
        elif action == "soak" and obj:
            facts.add(("soaked", obj))
        elif action == "freeze" and obj:
            facts.add(("frozen", obj))
            facts.discard(("not_frozen", obj))
        elif action == "unfreeze" and obj:
            facts.add(("not_frozen", obj))
            facts.discard(("frozen", obj))
        elif action == "cook" and obj:
            facts.add(("cooked", obj))
        elif action in {"walk", "navigate"} and obj:
            facts.add(("near", obj))
    return facts


def state_recall(predicted_state, gold_state):
    if not gold_state:
        return 1.0 if not predicted_state else 0.0
    return len(gold_state & predicted_state) / len(gold_state)

def score_predictions(name, predictions, examples, vocabulary, dfa=None, show=0):
    if not examples:
        return {
            "name": name,
            "examples": 0,
            "exact_sequence": 0.0,
            "token_accuracy": 0.0,
            "dfa_valid": 0.0,
            "gt_state_success": 0.0,
            "gt_state_recall": 0.0,
            "avg_pred_len": 0.0,
        }
    eos_label = vocabulary.eos_label
    exact = 0
    token_correct = 0
    token_total = 0
    dfa_valid = 0
    gt_state_success = 0
    gt_state_recall_total = 0.0
    pred_len_total = 0
    for idx, (sample, pred) in enumerate(zip(examples, predictions)):
        gold = gold_labels(sample, len(sample["target_action_labels"]))
        padded = pad_prediction(pred, gold, eos_label)
        pred_trimmed = trim_at_eos(pred, eos_label)
        gold_trimmed = trim_at_eos(gold, eos_label)
        exact += int(padded == gold)
        token_correct += sum(int(p == g) for p, g in zip(padded, gold))
        token_total += len(gold)
        dfa_valid += int(True if dfa is None else (dfa.accepts(pred_trimmed) or dfa.accepts(padded)))
        predicted_state = abstract_state_from_tokens(pred_trimmed, vocabulary)
        gold_state = abstract_state_from_tokens(gold_trimmed, vocabulary)
        recall = state_recall(predicted_state, gold_state)
        gt_state_success += int(recall >= 1.0)
        gt_state_recall_total += recall
        pred_len_total += len(pred)
        if idx < show:
            print()
            print(f"## {name} example {idx}: {sample.get('task_id', 'task')}")
            print(f"Instruction: {sample.get('natural_language_description') or sample.get('text')}")
            print(f"Gold: {labels_to_actions(gold_trimmed, vocabulary)}")
            print(f"Pred: {labels_to_actions(pred_trimmed, vocabulary)}")
            print(f"Gold state: {sorted(abstract_state_from_tokens(gold_trimmed, vocabulary))}")
            print(f"Pred state: {sorted(abstract_state_from_tokens(pred_trimmed, vocabulary))}")
    return {
        "name": name,
        "examples": len(examples),
        "exact_sequence": exact / len(examples),
        "token_accuracy": token_correct / token_total if token_total else 0.0,
        "dfa_valid": dfa_valid / len(examples),
        "gt_state_success": gt_state_success / len(examples),
        "gt_state_recall": gt_state_recall_total / len(examples),
        "avg_pred_len": pred_len_total / len(examples),
    }


def format_score(score):
    return (
        f"{score['name']}: examples={score['examples']} "
        f"exact_sequence={score['exact_sequence']:.4f} "
        f"token_accuracy={score['token_accuracy']:.4f} "
        f"dfa_valid={score['dfa_valid']:.4f} "
        f"gt_state_success={score['gt_state_success']:.4f} "
        f"gt_state_recall={score['gt_state_recall']:.4f} "
        f"avg_pred_len={score['avg_pred_len']:.2f}"
    )

def action_object_runtime_dfa(
    base_dfa,
    vocabulary,
    action_tokens,
    object_tokens,
    action_object_constraint_tokens=None,
    action_sequence_tokens=None,
):
    """Compose EAI action/object runtime constraints using shared DFA overlays."""
    from domiknows.generation import (
        compose_runtime_dfa,
        pending_token_allowed_set_overlay,
        token_class_sequence_overlay,
        token_set_sequence_overlay,
    )

    known_tokens = {vocabulary.token_for_label(label) for label in range(int(vocabulary.label_count))}
    action_tokens = [token for token in action_tokens if not isinstance(token, str) or token in known_tokens]
    object_tokens = [token for token in object_tokens if not isinstance(token, str) or token in known_tokens]
    action_sequence_tokens = [
        token
        for token in (action_sequence_tokens if action_sequence_tokens is not None else action_tokens)
        if not isinstance(token, str) or token in known_tokens
    ]
    if action_object_constraint_tokens:
        action_object_constraint_tokens = {
            action: [obj for obj in objects if not isinstance(obj, str) or obj in known_tokens]
            for action, objects in action_object_constraint_tokens.items()
            if not isinstance(action, str) or action in known_tokens
        }

    overlays = []
    if action_tokens and object_tokens:
        overlays.append(
            token_class_sequence_overlay(
                action_tokens,
                object_tokens,
                int(vocabulary.eos_label),
                vocabulary=vocabulary,
                name="eai_action_object_grammar",
            )
        )
    elif action_sequence_tokens:
        overlays.append(
            token_set_sequence_overlay(
                action_sequence_tokens,
                int(vocabulary.eos_label),
                vocabulary=vocabulary,
                name="eai_action_sequence_grammar",
            )
        )
    if action_object_constraint_tokens:
        overlays.append(
            pending_token_allowed_set_overlay(
                action_object_constraint_tokens,
                vocabulary=vocabulary,
                name="eai_action_object_compatibility",
            )
        )
    if not overlays:
        return base_dfa
    return compose_runtime_dfa(base_dfa, overlays)


def ActionObjectGrammarDFA(base_dfa, vocabulary, action_tokens, object_tokens):
    """Backward-compatible factory for the generic action/object grammar overlay."""
    return action_object_runtime_dfa(base_dfa, vocabulary, action_tokens, object_tokens)


def ActionObjectCompatibilityDFA(base_dfa, vocabulary, action_object_constraint_tokens):
    """Backward-compatible factory for the generic compatibility overlay."""
    from domiknows.generation import compose_runtime_dfa, pending_token_allowed_set_overlay

    return compose_runtime_dfa(
        base_dfa,
        [
            pending_token_allowed_set_overlay(
                action_object_constraint_tokens,
                vocabulary=vocabulary,
                name="eai_action_object_compatibility",
            )
        ],
    )


def NoImmediateEOSDFA(base_dfa, eos_label):
    """Backward-compatible factory that prevents empty plans by blocking EOS at step 0."""
    from domiknows.generation import RuntimeDFAOverlay, compose_runtime_dfa

    eos_label = int(eos_label)

    def step(emitted, label):
        label = int(label)
        if int(emitted) == 0 and label == eos_label:
            return None
        return 1

    overlay = RuntimeDFAOverlay(
        states=frozenset({0, 1}),
        alphabet=frozenset(base_dfa.alphabet),
        start_state=0,
        step_fn=step,
        accepting_states=frozenset({1}),
        name="no_immediate_eos",
    )
    return compose_runtime_dfa(base_dfa, [overlay])


def raw_qwen_predictions(args, examples, vocab):
    tokenizer, model = load_qwen(args.llm_backbone_path, args.device)
    predictions = []
    iterator = progress_bar(examples, total=len(examples), desc="raw Qwen")
    for sample in iterator:
        prompt = build_prompt(sample, vocab, args.max_steps)
        generated = generate_text(tokenizer, model, prompt, args)
        tokens = parse_generated_text(generated, vocab, args.max_steps)
        predictions.append(labels_to_ids(tokens, vocab))
    del model
    del tokenizer
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return predictions



def qwen_prompt_text(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def vocabulary_tokens(vocabulary):
    if hasattr(vocabulary, "label_count") and hasattr(vocabulary, "token_for_label"):
        return tuple(vocabulary.token_for_label(i) for i in range(int(vocabulary.label_count)))
    return tuple(vocabulary)


def qwen_label_logits(tokenizer, model, sample, vocabulary, prefix_labels, args, candidate_labels=None):
    """Score compact EAI labels under raw Qwen for the next decoding step."""
    tokens = vocabulary_tokens(vocabulary)
    label_count = len(tokens)
    device = args.device
    candidate_labels = list(range(label_count)) if candidate_labels is None else sorted(int(x) for x in candidate_labels)
    logits = torch.full((label_count,), -1e9, dtype=torch.float32, device=device)
    if not candidate_labels:
        return logits

    prefix_tokens = [tokens[int(label)] for label in prefix_labels if int(label) != int(getattr(vocabulary, "eos_label", 0))]
    prompt = build_prompt(sample, tokens, args.max_steps)
    if prefix_tokens:
        prompt = prompt + " " + " ".join(prefix_tokens)
    prompt_text = qwen_prompt_text(tokenizer, prompt)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])

    batch_size = max(1, int(getattr(args, "qwen_label_batch_size", 64)))
    for start in range(0, len(candidate_labels), batch_size):
        batch_labels = candidate_labels[start : start + batch_size]
        continuations = []
        continuation_ids = []
        for label in batch_labels:
            token = tokens[int(label)]
            text = " " + token
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if not ids:
                ids = tokenizer(token, add_special_tokens=False).input_ids
            continuation_ids.append(torch.tensor(ids, dtype=torch.long, device=device))
            continuations.append(text)
        max_cont = max(int(ids.numel()) for ids in continuation_ids)
        full_len = prompt_len + max_cont
        input_ids = torch.full(
            (len(batch_labels), full_len),
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, :prompt_len] = prompt_ids.expand(len(batch_labels), -1)
        attention_mask[:, :prompt_len] = 1
        lengths = []
        for row, ids in enumerate(continuation_ids):
            length = int(ids.numel())
            input_ids[row, prompt_len : prompt_len + length] = ids
            attention_mask[row, prompt_len : prompt_len + length] = 1
            lengths.append(length)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        for row, label in enumerate(batch_labels):
            total = torch.tensor(0.0, device=device)
            for offset in range(lengths[row]):
                target_pos = prompt_len + offset
                total = total + log_probs[row, target_pos - 1, input_ids[row, target_pos]]
            logits[int(label)] = total / max(1, lengths[row])
    return logits


def raw_qwen_label_model(args):
    tokenizer, model = load_qwen(args.llm_backbone_path, args.device)
    return tokenizer, model


def qwen_dfa_predictions(args, examples, dfa, vocabulary, desc="Qwen+DFA"):
    tokenizer, model = raw_qwen_label_model(args)
    predictions = []
    failures = 0
    iterator = progress_bar(examples, total=len(examples), desc=desc)
    for sample in iterator:
        state = dfa.start_state
        prefix = [int(vocabulary.eos_label)]
        labels = []
        for step in range(args.max_steps):
            try:
                allowed = set(dfa.allowed_tokens(state, remaining_steps=args.max_steps - step))
            except TypeError:
                allowed = set(dfa.allowed_tokens(state))
            if not allowed:
                failures += 1
                break
            step_logits = qwen_label_logits(tokenizer, model, sample, vocabulary, prefix, args, allowed)
            label = int(torch.argmax(step_logits).item())
            next_state = dfa.step(state, label)
            if next_state is None:
                failures += 1
                break
            labels.append(label)
            prefix.append(label)
            state = next_state
            if label == int(vocabulary.eos_label):
                break
        predictions.append(labels)
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(decode_failures=failures)
    del model
    del tokenizer
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    if failures:
        print(f"{desc}: decode_failures={failures}")
    return predictions


def build_no_training_graph(args, examples, constraint_mode="specific"):
    # Compile only the broad action->object constraint through DomiKnowS.  The
    # action-specific object compatibility is composed as a lightweight DFA
    # wrapper in build_no_training_dfa; compiling each compatibility rule as a
    # separate LC can make DFA construction stall on EAI.
    graph, bundle = create_generation_graph(
        max_steps=args.max_steps,
        vocab=generation_vocab_from_examples(examples),
        object_tokens=object_tokens_from_examples(examples),
        action_tokens=action_tokens_requiring_object_from_examples(examples),
        openable_object_tokens=None,
        action_object_constraint_tokens=None,
        enforce_action_object=True,
        enforce_action_object_constraints=False,
    )
    return graph, bundle


def build_no_training_dfa(args, examples, constraint_mode="specific"):
    from domiknows.generation import constraints_to_dfa_from_graph

    graph, bundle = build_no_training_graph(args, examples, constraint_mode=constraint_mode)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    dfa = action_object_runtime_dfa(
        dfa,
        bundle.vocabulary,
        action_tokens_requiring_object_from_examples(examples),
        object_tokens_from_examples(examples),
        action_object_constraint_tokens_from_examples(examples) if constraint_mode == "specific" else None,
        action_tokens_from_examples(examples),
    )
    return dfa, bundle.vocabulary, graph, bundle


def program_predictions(program, bundle, examples, max_steps, device, use_dfa=False, desc="program"):
    predictions = []
    failures = 0
    iterator = progress_bar(examples, total=len(examples), desc=desc)
    for sample in iterator:
        program.populate_one(sample, device=device)
        try:
            labels = dfa_constrained_sequence(program, bundle, sample, max_steps) if use_dfa else greedy_sequence(program, bundle, sample, max_steps)
        except ValueError as exc:
            if "DFA masking removed every label" not in str(exc):
                raise
            failures += 1
            invalid_label = int(bundle.vocabulary.label_count)
            labels = [invalid_label] * min(max_steps, len(sample["target_action_labels"]))
            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(decode_failures=failures)
        predictions.append(labels)
    if failures:
        print(f"{desc}: decode_failures={failures}")
    return predictions


def _align_hmm_to_vocabulary(data, vocabulary):
    alpha = np.asarray(data["alpha_exp"], dtype=np.float64)
    beta = np.asarray(data["beta"], dtype=np.float64)
    gamma = np.asarray(data["gamma"], dtype=np.float64)
    target_size = int(vocabulary.label_count)
    target_tokens = tuple(vocabulary.token_for_label(i) for i in range(target_size))

    if "tokens" not in data:
        return alpha[:target_size, :target_size], beta[:target_size, :target_size], gamma[:target_size]

    source_tokens = tuple(str(token) for token in data["tokens"])
    source_size = len(source_tokens)
    # Some older HMM files have one stale extra row/column beyond the saved
    # token list.  Trim arrays to the token list before remapping.
    alpha = alpha[:source_size, :source_size]
    beta = beta[:source_size, :source_size]
    gamma = gamma[:source_size]

    source_index = {token: idx for idx, token in enumerate(source_tokens)}
    missing = [token for token in target_tokens if token not in source_index]
    if not missing and source_tokens == target_tokens:
        return alpha, beta, gamma

    new_alpha = np.full((target_size, target_size), 1e-12, dtype=np.float64)
    new_beta = np.full((target_size, target_size), np.log(1e-12), dtype=np.float64)
    new_gamma = np.full((target_size,), np.log(1e-12), dtype=np.float64)

    for target_i, token_i in enumerate(target_tokens):
        source_i = source_index.get(token_i)
        if source_i is None:
            new_alpha[target_i, target_i] = 1.0
            new_beta[target_i, target_i] = 0.0
            continue
        new_gamma[target_i] = gamma[source_i]
        for target_j, token_j in enumerate(target_tokens):
            source_j = source_index.get(token_j)
            if source_j is None:
                continue
            new_alpha[target_i, target_j] = alpha[source_i, source_j]
            new_beta[target_i, target_j] = beta[source_i, source_j]

    new_alpha = new_alpha / new_alpha.sum(axis=1, keepdims=True)
    new_gamma = new_gamma - np.max(new_gamma)
    new_gamma = new_gamma - np.log(np.exp(new_gamma).sum())
    return new_alpha, new_beta, new_gamma


def load_hmm_generation_head(path, vocabulary, device):
    from domiknows.generation import HMMGenerationHead

    data = np.load(path, allow_pickle=True)
    alpha, beta, gamma = _align_hmm_to_vocabulary(data, vocabulary)
    label_count = int(vocabulary.label_count)
    head = HMMGenerationHead(
        label_count=label_count,
        state_count=int(gamma.shape[0]),
        pad_size=max(label_count, 1),
        label_to_token_id=tuple(range(label_count)),
        trainable=False,
    )
    with torch.no_grad():
        head.initial_logits.copy_(torch.as_tensor(gamma, dtype=head.initial_logits.dtype))
        head.transition_logits.copy_(torch.as_tensor(np.log(np.asarray(alpha, dtype=np.float64).clip(min=1e-30)), dtype=head.transition_logits.dtype))
        head.emission_logits.copy_(torch.as_tensor(beta, dtype=head.emission_logits.dtype))
    return head.to(device)


def _flat_ids(input_ids):
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2:
            return [int(x) for x in input_ids[0].detach().cpu().tolist()]
        return [int(x) for x in input_ids.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in input_ids]


def _mask_label_logits_local(logits, allowed_labels, fill_value=-1e9):
    if logits.dim() != 1:
        raise ValueError(f"expected compact label logits to be 1D, got shape {tuple(logits.shape)}")
    masked = torch.full_like(logits, fill_value)
    for label in allowed_labels:
        label = int(label)
        if 0 <= label < masked.numel():
            masked[label] = logits[label]
    if torch.all(masked <= fill_value / 2):
        raise ValueError("DFA mask removed all compact labels")
    return masked


def _select_label_local(logits, *, temperature, top_k, top_p, generator):
    if float(temperature) <= 0.0:
        label = int(torch.argmax(logits).item())
        log_probs = torch.log_softmax(logits, dim=-1)
        return label, float(log_probs[label].detach().item())
    probs = torch.softmax(logits / float(temperature), dim=-1)
    label = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    return label, float(torch.log(probs[label].clamp_min(torch.finfo(probs.dtype).tiny)).detach().item())


def _identity_emittable_labels(_model, vocabulary):
    return set(range(int(vocabulary.label_count)))


def _identity_token_id_for_label(_model, _vocabulary, label):
    return int(label)


def _invalid_prediction(bundle, sample, max_steps):
    invalid_label = int(bundle.vocabulary.label_count)
    return [invalid_label] * min(max_steps, len(sample["target_action_labels"]))


def _domiknows_hmm_dfa_predictions(args, dfa, bundle, generator, examples, desc="product HMM+DFA"):
    """Shared DomiKnowS generator + HMM + DFA decoding path.

    Both trained and no-training settings should differ only in which
    DomiKnowS generator/checkpoint is supplied. The decoder, HMM loading,
    DFA masking, and backend-logit integration stay identical.
    """
    if not args.hmm:
        raise ValueError("--hmm is required for HMM+DFA decoding")
    from domiknows.generation.applications.hybrid import HybridController

    scorer_head = load_hmm_generation_head(args.hmm, bundle.vocabulary, args.device)
    controller = HybridController(
        dfa=dfa,
        vocabulary=bundle.vocabulary,
        generator=generator,
        scorer_head=scorer_head,
        tokenizer=None,
    )
    predictions = []
    failures = 0
    iterator = progress_bar(examples, total=len(examples), desc=desc)
    for sample in iterator:
        prompt = torch.tensor([[int(bundle.vocabulary.eos_label)]], dtype=torch.long, device=args.device)
        results = controller.decode_hmm_dfa(
            prompt,
            search=args.hmm_search,
            num_return_sequences=1,
            beam_size=args.hmm_beam_size,
            max_new_tokens=args.max_steps,
            keep_rejected=args.hmm_keep_rejected,
            temperature=0.0 if args.hmm_search != "sample" else 1.0,
            hmm_weight=args.hmm_weight,
            hf_weight=args.hmm_hf_weight,
            lookahead_weight=args.hmm_lookahead_weight,
            lookahead_max_steps=args.hmm_lookahead_max_steps,
        )
        if results:
            predictions.append(results[0].labels)
        else:
            failures += 1
            predictions.append(_invalid_prediction(bundle, sample, args.max_steps))
            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(decode_failures=failures)
    if failures:
        print(f"{desc}: decode_failures={failures}")
    return predictions



def hmm_dfa_predictions(args, program, bundle, examples):
    from domiknows.generation import constraints_to_dfa_from_graph

    dfa = constraints_to_dfa_from_graph(program.graph, bundle)
    return _domiknows_hmm_dfa_predictions(
        args,
        dfa,
        bundle,
        program.autoregressive_head,
        examples,
        desc="4 product HMM+DFA",
    )

def selected_settings(args):
    alias = {"raw_dfa": "0", "raw": "1", "domiknows": "2", "dfa": "3", "hmm": "4"}
    settings = {alias.get(item, item) for item in args.settings}
    if "nt_suite" in settings:
        settings.discard("nt_suite")
        settings.update({"nt_domiknows", "nt_dfa", "nt_hmm_dfa"})
    if args.skip_raw_qwen:
        settings.discard("0")
        settings.discard("1")
        settings.discard("nt_domiknows")
        settings.discard("nt_dfa")
    return settings


def main():
    args = parse_args()
    settings = selected_settings(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_examples = load_examples(args, args.device)
    if not all_examples:
        raise ValueError("No EAI examples were loaded for evaluation.")
    examples = select_eval_examples(all_examples, args.eval_split, args.dev_fraction, args.eval_limit)
    if not examples:
        raise ValueError(f"No EAI examples selected for eval_split={args.eval_split!r}.")
    vocab = tuple(generation_vocab_from_examples(all_examples))

    raw_preds = None
    raw_needed = bool(settings & {"0", "1", "nt_domiknows"})
    if raw_needed:
        raw_preds = raw_qwen_predictions(args, examples, vocab)

    no_training_artifacts = {}
    nt_program = nt_bundle = None
    if "nt_hmm_dfa" in settings:
        nt_program, nt_bundle = build_trainable_program(args, all_examples, args.device)

    # Only build DFA artifacts for settings that actually need DFA decoding or
    # DFA validity scoring.  Raw no-training DomiKnowS/Qwen scoring can use the
    # lightweight RawVocab below and should not pay the ~20s DFA build cost.
    if settings & {"0", "nt_dfa", "nt_hmm_dfa"}:
        for constraint_mode in args.constraint_modes:
            if "nt_hmm_dfa" in settings:
                from domiknows.generation import constraints_to_dfa_from_graph as _constraints_to_dfa_from_graph

                dfa_i = _constraints_to_dfa_from_graph(nt_program.graph, nt_bundle)
                vocab_i = nt_bundle.vocabulary
                graph_i = nt_program.graph
                bundle_i = nt_bundle
            else:
                dfa_i, vocab_i, graph_i, bundle_i = build_no_training_dfa(
                    args, all_examples, constraint_mode=constraint_mode
                )
            artifact = {
                "dfa": dfa_i,
                "vocabulary": vocab_i,
                "graph": graph_i,
                "bundle": bundle_i,
            }
            if "nt_hmm_dfa" in settings:
                artifact["program"] = nt_program
            no_training_artifacts[constraint_mode] = artifact

    # Build from all loaded examples so graph/vocabulary matches the trained checkpoint,
    # then score only the selected evaluation examples.
    program = bundle = None
    if settings & {"2", "3", "4"}:
        if not args.model:
            raise ValueError("--model is required for settings 2, 3, and 4.")
        program, bundle = load_trained_program(args, all_examples, args.device)
    elif raw_preds is None and not (settings & {"nt_hmm_dfa"}):
        raise ValueError("No evaluation settings selected.")
    from domiknows.generation import constraints_to_dfa_from_graph

    dfa = constraints_to_dfa_from_graph(program.graph, bundle) if program is not None else None
    score_vocab = bundle.vocabulary if bundle is not None else type("RawVocab", (), {
        "eos_label": vocab.index(EOS_TOKEN),
        "label_count": len(vocab),
        "token_for_label": lambda self, idx: vocab[idx],
        "other_token": "other",
        "eos_token": EOS_TOKEN,
    })()
    scores = []

    if "0" in settings:
        artifact = no_training_artifacts[args.constraint_modes[-1]]
        scores.append(score_predictions("0 Qwen-1.5B zero-shot + DomiKnowS DFA (no training)", raw_preds, examples, artifact["vocabulary"], dfa=artifact["dfa"], show=args.show))

    for constraint_mode in args.constraint_modes:
        artifact = no_training_artifacts.get(constraint_mode)
        label = f"{constraint_mode} constraints"
        if "nt_domiknows" in settings:
            scores.append(score_predictions(f"NT DomiKnowS ({label}, original Qwen)", raw_preds, examples, score_vocab, dfa=None, show=args.show))
        if artifact is None:
            continue
        if "nt_dfa" in settings:
            dfa_preds = qwen_dfa_predictions(
                args,
                examples,
                artifact["dfa"],
                artifact["vocabulary"],
                desc=f"NT Qwen+DFA {constraint_mode}",
            )
            scores.append(score_predictions(f"NT DomiKnowS + DFA ({label}, original Qwen constrained)", dfa_preds, examples, artifact["vocabulary"], dfa=artifact["dfa"], show=args.show))
        if "nt_hmm_dfa" in settings:
            hmm_preds = hmm_dfa_predictions(args, artifact["program"], artifact["bundle"], examples)
            scores.append(score_predictions(f"NT DomiKnowS + HMM + DFA ({label}, distilled original Qwen HMM)", hmm_preds, examples, artifact["vocabulary"], dfa=artifact["dfa"], show=args.show))

    if "1" in settings:
        scores.append(score_predictions("1 Qwen-1.5B zero-shot", raw_preds, examples, score_vocab, dfa=dfa, show=args.show))

    if "2" in settings:
        preds = program_predictions(program, bundle, examples, args.max_steps, args.device, use_dfa=False, desc="2 DomiKnowS+Qwen")
        scores.append(score_predictions("2 DomiKnowS + Qwen-1.5B", preds, examples, bundle.vocabulary, dfa=dfa, show=args.show))

    if "3" in settings:
        dfa_preds = program_predictions(program, bundle, examples, args.max_steps, args.device, use_dfa=True, desc="3 DomiKnowS+Qwen+DFA")
        scores.append(score_predictions("3 DomiKnowS + Qwen-1.5B + DFA", dfa_preds, examples, bundle.vocabulary, dfa=dfa, show=args.show))

    if "4" in settings:
        hmm_preds = hmm_dfa_predictions(args, program, bundle, examples)
        scores.append(score_predictions("4 DomiKnowS + Qwen-1.5B + DFA + HMM", hmm_preds, examples, bundle.vocabulary, dfa=dfa, show=args.show))

    lines = [
        "EAI evaluation settings",
        f"dataset={args.dataset} eval_split={args.eval_split} examples={len(examples)} loaded_examples={len(all_examples)} max_steps={args.max_steps}",
        f"model={args.model}",
        f"hmm={args.hmm}",
        f"settings={','.join(sorted(settings))}",
        f"constraint_modes={','.join(args.constraint_modes)}",
        "",
    ]
    lines.extend(format_score(score) for score in scores)
    lines.append("")
    lines.append(json.dumps(scores, indent=2))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    for line in lines[:4] + [format_score(score) for score in scores]:
        print(line)
    print(f"saved_results={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
