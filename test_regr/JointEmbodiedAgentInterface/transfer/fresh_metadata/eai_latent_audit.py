"""Audit goal-success headroom in a fixed beam of legal EAI alternatives."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
from test_regr.EmbodiedAgentInterface.observed_reward import (
    prepare_observed_eai_goal,
    score_observed_eai_goal,
)

from ...utils.planner_context import planner_context
from ...utils.structured_inference import LabelSequenceCandidate, constrained_dfa_beam_search
from ..anonymous_grounding.planner_binding.verification_protocol import exact_policy_search
from ..protocol import digest, freeze, read, sha
from ..repair.common import build_runtime, fresh_planner
from ..repair.protocol import verify
from .train import FORMAT

BEAM_WIDTH = 8
FORWARD_BATCH = 4


def input_and_goal(item, world):
    public = {key: item[key] for key in ("tl_goal", "transition_model")}
    goal = prepare_observed_eai_goal(public, world_bundle=world)
    actor = goal.aliases.get("character", "character")
    if actor != "character":
        if not {"agent", "character"} & goal.types.get(actor, set()):
            raise ValueError("indexed EAI agent lacks an observed actor type")
        # VirtualHome names the sole actor in SimpleTL but declares its class in
        # PDDL. The executor records hand effects on the canonical actor.
        aliases = {key: ("character" if value == actor else value) for key, value in goal.aliases.items()}
        aliases[actor] = "character"
        types = dict(goal.types)
        types["character"] = types["character"] | types.pop(actor)

        def normalize(facts):
            return frozenset((fact[0], *("character" if name == actor else name for name in fact[1:]))
                             for fact in facts)

        goal = replace(goal, aliases=aliases, types=types, positive=normalize(goal.positive),
                       negative=normalize(goal.negative))
    prompt = planner_context("eai", {
        key: item[key] for key in ("causal_prompt_text", "instruction_text", "instruction", "goal", "tl_goal")
        if key in item
    })
    context = {key: prompt[key] for key in ("instruction", "goal")}
    context.update(goal.policy_context())
    return context, goal


def load_frozen(config, protocol, runtime):
    planner = fresh_planner(config, runtime, "joint")
    snapshot = torch.load(protocol["checkpoint"], map_location="cpu", weights_only=True)
    if (snapshot["format"], snapshot["epoch"], snapshot["settings_sha256"]) != (
        FORMAT, 5, protocol["checkpoint_settings_sha256"]
    ):
        raise ValueError("frozen planner does not match the E5 audit")
    expected = {
        name for name, _ in planner.named_parameters()
        if "lora_" in name or name.startswith("label_heads.")
    }
    if set(snapshot["planner"]) != expected:
        raise ValueError("checkpoint planner components differ")
    loaded = planner.load_state_dict(snapshot["planner"], strict=False)
    if loaded.unexpected_keys or expected.intersection(loaded.missing_keys):
        raise ValueError("checkpoint planner weights did not load")
    planner.requires_grad_(False)
    planner.eval()
    planner.free_decoding = False
    return planner


def next_logits(planner, vocabulary, context, prefixes, *, prefix_cache=None):
    """Batch text-only EAI prefixes with the planner's own label boundaries."""
    prepared = [
        planner._prepare_inputs_and_boundaries(
            "eai", context, [vocabulary.token_for_label(label) for label in prefix]
        )
        for prefix in prefixes
    ]
    keys = prepared[0][0].keys()
    if any(inputs.keys() != keys or any(value.ndim != 2 or value.shape[0] != 1 for value in inputs.values())
           for inputs, _ in prepared):
        raise ValueError("unexpected EAI processor inputs for batched search")
    groups = defaultdict(list)
    for index, (inputs, _) in enumerate(prepared):
        groups[inputs["input_ids"].shape[-1]].append(index)
    logits = {}
    for length, group in groups.items():
        for start in range(0, len(group), FORWARD_BATCH):
            indices = group[start:start + FORWARD_BATCH]
            batch = {key: torch.cat([prepared[i][0][key] for i in indices]) for key in keys}
            attention = batch.get("attention_mask")
            if attention is not None and bool(attention.all()):
                # Equal-length, unpadded rows can use the causal attention fast path.
                del batch["attention_mask"]
            prefix_length = 0
            cache = None
            if prefix_cache is not None and "attention_mask" not in batch:
                common_ids, initial_cache = prefix_cache
                common_length = common_ids.shape[-1]
                if (length > common_length and torch.equal(batch["input_ids"][:, :common_length],
                                                           common_ids.expand(len(indices), -1))
                        and all(prepared[i][1][-1] >= common_length for i in indices)):
                    prefix_length = common_length
                    batch = {key: value[:, prefix_length:] for key, value in batch.items()}
                    cache = deepcopy(initial_cache)
                    if len(indices) > 1:
                        cache.batch_repeat_interleave(len(indices))
            model_inputs = planner._inputs_to_model_device(batch)
            # EAI reads the graph-label head, so the native LM vocabulary logits
            # are needed only for HF's output contract.
            model_inputs["logits_to_keep"] = 1
            if cache is not None:
                model_inputs["past_key_values"] = cache
                positions = torch.arange(prefix_length, length, device=model_inputs["input_ids"].device)
                model_inputs["position_ids"] = positions[None].expand(len(indices), -1)
            result = planner._sampling_forward(model_inputs, use_cache=False)
            hidden = result.hidden_states[-1]
            positions = torch.tensor([min(prepared[i][1][-1], length - 1) - prefix_length for i in indices],
                                     device=hidden.device)
            batch_rows = torch.arange(len(indices), device=hidden.device)
            scores = planner.label_heads["eai"](hidden[batch_rows, positions].float().to(planner.device))
            logits.update(zip(indices, scores))
            del result, hidden
    return torch.stack([logits[i] for i in range(len(prepared))])


def prepare_prefix_cache(planner, context):
    """Prefill the stable public text once without trusting BPE prefix stability."""
    inputs, _ = planner._prepare_inputs_and_boundaries("eai", context, ())
    source = inputs["input_ids"][:, :-1]
    if source.shape[-1] == 0 or any(
        value.ndim != 2 or value.shape != inputs["input_ids"].shape for value in inputs.values()
    ):
        return None
    prefix = {key: value[:, :-1] for key, value in inputs.items()}
    if "attention_mask" in prefix and bool(prefix["attention_mask"].all()):
        del prefix["attention_mask"]
    model_inputs = planner._inputs_to_model_device(prefix)
    model_inputs["logits_to_keep"] = 1
    output = planner._sampling_forward(model_inputs, use_cache=True)
    cache = output.past_key_values
    if cache is None or not hasattr(cache, "batch_repeat_interleave"):
        return None
    return source, cache


def candidate_record(candidate, policy, goal, vocabulary, world):
    labels = candidate.labels
    if not labels or labels[-1] != vocabulary.eos_label or not policy.accepts(labels):
        raise ValueError("candidate fails accepting EOS or compiled DFA legality")
    score = score_observed_eai_goal(labels, goal, vocabulary, world_bundle=world)
    return {
        "labels": list(labels), "tokens": [vocabulary.token_for_label(i) for i in labels],
        "log_probability": candidate.log_probability, "goal_success": score["is_success"],
        "goal_recall": score["recall"],
    }


@torch.inference_mode()
def audit_case(planner, runtime, item, index, *, check_batch, shared_reranker=None):
    started = time.monotonic()
    world, vocabulary = runtime.world.eai, runtime.eai_vocabulary
    context, goal = input_and_goal(item, world)
    result = {
        "index": index, "task_id": str(item.get("task_id", "")), "domain": goal.domain,
        "observation_sha256": digest(context), "unresolved": list(goal.unresolved), "greedy": None,
        "beam": [], "search_errors": {}, "exact_search_retries": [],
    }
    if goal.unresolved:
        return result
    retries = result["exact_search_retries"]
    max_steps = runtime.max_eai_steps
    with runtime.domain_scope("eai"):
        policy = runtime.dfa_for("eai", context)
        with exact_policy_search(policy, retries):
            try:
                ids, probability = planner.sample_labels(
                    "eai", context, policy, max_steps=max_steps, deterministic=True
                )
                best = LabelSequenceCandidate(tuple(ids), float(probability))
                result["greedy"] = candidate_record(best, policy, goal, vocabulary, world)
            except SemanticSearchLimitExceeded as exc:
                result["search_errors"]["greedy"] = type(exc).__name__
            except RuntimeError as exc:
                if not str(exc).startswith(("eai planner DFA has no productive", "eai planner did not terminate")):
                    raise
                result["search_errors"]["greedy"] = str(exc)

            prefix_cache = prepare_prefix_cache(planner, context)

            if prefix_cache is not None and result["greedy"] and len(result["greedy"]["labels"]) > 2:
                prefix = tuple(result["greedy"]["labels"][:2])
                pair = [(), prefix]
                singles = next_logits(planner, vocabulary, context, pair)
                together = next_logits(planner, vocabulary, context, pair, prefix_cache=prefix_cache)
                if together.shape != singles.shape or not torch.isfinite(together).all():
                    raise ValueError("invalid batched EAI beam logits")
                delta = float((together - singles).abs().max())
                if delta > .25:
                    prefix_cache = None  # preserve full-prefix scoring if the cache interface diverges
                if check_batch or prefix_cache is None:
                    result["batch_alignment"] = {"max_absolute_logit_difference": delta,
                        "unmasked_top_label_matches": int((together.argmax(-1) == singles.argmax(-1)).sum())}
            result["shared_prefix_cache"] = prefix_cache is not None

            def batch_logits(prefixes):
                return next_logits(planner, vocabulary, context, prefixes, prefix_cache=prefix_cache)
            try:
                beam = constrained_dfa_beam_search(batch_logits, policy, vocabulary.eos_label,
                                                    max_steps, beam_width=BEAM_WIDTH)
                if shared_reranker is not None:
                    from ..shared_reranker.features import eai_public_goal
                    from ..shared_reranker.head import rerank_candidates

                    ranked = rerank_candidates(shared_reranker, beam, policy, vocabulary,
                                               goal=eai_public_goal(goal, world))
                    result["shared_reranker"] = [asdict(entry) for entry in ranked]
                result["beam"] = [candidate_record(x, policy, goal, vocabulary, world) for x in beam]
            except SemanticSearchLimitExceeded as exc:
                result["search_errors"]["beam"] = type(exc).__name__
            except RuntimeError as exc:
                if str(exc) != "DFA beam search produced no complete accepting path":
                    raise
                result["search_errors"]["beam"] = str(exc)
    result["seconds"] = time.monotonic() - started
    return result


def summarize(rows):
    def success(candidate):
        return bool(candidate and candidate["goal_success"])

    def measure(scope):
        scored = [row for row in scope if not row["unresolved"]]
        metrics = {
            "episodes": len(scope), "groundable": len(scored),
            "greedy_success": sum(success(row["greedy"]) for row in scored),
            "beam_first_success": sum(success(next(iter(row["beam"]), None)) for row in scored),
            "any_in_greedy_or_beam_success": sum(
                success(row["greedy"]) or any(map(success, row["beam"])) for row in scored
            ),
            "rescues_over_greedy": [row["index"] for row in scored
                                   if row["greedy"] and not success(row["greedy"])
                                   and any(map(success, row["beam"]))],
            "beam_rescues_without_terminated_greedy": [
                row["index"] for row in scored if row["greedy"] is None and any(map(success, row["beam"]))
            ],
            "completed_greedy": sum(row["greedy"] is not None for row in scored),
            "completed_beam": sum(bool(row["beam"]) for row in scored),
            "beam_candidate_counts": dict(Counter(len(row["beam"]) for row in scored)),
            "exact_search_exhausted": sum(
                "SemanticSearchLimitExceeded" in row["search_errors"].values() for row in scored
            ),
        }
        if any("shared_reranker" in row for row in scope):
            metrics["shared_reranker_success"] = sum(
                success(row["beam"][row["shared_reranker"][0]["index"]])
                for row in scored if row.get("shared_reranker")
            )
            metrics["shared_reranker_completed"] = sum(bool(row.get("shared_reranker")) for row in scored)
        return metrics

    return {
        "overall": measure(rows),
        "by_domain": {name: measure([row for row in rows if row["domain"] == name])
                      for name in sorted({row["domain"] for row in rows})},
        "unresolved_indices": [row["index"] for row in rows if row["unresolved"]],
        "beam_width": BEAM_WIDTH,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--shared-reranker", type=Path)
    args = parser.parse_args()
    root = args.root
    protocol, config = read(root / "protocol.json"), read(root / "config.json")
    verify(config)
    for file, expected in protocol["input_hashes"].items():
        if sha(file) != expected:
            raise ValueError(f"frozen audit input changed: {file}")
    if args.shards != protocol["shards"] or BEAM_WIDTH != protocol["beam_width"]:
        raise ValueError("frozen beam or sharding configuration differs")
    reranker_hash = sha(args.shared_reranker) if args.shared_reranker else None
    if protocol.get("shared_reranker_sha256") != reranker_hash:
        raise ValueError("shared reranker differs from the frozen audit protocol")
    reused = read(Path(protocol["completed_cases"])) if "completed_cases" in protocol else []
    if reranker_hash is not None and reused:
        raise ValueError("shared reranker audit requires a fresh complete candidate ranking")
    prior_indices = {row["index"] for row in reused}
    if len(prior_indices) != len(reused) or not prior_indices <= set(range(88)):
        raise ValueError("previously completed case identities differ")
    pending = [index for index in range(88) if index not in prior_indices]
    if args.collect:
        rows = list(reused)
        for shard in range(args.shards):
            source = root / f"shard{shard}"
            coverage = read(source / "complete.json")
            with (source / "cases.jsonl").open() as stream:
                cases = [json.loads(line) for line in stream]
            expected = pending[shard::args.shards]
            if (coverage["indices"] != expected or [row["index"] for row in cases] != expected
                    or coverage.get("shared_reranker_sha256") != reranker_hash):
                raise ValueError("unmatched canonical case coverage")
            rows.extend(cases)
        if len(rows) != 88 or len({row["index"] for row in rows}) != 88:
            raise ValueError("expected exactly 88 canonical validation rows")
        results = summarize(sorted(rows, key=lambda row: row["index"]))
        results["reused_cases"] = len(reused)
        results["memory_bounded_cases"] = len(pending)
        if reranker_hash is not None:
            results["shared_reranker_sha256"] = reranker_hash
        freeze(root / "results.json", results)
        print(json.dumps(results), flush=True)
        return
    if args.shard is None or not 0 <= args.shard < args.shards:
        raise ValueError("shard index missing or out of range")
    folder = root / f"shard{args.shard}"
    folder.mkdir(exist_ok=True)
    _, (runtime, _, valid, _, _) = build_runtime(config, folder / "runtime")
    if len(valid) != 88 or sha(folder / "runtime/eai_identities.json") != protocol["identities_sha256"]:
        raise ValueError("canonical EAI validation identities differ")
    planner = load_frozen(config, protocol, runtime)
    shared_reranker = None
    if args.shared_reranker is not None:
        from ..shared_reranker.head import SharedPlanReranker

        shared_reranker = SharedPlanReranker.load(args.shared_reranker)
    rows = []
    checked = False
    with (folder / "cases.jsonl").open("x") as stream:
        for index in pending[args.shard::args.shards]:
            row = audit_case(planner, runtime, valid[index], index, check_batch=not checked,
                             shared_reranker=shared_reranker)
            checked |= "batch_alignment" in row
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"[eai-latent-audit] shard {args.shard} case {index} complete", flush=True)
    completion = {"indices": [x["index"] for x in rows], "batch_alignment_checked": checked}
    if reranker_hash is not None:
        completion["shared_reranker_sha256"] = reranker_hash
    freeze(folder / "complete.json", completion)


if __name__ == "__main__":
    main()
