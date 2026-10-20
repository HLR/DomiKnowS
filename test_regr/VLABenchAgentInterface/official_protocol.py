"""Standard VLABench VLM-planning protocol (OpenMOSS/VLABench ``scripts/evaluate_vlm.py``).

Inputs, prompt, few-shot sampling, answer parsing and scoring follow the pinned
upstream revision so results can sit next to published VLM rows:

* every episode of every task directory is scored (no train/validation/test split);
* the model sees only the unlabeled image, the number-labeled image and the
  instruction, under the unmodified ``eval_vlm_en.txt`` prompt. No entity names;
* the answer is the first ``json`` fence of the reply; anything else is a
  ``format_error`` and scores zero;
* dependencies depend only on the task (Sequential / Seq-independent);
* ``total_score`` is ``0.4*SR + 0.4*PR`` (reported as ``legacy_score`` elsewhere),
  and a dimension score is its mean over the dimension's episodes.

Upstream's ``dim2task`` does not list every task of the released dataset, and its
script would raise on the unlisted ones. They are excluded from the official
dimension table here and reported separately under the extended mapping.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .planning_metrics import (
        _CONFIG, DIMENSIONS, OFFICIAL_TASK_DIMENSIONS, PlanningMetrics, default_dependency,
        score_planning_metrics,
    )
    from .world_graph import canonicalize_plan
except ImportError:  # pragma: no cover - flat-module execution
    from planning_metrics import (
        _CONFIG, DIMENSIONS, OFFICIAL_TASK_DIMENSIONS, PlanningMetrics, default_dependency,
        score_planning_metrics,
    )
    from world_graph import canonicalize_plan

PROTOCOL = "vlabench-vlm-official-v1"
PROMPT_PATH = Path(__file__).with_name("official_vlm_prompt_en.txt")
# configs/prompt/eval_vlm_en.txt at the revision pinned in planning_benchmark.json.
PROMPT_SHA256 = "e0dbd2973e73e6d091e765bd04f3ae172c22d415413cc56bb421abc36539e765"
ZERO_SCORES = {"SR": 0.0, "PR": 0.0, "SPR": 0.0, "PM": 0.0, "score": 0.0, "legacy_score": 0.0}


def load_official_prompt() -> str:
    data = PROMPT_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != PROMPT_SHA256:
        raise ValueError(f"{PROMPT_PATH.name} differs from the pinned upstream prompt; "
                         "official-protocol results would not be comparable")
    return data.decode("utf-8")


def official_dependency(task: str) -> str:
    """Upstream decides dependencies from the task name alone."""
    return default_dependency(task)


@dataclass(frozen=True)
class OfficialExample:
    episode_id: str
    task: str
    dimension: str | None  # None: upstream dim2task does not list the task
    instruction: str
    image_path: Path
    mask_path: Path
    reference_payload: Any  # the output JSON exactly as stored (few-shot answers print it)
    reference: tuple[dict[str, Any], ...]
    reference_sha256: str


def load_official_examples(root: str | Path, *, tasks: Sequence[str] | None = None,
                           limit: int | None = None) -> list[OfficialExample]:
    """Read ``<root>/**/<task>/example<N>/{input,output}`` without touching env configs."""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"official VLM evaluation data does not exist: {root}")
    wanted = set(tasks) if tasks else None
    examples: list[OfficialExample] = []
    for instruction_path in sorted(root.rglob("instruction.txt")):
        input_dir, example_dir = instruction_path.parent, instruction_path.parent.parent
        if input_dir.name != "input" or not example_dir.name.startswith("example"):
            continue
        task = example_dir.parent.name
        if wanted is not None and task not in wanted:
            continue
        outputs = sorted((example_dir / "output").glob("*.json"))
        image, mask = input_dir / "input.png", input_dir / "input_mask.png"
        if not outputs or not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"incomplete official episode: {example_dir}")
        raw = outputs[0].read_bytes()
        payload = json.loads(raw)
        plan = payload.get("skill_sequence", payload.get("operation_sequence")) if isinstance(payload, Mapping) else payload
        examples.append(OfficialExample(
            episode_id=example_dir.relative_to(root).as_posix(), task=task,
            dimension=OFFICIAL_TASK_DIMENSIONS.get(task),
            instruction=instruction_path.read_text(encoding="utf-8"),
            image_path=image, mask_path=mask, reference_payload=payload,
            reference=tuple(canonicalize_plan(plan)), reference_sha256=hashlib.sha256(raw).hexdigest(),
        ))
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no official VLM episodes found under {root}")
    return examples


def dataset_fingerprint(examples: Sequence[OfficialExample]) -> str:
    """Hash of episode ids, instructions and reference files (images are not hashed)."""
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.episode_id):
        digest.update(f"{example.episode_id}\t{hashlib.sha256(example.instruction.encode()).hexdigest()}"
                      f"\t{example.reference_sha256}\n".encode())
    return digest.hexdigest()


def sample_few_shots(examples: Sequence[OfficialExample], target: OfficialExample, count: int,
                     seed: int) -> list[OfficialExample]:
    """Mimic upstream: a random task, then a random episode of it, never the target itself."""
    if count <= 0:
        return []
    by_task: dict[str, list[OfficialExample]] = defaultdict(list)
    for example in examples:
        by_task[example.task].append(example)
    if sum(len(group) for group in by_task.values()) < 2:
        raise ValueError("few-shot evaluation needs at least two episodes")
    rng = random.Random(f"{seed}:{target.episode_id}")
    names = sorted(by_task)
    shots = []
    for _ in range(count):
        while True:
            shot = rng.choice(by_task[rng.choice(names)])
            if shot.episode_id != target.episode_id:
                break
        shots.append(shot)
    return shots


def build_ti_list(prompt: str, example: OfficialExample, shots: Sequence[OfficialExample] = (),
                  with_cot: bool = False) -> list[list[Any]]:
    """Upstream ``get_ti_list`` (English): ordered ``["text"|"image", value]`` items."""
    items: list[list[Any]] = [["text", prompt]]
    for number, shot in enumerate(shots):
        label = f"Example {number}"
        items += [["text", f"{label} input picture:"], ["image", shot.image_path],
                  ["text", f"{label} input picture with numbered tags"], ["image", shot.mask_path],
                  ["text", f"{label} language instruction:"], ["text", shot.instruction],
                  ["text", f"{label} output skill sequence"], ["text", json.dumps(shot.reference_payload)]]
    items += [["text", "Input picture"], ["image", example.image_path],
              ["text", "Input picture with numbered tags"], ["image", example.mask_path],
              ["text", "Language instruction:"], ["text", example.instruction],
              ["text", "Please give the output skill sequence"]]
    if with_cot:
        items.append(["text", "Please analyze the problem step by step and give the answer"])
    return items


def parse_official_output(text: str) -> tuple[bool, Any]:
    """First ```json fence, as upstream wrappers do; ``(False, None)`` is a format error."""
    try:
        return True, json.loads(text.split("```json")[1].split("```")[0])
    except Exception:  # noqa: BLE001 - upstream treats every failure as a format error
        return False, None


def _strict_plan(value: Any, coerce_ids: bool = False) -> list[dict[str, Any]]:
    """Validate like upstream (no name normalization) and reject what upstream cannot score.

    ``coerce_ids`` (supplementary only) reads digit strings such as ``"6"`` as ``6``.
    """
    if not isinstance(value, list):
        raise ValueError("skill_sequence must be a list")
    plan = []
    for skill in value:
        if not isinstance(skill, dict) or "name" not in skill or not isinstance(skill.get("params"), dict):
            raise ValueError("every skill needs a name and a params object")
        name = skill["name"]
        if not isinstance(name, str):
            raise ValueError("skill names must be strings")
        params = {}
        for key, argument in skill["params"].items():
            if isinstance(argument, float) and argument.is_integer():
                argument = int(argument)  # 2.0 == 2 under upstream's equality
            elif coerce_ids and isinstance(argument, str) and argument.strip().isdigit():
                argument = int(argument.strip())
            params[key] = argument
        # canonicalize_plan lowercases names; upstream compares them exactly, so a
        # non-canonical spelling is made unmatchable instead of silently repaired.
        if name != name.strip().lower().replace("-", "_"):
            name = "\0" + name
        plan.append({"name": name, "params": params})
    return plan


def score_official_answer(task: str, reference: Sequence[Mapping[str, Any]], ok: bool, sequence: Any,
                          coerce_ids: bool = False) -> dict[str, float]:
    """Per-episode scores; ``total_score`` of upstream is the ``legacy_score`` entry."""
    if not ok:
        return dict(ZERO_SCORES)
    try:
        plan = _strict_plan(sequence, coerce_ids)
    except ValueError:
        return dict(ZERO_SCORES)
    return score_planning_metrics(plan, list(reference), official_dependency(task))


def official_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Official dimension table (upstream-mapped tasks) plus the extended-mapping view."""
    official, extended = PlanningMetrics(), PlanningMetrics()
    per_task: dict[str, dict[str, float]] = {}
    unmapped: dict[str, int] = defaultdict(int)
    for row in rows:
        task, scores = row["task"], row["scores"]
        extended.update(scores, task=task)
        if task in OFFICIAL_TASK_DIMENSIONS:
            official.update(scores, task=task, dimension=OFFICIAL_TASK_DIMENSIONS[task])
        else:
            unmapped[task] += 1
        entry = per_task.setdefault(task, {"examples": 0, "total_score": 0.0})
        entry["examples"] += 1
        entry["total_score"] += scores["legacy_score"]
    official_result = official.result()
    table = {dimension: official_result["dimensions"][dimension]["legacy_score"] for dimension in DIMENSIONS}
    return {
        # Upstream ``six_dim_result`` total_score per dimension (0-100). Primary comparable output.
        "official_total_score_by_dimension": table,
        "official_dimension_examples": {d: official_result["dimensions"][d]["examples"] for d in DIMENSIONS},
        "official_dimensions": official_result["dimensions"],
        # Not an upstream quantity: unweighted mean of the six values above.
        "dimension_mean_of_total_score": official_result["legacy_Macro"],
        "official_mapped_examples": official_result["examples"],
        "official_mapped_overall": official_result["overall"],
        "unmapped_examples": dict(sorted(unmapped.items())),
        "extended_mapping": extended.result(),
        "per_task_total_score": {task: {"examples": item["examples"], "total_score": item["total_score"] / item["examples"]}
                                 for task, item in sorted(per_task.items())},
    }


def _read_lines(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["episode_id"]: row for row in rows}


def run_official_protocol(
    examples: Sequence[OfficialExample],
    generate: Callable[[list[list[Any]]], str],
    *,
    output_dir: str | Path,
    model_name: str,
    few_shot_num: int = 0,
    with_cot: bool = False,
    seed: int = 42,
    standard: bool = True,
    decoding: str = "greedy",
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Generate (resumably), score and summarize. ``generate`` maps a ti-list to reply text."""
    prompt = load_official_prompt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"protocol": PROTOCOL, "model": model_name, "few_shot_num": few_shot_num, "with_cot": with_cot,
              "seed": seed, "prompt_sha256": PROMPT_SHA256, "dataset_fingerprint": dataset_fingerprint(examples)}
    if decoding != "greedy":  # runs made before this option exist keep their recorded configuration
        config["decoding"] = decoding
    run_path = output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"{run_path} records a different configuration; use a new output directory")
    run_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs_path = output_dir / "outputs.jsonl"
    done = _read_lines(outputs_path)
    with outputs_path.open("a", encoding="utf-8") as stream:
        for index, example in enumerate(examples, 1):
            if example.episode_id in done:
                continue
            shots = sample_few_shots(examples, example, few_shot_num, seed)
            text = generate(build_ti_list(prompt, example, shots, with_cot))
            ok, sequence = parse_official_output(text)
            row = {"episode_id": example.episode_id, "task": example.task, "origin_output": text,
                   **({"skill_sequence": sequence} if ok else {"format_error": "format_error"})}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            done[example.episode_id] = row
            if index % 25 == 0 or index == len(examples):
                progress(f"[official-vlm] {index}/{len(examples)}")

    rows, coerced_rows = [], []
    for example in examples:
        answer = done[example.episode_id]
        ok = "skill_sequence" in answer
        sequence = answer.get("skill_sequence")
        rows.append({"episode_id": example.episode_id, "task": example.task, "format_error": not ok,
                     "scores": score_official_answer(example.task, example.reference, ok, sequence)})
        coerced_rows.append({"task": example.task,
                             "scores": score_official_answer(example.task, example.reference, ok, sequence, coerce_ids=True)})
    coerced = official_report(coerced_rows)
    (output_dir / "final_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        **config, "standard": standard, "examples": len(examples),
        "format_errors": sum(row["format_error"] for row in rows),
        "inputs": "unlabeled image, number-labeled image, instruction; no entity names",
        "source_revision": _CONFIG["source_revision"],
        **official_report(rows),
        # Not upstream: the same replies with digit-string ids read as integers. Upstream compares raw
        # values, so a reply that quotes its ids ("6") never matches the integer reference.
        "supplementary_integer_ids": {
            "official_total_score_by_dimension": coerced["official_total_score_by_dimension"],
            "dimension_mean_of_total_score": coerced["dimension_mean_of_total_score"],
            "answers_with_quoted_ids": sum(
                row["scores"] != crow["scores"] for row, crow in zip(rows, coerced_rows)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def make_vlm_generator(planner: Any, max_new_tokens: int = 768) -> Callable[[list[list[Any]]], str]:
    """Greedy, unconstrained generation from a ``FrozenNativePlanner`` over a ti-list."""
    import torch
    from PIL import Image

    @torch.no_grad()
    def generate(items: list[list[Any]]) -> str:
        images, content = [], []
        try:
            for kind, value in items:
                if kind == "text":
                    content.append({"type": "text", "text": value})
                else:
                    image = Image.open(value).convert("RGB")
                    images.append(image)
                    content.append({"type": "image", "image": image})
            text = planner.processor.apply_chat_template([{"role": "user", "content": content}],
                                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = planner.processor(text=[text], images=images, return_tensors="pt").to(next(planner.model.parameters()).device)
            length = inputs.input_ids.shape[1]
            output = planner.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)[0, length:]
            return planner.tokenizer.decode(output, skip_special_tokens=True)
        finally:
            for image in images:
                image.close()

    return generate


DFA_ENTITY_CAP = 64


def make_grammar_dfa_generator(planner: Any, max_new_tokens: int = 768, entity_cap: int = DFA_ENTITY_CAP,
                               max_operations: int = 16) -> Callable[[list[list[Any]]], str]:
    """Greedy generation under the DomiKnowS grammar DFA, with no scene information.

    The DFA enforces the JSON skill language, argument roles, the benchmark subtask
    patterns, integer ids below ``entity_cap`` and the operation bound. The reply is
    primed with an opening ```json fence and the accepted plan is closed with one, so
    the unchanged official parser applies. A generation the DFA cannot complete within
    the token budget is returned unfenced and counts as a format error.
    """
    import torch
    from PIL import Image
    from transformers import LogitsProcessorList

    try:
        from .native_planning import NativeConstraintProcessor, NativePlanDFA
    except ImportError:  # pragma: no cover - flat-module execution
        from native_planning import NativeConstraintProcessor, NativePlanDFA

    @torch.no_grad()
    def generate(items: list[list[Any]]) -> str:
        key = ("official-grammar", entity_cap, max_operations)
        dfa = planner.dfas.get(key)
        if dfa is None:
            dfa = planner.dfas[key] = NativePlanDFA(
                planner.token_bytes, planner.tokenizer.eos_token_id, entity_count=entity_cap,
                max_operations=max_operations, grammar_profile="benchmark")
        images, content = [], []
        try:
            for kind, value in items:
                if kind == "text":
                    content.append({"type": "text", "text": value})
                else:
                    image = Image.open(value).convert("RGB")
                    images.append(image)
                    content.append({"type": "image", "image": image})
            text = planner.processor.apply_chat_template([{"role": "user", "content": content}],
                                                         tokenize=False, add_generation_prompt=True,
                                                         enable_thinking=False) + "```json\n"
            inputs = planner.processor(text=[text], images=images, return_tensors="pt").to(next(planner.model.parameters()).device)
            length = inputs.input_ids.shape[1]
            output = planner.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                logits_processor=LogitsProcessorList([NativeConstraintProcessor(dfa, length, max_new_tokens)]),
                eos_token_id=planner.tokenizer.eos_token_id)[0, length:].tolist()
            plan = planner.tokenizer.decode(output, skip_special_tokens=True).strip()
            return f"```json\n{plan}\n```" if dfa.accepts(output) else plan
        finally:
            for image in images:
                image.close()

    return generate
