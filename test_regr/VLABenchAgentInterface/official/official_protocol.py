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
    from ..planning_metrics import (
        BENCHMARK_PATTERNS, _CONFIG, DIMENSIONS, OFFICIAL_TASK_DIMENSIONS, PlanningMetrics, default_dependency,
        score_planning_metrics,
    )
    from ..world_graph import canonicalize_plan
except ImportError:  # pragma: no cover - flat-module execution
    from planning_metrics import (
        BENCHMARK_PATTERNS, _CONFIG, DIMENSIONS, OFFICIAL_TASK_DIMENSIONS, PlanningMetrics, default_dependency,
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
    perception: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate (resumably), score and summarize. ``generate`` maps a ti-list to reply text."""
    prompt = load_official_prompt()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"protocol": PROTOCOL, "model": model_name, "few_shot_num": few_shot_num, "with_cot": with_cot,
              "seed": seed, "prompt_sha256": PROMPT_SHA256, "dataset_fingerprint": dataset_fingerprint(examples)}
    if decoding != "greedy":  # runs made before this option exist keep their recorded configuration
        config["decoding"] = decoding
    if perception is not None:  # the perception model, threshold and grounding policy of the run
        config["perception"] = dict(perception)
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
            "official_mapped_overall": coerced["official_mapped_overall"],
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
PRESS_MARK = "press-question:"
# One fixed question, asked once per episode. Three wordings were tried on 60 validation episodes
# (30 question tasks, 30 others) and this one was kept: it recognised 13 of 30 question tasks with no
# false positives, against 0 of 30 for a generic description of "choose one object and press its
# button" and 0 of 30 for a pick-or-press choice. It names the property topics of the benchmark's
# question tasks, so it encodes benchmark knowledge; the model, not a keyword rule, makes the call.
PRESS_QUESTION = (
    "Question: Does the instruction above ask a question about the objects' properties or behaviour "
    "(such as density, friction, magnetism, reflection, the speed of sound or thermal expansion) that is "
    "answered by choosing one object and pressing its button? Answer with one word, Yes or No."
)


# Asked about the number-labeled image only. On 150 validation episodes the largest number the model
# read was never below the largest id of the reference plan (the references were used only to check
# this, not by the decoder), and exceeded it by 2.7 on average.
ENTITY_QUESTION = (
    "This picture shows four camera views of a scene. Small black tags with white numbers mark objects. "
    "What is the largest number written on any tag? Answer with one integer only."
)


# Stage one of the pattern-first variant. It names no task and no benchmark topic: the patterns it
# chooses from are the ones the official prompt already lists.
PATTERN_QUESTION = (
    "Using only the sub-skill sequence patterns listed above, which pattern or patterns solve the language "
    "instruction? Answer with a JSON list of patterns, each pattern a list of skill names, and nothing else."
)


def parse_patterns(text: str, allowed: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """The distinct listed patterns in a stage-one reply (none if it is not a JSON list of patterns)."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return ()
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return ()
    known = {tuple(pattern) for pattern in allowed}
    if not isinstance(value, list) or (value and isinstance(value[0], str)):
        value = [value] if isinstance(value, list) else []
    chosen: list[tuple[str, ...]] = []
    for pattern in value:
        if isinstance(pattern, list) and all(isinstance(skill, str) for skill in pattern) and tuple(pattern) in known                 and tuple(pattern) not in chosen:
            chosen.append(tuple(pattern))
    return tuple(chosen)


def parse_entity_count(text: str, cap: int = DFA_ENTITY_CAP) -> int:
    """Object ids ``0..n-1`` for the largest tag number the model read; ``cap`` if it gave no number."""
    import re

    numbers = [int(match) for match in re.findall(r"\d+", text)]
    return min(max(numbers) + 1, cap) if numbers else cap


def prepare_inputs(planner: Any, items: list[list[Any]], prefix: str = ""):
    """Chat-template a ti-list. Returns ``(inputs, images)``; the caller closes the images."""
    from PIL import Image

    images, content = [], []
    for kind, value in items:
        if kind == "text":
            content.append({"type": "text", "text": value})
        else:
            image = Image.open(value).convert("RGB")
            images.append(image)
            content.append({"type": "image", "image": image})
    text = planner.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) + prefix
    inputs = planner.processor(text=[text], images=images, return_tensors="pt").to(next(planner.model.parameters()).device)
    return inputs, images


def make_grammar_dfa_generator(planner: Any, max_new_tokens: int = 768, entity_cap: int = DFA_ENTITY_CAP,
                               max_operations: int = 16, press_question: bool = False,
                               hmm: Any = None, weight: float = 0.0,
                               lookahead_score: str = "joint", scene_ids: bool = False, pattern_first: bool = False,
                               adapt: dict[str, Any] | None = None, seed: int = 42,
                               causal: bool = False, perception: Any = None,
                               research_knowledge: Any = None, disabled_formulas: Sequence[str] = (),
                               grounding_policy: str = "joint"
                               ) -> Callable[[list[list[Any]]], str]:
    """Greedy generation under the DomiKnowS grammar DFA, with no scene information.

    The DFA enforces the JSON skill language, argument roles, the benchmark subtask
    patterns, integer ids below ``entity_cap`` and the operation bound. The reply is
    primed with an opening ```json fence and the accepted plan is closed with one, so
    the unchanged official parser applies. A generation the DFA cannot complete within
    the token budget is returned unfenced and counts as a format error.

    ``press_question`` first asks the model :data:`PRESS_QUESTION`, ``scene_ids`` reads the largest tag
    number and ``pattern_first`` asks for the subtask patterns. The answers are observations of the
    LCs in :mod:`.decoding_graph` (a Yes allows one ``press``; ids stay within the tags read; the plan
    is made of the named patterns), compiled into a :class:`.knowledge_dfa.KnowledgePlanDFA`. The decision is
    recorded in an HTML comment ahead of the fence. ``hmm`` is an ``(initial, transition,
    emission)`` tuple; with ``weight`` > 0 its action-level lookahead (``native_lookahead``) adds
    ``weight * (log HMM mass - best log HMM mass)`` to the DFA-legal tokens that decide an action.
    The DFA stays the only authority on which tokens are legal, and the comment
    ``hmm-action: changed=x scored=y steps=z`` records how often the lookahead changed the greedy token.
    ``causal`` adds the id-only causal LCs of :mod:`.causal_graph` to the same DFA.
    ``perception`` is a frozen ``ImagePerceptionModel``: it reads the episode's two images and
    instruction, and its visible tags, predicted object kinds and role hypotheses become observations of the decoding
    and research LCs (:mod:`.perception_observations`); the result is recorded in a comment.
    """
    import torch
    from transformers import LogitsProcessorList

    try:
        from ..native_planning import NativeConstraintProcessor, NativePlanDFA, NativePlanFailure
        from .native_lookahead import DFAActionLookahead
    except ImportError:  # pragma: no cover - flat-module execution
        from native_planning import NativeConstraintProcessor, NativePlanDFA, NativePlanFailure
        from native_lookahead import DFAActionLookahead
    from domiknows.generation.applications.action_lookahead import TokenOperators

    try:
        from .knowledge_dfa import KnowledgePlanDFA
    except ImportError:  # pragma: no cover - flat-module execution
        from knowledge_dfa import KnowledgePlanDFA
    operators = TokenOperators(hmm[1], hmm[2]) if hmm is not None and weight > 0 else None
    adapting = operators is not None and adapt is not None and adapt.get("samples", 0) > 0 and adapt.get("lam", 0) > 0
    if adapting:
        try:
            from .official_adapt import adapt_emission, episode_seed, sample_plans
        except ImportError:  # pragma: no cover - flat-module execution
            from official_adapt import adapt_emission, episode_seed, sample_plans

    if perception is not None:
        try:
            from .perception_observations import PerceivedGrounding, observe_episode
            from ..knowledge.research_lcs import build_research_knowledge
        except ImportError:  # pragma: no cover - flat-module execution
            from perception_observations import PerceivedGrounding, observe_episode
            from knowledge.research_lcs import build_research_knowledge
        knowledge_lcs = research_knowledge or build_research_knowledge()

    def language(press_only: bool, bound: int | None, patterns: tuple = (), observed: Any = None):
        """The grammar DFA, with the episode's answers and the causal rules applied only through compiled LCs."""
        common = dict(entity_count=entity_cap, max_operations=max_operations, grammar_profile="benchmark")
        if observed is not None and observed.supplies_facts:
            # One episode's perception: built per episode and not cached.
            grounding = (PerceivedGrounding(observed.grounding, knowledge_lcs, disabled_formulas, grounding_policy)
                         if observed.grounding is not None else None)
            dfa = KnowledgePlanDFA(planner.token_bytes, planner.tokenizer.eos_token_id, causal=causal,
                                   press_answer=press_only, scene_bound=bound, chosen_patterns=patterns,
                                   visible_entities=observed.visible, grounding=grounding,
                                   object_candidates=observed.candidates, **common)
            return None, dfa
        knowledge = causal or press_only or bound is not None or bool(patterns)
        key = ("official-knowledge" if knowledge else "official-grammar", causal, press_only, bound, patterns,
               entity_cap, max_operations)
        if key not in planner.dfas:
            planner.dfas[key] = (KnowledgePlanDFA(planner.token_bytes, planner.tokenizer.eos_token_id, causal=causal,
                                                  press_answer=press_only, scene_bound=bound, chosen_patterns=patterns,
                                                  **common)
                                 if knowledge else
                                 NativePlanDFA(planner.token_bytes, planner.tokenizer.eos_token_id, **common))
        return key, planner.dfas[key]

    def lookahead_for(key, dfa):
        slot = ("action-lookahead", operators.size, id(operators)) + key
        if slot not in planner.tables:
            planner.tables[slot] = DFAActionLookahead(dfa, planner.tokenizer, operators)
        return planner.tables[slot]

    @torch.no_grad()
    def says_yes(items: list[list[Any]]) -> bool:
        inputs, images = prepare_inputs(planner, [*items, ["text", PRESS_QUESTION]])
        try:
            length = inputs.input_ids.shape[1]
            answer = planner.model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)[0, length:]
            return planner.tokenizer.decode(answer, skip_special_tokens=True).strip().lower().startswith("yes")
        finally:
            for image in images:
                image.close()

    @torch.no_grad()
    def read_entity_count(items: list[list[Any]]) -> int:
        """Object ids the number-labeled image offers, read by the same frozen model (no labels, no names)."""
        mask = [value for kind, value in items if kind == "image"][-1]
        inputs, images = prepare_inputs(planner, [["text", ENTITY_QUESTION], ["image", mask]])
        try:
            length = inputs.input_ids.shape[1]
            answer = planner.model.generate(**inputs, max_new_tokens=40, do_sample=False, use_cache=True)[0, length:]
            return parse_entity_count(planner.tokenizer.decode(answer, skip_special_tokens=True), entity_cap)
        finally:
            for image in images:
                image.close()

    @torch.no_grad()
    def choose_patterns(items: list[list[Any]]) -> tuple:
        """Stage one: the model names the subtask patterns (from the official prompt's list) for this episode."""
        inputs, images = prepare_inputs(planner, [*items, ["text", PATTERN_QUESTION]])
        try:
            length = inputs.input_ids.shape[1]
            answer = planner.model.generate(**inputs, max_new_tokens=96, do_sample=False, use_cache=True)[0, length:]
            return parse_patterns(planner.tokenizer.decode(answer, skip_special_tokens=True), BENCHMARK_PATTERNS)
        finally:
            for image in images:
                image.close()

    @torch.no_grad()
    def generate(items: list[list[Any]]) -> str:
        press_only = press_question and says_yes(items)
        bound = read_entity_count(items) if scene_ids else None
        chosen = choose_patterns(items) if pattern_first and not press_only else ()
        observed = observe_episode(perception, items) if perception is not None else None
        key, dfa = language(press_only, bound, tuple(sorted(chosen)), observed)
        steered = operators is not None and not press_only
        inputs, images = prepare_inputs(planner, items, prefix="```json\n")
        try:
            length = inputs.input_ids.shape[1]
            episode_hmm, episode_lookahead, kept = hmm, None, None
            if steered and adapting:
                # Per-instance adaptation: sample this prompt, re-estimate the emissions, rebuild the lookahead.
                sequences = sample_plans(planner, inputs, planner.tokenizer, samples=adapt["samples"],
                                         max_new_tokens=adapt.get("max_new_tokens", 320), seed=episode_seed(items, seed))
                kept = len(sequences)
                emission = adapt_emission(hmm[0].to(operators.emission.dtype), operators.transition, operators.emission,
                                          sequences, adapt["lam"])
                episode_hmm = (hmm[0], hmm[1], emission)
                episode_lookahead = DFAActionLookahead(dfa, planner.tokenizer, TokenOperators(operators.transition, emission))
                del emission
            elif steered:
                episode_lookahead = (lookahead_for(key, dfa) if key is not None else
                                     DFAActionLookahead(dfa, planner.tokenizer, operators))
            constraint = NativeConstraintProcessor(dfa, length, max_new_tokens, hmm=episode_hmm if steered else None,
                                                   weight=weight if steered else 0, action_lookahead=episode_lookahead,
                                                   lookahead_score=lookahead_score)
            try:
                output = planner.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    logits_processor=LogitsProcessorList([constraint]),
                    eos_token_id=planner.tokenizer.eos_token_id)[0, length:].tolist()
            except NativePlanFailure as failure:  # no legal completion: an unfenced reply, i.e. a format error
                output = list(failure.tokens)
            plan = planner.tokenizer.decode(output, skip_special_tokens=True).strip()
            reply = f"```json\n{plan}\n```" if dfa.accepts(output) else plan
            notes = []
            if press_question:
                notes.append(f"<!-- {PRESS_MARK} {'yes' if press_only else 'no'} -->")
            if kept is not None:
                notes.append(f"<!-- adapt: samples={kept} -->")
            if pattern_first:
                notes.append("<!-- pattern-first: " + json.dumps(["+".join(pattern) for pattern in chosen]) + " -->")
            if scene_ids:
                notes.append(f"<!-- scene-ids: {bound} -->")
            if observed is not None:
                notes.append(observed.note())
            if constraint.session is not None:
                stats = constraint.session.stats
                notes.append(f"<!-- hmm-action: changed={stats['changed']} scored={stats['scored_steps']} steps={stats['steps']} -->")
            return "\n".join([*notes, reply])
        finally:
            for image in images:
                image.close()

    return generate
