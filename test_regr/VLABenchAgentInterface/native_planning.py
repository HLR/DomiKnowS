"""Native-token frozen VLM planning and byte-exact plan-language projection."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch

from .world_graph import SKILL_ARGUMENTS, SUBTASK_PATTERNS
from .planning_metrics import BENCHMARK_PATTERNS
from .object_grounding import ObjectConstraints, scene_objects, instruction_object_constraints

# Native benchmark protocol is independent of graph checkpoint vocabulary.
BENCHMARK_ARGUMENTS = {**SKILL_ARGUMENTS, "push": ("target_container_name",)}
NATIVE_LANGUAGE_VERSION = "benchmark_json_grounding_v2"


def native_schema(profile):
    if profile == "benchmark":
        return BENCHMARK_ARGUMENTS, BENCHMARK_PATTERNS
    if profile == "legacy":
        return SKILL_ARGUMENTS, SUBTASK_PATTERNS
    raise ValueError(f"unknown native grammar profile {profile!r}")


class NativePlanFailure(ValueError):
    def __init__(self, message, *, tokens=(), raw=""):
        super().__init__(message)
        self.tokens, self.raw = list(tokens), raw


def save_lookahead_table(table, path, identity):
    """Persist belief-independent DP vectors for reuse across weights and prompts."""
    if not table._cache:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(table._cache)
    temporary = path.with_suffix(".tmp")
    torch.save({"identity": identity, "keys": keys,
        "vectors": torch.stack([table._cache[key] for key in keys]).cpu()}, temporary)
    os.replace(temporary, path)


def restore_lookahead_table(table, path, identity):
    if not Path(path).exists():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["identity"] != identity:
        return False
    vectors = payload["vectors"].to(table.transition.device)
    table._cache.update(zip(payload["keys"], vectors.unbind(0)))
    object.__setattr__(table, "_entries_computed", len(table._cache))
    return True


def tokenizer_checksum(tokenizer):
    return hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()


def byte_token_map(tokenizer):
    """Decode byte-level BPE surfaces without context-dependent Unicode replacement."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = visible.copy()
    missing = [value for value in range(256) if value not in visible]
    visible += missing
    characters += list(range(256, 256 + len(missing)))
    inverse = {chr(character): byte for byte, character in zip(visible, characters)}
    result = {}
    specials = set(tokenizer.all_special_ids)
    for surface, token_id in tokenizer.get_vocab().items():
        if token_id in specials:
            continue
        try:
            result[token_id] = bytes(inverse[character] for character in surface)
        except KeyError as exc:
            raise ValueError("native planning requires a byte-level BPE tokenizer") from exc
    return result


class NativePlanDFA:
    """Lazy deterministic JSON language with semantic subtask boundaries.

    Token transitions consume *all* token bytes, including tokens spanning JSON
    delimiters. Semantic states retain subtask-prefix possibilities rather than
    complete history, avoiding enumeration of all possible operation sequences.
    """

    def __init__(self, token_bytes, eos_id, *, entity_count, max_operations=None,
                 grammar_profile="benchmark", object_constraints=None):
        self.eos_label = int(eos_id)
        self.grammar_profile = grammar_profile
        self.skill_arguments, self.patterns = native_schema(grammar_profile)
        self.max_operations = max_operations if max_operations is not None else (16 if grammar_profile == "benchmark" else 8)
        if self.max_operations < 1 or entity_count < 0:
            raise ValueError("positive operation budget and nonnegative entity count required")
        self.entity_count = min(entity_count, 64) if grammar_profile == "legacy" else entity_count
        self.object_constraints = object_constraints or ObjectConstraints()
        self.states = frozenset()  # lazily projected states
        self.start_state = (0, ((),), "start", 0, "")
        self.operations = [{}]
        self.ends = {}
        self.possible = [set()]
        self.in_string = [False]
        self.whitespace_boundary = [False]
        for skill, roles in self.skill_arguments.items():
            values = self.object_constraints.ids(skill, roles[0], self.entity_count) if roles else [None]
            for pointer in values:
                if pointer is not None and (isinstance(pointer, bool) or not 0 <= pointer < self.entity_count):
                    raise ValueError("object constraint ID is outside the scene table")
                params = {role: pointer for role in roles}
                variants = [{"name": skill, "params": params}]
                if grammar_profile == "benchmark":
                    variants.append({"params": params, "name": skill})
                for value in variants:
                    text = json.dumps(value, separators=(",", ":"))
                    node, quoted = 0, False
                    self.possible[node].add(skill)
                    for byte in text.encode():
                        if byte == ord('"'):
                            quoted = not quoted
                        if byte not in self.operations[node]:
                            self.operations[node][byte] = len(self.operations)
                            self.operations.append({})
                            self.possible.append(set())
                            self.in_string.append(quoted)
                            self.whitespace_boundary.append(not quoted and (byte in b'{}:,"' or byte in b'0123456789'))
                        node = self.operations[node][byte]
                        self.possible[node].add(skill)
                    self.ends[node] = skill
        self.tokens = [{}]
        self.token_ends = {}
        for token_id, value in token_bytes.items():
            if not value or any(byte >= 128 for byte in value):
                continue  # canonical plan language is ASCII
            node = 0
            for byte in value:
                if byte not in self.tokens[node]:
                    self.tokens[node][byte] = len(self.tokens)
                    self.tokens.append({})
                node = self.tokens[node][byte]
            self.token_ends.setdefault(node, []).append(token_id)

    @lru_cache(maxsize=None)
    def next_semantics(self, prefixes, skill):
        next_prefixes = set()
        for prefix in prefixes:
            candidate = prefix + (skill,)
            for pattern in self.patterns:
                if tuple(pattern[:len(candidate)]) == candidate:
                    if len(candidate) == len(pattern):
                        next_prefixes.add(())
                    if len(candidate) < len(pattern):
                        next_prefixes.add(candidate)
        return tuple(sorted(next_prefixes))

    @lru_cache(maxsize=None)
    def allowed_skills(self, prefixes, count):
        remaining = self.max_operations - count - 1
        result = set()
        for skill in self.skill_arguments:
            next_prefixes = self.next_semantics(prefixes, skill)
            if () in next_prefixes or any(
                tuple(pattern[:len(prefix)]) == prefix and len(pattern)-len(prefix) <= remaining
                for prefix in next_prefixes for pattern in self.patterns
            ):
                result.add(skill)
        return frozenset(result)

    def byte_step(self, state, byte):
        count, prefixes, phase, node, pending = state
        flexible = self.grammar_profile == "benchmark"
        if flexible and phase == "operation" and pending.startswith("escape"):
            if pending == "escape":
                if byte == ord("u"):
                    return (count, prefixes, phase, node, "escape_u")
                escaped = {ord('"'): ord('"'), ord('\\'): ord('\\'), ord('/'): ord('/'),
                           ord('b'): 8, ord('f'): 12, ord('n'): 10, ord('r'): 13, ord('t'): 9}
                byte = escaped.get(byte, -1)
            else:
                if chr(byte) not in "0123456789abcdefABCDEF":
                    return None
                digits = pending.removeprefix("escape_u") + chr(byte)
                if len(digits) < 4:
                    return (count, prefixes, phase, node, "escape_u" + digits)
                byte = int(digits, 16)
            # Escaped quotes cannot end a string; only literal content is normalized.
            if byte in (ord('"'), ord('\\')) or byte < 32:
                return None
            pending = ""
        elif flexible and byte in b" \t\r\n":
            if phase in {"start", "separator", "closed"} or (phase == "operation" and
                    (node == 0 or self.whitespace_boundary[node])):
                return (count, prefixes, phase, node, "ws")
            return None
        elif flexible and phase == "operation" and self.in_string[node] and byte == ord('\\'):
            return (count, prefixes, phase, node, "escape")
        if pending == "ws" and phase == "operation" and byte in b"0123456789":
            # JSON whitespace cannot split an integer, e.g. 1 0 is not ID 10.
            # It may precede its first digit after a colon.
            if any(key in b"}," for key in self.operations[node]):
                return None
        if phase == "start":
            return (count, prefixes, "operation", 0, "") if byte == ord("[") else None
        if phase == "separator":
            if byte == ord("]") and () in prefixes:
                return (count, prefixes, "closed", 0, "")
            if byte == ord(",") and count < self.max_operations:
                return (count, prefixes, "operation", 0, "")
            return None
        if phase != "operation" or count >= self.max_operations:
            return None
        child = self.operations[node].get(byte)
        if child is None or not self.possible[child].intersection(self.allowed_skills(prefixes, count)):
            return None
        if child in self.ends:
            next_prefixes = self.next_semantics(prefixes, self.ends[child])
            return (count + 1, next_prefixes, "separator", 0, "") if next_prefixes else None
        return (count, prefixes, "operation", child, "")

    @lru_cache(maxsize=None)
    def transitions(self, state):
        if state[2] == "ended":
            return ()
        result = []
        def visit(token_node, current):
            result.extend((token_id, current) for token_id in self.token_ends.get(token_node, ()))
            for byte, child in self.tokens[token_node].items():
                next_state = self.byte_step(current, byte)
                if next_state is not None:
                    visit(child, next_state)
        visit(0, state)
        if state[2] == "closed":
            result.append((self.eos_label, (state[0], state[1], "ended", 0, "")))
        return tuple(result)

    def allowed_tokens(self, state, remaining_steps=None):
        return {label for label, target in self.transitions(state)
                if remaining_steps != 0 or self.is_accepting(target)}

    def step(self, state, label):
        return dict(self.transitions(state)).get(int(label))

    def is_accepting(self, state):
        return state is not None and state[2] == "ended"

    def accepts(self, labels):
        state = self.start_state
        for label in labels:
            state = self.step(state, label)
            if state is None:
                return False
        return self.is_accepting(state)


class NativeConstraintProcessor:
    """HF logits processor using the repository's CTRL-G success DP and belief update."""

    def __init__(self, dfa, prompt_length, max_tokens, *, hmm=None, weight=0, lookahead=50, table=None):
        self.dfa, self.prompt_length, self.max_tokens = dfa, prompt_length, max_tokens
        self.state = dfa.start_state
        self.consumed = 0
        self.weight = weight
        self.hmm = hmm
        self.table = table
        if hmm is not None and weight:
            from domiknows.generation.applications.hmm_dfa_decoder import HMMDFAStaticLookahead
            from domiknows.generation.dfa.stop_policy import StopPolicy
            self.initial, self.transition, self.emission = [value.double() for value in hmm]
            self.belief = self.initial.clone()
            if self.table is None:
                self.table = HMMDFAStaticLookahead(
                    transition=self.transition, emission=self.emission,
                    policy=StopPolicy(max_steps=max_tokens), eos_label=dfa.eos_label,
                    max_depth=lookahead,
                    allowed_transitions=lambda state, depth: tuple((label, target) for label, target in dfa.transitions(state)
                        if depth != 1 or dfa.is_accepting(target)),
                    is_accepting=dfa.is_accepting,
                )

    def __call__(self, input_ids, scores):
        from domiknows.generation.applications.hmm_utils import hmm_forward_update
        labels = input_ids[0, self.prompt_length:].tolist()
        for label in labels[self.consumed:]:
            self.state = self.dfa.step(self.state, label)
            if self.state is None:
                raise ValueError("generation left the native plan DFA")
            if self.hmm is not None and self.weight:
                self.belief = hmm_forward_update(self.belief, self.emission, self.transition, label)
        self.consumed = len(labels)
        remaining = self.max_tokens - len(labels) - 1
        allowed = self.dfa.allowed_tokens(self.state, remaining_steps=remaining)
        if not allowed:
            raise NativePlanFailure("native DFA has no completion within the token budget", tokens=labels)
        result = torch.full_like(scores, -torch.inf)
        for label in allowed:
            value = scores[0, label]
            if self.hmm is not None and self.weight:
                target = self.dfa.step(self.state, label)
                if self.dfa.is_accepting(target):
                    probability = self.belief.new_tensor(1.)
                else:
                    belief = hmm_forward_update(self.belief, self.emission, self.transition, label)
                    probability = self.table.success_after(belief, target, remaining)
                value = value + self.weight * probability.clamp_min(torch.finfo(probability.dtype).tiny).log().to(value.dtype)
            result[0, label] = value
        return result


def native_prompt(instruction, entities, grammar_profile="benchmark"):
    arguments, _ = native_schema(grammar_profile)
    return (
        ("Return only a compact JSON array of VLABench operations, with no explanation or markdown. "
         if grammar_profile == "legacy" else
         "Return only a JSON array of VLABench operations, with no explanation or markdown. ")
        +
        'Each operation has keys "name" and "params". Use integer object IDs. '
        'Example format: [{"name":"pick","params":{"target_entity_name":0}},'
        '{"name":"place","params":{"target_container_name":1}}].\n'
        f"Skills and required arguments: {json.dumps(dict(arguments))}\n"
        f"Objects: {json.dumps(dict(enumerate(entities)))}\nInstruction: {instruction}"
    )


class FrozenNativePlanner:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        self.tokenizer = processor.tokenizer
        self.token_bytes = byte_token_map(self.tokenizer)
        self.checksum = tokenizer_checksum(self.tokenizer)
        self.dfas, self.tables = {}, {}
        self.dp_cache_directory, self.dp_cache_identity = None, None
        self.persisted_tables = set()

    @classmethod
    def load(cls, model_path, load_in_4bit=True):
        from .models import resolve_vision_language_loader, load_sanitized_auto_config
        import transformers
        model_class, processor_class = resolve_vision_language_loader()
        kwargs = dict(dtype=torch.bfloat16, device_map="auto", local_files_only=True,
                      config=load_sanitized_auto_config(transformers, model_path, local_files_only=True))
        if load_in_4bit:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        model = model_class.from_pretrained(model_path, **kwargs).eval()
        model.requires_grad_(False)
        processor = processor_class.from_pretrained(model_path, local_files_only=True)
        return cls(model, processor)

    @torch.no_grad()
    def generate_teacher_batch(self, examples, *, seed=42, max_tokens=768, grammar_profile="benchmark"):
        """Batch teacher sampling only; evaluation configurations remain single-example."""
        from PIL import Image
        images, texts = [], []
        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for example in examples:
                row = [Image.open(path).convert("RGB") for path in (example.segmented_image_paths or example.image_paths)]
                images.extend(row)
                content = [{"type": "image", "image": image} for image in row]
                content.append({"type": "text", "text": native_prompt(example.instruction, example.entities, grammar_profile)})
                texts.append(self.processor.apply_chat_template([{"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(next(self.model.parameters()).device)
            length = inputs.input_ids.shape[1]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=.8, top_p=.95, use_cache=True)[:, length:].tolist()
            stops = self.model.generation_config.eos_token_id
            stops = {stops} if isinstance(stops, int) else set(stops or [self.tokenizer.eos_token_id])
            result = []
            for row in output:
                end = next((index+1 for index, token in enumerate(row) if token in stops), len(row))
                row = row[:end]
                result.append((self.tokenizer.decode(row, skip_special_tokens=True), row))
            return result
        finally:
            self.tokenizer.padding_side = previous_padding
            for image in images:
                image.close()

    @torch.no_grad()
    def generate(self, example, *, mode="raw", max_tokens=768, max_operations=16, seed=42, sample=False,
                 hmm=None, weight=0, grammar_profile="benchmark", object_constraints="instruction"):
        from PIL import Image
        from transformers import LogitsProcessorList
        paths = example.segmented_image_paths or example.image_paths
        images = [Image.open(path).convert("RGB") for path in paths]
        try:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": native_prompt(example.instruction, example.entities, grammar_profile)})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.processor(text=[text], images=images, return_tensors="pt").to(next(self.model.parameters()).device)
            length = inputs.input_ids.shape[1]
            kwargs = dict(max_new_tokens=max_tokens, do_sample=sample, use_cache=True)
            if sample:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                kwargs.update(temperature=0.8, top_p=0.95)
            constraint = None
            if mode != "raw":
                grounding = grounding_for_example(example, object_constraints)
                key = (len(example.entities), max_operations, grammar_profile, grounding.restrictions)
                if key not in self.dfas:
                    self.dfas[key] = NativePlanDFA(self.token_bytes, self.tokenizer.eos_token_id,
                        entity_count=key[0], max_operations=max_operations, grammar_profile=grammar_profile,
                        object_constraints=grounding)
                constraint = NativeConstraintProcessor(self.dfas[key], length, max_tokens,
                    hmm=hmm, weight=weight, table=self.tables.get(key))
                cache_path, cache_identity = None, None
                if constraint.table is not None and self.dp_cache_directory:
                    cache_identity = (self.dp_cache_identity, self.checksum, key, max_tokens,
                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                    key_hash = hashlib.sha256(repr(key).encode()).hexdigest()
                    cache_path = Path(self.dp_cache_directory) / f"language_{key_hash}.pt"
                    if key not in self.tables:
                        restore_lookahead_table(constraint.table, cache_path, cache_identity)
                kwargs["logits_processor"] = LogitsProcessorList([constraint])
                kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            output = self.model.generate(**inputs, **kwargs)[0, length:].tolist()
            if constraint is not None and constraint.table is not None:
                self.tables[key] = constraint.table
                if cache_path and key not in self.persisted_tables:
                    save_lookahead_table(constraint.table, cache_path, cache_identity)
                    self.persisted_tables.add(key)
            if mode != "raw" and not self.dfas[key].accepts(output):
                raise NativePlanFailure("native generation truncated before an accepting plan",
                    tokens=output, raw=self.tokenizer.decode(output, skip_special_tokens=True))
            return self.tokenizer.decode(output, skip_special_tokens=True), output
        except NativePlanFailure as exc:
            if not exc.raw:
                exc.raw = self.tokenizer.decode(exc.tokens, skip_special_tokens=True)
            raise
        finally:
            for image in images:
                image.close()


def grounding_for_example(example, mode="instruction"):
    if mode == "none":
        return ObjectConstraints(evidence=("disabled",))
    if mode != "instruction":
        raise ValueError(f"unknown object constraint mode {mode!r}")
    if example.env_config_path is None:
        return ObjectConstraints(evidence=("abstain: scene components unavailable",))
    config = json.loads(Path(example.env_config_path).read_text(encoding="utf-8"))
    return instruction_object_constraints(example.instruction, scene_objects(config, example.entities))


def validate_native_plan(plan, entities, *, grammar_profile="benchmark", max_operations=None, grounding=None):
    """Independent symbolic protocol diagnostic; graph validity remains separate."""
    from .world_graph import validate_plan
    def unique_members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result
    if isinstance(plan, str):
        try:
            plan = json.loads(plan, object_pairs_hook=unique_members)
        except (ValueError, TypeError) as exc:
            return False, [f"invalid native JSON: {exc}"]
    if not isinstance(plan, (list, tuple)):
        return False, ["native plan must be an array"]
    for operation in plan:
        if (not isinstance(operation, dict) or set(operation) != {"name", "params"}
                or not isinstance(operation["params"], dict) or not isinstance(operation["name"], str)):
            return False, ["native operations require exactly name and params members"]
    arguments, patterns = native_schema(grammar_profile)
    if max_operations is None:
        max_operations = 8 if grammar_profile == "legacy" else 16
    entity_count = min(len(entities), 64) if grammar_profile == "legacy" else len(entities)
    result = validate_plan(plan, entity_table=entities, skill_arguments=arguments, patterns=patterns)
    errors = list(result.errors)
    if len(plan) > max_operations:
        errors.append("native operation budget exceeded")
    for index, operation in enumerate(plan):
        if set(operation["params"]) != set(arguments.get(operation["name"], ())):
            errors.append(f"operation {index}: unexpected or missing native argument keys")
        for role, pointer in operation["params"].items():
            if isinstance(pointer, bool) or not isinstance(pointer, int) or not 0 <= pointer < entity_count:
                errors.append(f"operation {index}: native object IDs must be in-range integers")
            elif grounding is not None and pointer not in grounding.ids(operation["name"], role, len(entities)):
                errors.append(f"operation {index}: violates instruction object candidates")
    return not errors, errors
