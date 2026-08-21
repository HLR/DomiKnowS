#!/usr/bin/env python3
"""Scallop-style GraphQA execution with pluggable atomic concept scorers.

This is a deadline-oriented evaluator: it uses the same converted VQAR/GraphQA
query graph as the DomiKnowS adapter, keeps KB edges symbolic, and swaps the
scene predicate scorer.  The executable answer is computed from atomic concept
scores rather than from gold answers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from tqdm import tqdm
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

REPO = Path('/localscratch2/premsrit/DomiKnowS')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from test_regr.GraphQA.dataset import (  # noqa: E402
    DEFAULT_VQAR_ROOT,
    load_kb_facts,
    load_vqar_tasks,
    vqar_task_to_graphqa_instance,
)
from test_regr.GraphQA.execution import create_query_logic, materialize_bounded_facts  # noqa: E402
from test_regr.GraphQA.graph import SYMMETRIC_OBJECT_RELATIONS, alias_values, canonical_relation, collect_object_relations  # noqa: E402
from test_regr.GraphQA.modules import (  # noqa: E402
    GraphQAPredicateClassifier,
    _format_object_metadata,
    _object_pair_feature_prompt,
    _object_symbol_feature_prompt,
)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate Scallop-style GraphQA execution with oracle or Qwen atomic scorers.')
    p.add_argument('--root', type=Path, default=DEFAULT_VQAR_ROOT)
    p.add_argument('--task-path', type=Path, required=True)
    p.add_argument('--kb-dir', type=Path, default=None)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--kb-depth', type=int, default=2)
    p.add_argument('--scorer', choices=['oracle', 'qwen-logprob', 'qwen-heads', 'qwen-vl-logprob', 'qwen-vl-grouped'], default='oracle')
    p.add_argument('--model-path', default='/localscratch/premsrit/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-length', type=int, default=256)
    p.add_argument('--include-scene-facts', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-candidate-symbols', type=int, default=128)
    p.add_argument('--inference', choices=['local', 'ilp'], default='local')
    p.add_argument('--global-consistency', action=argparse.BooleanOptionalAction, default=False, help='Apply hard global grounding constraints during ILP decoding.')
    p.add_argument('--single-answer-only', action='store_true', help='Evaluate only examples with exactly one gold answer.')
    p.add_argument('--max-ilp-vars', type=int, default=50000, help='Skip an ILP instance if it creates more Boolean variables than this; 0 disables the cap.')
    p.add_argument('--max-ilp-constraints', type=int, default=200000, help='Skip an ILP instance if it creates more linear constraints than this; 0 disables the cap.')
    p.add_argument('--image-cache', type=Path, default=Path('/egr/research-hlr2/premsrit/VQAR_data/image_cache'))
    p.add_argument('--draw-boxes', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--group-size', type=int, default=12, help='Maximum name choices per grouped Qwen-VL call.')
    return p.parse_args()


class OracleScorer:
    def __init__(self, instance):
        self.facts = {(canonical_relation(p), str(l), _norm_payload(r)) for p, l, r in materialize_bounded_facts(instance)}
        # Oracle validation should not silently drop KB symbols for speed.
        self.max_candidate_symbols = 0

    def object_symbol(self, pred, obj, symbol):
        return 1.0 if (canonical_relation(pred), str(obj), str(symbol)) in self.facts else 0.0

    def object_pair(self, pred, src, dst):
        return 1.0 if (canonical_relation(pred), str(src), str(dst)) in self.facts else 0.0


class QwenLogprobScorer:
    def __init__(self, model_path, device='cuda', max_length=256, include_scene_facts=True, temperature=1.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.max_length = int(max_length)
        self.include_scene_facts = include_scene_facts
        self.temperature = max(float(temperature), 1e-6)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {'trust_remote_code': True, 'low_cpu_mem_usage': True}
        if str(device).startswith('cuda'):
            kwargs['torch_dtype'] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device)
        self.model.eval()
        self._cache = {}

    def bind(self, instance):
        return BoundQwenScorer(self, instance)

    @torch.no_grad()
    def yes_probability(self, prompt):
        if prompt in self._cache:
            return self._cache[prompt]
        yes = self._continuation_score(prompt, ' yes')
        no = self._continuation_score(prompt, ' no')
        logits = torch.tensor([no, yes], dtype=torch.float32) / self.temperature
        prob = float(torch.softmax(logits, dim=0)[1].item())
        self._cache[prompt] = prob
        return prob

    def _continuation_score(self, prompt, answer):
        prompt_ids = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_length).input_ids.to(self.device)
        answer_ids = self.tokenizer(answer, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, -self.max_length:]
            answer_len = min(answer_ids.shape[1], input_ids.shape[1] - 1)
        else:
            answer_len = answer_ids.shape[1]
        out = self.model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_scores = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return float(token_scores[:, -answer_len:].mean().item())


class BoundQwenScorer:
    def __init__(self, base: QwenLogprobScorer, instance):
        self.base = base
        self.instance = instance
        self.scene_text = _scene_text(instance) if base.include_scene_facts else ''

    def object_symbol(self, pred, obj, symbol):
        pred = canonical_relation(pred)
        prompt = '\n'.join([
            'Answer yes or no.',
            'Task: classify a GraphQA object-symbol atomic predicate.',
            f'Predicate: {pred}',
            f'Object: {obj}',
            f'Symbol: {symbol}',
            f'Query: {self.instance.get("query", {})}',
            f'Object metadata:\n{_format_object_metadata(self.instance, obj)}',
            self.scene_text,
            'Is the predicate true? Answer:',
        ])
        return self.base.yes_probability(prompt)

    def object_pair(self, pred, src, dst):
        pred = canonical_relation(pred)
        prompt = '\n'.join([
            'Answer yes or no.',
            'Task: classify a GraphQA object-object relation atomic predicate.',
            f'Predicate: {pred}',
            f'Source object: {src}',
            f'Destination object: {dst}',
            f'Query: {self.instance.get("query", {})}',
            f'Source metadata:\n{_format_object_metadata(self.instance, src)}',
            f'Destination metadata:\n{_format_object_metadata(self.instance, dst)}',
            self.scene_text,
            'Is the relation true? Answer:',
        ])
        return self.base.yes_probability(prompt)



class QwenHeadsScorer:
    """Fast trained-head atomic scorer.

    This follows the GraphQA predicate-classifier checkpoint format: Qwen encodes
    a predicate prompt once, then task heads score the DomiKnowS concept labels.
    """

    def __init__(self, model_path, checkpoint, device='cuda', max_length=128):
        if checkpoint is None:
            raise ValueError('--checkpoint is required for --scorer qwen-heads')
        checkpoint = Path(checkpoint)
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        program_state = 'label_spaces' not in ckpt
        if program_state:
            schema_path = Path(str(checkpoint) + '.schema.json')
            if not schema_path.exists():
                raise ValueError(f'DomiKnowS program checkpoint requires schema file: {schema_path}')
            spaces = json.loads(schema_path.read_text())
            ckpt_args = {}
        else:
            spaces = ckpt['label_spaces']
            ckpt_args = ckpt.get('args', {})
        if program_state and not ckpt_args:
            # Older DomiKnowS program checkpoints do not store CLI args. Infer
            # whether the saved shared learner used PEFT/LoRA from its key names;
            # otherwise a non-LoRA checkpoint is loaded into a LoRA-wrapped model
            # and hundreds of weights are silently dropped.
            uses_lora = any(".lora_A." in key or "base_model.model" in key for key in ckpt)
            ckpt_args = {
                "lora_r": 4 if uses_lora else 0,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
                "lora_target_modules": "q_proj,v_proj",
            }
        self.model = GraphQAPredicateClassifier(
            model_path=model_path,
            object_symbol_labels=spaces['object_symbol'],
            symbol_pair_labels=spaces['symbol_pair'],
            object_pair_labels=spaces['object_pair'],
            device=device,
            freeze_backbone=True,
            lora_r=ckpt_args.get('lora_r', 4),
            lora_alpha=ckpt_args.get('lora_alpha', 8),
            lora_dropout=ckpt_args.get('lora_dropout', 0.05),
            lora_target_modules=str(ckpt_args.get('lora_target_modules', 'q_proj,v_proj')).split(','),
            max_length=max_length,
            encode_batch_size=16,
        )
        if program_state:
            prefix = 'graphqa/object_symbol_pair/<object_symbol_relation>/modulelearner.shared.'
            state = {key[len(prefix):]: value for key, value in ckpt.items() if key.startswith(prefix)}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f'loaded_domiknows_program_checkpoint={checkpoint}', flush=True)
            print(f'program_checkpoint_missing={len(missing)} unexpected={len(unexpected)}', flush=True)
        else:
            self.model.object_symbol_head.load_state_dict(ckpt['object_symbol_head'])
            self.model.symbol_pair_head.load_state_dict(ckpt['symbol_pair_head'])
            self.model.object_pair_head.load_state_dict(ckpt['object_pair_head'])
            if 'backbone_lora' in ckpt:
                from peft import set_peft_model_state_dict

                set_peft_model_state_dict(self.model.backbone, ckpt['backbone_lora'])
        self.model.eval()

    def bind(self, instance):
        return BoundQwenHeadsScorer(self, instance)


class BoundQwenHeadsScorer:
    def __init__(self, base: QwenHeadsScorer, instance):
        self.base = base
        self.instance = instance
        self._object_symbol_cache = {}
        self._object_pair_cache = {}

    def preload_object_symbols(self, objects, symbols):
        pairs = [(str(o), str(sym)) for o in objects for sym in symbols]
        missing = [pair for pair in pairs if ('Name', pair[0], pair[1]) not in self._object_symbol_cache]
        if not missing:
            return
        examples = []
        labels = self.base.model.object_symbol_labels
        for obj, symbol in missing:
            prompt = _object_symbol_feature_prompt(self.instance, obj, symbol, self.instance.get('query', {}), labels)
            examples.append({'kind': 'object_symbol', 'prompt': prompt})
        with torch.no_grad():
            logits = self.base.model.forward_examples(examples)
            probs = torch.softmax(logits.float(), dim=-1).detach().cpu()
        labels = self.base.model.object_symbol_labels
        for (obj, symbol), row in zip(missing, probs):
            for idx, label in enumerate(labels):
                self._object_symbol_cache[(label, obj, symbol)] = float(row[idx].item())

    @torch.no_grad()
    def object_symbol(self, pred, obj, symbol):
        key = (canonical_relation(pred), str(obj), str(symbol))
        if key not in self._object_symbol_cache:
            self.preload_object_symbols([obj], [symbol])
        return self._object_symbol_cache.get(key, 0.0)

    @torch.no_grad()
    def object_pair(self, pred, src, dst):
        key = (canonical_relation(pred), str(src), str(dst))
        if key in self._object_pair_cache:
            return self._object_pair_cache[key]
        labels = self.base.model.object_pair_labels
        if key[0] not in labels:
            self._object_pair_cache[key] = 0.0
            return 0.0
        prompt = _object_pair_feature_prompt(self.instance, src, dst, self.instance.get('query', {}), labels)
        logits = self.base.model.forward_examples([{'kind': 'object_pair', 'prompt': prompt}])[0]
        probs = torch.softmax(logits.float(), dim=-1)
        score = float(probs[labels.index(key[0])].item())
        self._object_pair_cache[key] = score
        return score

def _image_cache_path(instance, cache_dir):
    image_id = instance.get('source_image_id') or instance.get('image_id')
    return Path(cache_dir) / f'{image_id}.jpg'


def _load_image_for_instance(instance, cache_dir):
    from PIL import Image
    path = _image_cache_path(instance, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        import requests
        url = None
        metadata = instance.get('object_metadata') or {}
        for item in metadata.values():
            if item.get('image_url'):
                url = item['image_url']
                break
        if not url:
            raise FileNotFoundError(f'No cached image and no image_url for {path}')
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    return Image.open(path).convert('RGB')


def _draw_object_boxes(image, instance, boxes):
    from PIL import ImageDraw
    image = image.copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    colors = ['red', 'blue', 'lime', 'yellow']
    metadata = instance.get('object_metadata') or {}
    for idx, obj in enumerate(boxes):
        meta = metadata.get(str(obj), {})
        box = meta.get('bbox')
        if not box:
            continue
        x, y, bw, bh = [float(v) for v in box]
        xyxy = [x * w, y * h, (x + bw) * w, (y + bh) * h]
        color = colors[idx % len(colors)]
        draw.rectangle(xyxy, outline=color, width=3)
        draw.text((xyxy[0], max(0, xyxy[1] - 12)), f'{obj}', fill=color)
    return image


class QwenVLLogprobScorer:
    """Image-based yes/no atomic predicate scorer for execution baselines."""

    def __init__(self, model_path, device='cuda', max_length=2048, image_cache=None, draw_boxes=True, temperature=1.0):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {'trust_remote_code': True, 'low_cpu_mem_usage': True}
        if str(device).startswith('cuda'):
            kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs).to(device)
        self.model.eval()
        self.process_vision_info = process_vision_info
        self.device = device
        self.max_length = int(max_length)
        self.image_cache = image_cache or Path('/egr/research-hlr2/premsrit/VQAR_data/image_cache')
        self.draw_boxes = bool(draw_boxes)
        self.temperature = max(float(temperature), 1e-6)
        self._cache = {}

    def bind(self, instance):
        return BoundQwenVLScorer(self, instance)

    def _messages(self, image, question):
        return [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': question},
            ],
        }]

    @torch.no_grad()
    def yes_probability(self, image, question):
        key = (id(image), question)
        if key in self._cache:
            return self._cache[key]
        messages = self._messages(image, question)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt').to(self.device)
        out = self.model(**inputs, return_dict=True)
        last = inputs['attention_mask'].sum(dim=1) - 1
        logits = out.logits[torch.arange(out.logits.shape[0], device=self.device), last, :][0]
        toks = ['No', 'Yes']
        ids = [self.processor.tokenizer.convert_tokens_to_ids(tok) for tok in toks]
        if any(idx is None or idx < 0 for idx in ids):
            toks = [' no', ' yes']
            ids = [self.processor.tokenizer.encode(tok, add_special_tokens=False)[-1] for tok in toks]
        score = logits[ids].float() / self.temperature
        prob = float(torch.softmax(score, dim=0)[1].detach().cpu().item())
        self._cache[key] = prob
        return prob

    @torch.no_grad()
    def choice_probabilities(self, image, question, choices):
        if not choices:
            return {}
        labels = [chr(ord('A') + index) for index in range(len(choices))]
        options = '\n'.join(f'{label}. {choice}' for label, choice in zip(labels, choices))
        prompt = f'{question}\n{options}\nAnswer with one option letter only.\nAnswer:'
        key = (id(image), prompt)
        if key in self._cache:
            return self._cache[key]
        messages = self._messages(image, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt').to(self.device)
        out = self.model(**inputs, return_dict=True)
        last = inputs['attention_mask'].sum(dim=1) - 1
        logits = out.logits[torch.arange(out.logits.shape[0], device=self.device), last, :][0]
        token_ids = [self.processor.tokenizer.encode(label, add_special_tokens=False)[-1] for label in labels]
        probabilities = torch.softmax(logits[token_ids].float() / self.temperature, dim=0).detach().cpu().tolist()
        result = dict(zip(choices, probabilities))
        self._cache[key] = result
        return result


class BoundQwenVLScorer:
    def __init__(self, base: QwenVLLogprobScorer, instance):
        self.base = base
        self.instance = instance
        self.max_candidate_symbols = 128
        self._image = _load_image_for_instance(instance, base.image_cache)
        self._cache = {}

    def object_symbol(self, pred, obj, symbol):
        pred = canonical_relation(pred)
        key = ('object_symbol', pred, str(obj), str(symbol))
        if key in self._cache:
            return self._cache[key]
        image = _draw_object_boxes(self._image, self.instance, [obj]) if self.base.draw_boxes else self._image
        prompt = '\n'.join([
            'Answer Yes or No only.',
            'The image shows object bounding boxes. The target object id is highlighted.',
            f'Predicate: {pred}({obj}, {symbol})',
            f'Question: Does target object {obj} have visual concept "{symbol}" as {pred}?',
            'Answer:',
        ])
        score = self.base.yes_probability(image, prompt)
        self._cache[key] = score
        return score

    def object_pair(self, pred, src, dst):
        pred = canonical_relation(pred)
        key = ('object_pair', pred, str(src), str(dst))
        if key in self._cache:
            return self._cache[key]
        image = _draw_object_boxes(self._image, self.instance, [src, dst]) if self.base.draw_boxes else self._image
        prompt = '\n'.join([
            'Answer Yes or No only.',
            'The image shows object bounding boxes. Source object is the first highlighted box; destination object is the second highlighted box.',
            f'Predicate: {pred}({src}, {dst})',
            f'Question: Is object {src} in relation "{pred}" to object {dst}?',
            'Answer:',
        ])
        score = self.base.yes_probability(image, prompt)
        self._cache[key] = score
        return score


class BoundGroupedQwenVLScorer(BoundQwenVLScorer):
    """Group leaf names only with mutually exclusive siblings from the KB."""

    def __init__(self, base, instance, group_size=12):
        super().__init__(base, instance)
        self.group_size = max(2, min(int(group_size), 25))
        if not hasattr(base, '_type_parents'):
            base._type_parents = {}
            base._type_children = {}
            for pred, child, parent in instance.get('kb_facts', []):
                if canonical_relation(pred) != 'TypeOf':
                    continue
                child, parent = str(child), str(parent)
                base._type_parents.setdefault(child, set()).add(parent)
                base._type_children.setdefault(parent, set()).add(child)
        self.parents = base._type_parents
        self.children = base._type_children

    def _sibling_choices(self, symbol):
        """Return leaf siblings under the most specific usable parent."""
        symbol = str(symbol)
        if symbol in self.children:
            return [symbol, 'none of these']
        aliases = set(alias_values('SemanticClass', symbol))
        groups = []
        for parent in self.parents.get(symbol, ()):
            siblings = {
                value for value in self.children.get(parent, ())
                if value not in aliases and value not in self.children
            }
            if siblings:
                groups.append((len(siblings), parent, siblings))
        if not groups:
            return [symbol, 'none of these']
        _size, _parent, candidates = min(groups, key=lambda item: (item[0], item[1]))
        ordered = [symbol] + sorted(candidates)
        return ordered[:self.group_size] + ['none of these']

    def preload_predicates(self, objects, name_symbols, attr_symbols):
        names = sorted(set(str(symbol) for symbol in name_symbols))
        for obj in objects:
            image = _draw_object_boxes(self._image, self.instance, [obj]) if self.base.draw_boxes else self._image
            groups = {}
            for symbol in names:
                choices = self._sibling_choices(symbol)
                groups.setdefault(tuple(choices), set()).add(symbol)
            for choices_tuple, targets in groups.items():
                choices = list(choices_tuple)
                question = '\n'.join([
                    'The target object is highlighted in the image.',
                    'Select the most specific object name. The choices are mutually exclusive siblings in a taxonomy.',
                ])
                probabilities = self.base.choice_probabilities(image, question, choices)
                for symbol in targets:
                    self._cache[('object_symbol', 'Name', str(obj), symbol)] = probabilities[symbol]

    def object_symbol(self, pred, obj, symbol):
        key = ('object_symbol', canonical_relation(pred), str(obj), str(symbol))
        if key not in self._cache and canonical_relation(pred) == 'Name':
            self.preload_predicates([str(obj)], [str(symbol)], [])
        return super().object_symbol(pred, obj, symbol)


class QwenVLGroupedScorer(QwenVLLogprobScorer):
    def __init__(self, *args, group_size=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size

    def bind(self, instance):
        return BoundGroupedQwenVLScorer(self, instance, self.group_size)


def _norm_payload(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return str(value)


def _scene_text(instance):
    facts = instance.get('visual_facts', [])
    if not facts:
        return ''
    shown = ', '.join(f'{p}({l},{r})' for p, l, r in facts[:120])
    return f'Scene facts: {shown}'


def index_kb(facts):
    by_src = {}
    by_rel_dst = {}
    for pred, left, right in facts:
        pred = canonical_relation(pred)
        left = str(left)
        right = str(right)
        if pred == 'TypeOf':
            by_src.setdefault(left, set()).add(right)
        by_rel_dst.setdefault((pred, right), set()).add(left)
    return by_src, by_rel_dst



_REVERSE_TYPE_INDEX_CACHE = {}


def _reverse_type_index(kb_by_src):
    cache_key = id(kb_by_src)
    cached = _REVERSE_TYPE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    reverse_type = {}
    for src, dsts in kb_by_src.items():
        for dst in dsts:
            reverse_type.setdefault(str(dst), set()).add(str(src))
    _REVERSE_TYPE_INDEX_CACHE[cache_key] = reverse_type
    return reverse_type


def _type_sources_for_targets(kb_by_src, targets, kb_depth=2):
    reverse_type = _reverse_type_index(kb_by_src)
    out = {str(target) for target in targets if target is not None}
    frontier = set(out)
    for _ in range(max(0, int(kb_depth))):
        new_frontier = set()
        for dst in frontier:
            for src in reverse_type.get(dst, set()):
                if src not in out:
                    out.add(src)
                    new_frontier.add(src)
        frontier = new_frontier
        if not frontier:
            break
    return out


def _semantic_symbols(kb_by_src, symbol, kb_depth=2):
    return _type_sources_for_targets(kb_by_src, alias_values('SemanticClass', symbol), kb_depth=kb_depth)


def _kg_destination_symbols(symbol):
    values = alias_values('SemanticClass', symbol)
    values.extend(alias_values('Attribute', symbol))
    return {str(value) for value in values if value is not None}


def needed_candidate_symbols(instance, kb_by_src, kb_depth=2, include_scene_names=False):
    def add_type_sources(symbol, out):
        out.update(_type_sources_for_targets(kb_by_src, [symbol], kb_depth=kb_depth))

    query = instance.get('query', {})
    groups = []
    if query.get('alternatives'):
        groups.extend(query.get('alternatives') or [])
    else:
        groups.append(query.get('conditions', []) or [])
    name_symbols = set()
    attr_symbols = set()
    target = query.get('target_type')
    if target and target != '__any_object__':
        for alias in alias_values('SemanticClass', target):
            add_type_sources(alias, name_symbols)
    for conditions in groups:
        for pred, _left, right in conditions:
            pred = canonical_relation(pred)
            if pred == 'Name':
                name_symbols.add(str(right))
            elif pred == 'Attribute':
                attr_symbols.update(str(x) for x in alias_values('Attribute', right))
            elif pred in {'ObjectType', 'ObjectCategory', 'SemanticClass'}:
                for alias in alias_values('SemanticClass' if pred == 'SemanticClass' else pred, right):
                    name_symbols.add(str(alias))
                    add_type_sources(alias, name_symbols)
            elif pred == 'KG':
                rel, dst = right
                dst_aliases = _kg_destination_symbols(dst)
                for _kg_pred, left, kg_dst in instance.get('kb_facts', []):
                    if canonical_relation(_kg_pred) == canonical_relation(rel) and str(kg_dst) in dst_aliases:
                        name_symbols.add(str(left))
                        add_type_sources(left, name_symbols)
    # Gold scene names are available only to the explicit oracle scorer.
    if include_scene_names:
        for pred, _obj, sym in instance.get('visual_facts', []):
            if canonical_relation(pred) == 'Name' and len(name_symbols) < 128:
                name_symbols.add(str(sym))
    return name_symbols, attr_symbols

def closure_scores(instance, scorer, kb_by_src, kb_depth=2, max_candidate_symbols=128):
    objects = [str(o) for o in instance.get('objects', [])]
    name_symbols, attr_symbols = needed_candidate_symbols(
        instance, kb_by_src, kb_depth=kb_depth, include_scene_names=isinstance(scorer, OracleScorer)
    )
    if max_candidate_symbols and len(name_symbols) > max_candidate_symbols:
        name_symbols = set(sorted(name_symbols)[:max_candidate_symbols])
    if max_candidate_symbols and len(attr_symbols) > max_candidate_symbols:
        attr_symbols = set(sorted(attr_symbols)[:max_candidate_symbols])
    if hasattr(scorer, 'preload_predicates'):
        scorer.preload_predicates(objects, sorted(name_symbols), sorted(attr_symbols))
    elif hasattr(scorer, 'preload_object_symbols'):
        scorer.preload_object_symbols(objects, sorted(name_symbols | attr_symbols))
    name = {(o, s): scorer.object_symbol('Name', o, s) for o in objects for s in sorted(name_symbols)}
    attr = {(o, s): scorer.object_symbol('Attribute', o, s) for o in objects for s in sorted(attr_symbols)}
    obj_type: Dict[Tuple[str, str], float] = {}
    obj_category: Dict[Tuple[str, str], float] = {}
    for o in objects:
        for src in sorted(name_symbols):
            base = name.get((o, src), 0.0)
            if base <= 0:
                continue
            for mid in kb_by_src.get(src, ()):
                obj_type[(o, mid)] = max(obj_type.get((o, mid), 0.0), base)
                if kb_depth >= 2:
                    for dst in kb_by_src.get(mid, ()):
                        obj_category[(o, dst)] = max(obj_category.get((o, dst), 0.0), base)
    return name, attr, obj_type, obj_category


def evaluate_instance(instance, scorer, kb_by_src, kb_by_rel_dst, kb_depth=2):
    query = instance['query']
    objects = [str(o) for o in instance.get('objects', [])]
    object_rels = collect_object_relations(instance)
    name, attr, obj_type, obj_category = closure_scores(instance, scorer, kb_by_src, kb_depth=kb_depth, max_candidate_symbols=getattr(scorer, 'max_candidate_symbols', 128))

    def object_symbol_score(pred, obj, sym):
        vals = []
        for alias in alias_values(pred, sym):
            if pred == 'Name':
                vals.append(name.get((obj, alias), 0.0))
            elif pred == 'Attribute':
                vals.append(attr.get((obj, alias), 0.0))
            elif pred == 'ObjectType':
                vals.append(obj_type.get((obj, alias), 0.0))
            elif pred == 'ObjectCategory':
                vals.append(obj_category.get((obj, alias), 0.0))
        return max(vals) if vals else 0.0

    def semantic_score(obj, sym):
        vals = []
        for alias in _semantic_symbols(kb_by_src, sym, kb_depth=kb_depth):
            vals.extend([
                name.get((obj, alias), 0.0),
                obj_type.get((obj, alias), 0.0),
                obj_category.get((obj, alias), 0.0),
            ])
        return max(vals) if vals else 0.0

    def kg_score(obj, payload):
        rel, dst = payload
        rel = canonical_relation(rel)
        dst_aliases = _kg_destination_symbols(dst)
        sources = set()
        for dst_alias in dst_aliases:
            sources.update(kb_by_rel_dst.get((rel, str(dst_alias)), set()))
        vals = []
        for src in sources:
            vals.extend([name.get((obj, src), 0.0), obj_type.get((obj, src), 0.0), obj_category.get((obj, src), 0.0)])
        return max(vals) if vals else 0.0

    @lru_cache(maxsize=None)
    def rel_score(pred, src, dst):
        return scorer.object_pair(pred, src, dst)

    def condition_score(obj, condition):
        pred, left, right = condition
        pred = canonical_relation(pred)
        if left != 'o':
            return 0.0
        if pred in {'Name', 'Attribute', 'ObjectType', 'ObjectCategory'}:
            return object_symbol_score(pred, obj, str(right))
        if pred == 'SemanticClass':
            return semantic_score(obj, str(right))
        if pred == 'KG':
            return kg_score(obj, right)
        if pred == 'RelationFrom':
            rel, anchors = right
            rel = canonical_relation(rel)
            if rel in SYMMETRIC_OBJECT_RELATIONS:
                return max([max(rel_score(rel, obj, str(anchor)), rel_score(rel, str(anchor), obj)) for anchor in anchors] or [0.0])
            return max([rel_score(rel, str(anchor), obj) for anchor in anchors] or [0.0])
        if pred == 'RelationTo':
            rel, anchors = right
            rel = canonical_relation(rel)
            if rel in SYMMETRIC_OBJECT_RELATIONS:
                return max([max(rel_score(rel, obj, str(anchor)), rel_score(rel, str(anchor), obj)) for anchor in anchors] or [0.0])
            return max([rel_score(rel, obj, str(anchor)) for anchor in anchors] or [0.0])
        if pred == 'OneOf':
            return 1.0 if obj in {str(x) for x in right} else 0.0
        if pred in object_rels:
            return rel_score(pred, obj, str(right))
        return 0.0

    def branch_score(obj, conditions):
        score = 1.0
        target = query.get('target_type')
        if target and target != '__any_object__':
            score *= object_symbol_score('ObjectCategory', obj, str(target))
        for condition in conditions:
            score *= condition_score(obj, condition)
        return score

    alternatives = query.get('alternatives') or [query.get('conditions', [])]
    scores = {obj: max(branch_score(obj, branch) for branch in alternatives) for obj in objects}
    return scores



def evaluate_instance_ilp(instance, scorer, kb_by_src, kb_by_rel_dst, kb_depth=2, max_ilp_vars=50000, max_ilp_constraints=200000, global_consistency=False):
    """Predicate-level ILP decoding for the bounded GraphQA executable.

    The local executor scores each candidate object by multiplying soft predicate
    probabilities. This decoder instead creates Boolean variables for relevant
    atomic predicates, imposes bounded TypeOf propagation as hard linear
    constraints, and selects exactly one executable answer object.
    """
    query = instance['query']
    objects = [str(o) for o in instance.get('objects', [])]
    object_rels = collect_object_relations(instance)
    name_symbols, attr_symbols = needed_candidate_symbols(
        instance, kb_by_src, kb_depth=kb_depth, include_scene_names=isinstance(scorer, OracleScorer)
    )
    max_candidate_symbols = getattr(scorer, 'max_candidate_symbols', 128)
    if max_candidate_symbols and len(name_symbols) > max_candidate_symbols:
        name_symbols = set(sorted(name_symbols)[:max_candidate_symbols])
    if max_candidate_symbols and len(attr_symbols) > max_candidate_symbols:
        attr_symbols = set(sorted(attr_symbols)[:max_candidate_symbols])
    if hasattr(scorer, 'preload_object_symbols'):
        scorer.preload_object_symbols(objects, sorted(name_symbols | attr_symbols))

    var_index = {}
    costs = []

    def add_var(key, prob=0.5):
        if key in var_index:
            return var_index[key]
        idx = len(costs)
        var_index[key] = idx
        p = float(prob)
        if not math.isfinite(p):
            p = 0.5
        p = min(max(p, 1e-5), 1.0 - 1e-5)
        costs.append(-math.log(p / (1.0 - p)))
        return idx

    def add_aux(key):
        return add_var(key, 0.5)

    for obj in objects:
        for sym in sorted(name_symbols):
            add_var(('Name', obj, sym), scorer.object_symbol('Name', obj, sym))
        for sym in sorted(attr_symbols):
            add_var(('Attribute', obj, sym), scorer.object_symbol('Attribute', obj, sym))

    def relation_var(pred, src, dst):
        pred = canonical_relation(pred)
        return add_var((pred, str(src), str(dst)), scorer.object_pair(pred, str(src), str(dst)))

    rows = []
    lbs = []
    ubs = []

    def constrain(coeffs, lb=-math.inf, ub=math.inf):
        rows.append(dict(coeffs))
        lbs.append(lb)
        ubs.append(ub)

    def le(left, rights):
        coeffs = {left: 1.0}
        for right in rights:
            coeffs[right] = coeffs.get(right, 0.0) - 1.0
        constrain(coeffs, ub=0.0)

    def ge_and(out, inputs):
        if not inputs:
            constrain({out: 1.0}, lb=1.0, ub=1.0)
            return
        for inp in inputs:
            constrain({out: 1.0, inp: -1.0}, ub=0.0)
        coeffs = {out: 1.0}
        for inp in inputs:
            coeffs[inp] = coeffs.get(inp, 0.0) - 1.0
        constrain(coeffs, lb=1.0 - len(inputs))

    def eq_or(out, inputs):
        if not inputs:
            constrain({out: 1.0}, lb=0.0, ub=0.0)
            return
        for inp in inputs:
            constrain({out: 1.0, inp: -1.0}, lb=0.0)
        le(out, inputs)

    if global_consistency:
        # Optional hard grounding consistency: every object has exactly one
        # visual Name assignment among the query-relevant candidate symbols.
        for obj in objects:
            name_vars = [add_var(('Name', obj, sym), scorer.object_symbol('Name', obj, sym)) for sym in sorted(name_symbols)]
            if name_vars:
                constrain({idx: 1.0 for idx in name_vars}, lb=1.0, ub=1.0)

    # Bounded KG propagation: Name -> ObjectType -> ObjectCategory.
    for obj in objects:
        type_sources = {}
        category_sources = {}
        for src in sorted(name_symbols):
            name_v = add_var(('Name', obj, src), scorer.object_symbol('Name', obj, src))
            for mid in kb_by_src.get(src, ()):
                type_v = add_aux(('ObjectType', obj, str(mid)))
                type_sources.setdefault(str(mid), []).append(name_v)
                constrain({type_v: 1.0, name_v: -1.0}, lb=0.0)
                if kb_depth >= 2:
                    for dst in kb_by_src.get(mid, ()):
                        cat_v = add_aux(('ObjectCategory', obj, str(dst)))
                        category_sources.setdefault(str(dst), []).append(type_v)
                        constrain({cat_v: 1.0, type_v: -1.0}, lb=0.0)
        for sym, sources in type_sources.items():
            le(add_aux(('ObjectType', obj, sym)), sources)
        for sym, sources in category_sources.items():
            le(add_aux(('ObjectCategory', obj, sym)), sources)

    def object_symbol_vars(pred, obj, sym):
        vals = []
        for alias in alias_values(pred, sym):
            alias = str(alias)
            if pred == 'Name' and alias in name_symbols:
                vals.append(add_var(('Name', obj, alias), scorer.object_symbol('Name', obj, alias)))
            elif pred == 'Attribute' and alias in attr_symbols:
                vals.append(add_var(('Attribute', obj, alias), scorer.object_symbol('Attribute', obj, alias)))
            elif pred in {'ObjectType', 'ObjectCategory'}:
                # Derived KB predicates are valid only when bounded propagation
                # created and constrained the variable. Creating a fresh aux
                # here would let ILP satisfy any query with an unconstrained fact.
                key = (pred, obj, alias)
                if key in var_index:
                    vals.append(var_index[key])
        return vals

    def condition_var(obj, condition, key_prefix):
        pred, left, right = condition
        pred = canonical_relation(pred)
        if left != 'o':
            v = add_aux((key_prefix, 'false'))
            constrain({v: 1.0}, lb=0.0, ub=0.0)
            return v
        alternatives = []
        if pred in {'Name', 'Attribute', 'ObjectType', 'ObjectCategory'}:
            alternatives.extend(object_symbol_vars(pred, obj, str(right)))
        elif pred == 'SemanticClass':
            for alias in _semantic_symbols(kb_by_src, right, kb_depth=kb_depth):
                alternatives.extend(object_symbol_vars('Name', obj, str(alias)))
                alternatives.extend(object_symbol_vars('ObjectType', obj, str(alias)))
                alternatives.extend(object_symbol_vars('ObjectCategory', obj, str(alias)))
        elif pred == 'KG':
            rel, dst = right
            rel = canonical_relation(rel)
            sources = set()
            for dst_alias in _kg_destination_symbols(dst):
                sources.update(kb_by_rel_dst.get((rel, str(dst_alias)), set()))
            for src in sources:
                alternatives.extend(object_symbol_vars('Name', obj, src))
                alternatives.extend(object_symbol_vars('ObjectType', obj, src))
                alternatives.extend(object_symbol_vars('ObjectCategory', obj, src))
        elif pred == 'RelationFrom':
            rel, anchors = right
            alternatives.extend(relation_var(rel, str(anchor), obj) for anchor in anchors)
        elif pred == 'RelationTo':
            rel, anchors = right
            alternatives.extend(relation_var(rel, obj, str(anchor)) for anchor in anchors)
        elif pred == 'OneOf':
            v = add_aux((key_prefix, 'oneof'))
            value = 1.0 if obj in {str(x) for x in right} else 0.0
            constrain({v: 1.0}, lb=value, ub=value)
            return v
        elif pred in object_rels:
            alternatives.append(relation_var(pred, obj, str(right)))
        out = add_aux((key_prefix, 'cond'))
        eq_or(out, alternatives)
        return out

    answer_vars = []
    for obj in objects:
        branch_vars = []
        branches = query.get('alternatives') or [query.get('conditions', [])]
        for branch_idx, conditions in enumerate(branches):
            atoms = []
            target = query.get('target_type')
            if target and target != '__any_object__':
                atoms.extend(object_symbol_vars('ObjectCategory', obj, str(target)))
            for cond_idx, condition in enumerate(conditions):
                atoms.append(condition_var(obj, condition, ('condition', obj, branch_idx, cond_idx)))
            branch_v = add_aux(('branch', obj, branch_idx))
            ge_and(branch_v, atoms)
            branch_vars.append(branch_v)
        member_v = add_aux(('member', obj))
        eq_or(member_v, branch_vars)
        answer_v = add_var(('answer', obj), 0.5)
        constrain({answer_v: 1.0, member_v: -1.0}, ub=0.0)
        # Small answer bonus makes the selected answer prefer satisfiable members;
        # predicate log-odds still dominate which facts are turned on.
        costs[answer_v] = -2.0
        answer_vars.append(answer_v)

    constrain({idx: 1.0 for idx in answer_vars}, lb=1.0, ub=1.0)
    n = len(costs)
    if max_ilp_vars and n > max_ilp_vars:
        return None
    if max_ilp_constraints and len(rows) > max_ilp_constraints:
        return None

    # Keep the ILP matrix sparse. A dense rows x variables matrix can explode
    # into hundreds of GB for GraphQA examples with many candidate symbols.
    if rows:
        row_ids = []
        col_ids = []
        data = []
        for row_id, row in enumerate(rows):
            for idx, value in row.items():
                row_ids.append(row_id)
                col_ids.append(idx)
                data.append(float(value))
        matrix = coo_matrix((data, (row_ids, col_ids)), shape=(len(rows), n)).tocsr()
        constraints = LinearConstraint(matrix, lbs, ubs)
    else:
        constraints = None
    result = milp(
        c=costs,
        integrality=[1] * n,
        bounds=Bounds([0.0] * n, [1.0] * n),
        constraints=constraints,
        options={'time_limit': 10.0, 'mip_rel_gap': 0.01},
    )
    # HiGHS reports success=False when the time limit is reached even when it
    # has found a feasible integer incumbent. The incumbent still satisfies
    # the hard constraints and is valid for constrained decoding.
    if result.x is None:
        return None
    assignment = result.x
    return {obj: float(assignment[var_index[('answer', obj)]]) for obj in objects}


def topk_recall(scores, gold, topk=5):
    if not gold:
        return math.nan
    ranked = [obj for obj, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topk]]
    return sum(1 for obj in gold if obj in ranked) / min(len(gold), topk)


def main():
    args = parse_args()
    tasks = load_vqar_tasks(args.task_path, limit=args.limit)
    kb_facts = load_kb_facts(args.kb_dir)
    kb_by_src, kb_by_rel_dst = index_kb(kb_facts)
    qwen = None
    if args.scorer == 'qwen-logprob':
        qwen = QwenLogprobScorer(args.model_path, args.device, args.max_length, args.include_scene_facts, args.temperature)
    elif args.scorer == 'qwen-heads':
        qwen = QwenHeadsScorer(args.model_path, args.checkpoint, args.device, args.max_length)
    elif args.scorer == 'qwen-vl-logprob':
        qwen = QwenVLLogprobScorer(args.model_path, args.device, args.max_length, args.image_cache, args.draw_boxes, args.temperature)
    elif args.scorer == 'qwen-vl-grouped':
        qwen = QwenVLGroupedScorer(args.model_path, args.device, args.max_length, args.image_cache, args.draw_boxes, args.temperature, group_size=args.group_size)

    total = 0
    exact = 0
    recalls = []
    unsupported = 0
    first_logic = None
    for task in tqdm(tasks, desc='Scallop-style GraphQA'):
        try:
            instance = vqar_task_to_graphqa_instance(task, kb_facts=kb_facts)
            if args.single_answer_only and len(instance.get('expected_answers', []) or []) != 1:
                continue
            if first_logic is None:
                first_logic = create_query_logic(instance)
            scorer = OracleScorer(instance) if args.scorer == 'oracle' else qwen.bind(instance)
            setattr(scorer, 'max_candidate_symbols', args.max_candidate_symbols)
            if args.inference == 'ilp':
                scores = evaluate_instance_ilp(instance, scorer, kb_by_src, kb_by_rel_dst, kb_depth=args.kb_depth, max_ilp_vars=args.max_ilp_vars, max_ilp_constraints=args.max_ilp_constraints, global_consistency=args.global_consistency)
                if scores is None:
                    unsupported += 1
                    continue
            else:
                scores = evaluate_instance(instance, scorer, kb_by_src, kb_by_rel_dst, kb_depth=args.kb_depth)
        except Exception:
            unsupported += 1
            continue
        gold = [str(x) for x in instance.get('expected_answers', [])]
        if not gold:
            continue
        positive_scores = {obj: score for obj, score in scores.items() if score > 0.0}
        ranked_scores = positive_scores if positive_scores else scores
        pred = max(ranked_scores, key=ranked_scores.get) if ranked_scores else None
        exact += int(pred in set(gold))
        recalls.append(topk_recall(ranked_scores, gold, args.topk))
        total += 1
    mean_recall = sum(r for r in recalls if not math.isnan(r)) / len(recalls) if recalls else 0.0
    print(f'scorer={args.scorer}')
    print(f'inference={args.inference}')
    print(f'global_consistency={args.global_consistency}')
    print(f'examples={len(tasks)} evaluated={total} unsupported={unsupported}')
    print(f'answer_acc={exact / total if total else 0.0:.6f}')
    print(f'recall_at_{args.topk}={mean_recall:.6f}')
    if first_logic:
        print('example_logic_str=')
        print(first_logic)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
