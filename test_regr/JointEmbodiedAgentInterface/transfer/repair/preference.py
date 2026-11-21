"""Compile a soft source-pointer preference and bind public visual evidence."""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

from ...generalization.executor_policy import SUBJECT
from ..protocol import sha
from .protocol import anonymous_entities


VISUAL_SOURCE = 'visual-source'
PREFERENCE_RULE = 'prefer_visually_supported_pick_source'
WEIGHT = 0.5
MIN_MARGIN = 0.02
PICK_SOURCE, MATCH, LOWER, UNKNOWN = ('pick_source', 'visual_match', 'visual_lower', 'visual_unknown')


@lru_cache(maxsize=1)
def source_preference():
    """Lower a DomiKnowS adjacency formula to a finite transition potential."""
    import torch
    from domiknows.generation import HMMFactorGraphEncoder, discover_transition_potentials
    from domiknows.graph.logicalConstrain import ifL, notL

    states = (PICK_SOURCE, MATCH, LOWER, UNKNOWN)
    graph, bundle = HMMFactorGraphEncoder(
        ['<eos>'], eos_token='<eos>', state_names=states,
        include_dp_factors=False, clear_graph=False, graph_name='visual_pick_preference',
    ).build_graph()
    ctx = bundle.context
    with graph:
        ifL(ctx.is_next_rel('edge'), ifL(
            ctx.latent_state_value(PICK_SOURCE, 'source', path=('edge', ctx.current_token)),
            notL(ctx.latent_state_value(LOWER, 'candidate', path=('edge', ctx.next_token))),
        ), name=PREFERENCE_RULE)
    compiled, = discover_transition_potentials(graph, bundle, strength=math.exp(-WEIGHT),
                                                on_unsupported='error')
    row = compiled.tensor_for(torch.ones((len(states), len(states)), dtype=torch.float64))[0]
    return dict(zip(states, row.log().tolist()))


@dataclass(frozen=True)
class VisualSourcePreference:
    evidence: dict[int, str]

    def log_bias(self, skill, role, allowed, vocabulary):
        if skill != 'pick' or role != SUBJECT or not self.evidence:
            return {}
        candidates = {}
        for label in allowed:
            token = vocabulary.token_for_label(int(label))
            if token.startswith('obj:'):
                candidates[int(label)] = int(token[4:])
        if MATCH not in (self.evidence.get(pointer) for pointer in candidates.values()):
            return {}
        scores = source_preference()
        return {label: scores[self.evidence.get(pointer, UNKNOWN)] for label, pointer in candidates.items()
                if scores[self.evidence.get(pointer, UNKNOWN)] != 0.}


def bind_visual_evidence(observation, entity_count):
    """Ground evidence classes without changing entity IDs or token legality."""
    if not observation['phrases']['source']:
        return VisualSourcePreference({})
    scores = {}
    for row in observation['diagnostics']['source']['ranked_candidates']:
        pointer, score = row['id'], row['score']
        if (type(pointer) is not int or not 0 <= pointer < entity_count or pointer in scores
                or not math.isfinite(score)):
            raise ValueError('invalid visual preference score or anonymous pointer')
        scores[pointer] = score
    if len(scores) < 2:
        return VisualSourcePreference({})
    order = sorted(scores, key=scores.__getitem__, reverse=True)
    if scores[order[0]] - scores[order[1]] < MIN_MARGIN:
        return VisualSourcePreference({})
    return VisualSourcePreference({pointer: MATCH if pointer == order[0] else LOWER for pointer in scores})


class FrozenVisualSource:
    """Score anonymous RGB regions from only the evaluator's public images and instruction."""

    def __init__(self, model_path, device):
        from transformers import AutoTokenizer
        from test_regr.VLABenchAgentInterface.simulation.controller import FrozenSigLIPEncoder

        model_path = Path(model_path)
        if not model_path.is_dir():
            raise ValueError('visual preference requires a pinned local SigLIP snapshot')
        self.model_files = {path.name: sha(path) for path in sorted(model_path.iterdir())
                            if path.is_file() and path.suffix in ('.json', '.model', '.safetensors', '.bin')}
        if 'config.json' not in self.model_files or not any(
            name.endswith(('.safetensors', '.bin')) for name in self.model_files
        ):
            raise ValueError('visual preference snapshot has no model weights or configuration')
        self.encoder = FrozenSigLIPEncoder(str(model_path), local_files_only=True, image_size=384).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.device = device
        self.settings = {'mode': VISUAL_SOURCE, 'weights': self.model_files, 'rule': PREFERENCE_RULE,
                         'log_penalty': -WEIGHT, 'cosine_margin': MIN_MARGIN,
                         'compiled_log_factors': source_preference(),
                         'observation': 'instruction and public numeric-tag RGB crops only'}

    def observe(self, public):
        from PIL import Image
        from ..anonymous_grounding.eval_observations import libero_markers
        from ..anonymous_grounding.object_fixes.grounding import paired_features, phrases

        count = len(public['entities'])
        model_input = {'entities': anonymous_entities(count), 'instruction': public['instruction'],
                       'images': tuple(public['images'])}
        query = phrases(model_input['instruction'])
        if not query['source']:
            return {'phrases': query, 'diagnostics': {}, 'status': 'no_source_phrase'}
        raw, numbered = model_input['images']
        binding = libero_markers(raw, numbered, count)
        with Image.open(raw) as image:
            width, height = image.size
        if width != height:
            raise ValueError('the public LIBERO marker adapter requires square camera views')
        if len(binding['markers']) < 2:
            return {'phrases': query, 'diagnostics': {'source': {'ranked_candidates': []}},
                    'visible_ids': binding['visible_ids'], 'status': 'insufficient_public_markers'}
        markers = [{**marker, 'box': [round(value * 480 / width) for value in marker['box']]}
                   for marker in binding['markers']]
        return paired_features(self.encoder, self.tokenizer, model_input,
                               {'binding': {'markers': markers}}, self.device)
