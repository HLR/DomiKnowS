"""Separate model observations from optional compiled-constraint bindings.

These functions route data; they do not implement planning constraints.
"""
from copy import deepcopy
import json

from .protocol import anonymous_entities

IMAGE_INSTRUCTION = 'image-instruction-v1'
METADATA_ASSISTED = 'metadata-assisted-v1'
INPUT_PROTOCOLS = (IMAGE_INSTRUCTION, METADATA_ASSISTED)


def protocol_from_settings(settings):
    # Earlier archived runs predate this field and used the metadata prompt.
    return settings.get('input_protocol', METADATA_ASSISTED)


def generation_context(public, vocabulary, protocol=IMAGE_INSTRUCTION):
    if protocol == METADATA_ASSISTED:
        from ..protocol import context
        from ..matched_baseline.protocol import prompt_core
        result = context(public, 'libero')
        result['_matched_core'] = prompt_core(public, vocabulary)
        result['_matched_roles'] = None
        return result
    if protocol != IMAGE_INSTRUCTION:
        raise ValueError('unknown evaluation input protocol')
    if not isinstance(public['instruction'], str):
        raise TypeError('public instruction must be text')
    # Anonymous pointer bindings are retained for serialization, but neither
    # the inventory nor its size is rendered to the model. No metadata lookup.
    entities = anonymous_entities(len(public['entities']))
    core = ('Return only a JSON array of robot operations, without markdown or explanation. '
        'Each operation has keys "name" and "params". '
        '"params" maps argument names to integer entity IDs shown by numeric tags in the images.\n'
        'Operation API (names and required argument keys only): '
        + json.dumps(dict(vocabulary.skill_arguments)) + '\n'
        'Instruction: ' + public['instruction'])
    return {'instruction': public['instruction'], 'images': tuple(public['images']),
            'entity_table': entities, '_matched_core': core, '_matched_roles': None}


def constraint_public(public):
    """Observed fixture evidence only; never goals, task aliases, or role hints."""
    originals = list(public['entities'])
    if len(set(originals)) != len(originals):
        raise ValueError('ambiguous observation-to-pointer bindings')
    entities = anonymous_entities(len(originals))
    # Missing evidence means unknown, not that every object is static.
    movable = set(public.get('movable', originals))
    if not movable.issubset(set(originals)):
        raise ValueError('unbound observed movability metadata')
    return {'entities': entities,
            'movable': [entities[i] for i, name in enumerate(originals) if name in movable]}


def constraint_context(public):
    observation = constraint_public(public)
    return {'entity_table': observation['entities'],
            'generation_static_fixture_ids': [i for i, name in enumerate(observation['entities'])
                                              if name not in observation['movable']]}


def contract_record(protocol):
    if protocol not in INPUT_PROTOCOLS:
        raise ValueError('unknown evaluation input protocol')
    strict = protocol == IMAGE_INSTRUCTION
    return {'input_protocol': protocol, 'primary_role_hints': False,
        'grounding_residual': False, 'model_scene_metadata': not strict,
        'model_entity_inventory': not strict, 'model_action_explanations': not strict,
        'generic_action_api': True, 'anonymous_image_tags': True,
        'scope': ('Instruction, images and output API; observed fixture evidence only through optional compiled DFA'
                  if strict else 'Historical metadata-assisted prompt; preserve as a separate protocol')}


def validate_resume_contract(saved, requested):
    if protocol_from_settings(saved) != requested:
        raise ValueError('evaluation input protocol differs; preserve old output and use a new directory')
    return deepcopy(saved)
