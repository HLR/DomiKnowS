"""Consistent source training pointers; numeric measurements are never relabeled."""
from copy import deepcopy
from dataclasses import replace
from .permutation import EntityPermutation
from .protocol import sha


def transform(example, record, *, epoch, augmented, pointer_keys):
    if example.episode_id != record['episode_id'] or tuple(example.entities) != tuple(record['canonical_entities']):
        raise ValueError('source identity or canonical entity ordering differs')
    if sha(record['scene_path']) != record['scene_sha256']:
        raise ValueError('native source scene changed')
    setting = record['augmentations'][epoch] if augmented else None
    permutation = EntityPermutation(setting['new_to_old'] if setting else tuple(range(len(example.entities))))
    image = setting['image'] if setting else record['canonical_image']
    if sha(image) != record['files'][image]: raise ValueError('source marker image changed')
    names = tuple(example.entities[i] for i in permutation.order)
    def pointer(value):
        if type(value) is int or (isinstance(value, str) and value.isdigit()):
            return permutation.encode_pointer(int(value))
        if value not in example.entities: raise ValueError('unknown canonical pointer')
        return value
    operations = deepcopy(example.operation_sequence)
    for operation in operations:
        for key in set(operation['params']) & set(pointer_keys):
            operation['params'][key] = pointer(operation['params'][key])
    def optional(value): return None if value is None else pointer(value)
    changes = dict(entities=names, operation_sequence=operations,
        image_paths=(image,), segmented_image_paths=(image,),
        task_target_container=optional(example.task_target_container),
        task_target_entity=optional(example.task_target_entity))
    if example.visible_entities is not None:
        changes['visible_entities'] = tuple(pointer(v) for v in example.visible_entities)
    if example.object_candidates is not None:
        changes['object_candidates'] = tuple((skill, role, tuple(pointer(v) for v in ids))
                                            for skill, role, ids in example.object_candidates)
    return replace(example, **changes)
