#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/26/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
from typing import Optional, Union, Callable, Sequence

import nltk
import numpy as np
from PIL import Image

import jacinle.io as io
import jaclearn.vision.coco.mask_utils as mask_utils

# from jacinle.logging import get_logger
# from jacinle.utils.container import GView
# from jactorch.data.dataset import FilterableDatasetUnwrapped, FilterableDatasetView
# from jactorch.data.dataloader import JacDataLoader
# from jactorch.data.collate import VarLengthCollateV2

# from concepts.benchmark.common.vocab import Vocab
# from concepts.benchmark.clevr.clevr_constants import g_attribute_concepts, g_relational_concepts
import logging
logger = logging.getLogger(__name__)

__all__ = ['CLEVRDatasetUnwrapped', 'CLEVRDatasetFilterableView', 'make_dataset', 'CLEVRCustomTransferDataset', 'make_custom_transfer_dataset', 'annotate_scene', 'canonize_answer', 'annotate_objects']


class CLEVRDatasetUnwrapped(FilterableDatasetUnwrapped):
    """The unwrapped CLEVR dataset."""

    def __init__(
        self,
        scenes_json: str, questions_json: str, image_root: str,
        image_transform: Callable,
        vocab_json: Optional[str], output_vocab_json: Optional[str],
        question_transform: Optional[Callable] = None,
        incl_scene: bool = True, incl_raw_scene: bool = False, size = -1
    ):
        """Initialize the CLEVR dataset.

        Args:
            scenes_json: the path to the scenes json file.
            questions_json: the path to the questions json file.
            image_root: the root directory of the images.
            image_transform: the image transform (torchvision transform).
            vocab_json: the path to the vocab json file. If None, the vocab will be built from the dataset.
            output_vocab_json: the path to the output vocab json file. If None, the output vocab will be built from the dataset.
            question_transform: the question transform (a callable). If None, no transform will be applied.
            incl_scene: whether to include the scene annotations (e.g., objects, relationships, etc.).
            incl_raw_scene: whether to include the raw scene annotations.
        """
        super().__init__()

        self.scenes_json = scenes_json
        self.questions_json = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.vocab_json = vocab_json
        self.output_vocab_json = output_vocab_json
        self.question_transform = question_transform

        self.incl_scene = incl_scene
        self.incl_raw_scene = incl_raw_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']
        if isinstance(self.questions_json, (tuple, list)):
            self.questions = list()
            for filename in self.questions_json:
                logger.info('Loading questions from: "{}".'.format(filename))
                self.questions.extend(io.load_json(filename)['questions'])
        else:
            logger.info('Loading questions from: "{}".'.format(self.questions_json))
            self.questions = io.load_json(self.questions_json)['questions']

        if self.vocab_json is not None:
            logger.info('Loading vocab from: "{}".'.format(self.vocab_json))
            self.vocab = Vocab.from_json(self.vocab_json)
        else:
            logger.info('Building the vocab.')
            self.vocab = Vocab.from_dataset(self, keys=['question_tokenized'])

        if output_vocab_json is not None:
            logger.info('Loading output vocab from: "{}".'.format(self.output_vocab_json))
            self.output_vocab = Vocab.from_json(self.output_vocab_json)
        else:
            logger.info('Building the output vocab.')
            self.output_vocab = Vocab.from_dataset(self, keys=['answer'], single_word=True)
        if size != -1:
            self.questions = self.questions[:size]
            self.scenes = self.scenes[:size]

    def _get_metainfo(self, index):
        question = self.questions[index]
        scene = self.scenes[question['image_index']]
        question['scene'] = scene
        question['program'] = question.pop('program', None)  # In CLEVR-Humans, there is no program.

        question['image_index'] = question['image_index']
        question['image_filename'] = self._get_image_filename(scene)
        question['question_index'] = index
        question['question_tokenized'] = nltk.word_tokenize(question['question'])
        question['question_type'] = get_question_type(question['program'])

        question["all_objects"] = scene['objects']
        # question['scene_complexity'] = len(scene['objects'])
        # question['program_complexity'] = len(question['program'])

        return question

    def _get_image_filename(self, scene: dict) -> str:
        return scene['image_filename']

    def __getitem__(self, index: int) -> dict:
        """Get a sample from the dataset.

        Returns:
            a dict of annotations, including:
                - scene: the scene annotations (raw dict).
                - objects: the bounding boxes of the objects (a Tensor of shape [N, 4]).
                - image_index: the index of the image (int).
                - image_filename: the filename of the image (str).
                - image: the image (a Tensor of shape [3, H, W]).
                - question_index: the index of the question (int).
                - question_raw: the raw question (str).
                - question_raw_tokenized: the tokenized raw question (list of str).
                - question: the tokenized question, and mapped to integers (a Tensor of shape [T]).
                - question_type: the type of the question (str).
                - answer: the answer to the question (bool, int, or str).
                - attribute_{attr_name}: the attribute concept id for each object (a Tensor of shape [N]).
                - attribute_relation_{attr_name}: the attribute relation concept id for each pair of objects (a Tensor of shape [N, N], then flattened to [N * N]).
                - relation_{attr_name}: the relational concept id for each pair of objects (a Tensor of shape [N, N, NR], then flattened to [N * N * NR]).
        """
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # metainfo annotations
        if self.incl_scene:
            feed_dict.update(annotate_objects(metainfo.scene))
            if 'objects' in feed_dict:
                # NB(Jiayuan Mao): in some datasets_v1, object information might be completely unavailable.
                feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(annotate_scene(metainfo.scene))

        if self.incl_raw_scene:
            feed_dict.scene = metainfo.scene

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None and feed_dict.image_filename is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image, feed_dict.objects = self.image_transform(feed_dict.image, feed_dict.objects)

        # question
        feed_dict.question_index = metainfo.question_index
        feed_dict.question_raw = metainfo.question
        feed_dict.question_raw_tokenized = metainfo.question_tokenized
        feed_dict.question = metainfo.question_tokenized
        feed_dict.question_type = metainfo.question_type
        feed_dict.answer = canonize_answer(metainfo.answer, None)
        feed_dict.all_objects = metainfo.all_objects
        if self.question_transform is not None:
            self.question_transform(feed_dict)
        feed_dict.question = np.array(self.vocab.map_sequence(feed_dict.question), dtype=np.int64)

        return feed_dict.raw()

    def __len__(self):
        return len(self.questions)

    def retain_data(self, fraction):
        ### shuffle questions
        import random
        random.shuffle(self.questions)
        self.questions = self.questions[:int(fraction * len(self.questions))]
    
class CLEVRDatasetFilterableView(FilterableDatasetView):
    def filter_program_size_raw(self, max_length: int):
        """Filter the questions by the length of the original CLEVR programs (in terms of the number of steps)."""
        def filt(question):
            return question['program'] is None or len(question['program']) <= max_length

        return self.filter(filt, 'filter-program-size-clevr[{}]'.format(max_length))

    def filter_scene_size(self, max_scene_size: int):
        """Filter the questions by the size of the scene (in terms of the number of objects)."""
        def filt(question):
            return len(question['scene']['objects']) <= max_scene_size

        return self.filter(filt, 'filter-scene-size[{}]'.format(max_scene_size))

    def filter_question_type(self, *, allowed=None, disallowed=None):
        """Filter the questions by the question type.

        Args:
            allowed: a set of allowed question types.
            disallowed: a set of disallowed question types. Only one of `allowed` and `disallowed` can be provided.

        Returns:
            a new dataset view.
        """
        def filt(question):
            if allowed is not None:
                return question['question_type'] in allowed
            elif disallowed is not None:
                return question['question_type'] not in disallowed

        if allowed is not None:
            return self.filter(filt, 'filter-question-type[allowed={' + (','.join(list(allowed))) + '}]')
        elif disallowed is not None:
            return self.filter(filt, 'filter-question-type[disallowed={' + (','.join(list(disallowed))) + '}]')
        else:
            raise ValueError('Must provide either allowed={...} or disallowed={...}.')

    def make_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool, nr_workers: int) -> JacDataLoader:
        """Make a dataloader for this dataset view.

        Args:
            batch_size: the batch size.
            shuffle: whether to shuffle the dataset.
            drop_last: whether to drop the remaining samples that are smaller than the batch size.
            nr_workers: the number of workers for the dataloader.

        Returns:
            a JacDataLoader instance.
        """
        collate_guide = {
            'scene': 'skip',
            'objects_raw': 'skip',
            'objects': 'concat',

            'image_index': 'skip',
            'image_filename': 'skip',
            'question_index': 'skip',
            'question_metainfo': 'skip',

            'question_raw': 'skip',
            'question_raw_tokenized': 'skip',
            'question': 'pad',

            'program_raw': 'skip',
            'program_seq': 'skip',
            'program_tree': 'skip',
            'program_qsseq': 'skip',
            'program_qstree': 'skip',

            'question_type': 'skip',
            'answer': 'skip',
            'parts': 'skip',
        }

        # Scene annotations.
        for attr_name in g_attribute_concepts:
            collate_guide['attribute_' + attr_name] = 'concat'
            collate_guide['attribute_relation_' + attr_name] = 'concat'
        for attr_name in g_relational_concepts:
            collate_guide['relation_' + attr_name] = 'concat'

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


class CLEVRCustomTransferDataset(FilterableDatasetUnwrapped):
    """The unwrapped CLEVR dataset for custom transfer learning."""

    def __init__(
        self,
        scenes_json: str, questions_json: str,
        image_root: str, image_transform: Callable,
        query_list_key: str, custom_fields: Sequence[str],
        output_vocab_json: Optional[str] = None,
        incl_scene: bool = True, incl_raw_scene: bool = False, size: int = -1
    ):
        """Initialize the CLEVR custom transfer dataset.

        Args:
            scenes_json: the path to the scenes json file.
            questions_json: the path to the questions json file.
            image_root: the root directory of the images.
            image_transform: the image transform (torchvision transform).
            query_list_key: the key of the query list in the questions json file (e.g., 'questions' or 'questions_human').
            custom_fields: the custom fields to be included in the dataset. These are fields in the scene annotations.
            output_vocab_json: the path to the output vocab json file. If None, the output vocab will be built from the dataset.
            incl_scene: whether to include the scene annotations (e.g., objects, relationships, etc.).
            incl_raw_scene: whether to include the raw scene annotations.
        """
        super().__init__()

        self.scenes_json = scenes_json
        self.questions_json = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.query_list_key = query_list_key
        self.custom_fields = custom_fields
        self.output_vocab_json = output_vocab_json

        self.incl_scene = incl_scene
        self.incl_raw_scene = incl_raw_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']
        if isinstance(self.questions_json, (tuple, list)):
            self.questions = list()
            for filename in self.questions_json:
                logger.info('Loading questions from: "{}".'.format(filename))
                self.questions.extend(io.load_json(filename)[query_list_key])
        else:
            logger.info('Loading questions from: "{}".'.format(self.questions_json))
            self.questions = io.load_json(self.questions_json)[query_list_key]

        if output_vocab_json is not None:
            logger.info('Loading output vocab from: "{}".'.format(self.output_vocab_json))
            self.output_vocab = Vocab.from_json(self.output_vocab_json)
        else:
            logger.info('Building the output vocab.')
            self.output_vocab = Vocab.from_dataset(self, keys=['answer'], single_word=True)
        if size != -1:
            self.questions = self.questions[:size]
            self.scenes = self.scenes[:size]
    def _get_scene_index_from_question(self, question):
        if 'image_index' in question:
            return question['image_index']
        if 'scene_index' in question:
            return question['scene_index']
        raise KeyError('Cannot find scene index from question.')

    def _get_metainfo(self, index):
        question = self.questions[index]
        scene = self.scenes[self._get_scene_index_from_question(question)]
        question['scene'] = scene
        question['program'] = question.pop('program', None)  # In CLEVR-Humans, there is no program.

        question['image_index'] = self._get_scene_index_from_question(question)
        question['image_filename'] = self._get_image_filename(scene)
        question['question_index'] = index
        question['question'] = question['question']
        question['answer'] = question['answer']
        question['question_type'] = question.get('question_type', self.query_list_key)

        for field_name in self.custom_fields:
            question[field_name] = scene[field_name]

        return question

    def _get_image_filename(self, scene: dict) -> str:
        return scene['image_filename']

    def __getitem__(self, index: int) -> dict:
        """Get a sample from the dataset.
        
        Returns:
            a dict of annotations, including:
                - scene: the scene annotations (raw dict).
                - objects: the bounding boxes of the objects (a Tensor of shape [N, 4]).
                - image_index: the index of the image (int).
                - image_filename: the filename of the image (str).
                - image: the image (a Tensor of shape [3, H, W]).
                - question_index: the index of the question (int).
                - question_raw: the raw question (str).
                - question_type: the type of the question (str).
                - answer: the answer to the question (bool, int, or str).
                - attribute_{attr_name}: the attribute concept id for each object (a Tensor of shape [N]).
                - attribute_relation_{attr_name}: the attribute relation concept id for each pair of objects (a Tensor of shape [N, N], then flattened to [N * N]).
                - relation_{attr_name}: the relational concept id for each pair of objects (a Tensor of shape [N, N, NR], then flattened to [N * N * NR]).
        """
                  
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # metainfo annotations
        feed_dict.update(annotate_objects(metainfo.scene))
        if 'objects' in feed_dict:
            # NB(Jiayuan Mao): in some datasets_v1, object information might be completely unavailable.
            feed_dict.objects_raw = feed_dict.objects.copy()

        if self.incl_scene:
            feed_dict.update(annotate_scene(metainfo.scene))
        if self.incl_raw_scene:
            feed_dict.scene = metainfo.scene

        # image
        
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None and feed_dict.image_filename is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image, feed_dict.objects = self.image_transform(feed_dict.image, feed_dict.objects)

        # question
        feed_dict.question_raw = metainfo.question
        feed_dict.question_type = metainfo.question_type
        feed_dict.answer = metainfo.answer
        return feed_dict.raw()

    def __len__(self):
        return len(self.questions)

    def make_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool, nr_workers: int) -> JacDataLoader:
        """Make a dataloader for this dataset view.

        Args:
            batch_size: the batch size.
            shuffle: whether to shuffle the dataset.
            drop_last: whether to drop the remaining samples that are smaller than the batch size.
            nr_workers: the number of workers for the dataloader.

        Returns:
            a JacDataLoader instance.
        """
        collate_guide = {
            'scene': 'skip',
            'objects_raw': 'skip',
            'objects': 'concat',

            'image_index': 'skip',
            'image_filename': 'skip',
            'question_index': 'skip',
            'question_metainfo': 'skip',

            'question_raw': 'skip',
            'question_type': 'skip',
            'answer': 'skip',
        }

        for field in self.custom_fields:
            collate_guide[field] = 'skip'

        # Scene annotations.
        for attr_name in g_attribute_concepts:
            collate_guide['attribute_' + attr_name] = 'concat'
            collate_guide['attribute_relation_' + attr_name] = 'concat'
        for attr_name in g_relational_concepts:
            collate_guide['relation_' + attr_name] = 'concat'

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


def make_dataset(
    scenes_json: str, questions_json: str, image_root: str, *,
    image_transform=None, vocab_json=None, output_vocab_json=None, filterable_view_cls=None, **kwargs
) -> CLEVRDatasetFilterableView:
    """Make a CLEVR dataset. See :class:`CLEVRDatasetUnwrapped` for more details.

    Args:
        scenes_json: the path to the scenes json file.
        questions_json: the path to the questions json file.
        image_root: the root directory of the images.
        image_transform: the image transform (torchvision transform). If None, a default transform will be used.
        vocab_json: the path to the vocab json file. If None, the vocab will be built from the dataset.
        output_vocab_json: the path to the output vocab json file. If None, the output vocab will be built from the dataset.
        filterable_view_cls: the filterable view class. If None, the default :class:`CLEVRDatasetFilterableView` will be used.
        **kwargs: other keyword arguments for the dataset.
    """
    if filterable_view_cls is None:
        filterable_view_cls = CLEVRDatasetFilterableView

    if image_transform is None:
        import jactorch.transforms.bbox as T
        image_transform = T.Compose([
            T.NormalizeBbox(),
            T.Resize(256),
            T.DenormalizeBbox(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return filterable_view_cls(CLEVRDatasetUnwrapped(
        scenes_json, questions_json, image_root, image_transform, vocab_json, output_vocab_json, **kwargs
    ))


def make_custom_transfer_dataset(
    scenes_json: str, questions_json: str, image_root: str, query_list_key: str, custom_fields: Sequence[str], *,
    image_transform=None, output_vocab_json=None, size: int = -1, **kwargs
) -> CLEVRCustomTransferDataset:
    """Make a CLEVR custom transfer dataset. See :class:`CLEVRCustomTransferDataset` for more details.

    Args:
        scenes_json: the path to the scenes json file.
        questions_json: the path to the questions json file.
        image_root: the root directory of the images.
        query_list_key: the key of the query list in the questions json file (e.g., 'questions' or 'questions_human').
        custom_fields: the custom fields to be included in the dataset. These are fields in the scene annotations.
        image_transform: the image transform (torchvision transform). If None, a default transform will be used.
        output_vocab_json: the path to the output vocab json file. If None, the output vocab will be built from the dataset.
        **kwargs: other keyword arguments for the dataset.

    Returns:
        a CLEVR custom transfer dataset.
    """
    if image_transform is None:
        import jactorch.transforms.bbox as T
        image_transform = T.Compose([
            T.NormalizeBbox(),
            T.Resize(256),
            T.DenormalizeBbox(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return CLEVRCustomTransferDataset(
        scenes_json, questions_json, image_root,
        query_list_key=query_list_key, custom_fields=custom_fields,
        image_transform=image_transform, output_vocab_json=output_vocab_json, **kwargs
    )


def annotate_scene(scene: dict) -> dict:
    """Annotate the scene with the attribute and relational concepts. This function will add the following fields to
    the scene annotations:

    - attribute_{attr_name}: the attribute concept id for each object.
    - attribute_relation_{attr_name}: the attribute relation concept id for each pair of objects.
    - relation_{attr_name}: the relational concept id for each pair of objects.

    Args:
        scene: the scene annotations.

    Returns:
        a dict of annotations of the attributes and relations.
    """
    feed_dict = dict()

    if not _is_object_annotation_available(scene):
        return feed_dict

    for attr_name, concepts in g_attribute_concepts.items():
        concepts2id = {v: i for i, v in enumerate(concepts)}
        values = list()
        for obj in scene['objects']:
            assert attr_name in obj
            values.append(concepts2id[obj[attr_name]])
        values = np.array(values, dtype='int64')
        feed_dict['attribute_' + attr_name] = values
        lhs, rhs = np.meshgrid(values, values)
        compare_label = (lhs == rhs).astype('float32')
        compare_label[np.diag_indices_from(compare_label)] = 0
        feed_dict['attribute_relation_' + attr_name] = compare_label.reshape(-1)

    nr_objects = len(scene['objects'])
    for attr_name, concepts in g_relational_concepts.items():
        concept_values = []
        for concept in concepts:
            values = np.zeros((nr_objects, nr_objects), dtype='float32')
            assert concept in scene['relationships']
            this_relation = scene['relationships'][concept]
            assert len(this_relation) == nr_objects
            for i, this_row in enumerate(this_relation):
                for j in this_row:
                    values[j, i] = 1
            concept_values.append(values)
        concept_values = np.stack(concept_values, -1)
        feed_dict['relation_' + attr_name] = concept_values.reshape(-1, concept_values.shape[-1])

    return feed_dict


def canonize_answer(answer: str, question_type: Optional[str] = None) -> Union[bool, int, str]:
    """Canonize the answer to a standard format.

    - For yes/no questions, the answer will be converted to a boolean.
    - For count questions, the answer will be converted to an integer.
    - For other questions, the answer will be kept as it is.

    Args:
        answer: the answer to be canonized.
        question_type: the question type. If None, the question type will be inferred from the answer.

    Returns:
        the canonized answer.
    """
    if answer in ('yes', 'no'):
        answer = (answer == 'yes')
    elif isinstance(answer, str) and answer.isdigit():
        answer = int(answer)
        assert 0 <= answer <= 10
    return answer


def annotate_objects(scene: dict) -> dict:
    """Annotate the scene with the object information. This function will add the following fields to the scene annotations:

    - objects: the bounding boxes of the objects.

    Args:
        scene: the scene annotations.

    Returns:
        a dict of annotations of the objects.
    """
    if 'objects' not in scene and 'objects_detection' not in scene:
        return dict()

    boxes = [mask_utils.toBbox(i['mask']) for i in _get_object_masks(scene)]
    if len(boxes) == 0:
        return {'objects': np.zeros((0, 4), dtype='float32')}
    boxes = np.array(boxes)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return {'objects': boxes.astype('float32')}


def _is_object_annotation_available(scene: dict) -> bool:
    if len(scene['objects']) > 0 and 'mask' in scene['objects'][0]:
        return True
    return False


def _get_object_masks(scene: dict) -> list:
    """Backward compatibility: in self-generated clevr scenes, the groundtruth masks are provided;
    while in the clevr test data, we use Mask R-CNN to detect all the masks, and stored in `objects_detection`."""
    if 'objects_detection' not in scene:
        return scene['objects']
    if _is_object_annotation_available(scene):
        return scene['objects']
    return scene['objects_detection']


g_last_op_to_question_type = {
    'query_color': 'query_attr',
    'query_shape': 'query_attr',
    'query_material': 'query_attr',
    'query_size': 'query_attr',
    'exist': 'exist',
    'count': 'count',
    'equal_integer': 'cmp_number',
    'greater_than': 'cmp_number',
    'less_than': 'cmp_number',
    'equal_color': 'cmp_attr',
    'equal_shape': 'cmp_attr',
    'equal_material': 'cmp_attr',
    'equal_size': 'cmp_attr',
}


def get_op_type(op: dict) -> str:
    """Get the type of the operation. This function is used to handle two different formats of the CLEVR programs."""
    if 'type' in op:
        return op['type']
    return op['function']


def get_question_type(program: list) -> str:
    """Get the question type from the full program. This function basically returns the type of the last operation in the program."""
    if program is None:
        return 'unk'
    return g_last_op_to_question_type[get_op_type(program[-1])]

