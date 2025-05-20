import os
import os.path as osp
from typing import Optional, Union, Callable, Sequence, List, Dict, Any, Tuple, Type

import nltk # For word_tokenize
import numpy as np
from PIL import Image

import json as json_io # Standard json library
import logging
import random
from collections import Counter

logger = logging.getLogger(__name__)

# Attempt to import pycocotools for mask processing
try:
    from pycocotools import mask as coco_mask_utils
    PYCOCOTOOLS_AVAILABLE = True
    logger.info("pycocotools found. Will use it for mask to bounding box conversion.")
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    # Log messages about pycocotools unavailability will be handled in toBbox_from_mask

# --- Global Constants (Placeholders - Define accurately for CLEVR) ---
g_attribute_concepts = {
    'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
    'material': ['rubber', 'metal'],
    'shape': ['cube', 'sphere', 'cylinder'],
    'size': ['small', 'large']
}

g_relational_concepts = {
    'spatial_relation': ['left', 'right', 'front', 'behind']
}

g_synonyms = {
    "thing": ["thing", "object"],
    "sphere": ["sphere", "ball", "spheres", "balls"],
    "cube": ["cube", "block", "cubes", "blocks"],
    "cylinder": ["cylinder", "cylinders"],
    "large": ["large", "big"],
    "small": ["small", "tiny"],
    "metal": ["metallic", "metal", "shiny"],
    "rubber": ["rubber", "matte"],
}

g_last_op_to_question_type: Dict[str, str] = {
    'query_color': 'query_attr', 'query_shape': 'query_attr',
    'query_material': 'query_attr', 'query_size': 'query_attr',
    'exist': 'exist', 'count': 'count',
    'equal_integer': 'cmp_number', 'greater_than': 'cmp_number', 'less_than': 'cmp_number',
    'equal_color': 'cmp_attr', 'equal_shape': 'cmp_attr',
    'equal_material': 'cmp_attr', 'equal_size': 'cmp_attr',
    'unique': 'query_attr',
    'scene': 'scene_interrogation'
}

# --- Utility Classes and Functions ---

class AttrDict(dict):
    """A dictionary that allows for attribute-style access (e.g., obj.key)."""
    def __init__(self, *args: Any, **kwargs: Any):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Vocab(object):
    """A simple vocabulary class."""
    def __init__(self, word2idx: Optional[Dict[str, int]] = None):
        self.special_tokens = {'<pad>': 0, '<unk>': 1}
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        for token, idx in self.special_tokens.items():
            self.word2idx[token] = idx
            self.idx2word[idx] = token
        
        current_max_idx = max(self.idx2word.keys()) if self.idx2word else -1

        if word2idx is not None:
            for word, idx_from_file in word2idx.items():
                if word in self.word2idx: 
                    if self.word2idx[word] != idx_from_file:
                        logger.warning(f"Vocab loading: Word '{word}' is a special token, but input index "
                                       f"{idx_from_file} differs from pre-defined {self.word2idx[word]}. "
                                       f"Using pre-defined index.")
                    continue 
                
                if idx_from_file in self.idx2word: 
                    logger.warning(f"Vocab loading: Index {idx_from_file} for word '{word}' collides with "
                                   f"existing word '{self.idx2word[idx_from_file]}'. Assigning new index for '{word}'.")
                    current_max_idx += 1
                    new_idx = current_max_idx
                    self.word2idx[word] = new_idx
                    # self.idx2word[new_idx] = word # This line was correct, mistake in thought process
                else:
                    self.word2idx[word] = idx_from_file
                    # self.idx2word[idx_from_file] = word # This line was correct
                    if idx_from_file > current_max_idx:
                        current_max_idx = idx_from_file
            # Rebuild idx2word to ensure it's perfectly synchronized with word2idx
            self.idx2word = {v: k for k, v in self.word2idx.items()}


    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = (max(self.idx2word.keys()) + 1) if self.idx2word else 0
            while idx in self.idx2word:
                idx +=1
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]

    def map_sequence(self, sequence: List[str]) -> List[int]:
        return [self.word2idx.get(word, self.special_tokens['<unk>']) for word in sequence]

    def __len__(self) -> int:
        return len(self.word2idx)

    @classmethod
    def from_json(cls, vocab_json_path: str) -> 'Vocab':
        try:
            with open(vocab_json_path, 'r', encoding='utf-8') as f:
                word2idx = json_io.load(f)
            return cls(word2idx)
        except Exception as e:
            logger.error(f"Error loading vocab from {vocab_json_path}: {e}. Returning new vocab.")
            return cls()

    @classmethod
    def from_dataset(cls, dataset_instance: Any, keys: List[str] = ['question_tokenized'], 
                     single_word: bool = False, max_size: Optional[int] = None) -> 'Vocab':
        vocab = cls() 
        counter = Counter()

        if not hasattr(dataset_instance, 'get_metainfo') or not callable(dataset_instance.get_metainfo) or \
           not hasattr(dataset_instance, 'questions'):
            logger.error("Dataset instance for Vocab.from_dataset must have 'get_metainfo' method and 'questions' attribute.")
            return vocab

        for i in range(len(dataset_instance.questions)):
            try:
                metainfo = dataset_instance.get_metainfo(i)
                for key in keys:
                    data = metainfo.get(key)
                    if data is not None:
                        if single_word:
                            current_val = str(data)
                            counter.update([current_val])
                        else: 
                            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                                counter.update(data)
            except Exception as e:
                logger.warning(f"Vocab.from_dataset: Error processing item {i} for key '{key}': {e}")
        
        current_vocab_size = len(vocab)
        num_words_to_add = None
        if max_size is not None:
            if max_size <= current_vocab_size:
                return vocab 
            num_words_to_add = max_size - current_vocab_size
        
        for word, _ in counter.most_common(num_words_to_add):
            if word not in vocab.word2idx:
                vocab.add_word(word)
        return vocab


def toBbox_from_mask(mask_data: Any) -> List[float]:
    """
    Converts mask data to a bounding box [x_min, y_min, width, height].
    Uses pycocotools if available.
    'mask_data' is expected to be in a format compatible with pycocotools (e.g., RLE dict).
    """
    if PYCOCOTOOLS_AVAILABLE:
        if isinstance(mask_data, dict) and 'counts' in mask_data and 'size' in mask_data:
            try:
                bbox_xywh = coco_mask_utils.toBbox(mask_data) 
                return [float(c) for c in bbox_xywh]
            except Exception as e:
                logger.debug(f"Error using pycocotools.toBbox on RLE mask_data (type: {type(mask_data)}): {e}. Falling back.")
        # else:
            # logger.debug(f"Mask data (type: {type(mask_data)}) is not in a recognized RLE format for pycocotools. Trying placeholder.")
            pass # Fall through if not RLE or other recognized pycocotools format

    # Fallback placeholder if pycocotools is not available or fails
    if isinstance(mask_data, dict): 
        if 'bbox' in mask_data and isinstance(mask_data['bbox'], (list, tuple)) and len(mask_data['bbox']) == 4:
            try:
                return [float(v) for v in mask_data['bbox']] # Assuming [x,y,w,h]
            except ValueError:
                pass

    if not PYCOCOTOOLS_AVAILABLE:
        logger.warning("pycocotools is not available, and mask was not a pre-computed bbox or known format. Returning DUMMY bounding box [0,0,10,10].")
    else:
        logger.warning(f"Failed to process mask (type: {type(mask_data)}) or find pre-computed bbox. Returning DUMMY bounding box [0,0,10,10].")
    return [0.0, 0.0, 10.0, 10.0]


def default_image_transform(pil_image: Image.Image, bboxes: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    original_size_w, original_size_h = pil_image.size
    
    shorter_edge_target = 256.0
    if original_size_w == 0 or original_size_h == 0:
        new_w, new_h = int(shorter_edge_target), int(shorter_edge_target)
    elif original_size_w < original_size_h:
        new_w = int(shorter_edge_target)
        new_h = int(original_size_h * (shorter_edge_target / original_size_w))
    else:
        new_h = int(shorter_edge_target)
        new_w = int(original_size_w * (shorter_edge_target / original_size_h))
    
    try:
        resized_image_pil = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except Exception as e:
        logger.error(f"Failed to resize image: {e}. Using original image.")
        resized_image_pil = pil_image
        new_w, new_h = original_size_w, original_size_h

    adjusted_bboxes_np: Optional[np.ndarray] = None
    if bboxes is not None:
        adjusted_bboxes_np = np.zeros((0, 4), dtype='float32')
        if len(bboxes) > 0:
            bboxes_np = np.array(bboxes, dtype=float)
            scale_w = new_w / float(original_size_w) if original_size_w > 0 else 1.0
            scale_h = new_h / float(original_size_h) if original_size_h > 0 else 1.0
            
            bboxes_np[:, 0] *= scale_w
            bboxes_np[:, 2] *= scale_w # x_max
            bboxes_np[:, 1] *= scale_h
            bboxes_np[:, 3] *= scale_h # y_max
            adjusted_bboxes_np = bboxes_np.astype('float32')

    img_array_hwc_f32 = np.array(resized_image_pil, dtype=np.float32) / 255.0
    
    if img_array_hwc_f32.ndim == 2:
        img_array_hwc_f32 = np.stack((img_array_hwc_f32,) * 3, axis=-1)
    elif img_array_hwc_f32.shape[-1] == 1:
         img_array_hwc_f32 = np.concatenate([img_array_hwc_f32] * 3, axis=-1)
    elif img_array_hwc_f32.shape[-1] == 4: 
        img_array_hwc_f32 = img_array_hwc_f32[..., :3]
    return img_array_hwc_f32, adjusted_bboxes_np


class CLEVRDatasetUnwrapped:
    """The unwrapped CLEVR dataset, with jacinle/jactorch dependencies removed."""
    def __init__(
        self,
        scenes_json: str, 
        questions_json: Union[str, Sequence[str]], 
        image_root: str,
        image_transform: Callable[[Image.Image, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]],
        vocab_json: Optional[str], 
        output_vocab_json: Optional[str],
        question_transform: Optional[Callable[[Dict[str, Any]], None]] = None,
        incl_scene: bool = True, 
        incl_raw_scene: bool = False, 
        size: int = -1
    ):
        self.scenes_json_path = scenes_json
        self.questions_json_path = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.vocab_json_path = vocab_json
        self.output_vocab_json_path = output_vocab_json
        self.question_transform = question_transform
        self.incl_scene = incl_scene
        self.incl_raw_scene = incl_raw_scene

        logger.info(f'Loading scenes from: "{self.scenes_json_path}".')
        try:
            with open(self.scenes_json_path, 'r', encoding='utf-8') as f:
                self.scenes: List[Dict[str, Any]] = json_io.load(f)['scenes']
        except Exception as e:
            logger.error(f"Failed to load scenes from {self.scenes_json_path}: {e}")
            self.scenes = []

        self.questions: List[Dict[str, Any]] = []
        q_files_to_load: Sequence[str] = [self.questions_json_path] if isinstance(self.questions_json_path, str) else self.questions_json_path
        
        for filename in q_files_to_load:
            logger.info(f'Loading questions from: "{filename}".')
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.questions.extend(json_io.load(f)['questions'])
            except Exception as e:
                logger.error(f"Failed to load questions from {filename}: {e}")
        
        if not self.scenes or not self.questions:
             logger.warning("Dataset initialized with no scenes or no questions. Subsequent operations might fail.")

        if size != -1:
            original_question_count = len(self.questions)
            original_scene_count = len(self.scenes)
            if 0 <= size < original_question_count:
                self.questions = self.questions[:size]
                logger.info(f"Truncated questions from {original_question_count} to {len(self.questions)}.")
            elif size < 0 and size != -1:
                logger.warning(f"Invalid size {size} for questions, using all {original_question_count} questions.")

            if 0 <= size < original_scene_count:
                 self.scenes = self.scenes[:size]
                 logger.info(f"Truncated scenes from {original_scene_count} to {len(self.scenes)}.")
            elif size < 0 and size != -1:
                 logger.warning(f"Invalid size {size} for scenes, using all {original_scene_count} scenes.")


        if self.vocab_json_path is not None:
            logger.info(f'Loading vocab from: "{self.vocab_json_path}".')
            self.vocab = Vocab.from_json(self.vocab_json_path)
        else:
            logger.info('Building the vocab from dataset questions.')
            self.vocab = Vocab.from_dataset(self, keys=['question_tokenized'])

        if self.output_vocab_json_path is not None:
            logger.info(f'Loading output vocab from: "{self.output_vocab_json_path}".')
            self.output_vocab = Vocab.from_json(self.output_vocab_json_path)
        else:
            logger.info('Building the output vocab from dataset answers.')
            self.output_vocab = Vocab.from_dataset(self, keys=['answer'], single_word=True)

    def get_metainfo(self, index: int) -> Dict[str, Any]:
        if not (0 <= index < len(self.questions)):
            raise IndexError(f"Index {index} out of bounds for questions (len: {len(self.questions)})")
        
        question_data = self.questions[index].copy()
        image_idx = question_data.get('image_index')
        scene_data: Dict[str, Any]

        if image_idx is None or not isinstance(image_idx, int) or not (0 <= image_idx < len(self.scenes)):
            logger.error(f"Invalid/missing image_index ({image_idx}) for question {index}. Scenes len: {len(self.scenes)}. Using dummy scene.")
            scene_data = {'image_filename': 'error_invalid_scene.png', 'objects': [], 'objects_detection': [], 'relationships': {}}
        else:
            scene_data = self.scenes[image_idx].copy()
        
        metainfo_dict = AttrDict(question_data) 
        metainfo_dict.scene = scene_data 
        metainfo_dict.program = metainfo_dict.pop('program', None)

        metainfo_dict.image_filename = self._get_image_filename(scene_data) 
        metainfo_dict.question_index = index 
        
        question_text = str(metainfo_dict.get('question', ''))
        try:
            metainfo_dict.question_tokenized = nltk.word_tokenize(question_text) 
        except LookupError: 
            logger.error("NLTK 'punkt' tokenizer model not found. Please run: import nltk; nltk.download('punkt')")
            metainfo_dict.question_tokenized = question_text.split()
        except Exception as e:
            logger.error(f"Error tokenizing question: '{question_text}'. Error: {e}")
            metainfo_dict.question_tokenized = [] 

        metainfo_dict.question_type = get_question_type(metainfo_dict.program) 
        metainfo_dict.all_objects = scene_data.get('objects', [])
        
        return metainfo_dict

    def _get_image_filename(self, scene: Dict[str, Any]) -> str:
        return scene.get('image_filename', "unknown_image.png") 

    def __getitem__(self, index: int) -> Dict[str, Any]:
        metainfo = self.get_metainfo(index) 
        feed_dict = AttrDict()

        # Get initial bounding boxes (objects_raw)
        # These are [x_min, y_min, x_max, y_max]
        obj_annotations = annotate_objects(metainfo.scene) # scene is a dict for one scene
        feed_dict.update(obj_annotations) # Adds 'objects' if found
        
        current_objects_for_transform: Optional[np.ndarray] = None
        if 'objects' in feed_dict and isinstance(feed_dict.objects, np.ndarray):
            feed_dict.objects_raw = feed_dict.objects.copy() # Store raw bboxes before transform
            current_objects_for_transform = feed_dict.objects
        else: # Ensure objects_raw exists even if empty
            feed_dict.objects_raw = np.zeros((0, 4), dtype='float32')


        if self.incl_scene: # Annotate attributes and relations based on scene['objects'] (GT properties)
            scene_annotations = annotate_scene(metainfo.scene) # Uses scene['objects']
            feed_dict.update(scene_annotations)

        if self.incl_raw_scene:
            feed_dict.scene = metainfo.scene

        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        feed_dict.image = None
        
        if self.image_root and feed_dict.image_filename and \
           feed_dict.image_filename not in ["unknown_image.png", "error_invalid_scene.png", "error_dummy_scene.png"]:
            try:
                image_path = osp.join(self.image_root, feed_dict.image_filename)
                pil_image = Image.open(image_path).convert('RGB')
                # image_transform receives the raw bboxes (current_objects_for_transform)
                # and returns the transformed image and transformed bboxes
                transformed_image, transformed_objects = self.image_transform(pil_image, current_objects_for_transform)
                feed_dict.image = transformed_image
                if transformed_objects is not None: 
                     feed_dict.objects = transformed_objects # This is now the transformed bboxes
                elif current_objects_for_transform is not None: # If transform doesn't return bboxes, keep raw (untransformed)
                     feed_dict.objects = current_objects_for_transform
                else: # Should not happen if objects_raw was initialized
                     feed_dict.objects = np.zeros((0,4), dtype='float32')

            except FileNotFoundError:
                logger.error(f"Image file not found: {image_path}")
                feed_dict.objects = current_objects_for_transform if current_objects_for_transform is not None else np.zeros((0,4), dtype='float32')
            except Exception as e:
                logger.error(f"Error loading/transforming image {feed_dict.image_filename}: {e}")
                feed_dict.objects = current_objects_for_transform if current_objects_for_transform is not None else np.zeros((0,4), dtype='float32')
        else: # No valid image, ensure 'objects' field matches 'objects_raw' (i.e., no transform applied)
            feed_dict.objects = feed_dict.objects_raw

        feed_dict.pil_image = pil_image if 'pil_image' not in feed_dict else None
        feed_dict.question_index = metainfo.question_index
        feed_dict.question_raw = metainfo.question
        feed_dict.question_raw_tokenized = metainfo.question_tokenized
        feed_dict.question = list(metainfo.question_tokenized)
        feed_dict.question_type = metainfo.question_type
        feed_dict.answer = canonize_answer(metainfo.answer, metainfo.question_type)
        feed_dict.all_objects = metainfo.all_objects # Raw object dicts from scene['objects']
        feed_dict.program = metainfo.program # Program dict (if available)
    
        if self.question_transform is not None:
            self.question_transform(feed_dict)

        if not (isinstance(feed_dict.question, list) and all(isinstance(s, str) for s in feed_dict.question)):
            feed_dict.question = list(feed_dict.question_raw_tokenized) 
            if not (isinstance(feed_dict.question, list) and all(isinstance(s, str) for s in feed_dict.question)):
                feed_dict.question = []

        feed_dict.question = np.array(self.vocab.map_sequence(feed_dict.question), dtype=np.int64)
        return dict(feed_dict) 

    def __len__(self) -> int:
        return len(self.questions)

    def retain_data(self, fraction: float):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Fraction must be between 0.0 and 1.0")
        random.shuffle(self.questions)
        num_to_retain = int(fraction * len(self.questions))
        self.questions = self.questions[:num_to_retain]
        logger.info(f"Retained {len(self.questions)} questions ({fraction*100:.2f}%).")

    
class CLEVRDatasetFilterableView:
    """A view for filtering CLEVR datasets."""
    def __init__(self, unwrapped_dataset: CLEVRDatasetUnwrapped):
        self.dataset = unwrapped_dataset
        self.indices: List[int] = list(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, view_index: int) -> Dict[str, Any]:
        if not 0 <= view_index < len(self.indices):
            raise IndexError(f"Index {view_index} out of bounds for filtered view of size {len(self.indices)}.")
        original_index = self.indices[view_index]
        return self.dataset[original_index]

    def _get_metainfo_for_filter(self, original_index: int) -> Dict[str, Any]:
        return self.dataset.get_metainfo(original_index)

    def filter(self, filter_func: Callable[[Dict[str, Any]], bool], filter_name: str = 'custom_filter') -> 'CLEVRDatasetFilterableView':
        new_original_indices = [
            original_idx for original_idx in self.indices 
            if filter_func(self._get_metainfo_for_filter(original_idx))
        ]
        logger.info(f"Filter '{filter_name}': {len(self.indices)} -> {len(new_original_indices)} items.")
        self.indices = new_original_indices
        return self

    def filter_program_size_raw(self, max_length: int) -> 'CLEVRDatasetFilterableView':
        def filt(q_metainfo: Dict[str, Any]) -> bool:
            program = q_metainfo.get('program')
            return program is None or len(program) <= max_length
        return self.filter(filt, f'filter-program-size-clevr[{max_length}]')

    def filter_scene_size(self, max_scene_size: int) -> 'CLEVRDatasetFilterableView':
        def filt(q_metainfo: Dict[str, Any]) -> bool:
            scene = q_metainfo.get('scene', {})
            objects = scene.get('objects', []) # Checks GT objects for scene size
            return len(objects) <= max_scene_size
        return self.filter(filt, f'filter-scene-size[{max_scene_size}]')

    def filter_question_type(self, *, allowed: Optional[Sequence[str]] = None, disallowed: Optional[Sequence[str]] = None) -> 'CLEVRDatasetFilterableView':
        if (allowed is None and disallowed is None) or (allowed is not None and disallowed is not None):
            raise ValueError('Must provide either allowed or disallowed, but not both or neither.')

        allowed_set = set(allowed) if allowed is not None else None
        disallowed_set = set(disallowed) if disallowed is not None else None

        def filt(q_metainfo: Dict[str, Any]) -> bool:
            q_type = q_metainfo.get('question_type')
            if allowed_set is not None:
                return q_type in allowed_set
            if disallowed_set is not None: 
                return q_type not in disallowed_set
            return False
        
        filter_name_suffix = ""
        if allowed is not None: filter_name_suffix = f"[allowed={{{','.join(list(allowed))}}}]"
        elif disallowed is not None: filter_name_suffix = f"[disallowed={{{','.join(list(disallowed))}}}]"
        return self.filter(filt, f'filter-question-type{filter_name_suffix}')
    
    def filter_relational_type(self, *args) -> 'CLEVRDatasetFilterableView':
        def convert_program_to_str(program: List[Dict[str, Any]]) -> str:
            """Converts a program to a string representation."""
            return ' '.join([str(op.get('type', op.get('function'))) for op in program])
        def filt(q_metainfo: Dict[str, Any]) -> bool:
            program_str = convert_program_to_str(q_metainfo["program"])
            if "query" in program_str or "same_" in program_str or "relate" in program_str or "than" in program_str or "count" in program_str:
            # if "query" in program_str or "than" in program_str or "count" in program_str:
                return False
            return True        
        return self.filter(filt, f'filter-program-type-concept-only') 
    
    def make_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool, nr_workers: int) -> Any:
        logger.warning("`make_dataloader` (JacDataLoader) is removed. Use a standard DataLoader (e.g., torch.utils.data.DataLoader) and define a custom collate_fn if needed.")
        raise NotImplementedError("JacDataLoader and VarLengthCollateV2 are removed.")


class CLEVRCustomTransferDataset:
    """The unwrapped CLEVR dataset for custom transfer learning."""
    def __init__(
        self,
        scenes_json: str, 
        questions_json: Union[str, Sequence[str]],
        image_root: str, 
        image_transform: Callable[[Image.Image, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]],
        query_list_key: str, 
        custom_fields: Sequence[str],
        output_vocab_json: Optional[str] = None,
        incl_scene: bool = True, 
        incl_raw_scene: bool = False, 
        size: int = -1
    ):
        self.scenes_json_path = scenes_json
        self.questions_json_path = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.query_list_key = query_list_key
        self.custom_fields = custom_fields
        self.output_vocab_json_path = output_vocab_json
        self.incl_scene = incl_scene
        self.incl_raw_scene = incl_raw_scene

        logger.info(f'Loading scenes from: "{self.scenes_json_path}".')
        try:
            with open(self.scenes_json_path, 'r', encoding='utf-8') as f:
                self.scenes: List[Dict[str, Any]] = json_io.load(f)['scenes']
        except Exception as e:
            logger.error(f"Failed to load scenes from {self.scenes_json_path}: {e}")
            self.scenes = []
            
        self.questions: List[Dict[str, Any]] = []
        q_files_to_load: Sequence[str] = [self.questions_json_path] if isinstance(self.questions_json_path, str) else self.questions_json_path

        for filename in q_files_to_load:
            logger.info(f'Loading questions (key: {query_list_key}) from: "{filename}".')
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_data = json_io.load(f)
                    if query_list_key not in loaded_data:
                        logger.error(f"Query list key '{query_list_key}' not found in {filename}.")
                        continue
                    self.questions.extend(loaded_data[query_list_key])
            except Exception as e:
                 logger.error(f"Failed to load questions from {filename} with key {query_list_key}: {e}")

        if not self.scenes or not self.questions:
             logger.warning("CustomTransferDataset initialized with no scenes or no questions.")

        if size != -1: # Apply size truncation
            original_question_count = len(self.questions)
            original_scene_count = len(self.scenes)
            if 0 <= size < original_question_count:
                self.questions = self.questions[:size]
                logger.info(f"Truncated questions from {original_question_count} to {len(self.questions)}.")
            elif size < 0 and size != -1:
                 logger.warning(f"Invalid size {size} for questions, using all {original_question_count} questions.")

            if 0 <= size < original_scene_count:
                self.scenes = self.scenes[:size]
                logger.info(f"Truncated scenes from {original_scene_count} to {len(self.scenes)}.")
            elif size < 0 and size != -1:
                 logger.warning(f"Invalid size {size} for scenes, using all {original_scene_count} scenes.")

        if self.output_vocab_json_path is not None:
            logger.info(f'Loading output vocab from: "{self.output_vocab_json_path}".')
            self.output_vocab = Vocab.from_json(self.output_vocab_json_path)
        else:
            logger.info('Building the output vocab from dataset answers for CustomTransferDataset.')
            # Need to provide self to from_dataset
            self.output_vocab = Vocab.from_dataset(self, keys=['answer'], single_word=True)


    def _get_scene_index_from_question(self, question: Dict[str, Any]) -> int:
        idx_val = question.get('image_index', question.get('scene_index'))
        if idx_val is None:
            raise KeyError(f'Cannot find scene index (image_index or scene_index) in question: {list(question.keys())}')
        if not isinstance(idx_val, int):
            raise TypeError(f"Scene index must be int, got {type(idx_val)} for question.")
        return idx_val

    def get_metainfo(self, index: int) -> Dict[str, Any]: # For Vocab.from_dataset compatibility
        if not (0 <= index < len(self.questions)):
            raise IndexError(f"Index {index} out of bounds for questions list (len: {len(self.questions)})")

        question_data = self.questions[index].copy()
        scene_data: Dict[str, Any]
        scene_idx = -1
        try:
            scene_idx = self._get_scene_index_from_question(question_data)
            if not (0 <= scene_idx < len(self.scenes)):
                 raise IndexError(f"Scene index {scene_idx} out of bounds (scenes len: {len(self.scenes)})")
            scene_data = self.scenes[scene_idx].copy()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error fetching scene for question {index} (scene_idx attempt: {scene_idx}): {e}")
            scene_data = {'image_filename': 'error_dummy_scene.png', 'objects': [], 'objects_detection':[], 'relationships': {}}

        metainfo_dict = AttrDict(question_data)
        metainfo_dict.scene = scene_data
        metainfo_dict.program = metainfo_dict.pop('program', None) # Not typically used here but for consistency
        metainfo_dict.image_index = scene_idx
            
        metainfo_dict.image_filename = self._get_image_filename(scene_data)
        metainfo_dict.question_index = index
        
        metainfo_dict.question = question_data.get('question', "") # Raw question string
        metainfo_dict.answer = question_data.get('answer', None)   # Raw answer
        metainfo_dict.question_type = question_data.get('question_type', self.query_list_key)
        metainfo_dict.all_objects = scene_data.get('objects', []) # GT object properties

        for field_name in self.custom_fields:
            metainfo_dict[field_name] = scene_data.get(field_name)
        return metainfo_dict

    def _get_image_filename(self, scene: dict) -> str:
        return scene.get('image_filename', "unknown_image.png")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        metainfo = self.get_metainfo(index)
        feed_dict = AttrDict()

        obj_annotations = annotate_objects(metainfo.scene)
        feed_dict.update(obj_annotations)
        
        current_objects_for_transform: Optional[np.ndarray] = None
        if 'objects' in feed_dict and isinstance(feed_dict.objects, np.ndarray):
            feed_dict.objects_raw = feed_dict.objects.copy()
            current_objects_for_transform = feed_dict.objects
        else:
            feed_dict.objects_raw = np.zeros((0,4), dtype='float32')
        
        if self.incl_scene:
            scene_annotations = annotate_scene(metainfo.scene) # Uses scene['objects'] for attributes
            feed_dict.update(scene_annotations)
        if self.incl_raw_scene:
            feed_dict.scene = metainfo.scene

        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        feed_dict.image = None

        if self.image_root and feed_dict.image_filename and \
           feed_dict.image_filename not in ["unknown_image.png", "error_dummy_scene.png"]:
            try:
                image_path = osp.join(self.image_root, feed_dict.image_filename)
                pil_image = Image.open(image_path).convert('RGB')
                transformed_image, transformed_objects = self.image_transform(pil_image, current_objects_for_transform)
                feed_dict.image = transformed_image
                if transformed_objects is not None:
                     feed_dict.objects = transformed_objects
                elif current_objects_for_transform is not None:
                     feed_dict.objects = current_objects_for_transform
                else:
                     feed_dict.objects = np.zeros((0,4), dtype='float32')
            except FileNotFoundError:
                logger.error(f"Image file not found: {image_path}")
                feed_dict.objects = current_objects_for_transform if current_objects_for_transform is not None else np.zeros((0,4), dtype='float32')
            except Exception as e:
                logger.error(f"Error loading or transforming image {feed_dict.image_filename}: {e}")
                feed_dict.objects = current_objects_for_transform if current_objects_for_transform is not None else np.zeros((0,4), dtype='float32')
        else:
            feed_dict.objects = feed_dict.objects_raw


        feed_dict.question_raw = metainfo.question # Raw question string
        feed_dict.question_type = metainfo.question_type
        feed_dict.answer = canonize_answer(metainfo.answer, metainfo.question_type) # Canonized answer
        feed_dict.all_objects = metainfo.all_objects # GT object properties
        
        for field_name in self.custom_fields:
            feed_dict[field_name] = metainfo.get(field_name)
        
        return dict(feed_dict)

    def __len__(self) -> int:
        return len(self.questions)


def make_dataset(
    scenes_json: str, 
    questions_json: Union[str, Sequence[str]], 
    image_root: str, *,
    image_transform: Optional[Callable[[Image.Image, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]]] = None, 
    vocab_json: Optional[str] = None, 
    output_vocab_json: Optional[str] = None, 
    filterable_view_cls: Optional[Type[CLEVRDatasetFilterableView]] = None, 
    **kwargs: Any
) -> CLEVRDatasetFilterableView:
    
    if filterable_view_cls is None:
        filterable_view_cls = CLEVRDatasetFilterableView

    if image_transform is None:
        logger.info("`image_transform` not provided to make_dataset, using `default_image_transform`.")
        image_transform = default_image_transform
        
    try:
        nltk.word_tokenize("test string")
    except LookupError:
        logger.warning("NLTK 'punkt' tokenizer package not found. Attempting to download.")
        try:
            nltk.download('punkt', quiet=True)
            nltk.word_tokenize("test string")
            logger.info("NLTK 'punkt' downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK 'punkt'. Word tokenization may fail. Error: {e}")
            logger.error("Please install it manually: import nltk; nltk.download('punkt')")

    unwrapped_dataset = CLEVRDatasetUnwrapped(
        scenes_json, questions_json, image_root, 
        image_transform, vocab_json, output_vocab_json, **kwargs
    )
    return filterable_view_cls(unwrapped_dataset)


def make_custom_transfer_dataset(
    scenes_json: str, 
    questions_json: Union[str, Sequence[str]], 
    image_root: str, 
    query_list_key: str, 
    custom_fields: Sequence[str], *,
    image_transform: Optional[Callable[[Image.Image, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]]] = None, 
    output_vocab_json: Optional[str] = None,
    **kwargs: Any
) -> CLEVRCustomTransferDataset:
    
    if image_transform is None:
        logger.info("`image_transform` not provided for make_custom_transfer_dataset, using `default_image_transform`.")
        image_transform = default_image_transform

    return CLEVRCustomTransferDataset(
        scenes_json, questions_json, image_root,
        image_transform=image_transform,
        query_list_key=query_list_key, 
        custom_fields=custom_fields,
        output_vocab_json=output_vocab_json, 
        **kwargs
    )

# --- Annotation Helper Functions ---

def annotate_objects(scene: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extracts bounding boxes [x_min, y_min, x_max, y_max] from scene object annotations.
    Prioritizes 'objects_detection' if pycocotools is available, otherwise uses _get_object_masks.
    """
    object_list_source: Optional[List[Dict[str, Any]]] = None

    if PYCOCOTOOLS_AVAILABLE and 'objects_detection' in scene and isinstance(scene['objects_detection'], list) and scene['objects_detection']:
        object_list_source = scene['objects_detection']
        # logger.debug("Using 'objects_detection' for bounding box annotation with pycocotools.")
    else:
        object_list_source = _get_object_masks(scene) # Fallback
        # if PYCOCOTOOLS_AVAILABLE:
        #     logger.debug("Using fallback from _get_object_masks for bounding box annotation with pycocotools.")
        # else:
        #     logger.debug("Using fallback from _get_object_masks for bounding box annotation (pycocotools not available).")


    if not object_list_source:
        # logger.debug("No suitable object list found in scene for bbox annotation.")
        return {'objects': np.zeros((0, 4), dtype='float32')}

    boxes: List[List[float]] = []
    for obj_data in object_list_source:
        if isinstance(obj_data, dict) and 'mask' in obj_data:
            x, y, w, h = toBbox_from_mask(obj_data['mask']) # Returns [x,y,w,h]
            boxes.append([x, y, x + w, y + h]) # Convert to [x_min, y_min, x_max, y_max]
        # else:
            # logger.debug(f"Object data (type: {type(obj_data)}) is not a dict or missing 'mask' key. Cannot generate bounding box.")
            
    if not boxes:
        return {'objects': np.zeros((0, 4), dtype='float32')}
    
    return {'objects': np.array(boxes, dtype='float32')}


def annotate_scene(scene: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Annotates scene with attribute and relational concepts based on scene['objects'] (GT properties)."""
    feed_dict: Dict[str, np.ndarray] = {}
    # Important: This function uses scene['objects'] for GT attributes,
    # while annotate_objects might use scene['objects_detection'] for bboxes.
    objects_data = scene.get('objects') # Use GT objects for semantic attributes

    if not objects_data or not isinstance(objects_data, list):
        return feed_dict 

    nr_objects = len(objects_data)
    if nr_objects == 0:
        return feed_dict

    # Attribute concepts from scene['objects']
    for attr_name, concepts_list in g_attribute_concepts.items():
        concepts2id = {v: i for i, v in enumerate(concepts_list)}
        values: List[int] = []
        valid_attr_for_all = True
        for obj in objects_data: # Iterate over GT objects
            if not isinstance(obj, dict) or attr_name not in obj:
                valid_attr_for_all = False; break
            attr_val = obj[attr_name]
            values.append(concepts2id.get(attr_val, -1)) 
        
        if not valid_attr_for_all:
            feed_dict[f'attribute_{attr_name}'] = np.array([], dtype='int64')
            feed_dict[f'attribute_relation_{attr_name}'] = np.array([], dtype='float32')
            continue

        values_np = np.array(values, dtype='int64')
        feed_dict[f'attribute_{attr_name}'] = values_np
        
        if nr_objects > 1:
            lhs, rhs = np.meshgrid(values_np, values_np, indexing='ij')
            compare_label = (lhs == rhs).astype('float32')
            np.fill_diagonal(compare_label, 0.0) # No self-relation for attribute comparison
            feed_dict[f'attribute_relation_{attr_name}'] = compare_label.reshape(-1)
        else: 
            feed_dict[f'attribute_relation_{attr_name}'] = np.zeros((0), dtype='float32')

    # Relational concepts from scene['relationships'] (which refers to indices in scene['objects'])
    relationships_data = scene.get('relationships')
    default_num_concepts_in_list = 1
    
    for rel_attr_name, rel_concepts_list in g_relational_concepts.items():
        num_concepts_for_this_attr = len(rel_concepts_list) if rel_concepts_list else default_num_concepts_in_list
        
        if not isinstance(relationships_data, dict):
            feed_dict[f'relation_{rel_attr_name}'] = np.zeros((nr_objects * nr_objects, num_concepts_for_this_attr), dtype='float32')
            continue

        all_concept_matrices_for_attr: List[np.ndarray] = []
        for concept_name_in_list in rel_concepts_list:
            concept_matrix = np.zeros((nr_objects, nr_objects), dtype='float32')
            if concept_name_in_list in relationships_data:
                relation_data_for_concept = relationships_data[concept_name_in_list]
                if isinstance(relation_data_for_concept, list) and len(relation_data_for_concept) == nr_objects:
                    for obj_idx_i, related_indices in enumerate(relation_data_for_concept):
                        if isinstance(related_indices, list):
                            for obj_idx_j in related_indices: # obj_idx_j is related to obj_idx_i
                                if isinstance(obj_idx_j, int) and 0 <= obj_idx_j < nr_objects:
                                    concept_matrix[obj_idx_j, obj_idx_i] = 1.0 # relation(j,i) is true
            all_concept_matrices_for_attr.append(concept_matrix)
        
        if not all_concept_matrices_for_attr:
             all_concept_matrices_for_attr = [np.zeros((nr_objects, nr_objects), dtype='float32')] * num_concepts_for_this_attr

        stacked_concept_matrices = np.stack(all_concept_matrices_for_attr, axis=-1) 
        feed_dict[f'relation_{rel_attr_name}'] = stacked_concept_matrices.reshape(-1, stacked_concept_matrices.shape[-1])
            
    return feed_dict


def canonize_answer(answer: Any, question_type: Optional[str] = None) -> Union[bool, int, str]:
    if isinstance(answer, str):
        ans_lower = answer.lower()
        if ans_lower == 'yes': return True
        if ans_lower == 'no': return False
        if answer.isdigit(): 
            val = int(answer)
            return val
        return answer 
    elif isinstance(answer, (bool, int)):
        return answer
    elif answer is None:
        return "<None>" 
    return str(answer)


def _is_object_annotation_available(scene: Dict[str, Any]) -> bool:
    """Checks if groundtruth object annotations (with masks) are likely present in scene['objects']."""
    objects_field = scene.get('objects')
    return (isinstance(objects_field, list) and 
            bool(objects_field) and 
            isinstance(objects_field[0], dict) and 
            'mask' in objects_field[0])


def _get_object_masks(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Determines the source of object representations if 'objects_detection' is not prioritized.
    Favors groundtruth 'objects' with masks, then 'objects_detection', then 'objects' without masks.
    """
    if _is_object_annotation_available(scene): # Checks scene['objects']
        return scene.get('objects', []) 
    
    objects_detection_field = scene.get('objects_detection')
    if isinstance(objects_detection_field, list) and objects_detection_field:
        # Check if items in objects_detection also have 'mask' for consistency with toBbox_from_mask
        if isinstance(objects_detection_field[0], dict) and 'mask' in objects_detection_field[0]:
            return objects_detection_field
    
    objects_field = scene.get('objects') # Fallback to scene['objects'] even if no masks
    if isinstance(objects_field, list):
        return objects_field
        
    return []


def get_op_type(op: Dict[str, Any]) -> str:
    """Gets the operation type from a CLEVR program step."""
    op_type = op.get('type', op.get('function'))
    if isinstance(op_type, str):
        return op_type
    return 'unknown_op'


def get_question_type(program: Optional[List[Dict[str, Any]]]) -> str:
    """Determines the question type from the last operation in a CLEVR functional program."""
    if not program : 
        return 'unk' 
    
    last_op = program[-1]
    if not isinstance(last_op, dict):
        return 'unk_program_format'
        
    op_type_str = get_op_type(last_op)
    return g_last_op_to_question_type.get(op_type_str, 'unk_op_mapping')


