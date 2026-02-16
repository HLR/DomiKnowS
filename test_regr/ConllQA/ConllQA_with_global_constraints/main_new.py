import sys
import torch
import torch.nn as nn
import os
from pathlib import Path

import optuna
from optuna.trial import TrialState

from domiknows.program.plugins.bert_unfreezing_plugin import create_optimizer_factory, create_optimizer_with_differential_lr

import argparse

#import logging
#logging.basicConfig(level=logging.INFO)
from domiknows import setProductionLogMode
setProductionLogMode()

from domiknows.program import CallbackProgram
from domiknows.program.lossprogram import GumbelInferenceProgram
from domiknows.program.model.pytorch import PoiModel
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, TorchSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor

from domiknows.program.plugins.callback_plugin_manager import create_standard_plugin_manager

from reader import conll4_reader
import numpy as np

import spacy

# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')  # English()

from transformers import BertTokenizerFast, BertModel

TRANSFORMER_MODEL = 'bert-base-uncased'

FEATURE_DIM = 768 + 96

_bert_model = None
_models = {}

# Locate data file in the same directory as this script
def find_data_file(filename):
    """Find data file in the same directory as this script."""
    current_dir = Path(__file__).parent
    data_path = current_dir / filename
    
    if data_path.exists():
        return str(data_path)
    
    raise FileNotFoundError(f"Could not find {filename} in {current_dir}")

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Getting the arguments passed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs")
    parser.add_argument("--evaluate", action='store_true', help="Only run evaluation on the test set")
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training sample")
    parser.add_argument("--train_portion", type=str, default="entities_with_relation", help="Training subset")
    parser.add_argument("--asking_type", type=str, help="ASKING_TYPE to filter for")
    parser.add_argument("--load_previous", action='store_true', help="Whether to load a previous model")
    parser.add_argument("--previous_portion", type=str, default="entities_only_with_1_things_YN", help="Previous Training subset to load")
    parser.add_argument("--previous_file", type=str, default="", help="File to load previous model from")
    parser.add_argument("--checked_acc", type=float, default=0, help="Accuracy to test")
    parser.add_argument("--data_path", type=str, default="conllQA_with_global.json", help="Path to data file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation")
    parser.add_argument("--classifier_lr", type=float, default=1e-3, help="Learning rate for classifier heads")
    
    # Optuna arguments
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=False, help="Run Optuna hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--tune_train_size", type=int, default=200, help="Number of samples to use during tuning")

    # BERT freezing
    parser.add_argument("--freeze_bert", type=str2bool, nargs='?', const=True, default=True, help="Keep BERT frozen throughout training")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Epochs to train with BERT frozen before unfreezing")
    parser.add_argument("--unfreeze_every", type=int, default=500, help="Unfreeze BERT layers every N steps")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of BERT layers to unfreeze per step")
    parser.add_argument("--bert_lr", type=float, default=1e-5, help="Learning rate for BERT layers")
    
    # Counting t-norm settings
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP", "default", "auto"],
                        default="L",
                        help="T-norm mode: G/P/L/SP = fixed t-norm, 'default' = per-type defaults, 'auto' = adaptive during training")
    
    # Gumbel-Softmax settings
    parser.add_argument("--use_gumbel", type=str2bool, nargs='?', const=True, default=False, help="Use Gumbel-Softmax for counting")
    parser.add_argument("--gumbel_temp_start", type=float, default=5.0, help="Initial Gumbel temperature")
    parser.add_argument("--gumbel_temp_end", type=float, default=0.5, help="Final Gumbel temperature")
    parser.add_argument("--gumbel_anneal_start", type=int, default=0, help="Epoch to start annealing temperature")
    parser.add_argument("--hard_gumbel", type=str2bool, nargs='?', const=True, default=True, help="Use hard Gumbel")

    # ========================================================================
    # ADD: Register callback plugin arguments
    # ========================================================================
    plugin_manager = create_standard_plugin_manager()
    plugin_manager.add_arguments_to_parser(parser)
    
    args = parser.parse_args()
    return args

class Tokenizer():
    def __init__(self, device='cpu') -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_MODEL)
        self.device = device

    def __call__(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text, padding=True, return_tensors='pt', return_offsets_mapping=True)

        ids = tokens['input_ids'].to(self.device)
        mask = tokens['attention_mask'].to(self.device)
        offset = tokens['offset_mapping'].to(self.device)

        idx = mask.nonzero()[:, 0].unsqueeze(-1)
        mapping = torch.zeros(
            idx.shape[0],
            int(idx.max().item()) + 1,
            device=self.device
        )        
        mapping.scatter_(1, idx, 1)

        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:, :, 0].masked_select(mask), offset[:, :, 1].masked_select(mask)), dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return mapping, ids, offset, tokens

class BERT(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        self.device = device
        self.module.to(self.device)
        self.unfrozen_layers = 0
        self.total_layers = len(self.module.encoder.layer)
        
        # Start frozen
        self.freeze_all()
    
    def freeze_all(self):
        """Freeze all BERT parameters."""
        for param in self.module.parameters():
            param.requires_grad = False
        self.unfrozen_layers = 0
        
    def unfreeze_layers(self, n_layers):
        """Unfreeze the last n_layers of BERT encoder."""
        if n_layers <= self.unfrozen_layers:
            return
        
        for layer in self.module.encoder.layer[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        if self.unfrozen_layers == 0 and n_layers > 0:
            for param in self.module.pooler.parameters():
                param.requires_grad = True
        
        self.unfrozen_layers = n_layers
        print(f"[BERT] Unfroze {n_layers}/{self.total_layers} layers")
    
    def step_unfreeze(self, unfreeze_every=500, layers_per_step=1):
        """Call this every training step to gradually unfreeze."""
        self.step_count += 1
        
        if self.step_count % unfreeze_every == 0:
            new_layers = min(
                self.unfrozen_layers + layers_per_step,
                self.total_layers
            )
            self.unfreeze_layers(new_layers)
    
    def _get_model_device(self):
        """Get the device of the model parameters."""
        return next(self.module.parameters()).device
    
    def forward(self, input):
        # Check if input is on a different device than model
        model_device = self._get_model_device()
        if model_device != input.device:
            self.module.to(input.device)
            self.device = input.device
        
        input = input.unsqueeze(0)
        _out = self.module(input)
        
        out, *_ = _out
        if isinstance(out, str):
            out = _out.last_hidden_state
        
        assert out.shape[0] == 1
        out = out.squeeze(0)
        return out

class Classifier(torch.nn.Sequential):
    def __init__(self, in_features, device='cpu') -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)
        self.to(device)


class InferenceProgramWithCallbacks(CallbackProgram, GumbelInferenceProgram):
    """InferenceProgram with callback support and proper loss dict handling."""
    
    def default_after_train_step(self, output=None):
        """Override to do nothing - GumbelInferenceProgram already handles backward."""
        # GumbelInferenceProgram.train_epoch already calls loss.backward() and opt.step()
        # So we don't need to do it again here
        pass
    
    def __init__(self, graph, Model, loss=None, **kwargs):
        """
        Initialize with proper handling of loss parameter.
        
        Args:
            graph: Knowledge graph
            Model: Model class (e.g., PoiModel, SolverModel)
            loss: Either a dict (for PoiModel) or callable (for other models)
            **kwargs: Additional arguments
        """
        # Separate loss parameter handling based on model type
        if loss is not None and isinstance(loss, dict):
            # For PoiModel with dict loss, we need to handle initialization carefully
            # Don't pass 'loss' to parent __init__ (it will go to InferenceModel)
            # Instead, initialize normally then set model.loss afterward
            
            # Initialize parent classes without dict loss
            super().__init__(graph, Model, **kwargs)
            
            # Now set the model's loss to the dict after initialization
            if hasattr(self, 'model') and self.model is not None:
                self.model.loss = loss
        else:
            # Standard initialization for non-dict loss
            super().__init__(graph, Model, loss=loss, **kwargs)
        
        # Ensure callback lists are properly initialized
        # This prevents the parent's default_after_train_step from being added twice
        self.after_train_step = [self.default_after_train_step]
        
        # Explicitly initialize all callback hooks (from CallbackProgram)
        self.before_train = []
        self.after_train = []
        self.before_train_epoch = []
        self.after_train_epoch = []
        self.before_train_step = []
        self.before_test = []
        self.after_test = []
        self.before_test_epoch = []
        self.after_test_epoch = []
        self.before_test_step = []
        self.after_test_step = []
    
def program_declaration(train, dev, args, device='cpu'):
    global _models
    global _bert_model
    global _adaptive_tnorm
    
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, \
        rel_sentence_contains_phrase

    graph.detach()  # FIX: Uncomment this!

    # Set device for all Sensors
    TorchSensor.set_default_device(device)

    phrase['text'] = ReaderSensor(keyword='tokens')

    def word2vec(text):
        texts = list(map(lambda x: ' '.join(x.split('/')), text))
        tokens_list = list(nlp.pipe(texts))
        return torch.tensor(np.array([tokens.vector for tokens in tokens_list]), device=device)

    phrase['w2v'] = FunctionalSensor('text', forward=word2vec)

    def merge_phrase(phrase_text):
        return [' '.join(phrase_text)], torch.ones((1, len(phrase_text)), device=device)

    sentence['text', rel_sentence_contains_phrase.reversed] = JointSensor(phrase['text'], forward=merge_phrase)

    # Create Tokenizer with device parameter
    tokenizer = Tokenizer(device=device)
    word[rel_sentence_contains_word, 'ids', 'offset', 'text'] = JointSensor(sentence['text'], forward=tokenizer)

    # Create BERT with device parameter
    bert_model = BERT(device=device)
    _bert_model = bert_model
    
    word['bert'] = ModuleSensor('ids', module=bert_model)

    def match_phrase(phrase, word_offset):
        def overlap(a_s, a_e, b_s, b_e):
            return (a_s <= b_s and b_s <= a_e) or (a_s <= b_e and b_e <= a_e)

        ph_offset = 0
        ph_word_overlap = []
        for ph in phrase:
            ph_len = len(ph)
            word_overlap = []
            for word_s, word_e in word_offset:
                if word_e - word_s <= 0:
                    word_overlap.append(False)
                else:
                    word_overlap.append(overlap(ph_offset, ph_offset + ph_len, word_s, word_e))
            ph_word_overlap.append(word_overlap)
            ph_offset += ph_len + 1
        return torch.tensor(ph_word_overlap, device=device)

    phrase[rel_phrase_contains_word.reversed] = EdgeSensor(phrase['text'], word['offset'],
                                                           relation=rel_phrase_contains_word.reversed,
                                                           forward=match_phrase)

    def phrase_bert(bert):
        return bert

    phrase['bert'] = FunctionalSensor(rel_phrase_contains_word.reversed(word['bert']), forward=phrase_bert)

    def concat_features(bert, w2v):
        return torch.cat((bert, w2v), dim=-1)

    phrase['emb'] = FunctionalSensor('bert', 'w2v', forward=concat_features)

    # Create classifiers, then use them in ModuleLearners
    classifiers = {
        'people': Classifier(FEATURE_DIM, device=device),
        'organization': Classifier(FEATURE_DIM, device=device),
        'location': Classifier(FEATURE_DIM, device=device),
        'other': Classifier(FEATURE_DIM, device=device),
        'o': Classifier(FEATURE_DIM, device=device),
        'work_for': Classifier(FEATURE_DIM * 2, device=device),
        'located_in': Classifier(FEATURE_DIM * 2, device=device),
        'live_in': Classifier(FEATURE_DIM * 2, device=device),
        'orgbase_on': Classifier(FEATURE_DIM * 2, device=device),
        'kill': Classifier(FEATURE_DIM * 2, device=device),
    }

    # Use the classifier instances from the dictionary
    phrase[people] = ModuleLearner('emb', module=classifiers['people'])
    phrase[organization] = ModuleLearner('emb', module=classifiers['organization'])
    phrase[location] = ModuleLearner('emb', module=classifiers['location'])
    phrase[other] = ModuleLearner('emb', module=classifiers['other'])
    phrase[o] = ModuleLearner('emb', module=classifiers['o'])
    
    # Create per-property binary label sensors so that find_poi() can pair
    # each classifier (ModuleLearner) with its label on the SAME property.
    # Each classifier outputs 2-class logits (yes/no), so each label is binary.
    #
    # Strategy: read entity_labels list (multi-class indices) into a ReaderSensor,
    # then derive per-type binary labels via FunctionalSensor.
    # Entity type index mapping (must match reader.py):
    # people=0, organization=1, location=2, other=3, o=4
    phrase['entity_labels'] = ReaderSensor(keyword='entity_labels')

    def _make_binary_entity_fn(target_class_idx):
        """Return a forward fn that maps multi-class labels → binary (is-target)."""
        def _fn(labels):
            t = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels.long()
            return (t == target_class_idx).long()
        return _fn

    phrase[people]       = FunctionalSensor('entity_labels', forward=_make_binary_entity_fn(0), label=True)
    phrase[organization] = FunctionalSensor('entity_labels', forward=_make_binary_entity_fn(1), label=True)
    phrase[location]     = FunctionalSensor('entity_labels', forward=_make_binary_entity_fn(2), label=True)
    phrase[other]        = FunctionalSensor('entity_labels', forward=_make_binary_entity_fn(3), label=True)
    phrase[o]            = FunctionalSensor('entity_labels', forward=_make_binary_entity_fn(4), label=True)

    def filter_pairs(phrase_text, arg1, arg2, data):
        for rel in data:
            if arg1.instanceID == rel['head'] and arg2.instanceID == rel['tail']:
                return True
        return False

    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateReaderSensor(
        phrase['text'],
        relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        keyword='relations',
        forward=filter_pairs)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('emb'), rel_pair_phrase2.reversed('emb'),
        forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))

    # FIX: Use the SAME classifier instances from the dictionary
    pair[work_for] = ModuleLearner('emb', module=classifiers['work_for'])
    pair[located_in] = ModuleLearner('emb', module=classifiers['located_in'])
    pair[live_in] = ModuleLearner('emb', module=classifiers['live_in'])
    pair[orgbase_on] = ModuleLearner('emb', module=classifiers['orgbase_on'])
    pair[kill] = ModuleLearner('emb', module=classifiers['kill'])
    
    # Relation label sensors using FunctionalSensor with mapping tensors.
    # After CompositionCandidateReaderSensor runs, the mapping tensors
    # pair[rel_pair_phrase1.reversed] (N_pairs, N_phrases) and
    # pair[rel_pair_phrase2.reversed] (N_pairs, N_phrases) tell us which
    # phrase each pair connects. We use these to look up relation labels.
    #
    # NOTE: pair[work_for] = FunctionalSensor(...) → self.concept = pair
    # (Property.sup = the concept owning the property, i.e. pair),
    # so string/Relation predecessors resolve against pair and would work.
    # We pass Property objects explicitly for clarity.
    #
    # Relation type index mapping (must match reader.py):
    # work_for=0, located_in=1, live_in=2, orgbase_on=3, kill=4
    sentence['relation_labels'] = ReaderSensor(keyword='relation_labels')

    # Grab Property object references.
    # These will be passed directly to FunctionalSensor as predecessors.
    _prop_arg1_map = pair[rel_pair_phrase1.reversed]   # Property holding (N_pairs, N_phrases) mapping
    _prop_arg2_map = pair[rel_pair_phrase2.reversed]   # Property holding (N_pairs, N_phrases) mapping
    _prop_rel_labels = sentence['relation_labels']     # Property holding dict (on sentence, not pair)

    def _make_binary_relation_fn(target_rel_idx):
        """Return a forward fn that maps relation_labels → binary labels per pair.

        Uses mapping tensors (from CompositionCandidateReaderSensor) to find
        (head_phrase_idx, tail_phrase_idx) for each pair, then looks up the
        relation label in the relation_labels dict.
        """
        def _fn(arg1_mapping, arg2_mapping, rel_labels):
            # arg1_mapping: (N_pairs, N_phrases) one-hot mapping for arg1
            # arg2_mapping: (N_pairs, N_phrases) one-hot mapping for arg2
            # rel_labels: dict {(head_phrase_idx, tail_phrase_idx): class_index}
            n_pairs = arg1_mapping.shape[0]
            labels = torch.zeros(n_pairs, dtype=torch.long, device=arg1_mapping.device)
            # Extract head/tail phrase indices from one-hot mapping
            head_indices = arg1_mapping.argmax(dim=1)  # (N_pairs,)
            tail_indices = arg2_mapping.argmax(dim=1)  # (N_pairs,)
            for k in range(n_pairs):
                h = head_indices[k].item()
                t = tail_indices[k].item()
                rel_class = rel_labels.get((h, t), -1)
                if rel_class == target_rel_idx:
                    labels[k] = 1
            return labels
        return _fn

    # Pass Property objects directly (not strings/Relations) to bypass concept lookup
    pair[work_for]    = FunctionalSensor(_prop_arg1_map, _prop_arg2_map, _prop_rel_labels,
                                         forward=_make_binary_relation_fn(0), label=True)
    pair[located_in]  = FunctionalSensor(_prop_arg1_map, _prop_arg2_map, _prop_rel_labels,
                                         forward=_make_binary_relation_fn(1), label=True)
    pair[live_in]     = FunctionalSensor(_prop_arg1_map, _prop_arg2_map, _prop_rel_labels,
                                         forward=_make_binary_relation_fn(2), label=True)
    pair[orgbase_on]  = FunctionalSensor(_prop_arg1_map, _prop_arg2_map, _prop_rel_labels,
                                         forward=_make_binary_relation_fn(3), label=True)
    pair[kill]        = FunctionalSensor(_prop_arg1_map, _prop_arg2_map, _prop_rel_labels,
                                         forward=_make_binary_relation_fn(4), label=True)

    _models['bert'] = bert_model
    _models['classifiers'] = classifiers

    # Build loss_dict keyed by (output_sensor, target_sensor) — the actual sensor
    # objects that find_poi() discovers.  PoiModel.poi_loss calls
    #   self.loss[output_sensor, target_sensor](logit, labels, mask)
    # so we wrap CrossEntropyLoss to accept the extra `mask` argument.
    class MaskedCrossEntropyLoss(nn.Module):
        """CrossEntropyLoss wrapper that accepts (logit, labels, mask)."""
        def __init__(self):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(reduction='none')
        def forward(self, logit, labels, mask=None):
            # logit: (N, 2)  labels: (N,) with values 0 or 1
            labels = labels.long()
            per_sample = self.ce(logit, labels)
            if mask is not None and mask.any():
                per_sample = per_sample * mask.float()
                return per_sample.sum() / mask.float().sum().clamp(min=1)
            return per_sample.mean()

    # find_poi() will discover (output_sensor, target_sensor) pairs automatically;
    # we just need the loss_dict to return a loss for any (output, target) key.
    # Use a defaultdict so every pair discovered by find_poi() gets a loss.
    from collections import defaultdict

    class _LossFactory(dict):
        """dict subclass that auto-creates MaskedCrossEntropyLoss for any key.
        __bool__ returns True so `if not self.loss:` in PoiModel.poi_loss
        doesn't short-circuit when the dict is initially empty."""
        def __missing__(self, key):
            val = MaskedCrossEntropyLoss()
            self[key] = val
            return val
        def __bool__(self):
            return True  # always truthy — losses will be created on demand

    loss_dict = _LossFactory()

    graph.constraint['label'] = ReaderSensor(keyword='logic_label', label=True)
    train_dataset = graph.compile_executable(train, logic_keyword='logic_str', logic_label_keyword='logic_label')

    # Use PoiModel with supervised loss (loss_dict) + constraint loss (via InferenceProgram)
    # The custom InferenceProgramWithCallbacks properly handles dict-based loss
    program = InferenceProgramWithCallbacks(
        graph, PoiModel,
        loss=loss_dict,
        poi=[phrase, sentence, word, pair, people, organization, location, graph.constraint],
        tnorm=args.counting_tnorm, 
        inferTypes=['local/argmax'], 
        device=device,
        # Gumbel-specific parameters
        use_gumbel=args.use_gumbel,           # Enable Gumbel-Softmax
        initial_temp=args.gumbel_temp_start,  # Start with higher temperature (softer)
        final_temp=args.gumbel_temp_end,      # End with lower temperature (sharper)
        anneal_start_epoch=args.gumbel_anneal_start,
        anneal_epochs=args.epochs - args.gumbel_anneal_start,  # Anneal over remaining epochs
        hard_gumbel=args.hard_gumbel,  # Use hard Gumbel (straight-through estimator)
    )
    
    dev_dataset = None
    if dev is not None and len(dev) > 0:
        dev_dataset = graph.compile_executable(dev, logic_keyword='logic_str', logic_label_keyword='logic_label')
    
    # Supervised learning:
    # - Binary label sensors are on the SAME property as each classifier so find_poi()
    #   discovers (output_sensor, target_sensor) pairs automatically.
    # - loss_dict maps those pairs to MaskedCrossEntropyLoss (handles mask arg).
    # - Training computes: total_loss = supervised_loss + beta * constraint_loss
    
    return program, train_dataset, dev_dataset

def log_training_config(args, models=None, train=None, dev=None, test=None, plugin_manager=None):
    """Log all training configuration parameters."""
    print("\n" + "=" * 60)
    if args.tune:
        print("OPTUNA HYPERPARAMETER TUNING CONFIGURATION")
    else:
        print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    # Data settings
    print("\n[Data]")
    print(f"  Data path:        {args.data_path}")
    print(f"  Train portion:    {args.train_portion}")
    print(f"  Train size:       {args.train_size if args.train_size != -1 else 'all'}")
    print(f"  Asking type filter: {args.asking_type if args.asking_type else 'None (all types)'}")
    if train is not None:
        print(f"  Train examples:   {len(train)}")
    if dev is not None:
        print(f"  Dev examples:     {len(dev)}")
    if test is not None:
        print(f"  Test examples:    {len(test)}")
    
    # Optuna settings
    if args.tune:
        print("\n[Optuna Tuning]")
        print(f"  Number of trials:     {args.n_trials}")
        print(f"  Epochs per trial:     {args.epochs}")
        print(f"  Freeze BERT:          {args.freeze_bert}")
        if args.freeze_bert:
            print("  Tuning params:        classifier_lr only")
        else:
            print("  Tuning params:        classifier_lr, bert_lr, warmup_epochs, unfreeze_layers")
    
    # Training settings
    print("\n[Training]")
    print(f"  Epochs:           {args.epochs}")
    if not args.freeze_bert:
        print(f"  Warmup epochs:    {args.warmup_epochs}")
    print(f"  Batch size:       1")
    print(f"  Device:           {args.device}")
    
    # Learning rates
    if not args.tune:
        print("\n[Learning Rates]")
        print(f"  Classifier LR:    {args.classifier_lr}")
        if not args.freeze_bert:
            print(f"  BERT LR:          {args.bert_lr}")
    
    # BERT settings
    if args.freeze_bert:
        print("\n[BERT]")
        print("  Mode:             Frozen (feature extraction only)")
    else:
        print("\n[BERT Unfreezing]")
        if not args.tune:
            print(f"  Unfreeze layers/epoch: {args.unfreeze_layers}")
        if models is not None:
            print(f"  Total BERT layers:     {models['bert'].total_layers}")
            print(f"  Initially frozen:      {models['bert'].unfrozen_layers == 0}")
    
    # Gumbel-Softmax settings
    print("\n[Gumbel-Softmax]")
    if args.use_gumbel:
        print(f"  Enabled:          Yes")
        print(f"  Initial temp:     {args.gumbel_temp_start}")
        print(f"  Final temp:       {args.gumbel_temp_end}")
        print(f"  Anneal start:     Epoch {args.gumbel_anneal_start}")
        print(f"  Hard Gumbel:      {args.hard_gumbel} {'(straight-through estimator)' if args.hard_gumbel else ''}")
    else:
        print(f"  Enabled:          No")
    
    # Constraint settings
    print("\n[Constraints]")
    print(f"  Counting t-norm:  {args.counting_tnorm} (initial)")
    
    # ========================================================================
    # ADD: Log plugin configurations
    # ========================================================================
    if plugin_manager:
        plugin_manager.log_all_configs(args)
    
    # Model info
    print("\n[Model]")
    if models is not None:
        bert_params = sum(p.numel() for p in models['bert'].parameters())
        bert_trainable = sum(p.numel() for p in models['bert'].parameters() if p.requires_grad)
        clf_params = sum(p.numel() for name, clf in models['classifiers'].items() for p in clf.parameters())
        
        print(f"  BERT params:      {bert_params:,} (trainable: {bert_trainable:,})")
        print(f"  Classifier params: {clf_params:,}")
        print(f"  Total params:     {bert_params + clf_params:,}")
    
    # Mode
    print("\n[Mode]")
    print(f"  Evaluate only:    {args.evaluate}")
    print(f"  Load previous:    {args.load_previous}")
    if args.load_previous:
        print(f"    Previous portion: {args.previous_portion}")
        if args.previous_file:
            print(f"    Previous file:    {args.previous_file}")
    
    print("\n" + "=" * 60 + "\n")

# Tuning objective function
def evaluate_with_counting_metrics(program, dataset, threshold=0.5, device=None):
    """Evaluate and capture both boolean and counting metrics."""
    
    eval_device = device if device else "cpu"
    train_eval = program.evaluate_condition(dataset, device=eval_device, threshold=threshold, return_dict=True)
    
    bool_acc = train_eval.get('boolean_accuracy', 0.0)
    if bool_acc is not None:
        bool_acc = bool_acc / 100.0  # Convert from percentage
    else:
        bool_acc = 0.0
    
    counting_acc = train_eval.get('counting_accuracy', 0.0)
    if counting_acc is not None:
        counting_acc = counting_acc / 100.0  # Convert from percentage
    else:
        counting_acc = 0.0
    
    return bool_acc, counting_acc

def create_objective(args, train, test, plugin_manager=None):
    """Create Optuna objective function optimizing for counting constraints."""
    
    def objective(trial: optuna.Trial) -> float:
        global _models, _bert_model
        
        os.environ["TQDM_DISABLE"] = "1"
        
        # Higher LR range - counting constraints often need stronger signal
        classifier_lr = trial.suggest_float('classifier_lr', 1e-4, 5e-3, log=True)
        
        if args.freeze_bert:
            bert_lr = 0.0
            warmup_epochs = args.epochs
            unfreeze_layers = 0
        else:
            bert_lr = trial.suggest_float('bert_lr', 1e-6, 1e-4, log=True)
            warmup_epochs = trial.suggest_int('warmup_epochs', 0, 3)
            unfreeze_layers = trial.suggest_int('unfreeze_layers', 1, 4)
        
        print(f"\n{'='*60}")
        print(f"TRIAL {trial.number} STARTED")
        print(f"{'='*60}")
        print(f"  classifier_lr: {classifier_lr:.2e}")
        if not args.freeze_bert:
            print(f"  bert_lr:       {bert_lr:.2e}")
            print(f"  warmup_epochs: {warmup_epochs}")
            print(f"  unfreeze_layers: {unfreeze_layers}")
        print(f"{'='*60}")
        
        args.classifier_lr = classifier_lr
        args.bert_lr = bert_lr
        args.warmup_epochs = warmup_epochs
        args.unfreeze_layers = unfreeze_layers
        
        try:
            from graph import graph
            graph.detach()
            
            graph.varContext = None
            graph._processed_lcs = set()
            graph._executableLCs.clear()
            graph.executableLCsLabels.clear()
            
            program, dataset, _ = program_declaration(train, None, args, device=args.device)
            
            _models['bert'].freeze_all()
            
            epoch_count = [0]
            def epoch_callback():
                epoch_count[0] += 1
                print(f"  [Trial {trial.number}] Epoch {epoch_count[0]}/{args.epochs} complete")
            
            program.after_train_epoch.append(epoch_callback)
            
            if not args.freeze_bert:
                def unfreeze_callback():
                        epoch = program.epoch or 0
                        if epoch <= args.warmup_epochs:
                            return
                        epochs_after_warmup = epoch - args.warmup_epochs
                        layers_to_unfreeze = min(epochs_after_warmup * args.unfreeze_layers, 12)
                        if layers_to_unfreeze > _models['bert'].unfrozen_layers:
                            _models['bert'].unfreeze_layers(layers_to_unfreeze)
                            program.opt = create_optimizer_with_differential_lr(
                                _models['bert'],
                                _models['classifiers'],
                                bert_lr=args.bert_lr,
                                classifier_lr=args.classifier_lr,
                                device=args.device  # ADD THIS
                            )
                program.before_train_epoch.append(unfreeze_callback)

            
            initial_optimizer_factory = create_optimizer_factory(
                _models['bert'],
                _models['classifiers'],
                bert_lr=0.0 if args.freeze_bert else args.bert_lr,
                classifier_lr=args.classifier_lr,
                device=args.device
            )
            
            print(f"  [Trial {trial.number}] Training started...")
            
            program.train(
                dataset,
                Optim=initial_optimizer_factory,
                train_epoch_num=args.epochs,
                c_lr=args.classifier_lr,
                c_warmup_iters=-1,
                batch_size=1,
                print_loss=False
            )
            
            print(f"  [Trial {trial.number}] Evaluating...")
            bool_acc, counting_acc = evaluate_with_counting_metrics(program, dataset, device=args.device)
            
            # OBJECTIVE: Focus on counting performance
            # Option 1: Pure counting accuracy
            objective_value = counting_acc
            
            # Option 3: Combined (uncomment to use)
            # objective_value = 0.3 * bool_acc + 0.7 * counting_acc
            
            print(f"\n  [Trial {trial.number}] SUMMARY:")
            print(f"    Boolean Acc:    {bool_acc*100:.2f}%")
            print(f"    Counting Acc:   {counting_acc*100:.2f}%  <-- OPTIMIZING THIS")
            print(f"{'='*60}\n")
            
            return objective_value
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"\n  [Trial {trial.number}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            os.environ["TQDM_DISABLE"] = "0"
    
    return objective

def run_optuna_tuning(args, train, test, n_trials=20, plugin_manager=None):
    """Run Optuna hyperparameter tuning."""
    
    # Use smaller subset for faster tuning
    tune_train = train
    if args.tune_train_size > 0 and len(train) > args.tune_train_size:
        import random
        random.seed(42)
        tune_train = random.sample(train, args.tune_train_size)
        print(f"[Optuna] Using {len(tune_train)}/{len(train)} samples for tuning")
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name=f'conll_tuning_{args.train_portion}'
    )
    
    # Pass tune_train instead of train
    objective = create_objective(args, tune_train, test, plugin_manager=plugin_manager)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=False  # We have our own logging now
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Complete trials: {len(complete_trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Test Accuracy): {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study results
    results_file = f"optuna_results_{args.train_portion}.txt"
    with open(results_file, 'w') as f:
        f.write("OPTUNA HYPERPARAMETER TUNING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Test Accuracy: {trial.value:.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n\nAll Trials:\n")
        for t in study.trials:
            f.write(f"  Trial {t.number}: {t.value} - {t.params} - {t.state}\n")
    
    print(f"\nResults saved to {results_file}")
    
    return study

def main(args):
    global _models
    global _bert_model
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o

    data_file_path = find_data_file(args.data_path)
    train, dev, test = conll4_reader(data_path=data_file_path, dataset_portion=args.train_portion, asking_type=args.asking_type)

    if args.train_size != -1:
        train = train[:args.train_size]
    
    suffix = "_curriculum_learning" if args.load_previous else ""

    # Create plugin manager and register plugins
    plugin_manager = create_standard_plugin_manager()
    
    # Initialize models and graph
    if args.tune:
        # Pass plugin_manager to log_training_config
        log_training_config(args, models=None, train=train, dev=dev, test=test, 
                           plugin_manager=plugin_manager)
        
        study = run_optuna_tuning(args, train, test, n_trials=args.n_trials, plugin_manager=plugin_manager)
        
        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        print("=" * 60)
        
        best_params = study.best_trial.params
        args.classifier_lr = best_params['classifier_lr']
        if not args.freeze_bert:
            args.bert_lr = best_params['bert_lr']
            args.warmup_epochs = best_params['warmup_epochs']
            args.unfreeze_layers = best_params['unfreeze_layers']
        
        args.tune = False
        
        graph.detach()
        graph.varContext = None
        graph._processed_lcs = set()
        graph._executableLCs.clear()
        graph.executableLCsLabels.clear()
    
    program, dataset, _ = program_declaration(
        train if not args.evaluate else test, 
        None,
        args, 
        device=args.device
    )
    
    # Pass plugin_manager to log_training_config
    log_training_config(args, _models, train=train, dev=None, test=test,
                       plugin_manager=plugin_manager)
    
    train_eval = None
    if not args.evaluate:
        _models['bert'].freeze_all()
        
        # Load previous checkpoint if resuming training
        if args.load_previous:
            if args.previous_file != "":
                checkpoint_path = args.previous_file
            else:                
                checkpoint_path = f"training_{args.epochs}_lr_{args.classifier_lr}_{args.previous_portion}.pth"
            
            print(f"\n[Checkpoint] Loading previous model from: {checkpoint_path}")
            program.load(checkpoint_path)
            print(f"[Checkpoint] Model loaded successfully")
            
            # IMPORTANT: After loading, BERT unfrozen state may have changed
            # Update our tracking
            if hasattr(_models['bert'], 'unfrozen_layers'):
                print(f"[Checkpoint] BERT state after load: {_models['bert'].unfrozen_layers} layers unfrozen")
            
            # Evaluate loaded checkpoint to show starting accuracy
            print(f"\n[Checkpoint] Evaluating loaded model on current dataset...")
            try:
                checkpoint_eval = program.evaluate_condition(dataset, device=args.device, threshold=0.5, return_dict=True)
                
                # Extract metrics (same logic as final evaluation)
                loaded_acc = checkpoint_eval.get('accuracy', 0.0)
                if loaded_acc is not None:
                    loaded_acc = loaded_acc * 100.0
                else:
                    loaded_acc = 0.0
                
                loaded_bool_acc = checkpoint_eval.get('boolean_accuracy', 0.0)
                if loaded_bool_acc is None:
                    loaded_bool_acc = 0.0
                
                loaded_counting_acc = checkpoint_eval.get('counting_accuracy', 0.0)
                if loaded_counting_acc is None:
                    loaded_counting_acc = 0.0
                
                
                print(f"\n[Checkpoint] Loaded Model Performance:")
                print(f"  Overall Accuracy:    {loaded_acc:.2f}%")
                print(f"  Boolean Accuracy:    {loaded_bool_acc:.2f}%")
                print(f"  Counting Accuracy:   {loaded_counting_acc:.2f}%")
                print(f"\n[Checkpoint] Continuing training from this checkpoint...\n")
                
            except Exception as e:
                print(f"[Checkpoint] Warning: Could not evaluate loaded model: {e}")
                print(f"[Checkpoint] Continuing with training anyway...\n")
                
        # Create optimizer factory and then configure all plugins
        from domiknows.program.plugins.bert_unfreezing_plugin import create_optimizer_factory, create_optimizer_with_differential_lr
        
        initial_optimizer_factory = create_optimizer_factory(
            _models['bert'],
            _models['classifiers'],
            bert_lr=0.0 if args.freeze_bert else args.bert_lr,
            classifier_lr=args.classifier_lr,
            device=args.device
        )
        
        # Configure all plugins at once
        plugin_manager.configure_all(
            program=program,
            models=_models,
            args=args,
            dataset=dataset,
            optimizer_factory=create_optimizer_with_differential_lr
        )
        
        # Load previous checkpoint if resuming training
        if args.load_previous:
            if args.previous_file != "":
                checkpoint_path = args.previous_file
            else:                
                checkpoint_path = f"training_{args.epochs}_lr_{args.classifier_lr}_{args.previous_portion}.pth"   
            
            program.load(checkpoint_path)
            
        # Start training
        program.train(
            dataset,
            Optim=initial_optimizer_factory,
            train_epoch_num=args.epochs,
            c_lr=args.classifier_lr,
            c_warmup_iters=-1,
            batch_size=1,
            print_loss=False
        )
        
        # Run final evaluation on FULL dataset
        print("\n[Running final evaluation on full dataset...]")
        train_eval = program.evaluate_condition(dataset, device=args.device, threshold=0.5, return_dict=True)
        
        # Display final evaluation results using plugin manager's display function
        plugin_manager.final_display_all(final_eval=train_eval)
        
        print("=" * 60 + "\n")
                
        # Save final model checkpoint
        program.save(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}.pth")
    else:
        # Evaluation mode
        program.load(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}.pth")
        train_eval = program.evaluate_condition(dataset, device=args.device, threshold=0.5, return_dict=True)

    # Print final evaluation results
    output_f = open("result.txt", 'a')
    print("BERT params device:", next(_models['bert'].parameters()).device)
    train_acc = train_eval['accuracy']
    train_bool_acc = train_eval['boolean_accuracy']
    train_counting_acc = train_eval['counting_accuracy']
    portion = "Training" if not args.evaluate else "Testing"
    
    print(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}", file=output_f)
    print(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}")
    print(f"{portion} Acc: {train_acc}", file=output_f)
    print(f"{portion} Acc: {train_acc}")
    print(f"{portion} Boolean Acc: {train_bool_acc}", file=output_f)
    print(f"{portion} Boolean Acc: {train_bool_acc}")
    print(f"{portion} Counting Acc: {train_counting_acc}", file=output_f)
    print(f"{portion} Counting Acc: {train_counting_acc}")
    print("#" * 40, file=output_f)
    print("#" * 40)

    if args.checked_acc:
        print(f"<acc>{train_acc}</acc>")
        assert train_acc > args.checked_acc

    return 0

if __name__ == '__main__':
    args = parse_arguments()
    main(args)