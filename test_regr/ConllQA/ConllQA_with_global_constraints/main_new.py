import sys
import torch
import os
from pathlib import Path

import optuna
from optuna.trial import TrialState
import random

sys.path.append('.')
sys.path.append('../../..')

import argparse

#import logging
#logging.basicConfig(level=logging.INFO)
from domiknows import setProductionLogMode
setProductionLogMode()

from domiknows.program import CallbackProgram
from domiknows.program.lossprogram import InferenceProgram, GumbelInferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, TorchSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor
from domiknows.solver.adaptiveTNormLossCalculator import AdaptiveTNormLossCalculator

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
        print(f"Using data file: {data_path}")
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
    parser.add_argument("--evaluate", action='store_true', help="Only run evaluation on the test set  needs to be set to true if --load_previous is set")
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training sample")
    parser.add_argument("--train_portion", type=str, default="entities_with_relation", help="Training subset")
    parser.add_argument("--asking_type", type=str, help="ASKING_TYPE to filter for (e.g. counting, atMost, Exactly, AtLeast, ATLeast)")
    parser.add_argument("--load_previous", action='store_true', help="Whether to load a previous model")
    parser.add_argument("--previous_portion", type=str, default="entities_only_with_1_things_YN", help="Previous Training subset to load (if --load_previous is set)")
    parser.add_argument("--previous_file", type=str, default="", help="File to load previous model from (if --load_previous is set)")
    
    parser.add_argument("--checked_acc", type=float, default=0, help="Accuracy to test")
    
    parser.add_argument("--data_path", type=str, default="conllQA_with_global.json", help="Path to data file (can be relative or absolute)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation (e.g., 'cuda', 'cpu', 'cuda:0')")
    
    # Learning rate arguments default: classifier_lr=1e-3
    parser.add_argument("--classifier_lr", type=float, default=1e-3, help="Learning rate for classifier heads")
    
    # Optuna arguments default OFF (can be enabled to perform hyperparameter tuning using Optuna, which may help find better learning rates and training schedules for improved model performance)
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=False, help="Run Optuna hyperparameter tuning (default: true)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--tune_train_size", type=int, default=200, help="Number of samples to use during tuning (smaller = faster)")

    # BERT freezing  defualt Frozen (can be set to false to enable gradual unfreezing during training)
    parser.add_argument("--freeze_bert", type=str2bool, nargs='?', const=True, default=True, help="Keep BERT frozen throughout training (default: true)")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Epochs to train with BERT frozen before unfreezing")
    parser.add_argument("--unfreeze_every", type=int, default=500, help="Unfreeze BERT layers every N steps")
    parser.add_argument("--unfreeze_layers", type=int, default=2,  help="Number of BERT layers to unfreeze per step")
    parser.add_argument("--bert_lr", type=float, default=1e-5, help="Learning rate for BERT layers (if not frozen)")
    
    # Evaluation settings default: evaluate on 20% of training data each epoch (for faster feedback during tuning, can be set to 0 to disable)
    parser.add_argument("--eval_fraction", type=float, default=0.2, help="Fraction of data for epoch evaluation (0.2 = 20%%)")
    
    # Counting t-norm settings default "L" (for Product t-norm, which is often effective for counting constraints, but can be set to "G", "P", or "SP" to test other t-norms)
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP"], default="L", help="The tnorm method to use for counting constraints.")
    
    # Adaptive t-norm settings default OFF (can be enabled to allow the program to automatically select the best t-norm for each constraint type during training, which may improve performance by tailoring the loss calculation to the specific characteristics of each constraint type)
    parser.add_argument("--adaptive_tnorm", type=str2bool, nargs='?', const=True, default=False, help="Enable adaptive t-norm selection per constraint")
    parser.add_argument("--tnorm_adaptation_interval", type=int, default=10, help="Steps between t-norm comparison (set lower for small datasets)")
    parser.add_argument("--tnorm_warmup_steps", type=int, default=5, help="Steps before adaptive t-norm selection begins")
    parser.add_argument("--tnorm_strategy", type=str, default="gradient_weighted", choices=["gradient_weighted", "loss_based", "rotating"], help="Strategy for selecting t-norms")
    
    # Counting schedule defaulr OFF (can be enabled to gradually introduce counting constraints during training, which may help with convergence and
    parser.add_argument("--use_counting_schedule", type=str2bool, nargs='?', const=True, default=False, help="Gradually introduce counting constraints during training")
    parser.add_argument("--counting_warmup_epochs", type=int, default=4, help="Epochs before introducing counting (default: half of total epochs)")
    
    # Gumbel-Softmax settings default OFF (can be enabled to use Gumbel-Softmax relaxation for counting constraints, which may improve training stability and convergence on counting tasks by providing smoother gradients compared to hard argmax)
    parser.add_argument("--use_gumbel", type=str2bool, nargs='?', const=True, default=False, help="Use Gumbel-Softmax for counting constraints")
    parser.add_argument("--gumbel_temp_start", type=float, default=5.0, help="Initial Gumbel temperature (higher = softer, better gradients)")
    parser.add_argument("--gumbel_temp_end", type=float, default=0.5, help="Final Gumbel temperature (lower = sharper, more discrete)")
    parser.add_argument("--gumbel_anneal_start", type=int, default=0, help="Epoch to start annealing temperature")
    parser.add_argument("--hard_gumbel", type=str2bool, nargs='?', const=True, default=True, help="Use hard Gumbel (straight-through estimator) - recommended for counting")

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
    """InferenceProgram with callback support for gradual unfreezing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Explicitly initialize all callback hooks (from CallbackProgram)
        self.before_train = []
        self.after_train = []
        self.before_train_epoch = []
        self.after_train_epoch = []
        self.before_train_step = []
        self.after_train_step = []
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

    def filter_pairs(phrase_text, arg1, arg2, data):
        for rel, (rel_arg1, *_), (rel_arg2, *_) in data:
            if arg1.instanceID == rel_arg1 and arg2.instanceID == rel_arg2:
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

    _models['bert'] = bert_model
    _models['classifiers'] = classifiers

    graph.constraint['label'] = ReaderSensor(keyword='logic_label', label=True)
    train_dataset = graph.compile_executable(train, logic_keyword='logic_str', logic_label_keyword='logic_label')

    program = InferenceProgramWithCallbacks(
        graph, SolverModel,
        poi=[phrase, sentence, word, people, organization, location, graph.constraint],
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
    
    return program, train_dataset, dev_dataset

def log_training_config(args, models=None, train=None, dev=None, test=None):
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
    if train is not None:
        print(f"  Train examples:   {len(train)}")
    if dev is not None:
        print(f"  Dev examples:     {len(dev)}")
    if test is not None:
        print(f"  Test examples:    {len(test)}")
    
    # Optuna settings (show first if tuning)
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
    
    # Learning rates (only show if not tuning, since they'll be searched)
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
            
    # Constraint settings
    print("\n[Constraints]")
    if args.adaptive_tnorm:
        print(f"  T-Norm adaptation: Enabled")
        print(f"    Strategy:       {args.tnorm_strategy}")
        print(f"    Adapt interval: {args.tnorm_adaptation_interval} steps")
        print(f"    Warmup steps:   {args.tnorm_warmup_steps}")
    else:
        print(f"  Counting t-norm:  {args.counting_tnorm}")
    
    # Model info
    print("\n[Model]")
    if models is not None:
        bert_params = sum(p.numel() for p in models['bert'].parameters())
        bert_trainable = sum(p.numel() for p in models['bert'].parameters() if p.requires_grad)
        clf_params = sum(p.numel() for name, clf in models['classifiers'].items() for p in clf.parameters())
        
        print(f"  BERT params:      {bert_params:,} (trainable: {bert_trainable:,})")
        print(f"  Classifier params: {clf_params:,}")
        print(f"  Total params:     {bert_params + clf_params:,}")
    else:
        # Estimate params before model creation
        bert_params_est = 110_000_000  # BERT base ~110M
        # 5 entity classifiers: (864*256 + 256) + (256*2 + 2) = 221,698 each
        # 5 relation classifiers: (1728*256 + 256) + (256*2 + 2) = 443,138 each
        entity_clf_params = 5 * ((FEATURE_DIM * 256 + 256) + (256 * 2 + 2))
        relation_clf_params = 5 * ((FEATURE_DIM * 2 * 256 + 256) + (256 * 2 + 2))
        clf_params_est = entity_clf_params + relation_clf_params
        
        print(f"  BERT params:      ~{bert_params_est:,} (trainable: {'0' if args.freeze_bert else 'varies'})")
        print(f"  Classifier params: ~{clf_params_est:,}")
        print(f"  Total params:     ~{bert_params_est + clf_params_est:,}")
    
    # Mode
    print("\n[Mode]")
    print(f"  Evaluate only:    {args.evaluate}")
    print(f"  Load previous:    {args.load_previous}")
    
    print("\n" + "=" * 60 + "\n")
   
# -- callback plugins 
def create_optimizer_with_differential_lr(bert_model, classifiers, 
                                          bert_lr=2e-5, classifier_lr=1e-6,
                                          device=None):
    """Create optimizer with different learning rates for BERT vs classifiers."""
    
    if device is not None:
        bert_model.to(device)
        for clf in classifiers.values():
            clf.to(device)
    
    param_groups = []
    
    bert_params = [p for p in bert_model.parameters() if p.requires_grad]
    if bert_params:
        param_groups.append({'params': bert_params, 'lr': bert_lr})
    
    clf_params = [p for clf in classifiers.values() 
                  for p in clf.parameters() if p.requires_grad]
    if clf_params:
        param_groups.append({'params': clf_params, 'lr': classifier_lr})
    
    if not param_groups:
        dummy_device = device if device else 'cpu'
        dummy = torch.nn.Parameter(torch.zeros(1, device=dummy_device))
        return torch.optim.Adam([dummy], lr=classifier_lr)
    
    return torch.optim.Adam(param_groups)

def create_optimizer_factory(bert_model, classifiers, bert_lr=2e-5, classifier_lr=1e-6, device=None):
    """Create optimizer factory that properly handles framework params."""
    
    if device is not None:
        bert_model.to(device)
        for clf in classifiers.values():
            clf.to(device)
    
    bert_param_ids = {id(p) for p in bert_model.parameters()}
    clf_param_ids = {id(p) for clf in classifiers.values() for p in clf.parameters()}
    
    def factory(params):
        # Convert generator to list to avoid exhausting it
        params_list = list(params) if params is not None else []
        
        if not params_list:
            return create_optimizer_with_differential_lr(
                bert_model, classifiers, bert_lr, classifier_lr, device
            )
        
        bert_group = []
        clf_group = []
        other_group = []
        
        for p in params_list:
            if not p.requires_grad:
                continue
            if id(p) in bert_param_ids:
                bert_group.append(p)
            elif id(p) in clf_param_ids:
                clf_group.append(p)
            else:
                other_group.append(p)
        
        param_groups = []
        if bert_group and bert_lr > 0:
            param_groups.append({'params': bert_group, 'lr': bert_lr})
        if clf_group:
            param_groups.append({'params': clf_group, 'lr': classifier_lr})
        if other_group:
            param_groups.append({'params': other_group, 'lr': classifier_lr})
        
        if not param_groups:
            return create_optimizer_with_differential_lr(
                bert_model, classifiers, bert_lr, classifier_lr, device
            )
        
        return torch.optim.Adam(param_groups)
    
    return factory

class GradualUnfreezeCallback:
    """Callback to gradually unfreeze BERT during training."""
    
    def __init__(self, bert_model, unfreeze_every=500, layers_per_step=1):
        self.bert_model = bert_model
        self.unfreeze_every = unfreeze_every
        self.layers_per_step = layers_per_step
        self.step = 0
    
    def on_batch_end(self):
        """Call after each batch."""
        self.step += 1
        if self.step % self.unfreeze_every == 0:
            new_layers = min(
                self.bert_model.unfrozen_layers + self.layers_per_step,
                self.bert_model.total_layers
            )
            self.bert_model.unfreeze_layers(new_layers)

def evaluate_with_counting_metrics(program, dataset, threshold=0.5, device=None):
    """Evaluate and capture both boolean and counting metrics."""
    
    eval_device = device if device else "cpu"
    train_eval = program.evaluate_condition(dataset, device=eval_device, threshold=threshold, return_dict=True)
    
    bool_acc = train_eval.get('boolean_accuracy', 0.0)
    if bool_acc is not None:
        bool_acc = bool_acc / 100.0  # Convert from percentage
    else:
        bool_acc = 0.0
    
    counting_mae = train_eval.get('counting_mae', float('inf'))
    if counting_mae is None:
        counting_mae = float('inf')
    
    counting_acc = train_eval.get('counting_accuracy', 0.0)
    if counting_acc is not None:
        counting_acc = counting_acc / 100.0  # Convert from percentage
    else:
        counting_acc = 0.0
    
    return bool_acc, counting_mae, counting_acc

def create_objective(args, train, test):
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
            bool_acc, counting_mae, counting_acc = evaluate_with_counting_metrics(program, dataset, device=args.device)
            
            # OBJECTIVE: Focus on counting performance
            # Option 1: Pure counting accuracy
            objective_value = counting_acc
            
            # Option 2: Negative MAE (minimize MAE = maximize negative MAE)
            # objective_value = -counting_mae
            
            # Option 3: Combined (uncomment to use)
            # objective_value = 0.3 * bool_acc + 0.7 * counting_acc
            
            print(f"\n  [Trial {trial.number}] SUMMARY:")
            print(f"    Boolean Acc:    {bool_acc*100:.2f}%")
            print(f"    Counting MAE:   {counting_mae:.3f}")
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

def run_optuna_tuning(args, train, test, n_trials=20):
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
    objective = create_objective(args, tune_train, test)
    
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

def create_adaptive_training_callback(program, models, args):
    """Create callback that tracks per-constraint-type metrics and applies t-norm switching."""
    
    adapt_interval = getattr(args, 'tnorm_adaptation_interval', 10)
    warmup = getattr(args, 'tnorm_warmup_steps', 5)
    auto_apply = getattr(args, 'adaptive_tnorm', False)
    
    adaptive_tracker = AdaptiveTNormLossCalculator(
        solver=None,
        tnorms=["L", "P", "SP", "G"],
        adaptation_interval=adapt_interval,
        warmup_steps=warmup,
        selection_strategy=getattr(args, 'tnorm_strategy', 'gradient_weighted'),
        auto_apply=auto_apply,
        min_observations=20,
    )
    
    step_counter = [0]
    
    def on_step_end(output):
        """Track metrics grouped by constraint type."""
        step_counter[0] += 1
        
        datanode = None
        if isinstance(output, (tuple, list)):
            for item in output:
                if item is not None and hasattr(item, 'calculateLcLoss'):
                    datanode = item
                    break
        
        if datanode is None:
            return
        
        try:
            current_tnorm = getattr(args, 'counting_tnorm', 'L')
            losses = datanode.calculateLcLoss(tnorm=current_tnorm)
            
            # Compute classifier gradient norm once
            grad_norm = 0.0
            for clf in models['classifiers'].values():
                for p in clf.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            for lc_name, loss_dict in losses.items():
                lc = loss_dict.get('lc')
                loss_tensor = loss_dict.get('loss')
                if loss_tensor is None:
                    continue
                
                if torch.is_tensor(loss_tensor):
                    loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                else:
                    loss_val = float(loss_tensor) if loss_tensor is not None else 0.0
                
                # Record into adaptive tracker (handles both per-type and per-constraint)
                adaptive_tracker.record_observation(lc_name, lc, loss_val, grad_norm, current_tnorm)
            
            # Compare t-norms at interval
            if step_counter[0] % adapt_interval == 0 and step_counter[0] >= warmup:
                for tnorm in adaptive_tracker.tnorms:
                    if tnorm == current_tnorm:
                        continue
                    try:
                        tnorm_losses = datanode.calculateLcLoss(tnorm=tnorm)
                        for lc_name, loss_dict in tnorm_losses.items():
                            lc = loss_dict.get('lc')
                            loss_tensor = loss_dict.get('loss')
                            if loss_tensor is not None and torch.is_tensor(loss_tensor):
                                loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                                adaptive_tracker.record_tnorm_comparison(lc_name, lc, tnorm, loss_val)
                    except Exception:
                        pass
                
        except Exception as e:
            if step_counter[0] <= 3:
                print(f"[Adaptive] Step {step_counter[0]} error: {e}")
    
    def on_epoch_end():
        """Print summary, compute recommendations, optionally apply t-norm switches."""
        adaptive_tracker.on_epoch_end(apply=auto_apply)
    
    return on_step_end, on_epoch_end, adaptive_tracker

def create_epoch_logging_callback(program, dataset, models, eval_fraction=0.2, min_samples=50, seed=42, device=None):
    """Create callback to log comprehensive training metrics after each epoch."""
    
    random.seed(seed)
    dataset_list = list(dataset)
    n_total = len(dataset_list)
    n_eval = max(min_samples, int(n_total * eval_fraction))
    n_eval = min(n_eval, n_total)
    
    eval_indices = sorted(random.sample(range(n_total), n_eval))
    eval_subset = [dataset_list[i] for i in eval_indices]
    
    print(f"[Eval] Using {n_eval}/{n_total} samples ({100*n_eval/n_total:.1f}%) for epoch evaluation")
    
    metrics_history = {
        'epoch': [],
        'overall_acc': [],
        'bool_acc': [],
        'counting_mae': [],
        'counting_acc': [],
        'clf_grad_norm': [],
        'accumulated_grad_norm': [],
    }
    
    accumulated_grad_norm = [0.0]
    grad_count = [0]
    
    def capture_gradients_before_step():
        total_norm = 0.0
        for name, clf in models['classifiers'].items():
            for p in clf.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 0:
            accumulated_grad_norm[0] += total_norm
            grad_count[0] += 1
    
    def log_epoch_metrics():
        epoch = program.epoch or 0
        
        print(f"\n[Epoch {epoch}] Starting evaluation...")
        
        eval_device = device if device else "cpu"
        
        try:
            train_eval = program.evaluate_condition(eval_subset, device=eval_device, threshold=0.5, return_dict=True)
            print(f"[Epoch {epoch}] Evaluation complete")
        except Exception as e:
            print(f"[Epoch {epoch}] ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return metrics_history
        
        # DEBUG: Print raw values from evaluate_condition
        print(f"\n[DEBUG] Raw eval results:")
        print(f"  accuracy: {train_eval.get('accuracy')}")
        print(f"  primary_metric: {train_eval.get('primary_metric')}")
        print(f"  boolean_accuracy: {train_eval.get('boolean_accuracy')}")
        print(f"  boolean_correct: {train_eval.get('boolean_correct')}")
        print(f"  boolean_total: {train_eval.get('boolean_total')}")
        print(f"  counting_accuracy: {train_eval.get('counting_accuracy')}")
        print(f"  counting_total: {train_eval.get('counting_total')}")
        print(f"  counting_errors length: {len(train_eval.get('counting_errors', []))}")
        
        # Calculate weights manually for verification
        bool_tot = train_eval.get('boolean_total', 0)
        count_tot = train_eval.get('counting_total', 0)
        if bool_tot + count_tot > 0:
            bool_weight = bool_tot / (bool_tot + count_tot)
            count_weight = count_tot / (bool_tot + count_tot)
            bool_acc_val = train_eval.get('boolean_accuracy', 0) or 0
            count_acc_val = train_eval.get('counting_accuracy', 0) or 0
            manual_primary = bool_weight * bool_acc_val + count_weight * count_acc_val
            print(f"  [VERIFY] boolean_weight: {bool_weight:.4f}, counting_weight: {count_weight:.4f}")
            print(f"  [VERIFY] manual primary_metric: {manual_primary:.2f}%")
        
        # 'accuracy' is already 0-1 range (divided by 100 in framework)
        overall_acc = train_eval.get('accuracy', 0.0)
        if overall_acc is None:
            overall_acc = 0.0
        else:
            overall_acc = overall_acc * 100.0  # Convert to percentage for display
        
        # 'boolean_accuracy' and 'counting_accuracy' are still percentages (0-100)
        bool_acc = train_eval.get('boolean_accuracy', 0.0)
        if bool_acc is None:
            bool_acc = 0.0
        
        counting_mae = train_eval.get('counting_mae', float('inf'))
        if counting_mae is None:
            counting_mae = float('inf')
        
        counting_acc = train_eval.get('counting_accuracy', 0.0)
        if counting_acc is None:
            counting_acc = 0.0
        
        avg_grad_norm = accumulated_grad_norm[0] / max(grad_count[0], 1)
        
        metrics_history['epoch'].append(epoch)
        metrics_history['overall_acc'].append(overall_acc)
        metrics_history['bool_acc'].append(bool_acc)
        metrics_history['counting_mae'].append(counting_mae)
        metrics_history['counting_acc'].append(counting_acc)
        metrics_history['accumulated_grad_norm'].append(avg_grad_norm)
        
        accumulated_grad_norm[0] = 0.0
        grad_count[0] = 0
        
        bert_status = f"frozen" if models['bert'].unfrozen_layers == 0 else f"{models['bert'].unfrozen_layers}L unfrozen"
        
        # Calculate deltas
        overall_delta = ""
        bool_delta = ""
        counting_delta = ""
        mae_delta = ""
        
        if len(metrics_history['overall_acc']) >= 2:
            overall_change = overall_acc - metrics_history['overall_acc'][-2]
            overall_delta = f" (Δ{overall_change:+.3f})"
            
            bool_change = bool_acc - metrics_history['bool_acc'][-2]
            bool_delta = f" (Δ{bool_change:+.3f})"
            
            counting_change = counting_acc - metrics_history['counting_acc'][-2]
            counting_delta = f" (Δ{counting_change:+.3f})"
            
            if counting_mae != float('inf') and metrics_history['counting_mae'][-2] != float('inf'):
                mae_change = counting_mae - metrics_history['counting_mae'][-2]
                mae_delta = f" (Δ{mae_change:+.3f})"
        
        print(f"\n[Epoch {epoch}] Metrics:")
        print(f"  Overall Acc:    {overall_acc:.4f}{overall_delta}")
        print(f"  Boolean Acc:    {bool_acc:.4f}{bool_delta}")
        print(f"  Counting Acc:   {counting_acc:.4f}{counting_delta}")
        mae_str = f"{counting_mae:.3f}" if counting_mae != float('inf') else "N/A"
        print(f"  Counting MAE:   {mae_str}{mae_delta}")
        print(f"  AvgGradNorm:    {avg_grad_norm:.6f}")
        print(f"  BERT:           {bert_status}")
        
        # Warnings
        if avg_grad_norm < 1e-7 and epoch > 1:
            print(f"  ⚠️  Gradients near zero - check t-norm choice!")
        if len(metrics_history['overall_acc']) >= 2:
            if overall_acc < metrics_history['overall_acc'][-2] - 0.02:
                print(f"  ⚠️  Overall accuracy dropped!")
            if bool_acc < metrics_history['bool_acc'][-2] - 0.02:
                print(f"  ⚠️  Boolean accuracy dropped!")
            if counting_acc < metrics_history['counting_acc'][-2] - 0.02:
                print(f"  ⚠️  Counting accuracy dropped!")
        
        print(f"[Epoch {epoch}] Logging complete\n")
        return metrics_history
    
    return log_epoch_metrics, metrics_history, eval_subset, capture_gradients_before_step

def create_gradient_flow_diagnostic_callback(program, models, check_every=100):
    """
    Diagnostic callback to check if gradients are flowing from sumL constraints
    back to entity classifiers.
    
    Checks:
    1. Are sumL losses being computed?
    2. Do sumL losses have gradients attached?
    3. Are classifier parameters receiving gradients from sumL?
    4. What's the magnitude of gradients from sumL vs other constraints?
    """
    
    step_counter = [0]
    sumL_stats = {
        'total_losses': [],
        'has_grad': [],
        'clf_grad_magnitudes': [],
        'constraint_type_grads': {'sumL': [], 'other': []}
    }
    
    def check_gradient_flow(output):
        """Check gradient flow after loss computation but before optimizer step."""
        step_counter[0] += 1
        
        # Only check periodically to avoid overhead
        if step_counter[0] % check_every != 0:
            return
        
        print(f"\n[Gradient Flow Check - Step {step_counter[0]}]")
        print("=" * 60)
        
        # Extract datanode from output
        datanode = None
        if isinstance(output, (tuple, list)):
            for item in output:
                if item is not None and hasattr(item, 'calculateLcLoss'):
                    datanode = item
                    break
        
        if datanode is None:
            print("  ⚠️  No datanode found in output")
            return
        
        # Calculate losses with gradient retention
        try:
            losses = datanode.calculateLcLoss(
                tnorm=getattr(program.graph, 'tnorm', 'L'),
                counting_tnorm=getattr(program.graph, 'counting_tnorm', None)
            )
        except Exception as e:
            print(f"  ⚠️  Failed to calculate losses: {e}")
            return
        
        # Analyze each constraint
        sumL_count = 0
        other_count = 0
        sumL_total_loss = 0.0
        other_total_loss = 0.0
        
        for lc_name, loss_dict in losses.items():
            lc = loss_dict.get('lc')
            loss_tensor = loss_dict.get('loss')
            
            if loss_tensor is None:
                continue
            
            # Determine if this is a sumL constraint
            is_sumL = False
            if lc and hasattr(lc, 'innerLC'):
                from domiknows.graph.logicalConstrain import sumL
                is_sumL = isinstance(lc.innerLC, sumL)
            
            if not torch.is_tensor(loss_tensor):
                continue
            
            loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
            
            if is_sumL:
                sumL_count += 1
                sumL_total_loss += loss_val
                sumL_stats['total_losses'].append(loss_val)
                sumL_stats['has_grad'].append(loss_tensor.requires_grad)
                
                # Check if gradient can be computed
                if loss_tensor.requires_grad:
                    print(f"  ✓ {lc_name}: sumL loss={loss_val:.4f}, requires_grad=True")
                else:
                    print(f"  ✗ {lc_name}: sumL loss={loss_val:.4f}, requires_grad=FALSE")
            else:
                other_count += 1
                other_total_loss += loss_val
        
        # Check classifier gradients
        print(f"\n  Constraint Summary:")
        print(f"    sumL constraints:   {sumL_count} (avg loss: {sumL_total_loss/max(sumL_count,1):.4f})")
        print(f"    Other constraints:  {other_count} (avg loss: {other_total_loss/max(other_count,1):.4f})")
        
        # Check if classifiers have gradients
        print(f"\n  Classifier Gradient Status:")
        total_grad_norm = 0.0
        has_any_grad = False
        
        for clf_name, clf in models['classifiers'].items():
            clf_grad_norm = 0.0
            param_count = 0
            grad_count = 0
            
            for param in clf.parameters():
                param_count += 1
                if param.grad is not None:
                    grad_count += 1
                    clf_grad_norm += param.grad.norm().item() ** 2
                    has_any_grad = True
            
            clf_grad_norm = clf_grad_norm ** 0.5
            total_grad_norm += clf_grad_norm
            
            if clf_grad_norm > 0:
                print(f"    {clf_name:15s}: grad_norm={clf_grad_norm:8.4f} ({grad_count}/{param_count} params)")
            else:
                print(f"    {clf_name:15s}: NO GRADIENTS ({grad_count}/{param_count} params)")
        
        if not has_any_grad:
            print(f"\n  ❌ CRITICAL: No classifier parameters have gradients!")
            print(f"     This means gradients are NOT flowing from constraints to classifiers.")
        else:
            print(f"\n  Total classifier grad norm: {total_grad_norm:.4f}")
        
        # Try to isolate sumL gradient contribution
        if sumL_count > 0:
            print(f"\n  Attempting to isolate sumL gradient contribution...")
            
            # Zero out classifier gradients
            for clf in models['classifiers'].values():
                clf.zero_grad()
            
            # Compute only sumL losses and backprop
            sumL_loss_sum = 0.0
            for lc_name, loss_dict in losses.items():
                lc = loss_dict.get('lc')
                if lc and hasattr(lc, 'innerLC'):
                    from domiknows.graph.logicalConstrain import sumL
                    if isinstance(lc.innerLC, sumL):
                        loss_tensor = loss_dict.get('loss')
                        if loss_tensor is not None and torch.is_tensor(loss_tensor):
                            sumL_loss_sum += loss_tensor
            
            if sumL_loss_sum != 0.0 and torch.is_tensor(sumL_loss_sum):
                try:
                    sumL_loss_sum.backward(retain_graph=True)
                    
                    sumL_grad_norm = 0.0
                    for clf in models['classifiers'].values():
                        for param in clf.parameters():
                            if param.grad is not None:
                                sumL_grad_norm += param.grad.norm().item() ** 2
                    sumL_grad_norm = sumL_grad_norm ** 0.5
                    
                    if sumL_grad_norm > 0:
                        print(f"    ✓ sumL contributes grad_norm={sumL_grad_norm:.4f} to classifiers")
                        sumL_stats['clf_grad_magnitudes'].append(sumL_grad_norm)
                    else:
                        print(f"    ✗ sumL does NOT contribute gradients to classifiers")
                        print(f"       Possible causes:")
                        print(f"       - Using argmax predictions (non-differentiable)")
                        print(f"       - Gradients blocked somewhere in computational graph")
                        print(f"       - sumL implemented without gradient flow")
                    
                    # Clear gradients again
                    for clf in models['classifiers'].values():
                        clf.zero_grad()
                        
                except Exception as e:
                    print(f"    ✗ Failed to backprop through sumL: {e}")
            else:
                print(f"    ⚠️  No sumL losses to backprop")
        
        print("=" * 60 + "\n")
    
    def print_summary():
        """Print summary statistics at end of epoch."""
        if not sumL_stats['total_losses']:
            print("\n[Gradient Flow Summary] No sumL constraints observed")
            return
        
        print("\n[Gradient Flow Summary - Epoch Complete]")
        print("=" * 60)
        print(f"  sumL Observations:     {len(sumL_stats['total_losses'])}")
        print(f"  Avg sumL Loss:         {sum(sumL_stats['total_losses'])/len(sumL_stats['total_losses']):.4f}")
        
        grad_enabled = sum(sumL_stats['has_grad'])
        print(f"  sumL with requires_grad: {grad_enabled}/{len(sumL_stats['has_grad'])}")
        
        if sumL_stats['clf_grad_magnitudes']:
            avg_grad = sum(sumL_stats['clf_grad_magnitudes']) / len(sumL_stats['clf_grad_magnitudes'])
            print(f"  Avg sumL→Classifier Gradient: {avg_grad:.4f}")
            
            if avg_grad < 1e-6:
                print(f"  ⚠️  Gradients are VANISHING - sumL not learning!")
            elif avg_grad > 1000:
                print(f"  ⚠️  Gradients are EXPLODING - may need clipping!")
        else:
            print(f"  ❌ No gradient flow detected from sumL to classifiers")
        
        print("=" * 60 + "\n")
        
        # Reset for next epoch
        sumL_stats['total_losses'].clear()
        sumL_stats['has_grad'].clear()
        sumL_stats['clf_grad_magnitudes'].clear()
    
    return check_gradient_flow, print_summary

def create_counting_weight_schedule(total_epochs, warmup_epochs=None):
    """
    Create a schedule that gradually introduces counting constraints.
    
    Epochs 1-warmup: counting_weight = 0.0 (boolean only)
    Epochs warmup+1 to end: counting_weight ramps from 0.01 to 1.0
    """
    if warmup_epochs is None:
        warmup_epochs = total_epochs // 2
    
    def get_counting_weight(epoch):
        if epoch <= warmup_epochs:
            return 0.0
        else:
            # Linear ramp from 0.01 to 0.1 over remaining epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.01 + progress * 0.09  # Max weight = 0.1 (10% of boolean)
    
    return get_counting_weight

def create_adaptive_loss_weighting_callback(program, total_epochs, args):
    """
    Apply adaptive weighting to counting vs boolean constraints.
    """
    from domiknows.graph.logicalConstrain import sumL
    
    get_counting_weight = create_counting_weight_schedule(
        total_epochs, 
        warmup_epochs=args.counting_warmup_epochs
    )
    
    def apply_loss_weights(output):
        """Reweight losses based on constraint type and training progress."""
        epoch = program.epoch or 1
        counting_weight = get_counting_weight(epoch)
        
        datanode = None
        if isinstance(output, (tuple, list)):
            for item in output:
                if item is not None and hasattr(item, 'calculateLcLoss'):
                    datanode = item
                    break
        
        if datanode is None:
            return
        
        # Get all losses
        all_losses = datanode.calculateLcLoss(
            tnorm=getattr(program.graph, 'tnorm', 'L'),
            counting_tnorm=getattr(program.graph, 'counting_tnorm', None)
        )
        
        weighted_loss = 0.0
        boolean_loss_sum = 0.0
        counting_loss_sum = 0.0
        
        for lc_name, loss_dict in all_losses.items():
            lc = loss_dict.get('lc')
            loss_tensor = loss_dict.get('loss')
            
            if loss_tensor is None or not torch.is_tensor(loss_tensor):
                continue
            
            # Check if counting constraint
            is_counting = False
            if lc and hasattr(lc, 'innerLC'):
                is_counting = isinstance(lc.innerLC, sumL)
            
            loss_val = loss_tensor if loss_tensor.numel() == 1 else loss_tensor.mean()
            
            if is_counting:
                # Apply reduced weight to counting
                weighted_loss += counting_weight * loss_val
                counting_loss_sum += loss_val
            else:
                # Full weight for boolean
                weighted_loss += loss_val
                boolean_loss_sum += loss_val
        
        # Log every 100 steps
        if not hasattr(program, '_loss_weight_step'):
            program._loss_weight_step = 0
        program._loss_weight_step += 1
        
        if program._loss_weight_step % 500 == 0:
            print(f"\n[Epoch {epoch}] Loss Weighting:")
            print(f"  Counting weight: {counting_weight:.3f}")
            print(f"  Boolean loss:    {boolean_loss_sum.item() if torch.is_tensor(boolean_loss_sum) else 0:.4f}")
            print(f"  Counting loss:   {counting_loss_sum.item() if torch.is_tensor(counting_loss_sum) else 0:.4f}")
            print(f"  Weighted total:  {weighted_loss.item() if torch.is_tensor(weighted_loss) else 0:.4f}\n")
        
        return weighted_loss
    
    def print_schedule():
        """Print the counting weight schedule at start of training."""
        print("\n[Counting Weight Schedule]")
        print("=" * 60)
        for epoch in range(1, total_epochs + 1):
            weight = get_counting_weight(epoch)
            status = "Boolean Only" if weight == 0 else f"Counting Weight: {weight:.3f}"
            print(f"  Epoch {epoch:2d}: {status}")
        print("=" * 60 + "\n")
    
    return apply_loss_weights, print_schedule

def create_gumbel_monitoring_callback(program):
    """Monitor Gumbel temperature and sampling behavior."""
    
    def log_gumbel_status():
        epoch = program.epoch or 0
        
        if hasattr(program, 'current_temp'):
            temp = program.current_temp
            print(f"  [Gumbel] Temperature: {temp:.4f}", end="")
            
            if temp > 1.0:
                print(" (soft - gradients flow well)")
            elif temp > 0.5:
                print(" (medium - balancing gradients and discreteness)")
            else:
                print(" (sharp - approaching discrete predictions)")
        
        if hasattr(program, 'hard_gumbel') and program.hard_gumbel:
            print(f"  [Gumbel] Using hard (straight-through) mode")
    
    return log_gumbel_status

def main(args):
    global _models
    global _bert_model
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o

    data_file_path = find_data_file(args.data_path)
    train, dev, test = conll4_reader(data_path=data_file_path, dataset_portion=args.train_portion)

    if args.train_size != -1:
        train = train[:args.train_size]
    
    suffix = "_curriculum_learning" if args.load_previous else ""

    # -- Initialize models and graph
    if args.tune:
        log_training_config(args, models=None, train=train, dev=dev, test=test)
        
        study = run_optuna_tuning(args, train, test, n_trials=args.n_trials)
        
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
    
    log_training_config(args, _models, train=train, dev=None, test=test)
    
    train_eval = None
    if not args.evaluate:
        _models['bert'].freeze_all()
        
        # -- Add epoch logging callback
        log_epoch_metrics, metrics_history, eval_subset, capture_gradients = create_epoch_logging_callback(
            program, dataset, _models,
            eval_fraction=args.eval_fraction,
            min_samples=50,
            seed=42,
            device=args.device
        )
        program.after_train_epoch.append(log_epoch_metrics)
        program.after_train_step.append(lambda _: capture_gradients())
        
        # -- Add adaptive t-norm callback
        on_step, on_epoch, tracker = create_adaptive_training_callback(program, _models, args)
        program.after_train_step.append(on_step)
        program.after_train_epoch.append(on_epoch)
        
        if args.adaptive_tnorm:
            print(f"[Adaptive T-Norm] Enabled with strategy '{args.tnorm_strategy}', auto-apply=True")
        else:
            print(f"[Adaptive T-Norm] Tracking only (use --adaptive_tnorm to enable auto-switching)")
            print(f"\n[T-norm] Using '{args.counting_tnorm}' for counting constraints")
            
        # -- Add gradient flow diagnostic
        check_gradient_flow, print_gradient_summary = create_gradient_flow_diagnostic_callback(
            program, _models, check_every=500  # Check every 500 steps
        )
        program.after_train_step.append(check_gradient_flow)
        program.after_train_epoch.append(lambda: print_gradient_summary())
        
        if args.freeze_bert:
            print("[Training] BERT frozen throughout training (--freeze_bert)")
        else:
            # -- Add dynamic unfreezing callback
            def unfreeze_callback():
                epoch = program.epoch or 0
                
                if epoch <= args.warmup_epochs:
                    print(f"[Epoch {epoch}] Warmup - BERT frozen")
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
                        device=args.device
                    )
                    print(f"[Epoch {epoch}] Unfroze {layers_to_unfreeze} layers, optimizer updated")
            
            #  - Add unfreeze callback at the start of each epoch after warmup 
            program.before_train_epoch.append(unfreeze_callback)
        
        # -- Initial optimizer setup (will be updated if unfreezing occurs)
        initial_optimizer_factory = create_optimizer_factory(
            _models['bert'],
            _models['classifiers'],
            bert_lr=0.0 if args.freeze_bert else args.bert_lr,
            classifier_lr=args.classifier_lr,
            device=args.device
        )
        
        # -- Add Gumbel monitoring callback
        gumbel_monitor = create_gumbel_monitoring_callback(program)
        program.after_train_epoch.append(gumbel_monitor)
            
        # -- Add adaptive loss weighting callback
        if args.use_counting_schedule:
            apply_loss_weights, print_schedule = create_adaptive_loss_weighting_callback(
                program, args.epochs, args
            )
            
            print_schedule()
            
            # Add callback
            program.after_train_step.append(apply_loss_weights)
            
        # -- Load previous checkpoint if resuming training
        if args.load_previous:
            if args.previous_file != "":
                checkpoint_path = args.previous_file
            else:                
                checkpoint_path = f"training_{args.epochs}_lr_{args.lr}_{args.previous_portion}.pth"   
            
            program.load(checkpoint_path)
            
        # -- Start training
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
        
        if len(metrics_history['epoch']) > 0:
            # Use metrics from FULL dataset evaluation, not subset
            initial_overall = metrics_history['overall_acc'][0]
            initial_bool = metrics_history['bool_acc'][0]
            initial_counting = metrics_history['counting_acc'][0]
            initial_mae = metrics_history['counting_mae'][0]
            
            # Use final_eval instead of last epoch's subset metrics
            final_overall = train_eval.get('accuracy', 0.0) * 100.0
            final_bool = train_eval.get('boolean_accuracy', 0.0) or 0.0
            final_counting = train_eval.get('counting_accuracy', 0.0) or 0.0
            final_mae = train_eval.get('counting_mae', float('inf')) or float('inf')
            
            # Overall metrics
            print("\n[Overall Metrics]")
            print(f"  Initial Accuracy:      {initial_overall:.2f}%")
            print(f"  Final Accuracy:        {final_overall:.2f}%")
            print(f"  Total Improvement:     {final_overall - initial_overall:+.2f}%")
            
            # Boolean vs Counting breakdown
            print("\n[Boolean Constraints]")
            print(f"  Initial Boolean Acc:   {initial_bool:.2f}%")
            print(f"  Final Boolean Acc:     {final_bool:.2f}%")
            print(f"  Boolean Improvement:   {final_bool - initial_bool:+.2f}%")
            
            print("\n[Counting Constraints]")
            print(f"  Initial Counting Acc:  {initial_counting:.2f}%")
            print(f"  Final Counting Acc:    {final_counting:.2f}%")
            print(f"  Counting Improvement:  {final_counting - initial_counting:+.2f}%")
            
            if final_mae != float('inf'):
                print(f"  Final Counting MAE:    {final_mae:.3f}")
                if initial_mae != float('inf'):
                    mae_change = final_mae - initial_mae
                    print(f"  MAE Change:            {mae_change:+.3f}")
            
            # Gradient analysis
            avg_grad = sum(metrics_history['accumulated_grad_norm']) / max(len(metrics_history['accumulated_grad_norm']), 1)
            print(f"\n[Gradient Analysis]")
            print(f"  Average Gradient Norm: {avg_grad:.6f}")
            
            # Learning assessment
            print("\n[Learning Assessment]")
            overall_improvement = final_overall - initial_overall
            bool_improvement = final_bool - initial_bool
            counting_improvement = final_counting - initial_counting
            
            if overall_improvement > 0.05:
                print("  ✅ Model is learning well!")
            elif overall_improvement > 0:
                print("  ⚠️  Model is learning slowly - consider more epochs or higher LR")
            else:
                print("  ❌ Model is NOT learning - check LR, data, or architecture")
            
            # Boolean vs counting comparison
            if bool_improvement > counting_improvement + 0.1:
                print("  ⚠️  Counting constraints underperforming - check t-norm selection")
            elif counting_improvement > bool_improvement + 0.1:
                print("  ⚠️  Boolean constraints underperforming")
            
            # Adaptive t-norm summary
            if hasattr(tracker, 'get_summary_stats'):
                print("\n[Adaptive T-Norm Analysis]")
                stats = tracker.get_summary_stats()
                
                # Export detailed per-ELC stats to CSV FIRST
                csv_filename = f"adaptive_tnorm_details_{args.train_portion}_epoch{args.epochs}.csv"
                csv_exported = False
                num_exported = 0
                if hasattr(tracker, 'export_detailed_stats_to_csv'):
                    try:
                        num_exported = tracker.export_detailed_stats_to_csv(csv_filename)
                        csv_exported = True
                    except ValueError as e:
                        # No data to export - this is expected early in training
                        pass
                    except Exception as e:
                        print(f"  Error exporting CSV: {e}")
                
                if csv_exported:
                    print(f"  Detailed constraint stats exported to: {csv_filename} ({num_exported} records)")
                else:
                    print(f"  No per-constraint stats to export yet (early in training)")
                
                # Coverage (only if available)
                if 'total_global_types' in stats:
                    print(f"\n  Total Global Constraint Types: {stats['total_global_types']}")
                if 'total_executable_constraints' in stats:
                    print(f"  Total Executable Constraints:   {stats['total_executable_constraints']}")
                
                # Final recommendations per TYPE only (not individual ELCs)
                if 'final_recommendations_by_type' in stats and stats['final_recommendations_by_type']:
                    print("\n  Final T-Norm Recommendations by Type:")
                    for ctype, tnorm in stats['final_recommendations_by_type'].items():
                        print(f"    {ctype:20s} -> {tnorm}")
                
                # Recommendation history (type level only)
                if 'recommendation_history' in stats and stats['recommendation_history']:
                    print("\n  Recommendation History (when changes occurred):")
                    for epoch, changes in stats['recommendation_history'].items():
                        print(f"    Epoch {epoch}:")
                        for ctype, tnorm in changes.items():
                            print(f"      {ctype:18s} -> {tnorm}")
                
                # Currently active t-norms (if auto-apply enabled)
                if args.adaptive_tnorm and 'active_tnorm_config' in stats and stats['active_tnorm_config']:
                    print("\n  Currently Active T-Norms (LossCalculator.TNORM_CONFIG):")
                    for ctype, tnorm in stats['active_tnorm_config'].items():
                        print(f"    {ctype:20s} -> {tnorm}")
                
                # Summary: Show only LC (type-level) stats, suppress ELC (instance-level)
                has_standard_keys = any(key in stats for key in ['total_global_types', 'final_recommendations_by_type', 'recommendation_history'])
                
                if not has_standard_keys and stats:
                    print("\n  Constraint Type Summary:")
                    
                    # Separate LC (constraint types) from ELC (executable instances)
                    lc_stats = {}
                    elc_count = 0
                    
                    for key, value in stats.items():
                        # Skip non-dict values
                        if not isinstance(value, dict):
                            continue
                        
                        # LC keys are constraint types (e.g., LC0, LC1, LC2)
                        if key.startswith('LC') and not key.startswith('LC_'):
                            lc_stats[key] = value
                        # ELC keys are executable constraint instances
                        elif key.startswith('ELC'):
                            elc_count += 1
                    
                    # Print LC type-level stats
                    if lc_stats:
                        for lc_name in sorted(lc_stats.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
                            lc_data = lc_stats[lc_name]
                            ctype = lc_data.get('constraint_type', 'unknown')
                            obs = lc_data.get('observations', 0)
                            avg_loss = lc_data.get('avg_loss', 0.0)
                            tnorm = lc_data.get('current_tnorm', 'N/A')
                            grad_health = lc_data.get('gradient_health', 'unknown')
                            
                            # Add warning emoji for problematic gradients
                            health_icon = ""
                            if grad_health == 'vanishing':
                                health_icon = " ⚠️"
                            elif grad_health == 'exploding':
                                health_icon = " 🔥"
                            
                            print(f"    {lc_name} ({ctype:12s}): {obs:3d} obs, loss={avg_loss:.4f}, tnorm={tnorm}{health_icon}")
                    
                    # Show ELC instance count
                    if elc_count > 0:
                        csv_ref = f"see {csv_filename}" if csv_exported else "CSV export available"
                        print(f"\n    [{elc_count} executable constraint instances - {csv_ref}]")
                    
                    # If no stats at all
                    if not lc_stats and elc_count == 0:
                        print("    No constraint statistics collected yet")
                        print("    (Stats are collected during training steps)")
                elif not stats:
                    print("\n  No statistics available yet")
        
        print("=" * 60 + "\n")
                
        #-- Save final model checkpoint
        program.save(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}.pth")
    else:
        #-- Evaluation mode: Load specified checkpoint and evaluate on test set
        program.load(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}.pth")
        train_eval = program.evaluate_condition(dataset, device=args.device, threshold=0.5, return_dict=True)

    # Print final evaluation results to console and result file
    output_f = open("result.txt", 'a')
    print("BERT params device:", next(_models['bert'].parameters()).device)
    train_acc = train_eval['accuracy']
    train_bool_acc = train_eval['boolean_accuracy']
    train_counting_mae = float('inf') if train_eval['counting_mae'] is None else train_eval['counting_mae']
    train_counting_acc = train_eval['counting_accuracy']
    portion = "Training" if not args.evaluate else "Testing"
    
    print(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}", file=output_f)
    print(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}")
    print(f"{portion} Acc: {train_acc}", file=output_f)
    print(f"{portion} Acc: {train_acc}")
    print(f"{portion} Boolean Acc: {train_bool_acc}", file=output_f)
    print(f"{portion} Boolean Acc: {train_bool_acc}")
    print(f"{portion} Counting MAE: {train_counting_mae}", file=output_f)
    print(f"{portion} Counting MAE: {train_counting_mae}")
    print(f"{portion} Counting Acc: {train_counting_acc}", file=output_f)
    print(f"{portion} Counting Acc: {train_counting_acc}")
    print("#" * 40, file=output_f)
    print("#" * 40)

    # -- Check if accuracy meets threshold (if specified)
    if args.checked_acc:
        print(f"<acc>{train_acc}</acc>")
        assert train_acc > args.checked_acc

    return 0

if __name__ == '__main__':
    args = parse_arguments()
    main(args)