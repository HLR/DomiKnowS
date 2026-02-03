import io
import re
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
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, TorchSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor

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

def find_data_file(filename):
    """Find data file in the same directory as this script."""
    current_dir = Path(__file__).parent
    data_path = current_dir / filename
    
    if data_path.exists():
        print(f"Using data file: {data_path}")
        return str(data_path)
    
    raise FileNotFoundError(f"Could not find {filename} in {current_dir}")

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
    
    def forward(self, input):
        if self.module.device != input.device:
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

class InferenceProgramWithCallbacks(CallbackProgram, InferenceProgram):
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
        device=device
    )
    
    dev_dataset = None
    if dev is not None and len(dev) > 0:
        dev_dataset = graph.compile_executable(dev, logic_keyword='logic_str', logic_label_keyword='logic_label')
    
    return program, train_dataset, dev_dataset

    # def find_label(label_type):
    #    def find(data):
    #        label = torch.tensor([item == label_type for item in data], device=device)
    #        return label
    #    return find

    # Normal Label
    # phrase[people] = FunctionalReaderSensor(keyword='label', forward=find_label('Peop'), label=True)
    # phrase[organization] = FunctionalReaderSensor(keyword='label', forward=find_label('Org'), label=True)
    # phrase[location] = FunctionalReaderSensor(keyword='label', forward=find_label('Loc'), label=True)
    # phrase[other] = FunctionalReaderSensor(keyword='label', forward=find_label('Other'), label=True)
    # phrase[o] = FunctionalReaderSensor(keyword='label', forward=find_label('O'), label=True)

    # Below Code is for relation

    # pair[work_for] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed],
    #                                         keyword='relation', forward=find_relation('Work_For'), label=True)
    # pair[located_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed],
    #                                           keyword='relation', forward=find_relation('Located_In'), label=True)
    # pair[live_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed],
    #                                        keyword='relation', forward=find_relation('Live_In'), label=True)
    # pair[orgbase_on] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed],
    #                                           keyword='relation', forward=find_relation('OrgBased_In'), label=True)
    # pair[kill] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed],
    #                                     keyword='relation', forward=find_relation('Kill'), label=True)

def create_optimizer_with_differential_lr(bert_model, classifiers, 
                                          bert_lr=2e-5, classifier_lr=1e-6):
    """Create optimizer with different learning rates for BERT vs classifiers."""
    
    param_groups = [
        # BERT parameters (lower LR)
        {
            'params': [p for p in bert_model.parameters() if p.requires_grad],
            'lr': bert_lr,
            'name': 'bert'
        },
        # Classifier parameters (higher LR)
        {
            'params': [p for name, clf in classifiers.items() 
                        for p in clf.parameters() if p.requires_grad],
            'lr': classifier_lr,
            'name': 'classifiers'
        }
    ]
    
    # Remove empty param groups
    param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
    
    return torch.optim.Adam(param_groups)

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

def evaluate_with_counting_metrics(program, dataset, threshold=0.5):
    """Evaluate and capture both boolean and counting metrics."""
    
    # Capture stdout to parse the printed metrics
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        result = program.evaluate_condition(dataset, threshold=threshold)
    finally:
        sys.stdout = old_stdout
    
    output = captured_output.getvalue()
    print(output)  # Still print it for visibility
    
    # Parse metrics from output
    # "Boolean accuracy: 70.24% (59/84)"
    bool_match = re.search(r'Boolean accuracy:\s*([\d.]+)%', output)
    bool_acc = float(bool_match.group(1)) / 100.0 if bool_match else 0.0
    
    # "Counting MAE: 9.449, Accuracy (±0.5): 4.31%"
    mae_match = re.search(r'Counting MAE:\s*([\d.]+)', output)
    counting_mae = float(mae_match.group(1)) if mae_match else float('inf')
    
    count_acc_match = re.search(r'Accuracy \(±0\.5\):\s*([\d.]+)%', output)
    counting_acc = float(count_acc_match.group(1)) / 100.0 if count_acc_match else 0.0
    
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
                            classifier_lr=args.classifier_lr
                        )
                program.before_train_epoch.append(unfreeze_callback)
            
            initial_optimizer = create_optimizer_with_differential_lr(
                _models['bert'],
                _models['classifiers'],
                bert_lr=0.0 if args.freeze_bert else args.bert_lr,
                classifier_lr=args.classifier_lr
            )
            
            print(f"  [Trial {trial.number}] Training started...")
            
            program.train(
                dataset,
                Optim=lambda params: initial_optimizer,
                train_epoch_num=args.epochs,
                c_lr=args.classifier_lr,
                c_warmup_iters=-1,
                batch_size=1,
                print_loss=False
            )
            
            print(f"  [Trial {trial.number}] Evaluating...")
            bool_acc, counting_mae, counting_acc = evaluate_with_counting_metrics(program, dataset)
            
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

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Getting the arguments passed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs")
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--load_previous", action='store_true')
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training sample")
    parser.add_argument("--train_portion", type=str, default="entities_with_relation", help="Training subset")
    parser.add_argument("--previous_portion", type=str, default="entities_only_with_1_things_YN",
                        help="Training subset")
    parser.add_argument("--checked_acc", type=float, default=0, help="Accuracy to test")
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP"], default="G",
                        help="The tnorm method to use for the counting constraints")
    parser.add_argument("--data_path", type=str,
                        default="conllQA_with_global.json",
                        help="Path to data file (can be relative or absolute)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for computation (e.g., 'cuda', 'cpu', 'cuda:0')")
    
    # Learning rate arguments
    parser.add_argument("--classifier_lr", type=float, default=1e-6,
                        help="Learning rate for classifier heads")
    # Optuna arguments
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=False,
                        help="Run Optuna hyperparameter tuning (default: true)")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--tune_train_size", type=int, default=200,
                    help="Number of samples to use during tuning (smaller = faster)")

    # BERT freezing
    parser.add_argument("--freeze_bert", type=str2bool, nargs='?', const=True, default=True,
                        help="Keep BERT frozen throughout training (default: true)")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Epochs to train with BERT frozen before unfreezing")
    parser.add_argument("--unfreeze_every", type=int, default=500, 
                        help="Unfreeze BERT layers every N steps")
    parser.add_argument("--unfreeze_layers", type=int, default=2, 
                        help="Number of BERT layers to unfreeze per step")
    parser.add_argument("--bert_lr", type=float, default=1e-5,
                        help="Learning rate for BERT layers (if not frozen)")
    
    # Evaluation settings
    parser.add_argument("--eval_fraction", type=float, default=0.2,
                    help="Fraction of data for epoch evaluation (0.2 = 20%)")
    

    args = parser.parse_args()
    return args

def create_epoch_logging_callback(program, dataset, models, eval_fraction=0.2, min_samples=50, seed=42):
    """Create callback to log training metrics after each epoch.
    
    Args:
        eval_fraction: Fraction of dataset to evaluate (0.2 = 20%)
        min_samples: Minimum number of samples to evaluate
        seed: Random seed for reproducible subset selection
    """
    
    # Create fixed evaluation subset for consistent comparisons
    random.seed(seed)
    dataset_list = list(dataset)
    n_total = len(dataset_list)
    n_eval = max(min_samples, int(n_total * eval_fraction))
    n_eval = min(n_eval, n_total)  # Don't exceed dataset size
    
    eval_indices = sorted(random.sample(range(n_total), n_eval))
    eval_subset = [dataset_list[i] for i in eval_indices]
    
    print(f"[Eval] Using {n_eval}/{n_total} samples ({100*n_eval/n_total:.1f}%) for epoch evaluation")
    
    metrics_history = {
        'epoch': [],
        'train_acc': [],
        'clf_grad_norm': [],
    }
    
    def log_epoch_metrics():
        epoch = program.epoch or 0
        
        # Fast evaluation on subset
        train_acc = program.evaluate_condition(eval_subset, threshold=0.5)
        
        # Calculate classifier gradient norms
        clf_grad_norm = 0.0
        for name, clf in models['classifiers'].items():
            for p in clf.parameters():
                if p.grad is not None:
                    clf_grad_norm += p.grad.data.norm(2).item() ** 2
        clf_grad_norm = clf_grad_norm ** 0.5
        
        metrics_history['epoch'].append(epoch)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['clf_grad_norm'].append(clf_grad_norm)
        
        bert_status = f"frozen" if models['bert'].unfrozen_layers == 0 else f"{models['bert'].unfrozen_layers}L unfrozen"
        
        # Compact single-line output
        acc_change = ""
        if len(metrics_history['train_acc']) >= 2:
            delta = train_acc - metrics_history['train_acc'][-2]
            acc_change = f" (Δ{delta:+.3f})"
        
        print(f"[Epoch {epoch}] Acc: {train_acc:.4f}{acc_change} | GradNorm: {clf_grad_norm:.4f} | BERT: {bert_status}")
        
        # Warnings only
        if clf_grad_norm < 1e-7:
            print(f"  ⚠️  Gradients near zero!")
        if len(metrics_history['train_acc']) >= 2 and train_acc < metrics_history['train_acc'][-2] - 0.02:
            print(f"  ⚠️  Accuracy dropped!")
        
        return metrics_history
    
    return log_epoch_metrics, metrics_history, eval_subset

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

    # Optuna tuning mode
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
    
    if not args.evaluate:
        _models['bert'].freeze_all()
        
        # Create epoch logging callback
        log_epoch_metrics, metrics_history, eval_subset = create_epoch_logging_callback(
            program, dataset, _models,
            eval_fraction=0.1,
            min_samples=50,
            seed=42
        )
        program.after_train_epoch.append(log_epoch_metrics)
        
        if args.freeze_bert:
            print("[Training] BERT frozen throughout training (--freeze_bert)")
        else:
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
                        classifier_lr=args.classifier_lr
                    )
                    print(f"[Epoch {epoch}] Unfroze {layers_to_unfreeze} layers, optimizer updated")
            
            program.before_train_epoch.append(unfreeze_callback)
        
        initial_optimizer = create_optimizer_with_differential_lr(
            _models['bert'],
            _models['classifiers'],
            bert_lr=0.0 if args.freeze_bert else args.bert_lr,
            classifier_lr=args.classifier_lr
        )
        
        program.train(
            dataset,
            Optim=lambda params: initial_optimizer,
            train_epoch_num=args.epochs,
            c_lr=args.classifier_lr,
            c_warmup_iters=-1,
            batch_size=1,
            print_loss=False
        )
        
        # Print final training summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 60)
        if len(metrics_history['epoch']) > 0:
            print(f"  Initial Accuracy: {metrics_history['train_acc'][0]:.4f}")
            print(f"  Final Accuracy:   {metrics_history['train_acc'][-1]:.4f}")
            print(f"  Total Improvement: {metrics_history['train_acc'][-1] - metrics_history['train_acc'][0]:+.4f}")
            
            # Check for proper learning
            if metrics_history['train_acc'][-1] > metrics_history['train_acc'][0] + 0.05:
                print("  ✅ Model is learning well!")
            elif metrics_history['train_acc'][-1] > metrics_history['train_acc'][0]:
                print("  ⚠️  Model is learning slowly - consider more epochs or higher LR")
            else:
                print("  ❌ Model is NOT learning - check LR, data, or architecture")
        print("=" * 60 + "\n")
                
        program.save(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}.pth")
    else:
        program.load(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}.pth")

    output_f = open("result.txt", 'a')
    print("BERT params device:", next(_models['bert'].parameters()).device)
    train_acc = program.evaluate_condition(dataset, threshold=0.5)
    portion = "Training" if not args.evaluate else "Testing"
    print(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}", file=output_f)
    print(f"training_{args.epochs}_lr_{args.classifier_lr}_{args.train_portion}{suffix}")
    print(f"{portion} Acc: {train_acc}", file=output_f)
    print(f"{portion} Acc: {train_acc}")
    print("#" * 40, file=output_f)
    print("#" * 40)

    if args.checked_acc:
        print(f"<acc>{train_acc}</acc>")
        assert train_acc > args.checked_acc

    return 0

if __name__ == '__main__':
    args = parse_arguments()
    main(args)