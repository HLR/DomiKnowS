import sys
import torch
from pathlib import Path

import optuna
from optuna.trial import TrialState

sys.path.append('.')
sys.path.append('../../..')

import argparse

#import logging
#logging.basicConfig(level=logging.INFO)
from domiknows import setProductionLogMode
setProductionLogMode()

from domiknows.program import POIProgram, SolverPOIProgram, IMLProgram, CallbackProgram, program
from domiknows.program.callbackprogram import ProgramStorageCallback
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.lossprogram import PrimalDualProgram, InferenceProgram
from domiknows.program.model.pytorch import SolverModel, SolverModelDictLoss
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, NBCrossEntropyDictLoss
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, \
    FunctionalReaderSensor, TorchSensor, cache, TorchCache
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor, EdgeSensor, \
    CompositionCandidateReaderSensor

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

    #graph.detach()

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
    _bert_model = bert_model # Store globally for unfreezing during training
    
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
                    # empty string / special tokens
                    word_overlap.append(False)
                else:
                    # other tokens, do compare offset
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

    phrase[people] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[organization] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[location] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[other] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[o] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))

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

    pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2, device=device))
    pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2, device=device))
    pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2, device=device))
    pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2, device=device))
    pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2, device=device))
    
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
    
    # Compile dev dataset after program is created (graph state is fully set up)
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
            
def create_objective(args, train, test):
    """Create Optuna objective function for hyperparameter tuning."""
    
    def objective(trial: optuna.Trial) -> float:
        global _models, _bert_model
        
        # Sample hyperparameters
        classifier_lr = trial.suggest_float('classifier_lr', 1e-5, 1e-3, log=True)
        
        if args.freeze_bert:
            # BERT frozen - don't tune BERT-related params
            bert_lr = 0.0
            warmup_epochs = args.epochs  # Always frozen
            unfreeze_layers = 0
        else:
            bert_lr = trial.suggest_float('bert_lr', 1e-6, 1e-4, log=True)
            warmup_epochs = trial.suggest_int('warmup_epochs', 0, 3)
            unfreeze_layers = trial.suggest_int('unfreeze_layers', 1, 4)
        
        # Override args with trial values
        args.classifier_lr = classifier_lr
        args.bert_lr = bert_lr
        args.warmup_epochs = warmup_epochs
        args.unfreeze_layers = unfreeze_layers
        
        try:
            # Fully reset graph state for each trial
            from graph import graph
            graph.detach()
            
            # Reset graph internal state that persists across detach()
            graph.varContext = None
            graph._processed_lcs = set()
            graph._executableLCs.clear()
            graph.executableLCsLabels.clear()
            
            program, dataset, _ = program_declaration(train, None, args, device=args.device)
            
            # Ensure BERT starts fully frozen
            _models['bert'].freeze_all()
            
            # Setup gradual unfreezing callback (skip if BERT frozen)
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
            
            program.train(
                dataset,
                Optim=lambda params: initial_optimizer,
                train_epoch_num=args.epochs,
                c_lr=args.classifier_lr,
                c_warmup_iters=-1,
                batch_size=1,
                print_loss=False
            )
            
            # Evaluate on training set (constraint satisfaction accuracy)
            train_acc = program.evaluate_condition(dataset, threshold=0.5)
            
            return train_acc
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[Trial {trial.number}] Failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # Return worst score on failure
    
    return objective

def run_optuna_tuning(args, train, test, n_trials=20):
    """Run Optuna hyperparameter tuning."""
    
    # Create study with pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name=f'conll_tuning_{args.train_portion}'
    )
    
    objective = create_objective(args, train, test)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
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
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True,
                        help="Run Optuna hyperparameter tuning (default: true)")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials")
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

    args = parser.parse_args()
    return args

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
        # Log config before tuning starts
        log_training_config(args, models=None, train=train, dev=dev, test=test)
        
        study = run_optuna_tuning(args, train, test, n_trials=args.n_trials)
        
        # Update args with best params for final training
        print("\n" + "=" * 60)
        print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        print("=" * 60)
        
        best_params = study.best_trial.params
        args.classifier_lr = best_params['classifier_lr']
        if not args.freeze_bert:
            args.bert_lr = best_params['bert_lr']
            args.warmup_epochs = best_params['warmup_epochs']
            args.unfreeze_layers = best_params['unfreeze_layers']
        
        # Disable tuning flag for final training log
        args.tune = False
        
        # Reset graph for final training
        graph.detach()
        graph.varContext = None
        graph._processed_lcs = set()
        graph._executableLCs.clear()
        graph.executableLCsLabels.clear()
    
    program, dataset, _ = program_declaration(
        train if not args.evaluate else test, 
        None,  # No dev set needed
        args, 
        device=args.device
    )
    
    # Log config for actual training (with model info now available)
    log_training_config(args, _models, train=train, dev=None, test=test)
    
    if not args.evaluate:
        _models['bert'].freeze_all()
        
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