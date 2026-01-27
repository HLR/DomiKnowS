import sys
import torch
from pathlib import Path

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
        mapping = torch.zeros(idx.shape[0], idx.max() + 1, device=self.device)
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
        
        # Start fully frozen
        self.freeze_all()
        
        # Track unfreezing state
        self.unfrozen_layers = 0
        self.total_layers = len(self.module.encoder.layer)  # Usually 12 for bert-base
        self.step_count = 0
    
    def freeze_all(self):
        """Freeze all BERT parameters."""
        for param in self.module.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, n_layers):
        """Unfreeze the last n_layers of BERT encoder."""
        if n_layers <= self.unfrozen_layers:
            return  # Already unfrozen
        
        # Unfreeze from the end
        for layer in self.module.encoder.layer[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Unfreeze pooler when we start unfreezing
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
        if input.device != self.device:
            input = input.to(self.device)
        
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

_bert_model = None
_models = {}

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
    

def program_declaration(train, args, device='auto'):
    global _models
    global _bert_model
    
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, \
        rel_sentence_contains_phrase

    graph.detach()

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

    pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    
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
    return program, train_dataset

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
                                          bert_lr=2e-5, classifier_lr=1e-4):
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting the arguments passed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
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
                        help="Device to use for computation (e.g., 'cuda', 'cpu', 'cuda:0', 'auto')")
    parser.add_argument("--unfreeze_every", type=int, default=500, 
                        help="Unfreeze BERT layers every N steps")
    parser.add_argument("--unfreeze_layers", type=int, default=2, 
                        help="Number of BERT layers to unfreeze per step")
    parser.add_argument("--bert_lr", type=float, default=2e-5, 
                        help="Learning rate for BERT layers")
    parser.add_argument("--classifier_lr", type=float, default=1e-4, 
                        help="Learning rate for classifier heads")
    # ...
    args = parser.parse_args()

    return args


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
            
def main(args):
    global _models
    global _bert_model
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o

    # Find the data file automatically (will use extracted file if exists)
    data_file_path = find_data_file(args.data_path)

    train, dev, test = conll4_reader(data_path=data_file_path, dataset_portion=args.train_portion)

    if args.train_size != -1:
        train = train[:args.train_size]

    program, dataset = program_declaration(train if not args.evaluate else test, args, device=args.device)

    suffix = "_curriculum_learning" if args.load_previous else ""
    if not args.evaluate:
        if args.load_previous:
            program.load(f"training_{args.epochs}_lr_{args.lr}_{args.previous_portion}.pth")
        
        # Setup gradual unfreezing callback
        def unfreeze_callback():
            epoch = program.epoch or 0
            layers_to_unfreeze = min(epoch * args.unfreeze_layers, 12)
            _models['bert'].unfreeze_layers(layers_to_unfreeze)
        
        # Setup optimizer update callback (for differential LR)
        def update_optimizer_callback():
            program.opt = create_optimizer_with_differential_lr(
                _models['bert'],
                _models['classifiers'],
                bert_lr=args.bert_lr,
                classifier_lr=args.classifier_lr
            )
        
        # Register callbacks
        program.after_train_epoch.append(unfreeze_callback)
        program.after_train_epoch.append(update_optimizer_callback)
        
        # Train - callbacks fire automatically after each epoch
        program.train(
            dataset, 
            Optim=lambda params: create_optimizer_with_differential_lr(
                _models['bert'],
                _models['classifiers'],
                bert_lr=args.bert_lr,
                classifier_lr=args.classifier_lr
            ),
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