import sys
import torch
from pathlib import Path

sys.path.append('.')
sys.path.append('../..')

import argparse
from domiknows.program import POIProgram, SolverPOIProgram, IMLProgram, CallbackProgram
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

import logging

logging.basicConfig(level=logging.INFO)

from transformers import BertTokenizerFast, BertModel

TRANSFORMER_MODEL = 'bert-base-uncased'

FEATURE_DIM = 768 + 96


def find_data_file(filename, train_portion=None):
    """Find data file by checking multiple possible locations"""
    current_dir = Path(__file__).parent

    # First, check if extracted portion file exists
    if train_portion:
        extracted_filename = f"{train_portion}.json"
        possible_extracted_paths = [
            current_dir / extracted_filename,
            current_dir / "data" / extracted_filename,
            Path.cwd() / extracted_filename,
            Path.cwd() / "data" / extracted_filename,
        ]
        
        for path in possible_extracted_paths:
            if path.exists():
                print(f"Using extracted data file: {path}")
                return str(path)

    # List of possible locations to check for main file
    possible_paths = [
        current_dir / filename,
        current_dir / "data" / filename,
        current_dir / ".." / filename,
        current_dir / ".." / "data" / filename,
        current_dir / ".." / ".." / filename,
        current_dir / ".." / ".." / "data" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"Could not find {filename} in any of the expected locations: {possible_paths}")


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
        # to freeze BERT, uncomment the following
        for param in self.module.base_model.parameters():
            param.requires_grad = False

    def forward(self, input):
        # Ensure input is on the correct device
        if input.device != self.device:
            input = input.to(self.device)
        
        input = input.unsqueeze(0)
        _out = self.module(input)

        out, *_ = _out

        if (isinstance(out, str)):  # Update for new transformers
            out = _out.last_hidden_state

        assert out.shape[0] == 1
        out = out.squeeze(0)
        return out
    
    def to(self, device):
        """Override to method to update self.device"""
        self.device = device
        return super().to(device)


class Classifier(torch.nn.Sequential):
    def __init__(self, in_features, device='cpu') -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)
        self.to(device)

def program_declaration(train, args, device='auto'):
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

    def find_label(label_type):
        def find(data):
            label = torch.tensor([item == label_type for item in data], device=device)
            return label

        return find

    train_dataset = graph.compile_logic(train, logic_keyword='logic_str', logic_label_keyword='logic_label')

    program = InferenceProgram(graph, SolverModel,
                               poi=[phrase, sentence, word, people, organization, location, graph.constraint],
                               tnorm=args.counting_tnorm, inferTypes=['local/argmax'], device=device)
    return program, train_dataset

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

    


def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting the arguments passed")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--load_previous", action='store_true')
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training sample")
    parser.add_argument("--train_portion", type=str, default="entities_with_relation", help="Training subset")
    parser.add_argument("--previous_portion", type=str, default="entities_only_with_1_things_YN", help="Training subset")
    parser.add_argument("--checked_acc", type=float, default=0, help="Accuracy to test")
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP"], default="G", help="The tnorm method to use for the counting constraints")
    parser.add_argument("--data_path", type=str, default="C:\\Users\\auszok\\git\\RelationalGraph\\test_regr\\ConllQA\\conllQA2.json", help="Path to data file (can be relative or absolute)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (e.g., 'cuda', 'cpu', 'cuda:0', 'auto')")
    args = parser.parse_args()

    return args


def main(args):
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o

    # Find the data file automatically (will use extracted file if exists)
    data_file_path = find_data_file(args.data_path, args.train_portion)

    train, dev, test = conll4_reader(data_path=data_file_path, dataset_portion=args.train_portion)

    if args.train_size != -1:
        train = train[:args.train_size]

    program, dataset = program_declaration(train if not args.evaluate else test, args, device=args.device)

    suffix = "_curriculum_learning" if args.load_previous else ""
    if not args.evaluate:
        if args.load_previous:
            program.load(f"training_{args.epochs}_lr_{args.lr}_{args.previous_portion}.pth")
        program.train(dataset, Optim=torch.optim.Adam, train_epoch_num=args.epochs, c_lr=args.lr, c_warmup_iters=-1,
                      batch_size=1, print_loss=False)
        program.save(f"training_{args.epochs}_lr_{args.lr}_{args.train_portion}{suffix}.pth")
    else:
        program.load(f"training_{args.epochs}_lr_{args.lr}_{args.train_portion}{suffix}.pth")

    output_f = open("result.txt", 'a')
    train_acc = program.evaluate_condition(dataset, threshold=0.5)
    portion = "Training" if not args.evaluate else "Testing"
    print(f"training_{args.epochs}_lr_{args.lr}_{args.train_portion}{suffix}", file=output_f)
    print(f"{portion} Acc: {train_acc}", file=output_f)
    print("#" * 40, file=output_f)

    if args.checked_acc:
        print(f"<acc>{train_acc}</acc>")
        assert train_acc > args.checked_acc

    return 0


if __name__ == '__main__':
    args = parse_arguments()
    main(args)