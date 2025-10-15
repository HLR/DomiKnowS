import sys
import torch

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
    FunctionalReaderSensor, cache, TorchCache
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


class Tokenizer():
    def __init__(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_MODEL)

    def __call__(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text, padding=True, return_tensors='pt', return_offsets_mapping=True)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        offset = tokens['offset_mapping']

        idx = mask.nonzero()[:, 0].unsqueeze(-1)
        mapping = torch.zeros(idx.shape[0], idx.max() + 1)
        mapping.scatter_(1, idx, 1)

        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:, :, 0].masked_select(mask), offset[:, :, 1].masked_select(mask)), dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return mapping, ids, offset, tokens


class BERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        # to freeze BERT, uncomment the following
        for param in self.module.base_model.parameters():
            param.requires_grad = False

    def forward(self, input):
        input = input.unsqueeze(0)
        _out = self.module(input)

        out, *_ = _out

        if (isinstance(out, str)):  # Update for new transformers
            out = _out.last_hidden_state

        assert out.shape[0] == 1
        out = out.squeeze(0)
        return out


class Classifier(torch.nn.Sequential):
    def __init__(self, in_features) -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)


def program_declaration(device='auto'):
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, \
        rel_sentence_contains_phrase

    graph.detach()

    phrase['text'] = ReaderSensor(keyword='tokens')

    def word2vec(text):
        texts = list(map(lambda x: ' '.join(x.split('/')), text))
        tokens_list = list(nlp.pipe(texts))
        return torch.tensor(np.array([tokens.vector for tokens in tokens_list]))

    phrase['w2v'] = FunctionalSensor('text', forward=word2vec)

    def merge_phrase(phrase_text):
        return [' '.join(phrase_text)], torch.ones((1, len(phrase_text)))

    sentence['text', rel_sentence_contains_phrase.reversed] = JointSensor(phrase['text'], forward=merge_phrase)

    word[rel_sentence_contains_word, 'ids', 'offset', 'text'] = JointSensor(sentence['text'], forward=Tokenizer())
    word['bert'] = ModuleSensor('ids', module=BERT())

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
        return torch.tensor(ph_word_overlap)

    phrase[rel_phrase_contains_word.reversed] = EdgeSensor(phrase['text'], word['offset'],
                                                           relation=rel_phrase_contains_word.reversed,
                                                           forward=match_phrase)

    def phrase_bert(bert):
        return bert

    phrase['bert'] = FunctionalSensor(rel_phrase_contains_word.reversed(word['bert']), forward=phrase_bert)
    phrase['emb'] = FunctionalSensor('bert', 'w2v', forward=lambda bert, w2v: torch.cat((bert, w2v), dim=-1))

    phrase[people] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[organization] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[location] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[other] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[o] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))

    def find_label(label_type):
        def find(data):
            label = torch.tensor([item == label_type for item in data])
            return label

        return find

    # Normal Label
    # phrase[people] = FunctionalReaderSensor(keyword='label', forward=find_label('Peop'), label=True)
    # phrase[organization] = FunctionalReaderSensor(keyword='label', forward=find_label('Org'), label=True)
    # phrase[location] = FunctionalReaderSensor(keyword='label', forward=find_label('Loc'), label=True)
    # phrase[other] = FunctionalReaderSensor(keyword='label', forward=find_label('Other'), label=True)
    # phrase[o] = FunctionalReaderSensor(keyword='label', forward=find_label('O'), label=True)

    # Below Code is for relation
    # def filter_pairs(phrase_text, arg1, arg2, data):
    #     for rel, (rel_arg1, *_), (rel_arg2, *_) in data:
    #         if arg1.instanceID == rel_arg1 and arg2.instanceID == rel_arg2:
    #             return True
    #     return False
    #
    # pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateReaderSensor(
    #     phrase['text'],
    #     relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
    #     keyword='relation',
    #     forward=filter_pairs)
    # pair['emb'] = FunctionalSensor(
    #     rel_pair_phrase1.reversed('emb'), rel_pair_phrase2.reversed('emb'),
    #     forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))
    #
    # pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    # pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    # pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    # pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    # pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    #
    # def find_relation(relation_type):
    #     def find(arg1m, arg2m, data):
    #         label = torch.zeros(arg1m.shape[0], dtype=torch.bool)
    #         for rel, (arg1, *_), (arg2, *_) in data:
    #             if rel == relation_type:
    #                 i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
    #                 label[i] = True
    #         return label  # torch.stack((~label, label), dim=1)
    #
    #     return find
    #
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

    program = SolverPOIProgram(graph, poi=[sentence, phrase, people, organization], inferTypes=["local/argmax"],
                               loss=MacroAverageTracker(NBCrossEntropyLoss()),
                               metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    return program


def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting the arguments passed")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()

    return args


def main(args):
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    program = program_declaration()

    train, dev, test = conll4_reader(data_path="conllQA.json", dataset_portion="entities_only_with_1_things_YN")

    train_dataset = graph.compile_logic(train, logic_keyword='logic_str', logic_label_keyword='logic_label')

    program = InferenceProgram(graph, SolverModel,
                               poi=[phrase, sentence, word, people, organization, location, graph.constraint],
                               tnorm="G", inferTypes=['local/argmax'])

    program.train(train_dataset, Optim=torch.optim.Adam, train_epoch_num=args.epochs, c_lr=args.lr, c_warmup_iters=-1,
                  batch_size=1, print_loss=False)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
