import sys
import torch

sys.path.append('.')
sys.path.append('../..')

from regr.program import POIProgram, SolverPOIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.model.pytorch import SolverModel
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, FunctionalReaderSensor, cache, TorchCache
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor, EdgeSensor, CompositionCandidateReaderSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor

from conll.data.data import SingletonDataLoader


import spacy
# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm') #English()


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
        mapping = torch.zeros(idx.shape[0], idx.max()+1)
        mapping.scatter_(1, idx, 1)

        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:,:,0].masked_select(mask), offset[:,:,1].masked_select(mask)), dim=-1)
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
        out, *_ = self.module(input)
        assert out.shape[0] == 1
        out = out.squeeze(0)
        return out


class Classifier(torch.nn.Sequential):
    def __init__(self, in_features) -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)


def model():
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase

    graph.detach()

    phrase['text'] = ReaderSensor(keyword='tokens')
    phrase['postag'] = ReaderSensor(keyword='postag')

    def word2vec(text):
        texts = list(map(lambda x: ' '.join(x.split('/')), text))
        tokens_list = list(nlp.pipe(texts))
        return torch.tensor([tokens.vector for tokens in tokens_list])

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
                    word_overlap.append(overlap(ph_offset, ph_offset+ph_len, word_s, word_e))
            ph_word_overlap.append(word_overlap)
            ph_offset += ph_len + 1
        return torch.tensor(ph_word_overlap)
    phrase[rel_phrase_contains_word.reversed] = EdgeSensor(phrase['text'], word['offset'], relation=rel_phrase_contains_word.reversed, forward=match_phrase)
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
            label = torch.tensor([item==label_type for item in data])
            return label # torch.stack((~label, label), dim=1)
        return find
    phrase[people] = FunctionalReaderSensor(keyword='label', forward=find_label('Peop'), label=True)
    phrase[organization] = FunctionalReaderSensor(keyword='label', forward=find_label('Org'), label=True)
    phrase[location] = FunctionalReaderSensor(keyword='label', forward=find_label('Loc'), label=True)
    phrase[other] = FunctionalReaderSensor(keyword='label', forward=find_label('Other'), label=True)
    phrase[o] = FunctionalReaderSensor(keyword='label', forward=find_label('O'), label=True)

    # pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateSensor(
    #     phrase['text'],
    #     relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
    #     forward=lambda *_, **__: True)
    def filter_pairs(phrase_text, arg1, arg2, data):
        for rel, (rel_arg1, *_), (rel_arg2, *_) in data:
            if arg1.instanceID == rel_arg1 and arg2.instanceID == rel_arg2:
                return True
        return False
    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateReaderSensor(
        phrase['text'],
        relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        keyword='relation',
        forward=filter_pairs)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('emb'), rel_pair_phrase2.reversed('emb'),
        forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))

    pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))

    def find_relation(relation_type):
        def find(arg1m, arg2m, data):
            label = torch.zeros(arg1m.shape[0], dtype=torch.bool)
            for rel, (arg1,*_), (arg2,*_) in data:
                if rel == relation_type:
                    i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
                    label[i] = True
            return label # torch.stack((~label, label), dim=1)
        return find
    pair[work_for] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Work_For'), label=True)
    pair[located_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Located_In'), label=True)
    pair[live_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Live_In'), label=True)
    pair[orgbase_on] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('OrgBased_In'), label=True)
    pair[kill] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Kill'), label=True)

    lbp = PrimalDualProgram(
        graph,Model=SolverModel ,poi=(sentence, phrase, pair), inferTypes=['ILP', 'local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        metric={
            'ILP': PRF1Tracker(DatanodeCMMetric()),
            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return lbp


def main():
    from graph import graph, sentence, word, phrase, pair
    program = model()

    split_id = 2
    train_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_train.corp')
    test_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_test.corp')
    program.train(train_reader, test_set=test_reader, train_epoch_num=10, Optim=lambda param: torch.optim.SGD(param, lr=.001), device='cuda:0')
    program.test(test_reader, device='cuda:0')
#     for datanode, data_item in program.populate(train_reader, device="cuda:0"):
#         print(datanode)
#         ph = len(datanode.getChildDataNodes(conceptName=phrase))
#         t = len(data_item['tokens'])
#         if ph != t:
#             print("this example is wrong")
            
#         if len(data_item['relation']):
#             print("I have relations")
            
#         if len(data_item['relation']) != len(datanode.findDatanodes(select = pair)):
#             print(data_item['relation'])
#             print(datanode.findDatanodes(select = pair))
#             print("this example is wrong")
#         print(data_item)
        
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    program.save(f'conll04-bert-iml-{split_id}-{now}.pt')


if __name__ == '__main__':
    main()
