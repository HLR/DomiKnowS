import torch

from regr.program import POIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor, EdgeSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor

from conll.data.data import SingletonDataLoader


import spacy
# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm') #English()


from transformers import BertTokenizerFast, BertModel

TRANSFORMER_MODEL = 'bert-base-uncased'

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

# emb_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
# to freeze BERT, uncomment the following
# for param in emb_model.base_model.parameters():
#     param.requires_grad = False
class BERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        # to freeze BERT, uncomment the following
        # for param in emb_model.base_model.parameters():
        #     param.requires_grad = False

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
        # softmax = torch.nn.Softmax(dim=-1)
        # super().__init__(linear, softmax)


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
    word['bert'] = ModuleLearner('ids', module=BERT())

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

    phrase[people] = ModuleLearner('w2v', module=Classifier(96))
    phrase[organization] = ModuleLearner('w2v', module=Classifier(96))
    phrase[location] = ModuleLearner('w2v', module=Classifier(96))
    phrase[other] = ModuleLearner('w2v', module=Classifier(96))
    phrase[o] = ModuleLearner('w2v', module=Classifier(96))

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

    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateSensor(
        phrase['w2v'],
        relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        forward=lambda *_, **__: True)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('w2v'), rel_pair_phrase2.reversed('w2v'),
        forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))

    pair[work_for] = ModuleLearner('emb', module=Classifier(96*2))
    pair[located_in] = ModuleLearner('emb', module=Classifier(96*2))
    pair[live_in] = ModuleLearner('emb', module=Classifier(96*2))
    pair[orgbase_on] = ModuleLearner('emb', module=Classifier(96*2))
    pair[kill] = ModuleLearner('emb', module=Classifier(96*2))

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

    lbp = POIProgram(
        graph,
        poi=(phrase, sentence, pair, word),
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        metric=PRF1Tracker())
    # lbp = IMLProgram(
    #     graph,
    #     loss=MacroAverageTracker(NBCrossEntropyIMLoss(lmbd=0.5)),
    #     metric=PRF1Tracker())

    return lbp


def main():
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill

    program = model()

    # Uncomment the following lines to enable training and testing
    # train_reader = SingletonDataLoader('data/conll04.corp_1_train.corp')
    #test_reader = SingletonDataLoader('data/conll04.corp_1_test.corp')
    # program.train(train_reader, train_epoch_num=2, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    #program.test(test_reader)

    reader = SingletonDataLoader('data/conll04.corp')

    for node in program.populate(reader, device='auto'):
        assert node.ontologyNode is sentence
        phrase_node = node.getChildDataNodes()[0]
        assert phrase_node.ontologyNode is phrase

        node.infer()

        if phrase_node.getAttribute(people) is not None:
            assert phrase_node.getAttribute(people, 'softmax') > 0
            node.inferILPResults(fun=None)
            
            ILPmetrics = node.getInferMetric()
            
            print("ILP metrics Total %s"%(ILPmetrics['Total']))
            
            assert phrase_node.getAttribute(people, 'ILP') >= 0
        else:
            print("%s phrases have no values for attribute people"%(node.getAttribute('text')))
            break


if __name__ == '__main__':
    main()
