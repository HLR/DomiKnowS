import torch

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor

from conll.data.data import SingletonDataLoader

import spacy
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm') #English()

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

    # word[rel_sentence_contains_word, 'text'] = JointSensor(sentence, )

    phrase[people] = ModuleLearner('w2v', module=Classifier(96))
    phrase[organization] = ModuleLearner('w2v', module=Classifier(96))
    phrase[location] = ModuleLearner('w2v', module=Classifier(96))
    phrase[other] = ModuleLearner('w2v', module=Classifier(96))
    phrase[o] = ModuleLearner('w2v', module=Classifier(96))

    def find_label(label):
        def find(data):
            return torch.tensor([item==label for item in data])
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

    def find_relation(label):
        def find(arg1m, arg2m, data):
            label = torch.zeros(arg1m.shape[0], dtype=torch.long)
            for rel, (arg1,*_), (arg2,*_) in data:
                i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
                label[i] = 1
            return label
        return find
    pair[work_for] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Work_For'), label=True)
    pair[located_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Located_In'), label=True)
    pair[live_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Live_In'), label=True)
    pair[orgbase_on] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('OrgBased_In'), label=True)
    pair[kill] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Kill'), label=True)

    # def find_relation(label):
    #     def find(*_, datanode, data):
    #         arg1 = datanode.relationLinks['arg1'][0]
    #         arg2 = datanode.relationLinks['arg2'][0]
    #         arg1.getAttribute('index')
    #         for arg1, arg2 in data:
    #             i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
    #             label[i] = 1
    #         return label
    #     return find
    # pair[work_for] = DataNodeReaderSensor(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed, keyword='relation', forward=find_relation('Work_For'), label=True)
    # pair[located_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Located_In'), label=True)
    # pair[live_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Live_In'), label=True)
    # pair[orgbase_on] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('OrgBased_In'), label=True)
    # pair[kill] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Kill'), label=True)

    lbp = POIProgram(graph, poi=(phrase, sentence, pair))
    return lbp


def main():
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill

    program = model()
    reader = SingletonDataLoader('data/conll04.corp')

    # program.train(reader, device='auto')

    for node in program.populate(reader, device='auto'):
        assert node.ontologyNode is sentence
        phrase_node = node.getChildDataNodes()[0]
        assert phrase_node.ontologyNode is phrase

        node.infer()
        assert phrase_node.getAttribute(people, 'softmax') > 0
        node.inferILPResults(fun=None)
        assert phrase_node.getAttribute(people, 'ILP') >= 0


if __name__ == '__main__':
    main()
