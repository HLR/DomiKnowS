import torch

from regr.program import POIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor

from conll.data.data import SingletonDataLoader


import spacy
# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm') #English()


class Classifier(torch.nn.Sequential):
    def __init__(self, in_features, out_features) -> None:
        linear = torch.nn.Linear(in_features, out_features)
        super().__init__(linear)
        # softmax = torch.nn.Softmax(dim=-1)
        # super().__init__(linear, softmax)
    
    def forward(self, input):
        return super().forward(input)


def model():
    from graph_multi import graph_multi, sentence, word, phrase, pair
    from graph_multi import entity_label, pair_label
    from graph_multi import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase

    graph_multi.detach()

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

    phrase[entity_label] = ModuleLearner('w2v', module=Classifier(96, 5))

    def find_label():
        def find(data):
            order = ["Peop", "Org", "Loc", "Other", "O"]
            label = torch.tensor([order.index(item) for item in data])
            return label # torch.stack((~label, label), dim=1)
        return find
    
    #TODO change the reader
    phrase[entity_label] = FunctionalReaderSensor(keyword='label', forward=find_label(), label=True)

    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateSensor(
        phrase['w2v'],
        relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        forward=lambda *_, **__: True)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('w2v'), rel_pair_phrase2.reversed('w2v'),
        forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))

    pair[pair_label] = ModuleLearner('emb', module=Classifier(96*2, 5))

    def find_relation():
        def find(arg1m, arg2m, data):
            rel_list = ["Work_For", "Located_In", "Live_In", "OrgBased_In", "Kill"]
            label = torch.zeros(arg1m.shape[0], dtype=torch.bool)
            for rel, (arg1,*_), (arg2,*_) in data:
                if rel in rel_list:
                    i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
                    label[i] = rel_list.index(rel)
            return label # torch.stack((~label, label), dim=1)
        return find
    
    #TODO change the reader
    pair[pair_label] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation(), label=True)

    lbp = POIProgram(
        graph_multi,
        poi=(phrase, sentence, pair),
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        metric=PRF1Tracker())
    # lbp = IMLProgram(
    #     graph_multi,
    #     loss=MacroAverageTracker(NBCrossEntropyIMLoss(lmbd=0.5)),
    #     metric=PRF1Tracker())

    return lbp


def main():
    from graph_multi import graph_multi, sentence, word, phrase, pair
    from graph_multi import entity_label
    from graph_multi import pair_label

    program = model()

    # Uncomment the following lines to enable training and testing
    #train_reader = SingletonDataLoader('data/conll04.corp_1_train.corp')
    #test_reader = SingletonDataLoader('data/conll04.corp_1_test.corp')
    #program.train(train_reader, train_epoch_num=2, Optim=lambda param: torch.optim.SGD(param, lr=.001))
    #program.test(test_reader)

    reader = SingletonDataLoader('data/conll04.corp')

    for node in program.populate(reader, device='auto'):
        assert node.ontologyNode is sentence
        phrase_node = node.getChildDataNodes()[0]
        assert phrase_node.ontologyNode is phrase

        node.infer()

        if phrase_node.getAttribute(entity_label) is not None:
            #assert phrase_node.getAttribute(entity_label, 'softmax') > 0
            node.inferILPResults(fun=None)
            
            ILPmetrics = node.getInferMetric()
            
            print("ILP metrics Total %s"%(ILPmetrics['Total']))
            
            #assert phrase_node.getAttribute(entity_label, 'ILP') >= 0
        else:
            print("%s phrases have no values for attribute people"%(node.getAttribute('text')))
            break


if __name__ == '__main__':
    main()
