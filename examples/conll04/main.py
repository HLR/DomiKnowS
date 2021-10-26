import sys
import torch

sys.path.append('.')
sys.path.append('../..')


from regr.program import POIProgram, SolverPOIProgram, IMLProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor, FunctionalReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor

from conll.data.data import SingletonDataLoader


import spacy
# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm') #English()

FEATURE_DIM = 96

import logging

logging.basicConfig(level=logging.INFO)

class Classifier(torch.nn.Sequential):
    def __init__(self, in_features) -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)
        # softmax = torch.nn.Softmax(dim=-1)
        # super().__init__(linear, softmax)


def model(use_ont):
    if use_ont:
        from graph_ont import graph_ont, sentence, word, phrase, pair
        from graph_ont import people, organization, location, other, o
        from graph_ont import work_for, located_in, live_in, orgbase_on, kill
        from graph_ont import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase
        
        graph_ont.detach()
    else:
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

    phrase[people] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
    phrase[organization] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
    phrase[location] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
    phrase[other] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
    phrase[o] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))

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

    POSTAGS_MAKE_PAIRS = ['NN', 'NNP', 'NNPS', 'NNS']
    def filter_pairs(phrase_postag, arg1, arg2):
        return arg1 is not arg2 and arg1.getAttribute('postag') in POSTAGS_MAKE_PAIRS and arg2.getAttribute('postag') in POSTAGS_MAKE_PAIRS
    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateSensor(
        phrase['postag'],
        relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        forward=filter_pairs)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('w2v'), rel_pair_phrase2.reversed('w2v'),
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

    if use_ont:
        lbp = SolverPOIProgram(graph_ont, poi=(sentence, phrase, pair), inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    else:
        lbp = SolverPOIProgram(graph, poi=(sentence, phrase, pair), inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric={'internal':PRF1Tracker(),'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})
        # lbp = IMLProgram(graph, poi=(sentence, phrase, pair), inferTypes=['ILP', 'local/argmax'], loss=MacroAverageTracker(NBCrossEntropyIMLoss(lmbd=0.5)), metric={'internal':PRF1Tracker(),'ILP':PRF1Tracker(DatanodeCMMetric()),'argmax':PRF1Tracker(DatanodeCMMetric('local/argmax'))})
    
    return lbp

def main():
    
    val = 'n'
    #while True:
    #    val = input("Use ontology constraint (y/n):")
    #   if val == 'y' or val == 'n':
    #break
        
    if val == 'y':
        use_ont = True
        from graph_ont import graph_ont, sentence, phrase, people, organization, location, other, o
    else:
        use_ont = False
        from graph import  sentence, phrase, people, organization, location, other, o

    program = model(use_ont)

    # Uncomment the following lines to enable training and testing
    # train_reader = SingletonDataLoader('data/conll04.corp_1_train.corp')
    # train_reader = SingletonDataLoader('data/conll04-one.corp')
    # test_reader = SingletonDataLoader('data/conll04.corp_1_test.corp')
    # program.train(train_reader, test_set=train_reader, train_epoch_num=1, Optim=lambda param: torch.optim.SGD(param, lr=.001), device='auto')
    # program.test(test_reader, device='auto')
    # from datetime import datetime
    # now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    # program.save(f'conll04-{now}.pt')

    reader = SingletonDataLoader('data/conll04-washington.corp')

    for i, node in enumerate(program.populate(reader, device='auto')):
        if i == 0:
            node.getChildDataNodes()[0].visualize('testnode')
        assert node.ontologyNode is sentence
        phrase_node = node.getChildDataNodes()[0]
        assert phrase_node.ontologyNode is phrase

        node.infer()
        node.inferLocal()
        argmax_metrics = node.getInferMetric(inferType='local/argmax')
        print("\nArgmax metrics Total %s"%(argmax_metrics['Total']))
        print("\nArgmax metrics work_for %s"%(argmax_metrics['work_for']))
        print("\nArgmax metrics people %s"%(argmax_metrics['people']))

        if phrase_node.getAttribute(people) is not None:
            assert phrase_node.getAttribute(people, 'softmax') > 0
            node.inferILPResults(fun=None)
            
            ILPmetrics = node.getInferMetric()
            
            print("\nILP metrics Total %s"%(ILPmetrics['Total']))
            print("ILP metrics work_for %s"%(ILPmetrics['work_for']))
            ILPmetrics = node.getInferMetric(people)
            print("ILP metrics people %s"%(ILPmetrics['people']))

            assert phrase_node.getAttribute(people, 'ILP') >= 0
        else:
            print("%s phrases have no values for attribute people"%(node.getAttribute('text')))
            break


if __name__ == '__main__':
    main()
