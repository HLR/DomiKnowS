import torch

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor

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

    lbp = POIProgram(graph, poi=(phrase, sentence, pair))
    return lbp


def main():
    program = model()
    reader = SingletonDataLoader('data/conll04.corp')

    for node in program.populate(reader, device='auto'):
        from graph import graph, sentence, word, phrase, pair
        assert node.ontologyNode is sentence


if __name__ == '__main__':
    main()
