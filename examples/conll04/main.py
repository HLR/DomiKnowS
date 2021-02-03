import torch

from regr.program import POIProgram
from regr.sensor.pytorch.sensors import JointSensor, ReaderSensor

from conll.data.data import SingletonDataLoader


def model():
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase

    graph.detach()

    phrase['text'] = ReaderSensor(keyword='tokens')
    phrase['postag'] = ReaderSensor(keyword='postag')

    def merge_phrase(phrase_text):
        return [' '.join(phrase_text)], torch.ones((1, len(phrase_text)))
    sentence[rel_sentence_contains_phrase.reversed, 'text'] = JointSensor(phrase['text'], forward=merge_phrase)

    word[rel_sentence_contains_word, 'text'] = JointSensor(sentence, )

    lbp = POIProgram(graph, poi=(phrase, sentence,))
    return lbp


def main():
    program = model()
    reader = SingletonDataLoader('data/conll04.corp')

    for node in program.populate(reader, device='auto'):
        node


if __name__ == '__main__':
    main()
