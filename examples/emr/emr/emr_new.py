import os
from regr.graph import Graph, Concept, Relation
from regr.sensor.allennlp.sensor import TokenSequenceSensor, LabelSequenceSensor
from regr.sensor.allennlp.learner import W2VLearner, LSTMLearner, LRLearner
from emr.data import Conll04SensorReader as Reader


Graph.clear()
Concept.clear()
Relation.clear()


def seed1():
    import random
    import numpy as np
    import torch

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

seed1()


with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        phrase = Concept(name='phrase')
        pair = Concept(name='pair')
        pair.has_a(phrase, phrase)
    with Graph('application') as app_graph:
        people = Concept(name='people')
        organization = Concept(name='organization')
        people.is_a(phrase)
        organization.is_a(phrase)
        people.not_a(organization)
        organization.not_a(people)
        work_for = Concept(name='work_for')
        work_for.is_a(pair)
        work_for.has_a(employee=people, employer=organization)


def main():
    graph.detach()

    phrase = graph['linguistic/phrase']
    people = graph['application/people']
    organization = graph['application/organization']

    reader = Reader()

    phrase['tokens'] = TokenSequenceSensor(reader, 'tokens')
    phrase['pos_tag'] = LabelSequenceSensor(reader, 'pos')
    phrase['w2v'] = W2VLearner(phrase['tokens'])
    phrase['emb'] = LSTMLearner(phrase['w2v'])

    people['label'] = LabelSequenceSensor(reader, 'Peop')
    organization['label'] = LabelSequenceSensor(reader, 'Org')

    people['label'] = LRLearner()
    organization['label'] = LRLearner()

    # data setting
    relative_path = "data/EntityMentionRelation"
    train_path = "conll04_train.corp"
    valid_path = "conll04_test.corp"

    reader.read(os.path.join(relative_path, train_path))

    # ...
    print('-' * 40)
    print(graph)


if __name__ == '__main__':
    main()