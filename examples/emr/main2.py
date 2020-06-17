from emr.data import NaiveDataLoader
from regr.sensor.torch.sensor import ReaderSensor, CartesianSensor, CartesianCandidateSensor, SpacyTokenizorSensor, LabelAssociateSensor, LabelRelationAssociateSensor, Key
from regr.sensor.torch.learner import RNNLearner, MLPLearner, LRLearner, NorminalEmbedderLearner
from emr.utils import seed


def ontology_declaration():
    from graph import graph
    return graph


def model_declaration(graph, config):
    graph.detach()

    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']
    pair = graph['linguistic/pair']

    people = graph['application/people']
    organization = graph['application/organization']
    location = graph['application/location']
    other = graph['application/other']
    o = graph['application/O']

    work_for = graph['application/work_for']
    located_in = graph['application/located_in']
    live_in = graph['application/live_in']
    orgbase_on = graph['application/orgbase_on']
    kill = graph['application/kill']

    # feature
    sentence['index'] = ReaderSensor(Key('sentence'))
    word['index'] = SpacyTokenizorSensor(sentence['index'])
    word['emb'] = NorminalEmbedderLearner(word['index'], **config.word.emb)
    word['ctx_emb'] = RNNLearner(word['emb'], **config.word.ctx_emb)
    word['feature'] = MLPLearner(word['ctx_emb'], **config.word.feature)

    pair['index'] = CartesianCandidateSensor(word['index'], word['index'])
    pair['emb'] = CartesianSensor(word['ctx_emb'], word['ctx_emb'])
    pair['feature'] = MLPLearner(pair['emb'], **config.pair.feature)

    # label
    word[people] = LabelAssociateSensor(word['index'], Key('tokens'), Key('label'), 'Peop')
    word[organization] = LabelAssociateSensor(word['index'], Key('tokens'), Key('label'), 'Org')
    word[location] = LabelAssociateSensor(word['index'], Key('tokens'), Key('label'), 'Loc')
    word[other] = LabelAssociateSensor(word['index'], Key('tokens'), Key('label'), 'Other')
    word[o] = LabelAssociateSensor(word['index'], Key('tokens'), Key('label'), 'O')

    word[people] = LRLearner(word['feature'], **config.word.lr)
    word[organization] = LRLearner(word['feature'], **config.word.lr)
    word[location] = LRLearner(word['feature'], **config.word.lr)
    word[other] = LRLearner(word['feature'], **config.word.lr)
    word[o] = LRLearner(word['feature'], **config.word.lr)

    # relation
    pair[work_for] = LabelRelationAssociateSensor(pair['index'], Key('tokens'), Key('relation'), 'Work_For')
    pair[live_in] = LabelRelationAssociateSensor(pair['index'], Key('tokens'), Key('relation'), 'Live_In')
    pair[located_in] = LabelRelationAssociateSensor(pair['index'], Key('tokens'), Key('relation'), 'Located_In')
    pair[orgbase_on] = LabelRelationAssociateSensor(pair['index'], Key('tokens'), Key('relation'), 'OrgBased_In')
    pair[kill] = LabelRelationAssociateSensor(pair['index'], Key('tokens'), Key('relation'), 'Kill')

    pair[work_for] = LRLearner(pair['feature'], **config.pair.lr)
    pair[live_in] = LRLearner(pair['feature'], **config.pair.lr)
    pair[located_in] = LRLearner(pair['feature'], **config.pair.lr)
    pair[orgbase_on] = LRLearner(pair['feature'], **config.pair.lr)
    pair[kill] = LRLearner(pair['feature'], **config.pair.lr)

    # program
    lbp = config.lbp.type(graph, config.lbp.model)
    return lbp


import os
os.environ['REGR_SOLVER'] = 'mini_prob_debug'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import logging
logging.basicConfig(level=logging.INFO)


def main():
    from config2 import CONFIG

    if CONFIG.seed is not None:
        seed(CONFIG.seed)

    graph = ontology_declaration()

    training_set = NaiveDataLoader(CONFIG.Data.train_path,
                                   batch_size=CONFIG.Data.batch_size,
                                   shuffle=True)

    valid_set = NaiveDataLoader(CONFIG.Data.valid_path,
                                batch_size=CONFIG.Data.batch_size,
                                shuffle=False)

    lbp = model_declaration(graph, CONFIG.Model)
    lbp.update_nominals(training_set)
    lbp.train(training_set, valid_set, **CONFIG.Train)


if __name__ == '__main__':
    main()
