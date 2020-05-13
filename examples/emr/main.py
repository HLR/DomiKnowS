from emr.data import ConllDataLoader
from emr.sensor.sensor import DataSensor, LabelSensor, CartesianSensor
from emr.sensor.learner import EmbedderLearner, RNNLearner, MLPLearner, LRLearner
from emr.utils import seed


def ontology_declaration():
    from graph import graph
    return graph


def model_declaration(graph, vocab, config):
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
    sentence['raw'] = DataSensor('token_raw')
    sentence['tensor'] = DataSensor('token')
    word['emb'] = EmbedderLearner(sentence['tensor'], padding_idx=vocab['token']['_PAD_'], num_embeddings=len(vocab['token']), **config.word.emb)
    word['ctx_emb'] = RNNLearner(word['emb'], **config.word.ctx_emb)
    word['feature'] = MLPLearner(word['ctx_emb'], **config.word.feature)

    pair['emb'] = CartesianSensor(word['ctx_emb'], word['ctx_emb'])
    pair['feature'] = MLPLearner(pair['emb'], **config.pair.feature)

    # label
    word[people] = LabelSensor('Peop')
    word[organization] = LabelSensor('Org')
    word[location] = LabelSensor('Loc')
    word[other] = LabelSensor('Other')
    word[o] = LabelSensor('O')

    word[people] = LRLearner(word['feature'], **config.word.lr)
    word[organization] = LRLearner(word['feature'], **config.word.lr)
    word[location] = LRLearner(word['feature'], **config.word.lr)
    word[other] = LRLearner(word['feature'], **config.word.lr)
    word[o] = LRLearner(word['feature'], **config.word.lr)

    # relation
    pair[work_for] = LabelSensor('Work_For')
    pair[live_in] = LabelSensor('Live_In')
    pair[located_in] = LabelSensor('Located_In')
    pair[orgbase_on] = LabelSensor('OrgBased_In')
    pair[kill] = LabelSensor('Kill')

    pair[work_for] = LRLearner(pair['feature'], **config.pair.lr)
    pair[live_in] = LRLearner(pair['feature'], **config.pair.lr)
    pair[located_in] = LRLearner(pair['feature'], **config.pair.lr)
    pair[orgbase_on] = LRLearner(pair['feature'], **config.pair.lr)
    pair[kill] = LRLearner(pair['feature'], **config.pair.lr)

    # program
    lbp = config.lbp.type(graph, config.lbp.model)
    return lbp


def main():
    from config import CONFIG

    if CONFIG.Train.seed is not None:
        seed(CONFIG.Train.seed)

    graph = ontology_declaration()

    training_set = ConllDataLoader(CONFIG.Data.train_path,
                                   batch_size=CONFIG.Train.batch_size,
                                   skip_none=CONFIG.Data.skip_none,
                                   shuffle=True)
    valid_set = ConllDataLoader(CONFIG.Data.valid_path,
                                batch_size=CONFIG.Train.batch_size,
                                skip_none=CONFIG.Data.skip_none,
                                vocab=training_set.vocab,
                                shuffle=False)

    lbp = model_declaration(graph, training_set.vocab, CONFIG.Model)
    lbp.train(training_set, valid_set, config=CONFIG.Train)

if __name__ == '__main__':
    main()
