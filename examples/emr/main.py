from itertools import chain

from emr.utils import seed
from emb.graph.torch import TorchModel, train, test

from config import Config


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
    sentence['raw'] = DataSensor('token')
    word['emb'] = EmbedderLearner(sentence['raw'])
    word['ctx_emb'] = RNNLearner(word['emb'])
    word['feature'] = MLPLearner(word['ctx_emb'])

    pair['emb'] = CartesianSensor(word['ctx_emb'], word['ctx_emb'])
    pair['feature'] = MLPLearner(word['emb'])

    # label
    word[people] = LabelSensor('Peop')
    word[organization] = LabelSensor('Org')
    word[location] = LabelSensor('Loc')
    word[other] = LabelSensor('Other')
    word[o] = LabelSensor('O')

    word[people] = LRLearner(word['feature'])
    word[organization] = LRLearner(word['feature'])
    word[location] = LRLearner(word['feature'])
    word[other] = LRLearner(word['feature'])
    word[o] = LRLearner(word['feature'])

    # relation
    pair[work_for] = LabelSensor('Work_For')
    pair[live_in] = LabelSensor('Live_In')
    pair[located_in] = LabelSensor('Located_In')
    pair[orgbase_on] = LabelSensor('OrgBased_In')
    pair[kill] = LabelSensor('Kill')

    pair[work_for] = LRLearner(pair['feature'])
    pair[live_in] = LRLearner(pair['feature'])
    pair[located_in] = LRLearner(pair['feature'])
    pair[orgbase_on] = LRLearner(pair['feature'])
    pair[kill] = LRLearner(pair['feature'])

    # program
    lbp = TorchModel(graph)
    return lbp


def main():
    graph = ontology_declaration()

    lbp = model_declaration(graph, Config.Model)

    seed()
    training_set = ConllDataLoader(config.Data.train_path,
                                   batch_size=config.Train.batch_size,
                                   skip_none=config.Data.skip_none)
    valid_set = ConllDataLoader(config.Data.valid_path,
                                  batch_size=config.Train.batch_size,
                                  skip_none=config.Data.skip_none)
    opt = torch.optim.Adam(lbp.parameters())
    for epoch in range(10):
        loss, metric, output = map(lambda x: chain(*x), zip(*train(lbp, training_set, opt)))
        valid_loss, valid_metric, valid_output = map(lambda x: chain(*x), zip(*test(lbp, valid_set, opt)))
        print(epoch, loss, valid_loss)
