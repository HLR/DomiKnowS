'''
# Example: Entity-Mention-Relation (EMR)
## Pipeline
This example follows the pipeline we discussed in our preliminary paper.
1. Ontology Declaration
2. Model Declaration
3. Explicit inference
'''

#### With `regr`, we assign sensors to properties of concept.
#### There are two types of sensor: `Sensor`s and `Learner`s.
#### `Sensor` is the more general term, while a `Learner` is a `Sensor` with learnable parameters.

#### `AllenNlpGraph` is a special subclass of `Graph` that wraps a `Graph` and adds computational functionalities to it.
from Graphs.Sensors.sentenceSensors import SentenceReaderSensor, SentenceBertEmbedderSensor, SentenceFlairEmbedderSensor, SentenceGloveEmbedderSensor, CallingSensor
from Graphs.Learners.conceptLearners import LSTMLearner


def ontology_declaration():
    from Graphs.graph import graph
    return graph


def reader_declaration(data_path, splitter_path):
    from data.reader import DataLoader
    reader = DataLoader(data_path, splitter_path)
    reader.fire()
    return reader


def model_declaration(graph, reader):
    graph.detach()

    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']
    # phrase = graph['linguistic/phrase']

    FAC = graph['application/FAC']
    GPE = graph['application/GPE']
    PER = graph['application/PER']
    ORG = graph['application/ORG']
    LOC = graph['application/LOC']
    VEH = graph['application/VEH']
    WEA = graph['application/WEA']

    print(" I am here")
    sentence['raw'] = SentenceReaderSensor(reader)
    sentence['bert'] = SentenceBertEmbedderSensor(sentence['raw'])
    sentence['glove'] = SentenceGloveEmbedderSensor(sentence['raw'])
    sentence['flair'] = SentenceFlairEmbedderSensor(sentence['raw'])

    # next(sentence['flair'].find(SentenceFlairEmbedderSensor))[1](context={})

    sentence['output'] = CallingSensor(sentence['bert'], sentence['glove'], sentence['flair'], output=sentence['raw'])
    # next(sentence['output'].find(CallingSensor))[1](context={})
    #
    sentence['encode'] = LSTMLearner(sentence['output'], input_dim=5220, hidden_dim=240, num_layers=1, bidirectional=True)
    #
    next(sentence['encode'].find(CallingSensor))[1](context={})

    # sentence['all'] = ConcatSensor(sentence['flair'], sentence['bert'], sentence['glove'])
    # #### `TokenInSequenceSensor` provides the ability to index tokens in a `TextField`.
    # #### Notice that the Conll04 dataset comes with phrase tokenized sentences.
    # #### Thus this is already a phrase-based sentence.
    # #### `TokenInSequenceSensor` takes the sentence `TextField` here and insert a token field to it.
    # #### Please also refer to AllenNLP `TextField` document for complicated relationship of it and its tokens.
    # word['raw'] = SentenceEmbedderSensor('word', config.pretrained_dims['word'], sentence['raw'], pretrained_file=config.pretrained_files['word'])
    # word['pos'] = SentenceEmbedderLearner('pos_tag', config.embedding_dim, sentence['raw'])
    # word['dep'] = SentenceEmbedderLearner('dep_tag', config.embedding_dim, sentence['raw'])
    # # possible to add more this kind
    # word['all'] = ConcatSensor(word['raw'], word['pos'], word['dep'])
    # #### `RNNLearner` takes a sequence of representations as input, encodes them with recurrent nerual networks (RNN), like LSTM or GRU, and provides the encoded output.
    # #### Here we encode the word2vec output further with an RNN.
    # #### The first argument indicates the dimensions of internal representations, and the second one incidates we will encode the output of `phrase['w2v']`.
    # #### More optional arguments are avaliable, like `bidirectional` defaulted to `True` for context from both sides, and `dropout` defaulted to `0.5` for tackling overfitting.
    # word['ngram'] = NGramSensor(config.ngram, word['all'])
    # word['encode'] = RNNLearner(word['ngram'], layers=config.rnn.layers, bidirectional=config.rnn.bidirectional, dropout=config.dropout)
    # #### `CartesianProductSensor` is a `Sensor` that takes the representation from `phrase['emb']`, makes all possible combination of them, and generates a concatenating result for each combination.
    # #### This process takes no parameters.
    # #### But there is still a PyTorch module associated with it.
    # word['compact'] = MLPLearner(config.compact.layers, word['encode'], activation=config.activation)
    # pair['cat'] = CartesianProductSensor(word['compact'])
    # pair['tkn_dist'] = TokenDistantSensor(config.distance_emb_size * 2, config.max_distance, sentence['raw'])
    # pair['tkn_dep'] = TokenDepSensor(sentence['raw'])
    # pair['tkn_dep_dist'] = TokenDepDistSensor(config.distance_emb_size, config.max_distance, sentence['raw'])
    # pair['onehots'] = ConcatSensor(pair['tkn_dist'], pair['tkn_dep'], pair['tkn_dep_dist'])
    # pair['emb'] = MLPLearner([config.relemb.emb_size,], pair['onehots'], activation=None)
    # pair['tkn_lca'] = TokenLcaSensor(sentence['raw'], word['compact'])
    # pair['all'] = ConcatSensor(pair['cat'], pair['tkn_lca'], pair['emb'])
    # pair['encode'] = ConvLearner(config.relconv.layers, config.relconv.kernel_size, pair['all'], activation=config.activation, dropout=config.dropout)
    #
    # #### Then we connect properties with ground-truth from `reader`.
    # #### `LabelSensor` takes the `reader` as argument to provide the ground-truth data.
    # #### The second argument indicates the key we used for each lable in reader.
    # #### The last keyword argument `output_only` indicates that these sensors are not to be used with forward computation.
    # people['label'] = LabelSensor(reader, 'Peop', output_only=True)
    # organization['label'] = LabelSensor(reader, 'Org', output_only=True)
    # location['label'] = LabelSensor(reader, 'Loc', output_only=True)
    # other['label'] = LabelSensor(reader, 'Other', output_only=True)
    # #o['label'] = LabelSensor(reader, 'O', output_only=True)
    #
    # #### We connect properties with learners that generate predictions.
    # #### Notice that we connect the predicting `Learner`s to the same properties as "ground-truth" `Sensor`s.
    # #### Multiple assignment is a feature in `regr` to allow each property to have multiple sources.
    # #### Value from different sources will be compared, to generate inconsistency error.
    # #### The training of this model is then based on this inconsistency error.
    # #### In this example, "ground-truth" `Sensor`s has no parameters to be trained, while predicting `Learner`s have all sets of paramters to be trained.
    # #### The error also propagate backward through the computational path to all modules as assigned above.
    # #### Here we use `LogisticRegressionLearner`s, which is binary classifiers.
    # #### Notice the first argument, the "input dimention", takes a `* 2` because the output from `phrase['emb']` is bidirectional, having two times dimentions.
    # #### The second argument is base on what the prediction will be made.
    # #### The constructors make individule modules for them with seperated parameters, though they take same arguments.
    # people['label'] = LogisticRegressionLearner(word['encode'])
    # organization['label'] = LogisticRegressionLearner(word['encode'])
    # location['label'] = LogisticRegressionLearner(word['encode'])
    # other['label'] = LogisticRegressionLearner(word['encode'])
    # #o['label'] = LogisticRegressionLearner(config.embedding_dim * 8, word['emb'])
    #
    # #### We repeat these on composed-concepts.
    # #### There is nothing different in usage thought they are higher ordered concepts.
    # work_for['label'] = LabelSensor(reader, 'Work_For', output_only=True)
    # live_in['label'] = LabelSensor(reader, 'Live_In', output_only=True)
    # located_in['label'] = LabelSensor(reader, 'Located_In', output_only=True)
    # orgbase_on['label'] = LabelSensor(reader, 'OrgBased_In', output_only=True)
    # kill['label'] = LabelSensor(reader, 'Kill', output_only=True)
    #
    # #### We also connect the predictors for composed-concepts.
    # #### Notice the first argument, the "input dimention", takes a `* 4` because `pair['emb']` from `CartesianProductSensor` has double dimention again over `phrase['emb']`.
    # work_for['label'] = LogisticRegressionLearner(pair['encode'])
    # live_in['label'] = LogisticRegressionLearner(pair['encode'])
    # located_in['label'] = LogisticRegressionLearner(pair['encode'])
    # orgbase_on['label'] = LogisticRegressionLearner(pair['encode'])
    # kill['label'] = LogisticRegressionLearner(pair['encode'])
    #
    # #### Lastly, we wrap these graph with `AllenNlpGraph` functionalities to get the full learning based program.
    # lbp = AllenNlpGraph(graph, **config.graph)
    # return lbp


#### The main entrance of the program.
def main():
    print("salam")
    graph = ontology_declaration()

    data_path = ["/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/bc/fp1/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/bc/fp2/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/bn/fp1/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/bn/fp2/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/cts/fp1/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/cts/fp2/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/nw/fp1/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/nw/fp2/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/un/fp1/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/un/fp2/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/wi/fp1/",
                 "/home/hfaghihi/LDC2006T06/ace_2005_td_v7/data/English/wi/fp2/", ]

    hint_path = "/home/hfaghihi/LDC2006T06/split"

    # reader = reader_declaration(data_path=data_path, splitter_path=hint_path)
    reader = None
    lbp = model_declaration(graph, reader)


    #### 3. Train and save the model
    #### "Explicit inference" is done automatically in every call to the model.
    #### To have better reproducibility, we initial the random seeds of all subsystems.
    # seed()
    # #### Train the model with inference functionality inside.
    # lbp.train(Config.Data, Config.Train)
    # #### Save the model, including vocabulary use to index the tokens.
    # save_to = Config.Train.trainer.serialization_dir or '/tmp/emr'
    # lbp.save(save_to, config=save_config)

####
"""
This example show a full pipeline how to work with `regr`.
"""
print("hello")
main()