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
from regr.sensor.allennlp.sensor import SentenceSensor, SentenceBertEmbedderSensor, LabelSensor, CartesianProductSensor, ConcatSensor, NGramSensor, TokenDistantSensor, TokenDepSensor, TokenLcaSensor, TokenDepDistSensor, CandidateReaderSensor
from regr.sensor.allennlp.learner import MLPLearner, ConvLearner, LogisticRegressionLearner

#### `AllenNlpGraph` is a special subclass of `Graph` that wraps a `Graph` and adds computational functionalities to it.
from regr.graph.allennlp import AllenNlpGraph

#### There are a few other components that are needed in common machine learning models.
#### * `Conll04SensorReader` is a AllenNLP `DatasetReader`.
#### We added a bit of magic to make it more compatible with `regr`.
#### See `data.py` for details.
#### * `Config` contains configurations for model, data, and training.
#### * `seed` is a useful function that resets random seed of all involving sub-systems: Python, numpy, and PyTorch, to make the performance of training consistent, as a demo.
#from .data import Conll04SensorReader as Reader
if __package__ is None or __package__ == '':
    from data_bert import Conll04BERTBinaryReader as Reader
    from config import Config
    from utils import seed
else:
    from .data_bert import Conll04BERTBinaryReader as Reader
    from .config import Config
    from .utils import seed


#### "*Ontology Declaration*" is the first step in our pipeline.
#### A graph of the concepts, representing the ontology in this application, is declared.
#### It can be compile from standard ontology formats like `OWL`, writen with python grammar directly, or combine both way.
#### Here we just import the graph from `graph.py`.
#### Please also refer to `graph.py` for details.
def ontology_declaration():
    if __package__ is None or __package__ == '':
        from graph import graph
    else:
        from .graph import graph
    return graph


#### "*Model Declaration*" comes after "Ontology Declaration" in our pipeline.
#### Sensors and learners are connected to the graph, what wraps the graph with functionalities to retieve data, forward computing, learning from error, and inference during all those processes.
#### `graph` is a `Graph` object retrieved from the "Ontology Declaration".
#### `config` is configurations relatred to the model.
def model_declaration(graph, config):
    #### `Graph` objects has some kind of global variables.
    #### Use `.detach()` to reset them to avoid baffling error.
    graph.detach()

    #### Retrieve concepts that are needed in this model.
    #### Notice that these concepts are already well defined in `graph.py`.
    #### Here we just retrieve them to use them as python variables.
    #### `sentence`, `phrase`, and `pair` are basic linguistic concepts in this demo.
    sentence = graph['linguistic/sentence']
    word = graph['linguistic/word']
    pair = graph['linguistic/pair']

    #### `people`, `organization`, `location`, `other`, and `O` are entities we want to extract in this demo.
    people = graph['application/people']
    organization = graph['application/organization']
    location = graph['application/location']
    other = graph['application/other']
    o = graph['application/O']

    #### `people`, `organization`, `location`, `other`, and `O` are entities we want to extract in this demo.
    work_for = graph['application/work_for']
    located_in = graph['application/located_in']
    live_in = graph['application/live_in']
    orgbase_on = graph['application/orgbase_on']
    kill = graph['application/kill']

    #### Create a `Conll04SensorReader` instance, to be assigned with properties, and allow the model to get corresponding data from it.
    reader = Reader()

    #### The most important part in "Model Declaration" is to connect sensors (and learners) to the graph.
    #### We start with linguistic concepts.
    #### `SequenceSensor` provides the ability to read from a `TextField` in AllenNLP.
    #### It takes two arguments, firstly the reader to read with, and secondly a `key` for the reader to refer to correct `TextField`.
    sentence['raw'] = SentenceSensor(reader, 'sentence')
    word['emb'] = SentenceBertEmbedderSensor('word', 'bert-base-uncased', sentence['raw'])
    word['encode'] = MLPLearner(config.bert_mlp, word['emb'])
    pair['cat'] = CartesianProductSensor(word['encode'])
    pair['encode'] = MLPLearner(config.bert_mlp, pair['cat'])

    #### Then we connect properties with ground-truth from `reader`.
    #### `LabelSensor` takes the `reader` as argument to provide the ground-truth data.
    #### The second argument indicates the key we used for each lable in reader.
    #### The last keyword argument `output_only` indicates that these sensors are not to be used with forward computation.
    people['label'] = LabelSensor(reader, 'Peop', output_only=True)
    organization['label'] = LabelSensor(reader, 'Org', output_only=True)
    location['label'] = LabelSensor(reader, 'Loc', output_only=True)
    other['label'] = LabelSensor(reader, 'Other', output_only=True)
    #o['label'] = LabelSensor(reader, 'O', output_only=True)

    #### We connect properties with learners that generate predictions.
    #### Notice that we connect the predicting `Learner`s to the same properties as "ground-truth" `Sensor`s.
    #### Multiple assignment is a feature in `regr` to allow each property to have multiple sources.
    #### Value from different sources will be compared, to generate inconsistency error.
    #### The training of this model is then based on this inconsistency error.
    #### In this example, "ground-truth" `Sensor`s has no parameters to be trained, while predicting `Learner`s have all sets of paramters to be trained.
    #### The error also propagate backward through the computational path to all modules as assigned above.
    #### Here we use `LogisticRegressionLearner`s, which is binary classifiers.
    #### Notice the first argument, the "input dimention", takes a `* 2` because the output from `phrase['emb']` is bidirectional, having two times dimentions.
    #### The second argument is base on what the prediction will be made.
    #### The constructors make individule modules for them with seperated parameters, though they take same arguments.
    people['label'] = LogisticRegressionLearner(word['encode'])
    organization['label'] = LogisticRegressionLearner(word['encode'])
    location['label'] = LogisticRegressionLearner(word['encode'])
    other['label'] = LogisticRegressionLearner(word['encode'])
    #o['label'] = LogisticRegressionLearner(word['encode'])

    #### We repeat these on composed-concepts.
    #### There is nothing different in usage thought they are higher ordered concepts.
    work_for['label'] = LabelSensor(reader, 'Work_For', output_only=True)
    live_in['label'] = LabelSensor(reader, 'Live_In', output_only=True)
    located_in['label'] = LabelSensor(reader, 'Located_In', output_only=True)
    orgbase_on['label'] = LabelSensor(reader, 'OrgBased_In', output_only=True)
    kill['label'] = LabelSensor(reader, 'Kill', output_only=True)

    #### We also connect the predictors for composed-concepts.
    #### Notice the first argument, the "input dimention", takes a `* 4` because `pair['emb']` from `CartesianProductSensor` has double dimention again over `phrase['emb']`.
    work_for['label'] = LogisticRegressionLearner(pair['encode'])
    live_in['label'] = LogisticRegressionLearner(pair['encode'])
    located_in['label'] = LogisticRegressionLearner(pair['encode'])
    orgbase_on['label'] = LogisticRegressionLearner(pair['encode'])
    kill['label'] = LogisticRegressionLearner(pair['encode'])

    pair['candidate'] = CandidateReaderSensor(reader, 'cancidate')

    #### Lastly, we wrap these graph with `AllenNlpGraph` functionalities to get the full learning based program.
    lbp = AllenNlpGraph(graph, **config.graph)
    return lbp


#### The main entrance of the program.
def main():
    save_config = Config.deepclone()
    #### 1. "Ontology Declaration" to get a graph, as a partial program.
    graph = ontology_declaration()

    #### 2. "Model Declaration" to connect sensors and learners and get the full program.
    lbp = model_declaration(graph, Config.Model)

    #### 3. Train and save the model
    #### "Explicit inference" is done automatically in every call to the model.
    #### To have better reproducibility, we initial the random seeds of all subsystems.
    seed()
    #### Train the model with inference functionality inside.
    lbp.train(Config.Data, Config.Train)
    #### Save the model, including vocabulary use to index the tokens.
    save_to = Config.Train.trainer.serialization_dir or '/tmp/emr'
    lbp.save(save_to, config=save_config)

####
"""
This example show a full pipeline how to work with `regr`.
"""
