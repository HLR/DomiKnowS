import abc
from typing import Dict, List
from collections import defaultdict, Counter
import torch
from regr.graph import Graph, Concept, DataNode, Trial
from regr.graph.allennlp import AllenNlpGraph
from regr.graph.allennlp.model import GraphModel
from regr.solver.ilpOntSolver import ilpOntSolver
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
from regr.sensor.allennlp.sensor import SentenceSensor
from regr.sensor import Sensor, Learner



DataInstance = Dict[str, torch.Tensor]


class AllennlpDataNodeSolver(ilpOntSolver):
    __metaclass__ = abc.ABCMeta

    def inferSelection(self, graph: Graph, context: DataInstance, vocab=None) -> DataInstance:
        for nodes, model_trial in self.populate(graph, context):
            import pdb; pdb.set_trace()

    def populate(self, graph, context):
        sentence = graph['linguistic/sentence']
        word = graph['linguistic/word']

        name, sentence_sensor = graph.get_sensors(SentenceSensor)[0]
        sentence_data = context[sentence_sensor.fullname]
        batch_size = len(sentence)

        counter = Counter()
        def next_id(concept):
            counter[concept] += 1
            return '{}-{}'.format(concept.name, counter[concept])

        for batch_index in range(batch_size):
            nodes = defaultdict(list)

            word_node_list = []
            for word_data in sentence_data:
                word_node  = DataNode(instanceID=next_id(word),
                                      instanceValue=word_data,
                                      ontologyNode=word,
                                      childInstanceNodes=None)
                nodes[word].append(word_node)
                word_node_list.append(word_node)
            sentence_node = DataNode(instanceID=next_id(sentence),
                                     instanceValue=sentence_data[batch_index],
                                     ontologyNode=sentence,
                                     childInstanceNodes=word_node_list)
            nodes[sentence].append(sentence_node)

            model_trial = Trial()  # model_trail should come from model run

            people = graph['application/people']
            concepts = [people,]
            for concept in concepts:
                label_prop = concept['label']
                #_, sensor = next(label_prop.find(Sensor, lambda x: not isinstance(x, Learner)))
                _, learner = next(label_prop.find(Learner))
                #sensor_data = context[sensor.fullname]
                learner_data = context[learner.fullname][batch_index]
                learner_data = torch.nn.functional.softmax(learner_data, dim=-1).clone().cpu().detach().numpy()
                for i, word_node in enumerate(word_node_list):
                    model_trial[concept, word_node] = learner_data[i, 1]

            work_for = graph['application/work_for']
            concepts = [work_for,]
            for concept in concepts:
                label_prop = concept['label']
                #_, sensor = next(label_prop.find(Sensor, lambda x: not isinstance(x, Learner)))
                _, learner = next(label_prop.find(Learner))
                #sensor_data = context[sensor.fullname]
                learner_data = context[learner.fullname][batch_index]
                learner_data = torch.nn.functional.softmax(learner_data, dim=-1).clone().cpu().detach().numpy()
                for i, word_node_1 in enumerate(word_node_list):
                    for j, word_node_2 in enumerate(word_node_list):
                        model_trial[concept, (word_node_1, word_node_2)] = learner_data[i, j, 1]

            yield nodes, model_trial


class AllenNlpDataNodeGraph(AllenNlpGraph):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        vocab = None # Vocabulary()
        self.model = GraphModel(self, vocab, *args, **kwargs)
        self.solver = ilpOntSolverFactory.getOntSolverInstance(self, AllennlpDataNodeSolver)
        self.solver_log_to(None)
        # do not invoke super().__init__() here
