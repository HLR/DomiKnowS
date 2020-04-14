from regr.solver.ilpOntSolver import ilpOntSolver
import abc
import torch
import numpy as np
from Graphs.graph import app_graph
from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
from regr.graph import Graph



class ACELogicalSolver(ilpOntSolver):
    __metaclass__ = abc.ABCMeta

    def inferILPConstrains(self, context, info):
        global_key = "global/linguistic/"
        root = "sentence"
        root_features = ["raw", ]
        predictions_on = "word"
        prediction_features = ["raw_ready"]
        predicates1 = ["FAC", "VEH", "PER", "ORG", "GPE", "LOC", "WEA"]
        predicates = []
        for item in info:
            if item in predicates1:
                predicates.append("<"+str(item)+">")

        pairs1 = ["ART", "GEN-AFF", "ORG-AFF", "PER-SOC", "METONYMY", "PART-WHOLE", "PHYS"]
        pairs = []
        for item in info:
            if item in pairs1:
                pairs.append("<"+str(item)+">")
        pairs_on = "pair"
        phrase_order1 = ["FAC", "GPE", "PER", "ORG", "LOC", "VEH", "WEA"]
        phrase_order = []
        for item in info:
            if item in phrase_order1:
                phrase_order.append(str(item))
        sentence = {"words": {}}
        with torch.no_grad():
            epsilon = 0.00001
            for item in predicates:
                _list = [_it.cpu().detach().numpy() for _it in context[global_key + predictions_on + "/" + item]]
                for _it in range(len(_list)):
                    if _list[_it][0] > 1-epsilon:
                        _list[_it][0] = 1-epsilon
                    elif _list[_it][1] > 1-epsilon:
                        _list[_it][1] = 1-epsilon
                    if _list[_it][0] < epsilon:
                        _list[_it][0] = epsilon
                    elif _list[_it][1] < epsilon:
                        _list[_it][1] = epsilon
                sentence[item.replace("<", "").replace(">", "")] = _list
                sentence['words'][item.replace("<", "").replace(">", "")] = np.log(np.array(_list))

            if len(pairs):
                sentence['phrase'] = {}
                sentence['phrase']['entity'] = {}
                sentence['phrase']['raw'] = context[global_key + "phrase/raw"]
                sentence['phrase']['tag'] = [_it.item() for _it in context[global_key + "phrase/tag"]]
                sentence['phrase']['tag_name'] = [phrase_order[t] for t in sentence['phrase']['tag']]
                sentence['phrase']['pair_index'] = context[global_key + "pair/index"]
                for item in predicates:
                    _list = []
                    for ph in sentence['phrase']['raw']:
                        _value = [1, 1]
                        for _range in range(ph[0], ph[1] + 1):
                            _value[0] = _value[0] * sentence[item.replace("<", "").replace(">", "")][_range][0]
                            _value[1] = _value[1] * sentence[item.replace("<", "").replace(">", "")][_range][1]
                        _list.append(np.log(np.array(_value)))
                    sentence['phrase']['entity'][item.replace("<", "").replace(">", "")] = np.array(_list)
                sentence['phrase']['relations'] = {}
                for item in pairs:
                    _list = [np.log(_it.cpu().detach().numpy()) for _it in context[global_key + pairs_on + "/" + item]]
                    _result = np.zeros((len(sentence['phrase']['raw']), len(sentence['phrase']['raw']), 2))
                    for _range in range(len(sentence['phrase']['pair_index'])):
                        indexes = sentence['phrase']['pair_index'][_range]
                        values = _list[_range]
                        _result[indexes[0]][indexes[1]][0] = values[0]
                        _result[indexes[1]][indexes[0]][0] = values[0]
                        _result[indexes[0]][indexes[1]][1] = values[1]
                        _result[indexes[1]][indexes[0]][1] = values[1]
                    for _ii in range(len(sentence['phrase']['raw'])):
                        _result[_ii][_ii] = np.log(np.array([0.999, 0.001]))
                    sentence['phrase']['relations'][item.replace("<", "").replace(">", "")] = np.array(_result)
        # import pickle
        # file = open('data.pkl', 'wb')
        #
        # # dump information to that file
        # pickle.dump(sentence, file)
        #
        # # close the file
        # file.close()
        # graph1 = Graph(iri='http://ontology.ihmc.us/ML/ACE.owl', local='./')
        # myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(graph1)

        #Using the lConstraints
        myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(app_graph)
        if len(pairs):
            phrases = [str(_it) for _it in range(len(sentence['phrase']['raw']))]
            results = myilpOntSolver.calculateILPSelection(phrases,
                                   sentence['phrase']['entity'],sentence['phrase']['relations'])
        else:
            tokens = [str(_it) for _it in range(len(sentence['FAC']))]

            results = myilpOntSolver.calculateILPSelection(tokens,
                                       sentence['words'])

        return self.transform_back(result=results, context=context, helper=sentence)

    def transform_back(self, result, context, helper):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if "phrase" in helper:
            pairs = ["ART", "GEN-AFF", "ORG-AFF", "PER-SOC", "METONYMY", "PART-WHOLE", "PHYS"]
            for val in pairs:
                _list = []
                for item in helper['phrase']['pair_index']:
                    if result[1][val][item[0]][item[1]] == 1 or result[1][val][item[1]][item[0]] == 1:
                        _list.append(1)
                    else:
                        _list.append(0)
                result[1][val] = torch.tensor(np.array(_list), device=self.device)

            for item in result[0]:
                count = 0
                _list = []
                last = -1
                for ph in helper['phrase']['raw']:

                    if ph[0] > last:
                        for i in range(last+1, ph[0]):
                            _list.append(0)
                    last = ph[1]
                    for i in range(ph[0], ph[1]+1):
                        _list.append(result[0][item][count])
                    count += 1
                if last != len(helper['words']['FAC']):
                    for i in range(last+1, len(helper['words']['FAC'])):
                            _list.append(0)
                result[0][item] = torch.tensor(_list, device = self.device)
        else:
            for item in result[0]:
                result[0][item] = torch.from_numpy(result[0][item]).float().to(self.device)
        return result




