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
        predicates1 = ["ORG", "FAC", "PER", "VEH", "LOC", "WEA", "GPE", "Airport", "Building-Grounds", "Path", "Plant",
                 "Subarea-Facility", "Continent", "County-or-District", "GPE-Cluster", "Nation", "Population-Center",
                 "Special", "State-or-Province", "Address", "Boundary", "Celestial", "Land-Region-Natural",
                 "Region-General", "Region-International", "Water-Body", "Commercial", "Educational", "Entertainment",
                 "Government", "Media", "Medical-Science", "Non-Governmental", "Religious", "Sports", "Group",
                 "Indeterminate", "Individual", "Land", "Subarea-Vehicle", "Underspecified", "Water", "Biological",
                 "Blunt", "Chemical", "Exploding", "Nuclear", "Projectile", "Sharp", "Shooting", "WEA-Underspecified"]
        predicates = []
        for item in info:
            if item in predicates1:
                predicates.append("<"+str(item)+">")

#         pairs1 = ["ART", "GEN-AFF", "ORG-AFF", "PER-SOC", "METONYMY", "PART-WHOLE", "PHYS"]
#         pairs = []
#         for item in info:
#             if item in pairs1:
#                 pairs.append("<"+str(item)+">")
#         pairs_on = "pair"
#         phrase_order1 = ["ORG", "FAC", "PER", "VEH", "LOC", "WEA", "GPE", "Airport", "Building-Grounds", "Path", "Plant",
#                  "Subarea-Facility", "Continent", "County-or-District", "GPE-Cluster", "Nation", "Population-Center",
#                  "Special", "State-or-Province", "Address", "Boundary", "Celestial", "Land-Region-Natural",
#                  "Region-General", "Region-International", "Water-Body", "Commercial", "Educational", "Entertainment",
#                  "Government", "Media", "Medical-Science", "Non-Governmental", "Religious", "Sports", "Group",
#                  "Indeterminate", "Individual", "Land", "Subarea-Vehicle", "Underspecified", "Water", "Biological",
#                  "Blunt", "Chemical", "Exploding", "Nuclear", "Projectile", "Sharp", "Shooting", "WEA-Underspecified"]
#         phrase_order = []
#         for item in info:
#             if item in phrase_order1:
#                 phrase_order.append(str(item))
        sentence = {"words": {}}
        with torch.no_grad():
            epsilon = 0.00001
            for item in predicates:
                _list = [_it.cpu().numpy() for _it in context[global_key + predictions_on + "/" + item]]
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
        for item in result[0]:
            result[0][item] = torch.from_numpy(result[0][item]).float().to(self.device)
        return result




