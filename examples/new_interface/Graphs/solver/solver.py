from regr.solver.ilpOntSolver import ilpOntSolver
import abc
import torch
import numpy as np


class ACELogicalSolver(ilpOntSolver):
    __metaclass__ = abc.ABCMeta

    def inferILPConstrains(self, context):
        global_key = "global/linguistic/"
        root = "sentence"
        root_features = ["raw", ]
        predictions_on = "word"
        prediction_features = ["raw_ready"]
        predicates = ["<FAC>", "<VEH>", "<PER>", "<ORG>", "<GPE>", "<LOC>", "<WEA>"]
        pairs = ["<ART>", "<GEN-AFF>", "<ORG-AFF>", "<PER-SOC>", "<METONYMY>", "<PART-WHOLE>", "<PHYS>"]
        pairs_on = "pair"
        phrase_order = ["FAC", "GPE", "PER", "ORG", "LOC", "VEH", "WEA"]
        sentence = {}
        with torch.no_grad():
            for item in predicates:
                sentence[item.replace("<", "").replace(">", "")] = [np.log(_it.cpu().numpy()) for _it in
                                                                    context[global_key + predictions_on + "/" + item]]
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
                    _list.append(_value)
                sentence['phrase']['entity'][item.replace("<", "").replace(">", "")] = [_val for _val in _list]
            sentence['phrase']['relations'] = {}
            for item in pairs:
                _list = [np.log(_it.cpu().numpy()) for _it in context[global_key + pairs_on + "/" + item]]
                _result = np.zeros((len(sentence['phrase']['raw']), len(sentence['phrase']['raw']), 2))
                for _range in range(len(sentence['phrase']['pair_index'])):
                    indexes = sentence['phrase']['pair_index'][_range]
                    values = _list[_range]
                    _result[indexes[0]][indexes[1]][0] = values[0]
                    _result[indexes[1]][indexes[0]][0] = values[0]
                    _result[indexes[0]][indexes[1]][1] = values[1]
                    _result[indexes[1]][indexes[0]][1] = values[1]
                sentence['phrase']['relations'][item.replace("<", "").replace(">", "")] = _result

        results = self.calculateILPSelection(sentence['phrase']['raw'],
                                   sentence['phrase']['entity'],
                                   sentence['phrase']['relations'])

        return self.transform_back(result=results, context=context, helper=sentence)


    def transform_back(self, result, context, helper):
        return result



