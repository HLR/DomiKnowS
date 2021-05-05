import os
from regr.data.reader import RegrReader
import torch


class ProparaReader(RegrReader):
    def getProcedureIDval(self, item):
        return [item["para_id"]]

    def getEntitiesval(self, item):
        return ' '.join(item['participants'])
        
    def getEntityval(self, item):
        return item["participants"]
    
    def getContextval(self, item):
        return item['sentence_paragraph']

    def getStepval(self, item):
        num_steps = len(item["sentence_texts"])
        rel = torch.ones(num_steps, 1)
        sentences = item["sentence_texts"]
        return rel, sentences
    
    def getActionval(self, item):
        actions = []
        for step, step_text in enumerate(item['sentence_texts']):
            actions.append([])
            for eid, entity in enumerate(item['participants']):
                actions[-1].append(item['action_probs'])

        return torch.tensor(actions)
