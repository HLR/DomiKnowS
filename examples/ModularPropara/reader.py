from regr.data.reader import RegrReader
import torch


class ProparaReader(RegrReader):
    def getProcedureIDval(self, item):
        return [item["para_id"]]

    def getEntitiesval(self, item):
        return [' '.join(item['participants'])]
        
    def getEntityval(self, item):
        return item["participants"]
    
    def getContextval(self, item):
        return [item['sentence_paragraph']]

    def getStepval(self, item):
        sentences = item["sentence_texts"]
        return  sentences
    
    def getActionval(self, item):
        return torch.tensor(item['action_probs'])
    
    def getbeforeval(self, item):
        b1s = []
        b2s = []
        for step in range(len(item["sentence_texts"])):
            b1 = torch.zeros(len(item["sentence_texts"]))
            b1[step] = 1
            for step1 in range(len(item["sentence_texts"])):
                b2 = torch.zeros(len(item["sentence_texts"]))
                b2[step1] = 1
                b1s.append(b1)
                b2s.append(b2)
        return torch.stack(b1s), torch.stack(b2s)

    def getbefore_trueval(self, item):
        num_steps = len(item["sentence_texts"])
        values = torch.zeros(num_steps * num_steps)
        for step in range(len(item["sentence_texts"])):
            for step1 in range(step + 1, len(item["sentence_texts"])):
                values[(step * num_steps) + step1] = 1
        return values
