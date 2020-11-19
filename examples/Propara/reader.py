import os
from regr.data.reader import RegrReader


class ProparaReader(RegrReader):
    def getprocedureIDval(self, item):
        return item['id']
    def getentitiesval(self, item):
        return item['entities']
    def getstepsval(self, item):
        num_steps = len(item['steps']) + 1
        rel = torch.ones(num_steps,1)
        return rel, ["step 0 information"].extend(item['steps'])
    def getentity_stepval(self, item):
        steps = len(item['steps']) + 1
        entities = len(item['entities'])
#         entity_step = []
        arg1s = []
        arg2s = []
        for entity,eraw in enumerate(item['entities']):
            arg1 = torch.zeros(entities)
            arg1[entity] = 1
            for step in range(steps):
                arg2 = torch.zeros(steps)
                arg2[step] = 1
                arg1s.append(arg1)
                arg2s.append(arg2)
#                 entity_step.append((eraw, sraw))
        return torch.stack(arg1s), torch.stack(arg2s)
    
    def getnon_existenceval(self, item):
        values = []
        for entity, eraw in enumerate(item['entities']):
            for step in range(len(item['steps']) + 1):
                values.append([1 - item['entity_step'][entity][step][2], item['entity_step'][entity][step][2]])
        return torch.tensor(values)
            
    def getknownval(self, item):
        values = []
        for entity, eraw in enumerate(item['entities']):
            for step in range(len(item['steps']) + 1):
                values.append([1 - item['entity_step'][entity][step][0], item['entity_step'][entity][step][0]])
        return torch.tensor(values)
    
    def getunkownval(self, item):
        values = []
        for entity, eraw in enumerate(item['entities']):
            for step in range(len(item['steps']) + 1):
                values.append([1 - item['entity_step'][entity][step][1], item['entity_step'][entity][step][1]])
        return torch.tensor(values)
    
    def getactionval(self, item):
        action1s = torch.diag(torch.ones( len(item['entities']) * len(item['steps']) ) )[:-1]
        action2s = torch.diag(torch.ones( len(item['entities']) * len(item['steps']) ) )[1:]
        return action1s, action2s
    
    def getcreateval(self, item):
        actions = []
        for eid, entity in enumerate(item['entities']):
            for sid, step in enumerate(item['steps']):
                o = 0
                c = 0
                d = 0
                if sid == 0:
                    prev_state = item['entity_step'][eid][sid]
                    continue
                else:
                    o += (prev_state[0] * item['entity_step'][eid][sid][0])
                    o += (prev_state[0] * item['entity_step'][eid][sid][1])
                    o += (prev_state[1] * item['entity_step'][eid][sid][0])
                    o += (prev_state[1] * item['entity_step'][eid][sid][1])
                    o += (prev_state[2] * item['entity_step'][eid][sid][2])
                    d += (prev_state[0] * item['entity_step'][eid][sid][2])
                    d += (prev_state[1] * item['entity_step'][eid][sid][2])
                    c += (prev_state[2] * item['entity_step'][eid][sid][1])
                    c += (prev_state[2] * item['entity_step'][eid][sid][0])
                    actions.append([1-c, c])
                    prev_state = item['entity_step'][eid][sid]
        return actions
                    
    def getdestroyval(self, item):
        actions = []
        for eid, entity in enumerate(item['entities']):
            for sid, step in enumerate(item['steps']):
                o = 0
                c = 0
                d = 0
                if sid == 0:
                    prev_state = item['entity_step'][eid][sid]
                    continue
                else:
                    o += (prev_state[0] * item['entity_step'][eid][sid][0])
                    o += (prev_state[0] * item['entity_step'][eid][sid][1])
                    o += (prev_state[1] * item['entity_step'][eid][sid][0])
                    o += (prev_state[1] * item['entity_step'][eid][sid][1])
                    o += (prev_state[2] * item['entity_step'][eid][sid][2])
                    d += (prev_state[0] * item['entity_step'][eid][sid][2])
                    d += (prev_state[1] * item['entity_step'][eid][sid][2])
                    c += (prev_state[2] * item['entity_step'][eid][sid][1])
                    c += (prev_state[2] * item['entity_step'][eid][sid][0])
                    actions.append([1-d, d])
                    prev_state = item['entity_step'][eid][sid]
        return actions
    
    def getotherval(self, item):
        actions = []
        for eid, entity in enumerate(item['entities']):
            for sid, step in enumerate(item['steps']):
                o = 0
                c = 0
                d = 0
                if sid == 0:
                    prev_state = item['entity_step'][eid][sid]
                    continue
                else:
                    o += (prev_state[0] * item['entity_step'][eid][sid][0])
                    o += (prev_state[0] * item['entity_step'][eid][sid][1])
                    o += (prev_state[1] * item['entity_step'][eid][sid][0])
                    o += (prev_state[1] * item['entity_step'][eid][sid][1])
                    o += (prev_state[2] * item['entity_step'][eid][sid][2])
                    actions.append([1-o, o])
                    prev_state = item['entity_step'][eid][sid]
        return actions
    
    def getbeforeval(self, item):
        b1s = []
        b2s = []
        for step in range(len(item['steps'])):
            b1 = torch.zeros(len(item['steps']))
            b1[step] = 1
            for step1 in range(step+1, len(item['steps'])):
                b2 = torch.zeros(len(item['steps']))
                b2[step1] = 1
                b1s.append(b1)
                b2s.append(b2)
        return torch.stack(b1s), torch.stack(b2s)
    