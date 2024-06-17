import os
from domiknows.data.reader import RegrReader
import torch


class ProparaReader(RegrReader):
    def getprocedureIDval(self, item):
        return [item["id"]]

    def getentitiesval(self, item):
        return item["entities"]

    def getstepsval(self, item):
        num_steps = len(item["steps"]) + 1
        rel = torch.ones(num_steps, 1)
        sentences = ["step 0 information"]
        sentences.extend(item["steps"])
        return rel, sentences

    def getnon_existenceval(self, item):
        values = []
        for step in range(len(item["steps"]) + 1):
            values.append(
                [1 - item["entity_step"][step][2], item["entity_step"][step][2]]
            )
        return torch.tensor(values)

    def getknownval(self, item):
        values = []
        for step in range(len(item["steps"]) + 1):
            values.append(
                [1 - item["entity_step"][step][0], item["entity_step"][step][0]]
            )
        return torch.tensor(values)

    def getunknownval(self, item):
        values = []
        for step in range(len(item["steps"]) + 1):
            values.append(
                [1 - item["entity_step"][step][1], item["entity_step"][step][1]]
            )
        return torch.tensor(values)

    def getactionval(self, item):
        action1s = torch.diag(torch.ones(len(item["steps"]) + 1))[:-1]
        action2s = torch.diag(torch.ones(len(item["steps"]) + 1))[1:]
        raw = torch.zeros(len(item["steps"]))
        return action1s, action2s

    def getcreateval(self, item):
        actions = []
        prev_state = item["entity_step"][0]
        for sid, step in enumerate(item["steps"]):
            o = 0
            c = 0
            d = 0
            o += prev_state[0] * item["entity_step"][sid + 1][0]
            o += prev_state[0] * item["entity_step"][sid + 1][1]
            o += prev_state[1] * item["entity_step"][sid + 1][0]
            o += prev_state[1] * item["entity_step"][sid + 1][1]
            o += prev_state[2] * item["entity_step"][sid + 1][2]
            d += prev_state[0] * item["entity_step"][sid + 1][2]
            d += prev_state[1] * item["entity_step"][sid + 1][2]
            c += prev_state[2] * item["entity_step"][sid + 1][1]
            c += prev_state[2] * item["entity_step"][sid + 1][0]
            actions.append([1 - c, c])
            prev_state = item["entity_step"][sid + 1]

        return torch.tensor(actions)

    def getdestroyval(self, item):
        actions = []
        prev_state = item["entity_step"][0]
        for sid, step in enumerate(item["steps"]):
            o = 0
            c = 0
            d = 0
            o += prev_state[0] * item["entity_step"][sid + 1][0]
            o += prev_state[0] * item["entity_step"][sid + 1][1]
            o += prev_state[1] * item["entity_step"][sid + 1][0]
            o += prev_state[1] * item["entity_step"][sid + 1][1]
            o += prev_state[2] * item["entity_step"][sid + 1][2]
            d += prev_state[0] * item["entity_step"][sid + 1][2]
            d += prev_state[1] * item["entity_step"][sid + 1][2]
            c += prev_state[2] * item["entity_step"][sid + 1][1]
            c += prev_state[2] * item["entity_step"][sid + 1][0]
            actions.append([1 - d, d])
            prev_state = item["entity_step"][sid + 1]
        return torch.tensor(actions)

    def getotherval(self, item):
        actions = []
        prev_state = item["entity_step"][0]
        for sid, step in enumerate(item["steps"]):
            o = 0
            c = 0
            d = 0
            o += prev_state[0] * item["entity_step"][sid + 1][0]
            o += prev_state[0] * item["entity_step"][sid + 1][1]
            o += prev_state[1] * item["entity_step"][sid + 1][0]
            o += prev_state[1] * item["entity_step"][sid + 1][1]
            o += prev_state[2] * item["entity_step"][sid + 1][2]
            d += prev_state[0] * item["entity_step"][sid + 1][2]
            d += prev_state[1] * item["entity_step"][sid + 1][2]
            c += prev_state[2] * item["entity_step"][sid + 1][1]
            c += prev_state[2] * item["entity_step"][sid + 1][0]
            actions.append([1 - o, o])
            prev_state = item["entity_step"][sid + 1]
        return torch.tensor(actions)

    def getbeforeval(self, item):
        b1s = []
        b2s = []
        for step in range(len(item["steps"]) + 1):
            b1 = torch.zeros(len(item["steps"]) + 1)
            b1[step] = 1
            for step1 in range(len(item["steps"]) + 1):
                b2 = torch.zeros(len(item["steps"]) + 1)
                b2[step1] = 1
                b1s.append(b1)
                b2s.append(b2)
        return torch.stack(b1s), torch.stack(b2s)

    def getbefore_trueval(self, item):
        num_steps = len(item["steps"]) + 1
        values = torch.zeros(num_steps * num_steps)
        for step in range(len(item["steps"]) + 1):
            for step1 in range(step + 1, len(item["steps"]) + 1):
                values[(step * num_steps) + step1] = 1
        return values
