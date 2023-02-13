import os
from regr.data.reader import RegrReader
import json
import torch


class ProparaReader(RegrReader):
    def parse_file(self):
        with open(self.file, "r") as f:
            lines = []
            for line in f:
                try:
                    if line != "\n":
                        lines.append(json.loads(str(line)))
                except:
                    raise
        items = lines
        final_dict = []
        for item in items:
            for i in range(len(item["participants"])):
                instance = item.copy()
                instance["participants"] = [item["participants"][i]]
                instance["states"] = item["states"][i]
                final_dict.append(instance)

        return final_dict

    #     def getDataval(self, item):
    #         return item

    def getParaIDval(self, item):
        return [item["para_id"]]

    def getSentencesval(self, item):
        data = ["step 0 goes here"]
        data.extend(item["sentence_texts"])
        return data

    def getEntityval(self, item):
        return item["participants"]

    def getnon_existenceval(self, item):
        values = []
        for value in item["states"]:
            if value == "-":
                values.append(1)
            else:
                values.append(0)
        return values

    def getunknownval(self, item):
        values = []
        for value in item["states"]:
            if value == "?":
                values.append(1)
            else:
                values.append(0)
        return values

    def getlocationval(self, item):
        values = []
        for value in item["states"]:
            if value != "?" and value != "-":
                values.append(1)
            else:
                values.append(0)
        return values

    def getLocationTextval(self, item):
        values = []
        for value in item["states"]:
            if value != "?" and value != "-":
                values.append(value)
            else:
                values.append("NAN")
        return values

    def getbeforeval(self, item):
        b1s = []
        b2s = []
        for step in range(len(item["states"]) + 1):
            b1 = torch.zeros(len(item["states"]) + 1)
            b1[step] = 1
            for step1 in range(len(item["states"]) + 1):
                b2 = torch.zeros(len(item["states"]) + 1)
                b2[step1] = 1
                b1s.append(b1)
                b2s.append(b2)
        return torch.stack(b1s), torch.stack(b2s)

    def getbefore_trueval(self, item):
        num_steps = len(item["states"]) + 1
        values = torch.zeros(num_steps * num_steps)
        for step in range(len(item["states"]) + 1):
            for step1 in range(step + 1, len(item["states"]) + 1):
                values[(step * num_steps) + step1] = 1
        return values