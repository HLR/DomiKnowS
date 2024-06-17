import sys

sys.path.append('.')
sys.path.append('../..')

from domiknows.data.reader import RegrReader


class SentimentReader(RegrReader):

    def gettweetval(self, item):
        return [item['tweet']]

    def getNegativeLabelval(self, item):

        if item['Label'] == "\"0\"":
            return [1]
        else:
            return [0]

    def getPositiveLabelval(self, item):

        if item['Label'] == "\"4\"":
            return [1]
        else:
            return [0]

    def parse_file(self):

        object_list = []

        with open(self.file, encoding="ISO-8859-1") as f:
            content = f.readlines()
        for l in content:
            item = {}
            substring = l.split(",")
            item["Label"] = substring[0]
            item["tweet"] = ','.join(substring[5:])
            object_list.append(item)
        return object_list
