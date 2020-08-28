from emr.conll import Conll04CorpusReader
from regr.data.reader import RegrReader

class conllReader(RegrReader):

    def getSentenceval(self, item):
        return item['Sentence']

    def getOrgLabelval(self,item):

        if item['Label'] == "Org":
            return 1
        else:
            return 0
    def parse_file(self):
        object_list = []
        x = Conll04CorpusReader()
        s, r = x(self.file)
        for l in s:
            item = {}
            item["Sentence"]=' '.join(l[0])
            item["Label"]= l[2]
            item["Pos"]= l[1]
            item["Tokens"]= l[0]
            object_list.append(item)
        return object_list
