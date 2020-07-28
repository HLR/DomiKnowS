import os
from regr.data.reader import RegrReader


class EmailSpamReader(RegrReader):

    def parse_file(self, ):
        folder = self.file
        data = []
        for file in [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.txt')]:
            with open(file, "r") as f:
                x = []
                for i in f:
                    x.append(i)
            data.append(x)
        final_data = []
        for dat in data:
            item = {'subject': dat[0].split(":")[1]}
            index = [i for i, v in enumerate(dat) if v.startswith('- - - - - - - - -')]
            if len(index):
                index = index[0]
                item['body'] = "".join(dat[1:index])
                item['forward'] = {}
                sub = [(i, v) for i, v in enumerate(dat[index:]) if v.startswith('subject')][0]
                item['forward']['subject'] = sub[1].split(":")[1]
                item['forward']['body'] = "".join(dat[index + sub[0] + 1:])
            else:
                item['body'] = item['body'] = ("").join(dat[1:])
            final_data.append(item)

        return final_data

    def getSubjectval(self, item):
        return item['subject']

    def getBodyval(self, item):
        return item['body']

    def getForwarSubjectdval(self, item):
        if 'forward' in item:
            return item['forward']['subject']
        else:
            return None

    def getForwardBodyval(self, item):
        if 'forward' in item:
            return item['forward']['body']
        else:
            return None