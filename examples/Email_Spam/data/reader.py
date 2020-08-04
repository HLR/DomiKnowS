import os
from regr.data.reader import RegrReader


class EmailSpamReader(RegrReader):

    def parse_file(self, ):
        folder = self.file
        data_spam = []
        data_ham = []
        for file in [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.txt')]:
            with open(file + "/spam", "r") as f:
                x = []
                for i in f:
                    x.append(i)
            data_spam.append(x)
        for file in [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.txt')]:
            with open(file + "/ham", "r") as f:
                x = []
                for i in f:
                    x.append(i)
            data_ham.append(x)
        final_data = []
        for dat in data_spam:
            item = {'subject': dat[0].split(":")[1]}
            index = [i for i, v in enumerate(dat) if v.startswith('- - - - - - - - -')]
            if len(index):
                index = index[0]
                item['body'] = "".join(dat[1:index])
                sub = [(i, v) for i, v in enumerate(dat[index:]) if v.startswith('subject')][0]
                item['forward_subject'] = sub[1].split(":")[1]
                item['forward_body'] = "".join(dat[index + sub[0] + 1:])
            else:
                item['body'] = item['body'] = ("").join(dat[1:])
            item['label'] = "spam"
            final_data.append(item)

        for dat in data_ham:
            item = {'subject': dat[0].split(":")[1]}
            index = [i for i, v in enumerate(dat) if v.startswith('- - - - - - - - -')]
            if len(index):
                index = index[0]
                item['body'] = "".join(dat[1:index])
                sub = [(i, v) for i, v in enumerate(dat[index:]) if v.startswith('subject')][0]
                item['forward_subject'] = sub[1].split(":")[1]
                item['forward_body'] = "".join(dat[index + sub[0] + 1:])
            else:
                item['body'] = item['body'] = ("").join(dat[1:])
            item['label'] = "ham"
            final_data.append(item)
        return final_data

    def getSubjectval(self, item):
        return item['subject']

    def getBodyval(self, item):
        return item['body']

    def getForwarSubjectdval(self, item):
        if 'forward_subject' in item:
            return item['forward_subject']
        else:
            return None

    def getForwardBodyval(self, item):
        if 'forward_body' in item:
            return item['forward_body']
        else:
            return None

    def getSpamval(self, item):
        if item['label'] == "spam":
            return 1
        else:
            return 0

    def getRegularval(self, item):
        if item['label'] == "ham":
            return 1
        else:
            return 0