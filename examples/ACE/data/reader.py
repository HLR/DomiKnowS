import torch
import numpy as np
import glob
from bs4 import BeautifulSoup
from flair.data import Sentence
from segtok.segmenter import split_single
import random
from flair.models import SequenceTagger


class DataLoader :
    def __init__(self, _paths, _hint):
        self.paths = _paths
        self.files = []
        self.inputs = []
        self.data = []
        self.train = []
        self.valid = []
        self.test = []
        self.hintFile = _hint
        self.splitTest = []
        self.splitValid = []
        self.splitTrain = []

    def listFiles(self):
        for path in self.paths:
            self.files = self.files + glob.glob(path + "*.sgm")
        print('Loaded {} files'.format(len(self.files)), end='\n')

    def splitHint(self):
        train = open(self.hintFile + "/train.txt", "r+")
        valid = open(self.hintFile + "/valid.txt", "r+")
        test = open(self.hintFile + "/test.txt", "r+")

        text_train = train.read()
        text_valid = valid.read()
        text_test = test.read()

        list_train = text_train.split("\n")
        for line in list_train:
            self.splitTrain.append(line + ".sgm")

        list_valid = text_valid.split("\n")
        for line in list_valid:
            self.splitValid.append(line + ".sgm")

        list_test = text_test.split("\n")
        for line in list_test:
            self.splitTest.append(line + ".sgm")

    def readData(self):
        print('Starting to Prepare {} files'.format(len(self.files)), end='\n')
        print(" ")
        iteration = 0
        for item in self.files:
            file1 = open(item, "r+")
            text = file1.read()
            sample = {"text": text, "id": item, "parse": BeautifulSoup(text, 'lxml')}
            sample['file_name'] = sample['id'].split("/")[-1]
            if sample['file_name'] in self.splitTrain:
                sample['type'] = "train"
            elif sample['file_name'] in self.splitValid:
                sample['type'] = "valid"
            elif sample['file_name'] in self.splitTest:
                sample['type'] = "test"
            else:
                sample['type'] = "ignore"
            text = sample['parse'].text.replace('\n\n', '. ').replace('\n', ' ')
            splitted = split_single(text)
            splitted = [data for data in splitted if data]
            sentences = [Sentence(sent) for sent in splitted]
            sentences = [sentence for sentence in sentences if len(sentence) >= 3]
            sample['sentences'] = sentences
            self.inputs.append(sample)
            file1.close()
            iteration += 1
            print('Prepared {} files'.format(iteration), end='\r')
        print('Preparation resulted in {} inputs'.format(len(self.inputs)))

    def prepareLables(self):
        print("starting to prepare lables", end='\n')
        print(" ")
        iteration = 0
        for sample in self.inputs:
            id = sample['id'].split('.sgm')[0]
            path = id + ".apf.xml"
            sample['annotation_path'] = path
            iteration += 1
            print('Prepared {} Lables'.format(iteration), end='\r')

    def readLables(self):
        print("starting to read lables", end='\n')
        iteration = 0
        total_ann = 0
        for sample in self.inputs:
            item = sample['annotation_path']
            # print(sample['id'])
            file1 = open(item, "r+")
            text = file1.read()
            temp = BeautifulSoup(text, 'lxml').find_all('entity')
            sample['annotated'] = [(x.select_one('entity_mention:nth-of-type(1) > charseq').string,
                                    x['type'],
                                    x['subtype'],
                                    x.select_one('entity_mention:nth-of-type(1) > extent > charseq').string.replace(
                                        "\n", " "))
                                   for x in temp]
            iteration += 1
            total_ann += len(sample['annotated'])
            print('Read {} inputs with total {} annotations'.format(iteration, total_ann), end='\r')
            file1.close()

    def lableInputs(self):
        print("starting to lable inputs", end='\n')
        print(" ")
        iteration = 0
        for sample in self.inputs:
            sample['inputs'] = []
            sample['phrases'] = []
            # words array cotains arrays of all sentences inside a file
            for sentence in sample['sentences']:
                annotated_sentence = []
                check = False
                #                 print(sentence)
                sentence1 = sentence
                sentence = sentence.to_tokenized_string().split(" ")
                #                 print(sentence)
                # temp array is a sentence of words
                # for token in sentence:
                #     check2 = False
                #     #                     string = token.
                #     for annotation in sample['annotated']:
                #         if len(annotation[0].split(' ')) > 1:
                #             if len(annotation[0].split(' ')) == 2:
                #                 check = True
                #                 check2 = True
                #                 _input = {"word": token, 'type': "L-" + annotation[1], 'subtype': annotation[2],
                #                           "mentioned": annotation[3], 'set': sample['type']}
                #                 annotated_sentence.append(_input)
                #                 break
                #         else:
                #             if token == annotation[0].split(' ')[0]:
                #                 check = True
                #                 check2 = True
                #                 _input = {"word": token, 'type': "L-"+ annotation[1], 'subtype': annotation[2],
                #                           "mentioned": annotation[3], 'set': sample['type']}
                #                 annotated_sentence.append(_input)
                #                 break
                #
                #     if not check2 and (token == " " or token == "."):
                #         _input = {"word": token, 'type': '-O-', 'subtype': '-O-', "mentioned": "NONE",
                #                   'set': sample['type']}
                #         annotated_sentence.append(_input)
                #     elif not check2:
                #         _input = {"word": token, 'type': '-O-', 'subtype': '-O-', "mentioned": "NONE",
                #                   'set': sample['type']}
                #         annotated_sentence.append(_input)

                phrases = []
                for annotation in sample['annotated']:
                    phrases.append({"text": annotation[0], "label": annotation[1]})
                    start = 0
                    _list = []
                    labels = annotation[0].split(' ')
                    # for _it in range(start, len(sentence)):
                    if labels[0] in sentence[start:]:
                        check = True
                        value_index = sentence.index(labels[0])
                    else:
                        continue
                    for _it in range(start, value_index):
                        _input = {"word": sentence[_it], 'type': '-O-', 'subtype': '-O-', "mentioned": "NONE",
                                  'set': sample['type']}
                        annotated_sentence.append(_input)
                    for _it in range(len(labels)):
                        if labels[_it] == sentence[start+_it]:
                            _list.append(sentence[sentence[start+_it]])
                        else:
                            break
                    if len(_list) == 1:
                        _input = {"word": _list[0], 'type': "L-" + annotation[1], 'subtype': annotation[2],
                                          "mentioned": annotation[3], 'set': sample['type']}
                        annotated_sentence.append(_input)
                    elif len(_list) == 2:
                        _input = {"word": _list[0], 'type': "B-" + annotation[1], 'subtype': annotation[2],
                                  "mentioned": annotation[3], 'set': sample['type']}
                        annotated_sentence.append(_input)
                        _input = {"word": _list[1], 'type': "L-" + annotation[1], 'subtype': annotation[2],
                                  "mentioned": annotation[3], 'set': sample['type']}
                        annotated_sentence.append(_input)
                    else:
                        _input = {"word": _list[0], 'type': "B-" + annotation[1], 'subtype': annotation[2],
                                  "mentioned": annotation[3], 'set': sample['type']}
                        annotated_sentence.append(_input)
                        _input = {"word": _list[-1], 'type': "L-" + annotation[1], 'subtype': annotation[2],
                                  "mentioned": annotation[3], 'set': sample['type']}
                        annotated_sentence.append(_input)
                        for item in range(1, len(_list) - 1):
                            _input = {"word": _list[item], 'type': "I-" + annotation[1], 'subtype': annotation[2],
                                      "mentioned": annotation[3], 'set': sample['type']}
                            annotated_sentence.append(_input)
                    start = value_index + len(_list)

                # annotated_sentence is an array of annotated words inside a sentence
                sample['inputs'].append(annotated_sentence)
                sample['phrases'].append(phrases)
                if check:
                    self.data.append([annotated_sentence, sentence1, phrases])
            iteration += 1
            print('Labled {} inputs with total number of sentences with entities of {} sentences'.format(iteration, len(
                self.data)), end='\r')

    def fire(self):
        self.listFiles()
        self.splitHint()
        self.readData()
        self.prepareLables()
        self.readLables()
        self.lableInputs()


class ACEReader :
    def __init__(self, _data):
        self.data = _data
        self.lables = []
        self.total = 0
        self.makeLables()
        self.lablesCount = np.zeros(len(self.lables))
        self.ratio = np.zeros(len(self.lables))
        self.weights = np.zeros(len(self.lables))
        self.lableCount()
        self.lableRatio()
        self.outputWeights()
        self.train = []
        self.valid = []
        self.postags = []
        self.test = []
        self.splitSet()
        self.posTagFinder()
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def makeLables(self):
        for item in self.data:
            for phrase in item[2]:
                if phrase['label'] not in self.lables:
                    self.lables.append(phrase['label'])

    def count(self):
        return len(self.data)

    def lablesCount(self):
        return len(self.lables)

    def lableToArray(self, lable):
        array = torch.zeros(1, len(self.lables))
        for k in range(len(self.lables)):
            if lable == self.lables[k]:
                array[0][k] = 1
                break
        return array

    def lableToInt(self, lable):
        for k in range(len(self.lables)):
            if lable == self.lables[k]:
                return k

    def readTrain(self):
        for item in self.train:
            sentence = item[1]
            boundaries = [x["type"] for x in item[0]]
            lables = item[2]
            yield sentence, boundaries, lables

    def readValid(self):
        for item in self.valid:
            sentence = item[1]
            boundaries = [x["type"] for x in item[0]]
            lables = item[2]
            yield sentence, boundaries, lables

    def readTest(self):
        for item in self.test:
            sentence = item[1]
            boundaries = [x["type"] for x in item[0]]
            lables = item[2]
            yield sentence, boundaries, lables

    def lableCount(self):
        for item in self.data:
            for phrase in item[2]:
                for k in range(len(self.lables)):
                    if phrase['label'] == self.lables[k]:
                        self.lablesCount[k] += 1
                        self.total += 1

    def lableRatio(self):
        for index in range(len(self.lables)):
            self.ratio[index] = self.lablesCount[index] / self.total

    def outputWeights(self):
        un = 1 / len(self.lables)
        for index in range(len(self.lables)):
            # self.weights[index] = un / self.ratio[index]
            self.weights[index] = 1 - self.ratio[index]
        # id1 = self.lableToInt('FAC')
        # id2 = self.lableToInt('-O-')
        # self.weights[id1] /= 2
        # self.weights[id2] *= 2

    def intToLable(self, integer):
        return self.lables[integer]

    def splitInputs(self):
        split_frac = 0.8
        random.shuffle(self.data)
        self.train = [x for x in self.data[0:int(split_frac * len(self.data))]]
        remaining = [x for x in self.data[int(split_frac * len(self.data)):]]
        self.valid = remaining[0:int(len(remaining) * 0.5)]
        self.test = remaining[int(len(remaining) * 0.5):]

    def splitSet(self):
        for item in self.data:
            if item[0][0]['set'] == "train":
                self.train.append(item)
            elif item[0][0]['set'] == "test":
                self.test.append(item)
            elif item[0][0]['set'] == "valid":
                self.valid.append(item)

    def posTagFinder(self):
        tagger = SequenceTagger.load('pos')
        _it = 0
        for item in self.data:
            if _it >= 20:
                break
            tagger.predict(item[1])
            _dict = item[1].to_dict(tag_type='pos')
            self.postags.extend([sample['type'] for sample in _dict['entities'] if sample['type'] not in self.postags])
            _it += 1
        self.postags = list(set(self.postags))

    def postagEncoder(self, pos):
        encode = torch.zeros(len(self.postags), device=self.device).view(len(self.postags), 1)
        for it in range(len(self.postags)):
            if pos == self.postags[it]:
                encode[it] = 1
                break
        return encode.view(1, len(self.postags))

