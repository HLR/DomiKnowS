import torch
import numpy as np
import glob
from bs4 import BeautifulSoup
from flair.data import Sentence
import random
from flair.models import SequenceTagger
import xml.etree.ElementTree as ET
import string
import re
import json


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
            text = sample['parse'].select_one("TEXT").prettify()
            text = re.sub('<[^<]+>', "\n",text)
            text = text.replace('\n\n', '. ').replace('\n', ' ')
            text = re.sub(' +', ' ', text)
            sample['parse'] = text
            splitted = text.split(".")
            splitted = [re.sub(' +', ' ', re.sub(r'[^\w\s]',' ',data)).replace(" t ", "'t ") for data in splitted if len(data) > 2]
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
            tree = ET.parse(sample['annotation_path'])
            root = tree.getroot()
            entities = root.findall(".//entity")
            _all = []
            for entity in entities:
                mentions = entity.findall(".//entity_mention")
                _all.extend([[mention.find("./head/charseq").text, mention.find("./head/charseq").attrib['START'], entity.attrib] for mention in mentions])
            _all = sorted(_all, key=lambda x:int(x[1]))
            sample['annotated'] = [(x[0],
                                    x[2]['TYPE'],
                                    x[2]['SUBTYPE'],
                                    x[1]
                                   )
                                   for x in _all]
            iteration += 1
            total_ann += len(sample['annotated'])
            print('Read {} inputs with total {} annotations'.format(iteration, total_ann), end='\r')

    def lableInputs(self):
        print("starting to lable inputs", end='\n')
        print(" ")
        iteration = 0
        for sample in self.inputs:
            sample['inputs'] = []
            sample['phrases'] = []
            # words array cotains arrays of all sentences inside a file
            checked = 0
            for sentence in sample['sentences']:
                annotated_sentence = []
                check = False
                #                 print(sentence)
                sentence1 = sentence
                sentence = sentence.to_tokenized_string().split(" ")
                phrases = []
                start = 0
                for iteration in range(len(sample['annotated'])):
#                     print(iteration)
                    if iteration < checked:
#                         print("passed")
                        continue
                    annotation = sample['annotated'][iteration]
#                     print(annotation)
#                     print(sentence[start:])
                    _list = []
                    labels = Sentence(annotation[0]).to_tokenized_string().split(" ")
#                     print(labels)
#                     print("Annotation is ")
#                     print(annotation)
#                     print("label are")
#                     print(labels)
                    # for _it in range(start, len(sentence)):
                    if labels[0] in sentence[start:]:
                        value_index = sentence[start:].index(labels[0])
                        if len(labels) > len(sentence[value_index:]):
#                             print("b1")
                            break
                    else:
#                         print("b2")
                        break

                    for _it in range(len(labels)):
                        if labels[_it] == sentence[start+value_index+_it]:
                            _list.append(sentence[start+value_index+_it])
                        else:
#                             print(value_index)
#                             print(sentence[start+value_index+_it])
#                             print(_it)
#                             print(labels[_it])
#                             print("b3")
                            break
                    if len(_list) != len(labels):
#                         print("b4")
                        break
                    check = True
                    example = {"text": annotation[0], "label": annotation[1], 'list': _list, 'value_index': value_index, 'before' : start}

                    for _it in range(start, start+value_index):
                        _input = {"word": sentence[_it], 'w-type': '-O-', 'type': '-O-', 'subtype': '-O-',
                                  'set': sample['type']}
                        annotated_sentence.append(_input)
                    if len(_list) == 1:
                        _input = {"word": _list[0], 'w-type': annotation[1], 'type': "L-" + annotation[1], 'subtype': annotation[2],
                                          'set': sample['type'], 'start': annotation[3]}
                        annotated_sentence.append(_input)
                    elif len(_list) == 2:
                        _input = {"word": _list[0], 'w-type': annotation[1], 'type': "B-" + annotation[1], 'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3]}
                        annotated_sentence.append(_input)
                        _input = {"word": _list[1], 'w-type': annotation[1], 'type': "L-" + annotation[1], 'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3]}
                        annotated_sentence.append(_input)
                    else:
                        _input = {"word": _list[0], 'w-type': annotation[1], 'type': "B-" + annotation[1], 'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3]}
                        annotated_sentence.append(_input)
                        _input = {"word": _list[-1], 'w-type': annotation[1], 'type': "L-" + annotation[1], 'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3]}
                        annotated_sentence.append(_input)
                        for item in range(1, len(_list) - 1):
                            _input = {"word": _list[item], 'w-type': annotation[1], 'type': "I-" + annotation[1], 'subtype': annotation[2],
                                      'set': sample['type'], 'start': annotation[3]}
                            annotated_sentence.append(_input)
                    start = start + value_index + len(_list)
                    example['after'] = start
                    checked = iteration+1
                    phrases.append(example)
                for _it in range(start, len(sentence)):
                        _input = {"word": sentence[_it], 'w-type': '-O-', 'type': '-O-', 'subtype': '-O-',
                                  'set': sample['type']}
                        annotated_sentence.append(_input)
                # annotated_sentence is an array of annotated words inside a sentence
                sample['inputs'].append(annotated_sentence)
                sample['phrases'].append(phrases)
                if check:
                    self.data.append([annotated_sentence, sentence1.to_tokenized_string(), phrases])
            iteration += 1
            print('Labled {} inputs with total number of sentences with entities of {} sentences'.format(iteration, len(
                self.data)), end='\r')
        self.data = [item for item in self.data if len(item[0]) != 0]
    def fire(self):
        self.listFiles()
        self.splitHint()
        self.readData()
        self.prepareLables()
        self.readLables()
        self.lableInputs()


class SimpleReader:
    def __init__(self, file):
        self.file = file
        with open(file, 'r') as myfile:
            data = myfile.read()
        # parse file
        self.obj = json.loads(data)

    def index_finder(self, _class):
        choices = ["ORG", "FAC", "PER", "VEH", "LOC", "WEA", "GPE", "-O-"]
        return choices.index(_class)

    def get_prob(self, words, label):
        _list = []
        for item in words:
            if item['w-type'] == label:
                _list.append(1)
            else:
                _list.append(0)
        return _list

    def data(self):
        for item in self.obj:
            _dict = {
                "raw": item['sentence'],
                "FAC": self.get_prob(item['words'], "FAC"),
                "GPE": self.get_prob(item['words'], "GPE"),
                "LOC": self.get_prob(item['words'], "LOC"),
                "PER": self.get_prob(item['words'], "PER"),
                "ORG": self.get_prob(item['words'], "ORG"),
                "VEH": self.get_prob(item['words'], "VEH"),
                "WEA": self.get_prob(item['words'], "WEA"),
            }
            yield _dict



