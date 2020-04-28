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
from pycorenlp import StanfordCoreNLP
import pickle



nlp = StanfordCoreNLP('http://localhost:9000')
properties={
  'annotators': 'ssplit',
  'outputFormat': 'json'
  }

def sentence_split(text, properties={'annotators': 'ssplit', 'outputFormat': 'json'}):
    """Split sentence using Stanford NLP"""
    annotated = nlp.annotate(text, properties)
    sentence_split = list()
    for sentence in annotated['sentences']:
        s = [t['word'] for t in sentence['tokens']]
        k = [item.lower() for item in s if item not in [",", ".", '...', '..']]
        sentence_split.append(" ".join(k))
    return sentence_split

class DataLoader:
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
        self.postags = []
        self.tagger = SequenceTagger.load('pos')

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
            text = re.sub('<[^<]+>', "\n", text)
            text = text.replace("Dr.", "Dr ").replace("Mr.", "Mr ").replace("Al-", "Al").replace("/", " ").replace("'s",
                                                                                                                   " s").replace(
                ".-", "-").replace("-", " ").replace("(", " ").replace(")", " ").replace("%", ' percent ')
            x = re.findall(r'[a-z]\.[A-Z]', text)
            for item in x:
                item = item.replace(".", "\.")
                y = item.replace("\.", " . ")
                text = re.sub(item, y, text)
            #             x = re.findall(r'[a-z]-[a-zA-Z]', text)
            #             for item in x:
            #                 y = item.replace("-", " ")
            #                 text = re.sub(item, y, text)
            text = re.sub(' +', ' ', text)
            sample['parse'] = text
            splitted = sentence_split(text)
            #             splitted = text.split(".")
            #             splitted = [re.sub(' +', ' ', re.sub(r'[^\w\s]',' ',data)).replace(" t ", "'t ") for data in splitted if len(data) > 2]
            splitted = [re.sub(' +', ' ', data).replace("'s", "' s").replace("' t ", "'t ") for data in splitted if
                        len(data) >= 1]
            sentences = [Sentence(re.sub(r'[^\w\s]', ' ', sent)) for sent in splitted if sent]
            sentences = [sentence for sentence in sentences if len(sentence) >= 1]
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

    def Sort(self, sub_li):

        # reverse = None (Sorts in Ascending order)
        # key is set to sort using second element of
        # sublist lambda has been used
        sub_li.sort(key=lambda x: x[3])
        return sub_li

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
                _all.extend([[mention.find("./head/charseq").text, mention.find("./head/charseq").attrib['START'],
                              entity.attrib, mention.attrib['ID'], mention.find("./head/charseq").attrib['END'],
                              mention.find("./extent/charseq").text] for mention in mentions])
            _all = sorted(_all, key=lambda x: int(x[1]))
            entities = [[x[0], x[2]['TYPE'], x[2]['SUBTYPE'], int(x[1]), x[3], int(x[4]), x[5]] for x in _all]
            sample['annotated'] = self.Sort(entities)
            relations = root.findall(".//relation")
            _all = []
            for relation in relations:
                mentions = relation.findall(".//relation_mention")
                _all.extend([[mention.findall("./relation_mention_argument")[0].attrib['REFID'],
                              mention.findall("./relation_mention_argument")[1].attrib['REFID'], relation.attrib,
                              mention.attrib["ID"]] for mention in mentions])
            sample['relations'] = [(x[0],
                                    x[1],
                                    x[2]['TYPE'],
                                    x[2]['SUBTYPE'],
                                    x[3]
                                    )
                                   for x in _all if x[2]['TYPE'] != "METONYMY"]
            sample['relations'].extend([(x[0],
                                         x[1],
                                         x[2]['TYPE'],
                                         "UNKNOWN",
                                         x[3]
                                         )
                                        for x in _all if x[2]['TYPE'] == "METONYMY"])
            #             sample['relations'] = _all
            iteration += 1
            total_ann += len(sample['annotated'])
            print('Read {} inputs with total {} annotations'.format(iteration, total_ann), end='\r')

    def match(self, s1, s2):
        if s1 == s2:
            return True
        elif s1 in s2.split("-"):
            return True
        else:
            if s1 == "u.s":
                s1 = "u.s."
            if s1 == "u.k":
                s1 = "u.k."
            if s1 == "u.n":
                s1 = "u.n."
            if s1 == "dr":
                s1 = "dr."
            if s1 == "mr":
                s1 = "mr."
            if s1 == "l":
                s1 == "led"
            if s1 == s2:
                return True
            elif s1 in s2.split("-"):
                return True
            if s2.count(s1):
                if len(s2) >= 3 and (len(s2) <= len(s1) + 1 or len(s1) / len(s2) >= 0.65):
                    return True
                if len(s1) >= 5 and len(s2) >= 8:
                    return True
                if len(s1) <= 4 and s1 == s2[0:len(s1)] or s1 == s2[len(s2) - len(s1):]:
                    return True
            #                 if s1 == "asia" and s2 == "asiaagent":
            #                     return True

            s1 = re.sub(r'[^\w\s]', '', s1)
            s2 = re.sub(r'[^\w\s]', '', s2)
            if s1 == s2:
                return True

        return False

    def lableInputs(self):
        print("starting to lable inputs", end='\n')
        print(" ")
        iteration = 0
        for sample in self.inputs:
            if sample['type'] == "ignore":
                continue
            total_phrases = 0
            passed = 0
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
                    if iteration < checked:
                        continue
                    annotation = sample['annotated'][iteration]
                    if iteration >= 1:
                        if annotation[3] <= sample['annotated'][iteration - 1][5]:
                            passed += 1
                            checked += 1
                            continue
                    #                             start = phrases[-1]['before']
                    _list = []
                    text = annotation[0]
                    x = re.findall(r'[a-z]\.[A-Z]', text)
                    for item in x:
                        item = item.replace(".", "\.")
                        y = item.replace("\.", " . ")
                        text = re.sub(item, y, text)
                    #                     x = re.findall(r'[a-z]-[A-Z]', text)
                    #                     for item in x:
                    #                         y = item.replace("-", " ")
                    #                         text = re.sub(item, y, text)
                    text = text.replace("Al-", "Al").replace("Dr.", "Dr ").replace("Mr.", "Mr ").replace("/",
                                                                                                         " ").replace(
                        "'s", " s").replace(".-", "-").replace("-", " ").replace("%", ' percent ')
                    text = re.sub(' +', ' ', text)
                    try:
                        labels = sentence_split(re.sub(r'[^\w\s]', ' ', text))[0]
                    except:
                        print("file name is ", sample['annotation_path'])
                        print("text is ", text)
                        print("iteration is ", iteration)
                        raise
                    labels = Sentence(labels).to_tokenized_string().replace("\n", " ").split(" ")
                    #                     if labels[0] == "u.s":
                    #                         labels[0] = "u.s."
                    #                     if labels[0] == "u.k":
                    #                         labels[0] = "u.k."
                    #                     if labels[0] == "u.n":
                    #                         labels[0] = "u.n."
                    exist_check = False
                    for _itt in range(start, len(sentence)):
                        if self.match(labels[0], sentence[_itt]):
                            exist_check = True
                            value_index = _itt - start
                            if len(labels) > len(sentence[_itt:]):
                                exist_check = False
                            else:
                                for _it in range(len(labels)):
                                    if self.match(labels[_it], sentence[start + value_index + _it]):
                                        _list.append(sentence[start + value_index + _it])
                                    else:
                                        break
                                if len(_list) != len(labels):
                                    _list = []
                                    exist_check = False
                        if exist_check:
                            break
                    if not exist_check:
                        break

                    check = True
                    example = {"text": annotation[0], "label": annotation[1], 'list': _list, 'value_index': value_index,
                               'before': start, 'id': annotation[4], 'start': annotation[3], 'end': annotation[5]}

                    for _it in range(start, start + value_index):
                        _input = {"word": sentence[_it], 'w-type': '-O-', 'type': '-O-', 'subtype': '-O-',
                                  'set': sample['type'], 'index': self.index_finder("-O-")}
                        annotated_sentence.append(_input)
                    if len(_list) == 1:
                        _input = {"word": _list[0], 'w-type': annotation[1], 'type': "L-" + annotation[1],
                                  'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3],
                                  'index': self.index_finder(annotation[1])}
                        annotated_sentence.append(_input)
                    elif len(_list) == 2:
                        _input = {"word": _list[0], 'w-type': annotation[1], 'type': "B-" + annotation[1],
                                  'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3],
                                  'index': self.index_finder(annotation[1])}
                        annotated_sentence.append(_input)
                        _input = {"word": _list[1], 'w-type': annotation[1], 'type': "L-" + annotation[1],
                                  'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3],
                                  'index': self.index_finder(annotation[1])}
                        annotated_sentence.append(_input)
                    else:
                        _input = {"word": _list[0], 'w-type': annotation[1], 'type': "B-" + annotation[1],
                                  'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3],
                                  'index': self.index_finder(annotation[1])}
                        annotated_sentence.append(_input)
                        _input = {"word": _list[-1], 'w-type': annotation[1], 'type': "L-" + annotation[1],
                                  'subtype': annotation[2],
                                  'set': sample['type'], 'start': annotation[3],
                                  'index': self.index_finder(annotation[1])}
                        annotated_sentence.append(_input)
                        for item in range(1, len(_list) - 1):
                            _input = {"word": _list[item], 'w-type': annotation[1], 'type': "I-" + annotation[1],
                                      'subtype': annotation[2],
                                      'set': sample['type'], 'start': annotation[3],
                                      'index': self.index_finder(annotation[1])}
                            annotated_sentence.append(_input)
                    start = start + value_index + len(_list)
                    example['after'] = start
                    example['boundary'] = [example['before'] + example['value_index'], example['after'] - 1]
                    checked = iteration + 1
                    phrases.append(example)
                for _it in range(start, len(sentence)):
                    _input = {"word": sentence[_it], 'w-type': '-O-', 'type': '-O-', 'subtype': '-O-',
                              'set': sample['type'], 'index': self.index_finder('-O-')}
                    annotated_sentence.append(_input)
                # annotated_sentence is an array of annotated words inside a sentence
                relations = []
                all_relations = []
                phrases_ids = [phrase['id'] for phrase in phrases]
                for phrase in phrases:
                    for rel in sample['relations']:
                        if rel in relations:
                            continue
                        if rel[0] == phrase['id'] and rel[1] in phrases_ids:
                            relations.append(rel)
                        elif (rel[1] not in phrases_ids or rel[0] not in phrases_ids) and (
                                rel[0] == phrase['id'] or rel[1] == phrase['id']):
                            all_relations.append(rel)
                #                 if(len(all_relations) != len(relations)):
                #                     print("relation not match ", sample['annotation_path'])

                sample['inputs'].append(annotated_sentence)
                sample['phrases'].append(phrases)
                total_phrases += len(phrases)
                if check:
                    self.data.append(
                        {"words": annotated_sentence, "sentence": sentence1.to_tokenized_string(), "phrases": phrases,
                         'relations': relations, 'all_relations': all_relations, 'file': sample['annotation_path']})
            iteration += 1
            if (total_phrases + passed != len(sample['annotated'])):
                print("phrase not match ", sample['annotation_path'])
            print('Labled {} inputs with total number of sentences with entities of {} sentences'.format(iteration, len(
                self.data)), end='\r')
        self.data = [item for item in self.data if len(item["words"]) != 0]

    def index_finder(self, _class, choices=["ORG", "FAC", "PER", "VEH", "LOC", "WEA", "GPE", "-O-"]):
        return choices.index(_class)

    def posTagFinder(self):
        _it = 0
        for _it in range(1500):
            item = random.choice(self.data)
            temp = Sentence(item['sentence'])
            self.tagger.predict(temp)
            _dict = temp.to_dict(tag_type='pos')
            self.postags.extend([sample['type'] for sample in _dict['entities'] if sample['type'] not in self.postags])
        self.postags = list(set(self.postags))

    def fire(self):
        self.listFiles()
        self.splitHint()
        self.readData()
        self.prepareLables()
        self.readLables()
        self.lableInputs()
#         self.posTagFinder()
#         self.posTagFinder()


def sentence_merger(data):
    update_list = []
    count = 0
    passing = 0
    removing = []
    print(len(data.data))
    for _iter in range(len(data.data)):
        if count < passing:
            count += 1
            continue
        else:
            count = 0
            passing = 0
        if len(data.data[_iter]['all_relations']) != 0:
            print("iteration is ", _iter)
            for relation in data.data[_iter]['all_relations']:
                if relation in data.data[_iter]['relations']:
                    print("here it is ")
                    continue
                search = None
                check = False
                if relation[0] in [phrase['id'] for phrase in data.data[_iter]['phrases']]:
                    search = relation[1]
                else:
                    search = relation[0]
                print(search)
                for j in range(1, 3):
                    if search in [phrase['id'] for phrase in data.data[_iter + j]['phrases']]:
                        print("it is in ", _iter + j)
                        check = True
                        new_start = len(data.data[_iter]['words'])
                        passing = j
                        for i in range(1, j + 1):
                            print("merging in process")
                            removing.append(_iter + i)
                            data.data[_iter]['words'].extend(data.data[_iter + i]['words'])
                            for phrase in data.data[_iter + i]['phrases']:
                                phrase['before'] += new_start
                            data.data[_iter]['phrases'].extend(data.data[_iter + i]['phrases'])
                            data.data[_iter]['relations'].extend(data.data[_iter + i]['relations'])
                            data.data[_iter]['all_relations'].extend(data.data[_iter + i]['all_relations'])
                            data.data[_iter]['sentence'] = " ".join(
                                [data.data[_iter]['sentence'], data.data[_iter + i]['sentence']])
                            new_start += len(data.data[_iter + i]['words'])
                        break
                if not check:
                    print(search)
                    #                 print(relation)
                    print(data.data[_iter]['sentence'])
                    #                 print(data.data[_iter]['relations'])
                    #                 print(data.data[_iter]['all_relations'])
                    print("error")
                break

    for remove in removing:
        del(data.data[remove])
    print(len(data.data))


class SimpleReader():
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
            for phrase in item['phrases']:
                phrase['index'] = phrase['value_index'] + phrase['before']
            phrase_bound = []
            for phrase in item['phrases']:
                phrase_bound.append([phrase['index'], phrase['after']-1])
            for relation in item['relations']:
                for phrase in item['phrases']:
                    if relation[0] == phrase['id']:
                        relation.append([phrase['index'], phrase['after']-1])
                    if relation[1] == phrase['id']:
                        relation.append([phrase['index'], phrase['after']-1])
            relation_locations = [[relation[5], relation[6], relation[2]] for relation in item['relations']]
            relations = {}
            keys = {"ART", "GEN-AFF", "METONYMY", "ORG-AFF", "PART-WHOLE", "PER-SOC", "PHYS"}
            for key in keys:
                relations[key]= []
            for relation_location in relation_locations:
                relations[relation_location[2]].append([relation_location[0], relation_location[1]])
            _dict = {
                "raw": item['sentence'],
                "FAC": self.get_prob(item['words'], "FAC"),
                "GPE": self.get_prob(item['words'], "GPE"),
                "LOC": self.get_prob(item['words'], "LOC"),
                "PER": self.get_prob(item['words'], "PER"),
                "ORG": self.get_prob(item['words'], "ORG"),
                "VEH": self.get_prob(item['words'], "VEH"),
                "WEA": self.get_prob(item['words'], "WEA"),
                "relations": relations,
                "boundaries": phrase_bound
            }
            yield _dict


class SubSimpleReader():
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
            if item['w-type'] == "WEA" and item['subtype'] == "Underspecified":
                item['subtype'] = "WEA-Underspecified"
            if item['w-type'] == label or item['subtype'] == label:
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
                "LOC" : self.get_prob(item['words'], "LOC"),
                "PER" : self.get_prob(item['words'], "PER"),
                "ORG" : self.get_prob(item['words'], "ORG"),
                "VEH" : self.get_prob(item['words'], "VEH"),
                "WEA" : self.get_prob(item['words'], "WEA"),
            }
            _subnames = ["Airport", "Building-Grounds", "Path", "Plant", "Subarea-Facility", "Continent", "County-or-District", "GPE-Cluster", "Nation", "Population-Center", "Special", "State-or-Province",
                        "Address", "Boundary", "Celestial", "Land-Region-Natural", "Region-General", "Region-International", "Water-Body", "Commercial", "Educational", "Entertainment", "Government",
                        "Media", "Medical-Science", "Non-Governmental", "Religious", "Sports", "Group", "Indeterminate", "Individual", "Air", "Land", "Subarea-Vehicle", "Underspecified", "Water",
                        "Biological", "Blunt", "Chemical", "Exploding", "Nuclear", "Projectile", "Sharp", "Shooting", "WEA-Underspecified"]
            for name in _subnames:
                _dict[name] = self.get_prob(item['words'], str(name))
            yield _dict

