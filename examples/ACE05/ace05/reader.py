import os
import re
import glob
import xml.etree.ElementTree as ET

from regr.graph import Concept

from .annotation import APFObject, Entity, Timex2, Value, Relation, Event


class RawReader():
    languages = ['Arabic', 'Chinese', 'English', '*']
    status = ['fp1', 'fp2', 'adj', 'timex2norm', '*']
    re_tag = re.compile(r'\<\/?.*?\>', flags=re.M|re.S)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def __call__(self, language='English', status='adj'):
        for doc_id, sgm_path, apf_path in self.docs(language=language, status=status):
            yield self.load(doc_id, sgm_path, apf_path)

    def docs(self, language='*', source='*', status='*'):
        for sgm_path in glob.glob(os.path.join(self.path, 'data', language, source, status, '*.sgm')):
            folder, basename = os.path.split(sgm_path)
            doc_id, ext = os.path.splitext(basename)
            apf_path = os.path.join(folder, f'{doc_id}.apf.xml')
            yield doc_id, sgm_path, apf_path

    def load(self, doc_id, sgm_path, apf_path):
        text = self.load_text(doc_id, sgm_path)
        spans, relations, events = self.load_anno(doc_id, apf_path, text)
        return {'text': text, 'spans': spans, 'relations': relations, 'events': events}

    def load_text(self, doc_id, path):
        # tree = ET.parse(path)
        # root = tree.getroot()
        # text = ''.join(itertools.chain(*root.itertext()))
        with open(path) as fin:
            text = fin.read()
        text = self.re_tag.sub('', text)
        return text

    def load_anno(self, doc_id, path, text):
        tree = ET.parse(path)
        root = tree.getroot()
        document = root.find('document')
        spans = {}
        relations = {}
        events = {}

        for node in document.findall(Entity.tag):
            entity = Entity(node, text)
            spans[entity.id] = entity

        for node in document.findall(Timex2.tag):
            timex2 = Timex2(node, text)
            spans[timex2.id] = timex2

        for node in document.findall(Value.tag):
            value = Value(node, text)
            spans[value.id] = value

        for node in document.findall(Relation.tag):
            relation = Relation(node, spans, text)
            relations[relation.id] = relation

        for node in document.findall(Event.tag):
            event = Event(node, spans, text)
            events[event.id] = event

        return spans, relations, events


class SplitRawReader(RawReader):
    def __init__(self, path, list_path=None, type=None):
        super().__init__(path)
        self.list_path = list_path
        self.type = type

    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, type):
        self._type = type
        id_list = []
        if type is not None:
            with open(self.list_path, 'r') as fin:
                fin.readline()
                for line in fin:
                    ltype, lpath = line.split(',')
                    if ltype == self._type:
                        lid = lpath.rsplit('/', 1)[-1].rstrip()
                        id_list.append(lid)
        self._id_list = id_list

    def docs(self, language='*', source='*', status='*'):
        if self._type is not None:
            yield from filter(lambda doc_output: doc_output[0] in self._id_list, super().docs(language=language, source=source, status=status))
        else:
            yield from super().docs(language=language, source=source, status=status)


class Reader():
    def __init__(self, path, list_path=None, type=None, language='English', status='adj'):
        self.path = path
        self.list_path = list_path
        self.type = type
        self.init_reader()
        self.language = language
        self.status = status

    def init_reader(self):
        self._raw_reader = SplitRawReader(path=self.path, list_path=self.list_path, type=self.type)

    @property
    def raw_reader(self):
        return self._raw_reader

    def __iter__(self):
        yield from self.raw_reader(language=self.language, status=self.status)

    def __len__(self):
        return len(list(iter(self)))


class DictReader(Reader):
    def _make_dict(self, obj):
        if isinstance(obj, APFObject):
            obj = obj.__dict__
        if isinstance(obj, dict) and not isinstance(obj, Concept):
            obj = {k: self._make_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            obj = [self._make_dict(v) for v in obj]
        return obj

    def __iter__(self):
        yield from map(self._make_dict, super().__iter__())
