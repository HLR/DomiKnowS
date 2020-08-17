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
        referables, relations, events = self.load_anno(doc_id, apf_path, text)
        return {'text': text, 'referables': referables, 'relations': relations, 'events': events}

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
        referables = {}
        relations = {}
        events = {}

        for node in document.findall(Entity.tag):
            entity = Entity(node, text)
            referables[entity.id] = entity

        for node in document.findall(Timex2.tag):
            timex2 = Timex2(node, text)
            referables[timex2.id] = timex2

        for node in document.findall(Value.tag):
            value = Value(node, text)
            referables[value.id] = value

        for node in document.findall(Relation.tag):
            relation = Relation(node, referables, text)
            relations[relation.id] = relation

        for node in document.findall(Event.tag):
            event = Event(node, referables, text)
            events[event.id] = event

        return referables, relations, events


class Reader():
    def __init__(self, path, language='English', status='adj'):
        self.path = path
        self.language = language
        self.status = status

    @property
    def path(self):
        return self._raw_reader.path

    @path.setter
    def path(self, path):
        self._raw_reader = RawReader(path)

    def __iter__(self):
        yield from self._raw_reader(language=self.language, status=self.status)


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
