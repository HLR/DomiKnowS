import os
import re
import glob
import xml.etree.ElementTree as ET

from domiknows.graph import Concept

from .annotation import APFObject, Entity, Timex2, Value, Trigger, Relation, Event


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
        return {'id': doc_id, 'text': text, 'spans': spans, 'relations': relations, 'events': events}

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

        base_types = [Entity, Timex2, Value, Trigger]
        for BaseType in base_types:
            for node in document.findall(BaseType.tag):
                span = BaseType.from_node(node, text)
                spans[span.id] = span

        for node in document.findall(Relation.tag):
            relation = Relation.from_node(node, text, spans)
            relations[relation.id] = relation

        for node in document.findall(Event.tag):
            event = Event.from_node(node, text, spans)
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


class ParagraphReader(Reader):
    def _filter_span(self, spans, start, end, offset=None):
        from copy import deepcopy
        if offset is None:
            offset = start
        new_spans = {}
        for span_id, span_ in spans.items():
            span = deepcopy(span_)
            for key, mention in list(span.mentions.items()):
                if not (start < mention.extent.start and mention.extent.start < end and
                    start < mention.extent.end and mention.extent.end < end):
                    del span.mentions[key]
            if span.mentions: # if there is any left
                span.apply_offset(offset)
                new_spans[span_id] = span
        return new_spans

    def _filter_relation(self, relations, spans, start, end, offset=None):
        from copy import deepcopy
        if offset is None:
            offset = start
        new_relations = {}
        for relation_id, relation in relations.items():
            relation = deepcopy(relation)
            for key, mention in list(relation.mentions.items()):
                if not (start < mention.extent.start and mention.extent.start < end and
                    start < mention.extent.end and mention.extent.end < end):
                    del relation.mentions[key]
            if relation.mentions: # if there is any left
                relation.apply_offset(offset)
                new_relations[relation_id] = relation
        return new_relations

    def _filter_event(self, events, spans, start, end, offset=None):
        from copy import deepcopy
        if offset is None:
            offset = start
        new_events = {}
        for event_id, event in events.items():
            event = deepcopy(event)
            for key, mention in list(event.mentions.items()):
                if not (start < mention.extent.start and mention.extent.start < end and
                    start < mention.extent.end and mention.extent.end < end):
                    del event.mentions[key]
            if event.mentions: # if there is any left
                event.apply_offset(offset)
                new_events[event_id] = event
        return new_events

    def __iter__(self):
        for doc_sample in super().__iter__():
            # {'text': text, 'spans': spans, 'relations': relations, 'events': events}
            doc_id = doc_sample['id']
            doc_text = doc_sample['text']
            paragraphs = doc_text.split('\n\n')
            offset = 0
            for text in paragraphs:
                start = offset
                end = offset + len(text)
                assert doc_text[start:end] == text
                spans = self._filter_span(doc_sample['spans'], start=start, end=end)
                relations = self._filter_relation(doc_sample['relations'], spans, start=start, end=end)
                events = self._filter_event(doc_sample['events'], spans, start=start, end=end)
                if text:
                    yield {'id': f'{doc_id}_{offset}', 'text': text, 'offset': offset, 'spans': spans, 'relations': relations, 'events': events}
                offset = end + 2


class DictParagraphReader(DictReader, ParagraphReader):
    pass
