import os
import re
import itertools
import glob
import difflib
import warnings
import xml.etree.ElementTree as ET
from xml.sax.saxutils import unescape

from .graph import ace05


class APFObject():
    tag = None

    def __init__(self, node, text):
        assert node.tag == self.tag, '{} must be created from "{}" node, "{}" is given.'.format(type(self), self.tag, node.tag)


class Charseq(APFObject):
    tag = 'charseq'
    differ = difflib.Differ()

    def __init__(self, node, text):
        super().__init__(node, text)
        self.start = int(node.attrib['START'])
        self.end = int(node.attrib['END']) + 1  # pythonic upper bound exclusion
        self.text = node.text
        a = unescape(text[self.start:self.end]) + '\n'
        b = unescape(self.text) + '\n'
        if a != b:
            warnings.warn(
                '<charseq> mismatch:\n %s' %
                ''.join(self.differ.compare(
                    a.splitlines(keepends=True),
                    b.splitlines(keepends=True))))



class Entity(APFObject):
    tag = 'entity'

    class Mention(APFObject):
        tag = 'entity_mention'

        def __init__(self, node, text):
            super().__init__(node, text)
            self.id = node.attrib['ID']
            self.type = node.attrib['TYPE']
            self.extent = Charseq(node.find('extent/charseq'), text)
            self.head = Charseq(node.find('head/charseq'), text)

    class Attribute(APFObject):
        tag = 'name'

        def __init__(self, node, text):
            super().__init__(node, text)
            self.name = node.attrib['NAME']
            self.text = Charseq(node.find('charseq'), text)

    def __init__(self, node, text):
        super().__init__(node, text)
        self.id = node.attrib['ID']
        self.type = ace05['Entities'][node.attrib['TYPE']]
        self.subtype = ace05['Entities']['{}-{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])]
        self.entity_class = node.attrib['CLASS']
        self.mentions = {}
        self.attributes = []
        for mention_node in node.findall('entity_mention'):
            self.mentions[mention_node.attrib['ID']] = self.Mention(mention_node, text)
        attributes_node = node.find('entity_attributes')
        if attributes_node:
            for name_node in attributes_node.findall('name'):
                self.attributes.append(self.Attribute(name_node, text))

class Timex2(APFObject):
    tag = 'timex2'

    class Mention(APFObject):
        tag = 'timex2_mention'

        def __init__(self, node, text):
            super().__init__(node, text)
            self.id = node.attrib['ID']
            self.extent = Charseq(node.find('extent/charseq'), text)

    def __init__(self, node, text):
        super().__init__(node, text)
        self.id = node.attrib['ID']
        self.mentions = {}
        for mention_node in node.findall(self.Mention.tag):
            self.mentions[mention_node.attrib['ID']] = self.Mention(mention_node, text)


class Value(Timex2):
    tag = 'value'

    class Mention(Timex2.Mention):
        tag = 'value_mention'

    def __init__(self, node, text):
        super().__init__(node, text)
        self.type = node.attrib['TYPE']
        self.subtype = node.attrib['SUBTYPE']


class Relation(APFObject):
    tag = 'relation'

    class Argument(APFObject):
        tag = 'relation_argument'

        def __init__(self, node, referables, text):
            super().__init__(node, text)
            self.refid = node.attrib['REFID']
            self.ref = referables[self.refid]
            self.role = node.attrib['ROLE']

    class Mention(APFObject):
        tag = 'relation_mention'

        class Argument(APFObject):
            tag = 'relation_mention_argument'

            def __init__(self, node, referables, text):
                super().__init__(node, text)
                self.refid = node.attrib['REFID']
                self.ref = referables[self.refid]
                self.role = node.attrib['ROLE']

        def __init__(self, node, referables, text):
            super().__init__(node, text)
            self.id = node.attrib['ID']
            self.lexical_condition = node.attrib['LEXICALCONDITION']
            self.extent = Charseq(node.find('extent/charseq'), text)
            self.arguments = [None, None]
            self.additional_arguments = []
            for argument_node in node.findall('relation_mention_argument'):
                referable = referables[argument_node.attrib['REFID'].rsplit('-',1)[0]]
                argument = self.Argument(argument_node, referable.mentions, text)
                if argument.role.startswith('Arg-'):
                    self.arguments[int(argument.role[-1])-1] = argument
                else:
                    self.additional_arguments.append(argument)

    def __init__(self, node, referables, text):
        super().__init__(node, text)
        self.id = node.attrib['ID']
        self.type = ace05['Relations'][node.attrib['TYPE']]
        subtype = node.attrib.get('SUBTYPE', None)
        self.subtype = ace05['Relations'][subtype] if subtype else None
        self.tense = node.attrib.get('TENSE')
        self.modality = node.attrib.get('MODALITY')
        self.arguments = [None, None]
        self.additional_arguments = []
        self.mentions = {}
        for argument_node in node.findall('relation_argument'):
            argument = self.Argument(argument_node, referables, text)
            if argument.role.startswith('Arg-'):
                self.arguments[int(argument.role[-1])-1] = argument
            else:
                self.additional_arguments.append(argument)
        for mention_node in node.findall('relation_mention'):
            self.mentions[mention_node.attrib['ID']] = self.Mention(mention_node, referables, text)


class Reader():
    languages = ['Arabic', 'Chinese', 'English', '*']
    status = ['fp1', 'fp2', 'adj', 'timex2norm', '*']
    re_tag = re.compile(r'\<\/?.*?\>', flags=re.M|re.S)

    def __init__(self, root):
        super().__init__()
        self.root = root

    def __call__(self, language='English', status='adj'):
        for doc_id, sgm_path, apf_path in self.docs(language=language, status=status):
            yield self.load(doc_id, sgm_path, apf_path)

    def docs(self, language='*', source='*', status='*'):
        for sgm_path in glob.glob(os.path.join(self.root, 'data', language, source, status, '*.sgm')):
            folder, basename = os.path.split(sgm_path)
            doc_id, ext = os.path.splitext(basename)
            apf_path = os.path.join(folder, f'{doc_id}.apf.xml')
            yield doc_id, sgm_path, apf_path

    def load(self, doc_id, sgm_path, apf_path):
        text = self.load_text(doc_id, sgm_path)
        anno = self.load_anno(doc_id, apf_path, text)
        return {'text': text, 'anno': anno}

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

        for node in document.findall('entity'):
            entity = Entity(node, text)
            referables[entity.id] = entity

        for node in document.findall('timex2'):
            timex2 = Timex2(node, text)
            referables[timex2.id] = timex2

        for node in document.findall('relation'):
            relation = Relation(node, referables, text)
            relations[relation.id] = relation

        return referables, relations, events
