import itertools
import difflib
import warnings
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


class Span(APFObject):
    class Mention(APFObject):
        def __init__(self, span, node, text):
            super().__init__(node, text)
            self.id = node.attrib['ID']
            self.extent = Charseq(node.find('extent/charseq'), text)
            self.head = self.extent
            self.span_basetype = span.basetype
            self.span_type = span.type
            self.span_subtype = span.subtype

    def __init__(self, node, text):
        super().__init__(node, text)
        self.id = node.attrib['ID']
        self.init_types(node, text)
        self.mentions = {}
        for mention_node in node.findall(self.Mention.tag):
            self.mentions[mention_node.attrib['ID']] = self.Mention(self, mention_node, text)

    def init_types(self, node, text):
        self.basetype = type(self).__name__.lower()
        type_str = node.attrib.get('TYPE', None)
        self.type = type_str and ace05['Entities'][type_str]
        self.subtype = None

class Entity(Span):
    tag = 'entity'

    class Mention(Span.Mention):
        tag = 'entity_mention'

        def __init__(self, span, node, text):
            super().__init__(span, node, text)
            self.type = node.attrib['TYPE']
            self.head = Charseq(node.find('head/charseq'), text)

    class Attribute(APFObject):
        tag = 'name'

        def __init__(self, node, text):
            super().__init__(node, text)
            self.name = node.attrib['NAME']
            self.text = Charseq(node.find('charseq'), text)

    def __init__(self, node, text):
        super().__init__(node, text)
        self.entity_class = node.attrib['CLASS']
        self.attributes = []
        attributes_node = node.find('entity_attributes')
        if attributes_node:
            for name_node in attributes_node.findall('name'):
                self.attributes.append(self.Attribute(name_node, text))

    def init_types(self, node, text):
        super().init_types(node, text)
        self.subtype = ace05['Entities']['{}-{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])]


class Timex2(Span):
    tag = 'timex2'

    class Mention(Span.Mention):
        tag = 'timex2_mention'

    def init_types(self, node, text):
        super().init_types(node, text)
        self.type = ace05['Entities']['Timex2']


class Value(Span):
    tag = 'value'

    class Mention(Span.Mention):
        tag = 'value_mention'

    def init_types(self, node, text):
        super().init_types(node, text)
        subtype_str = node.attrib.get('SUBTYPE', None)
        self.subtype = subtype_str and ace05['Entities'][subtype_str]


class Trigger(Span):
    tag = 'event'

    type_map = {
        'Business': 'Business-Event',
        'Sentence': 'Sentence-Event'
    }

    class Mention(Span.Mention):
        tag = 'event_mention'

    def init_types(self, node, text):
        self.basetype = type(self).__name__.lower()
        type_str = node.attrib.get('TYPE', None)
        type_str = self.type_map.get(type_str, type_str)
        self.type = type_str and ace05['Events'][type_str]
        subtype_str = node.attrib.get('SUBTYPE', None)
        subtype_str = self.type_map.get(subtype_str, subtype_str)
        self.subtype = subtype_str and ace05['Events'][subtype_str]


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
            for argument_node in node.findall(self.Argument.tag):
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
        for argument_node in node.findall(self.Argument.tag):
            argument = self.Argument(argument_node, referables, text)
            if argument.role.startswith('Arg-'):
                self.arguments[int(argument.role[-1])-1] = argument
            else:
                self.additional_arguments.append(argument)
        for mention_node in node.findall(self.Mention.tag):
            self.mentions[mention_node.attrib['ID']] = self.Mention(mention_node, referables, text)


class Event(APFObject):
    tag = 'event'

    type_map = {
        'Business': 'Business-Event',
        'Sentence': 'Sentence-Event'
    }

    class Argument(APFObject):
        tag = 'event_argument'

        def __init__(self, node, referables, text):
            super().__init__(node, text)
            self.refid = node.attrib['REFID']
            self.ref = referables[self.refid]
            self.role = node.attrib['ROLE']

    class Mention(APFObject):
        tag = 'event_mention'

        class Argument(APFObject):
            tag = 'event_mention_argument'

            def __init__(self, node, referables, text):
                super().__init__(node, text)
                self.refid = node.attrib['REFID']
                self.ref = referables[self.refid]
                self.role = node.attrib['ROLE']

        def __init__(self, node, referables, text):
            super().__init__(node, text)
            self.id = node.attrib['ID']
            self.extent = Charseq(node.find('extent/charseq'), text)
            self.ldc_scope = Charseq(node.find('ldc_scope/charseq'), text)
            self.anchor = Charseq(node.find('anchor/charseq'), text)
            self.arguments = []
            for argument_node in node.findall(self.Argument.tag):
                referable = referables[argument_node.attrib['REFID'].rsplit('-',1)[0]]
                argument = self.Argument(argument_node, referable.mentions, text)
                self.arguments.append(argument)

    def __init__(self, node, referables, text):
        super().__init__(node, text)
        self.id = node.attrib['ID']
        type_str = node.attrib['TYPE']
        type_str = self.type_map.get(type_str, type_str)
        self.type = ace05['Events'][type_str]
        subtype_str = node.attrib.get('SUBTYPE', None)
        subtype_str = self.type_map.get(subtype_str, subtype_str)
        self.subtype = ace05['Events'][subtype_str] if subtype_str else None
        self.modality = node.attrib['MODALITY']
        self.polarity = node.attrib['POLARITY']
        self.genericity = node.attrib['GENERICITY']
        self.tense = node.attrib['TENSE']
        self.arguments = []
        self.mentions = {}
        for argument_node in node.findall(self.Argument.tag):
            argument = self.Argument(argument_node, referables, text)
            self.arguments.append(argument)
        for mention_node in node.findall(self.Mention.tag):
            self.mentions[mention_node.attrib['ID']] = self.Mention(mention_node, referables, text)
