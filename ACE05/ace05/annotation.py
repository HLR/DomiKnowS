import itertools
import difflib
import warnings
from copy import deepcopy
from typing import overload
from xml.sax.saxutils import unescape

from .graph import ace05


class APFObject():
    tag = None

    @classmethod
    def from_node(cls, node, *args, **kwargs):
        assert node.tag == cls.tag, f'{cls} must be created from "{cls.tag}" node, "{node.tag}" is given.'
        node_args = cls.parse_node(node, *args, **kwargs)
        obj = cls(**node_args)
        return obj

    @classmethod
    def parse_node(cls, node):
        return {}

    def apply_offset(self, offset):
        raise NotImplementedError()


class Charseq(APFObject):
    tag = 'charseq'
    differ = difflib.Differ()

    @classmethod
    def parse_node(cls, node, text):
        node_args = super().parse_node(node)
        start = int(node.attrib['START'])
        node_args['start'] = start
        end = int(node.attrib['END']) + 1  # pythonic upper bound exclusion
        node_args['end'] = end
        node_args['text'] = node.text
        a = unescape(text[start:end]) + '\n'
        b = unescape(node.text) + '\n'
        if a != b:
            warnings.warn(
                '<charseq> mismatch:\n %s' %
                ''.join(cls.differ.compare(
                    a.splitlines(keepends=True),
                    b.splitlines(keepends=True))))
        return node_args

    def __init__(self, start, end, text, offset=0):
        super().__init__()
        self.start = start
        self.end = end
        self.text = text
        self.offset = offset

    def apply_offset(self, offset):
        assert self.offset == 0
        self.start -= offset
        self.end -= offset
        self.offset = offset


class Span(APFObject):
    class Mention(APFObject):
        @classmethod
        def parse_node(cls, node, text, span):
            node_args = super().parse_node(node)
            node_args['id'] = node.attrib['ID']
            extent = Charseq.from_node(node.find('extent/charseq'), text)
            node_args['extent'] = extent
            node_args['head'] = extent
            node_args['span_basetype'] = span.basetype
            node_args['span_type'] = span.type
            node_args['span_subtype'] = span.subtype
            return node_args

        def __init__(self, id, extent, head, span_basetype, span_type, span_subtype):
            super().__init__()
            self.id = id
            self.extent = extent
            self.head = head
            self.span_basetype = span_basetype
            self.span_type = span_type
            self.span_subtype = span_subtype
        
        def apply_offset(self, offset):
            self.extent.apply_offset(offset)  # head is a reference to extent

    @classmethod
    def from_node(cls, node, *args, **kwargs):
        obj = super().from_node(node, *args, **kwargs)
        obj = cls.parse_node_post(obj, node, *args, **kwargs)
        return obj

    @classmethod
    def parse_node(cls, node, text):
        node_args = super().parse_node(node)
        node_args['id'] = node.attrib['ID']
        basetype, type, subtype = cls.init_types(node, text)
        node_args['basetype'] = basetype
        node_args['type'] = type
        node_args['subtype'] = subtype
        node_args['mentions'] = {}
        return node_args

    @classmethod
    def parse_node_post(cls, obj, node, text):
        mentions = obj.mentions
        for mention_node in node.findall(cls.Mention.tag):
            mentions[mention_node.attrib['ID']] = cls.Mention.from_node(mention_node, text, obj)
        return obj

    @classmethod
    def init_types(cls, node, text):
        basetype = cls.__name__.lower()
        type_str = node.attrib.get('TYPE', None)
        type = type_str and ace05['Entities'][type_str]
        subtype = None
        return basetype, type, subtype

    def __init__(self, id, basetype, type, subtype, mentions):
        super().__init__()
        self.id = id
        self.basetype = basetype
        self.type = type
        self.subtype = subtype
        self.mentions = mentions

    def apply_offset(self, offset):
        for mention in self.mentions.values():
            mention.apply_offset(offset)

    # def __copy__(self):
    #     return type(self)(id=self.id, basetype=self.basetype, type=self.type, subtype=self.subtype, mentions=self.mentions)

    # def __deepcopy__(self, memo):
    #     return type(self)(id=self.id, basetype=self.basetype, type=self.type, subtype=self.subtype, mentions=dict(self.mentions))


class Entity(Span):
    tag = 'entity'

    class Mention(Span.Mention):
        tag = 'entity_mention'

        @classmethod
        def parse_node(cls, node, text, span):
            node_args = super().parse_node(node, text, span)
            node_args['type'] = node.attrib['TYPE']
            node_args['head'] = Charseq.from_node(node.find('head/charseq'), text)
            return node_args

        def __init__(self, id, extent, head, span_basetype, span_type, span_subtype, type):
            super().__init__(id, extent, head, span_basetype, span_type, span_subtype)
            self.type = type

        def apply_offset(self, offset):
            super().apply_offset(offset)
            self.head.apply_offset(offset)

    class Attribute(APFObject):
        tag = 'name'

        @classmethod
        def parse_node(cls, node, text):
            node_args = super().parse_node(node)
            node_args['name'] = node.attrib['NAME']
            node_args['text'] = Charseq.from_node(node.find('charseq'), text)
            return node_args

        def __init__(self, name, text):
            super().__init__()
            self.name = name
            self.text = text

        def apply_offset(self, offset):
            self.text.apply_offset(offset)

    def __init__(self, id, basetype, type, subtype, mentions, entity_class, attributes):
        super().__init__(id, basetype, type, subtype, mentions)
        self.entity_class = entity_class
        self.attributes = attributes

    def apply_offset(self, offset):
        super().apply_offset(offset)
        for attribute in self.attributes:
            attribute.apply_offset(offset)

    @classmethod
    def parse_node(cls, node, text):
        node_args = super().parse_node(node, text)
        node_args['entity_class'] = node.attrib['CLASS']
        node_args['attributes'] = []
        attributes_node = node.find('entity_attributes')
        if attributes_node:
            for name_node in attributes_node.findall('name'):
                node_args['attributes'].append(cls.Attribute.from_node(name_node, text))
        return node_args

    @classmethod
    def init_types(cls, node, text):
        basetype, type, subtype = super().init_types(node, text)
        subtype = ace05['Entities']['{}-{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])]
        return basetype, type, subtype

    # def __copy__(self):
    #     return type(self)(id=self.id, basetype=self.basetype, type=self.type, subtype=self.subtype, mentions=self.mentions, entity_class=self.entity_class, attributes=self.attributes)

    # def __deepcopy__(self, memo):
    #     return type(self)(id=self.id, basetype=self.basetype, type=self.type, subtype=self.subtype, mentions=dict(self.mentions), entity_class=self.entity_class, attributes=list(self.attributes))


class Timex2(Span):
    tag = 'timex2'

    class Mention(Span.Mention):
        tag = 'timex2_mention'

    @classmethod
    def init_types(cls, node, text):
        basetype, type, subtype = super().init_types(node, text)
        type = ace05['Entities']['Timex2']
        return basetype, type, subtype


class Value(Span):
    tag = 'value'

    class Mention(Span.Mention):
        tag = 'value_mention'

    @classmethod
    def init_types(cls, node, text):
        basetype, type, subtype = super().init_types(node, text)
        subtype_str = node.attrib.get('SUBTYPE', None)
        subtype = subtype_str and ace05['Entities'][subtype_str]
        return basetype, type, subtype


class Trigger(Span):
    tag = 'event'

    type_map = {
        'Business': 'Business-Event',
        'Sentence': 'Sentence-Event'
    }

    class Mention(Span.Mention):
        tag = 'event_mention'

        @classmethod
        def parse_node(cls, node, text, span):
            node_args = super().parse_node(node, text, span)
            node_args['head'] = Charseq.from_node(node.find('anchor/charseq'), text)
            return node_args

        def apply_offset(self, offset):
            super().apply_offset(offset)
            self.head.apply_offset(offset)

    @classmethod
    def init_types(cls, node, text):
        basetype = cls.__name__.lower()
        type_str = node.attrib.get('TYPE', None)
        type_str = cls.type_map.get(type_str, type_str)
        type = type_str and ace05['Events'][type_str]
        subtype_str = node.attrib.get('SUBTYPE', None)
        subtype_str = cls.type_map.get(subtype_str, subtype_str)
        subtype = subtype_str and ace05['Events'][subtype_str]
        return basetype, type, subtype


class BaseArgument(APFObject):
    @classmethod
    def parse_node(cls, node, text, spans):
        node_args = super().parse_node(node)
        refid = node.attrib['REFID']
        node_args['refid'] = refid
        node_args['ref'] = spans[refid]
        node_args['role'] = node.attrib['ROLE']
        return node_args

    def __init__(self, refid, ref, role):
        super().__init__()
        self.refid = refid
        self.ref = ref
        self.role = role

    def apply_offset(self, offset):
        pass  # should not update reference `ref`


class Relation(APFObject):
    tag = 'relation'

    class Argument(BaseArgument):
        tag = 'relation_argument'

    class Mention(APFObject):
        tag = 'relation_mention'

        class Argument(BaseArgument):
            tag = 'relation_mention_argument'

        @classmethod
        def parse_node(cls, node, text, spans):
            node_args = super().parse_node(node)
            node_args['id'] = node.attrib['ID']
            node_args['lexical_condition'] = node.attrib['LEXICALCONDITION']
            node_args['extent'] = Charseq.from_node(node.find('extent/charseq'), text)
            node_args['arguments'] = [None, None]
            node_args['additional_arguments'] = []
            for argument_node in node.findall(cls.Argument.tag):
                span = spans[argument_node.attrib['REFID'].rsplit('-',1)[0]]
                argument = cls.Argument.from_node(argument_node, text, span.mentions)
                if argument.role.startswith('Arg-'):
                    node_args['arguments'][int(argument.role[-1])-1] = argument
                else:
                    node_args['additional_arguments'].append(argument)
            return node_args

        def __init__(self, id, lexical_condition, extent, arguments, additional_arguments):
            super().__init__()
            self.id = id
            self.lexical_condition = lexical_condition
            self.extent = extent
            self.arguments = arguments
            self.additional_arguments = additional_arguments

        def apply_offset(self, offset):
            self.extent.apply_offset(offset)
            for argument in self.arguments:
                if argument:
                    argument.apply_offset(offset)
            for argument in self.additional_arguments:
                argument.apply_offset(offset)

    @classmethod
    def parse_node(cls, node, text, spans):
        node_args = super().parse_node(node)
        node_args['id'] = node.attrib['ID']
        node_args['type'] = ace05['Relations'][node.attrib['TYPE']]
        subtype_str = node.attrib.get('SUBTYPE', None)
        node_args['subtype'] = subtype_str and ace05['Relations'][subtype_str]
        node_args['tense'] = node.attrib.get('TENSE')
        node_args['modality'] = node.attrib.get('MODALITY')
        node_args['arguments'] = [None, None]
        node_args['additional_arguments'] = []
        node_args['mentions'] = {}
        for argument_node in node.findall(cls.Argument.tag):
            argument = cls.Argument.from_node(argument_node, text, spans)
            if argument.role.startswith('Arg-'):
                node_args['arguments'][int(argument.role[-1])-1] = argument
            else:
                node_args['additional_arguments'].append(argument)
        for mention_node in node.findall(cls.Mention.tag):
            node_args['mentions'][mention_node.attrib['ID']] = cls.Mention.from_node(mention_node, text, spans)
        return node_args

    def __init__(self, id, type, subtype, tense, modality, arguments, additional_arguments, mentions):
        self.id = id
        self.type = type
        self.subtype = subtype
        self.tense = tense
        self.modality = modality
        self.arguments = arguments
        self.additional_arguments = additional_arguments
        self.mentions = mentions

    def apply_offset(self, offset):
        for argument in self.arguments:
            if argument:
                argument.apply_offset(offset)
        for argument in self.additional_arguments:
            argument.apply_offset(offset)
        for mention in self.mentions.values():
            mention.apply_offset(offset)

    # def __copy__(self):
    #     return type(self)(id=self.id, type=self.type, subtype=self.subtype, tense=self.tense, modality=self.modality, arguments=self.arguments, additional_arguments=self.additional_arguments, mentions=self.mentions)

    # def __deepcopy__(self, memo):
    #     return type(self)(id=self.id, type=self.type, subtype=self.subtype, tense=self.tense, modality=self.modality, arguments=list(self.arguments), additional_arguments=list(self.additional_arguments), mentions=dict(self.mentions))


class Event(APFObject):
    tag = 'event'

    type_map = {
        'Business': 'Business-Event',
        'Sentence': 'Sentence-Event'
    }

    class Argument(BaseArgument):
        tag = 'event_argument'

    class Mention(APFObject):
        tag = 'event_mention'

        class Argument(BaseArgument):
            tag = 'event_mention_argument'

        @classmethod
        def parse_node(cls, node, text, spans):
            node_args = super().parse_node(node)
            id = node.attrib['ID']
            node_args['id'] = id
            node_args['extent'] = Charseq.from_node(node.find('extent/charseq'), text)
            node_args['ldc_scope'] = Charseq.from_node(node.find('ldc_scope/charseq'), text)
            node_args['anchor'] = Charseq.from_node(node.find('anchor/charseq'), text)
            span = spans[id.rsplit('-', 1)[0]]
            node_args['trigger'] = span.mentions[id]
            node_args['arguments'] = []
            for argument_node in node.findall(cls.Argument.tag):
                span = spans[argument_node.attrib['REFID'].rsplit('-',1)[0]]
                argument = cls.Argument.from_node(argument_node, text, span.mentions)
                node_args['arguments'].append(argument)
            return node_args

        def __init__(self, id, extent, ldc_scope, anchor, trigger, arguments):
            super().__init__()
            self.id = id
            self.extent = extent
            self.ldc_scope = ldc_scope
            self.anchor = anchor
            self.trigger = trigger
            self.arguments = arguments

        def apply_offset(self, offset):
            self.extent.apply_offset(offset)
            self.ldc_scope.apply_offset(offset)
            self.anchor.apply_offset(offset)
            for argument in self.arguments:
                argument.apply_offset(offset)

    @classmethod
    def parse_node(cls, node, text, spans):
        node_args = super().parse_node(node)
        id = node.attrib['ID']
        node_args['id'] = id
        type_str = node.attrib['TYPE']
        type_str = cls.type_map.get(type_str, type_str)
        node_args['type'] = ace05['Events'][type_str]
        subtype_str = node.attrib.get('SUBTYPE', None)
        subtype_str = cls.type_map.get(subtype_str, subtype_str)
        node_args['subtype'] = subtype_str and ace05['Events'][subtype_str]
        node_args['trigger'] = spans[id]
        node_args['modality'] = node.attrib['MODALITY']
        node_args['polarity'] = node.attrib['POLARITY']
        node_args['genericity'] = node.attrib['GENERICITY']
        node_args['tense'] = node.attrib['TENSE']
        node_args['arguments'] = []
        node_args['mentions'] = {}
        for argument_node in node.findall(cls.Argument.tag):
            argument = cls.Argument.from_node(argument_node, text, spans)
            node_args['arguments'].append(argument)
        for mention_node in node.findall(cls.Mention.tag):
            node_args['mentions'][mention_node.attrib['ID']] = cls.Mention.from_node(mention_node, text, spans)
        return node_args

    def __init__(self, id, type, subtype, trigger, modality, polarity, genericity, tense, arguments, mentions):
        self.id = id
        self.type = type
        self.subtype = subtype
        self.trigger = trigger
        self.modality = modality
        self.polarity = polarity
        self.genericity = genericity
        self.tense = tense
        self.arguments = arguments
        self.mentions = mentions

    def apply_offset(self, offset):
        for argument in self.arguments:
            if argument:
                argument.apply_offset(offset)
        for mention in self.mentions.values():
            mention.apply_offset(offset)

    # def __copy__(self):
    #     return type(self)(id=self.id, type=self.type, subtype=self.subtype, trigger=self.trigger, modality=self.modality, polarity=self.polarity, genericity=self.genericity, tense=self.tense, arguments=self.arguments, mentions=self.mentions)

    # def __deepcopy__(self, memo):
    #     return type(self)(id=self.id, type=self.type, subtype=self.subtype, trigger=deepcopy(self.trigger, memo), modality=self.modality, polarity=self.polarity, genericity=self.genericity, tense=self.tense, arguments=list(self.arguments), mentions=dict(self.mentions))
