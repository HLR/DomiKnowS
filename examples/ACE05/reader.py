from graph import ace05


class APFObject():
    tag = None

    def __init__(self, node, text):
        assert node.tag == self.tag, '{} must be created from {} node, {} is given.'.format(type(self), self.tag, node.tag)


class Charseq(APFObject):
    tag = 'charseq'

    def __init__(self, node, text):
        super().__init__(node, text)
        self.start = int(node.attrib['START'])
        self.end = int(node.attrib['END']) + 1  # pythonic upper bound exclusion
        self.text = node.text
        assert text[self.start:self.end] == self.text, 'Text not match in {}: (index) {} != (text) {}'.format(node, text[self.start:self.end], self.text)


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
            self.name = node['NAME']
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
        for name_node in attributes_node.find('name'):
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

        class Argument(Relation.Argument):
            tag = 'relation_mention_argument'

        def __init__(self, node, referables, text):
            super().__init__(node, text)
            self.id = node.attrib['ID']
            self.lexical_condition = node.attrib['LEXICALCONDITION']
            self.extent = create_charseq(node.find('extent/charseq'), text)
            self.arguments = [None, None]
            self.additional_arguments: []
            for argument_node in node.findall('relation_mention_argument'):
                referable = referables[argument_node.attrib['REFID'].rsplit('-',1)[0]]
                argument = self.Argument(argument_node, referable.mentions, text)
                if argument.role.startswith('Arg-'):
                    self.arguments[int(role_str[-1])-1] = argument
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
                self.arguments[int(role_str[-1])-1] = argument
            else:
                self.additional_arguments.append(argument)
        for mention_node in node.findall('relation_mention'):
            self.mentions[mention_node.attrib['ID']] = self.Mention(mention, referables, text)
