import warnings
from typing import List, Dict, Tuple, Generator
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer, PosTagIndexer, DepLabelIndexer, NerTagIndexer
from allennlp.data.fields import Field, TextField, SequenceLabelField, AdjacencyField
from regr.data.allennlp.reader import SensableReader, keep_keys
import cls
if __package__ is None or __package__ == '':
    from conll import Conll04CorpusReader
else:
    from .conll import Conll04CorpusReader


DELIM = '/'
NONE_LABEL = 'O'
REPL = {'COMMA': ',',
        '-LRB-': '(',
        '-RRB-': ')',
        }
MERGE = {'`': '\'\'',
         '\'': '\'\''}


def convert_index(index):
    return [index, index + 1]


def convert_relations(tokens, relations):
    tokens, _, _ = tokens
    for j, relation in enumerate(relations):
        (relation_type, (src_index, src_token), (dst_index, dst_token)) = relation
        src_index = convert_index(src_index)
        dst_index = convert_index(dst_index)
        relations[j] = (relation_type, src_index, dst_index)
    return relations


def convert_labels(tokens):
    _, _, labels = tokens
    new_labels = []
    for j, label in enumerate(labels):
        if label != NONE_LABEL:
            new_index = convert_index(j)
            new_labels.append((label, new_index))
    return new_labels


def extend_index(start, length, index):
    increace = length - 1  # 1 -> length
    if index[0] > start:
        index[0] += increace
    if index[1] > start:
        index[1] += increace
    return index


def extend_relations(start, length, relations):
    for j, relation in enumerate(relations):
        (relation_type, src_index, dst_index) = relation
        src_index = extend_index(start, length, src_index)
        dst_index = extend_index(start, length, dst_index)
        relations[j] = (relation_type, src_index, dst_index)
    return relations


def extend_labels(start, length, labels):
    for j, label in enumerate(labels):
        (label, index) = label
        index = extend_index(start, length, index)
        labels[j] = (label, index)
    return labels


def remove_index(start, length, index):
    if index[0] > start:
        index[0] = max(start, index[0] - length)
    # above should be equivalent to below
    '''
    if src_index[0] < start:
        pass
    elif src_index[0] < start + length:
        src_index[0] = start
    else:
        src_index[0] -= length
    '''
    last = index[1] - 1  # src_index[1] is stop index, not last index
    if last > start:
        last = max(start, last - length)
    index[1] = last + 1

    return index


def remove_relations(start, length, relations):
    for j, relation in enumerate(relations):
        (relation_type, src_index, dst_index) = relation
        src_index = remove_index(start, length, src_index)
        dst_index = remove_index(start, length, dst_index)
        relations[j] = (relation_type, src_index, dst_index)

    # descending order to avoid index change
    for j in reversed(range(len(relations))):
        (relation_type, src_index, dst_index) = relations[j]
        if src_index[0] >= src_index[1] or dst_index[0] >= dst_index[1]:
            assert False, 'Should not remove any relation.'
            del relations[j]

    return relations


def remove_labels(start, length, labels):
    for j, label in enumerate(labels):
        (label, index) = label
        index = remove_index(start, length, index)
        labels[j] = (label, index)

    for j in reversed(range(len(labels))):  # descending order to avoid index change
        (label, index) = labels[j]
        if index[0] >= index[1]:
            assert False, 'Should not remove any entity.'
            del labels[j]
    return labels


def add_tokens_index(tokens, index):
    return ' '.join([tokens[i] for i in range(index[0], index[1])])


def add_tokens_relations(tokens, relations):
    for j, relation in enumerate(relations):
        (relation_type, src_index, dst_index) = relation
        src_token = add_tokens_index(tokens, src_index)
        dst_token = add_tokens_index(tokens, dst_index)
        relations[j] = (relation_type,
                        (src_index, src_token),
                        (dst_index, dst_token))
    return relations


def add_tokens_labels(tokens, labels):
    for j, label in enumerate(labels):
        (label, index) = label
        token = add_tokens_index(tokens, index)
        labels[j] = (label, index, token)
    return labels


def reprocess(sentence, relations, keep_entity=False, first=True):
    new_tokens = []
    new_pos_tags = []
    new_labels = []
    offset = 0

    relations = convert_relations(sentence, relations)
    labels = convert_labels(sentence)

    for i, (token, pos_tag, label) in enumerate(zip(*sentence)):
        if ((keep_entity and label == NONE_LABEL) or (not keep_entity)) and DELIM in token and DELIM in pos_tag:
            new_token = token.split(DELIM)
            new_pos_tag = pos_tag.split(DELIM)
        else:
            new_token = [token]
            new_pos_tag = [pos_tag]
        if len(new_token) != len(new_pos_tag):
            warnings.warn(('POS tags do not match tokens when processing original CONLL04 data. '
                           '{}:{} -> {}:{}.').format(token, pos_tag, new_token, new_pos_tag),
                          stacklevel=2)
        for j, word in enumerate(new_token):
            if label != NONE_LABEL and word == '.' and j > 0:
                new_token[j - 1] = new_token[j - 1] + '.'
                # no need to change new_pos_tag[j-1]
                del new_token[j]
                del new_pos_tag[j]
        # TODO: the problem with '.' remain when it is a entity and it is not abbreviation

        new_label = [label,] * len(new_token)
        new_tokens.extend(new_token)
        new_pos_tags.extend(new_pos_tag)
        new_labels.extend(new_label)

        relations = extend_relations(i + offset, len(new_token), relations)
        labels = extend_labels(i + offset, len(new_token), labels)
        offset += len(new_token) - 1

    # handle some known issue
    for i, new_token in enumerate(new_tokens):
        if new_token in REPL:
            new_tokens[i] = REPL[new_token]
        #   seperated `` and ''
        if i > 0 and new_token in MERGE and new_tokens[i - 1] == new_token:
            if first:
                new_tokens[i - 1] = MERGE[new_token]
                new_tokens[i] = ''
            else:
                new_tokens[i - 1] = ''
                new_tokens[i] = ERGE[new_token]

    # remove empty ''
    for i in reversed(range(len(new_tokens))):
        if len(new_tokens[i].strip()) == 0:
            del new_tokens[i]
            del new_pos_tags[i]
            del new_labels[i]
            relations = remove_relations(i, 1, relations)
            labels = remove_labels(i, 1, labels)

    relations = add_tokens_relations(new_tokens, relations)
    labels = add_tokens_labels(new_tokens, labels)
    return new_tokens, labels, relations


def spacy_index(index, tokens, spacy_tokens):
    char_index = 0
    char_length = 0
    for index_cur, token in enumerate(tokens):
        if index_cur < index[0]:
            pass
        elif index_cur < index[1]:
            char_length += len(token) + 1
        else:
            break
        char_index += len(token) + 1
    # - 1 char_length has one more space, then we do not need -1
    start_char_index = char_index - char_length

    matched_spacy_tokens = []
    for spacy_token in spacy_tokens:
        if spacy_token.idx < start_char_index:
            pass
        elif spacy_token.idx < start_char_index + char_length:
            assert spacy_token.idx + \
                len(spacy_token) <= start_char_index + char_length
            matched_spacy_tokens.append(spacy_token)
        else:
            break
    return matched_spacy_tokens


def spacy_process(splitter, tokens, labels, relations):
    sentence = ' '.join(tokens)
    spacy_tokens = splitter.split_words(sentence)

    spacy_labels = []
    for label in labels:
        (label_type, index, token) = label
        token = spacy_index(index, tokens, spacy_tokens)
        spacy_label = (label_type, token)
        spacy_labels.append(spacy_label)

    spacy_relations = []
    for relation in relations:
        (relation_type, (src_index, src_token), (dst_index, dst_token)) = relation
        src_token = spacy_index(src_index, tokens, spacy_tokens)
        dst_token = spacy_index(dst_index, tokens, spacy_tokens)
        spacy_relation = (relation_type, src_token, dst_token)
        spacy_relations.append(spacy_relation)

    return spacy_tokens, spacy_labels, spacy_relations


class Conll04SpaCyReader(SensableReader):
    corpus_reader = Conll04CorpusReader()
    splitter = SpacyWordSplitter(
        'en_core_web_sm', True, True, True, keep_spacy_tokens=True)

    def __init__(self, entity_anckor='last') -> None:
        super().__init__()
        self.entity_anckor = entity_anckor

    def entity_anckor(self, entity_anckor):
        if entity_anckor in ('first', 'lass'):
            raise ValueError('"entity_anckor" must be one of "first" or "last".')
        if entity_anckor == 'first':
            self._entity_index = 0
        elif entity_anckor == 'last':
            self._entity_index = -1

    entity_anckor = property(fset=entity_anckor)

    def raw_read(self, file_path):
        sentences, relation_lists = self.corpus_reader(file_path)
        for sentence, relations in zip(sentences, relation_lists):
            tokens, labels, relations = reprocess(sentence, relations)
            spacy_tokens, spacy_labels, spacy_relations = spacy_process(
                self.splitter, tokens, labels, relations)
            yield spacy_tokens, spacy_labels, spacy_relations

    @cls.tokens('sentence')
    def update_sentence(
        self,
        fields: Dict,
        raw_sample
    ) -> List:
        tokens, labels, relations = raw_sample
        return tokens

    @cls.textfield('word')
    def update_sentence_raw(
        self,
        fields,
        tokens
    ) -> Field:
        indexers = {'word': SingleIdTokenIndexer(namespace='word', lowercase_tokens=True)}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('pos_tag')
    def update_sentence_pos(
        self,
        fields,
        tokens
    ) -> Field:
        indexers = {'pos_tag': PosTagIndexer(namespace='pos_tag')}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('dep_tag')
    def update_sentence_dep(
        self,
        fields,
        tokens
    ) -> Field:
        indexers = {'dep_tag': DepLabelIndexer(namespace='dep_tag')}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.field('labels')
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        tokens, labels, relations = raw_sample
        if labels is None:
            return None
        label_list = [NONE_LABEL,] * len(tokens)
        for label_type, token in labels:
            label_list[token[self._entity_index].i] = label_type
        return SequenceLabelField(label_list, fields[self.get_fieldname('word')])

    @cls.field('relation')
    def update_relations(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For', 'Kill'}
        tokens, labels, relations = raw_sample
        if relations is None:
            return None
        relation_indices = []
        relation_labels = []
        for relation_type, src_token, dst_token in relations:
            relation_indices.append((src_token[self._entity_index].i, dst_token[self._entity_index].i))
            relation_labels.append(relation_type)
        return AdjacencyField(
            relation_indices,
            fields[self.get_fieldname('phrase')],
            relation_labels,
            padding_value=-1  # multi-class label, use -1 for null class
        )


@keep_keys(exclude=['labels', 'relation'])
class Conll04SpaCyBinaryReader(Conll04SpaCyReader):
    label_names = {'Other', 'Loc', 'Peop', 'Org', 'O'}
    relation_names = {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For', 'Kill'}

    @cls.fields
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Generator[Tuple[str, Field], None, None]:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        tokens, labels, relations = raw_sample

        labels_dict = {}
        for label_type in self.label_names:
            labels_dict[label_type] = [False,] * len(tokens)

        for label_type, token in labels:
            labels_dict[label_type][token[self._entity_index].i] = True

        for label_type, label_list in labels_dict.items():
            yield label_type, SequenceLabelField(label_list, fields[self.get_fieldname('word')])

    @cls.fields
    def update_relations(
        self,
        fields: Dict,
        raw_sample
    ) -> Generator[Tuple[str, Field], None, None]:
        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For', 'Kill'}
        tokens, labels, relations = raw_sample

        relations_dict = {}
        for relation_type in self.relation_names:
            relations_dict[relation_type] = [] # just need indices

        for relation_type, src_token, dst_token in relations:
            try:
                relations_dict[relation_type].append(
                    (src_token[self._entity_index].i, dst_token[self._entity_index].i))
            except KeyError:
                msg = 'Relationship {} is unknown. Sentence: {} Raltions: {}'.format(relation_type, tokens, relations)
                warnings.warn(msg, stacklevel=3)

        for relation_type, indices in relations_dict.items():
            yield relation_type, AdjacencyField(
                indices,
                fields[self.get_fieldname('word')],
                padding_value=0
            )
