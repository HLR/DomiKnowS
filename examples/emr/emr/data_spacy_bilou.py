import warnings
from typing import Dict, Tuple, Generator
from allennlp.data.fields import Field, SequenceLabelField, AdjacencyField
from regr.data.allennlp.reader import keep_keys
import cls
from .data_spacy import Conll04SpaCyReader


@keep_keys(exclude=['labels', 'relation'])
class Conll04SpaCyBilouReader(Conll04SpaCyReader):
    def __init__(self, relation_anckor='last') -> None:
        super().__init__()
        self.relation_anckor = relation_anckor

    def relation_anckor(self, relation_anckor):
        if relation_anckor in ('first', 'lass'):
            raise ValueError('"relation_anckor" must be one of "first" or "last".')
        if relation_anckor == 'first':
            self._relation_index = 0
        elif relation_anckor == 'last':
            self._relation_index = -1

    relation_anckor = property(fset=relation_anckor)

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
        label_list = ['O',] * len(tokens)
        for label_type, token in labels:
            if len(token) == 1:
                label_list[token[0].i] = label_type + '-U'
            else:
                for i, word in enumerate(token):
                    if i == 0:
                        label_list[token[i].i] = label_type + '-B'
                    elif i < len(token)-1:
                        label_list[token[i].i] = label_type + '-I'
                    else:
                        label_list[token[i].i] = label_type + '-L'
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
            relation_indices.append((src_token[self._relation_index].i,
                                     dst_token[self._relation_index].i))
            relation_labels.append(relation_type)
        return AdjacencyField(
            relation_indices,
            fields[self.get_fieldname('phrase')],
            relation_labels,
            padding_value=-1  # multi-class label, use -1 for null class
        )


@keep_keys(exclude=['labels', 'relation'])
class Conll04SpaCyBilouSepReader(Conll04SpaCyBilouReader):
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
            labels_dict[label_type] = ['O', ] * len(tokens)

        for label_type, token in labels: # TODO: extend to BILOU
            if len(token) == 1:
                labels_dict[label_type][token[0].i] = 'U'
            else:
                for i, word in enumerate(token):
                    if i == 0:
                        labels_dict[label_type][token[i].i] = 'B'
                    elif i < len(token)-1:
                        labels_dict[label_type][token[i].i] = 'I'
                    else:
                        labels_dict[label_type][token[i].i] = 'L'

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
            relations_dict[relation_type] = []  # just need indices

        for relation_type, src_token, dst_token in relations:
            try:
                relations_dict[relation_type].append((src_token[self._relation_index].i,
                                                      dst_token[self._relation_index].i))
            except KeyError:
                msg = 'Relationship {} is unknown. Sentence: {} Raltions: {}'.format(relation_type, tokens, relations)
                warnings.warn(msg, stacklevel=3)

        for relation_type, indices in relations_dict.items():
            yield relation_type, AdjacencyField(
                indices,
                fields[self.get_fieldname('word')],
                padding_value=0
            )
