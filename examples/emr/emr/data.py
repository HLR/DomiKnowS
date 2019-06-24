from typing import Dict
from collections import defaultdict
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField, AdjacencyField
from regr.data.allennlp.reader import SensableReader, keep_fields
from .conll import Conll04CorpusReader


corpus_reader = Conll04CorpusReader()


class Conll04Reader(SensableReader):
    def __init__(self) -> None:
        super().__init__(lazy=False)

    def raw_read(self, file_path):
        yield from zip(*corpus_reader(file_path))

    @field('sentence')
    def update_sentence(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        (sentence, pos, labels), relations = raw_sample
        return TextField([Token(word) for word in sentence], self.get_token_indexers('sentence'))

    @field('labels')
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        (sentence, pos, labels), relations = raw_sample
        if labels is None:
            return None
        return SequenceLabelField(labels, fields[self.get_fieldname('sentence')])

    @field('relation')
    def update_relations(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}
        (sentence, pos, labels), relations = raw_sample
        if relations is None:
            return None
        relation_indices = []
        relation_labels = []
        for rel in relations:
            src_index = rel[1][0]
            dst_index = rel[2][0]
            relation_indices.append((src_index, dst_index))
            relation_labels.append(rel[0])
        return AdjacencyField(
            relation_indices,
            fields[self.get_fieldname('sentence')],
            relation_labels,
            padding_value=-1  # multi-class label, use -1 for null class
        )


@keep_fields('sentence')
class Conll04BinaryReader(Conll04Reader):
    label_names = {'Other', 'Loc', 'Peop', 'Org', 'O'}
    relation_names = {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}

    @fields
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        (sentence, pos, labels), relations = raw_sample
        for label_name in self.label_names:
            yield label_name, SequenceLabelField(
                [str(label == label_name) for label in labels],
                fields[self.get_fieldname('sentence')])

    @fields
    def update_relations(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}
        (sentence, pos, labels), relations = raw_sample
        if relations is None:
            return None
        relation_indices = []
        relation_labels = []
        for rel in relations:
            src_index = rel[1][0]
            dst_index = rel[2][0]
            relation_indices.append((src_index, dst_index))
            relation_labels.append(rel[0])

        for relation_name in self.relation_names:
            cur_indices = []
            for index, label in zip(relation_indices, relation_labels):
                if label == relation_name:
                    cur_indices.append(index)

            yield relation_name, AdjacencyField(
                cur_indices,
                fields[self.get_fieldname('sentence')],
                padding_value=0
            )


@keep_fields
class Conll04CandidateReader(Conll04Reader):
    def _is_candidate(
        self,
        word,
        p,
        label,
        i
    ) -> bool:
        # return (label != 'O') # this is too good and too strong ...
        candidate_p = {'NN', 'NNS', 'NNP', 'NNPS'}
        p_list = p.split('/')
        for cp in candidate_p:
            if cp in p_list:
                return True
        # also possible to add non-'O' not in candidate_p
        #   but it will be another strong bias that
        #   those not in candidate_p are something important
        return False

    @field('candidate')
    def update_candidate(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        (sentence, pos, labels), relations = raw_sample

        return SequenceLabelField([str(self._is_candidate(word, p, label, i))
                                   for i, (word, p, label) in enumerate(zip(sentence, pos, labels))],
                                  fields['sentence'])


@keep_fields
class Conll04CandidateFilteredReader(Conll04CandidateReader):
    def raw_read(self, file_path):
        for (sentence, pos, labels), relation in zip(sentences, relations):
            select = [self._is_candidate(word, p, label, i, relation)
                      for i, (word, p, label) in enumerate(zip(sentence, pos, labels))]
            select = np.array(select)
            if select.sum() == 0:
                # skip blank sentence after filter
                continue

            sentence = [val for val, sel in zip(sentence, select) if sel]
            pos = [val for val, sel in zip(pos, select) if sel]
            labels = [val for val, sel in zip(labels, select) if sel]
            new_relation = []
            for rel, (src, src_val), (dst, dst_val) in relation:
                if not select[src] or not select[dst]:
                    # skip the relation with filtered word
                    continue
                new_src = select[:src].sum()
                new_dst = select[:dst].sum()
                new_relation.append(
                    (rel, (new_src, src_val), (new_dst, dst_val)))
            relation = new_relation

            yield (sentence, pos, labels), relation


@keep_fields('sentence', 'candidate', 'update_labels()', 'update_relations()')
class Conll04CandidateBinaryReader(Conll04CandidateReader, Conll04BinaryReader):
    pass


@keep_fields('sentence', 'candidate', 'update_labels()', 'update_relations()')
class Conll04CandidateFilteredBinaryReader(Conll04CandidateFilteredReader, Conll04BinaryReader):
    pass


@keep_fields
class Conll04SensorReader(Conll04BinaryReader):
    pass
