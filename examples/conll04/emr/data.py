from typing import List, Dict, Generator, Tuple, Type
from collections import defaultdict
from allennlp.data import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, PosTagIndexer, DepLabelIndexer, NerTagIndexer
from allennlp.data.fields import Field, TextField, SequenceLabelField, AdjacencyField
from regr.data.allennlp.reader import SensableReader, keep_keys
import cls
from typing import Iterator
from allennlp.data import Instance
from .conll import Conll04CorpusReader
from typing import Iterator
from allennlp.data import Instance

corpus_reader = Conll04CorpusReader()


class Conll04Reader(SensableReader):
    def __init__(self) -> None:
        super().__init__(lazy=False)

    def raw_read(self, file_path, **kwargs):
        yield from zip(*corpus_reader(file_path))

    @cls.tokens('sentence')
    def update_sentence(
        self,
        fields: Dict,
        raw_sample
    ) -> List[Token]:
        (sentence, pos, labels), relations = raw_sample
        # NB: pos_ for coarse POS tags, and tag_ for fine-grained
        return [Token(word, pos_=pos_tag, tag_=pos_tag)
                    for word, pos_tag in zip(sentence, pos)]

    @cls.textfield('phrase')
    def update_sentence_raw(
        self,
        fields,
        tokens
    ) -> Field:
        indexers = {'phrase': SingleIdTokenIndexer(namespace='phrase')}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('pos_tags')
    def update_sentence_pos(
        self,
        fields,
        tokens
    ) -> Field:
        indexers = {'pos_tags': PosTagIndexer(namespace='pos_tags')} # PosTagIndexer(...coarse_tags:bool=False...)
        textfield = TextField(tokens, indexers)
        return textfield

    #@token_indexer('sentence', 'dep_tags')
    def update_sentence_dep(
        self,
    ) -> Type[TokenIndexer]:
        return DepLabelIndexer

    #@token_indexer('sentence', 'ner_tags')
    def update_sentence_ner(
        self,
    ) -> Type[TokenIndexer]:
        return NerTagIndexer

    @cls.field('labels')
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        (sentence, pos, labels), relations = raw_sample
        if labels is None:
            return None
        return SequenceLabelField(labels, fields[self.get_fieldname('phrase')])

    @cls.field('relation')
    def update_relations(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
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
            fields[self.get_fieldname('phrase')],
            relation_labels,
            padding_value=-1  # multi-class label, use -1 for null class
        )
      
    def _read(self, *args, **kwargs) -> Iterator[Instance]:
        for raw_sample in self.raw_read(*args, **kwargs):
            #raw_sample=self.negative_entity_generation(raw_sample)
            yield self._to_instance(raw_sample)

    def negative_entity_generation(self,raw_sample):
        temp_num=0
        label_dict={}
        label_dict['NONE']=[]

        sentence = " ".join(raw_sample[0][0])
        new_chunk=self.getChunk(sentence)

        for label in raw_sample[0][2]:

          try:
              label_dict[label].append(raw_sample[0][0][temp_num])
          except:
              label_dict[label]=[]
              label_dict[label].append(raw_sample[0][0][temp_num])
          temp_num += 1

        for chunk in new_chunk:

            for value in label_dict.items():

                if chunk in value[1]:
                    continue
                elif self.getHeadwords(chunk) in value[1]:
                    value[1].append(chunk)
                    raw_sample[0][0].append(chunk)
                    raw_sample[0][1].append(self.get_Postag(chunk))
                    raw_sample[0][2].append(value[0])

            if chunk not in value[1]:
                raw_sample[0][0].append(chunk)
                raw_sample[0][1].append(self.get_Postag(chunk))
                raw_sample[0][2].append("NONE")
        return raw_sample


@keep_keys('sentence', 'phrase', 'pos_tags')
class Conll04BinaryReader(Conll04Reader):
    label_names = {'Other', 'Loc', 'Peop', 'Org', 'O'}
    relation_names = {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For', 'Kill'}

    @cls.fields
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Generator[Tuple[str, Field], None, None]:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        (sentence, pos, labels), relations = raw_sample
        for label_name in self.label_names:
            yield label_name, SequenceLabelField(
                [str(label == label_name) for label in labels],
                fields[self.get_fieldname('phrase')])

    @cls.fields
    def update_relations(
        self,
        fields: Dict,
        raw_sample
    ) -> Generator[Tuple[str, Field], None, None]:
        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For', 'Kill'}
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
                fields[self.get_fieldname('phrase')],
                padding_value=0
            )


@keep_keys
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

    @cls.field('candidate')
    def update_candidate(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
        (sentence, pos, labels), relations = raw_sample

        return SequenceLabelField([str(self._is_candidate(word, p, label, i))
                                   for i, (word, p, label) in enumerate(zip(sentence, pos, labels))],
                                  fields[self.get_fieldname('phrase')])


@keep_keys
class Conll04CandidateFilteredReader(Conll04CandidateReader):
    def raw_read(self, file_path, **kwargs):
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


@keep_keys('sentence', 'candidate', 'update_labels()', 'update_relations()')
class Conll04CandidateBinaryReader(Conll04CandidateReader, Conll04BinaryReader):
    pass


@keep_keys('sentence', 'candidate', 'update_labels()', 'update_relations()')
class Conll04CandidateFilteredBinaryReader(Conll04CandidateFilteredReader, Conll04BinaryReader):
    pass


@keep_keys
class Conll04SensorReader(Conll04BinaryReader):
    pass
