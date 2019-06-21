from typing import Iterator, List, Dict, Set, Optional
from collections import OrderedDict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, AdjacencyField
from .conll import Conll04CorpusReader


corpus_reader = Conll04CorpusReader()


class SensableReaderMeta(type):
    # Steps constructing class:
    # 0. Resolving MRO entries
    # 1. Determining the appropriate metaclass
    # 2. Preparing the class namespace

    def __prepare__(name, bases, **kwds):
        namespace = OrderedDict()
        namespace['update'] = SensableReaderMeta.get_update(name)
        namespace['update_each'] = SensableReaderMeta.get_update_each(name)
        SensableReaderMeta.update_list[name] = OrderedDict()
        return namespace

    # 3. Executing the class body
    # 4. Creating the class object
    # def __call__(name, bases, namespace, **kwds)
    # def __new__(name, bases, namespace, **kwds) -> cls

    def __init__(cls, name, bases, namespace):
        super(SensableReaderMeta, cls).__init__(name, bases, namespace)

    # 5. Nothing more, ready to go

    update_list = {}  # just a global container

    update_each_suffix = '()'

    @staticmethod
    def get_update(name):
        def update(key):
            if key.endswith(SensableReaderMeta.update_each_suffix):
                raise ValueError('Data field name cannot end with "{}"'.format(
                    SensableReaderMeta.update_each_suffix))

            def up(func):
                def update_field(self_, fields, *args, **kwargs):
                    try:
                        field = func(self_, fields, *args, **kwargs)
                        if field is not None:
                            fields[key] = field
                    except KeyError:
                        pass
                    finally:
                        return fields
                SensableReaderMeta.update_list[name][key] = update_field
                return update_field
            return up
        return update

    @staticmethod
    def get_update_each(name):
        def update_each(func):
            def update_field(self_, fields, *args, **kwargs):
                for key, field in func(self_, fields, *args, **kwargs):
                    try:
                        if field is not None:
                            fields[key] = field
                    except KeyError:
                        pass
                return fields
            SensableReaderMeta.update_list[name][func.__name__ +
                                                 SensableReaderMeta.update_each_suffix] = update_field
            return update_field
        return update_each


update = None
update_each = None


def merge_parent_fields(cls, keys=[]):
    update_list = OrderedDict()
    # though same level, reversed order seems more expected
    for base in reversed(cls.__bases__):
        if issubclass(base, SensableReader):
            if len(keys) == 0:
                update_list.update(
                    SensableReaderMeta.update_list[base.__name__])
            else:
                for key in keys:
                    if key in SensableReaderMeta.update_list[base.__name__]:
                        update_list[key] = SensableReaderMeta.update_list[base.__name__][key]
    for key in keys:
        if key not in update_list:
            raise ValueError(
                'Cannot find key {} from any of the bases of {}'.format(key, cls.__name__))
    # update with current lastly
    update_list.update(SensableReaderMeta.update_list[cls.__name__])
    SensableReaderMeta.update_list[cls.__name__] = update_list


def optional_arg_decorator(fn):
    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args)
            return real_decorator
    return wrapped_decorator


@optional_arg_decorator
def inheritance_fields(cls, *keys):
    merge_parent_fields(cls, keys)
    return cls


class SensableReader(DatasetReader, metaclass=SensableReaderMeta):
    __metaclass__ = SensableReaderMeta

    def _to_instance(self, raw_sample) -> Instance:
        cls = type(self)
        fields = {}

        for key, update_field in SensableReaderMeta.update_list[cls.__name__].items():
            update_field(self, fields, raw_sample)
        return Instance(fields)

    def _read(self, *args, **kwargs) -> Iterator[Instance]:
        for raw_sample in self.raw_read(*args, **kwargs):
            yield self._to_instance(raw_sample)


class Conll04Reader(SensableReader):
    def __init__(self) -> None:
        super().__init__(lazy=False)
        self.token_indexers = {'phrase': SingleIdTokenIndexer('phrase')}

    @update('sentence')
    def update_sentence(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        (sentence, pos, labels), relations = raw_sample
        return TextField([Token(word) for word in sentence], self.token_indexers)

    @update('labels')
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        (sentence, pos, labels), relations = raw_sample
        if labels is None:
            return None
        return SequenceLabelField(labels, fields['sentence'])

    @update('relation')
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
            fields['sentence'],
            relation_labels,
            padding_value=-1  # multi-class label, use -1 for null class
        )

    def raw_read(self, file_path):
        yield from zip(*corpus_reader(file_path))


@inheritance_fields('sentence')
class Conll04BinaryReader(Conll04Reader):
    label_names = {'Other', 'Loc', 'Peop', 'Org', 'O'}
    relation_names = {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}

    @update_each
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
                fields['sentence'])

    @update_each
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
                fields['sentence'],
                padding_value=0
            )


@inheritance_fields
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

    @update('candidate')
    def update_candidate(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        (sentence, pos, labels), relations = raw_sample

        return SequenceLabelField([str(self._is_candidate(word, p, label, i))
                                   for i, (word, p, label) in enumerate(zip(sentence, pos, labels))],
                                  fields['sentence'])


@inheritance_fields
class Conll04CandidateFilteredReader(Conll04CandidateReader):
    def raw_read(self, file_path):
        yield from zip(*corpus_reader(file_path))

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


@inheritance_fields('sentence', 'candidate', 'update_labels()', 'update_relations()')
class Conll04CandidateBinaryReader(Conll04CandidateReader, Conll04BinaryReader):
    pass


@inheritance_fields('sentence', 'candidate', 'update_labels()', 'update_relations()')
class Conll04CandidateFilteredBinaryReader(Conll04CandidateFilteredReader, Conll04BinaryReader):
    pass

@inheritance_fields
class Conll04SensorReader(Conll04BinaryReader):
    pass
