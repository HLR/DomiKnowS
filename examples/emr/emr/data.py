from typing import Iterator, List, Dict, Set, Optional
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField
import numpy as np
from tqdm import tqdm


class Conll04CorpusReader():
    import logging
    logger = logging.getLogger(__name__)

    def sentence_start(self):
        Conll04CorpusReader.logger.debug('---- sentence start')
        self.sentence = []
        self.words = []
        self.poss = []
        self.labels = []

    def sentence_finish(self):
        Conll04CorpusReader.logger.debug('---- sentence finish')
        self.sentences.append((self.words, self.poss, self.labels))
        Conll04CorpusReader.logger.debug((self.words, self.poss, self.labels))

    def sentence_append(self, line):
        Conll04CorpusReader.logger.debug('---- sentence append')
        # new trunc in sentence
        '''
3    O    0    O    DT    The    O    O    O
3    O    1    O    JJ    long    O    O    O
3    Other    2    O    JJ    Palestinian    O    O    O
3    O    3    O    NN    uprising    O    O    O
3    O    4    O    VBZ    has    O    O    O
3    O    5    O    VBN    brought    O    O    O
3    O    6    O    NN    bitterness    O    O    O
        '''
        row = line.split()
        word = row[5]
        pos = row[4]
        label = row[1]  # row[1] if row[1] != 'O' else None
        self.words.append(word)
        self.poss.append(pos)
        self.labels.append(label)
        self.sentence.append((word, pos, label))
        Conll04CorpusReader.logger.debug(self.sentence[-1])

    def relation_start(self):
        Conll04CorpusReader.logger.debug('---- relation start')
        self.relation = []

    def relation_finish(self):
        Conll04CorpusReader.logger.debug('---- relation finish')
        self.relations.append(self.relation)
        Conll04CorpusReader.logger.debug(self.relation)

    def relation_append(self, line):
        Conll04CorpusReader.logger.debug('---- relation append')
        # new relation
        '''
6    8    Located_In
11    8    OrgBased_In
11    13    OrgBased_In
13    8    Located_In
        '''
        row = line.split()
        arg_1_idx = int(row[0])
        arg_2_idx = int(row[1])
        relation_type = row[2]
        arg_1 = (arg_1_idx, self.sentence[arg_1_idx])
        arg_2 = (arg_2_idx, self.sentence[arg_2_idx])
        self.relation.append((relation_type, arg_1, arg_2))
        Conll04CorpusReader.logger.debug(self.relation[-1])

    def sent2rel(self):
        self.sentence_finish()
        self.relation_start()

    def rel2sent(self):
        self.relation_finish()
        self.sentence_start()

    STATE_END = 0
    STATE_BEGIN = 1
    STATE_SENT = 2
    STATE_REL = 3
    #       empty                   non-empty           # input
    STT = [[(STATE_END, None),       (STATE_END, None)],  # STATE_END
           [(STATE_BEGIN, None),     (STATE_SENT, sentence_start)],  # STATE_BEGIN
           [(STATE_REL, sent2rel),   (STATE_SENT, None)],  # STATE_SENT
           [(STATE_SENT, rel2sent),  (STATE_REL, None)]]  # STATE_REL
    state_func = {STATE_END:   None,
                  STATE_BEGIN: None,
                  STATE_SENT:  sentence_append,
                  STATE_REL:   relation_append,
                  }

    def __call__(self, path):
        self.sentences = []
        self.relations = []
        # start from STATE_BEGIN (1), not 0 (a dead end)
        state = Conll04CorpusReader.STATE_BEGIN

        with open(path) as fin:
            lines = [line for line in fin]

        for line in tqdm(lines):
            line = line.strip()
            Conll04CorpusReader.logger.debug(line)
            state, trans_func = Conll04CorpusReader.STT[state][bool(line)]
            Conll04CorpusReader.logger.debug(state)
            if trans_func is not None:
                trans_func(self)
            if line and Conll04CorpusReader.state_func[state] is not None:
                Conll04CorpusReader.state_func[state](self, line)

        return self.sentences, self.relations


from typing import Iterable, List, Tuple
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, AdjacencyField


corpus_reader = Conll04CorpusReader()


class Conll04Reader(DatasetReader):
    def __init__(self) -> None:
        super().__init__(lazy=False)
        # 'tokens' could be just any name, and I don't know where it is need again
        # checkout modules used in word2vec, they need this name there
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

    def update_sentence(
        self,
        fields: Dict,
        sentence: List[str]
    ) -> Dict:
        fields['sentence'] = TextField(
            [Token(word) for word in sentence], self.token_indexers)
        return fields

    def update_labels(
        self,
        fields: Dict,
        labels: List[str]
    ) -> Dict:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        fields['label'] = SequenceLabelField(labels, fields['sentence'])
        return fields

    def update_relations(
        self,
        fields: Dict,
        relation_indices: List[Tuple[int, int]],
        relation_labels: Optional[List[str]]=None
    ) -> Dict:
        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}
        fields['relation'] = AdjacencyField(
            relation_indices,
            fields['sentence'],
            relation_labels
        )
        return fields

    def to_instance(
        self,
        sentence: List[str],
        labels: Optional[List[str]],
        relations: Optional[List[Tuple[str, Tuple[int, tuple], Tuple[int, tuple]]]]=None,
    ) -> Instance:
        fields = {}

        fields = self.update_sentence(fields, sentence)

        if labels is not None:
            fields = self.update_labels(fields, labels)

        if relations is not None:
            # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}
            relation_indices = []
            relation_labels = []
            for rel in relations:
                src_index = rel[1][0]
                dst_index = rel[2][0]
                relation_indices.append((src_index, dst_index))
                relation_labels.append(rel[0])
            fields = self.update_relations(fields, relation_indices, relation_labels)
        return Instance(fields)

    def _read(
        self,
        file_path: str
    ) -> Iterable[Instance]:
        sentences, relations = corpus_reader(file_path)
        for (sentence, pos, labels), relation in zip(sentences, relations):
            yield self.to_instance(sentence, labels, relation)


class Conll04TokenReader(Conll04Reader):
    def _read(
        self,
        file_path: str
    ) -> Iterable[Instance]:
        sentences, relations = corpus_reader(file_path)
        for (sentence, pos, labels), relation in zip(sentences, relations):
            for word, label in zip(sentence, labels):
                yield self.to_instance([word, ], [label, ], None)


class Conll04TokenBinaryReader(Conll04TokenReader):
    def __init__(
        self,
        label_names: Set[str]
    ) -> None:
        super().__init__()
        self.label_names = label_names

    def update_labels(
        self,
        fields: Dict,
        labels: List[str]
    ) -> Dict:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        for label_name in self.label_names:
            fields[label_name] = SequenceLabelField(
                [str(label == label_name) for label in labels],
                fields['sentence'])
        return fields


class Conll04BinaryReader(Conll04Reader):
    label_names = {'Other', 'Loc', 'Peop', 'Org', 'O'}
    relation_names = {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}

    def update_labels(
        self,
        fields: Dict,
        labels: List[str]
    ) -> Dict:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        for label_name in self.label_names:
            fields[label_name] = SequenceLabelField(
                [str(label == label_name) for label in labels],
                fields['sentence'])
        return fields

    def update_relations(
        self,
        fields: Dict,
        relation_indices: List[Tuple[int, int]],
        relation_labels: Optional[List[str]]=None
    ) -> Dict:
        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}
        for relation_name in self.relation_names:
            fields[relation_name] = AdjacencyField(
                relation_indices,
                fields['sentence'],
                # relation_labels # label is no need int binary case
            )
        return fields


from torch import Tensor
DataSource = List[Dict[str, Tensor]]
# should be consistent with the one in library
from allennlp.data.vocabulary import Vocabulary


class Data(object):
    def __init__(
        self,
        train_dataset: DataSource=None,
        valid_dataset: DataSource=None,
        test_dataset: DataSource=None,
    ) -> None:
        instances = []
        self.train_dataset = train_dataset
        if train_dataset is not None:
            instances += train_dataset

        self.valid_dataset = valid_dataset
        if valid_dataset is not None:
            instances += valid_dataset

        self.test_dataset = test_dataset
        if test_dataset is not None:
            instances += test_dataset

        vocab = Vocabulary.from_instances(instances)

        self.vocab = vocab

    def __getitem__(self, name: str) -> str:
        # return an identifier the module can use in forward function to get the data
        return name
