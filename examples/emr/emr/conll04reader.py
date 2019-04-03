from typing import Iterator, List, Dict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField
import numpy as np
from tqdm import tqdm


class Conll04Reader():
    import logging
    logger = logging.getLogger(__name__)

    def sentence_start(self):
        Conll04Reader.logger.debug('---- sentence start')
        self.sentence = []
        self.words = []
        self.poss = []
        self.labels = []

    def sentence_finish(self):
        Conll04Reader.logger.debug('---- sentence finish')
        self.sentences.append((self.words, self.poss, self.labels))
        Conll04Reader.logger.debug((self.words, self.poss, self.labels))

    def sentence_append(self, line):
        Conll04Reader.logger.debug('---- sentence append')
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
        Conll04Reader.logger.debug(self.sentence[-1])

    def relation_start(self):
        Conll04Reader.logger.debug('---- relation start')
        self.relation = []

    def relation_finish(self):
        Conll04Reader.logger.debug('---- relation finish')
        self.relations.append(self.relation)
        Conll04Reader.logger.debug(self.relation)

    def relation_append(self, line):
        Conll04Reader.logger.debug('---- relation append')
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
        Conll04Reader.logger.debug(self.relation[-1])

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
        state = Conll04Reader.STATE_BEGIN

        with open(path) as fin:
            lines = [line for line in fin]

        for line in tqdm(lines):
            line = line.strip()
            Conll04Reader.logger.debug(line)
            state, trans_func = Conll04Reader.STT[state][bool(line)]
            Conll04Reader.logger.debug(state)
            if trans_func is not None:
                trans_func(self)
            if line and Conll04Reader.state_func[state] is not None:
                Conll04Reader.state_func[state](self, line)

        return self.sentences, self.relations


conll04_reader = Conll04Reader()


class Conll04DatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(
                labels=['Other' if tag == 'O' else tag for tag in tags], sequence_field=sentence_field)  # avoid class 'O'
            #label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
            fields["labels_mask"] = ArrayField(
                np.array([0 if tag == 'O' else 1 for tag in tags], dtype=np.long))

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        sentences, relations = conll04_reader(file_path)
        for (sentence, pos, label), relation in zip(sentences, relations):
            yield self.text_to_instance([Token(word) for word in sentence], label)
