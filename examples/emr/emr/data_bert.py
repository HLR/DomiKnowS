from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.fields import Field, TextField

from typing import List, Dict, Tuple, Generator
from allennlp.data.fields import Field, TextField, SequenceLabelField, AdjacencyField
import cls

from regr.data.allennlp.reader import keep_keys
from .data_spacy import Conll04SpaCyBinaryReader


@keep_keys('sentence')
class Conll04BERTBinaryReader(Conll04SpaCyBinaryReader):
    splitter = BertBasicWordSplitter()

    @classmethod
    def match(cls, index, tokens, split_tokens):
        tk_idx = 0
        tk_tkn = tokens[tk_idx].lower()
        st_idx = 0
        st_tkn = split_tokens[st_idx].text.lower()

        matched_tokens = []
        while True:
            if index[0] <= tk_idx and tk_idx < index[1] and st_idx not in matched_tokens:
                matched_tokens.append(st_idx)

            if len(tk_tkn) < len(st_tkn):
                assert st_tkn.startswith(tk_tkn)
                st_tkn = st_tkn[len(tk_tkn):]
                tk_idx += 1
                tk_tkn = tokens[tk_idx].lower()
            elif len(tk_tkn) > len(st_tkn):
                assert tk_tkn.startswith(st_tkn)
                tk_tkn = tk_tkn[len(st_tkn):]
                st_idx += 1
                st_tkn = split_tokens[st_idx].text.lower()
            else:
                assert st_tkn == tk_tkn
                tk_idx += 1
                st_idx += 1
                if tk_idx == len(tokens):
                    assert st_idx == len(split_tokens)
                    break
                tk_tkn = tokens[tk_idx].lower()
                st_tkn = split_tokens[st_idx].text.lower()

        return matched_tokens

    @cls.textfield('bert')
    def update_sentence_raw(
        self,
        fields,
        tokens
    ) -> Field:
        indexers = {'bert': PretrainedBertIndexer(pretrained_model='bert-base-uncased')}
        textfield = TextField(tokens, indexers)
        #import pdb; pdb.set_trace()
        return textfield


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
            labels_dict[label_type][token[self._entity_index]] = True

        for label_type, label_list in labels_dict.items():
            #import pdb; pdb.set_trace()
            yield label_type, SequenceLabelField(label_list, fields[self.get_fieldname('bert')])

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
                    (src_token[self._entity_index], dst_token[self._entity_index]))
            except KeyError:
                msg = 'Relationship {} is unknown. Sentence: {} Raltions: {}'.format(relation_type, tokens, relations)
                warnings.warn(msg, stacklevel=3)

        for relation_type, indices in relations_dict.items():
            yield relation_type, AdjacencyField(
                indices,
                fields[self.get_fieldname('bert')],
                padding_value=0
            )
