from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.fields import Field, TextField

from typing import List, Dict, Tuple, Generator
from allennlp.data.fields import Field, TextField, SequenceLabelField, AdjacencyField
import cls

from regr.data.allennlp.reader import keep_keys

if __package__ is None or __package__ == '':
    from data_spacy import Conll04SpaCyBinaryReader
else:
    from .data_spacy import Conll04SpaCyBinaryReader


@keep_keys('sentence', 'update_labels()', 'update_relations()')
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

    @cls.textfield('word')
    def update_sentence_raw(
        self,
        fields,
        tokens
    ) -> Field:
        indexers = {'word': PretrainedBertIndexer(pretrained_model='bert-base-uncased')}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.field('cancidate')
    def update_relation_cancidate(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
        tokens, labels, relations = raw_sample
        if relations is None:
            return None
        relation_indices = []
        for relation_type, src_token, dst_token in relations:
            relation_indices.append((src_token[self._entity_index], dst_token[self._entity_index]))
        return AdjacencyField(
            relation_indices,
            fields[self.get_fieldname('word')],
            padding_value=0
        )
