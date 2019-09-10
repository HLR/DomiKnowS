import logging
from typing import Dict, List, Set

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
#from allennlp.data.tokenizers.token import Token
from spacy.tokens import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class Indexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents tokens by their part of speech tag, as determined by
    the ``pos_`` or ``tag_`` fields on ``Token`` (corresponding to spacy's coarse-grained and
    fine-grained POS tags, respectively).

    Parameters
    ----------
    namespace : ``str``, optional (default=``pos_tokens``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    coarse_tags : ``bool``, optional (default=``False``)
        If ``True``, we will use coarse POS tags instead of the default fine-grained POS tags.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace: str = 'feature_labels',
                 coarse_tags: bool = False,
                 token_min_padding_length: int = 0) -> None:

        super().__init__(token_min_padding_length)
        self._namespace = namespace
        self._coarse_tags = coarse_tags
        self._logged_errors: Set[str] = set()

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}



class PosTaggerIndexer(Indexer):
    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if self._coarse_tags:
            tag = token._.pos_
        else:
            tag = token._.tag_
        if not tag:
            if token.text not in self._logged_errors:
                logger.warning("Token had no POS tag: %s", token.text)
                self._logged_errors.add(token.text)
            tag = 'NONE'
        counter[self._namespace][tag] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        tags: List[str] = []

        for token in tokens:
            if self._coarse_tags:
                tag = token._.pos_
            else:
                tag = token._.tag_
            if not tag:
                tag = 'NONE'

            tags.append(tag)

        return {index_name: [vocabulary.get_token_index(tag, self._namespace) for tag in tags]}


class LemmaIndexer(Indexer):
    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        lemma_label = token._.lemma_
        if not lemma_label:
            if token.text not in self._logged_errors:
                logger.warning("Token had no Lemma label: %s", token.text)
                self._logged_errors.add(token.text)
            lemma_label = 'NONE'
        counter[self._namespace][lemma_label] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        lemma_labels = [token._.lemma_ or 'NONE' for token in tokens]

        return {index_name: [vocabulary.get_token_index(lemma_label, self._namespace) for lemma_label in lemma_labels]}



class DependencyIndexer(Indexer):
    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        dep_label = token._.dep_
        if not dep_label:
            if token.text not in self._logged_errors:
                logger.warning("Token had no dependency label: %s", token.text)
                self._logged_errors.add(token.text)
            dep_label = 'NONE'
        counter[self._namespace][dep_label] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        dep_labels = [token._.dep_ or 'NONE' for token in tokens]

        return {index_name: [vocabulary.get_token_index(dep_label, self._namespace) for dep_label in dep_labels]}


class HeadwordIndexer(Indexer):

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        headword_label = token._.headword_
        if not headword_label:
            if token.text not in self._logged_errors:
                logger.warning("Token had no headword label: %s", token.text)
                self._logged_errors.add(token.text)
            headword_label = 'NONE'
        counter[self._namespace][headword_label] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        headword_labels = [token._.headword_ or 'NONE' for token in tokens]

        return {index_name: [vocabulary.get_token_index(headword_label, self._namespace) for headword_label in headword_labels]}


class PhrasePosIndexer(Indexer):

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        headword_label = token._.phrasepos_
        if not headword_label:
            if token.text not in self._logged_errors:
                logger.warning("Token had no phrasepos label: %s", token.text)
                self._logged_errors.add(token.text)
            headword_label = 'NONE'
        counter[self._namespace][headword_label] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        headword_labels = [token._.phrasepos_ or 'NONE' for token in tokens]

        return {index_name: [vocabulary.get_token_index(headword_label, self._namespace) for headword_label in headword_labels]}