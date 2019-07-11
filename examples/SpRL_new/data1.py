
from typing import Iterator, List, Dict, Set, Optional, Tuple, Iterable
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField, AdjacencyField
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def sprlCorpusReader(sprlxmlfile):
    sprlXMLTree = ET.parse(sprlxmlfile)

    # get root of the xml tree
    sprlXMLRoot = sprlXMLTree.getroot()

    sentences_list = []

    # iterate news items
    for sentenceItem in sprlXMLRoot.findall('./SCENE/SENTENCE'):

        sentence_dic = {}

        sentence_dic["id"] = sentenceItem.attrib["id"]

        # iterate child elements of item
        for child in sentenceItem:

            if child.tag == 'TEXT':
                sentence_dic[child.tag] = child.text
            elif child.tag == 'LANDMARK' or child.tag == 'TRAJECTOR':
                if "text" in child.attrib:
                    sentence_dic[child.tag] = child.attrib["text"]
                    if "start" in child.attrib:
                        padded_str = ' ' * int(child.attrib["start"]) + child.attrib["text"]
                        sentence_dic[child.tag + "padded"] = padded_str

        sentences_list.append(sentence_dic)

    # create empty dataform for sentences

    return sentences_list


corpus_reader = sprlCorpusReader()

class SpRLReader(DatasetReader):

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
        if relation_labels is None:
            # giving none for label because user do not want label
            # return directly
            return fields
        fields['relation'] = AdjacencyField(
            relation_indices,
            fields['sentence'],
            relation_labels,
            padding_value=-1 # multi-class label, use -1 for null class
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
        for (sentence, labels), relation in zip(sentences, relations):
            yield self.to_instance(sentence, labels, relation)



