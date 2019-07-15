import xml.etree.ElementTree as ET
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import pandas as pd
import logging as logger
from tqdm import tqdm
from typing import Iterator, List, Dict, Set, Optional, Tuple, Iterable
from allennlp.data.fields import Field,TextField, SequenceLabelField,LabelField, ArrayField, AdjacencyField
from allennlp.data import Instance
from allennlp.data.tokenizers import Token
import numpy as np
# from feature1 import getChunk
# from feature1 import getHeadwords
from allennlp.data.token_indexers import SingleIdTokenIndexer, PosTagIndexer, DepLabelIndexer, NerTagIndexer
import spacy


#sklearn
from sklearn.feature_extraction.text import CountVectorizer

from allennlp.data.fields import TextField, MetadataField, ArrayField

nlpmodel = spacy.load("en_core_web_sm")
class SensableReader(DatasetReader):
    def __init__(self, lazy=False) -> None:
        super().__init__(lazy=lazy)

    @classmethod
    def update(cls, key):

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
            return update_field
        return up

    @classmethod
    def update_each(cls, func):
        def update_field(self_, fields, *args, **kwargs):
            for key, field in func(self_, fields, *args, **kwargs):
                try:
                    if field is not None:
                        fields[key] = field
                except KeyError:
                    pass
            return fields
        return update_field



class SpRLReader(DatasetReader):
    label_names = {'LANDMARK', 'TRAJECTOR'}
    relation_names = {'region', 'direction', 'distance'}

    def __init__(self) -> None:
        super().__init__(lazy=False)
        # 'tokens' could be just any name, and I don't know where it is need again
        # checkout modules used in word2vec, they need this name there

        self.token_indexers = {'phrase': SingleIdTokenIndexer('phrase')}

    @SensableReader.update('sentence')
    def update_sentence(
            self,
            fields: Dict,
            sentence: List[str]
    ) -> Dict:

        return TextField([Token(word) for word in sentence], self.token_indexers)

    @SensableReader.update_each
    def update_labels(
            self,
            fields: Dict,
            labels: List[str]
    ) -> Dict:

        # {'LANDMARK', 'TRAJECTOR', 'NONE'}
        for label_name in self.label_names:
            yield label_name, SequenceLabelField(
                [str(label == label_name) for label in labels],
                fields['sentence'])

    @SensableReader.update_each
    def update_relations(
            self,
            fields: Dict,
            relation_indices: List[Tuple[int, int]],
            relation_labels: Optional[List[str]] = None
    ) -> Dict:

        # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}
        if relation_labels is None:
            # giving none for label because user do not want label
            pass

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


    def to_instance(
            self,
            sentence: List[str],
            labels: Optional[List[str]],
            relations: Optional[List[Tuple[str, Tuple[int, tuple], Tuple[int, tuple]]]] = None
    ) -> Instance:


        fields = {}
        self.update_sentence(fields, sentence)

        if labels is not None:

            self.update_labels(fields, labels)

        if relations is not None:
            # {'Live_In', 'OrgBased_In', 'Located_In', 'Work_For'}
            relation_indices = []
            relation_labels = []
            for rel in relations:
                src_index = rel[1][0]
                dst_index = rel[2][0]
                if (src_index, dst_index) not in relation_indices:
                    relation_indices.append((src_index, dst_index))
                relation_labels.append(rel[0])


            self.update_relations(fields, relation_indices, relation_labels)


        return Instance(fields)



    def _read(
        self,
        file_path: str
    ) -> Iterable[Instance]:

        phrase_list=self.parseSprlXML(file_path)

        #改
        phrases,relations=self.getCorpus_without_none(phrase_list)# to add these two lines to process xml data

        #phrases=self.getCorpus_with_none(self.chunk_generation(phrase_list))

        for (sentence, labels),relation in zip(phrases,relations):

            yield self.to_instance(sentence,labels,relation)

        # for (phrase, label)in phrases:
        #      yield self.to_instance(phrase, label)

    def parseSprlXML(self,sprlxmlfile):

        # parse the xml tree object

        sprlXMLTree = ET.parse(sprlxmlfile)

        # get root of the xml tree
        sprlXMLRoot = sprlXMLTree.getroot()

        sentences_list = []

        # iterate news items
        type_list=[]
        for sentenceItem in sprlXMLRoot.findall('./SCENE/SENTENCE'):
             landmark = []
             trajector=[]
             relationship=[]
             id_text_list = {}
             id_label_list={}
             sentence_dic = {}
             sentence_dic["id"] = sentenceItem.attrib["id"]

             # iterate child elements of item

             for child in sentenceItem:


                 if child.tag == 'TEXT':
                     sentence_dic[child.tag] = child.text
                 if child.tag == 'LANDMARK' :
                    if "text" in child.attrib:
                         landmark.append(child.attrib['text'])
                         sentence_dic[child.tag] = landmark
                         id_text_list[child.attrib['id']]=child.attrib['text']
                         id_label_list[child.attrib['id']] = child.tag


                 if child.tag == 'TRAJECTOR':
                     if "text" in child.attrib:
                         trajector.append(child.attrib['text'])
                         sentence_dic[child.tag] = trajector
                         id_text_list[child.attrib['id']] = child.attrib['text']
                         id_label_list[child.attrib['id']] = child.tag


                 if child.tag == "RELATION":
                    if child.attrib['general_type'] not in type_list:
                        type_list.append(child.attrib['general_type'])

                    try:


                     if 'general_type' in child.attrib:
                         # if child.attrib['general_type'] not in type_list:
                         #     type_list.append(child.attrib['general_type'])
                         each_relationship=((child.attrib['general_type'],(list(id_text_list.values()).index(id_text_list[child.attrib['landmark_id']]),(id_text_list[child.attrib['landmark_id']],id_label_list[child.attrib['landmark_id']])),
                                             (list(id_text_list.values()).index(id_text_list[child.attrib['trajector_id']]),(id_text_list[child.attrib['trajector_id']],id_label_list[child.attrib['trajector_id']]))))

                         relationship.append(each_relationship)
                         sentence_dic[child.tag]=relationship
                    except:KeyError





             sentences_list.append(sentence_dic)


        # create empty dataform for sentences
        #sentences_df = pd.DataFrame(sentences_list)
        return sentences_list



    def getCorpus_with_none(self,sentences_list):


        output = list(self.label_names)

        final_corpus_list = []
        #
        for sents in sentences_list:


            phrase_list = []
            label_list = []
            corpus_list = []
            try:

                for land in sents[output[0]]:
                    phrase_list.append(land)
                    label_list.append(output[0])
                for traj in sents[output[1]]:
                    phrase_list.append(traj)
                    label_list.append(output[1])
                for none in sents[output[2]]:
                    phrase_list.append(none)
                    label_list.append(output[2])

            except:
                KeyError
            if len(phrase_list):

                corpus_list.append(phrase_list)
                corpus_list.append(label_list)
                final_corpus_list.append(corpus_list)


        return final_corpus_list




    def getCorpus_without_none(self,sentences_list):



        output = list(self.label_names)
        final_corpus_list=[]
        final_relationship=[]

        for sents in sentences_list:
            try:
                   final_relationship.append(sents['RELATION'])
            except:continue

            phrase_list=[]
            label_list=[]
            corpus_list=[]
            try:

                for land in sents[output[0]]:
                   phrase_list.append(land)
                   label_list.append(output[0])
                for traj in sents[output[1]]:
                   phrase_list.append(traj)
                   label_list.append(output[1])


            except:KeyError

            corpus_list.append(phrase_list)
            corpus_list.append(label_list)
            final_corpus_list.append(corpus_list)






        return  final_corpus_list,final_relationship





    #here just a example how to generate phrase from sentence

    def chunk_generation(self,phraselist):
        new_phraselist=[]

        for phrase in phraselist:
         try:
               chunklist=self.getChunk(phrase['TEXT'])
               tag1list=phrase['LANDMARK']
               tag2list=phrase['TRAJECTOR']
               phrase['NONE']=list()
               labelnone=phrase['NONE']

               tag1_headword=[]
               tag2_headword=[]

               for tag1 in tag1list:
                   tag1_headword.append(self.getHeadwords(tag1))
               for tag2 in tag2list:
                   tag2_headword.append(self.getHeadwords(tag2))

               for chunk in chunklist:
                        if chunk in tag1list or chunk in tag2list:
                            continue
                        if self.getHeadwords(chunk) in tag1_headword:
                            tag1list.append(chunk)
                        if self.getHeadwords(chunk) in tag2_headword:
                            tag2list.append(chunk)
                        else:
                            labelnone.append(chunk)
               new_phraselist.append(phrase)

         except:KeyError

        return new_phraselist

    def getChunk(self,sentence):
        pre_chunk = nlpmodel(sentence)
        new_chunk = []
        for chunk in pre_chunk.noun_chunks:
            new_chunk.append(chunk.text)
        return new_chunk

    def getHeadwords(self,phrase):
        docs = nlpmodel(phrase)

        for doc in docs.noun_chunks:
            return str(doc.root.text)
















#sp=SpRLReader()
#sp.parseSprlXML('data/newSprl2017_all.xml')
#sp.getCorpus_without_none(sp.parseSprlXML('data/newSprl2017_all.xml'))
#sp.read('data/newSprl2017_all.xml')

#sp._read(sp.getCorpus_with_none(sp.chunk_generation(sp.parseSprlXML('data/newSprl2017_all.xml'))))
#a=sp.parseSprlXML('data/newSprl2017_all.xml')

#sp.getCorpus(sp.parseSprlXML('data/newSprl2017_all.xml'))
# print(sentence_list)

# getCorpus(sentence_list)

