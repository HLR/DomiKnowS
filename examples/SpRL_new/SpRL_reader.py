import xml.etree.ElementTree as ET
from typing import Dict
from collections import defaultdict
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField,AdjacencyField

from regr.data.allennlp.reader import SensableReader, keep_fields
from typing import Iterator, List, Dict, Set, Optional, Tuple, Iterable
from allennlp.data import Instance
import spacy


#sklearn
from sklearn.feature_extraction.text import CountVectorizer

from allennlp.data.fields import TextField, MetadataField, ArrayField

nlpmodel = spacy.load("en_core_web_sm")


class SpRLReader(SensableReader):


    def __init__(self) -> None:
        super().__init__(lazy=False)

    @field('sentence')
    def update_sentence(
            self,
            fields: Dict,
            raw_sample
    ) -> Dict:

        (sentence,labels), relations = raw_sample
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



    def _read(
        self,
        file_path: str
    ) -> Iterable[Instance]:

        phrase_list=self.parseSprlXML(file_path)
        raw_examples=self.getCorpus_with_none(self.chunk_generation(phrase_list))

        for raw_sample in raw_examples:
            yield self._to_instance(raw_sample)

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
             spatialindicator=[]
             relationship=[]
             id_text_list = {}
             id_label_list={}
             sentence_dic = {}
             sentence_dic["id"] = sentenceItem.attrib["id"]

             # iterate child elements of item

             for child in sentenceItem:


                 if child.tag == 'TEXT':
                     sentence_dic[child.tag] = child.text

                 if child.tag == 'SPATIALINDICATOR' :
                    if "text" in child.attrib:
                         spatialindicator.append(child.attrib['text'])
                         sentence_dic[child.tag] = spatialindicator
                         id_text_list[child.attrib['id']]=child.attrib['text']
                         id_label_list[child.attrib['id']] = child.tag

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
                         each_relationship=((child.attrib['general_type'],[(id_text_list[child.attrib['landmark_id']],id_label_list[child.attrib['landmark_id']])],
                                             [(id_text_list[child.attrib['trajector_id']],id_label_list[child.attrib['trajector_id']])],
                                             [(id_text_list[child.attrib['spatial_indicator_id']],id_label_list[child.attrib['spatial_indicator_id']])]))

                         relationship.append(each_relationship)
                         sentence_dic[child.tag]=relationship
                    except:KeyError



             sentences_list.append(sentence_dic)


        # create empty dataform for sentences
        #sentences_df = pd.DataFrame(sentences_list)
        return sentences_list



    def getCorpus_with_none(self,sentences_list):

        output = list(self.label_names)
        final_relationship=[]

        for sents in sentences_list:
            phrase_list = []
            label_list = []
            temp_relation = []

            if "RELATION" not in sents.keys():
                continue

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

                for sptialindicator in sents[output[3]]:
                    phrase_list.append(sptialindicator)
                    label_list.append(output[3])

            except:
                KeyError


            for rel in sents['RELATION']:
                    temp_temp_element=[]
                    temp_temp_element.append(rel[0])

                    for element in rel[1:]:

                       element.insert(0,phrase_list.index(element[0][0]))
                       temp_temp_element.append(tuple(element))

                    temp_relation.append(tuple(temp_temp_element))

            relation_tuple=((phrase_list, label_list),temp_relation)
            final_relationship.append(relation_tuple)



        return final_relationship

    def chunk_generation(self, phraselist):

        new_phraselist = []

        for phrase in phraselist:
            try:

                chunklist = self.getChunk(phrase['TEXT'])
                tag1list = phrase['LANDMARK']
                tag2list = phrase['TRAJECTOR']
                tag3list = phrase['SPATIALINDICATOR']

                phrase['NONE'] = list()
                labelnone = phrase['NONE']

                tag1_headword = []
                tag2_headword = []
                tag3_headword = []

                for tag1 in tag1list:
                    tag1_headword.append(self.getHeadwords(tag1))
                for tag2 in tag2list:
                    tag2_headword.append(self.getHeadwords(tag2))
                for tag3 in tag3list:
                    tag3_headword.append(self.getHeadwords(tag3))

                chunklist = list(set(chunklist))

                for chunk in chunklist:
                    if chunk in tag1list or chunk in tag2list or chunk in tag3list:
                        continue
                    elif self.getHeadwords(chunk) in tag1_headword:
                        tag1list.append(chunk)

                    elif self.getHeadwords(chunk) in tag2_headword:
                        tag2list.append(chunk)

                    elif self.getHeadwords(chunk) in tag3_headword:
                        tag3list.append(chunk)

                    else:
                        labelnone.append(chunk)

                new_phraselist.append(phrase)



            except:
                KeyError

        return new_phraselist


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




@keep_fields('sentence')
class SpRLBinaryReader(SpRLReader):
    label_names = {'LANDMARK', 'TRAJECTOR', 'NONE', 'SPATIALINDICATOR'}
    relation_names = {'region', 'direction', 'distance'}

    @fields
    def update_labels(
        self,
        fields: Dict,
        raw_sample
    ) -> Dict:
        # {'Other', 'Loc', 'Peop', 'Org', 'O'}
        (sentence, labels), relations = raw_sample
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
        (sentence, labels), relations = raw_sample
        if relations is None:
            return None
        relation_indices = []
        relation_labels = []
        for rel in relations:
            src_index = rel[1][0]
           # mid_index=rel[2][0]
            dst_index = rel[3][0]
            if (src_index, dst_index) not in relation_indices:
                relation_indices.append((src_index,dst_index))

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
class SpRLSensorReader(SpRLBinaryReader):
    pass








#sp.parseSprlXML('data/newSprl2017_all.xml')
#sp.getCorpus_without_none(sp.parseSprlXML('data/newSprl2017_all.xml'))


#sp._read(sp.getCorpus_with_none(sp.chunk_generation(sp.parseSprlXML('data/newSprl2017_all.xml'))))
#a=sp.parseSprlXML('data/newSprl2017_all.xml')

#sp.getCorpus(sp.parseSprlXML('data/newSprl2017_all.xml'))
# print(sentence_list)

# getCorpus(sentence_list)

