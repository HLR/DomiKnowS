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
class SpRLReader(DatasetReader):

    def __init__(self) -> None:
        super().__init__(lazy=False)
        # 'tokens' could be just any name, and I don't know where it is need again
        # checkout modules used in word2vec, they need this name there

        self.token_indexers = {'phrase': SingleIdTokenIndexer('phrase')}

    def update_sentence(
        self,
        fields: Dict,
        phrase: str
    ) -> Dict:

        fields['sentence'] = TextField([Token(phrase)], self.token_indexers)
        return fields



    def update_labels_without_none(
            self,
            fields: Dict,
            labels: str
    ) -> Dict:
        labelsname={"Landmark":["True"],"Trajector":["False"]}
        if labels=="Landmark" :
            labelist1=labelsname['Landmark']
            labelist2=labelsname['Trajector']
            fields["Landmark"] = SequenceLabelField(labelist1, fields['sentence'])
            fields['Trajector']=SequenceLabelField(labelist2,fields['sentence'])
        if labels=="Trajector":
            labelist1 = labelsname['Trajector']
            labelist2 = labelsname['Landmark']
            fields["Landmark"] = SequenceLabelField(labelist1, fields['sentence'])
            fields['Trajector'] = SequenceLabelField(labelist2, fields['sentence'])

        return fields

    def update_labels_with_none(
            self,
            fields: Dict,
            labels: str
    ) -> Dict:

        labelsname = {"label1": ["True"], "label2": ["False"]}
        labelist1=labelsname['label1']
        labelist2=labelsname['label2']
        if labels == "Landmark":
            fields["Landmark"] = SequenceLabelField(labelist1, fields['sentence'])
            fields['Trajector'] = SequenceLabelField(labelist2, fields['sentence'])
            fields['None'] = SequenceLabelField(labelist2, fields['sentence'])

        if labels == "Trajector":
            fields["Landmark"] = SequenceLabelField(labelist2, fields['sentence'])
            fields['Trajector'] = SequenceLabelField(labelist1, fields['sentence'])
            fields['None'] = SequenceLabelField(labelist2, fields['sentence'])
        if labels=="None":
            fields["Landmark"] = SequenceLabelField(labelist2, fields['sentence'])
            fields['Trajector'] = SequenceLabelField(labelist2, fields['sentence'])
            fields['None'] = SequenceLabelField(labelist1, fields['sentence'])

        return fields





    def to_instance(
        self,
        sentence: List[str],
        label: Optional[List[str]],
    ) -> Instance:
        fields = {}
        fields = self.update_sentence(fields, sentence)#tokenization


        if label is not None:
            #改
           #fields = self.update_labels_without_none(fields, label)
           fields = self.update_labels_with_none(fields, label)


        return Instance(fields)

    # def _read(self,file_path):
    #     with open(file_path, "r") as data_file:
    #         logger.info("Reading instances from lines in file at: %s", file_path)
    #         for line_num, line in enumerate(tqdm(data_file.readlines())):
    #             line = line.strip("\n")
    #             print(line)
    #             if not line:
    #                 continue

    def _read(
        self,
        file_path: str
    ) -> Iterable[Instance]:
        phrase_list=self.parseSprlXML(file_path)
        #改
        #phrases=self.getCorpus_without_none(phrase_list)# to add these two lines to process xml data
        phrases=self.getCorpus_with_none(self.chunk_generation(phrase_list))



        for (phrase, label)in phrases:
             yield self.to_instance(phrase, label)



    def parseSprlXML(self,sprlxmlfile):

        # parse the xml tree object

        sprlXMLTree = ET.parse(sprlxmlfile)

        # get root of the xml tree
        sprlXMLRoot = sprlXMLTree.getroot()

        sentences_list = []

        # iterate news items
        for sentenceItem in sprlXMLRoot.findall('./SCENE/SENTENCE'):
             temp_landmark = []
             temp_trajector=[]
             sentence_dic = {}

             sentence_dic["id"] = sentenceItem.attrib["id"]

             # iterate child elements of item
             for child in sentenceItem:


                 if child.tag == 'TEXT':
                     sentence_dic[child.tag] = child.text
                 if child.tag == 'LANDMARK' :
                    if "text" in child.attrib:
                         temp_landmark.append(child.attrib['text'])
                         sentence_dic[child.tag] = temp_landmark
                 if child.tag == 'TRAJECTOR':
                     if "text" in child.attrib:
                         temp_trajector.append(child.attrib['text'])
                         sentence_dic[child.tag] = temp_trajector

                         # if "start" in child.attrib:
                         #     padded_str = ' ' * int(child.attrib["start"]) + child.attrib["text"]
                         #     sentence_dic[child.tag + "padded"] = padded_str

             sentences_list.append(sentence_dic)



        # create empty dataform for sentences
        #sentences_df = pd.DataFrame(sentences_list)
        return sentences_list


#gold label and generate chunk label
    def getCorpus_with_none(self,sentences_list):

        output = ["Landmark", "Trajector","None"]

       # print(sentences_list)
        # Combine landmarks and trajectors phrases and add answers
        landmark_list=[]
        trajector_list=[]
        none_list=[]


        corpus_landmarks={}
        corpus_trajecor={}
        corpus_none={}


        for sents in sentences_list:

            try:
                landmark_list+=sents['LANDMARK']
                trajector_list+=sents['TRAJECTOR']
                none_list+=sents['NONE']
            except:
                KeyError

        corpus_landmarks['Phrase']=landmark_list
        corpus_trajecor['Phrase']=trajector_list
        corpus_none['Phrase']=none_list

        landmark_df=pd.DataFrame(corpus_landmarks)
        trajector_df=pd.DataFrame(corpus_trajecor)
        none_df=pd.DataFrame(corpus_none)


        landmark_df = landmark_df[~landmark_df['Phrase'].isnull()]
        landmark_df['Output'] = landmark_df['Phrase'].apply(lambda x: output[0])

        trajector_df = trajector_df[~trajector_df['Phrase'].isnull()]
        trajector_df['Output'] = trajector_df["Phrase"].apply(lambda x: output[1])

        none_df = none_df[~none_df['Phrase'].isnull()]
        none_df['Output'] = none_df["Phrase"].apply(lambda x: output[2])

        corpus = landmark_df.append(trajector_df)
        corpus=corpus.append(none_df)

        corpus_list = np.array(corpus)  # np.ndarray()
        corpus_list = corpus_list.tolist()  # list

        # corpus.to_csv('data/corpus.txt', sep='\t', index=False)


        return corpus_list



# gold label
    def getCorpus_without_none(self,sentences_list):
        output = ["Landmark", "Trajector"]

       # print(sentences_list)
        # Combine landmarks and trajectors phrases and add answers
        landmark_list=[]
        trajector_list=[]


        corpus_landmarks={}
        corpus_trajecor={}

        for sents in sentences_list:

            try:
                landmark_list+=sents['LANDMARK']
                trajector_list+=sents['TRAJECTOR']

            except:
                KeyError

        corpus_landmarks['Phrase']=landmark_list
        corpus_trajecor['Phrase']=trajector_list

        landmark_df=pd.DataFrame(corpus_landmarks)
        trajector_df=pd.DataFrame(corpus_trajecor)


        landmark_df = landmark_df[~landmark_df['Phrase'].isnull()]
        landmark_df['Output'] = landmark_df['Phrase'].apply(lambda x: output[0])

        trajector_df = trajector_df[~trajector_df['Phrase'].isnull()]
        trajector_df['Output'] = trajector_df["Phrase"].apply(lambda x: output[1])


        corpus = landmark_df.append(trajector_df)

        corpus_list = np.array(corpus)  # np.ndarray()
        corpus_list = corpus_list.tolist()  # list

        # corpus.to_csv('data/corpus.txt', sep='\t', index=False)


        return corpus_list


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
# sp.getCorpus_with_none(sp.chunk_generation(sp.parseSprlXML('data/newSprl2017_all.xml')))
#a=sp.parseSprlXML('data/newSprl2017_all.xml')

#sp.getCorpus(sp.parseSprlXML('data/newSprl2017_all.xml'))
# print(sentence_list)

# getCorpus(sentence_list)

