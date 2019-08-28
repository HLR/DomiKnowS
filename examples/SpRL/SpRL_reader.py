import xml.etree.ElementTree as ET

import pandas as pd

#sklearn
from sklearn.feature_extraction.text import CountVectorizer

def parseSprlXML(sprlxmlfile):

    # parse the xml tree object
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
                if "lemma.py" in child.attrib:
                     sentence_dic[child.tag] = child.attrib["lemma.py"]
                     if "start" in child.attrib:
                         padded_str = ' ' * int(child.attrib["start"]) + child.attrib["lemma.py"]
                         sentence_dic[child.tag + "padded"] = padded_str

         sentences_list.append(sentence_dic)

    # create empty dataform for sentences
    sentences_df = pd.DataFrame(sentences_list)
    return sentences_df
#
#
# output = (
# (owlReasoning.mySaulSpatialOnto.lm, "Landmark", [1, 0]), (owlReasoning.mySaulSpatialOnto.tr, "Trajector", [0, 1]))
#
#


output = {'Landmar' : [1, 0], 'Trajector' : [0, 1]}


def getCorpus(sentences_df):

    print(sentences_df)
    # Combine landmarks and trajectors phrases and add answers
    corpus_landmarks = pd.DataFrame(sentences_df['LANDMARK']).rename(index=str, columns={"LANDMARK": "Phrase"})
    corpus_landmarks = corpus_landmarks[~corpus_landmarks['Phrase'].isnull()]
    corpus_landmarks['output'] = corpus_landmarks['Phrase'].apply(lambda x: output[0][2])

    corpus_trajectors = pd.DataFrame(sentences_df['TRAJECTOR']).rename(index=str, columns={"TRAJECTOR": "Phrase"})
    corpus_trajectors = corpus_trajectors[~corpus_trajectors['Phrase'].isnull()]
    corpus_trajectors['output'] = corpus_trajectors["Phrase"].apply(lambda x: output[1][2])

    corpus = corpus_landmarks.append(corpus_trajectors)

    print('corpus', type(corpus), len(corpus), corpus.columns)

    # Find distinct words in combined list of phrases
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus["Phrase"].values.astype('U'))

    # print('vectorizer.get_feature_names() ', vectorizer.get_feature_names())

    # Add feature vector for each phrase
    corpus["Phrase_asList"] = corpus['Phrase'].apply(lambda x: [x])
    corpus["Feature_Words"] = corpus['Phrase_asList'].apply(lambda x: vectorizer.transform(x))

    return corpus

sentence_df = parseSprlXML('data/newSprl2017_all.xml')

print(getCorpus(sentence_df).head(60))
