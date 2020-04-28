import xml.etree.ElementTree as ET
from typing import Dict
from collections import defaultdict
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, SequenceLabelField, AdjacencyField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from regr.data.allennlp.reader import SensableReader, keep_keys
#from regr.data.allennlp.reader import SensableReader, keep_keys
from typing import Iterator, List, Dict, Set, Optional, Tuple, Iterable
from allennlp.data import Instance
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.checks import ConfigurationError
import itertools
from itertools import product
import spacy
import torch
from featureIndexer import PosTaggerIndexer, LemmaIndexer, DependencyIndexer, HeadwordIndexer, PhrasePosIndexer
from feature import DataFeature_for_sentence, DataFeature_for_span
from dictionaries import dictionary


# sklearn
from allennlp.data.fields import TextField, MetadataField, ArrayField

nlpmodel = spacy.load("en_core_web_sm")
spatial_dict = []
with open('data/spatial_dic.txt') as f_sp:
    for each_sp_word in f_sp:
        spatial_dict.append(each_sp_word.strip())


def span_not_overlap(triplet):
    arg1, arg2, arg3 = triplet

    if (arg2.start <= arg1.start and arg1.start < arg2.end) or (arg3.start <= arg1.start and arg1.start < arg3.end) or \
        (arg3.start <= arg2.start and arg2.start < arg3.end) or (arg1.start <= arg2.start and arg2.start < arg1.end) or \
        (arg1.start <= arg3.start and arg3.start < arg1.end) or (arg2.start <= arg3.start and arg3.start < arg2.end):
        return False
    else:
        return True

def span_overlap(span1, span2):
    if (span1.start <= span2.start and span2.start < span1.end) or (span2.start <= span1.start and span1.start < span2.end):
        return True
    else:
        return False

def span_overlap_any(span, spanlist):
    for span2 in spanlist:
        if span_overlap(span, span2):
            return True
    return False

class SpRLReader(SensableReader):
    label_names = ['LANDMARK', 'TRAJECTOR', 'SPATIALINDICATOR', 'NONE']
    relation_names = ['region', 'direction', 'distance', 'relation_none']
    triplet_names = ['is_triplet', 'is_not_triplet']

    def __init__(self) -> None:
        super().__init__(lazy=False)

    @cls.tokens('sentence')
    def update_sentence(
            self,
            fields: Dict,
            raw_sample
    ) -> Dict:
        (phrases, labels), relations, sentence = raw_sample
        # newtoken = []
        # for phrase in phrases:
        #     #token = DataFeature(sentence, phrase).getPhraseTokens()
        #     # phrase.set_extension("lemma_", default=False, force=True)
        #     # phrase.set_extension('pos_', default=False, force=True)
        #     # phrase.set_extension('tag_', default=False, force=True)
        #     # phrase.set_extension('dep_', default=False, force=True)
        #     # phrase.set_extension('headword_', default=False, force=True)
        #     # phrase.set_extension('phrasepos_', default=False, force=True)
        #     # phrase.set_extension('sentence_', default=False, force=True)
        #     df = DataFeature_for_span(phrase)

        #     # token._.set('lemma_', df.getLemma())
        #     # token._.set('pos_', df.getPos())
        #     # token._.set('tag_', df.getTag())
        #     # token._.set('dep_', df.getDenpendency())
        #     # token._.set('headword_', df.getHeadword())
        #     # token._.set('phrasepos_', df.getPhrasepos())
        #     # token._.set('sentence_', df.getSentence())
        #     newtoken.append(df)
        return phrases

    @cls.textfield('word')
    def update_sentence_word(
            self,
            fields,
            tokens
    ) -> Field:
        tokens = [Token(each_df.span.text) for each_df in tokens]
        indexers = {'word': SingleIdTokenIndexer(namespace='word', lowercase_tokens=True)}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('pos_tag')
    def update_sentence_pos(
            self,
            fields,
            tokens
    ) -> Field:
        tokens = [Token(each_df.pos_) for each_df in tokens]
        indexers = {'pos_tag': SingleIdTokenIndexer(namespace='pos_tag', lowercase_tokens=True)}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('lemma_tag')
    def update_sentence_lemma(
            self,
            fields,
            tokens
    ) -> Field:
        tokens = [Token(each_df.lemma_) for each_df in tokens]
        indexers = {'lemma_tag': SingleIdTokenIndexer(namespace='lemma_tag', lowercase_tokens=True)}
        textfield = TextField(tokens, indexers)
        return textfield

    #
    @cls.textfield('dep_tag')
    def update_sentence_dep(
            self,
            fields,
            tokens
    ) -> Field:
        tokens = [Token(each_df.dep_) for each_df in tokens]
        indexers = {'dep_tag': SingleIdTokenIndexer(namespace='dep_tag', lowercase_tokens=True)}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('headword_tag')
    def update_sentence_dep(
            self,
            fields,
            tokens
    ) -> Field:
        tokens = [Token(each_df.headword_) for each_df in tokens]
        indexers = {'headword_tag': SingleIdTokenIndexer(namespace='headword_tag', lowercase_tokens=True)}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('phrasepos_tag')
    def update_sentence_dep(
            self,
            fields,
            tokens
    ) -> Field:
        tokens = [Token(each_df.phrasepos_) for each_df in tokens]
        indexers = {'phrasepos_tag': SingleIdTokenIndexer(namespace='phrasepos_tag', lowercase_tokens=True)}
        textfield = TextField(tokens, indexers)
        return textfield

    @cls.textfield('triplet_feature1')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))

        # just a dummy feature
        def dummy_feature(lm, tr, sp):
            return '::'.join((tokens[tr].lower_, tokens[sp].lower_, tokens[lm].lower_))

        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature1': SingleIdTokenIndexer(namespace='triplet_feature1')}
        textfield = TextField(encap_tokens, indexers)
        #import pdb; pdb.set_trace()
        return textfield

    @cls.textfield('triplet_feature2')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)
        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))

        # just a dummy feature
        def dummy_feature(lm, tr, sp):
            if tokens[sp].lower_ in spatial_dict or tokens[sp].lower_ in dictionary.prepositions:
                return "true"
            else:
                return "false"

        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature2': SingleIdTokenIndexer(namespace='triplet_feature2')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield

    @cls.textfield('triplet_feature3')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))

        # just a dummy feature
        def dummy_feature(lm, tr, sp):

            return '::'.join((tokens[tr].headword_.lower(), tokens[lm].headword_.lower(), tokens[sp].span.text.lower()))

        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature3': SingleIdTokenIndexer(namespace='triplet_feature3')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield

    @cls.textfield('triplet_feature4')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)
        sentence = tokens[0].span.doc.text

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))

        # just a dummy feature
        def dummy_feature(lm, tr, sp):
            entity1 = tokens[tr].headword_.lower()
            entity2 = tokens[sp].headword_.lower()
            entity1 = entity1.split(' ')[0]
            entity2 = entity2.split(' ')[0]
            shortestdepedenceylist = DataFeature_for_sentence(sentence=sentence).getShortestDependencyPath(entity1, entity2)
            shortestdepedenceylist.insert(-1, tokens[sp].span.text)
            return "::".join(shortestdepedenceylist)

        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature4': SingleIdTokenIndexer(namespace='triplet_feature4')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield

    @cls.textfield('triplet_feature5')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)
        sentence = tokens[0].span.doc.text

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))

        # just a dummy feature
        def dummy_feature(lm, tr, sp):
            entity1 = tokens[tr].headword_.lower()
            entity2 = tokens[sp].headword_.lower()
            entity1 = entity1.split(' ')[0]
            entity2 = entity2.split(' ')[0]
            shortestdepedenceylist = DataFeature_for_sentence(sentence=sentence).getShortestDependencyPath(entity1, entity2)
            shortestdepedenceylist.insert(-1, tokens[sp].span.text)
            return "::".join(shortestdepedenceylist)

        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature5': SingleIdTokenIndexer(namespace='triplet_feature5')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield

    def raw_read(
            self,
            file_path: str
    ) -> Iterable[Instance]:

        phrase_list = self.parseSprlXML(file_path)
       # raw_examples = self.getCorpus(self.negative_entity_generation_for(phrase_list))
        raw_examples = self.getCorpus(self.entity_candidate_generation_for_train(phrase_list))
        for raw_sample in raw_examples:
            yield raw_sample


    def parseSprlXML(self, sprlxmlfile):

        # parse the xml tree object

        sprlXMLTree = ET.parse(sprlxmlfile)

        # get root of the xml tree
        sprlXMLRoot = sprlXMLTree.getroot()

        sentences_list = []

        # iterate news items
        type_list = []
        def getspan(sentence, span_text, start_index):
            token_number_of_span = len(DataFeature_for_sentence(span_text).parse_sentence) 
            doc = DataFeature_for_sentence(sentence).parse_sentence
            for each_token in doc:
                 if str(each_token.idx) == start_index:
                     return DataFeature_for_span(doc[each_token.i:each_token.i + token_number_of_span])
                     

                     
        for sentenceItem in sprlXMLRoot.findall('./SCENE/SENTENCE'):
            landmark = []
            trajector = []
            spatialindicator = []
            relationship = []
            id_text_list = {}
            id_label_list = {}
            sentence_dic = {}
            sentence_dic["id"] = sentenceItem.attrib["id"]

            # iterate child elements of item

            for child in sentenceItem:

                if child.tag == 'TEXT':
                    chunklist = DataFeature_for_sentence(child.text).getChunks()
                    sentence_dic[child.tag] = child.text
                    sentence_dic["phrases"] = chunklist

                if child.tag == 'SPATIALINDICATOR':
                    if "text" in child.attrib: 
                        child.attrib['text'] = getspan(sentence_dic['TEXT'], child.attrib.get('text'), child.attrib['start'])
                        spatialindicator.append((child.attrib['text'], child.attrib['start']))
                        sentence_dic[child.tag] = spatialindicator
                        id_text_list[child.attrib['id']] = child.attrib['text']
                        id_label_list[child.attrib['id']] = child.tag

                if child.tag == 'LANDMARK':
                    if "text" in child.attrib:
                        child.attrib['text'] = getspan(sentence_dic['TEXT'], child.attrib.get('text'), child.attrib['start'])
                        landmark.append((child.attrib['text'], child.attrib['start']))
                        sentence_dic[child.tag] = landmark
                        id_text_list[child.attrib['id']] = child.attrib['text']
                        id_label_list[child.attrib['id']] = child.tag

                if child.tag == 'TRAJECTOR':
                    if "text" in child.attrib:
                        child.attrib['text'] = getspan(sentence_dic['TEXT'], child.attrib.get('text'), child.attrib['start'])
                        trajector.append((child.attrib['text'], child.attrib['start']))
                        sentence_dic[child.tag] = trajector
                        id_text_list[child.attrib['id']] = child.attrib['text']
                        id_label_list[child.attrib['id']] = child.tag

                if child.tag == "RELATION":
                    if child.attrib['general_type'] not in type_list:
                        type_list.append(child.attrib['general_type'])
                    #
                    try:

                        if 'general_type' in child.attrib:
                            # if child.attrib['general_type'] not in type_list:
                            #     type_list.append(child.attrib['general_type'])
                            each_relationship = ((child.attrib['general_type'],
                                                  [(id_text_list[child.attrib['landmark_id']],
                                                    id_label_list[child.attrib['landmark_id']])],
                                                  [(id_text_list[child.attrib['trajector_id']],
                                                    id_label_list[child.attrib['trajector_id']])],
                                                  [(id_text_list[child.attrib['spatial_indicator_id']],
                                                    id_label_list[child.attrib['spatial_indicator_id']])]))

                            relationship.append(each_relationship)
                            sentence_dic[child.tag] = relationship
                    except:
                        KeyError   
            sentences_list.append(sentence_dic)

        # create empty dataform for sentences
        # sentences_df = pd.DataFrame(sentences_list)
    
        
        return sentences_list

    def getCorpus(self, sentences_list):

        output = self.label_names
        final_relationship = []

        for sents in sentences_list:
            sentence = sents['TEXT']
            phrase_list = [''] * len(sentence)
            label_list = [''] * len(sentence)
            temp_relation = []

            if "RELATION" not in sents.keys():
                continue
            try:
                for land in sents[output[0]]:
                    phrase_list.insert(int(land[1]), land[0])
                    label_list.insert(int(land[1]), output[0])

                for traj in sents[output[1]]:
                    phrase_list.insert(int(traj[1]), traj[0])
                    label_list.insert(int(traj[1]), output[1])

                for sptialindicator in sents[output[2]]:
                    phrase_list.insert(int(sptialindicator[1]), sptialindicator[0])
                    label_list.insert(int(sptialindicator[1]), output[2])

                for none in sents[output[3]]:
                    phrase_list.insert(int(none[1]), none[0])
                    label_list.insert(int(none[1]), output[3])

            except:
                KeyError

            phrase_list = [phrase for phrase in phrase_list if phrase]
            label_list = [label for label in label_list if label]

            for rel in sents['RELATION']:
                temp_temp_element = []
                temp_temp_element.append(rel[0])

                for element in rel[1:]:
                    element.insert(0, phrase_list.index(element[0][0]))
                    temp_temp_element.append(tuple(element))

                temp_relation.append(tuple(temp_temp_element))

            relation_tuple = ((phrase_list, label_list), temp_relation, sents['TEXT'])
            
            relation_tuple = self.negative_relation_generation_for_test(relation_tuple, sents['phrases'])
            final_relationship.append(relation_tuple)
            
            
        # for i in final_relationship:
        #     numnum=0
        #     for m in i[0][0]:
        #         print(f"{m},{i[0][1][numnum]}")
        #         numnum +=1
        #     print('\n', end='')
       # print(self.negative_relation_generation_for_train(final_relationship)[0])
        
        return final_relationship
    
    def landmark_candidate_generation(self, phrase):
        landmark_pos_dic = ["PRON", "NOUN", "DET", "ADJ", "NUM"]
        landmark_pos = phrase.phrasepos_
        if landmark_pos in landmark_pos_dic:
            return phrase


    def trajector_candidate_generation(self, phrase):
        trajector_pos_dic = ["NOUN", "PRON", "ADJ","DET", "NUM", "VERB"]
        trajector_pos = phrase.phrasepos_
        if trajector_pos in trajector_pos_dic:
            return phrase

    def spatial_indicator_candidate_generation(self, phrase):
        spatial_pos = phrase.phrasepos_
        if phrase.span.text in spatial_dict or spatial_pos == "ADP":
            return phrase

    def negative_relation_generation_for_test(self, relation_tuple, generated_phrase_list):
        landmark_candidate_list = []
        trajector_candidate_list = []
        spatial_indicator_candidate_list = []
        for each_phrase in generated_phrase_list:
            if self.landmark_candidate_generation(each_phrase):
                landmark_candidate_list.append(self.landmark_candidate_generation(each_phrase))
            if self.trajector_candidate_generation(each_phrase):
                trajector_candidate_list.append(self.trajector_candidate_generation(each_phrase))
            if self.spatial_indicator_candidate_generation(each_phrase):
                spatial_indicator_candidate_list.append(self.spatial_indicator_candidate_generation(each_phrase))

        raw_triplet_candidates = list(product(landmark_candidate_list, trajector_candidate_list, spatial_indicator_candidate_list))
        new_triplet_candidates = []
        new_relation_triplet = []
        for arg1, arg2, arg3 in filter(span_not_overlap, raw_triplet_candidates):
            for each_relation in relation_tuple[1]:
                gold_arg1, gold_arg2, gold_arg3 = each_relation[1:]
                if span_overlap(arg1, gold_arg1[1][0]) and span_overlap(arg2, gold_arg2[1][0]) and span_overlap(arg3, gold_arg3[1][0]):
                    new_relation_triplet.append((each_relation[0], (relation_tuple[0][0].index(arg1), (arg1, "LANDMARK")), (relation_tuple[0][0].index(arg2), (arg2, "TRAJECTOR")), (relation_tuple[0][0].index(arg3), (arg3, "SPATIALINDICATOR"))))
                else:
                    new_relation_triplet.append(("relation_none", (relation_tuple[0][0].index(arg1), (arg1, "LANDMARK")), (relation_tuple[0][0].index(arg2), (arg2, "TRAJECTOR")), (relation_tuple[0][0].index(arg3), (arg3, "SPATIALINDICATOR"))))

        relation_tuple[1].extend(new_relation_triplet)

        return relation_tuple
        
    def negative_relation_generation_for_train(self, phrase_relation_list):
        for phrase_relation in phrase_relation_list:
            list_indices = list(range(len(phrase_relation[0][0])))
            list_indices = list(itertools.permutations(list_indices, 3))

            positive_list = []
            positive_text_list = []
            for rel in phrase_relation[1]:
                src_index = rel[1][0]
                src_text_index = rel[1][1][0]
                mid_index = rel[2][0]
                mid_text_index = rel[2][1][0]
                dst_index = rel[3][0]
                dst_text_index = rel[3][1][0]
                positive_list.append((src_index, mid_index, dst_index))
                positive_text_list.append((rel[0], (src_text_index, mid_text_index, dst_text_index)))

            label_indices_list = phrase_relation[0][1]
            text_list = phrase_relation[0][0]
            label_name_list = self.label_names

            for indice in list_indices:
                new_relationship = ()
                if indice in positive_list:
                    continue
                elif label_indices_list[indice[0]] == label_name_list[0] and \
                        label_indices_list[indice[1]] == label_name_list[1] and \
                        label_indices_list[indice[2]] == label_name_list[2]:
                    src_head = DataFeature("", text_list[indice[0]]).getHeadword()
                    mid_head = DataFeature("", text_list[indice[1]]).getHeadword()
                    dst_head = DataFeature("", text_list[indice[2]]).getHeadword()
                    label_set = set([src_head, mid_head, dst_head])
                    for positive_text in positive_text_list:
                        positive_src = DataFeature("", positive_text[1][0]).getHeadword()
                        positive_mid = DataFeature("", positive_text[1][1]).getHeadword()
                        positive_dst = DataFeature("", positive_text[1][2]).getHeadword()
                        positive_set = set([positive_src, positive_mid, positive_dst])

                        if label_set == positive_set:

                            new_relationship = ((positive_text[0],
                                                 (indice[0], (text_list[indice[0]], label_indices_list[indice[0]])),
                                                 (indice[1], (text_list[indice[1]], label_indices_list[indice[1]])),
                                                 (indice[2], (text_list[indice[2]], label_indices_list[indice[2]]))))
                            break
                        elif len(label_set.intersection(positive_set)) == 1 or len(
                                label_set.intersection(positive_set)) == 0:
                            break
                        else:
                            new_relationship = (("relation_none",
                                                 (indice[0], (text_list[indice[0]], label_indices_list[indice[0]])),
                                                 (indice[1], (text_list[indice[1]], label_indices_list[indice[1]])),
                                                 (indice[2], (text_list[indice[2]], label_indices_list[indice[2]]))))
                            break

                if new_relationship:
                    phrase_relation[1].append(new_relationship)
        # for i in phrase_relation_list:
        #     print(i)
        return phrase_relation_list

    def entity_candidate_generation_for_train(self, phraselist):
        gold_entity_landmark = 0
        gold_entity_trajector = 0
        gold_entity_spatialindicator = 0
        positive_entity_landmark = 0
        positive_entity_trajector = 0
        positive_entity_spatialindicator = 0
        negative_entity = 0

        new_phraselist = []

        for each_sentence in phraselist:
            chunklist = each_sentence.get('phrases')
            landmarklist = each_sentence.get('LANDMARK', '')
            trajectorlist = each_sentence.get('TRAJECTOR', '')
            spatialindicatorlist = each_sentence.get('SPATIALINDICATOR', '')
            

            tag1list = [land[0] for land in landmarklist]                            
            tag2list = [traj[0] for traj in trajectorlist]
            tag3list = [spat[0] for spat in spatialindicatorlist]

            gold_entity_landmark += len(tag1list)
            gold_entity_trajector += len(tag2list)
            gold_entity_spatialindicator += len(tag3list)

            each_sentence['NONE'] = list()
            labelnone = each_sentence['NONE']

            tag1_headword = []
            tag2_headword = []
            # tag3_headword = []

            for tag1 in tag1list:
                tag1_headword.append(tag1.headword_)
            for tag2 in tag2list:
                tag2_headword.append(tag2.headword_)
            # for tag3 in tag3list:
            #     tag3_headword.append(tag3.headword_)
            chunklist = list(set(chunklist))

            for chunk in chunklist:
                if chunk in tag1list or chunk in tag2list or chunk in tag3list:
                    continue
                elif chunk.headword_ in tag1_headword:
                    landmarklist.append((chunk, chunk.start))
                elif chunk.headword_ in tag2_headword:
                    trajectorlist.append((chunk, chunk.start))
                elif span_overlap_any(chunk, tag3list):
                    spatialindicatorlist.append((chunk, chunk.start))
                else:
                    labelnone.append((chunk, chunk.start))

            positive_entity_landmark += len(landmarklist)
            positive_entity_trajector += len(trajectorlist)
            positive_entity_spatialindicator += len(spatialindicatorlist)
            negative_entity += len(labelnone)
            new_phraselist.append(each_sentence)

        # print(f" landmark gold number is {gold_entity_landmark}, trajector gold number is {gold_entity_trajector}, spatialindicator gold number is {gold_entity_spatialindicator} , total gold number is {gold_entity_landmark+gold_entity_trajector+gold_entity_spatialindicator}")
        # print(
        #     f" landmark positive number is {positive_entity_landmark}, trajector positive number is {positive_entity_trajector}, spatialindicator positive number is {positive_entity_spatialindicator}, total positive number is {positive_entity_landmark+positive_entity_trajector+positive_entity_spatialindicator} ")
        # print(f" negative entity number is {negative_entity} ")

        return new_phraselist


    def negative_entity_generation(self, phraselist):
        gold_entity_landmark = 0
        gold_entity_trajector = 0
        gold_entity_spatialindicator = 0
        positive_entity_landmark = 0
        positive_entity_trajector = 0
        positive_entity_spatialindicator = 0
        negative_entity = 0

        new_phraselist = []

        for phrase in phraselist:
            try:
                sentence = phrase['TEXT']
                chunklist = DataFeature(sentence, "").getChunks()

                landmarklist = phrase['LANDMARK']
                trajectorlist = phrase['TRAJECTOR']
                spatialindicatorlist = phrase['SPATIALINDICATOR']

                tag1list = [land[0] for land in landmarklist]
                tag2list = [traj[0] for traj in trajectorlist]
                tag3list = [spat[0] for spat in spatialindicatorlist]

                gold_entity_landmark += len(tag1list)
                gold_entity_trajector += len(tag2list)
                gold_entity_spatialindicator += len(tag3list)

                phrase['NONE'] = list()
                labelnone = phrase['NONE']

                tag1_headword = []
                tag2_headword = []
                tag3_headword = []

                for tag1 in tag1list:
                    tag1_headword.append(DataFeature("", tag1).getHeadword())
                for tag2 in tag2list:
                    tag2_headword.append(DataFeature("", tag2).getHeadword())
                for tag3 in tag3list:
                    tag3_headword.append(DataFeature("", tag3).getHeadword())
                chunklist = list(set(chunklist))

                for chunk in chunklist:

                    if chunk.text in tag1list or chunk.text in tag2list or chunk.text in tag3list:
                        continue
                    elif DataFeature("", chunk.text).getHeadword() in tag1_headword and DataFeature("",
                                                                                                    chunk.text).getPhrasepos() != "VERB":

                        landmarklist.append((chunk.text, sentence.find(chunk.text)))

                    elif DataFeature("", chunk.text).getHeadword() in tag2_headword and DataFeature("",
                                                                                                    chunk.text).getPhrasepos() != "VERB":

                        trajectorlist.append((chunk.text, sentence.find(chunk.text)))
                    elif DataFeature("", chunk.text).getHeadword() in tag3_headword:

                        spatialindicatorlist.append((chunk.text, sentence.find(chunk.text)))

                    else:
                        labelnone.append((chunk.text, sentence.find(chunk.text)))

                positive_entity_landmark += len(landmarklist)
                positive_entity_trajector += len(trajectorlist)
                positive_entity_spatialindicator += len(spatialindicatorlist)
                negative_entity += len(labelnone)
                new_phraselist.append(phrase)

            except:
                KeyError

       # print(f" landmark gold number is {gold_entity_landmark}, trajector gold number is {gold_entity_trajector}, spatialindicator gold number is {gold_entity_spatialindicator} , total gold number is {gold_entity_landmark+gold_entity_trajector+gold_entity_spatialindicator}")
       # print(
           # f" landmark positive number is {positive_entity_landmark}, trajector positive number is {positive_entity_trajector}, spatialindicator positive number is {positive_entity_spatialindicator}, total positive number is {positive_entity_landmark+positive_entity_trajector+positive_entity_spatialindicator} ")
       # print(f" negative entity number is {negative_entity} ")

        return new_phraselist


@keep_keys(exclude=['labels', 'relation'])
class SpRLBinaryReader(SpRLReader):

    @cls.fields
    def update_labels(
            self,
            fields: Dict,
            raw_sample
    ) -> Dict:
        (phrase, labels), relations, sentence = raw_sample
        for label_name in self.label_names:
            yield label_name, SequenceLabelField(
                [str(label == label_name) for label in labels],
                fields[self.get_fieldname('word')])

    @cls.fields
    def update_relations(
            self,
            fields: Dict,
            raw_sample
    ) -> Dict:
        (phrase, labels), relations, sentence = raw_sample
        if relations is None:
            return None
        relation_indices = []
        relation_labels = []
        triplet_labels = []
        for rel in relations:
            src_index = rel[1][0]
            mid_index = rel[2][0]
            dst_index = rel[3][0]
            if (src_index, mid_index, dst_index) not in relation_indices:
                relation_indices.append((src_index, mid_index, dst_index))

            relation_labels.append(rel[0])
            if rel[0] != "relation_none":
                triplet_labels.append(self.triplet_names[0])
            else:
                triplet_labels.append(self.triplet_names[1])

        for relation_name in self.relation_names:
            cur_indices = []
            for index, label in zip(relation_indices, relation_labels):
                if label == relation_name:
                    cur_indices.append(index)

            yield relation_name, NewAdjacencyField(
                cur_indices,
                fields[self.get_fieldname('word')],
                padding_value=0
            )

        for triplet_name in self.triplet_names:
            cur_indices = []
            for index, label in zip(relation_indices, triplet_labels):
                if label == triplet_name:
                    cur_indices.append(index)

            yield triplet_name, NewAdjacencyField(
                cur_indices,
                fields[self.get_fieldname('word')],
                padding_value=0
            )


@keep_keys
class SpRLSensorReader(SpRLBinaryReader):
    @cls.field('triplet_mask')
    def update_relations(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
        (phrase, labels), relations, sentence = raw_sample
        if relations is None:
            return None
        relation_indices = set()
        for rel in relations:
            src_index = rel[1][0]
            mid_index = rel[2][0]
            dst_index = rel[3][0]
            relation_indices.add((src_index, mid_index, dst_index))
        return NewAdjacencyField(
            list(relation_indices),
            fields[self.get_fieldname('word')],
            padding_value=0
        )
      


class NewAdjacencyField(AdjacencyField):
    _already_warned_namespaces: Set[str] = set()

    def __init__(self,
                 indices: List[Tuple[int, int, int]],
                 sequence_field: SequenceField,
                 labels: List[str] = None,
                 label_namespace: str = 'labels',
                 padding_value: int = -1) -> None:
        self.indices = indices
        self.labels = labels
        self.sequence_field = sequence_field
        self._label_namespace = label_namespace
        self._padding_value = padding_value
        self._indexed_labels: List[int] = None

        self._maybe_warn_for_namespace(label_namespace)
        field_length = sequence_field.sequence_length()

        if len(set(indices)) != len(indices):
            raise ConfigurationError(f"Indices must be unique, but found {indices}")

        if not all([0 <= index[1] < field_length and 0 <= index[0] < field_length for index in indices]):
            raise ConfigurationError(f"Label indices and sequence length "
                                     f"are incompatible: {indices} and {field_length}")

        if labels is not None and len(indices) != len(labels):
            raise ConfigurationError(f"Labelled indices were passed, but their lengths do not match: "
                                     f" {labels}, {indices}")

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens = padding_lengths['num_tokens']
        tensor = torch.ones(desired_num_tokens, desired_num_tokens, desired_num_tokens) * self._padding_value
        labels = self._indexed_labels or [1 for _ in range(len(self.indices))]

        for index, label in zip(self.indices, labels):
            tensor[index] = label
        return tensor

    def empty_field(self) -> 'AdjacencyField':
        # pylint: disable=protected-access
        # The empty_list here is needed for mypy
        empty_list: List[Tuple[int, int, int]] = []
        adjacency_field = NewAdjacencyField(empty_list,
                                            self.sequence_field.empty_field(),
                                            padding_value=self._padding_value)
        return adjacency_field


sp = SpRLReader()
#sp.parseSprlXML('examples/SpRL/data/new_train.xml')
#sp.entity_candidate_generation_for_train(sp.parseSprlXML('examples/SpRL/data/new_train.xml'))
sp.getCorpus(sp.entity_candidate_generation_for_train(sp.parseSprlXML('data/new_train.xml')))
# sp.getCorpus(sp.parseSprlXML('data/newSprl2017_all.xml'))
# sp.getCorpus(sp.negative_entity_generation(sp.parseSprlXML('data/newSprl2017_all.xml')))

# sp.getCorpus(phraselist)
# sp.parseSprlXML('data/sprl2017_train.xml')

# sp.getCorpus(sp.parseSprlXML('data/newSprl2017_all.xml'))
# print(sentence_list)

# getCorpus(sentence_list)

import pickle

class JsonReader(SensableReader):
    def read(self, file_path, **kwargs):
        with open(file_path, 'rb') as fin:
            return pickle.load(fin)
