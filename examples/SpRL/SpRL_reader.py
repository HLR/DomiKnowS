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
import networkx as nx
from networkx.exception import NetworkXNoPath
import torch
import numpy as np
from featureIndexer import PosTaggerIndexer, LemmaIndexer, DependencyIndexer, HeadwordIndexer, PhrasePosIndexer
from feature import DataFeature_for_sentence, DataFeature_for_span
from dictionaries import dictionary


# sklearn
from allennlp.data.fields import TextField, MetadataField, ArrayField

nlpmodel = spacy.load("en_core_web_lg")
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
        tokens = [Token(each_df.text) for each_df in tokens]
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

    @cls.field('vec')
    def update_sentence_vec(
            self,
            fields,
            tokens
    ) -> Field:
        (tokens, _), _, _ = tokens
        dummy_vec = np.zeros_like(tokens[-1].headtoken_.vector)
        tokens_vec = np.array([each_df.headtoken_.vector if not each_df.is_dummy_ else dummy_vec for each_df in tokens])
        array = ArrayField(tokens_vec)
        return array
    
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
            return '::'.join((tokens[lm].lower_, tokens[tr].lower_, tokens[sp].lower_))

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

            return '::'.join((tokens[lm].headword_.lower(), tokens[tr].headword_.lower(), tokens[sp].lower_))

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
        sentence = tokens[0].doc

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))
            
        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(sentence.getShortestDependencyPath(tokens[lm], tokens[sp]))

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
        sentence = tokens[0].doc

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))
            
        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(sentence.getShortestDependencyPath(tokens[tr], tokens[sp]))

        indexers = {'triplet_feature5': SingleIdTokenIndexer(namespace='triplet_feature5')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield
    
    #phrasepos
    @cls.textfield('triplet_feature6')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)
        sentence = tokens[0].doc

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))
        
        def dummy_feature(lm, tr, sp):
            return '::'.join((tokens[lm].phrasepos_, tokens[tr].phrasepos_, tokens[sp].phrasepos_))
            
        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature6': SingleIdTokenIndexer(namespace='triplet_feature6')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield
    
    #pos
    @cls.textfield('triplet_feature7')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)
        sentence = tokens[0].doc

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))
        
        def dummy_feature(lm, tr, sp):
            return '::'.join((tokens[lm].pos_, tokens[tr].pos_, tokens[sp].pos_))
            
        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature7': SingleIdTokenIndexer(namespace='triplet_feature7')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield
    
    #dep
    @cls.textfield('triplet_feature8')
    def update_triplet_dummy(
            self,
            fields,
            tokens
    ) -> Field:
        length = len(tokens)
        sentence = tokens[0].doc

        # convert ijk index to 1-d array index
        def encap_index(*indices):
            return sum(i * length ** j for j, i in enumerate(reversed(indices)))
        
        def dummy_feature(lm, tr, sp):
            return '::'.join((tokens[lm].dep_, tokens[tr].dep_, tokens[sp].dep_))
            
        # prepare a empty list
        encap_tokens = [None] * (length ** 3)
        for lm, tr, sp in itertools.product(range(length), repeat=3):
            encap_tokens[encap_index(lm, tr, sp)] = Token(dummy_feature(lm, tr, sp))

        indexers = {'triplet_feature8': SingleIdTokenIndexer(namespace='triplet_feature8')}
        textfield = TextField(encap_tokens, indexers)
        # import pdb; pdb.set_trace()
        return textfield
    
    def raw_read(
            self,
            file_path: str,
            metas
    ) -> Iterable[Instance]:
        isTrain = metas and metas.get('dataset_type') == "train"
        phrase_list = self.parseSprlXML(file_path)
       # raw_examples = self.getCorpus(self.negative_entity_generation_for(phrase_list))
        phrase_list = self.entity_candidate_generation_for_train(phrase_list, isTrain)
        raw_examples = self.getCorpus(phrase_list)
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
        for scene in sprlXMLRoot.findall('./SCENE'):
            docno = scene.find('./DOCNO').text
            image = scene.find('./IMAGE').text
            for sentenceItem in scene.findall('./SENTENCE'):
                id_text_list = {}
                id_label_list = {}
                sentence = None

                sentence_dic = {}
                sentence_dic["id"] = sentenceItem.attrib["id"]
                sentence_dic["start"] = sentenceItem.attrib["start"]
                sentence_dic["end"] = sentenceItem.attrib["end"]
                sentence_dic['LANDMARK'] = []
                sentence_dic['TRAJECTOR'] = []
                sentence_dic['SPATIALINDICATOR'] = []
                sentence_dic['RELATION'] = []

                # iterate child elements of item

                for child in sentenceItem:

                    if child.tag == 'TEXT':
                        text = child.text
                        if not text:
                            break
                        sentence = DataFeature_for_sentence(text, id=sentence_dic["id"], docno=docno, image=image, start=sentence_dic['start'], end=sentence_dic['end'])
                        sentence_dic['TEXT'] = sentence

                    elif child.tag in ['LANDMARK', 'TRAJECTOR', 'SPATIALINDICATOR']:
                        id_ = child.attrib.get('id')
                        text = child.attrib.get('text')
                        start = int(child.attrib.get('start'))
                        end = int(child.attrib.get('end'))

                        if not text or start < 0 or end < 0:  # also skip empty
                            span = sentence.dummy
                        else:
                            span = sentence.getSpan(start, end)
                            assert span.text in text
                        sentence_dic[child.tag].append(span)
                        id_text_list[id_] = span
                        id_label_list[id_] = child.tag

                    elif child.tag == "RELATION":
                        # if child.attrib['general_type'] not in type_list:
                        #     type_list.append(child.attrib['general_type'])
                        # #
                        general_type = child.attrib.get('general_type')
                        landmark_id = child.attrib.get('landmark_id')
                        trajector_id = child.attrib.get('trajector_id')
                        spatial_indicator_id = child.attrib.get('spatial_indicator_id')
                        landmark = id_text_list.get(landmark_id, sentence.dummy)
                        trajector = id_text_list.get(trajector_id, sentence.dummy)
                        spatial_indicator = id_text_list.get(spatial_indicator_id, sentence.dummy)

                        if not general_type:
                            continue  # TODO: skip relation without general type? or add none relation?

                        each_relationship = (general_type, landmark, trajector, spatial_indicator)
                        sentence_dic['RELATION'].append(each_relationship)
                else:
                    sentence_dic['phrases'] = sentence.getChunks()
                    if len(sentence_dic['phrases']) == 1 and not sentence_dic['LANDMARK'] and not sentence_dic['TRAJECTOR'] and \
                        not sentence_dic['SPATIALINDICATOR'] and not sentence_dic['RELATION']:
                        continue
            
                    sentences_list.append(sentence_dic)

        return sentences_list

    def getCorpus(self, sentences_list):

        final_relationship = []

        for sents in sentences_list:
            sentence = sents['TEXT']
            phrase_list = sents['phrases']
            label_list = []
            for phrase in phrase_list:
                phrase_label = []
                for phrase_type in self.label_names:
                    if phrase in sents[phrase_type]:
                        phrase_label.append(phrase_type)
                label_list.append(phrase_label)
            
            relations = self.negative_relation_generation_for_test(sents['RELATION'], phrase_list)
            relations_indexed = []
            for r_type, lm, tr, si in relations:
                lmi = phrase_list.index(lm)
                tri = phrase_list.index(tr)
                sii = phrase_list.index(si)
                relations_indexed.append((r_type, lmi, tri, sii))
            relation_tuple = ((phrase_list, label_list), relations_indexed, sentence)
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
        if phrase.text in spatial_dict or spatial_pos == "ADP":
            return phrase

    def negative_relation_generation_for_test(self, relations, generated_phrase_list):
        landmark_candidate_list = []
        trajector_candidate_list = []
        spatial_indicator_candidate_list = []
        for each_phrase in generated_phrase_list:
            if self.landmark_candidate_generation(each_phrase) or each_phrase.is_dummy_:
                landmark_candidate_list.append(each_phrase)
            if self.trajector_candidate_generation(each_phrase) or each_phrase.is_dummy_:
                trajector_candidate_list.append(each_phrase)
            if self.spatial_indicator_candidate_generation(each_phrase) or each_phrase.is_dummy_:
                spatial_indicator_candidate_list.append(each_phrase)

        raw_triplet_candidates = list(product(landmark_candidate_list, trajector_candidate_list, spatial_indicator_candidate_list))
        new_relation_triplet = []
        for arg1, arg2, arg3 in filter(span_not_overlap, raw_triplet_candidates):
            isNone = True
            for each_relation in relations:
                gold_arg1, gold_arg2, gold_arg3 = each_relation[1:]
                if (arg1.share_headtoken(gold_arg1) and arg2.share_headtoken(gold_arg2) and arg3.overlap(gold_arg3)):
                    isNone = False
                    new_relation_triplet.append((each_relation[0], arg1, arg2, arg3))
            if isNone:
                new_relation_triplet.append(("relation_none", arg1, arg2, arg3))

        return new_relation_triplet
        

    def entity_candidate_generation_for_train(self, phraselist, isTrain=False):
        gold_entity_landmark = 0
        gold_entity_trajector = 0
        gold_entity_spatialindicator = 0
        positive_entity_landmark = 0
        positive_entity_trajector = 0
        positive_entity_spatialindicator = 0
        negative_entity = 0

        new_phraselist = []

        for each_sentence in phraselist:
            sentence = each_sentence.get('TEXT')
            chunklist = each_sentence.get('phrases', [])

            landmarklist = each_sentence.get('LANDMARK', [])
            trajectorlist = each_sentence.get('TRAJECTOR', [])
            spatialindicatorlist = each_sentence.get('SPATIALINDICATOR', [])
            each_sentence['NONE'] = []
            labelnone = each_sentence.get('NONE', [])

            gold_entity_landmark += len(landmarklist)
            gold_entity_trajector += len(trajectorlist)
            gold_entity_spatialindicator += len(spatialindicatorlist)

            landmark_gt = landmarklist.copy()
            trajector_gt = trajectorlist.copy()
            spatialindicator_gt = spatialindicatorlist.copy()
            landmarklist.clear()
            trajectorlist.clear()
            spatialindicatorlist.clear()


            if isTrain:
                for groundtruthlist in (landmark_gt, trajector_gt, spatialindicator_gt):
                    for phrase in groundtruthlist:
                        if phrase not in chunklist:
                            chunklist.append(phrase)
            chunklist.sort(key=lambda chunk: chunk.start)

        
            for chunk in chunklist:
                isNone = True
                if chunk.share_headtoken_any(landmark_gt):
                    landmarklist.append(chunk)
                    isNone = False
                if chunk.share_headtoken_any(trajector_gt):
                    trajectorlist.append(chunk)
                    isNone = False
                if chunk.overlap_any(spatialindicator_gt):
                    spatialindicatorlist.append(chunk)
                    isNone = False
                if isNone:
                    labelnone.append(chunk)

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
                [str(label_name in label) for label in labels],
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

        for relation_name in self.relation_names:
            cur_indices = []
            for rel in relations:
                if rel[0] == relation_name:
                    cur_indices.append(rel[1:])

            yield relation_name, NewAdjacencyField(
                set(cur_indices),
                fields[self.get_fieldname('word')],
                padding_value=0
            )

        triplet_indices = []
        none_indices = []
        for rel in relations:
            if rel[0] == "relation_none":
                none_indices.append(rel[1:])
            else:
                triplet_indices.append(rel[1:])

        yield self.triplet_names[0], NewAdjacencyField(
            set(triplet_indices),
            fields[self.get_fieldname('word')],
            padding_value=0
        )
        yield self.triplet_names[1], NewAdjacencyField(
            set(none_indices),
            fields[self.get_fieldname('word')],
            padding_value=0
        )

@keep_keys
class SpRLSensorReader(SpRLBinaryReader):
    @cls.field('entity_mask')
    def update_entity(
        self,
        fields: Dict,
        raw_sample
    ) -> Field:
        (phrase, labels), relations, sentence = raw_sample
        return SequenceLabelField(
                [str(not phrase.is_dummy_) for phrase in phrase],
                fields[self.get_fieldname('word')])

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
            src_index = rel[1]
            mid_index = rel[2]
            dst_index = rel[3]
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


def test():    
    sp = SpRLReader()
    #sp.parseSprlXML('examples/SpRL/data/new_train.xml')
    #sp.entity_candidate_generation_for_train(sp.parseSprlXML('examples/SpRL/data/new_train.xml'))
    #plist = sp.parseSprlXML('data/new_train.xml')
    plist = sp.parseSprlXML('data/new_train.xml')
    ecandidate = sp.entity_candidate_generation_for_train(plist)
    sp.getCorpus(ecandidate)
    # sp.getCorpus(sp.parseSprlXML('data/newSprl2017_all.xml'))
    # sp.getCorpus(sp.negative_entity_generation(sp.parseSprlXML('data/newSprl2017_all.xml')))

    # sp.getCorpus(phraselist)
    # sp.parseSprlXML('data/sprl2017_train.xml')

    # sp.getCorpus(sp.parseSprlXML('data/newSprl2017_all.xml'))
    # print(sentence_list)

    # getCorpus(sentence_list)

if __name__ == '__main__':
    test()

import pickle

class PickleReader(SensableReader):
    def read(self, file_path, **kwargs):
        with open(file_path, 'rb') as fin:
            data = pickle.load(fin)
            return data
