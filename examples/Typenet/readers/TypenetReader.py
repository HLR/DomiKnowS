import gc
import sys
sys.path.append('../../../')

import numpy as np
import pickle
import joblib
import os
import h5py
from tqdm.autonotebook import tqdm
import json

from regr.data.reader import RegrReader

import config

skip_tokens = set(["<em>", "</em>"])

def process(vocab_dict, tokens, flag_wiki = False, encoder = "rnn_phrase"):
        '''
        Takes the tokens of a mention sentence in the form of a list of strings, and outputs a left context and a right context list along with the mention surface form
        '''

        def process_tokens(tokens):
            p_tokens = []
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok == "<target>":
                    p_tokens.append(tok)
                    p_tokens += tokens[i+1].split("_")
                    i += 1
                elif tok not in skip_tokens:
                    p_tokens.append(tok)
                i += 1
            return p_tokens

        def find_str(text, str):
            for i, text in enumerate(text):
                if (text == str):
                    return i
            return -1

        def getid(token):
            if token not in vocab_dict:
                return OOV
            else:
                return vocab_dict[token]


        if flag_wiki:
            tokens = process_tokens(tokens)

        i = find_str(tokens, "<target>")  # i+1..j-1 is the mention
        j = find_str(tokens, "</target>")
        assert(i != -1)
        assert(j != -1)
        # surface form of the mention is everything from i+1..j-1
        sfm_mention = tokens[i+1:j]


        left_context = tokens[: i+1] + sfm_mention
        right_context = sfm_mention + tokens[j:]

        left_context =  [getid(tok) for tok in left_context]  # s1... <target> m1 ... mN
        right_context = [getid(tok) for tok in right_context] # m1 ... mN </target> s_k+1 ... sN


        if encoder == "position_cnn" or encoder == "rnn_phrase":
            position_left = [MAX_SENT-dist for dist in xrange(i, 0, -1)]
            position_right = [MAX_SENT+dist - j for dist in xrange(j+1, len(tokens))]

            position_mention = [MAX_SENT]*(j-i -1)


            return [getid(tok) for tok in tokens if tok not in target_mention_markers], \
                [getid(tok) for tok in sfm_mention], position_left + position_mention + position_right, [i], [j-2]

        else:
            return [getid(tok) for tok in tokens], \
                [getid(tok) for tok in sfm_mention], [getid(tok) for tok in tokens], -1, -1

def get_curr_sentence(mention_raw):
    mention_raw = mention_raw.split("\t")
    assert(len(mention_raw) == 6)
    title, sfm_mention, entity, prev_sentence, curr_sentence, next_sentence = mention_raw

    return curr_sentence.split(" "), sfm_mention, entity

def read_entities(filename, start = 0.0, end = 1.0):
    f = open(filename, encoding='UTF-8')
    entities = set()

    for line in f:
        line = line.strip().split("\t")
        entities.add(line[0])

    start_idx = int(start*len(entities))
    end_idx  = int(end*len(entities))
    return set(list(entities)[start_idx: end_idx])

import re
pattern = re.compile('[^\s\w]+')

OOV=1

def get_candidates(sfm_mention, entity_dict, gold_entity, crosswikis, train, hierarchy):
    '''
        Use cross-wikis to retrieve top-100 entities for this mention
    '''
    candidate_probab_list = crosswikis[sfm_mention] if sfm_mention in crosswikis else []
    # the list is organized as (ent, prob), and we sort it in decreasing order of P(ent | sfm_mention)
    candidate_probab_list.sort(key = lambda item : -item[1])

    gold_id = entity_dict[gold_entity]

    gold_ids = set([gold_id])
    if hierarchy is not None:
        ancestors = hierarchy[gold_id] if gold_id in hierarchy else set()

        gold_ids |= ancestors

    if train:
        # take the top 100 - len(gold_ids) and add to it the gold entity at train time
        crosswiki_data = [ (entity_dict[ent], prob) for (ent, prob) in candidate_probab_list if  ent in entity_dict and entity_dict[ent] not in gold_ids][:100]

        neg_cands = len(crosswiki_data)
        ent_gold_prob = {_id : 0.0 for _id in gold_ids}
        for ent, prob in candidate_probab_list:
            if ent not in entity_dict:
                continue
            ent_id = entity_dict[ent]
            if ent_id in gold_ids:
                if ent_gold_prob[ent_id] == 0.0:
                    ent_gold_prob[ent_id] = prob


        crosswiki_data += ent_gold_prob.items()
        candidates, priors = zip(*crosswiki_data)


        return candidates, priors, [0]*neg_cands + [1]*len(gold_ids)

    else:
        # take the top 100 entities and hope that the gold is somewhere in this
        crosswiki_data = [ [entity_dict[ent], prob] for (ent, prob) in candidate_probab_list if ent in entity_dict][:100]
        if (len(crosswiki_data) != 0):
            candidates, priors = zip(*crosswiki_data)
        else:
            candidates, priors = [], []

        return candidates, priors, gold_id

def getLnrm(arg, pattern):
    """Normalizes the given arg by stripping it of diacritics, lowercasing, and
    removing all non-alphanumeric characters.
    """
    arg = pattern.sub('', arg)
    arg = arg.lower()

    return arg

def filter_fb_types(type_dict, entity_type_dict, typenet_matrix_orig):
    fb_types = [_type for _type in type_dict if not _type.startswith("Synset")]
    wordnet_types = [_type for _type in type_dict if _type.startswith("Synset")]

    '''def fix_name(orig):
                    orig = orig.replace('-', '_')
                    if orig[:7] == 'Synset(':
                        new = 'Synset__' + orig[8:-2]
                        return new.replace('.', '__')
                    else:
                        return orig.replace('/', '__')
            
                import json
                with open('labels.json', 'w+') as file_out:
                    json.dump([fix_name(_type) for _type in fb_types], file_out)'''

    # reorder types to make fb types appear first
    all_types = fb_types + wordnet_types

    orig_idx2type = {idx : _type for (_type, idx) in type_dict.items()}
    type2idx = {_type : idx for (idx, _type) in enumerate(all_types)}
    orig2new = {idx : type2idx[_type] for (_type,idx) in type_dict.items()}
    typenet_matrix = np.zeros(typenet_matrix_orig.shape)
    for i,j in zip(*np.where(typenet_matrix_orig == 1)):
        typenet_matrix[orig2new[i],orig2new[j]] = 1

    # 1. filter out only fb types and 2. change fb IDs according to type2idx
    fb_entity_type_dict = {}
    for ent in entity_type_dict:
        curr = []
        for type_id in entity_type_dict[ent]:
            orig_type = orig_idx2type[type_id]

            if not orig_type.startswith("Synset") or not config.freebase_only:
                curr.append(orig_type)

        assert(len(curr) != 0)
        fb_entity_type_dict[ent] = set(curr) # easy to search


    return type2idx, fb_entity_type_dict, len(fb_types), typenet_matrix, orig2new, fb_types, all_types, len(all_types)

def underscore_to_slash(orig):
    res = orig.replace('__', '/')
    res = res.replace('_', '-')
    return res

def fix_name(orig):
    orig = orig.replace('-', '_')
    if orig[:7] == 'Synset(':
        new = 'Synset__' + orig[8:-2]
        return new.replace('.', '__')
    else:
        return orig.replace('/', '__')

with open('depth_classes.json', 'r') as class_file:
    classes = json.load(class_file)

class WikiReader(RegrReader):
    def __init__(self, file, type, file_data, bag_size=3, mode='train', limit_size=None, vocab_map={}, prune=False):
        self.bag_size = bag_size
        
        self.mode = mode

        self.limit_size = limit_size

        self.train_bags = file_data['train_bags']
        self.embeddings = file_data['embeddings']
        self.typenet_matrix_orig = file_data['typenet_matrix_orig']
        self.vocab_dict = file_data['vocab_dict']
        self.entity_dict = file_data['entity_dict']
        self.type_dict = file_data['type_dict']
        self.entity_type_dict = file_data['entity_type_dict']

        self.class_counts = {}
        self.total_items = 0
        self.vocab_map = vocab_map
        self.prune = prune

        super().__init__(file=file, type=type)

    def make_object(self, item):
        self.subsample_ids = []
        for i in range(config.batch_size):
            self.subsample_ids.append(np.random.choice(len(item['context'][i]), self.bag_size, replace=False))

        result = super().make_object(item)

        for lbl in self.all_types:
            result[lbl] = []

        for data_types in item['gold_types_pos']:
            for lbl in self.all_types:
                if lbl in data_types:
                    result[lbl].append([1])
                else:
                    result[lbl].append([0])

        return result
    
    def add_to_dict(self, dict_result, new_item):
        for key, val in new_item.items():
            if not key in dict_result:
                dict_result[key] = []
            dict_result[key].append(val)

    def get_class_weights(self):
        # calculate class weights
        class_weights = {}

        for type, num in self.class_counts.items():
            class_weights[type] = self.total_items/num

        return class_weights

    def parse_file(self):
        # load entities
        train_entities = read_entities(self.file, end=0.05)

        # process
        print('WikiReader: processing data')
        type_dict, entity_type_dict, fb_type_size, typenet_matrix, map_old_to_new, fb_types, typenet_types, typenet_types_size = filter_fb_types(self.type_dict, self.entity_type_dict, self.typenet_matrix_orig)
        
        print('num fb types:', fb_type_size)
        print('num total types:', typenet_types_size)

        if config.freebase_only:
            types_used = fb_types
        else:
            types_used = typenet_types

        assert config.num_types + 1 == len(types_used) # one of the types is NO_TYPES, so add 1 to config

        # get list of all types use for prediction
        self.all_types = []
        for tp in types_used:
            if not tp == 'NO_TYPES':
                self.all_types.append(fix_name(tp))

        self.dataset_all = []

        # get total for tqdm
        total = None
        if self.limit_size == None:
            total = len(train_entities)
        else:
            total = min(self.limit_size, len(train_entities))

        batch = []

        # process bag of mention contexts per entity
        # each training iteration uses one entity (one bag of mention contexts contexts)
        for i, ent in tqdm(enumerate(train_entities), total=total):
            if i == self.limit_size:
                break

            if ent in self.train_bags:
                all_mentions = self.train_bags[ent]

                if len(all_mentions) >= self.bag_size:

                    # process each selected mention in bag
                    all_mention_data = {}
                    for curr_data in all_mentions:
                        data = {}

                        if isinstance(curr_data, bytes):
                            curr_data = curr_data.decode("utf-8")

                        curr_sentence, gold_mention, gold_ent = get_curr_sentence(curr_data)
                        gold_ent     = gold_ent[7:-4]
                        gold_mention = " ".join(gold_mention[9:].split("_"))

                        sentence, sfm_mention, position_embedding, st_id, en_id = process(self.vocab_dict, curr_sentence, flag_wiki=True, encoder='basic')

                        mention_representation = np.array([self.embeddings[_id] for _id in sfm_mention]).mean(axis = 0)

                        data['mention_representation'] = mention_representation
                        data['context'] = np.array(sentence)

                        for word_idx in data['context']:
                            if word_idx not in self.vocab_map:
                                self.vocab_map[word_idx] = len(self.vocab_map)

                        if self.prune:
                            data['context'] = np.fromiter((self.vocab_map[x] for x in data['context']), data['context'].dtype)

                        data['position_embeddings'] = position_embedding
                        data['st_ids'] = st_id
                        data['en_ids'] = en_id
                        #data['entity_candidates'] = all_candidates
                        #data['priors'] = priors
                        #data['gold_ids'] = gold_id

                        assert(len(position_embedding) == len(sentence))

                        self.add_to_dict(all_mention_data, data)

                    #bit_vec = [0]*config.num_types  # predictions are made only for freebase types
                    
                    labels = []
                    if entity_type_dict is not None:
                        labels = entity_type_dict[ent]

                    gold_types = []
                    for gt in labels:
                        gt = fix_name(gt)
                        gold_types.append(gt)

                    all_mention_data['gold_types_pos'] = set(gold_types)

                    for gt in gold_types:
                        if not gt in self.class_counts:
                            self.class_counts[gt] = 0
                        self.class_counts[gt] += 1

                    self.total_items += 1

                    batch.append(all_mention_data)

                    if len(batch) == config.batch_size:
                        batch_combined = {}
                        batch_combined['mention_representation'] = [item['mention_representation'] for item in batch]
                        batch_combined['context'] = [item['context'] for item in batch]
                        batch_combined['gold_types_pos'] = [item['gold_types_pos'] for item in batch]

                        self.dataset_all.append(batch_combined)

                        batch = []

        del train_entities
        gc.collect()

        print('WikiReader: finished processing data')
        return self.dataset_all

    def save_cache(self, fp):
        cache = {
            'bag_size': self.bag_size,
            'class_counts': self.class_counts,
            'total_items': self.total_items,
            'all_types': self.all_types,
            'dataset_all': self.dataset_all
        }

        with open(fp, 'wb') as file_out:
            pickle.dump(cache, file_out)

        print('Saved wiki cache to', fp)

    def get_pruned_embeddings(self):
        sorted_word_idx = sorted(list(self.vocab_map), key=lambda idx: self.vocab_map[idx])

        return self.embeddings[sorted_word_idx]

    def subsample(self, item_list, batch_idx):
        return [item_list[idx] for idx in self.subsample_ids[batch_idx]]
    
    def subsample_batch(self, item_batch):
        subsampled_batch = []
        for i, item in enumerate(item_batch):
            subsampled_batch.append(self.subsample(item, i))
        return subsampled_batch

    def getMentionRepresentationval(self, item):
        subsampled = self.subsample_batch(item['mention_representation'])
        return [subsampled]
    
    def getContextval(self, item):
        return [self.subsample_batch(item['context'])]

class CachedWikiReader(WikiReader):
    def __init__(self, file, type, mode='train'):
        self.mode = mode

        RegrReader.__init__(self, file=file, type=type)

    def parse_file(self):
        with open(self.file, 'rb') as file_in:
            self.cache_data = pickle.load(file_in)

        self.bag_size = self.cache_data['bag_size']

        self.class_counts = self.cache_data['class_counts']
        self.total_items = self.cache_data['total_items']

        self.all_types = self.cache_data['all_types']

        print('WikiReader: finished loading cache')

        return self.cache_data['dataset_all']