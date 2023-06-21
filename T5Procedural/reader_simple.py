import torch
import random

from domiknows.data.reader import RegrReader


# The ProparaReader class is a subclass of RegrReader that defines methods for parsing and extracting
# various values from Propara dataset files.
class ProparaReader(RegrReader):

    def parse_file(self):
        """
        This function loads data from a file using PyTorch and returns a list of the values in the data.
        :return: a list of values from the data loaded from a file using PyTorch's `torch.load()`
        function.
        """
        data = torch.load(self.file)
        new_data = []
        for key in data.keys():
            val = data[key]
            val['pid'] = key
            new_data.append(val)
        return list(new_data)

    def getProcedureIDval(self, item):
        # return [item["para_id"]]
        ### generate a random number and add to the following sample id
        return [f"sample_id {item['pid']}"]

    def getEntitiesval(self, item):
        return [item['entities']]
        # return [[item['entities'][0]]]
        
    def getEntityval(self, item):
        return item["entities"]
        # return [item['entities'][0]]
    
    def getContextval(self, item):
        return [item['sentence_texts']]

    def getStepval(self, item):
        sentences = item["sentence_texts"]
        return  sentences
    
    def getLocationsval(self, item):
        locations = []
        for loc in item['loc_candidates']:
            if 1 in loc:
                loc.remove(1)
            loc = [str(i) for i in loc]
            loc = " ".join(loc)
            locations.append(loc)
        assert len(locations) == len(set(locations))
        return [locations]
    
    def getLocationval(self, item):
        locations = []
        for loc in item['loc_candidates']:
            if 1 in loc:
                loc.remove(1)
            loc = [str(i) for i in loc]
            loc = " ".join(loc)
            locations.append(loc)
        
        return locations
        
    def getbeforeval(self, item):
        """
        This function returns two torch tensors containing binary values indicating the positions of two
        sentences in a list of sentences.
        
        :param item: The `item` parameter is a dictionary containing information about a sentence,
        including its text and other relevant features. The function `getbeforeval` uses this
        information to create two tensors `b1s` and `b2s`, which represent the positions of two
        sentences relative to each other in the
        :return: two tensors, `b1s` and `b2s`, stacked together using `torch.stack()`. These tensors are
        created by iterating over the `sentence_texts` key of the `item` dictionary and creating binary
        tensors `b1` and `b2` with 1s at the indices corresponding to the current and next sentence
        respectively. These binary tensors are then appended
        """
        b1s = []
        b2s = []
        for step in range(len(item['sentence_texts'])):
            for step1 in range(len(item['sentence_texts'])):
                if step < step1:
                    b1 = torch.zeros(len(item["sentence_texts"]))
                    b1[step] = 1
                    b2 = torch.zeros(len(item["sentence_texts"]))
                    b2[step1] = 1
                    b1s.append(b1)
                    b2s.append(b2)
                    
                    
        return torch.stack(b1s), torch.stack(b2s)
        

    def getbefore_trueval(self, item):
        """
        The function returns a tensor of zeros with ones in positions corresponding to pairs of steps in
        a given item's sentence texts.
        
        :param item: The "item" parameter is likely a dictionary containing information about a sequence
        of sentences. It seems to have a key called "sentence_texts" which is a list of strings
        representing the text of each sentence in the sequence. The function is creating a tensor of
        zeros with a length of num_steps * num
        :return: a tensor of zeros with a length of `num_steps * num_steps`. The function iterates over
        the `sentence_texts` in the input `item` and sets the values in the tensor to 1 for all pairs of
        steps where the second step comes after the first step.
        """
        num_steps = len(item["sentence_texts"])
        values = torch.zeros(num_steps * num_steps)
        for step in range(len(item["sentence_texts"])):
            for step1 in range(step + 1, len(item["sentence_texts"])):
                values[(step * num_steps) + step1] = 1
        return values
    
    def getexact_beforeval(self, item):
        """
        This function returns two torch tensors containing binary values indicating the positions of two
        adjacent sentences in a list of sentences.
        
        :param item: The parameter "item" is a dictionary containing information about a particular
        item. It has a key "sentence_texts" which is a list of strings representing the sentences in the
        item
        :return: two tensors, `b1s` and `b2s`, which are stacks of one-hot encoded tensors. `b1s`
        contains tensors where the value at the index corresponding to the current sentence in
        `item['sentence_texts']` is 1 and all other values are 0. `b2s` contains tensors where the value
        at the index corresponding to the
        """
        b1s = []
        b2s = []
        for step in range(len(item['sentence_texts'])):
            step1 = step + 1
            if step1 < len(item['sentence_texts']):
                b1 = torch.zeros(len(item["sentence_texts"]))
                b1[step] = 1
                b2 = torch.zeros(len(item["sentence_texts"]))
                b2[step1] = 1
                b1s.append(b1)
                b2s.append(b2)
                    
                    
        return torch.stack(b1s), torch.stack(b2s)
    
    ### Getting the after location probs and value
    def getAfterLocationProbval(self, item):
        decision, _ = self.compute_iterative_probs(item, 'after_location')
        return decision
    def getAfterLocationTrueval(self, item):
        _, ground_truth = self.compute_iterative_probs(item, 'after_location')
        return ground_truth
    
    ### Getting the before location probs and value
    def getBeforeLocationProbval(self, item):
        decision, _ = self.compute_iterative_probs(item, 'before_location')
        return decision
    
    def getBeforeLocationTrueval(self, item):
        _, ground_truth = self.compute_iterative_probs(item, 'before_location')
        return ground_truth
    
    ### Getting the multi-action probs and value
    def getMultiActionProbval(self, item):
        decision, _ = self.compute_iterative_probs(item, 'multi_action')
        return decision
    
    def getMultiActionTrueval(self, item):
        _, ground_truth = self.compute_iterative_probs(item, 'multi_action')
        return ground_truth
    
    def compute_iterative_probs(self, item, key):
        """
        This function takes in an item and a key, loops over all entities in the processed
        probabilities, formats the predictions and ground truths, and returns them as torch tensors.
        
        :param item: The input item containing processed probabilities for different entities and their
        corresponding predictions and ground truths
        :param key: The "key" parameter is a string that specifies which type of predictions to gather
        from the "processed_probs" dictionary in the "item" object. It could be "bool" for binary
        predictions (e.g. yes/no), "multi" for multi-class predictions (e.g. create/exist
        :return: a tuple of two tensors. The first tensor contains all the decisions made by the model
        for each entity in the processed probabilities, and the second tensor contains the corresponding
        ground truth values for each decision.
        """
        ### loop over all entities in the processed probs and gather the after_location values
        all_decisions = []
        all_ground_truths = []
        count = 0
        for eid in item['processed_probs']:
            # if count != 0:
            #     break
            # count += 1
            predictions = item['processed_probs'][eid][key]
            formatted_preds = []
            formatted_gt = []
            for sid, val in enumerate(predictions):
                if not torch.is_tensor(val[0]):
                    formatted_preds.append(torch.stack(val[0]))
                else:
                    formatted_preds.append(val[0])
                if 'bool' in key:
                    tval = 0
                    if val[1] == 'yes':
                        tval = 1
                    formatted_gt.append(tval)
                elif "multi" in key:
                    _cand_list = ["create", "exist","move", "destroy", "prior", "post"]
                    tval = _cand_list.index(val[1])
                    formatted_gt.append(tval)
                # elif 'location' in key:
                #     _cand_list = item['loc_candidates']
                #     tval = _cand_list.index(val[1])
                #     formatted_gt.append(tval)
                else:
                    if "location" in key:
                        answer = val[1]
                        if 1 in answer:
                            answer.remove(1)
                        # if answer[0] == 3 and len(answer) > 1:
                        #     answer = answer[1:]
                        if answer[0] == 3 and len(answer) == 1:
                            answer = [5839]
                        # answer = [str(x) for x in answer]
                        # answer = " ".join(answer)
                        check = -1
                        for lid, lcand in enumerate(item['loc_candidates']):
                            if 1 in lcand:
                                lcand.remove(1)
                            if answer == lcand:
                                check = lid
                                break
                        answer = check
                        formatted_gt.append(answer)
                    else:
                        formatted_gt.append(val[1])
            
            all_decisions.append(torch.stack(self.process_prob_vectors(formatted_preds, key)))
            all_ground_truths.append(formatted_gt)
        # if "location" in key:
        #     return torch.stack(all_decisions), all_ground_truths
        # else:
        return torch.stack(all_decisions), torch.tensor(all_ground_truths, dtype=torch.float32)
    
    def computer_single_probs(self, item, key):
        """
        The function takes in an item and a key, extracts predictions and ground truths from the item's
        processed probabilities, and returns them as torch tensors.
        
        :param item: The input data item that contains processed probabilities for a particular task
        :param key: The "key" parameter is a string that specifies which type of predictions to retrieve
        from the "processed_probs" dictionary in the "item" object. The function uses this key to
        extract the relevant predictions and ground truths for a particular type of problem (e.g.
        input/output, when, etc.)
        :return: two tensors - `torch.stack(all_decisions)` and `torch.tensor(all_ground_truths)`.
        """
        all_decisions = []
        all_ground_truths = []
        count = 0
        for eid in item['processed_probs']:
            # if count != 0:
            #     break
            # count += 1
            predictions = item['processed_probs'][eid][key]   
            all_decisions.append(self.process_prob_vectors(predictions[0]))
            if 'input' in key or 'output' in key:
                tval = 0
                if predictions[1] == 'yes':
                    tval = 1
                all_ground_truths.append(tval)
            elif 'when' in key:
                ### find which index should be selected, never is always the index 0
                if predictions[1] == 'never':
                    step_of_pred = 0
                else:
                    step_of_pred = int(predictions[1].split(' ')[1])
                all_ground_truths.append(step_of_pred)
            else:
                all_ground_truths.append(predictions[1])
        return torch.stack(all_decisions), torch.tensor(all_ground_truths)
    
    def process_prob_vectors(self, vectors, key):
        acc = {"multi_action": 73.05, "before_location": 68.21, "after_location": 68.21}
        # return vectors
        multiplier = pow(acc[key], 4)
        final_vec = []
        if torch.is_tensor(vectors):
            vector = torch.clamp(vectors, min=1e-12, max=1 - 1e-12)
            entropy = torch.distributions.Categorical(torch.log(vector)).entropy() / vector.shape[0]
            # vector = (1/entropy.item()) * (vector/torch.mean(vector))
            final_vec = vector * multiplier
        else:
            for vector in vectors:
                vector = torch.clamp(vector, min=1e-12, max=1 - 1e-12)
                entropy = torch.distributions.Categorical(torch.log(vector)).entropy() / vector.shape[0]
                # vector = (1/entropy.item()) * (vector/torch.mean(vector))
                final_vec.append(vector * multiplier)
        return final_vec

    def getSameMentionsval(self, item):
        entities = item['entities_tokens']
        location = item['loc_candidates']
        matchings = []
        # for eid, ent in enumerate(entities):
        #     ### ent is a list of lists, we want to flattent the list here
        #     ent = [_it for sublist in ent for _it in sublist]
        #     ### Finding the indexes of location list, where its value is inside ent
        #     loc_indexes = [location.index(_it) for _it in ent if _it in location]
        #     for loc_index in loc_indexes:
        #         if (eid, loc_index) not in matchings:
        #             matchings.append((eid, loc_index))
        for lid, loc in enumerate(location):
            ### Finding the indexes of entities list, where its value is inside location
            if 1 in loc:
                loc.remove(1)
            check = False
            for eid, ent in enumerate(entities):
                if loc in ent:
                    if (eid, lid) not in matchings:
                        matchings.append((eid, lid))
                        check = True
                        break
            # if not check:
            #     ### check if location is part of any of these entities
            #     ### find the largest match 
            #     ### if it only matches one entity, then it is a match
            #     matched = -1
            #     for eid, ent in enumerate(entities):
            #         for en in ent:
            #             if set(loc).issubset(set(en)):
            #                 if matched == -1:
            #                     matched = eid
            #                 else:
            #                     matched = -1
            #                     break
            #     if matched != -1:
            #         if (matched, lid) not in matchings:
            #             matchings.append((matched, lid))
        ### matching is [(eid, loc_index), ...)], if the same `loc_index` is repeated for more than one eid, then the smallest matching should be selected
        # a = matchings.copy()
        # for match in a:
        #     if not match in matchings:
        #         continue
        #     ent1 = entities[match[0]][0]
        #     loc1 = location[match[1]]
        #     for match2 in a:
        #         if not match2 in matchings:
        #             continue
        #         if match2 == match:
        #             continue
        #         ent2 = entities[match2[0]][0]
        #         loc2 = location[match2[1]]
        #         if loc1 == loc2:
        #             if loc1 == ent1[0] and loc1 != ent2[0]:
        #                 matchings.remove(match2)
        #             elif loc1 == ent2[0] and loc1 != ent1[0]:
        #                 matchings.remove(match)
        #             elif loc1 == ent1[0] and loc1 == ent2[0]:
        #                 if len(ent1) > len(ent2):
        #                     if match in matchings:
        #                         matchings.remove(match)
        #                 else:
        #                     if match2 in matchings:
        #                         matchings.remove(match2)
                        


        connection_entities = torch.zeros(len(matchings), len(entities))
        connection_locations = torch.zeros(len(matchings), len(location))
        ### putting one for the corresponding entity and location
        for match_id, match in enumerate(matchings):
            connection_entities[match_id, match[0]] = 1
            connection_locations[match_id, match[1]] = 1
        check = ((connection_locations == 1).nonzero()[...,-1:]).flatten().tolist()
        if len(set(check)) != len(check):
            print("multiple entities matched a location")
        return connection_entities, connection_locations
            


