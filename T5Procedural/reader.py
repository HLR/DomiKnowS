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
        return list(data.values())

    def getProcedureIDval(self, item):
        # return [item["para_id"]]
        ### generate a random number and add to the following sample id
        return [f"sample_id {random.randint(0, 1000)}"]

    def getEntitiesval(self, item):
        return [item['entities']]
        
    def getEntityval(self, item):
        return item["entities"]
    
    def getContextval(self, item):
        return [item['sentence_texts']]

    def getStepval(self, item):
        sentences = item["sentence_texts"]
        return  sentences
    
    def getLocationsval(self, item):
        return [item['loc_candidates']]
    
    def getLocationval(self, item):
        return item['loc_candidates']
    
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
    
    ### Getting Boolean probs and value for create, destroy, and move
    def getCreateProbval(self, item):
        decision, _ = self.compute_iterative_probs(item, 'bool_create')
        return decision
    
    def getCreateTrueval(self, item):
        _, ground_truth = self.compute_iterative_probs(item, 'bool_create')
        return ground_truth
    
    def getDestroyProbval(self, item):
        decision, _ = self.compute_iterative_probs(item, 'bool_destroy')
        return decision
    
    def getDestroyTrueval(self, item):
        _, ground_truth = self.compute_iterative_probs(item, 'bool_destroy')
        return ground_truth
    
    def getMoveProbval(self, item):
        decision, _ = self.compute_iterative_probs(item, 'bool_move')
        return decision
    
    def getMoveTrueval(self, item):
        _, ground_truth = self.compute_iterative_probs(item, 'bool_move')
        return ground_truth
    
    ### get bool-change probs and value
    def getChangeProbval(self, item):
        decision, _ = self.compute_iterative_probs(item, 'bool_change')
        return decision
    
    def getChangeTrueval(self, item):
        _, ground_truth = self.compute_iterative_probs(item, 'bool_change')
        return ground_truth
    
    ### Get the input and input_alt value for the single entity
    def getInputProbval(self, item):
        decision, _ = self.computer_single_probs(item, 'input')
        return decision

    def getInputTrueval(self, item):
        _, ground_truth = self.computer_single_probs(item, 'input')
        return ground_truth
    
    def getAltInputProbval(self, item):
        decision, _ = self.computer_single_probs(item, 'input_alt')
        return decision
    
    def getAltInputTrueval(self, item):
        _, ground_truth = self.computer_single_probs(item, 'input_alt')
        return ground_truth
    
    ### Get the output and output_alt value for the single entity
    def getOutputProbval(self, item):
        decision, _ = self.computer_single_probs(item, 'output')
        return decision
    
    def getOutputTrueval(self, item):
        _, ground_truth = self.computer_single_probs(item, 'output')
        return ground_truth
    
    def getAltOutputProbval(self, item):
        decision, _ = self.computer_single_probs(item, 'output_alt')
        return decision
    
    def getAltOutputTrueval(self, item):
        _, ground_truth = self.computer_single_probs(item, 'output_alt')
        return ground_truth
    
    ### Get the when-create and when_destroy probs and value for the single entity
    def getWhenCreateProbval(self, item):
        decision, _ = self.computer_single_probs(item, 'when_create')
        return decision
    
    def getWhenCreateTrueval(self, item):
        _, ground_truth = self.computer_single_probs(item, 'when_create')
        return ground_truth
    
    def getWhenDestroyProbval(self, item):
        decision, _ = self.computer_single_probs(item, 'when_destroy')
        return decision
    
    def getWhenDestroyTrueval(self, item):
        """
        This function returns the ground truth value for the "when_destroy" attribute of a given item.
        
        :param item: The "item" parameter is not defined in the given code snippet. It is likely that it
        is defined elsewhere in the code and passed as an argument to this function
        :return: The function `getWhenDestroyTrueval` is returning the ground truth value for the
        `when_destroy` label of a given `item`.
        """
        _, ground_truth = self.computer_single_probs(item, 'when_destroy')
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
        for eid in item['processed_probs']:
            predictions = item['processed_probs'][eid][key]
            formatted_preds = []
            formatted_gt = []
            for sid, val in enumerate(predictions):
                formatted_preds.append(val[0])
                if 'bool' in key:
                    tval = 0
                    if val[1] == 'yes':
                        tval = 1
                    formatted_gt.append(tval)
                elif "multi" in key:
                    _cand_list = ["create", "exist","move", "destroy", "outside"]
                    tval = _cand_list.index(val[1])
                    formatted_gt.append(tval)
                # elif 'location' in key:
                #     _cand_list = item['loc_candidates']
                #     tval = _cand_list.index(val[1])
                #     formatted_gt.append(tval)
                else:
                    formatted_gt.append(val[1])
            all_decisions.append(torch.stack(formatted_preds))
            all_ground_truths.append(formatted_gt)
        return torch.stack(all_decisions), torch.tensor(all_ground_truths)
    
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
        for eid in item['processed_probs']:
            predictions = item['processed_probs'][eid][key]
            all_decisions.append(predictions[0])
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
