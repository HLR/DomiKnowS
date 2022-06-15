from stemming.porter2 import stem
import csv
import torch
import re


def find_allzero_rows(vector) -> torch.BoolTensor:
    """
    Find all-zero rows of a given tensor, which is of size (batch, max_sents, max_tokens).
    This function is used to find unmentioned sentences of a certain entity/location.
    So the input tensor is typically a entity_mask or loc_mask.
    Return:
        a BoolTensor indicating that a all-zero row is True. Convenient for masked_fill.
    """
    column_sum = torch.sum(vector, dim = -1)
    return column_sum == 0

def fix_rel_name(name):
    name = ''.join([i for i in name if not i.isdigit()])
    name = name.replace("-", " ")
    name = re.sub(r'\bres\b', 'result', name)
    name = re.sub(r'\bassoc\b', 'associated', name)
    name = re.sub(r'\bloc\b', 'location', name)
    name = re.sub(r'\bsuchthat\b', 'such that', name)
    name = re.sub(r'\brel\b', 'relation', name)
    name = re.sub(r'\brefset\b', 'referral set', name)
    return name


def location_match(p_loc, g_loc):
    if p_loc == g_loc:
        return True

    p_string = " %s " % " ".join(
        [stem(x) for x in p_loc.lower().replace('"', "").split()]
    )
    g_string = " %s " % " ".join(
        [stem(x) for x in g_loc.lower().replace('"', "").split()]
    )

    if p_string in g_string:
        # print ("%s === %s" % (p_loc, g_loc))
        return True

    return False


import re


def removearticles(text):
    articles = {"a": "", "an": "", "and": "", "the": ""}
    rest = []
    for word in text.split():
        if word not in articles:
            rest.append(word)
    return " ".join(rest)


def load_preds(file="../experiments/test/predictions/test_0_output.tsv"):
    preds = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        for row in csv_reader:
            preds.append(row)
    return preds


def make_eval_entry_logical(inputs):
    data = []
    destroy = 0
    created = 0
    exists = 0
    from_zero = 0         
    for _id, inp in enumerate(inputs):
        item = {}
        item['para_id'] = inp[0]
        item['step'] = inp[1]
        step = int(item['step'])
        item['entity'] = inp[2]
        if step == 1:
            item['before'] = inp[4]
            if item['before'] != "-":
                exists = 1
                from_zero = 1
        else:
            item['before'] = data[-1]['after']
            
        if inp[3].lower() == "none" or inp[3].lower() == "nochange":
            item['action'] = "NONE"
            item['after'] = item['before']
            
        elif inp[3].lower() == "create":
            if exists:
                if len(inp[5]) and inp[5] != "?" and inp[5] != item['before']:
                    item['action'] = "MOVE"
                    item['after'] = inp[5]
                else:
                    item['action'] = "NONE"
                    item['after'] = item['before']
            else:
                exists = 1
                item['action'] = "CREATE"
                item['before'] = "-"
                if len(inp[5]):
                    item['after'] = inp[5]
                else:
                    item['after'] = "?"
                    
        elif inp[3].lower() == "destroy":
            later_check = True
            move_loc = ''
            for later in inputs[_id+1:]:
                if later[3].lower() == "create":
                    break
                elif later[3].lower() == "destroy":
                    later_check = False
                    break
            if exists and later_check :
                item['action'] = "DESTROY"
                exists = 0
                item['after'] = "-"
            else:
                item['action'] = "NONE"
                item['after'] = item['before']
                
        elif inp[3].lower() == "move":
            if exists:
                if inp[5] == "?":
                    item['action'] = "MOVE"
                    after = "?"
                    item['after'] = after
                elif not location_match(inp[5], item['before']):
                    item['action'] = "MOVE"
                    if inp[5]:
                        item['after'] = inp[5]
                    else:
                        item['after'] = "?"
                else:
                    item['action'] = "NONE"
                    item['after'] = item['before'] 
                
            else:
                item['action'] = "NONE"
                item['after'] = item['before']
                
        data.append(item)
    return data



def make_sequential_entry_logical(inputs, step_0):
    data = []
    destroy = 0
    created = 0
    exists = 0
    for _id, inp in enumerate(inputs):
        if inp[1] == "0":
            continue
        
        item = {}
        item['para_id'] = inp[0]
        item['step'] = inp[1]
        step = int(item['step'])
        item['entity'] = inp[2]
        if step == 1:
            if inp[3] == "O_C":
                item['action'] = "NONE"
                item['before'] = "-"
                item['after'] = "-"
            elif inp[3] == "O_D":
                item['action'] = "NONE"
                item['before'] = "-"
                item['after'] = "-"
            elif inp[3] == "E":
                exists = 1
                item['action'] = "NONE"
                item['before'] = step_0[5]
                item['after'] = step_0[5]
            elif inp[3] == "C":
                exists = 1
                item['action'] = "CREATE"
                item['before'] = "-"
                item['after'] = inp[5]
            elif inp[3] == "D":
                item['action'] = "DESTROY"
                item['before'] = step_0[5]
                item['after'] = "-"
            elif inp[3] == "M":
                exists = 1
                # if step_0[5] != inp[5] or inp[5] == "?":
                item['action'] = "MOVE"
                item['before'] = step_0[5]
                item['after'] = inp[5]
                # else:
                #     inp[3] = "E"
                #     item['action'] = "NONE"
                #     item['before'] = inp[5]
                #     item['after'] = inp[5]
        else:
            item['before'] = data[-1]['after']
            
            if inp[3] == "O_C":
                item['action'] = "NONE"
                if exists:
                    item['after'] = item['before']
                else:                
                    item['after'] = "-"

            elif inp[3] == "O_D":
                item['action'] = "NONE"
                if exists:
                    item['after'] = item['before']
                else:
                    item['after'] = "-"

            elif inp[3] == "E":
                item['action'] = "NONE"
                if exists:
                    item['after'] = item['before']
                else:
                    item['after'] = "-"

            elif inp[3] == "D":
                if exists:
                    exists = 0
                    item['after'] = "-"
                    item['action'] = "DESTROY"
                else:
                    item['after'] = "-"
                    item['action'] = "NONE"

            elif inp[3] == "C":
                if exists:
                    if inp[5] != item['before'] or inp[5] == "?":
                        item['after'] = inp[5]
                        item['action'] = "MOVE"
                    else:
                        item['after'] = item['before']
                        item['action'] = "NONE"
                else:
                    exists = 1
                    item['action'] = "CREATE"
                    item['after'] = inp[5]

            elif inp[3] == "M":
                if exists:
                    if inp[5] != item['before'] or inp[5] == "?":
                        item["action"] = "MOVE"
                        item['after'] = inp[5]
                    else:
                        item['action'] = "NONE"
                        item["after"] = item['before']
                else:
                    item["after"] = "-"
                    item['action'] = "NONE"
                
        data.append(item)
        
    return data



from typing import List, Dict

def get_output(entity_name, para_id, pred_state_seq: List[str], pred_loc_seq: List[str]) -> Dict:
    """
    Get the predicted output from generated sequences by the model.
    """
    pred_state_seq, pred_loc_seq = predict_consistent_loc(pred_state_seq = pred_state_seq, pred_loc_seq = pred_loc_seq,
                                                          para_id = para_id, entity = entity_name)
    prediction = format_final_prediction(pred_state_seq = pred_state_seq, pred_loc_seq = pred_loc_seq)

    result = {'id': para_id,
              'entity': entity_name,
              'prediction': prediction
              }
    return result


def format_final_prediction(pred_state_seq: List[str], pred_loc_seq: List[str]) -> List:
    """
    Final format: (state, loc_before, location_after) for each timestep (each sentence)
    """
#     assert len(pred_state_seq) + 1 == len(pred_loc_seq)
    num_sents = len(pred_state_seq)
    prediction = []
    tag2state = {'O_C': 'NONE', 'O_D': 'NONE', 'C': 'CREATE', 'E': 'NONE', 'M': 'MOVE', 'D': 'DESTROY'}

    for i in range(num_sents):
        state_tag = tag2state[pred_state_seq[i]]
        prediction.append( hard_constraint(state_tag, pred_loc_seq[i], pred_loc_seq[i+1]) )

    return prediction


def hard_constraint(state: str, loc_before: str, loc_after: str):
    """
    Some commonsense hard constraints on the predictions.
    P.S. These constraints are only defined for evaluation, not for state sequence prediction.
    1. For state NONE, loc_after must be the same with loc_before
    2. For state MOVE and DESTROY, loc_before must not be '-'.
    3. For state CREATE, loc_before should be '-'.
    """
    if state == 'NONE' and loc_before != loc_after:
        if loc_after == '-':
            state = 'DESTROY'
        else:
            print('WHAT THE HELL?')
        # no other possibility
    if state == 'MOVE' and loc_before == '-':
        state = 'CREATE'
    # if state == 'MOVE' and loc_before == loc_after:
    #     state = 'NONE'
    if state == 'DESTROY' and loc_before == '-':
        state = 'NONE'
    if state == 'CREATE' and loc_before != '-':
        if loc_before == loc_after:
            state = 'NONE'
        elif loc_before != loc_after:
            state = 'MOVE'
    return state, loc_before, loc_after


def predict_consistent_loc(pred_state_seq: List[str], pred_loc_seq: List[str], para_id: int, entity: str):
    """
    1. Only keep the location predictions at state "C" or "M"
    2. For "O_C", "O_D", and "D", location should be "-"
    3. For "E", location should be the same with previous timestep
    4. For state0: if state1 is "E", "M" or "D", then state0 should be "?";
       if state1 is "O_C", "O_D" or "C", then state0 should be "-"
    """
    num_sents = len(pred_state_seq)
    consist_state_seq = []
    consist_loc_seq = []

    for sent_i in range(num_sents):

        state = pred_state_seq[sent_i]
        location = pred_loc_seq[sent_i + 1]

        #if 'D' is followed by a 'D'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'D' and state == 'D':
            state = 'E'

        # if 'D' is followed by a 'M'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'M' and state == 'D':
            state = 'E'

        # if 'E' is followed by a 'O_D'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'O_D' and state in ['E', 'M']:
            state = 'D'

        # if 'C' is followed by a 'C'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'C' and state == 'C':
            state = 'O_C'

        # if the state before O_C is not O_C
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'O_C' and state != 'O_C':
            temp_idx = sent_i + 1
            while temp_idx != num_sents and pred_state_seq[temp_idx] in ['O_C', 'O_D']:
                temp_idx += 1
            # pred_state_seq[temp_idx]: first state after O_C
            # state: last state before O_C
            if temp_idx != num_sents and pred_state_seq[temp_idx] == 'C':
                for idx in range(0, sent_i):
                    consist_state_seq[idx] = 'O_C'
                    consist_loc_seq[idx] = '-'
                state = 'O_C'
                if sent_i > 0:
                    consist_loc_seq[sent_i] = '-'
            else:
                for idx in range(sent_i+1, temp_idx):
                    pred_state_seq[idx] = 'E'

        # if 'O_C' is followed by a 'E'
        if sent_i > 0 and consist_state_seq[sent_i - 1] == 'O_C' and state == 'E':
            temp_idx = sent_i + 1
            while temp_idx != num_sents and pred_state_seq[temp_idx] == 'E':
                temp_idx += 1

            if temp_idx != num_sents and pred_state_seq[temp_idx] == 'C':
                for idx in range(sent_i, temp_idx):
                    pred_state_seq[idx] = 'O_C'
                state = 'O_C'
            else:
                state = 'C'

        # if 'O_C' is followed by a 'D'
        if sent_i > 0 and consist_state_seq[sent_i - 1] == 'O_C' and state == 'D' :
            for idx in range(0, sent_i):
                if pred_state_seq[idx] != 'O_C':
                    raise ValueError
                else:
                    consist_state_seq[idx] = 'E'
                    consist_loc_seq[idx] = pred_loc_seq[0]
            consist_loc_seq[sent_i] = pred_loc_seq[0]

        # set location according to state
        if sent_i == 0:
            location_0 = predict_loc0(state1 = state, loc0 = pred_loc_seq[0])
            consist_loc_seq.append(location_0)

        if state in ['O_C', 'O_D', 'D']:
            cur_location = '-'
        elif state == 'E':
            cur_location = consist_loc_seq[sent_i]  # this is the previous location since we add a location_0
        elif state in ['C', 'M']:
            cur_location = location

        consist_state_seq.append(state)
        consist_loc_seq.append(cur_location)

    return consist_state_seq, consist_loc_seq


def predict_loc0(state1: str, loc0: str) -> str:
    if state1 in ['E', 'M', 'D']:
        loc0 = loc0
    elif state1 in ['O_C', 'O_D', 'C']:
        loc0 = '-'

    return loc0

def get_koala_fix(preds):
    items = dict()
    for pred in preds:
        if pred[0] not in items:
            items[pred[0]] = {"para_id": int(pred[0]), "entities": dict()}

        item = items[pred[0]]
        if pred[2] not in item['entities']:
            item['entities'][pred[2]] = {"locations": [], "states": []}

        eitem = item['entities'][pred[2]]
        eitem['locations'].append(pred[5])
        if pred[3] != "-":
            eitem['states'].append(pred[3])

    for pred in preds:
        try:
            assert items[pred[0]]['entities'][pred[2]]['locations'][int(pred[1])] == pred[5]
        except:
#             print(items[pred[0]])
            print(pred)
            print(items[pred[0]]['entities'][pred[2]]['locations'][int(pred[1])])
            raise
            
        if pred[1] != '0':
            assert items[pred[0]]['entities'][pred[2]]['states'][int(pred[1]) - 1] == pred[3]

    for para_id, item in items.items():
        for entity in item['entities']:
            results = get_output(entity, item['para_id'], item['entities'][entity]['states'], item['entities'][entity]['locations'])
            item['entities'][entity]['results'] = results
            
    final_data = []
    for key, item in items.items():
        for entity in item['entities']:
            for step, res in enumerate(item['entities'][entity]['results']['prediction']):
                final_data.append({"para_id": item['para_id'], "step": step + 1, "entity": entity, "action": res[0], "before": res[1], "after": res[2]})
    
    return final_data