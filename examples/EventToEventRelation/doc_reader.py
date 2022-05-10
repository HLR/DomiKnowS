import os
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')
from xml.etree import ElementTree
import nltk
from nltk.tokenize import sent_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")


# Original Code from document_reader.py in

def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span


def id_lookup(span_SENT, start_char):
    # this function is applicable to RoBERTa subword or token from ltf/spaCy
    # id: start from 0
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= start_char:
            return token_id
    raise ValueError("Nothing is found.")


def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    # token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        # assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        # token_count += 1
    return token_span_DOC


def RoBERTa_list(content, token_list=None, token_span_SENT=None):
    encoded = tokenizer.encode(content)
    roberta_subword_to_ID = encoded
    # input_ids = torch.tensor(encoded).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    roberta_subwords = []
    roberta_subwords_no_space = []
    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        roberta_subwords.append(r_token)
        if r_token[0] == " ":
            roberta_subwords_no_space.append(r_token[1:])
        else:
            roberta_subwords_no_space.append(r_token)

    roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1])  # w/o <s> and </s>
    roberta_subword_map = []
    if token_span_SENT is not None:
        roberta_subword_map.append(-1)  # "<s>"
        for subword in roberta_subword_span:
            roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
        roberta_subword_map.append(-1)  # "</s>"
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
    else:
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1


def sent_id_lookup(my_dict, start_char, end_char=None):
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']


def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index


########################################## REIMPLEMENT ABOVE ##############################################

def read_matres_relation(MATRES_file, events_to_relation, file_to_event_trigger):
    with open(MATRES_file, "r") as file:
        for line in file:
            file_name, trigger1, trigger2, eiid1, eiid2, tempRel = line.strip().split("\t")
            eiid1, eiid2 = int(eiid1), int(eiid2)
            if file_name not in events_to_relation:
                events_to_relation[file_name] = {}
                file_to_event_trigger[file_name] = {}
            if eiid1 not in events_to_relation[file_name]:
                file_to_event_trigger[file_name][eiid1] = trigger1
            if eiid2 not in events_to_relation[file_name]:
                file_to_event_trigger[file_name][eiid2] = trigger2
            events_to_relation[file_name][(eiid1, eiid2)] = tempRel


def read_matres_info(dirname, file_name, events_to_relation, file_to_event_trigger):
    return_dict = {}
    return_dict["event"] = {}
    return_dict["eiid"] = {}
    return_dict["doc_id"] = file_name.replace(".tml", "")
    tree = ElementTree.parse(os.path.join(dirname, file_name))
    root = tree.getroot()
    for makeinstance in root.findall('MAKEINSTANCE'):
        attributes = makeinstance.attrib
        eID = attributes["eventID"]
        eiid = int(attributes["eiid"][2:])
        # Confirm existance of document and event trigger
        if return_dict["doc_id"] in file_to_event_trigger:
            if eiid in file_to_event_trigger[return_dict["doc_id"]]:
                # Using dictionary for looking up between eID and eiid
                return_dict["event"][eID] = {"eiid": eiid, "word": file_to_event_trigger[return_dict["doc_id"]][eiid]}
                return_dict["eiid"][eiid] = {"eID": eID}

    # Cleaning up the text provided in original code from document_reader.py in JointConstrainLearning Paper
    #
    content = root.find("TEXT")
    MY_STRING = str(ElementTree.tostring(content))
    start = MY_STRING.find("<TEXT>") + 6
    end = MY_STRING.find("</TEXT>")
    MY_TEXT = MY_STRING[start:end]
    while MY_TEXT[0] == " ":
        MY_TEXT = MY_TEXT[1:]
    MY_TEXT = MY_TEXT.replace("\\n", " ")
    MY_TEXT = MY_TEXT.replace("\\'", "\'")
    MY_TEXT = MY_TEXT.replace("  ", " ")
    MY_TEXT = MY_TEXT.replace(" ...", "...")

    # ========================================================
    #    Load position of events, in the meantime replacing
    #    "<EVENT eid="e1" class="OCCURRENCE">turning</EVENT>"
    #    with "turning"
    # ========================================================
    while MY_TEXT.find("<") != -1:
        start = MY_TEXT.find("<")
        end = MY_TEXT.find(">")
        if MY_TEXT[start + 1] == "E":
            event_description = MY_TEXT[start:end].split(" ")
            eID = (event_description[1].split("="))[1].replace("\"", "")
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
            if eID in return_dict["event"].keys():
                return_dict["event"][eID]["start_char"] = start  # loading position of events
        else:
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]

    # Text processing part
    return_dict["content"] = MY_TEXT
    return_dict["sentences"] = []
    return_dict["relations"] = {}
    sent_tokenized_text = sent_tokenize(return_dict["content"])
    sent_span = tokenized_to_origin_span(return_dict["content"], sent_tokenized_text)
    count_sent = 0
    # PART BELOW IS COPYING DIRECTLY FROM PROJECT
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])

        sent_dict["roberta_subword_span_DOC"] = \
            span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])

        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

        return_dict["sentences"].append(sent_dict)

        # Add sent_id as an attribute of event
    for event_id, event_dict in return_dict["event"].items():
        return_dict["event"][event_id]["sent_id"] = sent_id = \
            sent_id_lookup(return_dict, event_dict["start_char"])
        return_dict["event"][event_id]["token_id"] = \
            id_lookup(return_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        return_dict["event"][event_id]["roberta_subword_id"] = \
            id_lookup(return_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1
        # updated on Mar 20, 2021
    return return_dict


def matres_reader():
    path_TB = os.path.join("data/MATRES/TBAQ-cleaned/TimeBank")
    TB_files = set([file for file in os.listdir(path_TB) if os.path.isfile(os.path.join(path_TB, file))])
    path_AQ = os.path.join("data/MATRES/TBAQ-cleaned/AQUAINT")
    AQ_files = set([file for file in os.listdir(path_AQ) if os.path.isfile(os.path.join(path_AQ, file))])
    path_PL = os.path.join("data/MATRES/te3-platinum")
    PL_files = set([file for file in os.listdir(path_PL) if os.path.isfile(os.path.join(path_PL, file))])
    MATRES_TB = os.path.join("data/MATRES/timebank.txt")
    MATRES_AQ = os.path.join("data/MATRES/aquaint.txt")
    MATRES_PL = os.path.join("data/MATRES/platinum.txt")
    events_to_relation = {}
    file_to_event_trigger = {}
    # Reading relation from MATRES dataset
    read_matres_relation(MATRES_TB, events_to_relation, file_to_event_trigger)
    read_matres_relation(MATRES_AQ, events_to_relation, file_to_event_trigger)
    read_matres_relation(MATRES_PL, events_to_relation, file_to_event_trigger)
    # Reading text information from original file
    for file in file_to_event_trigger.keys():
        file_name = file + ".tml"
        dirname = path_TB if file_name in TB_files else \
            path_AQ if file_name in AQ_files else \
                path_PL if file_name in PL_files else None
        if dirname is None:
            continue
        result_dict = read_matres_info(dirname, file_name, events_to_relation, file_to_event_trigger)
        print(result_dict)
        break


if __name__ == "__main__":
    matres_reader()
