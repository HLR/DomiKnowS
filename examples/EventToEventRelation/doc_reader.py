import os
from xml.etree import ElementTree
import nltk
import re
from nltk.tokenize import sent_tokenize
import spacy
import tqdm
import torch
from models import *
from utils import *

nlp = spacy.load("en_core_web_sm")
tokenlize = Roberta_Tokenizer()


# Original Code from document_reader.py in

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
    MY_STRING = str(ElementTree.tostring(root))
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
            eID_index = 0
            for i, event_attr in enumerate(event_description):
                if re.search('eid\w*', event_attr):
                    eID_index = i
                    break
            eID = (event_description[eID_index].split("="))[1].replace("\"", "")
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
            tokenlize(sent_dict["content"], sent_dict["token_span_SENT"])

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
    for file in tqdm.tqdm(file_to_event_trigger.keys()):
        file_name = file + ".tml"
        dirname = path_TB if file_name in TB_files else \
            path_AQ if file_name in AQ_files else \
                path_PL if file_name in PL_files else None
        if dirname is None:
            continue
        # TODO: Creating the train and test file from this
        # Events to relation
        result_dict = read_matres_info(dirname, file_name, events_to_relation, file_to_event_trigger)
        eiid_to_event_trigger = file_to_event_trigger[file]
        all_event_pairs = events_to_relation[file]
        train_set = []
        for eiid1, eiid2 in all_event_pairs:
            # TODO: Check which variable need to be used in the model to train from original paper
            x = result_dict["eiid"][eiid1]["eID"]
            y = result_dict["eiid"][eiid2]["eID"]

            x_sent_id = result_dict["event"][x]["sent_id"]
            y_sent_id = result_dict["event"][y]["sent_id"]

            x_sent = padding(result_dict["sentences"][x_sent_id]["roberta_subword_to_ID"])  # A
            y_sent = padding(result_dict["sentences"][y_sent_id]["roberta_subword_to_ID"])  # B

            x_position = result_dict["event"][x]["roberta_subword_id"]  # A_pos
            y_position = result_dict["event"][y]["roberta_subword_id"]  # B_pos

            x_sent_pos = padding(result_dict["sentences"][x_sent_id]["roberta_subword_pos"], pos=True)
            y_sent_pos = padding(result_dict["sentences"][y_sent_id]["roberta_subword_pos"], pos=True)
            # Related relation need to group together for constrain learning
    # TODO: Continue on this after finishing the DomiKnows Model

    # Notes:
    #     TB files goto training set
    #     Others goto testing set
    #     break
    # return train_set, test_set


if __name__ == "__main__":
    matres_reader()
