import json
import torch
import re
import spacy
import copy
import numpy as np

nlp = spacy.load('en_core_web_sm')  # English()

# label: list based on token
# relation: token * token

ASKING_TYPE = {
    "AtMost": "atMostAL",
    "Exactly": "exactAL",
    "AtLeast": "atLeastAL",
    "counting": "sumL"
}

ENTITIES_NAME = {
    "Org": "organization",
    "Peop": "people",
    "Loc": "location"
}


def create_query(question, question_type="YN"):
    
    asked_entities = ",".join([ENTITIES_NAME[entity] for entity in question["entity_asking"]])
    asked_type = question["count_ask"]
    
    # For Counting questions, the label is the actual count number
    if question_type == "Counting":
        asked_number = question["label"]  # The label IS the number for counting
        label = [asked_number]  # Return the count as label
    # For YN questions
    elif question_type == "YN":
        asked_number = int(re.findall(r'[+-]?\d+', question['question'])[0])
        label = [int(question["label"] == "YES")]
    else:
        raise Exception("Only Support YN and Counting Question Currently")
    
    str_query = f"{ASKING_TYPE[asked_type]}({asked_entities}, {asked_number})"
    return str_query, label, asked_number


def conll4_reader(data_path, dataset_portion):
    with open(data_path, 'r') as f:
        full_data = json.load(f)
    
    # Determine question type from portion name
    question_type = "Counting" if "Counting" in dataset_portion else "YN"
    
    # Check if the file is an extracted single-portion file
    # Extracted files have a single key that matches the portion name
    if len(full_data) == 1 and dataset_portion in full_data:
        dataset = full_data[dataset_portion]
        print(f"Using extracted portion file: {data_path}")
    elif dataset_portion in full_data:
        dataset = full_data[dataset_portion]
        print(f"Using portion '{dataset_portion}' from full data file: {data_path}")
    else:
        # Assume it's already an extracted file containing only the portion data directly
        dataset = full_data
        print(f"Using direct portion data from: {data_path}")

    train = []
    test = []
    dev = []
    for portion in ["train", "validation", "test"]:
        current_portion = []
        for data in dataset[portion]:
            entities = data['entities']

            index = 0
            label = []
            tokens = []
            for entity in entities:
                label.extend(['O'] * (entity['start'] - index))
                tokens.extend(data["tokens"][index: entity['start']])
                index = entity['end']
                label.append(entity['type'])
                tokens.append("/".join(data['tokens'][entity['start']: entity['end']]))

            str_query, label_query, asked_number = create_query(data["qa_questions"][0], question_type)
            if str_query == "":
                continue

            # if asked_number == 0 or asked_number == len(tokens):
            #     continue

            current_portion.append({
                "tokens": tokens,
                "label": label,
                "logic_str": str_query,
                "logic_label": label_query
            })

        if portion == "train":
            train = copy.deepcopy(current_portion)
        elif portion == "validation":
            dev = copy.deepcopy(current_portion)
        else:
            test = copy.deepcopy(current_portion)
    return train, dev, test


if __name__ == '__main__':
    conll4_reader("conllQA.json", "entities_only_with_1_things_YN")