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
    "AtLeast": "atLeastAL"
}

ENTITIES_NAME = {
    "Org": "organization",
    "Peop": "people",
    "Loc": "location"
}


def create_query(question, question_type="YN"):
    if question_type != "YN":
        raise Exception("Only Support YN Question Currently")
    asked_number = int(re.findall(r'[+-]?\d+', question['question'])[0])
    asked_type = question["count_ask"]
    asked_entity = ",".join([ENTITIES_NAME[entity] for entity in question["entity_asking"]])
    str_query = f"{ASKING_TYPE[asked_type]}({asked_entity}, {asked_number})"
    label = [int(question["label"] == "YES")]
    return str_query, label, asked_number


def conll4_reader(data_path, dataset_portion):
    with open(data_path, 'r') as f:
        dataset = json.load(f)[dataset_portion]

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

            str_query, label_query, asked_number = create_query(data["qa_questions"][0])
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
