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

RELATIONS_NAME = {
    'Located_In': 'located_in',
    'Work_For': 'work_for',
    'OrgBased_In': 'orgbase_on',
    'Live_In': 'live_in',
    'Kill': 'kill'
}

def create_query(question, question_type="YN"):

    # Prepare the entities to ask
    if question_type == "Counting-Entity-Relation":
        asked_entities = ""
        all_obj = []
        define_character = ord("a")
        for entity_relation in question["entity_asking"]:
            entity1, rel1, entity2 = entity_relation.split("-")
            entity1 = ENTITIES_NAME[entity1]
            entity2 = ENTITIES_NAME[entity2]
            rel1 = RELATIONS_NAME[rel1]
            object_name = f"andL({entity1}('{chr(define_character)}'), {rel1}('{chr(define_character+1)}', path=('{chr(define_character)}', rel_pair_phrase1.reversed)), {entity2}('{chr(define_character+2)}', path=('{chr(define_character+1)}', rel_pair_phrase2)))"
            print(object_name)
            define_character += 3
            all_obj.append(object_name)
        asked_entities += ",".join(all_obj)
    else:
        asked_entities = ",".join([ENTITIES_NAME[entity] for entity in question["entity_asking"]])

    asked_type = question["count_ask"]
    
    # For Counting questions, the label is the actual count number
    if question_type == "Counting" or question_type == "Counting-Entity-Relation":
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
    if "conllQA2" in data_path:
        question_type = "Counting-Entity-Relation"
    
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
            relations = data.get('relations', [])  # Get relations if they exist

            all_pos_relations = [["" for _ in range(len(entities))] for _ in range(len(entities))]
            for relation in relations:
                head = relation['head']
                tail = relation['tail']
                rel_text = relation['type']
                all_pos_relations[head][tail] = rel_text

            relation = []
            for i in range(len(entities)):
                for j in range(len(entities)):
                    if all_pos_relations[i][j] == "":
                        continue
                    cur_rel = (all_pos_relations[i][j], i, j)
                    relation.append(cur_rel)

            index = 0
            label = []
            tokens = []
            for entity in entities:
                label.extend(['O'] * (entity['start'] - index))
                tokens.extend(data["tokens"][index: entity['start']])
                index = entity['end']
                label.append(entity['type'])
                tokens.append("/".join(data['tokens'][entity['start']: entity['end']]))

            # Add remaining tokens after last entity
            if index < len(data['tokens']):
                label.extend(['O'] * (len(data['tokens']) - index))
                tokens.extend(data['tokens'][index:])

            # Process relations if they exist
            relation_labels = []
            if relations:
                for rel in relations:
                    head_idx = rel['head']
                    tail_idx = rel['tail']
                    rel_type = rel['type']
                    relation_labels.append({
                        'head': head_idx,
                        'tail': tail_idx,
                        'type': rel_type
                    })

            if 'qa_questions' not in data or not data["qa_questions"]:
                continue

            str_query, label_query, asked_number = create_query(data["qa_questions"][0], question_type)
            if str_query == "":
                continue

            # if asked_number == 0 or asked_number == len(tokens):
            #     continue

            current_portion.append({
                "tokens": tokens,
                "label": label,
                "relations": relation_labels,  # Add relations to output
                "relation": relation,
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
    conll4_reader("conllQA2.json", "entities_with_relation")