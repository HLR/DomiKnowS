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

def create_query(question, question_type):

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


def conll4_reader(data_path, dataset_portion, asking_type=None):
    # Validate asking_type if provided
    if asking_type is not None:
        valid_types = list(ASKING_TYPE.keys())
        if asking_type not in valid_types:
            raise ValueError(f"Invalid asking_type '{asking_type}'. Must be one of: {valid_types}")        
    
    with open(data_path, 'r') as f:
        full_data = json.load(f)["data"]

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

    # Track which ASKING_TYPEs have been logged
    logged_asking_types = set()
    print("\n=== Example Queries by ASKING_TYPE ===")

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

            # Build tokens/labels first so entity_to_phrase_idx is populated
            # before the relation loop uses it.
            index = 0
            label = []
            tokens = []
            entity_to_phrase_idx = {}  # map entity array index → phrase list index
            for ent_idx, entity in enumerate(entities):
                num_o_before = entity['start'] - index
                label.extend(['O'] * num_o_before)
                tokens.extend(data["tokens"][index: entity['start']])
                entity_to_phrase_idx[ent_idx] = len(tokens)  # phrase idx for this entity
                index = entity['end']
                label.append(entity['type'])
                tokens.append("/".join(data['tokens'][entity['start']: entity['end']]))

            # Add remaining tokens after last entity
            if index < len(data['tokens']):
                label.extend(['O'] * (len(data['tokens']) - index))
                tokens.extend(data['tokens'][index:])

            relation = []
            for i in range(len(entities)):
                for j in range(len(entities)):
                    if all_pos_relations[i][j] == "":
                        continue
                    # Use phrase indices so filter_pairs can match DataNode instanceIDs
                    phrase_i = entity_to_phrase_idx.get(i, i)
                    phrase_j = entity_to_phrase_idx.get(j, j)
                    cur_rel = (all_pos_relations[i][j], phrase_i, phrase_j)
                    relation.append(cur_rel)

            # Build relation data for CompositionCandidateReaderSensor's filter_pairs.
            # Must be a list of dicts with 'head'/'tail' keys (phrase indices).
            relation_labels = []
            if relations:
                for rel in relations:
                    head_ent_idx = rel['head']
                    tail_ent_idx = rel['tail']
                    rel_type = rel['type']
                    h = entity_to_phrase_idx.get(head_ent_idx, head_ent_idx)
                    t = entity_to_phrase_idx.get(tail_ent_idx, tail_ent_idx)
                    relation_labels.append({
                        'head': h,
                        'tail': t,
                        'type': rel_type
                    })

            if 'qa_questions' not in data or not data["qa_questions"]:
                continue

            str_query, label_query, asked_number = create_query(data["qa_questions"][0], data["question_type"])
            if str_query == "":
                continue

            # if asked_number == 0 or asked_number == len(tokens):
            #     continue

            # Get the asking type from the question
            question_asking_type = data["qa_questions"][0]["count_ask"]
            
            # Filter by asking_type if specified
            if asking_type is not None and question_asking_type != asking_type:
                continue

            # Log example queries for each ASKING_TYPE (only once per type)
            if question_asking_type not in logged_asking_types:
                logged_asking_types.add(question_asking_type)
                print(f"\n{question_asking_type} ({ASKING_TYPE[question_asking_type]}):")
                print(f"  Question: {data['qa_questions'][0].get('question', 'N/A')}")
                print(f"  Query: {str_query}")
                print(f"  Label: {label_query}")
                print(f"  Question Type: {data['question_type']}")

            # Generate structured entity labels as lists (one per phrase)
            # Convert entity labels to class indices: people=0, organization=1, location=2, other=3, o=4
            entity_type_mapping = {
                'Peop': 0,  # people
                'Org': 1,   # organization
                'Loc': 2,   # location
                'Other': 3, # other
                'O': 4      # o (no entity)
            }
            
            entity_class_labels = []
            for entity_label in label:
                if entity_label in entity_type_mapping:
                    entity_class_labels.append(entity_type_mapping[entity_label])
                else:
                    # Default to 'other' for unknown types
                    entity_class_labels.append(3)
            
            # Generate structured relation labels as a dictionary
            # Key: (head_idx, tail_idx), Value: relation_class_index
            # Relation mapping: work_for=0, located_in=1, live_in=2, orgbase_on=3, kill=4
            relation_type_mapping = {
                'work_for': 0,
                'located_in': 1,
                'live_in': 2,
                'orgbase_on': 3,
                'kill': 4
            }
            
            relation_class_labels = {}
            for rel_info in relation:
                rel_type, head_phrase_idx, tail_phrase_idx = rel_info
                rel_name = RELATIONS_NAME.get(rel_type, None)
                if rel_name and rel_name in relation_type_mapping:
                    # Keys are phrase indices (matching DataNode instanceIDs)
                    relation_class_labels[(head_phrase_idx, tail_phrase_idx)] = relation_type_mapping[rel_name]

            # Create individual label fields for each phrase (phrase_0_entity_label, phrase_1_entity_label, etc.)
            data_item = {
                "tokens": tokens,
                "label": label,
                "relations": relation_labels,
                "relation": relation,
                "logic_str": str_query,
                "logic_label": label_query,
                # Add structured labels for supervised learning
                "entity_labels": entity_class_labels,  # List of class indices, one per phrase
                "relation_labels": relation_class_labels  # Dict: (head, tail) -> class_index
            }
            
            # Add individual entity label fields: phrase_0_entity_label, phrase_1_entity_label, etc.
            for phrase_idx, entity_class in enumerate(entity_class_labels):
                data_item[f"phrase_{phrase_idx}_entity_label"] = entity_class
            
            # Add individual relation label fields: pair_0_0_relation_label, pair_0_1_relation_label, etc.
            # Use a counter for pair indices to make them sequential
            pair_idx = 0
            for (head_idx, tail_idx), relation_class in relation_class_labels.items():
                data_item[f"pair_{head_idx}_{tail_idx}_relation_label"] = relation_class
                pair_idx += 1
            
            current_portion.append(data_item)

        if portion == "train":
            train = copy.deepcopy(current_portion)
        elif portion == "validation":
            dev = copy.deepcopy(current_portion)
        else:
            test = copy.deepcopy(current_portion)
    
    print(f"\n=== Query Logging Summary ===")
    print(f"Logged ASKING_TYPEs: {', '.join(sorted(logged_asking_types))}")
    missing_types = set(ASKING_TYPE.keys()) - logged_asking_types
    if missing_types:
        print(f"Missing ASKING_TYPEs: {', '.join(sorted(missing_types))}")
    print("=" * 35 + "\n")
    
    return train, dev, test


if __name__ == '__main__':
    conll4_reader("conllQA_with_global.json", "entities_with_relation")