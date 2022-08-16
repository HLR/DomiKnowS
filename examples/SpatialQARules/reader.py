import json
import tqdm

VOCABULARY = {
    "LEFT": ["to the left of"],
    "RIGHT": ["to the right of"],
    "ABOVE": ["above"],
    "BELOW": ["below"],
    "BEHIND": ["behind"],
    "FRONT": ["in front of"],
    "NEAR": ["near to"],
    "FAR": ["far from"],
    "DC": "disconnected from",
    "EC": "touch[es]",
    "PO": "overlap[s]",
    "TPP": ["covered by"],
    "NTPP": ["inside"],
    "TPPI": ["cover[s]"],
    "NTPPI": ["contain[s]"]
}


def create_key(obj1, obj2, relation):
    key = str(obj1) + ":" + str(obj2) + ":" + str(relation)
    return key


def create_simple_question(obj1, obj2, relation, obj_info):
    return "Is " + obj_info[obj1]["full_name"] + " " + \
           (VOCABULARY[relation][0] if isinstance(VOCABULARY[relation], list) else VOCABULARY[relation]) \
           + " " + obj_info[obj2]["full_name"] + "?"


def train_reader(file, question_type, size=None, upward_level=0):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 300000 if not size else size
    print("level:", upward_level)

    dataset = []
    count = 0
    count_original = 0
    all_batch_dynamic_info = {}
    for story in data["data"]:
        story_txt = story['story'][0]
        facts_info = story['facts_info']
        obj_info = story["objects_info"]
        relation_info = {}
        question_id = {}
        run_id = 0

        for question in story["questions"]:
            if count >= size:
                break
            question_txt = question["question"]

            q_type = question["q_type"]
            if q_type != question_type:
                continue

            candidates = question['candidate_answers']
            target_relation = question['question_info']['target_relation'][0].upper()
            asked_relation = question['question_info']['asked_relation'][0].upper()
            count_original += 1
            obj1, obj2 = question['query']
            target_question = (obj1, obj2, target_relation)
            asked_question = (obj1, obj2, asked_relation)
            current_key = create_key(*asked_question)

            added_questions = []
            deep_level = upward_level

            if current_key not in question_id:
                question_id[current_key] = run_id
                run_id += 1
            added_questions.append((question_txt, question['answer'][0], current_key))

            if question['answer'][0] == "No":
                target_key = create_key(*target_question)
                added_questions.append((create_simple_question(*target_question, obj_info), "Yes", target_key))

                if target_key not in question_id:
                    question_id[target_key] = run_id
                    run_id += 1
                relation_info[current_key] = "reverse," + str(question_id[target_key])

                deep_level -= 1

            current_level = [target_question]
            for _ in range(deep_level):
                new_level = []
                for current_fact in current_level:

                    current_key = create_key(*current_fact)
                    previous_ids = []
                    if current_key not in question_id:
                        question_id[current_key] = run_id
                        run_id += 1
                    previous_facts = facts_info[current_key]["previous"]

                    for previous in previous_facts:
                        previous_key = create_key(*previous)
                        if previous_key not in question_id:
                            question_id[previous_key] = run_id
                            run_id += 1
                        previous_ids.append(question_id[previous_key])
                        new_level.append(previous)
                        added_questions.append((create_simple_question(*previous, obj_info), "Yes", previous_key))
                        current_level = new_level

                    size_relation = len(previous_ids)
                    if size_relation == 0:
                        relation_info[current_key] = ""
                        continue
                    relation_type = "symmetric" if size_relation == 1 \
                        else "transitve" if size_relation == 2 \
                        else "transitive_topo"
                    for previous_id in previous_ids:
                        relation_type += "," + str(previous_id)
                    relation_info[current_key] = relation_type

            if len(added_questions) not in all_batch_dynamic_info:
                all_batch_dynamic_info[len(added_questions)] = 0
            all_batch_dynamic_info[len(added_questions)] += 1

            # dataset.append(added_questions[::-1])
            batch_question = []
            for added_question, label, question_key in added_questions[::-1]:
                batch_question.append((added_question, story_txt, q_type,
                                       candidates,
                                       relation_info[question_key] if question_key in relation_info else "",
                                       label, question_id[question_key]))
                count += 1
            dataset.append(batch_question)

    print("Original questions", count_original)
    print("Total questions", count)
    print(all_batch_dynamic_info)
    # Return Type need to be list of dict with name of variable as key
    return dataset


def DomiKnowS_reader(file, question_type, size=None, upward_level=0, train=True, batch_size=8):
    dataset = train_reader(file, question_type, size, upward_level) if train else None  # TODO: Fix this later

    return_dataset = []
    current_batch_size = 0
    batch_data = {'questions': [], 'stories': [], 'relation': [], 'labels': [], "question_ids": []}
    if train:
        for batch in tqdm.tqdm(dataset):
            if current_batch_size + len(batch) > batch_size and current_batch_size != 0:
                current_batch_size = 0
                return_dataset.append({"questions": "@@".join(batch_data['questions']),
                                       "stories": "@@".join(batch_data['stories']),
                                       "relation": "@@".join(batch_data['relation']),
                                       "question_ids": "@@".join(batch_data['question_ids']),
                                       "labels": "@@".join(batch_data['labels'])})
                batch_data = {'questions': [], 'stories': [], 'relation': [], 'labels': [], "question_ids": []}
            for data in batch:
                question_txt, story_txt, q_type, candidates_answer, relation, label, id = data
                batch_data["questions"].append(question_txt)
                batch_data["stories"].append(story_txt)
                batch_data["relation"].append(relation)
                batch_data["question_ids"].append(str(id))
                batch_data["labels"].append(label)
            current_batch_size += len(batch)
        if current_batch_size != 0:
            return_dataset.append({"questions": "@@".join(batch_data['questions']),
                                   "stories": "@@".join(batch_data['stories']),
                                   "relation": "@@".join(batch_data['relation']),
                                   "question_ids": "@@".join(batch_data['question_ids']),
                                   "labels": "@@".join(batch_data['labels'])})

    return return_dataset


if __name__ == "__main__":
    # Maximum deep for target = 7 + 1 (No answer) right now batch size is dynamic for training (It needs to cover all
    # related fact) <- Right now can add the option later For training batch size if fix
    dataset = DomiKnowS_reader("DataSet/train_with_rules.json", "YN", upward_level=8, batch_size=8)
    print(len(dataset))
