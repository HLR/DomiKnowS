import json
import tqdm
import random

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

LABELS_INT = {
    "LEFT": 1,
    "RIGHT": 2,
    "ABOVE": 4,
    "BELOW": 8,
    "BEHIND": 16,
    "FRONT": 32,
    "NEAR": 64,
    "FAR": 128,
    "DC": 256,
    "EC": 512,
    "PO": 1024,
    "TPP": 2048,
    "NTPP": 4096,
    "TPPI": 8192,
    "NTPPI": 16384
}


def create_key(obj1, obj2, relation, question_type):
    if question_type == "YN":
        return str(obj1) + ":" + str(obj2) + ":" + relation
    return str(obj1) + ":" + str(obj2)


def create_simple_question(obj1, obj2, relation, obj_info, question_type):
    if question_type == "YN":
        return "Is " + obj_info[obj1]["full_name"] + " " + \
               (VOCABULARY[relation][0] if isinstance(VOCABULARY[relation], list) else VOCABULARY[relation]) \
               + " " + obj_info[obj2]["full_name"] + "?"

    question_fr1 = "Where is {:} relative to the {:}?".format(obj_info[obj1]["full_name"],
                                                              obj_info[obj2]["full_name"])
    question_fr2 = "What is the position of the {:} regarding {:}".format(obj_info[obj1]["full_name"],
                                                                          obj_info[obj2]["full_name"])
    return question_fr1 if random.random() < 0.5 else question_fr2


def label_fr_to_int(labels: list):
    result = 0
    for label in labels:
        result += LABELS_INT[label.upper()]
    return result


def train_reader(file, question_type, *, limit_questions=300000, upward_level=0):
    with open(file) as json_file:
        data = json.load(json_file)
    print("level:", upward_level)
    print("USING THIS")
    dataset = []
    count_questions = 0
    count_original = 0
    all_batch_dynamic_info = {}
    for story in data["data"]:
        story_txt = story['story'][0]
        facts_info = story['facts_info']
        obj_info = story["objects_info"]
        relation_info = {}
        question_id = {}
        run_id_within_q = 0

        for question in story["questions"]:
            if count_questions >= limit_questions:
                break

            question_txt = question["question"]

            q_type = question["q_type"]
            if q_type != question_type:
                continue

            candidates = question['candidate_answers']

            # Finding the target relation (Can be more than 1?)
            target_relation = question['question_info']['target_relation'][0] \
                if isinstance(question['question_info']['target_relation'], list) \
                else question['question_info']['target_relation']
            target_relation = target_relation.upper()

            # Finding the asked relation (Can be more than 1?)
            asked_relation = question['question_info']['asked_relation'][0] \
                if isinstance(question['question_info']['asked_relation'], list) \
                else question['question_info']['asked_relation']
            asked_relation = asked_relation.upper()

            count_original += 1
            obj1, obj2 = question['query']
            target_question = (obj1, obj2, target_relation)
            asked_question = (obj1, obj2, asked_relation)
            current_key = create_key(*asked_question, question_type)

            added_questions = []  # questions to be added to the model
            reasoning_steps_from_target = upward_level

            # Create question id of current answer
            if current_key not in question_id:
                question_id[current_key] = run_id_within_q
                run_id_within_q += 1

            if question_type == "YN":
                label = question["answer"][0]
            else:
                label = label_fr_to_int(question["answer"])

            added_questions.append((question_txt, label, current_key))

            if question_type == "YN":
                # If the answer of question is no, adding another question asking the same thing but "Yes" input
                if question['answer'][0].lower() == "no":
                    target_key = create_key(*target_question, question_type)
                    added_questions.append((create_simple_question(*target_question, obj_info, question_type),
                                            "Yes",
                                            target_key))

                    if target_key not in question_id:
                        question_id[target_key] = run_id_within_q
                        run_id_within_q += 1
                    relation_info[current_key] = "reverse," + str(question_id[target_key])

                    reasoning_steps_from_target -= 1

            current_level = [target_question]
            for _ in range(reasoning_steps_from_target):
                new_level = []
                for current_fact in current_level:

                    current_key = create_key(*current_fact, question_type)
                    fact_info_key = create_key(*current_fact, "")
                    previous_ids = []
                    if current_key not in question_id:
                        question_id[current_key] = run_id_within_q
                        run_id_within_q += 1
                    previous_facts = facts_info[fact_info_key][current_fact[2]]["previous"]

                    for previous in previous_facts:
                        previous_key = create_key(*previous, question_type)
                        fact_info_prev_key = create_key(*previous, "")
                        if previous_key not in question_id:
                            question_id[previous_key] = run_id_within_q
                            run_id_within_q += 1
                        previous_ids.append(str(question_id[previous_key]))
                        new_level.append(previous)
                        if question_type == "YN":
                            added_questions.append((create_simple_question(*previous, obj_info, question_type),
                                                    "Yes",
                                                    previous_key))
                        else:
                            added_questions.append((create_simple_question(*previous, obj_info, question_type),
                                                    label_fr_to_int(list(facts_info[fact_info_prev_key].keys())),
                                                    previous_key))
                        current_level = new_level

                    size_relation = len(previous_ids)
                    if size_relation == 0:
                        relation_info[current_key] = ""
                        continue
                    relation_type = "symmetric" if size_relation == 1 \
                        else "transitive" if size_relation == 2 \
                        else "transitive_topo"
                    relation_type = relation_type + ',' + ','.join(previous_ids)
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
                count_questions += 1
            dataset.append(batch_question)

    print("Original questions", count_original)
    print("Total questions", count_questions)
    print(all_batch_dynamic_info)
    # Return Type need to be list of dict with name of variable as key
    return dataset


def general_reader(file, question_type, size=None):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 10 ** 6 if not size else size

    dataset = []
    count = 0
    for story in data["data"]:
        story_txt = " ".join(story['story'])
        question_id = {}
        run_id = 0

        for question in story["questions"]:
            if count >= size:
                break
            question_txt = question["question"]

            q_type = question["q_type"]
            if q_type != question_type:
                continue
            if q_type == "YN":
                # Variable need
                candidates = question['candidate_answers']
                # asked_relation = question['question_info']['asked_relation'][0] \
                #     if isinstance(question['question_info']['asked_relation'], list) \
                #     else question['question_info']['asked_relation']
                # asked_relation = asked_relation.upper()
                # obj1, obj2 = question['query']
                # asked_question = (obj1, obj2, asked_relation)
                # current_key = create_key(*asked_question)
                label = question["answer"][0]
                if label == "DK":
                    label = "No"
                dataset.append([[question_txt, story_txt, q_type, candidates, "", label, run_id]])
                run_id += 1
                count += 1
            elif q_type == "FR":
                # Variable need
                candidates = question['candidate_answers']
                # asked_relation = question['question_info']['asked_relation'][0] \
                #     if isinstance(question['question_info']['asked_relation'], list) \
                #     else question['question_info']['asked_relation']
                # asked_relation = asked_relation.upper()
                # obj1, obj2 = question['query']
                # asked_question = (obj1, obj2, asked_relation)
                # current_key = create_key(*asked_question)
                label = question["answer"]
                dataset.append([[question_txt, story_txt, q_type, candidates, "", label_fr_to_int(label), run_id]])
                run_id += 1
                count += 1

    return dataset


def RESQ_reader(file, question_type, size=None, reasoning=None):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 300000 if not size else size

    dataset = []
    count = 0
    for story in data["data"]:
        story_txt = " ".join(story['story'])
        run_id = 0
        for question in story["questions"]:
            if count >= size:
                break
            if reasoning is not None:
                if reasoning == 0 and isinstance(question["step_of_reasoning"], int):
                    continue
                if reasoning != 0 and question["step_of_reasoning"] != reasoning:
                    continue
            question_txt = question["question"]
            candidates = question['candidate_answers']
            label = question["answer"][0] if question["answer"][0] != "DK" else "NO"
            dataset.append([[question_txt, story_txt, "YN", candidates, "", label, run_id]])
            run_id += 1
            count += 1

    return dataset


def boolQ_reader(file, size=None):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 300000 if not size else size

    dataset = []
    for story in data["data"][:size]:
        story_txt = story['passage'][:1000]
        run_id = 0
        question_txt = story['question']
        # Variable need
        candidates = ["Yes", "No"]
        label = story['answer']
        dataset.append([[question_txt, story_txt, "YN", candidates, "", label, run_id]])
        run_id += 1
    return dataset


def StepGame_reader(prefix, train_dev_test="train", size=None, file_number=None):
    if train_dev_test == "train":
        files = ["train.json"]
    elif train_dev_test == "dev":
        if file_number is None:
            files = ["qa" + str(i + 1) + "_valid.json" for i in range(5)]
        else:
            files = ["qa" + str(file_number + 1) + "_valid.json"]
    else:
        if file_number is None:
            files = ["qa" + str(i + 1) + "_test.json" for i in range(10)]
        else:
            files = ["qa" + str(file_number + 1) + "_test.json"]

    dataset = []
    print(prefix, files)
    for file in files:
        with open(prefix + "/" + file) as json_file:
            data = json.load(json_file)
        size = 300000 if not size else size
        run_id = 0
        for story_ind in list(data)[:size]:
            story = data[story_ind]
            story_txt = " ".join(story["story"])

            question_txt = story["question"]
            # Variable need
            candidates = ["left", "right", "above", "below", "lower-left",
                          "lower-right", "upper-left", "upper-right", "overlap"]
            label = story["label"]
            dataset.append([[question_txt, story_txt, "FR", candidates, "", label, run_id]])
            run_id += 1

    return dataset


def DomiKnowS_reader(file, question_type, size=300000, *,
                     type_dataset=None,
                     upward_level=0,
                     augmented=True,
                     batch_size=8,
                     rule_text=False,
                     reasoning_steps=None,
                     STEPGAME_status="train"):
    if type_dataset == "STEPGAME":
        dataset = StepGame_reader(file, STEPGAME_status, size, file_number=reasoning_steps)
    elif type_dataset == "BOOLQ":
        dataset = boolQ_reader(file, size)
    elif type_dataset == "RESQ":
        dataset = RESQ_reader(file, size, reasoning=reasoning_steps)
    elif augmented:  # Refer to SPARTUN with chain of reasoning when training
        dataset = train_reader(file, question_type, limit_questions=size, upward_level=upward_level)
    else:
        dataset = general_reader(file, question_type, size)

    additional_text = ""
    if rule_text:
        with open("DataSet/rules.txt", 'r') as rules:
            additional_text = rules.readline()
    return_dataset = []
    current_batch_size = 0
    count_question = 0
    batch_data = {'questions': [], 'stories': [], 'relation': [], 'labels': [], "question_ids": []}
    for batch in tqdm.tqdm(dataset, desc="Reading " + file + " " + (
            str(STEPGAME_status) if STEPGAME_status is not None else "")):
        count_question += len(batch)
        # Checking each batch have same story, prevent mixing IDs
        check_same_story = current_batch_size != 0 and batch[0][1] == batch_data["stories"][0]
        if (current_batch_size + len(batch) > batch_size) and current_batch_size != 0:
            current_batch_size = 0
            return_dataset.append({"questions": "@@".join(batch_data['questions']),
                                   "stories": "@@".join(batch_data['stories']),
                                   "relation": "@@".join(batch_data['relation']),
                                   "question_ids": "@@".join(batch_data['question_ids']),
                                   "labels": "@@".join(batch_data['labels'])})
            batch_data = {'questions': [], 'stories': [], 'relation': [], 'labels': [], "question_ids": []}
        for data in batch:
            question_txt, story_txt, q_type, candidates_answer, relation, label, id = data
            batch_data["questions"].append(question_txt + additional_text)
            batch_data["stories"].append(story_txt)
            batch_data["relation"].append(relation)
            batch_data["question_ids"].append(str(id))
            batch_data["labels"].append(str(label))
        current_batch_size += len(batch)
    if current_batch_size != 0:
        return_dataset.append({"questions": "@@".join(batch_data['questions']),
                               "stories": "@@".join(batch_data['stories']),
                               "relation": "@@".join(batch_data['relation']),
                               "question_ids": "@@".join(batch_data['question_ids']),
                               "labels": "@@".join(batch_data['labels'])})
    print("Total question:", count_question)
    return return_dataset


if __name__ == "__main__":
    # print(os.path.abspath("."))

    # dataset = train_reader("DataSet/train_v3.json", "FR", limit_questions=300000, upward_level=100)
    datasets = DomiKnowS_reader("DataSet/train_v3.json", "FR", upward_level=14, augmented=True, batch_size=8)
    print(len(datasets))
    print(datasets[0])
