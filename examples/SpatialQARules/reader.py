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


def create_key(obj1, obj2, relation):
    key = str(obj1) + ":" + str(obj2) + ":" + str(relation)
    return key


def create_simple_question(obj1, obj2, relation, obj_info):
    return "Is " + obj_info[obj1]["full_name"] + " " + \
           (VOCABULARY[relation][0] if isinstance(VOCABULARY[relation], list) else VOCABULARY[relation]) \
           + " " + obj_info[obj2]["full_name"] + "?"


def label_fr_to_int(labels: list):
    result = 0
    for label in labels:
        result += LABELS_INT[label]
    return result


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

            target_relation = question['question_info']['target_relation'][0] \
                if isinstance(question['question_info']['target_relation'], list) \
                else question['question_info']['target_relation']
            target_relation = target_relation.upper()

            asked_relation = question['question_info']['asked_relation'][0] \
                if isinstance(question['question_info']['asked_relation'], list) \
                else question['question_info']['asked_relation']
            asked_relation = asked_relation.upper()

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
                        else "transitive" if size_relation == 2 \
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

    # print("Original questions", count_original)
    # print("Total questions", count)
    # print(all_batch_dynamic_info)
    # Return Type need to be list of dict with name of variable as key
    return dataset


def test_reader(file, question_type, size=None):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 300000 if not size else size

    dataset = []
    count = 0
    for story in data["data"]:
        story_txt = story['story'][0]
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


def StepGame_reader(prefix, train_dev_test="train", size=None):
    if train_dev_test == "train":
        files = ["train.json"]
    elif train_dev_test == "dev":
        files = ["qa" + str(i + 1) + "_valid.json" for i in range(5)]
    else:
        files = ["qa" + str(i + 1) + "_test.json" for i in range(10)]

    dataset = []
    for file in files:
        with open(prefix+ "/" + file) as json_file:
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


def DomiKnowS_reader(file, question_type, size=None, upward_level=0, augmented=True, boolQL=False, batch_size=8,
                     rule=False, StepGame_status=None):
    dataset = StepGame_reader(file, StepGame_status, size) if StepGame_status \
        else train_reader(file, question_type, size, upward_level) if augmented \
        else boolQ_reader(file, size) if boolQL else test_reader(file, question_type, size)
    additional_text = ""
    if rule:
        with open("DataSet/rules.txt", 'r') as rules:
            additional_text = rules.readline()
    return_dataset = []
    current_batch_size = 0
    batch_data = {'questions': [], 'stories': [], 'relation': [], 'labels': [], "question_ids": []}
    for batch in tqdm.tqdm(dataset, desc="Reading " + file + " " + (str(StepGame_status) if StepGame_status is not None else "")):
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

    return return_dataset
