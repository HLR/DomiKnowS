import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm


def DataReader(file, size):
    df = pd.read_csv(file).dropna()
    return_data = []
    sample = resample(df, n_samples=size, stratify=df['label']) if size != len(df) else df
    for index, item in sample.iterrows():
        data = {'premise': [item['premise']], 'hypothesis': [item['hypothesis']],
                'entailment': 1 if item['label'] == 0 else 0,
                'neutral': 1 if item['label'] == 1 else 0,
                'contradiction': 1 if item['label'] == 2 else 0
                }
        return_data.append(data)
    return return_data


def DataReaderMultiRelation(file, size, *, batch_size=8, augment_file=None):
    if file is None and augment_file is None:
        return None

    def append_data_from_id(cur_id, cur_data, data_from_id, check):
        if cur_id in check:
            return
        check[cur_id] = True
        current_data, is_augmented = data_from_id[cur_id]
        premise_key = "premise" if not is_augmented else "sentence1"
        hypothesis_key = "hypothesis" if not is_augmented else "sentence2"
        cur_data['premise'].append(item[premise_key])
        cur_data['hypothesis'].append(item[hypothesis_key])

        if not augment_data:
            current_label = item['label'] if item['label'] != -1 else 1
        else:
            current_label = 0 if item['gold_label'] == "entailment" else \
                2 if item['gold_label'] == "contradiction" else 1

        cur_data['label'].append(str(current_label))
        return

    target_batch = None
    df = pd.read_csv(file).dropna() if file else None
    df_augment = pd.read_json(augment_file, lines=True).dropna() if augment_file else None
    # Default will make the size equal to the maximum size of data
    size = min(size, df.shape[0]) if file else 0
    return_data = []
    # Doing the basic batch size without relationship, first
    current_size = 0
    data = {'premise': [], 'hypothesis': [], 'label': []}
    index = 0
    data_id_sample = {}
    sample = df.iloc[:size, :] if file else None


    # Making combined of files, False -> Original dataset, True -> Augmented proposed by reference paper
    if file:
        for _, item in sample.iterrows():
            data_id_sample[index] = (item, False)
            index += 1
    count = 0
    if augment_file:
        for _, item in df_augment.iterrows():
            data_id_sample[index] = (item, True)
            index += 1

    symmetric_check = {}
    transitive_check = {}
    check_id = {}
    total_size = 0
    # Creating key from symmetric pair
    for id, pair in data_id_sample.items():
        item, augment_data = pair
        premise = "premise" if not augment_data else "sentence1"
        hypothesis = "hypothesis" if not augment_data else "sentence2"
        pre = item[premise]
        hypo = item[hypothesis]
        # If any pair have the same
        key = pre + ',' + hypo if pre <= hypo else hypo + ',' + pre
        if key not in symmetric_check:
            symmetric_check[key] = []
        if pre not in transitive_check:
            transitive_check[pre] = {}
        if hypo not in transitive_check[pre]:
            transitive_check[pre][hypo] = []
        transitive_check[pre][hypo].append(id)
        symmetric_check[key].append(id)
    for group_sym in tqdm(symmetric_check.values()):
        for ind, id in enumerate(group_sym):
            if id in check_id:
                continue
            item, augment_data = data_id_sample[id]
            premise = "premise" if not augment_data else "sentence1"
            hypothesis = "hypothesis" if not augment_data else "sentence2"
            current_premise = item[premise]
            current_hypo = item[hypothesis]
            append_data_from_id(id, data, data_id_sample, check_id)
            current_size += 1
            # If there is any pair start with current hypothesis (Potential transitive)
            if current_hypo in transitive_check:
                for last_hypothesis in transitive_check[current_hypo]:
                    if last_hypothesis in transitive_check[current_premise]:
                        for cur_id in transitive_check[current_hypo][last_hypothesis]:
                            if cur_id in check_id:
                                continue
                            append_data_from_id(cur_id, data, data_id_sample, check_id)
                            current_size += 1
                        for cur_id in transitive_check[current_premise][last_hypothesis]:
                            if cur_id in check_id:
                                continue
                            append_data_from_id(cur_id, data, data_id_sample, check_id)
                            current_size += 1

        if current_size >= batch_size:
            total_size += current_size
            current_size = 0
            return_data.append({"premises": "@@".join(data['premise']),
                                "hypothesises": "@@".join(data['hypothesis']),
                                "label_list": "@@".join(data['label'])})
            # Reset data
            for key in data.keys():
                data[key] = []
    if current_size != 0:
        total_size += current_size
        return_data.append({"premises": "@@".join(data['premise']),
                            "hypothesises": "@@".join(data['hypothesis']),
                            "label_list": "@@".join(data['label'])})
    return return_data
