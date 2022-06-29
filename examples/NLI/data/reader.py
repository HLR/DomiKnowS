import pandas as pd
from sklearn.utils import resample


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

    df = pd.read_csv(file).dropna() if file else None
    df_augment = pd.read_json(augment_file, lines=True).dropna() if augment_file else None
    # Default will make the size equal to the maximum size of data
    size = min(size, df.shape[0]) if file else 0
    return_data = []
    # Doing the basic batch size without relationship, first
    current_size = 0
    data = {'premise': [], 'hypothesis': [], 'entailment': [], 'contradiction': [], 'neutral': []}
    index = 0
    data_id_sample = {}
    sample = df.iloc[:size, :] if file else None
    if file:
        for _, item in sample.iterrows():
            data_id_sample[index] = (item, False)
            index += 1
    count = 0
    if augment_file:
        for _, item in df_augment.iterrows():
            data_id_sample[index] = (item, True)
            index += 1
            count += 1
            if count == 4:
                break

    symmetric = {}
    check_id = {}
    for id, pair in data_id_sample.items():
        item, augment_data = pair
        premise = "premise" if not augment_data else "sentence1"
        hypothesis = "hypothesis" if not augment_data else "sentence2"
        pre = item[premise]
        hypo = item[hypothesis]
        key = pre + ',' + hypo if pre <= hypo else hypo + ',' + pre
        if key not in symmetric:
            symmetric[key] = []
        symmetric[key].append(id)

    for group_sym in symmetric.values():
        for ind, id in enumerate(group_sym):
            if id in check_id:
                continue
            check_id[id] = True
            item, augment_data = data_id_sample[id]
            premise = "premise" if not augment_data else "sentence1"
            hypothesis = "hypothesis" if not augment_data else "sentence2"
            data['premise'].append(item[premise])
            data['hypothesis'].append(item[hypothesis])
            if not augment_data:
                data['entailment'].append('1' if item['label'] == 0 else '0')
                data['contradiction'].append('1' if item['label'] == 2 else '0')
                data['neutral'].append('1' if data['entailment'][-1] == '0' and data['contradiction'][-1] == '0' else '0')
            else:
                data['entailment'].append('1' if item['gold_label'] == "entailment" else '0')
                data['contradiction'].append('1' if item['gold_label'] == "contradiction" else '0')
                data['neutral'].append('1' if data['entailment'][-1] == '0' and data['contradiction'][-1] == '0' else '0')
            current_size += 1
            # To prevent separation between symmetric sentence
            if ind + 3 == len(group_sym) or current_size == batch_size:
                current_size = 0
                return_data.append({"premises": "@@".join(data['premise']),
                                    "hypothesises": "@@".join(data['hypothesis']),
                                    "entailment_list": "@@".join(data['entailment']),
                                    "contradiction_list": "@@".join(data['contradiction']),
                                    "neutral_list": "@@".join(data['neutral'])})
                # Reset data
                for key in data.keys():
                    data[key] = []
    if current_size != 0:
        return_data.append({"premises": "@@".join(data['premise']),
                            "hypothesises": "@@".join(data['hypothesis']),
                            "entailment_list": "@@".join(data['entailment']),
                            "contradiction_list": "@@".join(data['contradiction']),
                            "neutral_list": "@@".join(data['neutral'])})
    return return_data
