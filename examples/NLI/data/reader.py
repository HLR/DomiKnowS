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


def DataReaderMulti(file, size, *, batch_size=8, adver_data_set=False, adver_file=""):
    df = pd.read_json(file, lines=True).dropna()
    # Default will make the size equal to the maximum size of data
    size = min(size, df.shape[0])
    return_data = []
    sample = df.iloc[:size, :]
    # Doing the basic batch size without relationship, first
    current_size = 0
    data = {'premise': [], 'hypothesis': [], 'entailment': [], 'contradiction': [], 'neutral': []}

    for index, item in sample.iterrows():
        data['premise'].append(item["sentence1"])
        data['hypothesis'].append(item["sentence2"])
        data['entailment'].append('1' if item['gold_label'] == "entailment" else '0')
        data['neutral'].append('1' if item['gold_label'] == "neutral" else '0')
        data['contradiction'].append('1' if item['gold_label'] == "contradiction" else '0')
        current_size += 1
        if current_size == batch_size:
            current_size = 0
            return_data.append({"premises": "@@".join(data['premise']),
                                "hypothesises": "@@".join(data['hypothesis']),
                                "entailment_list": "@@".join(data['entailment']),
                                "contradiction_list": "@@".join(data['contradiction']),
                                "neutral_list": "@@".join(data['neutral'])})
            # Reset data
            for key in data.keys():
                data[key] = []
    # Using Adversarially Regularising dataset from
    # https://arxiv.org/pdf/1808.08609.pdf
    adv_df = pd.read_json(adver_file, lines=True).dropna() if adver_data_set else pd.Series()
    for index, item in adv_df.iterrows():
        data['premise'].append(item["sentence1"])
        data['hypothesis'].append(item["sentence2"])
        data['entailment'].append('1' if item['gold_label'] == "entailment" else '0')
        data['neutral'].append('1' if item['gold_label'] == "neutral" else '0')
        data['contradiction'].append('1' if item['gold_label'] == "contradiction" else '0')
        current_size += 1
        if current_size == batch_size:
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