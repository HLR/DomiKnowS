import pandas as pd
from sklearn.utils import resample


def DataReader(file, size, *, batch_size = 16):
    df = pd.read_csv(file)
    return_data = []
    count = 0
    accumulate_data = {'premise': [], 'hypothesis': [], 'entailment': [], 'neutral': [], 'contradiction': []}
    sample = resample(df, n_samples=size, stratify=df['label']) if size != len(df) else df
    for index, item in sample.iterrows():
        data = {'premise': [item['premise']], 'hypothesis': [item['hypothesis']],
                'entailment': 1 if item['label'] == 0 else 0,
                'neutral': 1 if item['label'] == 1 else 0,
                'contradiction': 1 if item['label'] == 2 else 0
                }
        return_data.append(data)
    #     if count == batch_size or (len(accumulate_data['premise']) != 0 and accumulate_data['premise'][-1] != item['premise']):
    #         return_data.append(accumulate_data)
    #         count = 0
    #         accumulate_data = {'premise': [], 'hypothesis': [], 'useful': [], 'useless': []}
    #     accumulate_data['premise'].append(item['premise'])
    #     accumulate_data['hypothesis'].append(item['hypothesis'])
    #     accumulate_data['useful'].append(1 if item['label'] == 0 else 0)
    #     accumulate_data['useless'].append(1 if item['label'] != 0 else 0)
    #     count += 1
    #
    # if len(accumulate_data['premise']):
    #     return_data.append(accumulate_data)

    return return_data
