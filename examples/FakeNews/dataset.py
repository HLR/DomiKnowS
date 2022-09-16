import pandas as pd
import numpy as np


def load_annodata(anno_data):
    anno_data = pd.read_csv(anno_data, encoding="utf-8")
    final_reader = []
    anno_dict = {}
    for idx, row in anno_data.iterrows():
        labels = [row["noAnno"], row["hasAnno"]]
        anno_dict["Label"] = [np.array(labels).argmax(axis=0)]
        parent_labels = row[4:16].astype(int).tolist()
        parent_labels += [1 if sum(parent_labels) == 0 else 0]
        anno_dict["Parent Labels"] = [np.array(parent_labels).argmax(axis=0)]

        sub_labels = row[16:].astype(int).tolist()
        sub_labels += [1 if sum(sub_labels) == 0 else 0]
        anno_dict["Sub Labels"] = [np.array(sub_labels).argmax(axis=0)]
        anno_dict["Text"] = [row["Texts"]]
        final_reader.append(anno_dict)
        anno_dict = {}
    return final_reader
