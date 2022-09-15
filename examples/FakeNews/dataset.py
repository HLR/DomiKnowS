import pandas as pd

def load_annodata(anno_data):
    anno_data = pd.read_csv(anno_data, encoding="utf-8")
    final_reader = []
    anno_dict = {}
    for idx, row in anno_data.iterrows():
        anno_dict["Text"] = [row["Texts"]]
        anno_dict["Label"] = [[row["noAnno"], row["hasAnno"]]]
        anno_dict["Parent Labels"] = [row[4:16].astype(int).tolist()]
        anno_dict["Sub Labels"] = [row[16:].astype(int).tolist()]
        final_reader.append(anno_dict)
        anno_dict = {}
    return final_reader
