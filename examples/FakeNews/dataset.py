import pandas as pd

def load_annodata(anno_data):
    anno_data = pd.read_csv(anno_data, encoding="utf-8")
    final_reader = []
    anno_dict = {}
    for idx, row in anno_data.iterrows():
        text = row["Texts"]
        binary_label = [row["noAnno"], row["hasAnno"]]
        parent_labels = [1]+row[4:16].astype(int).tolist()

        anno_dict["Text"] = [text]
        anno_dict["Label"] = [binary_label]
        anno_dict["Parent Labels"] = [parent_labels]
        
        final_reader.append(anno_dict)
        anno_dict = {}
    return final_reader
