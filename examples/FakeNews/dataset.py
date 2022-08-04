import pandas as pd

def load_annodata(anno_data):
    anno_data = pd.read_csv(anno_data)
    final_reader = []
    anno_dict = {}
    for idx, row in anno_data.iterrows():
        text = row["Texts"]
        binary_label = row["hasAnno"]
        parent_texts = [text] * 12
        parent_labels = row[4:16].tolist()

        anno_dict["Text"] = text
        anno_dict["Label"] = binary_label
        anno_dict["Parent Text"] = [parent_texts]
        anno_dict["Parent Labels"] = [parent_labels]
        
        final_reader.append(anno_dict)
        anno_dict = {}

    return final_reader