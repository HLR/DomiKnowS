import torch
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def collate_label_set(data):
    level1 = [ 
        "comp", "rec", "sci", "misc", "talk", "alt", "soc", "None"
    ]
    level2 = [
        "os", "sys", "windows", "graphics", "motorcycles", "sport", "autos", "religion",
        "electronics", "med", "space", "forsale", "politics", "crypt", "None"
    ]
    level2_pass = {
        "soc", "alt" 
    }
    level3_pass = {
        "windows", "os", "religion"
    }
    level3 = [
        "misc", "guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"
    ]
    for item in data:
        if item['text'] is None:
            item['text'] = " "
        try:
            label = item['label']
            if not label:
                label = "None"
                item['level1'] = level1.index(label)
                item['level2'] = level2.index("None")
                item['level3'] = level3.index("None")
            else:
                label = label.split(".")
                if label[0] not in level1:
                    print(label[0])
                item['level1'] = level1.index(label[0])
                if len(label) > 1:
                    if label[0] in level2_pass:
                        item['level2'] = level2.index("None")
                        item['level3'] = level3.index("None")
                    else:
                        if label[1] not in level2:
                            print(label[1])
                            item['level2'] = level2.index("None")
                            item['level3'] = level3.index("None")
                        else:
                            item['level2'] = level2.index(label[1])
                            if len(label) > 2:
                                if label[1] in level3_pass:
                                    item['level3'] = level3.index("None")
                                else:
                                    if label[2] not in level3:
                                        print(label[2])
                                        item['level3'] = level3.index("None")
                                    else:
                                        item['level3'] = level3.index(label[2])
                            else:
                                item['level3'] = level3.index("None")
                else:
                    item['level2'] = level2.index("None")
                    item['level3'] = level3.index("None")
        except:
            print(label, item['text'])
            raise
            
    final_data = {"id": None, "text": None, "level1": None, "level2": None, "level3": None}
    final_data['id'] = torch.tensor([item['id'] for item in data])
    final_data['text'] = [item['text'] for item in data]
    final_data['level1'] = torch.tensor([item['level1'] for item in data])
    final_data['level2'] = torch.tensor([item['level2'] for item in data])
    final_data['level3'] = torch.tensor([item['level3'] for item in data])
    try:
        x = tokenizer.batch_encode_plus(final_data['text'], return_tensors="pt", padding="longest", max_length=512, truncation=True)
    except:
        print(final_data['text'])
        raise
    for key in x:
        final_data[key] = x[key]
        
    return final_data