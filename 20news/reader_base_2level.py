import torch
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def collate_label_set(data):
    level1 = [
        "comp.os", "comp.sys", "comp.windows", "comp.graphics", "rec.motorcycles", "rec.sport", "rec.autos", "talk.religion",
        "sci.electronics", "sci.med", "sci.space", "misc.forsale", "talk.politics", "sci.crypt", "alt.atheism", "soc.religion", "None"
    ]
    level2_pass = {
        "comp.windows", "comp.os", "talk.religion", "soc.religion"
    }
    level2 = [
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
            else:
                label = label.split(".")
                label_two = ".".join(label[:2])
                if label_two not in level1:
                    print(label[0])
                else:
                    item['level1'] = level1.index(label_two)
                if len(label) > 2:
                    if label_two in level2_pass:
                        item['level2'] = level2.index("None")
                    else:
                        if label[2] not in level2:
                            print(label[2])
                            item['level2'] = level2.index("None")
                        else:
                            item['level2'] = level2.index(label[2])
                else:
                    item['level2'] = level2.index("None")
        except:
            print(label, item['text'])
            raise
            
    final_data = {"id": None, "text": None, "level1": None, "level2": None}
    final_data['id'] = torch.tensor([item['id'] for item in data])
    final_data['text'] = [item['text'] for item in data]
    final_data['level1'] = torch.tensor([item['level1'] for item in data])
    final_data['level2'] = torch.tensor([item['level2'] for item in data])
    try:
        x = tokenizer.batch_encode_plus(final_data['text'], return_tensors="pt", padding="longest", max_length=512, truncation=True)
    except:
        print(final_data['text'])
        raise
    for key in x:
        final_data[key] = x[key]
        
    return final_data