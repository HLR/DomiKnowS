import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import json
import numpy as np
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from sg_model import SceneGraphModel
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import exactL
from itertools import chain
from domiknows.sensor.pytorch.sensors import FunctionalSensor
from domiknows.program import SolverPOIProgram
import torch
from tqdm import tqdm

from domiknows import setProductionLogMode
setProductionLogMode(no_UseTimeLog=True)


# load is_a constraints
isa_file = '../../data/knowledge_base/is_a.facts'

isa_facts = {}

with open(isa_file) as f_in:
    for line in f_in:
        child_n, parent_n = line.split('\t')
        isa_facts[child_n.strip()] = parent_n.strip()

# load metadata
meta_info = json.load(open('../../data/gqa_info.json', 'r'))
name1_meta = json.load(open('./rescources/new_label_data/labels/name1.json', 'r'))
name2_meta = json.load(open('./rescources/new_label_data/labels/name2.json', 'r'))
name3_meta = json.load(open('./rescources/new_label_data/labels/name3.json', 'r'))
name4_meta = json.load(open('./rescources/new_label_data/labels/name4.json', 'r'))

name_meta = {
    1: name1_meta,
    2: name2_meta,
    3: name3_meta,
    4: name4_meta
}

# load features
def fake_features():
    import pickle
    with open('./keys.pkl', 'rb') as f_in:
        keys = pickle.load(f_in)
    print(len(keys))
    return {
        k: np.zeros((2048,), dtype=np.float32) for k in tqdm(keys[:100])
    }

obj_feats = np.load('../../data/features.npy', allow_pickle=True).item()
#obj_feats = fake_features()
feat_dim = list(obj_feats.values())[0].shape[0]
print(feat_dim)

sg_model_dict = SceneGraphModel(
    feat_dim=feat_dim,
    n_names=meta_info['name']['num'],
    n_attrs=meta_info['attr']['num'],
    n_rels=meta_info['rel']['num'],
    device='cuda:0',
    model_dir='./model_ckpts_sg_test/'
)

sg_model = {
    1: sg_model_dict.models["name1"].eval(),
    2: sg_model_dict.models["name2"].eval(),
    3: sg_model_dict.models["name3"].eval(),
    4: sg_model_dict.models["name4"].eval()
}

name_to_obj = {}

name_to_idx = {
    1: {},
    2: {},
    3: {},
    4: {}
}

with Graph('graph') as graph:
    name = Concept(name='name')

    for depth, meta in name_meta.items():
        for name_idx, name_str in meta.items():
            name_idx = int(name_idx)

            name_concept = Concept(name=name_str)

            name_to_obj[name_str] = name_concept
            name_to_idx[depth][name_str] = name_idx
    
            if depth == 1:
                name_concept.is_a(name)

    not_added_names = list(name_to_obj.keys())

    for depth in range(1, 5):
        exactL(*[name_to_obj[name_str] for name_str in name_meta[depth].values()])

    for child_name, parent_name in isa_facts.items():
        if child_name in name_to_obj and parent_name in name_to_obj:
            not_added_names.remove(child_name)

            name_to_obj[child_name].is_a(name_to_obj[parent_name])
            #print(f'{child_name} < {parent_name}')
        else:
            pass
            #print(f'skipping {child_name} < {parent_name}')
    
    for name_str in not_added_names:
        name_to_obj[name_str].is_a(name)

    print(not_added_names)

from sg_data_loader import SceneGraphLoader

sg_test_loader = SceneGraphLoader(
    meta_info=meta_info,
    data_f='./rescources/new_label_data/val.pickle',
    obj_feats=obj_feats,
    type='name',
    batch_size=1,
    drop_last=False
)
def row_to_dataitem(row):
    result = {}
    result['features'] = row[0]
    
    for depth in range(1, 5):
        result[f'labels_{depth}'] = row[depth + 1]

    return result

def label_to_binary(index):
    def _label_to_binary(label):
        #print(index, label[0], torch.equal(label[0], torch.tensor(int(index))))
        return [int(int(index) == label.cpu().item())]
    return _label_to_binary

def logits_to_binary(index):
    def _logits_to_binary(logits):
        probs = torch.softmax(logits, dim=-1)
        
        result = torch.tensor([1 - probs[:, int(index)], probs[:, int(index)]])

        result = torch.log(result)
        
        return result.unsqueeze(0)

    return _logits_to_binary

name['features'] = ReaderSensor(keyword='features')

for depth in range(1, 5):
    name[f'labels_{depth}'] = ReaderSensor(keyword=f'labels_{depth}')
    name[f'logits_{depth}'] = ModuleLearner('features', module=sg_model[depth])

for depth in range(1, 5):
    for name_idx, name_str in name_meta[depth].items():
        name[name_to_obj[name_str]] = FunctionalSensor(
            f'logits_{depth}',
            forward=logits_to_binary(name_idx)
        )

        name[name_to_obj[name_str]] = FunctionalSensor(
            f'labels_{depth}',
            forward=label_to_binary(name_idx),
            label=True
        )

suffixes = ['local/argmax', 'ILP']

program = SolverPOIProgram(
    graph,
    poi=(name,),
    inferTypes=suffixes,
    metric={}
)

all_results = {
    s: {
        depth: [] for depth in range(1, 5)
    } for s in suffixes + ['prob', 'label']
}

for ex in tqdm(sg_test_loader):
    data_item = row_to_dataitem(ex)
    
    node = program.populate_one(data_item)

    result = {
        s: {
            depth: torch.zeros((len(name_meta[depth].values()),)) for depth in range(1, 5)
        } for s in suffixes + ['prob', 'label']
    }

    for depth in range(1, 5):
        for idx, name_str in name_meta[depth].items():
            idx = int(idx)

            attributes = node.getAttributes()
            
            prob = attributes[f'<{name_str}>'][1]
            result['prob'][depth][idx] = prob

            lbl = attributes[f'<{name_str}>/label'][0]
            if lbl == 1:
                result['label'][depth][idx] = 1

            if 'local/argmax' in suffixes:
                argmax_key = f'<{name_str}>/local/argmax'

                assert argmax_key in attributes
                pred = torch.argmax(attributes[argmax_key])
                result['local/argmax'][depth][idx] = pred.cpu().item()
            
            if 'ILP' in suffixes:
                ILP_key = f'<{name_str}>/ILP'
                if ILP_key in attributes:
                    pred = attributes[ILP_key]
                    result['ILP'][depth][idx] = int(pred.cpu().item())

        label = ex[depth + 1]

        for s in suffixes + ['label', 'prob']:
            print(f'{s}\t| depth: {depth}, pred: {torch.argmax(result[s][depth])}, label: {label.item()}')
            all_results[s][depth].append(torch.argmax(result[s][depth]).item())

        print(f'prob\t| depth: {depth}, pred: {torch.argmax(result["prob"][depth])}, label: {label.item()}')
        
        print(f'label\t| depth: {depth}, pred: {torch.argmax(result["label"][depth])}, label: {label.item()}')

    with open('results_all.json', 'w') as file_out:
        json.dump(all_results, file_out)
