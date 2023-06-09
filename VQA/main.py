import argparse
import os,sys
import torch
import torch.nn as nn
import numpy as np
currentdir = os.path.dirname(os.getcwd())
root = os.path.dirname(currentdir)
sys.path.append(root)


from domiknows.program.model_program import SolverPOIProgram, SolverModel
from domiknows.program.model.gbi import GBIModel
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram, GBIProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.sensor.pytorch.learners import ModuleLearner
from model import Net

# Enable skeleton DataNode
def main(device, models):
    from graph import graph, image_group_contains, image, level1, level2, level3, level4, image_group, structure

    image_group['reps'] = FunctionalReaderSensor(keyword='reps', forward=lambda data: data.unsqueeze(0) ,device=device)

    image[image_group_contains, "reps"] = JointSensor(image_group['reps'], forward=lambda x: (torch.ones(x.shape[1], 1), x.squeeze(0)))

    def get_label(*inputs, data):
        data = data.unsqueeze(-1)
        return data

    image['features'] = ReaderSensor(keyword='features')

    image[level1] = ModuleLearner(image_group_contains, "reps", "features", module=Net(models, 1), label=False)
    image[level1] = FunctionalReaderSensor(keyword='level1_label', forward=get_label, label=True)

    image[level2] = ModuleLearner(image_group_contains, "reps", "features", module=Net(models, 2), label=False)
    image[level2] = FunctionalReaderSensor(keyword='level2_label', forward=get_label, label=True)

    image[level3] = ModuleLearner(image_group_contains, "reps", "features", module=Net(models, 3), label=False)
    image[level3] = FunctionalReaderSensor(keyword='level3_label', forward=get_label, label=True)

    image[level4] = ModuleLearner(image_group_contains, "reps", "features", module=Net(models, 4), label=False)
    image[level4] = FunctionalReaderSensor(keyword='level4_label', forward=get_label, label=True)

    f = open("logger.txt", "w")
    program = GBIProgram(graph, SolverModel,
        inferTypes=[
            'local/argmax', 'GBI'
        ],
        poi = (image_group, image, level1, level2, level3, level4),
        metric={
            # 'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))
        },
        f=f
    )

    # class CallbackProgram(SampleLossProgram):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self.after_train_epoch = []

    #     def call_epoch(self, name, dataset, epoch_fn, **kwargs):
    #         if name == 'Testing':
    #             for fn in self.after_train_epoch:
    #                 fn(kwargs)
    #         else:
    #             super().call_epoch(name, dataset, epoch_fn, **kwargs)

    # program = CallbackProgram(
    #     graph,
    #     SolverModel,
    #     inferTypes=[
    #         'local/argmax'
    #     ],
    #     poi = (image_group, image, level1, level2, level3, level4),
    #     loss=MacroAverageTracker(NBCrossEntropyLoss()),
    #     metric={
    #         # 'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))
    #     },
    #     f=f
    # )

    return program


if __name__ == '__main__':
    from domiknows.utils import setProductionLogMode
    productionMode = True
    if productionMode:
        setProductionLogMode(no_UseTimeLog=True)
    from domiknows.utils import setDnSkeletonMode
    setDnSkeletonMode(True)
    import logging
    logging.basicConfig(level=logging.INFO)

    from torch.utils.data import Dataset, DataLoader
    from graph import *

    from sg_data_loader import SceneGraphLoader
    import json

    import numpy as np
    from functools import partial
    from tqdm import tqdm

    from model import init_model

    """
    Train Config
    """
    do_train = False
    train_batch_size = 32

    """
    Eval Config
    """
    checkpoint_dir = 'data/model_ckpts_sg_test1/'
    eval_file = 'eval_gbi_domiknows.csv'

    
    """
    Load data
    """
    meta_info = json.load(open('data/gqa_info.json', 'r'))

    print('loading object features')
    obj_feats = np.load('data/features_mini.npy', allow_pickle=True).item()

    sg_test_loader = SceneGraphLoader(
        meta_info=meta_info,
        data_f='data/rescources/new_label_data/val.pickle',
        obj_feats=obj_feats,
        type='name',
        batch_size=1,
        drop_last=False,
        shuffle=False
    )

    num_valid = len(sg_test_loader)

    label_sizes = {
        2: torch.tensor(158),
        3: torch.tensor(63),
        4: torch.tensor(8)
    }

    def fix_label(label_list, depth):
        """
        Convert None label (-1) to index of None prediction (last label)
        """
        return [l if l != -1 else label_sizes[depth] - 1 for l in label_list]

    def row_to_dataitem(row, batch_size=1):
        """
        Convert sg_test_loader item to domiknows dataitem dict.
        """
        result = {}
        result['features'] = row[0]

        for depth in range(1, 5):
            print(fix_label(row[depth + 1], depth))
            result[f'level{depth}_label'] = torch.tensor(fix_label(row[depth + 1], depth))
        
        result['reps'] = torch.randn((batch_size))

        return result
    
    feat_dim = list(obj_feats.values())[0].shape[0]

    """
    Load models
    """
    sg_model_dict, models = init_model(
        feat_dim,
        meta_info,
        checkpoint_dir=None if do_train else checkpoint_dir,
        train=do_train
    )

    program = main('cpu', models)

    if do_train:
        epoch_num = 0
        def checkpoint_fn(*args):
            global epoch_num
            sg_model_dict.save_models(model_dir=f'checkpoints_{epoch_num}/')
            epoch_num += 1
        program.after_train_epoch = [checkpoint_fn]

        sg_train_loader = SceneGraphLoader(
            meta_info=meta_info,
            data_f='data/rescources/new_label_data/train.pickle',
            obj_feats=obj_feats,
            type='name',
            batch_size=train_batch_size,
            drop_last=True,
            shuffle=True
        )

        num_train = len(sg_train_loader)

        program.train(
            tqdm(map(partial(row_to_dataitem, batch_size=train_batch_size), sg_train_loader), total=num_train),
            # valid_set=map(partial(row_to_dataitem, batch_size=1), sg_test_loader),
            device='cpu',
            train_epoch_num=10,
            test_every_epoch=True,
        )
    else:
        eval_f = open(eval_file, 'w', buffering=1)

        eval_f.write('index,level1,level2,level3,level4,pred_type\n')

        for row_idx, row in enumerate(sg_test_loader):
            dataitem = row_to_dataitem(row)

            node = program.populate_one(dataitem)

            node.inferLocal()

            for child in node.getChildDataNodes('image'):
                eval_row = [str(row_idx)]
                eval_row_GBI = [str(row_idx)]
                #eval_row_ILP = [str(row_idx)]
                eval_row_label = [str(row_idx)]
                for _concept in ['level1', 'level2', 'level3', 'level4']:
                    label = child.getAttribute(_concept, 'label').item()
                    pred = child.getAttribute(_concept, 'local/argmax').argmax().item()
                    pred_GBI = child.getAttribute(_concept, 'GBI').argmax().item()
                    #pred_ILP = child.getAttribute(_concept, 'ILP').argmax().item()

                    eval_row_label.append(str(label))
                    eval_row.append(str(pred))
                    eval_row_GBI.append(str(pred_GBI))
                    #eval_row_ILP.append(str(pred_ILP))

                    print(_concept, pred, label)
                
                print()

                eval_f.write(','.join(eval_row_label + ['label']) + '\n')
                eval_f.write(','.join(eval_row + ['argmax']) + '\n')
                eval_f.write(','.join(eval_row_GBI + ['GBI']) + '\n')
                #eval_f.write(','.join(eval_row_ILP + ['ILP']) + '\n')
        
        eval_f.close()
