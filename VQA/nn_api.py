import argparse
import os
import numpy as np
import torch
import random
import json
import pickle
from supervised_learning.sg_model import SceneGraphModel

def load_sg_model_dict(args):
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    meta_info = json.load(open(args.meta_f, 'r'))
    name_embeddings = pickle.load(open(args.name_glove_f, 'rb'))        # numpy array of shape (n_names, glove_dim)
    attribute_embeddings = pickle.load(open(args.attr_glove_f, 'rb'))   # numpy array of shape (n_attrs+1, glove_dim)
    relate_embeddings = pickle.load(open(args.rel_glove_f, 'rb'))

    sg_model_dict = SceneGraphModel(
        feat_dim=args.feat_dim,
        n_names=meta_info['name']['num'],
        n_attrs=meta_info['attr']['num'],
        n_rels=meta_info['rel']['num'],
        device=device,
        model_dir=args.model_dir
    )

    # sg_model_dict = SceneGraphModel(
    #     feat_dim=args.feat_dim,
    #     n_names=meta_info['name']['num'],
    #     n_attrs=meta_info['attr']['num'],
    #     n_rels=meta_info['rel']['num'],
    #     name_embeddings=name_embeddings,
    #     attribute_embeddings=attribute_embeddings,
    #     relate_embeddings=relate_embeddings,
    #     model_dir=args.model_dir,
    #     device=device
    # )
    return sg_model_dict

if __name__ == '__main__':
    # DATA_ROOT = os.getenv('HOME') + '/project_data/CRIC'
    # DATA_ROOT = '/localscratch/bchen346/project_data/CRIC'
    DATA_ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--feat_dim', type=int, default=2048)
    # parser.add_argument('--type', default='name')
    parser.add_argument('--model_dir', default=DATA_ROOT+'/model_ckpts')
    # parser.add_argument('--feat_f', default=DATA_ROOT+'/scene_graph_data/features.npy')
    # parser.add_argument('--test_f', default=DATA_ROOT+'/scene_graph_data/test_split.pkl')
    parser.add_argument('--meta_f', default=DATA_ROOT+'/preprocessing/meta.json')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    sg_models = load_sg_model_dict(args)

    obj_feat_np_array = np.random.randn(args.feat_dim)
    sub_feat_np_array = np.random.randn(args.feat_dim)
    sub_bbox_np_array = np.random.uniform(low=0, high=1, size=4)
    obj_bbox_np_array = np.random.uniform(low=0, high=1, size=4)
    group_id = 5 #np.random.randint(50)
    # type == 'name', inputs == [obj_feat_np_array]
    # type == 'rel', inputs == [sub_feat_np_array, obj_feat_np_array, sub_bbox_np_array, obj_bbox_np_array]
    # type == 'attr:%d' % group_id, inputs == [obj_feat_np_array]
    name_probs = sg_models.predict(
        type='name',
        inputs=[obj_feat_np_array]
    )
    rel_probs = sg_models.predict(
        type='rel',
        inputs=[sub_feat_np_array, obj_feat_np_array, sub_bbox_np_array, obj_bbox_np_array]
    )
    attr_probs = sg_models.predict(
        type='attr:%d' % group_id,
        inputs=[obj_feat_np_array]
    )
    print('end')