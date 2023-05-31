from sg_model import SceneGraphModel
from torch import nn


class Net(nn.Module):
    def __init__(self, sg_models, depth):
        super().__init__()

        self.model = sg_models[depth]
    
    def forward(self, *inputs):
        data = inputs[-1]

        out = self.model(data)

        return out


def init_model(feat_dim, meta_info, checkpoint_dir='./model_ckpts_sg_test/', train=False):
    sg_model_dict = SceneGraphModel(
        feat_dim=feat_dim,
        n_names=meta_info['name']['num'],
        n_attrs=meta_info['attr']['num'],
        n_rels=meta_info['rel']['num'],
        device='cpu',
        model_dir=checkpoint_dir
    )
    
    if train:
        sg_model = {
            1: sg_model_dict.models["name1"].train(),
            2: sg_model_dict.models["name2"].train(),
            3: sg_model_dict.models["name3"].train(),
            4: sg_model_dict.models["name4"].train()
        }
    else:
        sg_model = {
            1: sg_model_dict.models["name1"].eval(),
            2: sg_model_dict.models["name2"].eval(),
            3: sg_model_dict.models["name3"].eval(),
            4: sg_model_dict.models["name4"].eval()
        }
    
    return sg_model_dict, sg_model