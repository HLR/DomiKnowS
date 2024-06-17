import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
torch.autograd.set_detect_anomaly(True)

def load_model(model, model_f, device):
    print('loading model from %s' % model_f)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, n_layers, dropout_rate):
        super(MLPClassifier, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(latent_dim))
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(latent_dim))
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(latent_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        return logits

class SceneGraphModel:
    def __init__(self, feat_dim, n_names, n_attrs, n_rels, device, model_dir=None):
        self.feat_dim = feat_dim
        self.n_names1 = 274
        self.n_names2 = 158
        self.n_names3 = 63
        self.n_names4 = 8
        self.n_attrs = n_attrs
        self.n_rels = n_rels
        self.device = device

        self._init_models()
        if model_dir is not None:
            self._load_models(model_dir)

    def _load_models(self, model_dir):
        for id, type in enumerate(['name1', 'name2', 'name3', 'name4']):
            load_model(
                model=self.models[type],
                model_f=model_dir+'class%s'%str(id+1)+'/name_best_epoch.pt',
                device=self.device
            )

    def save_models(self, model_dir='checkpoints/'):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        else:
            print(f'{model_dir} already exists.')

        for id, type in enumerate(['name1', 'name2', 'name3', 'name4']):
            sub_dir = model_dir+'class%s'%str(id+1)

            if not os.path.isdir(sub_dir):
                os.mkdir(sub_dir)
            else:
                print(f'{sub_dir} already exists.')

            out_path = sub_dir+'/name_best_epoch.pt'
            torch.save(self.models[type].state_dict(), out_path)  

    def _init_models(self):
        name_clf1 = MLPClassifier(
            input_dim=self.feat_dim,
            output_dim=self.n_names1,
            latent_dim=1024,
            n_layers=2,
            dropout_rate=0.3
        )
        name_clf2 = MLPClassifier(
            input_dim=self.feat_dim,
            output_dim=self.n_names2,
            latent_dim=1024,
            n_layers=2,
            dropout_rate=0.3
        )
        name_clf3 = MLPClassifier(
            input_dim=self.feat_dim,
            output_dim=self.n_names3,
            latent_dim=1024,
            n_layers=2,
            dropout_rate=0.3
        )
        name_clf4 = MLPClassifier(
            input_dim=self.feat_dim,
            output_dim=self.n_names4,
            latent_dim=1024,
            n_layers=2,
            dropout_rate=0.3
        )

        rela_clf = MLPClassifier(
            input_dim=(self.feat_dim+4)*2,  # 4: bbox
            output_dim=self.n_rels+1,       # 1: None
            latent_dim=1024,
            n_layers=1,
            dropout_rate=0.5
        )

        attr_clf = MLPClassifier(
            input_dim=self.feat_dim,
            output_dim=self.n_attrs,
            latent_dim=1024,
            n_layers=1,
            dropout_rate=0.3
        )

        self.models = {
            'attribute': attr_clf,
            'relation': rela_clf,
            'name1': name_clf1,
            'name2': name_clf2,
            'name3': name_clf3,
            'name4': name_clf4,
        }

    def predict(self, type, inputs):
        # type == 'name', inputs == (obj_feat_np_array)
        # type == 'relation', inputs == (sub_feat_np_array, obj_feat_np_array, sub_bbox_np_array, obj_bbox_np_array)
        # type == 'attribute', inputs == (obj_feat_np_array)

        model = self.models[type].to(self.device)
        inputs = torch.cat([torch.from_numpy(x).float() for x in inputs]).reshape(len(inputs), -1).to(self.device)
        logits = model(inputs)

        if type == 'attribute':
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        return logits, probs

    def batch_predict(self, type, inputs, batch_split):

        model = self.models[type].to(self.device)
        inputs = torch.cat([torch.from_numpy(x).float() for x in inputs]).reshape(len(inputs), -1).to(self.device)
        logits = model(inputs)

        if type == 'attribute':
            probs = torch.sigmoid(logits)
        else:
            current_split = 0
            probs = []
            for split in batch_split:
                current_logits = logits[current_split:split]
                # batched_logits = logits.reshape(batch_shape[0], batch_shape[1], -1)
                current_probs = F.softmax(current_logits, dim=1)
                # probs = probs.reshape(inputs.shape[0], -1)
                probs.append(current_probs)
                current_split = split

            probs = torch.cat(probs).reshape(inputs.shape[0], -1)
        return logits, probs