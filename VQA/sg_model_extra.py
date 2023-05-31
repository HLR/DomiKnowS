import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


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

class NameClassifier(nn.Module):
    def __init__(self, feat_dim, latent_dim, name_embeddings, device):
        super(NameClassifier, self).__init__()

        self.name_embed = torch.from_numpy(name_embeddings).to(device)     # n_names * glove_dim
        glove_dim = self.name_embed.shape[1]

        self.fc1 = nn.Linear(feat_dim, latent_dim)
        self.fc2 = nn.Linear(glove_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1)

    def forward(self, objs):                                    # n_objs * feat_dim
        x = self.fc1(objs).unsqueeze(dim=1)                     # n_objs * 1 * latent_dim
        c = self.fc2(self.name_embed).unsqueeze(dim=0)          # 1 * n_names * latent_dim
        xc = x * c                                              # n_objs * n_names * latent_dim
        logits = self.fc3(xc).squeeze(dim=2)                    # n_objs * n_names

        return logits

class AttributeClassifier(nn.Module):
    def __init__(self, feat_dim, latent_dim, attribute_embeddings, device):
        super(AttributeClassifier, self).__init__()

        self.attr_embed = torch.from_numpy(attribute_embeddings).to(device)    # (n_attrs+1) * glove_dim
        glove_dim = self.attr_embed.shape[1]

        self.fc1 = nn.Linear(feat_dim, latent_dim)
        self.fc2 = nn.Linear(glove_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1)

    def forward(self, objs):                                    # n_objs * feat_dim
        x = self.fc1(objs).unsqueeze(dim=1)                     # n_objs * 1 * latent_dim
        c = self.fc2(self.attr_embed).unsqueeze(dim=0)          # 1 * n_attrs * latent_dim
        xc = x * c                                              # n_objs * n_attrs * latent_dim
        logits = self.fc3(xc).squeeze(dim=2)                    # n_objs * n_attrs

        return logits

class RelateClassifier(nn.Module):
    def __init__(self, latent_dim, relate_embeddings, device):
        super(RelateClassifier, self).__init__()

        self.rel_embed = torch.from_numpy(relate_embeddings).to(device)    # n_rels * glove_dim
        glove_dim = self.rel_embed.shape[1]

        self.fc1 = nn.Linear(4, latent_dim)
        self.fc2 = nn.Linear(4, latent_dim)
        self.fc3 = nn.Linear(glove_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 1)

    def forward(self, bboxs):                                   # n_pairs * 4 * 2
        bboxs = bboxs.view(-1, 4, 2)                            # n_pairs * 4 * 2
        sub_emb = self.fc1(bboxs[:,:,0]).unsqueeze(dim=1)       # n_pairs * 1 * latent_dim
        obj_emb = self.fc2(bboxs[:,:,1]).unsqueeze(dim=1)       # n_pairs * 1 * latent_dim
        c = self.fc3(self.rel_embed).unsqueeze(dim=0)           # 1 * n_rels * latent_dim
        xc = sub_emb * obj_emb * c                              # n_pairs * n_rels * latent_dim
        logits = self.fc4(xc).squeeze(dim=2)                    # n_pairs * n_rels

        return logits

class SceneGraphModel:
    def __init__(self, feat_dim, n_names, n_attrs, n_rels, name_embeddings, attribute_embeddings, relate_embeddings,
                 device, model_dir=None):
        self.feat_dim = feat_dim
        self.n_names = n_names
        self.n_attrs = n_attrs
        self.n_rels = n_rels
        self.name_embeddings = name_embeddings
        self.attribute_embeddings = attribute_embeddings
        self.relate_embeddings = relate_embeddings
        self.device = device

        self._init_models()
        if model_dir is not None:
            self._load_models(model_dir)

    def _load_models(self, model_dir):
        for type in ['name', 'relation', 'attribute']:
            load_model(
                model=self.models[type],
                model_f=model_dir+'/%s_best_epoch.pt' % type,
                device=self.device
            )

    def _init_models(self):
        # name_clf = MLPClassifier(
        #     input_dim=self.feat_dim,
        #     output_dim=self.n_names,
        #     latent_dim=1024,
        #     n_layers=2,
        #     dropout_rate=0.3
        # )
        #
        # rela_clf = MLPClassifier(
        #     input_dim=(self.feat_dim+4)*2,  # 4: bbox
        #     output_dim=self.n_rels+1,       # 1: None
        #     latent_dim=1024,
        #     n_layers=1,
        #     dropout_rate=0.5
        # )
        #
        # attr_clf = MLPClassifier(
        #     input_dim=self.feat_dim,
        #     output_dim=self.n_attrs,
        #     latent_dim=1024,
        #     n_layers=1,
        #     dropout_rate=0.3
        # )

        name_clf = NameClassifier(
            feat_dim=self.feat_dim,
            latent_dim=1024,
            name_embeddings=self.name_embeddings,
            device=self.device
        )

        attr_clf = AttributeClassifier(
            feat_dim=self.feat_dim,
            latent_dim=1024,
            attribute_embeddings=self.attribute_embeddings,
            device=self.device
        )

        rela_clf = RelateClassifier(
            latent_dim=1024,
            relate_embeddings=self.relate_embeddings,
            device=self.device
        )

        self.models = {
            'name': name_clf,
            'attribute': attr_clf,
            'relation': rela_clf
        }

    def predict(self, type, inputs):
        # type == 'name', inputs == (obj_feat_np_array)
        # old: type == 'relation', inputs == (sub_feat_np_array, obj_feat_np_array, sub_bbox_np_array, obj_bbox_np_array)
        # new: type == 'relation', inputs == (sub_bbox_np_array, obj_bbox_np_array)
        # type == 'attribute', inputs == (obj_feat_np_array)

        model = self.models[type].to(self.device)
        inputs = torch.cat([torch.from_numpy(x).float() for x in inputs]).reshape(len(inputs), -1).to(self.device)
        logits = model(inputs)
        if type == 'attribute':
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        return logits, probs