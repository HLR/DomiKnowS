import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
from domiknows.sensor.pytorch import TorchLearner
import torch
import torchvision.transforms as T
import torchvision.models as models
from torchvision.ops import roi_align
from typing import List, Union, Optional, Tuple
from PIL import Image
import numpy as np
import jactorch.models.vision.resnet as resnet
import jactorch
from pathlib import Path
from functional_2d import generate_intersection_map, generate_union_box



class ResNetPatcher(torch.nn.Module):
    """
    A PyTorch Module to extract features from image patches using a ResNet model.
    """
    def __init__(self, resnet_model_name: str = 'resnet50', pretrained: bool = True, device: Optional[str] = None):
        super().__init__()

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_model_name = resnet_model_name
        # Load the specified ResNet model
        if resnet_model_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif resnet_model_name == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif resnet_model_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif resnet_model_name == 'resnet101':
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        elif resnet_model_name == 'dummy': ## this is a dummy model for testing
            resnet = None
        else:
            raise ValueError(f"Unsupported ResNet model name: {resnet_model_name}. Choose from resnet18, resnet34, resnet50, resnet101.")

        if resnet_model_name != 'dummy':
            # Remove the final fully connected layer (classifier)
            self.features_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
            
        else:
            ## a linear layer for testing
            # Dummy model for testing
            self.features_extractor = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(224*224*3, 2048),
            )
            
        self.features_extractor.eval() # Set to evaluation mode
        self.features_extractor.to(self.device)
        # Standard ImageNet normalization
        self.preprocess = T.Compose([
            T.Resize((224, 224)),  # Resize to ResNet input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._ram_cache: dict[str, torch.Tensor] = {}

    def _path(self, sample_id) -> Path:
        return self.cache_dir / f"{self.resnet_model_name}_{sample_id}.pt"

    def _load_if_present(self, sample_id):
        if sample_id in self._ram_cache:
            return self._ram_cache[sample_id]

        p = self._path(sample_id)
        if p.exists():
            feat = torch.load(p, map_location=self.device)
            self._ram_cache[sample_id] = feat
            return feat
        return None
    
    @torch.no_grad()
    def forward(self,
                sample_id,
                image,  # PIL.Image | np.ndarray | tensor –– as before
                bboxes_xyxy):
        """
        * If features for `sample_id` already exist → return them.
        * Otherwise compute, store (RAM + disk) and return.
        """
        if isinstance(sample_id, list):
            sample_id = sample_id[0]

        cached = self._load_if_present(sample_id)
        if cached is not None:
            return cached  # <-- fast path

        # --------- slow path: compute ---------------------------------------
        if isinstance(image, list):  # your original code used image[0]
            image = image[0]

        feats = []
        for bbox in bboxes_xyxy:
            x_min, y_min, x_max, y_max = map(int, bbox)

            patch = image.crop((x_min, y_min, x_max, y_max))
            patch_tensor = self.preprocess(patch).unsqueeze(0).to(self.device)

            features = self.features_extractor(patch_tensor)
            feats.append(features.squeeze())

        stacked = torch.stack(feats).to(self.device)

        # --------- store in cache -------------------------------------------
        self._ram_cache[sample_id] = stacked
        # write atomically: first to tmp, then rename – avoids half-written files
        tmp_path = self._path(f"{self.resnet_model_name}_{sample_id}.tmp")
        torch.save(stacked.cpu(), tmp_path)
        tmp_path.replace(self._path(sample_id))

        return stacked


class PrRoIPoolApprox(torch.nn.Module):
    def __init__(self, output_size=(32, 32), spatial_scale=1.0/16):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        """
        Args:
            features: Tensor of shape (B, C, H, W)
            rois: Tensor of shape (N, 5) with (batch_idx, x1, y1, x2, y2)

        Returns:
            Tensor of shape (N, C, output_size[0], output_size[1])
        """
        return roi_align(
            input=features,
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1,  # let PyTorch choose automatically
            aligned=True        # bilinear interpolation
        )
    
class ResnetLEFT(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.resnet = resnet.resnet34(pretrained=True, incl_gap=False, num_classes=None)
        self.resnet.layer4 = jactorch.nn.Identity()

        self.preprocessor = T.Compose([
            T.Resize((224, 224)),  # Resize to ResNet input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = device


    def forward(self, sample_id, image):
        # Cache here
        if isinstance(image, list):
            image = image[0]
        x = self.preprocessor(image).unsqueeze(0).to(self.device)
        feature_emb = self.resnet(x)
        return feature_emb


class LEFTObjectEMB(torch.nn.Module):
    def __init__(self, resnet_model_name: str = 'resnet50', pretrained: bool = True, device: Optional[str] = None):
        super().__init__()

        self.pool_size = 32
        downsample_rate = 16

        self.context_roi_pool = PrRoIPoolApprox((self.pool_size, self.pool_size), 1.0 / downsample_rate).to(device)
        self.object_roi_pool = PrRoIPoolApprox((self.pool_size, self.pool_size), 1.0 / downsample_rate).to(device)

        self.object_feature_extract = torch.nn.Conv2d(256, 256, 1).to(device)
        self.object_feature_fuse = torch.nn.Conv2d(256 * 2, 128, 1).to(device)
        # 

        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._ram_cache: dict[str, torch.Tensor] = {}

        
    def forward(self, scene, box):
        
        this_object_features = self.object_feature_extract(scene)

        with torch.no_grad():
            image_h, image_w = scene.size(2) * 16, scene.size(3) * 16
            image_box = torch.cat((
                torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                image_w + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device),
                image_h + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)
            ), dim=-1)

            box_context_imap = generate_intersection_map(box, image_box, 32)

            batch_ind = 0 + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)

        this_context_features = self.context_roi_pool(this_object_features, torch.cat([batch_ind, image_box], dim=-1))
        x, y = this_context_features.chunk(2, dim=1)

        this_context_features = torch.cat((self.object_roi_pool(scene, torch.cat([batch_ind, box], dim=-1)), x, y * box_context_imap), dim=1)
        this_context_features = self.object_feature_fuse(this_context_features)
        # def _norm(x):
        #     # if self.norm:
        #     return x / x.norm(2, dim=-1, keepdim=True)
        #     # return x

        # object_features_emb = _norm(self.object_fc(this_context_features.view(box.size(0), -1)))

        return this_context_features
    
class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device="cpu"):
        super().__init__()
        self.device = device
        # 128 * 32 * 32, 2048
        self.object_fc = torch.nn.Sequential(torch.nn.ReLU(True), torch.nn.Linear(input_dim, output_dim))
        self.object_fc.to(self.device)

    
    def _norm(self, x):
        # if self.norm:
        # return x
        return x / x.norm(2, dim=-1, keepdim=True)
            
    def forward(self, feature, box):
        emb = self._norm(self.object_fc(feature.view(box.size(0), -1)))
        return emb


class LEFTRelationEMB(torch.nn.Module):
    def __init__(self, resnet_model_name: str = 'resnet50', pretrained: bool = True, device: Optional[str] = None):
        super().__init__()
        
        self.pool_size = 32
        downsample_rate = 16

        self.relation_roi_pool = PrRoIPoolApprox((self.pool_size, self.pool_size), 1.0 / downsample_rate).to(device)

        feature_dim = 256
        output_dims = [None, 128, 128, 128]
        self.relation_feature_extract = torch.nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1).to(device)
        self.relation_feature_fuse = torch.nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1).to(device)
        self.relation_feature_fc = torch.nn.Sequential(torch.nn.ReLU(True), 
                                                       torch.nn.Linear(output_dims[2] * self.pool_size ** 2, 128)).to(device)

        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._ram_cache: dict[str, torch.Tensor] = {}

        
    def forward(self, scene, box, object_features):

        relation_features = self.relation_feature_extract(scene)

        with torch.no_grad():
            sub_id, obj_id = jactorch.meshgrid(torch.arange(box.size(0), dtype=torch.int64, device=box.device), dim=0)
            sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)
            sub_box, obj_box = jactorch.meshgrid(box, dim=0)
            sub_box = sub_box.contiguous().view(box.size(0) ** 2, 4)
            obj_box = obj_box.contiguous().view(box.size(0) ** 2, 4)

            # union box
            union_box = generate_union_box(sub_box, obj_box)
            rel_batch_ind = 0 + torch.zeros(union_box.size(0), 1, dtype=box.dtype, device=box.device)

            # intersection maps
            sub_union_imap = generate_intersection_map(sub_box, union_box, self.pool_size)
            obj_union_imap = generate_intersection_map(obj_box, union_box, self.pool_size)

        def _norm(x):
            # if self.norm:
            return x / x.norm(2, dim=-1, keepdim=True)
            # return x

        this_relation_features = self.relation_roi_pool(relation_features, torch.cat((rel_batch_ind, union_box), dim=-1))
        x, y, z = this_relation_features.chunk(3, dim=1)
        # print()
        this_relation_features = self.relation_feature_fuse(torch.cat((object_features[sub_id], object_features[obj_id], x, y * sub_union_imap, z * obj_union_imap), dim=1))

        relation_features_emb = _norm(self.relation_feature_fc(this_relation_features.view(box.size(0) * box.size(0), -1)))
        # print(relation_features_emb.shape)

        return relation_features_emb
        
class DummyLinearLearnerold(TorchLearner):
    def __init__(self, *pre,current_attribute=None):
        TorchLearner.__init__(self, *pre)
        self.current_attribute = current_attribute

    def forward(self, x,properties):
        result = torch.zeros(len(x), 2)
        for idx in range(len(x)):
            if self.current_attribute[3:] in [v for k,v in properties[idx].items()]:
                result[idx, 1] = 10
            else:
                result[idx, 0] = 10
        return result


class DummyLinearLearner(TorchLearner):
    def __init__(self, *pre, current_attribute=None):
        TorchLearner.__init__(self, *pre)
        self.current_attribute = current_attribute
        self.head = torch.nn.Linear(1, 2)

    @torch.no_grad()
    def _attr_mask(self, properties):
        attr = self.current_attribute[3:]
        return torch.tensor(
            [attr in p.values() for p in properties],
            dtype=torch.bool
        )

    def forward(self, x, properties):
        logits = self.head(x)

        mask = self._attr_mask(properties)
        if mask is not None:
            logits[mask, 1] += 3.0
            logits[~mask, 0] += 3.0

        return logits
