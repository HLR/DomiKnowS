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
from pathlib import Path
from functional_2d import generate_intersection_map, generate_union_box

def meshgrid_single(tensor, dim=0):
    """
    Replacement for jactorch.meshgrid - creates pairwise combinations.
    For 1D tensor of size N: returns two (N, N) tensors
    For 2D tensor of size (N, D): returns two (N, N, D) tensors
    """
    n = tensor.size(0)
    if tensor.dim() == 1:
        a = tensor.unsqueeze(1).expand(n, n)
        b = tensor.unsqueeze(0).expand(n, n)
    else:
        a = tensor.unsqueeze(1).expand(n, n, -1)
        b = tensor.unsqueeze(0).expand(n, n, -1)
    return a, b

class ResnetLEFT(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        # Load pretrained resnet34 and modify it
        base_resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Keep layers up to layer3, replace layer4 with Identity
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = torch.nn.Identity()  # Replace layer4
        # No avgpool or fc (incl_gap=False, num_classes=None)

        self.preprocessor = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = device

    def forward(self, sample_id, image):
        if isinstance(image, list):
            image = image[0]
        x = self.preprocessor(image).unsqueeze(0).to(self.device)
        
        # Forward through modified resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


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
        
        base_resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = torch.nn.Identity()

        self.preprocessor = T.Compose([
            T.Resize((224, 224)),  # Resize to ResNet input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = device

    def forward(self, sample_id, image):
        if isinstance(image, list):
            image = image[0]
        x = self.preprocessor(image).unsqueeze(0).to(self.device)
        
        # Forward through individual layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


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
    def __init__(self, input_size: int, output_size: int, pretrained: bool = True, device: Optional[str] = None):
        super().__init__()
        
        self.pool_size = 32
        downsample_rate = 16

        self.relation_roi_pool = PrRoIPoolApprox((self.pool_size, self.pool_size), 1.0 / downsample_rate).to(device)

        feature_dim = input_size
        output_dims = [None, 128, 128, 128]
        self.relation_feature_extract = torch.nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1).to(device)
        self.relation_feature_fuse = torch.nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1).to(device)
        self.relation_feature_fc = torch.nn.Sequential(
            torch.nn.ReLU(True), 
            torch.nn.Linear(output_dims[2] * self.pool_size ** 2, output_size)
        ).to(device)

        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._ram_cache: dict[str, torch.Tensor] = {}

    def forward(self, scene, box, object_features):
        relation_features = self.relation_feature_extract(scene)

        with torch.no_grad():
            # Replace jactorch.meshgrid with custom implementation
            sub_id, obj_id = meshgrid_single(torch.arange(box.size(0), dtype=torch.int64, device=box.device))
            sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)
            
            sub_box, obj_box = meshgrid_single(box)
            sub_box = sub_box.contiguous().view(box.size(0) ** 2, 4)
            obj_box = obj_box.contiguous().view(box.size(0) ** 2, 4)

            # union box
            union_box = generate_union_box(sub_box, obj_box)
            rel_batch_ind = 0 + torch.zeros(union_box.size(0), 1, dtype=box.dtype, device=box.device)

            # intersection maps
            sub_union_imap = generate_intersection_map(sub_box, union_box, self.pool_size)
            obj_union_imap = generate_intersection_map(obj_box, union_box, self.pool_size)

        def _norm(x):
            return x / x.norm(2, dim=-1, keepdim=True)

        this_relation_features = self.relation_roi_pool(relation_features, torch.cat((rel_batch_ind, union_box), dim=-1))
        x, y, z = this_relation_features.chunk(3, dim=1)
        this_relation_features = self.relation_feature_fuse(
            torch.cat((object_features[sub_id], object_features[obj_id], x, y * sub_union_imap, z * obj_union_imap), dim=1)
        )

        relation_features_emb = _norm(self.relation_feature_fc(this_relation_features.view(box.size(0) * box.size(0), -1)))

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
