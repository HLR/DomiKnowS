import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
from domiknows.sensor.pytorch import TorchLearner
import torch
import torchvision.transforms as T
import torchvision.models as models
from typing import List, Union, Optional, Tuple
from PIL import Image
import numpy as np
from pathlib import Path



class ResNetPatcher(torch.nn.Module):
    """
    A PyTorch Module to extract features from image patches using a ResNet model.
    """
    def __init__(self, resnet_model_name: str = 'resnet50', pretrained: bool = True, device: Optional[str] = None):
        super().__init__()

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        return self.cache_dir / f"{sample_id}.pt"

    def _load_if_present(self, sample_id):
        if sample_id in self._ram_cache:
            return self._ram_cache[sample_id]

        p = self._path(sample_id)
        if p.exists():
            feat = torch.load(p, map_location="cpu")
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
            feats.append(features.squeeze().cpu())

        stacked = torch.stack(feats)

        # --------- store in cache -------------------------------------------
        self._ram_cache[sample_id] = stacked
        # write atomically: first to tmp, then rename – avoids half-written files
        tmp_path = self._path(f"{sample_id}.tmp")
        torch.save(stacked, tmp_path)
        tmp_path.replace(self._path(sample_id))

        return stacked



class DummyLinearLearner(TorchLearner):
    def __init__(self, *pre,current_attribute=None):
        TorchLearner.__init__(self, *pre)
        self.current_attribute = current_attribute

    def forward(self, x,properties):
        result = torch.zeros(len(x), 2)
        for idx in range(len(x)):
            if self.current_attribute[3:] in [v for k,v in properties[idx].items()]:
                result[idx, 1] = 1000
            else:
                result[idx, 0] = 1000
        return result
