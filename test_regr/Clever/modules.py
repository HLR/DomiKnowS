import torch
import torch
import torchvision.transforms as T
import torchvision.models as models
from typing import List, Union, Optional, Tuple
from PIL import Image
import numpy as np

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

    def forward(self, image: Image, bboxes_xyxy: List[Union[List[List[float]], np.ndarray]]) -> List[List[torch.Tensor]]:

        image=image[0]
        all_features: List[torch.Tensor] = []
        for bbox in bboxes_xyxy:
            x_min, y_min, x_max, y_max = map(int, bbox)  # Ensure integer coordinates for cropping

            # patches = image[y_min:y_max, x_min:x_max]
            patches = image.crop((x_min, y_min, x_max, y_max))  # Crop the image using PIL
            # Preprocess the patch
            patch_tensor = self.preprocess(patches).unsqueeze(0).to(self.device)  # Add batch dimension

            # Extract features
            with torch.no_grad():  # No need to track gradients
                features = self.features_extractor(patch_tensor)

            all_features.append(features.squeeze().cpu())  # Remove batch dim and move to CPU

        return torch.stack(all_features)