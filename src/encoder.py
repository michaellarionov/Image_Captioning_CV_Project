import torch
import torch.nn as nn
from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=768, freeze=True):
        super().__init__()

        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final classification layer
        # We want features, not class predictions
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Project ResNet output (2048) → GPT-2 input size (768)
        self.projection = nn.Linear(2048, 768)

        # Freeze ResNet weights to save memory during Phase 1 training
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images):
        features = self.backbone(images)            # [batch, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1) # [batch, 2048]
        features = self.projection(features)        # [batch, 768]
        return features.unsqueeze(1)               # [batch, 1, 768]