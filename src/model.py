
import torch.nn as nn
from torchvision import models

def create_model(num_classes=14, freeze_backbone = True):
    model = models.resnet50(weights='IMAGENET1K_V1')

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

