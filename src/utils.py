"""
Train and Validation transformations for images 
"""

import torch
from torchvision.transforms import v2

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(224,224)),
    v2.RGB(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomAffine(degrees=(10, 20), translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Normalizing using ImageNet stats
])

val_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=(224,224)),
    v2.RGB(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Normalizing using ImageNet stats
])