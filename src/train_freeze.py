import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from dataset import ChestXrayDataset
from utils import train_transforms, val_transforms
from torchvision import models
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from collections import Counter
from model import create_model

cudnn.benchmark = True
plt.ion()   # interactive mode


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_dir = '../data'

train_ds = load_dataset(
    "alkzar90/NIH-Chest-X-ray-dataset",
    "image-classification",
    split="train",
    cache_dir = data_dir,
    trust_remote_code = True,
    )
split = train_ds.train_test_split(test_size=0.1, seed=42)
train_dataset = ChestXrayDataset(split["train"], transform=train_transforms)
val_dataset = ChestXrayDataset(split["test"], transform=val_transforms)

dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True),
               'val': torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=False, pin_memory=True)}
dataset_sizes = {'train':train_dataset.__len__(),
               'val': val_dataset.__len__()}

class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def get_pos_weight(dataset):
    all_labels = dataset["labels"]
    weights=[]
    count = Counter()
    for label in all_labels:
        count.update(label)
    total_samples = len(all_labels)

    for i in range(1,15):
        label_count = count[i]
        weights.append((total_samples - label_count)/label_count)
    return torch.tensor(weights)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, val = True):
    since = time.time()

    # Create a temporary directory to save training checkpoints

    best_model_params_path = os.path.join('../models', 'best_model_params_freeze.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_auc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        phases = ['train', 'val'] if val else ['train']

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            if phase == 'val':
                model.eval()   # Set model to evaluate mode
                all_labels = []
                all_outputs = []

            running_loss = 0.0
            

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'val':
                    all_labels.append(labels.cpu())
                    all_outputs.append(torch.sigmoid(outputs).cpu().detach())
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}')
            

            # deep copy the model
            if phase == 'val':
                all_labels = torch.cat(all_labels).numpy()
                all_outputs = torch.cat(all_outputs).numpy()
                auc = roc_auc_score(all_labels, all_outputs, average='macro')
                print(f'Val AUC: {auc:.4f}')
                if auc>best_auc:
                    best_auc = auc
                    torch.save(model.state_dict(), best_model_params_path)
                for i, name in enumerate(class_names):
                    class_auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                    print(f'{name}: {class_auc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best auc: {best_auc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

# Phase 1
model = create_model(freeze_backbone=True).to(device)
optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3)

# Train for 5 epochs
pos_weight = get_pos_weight(train_ds).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Train model

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=5, val=False)

# Train for 10 epochs

for param in model_ft.parameters():
    param.requires_grad = True
optimizer = optim.AdamW(model_ft.parameters(), lr=1e-5)
exp_lr_scheduler_2 = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_ft = train_model(model_ft,criterion, optimizer, exp_lr_scheduler_2,
                       num_epochs=10, val=True)