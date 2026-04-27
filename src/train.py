import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from dataset import ChestXrayDataset
from utils import train_transforms, val_transforms
from torchvision import models
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

cudnn.benchmark = True
plt.ion()   # interactive mode


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_dir = '../data'

train_ds = load_dataset(
    "alkzar90/NIH-Chest-X-ray-dataset",
    "image-classification",
    split="train[:500]",
    cache_dir = data_dir,
    trust_remote_code = True,
    )
train_dataset = ChestXrayDataset(train_ds, transform=train_transforms)

val_ds = load_dataset(
    "alkzar90/NIH-Chest-X-ray-dataset",
    "image-classification",
    split="test[:100]",
    cache_dir = data_dir,
    trust_remote_code = True,
    )
val_dataset = ChestXrayDataset(val_ds, transform=val_transforms)

dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True),
               'val': torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)}
dataset_sizes = {'train':train_dataset.__len__(),
               'val': val_dataset.__len__()}



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints

    best_model_params_path = os.path.join('../models', 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_auc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                all_labels = []
                all_outputs = []

            running_loss = 0.0
            

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
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

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best auc: {best_auc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 14)

model_ft = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Train model

model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=2)

