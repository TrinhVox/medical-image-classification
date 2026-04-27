"""
Wrap HuggingDace dataset in PyTorch dataset class
Transform label list into binary vector
"""


import torch
from torch.utils.data import Dataset



class ChestXrayDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        raw_labels = self.data[idx]["labels"]
        adjusted = [l - 1 for l in raw_labels if l != 0]
        label = torch.zeros(14, dtype=torch.float)
        if adjusted:
            label.scatter_(0, torch.tensor(adjusted), 1)
        image = self.data[idx]["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
        

        
