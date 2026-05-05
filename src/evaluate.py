import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from dataset import ChestXrayDataset
from utils import val_transforms
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from model import create_model
import argparse

cudnn.benchmark = True
plt.ion()   # interactive mode


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_dir = '../data'

test_ds = load_dataset(
    "alkzar90/NIH-Chest-X-ray-dataset",
    "image-classification",
    split="test",
    cache_dir = data_dir,
    trust_remote_code = True,
    )
test_dataset = ChestXrayDataset(test_ds, transform=val_transforms)

dataloaders = {'test':torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False, pin_memory=True)}
dataset_sizes = {'test':test_dataset.__len__()}

class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']



def test_model(model_path):
    since = time.time()

    # Create a temporary directory to save training checkpoints

    best_model_params_path = os.path.join('../models', model_path)

    model = create_model()
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    model.to(device)
       
    model.eval()   # Set model to evaluate mode
    all_labels = []
    all_outputs = []

        

    for inputs, labels in tqdm(dataloaders['test'], desc = 'test'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)

        all_labels.append(labels.cpu())
        all_outputs.append(torch.sigmoid(outputs).cpu().detach())
    

    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs).numpy()
    auc = roc_auc_score(all_labels, all_outputs, average='macro')

    print(f'Val AUC: {auc:.4f}')
    for i, name in enumerate(class_names):
        class_auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        print(f'{name}: {class_auc:.4f}')

    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model filename in ../models/')
args = parser.parse_args()
model_ft = test_model(args.model)