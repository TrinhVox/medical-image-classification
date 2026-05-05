# Chest X-ray Multi-Label Disease Classification

Multi-label classification of 14 thoracic diseases from chest X-rays using ResNet-50, 
achieving **0.77 macro AUC-ROC** on NIH ChestX-ray14 (~112k images).

This project establishes a ResNet-50 baseline as a stepping stone toward the DenseNet-121 architecture used in CheXNet (Rajpurkar et al., 0.841 macro AUC), and investigates why diffuse pathologies like Pneumonia remain significantly harder to classify than focal conditions like Cardiomegaly.

![Grad-CAM comparison](outputs/comparison/gradcam_Pneumonia_incorrect_273.png)



## Key Findings

**Transfer learning strategy matters.** Freeze/unfreeze fine-tuning consistently outperformed training from a frozen backbone across 13 of 14 classes, improving macro AUC from 0.75 → 0.77. The exception was Hernia (0.90 → 0.89), likely because the limited class frequency made full fine-tuning prone to overfitting on a small positive set.

**Focal vs. diffuse pathology is the core challenge.** Grad-CAM analysis reveals a consistent pattern: the model correctly localises focal, morphologically distinct conditions (Cardiomegaly: 0.86 AUC, Hernia: 0.90 AUC) by attending to well-defined anatomical structures. For diffuse conditions (Pneumonia: 0.67, Infiltration: 0.68), the model attends to peripheral regions — shoulders, image borders — rather than anomical structures such as the lungs.


## Results

| Class | Baseline | Freeze/Unfreeze |
|-------|----------|-----------------|
| Atelectasis | 0.7288 | 0.7686 |
| Cardiomegaly | 0.8312 | 0.8622 |
| Effusion | 0.7775 | 0.7935 |
| Infiltration | 0.6758 | 0.6845 |
| Mass | 0.7013 | 0.7451 |
| Nodule | 0.6824 | 0.6932 |
| Pneumonia | 0.6607 | 0.6705 |
| Pneumothorax | 0.8075 | 0.8179 |
| Consolidation | 0.7035 | 0.7145 |
| Edema | 0.8166 | 0.8278 |
| Emphysema | 0.7864 | 0.8080 |
| Fibrosis | 0.7674 | 0.7815 |
| Pleural_Thickening | 0.7235 | 0.7323 |
| Hernia | 0.9023 | 0.8949 |
| **Macro AUC** | **0.7546** | **0.7686** |

*Evaluated on the official NIH ChestX-ray14 test split. AUC-ROC reported per class and macro-averaged. CheXNet (DenseNet-121) reports 0.841 macro AUC for reference.*

## Approach

- **Dataset**: NIH ChestX-ray14 (~112k images, 14 disease labels)
- **Architecture**: ResNet-50 pretrained on ImageNet
- **Loss**: BCEWithLogitsLoss with class-frequency pos_weight to handle imbalance
- **Training strategy**: 
  - Phase 1: Freeze backbone, train FC layer (5 epochs, lr=1e-3)
  - Phase 2: Unfreeze all layers, fine-tune (10 epochs, lr=1e-5)
- **Evaluation**: Per-class and macro-averaged AUC-ROC

## Interpretability

Grad-CAM visualizations on the final convolutional layer comparing 
baseline vs freeze/unfreeze models across strong and weak performing classes.

Strong Performing Classes:

![Cardiomegalyn](outputs/comparison/gradcam_Cardiomegaly_(correct)_35.png)

![Hernia](outputs/comparison/gradcam_Hernia_(correct)_0.png)

Weak Performing Classes:

![Pneumonia](outputs/comparison/gradcam_Pneumonia_incorrect_273.png)

![Infiltration](outputs/comparison/gradcam_Infiltration_(incorrect)_31.png)

For the strong performing classes such as Cardiomegalyn and Hernia, both models correctly attend to the cardiac silhouette to classify. For Pneumonia, the freeze/unfreeze gives better attention to the lung opacities hence delivered a better result compared to the baseline model. However, for Infiltration, both models focus on shoulders region rather than actual organs therefore both delivered considerably weak results. 

## Project Structure

```
├── src/
│   ├── dataset.py          # PyTorch Dataset class for ChestX-ray14
│   ├── model.py            # ResNet-50 model with freeze/unfreeze support
│   ├── train.py             # Baseline training script
│   ├── train_freeze.py      # Freeze/unfreeze training script
│   ├── evaluate.py          # Test set evaluation with per-class AUC
│   ├── utils.py             # Transforms and helper functions
│   └── download_data.py     # Dataset download script
├── notebooks/
│   ├── exploratory-data-analysis.ipynb
│   └── grad-cam-analysis.ipynb
├── outputs/                 # Grad-CAM visualizations and results
├── models/                  # Saved model checkpoints (not tracked)
├── data/                    # Dataset cache (not tracked)
├── requirements.txt
└── README.md
```

## Setup & Reproduce

```bash
# Clone the repo
git clone https://github.com/TrinhVox/medical-image-classification.git
cd medical-image-classification

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Download dataset (5k sample for dev, --full for entire dataset)
cd src
python download_data.py
python download_data.py --full  # ~45GB

# Train baseline
python train.py

# Train freeze/unfreeze
python train_freeze.py

# Evaluate on test set
python evaluate.py --model best_model_params.pt
python evaluate.py --model best_model_params_freeze.pt
```

## Future Work

The 0.19 AUC gap between Pneumonia (0.67) and Cardiomegaly (0.86) is the central open problem. Three experiments follow directly from the Grad-CAM analysis:

1. **DenseNet-121 backbone** — CheXNet uses dense connectivity to preserve fine-grained spatial features across layers. The hypothesis is that richer feature reuse will improve localisation of diffuse opacities without explicit attention supervision.

2. **Spatial attention / transformer head** — Replacing global average pooling with a self-attention pooling layer to retain spatial structure through the classification head. Motivated directly by the Grad-CAM finding that diffuse pathologies require distributed spatial reasoning rather than localised feature detection.


## References

- [Original NIH paper](https://arxiv.org/abs/1705.02315)
- [CheXNet (Rajpurkar et al.)](https://arxiv.org/abs/1711.05225)