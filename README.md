# Skin Cancer Classification — Transfer Learning + Explainable AI

> **College Project** | EfficientNet-B0 + Grad-CAM | Binary Classification: Malignant vs Benign

## Overview

This project classifies skin cancer images as **malignant** or **benign** using:
- **Transfer Learning** with EfficientNet-B0 (pretrained on ImageNet, fine-tuned on ISIC dataset)
- **Explainable AI (XAI)** using Grad-CAM to visualize which skin regions influenced the model's prediction

## Project Pipeline

```
Load Data → Preprocess → Train EfficientNet-B0 → Evaluate → Apply Grad-CAM
```

1. `load_data.py` — Loads ISIC dataset image paths and generates malignant/benign labels
2. `load_image.py` — Reads, decodes, resizes and normalizes images to tensors
3. `pre_processing.py` — Train/validation/test split + augmentation (RandomFlip, RandomZoom)
4. `model_efficientnet_b0.py` — Builds, trains and evaluates the EfficientNet-B0 model
5. `xai_gradcam_reference.py` — Reference XAI implementation with Grad-CAM and LIME

## Dataset

[ISIC Skin Cancer Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) from Kaggle.
Data is organized into `data/train/` and `data/test/` folders with subfolders named `malignant` and `benign`.

> Update the paths in `load_data.py` to match your local or cloud environment.

## Requirements

```bash
pip install tensorflow==2.16 keras scikit-learn matplotlib seaborn pandas numpy
pip install tf-keras-vis lime
```

## How to Run

```bash
# 1. Update dataset paths in load_data.py
# 2. Train and evaluate
python model_efficientnet_b0.py
```

## Results

The model produces:
- Confusion Matrix
- ROC Curve
- Classification Report (Precision, Recall, F1)
- Grad-CAM heatmaps highlighting the skin regions driving predictions

## Tech Stack

| Component | Tool |
|-----------|------|
| Model | EfficientNet-B0 (Keras) |
| Framework | TensorFlow 2.16 |
| XAI | Grad-CAM (tf-keras-vis) |
| Dataset | ISIC (Kaggle) |
