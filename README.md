# Skin Cancer Classification — Transfer Learning + Explainable AI

> **College Project** | EfficientNet-B0 + Grad-CAM | Binary Classification: Malignant vs Benign

## Overview

This project classifies skin cancer images as **malignant** or **benign** using:
- **Transfer Learning** with EfficientNet-B0 (pretrained on ImageNet, fine-tuned on ISIC dataset)
- **Explainable AI (XAI)** using Grad-CAM to visualize which skin regions influenced the model's prediction

## Project Structure

```
skin-cancer-efficientnet-xai/
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
├── models/
├── results/
│   ├── metrics/
│   ├── gradcam/
│   └── inference/
├── load_data.py
├── load_image.py
├── pre_processing.py
├── model_efficientnet_b0.py
├── xai_gradcam.py
├── inference.py
├── requirements.txt
└── README.md
```

## Files

- **`load_data.py`** — Loads image paths and generates malignant/benign labels
- **`load_image.py`** — Reads, decodes, resizes, and normalizes images to tensors
- **`pre_processing.py`** — Train/validation split + augmentation (`RandomFlip`, `RandomZoom`); also provides `pre_process_df()` for sampled data
- **`model_efficientnet_b0.py`** — Builds EfficientNet-B0 with ImageNet weights, trains, evaluates, and saves the best model
- **`xai_gradcam.py`** — Generates Grad-CAM heatmaps and overlays for sampled test images
- **`inference.py`** — Single-image inference with Grad-CAM overlay

## Dataset

[ISIC Skin Cancer Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) from Kaggle.
Organize it into `data/train/` and `data/test/` with `benign/` and `malignant/` subfolders.

## Google Colab Setup

1. **Mount Drive** (if storing data on Google Drive):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Clone or upload this repo** to `/content/skin-cancer-efficientnet-xai`.

3. **Symlink or copy data**:
   ```python
   !mkdir -p /content/skin-cancer-efficientnet-xai/data
   !cp -r /content/drive/MyDrive/ISIC_data/train /content/skin-cancer-efficientnet-xai/data/
   !cp -r /content/drive/MyDrive/ISIC_data/test /content/skin-cancer-efficientnet-xai/data/
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run training**:
   ```bash
   cd /content/skin-cancer-efficientnet-xai
   python model_efficientnet_b0.py
   ```

   Trained model is saved to `models/efficientnet_b0_best.keras`.
   Metrics are saved to `results/metrics/`.

6. **Run Grad-CAM on sampled test images**:
   ```bash
   python xai_gradcam.py
   ```
   Outputs go to `results/gradcam/`.

7. **Run single-image inference**:
   ```bash
   python inference.py --image data/test/malignant/example.jpg
   ```
   Output goes to `results/inference/`.

## Local Setup

```bash
pip install -r requirements.txt
python model_efficientnet_b0.py
python xai_gradcam.py
python inference.py --image path/to/image.jpg
```

## Results

- **Training**: confusion matrix, ROC curve, classification report
- **Grad-CAM**: per-image original, heatmap, and overlay saved under `results/gradcam/`
- **Inference**: single-image prediction + Grad-CAM overlay saved under `results/inference/`

## Tech Stack

| Component | Tool |
|-----------|------|
| Model | EfficientNet-B0 (Keras Applications) |
| Framework | TensorFlow 2.x |
| XAI | Grad-CAM (manual `tf.GradientTape` implementation) |
| Dataset | ISIC (Kaggle) |
