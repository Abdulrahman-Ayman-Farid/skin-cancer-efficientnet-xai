# Skin Cancer Classification — Transfer Learning + Explainable AI

> **College Project** | EfficientNet-B0 + Grad-CAM | Binary Classification: Malignant vs Benign

## Overview

This project classifies skin cancer images as **malignant** or **benign** using:
- **Transfer Learning** with EfficientNet-B0 (pretrained on ImageNet, fine-tuned on ISIC dataset)
- **Explainable AI (XAI)** using Grad-CAM to visualize which skin regions influenced the model's prediction

## Project Structure

```
skin-cancer-efficientnet-xai/
├── data/                           # Dataset (created by setup_colab.py)
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
├── models/                         # Saved models
├── results/                        # All outputs
│   ├── metrics/                    # Confusion matrix, ROC curves
│   ├── gradcam/                    # Grad-CAM visualizations
│   └── inference/                  # Single-image inference outputs
├── load_data.py                    # Data loading
├── load_image.py                   # Image preprocessing
├── pre_processing.py               # Train/val split, augmentation
├── model_efficientnet_b0.py        # Training script
├── xai_gradcam.py                  # Grad-CAM generation
├── inference.py                    # Single-image inference
├── setup_colab.py                  # Kaggle dataset downloader
├── run_on_colab.ipynb              # Colab notebook (run this on Colab!)
├── requirements.txt
└── README.md
```

## Files

- **`load_data.py`** — Loads image paths and generates malignant/benign labels
- **`load_image.py`** — Reads, decodes, resizes, and normalizes images to tensors
- **`pre_processing.py`** — Train/validation split + augmentation; provides `pre_process_df()` for sampled data
- **`model_efficientnet_b0.py`** — Trains EfficientNet-B0 with ImageNet weights
- **`xai_gradcam.py`** — Generates Grad-CAM heatmaps and overlays
- **`inference.py`** — Single-image inference with Grad-CAM overlay
- **`setup_colab.py`** — Downloads ISIC dataset from Kaggle and organizes into `data/` folders
- **`run_on_colab.ipynb`** — **Main Colab notebook** — run this on Google Colab

## Dataset

[ISIC Skin Cancer Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) from Kaggle.
Organize it into `data/train/` and `data/test/` with `benign/` and `malignant/` subfolders.

## Google Colab Setup (Easy Method)

**Option 1: Use the Notebook (Recommended)**

1. Upload the entire repo to Google Colab or GitHub
2. Open `run_on_colab.ipynb` in Colab
3. Run cells sequentially — it handles everything:
   - Kaggle authentication
   - Dataset download
   - Model training
   - Grad-CAM generation
   - Results download

**Option 2: Manual Setup**

```bash
# 1. Upload repo to Colab
# 2. Configure Kaggle API (get kaggle.json from kaggle.com → Account → API)
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 3. Download and organize dataset
!python setup_colab.py

# 4. Train model
!python model_efficientnet_b0.py

# 5. Generate Grad-CAM
!python xai_gradcam.py

# 6. Test inference
!python inference.py --image data/test/malignant/example.jpg
```

**Outputs:**
- `models/efficientnet_b0_best.keras` — trained model
- `results/metrics/` — confusion matrix & ROC curves
- `results/gradcam/` — Grad-CAM visualizations
- `results/inference/` — single-image inference results

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
