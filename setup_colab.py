"""
Setup script for Google Colab.
Downloads ISIC dataset from Kaggle and organizes into benign/malignant folders.
"""
import os
import shutil
from pathlib import Path
import random

def setup_isic_data():
    """Download ISIC dataset and organize for binary classification."""
    print("Setting up ISIC dataset for binary classification...")
    
    # Try importing kagglehub, install if needed
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system("pip install -q kagglehub")
        import kagglehub
    
    # Download dataset
    print("Downloading dataset from Kaggle (this may take a few minutes)...")
    dataset_path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
    print(f"Downloaded to: {dataset_path}")
    
    # Define class mapping to benign/malignant
    # Based on ISIC archive medical classification
    benign_classes = ['nv', 'bkl', 'df', 'vasc']  # Nevus, Benign Keratosis, Dermatofibroma, Vascular
    malignant_classes = ['mel', 'bcc', 'akiec']    # Melanoma, Basal Cell Carcinoma, Actinic Keratosis
    
    # Create directory structure
    base_dir = Path("data")
    for split in ['train', 'test']:
        for label in ['benign', 'malignant']:
            (base_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Process training data
    train_src = Path(dataset_path) / "train"
    if train_src.exists():
        for class_name in benign_classes:
            src_dir = train_src / class_name
            if src_dir.exists():
                for img in src_dir.glob("*.jpg"):
                    shutil.copy(img, base_dir / "train" / "benign" / img.name)
        
        for class_name in malignant_classes:
            src_dir = train_src / class_name
            if src_dir.exists():
                for img in src_dir.glob("*.jpg"):
                    shutil.copy(img, base_dir / "train" / "malignant" / img.name)
    
    # Process test data
    test_src = Path(dataset_path) / "test"
    if test_src.exists():
        for class_name in benign_classes:
            src_dir = test_src / class_name
            if src_dir.exists():
                for img in src_dir.glob("*.jpg"):
                    shutil.copy(img, base_dir / "test" / "benign" / img.name)
        
        for class_name in malignant_classes:
            src_dir = test_src / class_name
            if src_dir.exists():
                for img in src_dir.glob("*.jpg"):
                    shutil.copy(img, base_dir / "test" / "malignant" / img.name)
    
    # Print statistics
    train_benign = len(list((base_dir / "train" / "benign").glob("*.jpg")))
    train_malignant = len(list((base_dir / "train" / "malignant").glob("*.jpg")))
    test_benign = len(list((base_dir / "test" / "benign").glob("*.jpg")))
    test_malignant = len(list((base_dir / "test" / "malignant").glob("*.jpg")))
    
    print(f"\nDataset organized:")
    print(f"  Train - Benign: {train_benign}, Malignant: {train_malignant}")
    print(f"  Test  - Benign: {test_benign}, Malignant: {test_malignant}")
    print(f"  Total: {train_benign + train_malignant + test_benign + test_malignant} images")

if __name__ == "__main__":
    setup_isic_data()
