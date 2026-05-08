"""
Load dataset paths and labels into DataFrames.
Expects data/train/ and data/test/ folders with benign/ and malignant/ subfolders.
"""

# Import required packages
from pathlib import Path
import pandas as pd


def load_data(data_dir: str = "data"):
    """Load image paths and labels into DataFrames.

    Args:
        data_dir (str): Root data directory containing train/ and test/.

    Returns:
        tuple: (train_df, test_df) with columns image_path, label, label_encoded.
    """
    data_path = Path(data_dir)
    path_train = data_path / "train"
    path_test = data_path / "test"

    # Collect all jpg images recursively
    train_images = sorted([str(p) for p in path_train.rglob("*.jpg")])
    test_images = sorted([str(p) for p in path_test.rglob("*.jpg")])

    print(f"train samples count: {len(train_images)}")
    print(f"test samples count: {len(test_images)}")

    # Derive label from parent folder name
    train_labels = [Path(p).parent.name for p in train_images]
    test_labels = [Path(p).parent.name for p in test_images]

    train_df = pd.DataFrame({
        "image_path": train_images,
        "label": train_labels
    })

    test_df = pd.DataFrame({
        "image_path": test_images,
        "label": test_labels
    })

    # Encode labels: malignant -> 1, benign -> 0
    train_df["label_encoded"] = (train_df["label"] == "malignant").astype(int)
    test_df["label_encoded"] = (test_df["label"] == "malignant").astype(int)

    return train_df, test_df
