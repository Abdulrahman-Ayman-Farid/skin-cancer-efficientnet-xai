"""
Single-image inference script with Grad-CAM overlay.

Example:
    python inference.py --image path/to/image.jpg
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

from load_image import load_image
from xai_gradcam import make_gradcam_heatmap, get_last_conv_layer_name, normalize

os.makedirs("results/inference", exist_ok=True)

LABELS = {0: "benign", 1: "malignant"}


def inference_pipeline(image_path: str, model_path: str = "models/efficientnet_b0_best.keras"):
    """Run inference + Grad-CAM on a single image.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved .keras model.

    Returns:
        dict: Prediction result with label, confidence, and output path.
    """
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    img_tensor = load_image(image_path)  # (224, 224, 3), float32 [0,1]
    img_batch = tf.expand_dims(img_tensor, axis=0)  # (1, 224, 224, 3)

    # Predict
    preds = model.predict(img_batch, verbose=0)
    pred_index = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_index])
    pred_label = LABELS[pred_index]

    # Grad-CAM
    last_conv_layer = get_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer, pred_index)

    # Overlay
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], [224, 224], method="bilinear"
    ).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    heatmap_uint8 = np.uint8(255 * normalize(heatmap_resized))
    jet_heatmap = jet_colors[heatmap_uint8]

    original_disp = np.uint8(255 * img_tensor.numpy())
    overlay = np.uint8(jet_heatmap * 0.4 + original_disp * 0.6)

    # Save figure
    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join("results/inference", f"{basename}_gradcam.png")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original_disp)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(heatmap_resized, cmap="jet")
    axs[1].set_title("Grad-CAM Heatmap")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title(f"Prediction: {pred_label} ({confidence:.2%})")
    axs[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "label": pred_label,
        "confidence": confidence,
        "output_path": out_path,
    }
    print(f"Prediction: {pred_label} ({confidence:.2%})")
    print(f"Saved overlay to: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Cancer Inference + Grad-CAM")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        default="models/efficientnet_b0_best.keras",
        help="Path to saved model",
    )
    args = parser.parse_args()

    inference_pipeline(args.image, model_path=args.model)
