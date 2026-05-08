"""
Standalone Grad-CAM module for EfficientNet-B0.
Samples test images, generates heatmaps/overlays, and saves results.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image

from load_data import load_data
from pre_processing import pre_process_df

os.makedirs("results/gradcam", exist_ok=True)


def normalize(array):
    """Min-max normalize array to [0, 1]."""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val > min_val:
        return (array - min_val) / (max_val - min_val)
    return array


def get_last_conv_layer_name(model):
    """Auto-detect the last 4D convolutional layer name in the model."""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No 4D convolutional layer found in model.")


def make_gradcam_heatmap(img_array, model, layer_name, class_idx):
    """Generate a Grad-CAM heatmap for a given image and class index.

    Args:
        img_array (tf.Tensor): 4D batch tensor (1, H, W, C).
        model (tf.keras.Model): Trained model.
        layer_name (str): Target convolutional layer name.
        class_idx (int): Index of the class to explain.

    Returns:
        np.ndarray: 2D normalized heatmap.
    """
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam_outputs(
    original_img, heatmap, pred_label, confidence, true_label, basename
):
    """Save original image, heatmap, and overlay side-by-side."""
    # Resize heatmap to original image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], [224, 224], method="bilinear"
    ).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    # Overlay using jet colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    heatmap_uint8 = np.uint8(255 * normalize(heatmap_resized))
    jet_heatmap = jet_colors[heatmap_uint8]

    # Original image for display (already float32 [0,1])
    if original_img.max() <= 1.0:
        original_disp = np.uint8(255 * original_img)
    else:
        original_disp = np.uint8(original_img)

    overlay = np.uint8(jet_heatmap * 0.4 + original_disp * 0.6)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original_disp)
    axs[0].set_title(f"Original\nTrue: {true_label}")
    axs[0].axis("off")

    axs[1].imshow(heatmap_resized, cmap="jet")
    axs[1].set_title("Grad-CAM Heatmap")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title(f"Overlay\nPred: {pred_label} ({confidence:.2%})")
    axs[2].axis("off")

    fig.tight_layout()
    save_path = os.path.join("results/gradcam", f"{basename}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_gradcam_on_model(model_path="models/efficientnet_b0_best.keras"):
    """Load trained model, sample test images, and generate Grad-CAM outputs."""
    # Load data
    _, test_df = load_data()

    # Sample 5 benign (0) and 5 malignant (1)
    class0 = test_df[test_df["label_encoded"] == 0].sample(5, random_state=10)
    class1 = test_df[test_df["label_encoded"] == 1].sample(5, random_state=10)
    sampled_df = pd.concat([class0, class1]).reset_index(drop=True)

    # Preprocess sampled data (no augmentation)
    sampled_dataset = pre_process_df(sampled_df, augmentation=False)

    # Load model
    model = tf.keras.models.load_model(model_path)
    last_conv_layer = get_last_conv_layer_name(model)
    print(f"Using last conv layer for Grad-CAM: {last_conv_layer}")

    label_names = {0: "benign", 1: "malignant"}

    # Process each sampled image
    for i, (images, labels) in enumerate(sampled_dataset.unbatch().take(10)):
        img_array = tf.expand_dims(images, axis=0)  # add batch dim
        true_label = int(np.argmax(labels.numpy()))

        preds = model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_index])

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_index)

        basename = f"sample_{i+1:02d}_{label_names[true_label]}"
        save_gradcam_outputs(
            images.numpy(),
            heatmap,
            label_names[pred_index],
            confidence,
            label_names[true_label],
            basename,
        )
        print(f"Saved Grad-CAM for {basename}")

    # Create a summary grid
    grid_path = os.path.join("results/gradcam", "grid.png")
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    axes = axes.flatten()

    image_files = sorted(
        [f for f in os.listdir("results/gradcam") if f.endswith(".png") and f != "grid.png"]
    )
    for ax, img_file in zip(axes, image_files):
        img = Image.open(os.path.join("results/gradcam", img_file))
        ax.imshow(img)
        ax.set_title(img_file.replace(".png", ""), fontsize=8)
        ax.axis("off")

    # Hide unused subplots
    for ax in axes[len(image_files) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary grid to {grid_path}")


if __name__ == "__main__":
    run_gradcam_on_model()
