"""
EfficientNet-B0 transfer-learning training & evaluation script.
Saves model to models/ and metrics plots to results/metrics/.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetB0

from load_data import load_data
from pre_processing import pre_processing

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)


def efficientnet_model():
    """Build EfficientNet-B0 with ImageNet weights and a custom classification head.

    Returns:
        tf.keras.Model: compiled EfficientNet-B0 model.
    """
    inputs = layers.Input(shape=(224, 224, 3), dtype=tf.float32, name="input_image")

    # Load pretrained EfficientNet-B0 (exclude original top classifier)
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

    # Custom head for 2-class skin-cancer classification
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs)

    tf.random.set_seed(42)

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(model, train_dataset, val_dataset, test_dataset):
    """Train with early stopping + checkpointing, then evaluate on test set.

    Returns:
        tuple: (history, test_probabilities, test_predictions)
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/efficientnet_b0_best.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        ),
    ]

    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        validation_steps=int(len(val_dataset)),
        callbacks=callbacks,
    )

    # Evaluate on test set
    model.evaluate(test_dataset)

    test_prob = model.predict(test_dataset, verbose=1)
    test_pred = tf.argmax(test_prob, axis=1).numpy()

    return history, test_prob, test_pred


def evaluate_model(test_df, test_prob, test_pred):
    """Generate confusion matrix, ROC curve, and classification report."""
    cm = confusion_matrix(test_df.label_encoded, test_pred)

    fig_cm = plt.figure(figsize=(4, 4))
    disp = sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        annot_kws={"size": 6},
        fmt="g",
        linewidths=1,
        linecolor="black",
        clip_on=False,
        xticklabels=["benign", "malignant"],
        yticklabels=["benign", "malignant"],
    )
    disp.set_title("EfficientNet - Confusion Matrix", fontsize=14)
    disp.set_xlabel("Predicted Label", fontsize=10)
    disp.set_ylabel("True Label", fontsize=10)
    plt.yticks(rotation=0)
    fig_cm.savefig("results/metrics/EfficientNet_CM.pdf", bbox_inches="tight")
    plt.close(fig_cm)

    # ROC curve (malignant class = index 1)
    fpr, tpr, _ = roc_curve(test_df.label_encoded, test_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    fig_roc, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("EfficientNet - ROC Curve")
    ax.legend(loc="lower right")
    fig_roc.savefig("results/metrics/EfficientNet_ROC.pdf", bbox_inches="tight")
    plt.close(fig_roc)

    print(
        classification_report(
            test_df.label_encoded,
            test_pred,
            target_names=["Benign", "Malignant"],
        )
    )


if __name__ == "__main__":
    # Load and preprocess data
    train_df, test_df = load_data()
    train_dataset, val_dataset, test_dataset = pre_processing(train_df, test_df)

    # Build model
    model = efficientnet_model()
    model.summary()

    # Train & evaluate
    history, test_prob, test_pred = train_and_evaluate(
        model, train_dataset, val_dataset, test_dataset
    )

    evaluate_model(test_df, test_prob, test_pred)