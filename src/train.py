"""
Training script for Plant Disease CNN Classifier.
Trains on PlantVillage dataset with data augmentation and validation tracking.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import matplotlib.pyplot as plt
import keras
from keras import callbacks

from src.data_loader import get_data_splits, IMAGE_SIZE
from src.model import build_plant_cnn, compile_model


def plot_history(history, save_path="reports/training_curves.png"):
    """Plot and save training & validation accuracy and loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", label="Training Accuracy", color="#2ecc71")
    ax1.plot(epochs_range, val_acc, "s--", label="Validation Accuracy", color="#3498db")
    ax1.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", label="Training Loss", color="#e74c3c")
    ax2.plot(epochs_range, val_loss, "s--", label="Validation Loss", color="#f39c12")
    ax2.set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss (Crossentropy)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves saved to {save_path}")


def train(dataset_dir="dataset", 
          model_dir="model", 
          epochs=10, 
          batch_size=32, 
          max_samples_per_class=400,
          learning_rate=0.001):
    """
    Execute model training workflow.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("=" * 60)
    print("PLANT VILLAGE DISEASE CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Dataset path: {dataset_dir}")
    print(f"Max samples per class: {max_samples_per_class}")
    print(f"Batch size: {batch_size} | Epochs: {epochs}")

    # Load data splits
    train_ds, val_ds, class_names, class_to_idx = get_data_splits(
        dataset_dir=dataset_dir,
        max_samples_per_class=max_samples_per_class,
        train_ratio=0.8,
        batch_size=batch_size,
        seed=42
    )

    num_classes = len(class_names)
    print(f"\nDiscovered {num_classes} classes: {class_names}")
    print(f"Training batches: {len(train_ds)} | Validation batches: {len(val_ds)}")

    # Save class mapping
    class_meta_path = os.path.join(model_dir, "class_indices.json")
    with open(class_meta_path, "w") as f:
        json.dump({
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "image_size": list(IMAGE_SIZE),
            "num_classes": num_classes
        }, f, indent=4)
    print(f"Saved class metadata to {class_meta_path}")

    # Build CNN
    print("\nBuilding CNN Architecture...")
    model = build_plant_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=num_classes)
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    # Training callbacks
    model_save_path = os.path.join(model_dir, "plant_disease_model.keras")
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1
    )

    # Save final model if not already saved by checkpoint
    if not os.path.exists(model_save_path):
        model.save(model_save_path)
    print(f"\nTrained model saved to {model_save_path}")

    # Plot training metrics
    plot_history(history, save_path="reports/training_curves.png")

    # Save history stats
    history_path = "reports/history.json"
    with open(history_path, "w") as f:
        # Convert any float32 to float
        hist_clean = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(hist_clean, f, indent=4)
    print(f"Training metrics saved to {history_path}")

    return model, history, val_ds, class_names


if __name__ == "__main__":
    train(epochs=10, max_samples_per_class=400)
