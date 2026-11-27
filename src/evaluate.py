"""
Evaluation and performance assessment script for Plant Disease CNN model.
Computes:
- Overall Accuracy & Loss
- Precision, Recall, and F1-Score per class
- Confusion Matrix Heatmap
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras

from src.data_loader import get_data_splits


def evaluate_model(model_path="model/plant_disease_model.keras", 
                   class_indices_path="model/class_indices.json",
                   dataset_dir="dataset",
                   output_dir="reports"):
    """
    Run evaluation on the trained CNN model.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PLANT VILLAGE DISEASE CLASSIFIER - MODEL EVALUATION")
    print("=" * 60)

    # 1. Load Metadata
    with open(class_indices_path, "r") as f:
        meta = json.load(f)
    class_names = meta["class_names"]

    # 2. Load Validation Data
    _, val_ds, _, _ = get_data_splits(
        dataset_dir=dataset_dir,
        max_samples_per_class=400,
        train_ratio=0.8,
        batch_size=32,
        seed=42
    )

    # 3. Load Trained Model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    # 4. Gather ground truth and predictions
    print("Running inference across validation batches...")
    y_true_all = []
    y_pred_probs_all = []

    for i in range(len(val_ds)):
        x_batch, y_batch = val_ds[i]
        preds = model.predict(x_batch, verbose=0)
        y_true_all.extend(y_batch)
        y_pred_probs_all.append(preds)

    y_true = np.array(y_true_all)
    y_pred_probs = np.vstack(y_pred_probs_all)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Metrics calculation
    overall_acc = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    clf_report_text = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- CLASSIFICATION REPORT ---")
    print(clf_report_text)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%\n")

    # 6. Plot Confusion Matrix
    clean_labels = [name.replace("___", " - ").replace("_", " ") for name in class_names]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Greens",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar=True
    )
    plt.title("Plant Disease Classification - Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Disease Class", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_save_path}")

    # 7. Save metrics report JSON
    eval_results = {
        "overall_accuracy": float(overall_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report
    }
    report_json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(report_json_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Saved evaluation metrics to {report_json_path}")

    return overall_acc, clf_report, cm


if __name__ == "__main__":
    evaluate_model()
