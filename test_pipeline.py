"""
End-to-end automated verification script for Plant Disease Detection pipeline.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import cv2
import numpy as np
import keras

from src.data_loader import preprocess_image_cv2, get_opencv_visualizations, IMAGE_SIZE
from app import predict_leaf


def test_pipeline():
    print("=" * 60)
    print("VERIFYING PLANT DISEASE DETECTION PIPELINE")
    print("=" * 60)

    # 1. Dataset verification
    dataset_dir = "dataset"
    assert os.path.exists(dataset_dir), "Dataset folder missing!"
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"✓ Dataset verified: {len(classes)} classes found -> {classes}")

    # 2. Sample image verification
    sample_dir = "sample_images"
    assert os.path.exists(sample_dir), "sample_images folder missing!"
    sample_files = os.listdir(sample_dir)
    assert len(sample_files) > 0, "No sample images found!"
    print(f"✓ Sample images verified: {len(sample_files)} sample leaves ready for testing.")

    # 3. Model & Metadata verification
    model_path = "model/plant_disease_model.keras"
    meta_path = "model/class_indices.json"
    assert os.path.exists(model_path), "Trained model file missing!"
    assert os.path.exists(meta_path), "Class metadata missing!"
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    print(f"✓ Class metadata verified: {len(class_names)} classes.")

    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully: {model.name}")

    # 4. OpenCV preprocessing & inference verification
    test_img_path = os.path.join(sample_dir, sample_files[0])
    img_bgr = cv2.imread(test_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    norm_img, resized = preprocess_image_cv2(img_rgb, target_size=IMAGE_SIZE)
    assert norm_img.shape == (128, 128, 3), f"Incorrect shape: {norm_img.shape}"
    assert 0.0 <= norm_img.max() <= 1.0, "Normalization bounds violated!"
    print(f"✓ OpenCV preprocessing verified (Normalized min={norm_img.min():.2f}, max={norm_img.max():.2f})")

    cv_vis = get_opencv_visualizations(resized)
    assert "grayscale" in cv_vis and "edges" in cv_vis and "lesion_mask" in cv_vis
    print("✓ OpenCV visualization suite (Grayscale, Canny Edges, HSV Necrosis) verified.")

    # 5. Prediction verification
    pred_class, confidence, all_probs, _ = predict_leaf(model, class_names, img_rgb)
    assert pred_class in class_names, "Predicted class not in class names!"
    assert 0.0 <= confidence <= 1.0, "Confidence out of range!"
    print(f"✓ Model inference verified on '{sample_files[0]}':")
    print(f"   -> Predicted Class: {pred_class} (Confidence: {confidence * 100:.2f}%)")

    # 6. Evaluation metrics verification
    eval_json = "reports/evaluation_results.json"
    curves_img = "reports/training_curves.png"
    cm_img = "reports/confusion_matrix.png"
    assert os.path.exists(eval_json), "Evaluation results JSON missing!"
    assert os.path.exists(curves_img), "Training curves image missing!"
    assert os.path.exists(cm_img), "Confusion matrix image missing!"
    
    with open(eval_json, "r") as f:
        eval_data = json.load(f)
    print(f"✓ Evaluation report verified: Accuracy = {eval_data['overall_accuracy'] * 100:.2f}%")
    print(f"✓ Confusion Matrix & Training Curves verified in reports/ directory.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED: PIPELINE READY FOR PRODUCTION / STREAMLIT")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
