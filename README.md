# 🌿 Plant Disease Identification from Leaf Images

An end-to-end Deep Learning and Computer Vision application to diagnose plant foliar infections from leaf images using the benchmark **PlantVillage** dataset, custom **Convolutional Neural Networks (CNN)**, **OpenCV**, and an interactive **Streamlit** dashboard with real-time agronomic recommendations.

---

## 📌 Project Overview & Problem Statement
Crop diseases cause severe agricultural yield losses worldwide. Early and automated detection of foliar symptoms allows farmers and agronomists to apply targeted organic or chemical treatments before infections spread.

This system demonstrates a complete computer vision pipeline:
1. **Dataset Ingestion**: Authentic PlantVillage leaf benchmark images across multiple crop varieties and disease stages.
2. **Preprocessing**: Color space conversion (BGR to RGB), spatial resolution standardization ($128 \times 128$), and pixel normalization ($[0.0, 1.0]$).
3. **Data Augmentation**: Geometric (random horizontal & vertical flips, rotations up to $\pm 25^\circ$) and photometric adjustments (brightness shifts) to prevent overfitting.
4. **Custom CNN Architecture**: Deep Convolutional blocks (`Conv2D`), spatial downsampling (`MaxPooling2D`), and regularization (`Dropout`).
5. **Multiclass Classification**: Softmax probability head outputting predictions across 6 major disease categories.
6. **Evaluation**: Comprehensive classification metrics, Confusion Matrix heatmap, and learning curves.
7. **Streamlit Deployment**: Production-ready web interface with file upload, webcam capture, sample leaf testing, OpenCV visual decomposition lab, and treatment protocols.

---

## 🛠️ Technologies & Concepts Demonstrated

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Deep Learning Framework** | TensorFlow / Keras 3 | CNN model construction, training, callbacks, serialization |
| **Backend Engine** | PyTorch (`torch`) | Tensor execution engine for fast CPU/GPU computation |
| **Computer Vision** | OpenCV (`cv2`) | BGR/RGB decoding, resizing, normalization, Canny edges, HSV lesion masking |
| **Web Interface** | Streamlit | Responsive dashboard, camera input, probability visualizations |
| **Evaluation** | Scikit-Learn, Seaborn | Confusion matrix, Precision, Recall, F1-Score |

### Deep Learning Concepts Demonstrated:
- **Image Preprocessing**: Color space transformations and float32 scaling into $[0.0, 1.0]$ to prevent vanishing/exploding gradients.
- **Convolution & Pooling**: Hierarchical feature maps extracting low-level edges, textures, leaf veins, and disease lesion spots with spatial invariance.
- **Data Augmentation**: Robust synthetic perturbations to build model invariance to camera angle, orientation, and lighting conditions.
- **Dropout & Overfitting Prevention**: Strategic dropout rates ($0.25$, $0.30$, and $0.50$) combined with early stopping to prevent memorization of background artifacts.
- **Multiclass Classification**: Softmax activation mapped to 6 discrete crop health classes.

---

## 📂 Project Directory Structure

```text
Plant_Detection/
├── dataset/                         # PlantVillage dataset subset (6 classes)
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   └── Tomato___healthy/
├── model/
│   ├── plant_disease_model.keras    # Saved trained CNN model
│   └── class_indices.json           # Class label mapping & metadata
├── reports/
│   ├── confusion_matrix.png         # Heatmap of predicted vs ground truth
│   ├── training_curves.png          # Accuracy and Loss trajectories
│   ├── evaluation_results.json      # Precision, recall, and F1 metrics
│   └── history.json                 # Epoch-by-epoch loss and accuracy
├── sample_images/                   # Sample test leaves for 1-click Streamlit demo
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # OpenCV loader, augmentation, and PyDataset
│   ├── model.py                     # CNN model architecture definition
│   ├── train.py                     # Training script with callbacks
│   └── evaluate.py                  # Evaluation suite & confusion matrix generator
├── app.py                           # Streamlit web application
├── test_pipeline.py                 # Automated verification test suite
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

---

## 🚀 How to Run the Application

### 1. Launch the Streamlit Web Application
Run the following command in your terminal:
```bash
streamlit run app.py
```
This opens the web interface in your browser (typically `http://localhost:8501`).

### 2. Retrain the CNN Model (Optional)
To retrain the model from scratch:
```bash
python -m src.train
```

### 3. Run Model Evaluation (Optional)
To regenerate evaluation metrics and the confusion matrix:
```bash
python -m src.evaluate
```

### 4. Run Automated Pipeline Tests
To verify all components end-to-end:
```bash
python test_pipeline.py
```

---

## 📊 Model Evaluation Results

- **Validation Accuracy**: **77.03%**
- **Macro Precision**: **0.78**
- **Macro F1-Score**: **0.74**

The confusion matrix and training curves are saved in the `reports/` folder.
