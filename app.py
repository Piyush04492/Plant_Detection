"""
Streamlit Web Application for Plant Disease Detection
Powered by TensorFlow / Keras CNN and OpenCV.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import keras

from src.data_loader import preprocess_image_cv2, get_opencv_visualizations, IMAGE_SIZE

# --- Page Configuration ---
st.set_page_config(
    page_title="PlantGuard AI - Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2d6a4f;
        margin-bottom: 1rem;
    }
    .diagnosis-badge {
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-healthy {
        background-color: #d8f3dc;
        color: #1b4332;
        border: 1px solid #74c69d;
    }
    .badge-diseased {
        background-color: #ffe3e3;
        color: #9b2226;
        border: 1px solid #f28482;
    }
</style>
""", unsafe_allow_html=True)

# --- Agronomic Disease Knowledge Base ---
DISEASE_INFO = {
    "Potato___Early_blight": {
        "crop": "Potato (Solanum tuberosum)",
        "disease": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "status": "Diseased",
        "severity": "High",
        "symptoms": "Dark brown, concentric ring spots ('target board' pattern) appearing first on older lower leaves, leading to yellowing (chlorosis) and defoliation.",
        "causes": "Warm temperatures (24-29°C) combined with high humidity, heavy dew, or prolonged leaf wetness.",
        "organic_treatment": [
            "Apply copper-based fungicidal soap or liquid copper spray.",
            "Spray Bacillus subtilis or Serenade bio-fungicide weekly.",
            "Remove and destroy severely affected lower foliage to prevent spore dispersal."
        ],
        "chemical_treatment": [
            "Foliar spray with Mancozeb (2g/L) or Chlorothalonil (2ml/L).",
            "Apply systemic fungicides such as Azoxystrobin or Difenoconazole at initial symptom onset."
        ],
        "prevention": [
            "Adopt 3-4 year crop rotation away from solanaceous plants.",
            "Use drip irrigation rather than overhead sprinklers to keep leaves dry.",
            "Ensure proper plant spacing for sunlight penetration and airflow."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato (Solanum tuberosum)",
        "disease": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "status": "Diseased",
        "severity": "Critical",
        "symptoms": "Water-soaked irregular black/dark-brown lesions on leaves and stems, with white cottony mildew under leaf surfaces in humid weather. Rapidly blights entire plant canopy.",
        "causes": "Cool, damp weather with temperatures between 15-22°C and relative humidity above 90%.",
        "organic_treatment": [
            "Prophylactic application of Bordeaux mixture (copper sulfate + slaked lime).",
            "Immediate pruning of infected foliage; dispose of infected tubers away from field."
        ],
        "chemical_treatment": [
            "Metalaxyl + Mancozeb (Ridomil MZ @ 2.5g/L) as curative treatment.",
            "Cymoxanil + Mancozeb or Dimethomorph during active disease outbreaks."
        ],
        "prevention": [
            "Plant certified disease-free, resistant seed tubers.",
            "Destroy volunteer potato cull piles near fields.",
            "Apply preventative fungicides before forecast rainy periods."
        ]
    },
    "Potato___healthy": {
        "crop": "Potato (Solanum tuberosum)",
        "disease": "Healthy Leaf",
        "pathogen": "None (Plant is robust and pathogen-free)",
        "status": "Healthy",
        "severity": "None",
        "symptoms": "Leaves are vibrant green with smooth edges, intact cellular structure, and no necrotic lesions or chlorosis.",
        "causes": "Good nutrition, balanced soil moisture, and optimal agronomic management.",
        "organic_treatment": ["Maintain regular compost top-dressing and neem oil preventive sprays."],
        "chemical_treatment": ["No chemical treatment required."],
        "prevention": ["Continue balanced N-P-K fertilization and regular scouting for pests/blight."]
    },
    "Tomato___Early_blight": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "status": "Diseased",
        "severity": "Moderate to High",
        "symptoms": "Brown to black spots with concentric rings surrounded by yellow halos on mature leaves. Leads to premature leaf drop and collar rot on stems.",
        "causes": "High humidity, rain splash carrying spores from soil, and temperature above 24°C.",
        "organic_treatment": [
            "Apply copper octanoate or neem extract spray every 7-10 days.",
            "Mulch heavily around the base to prevent soil splash onto lower foliage."
        ],
        "chemical_treatment": [
            "Chlorothalonil or Mancozeb sprays at early disease detection.",
            "Pyraclostrobin or Boscalid for resistant commercial crops."
        ],
        "prevention": [
            "Stake and prune indeterminate tomatoes to maintain upward airflow.",
            "Avoid working in tomato fields when leaves are wet.",
            "Clean and sanitize pruning shears with 70% alcohol."
        ]
    },
    "Tomato___Late_blight": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "status": "Diseased",
        "severity": "Critical",
        "symptoms": "Rapidly enlarging dark brown, greasy, water-soaked patches on leaves and green fruit; leaves turn papery brown and die rapidly.",
        "causes": "High relative humidity (>90%) with cool moderate temperatures (12-24°C).",
        "organic_treatment": [
            "Apply fixed copper fungicide before infection takes hold.",
            "Immediately remove and bag all infected plant tissue (never compost)."
        ],
        "chemical_treatment": [
            "Apply systemic oomycide fungicides like Mandipropamid (Revus) or Fluopicolide.",
            "Protective Mancozeb / Chlorothalonil application."
        ],
        "prevention": [
            "Select resistant cultivars (e.g., Defiant, Mountain Magic).",
            "Ensure wide row spacing and drip lines.",
            "Inspect daily during overcast, wet spells."
        ]
    },
    "Tomato___healthy": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Healthy Leaf",
        "pathogen": "None (Plant is robust and disease-free)",
        "status": "Healthy",
        "severity": "None",
        "symptoms": "Lush, uniformly green compound leaves without chlorosis, fungal sporulation, or leaf spots.",
        "causes": "Balanced soil fertility, appropriate watering, and sunlight exposure.",
        "organic_treatment": ["Maintain balanced organic manure and calcium levels to prevent blossom end rot."],
        "chemical_treatment": ["No chemical treatment required."],
        "prevention": ["Maintain consistent watering schedule and prophylactic pest scouting."]
    }
}


@st.cache_resource
def load_disease_classifier():
    """Load model and class indices once and cache in memory."""
    model_path = "model/plant_disease_model.keras"
    meta_path = "model/class_indices.json"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    model = keras.models.load_model(model_path)
    return model, meta["class_names"]


def predict_leaf(model, class_names, image_array):
    """
    Run CNN forward pass and return predicted class, confidence, and all probabilities.
    """
    norm_img, resized_img = preprocess_image_cv2(image_array, target_size=IMAGE_SIZE)
    # Expand batch dimension: (1, 128, 128, 3)
    batch_input = np.expand_dims(norm_img, axis=0)
    
    preds = model.predict(batch_input, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    predicted_class = class_names[pred_idx]
    
    all_probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return predicted_class, confidence, all_probs, resized_img


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/cotton/128/plant--v1.png", width=90)
    st.sidebar.title("PlantGuard AI")
    st.sidebar.markdown("**Deep Learning Plant Disease Diagnostic System**")
    
    app_mode = st.sidebar.radio(
        "Navigation",
        ["🔍 Disease Diagnosis", "📊 Model Performance & Metrics", "🔬 OpenCV Preprocessing Lab", "ℹ️ About & Architecture"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Info")
    st.sidebar.caption("• Model: Custom Deep CNN (Keras 3)")
    st.sidebar.caption("• Dataset: PlantVillage Benchmark")
    st.sidebar.caption("• Image Preprocessor: OpenCV")
    st.sidebar.caption("• Target Resolution: 128x128 RGB")
    
    model, class_names = load_disease_classifier()
    
    if model is None:
        st.error("⚠️ Model or class metadata not found. Please train the model first by running `python -m src.train`.")
        return

    # --- Mode 1: Disease Diagnosis ---
    if app_mode == "🔍 Disease Diagnosis":
        st.markdown('<div class="main-header">🌿 Plant Disease Detection & Advisory</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Identify plant foliage infections with deep convolutional neural networks and get instant agronomic remedies.</div>', 
            unsafe_allow_html=True
        )
        
        # Input selection method
        input_source = st.radio(
            "Select Image Input Method:",
            ["📁 Upload Image", "📸 Use Camera", "🖼️ Try Sample Leaf Images"],
            horizontal=True
        )
        
        leaf_img = None
        img_name = "Leaf Image"

        if input_source == "📁 Upload Image":
            uploaded_file = st.file_uploader("Upload leaf photograph (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                leaf_img = Image.open(uploaded_file).convert("RGB")
                img_name = uploaded_file.name

        elif input_source == "📸 Use Camera":
            camera_file = st.camera_input("Capture leaf using your webcam / mobile camera:")
            if camera_file is not None:
                leaf_img = Image.open(camera_file).convert("RGB")
                img_name = "Camera Capture"

        elif input_source == "🖼️ Try Sample Leaf Images":
            sample_dir = "sample_images"
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                selected_sample = st.selectbox("Choose a sample leaf from the PlantVillage dataset:", sample_files)
                if selected_sample:
                    sample_path = os.path.join(sample_dir, selected_sample)
                    leaf_img = Image.open(sample_path).convert("RGB")
                    img_name = selected_sample

        # If image available, run diagnosis
        if leaf_img is not None:
            img_np = np.array(leaf_img)
            
            col_img, col_diag = st.columns([1, 1.2])
            
            with col_img:
                st.subheader("📷 Inspected Leaf")
                st.image(leaf_img, caption=img_name, use_container_width=True)
                st.caption(f"Original Resolution: {leaf_img.size[0]} x {leaf_img.size[1]} px")

            with col_diag:
                st.subheader("🧪 AI Diagnostic Result")
                with st.spinner("Analyzing leaf textures and lesion spots..."):
                    pred_class, confidence, all_probs, resized_img = predict_leaf(model, class_names, img_np)
                
                info = DISEASE_INFO.get(pred_class, {})
                status_color = "badge-healthy" if info.get("status") == "Healthy" else "badge-diseased"
                
                # Disease status header
                st.markdown(
                    f'<div class="diagnosis-badge {status_color}">Status: {info.get("status", "Unknown").upper()} | Severity: {info.get("severity", "N/A")}</div>',
                    unsafe_allow_html=True
                )
                
                clean_title = pred_class.replace("___", " - ").replace("_", " ")
                st.markdown(f"### **{clean_title}**")
                
                # Confidence Progress
                st.metric("Model Confidence", f"{confidence * 100:.1f}%")
                st.progress(float(confidence))

                st.markdown(f"**Crop Type:** {info.get('crop', 'N/A')}")
                st.markdown(f"**Causal Pathogen:** *{info.get('pathogen', 'N/A')}*")

            # Probability Breakdown
            st.markdown("---")
            st.subheader("📊 Class Probability Distribution")
            
            # Format dataframe for bar chart
            prob_items = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            chart_data = {
                "Class": [k.replace("___", " - ").replace("_", " ") for k, _ in prob_items],
                "Probability (%)": [v * 100 for _, v in prob_items]
            }
            st.bar_chart(data=chart_data, x="Class", y="Probability (%)", color="#2d6a4f")

            # Agronomic Treatment Guide
            st.markdown("---")
            st.subheader("🌾 Agronomic Advisory & Treatment Protocol")
            
            tab_symptoms, tab_organic, tab_chemical, tab_prevention = st.tabs([
                "🔍 Symptoms & Cause", "🍃 Organic Treatment", "🧪 Chemical Sprays", "🛡️ Cultural Prevention"
            ])
            
            with tab_symptoms:
                st.write(f"**Symptoms:** {info.get('symptoms', 'None reported.')}")
                st.write(f"**Favorable Environmental Conditions:** {info.get('causes', 'N/A')}")
            
            with tab_organic:
                for item in info.get("organic_treatment", []):
                    st.markdown(f"- {item}")
            
            with tab_chemical:
                for item in info.get("chemical_treatment", []):
                    st.markdown(f"- {item}")
            
            with tab_prevention:
                for item in info.get("prevention", []):
                    st.markdown(f"- {item}")

        else:
            st.info("👆 Please select or upload a leaf image above to begin real-time diagnosis.")

    # --- Mode 2: Model Performance & Metrics ---
    elif app_mode == "📊 Model Performance & Metrics":
        st.markdown('<div class="main-header">📊 CNN Performance Evaluation</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Evaluation benchmarks, confusion matrix, and training trajectories.</div>', unsafe_allow_html=True)

        report_json_path = "reports/evaluation_results.json"
        if os.path.exists(report_json_path):
            with open(report_json_path, "r") as f:
                eval_data = json.load(f)
            
            overall_acc = eval_data.get("overall_accuracy", 0.0)
            clf_rep = eval_data.get("classification_report", {})
            macro_f1 = clf_rep.get("macro avg", {}).get("f1-score", 0.0)
            weighted_prec = clf_rep.get("weighted avg", {}).get("precision", 0.0)
            weighted_rec = clf_rep.get("weighted avg", {}).get("recall", 0.0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{overall_acc * 100:.2f}%")
            c2.metric("Macro F1-Score", f"{macro_f1 * 100:.2f}%")
            c3.metric("Weighted Precision", f"{weighted_prec * 100:.2f}%")
            c4.metric("Weighted Recall", f"{weighted_rec * 100:.2f}%")
            
            st.markdown("---")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("Confusion Matrix Heatmap")
            cm_img_path = "reports/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrix over Validation Dataset", use_container_width=True)
            else:
                st.warning("Confusion matrix plot not found. Run `python -m src.evaluate` to generate.")

        with col_m2:
            st.subheader("Training & Validation Trajectory")
            curves_path = "reports/training_curves.png"
            if os.path.exists(curves_path):
                st.image(curves_path, caption="Accuracy and Loss Learning Curves", use_container_width=True)
            else:
                st.warning("Training curves plot not found.")

        # Per-Class Metrics Table
        if os.path.exists(report_json_path):
            st.markdown("---")
            st.subheader("📋 Per-Class Classification Report")
            
            rows = []
            for cls_key in class_names:
                if cls_key in clf_rep:
                    row_data = clf_rep[cls_key]
                    rows.append({
                        "Class Name": cls_key.replace("___", " - ").replace("_", " "),
                        "Precision": f"{row_data['precision']:.3f}",
                        "Recall": f"{row_data['recall']:.3f}",
                        "F1-Score": f"{row_data['f1-score']:.3f}",
                        "Support (Samples)": int(row_data['support'])
                    })
            st.dataframe(rows, use_container_width=True)

    # --- Mode 3: OpenCV Preprocessing Lab ---
    elif app_mode == "🔬 OpenCV Preprocessing Lab":
        st.markdown('<div class="main-header">🔬 OpenCV Leaf Computer Vision Lab</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Inspect how OpenCV transforms raw RGB photographs into edge maps and color segmentations.</div>', unsafe_allow_html=True)

        sample_dir = "sample_images"
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(sample_dir) else []
        
        selected_sample = st.selectbox("Select a leaf sample to inspect:", sample_files)
        if selected_sample:
            sample_path = os.path.join(sample_dir, selected_sample)
            raw_bgr = cv2.imread(sample_path)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            norm_img, resized_rgb = preprocess_image_cv2(raw_rgb, target_size=IMAGE_SIZE)
            cv_features = get_opencv_visualizations(resized_rgb)

            st.markdown("### 1. Color Space & Edge Decompositions")
            c1, c2, c3, c4 = st.columns(4)
            c1.image(resized_rgb, caption="1. RGB Normalized (128x128)", use_container_width=True)
            c2.image(cv_features["grayscale"], caption="2. OpenCV Grayscale", use_container_width=True, clamp=True)
            c3.image(cv_features["edges"], caption="3. Canny Edges (Lesion Margins)", use_container_width=True, clamp=True)
            c4.image(cv_features["lesion_highlight"], caption="4. HSV Necrosis Segment", use_container_width=True)

            st.markdown("---")
            st.markdown("### 2. OpenCV Operations Demonstrated")
            st.markdown("""
            - **`cv2.resize`**: Standardizes variable camera resolutions into a uniform 128x128 tensor input.
            - **`cv2.cvtColor`**: Transforms color spaces (BGR to RGB, RGB to Grayscale, and RGB to HSV).
            - **`cv2.Canny` & `cv2.GaussianBlur`**: Delineates leaf boundary geometry and perimeter margins of necrotic lesion spots.
            - **`cv2.inRange` & `cv2.bitwise_and`**: Masks yellow chlorotic halos and brown necrotic dry spots using HSV hue ranges.
            - **Pixel Normalization**: Scales pixel intensities from $[0, 255]$ integer values into $[0.0, 1.0]$ float32 values.
            """)

    # --- Mode 4: About & Architecture ---
    elif app_mode == "ℹ️ About & Architecture":
        st.markdown('<div class="main-header">ℹ️ System Architecture & Deep Learning Concepts</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🧠 Concepts Demonstrated in this Implementation:
        
        1. **Image Preprocessing**:
           - Normalized pixel values ($[0, 255] \to [0.0, 1.0]$) to maintain stable gradient magnitudes during backpropagation.
           - Image resolution alignment to $128 \\times 128 \\times 3$.
        
        2. **Convolution & Max Pooling**:
           - **Conv2D**: Uses $3 \\times 3$ sliding kernels with ReLU non-linearity to extract hierarchical representations (edges $\\to$ leaf veins $\\to$ necrotic lesions).
           - **MaxPooling2D**: Downsamples spatial resolution ($128 \\to 64 \\to 30 \\to 15$), enforcing translation invariance and reducing compute.
        
        3. **Data Augmentation**:
           - Geometric transforms (Random horizontal flips, vertical flips, $-25^\\circ$ to $+25^\\circ$ rotations).
           - Photometric transforms (dynamic brightness scaling) to build model robustness against field sunlight variations.
        
        4. **Dropout & Combating Overfitting**:
           - Dropout rates of $0.25$, $0.30$, and $0.50$ inserted between convolutional blocks and before the dense classification layer.
           - Ensures the network does not memorize specific training leaf backgrounds.
        
        5. **Multiclass Classification**:
           - Output layer with 6 units and **Softmax** activation producing true probability distributions summing to $1.0$.
           - Trained using **Sparse Categorical Crossentropy** loss and the **Adam** adaptive learning rate optimizer.
        """)


if __name__ == "__main__":
    main()
