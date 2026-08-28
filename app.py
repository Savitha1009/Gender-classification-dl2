import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Facial Gender Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual design
st.markdown("""
    <style>
    /* Main Theme Overrides */
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header Card Styling */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
    }
    
    /* Status Badge */
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-warning {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Prediction Card */
    .pred-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 1rem;
    }
    
    /* Tech Stack Pills */
    .tech-pill {
        display: inline-block;
        background-color: #334155;
        color: #f8fafc;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Ethics Box */
    .ethics-box {
        background-color: #1e1b4b;
        border: 1px solid #4338ca;
        border-radius: 10px;
        padding: 1.2rem;
        color: #c7d2fe;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Add project root path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.predict import GenderPredictor
from src.face_detection import detect_and_crop_faces

@st.cache_resource
def get_predictor():
    """Cache loaded predictor to prevent reloading model on every rerun."""
    try:
        predictor = GenderPredictor()
        return predictor, None
    except Exception as e:
        return None, str(e)

def main():
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.sidebar.title("Gender Recognition")
    st.sidebar.caption("MobileNetV2 Deep Learning System")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🖼️ Image Prediction", "📷 Webcam Live", "📊 Model Performance"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Check model status
    models_dir = os.path.join(ROOT_DIR, 'models')
    model_file = os.path.join(models_dir, 'gender_classifier.keras')
    model_exists = os.path.exists(model_file)
    
    if model_exists:
        st.sidebar.markdown('<span class="badge-success">● Model Ready</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge-warning">▲ Model Not Trained</span>', unsafe_allow_html=True)
        st.sidebar.warning("Run `python src/train.py` to train the model.")
        
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Student Viva Tip:** MobileNetV2 uses depthwise separable convolutions to reduce parameter count while preserving feature extraction efficiency.")

    predictor, load_err = get_predictor()

    # ----------------------------------------------------
    # PAGE 1: HOME OVERVIEW
    # ----------------------------------------------------
    if page == "🏠 Home":
        st.markdown("""
            <div class="header-card">
                <h1>👤 Facial Gender Recognition</h1>
                <p style="font-size: 1.1rem; color: #94a3b8;">
                    An efficient computer vision system using <b>MobileNetV2 Transfer Learning</b>, 
                    <b>OpenCV Haar Cascade face detection</b>, and <b>TensorFlow/Keras</b>.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📌 System Architecture Workflow")
            st.code("""
Kaggle Dataset / Input Image
           ↓
Data Inspection & Preprocessing (Resize 224x224, Normalize, Augmentation)
           ↓
OpenCV Haar Cascade Face Detection & Cropping
           ↓
MobileNetV2 Transfer Learning Base (Pre-trained ImageNet)
           ↓
Global Average Pooling 2D Layer
           ↓
Dense Layer (128 units, ReLU activation)
           ↓
Dropout Regularization (0.5 rate)
           ↓
Output Layer (Softmax Probability Distribution)
           ↓
Predicted Class Label & Confidence Score
            """, language="text")
            
            st.subheader("🛠️ Technologies & Libraries Used")
            tech_stack = ["Python 3.13", "TensorFlow / Keras", "MobileNetV2", "OpenCV", "NumPy", "Pandas", "Scikit-Learn", "Matplotlib", "Streamlit"]
            pills_html = "".join([f'<span class="tech-pill">{t}</span>' for t in tech_stack])
            st.markdown(pills_html, unsafe_allow_html=True)
            
        with col2:
            st.subheader("📊 Dataset & Model Status")
            dataset_dir = os.path.join(ROOT_DIR, 'dataset')
            
            if os.path.exists(dataset_dir):
                classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                st.metric("Detected Dataset Classes", len(classes) if classes else "Subfolder auto-detect")
            else:
                st.metric("Dataset Status", "Not Found (Run train script)")
                
            if model_exists:
                st.success("✅ Model file (`gender_classifier.keras`) is loaded and ready.")
                if predictor and hasattr(predictor, 'class_names'):
                    st.write(f"**Detected Dynamic Classes:** `{predictor.class_names}`")
            else:
                st.error("❌ Model missing. Train model first using command `python src/train.py`.")
                
            st.markdown("""
            ### 🎯 Key Highlights
            - **Dynamic Class Detection**: Automatically reads labels without hardcoding.
            - **Robust Data Cleaning**: Filters missing/corrupted image files automatically.
            - **Multi-Face Detection**: Handles single and multiple faces in a single frame.
            - **Transfer Learning Efficiency**: Fast convergence with high accuracy using MobileNetV2.
            """)
            
        # Ethics & Limitations Notice (Mandatory Requirement)
        st.markdown("""
            <div class="ethics-box">
                <h4>⚠️ Ethical Disclaimer & Model Limitations</h4>
                <p>
                    <b>Important Notice:</b> This project performs <b>dataset-based visual image classification</b> 
                    on facial characteristics. It is not a method for determining an individual's actual gender identity.
                </p>
                <ul>
                    <li><b>Dataset Bias:</b> Model predictions reflect the statistical distribution and demographics of training datasets.</li>
                    <li><b>Environmental Factors:</b> Lighting conditions, facial angles, occlusion, makeup, and image resolution significantly impact prediction confidence.</li>
                    <li><b>Demographic Representation:</b> Performance may vary across different age groups, ethnicities, and facial features.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # PAGE 2: IMAGE PREDICTION
    # ----------------------------------------------------
    elif page == "🖼️ Image Prediction":
        st.title("🖼️ Facial Image Prediction")
        st.write("Upload an image containing one or more faces to perform automated detection and gender prediction.")
        
        if not model_exists or predictor is None:
            st.error(f"Cannot perform prediction. Model not found or failed to load. Error: {load_err}")
            st.info("Please train the model first by running `python src/train.py` in your terminal.")
            return

        # Preset Sample Image Option
        use_sample = st.checkbox("Use a built-in sample image", value=False)
        uploaded_file = None
        
        if use_sample:
            dataset_dir = os.path.join(ROOT_DIR, 'dataset')
            sample_files = []
            if os.path.exists(dataset_dir):
                for r, d, f in os.walk(dataset_dir):
                    for file in f:
                        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            sample_files.append(os.path.join(r, file))
                            
            if sample_files:
                selected_sample = st.selectbox("Select Sample Image:", sample_files, format_func=lambda x: os.path.basename(x))
                if selected_sample:
                    image_pil = Image.open(selected_sample).convert("RGB")
                    uploaded_file = selected_sample
            else:
                st.warning("No sample images found in `dataset/`. Please upload an image below.")
                
        if not use_sample:
            uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG, WEBP)...", type=["jpg", "png", "jpeg", "webp"])
            if uploaded_file is not None:
                image_pil = Image.open(uploaded_file).convert("RGB")
                
        if uploaded_file is not None:
            image_np = np.array(image_pil)
            
            with st.spinner("Processing face detection and MobileNetV2 classification..."):
                results, annotated_image, status_msg = predictor.process_and_predict(image_np)
                
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.subheader("Original Input Image")
                st.image(image_pil, use_container_width=True)
                
            with col_img2:
                st.subheader("Annotated Predictions")
                st.image(annotated_image, use_container_width=True)
                
            st.markdown("---")
            st.subheader("🔍 Prediction Results Breakdown")
            st.info(status_msg)
            
            # Display result cards for each face
            for idx, res in enumerate(results):
                st.markdown(f"#### Face #{idx+1}")
                res_col1, res_col2 = st.columns([1, 3])
                
                with res_col1:
                    st.image(res['cropped'], caption=f"Cropped Face #{idx+1}", width=160)
                    
                with res_col2:
                    label = res['label']
                    confidence = res['confidence']
                    probs = res['probabilities']
                    
                    st.markdown(f"**Predicted Class:** `<span class='badge-success'>{label.upper()}</span>`", unsafe_allow_html=True)
                    st.markdown(f"**Confidence Level:** `{confidence * 100:.2f}%`")
                    st.progress(float(confidence))
                    
                    st.markdown("**Class Probabilities:**")
                    for cls_name, prob in probs.items():
                        st.write(f"- `{cls_name.capitalize()}`: {prob * 100:.2f}%")
                        
                st.markdown("---")

    # ----------------------------------------------------
    # PAGE 3: WEBCAM LIVE
    # ----------------------------------------------------
    elif page == "📷 Webcam Live":
        st.title("📷 Real-Time Webcam Gender Recognition")
        st.write("Capture a frame from your webcam to run instant face detection and prediction.")
        
        if not model_exists or predictor is None:
            st.error("Model file not found. Train the model first.")
            return
            
        cam_file = st.camera_input("Take a snapshot from webcam")
        
        if cam_file is not None:
            image_pil = Image.open(cam_file).convert("RGB")
            image_np = np.array(image_pil)
            
            with st.spinner("Analyzing snapshot..."):
                results, annotated_image, status_msg = predictor.process_and_predict(image_np)
                
            st.image(annotated_image, caption="Webcam Prediction", use_container_width=True)
            st.success(status_msg)
            
            for idx, res in enumerate(results):
                st.write(f"**Face #{idx+1}:** `{res['label'].upper()}` ({res['confidence']*100:.1f}% confidence)")

    # ----------------------------------------------------
    # PAGE 4: PERFORMANCE & METRICS
    # ----------------------------------------------------
    elif page == "📊 Model Performance":
        st.title("📊 Model Performance & Metrics")
        st.write("Evaluation results, accuracy/loss graphs, and confusion matrix.")
        
        eval_path = os.path.join(models_dir, 'eval_metrics.json')
        cm_path = os.path.join(models_dir, 'confusion_matrix.png')
        history_plot_path = os.path.join(models_dir, 'training_history.png')
        
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                metrics = json.load(f)
                
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            m_col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
            m_col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
            m_col4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
            
            st.markdown("---")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("📈 Training History (Loss & Accuracy)")
                if os.path.exists(history_plot_path):
                    st.image(history_plot_path, use_container_width=True)
                else:
                    st.info("Training history plot not generated yet.")
                    
            with col_g2:
                st.subheader("🧩 Test Confusion Matrix")
                if os.path.exists(cm_path):
                    st.image(cm_path, use_container_width=True)
                else:
                    st.info("Confusion matrix plot not generated yet.")
                    
            st.markdown("---")
            st.subheader("📋 Classification Report Details")
            if 'classification_report' in metrics:
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)
        else:
            st.warning("No evaluation metrics found in `models/eval_metrics.json`.")
            st.info("Run `python src/train.py` to train the model and calculate evaluation metrics.")

if __name__ == '__main__':
    main()
