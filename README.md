# 👤 Facial Gender Recognition Using Deep Learning

A simple, modular, and efficient Deep Learning project for **Facial Gender Recognition** built using **Python, TensorFlow/Keras, MobileNetV2, OpenCV, and Streamlit**.

---

## 📌 Project Architecture

```text
Kaggle Dataset / Input Image
      ↓
Data Inspection & Preprocessing (Dynamic Folder Discovery, Sanity Verification)
      ↓
OpenCV Haar Cascade Face Detection & Cropping
      ↓
Resize 224×224 + Normalize + Data Augmentation
      ↓
MobileNetV2 Base Model (Pre-trained ImageNet Transfer Learning)
      ↓
Global Average Pooling 2D
      ↓
Dense Layer (128 units, ReLU activation)
      ↓
Dropout Regularization (0.5 rate)
      ↓
Output Layer (Softmax Probability Distribution)
      ↓
Class Prediction & Confidence Score
```

---

## 🚀 Key Features

1. **Dynamic Dataset Discovery**: Automatically scans subdirectories or CSV files inside `dataset/`, infers class names dynamically, and handles missing or corrupted images.
2. **Transfer Learning Architecture**: Leverages pre-trained **MobileNetV2** for fast convergence and lightweight execution.
3. **OpenCV Face Detection**: Integrated `haarcascade_frontalface_default.xml` detector supporting single and multi-face recognition with bounding box visualization.
4. **Interactive Streamlit Web App**:
   - **Home**: Architecture breakdown, tech stack, and ethical disclaimer.
   - **Image Prediction**: File uploader with bounding boxes, cropped thumbnails, and confidence progress bars.
   - **Webcam Live**: Snapshot capture for instant live predictions.
   - **Model Performance**: Accuracy, precision, recall, F1-score, loss/accuracy training graphs, and confusion matrix.
5. **Ready Out-of-the-Box**: Includes an automatic sample data generator (`generate_sample_data.py`) for zero-setup presentation and testing.

---

## 🛠️ Project Structure

```text
facial-gender-recognition/
├── dataset/                    # Dataset directory (male/female subfolders or Kaggle data)
├── models/                     # Saved model artifacts & evaluation metrics
│   ├── gender_classifier.keras
│   ├── class_names.json
│   ├── eval_metrics.json
│   ├── history.json
│   ├── confusion_matrix.png
│   └── training_history.png
├── src/                        # Modular source code
│   ├── preprocessing.py        # Dataset loading, verification, and tf.data pipelines
│   ├── face_detection.py       # OpenCV Haar Cascade face detection & annotation
│   ├── train.py                # MobileNetV2 training & fine-tuning script
│   ├── predict.py              # Inference pipeline & face cropping wrapper
│   └── evaluate.py             # Evaluation metrics & confusion matrix generator
├── generate_sample_data.py     # Synthetic sample dataset generator for instant dry-runs
├── app.py                      # Interactive Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Option A: Download the [Kaggle Facial Gender Recognition Dataset](https://www.kaggle.com/competitions/facial-gender-recognition/data) and extract files into the `dataset/` directory.

Option B: Generate sample data automatically for testing:
```bash
python generate_sample_data.py
```

### 3. Train the Model
Run the complete training pipeline (includes validation, fine-tuning, metric saving, and evaluation):
```bash
python src/train.py
```

### 4. Launch the Web Application
Launch the interactive Streamlit interface:
```bash
streamlit run app.py
```

---

## 📊 Model Evaluation Metrics

Upon completing training, metrics are calculated on the test split:
- **Accuracy**: Overall classification accuracy.
- **Precision**: Weighted precision across classes.
- **Recall**: Weighted recall score.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix Heatmap**: Visual breakdown of true vs predicted classes saved at `models/confusion_matrix.png`.

---

## ⚠️ Ethical & Technical Disclaimer

> **Important Notice:** This project performs **dataset-based visual facial classification** on images. It is designed purely as a computer vision demonstration and is **not a method for determining an individual's gender identity**.
> 
> **Key Limitations to Explain in Viva:**
> - **Dataset Bias**: Predictions depend on the demographics of training datasets.
> - **Lighting & Pose**: Variations in shadow, angle, makeup, or facial hair can affect predictions.
> - **Image Resolution**: Low-resolution images or distance from camera impact OpenCV Haar Cascade detection accuracy.

---

## 🎓 Viva Presentation Tips for Students

1. **Why MobileNetV2?** 
   MobileNetV2 uses **depthwise separable convolutions** and inverted residuals, significantly reducing floating-point operations (FLOPs) and parameters while maintaining high accuracy.
2. **Why Global Average Pooling?**
   Global Average Pooling reduces spatial dimensions (7x7 -> 1x1) without adding trainable parameters, reducing overfitting compared to traditional Flatten layers.
3. **How does OpenCV Haar Cascade work?**
   It uses Haar-like features (contrast edge/line detectors) with an AdaBoost cascade of weak classifiers for fast real-time face detection before feeding cropped regions to the neural network.
