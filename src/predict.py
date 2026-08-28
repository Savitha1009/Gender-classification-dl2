import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.face_detection import detect_and_crop_faces, draw_predictions_on_image

class GenderPredictor:
    def __init__(self, model_path=None, class_names_path=None):
        if model_path is None:
            model_path = os.path.join(ROOT_DIR, 'models', 'gender_classifier.keras')
        if class_names_path is None:
            class_names_path = os.path.join(ROOT_DIR, 'models', 'class_names.json')
            
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = ['female', 'male']  # Default fallback
        self._load_model_and_classes()
        
    def _load_model_and_classes(self):
        """Load Keras model and dynamic class names mapping."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'. Please run training script first.")
            
        print(f"[Predictor] Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path, 'r') as f:
                data = json.load(f)
                self.class_names = data.get('class_names', self.class_names)
            print(f"[Predictor] Loaded dynamic class map: {self.class_names}")
            
    def predict_image(self, image_rgb):
        """
        Preprocess a single image (RGB numpy array) and get classification probabilities.
        """
        # Resize to 224x224
        resized = cv2.resize(image_rgb, (224, 224))
        
        # Expand dims & preprocess for MobileNetV2
        input_tensor = np.expand_dims(resized, axis=0).astype(np.float32)
        input_tensor = preprocess_input(input_tensor)
        
        # Inference
        probs = self.model.predict(input_tensor, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_label = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        return pred_label, confidence, prob_dict

    def process_and_predict(self, image_rgb):
        """
        Full workflow: detect faces -> crop -> predict each face -> annotate image.
        If no faces detected, fall back to whole image classification.
        """
        faces_data, status_msg = detect_and_crop_faces(image_rgb)
        
        if not faces_data:
            # Fallback to whole image classification
            pred_label, confidence, prob_dict = self.predict_image(image_rgb)
            fallback_res = [{
                'box': (0, 0, image_rgb.shape[1], image_rgb.shape[0]),
                'label': pred_label,
                'confidence': confidence,
                'probabilities': prob_dict,
                'cropped': image_rgb,
                'is_fallback': True
            }]
            return fallback_res, image_rgb, "No face detected by OpenCV Haar cascade. Applied fallback whole-image classification."
            
        results = []
        for face_info in faces_data:
            cropped_face = face_info['cropped']
            pred_label, confidence, prob_dict = self.predict_image(cropped_face)
            
            results.append({
                'box': face_info['box'],
                'label': pred_label,
                'confidence': confidence,
                'probabilities': prob_dict,
                'cropped': cropped_face,
                'is_fallback': False
            })
            
        annotated_image = draw_predictions_on_image(image_rgb, results)
        return results, annotated_image, f"Successfully detected and classified {len(results)} face(s)."
