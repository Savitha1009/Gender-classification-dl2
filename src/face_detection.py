import os
import cv2
import numpy as np

def get_cascade_classifier():
    """
    Load OpenCV Haar Cascade Frontal Face Classifier with multiple search locations and auto-download.
    """
    cascade_filename = 'haarcascade_frontalface_default.xml'
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', cascade_filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), cascade_filename),
        cascade_filename,
        os.path.join(getattr(cv2, 'data', None).haarcascades if hasattr(cv2, 'data') else '', cascade_filename)
    ]
    
    cascade_path = None
    for path in possible_paths:
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            cascade_path = path
            break
            
    if cascade_path is None:
        # Download XML if missing
        target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', cascade_filename)
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        print(f"[Face Detection] Haar Cascade XML not found. Downloading from {url}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, target_path)
            cascade_path = target_path
        except Exception as e:
            print(f"[Face Detection] Warning downloading Haar Cascade XML: {e}")
            
    if cascade_path and os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if not face_cascade.empty():
            return face_cascade
            
    print("[Face Detection] Warning: OpenCV Cascade Classifier empty or missing.")
    return None


def detect_and_crop_faces(image_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), padding=0.1):
    """
    Detect faces in an RGB image using OpenCV Haar Cascade.
    
    Returns:
        faces_data: List of dicts with key 'box' (x, y, w, h) and 'cropped' (RGB cropped face image)
        status_message: Explanation string
    """
    face_cascade = get_cascade_classifier()
    if face_cascade is None or face_cascade.empty():
        return [], "Face detector unavailable."
        
    # Convert RGB to Grayscale for OpenCV Haar Cascade

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Equalize histogram for better detection under different lighting
    gray_eq = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        # Retry on original gray image if equalized missed
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=minSize
        )
        
    if len(faces) == 0:
        return [], "No faces detected in the image."
        
    h_img, w_img = image_rgb.shape[:2]
    faces_data = []
    
    for (x, y, w, h) in faces:
        # Add padding margin
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)
        
        cropped_face = image_rgb[y1:y2, x1:x2]
        
        faces_data.append({
            'box': (x, y, w, h),
            'padded_box': (x1, y1, x2 - x1, y2 - y1),
            'cropped': cropped_face
        })
        
    return faces_data, f"Detected {len(faces_data)} face(s)."

def draw_predictions_on_image(image_rgb, predictions_list):
    """
    Draw bounding boxes and class label banners over detected faces.
    
    predictions_list: list of dicts with 'box', 'label', 'confidence'
    """
    annotated = image_rgb.copy()
    
    for item in predictions_list:
        x, y, w, h = item['box']
        label = item.get('label', 'Unknown')
        conf = item.get('confidence', 0.0)
        
        # Color coding: Cyan for female, Lime for male, or default bright green
        if label.lower() in ['female', 'woman']:
            color = (235, 64, 180)  # Bright Pink / Magenta RGB
        elif label.lower() in ['male', 'man']:
            color = (30, 160, 240)  # Bright Blue RGB
        else:
            color = (50, 205, 50)   # Lime Green
            
        # Draw face bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Prepare text label string
        text_str = f"{label} ({conf * 100:.1f}%)"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, w / 200.0)
        thickness = max(1, int(font_scale * 2))
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text_str, font, font_scale, thickness)
        
        # Text box coordinates
        text_y1 = max(0, y - text_height - 10)
        text_y2 = y
        text_x1 = x
        text_x2 = x + text_width + 10
        
        # Fill banner background
        cv2.rectangle(annotated, (text_x1, text_y1), (text_x2, text_y2), color, -1)
        
        # Write text in white
        cv2.putText(annotated, text_str, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return annotated
