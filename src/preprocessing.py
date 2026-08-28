import os
import re
import glob
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Normalize various folder/label names across Kaggle datasets to canonical labels
LABEL_MAPPING = {
    'male': 'male', 'man': 'male', 'men': 'male', 'm': 'male', '0': 'male',
    'female': 'female', 'woman': 'female', 'women': 'female', 'f': 'female', 'w': 'female', '1': 'female'
}

def is_valid_image(filepath, min_size=(20, 20), check_blur=False, blur_threshold=15.0):
    """
    Check if image file exists, is non-zero, valid, meets size thresholds, and optional blurriness check.
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
            if img.width < min_size[0] or img.height < min_size[1]:
                return False
                
        if check_blur:
            img_bgr = cv2.imread(filepath)
            if img_bgr is None:
                return False
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < blur_threshold:
                return False
                
        return True
    except Exception:
        return False

def extract_utkface_gender(filename):
    """
    Parse UTKFace / face-age-gender-dataset filename format: [age]_[gender]_[race]_[date].jpg
    Gender: 0 = male, 1 = female
    """
    basename = os.path.basename(filename)
    match = re.match(r"^(\d+)_([01])_", basename)
    if match:
        gender_code = match.group(2)
        return 'male' if gender_code == '0' else 'female'
    return None

def discover_dataset(dataset_dir, check_blur=False):
    """
    Dynamically discover dataset structure across 4 Kaggle dataset formats:
    1. Kaggle competition format (train/val subfolders or CSV annotations).
    2. UTKFace / face-age-gender-dataset filename encoded labels ([age]_[gender]_[race]).
    3. graphical27/gender-detection subfolders (Training/Validation/male/female/man/woman).
    4. mustafahabeeb90/gender-classification format (male/female).
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    image_paths = []
    labels = []
    
    # ----------------------------------------------------
    # Method 1: UTKFace Filename Pattern Match
    # ----------------------------------------------------
    print(f"[Dataset Inspector] Checking dataset path: '{dataset_dir}'")
    all_files = glob.glob(os.path.join(dataset_dir, "**", "*.*"), recursive=True)
    valid_files = [f for f in all_files if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
    
    utk_matched = 0
    for f in valid_files:
        gender_lbl = extract_utkface_gender(f)
        if gender_lbl and is_valid_image(f, check_blur=check_blur):
            image_paths.append(f)
            labels.append(gender_lbl)
            utk_matched += 1
            
    if utk_matched > 10:
        print(f"[Dataset Inspector] Recognized UTKFace / face-age-gender filename format. Loaded {utk_matched} images.")
    else:
        # Reset if UTKFace wasn't the main format
        image_paths = []
        labels = []
        
        # ----------------------------------------------------
        # Method 2: CSV Annotation Parsing
        # ----------------------------------------------------
        csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
        if csv_files:
            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                    file_col, label_col = None, None
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(k in col_lower for k in ['file', 'image', 'path', 'id', 'img']):
                            file_col = col
                        if any(k in col_lower for k in ['label', 'gender', 'class', 'target', 'category']):
                            label_col = col
                            
                    if file_col and label_col:
                        base_folder = os.path.dirname(csv_path)
                        for _, row in df.iterrows():
                            fname = str(row[file_col])
                            raw_lbl = str(row[label_col]).strip().lower()
                            canonical_lbl = LABEL_MAPPING.get(raw_lbl, raw_lbl)
                            
                            possible_paths = [
                                os.path.join(base_folder, fname),
                                os.path.join(dataset_dir, fname),
                                os.path.join(base_folder, "images", fname)
                            ]
                            for path in possible_paths:
                                if is_valid_image(path, check_blur=check_blur):
                                    image_paths.append(path)
                                    labels.append(canonical_lbl)
                                    break
                        if image_paths:
                            print(f"[Dataset Inspector] Loaded {len(image_paths)} images via CSV: {csv_path}")
                            break
                except Exception as e:
                    print(f"[Dataset Inspector] CSV inspection warning: {e}")

        # ----------------------------------------------------
        # Method 3: Dynamic Folder & Subdirectory Traversal
        # ----------------------------------------------------
        if not image_paths:
            print("[Dataset Inspector] Traversing subfolders for class categories...")
            for root, dirs, files in os.walk(dataset_dir):
                img_files = [f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
                if not img_files:
                    continue
                    
                folder_name = os.path.basename(root).lower().strip()
                canonical_lbl = LABEL_MAPPING.get(folder_name, None)
                
                if canonical_lbl is None:
                    # Check parent folder if current is nested e.g., 'Crop/male'
                    parent_name = os.path.basename(os.path.dirname(root)).lower().strip()
                    canonical_lbl = LABEL_MAPPING.get(parent_name, folder_name)
                    
                generic_containers = {'train', 'training', 'test', 'testing', 'val', 'validation', 'dataset', 'images', 'data', 'crop', 'cropped'}
                if canonical_lbl in generic_containers:
                    continue
                    
                for f in img_files:
                    full_path = os.path.join(root, f)
                    if is_valid_image(full_path, check_blur=check_blur):
                        image_paths.append(full_path)
                        labels.append(canonical_lbl)

    if not image_paths:
        raise ValueError(f"No valid image files found in '{dataset_dir}'. Supported formats: subfolders (male/female), UTKFace filenames, or CSV annotations.")

    unique_classes = sorted(list(set(labels)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    numeric_labels = [class_to_idx[lbl] for lbl in labels]

    print(f"[Dataset Inspector] Detected {len(unique_classes)} dynamic classes: {unique_classes}")
    print(f"[Dataset Inspector] Total verified valid images: {len(image_paths)}")
    
    return image_paths, numeric_labels, unique_classes, class_to_idx

def prepare_splits(image_paths, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into stratified Train, Validation, and Test sets.
    """
    total_val_test = test_size + val_size
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=total_val_test, random_state=random_state, stratify=labels
    )
    
    val_ratio = val_size / total_val_test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1.0 - val_ratio), random_state=random_state, stratify=temp_labels
    )
    
    print(f"[Splits] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def get_class_weights_dict(labels):
    """Compute balanced class weights to handle imbalanced datasets."""
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[Dataset Info] Computed Class Weights: {class_weight_dict}")
    return class_weight_dict

def load_and_preprocess_image(path_tensor, img_size=(224, 224)):
    """TF tensor image loader, decoder, and MobileNetV2 preprocessor."""
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = preprocess_input(img)
    return img

def augment_image(img):
    """Advanced image data augmentation for robust training."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    
    # Random slight hue adjustment
    img = tf.image.random_hue(img, max_delta=0.02)
    return img

def create_tf_dataset(image_paths, labels, num_classes, img_size=(224, 224), batch_size=32, is_training=False, cache=False):
    """
    High performance tf.data.Dataset pipeline with dynamic augmentation, caching, and prefetching.
    """
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    def process_path_and_label(path, label):
        img = load_and_preprocess_image(path, img_size=img_size)
        if is_training:
            img = augment_image(img)
        one_hot_label = tf.one_hot(label, depth=num_classes)
        return img, one_hot_label
    
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 5000))
        
    dataset = dataset.map(process_path_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    
    if cache:
        dataset = dataset.cache()
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
