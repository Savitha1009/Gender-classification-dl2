import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def is_valid_image(filepath):
    """Check if file exists, is readable, and can be opened by PIL."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def discover_dataset(dataset_dir):
    """
    Dynamically discover dataset structure, extract image paths and dynamic class names.
    Supports directory-based structure (subfolders as classes) or CSV-annotated datasets.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    image_paths = []
    labels = []
    
    # Check for CSV annotation files (e.g., train.csv, labels.csv, dataset.csv)
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    if csv_files:
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                # Look for filename and label columns
                cols = [c.lower() for c in df.columns]
                file_col = None
                label_col = None
                
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
                        lbl = str(row[label_col]).strip()
                        
                        # Find matching file path
                        possible_paths = [
                            os.path.join(base_folder, fname),
                            os.path.join(dataset_dir, fname),
                            os.path.join(base_folder, "images", fname)
                        ]
                        for path in possible_paths:
                            if is_valid_image(path):
                                image_paths.append(path)
                                labels.append(lbl)
                                break
                    if image_paths:
                        print(f"[Dataset Inspector] Loaded {len(image_paths)} images from CSV: {csv_path}")
                        break
            except Exception as e:
                print(f"[Dataset Inspector] Warning reading CSV {csv_path}: {e}")

    # Fallback to dynamic directory-based discovery if CSV didn't yield images
    if not image_paths:
        print("[Dataset Inspector] Scanning directory tree for class folders...")
        for root, dirs, files in os.walk(dataset_dir):
            valid_files = [f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSIONS)]
            if not valid_files:
                continue
            
            # Infer class name from parent directory name
            folder_name = os.path.basename(root)
            parent_name = os.path.basename(os.path.dirname(root))
            
            # Ignore generic container folders like 'train', 'test', 'dataset', 'val', 'val_set'
            generic_names = {'train', 'test', 'val', 'validation', 'dataset', 'images', 'data', 'crop', 'cropped'}
            if folder_name.lower() in generic_names:
                continue
                
            class_name = folder_name
            
            for f in valid_files:
                full_path = os.path.join(root, f)
                if is_valid_image(full_path):
                    image_paths.append(full_path)
                    labels.append(class_name)

    if not image_paths:
        raise ValueError(f"No valid image files found in '{dataset_dir}'. Please verify dataset contents.")

    unique_classes = sorted(list(set(labels)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    numeric_labels = [class_to_idx[lbl] for lbl in labels]

    print(f"[Dataset Inspector] Detected {len(unique_classes)} classes dynamically: {unique_classes}")
    print(f"[Dataset Inspector] Total valid images found: {len(image_paths)}")
    
    return image_paths, numeric_labels, unique_classes, class_to_idx

def prepare_splits(image_paths, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into stratified Train, Validation, and Test sets.
    """
    total_val_test = test_size + val_size
    
    # Train vs (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=total_val_test, random_state=random_state, stratify=labels
    )
    
    # Val vs Test split ratio relative to temp set
    val_ratio = val_size / total_val_test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1.0 - val_ratio), random_state=random_state, stratify=temp_labels
    )
    
    print(f"[Splits] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def load_and_preprocess_image(path_tensor, img_size=(224, 224)):
    """TF tensor image loader, decoder, and MobileNetV2 preprocessor."""
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    # Apply MobileNetV2 scaling [-1, 1]
    img = preprocess_input(img)
    return img

def augment_image(img):
    """Data augmentation for training images."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img

def create_tf_dataset(image_paths, labels, num_classes, img_size=(224, 224), batch_size=32, is_training=False):
    """
    Create a tf.data.Dataset pipeline with dynamic augmentation and prefetching.
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
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        
    dataset = dataset.map(process_path_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
