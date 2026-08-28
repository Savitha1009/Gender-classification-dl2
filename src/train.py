import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add root directory to sys.path to allow src imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import discover_dataset, prepare_splits, create_tf_dataset
from src.evaluate import evaluate_and_save_results

def build_mobilenet_model(num_classes, input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Build Transfer Learning model with MobileNetV2 base + Custom Classifier Head.
    
    Architecture:
      Input (224x224x3)
           ↓
      MobileNetV2 (Pre-trained ImageNet, Frozen)
           ↓
      GlobalAveragePooling2D
           ↓
      Dense(128, activation='relu')
           ↓
      Dropout(0.5)
           ↓
      Output Layer (Dense(num_classes), Softmax)
    """
    # Base pretrained model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='GenderClassifier_MobileNetV2')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def plot_history(history, save_path):
    """Plot and save loss and accuracy training metrics over epochs."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#2b5c8f', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2)
    plt.title('Model Accuracy vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#2b5c8f', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    plt.title('Model Loss vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Training] Saved loss & accuracy plots to {save_path}")

def run_training(dataset_dir, models_dir, epochs=15, batch_size=32, fine_tune=True):
    """
    Main training workflow for MobileNetV2 gender recognition model.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Step 1: Discover dataset & detect classes dynamically
    image_paths, numeric_labels, unique_classes, class_to_idx = discover_dataset(dataset_dir)
    num_classes = len(unique_classes)
    
    # Save dynamic class mapping to models directory
    class_names_path = os.path.join(models_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump({
            'class_names': unique_classes,
            'class_to_idx': class_to_idx,
            'num_classes': num_classes
        }, f, indent=4)
    print(f"[Dataset] Saved dynamic class mapping to {class_names_path}")
    
    # Step 2: Train / Validation / Test stratified split
    (train_paths, train_lbls), (val_paths, val_lbls), (test_paths, test_lbls) = prepare_splits(image_paths, numeric_labels)
    
    # Step 3: Create tf.data pipelines
    train_ds = create_tf_dataset(train_paths, train_lbls, num_classes, batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_lbls, num_classes, batch_size=batch_size, is_training=False)
    test_ds = create_tf_dataset(test_paths, test_lbls, num_classes, batch_size=batch_size, is_training=False)
    
    # Step 4: Build MobileNetV2 Model
    model, base_model = build_mobilenet_model(num_classes=num_classes)
    model.summary()
    
    model_save_path = os.path.join(models_dir, 'gender_classifier.keras')
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n--- PHASE 1: Training Transfer Learning Classifier Head ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Optional Fine-Tuning Phase
    if fine_tune:
        print("\n--- PHASE 2: Fine-Tuning Top Layers of Base MobileNetV2 ---")
        base_model.trainable = True
        # Freeze all layers before layer 100
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        ft_epochs = max(5, epochs // 2)
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=ft_epochs,
            callbacks=callbacks
        )
        
        # Combine history metrics
        for k in history.history:
            history.history[k].extend(history_ft.history[k])
            
    # Save training history metrics
    history_json_path = os.path.join(models_dir, 'history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history.history, f, indent=4)
        
    # Save training plots
    plot_save_path = os.path.join(models_dir, 'training_history.png')
    plot_history(history, plot_save_path)
    
    # Step 5: Evaluate on Test Set
    print("\n--- PHASE 3: Evaluating Model on Test Split ---")
    best_model = tf.keras.models.load_model(model_save_path)
    evaluate_and_save_results(best_model, test_ds, unique_classes, models_dir)
    
    print("\n[Training Complete] Best model saved to:", model_save_path)
    return best_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Facial Gender Recognition Model")
    parser.add_argument('--dataset', type=str, default=os.path.join(ROOT_DIR, 'dataset'), help="Path to dataset directory")
    parser.add_argument('--models', type=str, default=os.path.join(ROOT_DIR, 'models'), help="Path to save output models")
    parser.add_argument('--epochs', type=int, default=10, help="Number of initial training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--no_fine_tune', action='store_true', help="Disable fine-tuning phase")
    
    args = parser.parse_args()
    
    # Check if dataset exists, if not, generate sample dataset automatically
    if not os.path.exists(args.dataset) or len(os.listdir(args.dataset)) == 0:
        print("[Notice] Dataset directory is empty. Generating sample dataset automatically...")
        from generate_sample_data import main as generate_sample
        generate_sample()
        
    run_training(
        dataset_dir=args.dataset,
        models_dir=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune=not args.no_fine_tune
    )
