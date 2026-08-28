import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

def evaluate_and_save_results(model, test_ds, class_names, output_dir):
    """
    Evaluate model performance on test dataset and save metrics/confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = []
    y_pred_probs = []
    
    # Collect all ground truth and predictions batch by batch
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    print("\n" + "="*50)
    print("           MODEL EVALUATION SUMMARY           ")
    print("="*50)
    print(f" Test Accuracy  : {acc * 100:.2f}%")
    print(f" Precision      : {precision * 100:.2f}%")
    print(f" Recall         : {recall * 100:.2f}%")
    print(f" F1-Score       : {f1 * 100:.2f}%")
    print("-" * 50)
    print("Classification Report:\n", report_str)
    print("="*50)
    
    # Save Metrics JSON
    eval_metrics = {
        'accuracy': acc,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'class_names': class_names,
        'classification_report': report_dict
    }
    
    metrics_path = os.path.join(output_dir, 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    # Plot & Save Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1
    )
    plt.title('Test Set Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix heatmap to {cm_path}")
    
    return eval_metrics

if __name__ == '__main__':
    import sys
    import tensorflow as tf
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
        
    from src.preprocessing import discover_dataset, prepare_splits, create_tf_dataset
    
    dataset_dir = os.path.join(ROOT_DIR, 'dataset')
    models_dir = os.path.join(ROOT_DIR, 'models')
    model_path = os.path.join(models_dir, 'gender_classifier.keras')
    class_map_path = os.path.join(models_dir, 'class_names.json')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run training first.")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    image_paths, numeric_labels, unique_classes, _ = discover_dataset(dataset_dir)
    _, _, (test_paths, test_lbls) = prepare_splits(image_paths, numeric_labels)
    test_ds = create_tf_dataset(test_paths, test_lbls, len(unique_classes), is_training=False)
    
    evaluate_and_save_results(model, test_ds, unique_classes, models_dir)

