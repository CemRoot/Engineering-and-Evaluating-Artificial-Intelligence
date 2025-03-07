import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)


def safe_filename(stage_name):
    """
    Replace characters that are problematic in filenames
    """
    return stage_name.replace('/', '_').replace('\\', '_').replace(':', '_')


def detailed_error_analysis(y_true, y_pred, class_names=None, stage_name=""):
    """
    Perform detailed error analysis on predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (if None, will be extracted from unique values)
        stage_name: Name of the stage being analyzed (for output labeling)
    """
    if class_names is None:
        class_names = np.unique(np.concatenate((y_true, y_pred)))

    # If there are too many classes, skip visualization
    if len(class_names) > 20:
        logging.info(f"Too many classes ({len(class_names)}) for detailed visualization")
        return

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Ensure output directory exists
    os.makedirs("error_analysis", exist_ok=True)

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f',
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix - {stage_name}')
    plt.tight_layout()
    file_name = f'error_analysis/confusion_matrix_{safe_filename(stage_name)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(file_name)
    plt.close()
    logging.info(f"Saved confusion matrix to {file_name}")

    # Identify most confused classes
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((class_names[i], class_names[j], cm[i, j],
                               cm_norm[i, j] if i < len(cm_norm) and j < len(cm_norm[i]) else 0))

    errors.sort(key=lambda x: x[2], reverse=True)

    logging.info(f"Top confused classes for {stage_name}:")
    for true_class, pred_class, count, normalized in errors[:min(10, len(errors))]:
        logging.info(f"True: {true_class}, Predicted: {pred_class}, Count: {count}, Rate: {normalized:.2f}")

    # Calculate per-class metrics
    correct = np.diag(cm)
    total_per_class = np.sum(cm, axis=1)
    accuracy_per_class = np.divide(correct, total_per_class,
                                   out=np.zeros_like(correct, dtype=float),
                                   where=total_per_class != 0)

    logging.info(f"\nPer-class accuracy for {stage_name}:")
    for i, class_name in enumerate(class_names):
        if i < len(accuracy_per_class):
            logging.info(f"{class_name}: {accuracy_per_class[i]:.2f} ({correct[i]}/{total_per_class[i]})")