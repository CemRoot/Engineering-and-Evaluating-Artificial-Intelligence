import numpy as np
from sklearn.utils import resample
import logging

logging.basicConfig(level=logging.INFO)


def handle_imbalance(X_train, y_train, strategy='simple'):
    """
    Handle class imbalance using simpler techniques that work with small datasets.

    Args:
        X_train: Training features
        y_train: Training labels
        strategy: Resampling strategy ('simple', 'none')

    Returns:
        Resampled features and labels
    """
    # Skip if too few samples or only one class
    if len(np.unique(y_train)) <= 1 or len(y_train) < 6:
        logging.warning("Insufficient samples or classes for resampling. Using original data.")
        return X_train, y_train

    if strategy == 'none':
        return X_train, y_train

    try:
        # Count labels
        unique_labels, counts = np.unique(y_train, return_counts=True)

        # Find majority class and its count
        majority_label = unique_labels[counts.argmax()]
        majority_count = counts.max()

        # Create new balanced dataset
        X_resampled = []
        y_resampled = []

        # For each class
        for label in unique_labels:
            # Get samples of this class
            indices = np.where(y_train == label)[0]
            X_class = X_train[indices]
            y_class = y_train[indices]

            # If minority class, resample with replacement to match majority
            if len(indices) < majority_count and len(indices) >= 3:
                resampled_idx = np.random.choice(indices, size=majority_count, replace=True)
                X_resampled.append(X_train[resampled_idx])
                y_resampled.append(y_train[resampled_idx])
            # If too small, just include original samples
            elif len(indices) < 3:
                X_resampled.append(X_class)
                y_resampled.append(y_class)
            # If majority class, include as is
            else:
                X_resampled.append(X_class)
                y_resampled.append(y_class)

        # Concatenate all resampled classes
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)

        # Log resampling results
        logging.info(f"Original class distribution: {dict(zip(unique_labels, counts))}")
        unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
        logging.info(f"Resampled class distribution: {dict(zip(unique_resampled, counts_resampled))}")
        logging.info(f"Resampled data shape: {X_resampled.shape}, Original: {X_train.shape}")

        return X_resampled, y_resampled
    except Exception as e:
        logging.warning(f"Resampling failed: {str(e)}. Using original data.")
        return X_train, y_train