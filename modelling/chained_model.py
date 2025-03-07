from modelling.randomforest import RandomForest
from Config import Config
import numpy as np
import warnings
import os
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logging

# Import our utility modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.imbalance_handling import handle_imbalance
    from utils.error_analysis import detailed_error_analysis

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("Utils modules not found. Using basic implementation.")

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def filter_rare_labels(series, threshold=5):
    """
    Replace values in the series that occur fewer than 'threshold' times with "Other".
    """
    counts = Counter(series)
    return series.apply(lambda x: x if counts[x] >= threshold else "Other")


def compute_chained_accuracy(true_stage1, pred_stage1, true_stage2, pred_stage2, true_stage3, pred_stage3):
    """
    Compute overall chained accuracy per instance:
      - 33.33% credit if Stage 1 is correct.
      - Additional 33.33% if Stage 2 is correct (given Stage 1 is correct).
      - Additional 33.34% if Stage 3 is correct (given Stages 1 and 2 are correct).
    Returns:
        float: Average accuracy across all instances.
    """
    total_scores = []
    true_stage1 = [str(x) for x in true_stage1]
    pred_stage1 = [str(x) for x in pred_stage1]
    true_stage2 = [str(x) for x in true_stage2]
    pred_stage2 = [str(x) for x in pred_stage2]
    true_stage3 = [str(x) for x in true_stage3]
    pred_stage3 = [str(x) for x in pred_stage3]

    for i in range(len(true_stage1)):
        score = 0.0
        if pred_stage1[i] == true_stage1[i]:
            score += 33.33
            if pred_stage2[i] == true_stage2[i]:
                score += 33.33
                if pred_stage3[i] == true_stage3[i]:
                    score += 33.34
        total_scores.append(score)
    overall_accuracy = sum(total_scores) / len(total_scores)
    return overall_accuracy


def chained_model_predict(data, df, group_name):
    """
    Implements chained multi-output classification:
      - Stage 1: Predict Type2.
      - Stage 2: Predict concatenated target (Type2+Type3).
      - Stage 3: Predict concatenated target (Type2+Type3+Type4).

    Targets are built using a clear separator, with missing values replaced by "Unknown"
    and rare labels (fewer than 5 occurrences) merged to "Other".

    For Stage 2 and Stage 3, we use logistic regression as an alternative classifier.

    IMPORTANT: The group DataFrame index is reset so that indices align with the embedding matrix.

    Computes and prints the overall chained accuracy for the group.
    """
    df = df.reset_index(drop=True)

    for col in Config.TYPE_COLS:
        df[col] = df[col].fillna("Unknown").replace("nan", "Unknown")

    df['true_stage1'] = df[Config.TYPE_COLS[0]].astype(str)
    df['true_stage2'] = df[Config.TYPE_COLS[0]].astype(str) + "_" + df[Config.TYPE_COLS[1]].astype(str)
    df['true_stage3'] = df[Config.TYPE_COLS[0]].astype(str) + "_" + df[Config.TYPE_COLS[1]].astype(str) + "_" + df[
        Config.TYPE_COLS[2]].astype(str)

    # Apply rare label filtering with adjusted threshold
    rare_threshold = 4 if len(df) < 100 else 5
    df['true_stage2'] = filter_rare_labels(df['true_stage2'], threshold=rare_threshold)
    df['true_stage3'] = filter_rare_labels(df['true_stage3'], threshold=rare_threshold)

    # Split data into train and test sets
    X = data.embeddings
    test_size = 0.2
    train_indices, test_indices = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=0,
        stratify=df[Config.TYPE_COLS[0]]
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    stages = [
        ("Stage 1: Type2", 'true_stage1'),
        ("Stage 2: Type2+Type3", 'true_stage2'),
        ("Stage 3: Type2+Type3+Type4", 'true_stage3')
    ]

    stage_predictions = {}
    stage_true_labels = {}

    for stage_name, target_col in stages:
        print("\n=== {} ===".format(stage_name))
        train_labels = train_df[target_col].astype(str).to_numpy()
        test_labels = test_df[target_col].astype(str).to_numpy()
        stage_true_labels[target_col] = test_labels

        # Handle class imbalance if utils are available
        if UTILS_AVAILABLE:
            if stage_name == "Stage 1: Type2":
                resampling_strategy = 'simple'
            else:
                resampling_strategy = 'none'  # Skip resampling for later stages

            X_train_balanced, train_labels_balanced = handle_imbalance(
                X_train, train_labels, strategy=resampling_strategy
            )
        else:
            X_train_balanced, train_labels_balanced = X_train, train_labels

        # Choose model based on stage
        if stage_name == "Stage 1: Type2":
            # Use RandomForest for Stage 1 - more reliable with small dataset
            model = RandomForest("RandomForest_" + stage_name, X_train_balanced, train_labels_balanced)
            # Adjust parameters for smaller datasets
            if len(X_train_balanced) < 100:
                model.mdl.n_estimators = 100  # Reduce complexity for small datasets

            model.mdl.fit(X_train_balanced, train_labels_balanced)
            preds = model.mdl.predict(X_test)
        else:
            # Use LogisticRegression for Stages 2 and 3
            model = RandomForest("RandomForest_" + stage_name, X_train_balanced, train_labels_balanced)
            if len(np.unique(train_labels_balanced)) > 10:
                # More solver iterations for complex problems with many classes
                model.mdl = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=0, solver='saga')
            else:
                model.mdl = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=0)

            model.mdl.fit(X_train_balanced, train_labels_balanced)
            preds = model.mdl.predict(X_test)

        # Store predictions
        stage_predictions[target_col] = preds

        # Print classification report
        print(classification_report(test_labels, preds, zero_division=0))

        # Perform detailed error analysis if available
        if UTILS_AVAILABLE:
            try:
                detailed_error_analysis(test_labels, preds,
                                        class_names=np.unique(np.concatenate((test_labels, preds))),
                                        stage_name=stage_name)
            except Exception as e:
                logging.warning(f"Error analysis failed: {str(e)}")

    # Calculate and print overall accuracy
    overall_acc = compute_chained_accuracy(
        stage_true_labels['true_stage1'], stage_predictions['true_stage1'],
        stage_true_labels['true_stage2'], stage_predictions['true_stage2'],
        stage_true_labels['true_stage3'], stage_predictions['true_stage3']
    )
    print("\nOverall Chained Accuracy for group '{}': {:.2f}%".format(group_name, overall_acc))