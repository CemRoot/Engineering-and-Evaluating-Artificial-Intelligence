from modelling.randomforest import RandomForest
from Config import Config
import numpy as np
import warnings
import os
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Import our new utility modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.imbalance_handling import handle_imbalance
    from utils.error_analysis import detailed_error_analysis
    from utils.hyperparameter_tuning import optimize_hyperparameters

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("Utils modules not found. Using basic implementation.")

# Try importing XGBoost model
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def filter_rare_labels(series, threshold=5):
    """
    Replace values in the series that occur fewer than 'threshold' times with "Other".
    """
    counts = Counter(series)
    return series.apply(lambda x: x if counts[x] >= threshold else "Other")


def compute_hierarchical_accuracy(type2_correct, type3_correct, type4_correct):
    """
    Compute overall hierarchical accuracy:
      - 33.33% credit if Type 2 is correct.
      - Additional 33.33% if Type 3 is correct (given Type 2 is correct).
      - Additional 33.34% if Type 4 is correct (given Types 2 and 3 are correct).
    Returns:
        float: Average accuracy score.
    """
    total_score = 0.0

    if type2_correct:
        total_score += 33.33
        if type3_correct:
            total_score += 33.33
            if type4_correct:
                total_score += 33.34

    return total_score


def hierarchical_model_predict(data, df, group_name):
    """
    Implements hierarchical multi-output classification:
      - Stage 1: Train a model to predict Type 2.
      - Stage 2: For each unique value in Type 2, filter data and train a separate model for Type 3.
      - Stage 3: For each Type 2 + Type 3 combination, filter data and train a model for Type 4.

    IMPORTANT: The group DataFrame index is reset so that indices align with the embedding matrix.
    """
    df = df.reset_index(drop=True)

    # Fill missing values
    for col in Config.TYPE_COLS:
        df[col] = df[col].fillna("Unknown").replace("nan", "Unknown")

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

    # Stage 1: Train a model for Type 2
    print("\n=== Stage 1: Type 2 ===")

    # Handle class imbalance for Type 2 if utils are available
    if UTILS_AVAILABLE:
        type2_X_train, type2_y_train = handle_imbalance(
            X_train, train_df[Config.TYPE_COLS[0]].values, strategy='smote'
        )
    else:
        type2_X_train, type2_y_train = X_train, train_df[Config.TYPE_COLS[0]].values

    # Choose model for Type 2
    if XGBOOST_AVAILABLE:
        # Use XGBoost with label encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(type2_y_train)

        # Create and train XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            random_state=0,
            eval_metric='mlogloss'
        )
        xgb_model.fit(type2_X_train, y_encoded)

        # Make predictions and decode them back to original labels
        encoded_preds = xgb_model.predict(X_test)
        type2_preds = label_encoder.inverse_transform(encoded_preds)
    else:
        # Use RandomForest otherwise
        type2_model = RandomForest("Type2_Model", type2_X_train, type2_y_train)
        type2_model.mdl.fit(type2_X_train, type2_y_train)
        type2_preds = type2_model.mdl.predict(X_test)

    type2_true = test_df[Config.TYPE_COLS[0]].values

    print(classification_report(type2_true, type2_preds, zero_division=0))

    # Perform detailed error analysis for Type 2 if available
    if UTILS_AVAILABLE:
        try:
            detailed_error_analysis(type2_true, type2_preds,
                                    stage_name="Stage 1: Type 2")
        except Exception as e:
            logging.warning(f"Error analysis failed: {str(e)}")

    # Get unique values for Type 2 in the training set
    type2_values = train_df[Config.TYPE_COLS[0]].unique()
    print(f"\nUnique values for Type 2: {type2_values}")

    # Stage 2: Train models for Type 3 for each Type 2 value
    type3_models = {}
    for type2_val in type2_values:
        print(f"\n=== Stage 2: Type 3 for Type 2 = {type2_val} ===")
        # Filter training data for this Type 2 value
        type2_filter = train_df[Config.TYPE_COLS[0]] == type2_val
        if sum(type2_filter) < 5:  # Skip if too few samples
            print(f"Skipping {type2_val} due to insufficient samples")
            continue

        # Get filtered data
        filtered_X_train = X_train[type2_filter]
        filtered_y_train = train_df[type2_filter][Config.TYPE_COLS[1]].values

        # Skip if only one class
        if len(np.unique(filtered_y_train)) < 2:
            print(f"Skipping {type2_val} due to insufficient class diversity")
            continue

        # Handle class imbalance for Type 3 if utils are available
        if UTILS_AVAILABLE:
            filtered_X_train, filtered_y_train = handle_imbalance(
                filtered_X_train, filtered_y_train, strategy='smote'
            )

        # Train a model for Type 3 on the filtered data
        type3_model = RandomForest(f"Type3_Model_{type2_val}", filtered_X_train, filtered_y_train)
        # Train directly with filtered data
        type3_model.mdl.fit(filtered_X_train, filtered_y_train)
        type3_models[type2_val] = type3_model

        # Test the model on relevant test samples
        test_type2_filter = test_df[Config.TYPE_COLS[0]] == type2_val
        if sum(test_type2_filter) > 0:
            filtered_test_X = X_test[test_type2_filter]
            filtered_test_y = test_df[test_type2_filter][Config.TYPE_COLS[1]].values
            if len(filtered_test_X) > 0:
                preds = type3_model.mdl.predict(filtered_test_X)
                print(classification_report(filtered_test_y, preds, zero_division=0))

                # Perform detailed error analysis for this Type 3 model if available
                if UTILS_AVAILABLE and len(np.unique(filtered_test_y)) > 1:
                    try:
                        detailed_error_analysis(filtered_test_y, preds,
                                                stage_name=f"Stage 2: Type3 for {type2_val}")
                    except Exception as e:
                        logging.warning(f"Error analysis failed: {str(e)}")

    # Continue with Stage 3 as before...
    # Stage 3: Train models for Type 4 for each Type 2 + Type 3 combination
    type4_models = {}
    for type2_val in type2_values:
        if type2_val not in type3_models:
            continue

        # Get Type 3 values for this Type 2 value
        type2_filter = train_df[Config.TYPE_COLS[0]] == type2_val
        type3_values = train_df[type2_filter][Config.TYPE_COLS[1]].unique()

        for type3_val in type3_values:
            print(f"\n=== Stage 3: Type 4 for Type 2 = {type2_val}, Type 3 = {type3_val} ===")

            # Filter training data for this Type 2 + Type 3 combination
            type23_filter = (train_df[Config.TYPE_COLS[0]] == type2_val) & \
                            (train_df[Config.TYPE_COLS[1]] == type3_val)

            if sum(type23_filter) < 5:  # Skip if too few samples
                print(f"Skipping {type2_val}_{type3_val} due to insufficient samples")
                continue

            # Get filtered data
            filtered_X_train = X_train[type23_filter]
            filtered_y_train = train_df[type23_filter][Config.TYPE_COLS[2]].values

            # Skip if only one class
            if len(np.unique(filtered_y_train)) < 2:
                print(f"Skipping {type2_val}_{type3_val} due to insufficient class diversity")
                continue

            # Handle class imbalance for Type 4 if utils are available
            if UTILS_AVAILABLE:
                filtered_X_train, filtered_y_train = handle_imbalance(
                    filtered_X_train, filtered_y_train, strategy='smote'
                )

            # Train a model for Type 4 on the filtered data
            type4_model = RandomForest(f"Type4_Model_{type2_val}_{type3_val}",
                                       filtered_X_train, filtered_y_train)
            # Train directly with filtered data
            type4_model.mdl.fit(filtered_X_train, filtered_y_train)
            type4_models[f"{type2_val}_{type3_val}"] = type4_model

            # Test the model on relevant test samples
            test_type23_filter = (test_df[Config.TYPE_COLS[0]] == type2_val) & \
                                 (test_df[Config.TYPE_COLS[1]] == type3_val)
            if sum(test_type23_filter) > 0:
                filtered_test_X = X_test[test_type23_filter]
                filtered_test_y = test_df[test_type23_filter][Config.TYPE_COLS[2]].values
                if len(filtered_test_X) > 0:
                    preds = type4_model.mdl.predict(filtered_test_X)
                    print(classification_report(filtered_test_y, preds, zero_division=0))

                    # Perform detailed error analysis for this Type 4 model if available
                    if UTILS_AVAILABLE and len(np.unique(filtered_test_y)) > 1:
                        try:
                            detailed_error_analysis(filtered_test_y, preds,
                                                    stage_name=f"Stage 3: Type4 for {type2_val}_{type3_val}")
                        except Exception as e:
                            logging.warning(f"Error analysis failed: {str(e)}")

    # Evaluate hierarchical accuracy on test data
    total_scores = []

    for i, (idx, row) in enumerate(test_df.iterrows()):
        true_type2 = row[Config.TYPE_COLS[0]]
        true_type3 = row[Config.TYPE_COLS[1]]
        true_type4 = row[Config.TYPE_COLS[2]]

        pred_type2 = type2_preds[i]

        # Check if Type 2 is predicted correctly
        type2_correct = (pred_type2 == true_type2)
        type3_correct = False
        type4_correct = False

        if type2_correct and pred_type2 in type3_models:
            # Predict Type 3 using the model for the predicted Type 2
            type3_model = type3_models[pred_type2]
            pred_type3 = type3_model.mdl.predict(X_test[i:i + 1])[0]

            # Check if Type 3 is predicted correctly
            type3_correct = (pred_type3 == true_type3)

            if type3_correct and f"{pred_type2}_{pred_type3}" in type4_models:
                # Predict Type 4 using the model for the predicted Type 2 + Type 3
                type4_model = type4_models[f"{pred_type2}_{pred_type3}"]
                pred_type4 = type4_model.mdl.predict(X_test[i:i + 1])[0]

                # Check if Type 4 is predicted correctly
                type4_correct = (pred_type4 == true_type4)

        # Compute hierarchical accuracy for this instance
        score = compute_hierarchical_accuracy(type2_correct, type3_correct, type4_correct)
        total_scores.append(score)

    # Calculate overall hierarchical accuracy
    overall_accuracy = sum(total_scores) / len(total_scores)
    print(f"\nOverall Hierarchical Accuracy for group '{group_name}': {overall_accuracy:.2f}%")