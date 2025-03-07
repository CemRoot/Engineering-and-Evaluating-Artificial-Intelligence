from modelling.randomforest import RandomForest
from Config import Config
import numpy as np
import warnings
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

    df['true_stage2'] = filter_rare_labels(df['true_stage2'], threshold=5)
    df['true_stage3'] = filter_rare_labels(df['true_stage3'], threshold=5)

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

        # Create and train model directly instead of using the data object methods
        if stage_name == "Stage 1: Type2":
            model = RandomForest("RandomForest_" + stage_name, X_train, train_labels)
            model.mdl.fit(X_train, train_labels)
        else:
            # Use LogisticRegression for stages 2 and 3
            model = RandomForest("RandomForest_" + stage_name, X_train, train_labels)
            model.mdl = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=0)
            model.mdl.fit(X_train, train_labels)

        # Get predictions directly using the trained model
        preds = model.mdl.predict(X_test)
        stage_predictions[target_col] = preds

        print(classification_report(test_labels, preds, zero_division=0))

    overall_acc = compute_chained_accuracy(
        stage_true_labels['true_stage1'], stage_predictions['true_stage1'],
        stage_true_labels['true_stage2'], stage_predictions['true_stage2'],
        stage_true_labels['true_stage3'], stage_predictions['true_stage3']
    )
    print("\nOverall Chained Accuracy for group '{}': {:.2f}%".format(group_name, overall_acc))