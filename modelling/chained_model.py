from modelling.randomforest import RandomForest
from Config import Config
import numpy as np
import warnings
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning

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
      - Additional 33.33% if Stage 2 is correct (and Stage 1 was correct).
      - Additional 33.34% if Stage 3 is correct (with Stage 1 and 2 correct).
    The overall accuracy is the average of per-instance scores.
    """
    total_scores = []
    # Convert all values to strings for consistency.
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
                    score += 33.34  # Totals 100%
        total_scores.append(score)
    overall_accuracy = sum(total_scores) / len(total_scores)
    return overall_accuracy


def chained_model_predict(data, df, group_name):
    """
    Implements chained multi-output classification in three stages:
      - Stage 1: Predict Type2.
      - Stage 2: Predict concatenated Type2+Type3.
      - Stage 3: Predict concatenated Type2+Type3+Type4.
    Then applies rare-label filtering to reduce class sparsity in Stage 2 and Stage 3.
    Computes and prints overall chained accuracy for the group.
    """
    # Create target columns (casting to string)
    df['true_stage1'] = df[Config.TYPE_COLS[0]].astype(str)
    df['true_stage2'] = (df[Config.TYPE_COLS[0]].astype(str) + "_" + df[Config.TYPE_COLS[1]].astype(str))
    df['true_stage3'] = (df[Config.TYPE_COLS[0]].astype(str) + "_" + df[Config.TYPE_COLS[1]].astype(str) + "_" + df[
        Config.TYPE_COLS[2]].astype(str))

    # Apply rare-label filtering to Stage 2 and Stage 3 targets
    df['true_stage2'] = filter_rare_labels(df['true_stage2'], threshold=5)
    df['true_stage3'] = filter_rare_labels(df['true_stage3'], threshold=5)

    stages = [
        ("Stage 1: Type2", 'true_stage1'),
        ("Stage 2: Type2+Type3", 'true_stage2'),
        ("Stage 3: Type2+Type3+Type4", 'true_stage3')
    ]

    # Dictionaries to store true labels and predictions for each stage.
    stage_predictions = {}
    stage_true_labels = {}

    # Filter embeddings to include only the rows corresponding to the current group.
    filtered_embeddings = data.embeddings[df.index, :]

    for stage_name, target_col in stages:
        print("\n=== {} ===".format(stage_name))
        # Get true labels as strings.
        true_labels = df[target_col].astype(str).to_numpy()
        stage_true_labels[target_col] = true_labels

        # Initialize a RandomForest model using the filtered embeddings.
        model = RandomForest("RandomForest_" + stage_name, filtered_embeddings, true_labels)
        model.train(data)
        model.predict(filtered_embeddings)

        # Convert predictions to strings.
        preds = np.array(model.predictions).astype(str)
        stage_predictions[target_col] = preds

        # Print the classification report with zero_division parameter.
        print(classification_report(true_labels, preds, zero_division=0))

    # Compute overall chained accuracy using the stored true labels and predictions.
    overall_acc = compute_chained_accuracy(
        stage_true_labels['true_stage1'], stage_predictions['true_stage1'],
        stage_true_labels['true_stage2'], stage_predictions['true_stage2'],
        stage_true_labels['true_stage3'], stage_predictions['true_stage3']
    )

    print("\nOverall Chained Accuracy for group '{}': {:.2f}%".format(group_name, overall_acc))
