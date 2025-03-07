import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import logging
import os

# Initialize optional Optuna if available
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Using default hyperparameters.")

logging.basicConfig(level=logging.INFO)


def optimize_hyperparameters(X, y, n_trials=50, model_type='rf'):
    """
    Find optimal hyperparameters using Optuna if available, otherwise return default params.

    Args:
        X: Features
        y: Labels
        n_trials: Number of optimization trials
        model_type: Model type ('rf', 'xgb', or 'lr')

    Returns:
        Best parameters dict
    """
    # Default parameters if Optuna is not available
    if model_type == 'rf':
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced_subsample',
            'random_state': 0
        }
    elif model_type == 'xgb':
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 0
        }
    else:  # logistic regression
        default_params = {
            'C': 1.0,
            'solver': 'liblinear',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 0
        }

    if not OPTUNA_AVAILABLE or n_trials <= 0:
        return default_params

    # Skip optimization if dataset is too small
    if len(y) < 50:
        logging.info(f"Dataset too small for hyperparameter optimization. Using default parameters.")
        return default_params

    # Define the objective function for Optuna
    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
                'random_state': 0
            }
            model = RandomForestClassifier(**params)

        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 0
            }
            model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')

        elif model_type == 'lr':
            params = {
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 0
            }
            model = LogisticRegression(**params)

        # Use 3-fold cross-validation
        try:
            score = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=-1)
            return score.mean()
        except Exception as e:
            logging.warning(f"Error during cross-validation: {str(e)}")
            return 0.0

    try:
        # Create a study object and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        logging.info(f"Best parameters for {model_type}: {study.best_params}")
        logging.info(f"Best F1 score: {study.best_value}")

        # Merge default params with optimized params
        best_params = default_params.copy()
        best_params.update(study.best_params)
        return best_params
    except Exception as e:
        logging.warning(f"Hyperparameter optimization failed: {str(e)}. Using default parameters.")
        return default_params