import numpy as np
import pandas as pd
from modelling.base import BaseModel
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import logging

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Will use RandomForest instead.")
    from sklearn.ensemble import RandomForestClassifier


class XGBoostModel(BaseModel):
    """
    XGBoost model implementing the BaseModel interface.
    Falls back to RandomForest if XGBoost is not available.
    """

    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(XGBoostModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.label_encoder = LabelEncoder()

        if XGBOOST_AVAILABLE:
            self.mdl = xgb.XGBClassifier(
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
        else:
            # Fallback to RandomForest if XGBoost is not available
            from sklearn.ensemble import RandomForestClassifier
            self.mdl = RandomForestClassifier(
                n_estimators=200,
                random_state=0,
                class_weight='balanced_subsample'
            )

        self.predictions = None

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.get_X_train(), data.get_type_y_train())

    def predict(self, X_test) -> None:
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data) -> None:
        if self.predictions is None:
            print("No predictions available. Run predict() first.")
        else:
            print(classification_report(data.get_type_y_test(), self.predictions, zero_division=0))