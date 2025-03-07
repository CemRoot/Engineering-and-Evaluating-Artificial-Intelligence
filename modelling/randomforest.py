import numpy as np
import pandas as pd
from modelling.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

class RandomForest(BaseModel):
    """
    RandomForest model implementing the BaseModel interface.
    """
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
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