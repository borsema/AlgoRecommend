import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    r2_score, mean_squared_error, mean_absolute_error
)


class KnnModel:
    def __init__(self, X, y, encoder):
        self.X = X
        self.y = y
        self.encoder = encoder
        self.task_type = self._detect_task_type()
        self.model = None
        self.class_names = encoder.class_names

    def _detect_task_type(self):
        if pd.api.types.is_numeric_dtype(self.y) and len(np.unique(self.y)) > 20:
            return "regression"
        return "classification"

    def train(self, **params):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = KNeighborsClassifier(**params) if self.task_type == "classification" else KNeighborsRegressor(**params)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        self._evaluate(y_test)
        return self

    def _evaluate(self, y_test):
        if self.task_type == "classification":
            acc = accuracy_score(y_test, self.y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, self.y_pred, average="weighted", zero_division=0)
            self.metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
        else:
            mse = mean_squared_error(y_test, self.y_pred)
            self.metrics = {"r2": r2_score(y_test, self.y_pred), "mse": mse, "rmse": np.sqrt(mse),
                            "mae": mean_absolute_error(y_test, self.y_pred)}

    def get_results(self):
        return {
            "model": self.model,
            "model_type": self.task_type,
            "metrics": self.metrics,
            "prediction": self.y_pred,
            "class_names": self.class_names
        }