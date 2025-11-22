import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class NaiveBayesModel:
    def __init__(self, X, y, encoder):
        self.X = X
        self.y = y
        self.encoder = encoder
        self.task_type = "classification"
        self.model = None
        self.class_names = encoder.class_names

    def train(self, **params):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = GaussianNB(**params)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        self._evaluate(y_test)
        return self

    def _evaluate(self, y_test):
        acc = accuracy_score(y_test, self.y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, self.y_pred, average="weighted", zero_division=0)
        self.metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}

    def get_results(self):
        return {
            "model": self.model,
            "model_type": self.task_type,
            "metrics": self.metrics,
            "prediction": self.y_pred,
            "class_names": self.class_names
        }
