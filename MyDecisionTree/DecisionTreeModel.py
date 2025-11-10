import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    r2_score, mean_squared_error, mean_absolute_error
)


class DecisionTreeModel:
    def __init__(self, X, y, features, target, encoder):
        self.X = X
        self.y = y
        self.features = features
        self.target = target
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

        self.model = DecisionTreeClassifier(**params) if self.task_type == "classification" else DecisionTreeRegressor(**params)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        self._evaluate(y_test)
        self.feature_importance = pd.DataFrame({
            "feature": self.features,
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=False)
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
            "class_names": self.class_names,
            "feature_importance": self.feature_importance
        }


'''
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    r2_score, mean_squared_error, mean_absolute_error
)


class DecisionTreeModel:
    def __init__(self, df: pd.DataFrame, independent_cols: list, dependent_col: str):
        """
        A unified Decision Tree model that auto-detects task type (Classification / Regression)
        and provides evaluation metrics and predictions.
        """
        self.df = df.dropna(subset=independent_cols + [dependent_col])
        self.dependent_col = dependent_col
        self.independent_cols = independent_cols
        self.model = None
        self.task_type = self._detect_task_type()
        self.metrics = {}
        self.y_pred = None

    def _detect_task_type(self) -> str:
        """Detect whether classification or regression based on target data."""
        y = self.df[self.dependent_col]
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
            return "Decision Tree Regression"
        return "Decision Tree Classification"

    def _prepare_data(self):
        """Prepare and encode data for training."""
        X = self.df[self.independent_cols].copy()
        y = self.df[self.dependent_col].copy()

        # Encode categorical features
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype("category").cat.codes

        # Encode categorical target
        if not pd.api.types.is_numeric_dtype(y):
            y = y.astype("category").cat.codes

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, **params):
        """Train the Decision Tree model (Classifier or Regressor)."""
        X_train, X_test, y_train, y_test = self._prepare_data()

        if self.task_type == "Decision Tree Classification":
            self.model = DecisionTreeClassifier(**params)
        else:
            self.model = DecisionTreeRegressor(**params)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        self._evaluate(y_test, y_pred)

        return self

    def _evaluate(self, y_test, y_pred):
        """Compute performance metrics."""
        if self.task_type == "Decision Tree Classification":
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )
            self.metrics = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            }
        else:
            self.metrics = {
                "r2": round(r2_score(y_test, y_pred), 4),
                "mse": round(mean_squared_error(y_test, y_pred), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                "mae": round(mean_absolute_error(y_test, y_pred), 4),
            }

    def get_results(self):
        """Return model, metrics, type, and predictions."""
        return {
            "model_type": self.task_type,
            "model": self.model,
            "metrics": self.metrics,
        }
'''
