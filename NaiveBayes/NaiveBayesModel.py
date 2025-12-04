import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class NaiveBayesModel:
    def __init__(self, X, y, encoder):
        self.X = X
        self.y = y
        self.encoder = encoder
        self.class_names = encoder.class_names
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.gaussian_model = GaussianNB()
        self.multinomial_model = None
        self.bernoulli_model = BernoulliNB()
        self.categorical_model = None
        self.complement_model = None
        self.gaussian_pred = None
        self.multinomial_pred = None
        self.bernoulli_pred = None
        self.categorical_pred = None
        self.complement_pred = None

    def _transform_for_multinomial(self, X):
        """Transform data to non-negative for MultinomialNB"""
        X_transformed = X.copy()
        if X_transformed.min().min() < 0:
            X_transformed = X_transformed - X_transformed.min().min()
        return X_transformed
    
    def _transform_for_categorical(self, X):
        """Transform data to non-negative integers for CategoricalNB"""
        from sklearn.preprocessing import MinMaxScaler
        X_transformed = X.copy()
        scaler = MinMaxScaler(feature_range=(0, 10))
        X_transformed = pd.DataFrame(scaler.fit_transform(X_transformed), columns=X.columns, index=X.index)
        X_transformed = X_transformed.round().astype(int)
        return X_transformed

    def train_models(self, gaussian_var_smoothing=1e-9, multinomial_alpha=1.0, bernoulli_alpha=1.0, bernoulli_binarize=0.0, categorical_alpha=1.0, complement_alpha=1.0):
        # Gaussian NB - works with any data
        self.gaussian_model = GaussianNB(var_smoothing=gaussian_var_smoothing)
        self.gaussian_model.fit(self.X_train, self.y_train)
        self.gaussian_pred = self.gaussian_model.predict(self.X_test)
        
        # Bernoulli NB - works with any data (binarizes automatically)
        self.bernoulli_model = BernoulliNB(alpha=bernoulli_alpha, binarize=bernoulli_binarize)
        self.bernoulli_model.fit(self.X_train, self.y_train)
        self.bernoulli_pred = self.bernoulli_model.predict(self.X_test)
        
        # MultinomialNB - transform to non-negative
        try:
            X_train_multi = self._transform_for_multinomial(self.X_train)
            X_test_multi = self._transform_for_multinomial(self.X_test)
            self.multinomial_model = MultinomialNB(alpha=multinomial_alpha)
            self.multinomial_model.fit(X_train_multi, self.y_train)
            self.multinomial_pred = self.multinomial_model.predict(X_test_multi)
        except:
            pass
        
        # ComplementNB - transform to non-negative
        try:
            X_train_comp = self._transform_for_multinomial(self.X_train)
            X_test_comp = self._transform_for_multinomial(self.X_test)
            self.complement_model = ComplementNB(alpha=complement_alpha)
            self.complement_model.fit(X_train_comp, self.y_train)
            self.complement_pred = self.complement_model.predict(X_test_comp)
        except:
            pass
        
        # CategoricalNB - transform to non-negative integers
        try:
            X_train_cat = self._transform_for_categorical(self.X_train)
            X_test_cat = self._transform_for_categorical(self.X_test)
            self.categorical_model = CategoricalNB(alpha=categorical_alpha)
            self.categorical_model.fit(X_train_cat, self.y_train)
            self.categorical_pred = self.categorical_model.predict(X_test_cat)
        except:
            pass
        
        return self

    def calculate_metrics(self):
        metrics = {}
        
        acc = accuracy_score(self.y_test, self.gaussian_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, self.gaussian_pred, average="weighted", zero_division=0)
        metrics["gaussian_accuracy"] = acc
        metrics["gaussian_precision"] = prec
        metrics["gaussian_recall"] = rec
        metrics["gaussian_f1"] = f1
        
        if self.multinomial_pred is not None:
            acc = accuracy_score(self.y_test, self.multinomial_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, self.multinomial_pred, average="weighted", zero_division=0)
            metrics["multinomial_accuracy"] = acc
            metrics["multinomial_precision"] = prec
            metrics["multinomial_recall"] = rec
            metrics["multinomial_f1"] = f1
        
        acc = accuracy_score(self.y_test, self.bernoulli_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, self.bernoulli_pred, average="weighted", zero_division=0)
        metrics["bernoulli_accuracy"] = acc
        metrics["bernoulli_precision"] = prec
        metrics["bernoulli_recall"] = rec
        metrics["bernoulli_f1"] = f1
        
        if self.categorical_pred is not None:
            acc = accuracy_score(self.y_test, self.categorical_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, self.categorical_pred, average="weighted", zero_division=0)
            metrics["categorical_accuracy"] = acc
            metrics["categorical_precision"] = prec
            metrics["categorical_recall"] = rec
            metrics["categorical_f1"] = f1
        
        if self.complement_pred is not None:
            acc = accuracy_score(self.y_test, self.complement_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, self.complement_pred, average="weighted", zero_division=0)
            metrics["complement_accuracy"] = acc
            metrics["complement_precision"] = prec
            metrics["complement_recall"] = rec
            metrics["complement_f1"] = f1
        
        return metrics
