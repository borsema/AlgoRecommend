import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample

class MyRegression:
    def __init__(self, df, X, y, test_size=0.2, random_state=42):
        self.df = df
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.linear_model = LinearRegression()
        self.lasso_model = Lasso(alpha=0.1)
        self.ridge_model = Ridge(alpha=0.1)

        self.y_pred = None
        self.lasso_pred = None
        self.ridge_pred = None

    def train_models(self):
        self.linear_model.fit(self.X_train, self.y_train)
        self.lasso_model.fit(self.X_train, self.y_train)
        self.ridge_model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.linear_model.predict(self.X_test)
        self.lasso_pred = self.lasso_model.predict(self.X_test)
        self.ridge_pred = self.ridge_model.predict(self.X_test)

    def calculate_metrics(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)

        lasso_r2 = r2_score(self.y_test, self.lasso_pred)
        ridge_r2 = r2_score(self.y_test, self.ridge_pred)

        bias, variance = self._calculate_bias_variance()

        return {
            "linear_r2": r2,
            "lasso_r2": lasso_r2,
            "ridge_r2": ridge_r2,
            "mse": mse,
            "rmse": rmse,
            "bias": bias,
            "variance": variance
        }

    def _calculate_bias_variance(self, n_bootstraps=100):
        predictions = []
        for _ in range(n_bootstraps):
            X_sample, y_sample = resample(self.X_train, self.y_train)
            self.linear_model.fit(X_sample, y_sample)
            predictions.append(self.linear_model.predict(self.X_test))
        predictions = np.array(predictions)
        bias = np.mean((np.mean(predictions, axis=0) - self.y_test) ** 2)
        variance = np.mean(np.var(predictions, axis=0))
        return bias, variance

    # def cross_validate_model(self, model_type="Linear", cv_splits=5):
    #     if model_type == "Linear":
    #         model = LinearRegression()
    #     elif model_type == "Lasso":
    #         model = Lasso(alpha=0.1)
    #     else:
    #         model = Ridge(alpha=0.1)
    #
    #     kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
    #     r2_scores = cross_val_score(model, self.X, self.y, cv=kf, scoring="r2")
    #     mse_scores = -cross_val_score(model, self.X, self.y, cv=kf, scoring="neg_mean_squared_error")
    #     return {
    #         "cv_r2_mean": np.mean(r2_scores),
    #         "cv_r2_std": np.std(r2_scores),
    #         "cv_mse_mean": np.mean(mse_scores),
    #         "cv_mse_std": np.std(mse_scores)
    #     }
