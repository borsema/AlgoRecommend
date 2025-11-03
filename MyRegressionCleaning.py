import pandas as pd
import numpy as np

def clean_regression_data(df, selected_features, target_col, one_hot_threshold=15):
    """
    Clean dataset for regression:
    - Keep only selected features and target
    - Fill missing values (numeric -> mean, categorical -> mode)
    - Encode categorical features:
        * One-hot encode if unique values <= one_hot_threshold
        * Label encode if unique values > threshold
    - Ensure numeric target

    Returns:
        X (pd.DataFrame) - cleaned features
        y (pd.Series) - numeric target
        cleaned_df (pd.DataFrame) - combined cleaned dataset
    """
    # --- Separate target and features ---
    X = df[selected_features].copy()
    y = df[target_col].copy()

    # --- Handle target column ---
    if y.isna().any():
        y = y.fillna(y.mean()) if pd.api.types.is_numeric_dtype(y) else y.fillna(y.mode()[0])
    if y.dtype == 'object':
        y = pd.to_numeric(y, errors='coerce')
        if y.isna().any():
            raise ValueError(f"Target column '{target_col}' contains non-numeric values.")

    # --- Handle missing values in features ---
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

    # --- Encode categorical features ---
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if X[col].nunique() <= one_hot_threshold:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X[col] = pd.factorize(X[col])[0]

    # --- Ensure numeric and fill remaining NaN (if any) ---
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- Final cleaned dataset ---
    cleaned_df = X.copy()
    cleaned_df[target_col] = y

    return X, y, cleaned_df
