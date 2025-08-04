import pandas as pd

def clean_regression_data(df, selected_features, target_col):
    """
    Cleans dataset for regression:
    - Keeps only selected features + target column
    - Handles missing values
    - Converts string categorical feature columns to one-hot encoding
    - Ensures target column is numeric
    Returns:
        X (numpy array), y (numpy array), cleaned_df (for reference)
    """

    # ---- Keep only required columns ----
    df = df[selected_features + [target_col]].copy()

    # ---- Handle Missing Values ----
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ---- Handle categorical features (string/object) ----
    categorical_cols = [col for col in selected_features if df[col].dtype == 'object']
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # ---- Ensure target column is numeric ----
    if df[target_col].dtype == 'object':
        try:
            df[target_col] = pd.to_numeric(df[target_col], errors='raise')
        except Exception:
            raise ValueError(f"Target column '{target_col}' is categorical (string). "
                             "Regression requires a numeric target.")

    # ---- Separate X and y ----
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    return X, y, df
