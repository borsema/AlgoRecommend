import streamlit as st
import pandas as pd

# ================================================================
# 🧹 DATA CLEANING & ENCODING
# ================================================================
def clean_regression_data(df, selected_features, target_col, one_hot_threshold=15):
    """
    Clean dataset for regression or encoded categorical targets.
    """
    X = df[selected_features].copy()

    # Drop entirely empty feature columns
    empty_cols = [col for col in X.columns if X[col].isna().all() or (X[col] == "").all()]
    if empty_cols:
        st.warning(f"⚠️ Dropped empty feature columns: {', '.join(empty_cols)}")
        X = X.drop(columns=empty_cols)

    # Check if target is completely empty
    if df[target_col].isna().all():
        st.error(f"❌ Target column '{target_col}' is completely empty!")
        return None, None, None, None
        # raise ValueError(f"Target column '{target_col}' has no valid data.")

    # Combine X and target before dropping NaNs (to keep row alignment)
    combined_df = pd.concat([X, df[target_col]], axis=1)
    combined_df = combined_df.dropna()

    # Separate back
    X = combined_df[[col for col in selected_features if col in combined_df.columns]]
    y = combined_df[target_col]



    # Encode target if categorical
    target_mapping = None
    if pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors='coerce')
    else:
        y, uniques = pd.factorize(y)
        target_mapping = {i: label for i, label in enumerate(uniques)}


    # Encode categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if X[col].nunique() <= one_hot_threshold:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
        else:
            X.loc[:, col] = pd.factorize(X[col])[0]

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    cleaned_df = X.copy()
    cleaned_df[target_col] = y

    if X.shape[1] == 0:
        st.error("❌ No valid features left after cleaning! Please select non-empty columns.")
        return None, None, None, None

    return X, y, cleaned_df, target_mapping



'''



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


'''
