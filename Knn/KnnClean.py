import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

def clean_and_encode(df, selected_features, target_col):
    # Step 1: Prepare X and y
    X = df[selected_features].copy()
    y = df[target_col].copy()

    # Step 2: Drop completely empty columns
    empty_cols = []
    for col in X.columns:
        if X[col].isna().all() or (X[col].astype(str).str.strip() == "").all():
            empty_cols.append(col)

    if empty_cols:
        st.warning(f"⚠️ Dropped empty feature columns: {', '.join(empty_cols)}")
        X = X.drop(columns=empty_cols)

    # Step 3: If no valid feature columns left → stop
    if X.shape[1] == 0:
        st.error("❌ No valid feature columns remaining after cleaning!")
        return None, None, None, None

    # Step 4: Combine X and y, remove NaN rows
    df_clean = pd.concat([X, y], axis=1)
    df_clean = df_clean.dropna(subset=[target_col])  # must have target

    # Step 5: Check if target became empty
    if df_clean[target_col].isna().all():
        st.error(f"❌ Target column '{target_col}' is empty!")
        return None, None, None, None

    # Step 6: Remove rows where all features are NaN
    df_clean = df_clean.dropna(subset=X.columns, how="all")

    # Step 7: Recreate cleaned X and y
    X_clean = df_clean[X.columns]
    y_clean = df_clean[target_col]

    # Step 8: Encode features + target
    encoder = ConsistentEncoder()
    X_encoded = encoder.fit_transform_X(X_clean)
    y_encoded = encoder.fit_transform_y(y_clean)
    
    # Step 9: Apply StandardScaler for KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_final = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X_encoded.index)
    
    # Step 10: Final NaN check and removal
    if X_final.isna().any().any() or pd.Series(y_encoded).isna().any():
        combined = pd.concat([X_final, pd.Series(y_encoded, index=X_final.index)], axis=1)
        combined = combined.dropna()
        if combined.empty:
            st.error("❌ No data remaining after final cleaning!")
            return None, None, None, None
        X_final = combined.iloc[:, :-1]
        y_encoded = combined.iloc[:, -1].values

    return X_final, y_encoded, df_clean, encoder


class ConsistentEncoder:
    def __init__(self):
        self.feature_encoders = {}
        self.target_encoder = None
        self.class_names = None

    def fit_transform_X(self, X):
        X = X.copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                cats = X[col].astype("category")
                X[col] = cats.cat.codes
                self.feature_encoders[col] = list(cats.cat.categories)
        return X

    def fit_transform_y(self, y):
        y = y.copy()
        if not pd.api.types.is_numeric_dtype(y):
            cats = y.astype("category")
            self.class_names = list(cats.cat.categories)
            self.target_encoder = {name: i for i, name in enumerate(self.class_names)}
            return cats.cat.codes
        else:
            unique_vals = np.unique(y)
            self.class_names = [str(v) for v in unique_vals]
            self.target_encoder = {v: i for i, v in enumerate(unique_vals)}
            return pd.Categorical(y, categories=unique_vals).codes