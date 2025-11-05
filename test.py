# ============================================================
# 🌸 app.py — Generic Streamlit App (Classification + Regression)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from plot_utils import plot_decision_boundaries, plot_regression_surfaces

# Streamlit setup
st.set_page_config(page_title="Generic Decision Tree App", layout="wide")
st.title("🌳 Generic Decision Tree Classifier & Regressor with Top 6 Feature Combinations")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Select target and task
    target_col = st.selectbox("🎯 Select target column", df.columns)
    task_type = st.selectbox("⚙️ Select task type", ["Classification", "Regression"])

    # Separate features and target
    X_df = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical target for classification
    le = None
    if task_type == "Classification":
        if y.dtype == "object" or y.nunique() < 15:
            le = LabelEncoder()
            y = le.fit_transform(y)

    # Select numeric feature columns
    X = X_df.select_dtypes(include=["float64", "int64"])
    if X.shape[1] == 0:
        st.error("❌ No numeric feature columns found.")
        st.stop()

    feature_cols = X.columns.tolist()

    # --------------------------------------------------------
    # 🌿 CLASSIFICATION
    # --------------------------------------------------------
    if task_type == "Classification":
        st.header("🌿 Decision Tree Classification")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**✅ Accuracy:** {acc:.3f}")

        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": clf.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.write("📈 Feature Importance")
        st.dataframe(imp_df)

        # Select top 4 features for classification
        top_features = imp_df["Feature"].head(4).tolist() if len(feature_cols) > 4 else feature_cols

        st.subheader("🎨 Decision Boundary Plots (Top Feature Pairs)")
        figs = plot_decision_boundaries(X, y, le.classes_ if le else None, top_features)
        for fig in figs:
            st.pyplot(fig)

    # --------------------------------------------------------
    # 📈 REGRESSION
    # --------------------------------------------------------
    else:
        st.header("📈 Decision Tree Regression")

        reg = DecisionTreeRegressor(max_depth=4, random_state=42)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        st.write(f"**✅ R² Score:** {r2:.3f}")

        reg_imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": reg.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.write("📊 Feature Importance")
        st.dataframe(reg_imp_df)

        # Top 4 or all
        top_features = reg_imp_df["Feature"].head(4).tolist() if len(feature_cols) > 4 else feature_cols

        st.subheader("📉 Regression Decision Curves (Top Feature Pairs)")
        reg_figs = plot_regression_surfaces(X, y, top_features)
        for fig in reg_figs:
            st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to begin.")
