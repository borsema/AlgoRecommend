import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

def display_metrics(metrics):
    """Display metrics for Linear, Lasso, and Ridge from MyRegression."""
    st.subheader("📊 Model Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Linear R²", round(metrics["linear_r2"], 4))
        st.metric("Variance", round(metrics["variance"], 4))
    with col2:
        st.metric("Lasso R²", round(metrics["lasso_r2"], 4))
        st.metric("Bias", round(metrics["bias"], 4))
    with col3:
        st.metric("Ridge R²", round(metrics["ridge_r2"], 4))
    with col4:
        st.metric("Linear RMSE", round(metrics["rmse"], 4))
    with col5:
        st.metric("MSE", round(metrics["mse"], 4))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_all_graphs_horizontal(model, df, target_col):
    """
    Display combined predicted vs actual, residuals plot,
    and all feature vs target plots horizontally with best fit lines.
    """
    st.subheader("📊 Combined, Residual & Individual Feature Plots")

    figs = []

    # --- Combined Features vs Target (Predicted vs Actual) ---
    fig_combined, ax_combined = plt.subplots(figsize=(3, 3))
    ax_combined.scatter(model.y_test, model.y_pred, alpha=0.7, color="green", s=15)
    ax_combined.plot([model.y_test.min(), model.y_test.max()],
                     [model.y_test.min(), model.y_test.max()],
                     color="red", linestyle="--", label="Perfect Fit")
    ax_combined.set_xlabel("Actual", fontsize=8)
    ax_combined.set_ylabel("Predicted", fontsize=8)
    ax_combined.set_title("Predicted vs Actual", fontsize=10)
    ax_combined.legend(fontsize=6)
    ax_combined.grid(True)
    figs.append(fig_combined)

    # --- Residuals vs Predicted ---
    residuals = model.y_test - model.y_pred
    fig_residuals, ax_residuals = plt.subplots(figsize=(3, 3))
    ax_residuals.scatter(model.y_pred, residuals, alpha=0.7, s=15)
    ax_residuals.axhline(0, color="red", linestyle="--")
    ax_residuals.set_xlabel("Predicted", fontsize=8)
    ax_residuals.set_ylabel("Residuals", fontsize=8)
    ax_residuals.set_title("Residuals vs Predicted", fontsize=10)
    figs.append(fig_residuals)

    # --- Individual Feature vs Target (with best fit line) ---
    numeric_df = df.select_dtypes(include=[np.number])
    features = [col for col in numeric_df.columns if col != target_col]

    for feature in features:
        x = df[feature].values
        y = df[target_col].values

        coeffs = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = coeffs[0] * line_x + coeffs[1]

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(x, y, alpha=0.7, s=15)
        ax.plot(line_x, line_y, color="red", linestyle="--", label="Best Fit")
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel(target_col, fontsize=8)
        ax.set_title(f"{feature} vs {target_col}", fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True)
        figs.append(fig)

    # --- Display horizontally (4 per row) ---
    n_cols = 4
    rows = (len(figs) + n_cols - 1) // n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for i in range(n_cols):
            idx = r * n_cols + i
            if idx >= len(figs):
                break
            with cols[i]:
                st.pyplot(figs[idx])

def download_predictions(model, feature_columns, target_col):
    """
    Display predictions for all three models along with actual tagged column
    and all selected feature values. Also provides a download option.

    Parameters:
        model: Trained MyRegression model
        feature_columns: List of selected feature column names
        target_col: Name of the target column
    """
    # Convert X_test to DataFrame (if it's numpy)
    if not isinstance(model.X_test, pd.DataFrame):
        X_test_df = pd.DataFrame(model.X_test, columns=feature_columns)
    else:
        X_test_df = model.X_test.copy()

    # Combine features, actual values, and predictions
    results_df = X_test_df.copy()
    results_df[target_col] = model.y_test
    results_df["Predicted_Linear"] = model.y_pred
    results_df["Predicted_Lasso"] = model.lasso_pred
    results_df["Predicted_Ridge"] = model.ridge_pred

    # ---- Display top 10 rows ----
    st.subheader("📄 Predictions ")
    st.dataframe(results_df.head(5), use_container_width=True)

    # ---- Download full dataset as Excel ----
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False)
    st.download_button(
        label="📥 Download Full Predictions",
        data=output.getvalue(),
        file_name="regression_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
