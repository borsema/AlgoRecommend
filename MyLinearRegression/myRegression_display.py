import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from sklearn.metrics import r2_score
from MyLinearRegression.myRegressionCleaning import clean_regression_data
from MyLinearRegression.myRegression import MyRegression

# ================================================================
# 🎛️ STREAMLIT DISPLAY
# ================================================================
@st.fragment
def DisplayRegression(df: pd.DataFrame, selected_features: list, tagged_column: str):

    X, y, cleaned_df, target_mapping = clean_regression_data(df, selected_features, tagged_column)
    if X is None:
        st.info("Skipping this section because the target column or all features are empty.")
        return

    with st.expander("📈 Linear Regression", expanded=False):

        # Step 1: Train initial model
        model = MyRegression(cleaned_df, X, y)
        model.train_models()
        model.predict()
        metrics = model.calculate_metrics()

        # Step 2: Define default hyperparameters
        default_params = {
            "lasso_alpha": 0.1,
            "ridge_alpha": 0.1,
            "poly_degree": 2,
            "model": model,
            "metrics": metrics
        }

        # Initialize session state (similar to Decision Tree logic)
        if "regression_state" not in st.session_state:
            st.session_state.regression_state = default_params.copy()

        # Step 3: Hyperparameter tuning form
        with st.expander("⚙️ Regression - Hyperparameter Tuning", expanded=False):
            with st.form("regression_param_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_lasso_alpha = st.slider("Lasso α", 0.001, 2.0, default_params["lasso_alpha"], step=0.05)
                with col2:
                    new_ridge_alpha = st.slider("Ridge α", 0.001, 2.0, default_params["ridge_alpha"], step=0.05)
                with col3:
                    new_poly_degree = st.number_input("Polynomial Degree", 1, 6, int(default_params["poly_degree"]))

                submit = st.form_submit_button("🔄 Re-run Regression")

                if submit:
                    with st.spinner("Updating regression models with new parameters..."):
                        model = MyRegression(
                            cleaned_df, X, y,
                            lasso_alpha=new_lasso_alpha,
                            ridge_alpha=new_ridge_alpha,
                            poly_degree=int(new_poly_degree)
                        )
                        model.train_models()
                        model.predict()
                        metrics = model.calculate_metrics()

                        # Update session state
                        st.session_state.regression_state.update({
                            "lasso_alpha": new_lasso_alpha,
                            "ridge_alpha": new_ridge_alpha,
                            "poly_degree": new_poly_degree,
                            "model": model,
                            "metrics": metrics
                        })

                    # Refresh local variables after retraining
                    reg_state = st.session_state.regression_state
                    model = reg_state["model"]
                    metrics = reg_state["metrics"]
                    st.toast("✅ Regression models retrained successfully!", icon="🚀")

        # Step 4: Display Metrics
        st.markdown("#### 📊 Model Metrics")
        display_metrics(metrics)

        with st.expander(" 📈 Graphical Representations"):
            # Step 5: Visualization
            st.markdown("##### 📈 Model Visualizations")
            plot_all_graphs_horizontal(model, cleaned_df, tagged_column)

            # Step 6: Predictions & Download
            st.markdown("##### 📄 Predictions & Download")
            download_predictions(model, selected_features, tagged_column, target_mapping)

# ================================================================
# 📊 METRICS DISPLAY
# ================================================================
def display_metrics(metrics):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Linear R²", round(metrics["linear_r2"], 4))
        st.metric("RMSE", round(metrics["rmse"], 4))
    with col2:
        st.metric("Lasso R²", round(metrics["lasso_r2"], 4))
        st.metric("Bias", round(metrics["bias"], 4))
    with col3:
        st.metric("Ridge R²", round(metrics["ridge_r2"], 4))
        st.metric("Variance", round(metrics["variance"], 4))
    with col4:
        st.metric("Poly R²", round(metrics["poly_r2"], 4))
    with col5:
        st.metric("MSE", round(metrics["mse"], 4))


# ================================================================
# 📉 PLOTS (5 per row)
# ================================================================
def plot_all_graphs_horizontal(model, df, target_col):

    figs = []

    # Predicted vs Actual
    fig_pred, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(model.y_test, model.y_pred, s=20, alpha=0.7, label=f"Linear (R²={r2_score(model.y_test, model.y_pred):.2f})")
    ax.scatter(model.y_test, model.lasso_pred, s=20, alpha=0.7, label=f"Lasso (R²={r2_score(model.y_test, model.lasso_pred):.2f})")
    ax.scatter(model.y_test, model.ridge_pred, s=20, alpha=0.7, label=f"Ridge (R²={r2_score(model.y_test, model.ridge_pred):.2f})")
    ax.scatter(model.y_test, model.poly_pred, s=20, alpha=0.7, label=f"Poly (R²={r2_score(model.y_test, model.poly_pred):.2f})")
    min_y, max_y = model.y_test.min(), model.y_test.max()
    ax.plot([min_y, max_y], [min_y, max_y], 'r--', label="Perfect Fit")
    ax.set_xlabel("Actual", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    ax.legend(fontsize=6)
    figs.append(fig_pred)

    # Residuals
    fig_res, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(model.y_pred, model.y_test - model.y_pred, s=15, alpha=0.7, label="Linear")
    ax.scatter(model.poly_pred, model.y_test - model.poly_pred, s=15, alpha=0.7, label="Poly")
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Residuals", fontsize=8)
    ax.legend(fontsize=6)
    figs.append(fig_res)

    numeric_df = df.select_dtypes(include=[np.number])
    features = [col for col in numeric_df.columns if col != target_col]

    for feature in features:
        x = numeric_df[feature]
        y = numeric_df[target_col]
        unique_vals = np.unique(x)
        fig, ax = plt.subplots(figsize=(3.2, 3.2))

        if len(unique_vals) <= 10:
            categories = sorted(unique_vals)
            preds_mean = {"Linear": [], "Lasso": [], "Ridge": [], "Poly": []}
            for cat in categories:
                baseline = model.X_test.mean(axis=0).to_frame().T
                baseline[feature] = cat
                X_input = baseline[model.X_train.columns]
                preds_mean["Linear"].append(model.linear_model.predict(X_input)[0])
                preds_mean["Lasso"].append(model.lasso_model.predict(X_input)[0])
                preds_mean["Ridge"].append(model.ridge_model.predict(X_input)[0])
                preds_mean["Poly"].append(model.poly_pipeline.predict(X_input)[0])

            ax.scatter(x, y, alpha=0.5, s=25, color="gray", label="Actual")
            ax.plot(categories, preds_mean["Linear"], "--", label="Linear")
            ax.plot(categories, preds_mean["Lasso"], "-.", label="Lasso")
            ax.plot(categories, preds_mean["Ridge"], ":", label="Ridge")
            ax.plot(categories, preds_mean["Poly"], "-", label=f"Poly (deg={model.poly_degree})")

        else:
            x_min, x_max = x.min(), x.max()
            x_range = np.linspace(x_min, x_max, 100)
            baseline = model.X_test.mean(axis=0).to_frame().T
            preds = {"Linear": [], "Lasso": [], "Ridge": [], "Poly": []}
            for val in x_range:
                row = baseline.copy()
                row[feature] = val
                X_input = row[model.X_train.columns]
                preds["Linear"].append(model.linear_model.predict(X_input)[0])
                preds["Lasso"].append(model.lasso_model.predict(X_input)[0])
                preds["Ridge"].append(model.ridge_model.predict(X_input)[0])
                preds["Poly"].append(model.poly_pipeline.predict(X_input)[0])

            ax.scatter(x, y, s=20, alpha=0.5, color="gray", label="Actual")
            ax.plot(x_range, preds["Linear"], "--", label="Linear")
            ax.plot(x_range, preds["Lasso"], "-.", label="Lasso")
            ax.plot(x_range, preds["Ridge"], ":", label="Ridge")
            ax.plot(x_range, preds["Poly"], "-", label=f"Poly (deg={model.poly_degree})")

        ax.set_title(feature, fontsize=9)
        ax.legend(fontsize=6)
        figs.append(fig)

    # Display 5 per row
    n_cols = 5
    for i in range(0, len(figs), n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            if i + j < len(figs):
                with cols[j]:
                    st.pyplot(figs[i + j])


# ================================================================
# 📥 DOWNLOAD
# ================================================================
def download_predictions(model, feature_columns, target_col, mappings):
    X_test_df = pd.DataFrame(model.X_test, columns=model.X.columns)
    results_df = X_test_df.copy()
    results_df[target_col] = model.y_test

    if mappings:
        results_df[f"{target_col}_decoded"] = results_df[target_col].map(mappings)

    results_df["Predicted_Linear"] = model.y_pred
    results_df["Predicted_Lasso"] = model.lasso_pred
    results_df["Predicted_Ridge"] = model.ridge_pred
    results_df["Predicted_Poly"] = model.poly_pred

    st.dataframe(results_df.head(5), use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False)

    st.download_button(
        "📥 Download Predictions",
        data=output.getvalue(),
        file_name="regression_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )




'''
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io


# -----------------------------
# METRICS DISPLAY
# -----------------------------
def display_metrics(metrics):
    """Display metrics for Linear, Lasso, Ridge, and Polynomial from MyRegression."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Linear R²", round(metrics["linear_r2"], 4))
        st.metric("Linear RMSE", round(metrics["rmse"], 4))
    with col2:
        st.metric("Lasso R²", round(metrics["lasso_r2"], 4))
        st.metric("Bias", round(metrics["bias"], 4))
    with col3:
        st.metric("Ridge R²", round(metrics["ridge_r2"], 4))
        st.metric("Variance", round(metrics["variance"], 4))
    with col4:
        st.metric("Poly R²", round(metrics.get("poly_r2", 0.0), 4))
    with col5:
        st.metric("MSE", round(metrics["mse"], 4))


# -----------------------------
# PLOTS DISPLAY
# -----------------------------
def plot_all_graphs_horizontal(model, df, target_col):
    """
    Display combined predicted vs actual, residuals, and feature plots.
    Now includes Polynomial Regression visualizations.
    """
    st.subheader("📊 Combined, Residual & Feature Plots (All Models)")

    figs = []

    # --- Combined Features vs Target (Predicted vs Actual) ---
    fig_combined, ax_combined = plt.subplots(figsize=(3, 3))
    ax_combined.scatter(model.y_test, model.y_pred, alpha=0.6, color="green", s=15, label="Linear")
    ax_combined.scatter(model.y_test, model.poly_pred, alpha=0.6, color="purple", s=15, label="Polynomial")
    ax_combined.plot([model.y_test.min(), model.y_test.max()],
                     [model.y_test.min(), model.y_test.max()],
                     color="red", linestyle="--", label="Perfect Fit")
    ax_combined.set_xlabel("Actual", fontsize=8)
    ax_combined.set_ylabel("Predicted", fontsize=8)
    ax_combined.set_title("Predicted vs Actual", fontsize=10)
    ax_combined.legend(fontsize=6)
    ax_combined.grid(True)
    figs.append(fig_combined)

    # --- Residuals vs Predicted (Linear vs Polynomial) ---
    residuals_linear = model.y_test - model.y_pred
    residuals_poly = model.y_test - model.poly_pred
    fig_residuals, ax_residuals = plt.subplots(figsize=(3, 3))
    ax_residuals.scatter(model.y_pred, residuals_linear, alpha=0.6, s=15, label="Linear")
    ax_residuals.scatter(model.poly_pred, residuals_poly, alpha=0.6, s=15, label="Polynomial")
    ax_residuals.axhline(0, color="red", linestyle="--")
    ax_residuals.set_xlabel("Predicted", fontsize=8)
    ax_residuals.set_ylabel("Residuals", fontsize=8)
    ax_residuals.set_title("Residuals Comparison", fontsize=10)
    ax_residuals.legend(fontsize=6)
    figs.append(fig_residuals)

    # --- Individual Feature vs Target (with best fit line) ---
    numeric_df = df.select_dtypes(include=[np.number])
    features = [col for col in numeric_df.columns if col != target_col]

    for feature in features:
        x = df[feature].values
        y = df[target_col].values

        # Linear fit line
        coeffs_lin = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y_lin = coeffs_lin[0] * line_x + coeffs_lin[1]

        # Polynomial (degree 2 for visual)
        coeffs_poly = np.polyfit(x, y, 2)
        line_y_poly = coeffs_poly[0] * line_x ** 2 + coeffs_poly[1] * line_x + coeffs_poly[2]

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(x, y, alpha=0.7, s=15, label="Actual")
        ax.plot(line_x, line_y_lin, color="green", linestyle="--", label="Linear Fit")
        ax.plot(line_x, line_y_poly, color="purple", linestyle="-.", label="Poly Fit (deg 2)")
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



# -----------------------------
# DOWNLOAD PREDICTIONS
# -----------------------------
def download_predictions(model, feature_columns, target_col, target_mapping=None):
    """
    Display predictions (Linear, Lasso, Ridge, Polynomial)
    and allow Excel download with category labels if mapping exists.
    """
    # --- Safe dataframe creation ---
    if isinstance(model.X_test, pd.DataFrame):
        X_test_df = model.X_test.copy()
    else:
        if hasattr(model, "X") and isinstance(model.X, pd.DataFrame):
            X_test_df = pd.DataFrame(model.X_test, columns=model.X.columns)
        else:
            n_features = model.X_test.shape[1]
            if len(feature_columns) == n_features:
                X_test_df = pd.DataFrame(model.X_test, columns=feature_columns)
            else:
                X_test_df = pd.DataFrame(model.X_test, columns=[f"feature_{i}" for i in range(n_features)])

    # --- Combine features + target ---
    results_df = X_test_df.copy()
    results_df[target_col] = model.y_test

    # --- Add categorical label if mapping exists ---
    if target_mapping is not None:
        def get_label(x):
            try:
                return target_mapping[int(x)]
            except Exception:
                return None

        results_df[f"{target_col}_category"] = results_df[target_col].apply(get_label)
        col_order = list(X_test_df.columns) + [target_col, f"{target_col}_category"]
    else:
        col_order = list(X_test_df.columns) + [target_col]

    # --- Add predictions ---
    results_df["Predicted_Linear"] = model.y_pred
    results_df["Predicted_Lasso"] = model.lasso_pred
    results_df["Predicted_Ridge"] = model.ridge_pred
    results_df["Predicted_Poly"] = model.poly_pred

    results_df = results_df[col_order + [
        "Predicted_Linear", "Predicted_Lasso", "Predicted_Ridge", "Predicted_Poly"
    ]]

    # ---- Display sample predictions ----
    st.subheader("📄 Predictions (Linear, Lasso, Ridge, Polynomial)")
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

'''


'''

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

def display_metrics(metrics):
    """Display metrics for Linear, Lasso, and Ridge from MyRegression."""
    # st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Linear R²", round(metrics["linear_r2"], 4))
        st.metric("Linear RMSE", round(metrics["rmse"], 4))
    with col2:
        st.metric("Lasso R²", round(metrics["lasso_r2"], 4))
        st.metric("Bias", round(metrics["bias"], 4))
    with col3:
        st.metric("Ridge R²", round(metrics["ridge_r2"], 4))
        st.metric("Variance", round(metrics["variance"], 4))
    with col4:
        st.metric("MSE", round(metrics["mse"], 4))

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

def download_predictions(model, feature_columns, target_col, target_mapping=None):
    """
    Display predictions and allow download with original categorical labels if available.
    Handles case where model.X_test has encoded (dummy) feature columns.
    """
    # Try to use column names from model.X_test if available
    if isinstance(model.X_test, pd.DataFrame):
        X_test_df = model.X_test.copy()
    else:
        # Try to use model.X.columns if available
        if hasattr(model, "X") and isinstance(model.X, pd.DataFrame):
            X_test_df = pd.DataFrame(model.X_test, columns=model.X.columns)
        else:
            # Fallback to given feature names (if same shape)
            n_features = model.X_test.shape[1]
            if len(feature_columns) == n_features:
                X_test_df = pd.DataFrame(model.X_test, columns=feature_columns)
            else:
                # Last fallback — use generic names to prevent KeyError
                X_test_df = pd.DataFrame(model.X_test, columns=[f"feature_{i}" for i in range(n_features)])

    # Combine features + target
    results_df = X_test_df.copy()
    results_df[target_col] = model.y_test

    # --- Add categorical target label column if mapping exists ---
    if target_mapping is not None:
        def get_label(x):
            try:
                return target_mapping[int(x)]
            except Exception:
                return None

        results_df[f"{target_col}_category"] = results_df[target_col].apply(get_label)
        col_order = list(X_test_df.columns) + [target_col, f"{target_col}_category"]
    else:
        col_order = list(X_test_df.columns) + [target_col]

    # --- Add predictions after target/category columns ---
    results_df["Predicted_Linear"] = model.y_pred
    results_df["Predicted_Lasso"] = model.lasso_pred
    results_df["Predicted_Ridge"] = model.ridge_pred

    results_df = results_df[col_order + ["Predicted_Linear", "Predicted_Lasso", "Predicted_Ridge"]]

    # ---- Display top rows ----
    st.subheader("📄 Predictions")
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

'''