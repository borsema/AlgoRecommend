import streamlit as st
from LogisticRegression.LogisticRegressionClean import clean_and_encode
from LogisticRegression.LogisticRegressionModel import LogisticRegressionModel
from LogisticRegression.logistic_regression_plot_utils import plot_logistic_regression_boundaries
import numpy as np


@st.fragment
def DisplayLogisticRegression(df, features, target):
    X, y, df_clean, encoder = clean_and_encode(df, features, target)
    if X is None:
        return

    with st.expander("🎯 Logistic Regression"):
        model = LogisticRegressionModel(X, y, encoder)
        st.session_state.lr_result = model.train(max_iter=1000, solver='lbfgs').get_results()
        current_result = st.session_state.lr_result
        st.caption(f"###### **Detected Model Type:** {current_result['model_type']}")

        default_params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "penalty": "l2"
        }

        if "lr_params" not in st.session_state:
            st.session_state.lr_params = default_params.copy()

        with st.expander(f"⚙️ {current_result['model_type']} - Hyperparameter Tuning", expanded=False):
            with st.form("lr_param_form"):
                st.caption("##### Adjust Hyperparameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_C = st.number_input("C (Regularization)", 0.01, 100.0, value=1.0, step=0.1)
                with col2:
                    new_solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"], index=0)
                with col3:
                    new_penalty = st.selectbox("Penalty", ["l2", "l1", "none"], index=0)

                col4 = st.columns(1)[0]
                with col4:
                    new_max_iter = st.number_input("Max Iterations", 100, 5000, value=1000, step=100)

                submit = st.form_submit_button("🔄 Re-run Logistic Regression")
                if submit:
                    with st.spinner("Updating Logistic Regression with new parameters..."):
                        new_params = {
                            "C": new_C,
                            "max_iter": new_max_iter,
                            "solver": new_solver,
                            "penalty": new_penalty if new_penalty != "none" else None
                        }
                        st.session_state.lr_params = new_params
                        model = LogisticRegressionModel(X, y, encoder)
                        st.session_state.lr_result = model.train(**new_params).get_results()
                        st.toast("✅ Logistic Regression retrained successfully!", icon="🚀")
            current_result = st.session_state.lr_result

        st.markdown("#### 📊 Model Metrics")
        metrics = current_result["metrics"]
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
                st.metric(label=key.upper(), value=formatted_value)

        with st.expander("📈 Graphical Representations"):
            display_logistic_regression_boundaries(X, y, current_result)


def display_logistic_regression_boundaries(X, y, current_result):
    if X.shape[1] < 2:
        st.info("⚠️ Need at least two features to plot decision boundaries.")
        return

    top_features_df = current_result["feature_importance"][["feature", "importance"]].head(4)
    if len(top_features_df) < 2:
        st.info("⚠️ Need at least two valid features to plot decision boundaries.")
        return
    
    top_features = top_features_df["feature"].tolist()

    st.markdown("##### 🎯 Logistic Regression Decision Boundaries")
    st.caption(f"###### Top priority features (max 04): {{ {', '.join(f'{f}: {i:.5f}' for f, i in zip(top_features_df.feature, top_features_df.importance))} }}")

    figs = plot_logistic_regression_boundaries(
        X, y,
        top_features=top_features,
        params=st.session_state.lr_params,
        class_names=current_result["class_names"]
    )

    if figs == 1:
        st.info("No Valid Feature")
        return

    cols = st.columns(6)
    for i, fig in enumerate(figs):
        with cols[i % 6]:
            st.pyplot(fig, use_container_width=True)
