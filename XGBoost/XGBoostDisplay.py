import streamlit as st
from XGBoost.XGBoostClean import clean_and_encode
from XGBoost.XGBoostModel import XGBoostModel
from XGBoost.xgboost_plot_utils import plot_xgboost_boundaries, plot_xgboost_surfaces
import numpy as np


@st.fragment
def DisplayXGBoost(df, features, target):
    X, y, df_clean, encoder = clean_and_encode(df, features, target)
    if X is None:
        return

    with st.expander("🚀 XGBoost"):
        model = XGBoostModel(X, y, encoder)
        st.session_state.xgb_result = model.train(n_estimators=100, max_depth=6, learning_rate=0.1).get_results()
        current_result = st.session_state.xgb_result
        model_type = str(current_result["model_type"]).lower()
        st.caption(f"###### **Detected Model Type:** {current_result['model_type']}")

        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0,
            "min_child_weight": 1
        }

        if "xgb_params" not in st.session_state:
            st.session_state.xgb_params = default_params.copy()

        with st.expander(f"⚙️ {current_result['model_type']} - Hyperparameter Tuning", expanded=False):
            with st.form("xgb_param_form"):
                st.caption("##### Adjust Hyperparameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_n_estimators = st.number_input("N Estimators", 10, 500, value=100, step=10)
                with col2:
                    new_max_depth = st.number_input("Max Depth", 1, 20, value=6)
                with col3:
                    new_learning_rate = st.number_input("Learning Rate", 0.01, 1.0, value=0.1, step=0.01)

                col4, col5, col6 = st.columns(3)
                with col4:
                    new_subsample = st.slider("Subsample", 0.1, 1.0, value=1.0, step=0.1)
                with col5:
                    new_colsample_bytree = st.slider("Colsample Bytree", 0.1, 1.0, value=1.0, step=0.1)
                with col6:
                    new_gamma = st.number_input("Gamma", 0.0, 10.0, value=0.0, step=0.1)

                col7 = st.columns(1)[0]
                with col7:
                    new_min_child_weight = st.number_input("Min Child Weight", 1, 20, value=1)

                submit = st.form_submit_button("🔄 Re-run XGBoost")
                if submit:
                    with st.spinner("Updating XGBoost with new parameters..."):
                        new_params = {
                            "n_estimators": new_n_estimators,
                            "max_depth": new_max_depth,
                            "learning_rate": new_learning_rate,
                            "subsample": new_subsample,
                            "colsample_bytree": new_colsample_bytree,
                            "gamma": new_gamma,
                            "min_child_weight": new_min_child_weight
                        }
                        st.session_state.xgb_params = new_params
                        model = XGBoostModel(X, y, encoder)
                        st.session_state.xgb_result = model.train(**new_params).get_results()
                        st.toast("✅ XGBoost retrained successfully!", icon="🚀")
                current_result = st.session_state.xgb_result

        st.markdown("#### 📊 Model Metrics")
        metrics = current_result["metrics"]
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
                st.metric(label=key.upper(), value=formatted_value)

        with st.expander("📈 Graphical Representations"):
            display_xgboost_boundaries(X, y, current_result, model_type)


def display_xgboost_boundaries(X, y, current_result, model_type):
    if X.shape[1] < 2:
        st.info("⚠️ Need at least two features to plot decision boundaries.")
        return

    top_features_df = current_result["feature_importance"][["feature", "importance"]].head(4)
    if len(top_features_df) < 2:
        st.info("⚠️ Need at least two valid features to plot decision boundaries.")
        return
    
    top_features = top_features_df["feature"].tolist()

    st.markdown("##### 🚀 XGBoost Decision Boundaries")
    st.caption(f"###### Top priority features (max 04): {{ {', '.join(f'{f}: {i:.5f}' for f, i in zip(top_features_df.feature, top_features_df.importance))} }}")

    if "class" in model_type:
        figs = plot_xgboost_boundaries(
            X, y,
            top_features=top_features,
            params=st.session_state.xgb_params,
            class_names=current_result["class_names"]
        )
    else:
        figs = plot_xgboost_surfaces(
            X, y,
            top_features=top_features,
            params=st.session_state.xgb_params
        )

    if figs == 1:
        st.info("No Valid Feature")
        return

    cols = st.columns(6)
    for i, fig in enumerate(figs):
        with cols[i % 6]:
            st.pyplot(fig, use_container_width=True)
