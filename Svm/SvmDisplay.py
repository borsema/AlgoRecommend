import streamlit as st
from Svm.SvmClean import clean_and_encode
from Svm.SvmModel import SvmModel
from Svm.svm_plot_utils import plot_svm_boundaries, plot_svm_surfaces
import numpy as np


@st.fragment
def DisplaySVM(df, features, target):
    X, y, df_clean, encoder = clean_and_encode(df, features, target)
    if X is None:
        return

    with st.expander("🎯 Support Vector Machine"):
        model = SvmModel(X, y, encoder)
        st.session_state.svm_result = model.train(kernel='rbf', C=1.0).get_results()
        current_result = st.session_state.svm_result
        model_type = str(current_result["model_type"]).lower()
        st.caption(f"###### **Detected Model Type:** {current_result['model_type']}")

        default_params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "degree": 3
        }

        if "svm_params" not in st.session_state:
            st.session_state.svm_params = default_params.copy()

        with st.expander(f"⚙️ {current_result['model_type']} - Hyperparameter Tuning", expanded=False):
            with st.form("svm_param_form"):
                st.caption("##### Adjust Hyperparameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
                with col2:
                    new_C = st.number_input("C (Regularization)", 0.01, 100.0, value=1.0, step=0.1)
                with col3:
                    new_gamma = st.selectbox("Gamma", ["scale", "auto"], index=0)

                col4 = st.columns(1)[0]
                with col4:
                    new_degree = st.number_input("Degree (for poly)", 1, 10, value=3)

                submit = st.form_submit_button("🔄 Re-run SVM")
                if submit:
                    with st.spinner("Updating SVM with new parameters..."):
                        new_params = {
                            "kernel": new_kernel,
                            "C": new_C,
                            "gamma": new_gamma,
                            "degree": new_degree
                        }
                        st.session_state.svm_params = new_params
                        model = SvmModel(X, y, encoder)
                        st.session_state.svm_result = model.train(**new_params).get_results()
                        st.toast("✅ SVM retrained successfully!", icon="🚀")
                current_result = st.session_state.svm_result

        st.markdown("#### 📊 Model Metrics")
        metrics = current_result["metrics"]
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
                st.metric(label=key.upper(), value=formatted_value)

        with st.expander("📈 Graphical Representations"):
            display_svm_boundaries(X, y, current_result, model_type)


def display_svm_boundaries(X, y, current_result, model_type):
    if X.shape[1] < 2:
        st.info("⚠️ Need at least two features to plot decision boundaries.")
        return

    top_features_df = current_result["feature_importance"][["feature", "importance"]].head(4)
    if len(top_features_df) < 2:
        st.info("⚠️ Need at least two valid features to plot decision boundaries.")
        return
    
    top_features = top_features_df["feature"].tolist()

    st.markdown("##### 🎯 SVM Decision Boundaries")
    st.caption(f"###### Top priority features (max 04): {{ {', '.join(f'{f}: {i:.5f}' for f, i in zip(top_features_df.feature, top_features_df.importance))} }}")

    if "class" in model_type:
        figs = plot_svm_boundaries(
            X, y,
            top_features=top_features,
            params=st.session_state.svm_params,
            class_names=current_result["class_names"]
        )
    else:
        figs = plot_svm_surfaces(
            X, y,
            top_features=top_features,
            params=st.session_state.svm_params
        )

    if figs == 1:
        st.info("No Valid Feature")
        return

    cols = st.columns(6)
    for i, fig in enumerate(figs):
        with cols[i % 6]:
            st.pyplot(fig, use_container_width=True)
