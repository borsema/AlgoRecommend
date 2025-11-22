import streamlit as st
from NaiveBayes.NaiveBayesClean import clean_and_encode
from NaiveBayes.NaiveBayesModel import NaiveBayesModel
from NaiveBayes.naive_bayes_plot_utils import plot_naive_bayes_boundaries
import numpy as np


@st.fragment
def DisplayNaiveBayes(df, features, target):
    X, y, df_clean, encoder = clean_and_encode(df, features, target)
    if X is None:
        return

    with st.expander("🎯 Naive Bayes"):
        model = NaiveBayesModel(X, y, encoder)
        st.session_state.nb_result = model.train().get_results()
        current_result = st.session_state.nb_result
        st.caption(f"###### **Detected Model Type:** {current_result['model_type']}")

        default_params = {
            "var_smoothing": 1e-9
        }

        if "nb_params" not in st.session_state:
            st.session_state.nb_params = default_params.copy()

        with st.expander(f"⚙️ {current_result['model_type']} - Hyperparameter Tuning", expanded=False):
            with st.form("nb_param_form"):
                st.caption("##### Adjust Hyperparameters")
                col1 = st.columns(1)[0]
                with col1:
                    new_var_smoothing = st.number_input("Var Smoothing", 1e-12, 1e-5, value=1e-9, format="%.1e")

                submit = st.form_submit_button("🔄 Re-run Naive Bayes")
                if submit:
                    with st.spinner("Updating Naive Bayes with new parameters..."):
                        new_params = {"var_smoothing": new_var_smoothing}
                        st.session_state.nb_params = new_params
                        model = NaiveBayesModel(X, y, encoder)
                        st.session_state.nb_result = model.train(**new_params).get_results()
                        st.toast("✅ Naive Bayes retrained successfully!", icon="🚀")
                current_result = st.session_state.nb_result

        st.markdown("#### 📊 Model Metrics")
        metrics = current_result["metrics"]
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
                st.metric(label=key.upper(), value=formatted_value)

        with st.expander("📈 Graphical Representations"):
            display_naive_bayes_boundaries(X, y, current_result)


def display_naive_bayes_boundaries(X, y, current_result):
    if X.shape[1] < 2:
        st.info("⚠️ Need at least two features to plot decision boundaries.")
        return

    top_features = list(X.columns)[:4]

    st.markdown("##### 🎯 Naive Bayes Decision Boundaries")
    st.caption(f"###### Using features: {', '.join(top_features)}")

    figs = plot_naive_bayes_boundaries(
        X, y,
        top_features=top_features,
        params=st.session_state.nb_params,
        class_names=current_result["class_names"]
    )

    if figs == 1:
        st.info("No Valid Feature")
        return

    cols = st.columns(6)
    for i, fig in enumerate(figs):
        with cols[i % 6]:
            st.pyplot(fig, use_container_width=True)
