import streamlit as st
from NaiveBayes.NaiveBayesClean import clean_and_encode
from NaiveBayes.NaiveBayesModel import NaiveBayesModel
from NaiveBayes.naive_bayes_plot_utils import plot_naive_bayes_boundaries
import numpy as np
import matplotlib.pyplot as plt


@st.fragment
def DisplayNaiveBayes(df, features, target):
    X, y, df_clean, encoder = clean_and_encode(df, features, target)
    if X is None:
        return

    with st.expander("🎯 Naive Bayes", expanded=False):
        model = NaiveBayesModel(X, y, encoder)
        model.train_models()
        metrics = model.calculate_metrics()
        st.session_state.nb_state = {
            "gaussian_var_smoothing": 1e-9,
            "multinomial_alpha": 1.0,
            "bernoulli_alpha": 1.0,
            "bernoulli_binarize": 0.0,
            "categorical_alpha": 1.0,
            "complement_alpha": 1.0,
            "model": model,
            "metrics": metrics
        }
        st.session_state.nb_result = {"metrics": metrics}
        current_state = st.session_state.nb_state

        default_params = {
            "gaussian_var_smoothing": 1e-9,
            "multinomial_alpha": 1.0,
            "bernoulli_alpha": 1.0,
            "bernoulli_binarize": 0.0,
            "categorical_alpha": 1.0,
            "complement_alpha": 1.0
        }

        if "nb_params" not in st.session_state:
            st.session_state.nb_params = {"var_smoothing": 1e-9}

        with st.expander("⚙️ Naive Bayes - Hyperparameter Tuning", expanded=False):
            with st.form("nb_param_form"):
                st.caption("##### Adjust Hyperparameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_multinomial_alpha = st.slider("Multinomial α", 0.01, 5.0, current_state["multinomial_alpha"], step=0.1)
                    new_bernoulli_alpha = st.slider("Bernoulli α", 0.01, 5.0, current_state["bernoulli_alpha"], step=0.1)
                with col2:
                    new_bernoulli_binarize = st.slider("Bernoulli Binarize", 0.0, 1.0, current_state["bernoulli_binarize"], step=0.1)
                    new_categorical_alpha = st.slider("Categorical α", 0.01, 5.0, current_state["categorical_alpha"], step=0.1)
                with col3:
                    new_complement_alpha = st.slider("Complement α", 0.01, 5.0, current_state["complement_alpha"], step=0.1)

                submit = st.form_submit_button("🔄 Re-run Naive Bayes")
                if submit:
                    with st.spinner("Updating Naive Bayes with new parameters..."):
                        model = NaiveBayesModel(X, y, encoder)
                        model.train_models(
                            multinomial_alpha=new_multinomial_alpha,
                            bernoulli_alpha=new_bernoulli_alpha,
                            bernoulli_binarize=new_bernoulli_binarize,
                            categorical_alpha=new_categorical_alpha,
                            complement_alpha=new_complement_alpha
                        )
                        metrics = model.calculate_metrics()
                        st.session_state.nb_params = {"var_smoothing": 1e-9}
                        st.session_state.nb_state = {
                            "gaussian_var_smoothing": 1e-9,
                            "multinomial_alpha": new_multinomial_alpha,
                            "bernoulli_alpha": new_bernoulli_alpha,
                            "bernoulli_binarize": new_bernoulli_binarize,
                            "categorical_alpha": new_categorical_alpha,
                            "complement_alpha": new_complement_alpha,
                            "model": model,
                            "metrics": metrics
                        }
                        st.session_state.nb_result = {"metrics": metrics}
                        st.toast("✅ Naive Bayes retrained successfully!", icon="🚀")
            current_state = st.session_state.nb_state

        st.markdown("#### 📊 Model Metrics")
        
        # Show which models are available
        available_models = []
        if "gaussian_accuracy" in current_state["metrics"]:
            available_models.append("Gaussian")
        if "multinomial_accuracy" in current_state["metrics"]:
            available_models.append("Multinomial")
        if "bernoulli_accuracy" in current_state["metrics"]:
            available_models.append("Bernoulli")
        if "categorical_accuracy" in current_state["metrics"]:
            available_models.append("Categorical")
        if "complement_accuracy" in current_state["metrics"]:
            available_models.append("Complement")
        
        st.caption(f"Available models: {', '.join(available_models)}")
        
        display_nb_metrics(current_state["metrics"])

        with st.expander("📈 Graphical Representations"):
            display_naive_bayes_boundaries(X, y, current_state)


def display_nb_metrics(metrics):
    models_info = [
        ("Gaussian", "gaussian"),
        ("Multinomial", "multinomial"),
        ("Bernoulli", "bernoulli"),
        ("Categorical", "categorical"),
        ("Complement", "complement")
    ]
    
    for name, prefix in models_info:
        if f"{prefix}_accuracy" in metrics:
            st.caption(f"##### {name} NB")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics[f'{prefix}_accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics[f'{prefix}_precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics[f'{prefix}_recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics[f'{prefix}_f1']:.4f}")


def display_naive_bayes_boundaries(X, y, current_state):
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
        class_names=current_state["model"].class_names
    )

    if figs == 1:
        st.info("No Valid Feature")
        return

    cols = st.columns(6)
    for i, fig in enumerate(figs):
        with cols[i % 6]:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
