import streamlit as st
from Knn.KnnClean import clean_and_encode
from Knn.KnnModel import KnnModel
from Knn.knn_plot_utils import plot_knn_boundaries, plot_knn_surfaces
import numpy as np


@st.fragment
def DisplayKNN(df, features, target):
    X, y, df_clean, encoder = clean_and_encode(df, features, target)
    if X is None:
        return

    with st.expander("🎯 K-Nearest Neighbors"):
        # Step 1: Train initial model
        model = KnnModel(X, y, encoder)
        st.session_state.knn_result = model.train(n_neighbors=5).get_results()
        current_result = st.session_state.knn_result
        model_type = str(current_result["model_type"]).lower()
        st.caption(f"###### **Detected Model Type:** {current_result['model_type']}")

        # Step 2: Define default hyperparameters
        default_params = {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski"
        }

        if "knn_params" not in st.session_state:
            st.session_state.knn_params = default_params.copy()

        # Step 3: Hyperparameter tuning form
        with st.expander(f"⚙️ {current_result['model_type']} - Hyperparameter Tuning", expanded=False):
            with st.form("knn_param_form"):
                st.caption("##### Adjust Hyperparameters")
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_n_neighbors = st.number_input("N Neighbors", 1, 50, value=5)
                with col2:
                    new_weights = st.selectbox("Weights", ["uniform", "distance"], index=0)
                with col3:
                    new_algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0)

                col4, col5, col6 = st.columns(3)
                with col4:
                    new_leaf_size = st.number_input("Leaf Size", 1, 100, value=30)
                with col5:
                    new_p = st.number_input("P (Minkowski)", 1, 5, value=2)
                with col6:
                    new_metric = st.selectbox("Metric", ["minkowski", "euclidean", "manhattan"], index=0)

                submit = st.form_submit_button("🔄 Re-run KNN")
                if submit:
                    with st.spinner("Updating KNN with new parameters..."):
                        new_params = {
                            "n_neighbors": new_n_neighbors,
                            "weights": new_weights,
                            "algorithm": new_algorithm,
                            "leaf_size": new_leaf_size,
                            "p": new_p,
                            "metric": new_metric
                        }
                        st.session_state.knn_params = new_params
                        model = KnnModel(X, y, encoder)
                        st.session_state.knn_result = model.train(**new_params).get_results()
                        st.toast("✅ KNN retrained successfully!", icon="🚀")
                current_result = st.session_state.knn_result

        # Step 4: Display Model Metrics
        st.markdown("#### 📊 Model Metrics")
        metrics = current_result["metrics"]
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
                st.metric(label=key.upper(), value=formatted_value)

        # Step 5: Graphical Representations
        with st.expander("📈 Graphical Representations"):
            display_knn_boundaries(X, y, current_result, model_type)


def display_knn_boundaries(X, y, current_result, model_type):
    """Visualize KNN decision boundaries using top features."""
    if X.shape[1] < 2:
        st.info("⚠️ Need at least two features to plot decision boundaries.")
        return

    # Use first 4 features for KNN (no feature importance)
    top_features = list(X.columns)[:4]

    st.markdown("##### 🎯 KNN Decision Boundaries")
    st.caption(f"###### Using features: {', '.join(top_features)}")

    if "class" in model_type:
        figs = plot_knn_boundaries(
            X, y,
            top_features=top_features,
            params=st.session_state.knn_params,
            class_names=current_result["class_names"]
        )
    else:
        figs = plot_knn_surfaces(
            X, y,
            top_features=top_features,
            params=st.session_state.knn_params
        )

    if figs == 1:
        st.info("No Valid Feature")
        return

    cols = st.columns(6)
    for i, fig in enumerate(figs):
        with cols[i % 6]:
            st.pyplot(fig, use_container_width=True)