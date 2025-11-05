import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
import re
import streamlit.components.v1 as components
from DecisionTreeModel import DecisionTreeModel
from plot_utils import plot_decision_boundaries, plot_regression_surfaces


@st.fragment
def DisplayDT(df: pd.DataFrame, selected_features: list, tagged_column: str):
    """
    Streamlit component for visualizing and tuning a Decision Tree model.

    Args:
        df (pd.DataFrame): Input dataset.
        selected_features (list): List of selected feature column names.
        tagged_column (str): Target column name.
    """
    with st.expander("🌲 Decision Tree", expanded=True):

        # --- Step 1: Train initial model with default settings ---
        model = DecisionTreeModel(df, selected_features, tagged_column)
        st.session_state.dt_result = model.train(max_depth=7, random_state=42).get_results()
        current_result = st.session_state.dt_result

        model_type = str(current_result["model_type"]).lower()
        st.info(f"**Detected Model Type:** {current_result['model_type']}")

        # --- Step 2: Define default hyperparameters ---
        default_params = {
            "criterion": "gini" if "class" in model_type else "squared_error",
            "splitter": "best",
            "max_features": None,
            "max_depth": 5,
            "random_state": 42,
            "max_leaf_nodes": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0.0,
        }

        # if "dt_params" not in st.session_state:
        #     st.session_state.dt_params = default_params.copy()

        if "dt_params" not in st.session_state:
            st.session_state.dt_params = default_params.copy()
        else:
            # Validate criterion
            current_criterion = st.session_state.dt_params.get("criterion", "")
            if ("class" in model_type and current_criterion not in ["gini", "entropy", "log_loss"]) or \
                    ("class" not in model_type and current_criterion not in ["squared_error", "friedman_mse",
                                                                             "absolute_error", "poisson"]):
                st.session_state.dt_params = default_params.copy()
                st.toast("🔄 Reset incompatible Decision Tree parameters for new model type", icon="⚙️")


        params = st.session_state.dt_params

        # --- Step 3: Hyperparameter tuning form ---
        with st.expander(f"⚙️ {current_result['model_type']} - Hyperparameter Tuning", expanded=False):
            with st.form("dt_param_form"):
                st.markdown("### Adjust Hyperparameters")

                col1, col2, col3 = st.columns(3)
                with col1:
                    new_criterion = st.selectbox(
                        "Criterion",
                        ["gini", "entropy", "log_loss"] if "class" in model_type
                        else ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                        index=0,
                    )
                with col2:
                    new_splitter = st.selectbox("Splitter", ["best", "random"], index=0)
                with col3:
                    new_max_features = st.selectbox("Max Features", [None, "sqrt", "log2"], index=0)

                col4, col5, col6 = st.columns(3)
                with col4:
                    new_max_depth = st.number_input("Max Depth", 1, 50, value=5)
                with col5:
                    new_random_state = st.number_input("Random State", 0, 9999, value=42, step=1)
                with col6:
                    new_max_leaf_nodes = st.number_input("Max Leaf Nodes (0 = None)", 0, 1000, value=0, step=1)
                    if new_max_leaf_nodes == 0:
                        new_max_leaf_nodes = None

                col7, col8, col9 = st.columns(3)
                with col7:
                    new_min_samples_split = st.slider("Min Samples Split", 2, 100, 2)
                with col8:
                    new_min_samples_leaf = st.slider("Min Samples Leaf", 1, 50, 1)
                with col9:
                    new_min_impurity_decrease = st.slider("Min Impurity Decrease", 0.0, 1.0, 0.0, step=0.01)

                submit = st.form_submit_button("🔄 Re-run Decision Tree")

                if submit:
                    new_params = {
                        "criterion": new_criterion,
                        "splitter": new_splitter,
                        "max_features": new_max_features,
                        "max_depth": new_max_depth,
                        "random_state": new_random_state,
                        "max_leaf_nodes": new_max_leaf_nodes,
                        "min_samples_split": new_min_samples_split,
                        "min_samples_leaf": new_min_samples_leaf,
                        "min_impurity_decrease": new_min_impurity_decrease,
                    }

                    st.session_state.dt_params = new_params
                    model = DecisionTreeModel(df, selected_features, tagged_column)
                    st.session_state.dt_result = model.train(**new_params).get_results()
                    current_result = st.session_state.dt_result
                    st.toast("✅ Decision Tree retrained successfully!", icon="🚀")

        # --- Step 4: Display Model Metrics ---
        st.markdown("---")
        st.markdown("### 📊 Model Metrics")

        metrics = current_result["metrics"]
        cols = st.columns(len(metrics))

        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                formatted_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
                st.metric(label=key.upper(), value=formatted_value)

        ########################################## Graphical Representation of data #############################################
        # --- Step 5: Tree Graph Visualization ---
        with st.expander("🌳 Decision Tree Graph", expanded=True):
            model = current_result["model"]
            depth = model.get_depth()
            scale_factor = 0.5
            is_classifier = "class" in model_type

            # ✅ Use original categorical names for classification
            class_names = None
            if is_classifier:
                class_names = current_result.get("class_names", None)
                if not class_names and hasattr(model, "classes_"):
                    class_names = [str(c) for c in model.classes_]

            # Generate preview tree (up to depth 4)
            preview_depth = 4 if depth > 4 else None
            preview_dot = tree.export_graphviz(
                model,
                out_file=None,
                feature_names=selected_features,
                class_names=class_names if is_classifier else None,
                filled=True,
                rounded=True,
                special_characters=True,
                proportion=True,
                precision=2,
                max_depth=preview_depth,
                fontname="Helvetica",
            )

            preview_graph = graphviz.Source(preview_dot, format="svg")
            preview_svg = preview_graph.pipe(format="svg").decode("utf-8").strip()
            preview_svg = re.sub(r"<title>.*?</title>", "", preview_svg, flags=re.DOTALL).strip()

            st.markdown("### 🌳 Decision Tree (Preview)")
            st.caption("Showing top 4 levels. Expand below to view the full tree.")
            preview_html = f"""
            <div style="width:100%;height:320px;overflow:auto;background-color:#fafafa;display:flex;justify-content:center;">
                <div style='transform:scale({scale_factor});transform-origin:top center;'>
                    {preview_svg}
                </div>
            </div>
            """
            components.html(preview_html, height=360, scrolling=False)

            # Full tree visualization
            with st.expander("🔍 View Full Tree", expanded=False):
                full_dot = tree.export_graphviz(
                    model,
                    out_file=None,
                    feature_names=selected_features,
                    class_names=class_names if is_classifier else None,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    proportion=True,
                    precision=2,
                    fontname="Helvetica",
                )

                full_graph = graphviz.Source(full_dot, format="svg")
                full_svg_bytes = full_graph.pipe(format="svg")
                full_svg = full_svg_bytes.decode("utf-8").strip()
                full_svg = re.sub(r"<title>.*?</title>", "", full_svg, flags=re.DOTALL).strip()

                full_html = f"""
                <div style="width:100%;height:700px;overflow:auto;border:1px solid #ccc;background-color:#fafafa;">
                    {full_svg}
                </div>
                """
                components.html(full_html, height=720, scrolling=False)

                st.download_button(
                    label="⬇️ Download Full Tree (SVG)",
                    data=full_svg_bytes,
                    file_name="decision_tree_full.svg",
                    mime="image/svg+xml",
                )

        ########################################## 2D Scatter Plot #############################################

            top_features_df = current_result["feature_importance"][["feature", "importance"]].head(4)
            if len(top_features_df) < 2:
                st.info("⚠️ Need at least two valid features to plot decision boundaries.")
            else:
                st.info(f"Top priority features (max 04): {{ {', '.join(f'{f}: {i:.5f}' for f, i in zip(top_features_df.feature, top_features_df.importance))} }}")
            top_features = top_features_df["feature"].tolist()
            # top_features = current_result["feature_importance"]["feature"].head(4).tolist() if len(selected_features) > 4 else selected_features
            if "class" in model_type:
                figs = plot_decision_boundaries(df, df[tagged_column], current_result["class_names"], top_features, **st.session_state.dt_params)
                cols = st.columns(6)
                for i, fig in enumerate(figs):
                    with cols[i % 6]:
                        st.pyplot(fig, use_container_width=True)
            else:
                figs = plot_regression_surfaces(df, df[tagged_column], top_features, **st.session_state.dt_params)
                cols = st.columns(6)
                for i, fig in enumerate(figs):
                    with cols[i % 6]:
                        st.pyplot(fig, use_container_width=True)



'''
#########2D scatter plot decision boundaries ########################################
# --- Step 5: 2D Scatter Graphs with Decision Boundaries ---
        st.markdown("---")
        st.markdown("### 🎨 Decision Tree Boundaries (2D Scatter Plots)")
        st.caption("Each plot shows a feature vs. target relationship with the learned tree boundaries.")

        n_features = len(selected_features)
        cols = st.columns(3)

        for i, feature in enumerate(selected_features):
            col = cols[i % 3]
            with col:
                # Clean and prepare feature-target pair
                data = df[[feature, tagged_column]].replace([np.inf, -np.inf], np.nan).dropna()
                if data.empty:
                    st.warning(f"⚠️ Skipping '{feature}' — no valid data.")
                    continue

                X = data[feature]
                y = data[tagged_column]

                if not pd.api.types.is_numeric_dtype(X):
                    X = X.astype("category").cat.codes
                X = np.array(X).reshape(-1, 1)

                if "class" in model_type and not pd.api.types.is_numeric_dtype(y):
                    y = y.astype("category").cat.codes
                y = np.array(y)

                # Skip constant features
                x_min, x_max = float(X.min()), float(X.max())
                x_range = x_max - x_min
                if x_range == 0:
                    st.warning(f"⚠️ Skipping '{feature}' — constant feature (no variation).")
                    continue

                X_grid = np.linspace(x_min - 0.05 * x_range, x_max + 0.05 * x_range, 400).reshape(-1, 1)

                # Safe parameter copy and model instantiation
                safe_params = params.copy()
                if "class" in model_type:
                    safe_params["criterion"] = safe_params.get("criterion", "gini")
                    temp_model = DecisionTreeClassifier(**safe_params)
                else:
                    safe_params["criterion"] = safe_params.get("criterion", "squared_error")
                    temp_model = DecisionTreeRegressor(**safe_params)

                try:
                    temp_model.fit(X, y)
                    y_pred = temp_model.predict(X_grid)
                except Exception as e:
                    st.warning(f"⚠️ Skipping '{feature}' due to training error: {e}")
                    continue

                # Plot the results
                fig, ax = plt.subplots(figsize=(4, 3))
                if "class" in model_type:
                    ax.scatter(X, y, c=y, cmap="tab10", edgecolor="k", s=40, alpha=0.8)
                    ax.plot(X_grid, y_pred, color="black", linewidth=1.2)
                else:
                    ax.scatter(X, y, color="blue", edgecolor="k", s=40, alpha=0.6)
                    ax.plot(X_grid, y_pred, color="red", linewidth=2)

                ax.set_title(feature, fontsize=10)
                ax.set_xlabel(feature)
                ax.set_ylabel(tagged_column)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)

            if (i + 1) % 3 == 0 and (i + 1) < n_features:
                st.markdown("")
                cols = st.columns(3)
                '''

'''
import streamlit as st
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
import base64
import re
import streamlit.components.v1 as components
from DecisionTreeModel import DecisionTreeModel


@st.fragment
def DisplayDT(df: pd.DataFrame, selected_features: list, tagged_column: str):

    st.subheader("🌲 Decision Tree Model")

    # --- Step 1: Initial model run with defaults ---
    model = DecisionTreeModel(df, selected_features, tagged_column)
    st.session_state.dt_result = model.train(**{"max_depth": 7, "random_state": 42}).get_results()
    current_result = st.session_state.dt_result

    model_type = current_result["model_type"]
    st.info(f"**Detected Model Type:** {model_type}")

    # --- Step 2: Default hyperparameters ---
    default_params = {
        "criterion": "gini" if "Classifier" in model_type else "squared_error",
        "splitter": "best",
        "max_features": None,
        "max_depth": 5,
        "random_state": 42,
        "max_leaf_nodes": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_impurity_decrease": 0.0,
    }

    if "dt_params" not in st.session_state:
        st.session_state.dt_params = default_params.copy()

    params = st.session_state.dt_params

    # --- Step 3: Hyperparameter tuning form ---
    with st.expander(f"⚙️ {model_type} - Hyperparameter Tuning", expanded=True):
        with st.form("dt_param_form"):
            st.markdown("### Adjust Hyperparameters")

            col1, col2, col3 = st.columns(3)
            with col1:
                new_criterion = st.selectbox(
                    "Criterion",
                    ["gini", "entropy", "log_loss"]
                    if "Classifier" in model_type
                    else ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    index=0,
                )

            with col2:
                new_splitter = st.selectbox("Splitter", ["best", "random"], index=0)

            with col3:
                new_max_features = st.selectbox(
                    "Max Features", [None, "sqrt", "log2"], index=0
                )

            col4, col5, col6 = st.columns(3)
            with col4:
                new_max_depth = st.number_input("Max Depth", 1, 50, value=5)
            with col5:
                new_random_state = st.number_input("Random State", 0, 9999, value=42, step=1)
            with col6:
                new_max_leaf_nodes = st.number_input("Max Leaf Nodes (0 = None)", 0, 1000, value=0, step=1)
                if new_max_leaf_nodes == 0:
                    new_max_leaf_nodes = None

            col7, col8, col9 = st.columns(3)
            with col7:
                new_min_samples_split = st.slider("Min Samples Split", 2, 100, 2)
            with col8:
                new_min_samples_leaf = st.slider("Min Samples Leaf", 1, 50, 1)
            with col9:
                new_min_impurity_decrease = st.slider(
                    "Min Impurity Decrease", 0.0, 1.0, 0.0, step=0.01
                )

            submit = st.form_submit_button("🔄 Re-run Decision Tree")

            if submit:
                new_params = {
                    "criterion": new_criterion,
                    "splitter": new_splitter,
                    "max_features": new_max_features,
                    "max_depth": new_max_depth,
                    "random_state": new_random_state,
                    "max_leaf_nodes": new_max_leaf_nodes,
                    "min_samples_split": new_min_samples_split,
                    "min_samples_leaf": new_min_samples_leaf,
                    "min_impurity_decrease": new_min_impurity_decrease,
                }

                st.session_state.dt_params = new_params
                model = DecisionTreeModel(df, selected_features, tagged_column)
                st.session_state.dt_result = model.train(**new_params).get_results()
                current_result = st.session_state.dt_result
                st.toast("✅ Decision Tree retrained successfully!", icon="🚀")

        # --- Step 4: Metrics ---
        st.markdown("---")
        st.markdown("### 📊 Model Metrics")
        metrics = current_result["metrics"]
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(label=key.upper(), value=value)
###########################Tree Graph #############################################

        with st.expander("🌳 Decision Tree Graph", expanded=False):

            # --- Assume you already have: model, selected_features, tagged_column ---
            model = current_result["model"]
            depth = model.get_depth()
            scale_factor = 0.5

            # --- Preview (top 5 levels only) ---
            preview_depth = 4 if depth > 4 else None
            preview_dot = tree.export_graphviz(
                model,
                out_file=None,
                feature_names=selected_features,
                class_names=[tagged_column],
                filled=True,
                rounded=True,
                special_characters=True,
                proportion=True,
                precision=2,
                max_depth=preview_depth,
                fontname="Helvetica"
            )

            preview_graph = graphviz.Source(preview_dot, format="svg")
            preview_svg = preview_graph.pipe(format="svg").decode("utf-8").strip()
            preview_svg = re.sub(r"<title>.*?</title>", "", preview_svg, flags=re.DOTALL).strip()

            # --- Display compact preview ---
            st.markdown("### 🌳 Decision Tree (Preview)")
            st.caption("Showing top 5 levels. Click below to view full tree.")

            preview_html = f"""
            <div style="
                width:100%;
                height:320px;
                overflow:auto;
                border:0px solid #ccc;
                background-color:#fafafa;
                display:flex;
                justify-content:center;   
                line-height:0;            
            ">
                <div style='
                    transform:scale({scale_factor});
                    transform-origin:top center;
                    margin:0;
                    padding:0;
                    line-height:5;    
                '>
                    {preview_svg}
                </div>
            </div>
            """
            st.components.v1.html(preview_html, height=360, scrolling=False)

            # --- Full-depth tree view ---
            with st.expander("🔍 View Full Tree", expanded=False):
                full_dot = tree.export_graphviz(
                    model,
                    out_file=None,
                    feature_names=selected_features,
                    class_names=[tagged_column],
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    proportion=True,
                    precision=2,
                    fontname="Helvetica"
                )

                full_graph = graphviz.Source(full_dot, format="svg")
                full_svg_bytes = full_graph.pipe(format="svg")
                full_svg = full_svg_bytes.decode("utf-8").strip()
                full_svg = re.sub(r"<title>.*?</title>", "", full_svg, flags=re.DOTALL).strip()

                full_html = f"""
                <div style="
                    width:100%;
                    height:700px;
                    overflow:auto;
                    border:1px solid #ccc;
                    background-color:#fafafa;">
                    {full_svg}
                </div>
                """

                st.components.v1.html(full_html, height=720, scrolling=False)

                st.download_button(
                    label="⬇️ Download Full Tree (SVG)",
                    data=full_svg_bytes,
                    file_name="decision_tree_full.svg",
                    mime="image/svg+xml"
                )


'''