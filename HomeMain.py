import streamlit as st
import pandas as pd
from knn_module import run_knn_classifier
from random_forest_module import run_random_forest_classifier
from blue_theme import apply_blue_theme
import myRegression
from RegressionCleaning import clean_regression_data
from myRegression_display import display_metrics, plot_all_graphs_horizontal, download_predictions
# ---------- Theme & Config ----------
apply_blue_theme()
st.set_page_config(page_title="Algorithm Recommendation Tool", layout="wide")

# ---------- Header ----------
st.markdown("<h1 style='text-align: center;'>⋈ Algorithm Recommendation Tool</h1>", unsafe_allow_html=True)

# ---------- Upload Section ----------
with st.expander("📁 Upload CSV File", expanded=True):
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_columns = df.columns.tolist()

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- Step 1 & 2 with Vertical Divider ----------
    left_col, divider_col, right_col = st.columns([3, 0.2, 3])

    with left_col:
        st.markdown("<h3>Step 1: Select Feature Columns (Mandatory)</h3>", unsafe_allow_html=True)
        selected_features = st.multiselect("Select one or more feature columns", options=all_columns)

    with divider_col:
        # Dashed vertical divider
        st.markdown("""
               <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                   <div style="border-left: 1.5px dashed #295e27; height: 250px;"></div>
               </div>
           """, unsafe_allow_html=True)

    with right_col:
        st.markdown("<h3>Step 2: Select Tagged Column (Optional)</h3>", unsafe_allow_html=True)
        remaining_cols = [col for col in all_columns if col not in selected_features]
        tagged_column = st.selectbox("Select a tagged column (optional)", options=["None"] + remaining_cols)

    # ---------- Step 3 ----------
    st.markdown("<h3>Step 3: Start Algorithm Scanning</h3>", unsafe_allow_html=True)

    btn_cols = st.columns([2.2, 1, 2])
    run_clicked = btn_cols[1].button("🔍 Start Algo Scanning")

    if run_clicked:
        if not selected_features:
            st.warning("Please select at least one feature to proceed.")
        elif tagged_column == "None":
            st.warning("Please select a tagged column to run classification.")
        else:
            st.success("🔄 Running Algorithm...")

            X, y, cleaned_df = clean_regression_data(df, selected_features, tagged_column)
            model = myRegression.MyRegression(cleaned_df, X, y)
            model.train_models()
            model.predict()
            metrics = model.calculate_metrics()

            display_metrics(metrics)
            plot_all_graphs_horizontal(model, cleaned_df, tagged_column)
            download_predictions(model,selected_features, tagged_column)
else:
    st.info("👈 Upload a file to begin.")
