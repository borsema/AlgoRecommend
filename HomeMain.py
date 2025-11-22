import streamlit as st
import pandas as pd
from blue_theme import apply_blue_theme
from MyLinearRegression.myRegression_display import DisplayRegression

# Import the fragment function
from MyDecisionTree.DecisionTreeDisplay import DisplayDT
from Knn.KnnDisplay import DisplayKNN


# ---------- Theme & Config ----------
apply_blue_theme()
st.set_page_config(page_title="Algorithm Recommendation Tool", layout="wide")

# ---------- Header ----------
st.markdown("<h1 style='text-align: center;'> ⋈ Algorithm Recommendation Tool ⋈</h1>", unsafe_allow_html=True)

# ---------- Upload Section ----------
with st.expander("📁 Upload CSV File", expanded=True):
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    all_columns = df.columns.tolist()

    st.subheader("👀 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- Step 1 & 2 with Vertical Divider----------
    left_col, divider_col, right_col = st.columns([3, 0.2, 3])

    with left_col:
        st.markdown("<h3>Step 1: Select Feature columns</h3>", unsafe_allow_html=True)
        selected_features = st.multiselect("Select one or more feature columns (Mandatory)", options=all_columns)

    with divider_col:
        st.markdown("""
               <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                   <div style="border-left: 1.5px dashed #295e27; height: 250px;"></div>
               </div>
           """, unsafe_allow_html=True)

    with right_col:
        st.markdown("<h3>Step 2: Select Tagged column</h3>", unsafe_allow_html=True)
        remaining_cols = [col for col in all_columns if col not in selected_features]
        tagged_column = st.selectbox("Select a tagged column (Mandatory)", options=["None"] + remaining_cols)

    # ---------- Step 3 ----------
    st.markdown("<h3 style='text-align: center;'>Step 3: Start Algorithm Scanning</h3>", unsafe_allow_html=True)

    btn_cols = st.columns([2.2, 1, 2])
    run_clicked = btn_cols[1].button("▶️ Start Algo Scanning", key="main_run_button")

    # --- MAIN LOGIC EXECUTION ---
    if run_clicked:
        if not selected_features:
            st.warning("Please select at least one feature to proceed.")
        elif tagged_column == "None":
            st.warning("Please select a tagged column to run classification/regression.")
        else:

            with st.spinner("Running Algorithms..."):

                # Linear Regression
                st.subheader("Linear Regression")
                DisplayRegression(df, selected_features, tagged_column)

                # Decision Tree
                st.subheader("Decision Tree")
                DisplayDT(df, selected_features, tagged_column)

                # KNN Algorithm
                st.subheader("Knn Algorithm")
                DisplayKNN(df, selected_features, tagged_column)


                # END
                st.success("✅ All algorithm ran successfully. !")


else:
    st.info("ℹ️ Upload a file to begin.")