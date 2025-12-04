import streamlit as st
import pandas as pd
from blue_theme import apply_blue_theme
from MyLinearRegression.myRegression_display import DisplayRegression
from auth import show_login, show_logout, is_authenticated

# Import the fragment function
from MyDecisionTree.DecisionTreeDisplay import DisplayDT
from Knn.KnnDisplay import DisplayKNN
from Svm.SvmDisplay import DisplaySVM
from NaiveBayes.NaiveBayesDisplay import DisplayNaiveBayes
from LogisticRegression.LogisticRegressionDisplay import DisplayLogisticRegression
from XGBoost.XGBoostDisplay import DisplayXGBoost


# ---------- Theme & Config ----------
apply_blue_theme()
st.set_page_config(page_title="Algorithm Recommendation Tool", layout="wide")

# ---------- Header ----------
st.markdown("<h1 style='text-align: center;'> ⋈ Algorithm Recommendation Tool ⋈</h1>", unsafe_allow_html=True)

# Show logout button for authenticated users
# if is_authenticated():
#     show_logout()

# ---------- Upload Section ----------
with st.expander("📁 Upload File", expanded=True):
    uploaded_file = st.file_uploader("Upload your CSV or Excel file here", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
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
        # Check authentication first
        # if not is_authenticated():
        #     st.warning("⚠️ Please login to run algorithm scanning")
        #     show_login()
        #     st.stop()
        
        if not selected_features:
            st.warning("Please select at least one feature to proceed.")
        elif tagged_column == "None":
            st.warning("Please select a tagged column to run classification/regression.")
        else:

            with st.spinner("Running Algorithms..."):
                algorithm_scores = {}

                # Linear Regression
                st.subheader("Linear Regression")
                DisplayRegression(df, selected_features, tagged_column)
                if "regression_state" in st.session_state:
                    metrics = st.session_state.regression_state.get("metrics", {})
                    algorithm_scores["Linear Regression"] = metrics.get("linear_r2", 0)

                # Logistic Regression
                st.subheader("Logistic Regression")
                DisplayLogisticRegression(df, selected_features, tagged_column)
                if "lr_result" in st.session_state:
                    metrics = st.session_state.lr_result.get("metrics", {})
                    algorithm_scores["Logistic Regression"] = metrics.get("accuracy", 0)

                # Decision Tree
                st.subheader("Decision Tree")
                DisplayDT(df, selected_features, tagged_column)
                if "dt_result" in st.session_state:
                    metrics = st.session_state.dt_result.get("metrics", {})
                    algorithm_scores["Decision Tree"] = metrics.get("accuracy", metrics.get("r2", 0))

                # KNN Algorithm
                st.subheader("Knn Algorithm")
                DisplayKNN(df, selected_features, tagged_column)
                if "knn_result" in st.session_state:
                    metrics = st.session_state.knn_result.get("metrics", {})
                    algorithm_scores["KNN"] = metrics.get("accuracy", metrics.get("r2", 0))

                # SVM Algorithm
                st.subheader("SVM Algorithm")
                DisplaySVM(df, selected_features, tagged_column)
                if "svm_result" in st.session_state:
                    metrics = st.session_state.svm_result.get("metrics", {})
                    algorithm_scores["SVM"] = metrics.get("accuracy", metrics.get("r2", 0))

                #NaiveBayes Algorithm
                st.subheader("NaiveBayes Algorithm")
                DisplayNaiveBayes(df, selected_features, tagged_column)
                if "nb_state" in st.session_state:
                    metrics = st.session_state.nb_state.get("metrics", {})
                    algorithm_scores["Naive Bayes"] = metrics.get("gaussian_accuracy", 0)

                #XG Boost
                st.subheader("XG Boost")
                DisplayXGBoost(df, selected_features, tagged_column)
                if "xgb_result" in st.session_state:
                    metrics = st.session_state.xgb_result.get("metrics", {})
                    algorithm_scores["XGBoost"] = metrics.get("accuracy", metrics.get("r2", 0))

                # END
                st.success("✅ All algorithm ran successfully. !")

                # Recommendation
                if algorithm_scores:
                    best_algo = max(algorithm_scores, key=algorithm_scores.get)
                    best_score = algorithm_scores[best_algo]
                    
                    st.markdown("---")
                    st.markdown("### 🏆 Algorithm Recommendation")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"#### **Recommended Algorithm: {best_algo}**")
                        st.markdown(f"**Score: {best_score:.4f}**")
                    
                    with col2:
                        st.metric("Best Score", f"{best_score:.4f}")
                    
                    st.markdown("##### 📊 All Algorithm Scores:")
                    scores_df = pd.DataFrame(list(algorithm_scores.items()), columns=["Algorithm", "Score"])
                    scores_df = scores_df.sort_values("Score", ascending=False).reset_index(drop=True)
                    st.dataframe(scores_df, use_container_width=True, hide_index=True)


else:
    st.info("ℹ️ Upload a file to begin.")