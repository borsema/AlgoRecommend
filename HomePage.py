import streamlit as st
import pandas as pd

# --------- Custom Page Config ---------
st.set_page_config(page_title="Algorithm Recommendation Tool", layout="wide")

# --------- Sidebar ---------
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Upload an Excel file and select features.")

# --------- Main Layout ---------
st.markdown("<h1 style='color:#4A90E2;'>📊 Algorithm Recommendation Tool</h1>", unsafe_allow_html=True)
st.write("This app helps you to select best suitable algorithm for your data.")

# --------- Upload Section ---------
with st.expander("📁 Upload Excel File", expanded=True):
    uploaded_file = st.file_uploader("Upload your Excel file here", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.markdown("## ✅ Step 1: Select Features")
    selected_features = []

    # Use Columns to List Checkboxes Nicely
    cols = st.columns(3)
    for i, column in enumerate(df.columns):
        if cols[i % 3].checkbox(f"{column}", key=column):
            selected_features.append(column)

    if selected_features:
        st.success(f"Selected Features: {selected_features}")

        st.markdown("## ✅ Step 2: Choose One Main Feature")
        main_feature = st.selectbox("Select one feature as main focus:", selected_features)

        st.info(f"Main Feature Selected: `{main_feature}`")

        with st.expander("📌 Main Feature Data"):
            st.write(df[[main_feature]])

    else:
        st.warning("Please select at least one column to proceed.")
else:
    st.info("👈 Upload a file to begin.")
