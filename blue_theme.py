import streamlit as st

def apply_blue_theme():
    st.markdown(
        """
        <style>
            :root {
                --blue-bg-light: #f7fbff;       /* very light blue background */
                --blue-primary:  #2166af;       /* primary accent color */
                --blue-dark:     #184a7c;       /* hover/dark state */
                --text-color:    #102841;       /* deep navy for headings/text */
                --tag-bg:        #e6f0ff;       /* tag background */
            }

            /* ----- General background and text ----- */
            html, body, .stApp, .block-container, .main, footer {
                background-color: var(--blue-bg-light) !important;
                color: var(--text-color) !important;
            }

            /* ----- Main content block ----- */
            .main > div {
                background-color: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 0 12px rgba(33, 102, 175, 0.08);
            }

            /* ----- Heading colors ----- */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-color) !important;
            }

            /* ----- Button styling ----- */
            .stButton > button,
            .stDownloadButton > button {
                background-color: var(--blue-primary) !important;
                color: white !important;
                font-weight: 600;
                border-radius: 6px;
                transition: background 0.3s ease;
            }
            .stButton > button:hover,
            .stDownloadButton > button:hover {
                background-color: var(--blue-dark) !important;
            }

            /* ----- File uploader styling ----- */
            .stFileUploader label div[data-testid="stFileUploaderDropzone"] {
                background-color: var(--blue-primary) !important;
                color: white !important;
                font-weight: 600;
                border-radius: 6px;
            }
            .stFileUploader label div[data-testid="stFileUploaderDropzone"]:hover {
                background-color: var(--blue-dark) !important;
            }
            .stFileUploader input[type="file"]::-webkit-file-upload-button,
            .stFileUploader input[type="file"]::file-selector-button {
                background: var(--blue-primary) !important;
                color: #fff !important;
                border: none;
                border-radius: 6px;
                padding: 0.4rem 0.9rem;
                cursor: pointer;
            }
            .stFileUploader input[type="file"]::-webkit-file-upload-button:hover,
            .stFileUploader input[type="file"]::file-selector-button:hover {
                background: var(--blue-dark) !important;
            }

            /* ----- Input, multiselect, selectbox border focus/hover ----- */
            input:focus,
            input:hover,
            .stTextInput input:focus,
            .stTextInput input:hover,
            .stSelectbox div[data-baseweb="select"] > div:focus-within,
            .stSelectbox div[data-baseweb="select"]:hover,
            .stMultiSelect div[data-baseweb="select"] > div:focus-within,
            .stMultiSelect div[data-baseweb="select"]:hover {
                border: 2px solid var(--blue-primary) !important;
                box-shadow: 0 0 0 3px rgba(33, 102, 175, 0.2) !important;
            }

            /* ----- Text colors inside dropdowns ----- */
            .stTextInput > div > div > input,
            .stSelectbox > div > div > div > div,
            .stMultiSelect > div > div > div {
                color: var(--text-color) !important;
            }

            /* ----- Selected tags in multiselect dropdowns ----- */
            div[data-baseweb="select"] span {
                background-color: var(--tag-bg) !important;
                color: var(--blue-primary) !important;
                font-weight: 600;
                border-radius: 4px;
                padding: 2px 6px;
            }

            /* ----- Single select visible dropdown area ----- */
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: var(--tag-bg) !important;
                color: var(--blue-primary) !important;
                font-weight: 600;
                border-radius: 6px;
            }

            /* ----- Dashed vertical line for divider ----- */
            .divider-blue {
                border-left: 2px dashed var(--blue-primary);
                height: 250px;
            }

            /* ----- Remove Streamlit footer ----- */
            footer:after {
                content: "";
            }
        </style>
        """,
        unsafe_allow_html=True
    )
