import streamlit as st
import time
import os

# Set page config
st.set_page_config(
    page_title="CUSTORA - Customer Personality Analysis",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State variables
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "input_data" not in st.session_state:
    st.session_state["input_data"] = None

# Inject custom CSS to center content and style splash page
st.markdown("""
<style>
/* Hide sidebar on splash page */
[data-testid="stSidebar"] {
    display: none;
}

/* Center all main content container */
.main-splash-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 80px 20px 20px 20px;
    margin-top: 10vh;
}

.splash-icon {
    font-size: 90px;
    margin-bottom: 20px;
    animation: bounce 2s infinite;
}

.splash-logo {
    font-size: 80px;
    font-weight: 800;
    letter-spacing: 6px;
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
    animation: fadeIn 1s ease-in-out;
}

.splash-subtitle {
    font-size: 24px;
    color: #475569;
    font-weight: 400;
    max-width: 700px;
    line-height: 1.6;
    animation: fadeIn 1.5s ease-in-out;
}

.loading-spinner {
    margin-top: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #2563eb;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    animation: spin 1s linear infinite;
    display: inline-block;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# Render splash container
st.markdown("""
<div class="main-splash-container">
    <div class="splash-icon">👤</div>
    <div class="splash-logo">CUSTORA</div>
    <div class="splash-subtitle">
        Next-Generation Customer Personality Analysis & Segmentation Platform.
    </div>
    <div class="loading-spinner"></div>
</div>
""", unsafe_allow_html=True)

# Pause to let the user see the centered CUSTORA splash screen
time.sleep(2.0)

# Automatically switch to the Home dashboard
st.switch_page("pages/Home.py")
