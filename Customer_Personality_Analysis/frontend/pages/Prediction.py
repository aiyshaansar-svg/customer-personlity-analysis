import streamlit as st
import os

# Helper to load custom CSS
def load_css():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "..", "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Render High-tech Sidebar branding and indicators
import sys
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import prediction_backend
importlib.reload(prediction_backend)
from prediction_backend import render_sidebar
render_sidebar()

# Render Header Bar
st.markdown("""
<div class="header-bar">
    <div class="header-title">🔮 CUSTORA — Predict Customer Segment</div>
    <div class="header-subtitle">Enter customer details below to run the K-Means clustering model and retrieve custom business strategies.</div>
</div>
""", unsafe_allow_html=True)

# Categorical mapping dictionaries
EDUCATION_MAP = {
    "2n Cycle": 0,
    "Basic": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4
}

MARITAL_STATUS_MAP = {
    "Absurd": 0,
    "Alone": 1,
    "Divorced": 2,
    "Married": 3,
    "Single": 4,
    "Together": 5,
    "Widow": 6,
    "YOLO": 7
}


# Model API configuration map (PCA versions)
MODEL_API_MAP = {
    "Best Performing Model (Auto-Selected)": "best",
    "K-Means Clustering (PCA)": "kmeans",
    "Gaussian Mixture Model (PCA GMM)": "gmm",
    "Birch Hierarchical Clustering (PCA)": "birch",
    "MiniBatch K-Means (PCA)": "minibatch_kmeans"
}

# Input Form
with st.form("prediction_form"):
    st.markdown("### 📋 Customer Profile Input Form")
    
    # Model Selection Selectbox
    model_display = st.selectbox(
        "🧠 Select Clustering Model",
        options=list(MODEL_API_MAP.keys()),
        index=0,
        help="Choose the algorithm to perform customer segmentation."
    )
    st.markdown("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #cbd5e1;'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        education_str = st.selectbox(
            "Education Level", 
            options=list(EDUCATION_MAP.keys()), 
            index=2,
            help="Select the customer's highest level of education."
        )
        marital_str = st.selectbox(
            "Marital Status", 
            options=list(MARITAL_STATUS_MAP.keys()), 
            index=3,
            help="Select the customer's marital status."
        )
        age = st.number_input(
            "Age", 
            min_value=18, 
            max_value=120, 
            value=56, 
            step=1,
            help="Age of the customer calculated as 2026 minus birth year."
        )
        children = st.number_input(
            "Number of Children", 
            min_value=0, 
            max_value=10, 
            value=1, 
            step=1,
            help="Total children (Kidhome + Teenhome) in the customer's household."
        )
        customer_days = st.number_input(
            "Customer Days", 
            min_value=0, 
            max_value=15000, 
            value=4555, 
            step=1,
            help="Days since the customer registered (e.g. relative to 2026-01-01)."
        )

    with col2:
        st.markdown("#### 📈 Engagement & History")
        income = st.number_input(
            "Annual Income ($)", 
            min_value=0.0, 
            max_value=1000000.0, 
            value=51381.0, 
            step=500.0,
            help="Annual household income of the customer."
        )
        recency = st.number_input(
            "Recency (Days)", 
            min_value=0, 
            max_value=150, 
            value=49, 
            step=1,
            help="Number of days since customer's last purchase."
        )
        num_deals = st.number_input(
            "Deals Purchases", 
            min_value=0, 
            max_value=30, 
            value=2, 
            step=1,
            help="Number of purchases made using a discount/deal."
        )
        web_visits = st.number_input(
            "Web Visits per Month", 
            min_value=0, 
            max_value=30, 
            value=6, 
            step=1,
            help="Number of visits to the company's website in the last month."
        )
        complain_choice = st.selectbox(
            "Complained in Last 2 Years?", 
            options=["No", "Yes"], 
            index=0,
            help="Has the customer complained in the last 2 years?"
        )

    with col3:
        st.markdown("#### 💰 Spending & Campaigns")
        total_spending = st.number_input(
            "Total Spending ($)", 
            min_value=0.0, 
            max_value=50000.0, 
            value=397.0, 
            step=10.0,
            help="Total amount spent on all product categories in the last 2 years."
        )
        total_purchases = st.number_input(
            "Total Purchases Count", 
            min_value=0, 
            max_value=100, 
            value=12, 
            step=1,
            help="Total number of web, catalog, and store purchases combined."
        )
        accepted_campaigns = st.number_input(
            "Accepted Campaigns", 
            min_value=0, 
            max_value=5, 
            value=0, 
            step=1,
            help="Number of promotional campaigns accepted by the customer (excluding response)."
        )
        response_choice = st.selectbox(
            "Accepted Last Offer (Response)?", 
            options=["No", "Yes"], 
            index=0,
            help="Did the customer accept the offer in the last promotional campaign?"
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(
        "🔮 Run Segment Clustering Prediction",
        use_container_width=True
    )

if submit_button:
    # Compile raw inputs to payload dict with proper numeric encoding
    payload = {
        "model": MODEL_API_MAP[model_display],
        "Education": EDUCATION_MAP[education_str],
        "Marital_Status": MARITAL_STATUS_MAP[marital_str],
        "Income": float(income),
        "Recency": int(recency),
        "NumDealsPurchases": int(num_deals),
        "NumWebVisitsMonth": int(web_visits),
        "Complain": 1 if complain_choice == "Yes" else 0,
        "Response": 1 if response_choice == "Yes" else 0,
        "Age": int(age),
        "Customer_Days": int(customer_days),
        "Children": int(children),
        "Total_Spending": float(total_spending),
        "AcceptedCampaigns": int(accepted_campaigns),
        "TotalPurchases": int(total_purchases)
    }

    # Store user-facing string representations for the Results page display
    user_inputs = {
        "Selected Model": model_display,
        "Education": education_str,
        "Marital Status": marital_str,
        "Annual Income": f"${income:,.2f}",
        "Recency (Days)": recency,
        "Deals Purchases": num_deals,
        "Web Visits/Month": web_visits,
        "Complained?": complain_choice,
        "Response (Last Campaign)?": response_choice,
        "Age": age,
        "Customer Days": customer_days,
        "Children": children,
        "Total Spending": f"${total_spending:,.2f}",
        "Accepted Campaigns": accepted_campaigns,
        "Total Purchases": total_purchases
    }

    with st.spinner("Running segmentation classification locally..."):
        try:
            from prediction_backend import predict_customer
            result = predict_customer(payload, model_name=payload["model"])
            
            # Save results in session state
            st.session_state["prediction_result"] = result
            st.session_state["input_data"] = user_inputs
            
            st.success("Analysis complete! Redirecting to results...")
            st.switch_page("pages/Results.py")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
