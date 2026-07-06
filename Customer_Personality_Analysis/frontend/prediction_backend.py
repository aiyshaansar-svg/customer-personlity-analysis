import os
import joblib
import json
import pandas as pd
import streamlit as st

# Resolve paths to backend/model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend", "model"))

PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
KMEANS_PCA_PATH = os.path.join(MODEL_DIR, "kmeans_pca_model.pkl")
GMM_PCA_PATH = os.path.join(MODEL_DIR, "gmm_pca_model.pkl")
BIRCH_PCA_PATH = os.path.join(MODEL_DIR, "birch_pca_model.pkl")
MINIBATCH_PCA_PATH = os.path.join(MODEL_DIR, "minibatch_kmeans_pca_model.pkl")
MAPPINGS_PCA_PATH = os.path.join(MODEL_DIR, "model_mappings_pca.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# Cached resource loading (loads once across all sessions)
@st.cache_resource
def load_ml_assets():
    pca = joblib.load(PCA_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(COLS_PATH)
    
    models = {
        "best": joblib.load(BEST_MODEL_PATH),
        "kmeans": joblib.load(KMEANS_PCA_PATH),
        "gmm": joblib.load(GMM_PCA_PATH),
        "birch": joblib.load(BIRCH_PCA_PATH),
        "minibatch_kmeans": joblib.load(MINIBATCH_PCA_PATH)
    }
    
    with open(MAPPINGS_PCA_PATH, "r") as f:
        model_mappings = json.load(f)
        
    return pca, scaler, feature_columns, models, model_mappings

# Cluster descriptions and recommendations mapping
CLUSTER_METADATA = {
    0: {
        "segment": "Regular Customer",
        "description": "Moderate income and moderate spending customer who typically has children. Open to value-oriented offers and deals.",
        "recommendation": "Target with discount codes, bundle offers, and email newsletters detailing new products."
    },
    1: {
        "segment": "High-Value Customer",
        "description": "High income customer with strong purchasing behavior.",
        "recommendation": "Offer VIP membership, exclusive discounts, premium campaigns, and personalized marketing."
    },
    2: {
        "segment": "Low-Value Customer",
        "description": "Low income customer with low spending and fewer purchases.",
        "recommendation": "Focus on low-cost digital marketing, budget offers, and basic membership retention."
    },
    3: {
        "segment": "Premium Loyal Customer",
        "description": "Very high income customer with the highest spending and purchase volumes, very few children.",
        "recommendation": "Provide dedicated relationship managers, early access to new collections, and luxury brand events."
    }
}

def predict_customer(data, model_name="best"):
    """
    Direct in-memory prediction using cached models.
    Validates, scales, projects to PCA subspace, classifies, and returns segment details.
    """
    # Load cached ML models
    pca, scaler, feature_columns, models, model_mappings = load_ml_assets()
    
    if model_name not in models:
        raise ValueError(f"Invalid model '{model_name}'. Select from {list(models.keys())}.")
        
    selected_model = models[model_name]
    
    # Calculate Age
    birth_year = data.get("Year_Birth", 1970)
    age = 2026 - birth_year # Reference year 2026
    
    # Calculate Days as Customer
    enroll_date = pd.to_datetime(data.get("Dt_Customer", "2013-01-01"))
    ref_date = pd.to_datetime("2015-01-01") # Reference date
    customer_days = (ref_date - enroll_date).days
    
    # Calculate Children
    children = int(data.get("Kidhome", 0)) + int(data.get("Teenhome", 0))
    
    # Calculate Total Spending
    total_spending = (
        float(data.get("MntWines", 0)) + float(data.get("MntFruits", 0)) +
        float(data.get("MntMeatProducts", 0)) + float(data.get("MntFishProducts", 0)) +
        float(data.get("MntSweetProducts", 0)) + float(data.get("MntGoldProds", 0))
    )
    
    # Calculate Total Campaigns
    accepted_campaigns = (
        int(data.get("AcceptedCmp1", 0)) + int(data.get("AcceptedCmp2", 0)) +
        int(data.get("AcceptedCmp3", 0)) + int(data.get("AcceptedCmp4", 0)) +
        int(data.get("AcceptedCmp5", 0))
    )
    
    # Calculate Total Purchases
    total_purchases = (
        int(data.get("NumWebPurchases", 0)) + int(data.get("NumCatalogPurchases", 0)) +
        int(data.get("NumStorePurchases", 0))
    )
    
    # Construct feature row
    clean_record = {
        "Education": int(data.get("Education", 2)),
        "Marital_Status": int(data.get("Marital_Status", 3)),
        "Income": float(data.get("Income", 50000.0)),
        "Recency": int(data.get("Recency", 30)),
        "NumDealsPurchases": int(data.get("NumDealsPurchases", 1)),
        "NumWebVisitsMonth": int(data.get("NumWebVisitsMonth", 5)),
        "Complain": int(data.get("Complain", 0)),
        "Response": int(data.get("Response", 0)),
        "Age": age,
        "Customer_Days": customer_days,
        "Children": children,
        "Total_Spending": total_spending,
        "AcceptedCampaigns": accepted_campaigns,
        "TotalPurchases": total_purchases
    }
    
    # Order columns
    df = pd.DataFrame([clean_record])
    df = df[feature_columns]
    
    # Standard scale
    scaled_features = scaler.transform(df)
    
    # Apply PCA projection
    pca_features = pca.transform(scaled_features)
    
    # Run classification prediction
    raw_cluster = int(selected_model.predict(pca_features)[0])
    
    # Map the arbitrary cluster index to business labels
    model_mapping = model_mappings.get(model_name, {})
    aligned_cluster = model_mapping.get(str(raw_cluster), raw_cluster)
    
    # Fetch persona metadata details
    meta = CLUSTER_METADATA.get(aligned_cluster, {
        "segment": "Unknown Customer",
        "description": "No metadata matches the predicted cluster.",
        "recommendation": "Perform standard customer engagement."
    })
    
    return {
        "cluster": aligned_cluster,
        "raw_cluster": raw_cluster,
        "model_used": model_name,
        "pca_coordinates": [float(c) for c in pca_features[0]],
        "segment": meta["segment"],
        "description": meta["description"],
        "recommendation": meta["recommendation"]
    }

def render_sidebar():
    """
    Renders high-tech headers, status indicators, and footers inside the Streamlit sidebar.
    """
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 25px 0 15px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
        <span style="font-size: 28px; font-weight: 800; letter-spacing: 5px; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">CUSTORA</span>
        <p style="font-size: 11px; color: #64748b; margin-top: 5px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">CPA Console</p>
    </div>
    
    <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 12px; padding: 10px; text-align: center; margin: 10px 10px 30px 10px; box-shadow: 0 0 15px rgba(34, 197, 94, 0.05);">
        <span style="color: #4ade80; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;">⚡ CONSOLIDATED ENGINE: ONLINE</span>
    </div>
    
    <div style="margin: 0 10px; padding: 15px; background-color: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255,255,255,0.03); border-radius: 12px; margin-bottom: 30px;">
        <h5 style="margin-top:0; color:#93c5fd; font-size:13px; font-weight:600; margin-bottom:8px;">📌 Project Objective</h5>
        <p style="font-size:12px; color:#94a3b8; line-height:1.5; margin:0;">Segment new prospects based on demographic, spending, and historical factors to trigger personalized CRM action plans.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sticky footer inside the sidebar container
    st.sidebar.markdown("""
    <div style="margin-top: 40px; padding: 15px; border-top: 1px solid rgba(255,255,255,0.05); text-align: center; margin-left: 10px; margin-right: 10px;">
        <p style="font-size: 11px; color: #64748b; margin: 0; font-weight: 500;">CUSTORA v2.0.0</p>
        <p style="font-size: 10px; color: #475569; margin-top: 4px; line-height: 1.4;">Customer Personality Analysis using Dimensionality Reduction</p>
    </div>
    """, unsafe_allow_html=True)

