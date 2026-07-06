import streamlit as st
import pandas as pd
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
    <div class="header-title">📊 CUSTORA — Customer Analysis Results</div>
    <div class="header-subtitle">Detailed business classification, traits profiles, and tactical recommendations.</div>
</div>
""", unsafe_allow_html=True)

# Guard against accessing page without prediction
if not st.session_state.get("prediction_result") or not st.session_state.get("input_data"):
    st.warning("⚠️ No prediction result found. Please submit a customer profile first.")
    if st.button("⬅️ Go to Prediction Page", use_container_width=True):
        st.switch_page("pages/Prediction.py")
    st.stop()

# Retrieve stored values
result = st.session_state["prediction_result"]
user_inputs = st.session_state["input_data"]

cluster_num = result["cluster"]
segment_name = result["segment"]
description = result["description"]
recommendation = result["recommendation"]

# Define CSS class mapping based on cluster
class_map = {
    0: "segment-regular",
    1: "segment-high",
    2: "segment-low",
    3: "segment-premium"
}
segment_class = class_map.get(cluster_num, "segment-regular")

# Render Segment Details Card
model_display_name = user_inputs.get("Selected Model", "K-Means Clustering")
raw_cluster = result.get("raw_cluster", cluster_num)

st.markdown(f"""
<div class="segment-card {segment_class}">
    <div class="segment-title">{segment_name}</div>
    <div class="segment-meta"><b>Model:</b> {model_display_name} | <b>Aligned Cluster:</b> {cluster_num} | <b>Raw Model Output:</b> {raw_cluster}</div>
    <div class="segment-desc"><b>Profile Description:</b> {description}</div>
    <div class="segment-recommendation-box">
        <div class="recommendation-title">🎯 Business Recommendation Action Plan</div>
        <div class="recommendation-text">{recommendation}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Comparison with Cluster Averages
st.markdown("### 📊 Personal Profile vs Segment Average")

current_dir = os.path.dirname(os.path.abspath(__file__))
segments_path = os.path.join(current_dir, "..", "..", "customer_segments.csv")

# Clean numeric versions of input data for comparison
try:
    df_seg = pd.read_csv(segments_path)
    df_cluster = df_seg[df_seg["Cluster"] == cluster_num]
    
    # Calculate Cluster Means
    avg_income = df_cluster["Income"].mean()
    avg_spending = df_cluster["Total_Spending"].mean()
    avg_purchases = df_cluster["TotalPurchases"].mean()
    
    # Extract customer values
    cust_income = float(user_inputs["Annual Income"].replace("$", "").replace(",", ""))
    cust_spending = float(user_inputs["Total Spending"].replace("$", "").replace(",", ""))
    cust_purchases = float(user_inputs["Total Purchases"])
    
    # Render KPI Comparison Metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    
    with mcol1:
        diff_inc = cust_income - avg_income
        st.metric(
            label="Annual Income vs. Segment Average",
            value=f"${cust_income:,.2f}",
            delta=f"${diff_inc:,.2f} ({'above' if diff_inc >= 0 else 'below'} avg)",
            delta_color="normal"
        )
        st.markdown(f"<p style='font-size: 13px; color: #64748b;'>Segment Mean: <b>${avg_income:,.2f}</b></p>", unsafe_allow_html=True)
        
    with mcol2:
        diff_spd = cust_spending - avg_spending
        st.metric(
            label="Total Spending vs. Segment Average",
            value=f"${cust_spending:,.2f}",
            delta=f"${diff_spd:,.2f} ({'above' if diff_spd >= 0 else 'below'} avg)",
            delta_color="normal"
        )
        st.markdown(f"<p style='font-size: 13px; color: #64748b;'>Segment Mean: <b>${avg_spending:,.2f}</b></p>", unsafe_allow_html=True)
        
    with mcol3:
        diff_pur = cust_purchases - avg_purchases
        st.metric(
            label="Total Purchases vs. Segment Average",
            value=f"{cust_purchases:.0f} purchases",
            delta=f"{diff_pur:+.1f} ({'above' if diff_pur >= 0 else 'below'} avg)",
            delta_color="normal"
        )
        st.markdown(f"<p style='font-size: 13px; color: #64748b;'>Segment Mean: <b>{avg_purchases:.1f}</b></p>", unsafe_allow_html=True)

except Exception as e:
    st.warning("Comparison to cluster averages is currently unavailable. Ensure customer_segments.csv exists.")

# Render PCA coordinates card
if "pca_coordinates" in result:
    pc1, pc2, pc3 = result["pca_coordinates"]
    st.markdown(f"""
    <div class="custom-card" style="border-left: 5px solid #3b82f6;">
        <h4 style="margin-top:0; color:#93c5fd; font-size:16px;">🧬 Projection in PCA Subspace (3D Coordinates)</h4>
        <p style="font-size:14px; margin-bottom:12px; color:#94a3b8;">To perform clustering in reduced dimensions, the 14 raw profile parameters are projected into a 3D Principal Component subspace:</p>
        <div style="display:flex; justify-content:space-around; background-color:rgba(15, 23, 42, 0.4); padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
            <div><span style="color:#94a3b8;">PC1:</span> <code style="color:#60a5fa; font-weight:600; background:none; border:none; padding:0;">{pc1:.4f}</code></div>
            <div><span style="color:#94a3b8;">PC2:</span> <code style="color:#60a5fa; font-weight:600; background:none; border:none; padding:0;">{pc2:.4f}</code></div>
            <div><span style="color:#94a3b8;">PC3:</span> <code style="color:#60a5fa; font-weight:600; background:none; border:none; padding:0;">{pc3:.4f}</code></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: 0; height: 1px; background: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)

# Input parameters summary table
st.markdown("### 📋 Evaluated Profile Input Parameters")
col_left, col_right = st.columns(2)

# Convert user_inputs dictionary to a clean DataFrame for display
input_items = list(user_inputs.items())
halfway = len(input_items) // 2

with col_left:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    df_inputs_1 = pd.DataFrame(input_items[:halfway], columns=["Parameter", "Submitted Value"])
    st.dataframe(df_inputs_1, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    df_inputs_2 = pd.DataFrame(input_items[halfway:], columns=["Parameter", "Submitted Value"])
    st.dataframe(df_inputs_2, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Add a button to run another prediction
if st.button("🔮 Analyze Another Customer Profile", use_container_width=True):
    st.session_state["prediction_result"] = None
    st.session_state["input_data"] = None
    st.switch_page("pages/Prediction.py")
