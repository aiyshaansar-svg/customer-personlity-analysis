import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

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
    <div class="header-title">👤 CUSTORA</div>
    <div class="header-subtitle">Advanced Customer Segmentation and Business Analytics using K-Means Clustering</div>
</div>
""", unsafe_allow_html=True)

# Objectives & Workflow (2 Columns)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="custom-card">
        <h3>🎯 Project Objective</h3>
        <p>The goal of this project is to perform <b>Customer Personality Analysis</b> to segment a company's customer base into distinct groups based on demographic details, spending habits, response to campaigns, and purchasing behavior.</p>
        <p>By identifying distinct customer segments, businesses can tailor their product offerings, marketing campaigns, and service strategies to match the unique needs of each profile, maximizing customer value and acquisition efficiency.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="custom-card">
        <h3>🔄 Machine Learning Workflow</h3>
        <p>Our modular full-stack pipeline performs the following steps:</p>
        <ol>
            <li><b>Data Cleaning</b>: Handled missing values, outliers, and duplicates.</li>
            <li><b>Feature Engineering</b>: Created seniority, total children, and total spending metrics.</li>
            <li><b>Scaling</b>: Applied <code>StandardScaler</code> to normalize numerical features.</li>
            <li><b>Clustering</b>: Trained a K-Means model to classify customers into 4 distinct groups.</li>
            <li><b>Prediction REST API</b>: Flask server takes fresh inputs, scales them, and outputs predictions.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Load Dataset for Summary statistics & Visuals
current_dir = os.path.dirname(os.path.abspath(__file__))
segments_path = os.path.join(current_dir, "..", "..", "customer_segments.csv")

try:
    df_seg = pd.read_csv(segments_path)
    
    # Map cluster integers to names
    cluster_names = {
        0: "Regular Customer",
        1: "High-Value Customer",
        2: "Low-Value Customer",
        3: "Premium Loyal Customer"
    }
    df_seg["Segment"] = df_seg["Cluster"].map(cluster_names)

    # Dataset Summary Metrics (KPIs)
    total_customers = len(df_seg)
    avg_income = df_seg["Income"].median()
    avg_spending = df_seg["Total_Spending"].median()
    
    st.markdown("### 📊 Dataset Key Performance Indicators (KPIs)")
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-label">Total Customers Analyzed</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${avg_income:,.2f}</div>
            <div class="kpi-label">Median Annual Income</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${avg_spending:,.2f}</div>
            <div class="kpi-label">Median Customer Spending</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">4</div>
            <div class="kpi-label">Target Segments</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visualizations (2 Columns)
    st.markdown("### 📈 Interactive Segment Profiling")
    vcol1, vcol2 = st.columns([1, 1.2])

    with vcol1:
        # Pie chart for segment representation
        seg_counts = df_seg["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        
        fig_pie = px.pie(
            seg_counts, 
            values="Count", 
            names="Segment",
            title="Customer Segment Breakdown",
            color="Segment",
            color_discrete_map={
                "Regular Customer": "#0284c7",
                "High-Value Customer": "#16a34a",
                "Low-Value Customer": "#ea580c",
                "Premium Loyal Customer": "#9333ea"
            },
            hole=0.4
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Outfit",
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with vcol2:
        # 3D Scatter plot or 2D Scatter plot
        # Let's make a beautiful 2D scatter plot with Income vs Total Spending
        fig_scatter = px.scatter(
            df_seg,
            x="Income",
            y="Total_Spending",
            color="Segment",
            hover_data=["Age", "Children", "TotalPurchases"],
            title="Income vs. Total Spending by Segment",
            labels={"Income": "Annual Income ($)", "Total_Spending": "Total Spending ($)"},
            color_discrete_map={
                "Regular Customer": "#0284c7",
                "High-Value Customer": "#16a34a",
                "Low-Value Customer": "#ea580c",
                "Premium Loyal Customer": "#9333ea"
            },
            opacity=0.7
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Outfit",
            margin=dict(t=50, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 3D Visualizations Section
    st.markdown("### 🗺️ 3D Customer Cluster Projections (Interactive)")
    
    projection_choice = st.radio(
        "Select Projection Space (3D):",
        options=["PCA (Principal Component Analysis)", "t-SNE (t-Distributed Stochastic Neighbor Embedding)", "UMAP (Uniform Manifold Approximation and Projection)"],
        horizontal=True,
        help="Choose the 3D space to project and inspect the customer clusters."
    )
    
    if "PCA" in projection_choice:
        x_col, y_col, z_col = "pca_1", "pca_2", "pca_3"
        plot_title = "3D Customer Clusters in PCA Space"
        x_label, y_label, z_label = "PC1", "PC2", "PC3"
    elif "t-SNE" in projection_choice:
        x_col, y_col, z_col = "tsne_1", "tsne_2", "tsne_3"
        plot_title = "3D Customer Clusters in t-SNE Space"
        x_label, y_label, z_label = "Dimension 1", "Dimension 2", "Dimension 3"
    else:
        x_col, y_col, z_col = "umap_1", "umap_2", "umap_3"
        plot_title = "3D Customer Clusters in UMAP Space"
        x_label, y_label, z_label = "UMAP 1", "UMAP 2", "UMAP 3"
        
    if x_col in df_seg.columns:
        fig_3d = px.scatter_3d(
            df_seg,
            x=x_col,
            y=y_col,
            z=z_col,
            color="Segment",
            hover_data=["Income", "Total_Spending", "Age", "Children", "TotalPurchases"],
            title=plot_title,
            labels={x_col: x_label, y_col: y_label, z_col: z_label},
            color_discrete_map={
                "Regular Customer": "#0284c7",
                "High-Value Customer": "#16a34a",
                "Low-Value Customer": "#ea580c",
                "Premium Loyal Customer": "#9333ea"
            },
            opacity=0.75,
            height=600
        )
        fig_3d.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Outfit",
            margin=dict(t=50, b=0, l=0, r=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)", showbackground=True)
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.warning("3D Projection coordinates not found in customer_segments.csv. Run training script to generate them.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Detailed Cluster Profile Overview
    st.markdown("### 👥 Segment Profiles & Business Action Plans")
    
    # Let's show profiles in columns
    pcol0, pcol1, pcol2, pcol3 = st.columns(4)
    
    with pcol0:
        st.markdown("""
        <div class="custom-card" style="border-top: 5px solid #0284c7; height: 100%;">
            <h4 style="color:#0284c7; margin-bottom:5px;">Regular Customer</h4>
            <span class="segment-meta">Cluster 0</span>
            <p style="font-size:14px; color:#555;"><b>Avg. Income</b>: ~$56,160<br><b>Avg. Spending</b>: ~$689</p>
            <p style="font-size:13px; color:#666;">Moderate-income household with children. They respond well to discounts, sales events, and bundle deals.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pcol1:
        st.markdown("""
        <div class="custom-card" style="border-top: 5px solid #16a34a; height: 100%;">
            <h4 style="color:#16a34a; margin-bottom:5px;">High-Value Customer</h4>
            <span class="segment-meta">Cluster 1</span>
            <p style="font-size:14px; color:#555;"><b>Avg. Income</b>: ~$73,137<br><b>Avg. Spending</b>: ~$1,265</p>
            <p style="font-size:13px; color:#666;">High income, strong purchases, fewer kids. Respond well to premium service levels and customized messaging.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with pcol2:
        st.markdown("""
        <div class="custom-card" style="border-top: 5px solid #ea580c; height: 100%;">
            <h4 style="color:#ea580c; margin-bottom:5px;">Low-Value Customer</h4>
            <span class="segment-meta">Cluster 2</span>
            <p style="font-size:14px; color:#555;"><b>Avg. Income</b>: ~$34,629<br><b>Avg. Spending</b>: ~$94</p>
            <p style="font-size:13px; color:#666;">Budget-conscious buyers. High web visit rates but low conversions. Target with low-cost retention and entry-level items.</p>
        </div>
        """, unsafe_allow_html=True)

    with pcol3:
        st.markdown("""
        <div class="custom-card" style="border-top: 5px solid #9333ea; height: 100%;">
            <h4 style="color:#9333ea; margin-bottom:5px;">Premium Loyal Customer</h4>
            <span class="segment-meta">Cluster 3</span>
            <p style="font-size:14px; color:#555;"><b>Avg. Income</b>: ~$79,988<br><b>Avg. Spending</b>: ~$1,598</p>
            <p style="font-size:13px; color:#666;">Highest spenders with maximum purchase frequency and zero complains. Offer white-glove support and VIP benefits.</p>
        </div>
        """, unsafe_allow_html=True)

    # Load PCA model comparison data
    comp_path = os.path.join(current_dir, "..", "..", "backend", "model", "model_comparison.json")
    if os.path.exists(comp_path):
        st.markdown("<hr style='border: 0; height: 1px; background: #e2e8f0; margin: 40px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🧬 PCA Subspace & Model Clustering Performance Comparison")
        
        with open(comp_path, "r") as f:
            comp_data = json.load(f)
            
        explained_var = comp_data.get("explained_variance", [0, 0, 0])
        cum_var = comp_data.get("cumulative_variance", 0)
        scores = comp_data.get("silhouette_scores", {})
        best_model_name = comp_data.get("best_model", "kmeans")
        
        ccol1, ccol2 = st.columns([1, 1.2])
        
        with ccol1:
            st.markdown(f"""
            <div class="custom-card" style="height: 100%;">
                <h4>📐 PCA Dimensionality Reduction</h4>
                <p>Principal Component Analysis (PCA) was fitted on the 14 scaled customer features to reduce noise and project the data into a 3D subspace.</p>
                <ul>
                    <li><b>PC1 (Principal Component 1):</b> explains {explained_var[0]*100:.2f}% of data variance.</li>
                    <li><b>PC2 (Principal Component 2):</b> explains {explained_var[1]*100:.2f}% of data variance.</li>
                    <li><b>PC3 (Principal Component 3):</b> explains {explained_var[2]*100:.2f}% of data variance.</li>
                    <li><b>Cumulative Variance Preserved:</b> <code style='font-size:14px; font-weight:600; color:#2563eb;'>{cum_var*100:.2f}%</code>.</li>
                </ul>
                <p>This projection allows high-performance clustering while avoiding the curse of dimensionality.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with ccol2:
            # Render Silhouette Scores comparison chart
            model_names_display = {
                "kmeans": "K-Means (PCA)",
                "gmm": "GMM (PCA)",
                "birch": "Birch (PCA)",
                "minibatch_kmeans": "MiniBatch K-Means (PCA)"
            }
            
            score_df = pd.DataFrame({
                "Algorithm": [model_names_display.get(k, k) for k in scores.keys()],
                "Silhouette Score": list(scores.values()),
                "Is Best": ["Best Model" if k == best_model_name else "Other" for k in scores.keys()]
            })
            
            fig_bar = px.bar(
                score_df,
                x="Silhouette Score",
                y="Algorithm",
                color="Is Best",
                orientation="h",
                title="Clustering Performance (Silhouette Score - Higher is Better)",
                color_discrete_map={
                    "Best Model": "#16a34a",
                    "Other": "#3b82f6"
                },
                text="Silhouette Score"
            )
            fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Outfit",
                showlegend=False,
                margin=dict(t=50, b=0, l=0, r=50),
                xaxis=dict(range=[0, 0.45], gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

except Exception as e:
    st.error(f"Error loading segments visualization file: {str(e)}")
    st.warning("Ensure customer_segments.csv is placed in the project root directory.")
