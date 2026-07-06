import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "..", "customer_cleaned.csv")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
COLS_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

# Output model paths
KMEANS_PATH = os.path.join(BASE_DIR, "model", "kmeans_model.pkl")
GMM_PATH = os.path.join(BASE_DIR, "model", "gmm_model.pkl")
BIRCH_PATH = os.path.join(BASE_DIR, "model", "birch_model.pkl")
MINIBATCH_PATH = os.path.join(BASE_DIR, "model", "minibatch_kmeans_model.pkl")
MAPPINGS_PATH = os.path.join(BASE_DIR, "model", "model_mappings.json")

def main():
    print("Loading datasets and scaler...")
    # Load dataset
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # Engineer the missing columns for the training features
    df['AcceptedCampaigns'] = (
        df['AcceptedCmp1'] + df['AcceptedCmp2'] + 
        df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5']
    )
    df['TotalPurchases'] = (
        df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
    )
    
    # Load feature columns and scaler
    feature_columns = joblib.load(COLS_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Prepare features
    X_raw = df[feature_columns]
    X_scaled = scaler.transform(X_raw)
    
    # Train 4 models with 4 clusters on scaled data
    print("Training K-Means Model...")
    kmeans_model = KMeans(n_clusters=4, random_state=42)
    kmeans_model.fit(X_scaled)
    
    print("Training Gaussian Mixture Model (GMM)...")
    gmm_model = GaussianMixture(n_components=4, random_state=42)
    gmm_model.fit(X_scaled)
    
    print("Training Birch Hierarchical Clustering...")
    birch_model = Birch(n_clusters=4)
    birch_model.fit(X_scaled)
    
    print("Training MiniBatch K-Means...")
    minibatch_model = MiniBatchKMeans(n_clusters=4, random_state=42)
    minibatch_model.fit(X_scaled)
    
    # Save the models
    print("Saving models...")
    joblib.dump(kmeans_model, KMEANS_PATH)
    joblib.dump(gmm_model, GMM_PATH)
    joblib.dump(birch_model, BIRCH_PATH)
    joblib.dump(minibatch_model, MINIBATCH_PATH)
    
    # Build cluster mapping alignment based on Total_Spending
    # Target Aligned Clusters:
    # 0 -> Regular Customer (Moderate spending)
    # 1 -> High-Value Customer (High spending)
    # 2 -> Low-Value Customer (Low spending)
    # 3 -> Premium Loyal Customer (Highest spending)
    
    mappings = {}
    
    def get_alignment_mapping(preds):
        temp_df = pd.DataFrame({'pred': preds, 'spending': df['Total_Spending']})
        mean_spending = temp_df.groupby('pred')['spending'].mean().sort_values().index.tolist()
        
        # mean_spending is ordered [lowest, second, third, highest]
        mapping = {}
        # Lowest spending -> 2
        mapping[int(mean_spending[0])] = 2
        # Second -> 0
        mapping[int(mean_spending[1])] = 0
        # Third -> 1
        mapping[int(mean_spending[2])] = 1
        # Highest -> 3
        mapping[int(mean_spending[3])] = 3
        return mapping

    print("\nComputing mappings...")
    kmeans_preds = kmeans_model.predict(X_scaled)
    mappings['kmeans'] = get_alignment_mapping(kmeans_preds)
    print("K-Means Mappings:", mappings['kmeans'])
    
    gmm_preds = gmm_model.predict(X_scaled)
    mappings['gmm'] = get_alignment_mapping(gmm_preds)
    print("GMM Mappings:", mappings['gmm'])
    
    birch_preds = birch_model.predict(X_scaled)
    mappings['birch'] = get_alignment_mapping(birch_preds)
    print("Birch Mappings:", mappings['birch'])
    
    minibatch_preds = minibatch_model.predict(X_scaled)
    mappings['minibatch_kmeans'] = get_alignment_mapping(minibatch_preds)
    print("MiniBatch K-Means Mappings:", mappings['minibatch_kmeans'])
    
    # Save mappings JSON
    with open(MAPPINGS_PATH, "w") as f:
        json.dump(mappings, f, indent=4)
        
    print("\nAll 4 models trained on scaled features and saved successfully with mappings!")

if __name__ == "__main__":
    main()
