import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "..", "customer_cleaned.csv")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
COLS_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

# Output paths
PCA_OBJ_PATH = os.path.join(BASE_DIR, "model", "pca.pkl")
KMEANS_PCA_PATH = os.path.join(BASE_DIR, "model", "kmeans_pca_model.pkl")
GMM_PCA_PATH = os.path.join(BASE_DIR, "model", "gmm_pca_model.pkl")
BIRCH_PCA_PATH = os.path.join(BASE_DIR, "model", "birch_pca_model.pkl")
MINIBATCH_PCA_PATH = os.path.join(BASE_DIR, "model", "minibatch_kmeans_pca_model.pkl")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")

MAPPINGS_PCA_PATH = os.path.join(BASE_DIR, "model", "model_mappings_pca.json")
COMPARISON_PATH = os.path.join(BASE_DIR, "model", "model_comparison.json")
SEGMENTS_OUT_PATH = os.path.join(BASE_DIR, "..", "customer_segments.csv")

def main():
    print("Loading datasets, feature columns list, and scaler...")
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
    
    # Prepare features and scale them
    X_raw = df[feature_columns]
    X_scaled = scaler.transform(X_raw)
    
    # 1. Fit PCA with 3 components
    print("Fitting PCA dimensionality reduction (3 components)...")
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Save PCA object
    joblib.dump(pca, PCA_OBJ_PATH)
    print("Saved PCA transformer object to pca.pkl")
    print(f"PCA Cumulative Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 2. Fit t-SNE with 3 components
    print("Fitting t-SNE dimensionality reduction (3 components)...")
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    print("t-SNE coordinates computed.")
    
    # 3. Fit UMAP with 3 components
    print("Fitting UMAP dimensionality reduction (3 components)...")
    umap_reducer = umap.UMAP(n_components=3, random_state=42)
    X_umap = umap_reducer.fit_transform(X_scaled)
    print("UMAP coordinates computed.")
    
    # 4. Train 4 models with 4 clusters in the PCA subspace
    print("\nTraining K-Means on PCA data...")
    kmeans_model = KMeans(n_clusters=4, random_state=42)
    kmeans_model.fit(X_pca)
    kmeans_preds = kmeans_model.predict(X_pca)
    kmeans_sil = float(silhouette_score(X_pca, kmeans_preds))
    print(f"K-Means Silhouette Score: {kmeans_sil:.4f}")
    
    print("Training Gaussian Mixture Model (GMM) on PCA data...")
    gmm_model = GaussianMixture(n_components=4, random_state=42)
    gmm_model.fit(X_pca)
    gmm_preds = gmm_model.predict(X_pca)
    gmm_sil = float(silhouette_score(X_pca, gmm_preds))
    print(f"GMM Silhouette Score: {gmm_sil:.4f}")
    
    print("Training Birch Hierarchical Clustering on PCA data...")
    birch_model = Birch(n_clusters=4)
    birch_model.fit(X_pca)
    birch_preds = birch_model.predict(X_pca)
    birch_sil = float(silhouette_score(X_pca, birch_preds))
    print(f"Birch Silhouette Score: {birch_sil:.4f}")
    
    print("Training MiniBatch K-Means on PCA data...")
    minibatch_model = MiniBatchKMeans(n_clusters=4, random_state=42)
    minibatch_model.fit(X_pca)
    minibatch_preds = minibatch_model.predict(X_pca)
    minibatch_sil = float(silhouette_score(X_pca, minibatch_preds))
    print(f"MiniBatch K-Means Silhouette Score: {minibatch_sil:.4f}")
    
    # 5. Determine the best model
    scores = {
        "kmeans": kmeans_sil,
        "gmm": gmm_sil,
        "birch": birch_sil,
        "minibatch_kmeans": minibatch_sil
    }
    
    best_model_name = max(scores, key=scores.get)
    print(f"\n---> Best Model: {best_model_name} (Silhouette Score: {scores[best_model_name]:.4f})")
    
    # Save PCA models
    joblib.dump(kmeans_model, KMEANS_PCA_PATH)
    joblib.dump(gmm_model, GMM_PCA_PATH)
    joblib.dump(birch_model, BIRCH_PCA_PATH)
    joblib.dump(minibatch_model, MINIBATCH_PCA_PATH)
    
    # Save the best model to the alias path
    best_models_map = {
        "kmeans": kmeans_model,
        "gmm": gmm_model,
        "birch": birch_model,
        "minibatch_kmeans": minibatch_model
    }
    joblib.dump(best_models_map[best_model_name], BEST_MODEL_PATH)
    print(f"Saved the best model to best_model.pkl")
    
    # 6. Compute cluster mapping alignments based on Total_Spending ranks
    mappings = {}
    
    def get_alignment_mapping(preds):
        temp_df = pd.DataFrame({'pred': preds, 'spending': df['Total_Spending']})
        mean_spending = temp_df.groupby('pred')['spending'].mean().sort_values().index.tolist()
        
        mapping = {}
        mapping[int(mean_spending[0])] = 2
        mapping[int(mean_spending[1])] = 0
        mapping[int(mean_spending[2])] = 1
        mapping[int(mean_spending[3])] = 3
        return mapping

    print("\nComputing mappings...")
    mappings['kmeans'] = get_alignment_mapping(kmeans_preds)
    mappings['gmm'] = get_alignment_mapping(gmm_preds)
    mappings['birch'] = get_alignment_mapping(birch_preds)
    mappings['minibatch_kmeans'] = get_alignment_mapping(minibatch_preds)
    mappings['best'] = mappings[best_model_name]
    
    with open(MAPPINGS_PCA_PATH, "w") as f:
        json.dump(mappings, f, indent=4)
    print("Saved mappings JSON to model_mappings_pca.json")
    
    # 7. Write projections and aligned cluster predictions back to customer_segments.csv
    print("\nSaving 3D coordinates and aligned cluster segments to customer_segments.csv...")
    df['Cluster'] = [mappings['kmeans'][int(p)] for p in kmeans_preds]
    
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    df['pca_3'] = X_pca[:, 2]
    
    df['tsne_1'] = X_tsne[:, 0]
    df['tsne_2'] = X_tsne[:, 1]
    df['tsne_3'] = X_tsne[:, 2]
    
    df['umap_1'] = X_umap[:, 0]
    df['umap_2'] = X_umap[:, 1]
    df['umap_3'] = X_umap[:, 2]
    
    df.to_csv(SEGMENTS_OUT_PATH, index=False)
    print(f"Segments dataset successfully saved with t-SNE/UMAP dimensions. Total rows: {len(df)}")
    
    # 8. Save comparison data
    comparison_data = {
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance": float(sum(pca.explained_variance_ratio_)),
        "silhouette_scores": scores,
        "best_model": best_model_name,
        "mappings": mappings
    }
    
    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison_data, f, indent=4)
    print("Saved comparisons metrics to model_comparison.json")
    
    print("\nPCA, t-SNE, and UMAP integration training flow completed successfully!")

if __name__ == "__main__":
    main()
