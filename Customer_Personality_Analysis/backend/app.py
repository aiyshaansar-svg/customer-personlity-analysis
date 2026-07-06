import os
import joblib
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from utils import validate_input, logger

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Resolve paths to model pickles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCA_PATH = os.path.join(BASE_DIR, "model", "pca.pkl")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
KMEANS_PCA_PATH = os.path.join(BASE_DIR, "model", "kmeans_pca_model.pkl")
GMM_PCA_PATH = os.path.join(BASE_DIR, "model", "gmm_pca_model.pkl")
BIRCH_PCA_PATH = os.path.join(BASE_DIR, "model", "birch_pca_model.pkl")
MINIBATCH_PCA_PATH = os.path.join(BASE_DIR, "model", "minibatch_kmeans_pca_model.pkl")
MAPPINGS_PCA_PATH = os.path.join(BASE_DIR, "model", "model_mappings_pca.json")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
COLS_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

# Load models and mappings once at startup
try:
    logger.info("Loading pre-trained machine learning artifacts with PCA...")
    pca = joblib.load(PCA_PATH)
    models = {
        "best": joblib.load(BEST_MODEL_PATH),
        "kmeans": joblib.load(KMEANS_PCA_PATH),
        "gmm": joblib.load(GMM_PCA_PATH),
        "birch": joblib.load(BIRCH_PCA_PATH),
        "minibatch_kmeans": joblib.load(MINIBATCH_PCA_PATH)
    }
    
    # Load model alignment mappings
    with open(MAPPINGS_PCA_PATH, "r") as f:
        model_mappings = json.load(f)
        
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(COLS_PATH)
    logger.info("All 5 PCA-based models, PCA transformer, and mappings loaded successfully!")
    logger.info(f"Expected features: {feature_columns}")
except Exception as e:
    logger.error(f"Failed to load machine learning artifacts: {str(e)}")
    raise RuntimeError(f"Startup failed: {str(e)}")

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

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "pca_loaded": pca is not None,
        "scaler_loaded": scaler is not None
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to receive customer features, transform them to PCA space,
    predict using selected model, align cluster index, and return results.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json."}), 400

    data = request.get_json()
    
    # 1. Extract and validate selected model (default to 'best')
    model_name = data.get("model", "best")
    if model_name not in models:
        return jsonify({"error": f"Invalid model '{model_name}'. Select from {list(models.keys())}."}), 400
    
    # 2. Input validation
    is_valid, err_msg, cleaned_data = validate_input(data)
    if not is_valid:
        logger.warning(f"Validation failed: {err_msg}")
        return jsonify({"error": err_msg}), 400

    try:
        # 3. Convert to DataFrame
        df = pd.DataFrame([cleaned_data])

        # 4. Reorder columns to match scaler and model expectations
        df = df[feature_columns]

        # 5. Scale inputs using the StandardScaler
        scaled_features = scaler.transform(df)

        # 6. Apply PCA Dimensionality Reduction
        pca_features = pca.transform(scaled_features)

        # 7. Predict using the selected model
        selected_model = models[model_name]
        
        # Check if the model is GMM or birch/kmeans
        if hasattr(selected_model, "predict"):
            raw_cluster = int(selected_model.predict(pca_features)[0])
        else:
            # GMM also has predict
            raw_cluster = int(selected_model.predict(pca_features)[0])

        # 8. Apply cluster alignment mapping
        model_mapping = model_mappings.get(model_name, {})
        aligned_cluster = model_mapping.get(str(raw_cluster), raw_cluster)

        # 9. Retrieve metadata based on aligned cluster
        metadata = CLUSTER_METADATA.get(aligned_cluster, {
            "segment": "Unknown Customer",
            "description": "No metadata matches the predicted cluster.",
            "recommendation": "Perform standard customer engagement."
        })

        response = {
            "cluster": aligned_cluster,
            "raw_cluster": raw_cluster,
            "model_used": model_name,
            "pca_coordinates": [float(c) for c in pca_features[0]],
            "segment": metadata["segment"],
            "description": metadata["description"],
            "recommendation": metadata["recommendation"]
        }
        
        logger.info(f"PCA Model: {model_name} | Predicted: {response['segment']} (Raw: {raw_cluster} -> Aligned: {aligned_cluster})")
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error occurred during prediction flow")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask application on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
