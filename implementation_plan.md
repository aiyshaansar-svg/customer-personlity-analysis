# Implementation Plan - Customer Personality Analysis Web Application

This implementation plan details the creation of a production-ready, premium full-stack machine learning web application for Customer Personality Analysis. We will use the pre-trained KMeans model, standard scaler, and feature columns definition.

## User Review Required

> [!IMPORTANT]
> The pre-trained model and scaler files (`kmeans_model.pkl` and `scaler.pkl`) were serialized using `joblib` rather than standard `pickle`. We will load them using `joblib` in the Flask backend to avoid deserialization errors.

> [!NOTE]
> The label encoder mappings for the categorical fields are as follows:
> - **Education**: 2n Cycle (0), Basic (1), Graduation (2), Master (3), PhD (4)
> - **Marital Status**: Absurd (0), Alone (1), Divorced (2), Married (3), Single (4), Together (5), Widow (6), YOLO (7)

---

## Proposed Changes

We will create a root project directory named `Customer_Personality_Analysis` with a clean, modular structure.

### Root Files

#### [NEW] [README.md](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/README.md)
Documentation of the project structure, setup instructions, example API request/response payload, and deployment instructions.

#### [NEW] [.gitignore](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/.gitignore)
Standard ignore patterns for python, virtual environments, cache folders, and data files (keeping model pickles excluded from git tracking if necessary).

---

### Backend Component

The backend will be built with Flask. It will load the models once at startup and expose a single prediction endpoint.

#### [NEW] [backend/requirements.txt](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/backend/requirements.txt)
Dependency list including `Flask`, `scikit-learn`, `pandas`, `joblib`, and `flask-cors`.

#### [NEW] [backend/utils.py](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/backend/utils.py)
Utility functions for logging, custom input validation, and formatting prediction responses.

#### [NEW] [backend/app.py](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/backend/app.py)
The Flask server entrypoint:
- Loads the models once.
- Validates the incoming JSON request parameters.
- Converts JSON input to a pandas DataFrame and reorders columns according to `feature_columns.pkl`.
- Transforms features with the standard scaler.
- Performs prediction using the KMeans model.
- Maps the predicted cluster (0-3) to its description and recommendations, and returns it as JSON.

---

### Frontend Component

The frontend will be built with Streamlit and styled with a custom CSS template. We will use a multi-page app structure.

#### [NEW] [frontend/requirements.txt](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/frontend/requirements.txt)
Streamlit dashboard dependencies: `streamlit`, `pandas`, `plotly`, and `requests`.

#### [NEW] [frontend/assets/style.css](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/frontend/assets/style.css)
Custom corporate CSS styles defining rounded cards, soft shadows, custom typography (e.g., Roboto/Inter), and hover effects.

#### [NEW] [frontend/app.py](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/frontend/app.py)
Streamlit application entry point:
- Sets up standard sidebar and page navigation.
- Injects global custom CSS stylesheet.
- Initializes session state variables for input features and prediction outputs to facilitate sharing data across pages.

#### [NEW] [frontend/pages/Home.py](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/frontend/pages/Home.py)
Displays:
- Project title, objectives, and workflow description.
- Interactive Plotly figures showing dataset distributions and a 3D scatter plot of customer segments in `customer_segments.csv`.
- KPI metrics and cards explaining each customer cluster.

#### [NEW] [frontend/pages/Prediction.py](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/frontend/pages/Prediction.py)
Constructs a well-organized input form with logical groups (Demographics, Spending & Purchases, and Campaigns/Visits). It provides default values based on the dataset medians. When submitted, it:
1. Calls the Flask REST API.
2. Saves the API response in session state.
3. Programmatically routes the user to the Results page.

#### [NEW] [frontend/pages/Results.py](file:///c:/Users/user/OneDrive/Desktop/internship/Customer_Personality_Analysis/frontend/pages/Results.py)
Renders a custom styled segment card with color coding based on the customer segment:
- **Cluster 0 (Regular)**: Soft Cyan/Blue
- **Cluster 1 (High-Value)**: Soft Green
- **Cluster 2 (Low-Value)**: Soft Orange
- **Cluster 3 (Premium Loyal)**: Premium Purple/Gold
It also shows the input parameters and compares them against the average characteristics of the predicted cluster.

---

## Verification Plan

### Automated Tests
We will verify API connectivity and correctness by:
1. Running the Flask server and testing it using a temporary python script that makes requests with realistic inputs.
2. Checking page loading and routing in Streamlit.

### Manual Verification
1. Launching the backend and frontend.
2. Filling out the form on the Prediction page, clicking Predict, and checking the redirection to the Results page.
3. Reviewing the visuals and styling on the Home page and ensuring interactive Plotly charts load correctly.
