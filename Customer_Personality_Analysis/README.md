# CUSTORA — Customer Personality Analysis using K-Means Clustering

A complete, production-ready full-stack machine learning web application built as an internship project to segment customers and generate tailored marketing recommendations. The application uses a pre-trained K-Means clustering model to classify new customer records into one of four key segments.

---

## 📂 Project Structure

The project has been organized with a clean, modular folder layout:

```text
Customer_Personality_Analysis/
│
├── backend/
│   ├── app.py                  # Flask REST API server
│   ├── requirements.txt        # Backend python dependencies
│   ├── utils.py                # Logging and input validation helpers
│   └── model/
│       ├── kmeans_model.pkl     # Pre-trained K-Means model (joblib format)
│       ├── scaler.pkl           # Pre-trained StandardScaler (joblib format)
│       └── feature_columns.pkl  # Exact column names and ordering (joblib format)
│
├── frontend/
│   ├── app.py                  # Streamlit entry point and state initialization
│   ├── requirements.txt        # Frontend python dependencies
│   ├── assets/
│   │   └── style.css           # Premium corporate CSS styling sheet
│   └── pages/
│       ├── Home.py             # Landing page with KPIs and dataset graphs
│       ├── Prediction.py       # Input form for customer profiling
│       └── Results.py          # Custom styled segment cards and recommendations
│
├── customer_cleaned.csv         # Cleaned reference dataset
├── customer_segments.csv        # Segmented reference dataset with Cluster column
├── README.md                   # Setup and deployment manual (this file)
└── .gitignore                  # Standard git exclusions
```

---

## 🚀 Setup & Execution Instructions

Follow these steps to run the application locally.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
You can install dependencies for both the backend and frontend. It is recommended to use a virtual environment.

```bash
# Install backend requirements
pip install -r backend/requirements.txt

# Install frontend requirements
pip install -r frontend/requirements.txt
```

### 3. Run the Flask REST API (Backend)
Navigate to the root directory and run:

```bash
python backend/app.py
```
By default, the server will start on `http://localhost:5000`.

### 4. Run the Streamlit Dashboard (Frontend)
In a separate terminal, navigate to the root directory and run:

```bash
streamlit run frontend/app.py
```
This will automatically launch the web interface in your browser (usually at `http://localhost:8501`).

---

## 🔌 API Usage (Example Request & Response)

### Endpoint
*   **URL**: `/predict`
*   **Method**: `POST`
*   **Headers**: `Content-Type: application/json`

### Example Request JSON Payload
```json
{
  "Education": 2,
  "Marital_Status": 3,
  "Income": 60000.0,
  "Recency": 30,
  "NumDealsPurchases": 1,
  "NumWebVisitsMonth": 4,
  "Complain": 0,
  "Response": 0,
  "Age": 45,
  "Customer_Days": 4500,
  "Children": 1,
  "Total_Spending": 500.0,
  "AcceptedCampaigns": 1,
  "TotalPurchases": 15
}
```

### Example Response JSON
```json
{
  "cluster": 2,
  "segment": "Low-Value Customer",
  "description": "Low income customer with low spending and fewer purchases.",
  "recommendation": "Focus on low-cost digital marketing, budget offers, and basic membership retention."
}
```

---

## 🎯 Target Customer Segments

The K-Means clustering model segments customers into four groups:

| Cluster | Segment Name | Typical Characteristics | Business Strategy |
| :--- | :--- | :--- | :--- |
| **0** | **Regular Customer** | Moderate income, moderate spending, has kids. | Target with discount codes, bundle offers, and newsletters. |
| **1** | **High-Value Customer** | High income, strong purchases, few kids. | VIP memberships, exclusive discounts, premium campaigns. |
| **2** | **Low-Value Customer** | Low income, low spending, has kids. | Focus on low-cost digital marketing, budget retention. |
| **3** | **Premium Loyal Customer** | Highest income & spending, zero complaints. | Dedicated relationship managers, luxury events, early access. |

---

## ☁️ Deployment Guidelines

For deploying this application in production:

### Option A: Docker Compose (Recommended)
You can create docker files to deploy both services easily.

1.  **Backend Dockerfile** (`backend/Dockerfile`):
    ```dockerfile
    FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 5000
    CMD ["python", "app.py"]
    ```
2.  **Frontend Dockerfile** (`frontend/Dockerfile`):
    ```dockerfile
    FROM python:3.10-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 8501
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ```
3.  **Docker Compose** (`docker-compose.yml`):
    ```yaml
    version: '3.8'
    services:
      backend:
        build: ./backend
        ports:
          - "5000:5000"
        environment:
          - PORT=5000
      frontend:
        build: ./frontend
        ports:
          - "8501:8501"
        environment:
          - BACKEND_API_URL=http://backend:5000/predict
        depends_on:
          - backend
    ```

### Option B: Cloud Services (e.g., Render, Heroku)
-   Deploy the **Backend** as a web service. Set the build command to `pip install -r backend/requirements.txt` and start command to `python backend/app.py`.
-   Deploy the **Frontend** as a static/web service. Set environment variable `BACKEND_API_URL` to your live backend endpoint.
