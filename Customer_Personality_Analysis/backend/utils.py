import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("backend_utils")

REQUIRED_FEATURES = [
    "Education", "Marital_Status", "Income", "Recency", "NumDealsPurchases",
    "NumWebVisitsMonth", "Complain", "Response", "Age", "Customer_Days",
    "Children", "Total_Spending", "AcceptedCampaigns", "TotalPurchases"
]

def validate_input(data):
    """
    Validates prediction input parameters.
    Returns: (is_valid, error_message, cleaned_dict)
    """
    if not isinstance(data, dict):
        return False, "Input data must be a JSON object.", None

    cleaned = {}
    missing = [feat for feat in REQUIRED_FEATURES if feat not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}", None

    try:
        # 1. Categorical / Integer check
        for int_field in ["Education", "Marital_Status", "Recency", "NumDealsPurchases", 
                          "NumWebVisitsMonth", "Complain", "Response", "Age", 
                          "Customer_Days", "Children", "AcceptedCampaigns", "TotalPurchases"]:
            val = data[int_field]
            try:
                cleaned[int_field] = int(float(val))
            except (ValueError, TypeError):
                return False, f"Field '{int_field}' must be an integer (received: {val}).", None

        # 2. Float check
        for float_field in ["Income", "Total_Spending"]:
            val = data[float_field]
            try:
                cleaned[float_field] = float(val)
            except (ValueError, TypeError):
                return False, f"Field '{float_field}' must be a numeric value (received: {val}).", None

        # 3. Value constraints & bounds checks
        if not (0 <= cleaned["Education"] <= 4):
            return False, f"Education must be between 0 and 4 (received: {cleaned['Education']}).", None

        if not (0 <= cleaned["Marital_Status"] <= 7):
            return False, f"Marital_Status must be between 0 and 7 (received: {cleaned['Marital_Status']}).", None

        if cleaned["Income"] < 0:
            return False, f"Income cannot be negative (received: {cleaned['Income']}).", None

        if not (0 <= cleaned["Recency"] <= 150):
            return False, f"Recency must be between 0 and 150 (received: {cleaned['Recency']}).", None

        if cleaned["NumDealsPurchases"] < 0:
            return False, f"NumDealsPurchases cannot be negative.", None

        if cleaned["NumWebVisitsMonth"] < 0:
            return False, f"NumWebVisitsMonth cannot be negative.", None

        if cleaned["Complain"] not in [0, 1]:
            return False, f"Complain must be 0 or 1.", None

        if cleaned["Response"] not in [0, 1]:
            return False, f"Response must be 0 or 1.", None

        if not (18 <= cleaned["Age"] <= 120):
            return False, f"Age must be between 18 and 120 (received: {cleaned['Age']}).", None

        if cleaned["Customer_Days"] < 0:
            return False, f"Customer_Days cannot be negative.", None

        if cleaned["Children"] < 0:
            return False, f"Children count cannot be negative.", None

        if cleaned["Total_Spending"] < 0:
            return False, f"Total_Spending cannot be negative.", None

        if not (0 <= cleaned["AcceptedCampaigns"] <= 5):
            return False, f"AcceptedCampaigns must be between 0 and 5.", None

        if cleaned["TotalPurchases"] < 0:
            return False, f"TotalPurchases cannot be negative.", None

        return True, None, cleaned

    except Exception as e:
        logger.exception("Error during input validation")
        return False, f"An unexpected validation error occurred: {str(e)}", None
