import os
import pandas as pd
import joblib

# ------------------------------
# Paths to artifacts
# ------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Ensure artifacts folder exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ------------------------------
# Load models and scalers
# ------------------------------
try:
    model_young = joblib.load(os.path.join(ARTIFACTS_DIR, "model_young.joblib"))
    model_rest = joblib.load(os.path.join(ARTIFACTS_DIR, "model_rest.joblib"))
    scaler_young = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_young.joblib"))
    scaler_rest = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_rest.joblib"))
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"Missing module required to load models/scalers: {e.name}")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model/scaler file not found: {e.filename}")

# ------------------------------
# Helper functions
# ------------------------------

def calculate_normalized_risk(medical_history: str) -> float:
    """Calculate a normalized risk score from medical history"""
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0,
    }

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)

    max_score = 14  # 8 (heart) + 6 (diabetes/high BP)
    min_score = 0

    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score

def handle_scaling(age: int, df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric features based on age group"""
    scaler_object = scaler_young if age <= 25 else scaler_rest
    cols_to_scale = scaler_object["cols_to_scale"]
    scaler = scaler_object["scaler"]

    # Ensure no None values
    df["income_level"] = 0
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop("income_level", axis="columns", inplace=True)

    return df

def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """Convert raw input dictionary into model-ready DataFrame"""
    expected_columns = [
        "age",
        "number_of_dependants",
        "income_lakhs",
        "insurance_plan",
        "genetical_risk",
        "normalized_risk_score",
        "gender_Male",
        "region_Northwest",
        "region_Southeast",
        "region_Southwest",
        "marital_status_Unmarried",
        "bmi_category_Obesity",
        "bmi_category_Overweight",
        "bmi_category_Underweight",
        "smoking_status_Occasional",
        "smoking_status_Regular",
        "employment_status_Salaried",
        "employment_status_Self-Employed",
    ]

    insurance_plan_encoding = {"Bronze": 1, "Silver": 2, "Gold": 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Map input values to columns
    for key, value in input_dict.items():
        if key == "Gender" and value == "Male":
            df["gender_Male"] = 1
        elif key == "Region":
            if value in ["Northwest", "Southeast", "Southwest"]:
                df[f"region_{value}"] = 1
        elif key == "Marital Status" and value == "Unmarried":
            df["marital_status_Unmarried"] = 1
        elif key == "BMI Category" and value in ["Obesity", "Overweight", "Underweight"]:
            df[f"bmi_category_{value}"] = 1
        elif key == "Smoking Status" and value in ["Occasional", "Regular"]:
            df[f"smoking_status_{value}"] = 1
        elif key == "Employment Status" and value in ["Salaried", "Self-Employed"]:
            df[f"employment_status_{value}"] = 1
        elif key == "Insurance Plan":
            df["insurance_plan"] = insurance_plan_encoding.get(value, 1)
        elif key == "Age":
            df["age"] = value
        elif key == "Number of Dependants":
            df["number_of_dependants"] = value
        elif key == "Income in Lakhs":
            df["income_lakhs"] = value
        elif key == "Genetical Risk":
            df["genetical_risk"] = value

    df["normalized_risk_score"] = calculate_normalized_risk(input_dict["Medical History"])
    df = handle_scaling(input_dict["Age"], df)

    return df

def predict(input_dict: dict) -> int:
    """Predict health insurance cost based on input dictionary"""
    input_df = preprocess_input(input_dict)
    model = model_young if input_dict["Age"] <= 25 else model_rest
    prediction = model.predict(input_df)
    return int(prediction[0])
