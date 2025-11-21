import os
import io
import json
import logging
from typing import Any, Optional, Tuple, List
# from flask import Flask, request, jsonify
import joblib
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

from project.models.PredictionInput import PredictionInput

from sklearn.pipeline import Pipeline
from project.scripts.preprocessing import create_advanced_features

try:
    import xgboost as xgb
except Exception:
    xgb = None

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict-app")

MODEL_PATH = os.path.join("project", "fitted_pipeline.joblib")


# app = Flask(__name__)
app = FastAPI(title="ML pipeline service")

def load_model(path: str) -> Tuple[Any, str]:
    """Load a model from a given path and return the model and its type.
    Supports xgboost JSON (.json), joblib (.joblib) and pickle (.pkl/.pickle).
    """
    if path.endswith(".json"):
        if xgb is None:
            raise ImportError("xgboost is required to load JSON models. Install 'xgboost'.")
        booster = xgb.Booster()
        booster.load_model(path)  # loads JSON-format model file
        return booster, "xgboost"

    if path.endswith(".joblib"):
        model = joblib.load(path)
        model_type = "xgboost" if (xgb is not None and isinstance(model, xgb.Booster)) else "sklearn"
        return model, model_type

    # if path.endswith(".pkl") or path.endswith(".pickle"):
    #     with open(path, "rb") as f:
    #         model = pickle.load(f)
    #     model_type = "xgboost" if (xgb is not None and isinstance(model, xgb.Booster)) else "sklearn"
    #     return model, model_type

    # raise ValueError("Unsupported model file format. Supported: .json, .joblib, .pkl, .pickle")

def build_preprocessor(
    num_cols: Optional[list] = None,
    cat_cols: Optional[list] = None,
    smoothing: float = 25.0,
):
    """
    Build and return a ColumnTransformer preprocessor plus the column lists.
    If num_cols or cat_cols are not provided, defaults from the original selection are used.
    """

    if num_cols is None:
        num_cols = [
            "annual_income",
            "debt_to_income_ratio",
            "credit_score",
            "loan_amount",
            "interest_rate"
        ]

    if cat_cols is None:
        cat_cols = [
            "gender",
            "marital_status",
            "education_level",
            "employment_status",
            "loan_purpose",
            "grade_subgrade"
        ]

    cols_to_encode = cat_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(cols=cols_to_encode, smoothing=smoothing), cols_to_encode),
            ("scaler", StandardScaler(), num_cols)
        ],
        remainder="drop"
    )

    return preprocessor

def prepare_input_data(request_data: dict) -> pd.DataFrame:
    """Convert incoming JSON data to a pandas DataFrame."""
    # if "instances" in request_data:
    #     data = request_data["instances"]
    # else:
    #     data = request_data
    if hasattr(request_data, 'dict'):
        request_data = request_data.dict()

    # The fix: wrap every single value in a list of length one
    data_for_df = {k: [v] for k, v in request_data.items()}
    
    # Create the DataFrame. No need for index=[0] now.
    input_df = pd.DataFrame(data_for_df)
    return input_df

@app.post("/predict")
def predict(input_data: PredictionInput) -> List[float]:
    """Make predictions using the loaded model and preprocessor."""
    input_data = prepare_input_data(input_data)
    input_data = create_advanced_features(input_data)

    model = joblib.load(MODEL_PATH)

    preds = model.predict_proba(input_data)[:, 1]

    return {"y_pred":preds.tolist()}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    info = {"status": "ok"}
    print("Checking model at path:", MODEL_PATH)
    # Check for a common model file and report status
    if os.path.exists(MODEL_PATH):
        try:
            model_obj, model_type = load_model(MODEL_PATH)
            info["model"] = {
                "model_loaded": True,
                "model_path": MODEL_PATH,
                "model_type": model_type,
            }
        except Exception as e:
            # Model file exists but failed to load
            info["model"] = {
                "model_loaded": False,
                "model_path": MODEL_PATH,
                "error": str(e),
            }
    else:
        info["model"] = {"model_loaded": False, "model_path": None}

    return info
if __name__ == "__main__":
    print(health_check())

    client = {
        "id": 593994,
        "annual_income": 28781.05,
        "debt_to_income_ratio": 0.049,
        "credit_score": 626,
        "loan_amount": 11461.42,
        "interest_rate": 14.73,
        "gender": "Female",
        "marital_status": "Single",
        "education_level": "High School",
        "employment_status": "Employed",
        "loan_purpose": "Other",
        "grade_subgrade": "D5"
    }

    preds = predict(client)
    print("Prediction:", preds)