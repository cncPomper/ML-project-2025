import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

import bentoml

if __name__ == "__main__":

    num_cols = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate"
    ]
    
    cat_cols = [
        "gender",
        "marital_status",
        "education_level",
        "employment_status",
        "loan_purpose",
        "grade_subgrade"
    ]
    
    # You can add binary flags (if any exist)
    bool_cols = []  
    
    # Columns to encode using Target Encoding (categorical + bools)
    cols_to_encode = cat_cols + bool_cols
    
    # ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(cols=cols_to_encode, smoothing=25.0), cols_to_encode),
            ("scaler", StandardScaler(), num_cols)
        ],
        remainder="drop"
    )

    X = pd.read_csv("project/X_train.csv")
    y = pd.read_csv("project/y_train.csv")

    # Ensure all object columns are converted to 'category' type
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    cv_results = {}
    
    # Classification metrics
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC_AUC": "roc_auc"
    }
    
    # # XGBoost Hyperparameters (Tuned for 80/20 Imbalance)
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # Faster training
        'enable_categorical': True  # Enable categorical support
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # oof_preds = np.zeros(len(X))
    scores = []
    
    model = XGBClassifier(**params)
    models = {"XGBoost":model}
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    
        cv_scores = cross_validate(
            pipeline,
            X,
            y,
            cv=kf,
            scoring=scoring,
            n_jobs=-1
        )
    
        cv_results[name] = {metric: np.mean(scores) for metric, scores in cv_scores.items() if "test_" in metric}
    
        print(f"Accuracy:  {cv_results[name]['test_Accuracy']:.4f}")
        print(f"Precision: {cv_results[name]['test_Precision']:.4f}")
        print(f"Recall:    {cv_results[name]['test_Recall']:.4f}")
        print(f"F1-score:  {cv_results[name]['test_F1']:.4f}")
        print(f"ROC-AUC:   {cv_results[name]['test_ROC_AUC']:.4f}")
    
    results_df = pd.DataFrame({
        model: {
            "Accuracy": cv_results[model]["test_Accuracy"],
            "Precision": cv_results[model]["test_Precision"],
            "Recall": cv_results[model]["test_Recall"],
            "F1": cv_results[model]["test_F1"],
            "ROC_AUC": cv_results[model]["test_ROC_AUC"]
        } for model in cv_results.keys()
    }).T.round(4)

    best_model_name = results_df['ROC_AUC'].idxmax()
    best_model = models[best_model_name]

    # Fit the best model before saving
    if not hasattr(best_model, 'fit'):
        raise ValueError(f"The selected model '{best_model_name}' does not support fitting.")

    best_model = model
    best_model_name = "XGBoost"
    print(f"Fitting the best model: {best_model_name}...")
    best_model.fit(X, y)

    # Save the fitted model
    best_model.save_model('project/model.json')
