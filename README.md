# 🚀 ML Project 2025: Predicting Loan Payback

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-available-blue.svg)

## 📌 Problem Description

### 🎯 Objective
The primary goal of this project is to predict whether a borrower will repay their loan. This is a **binary classification problem** where the model estimates the probability of the target variable `loan_paid_back` (1 for repaid, 0 for not repaid).

### 📖 Context
This project is based on the **[Kaggle Playground Series - Season 5, Episode 11](https://www.kaggle.com/competitions/playground-series-s5e11/overview)** competition. The dataset consists of synthetically generated data that mirrors real-world loan attributes, such as credit scores, income levels, and debt ratios.

### 💼 Business Value
By accurately predicting loan repayment, financial institutions can:
* **Minimize Risk:** Identify high-risk borrowers before approving loans.
* **Optimize Lending Strategies:** Adjust interest rates or loan amounts based on predicted risk.
* **Automate Decision Making:** Streamline the approval process for low-risk candidates.

### 📏 Evaluation Metric
The model performance is evaluated using **ROC AUC (Area Under the Receiver Operating Characteristic Curve)**.
* This metric was chosen because it effectively measures the model's ability to distinguish between the positive class (Repaid) and the negative class (Defaulted), providing a robust performance measure even if the classes are imbalanced.

## 💾 Dataset
The dataset consists of synthetic loan data. It is split into training and testing sets, with the training set containing the target variable `loan_paid_back`.

* **Source:** [Kaggle Playground Series S5E11 Data](https://www.kaggle.com/competitions/playground-series-s5e11/data)
* **Type:** Tabular / Binary Classification
* **Size:** ~594k rows (Train), ~255k rows (Test)

### Features
The dataset includes a mix of numerical and categorical features describing the borrower's financial status:
* **Target:** `loan_paid_back` (Binary: `1` if repaid, `0` if not)
* **Key Numerical Features:** `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`.
* **Key Categorical Features:** `grade_subgrade`, `loan_purpose`, `employment_status`, `education_level`.

### Accessing the Data
**Option 1: Kaggle API**
You can download the data directly using the Kaggle API:
```bash
kaggle competitions download -c playground-series-s5e11
unzip playground-series-s5e11.zip -d data/
```

## 📂 Repository Structure

```text
├── data/                  # CSV files or download scripts  
├── scripts/
│   └── notebook.ipynb     # Data cleaning, EDA, Model selection, Parameter tuning
├── train.py               # Script to train the final model and save to pickle/Bento
└── predict.py             # Flask/BentoML script for serving predictions
├── Dockerfile             # Docker container definition
├── Pipenv & Pipenv.lock   # Dependency management
├── requirements.txt       # Alternative dependency file
└── README.md              # Project documentation
```

```bash
git clone [https://github.com/cncPomper/ML-project-2025.git](https://github.com/cncPomper/ML-project-2025.git)
cd ML-project-2025
```

```bash
conda create -n ml
conda activate ml

# install package
pip3 install --upgrade pip
pip3 install -e .

# install lock
poetry build
poetry install
```

```
hatch run python -c "import kaggle; from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); api.model_list_cli()"
```

With `kaggle API - download data (train | test | submission)`:

```bash
mkdir data && kaggle competitions download -c playground-series-s5e11 && mv playground-series-s5e11.* data/
```


```bash
unzip data/playground-series-s5e11.zip -d data/
```

## Building an image
```bash
cp Dockerfile_base Dockerfile
docker build -t mlzoomcamp -f Dockerfile .
```

```bash
docker run -it --rm -p 8000:8000 mlzoomcamp
```