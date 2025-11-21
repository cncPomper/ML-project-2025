import requests


url_healhcheck = "http://127.0.0.1:8000/health"  # or YOUR_URL
resp = requests.get(url_healhcheck)
print(resp.status_code, resp.json())

url = "http://127.0.0.1:8000/predict"  # or YOUR_URL
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

resp = requests.post(url, json=client)
print(resp.status_code, resp)