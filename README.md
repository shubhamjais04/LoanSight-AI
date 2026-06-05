# 🏦 LoanSight AI — Bank Loan Default Analytics

> An end-to-end machine learning platform for predicting loan default risk, powered by XGBoost, SQL analytics, and an interactive Streamlit dashboard.

🔗 **Live App:** [loansight-amdox.streamlit.app](https://loansight-amdox.streamlit.app)  
📁 **Dataset:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

---

## 📌 Project Overview

LoanSight AI analyzes **307,511 real loan applications** from Home Credit to predict which applicants are likely to default. Built as part of the Amdox Technologies Data Science internship (Month 2), this project demonstrates a complete ML pipeline from raw data to a deployed web application.

---

## 🎯 Key Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **XGBoost** ✅ | **91.50%** | **0.7222** |
| Random Forest | 91.14% | 0.6678 |
| Logistic Regression | 79.72% | 0.6399 |
| Decision Tree | 76.02% | 0.6396 |

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview and key metrics |
| 📊 Overview | Portfolio summary and distributions |
| ⚠️ Risk Analysis | Default patterns by age, gender, credit |
| 🔮 Loan Predictor | Real-time default risk prediction |
| 🗄️ SQL Explorer | Live SQL queries on loan database |
| 📈 Model Report | Full evaluation — ROC, confusion matrix, feature importance |

---

## 🗂️ Project Structure

```
BankLoan/
├── data/
│   ├── raw/                  # Original Kaggle CSVs (gitignored)
│   └── processed/            # Cleaned data, sample CSV, SQLite DB
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_SQL_Analytics.ipynb
│   └── 05_Model_Evaluation.ipynb
├── models/
│   ├── best_model.pkl        # Trained XGBoost model
│   ├── scaler.pkl
│   └── feature_names.pkl
├── app/
│   ├── main.py               # Streamlit home page
│   └── pages/                # Multi-page app
├── images/                   # Saved EDA and evaluation charts
├── screenshots/              # App screenshots
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML Models | XGBoost, Random Forest, Logistic Regression, Decision Tree |
| Imbalance | SMOTE (imbalanced-learn) |
| Database | SQLite + SQLAlchemy |
| Dashboard | Streamlit |
| Charts | Plotly, Seaborn, Matplotlib |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally

```bash
git clone https://github.com/shubhamjais04/BankLoan.git
cd BankLoan
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

> ⚠️ Raw data files are not included in the repo. Download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place in `data/raw/`.

---

## 📊 Sample Visualizations

<!-- Add screenshots here after running the app -->
| Overview | Risk Analysis | Loan Predictor |
|----------|---------------|----------------|
| ![Overview](screenshots/overview.png) | ![Risk](screenshots/risk.png) | ![Predictor](screenshots/predictor.png) |

---

## 👤 Author

**Shubham Jaiswal**   
🔗 [GitHub](https://github.com/shubhamjais04) · [LinkedIn](https://linkedin.com/in/shubhamjaiswal04)