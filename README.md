# 🏦 LoanSight AI — Bank Loan Default Risk Analytics Platform

> **An end-to-end bank loan default risk analytics platform combining SQL-powered portfolio analysis, machine learning-based default prediction, and a 6-page interactive Streamlit dashboard — built on the Home Credit Default Risk dataset.**

---

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

> **🚀 Live App:** [https://loansight-ai.streamlit.app](https://loansight-ai.streamlit.app)

---

## 📌 Project Overview

An end-to-end **SQL-powered analytics and machine learning platform** built on the **Home Credit Default Risk** dataset (307,511 loan applications). LoanSight AI predicts loan default risk using XGBoost, performs SQL-powered portfolio analytics, and provides an interactive risk assessment tool for banking professionals.

---

## 🔢 Dataset

- **Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
- **Total Records:** 307,511 loan applications
- **Target:** Binary — `1` = Default, `0` = Repaid
- **Class Imbalance:** ~8.07% positive class
- **Features Used:** 10 engineered features (EXT_SOURCE_1/2/3, AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, DAYS_BIRTH, DAYS_EMPLOYED, CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO)

---

## 🗄️ SQL Analytics — Core Component

A dedicated SQLite database powers the business intelligence layer of LoanSight AI — answering real banking queries through structured SQL:

- 📊 Default rate analysis by income bracket and loan type
- 💰 Average loan amount segmentation by risk category
- 📈 Portfolio-level exposure and risk distribution queries
- 🔍 High-risk applicant identification using multi-condition filters
- 📋 Repayment behavior patterns across demographic segments
- 🏦 Credit bureau data correlation with default probability

> All SQL queries are structured, documented, and run directly on the SQLite database inside the notebook — demonstrating real-world database analytics skills.

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC | Status |
|-------|----------|---------|--------|
| **XGBoost** | **91.50%** | **0.7222** | 🥇 Best |
| Random Forest | 91.14% | 0.6678 | 2nd |
| Logistic Regression | 79.72% | 0.6399 | 3rd |
| Decision Tree | 76.02% | 0.6396 | 4th |

> XGBoost selected as the final model — best accuracy and ROC-AUC across all models.

---

## ✨ What's Built

- 📥 **Data Pipeline** — Loading and inspecting Home Credit Default Risk dataset (50K sample)
- 🔍 **Deep EDA** — Target distribution, missing value analysis, feature correlations
- ⚙️ **Feature Engineering** — Credit risk features, encoding, imputation, StandardScaler
- 🗄️ **SQL Analytics** — SQLite database with business-grade portfolio queries
- 🤖 **5 Model Comparison** — Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- 📊 **Model Evaluation** — Accuracy, ROC-AUC, Precision, Recall, Confusion Matrix
- 💾 **Model Persistence** — Saved with Joblib, rule-based fallback predictor
- 🖥️ **Streamlit Dashboard** — 6-page interactive credit risk intelligence platform

---

## 🗂️ Project Structure

```
BankLoan/
├── app/
│   ├── .streamlit/
│   │   └── config.toml          # Light theme config
│   └── app.py                   # Main Streamlit app (single file)
├── assets/                      # Logo / icons
├── data/
│   ├── processed/
│   │   └── sample_data.csv      # Processed dataset used by app
│   └── raw/                     # Raw Kaggle CSVs (not pushed — too large)
├── images/                      # EDA & model plots from notebooks
├── models/
│   ├── best_model.pkl           # Trained XGBoost model
│   ├── feature_names.pkl        # Feature names list
│   └── scaler.pkl               # StandardScaler
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_SQL_Analytics.ipynb
│   └── 05_Model_Evaluation.ipynb
├── power_bi/
│   ├── dashboard_preview_loansight
│   ├── loansight_dashboard
├── screenshots/                 # App screenshots
├── .gitignore
├── README.md
└── requirements.txt
```

> **📌 Note on Data Files:** Raw data files (`data/raw/`) and large processed files are excluded from this repository due to GitHub's file size limits. Download the original dataset from [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk). The app runs on `data/processed/sample_data.csv` which is included.

---

## 🖥️ Dashboard Pages

- **🏠 Home** — Project overview, KPI metrics, tech stack, and dataset info
- **📊 Data Overview & EDA** — Target distribution, credit/income/age analysis, default rates by gender, education, and contract type
- **⚠️ Risk Analysis** — Interactive filters, risk level breakdown, default heatmap (income × education), credit-to-income scatter
- **🔮 Loan Default Predictor** — Real-time default probability prediction with gauge chart, risk classification, and recommendation
- **🗄️ SQL Explorer** — Run live SQL queries on the loan portfolio with auto-chart generation
- **📝 Model Report** — Model comparison table, accuracy/ROC-AUC charts, feature importance, radar chart, and hyperparameter details

---

## 📸 App Screenshots

| Home | Overview |
|------|----------|
| ![Home](screenshots/home.png) | ![Overview](screenshots/overview.png) |

| Risk Analysis | Loan Predictor |
|---------------|----------------|
| ![Risk Analysis](screenshots/risk_analysis.png) | ![Loan Predictor](screenshots/loan_predictor.png) |

| SQL Explorer | Model Report |
|--------------|--------------|
| ![SQL Explorer](screenshots/sql_explorer.png) | ![Model Report](screenshots/model_report.png) |

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/shubhamjais04/LoanSight-AI.git
cd LoanSight-AI
```

**2. Create virtual environment**
```bash
py -3.12 -m venv venv
.\venv\Scripts\activate       # Windows
source venv/bin/activate      # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
cd app
streamlit run app.py
```

**Or visit the live demo directly**

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Click%20Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://loansight-ai.streamlit.app)

---

## 👤 Author

**Shubham Jaiswal**  
*Data engineer & ML developer | Combining SQL and machine learning to power smarter credit decisions*

---

## 📬 Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shubhjais04-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/shubhjais04)
[![Gmail](https://img.shields.io/badge/Gmail-shubhjais.in@gmail.com-D14836?style=flat&logo=gmail)](mailto:shubhjais.in@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-shubhamjais04-181717?style=flat&logo=github)](https://github.com/shubhamjais04)
[![Kaggle](https://img.shields.io/badge/Kaggle-shubhamjaiswal04-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/shubhamjaiswal04)
