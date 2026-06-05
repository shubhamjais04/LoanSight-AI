import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Loan Predictor", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a3a5c 0%, #2d6a9f 100%); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .section-title { font-size:1.3rem; font-weight:600; color:#1a3a5c; border-bottom:2px solid #2d6a9f; padding-bottom:8px; margin-bottom:16px; }
    .result-high   { background:#ffe0e0; border-left:4px solid #e74c3c; padding:20px; border-radius:0 12px 12px 0; }
    .result-low    { background:#d4edda; border-left:4px solid #27ae60; padding:20px; border-radius:0 12px 12px 0; }
    .result-medium { background:#fff3cd; border-left:4px solid #f39c12; padding:20px; border-radius:0 12px 12px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔮 Loan Default Predictor")
st.markdown("Fill in the applicant details below to get a real-time default risk assessment.")

@st.cache_resource
def load_model():
    with open('../models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Applicant Info**")
    age          = st.slider("Age (years)",           18, 70, 35)
    years_emp    = st.slider("Years Employed",         0, 40,  5)
    cnt_children = st.number_input("Number of Children", 0, 10, 0)
    cnt_fam      = st.number_input("Family Members",      1, 10, 2)

with col2:
    st.markdown("**💰 Financial Info**")
    income       = st.number_input("Annual Income (₹)",    10000, 10000000, 150000, step=10000)
    credit       = st.number_input("Credit Amount (₹)",    10000, 5000000,  500000, step=10000)
    annuity      = st.number_input("Annuity Amount (₹)",   1000,  200000,   25000,  step=1000)
    goods_price  = st.number_input("Goods Price (₹)",      10000, 5000000,  450000, step=10000)

with col3:
    st.markdown("**📊 Credit Scores**")
    ext1 = st.slider("External Score 1", 0.0, 1.0, 0.5, 0.01)
    ext2 = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
    ext3 = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)

st.markdown("---")

if st.button("🔍 Predict Default Risk", use_container_width=True):
    # Build input dict with all feature names defaulting to 0
    input_dict = {f: 0 for f in feature_names}

    # Fill known values
    input_dict.update({
         'AGE_YEARS'            : age,
        'YEARS_EMPLOYED'       : years_emp,
        'CNT_CHILDREN'         : cnt_children,
        'CNT_FAM_MEMBERS'      : cnt_fam,
        'AMT_INCOME_TOTAL'     : income,
        'AMT_CREDIT'           : credit,
        'AMT_ANNUITY'          : annuity,
        'AMT_GOODS_PRICE'      : goods_price,
        'EXT_SOURCE_1'         : ext1,
        'EXT_SOURCE_2'         : ext2,
        'EXT_SOURCE_3'         : ext3,
        'CREDIT_INCOME_RATIO'  : credit / income if income > 0 else 0,
        'ANNUITY_INCOME_RATIO' : annuity / income if income > 0 else 0,
        'CREDIT_GOODS_RATIO'   : credit / goods_price if goods_price > 0 else 0,
        'INCOME_PER_PERSON'    : income / cnt_fam if cnt_fam > 0 else 0,
        'EMPLOYMENT_AGE_RATIO' : years_emp / age if age > 0 else 0,
    })

    input_df   = pd.DataFrame([input_dict])[feature_names]
    prob       = model.predict_proba(input_df)[0][1]
    risk_pct   = prob * 100

    col1, col2 = st.columns([1, 1])

   with col1:
        if risk_pct >= 50:
            st.markdown(f"""<div class="result-high">
                <h3 style="color:#c0392b; margin:0">⛔ High Default Risk</h3>
                <h1 style="color:#c0392b; margin:8px 0">{risk_pct:.1f}%</h1>
                <p style="color:#721c24; margin:0">This applicant has a high probability of defaulting. Loan approval not recommended.</p>
            </div>""", unsafe_allow_html=True)
        elif risk_pct >= 25:
            st.markdown(f"""<div class="result-medium">
                <h3 style="color:#856404; margin:0">⚠️ Medium Default Risk</h3>
                <h1 style="color:#856404; margin:8px 0">{risk_pct:.1f}%</h1>
                <p style="color:#533f03; margin:0">Moderate risk — consider additional verification before approving.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-low">
                <h3 style="color:#155724; margin:0">✅ Low Default Risk</h3>
                <h1 style="color:#155724; margin:8px 0">{risk_pct:.1f}%</h1>
                <p style="color:#0b3d21; margin:0">This applicant is likely to repay. Loan approval recommended.</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**📋 Input Summary**")
        summary = {
            'Age'              : f"{age} years",
            'Years Employed'   : f"{years_emp} years",
            'Annual Income'    : f"₹{income:,}",
            'Credit Amount'    : f"₹{credit:,}",
            'Credit/Income'    : f"{credit/income:.2f}x",
            'Ext Score 2'      : f"{ext2:.2f}",
            'Ext Score 3'      : f"{ext3:.2f}",
        }
        st.table(pd.DataFrame(summary.items(), columns=['Feature', 'Value']))
