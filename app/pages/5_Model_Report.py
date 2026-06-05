import streamlit as st
import pandas as pd
from PIL import Image
import os

st.set_page_config(page_title="Model Report", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a3a5c 0%, #2d6a9f 100%); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .section-title { font-size:1.3rem; font-weight:600; color:#1a3a5c; border-bottom:2px solid #2d6a9f; padding-bottom:8px; margin-bottom:16px; }
    .metric-card { background:#ffffff; border-radius:12px; padding:20px; box-shadow:0 2px 12px rgba(0,0,0,0.08); border-left:4px solid #2d6a9f; margin-bottom:16px; }
    .insight-box { background:#e8f4fd; border-left:4px solid #2d6a9f; padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; color:#1a3a5c; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 Model Evaluation Report")

# Model metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model",  "XGBoost")
col2.metric("Accuracy",    "91.50%")
col3.metric("ROC-AUC",     "0.7222")
col4.metric("Threshold",   "0.40")

st.markdown("---")

# Model comparison table
st.markdown('<div class="section-title">📊 All Models Comparison</div>', unsafe_allow_html=True)
comparison = pd.DataFrame({
    'Model'    : ['XGBoost', 'Random Forest', 'Logistic Regression', 'Decision Tree'],
    'Accuracy' : ['91.50%',  '91.14%',        '79.72%',             '76.02%'],
    'ROC-AUC'  : ['0.7222',  '0.6678',        '0.6399',             '0.6396'],
    'Status'   : ['🏆 Best', '2nd',           '3rd',                '4th']
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.markdown("---")

# Display saved charts
image_dir = '../images/'
charts = {
    'ROC & Precision-Recall Curves' : 'roc_pr_curves.png',
    'Confusion Matrix (Tuned)'       : 'confusion_matrix_tuned.png',
    'Feature Importance'             : 'feature_importance_final.png',
    'Threshold Tuning'               : 'threshold_tuning.png',
}

for title, filename in charts.items():
    path = os.path.join(image_dir, filename)
    if os.path.exists(path):
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        st.image(path, use_column_width=True)
    else:
        st.warning(f"{filename} not found — run Notebook 5 first.")

st.markdown('<div class="insight-box">💡 <b>XGBoost</b> outperformed all other models with 91.50% accuracy and 0.7222 ROC-AUC on the Home Credit Default Risk dataset.</div>', unsafe_allow_html=True)
