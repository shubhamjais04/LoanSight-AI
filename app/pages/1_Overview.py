import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a3a5c 0%, #2d6a9f 100%); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .section-title { font-size:1.3rem; font-weight:600; color:#1a3a5c; border-bottom:2px solid #2d6a9f; padding-bottom:8px; margin-bottom:16px; }
    .insight-box { background:#e8f4fd; border-left:4px solid #2d6a9f; padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; color:#1a3a5c; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Portfolio Overview")

@st.cache_data
def load_data():
    return pd.read_csv('../data/processed/sample_data.csv')

df = load_data()

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",     f"{len(df):,}")
col2.metric("Default Rate",      f"{df['TARGET'].mean()*100:.2f}%")
col3.metric("Avg Credit Amount", f"₹{df['AMT_CREDIT'].mean():,.0f}")
col4.metric("Avg Income",        f"₹{df['AMT_INCOME_TOTAL'].mean():,.0f}")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">🎯 Target Distribution</div>', unsafe_allow_html=True)
    target_counts = df['TARGET'].value_counts().reset_index()
    target_counts.columns = ['Status', 'Count']
    target_counts['Status'] = target_counts['Status'].map({0: 'Repaid', 1: 'Defaulted'})
    fig = px.pie(target_counts, values='Count', names='Status',
                 color_discrete_sequence=['#2d6a9f', '#e74c3c'],
                 hole=0.45)
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-title">💰 Credit Amount Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x='AMT_CREDIT', nbins=50, color_discrete_sequence=['#2d6a9f'])
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f8f9fa',
                      xaxis_title='Credit Amount', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-title">📋 Sample Data Preview</div>', unsafe_allow_html=True)
st.dataframe(df[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                  'AGE_YEARS','YEARS_EMPLOYED','TARGET']].head(20),
             use_container_width=True)