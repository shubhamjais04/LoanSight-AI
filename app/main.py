import streamlit as st

st.set_page_config(
    page_title="LoanSight AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global light theme */
    .stApp { background-color: #F8F9FA; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #2d6a9f 100%);
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    
    /* Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #2d6a9f;
        margin-bottom: 16px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3a5c;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #b8d4f0; margin: 0.5rem 0 0; font-size: 1.1rem; }
    
    /* Section titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    /* Insight box */
    .insight-box {
        background: #e8f4fd;
        border-left: 4px solid #2d6a9f;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        color: #1a3a5c;
    }
    
    /* Risk badges */
    .badge-high   { background:#ffe0e0; color:#c0392b; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-medium { background:#fff3cd; color:#856404; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-low    { background:#d4edda; color:#155724; padding:4px 12px; border-radius:20px; font-weight:600; }
    
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏦 LoanSight AI</h1>
    <p>Bank Loan Default Risk Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Total Applications</div>
        <div class="metric-value">307,511</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="metric-card" style="border-left-color:#e74c3c">
        <div class="metric-label">Default Rate</div>
        <div class="metric-value" style="color:#e74c3c">8.07%</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="metric-card" style="border-left-color:#27ae60">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value" style="color:#27ae60">91.50%</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown("""<div class="metric-card" style="border-left-color:#f39c12">
        <div class="metric-label">ROC-AUC Score</div>
        <div class="metric-value" style="color:#f39c12">0.7222</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">📌 About This Project</div>
<div class="insight-box">
    <b>LoanSight AI</b> is an end-to-end machine learning platform built on the Home Credit Default Risk dataset (307,511 loan applications).
    It predicts loan default risk using XGBoost, performs SQL-powered portfolio analytics, and provides an interactive risk assessment tool for banking professionals.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">🗂️ Navigate the App</div>', unsafe_allow_html=True)
    st.markdown("""
    | Page | Description |
    |------|-------------|
    | 📊 Overview | Portfolio summary and key metrics |
    | ⚠️ Risk Analysis | Default patterns and risk segments |
    | 🔮 Loan Predictor | Real-time default risk prediction |
    | 🗄️ SQL Explorer | Live SQL queries on loan data |
    | 📈 Model Report | Full model evaluation and charts |
    """)

with col2:
    st.markdown('<div class="section-title">⚙️ Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    | Component | Technology |
    |-----------|------------|
    | Language | Python 3.12 |
    | ML Model | XGBoost |
    | Database | SQLite |
    | Dashboard | Streamlit |
    | Charts | Plotly |
    | Data | Home Credit (Kaggle) |
    """)

st.markdown("""
<div class="insight-box" style="margin-top:1rem">
    👈 <b>Use the sidebar</b> to navigate between pages.
</div>
""", unsafe_allow_html=True)