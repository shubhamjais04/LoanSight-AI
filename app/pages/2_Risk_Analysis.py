import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Risk Analysis", page_icon="⚠️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a3a5c 0%, #2d6a9f 100%); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .section-title { font-size:1.3rem; font-weight:600; color:#1a3a5c; border-bottom:2px solid #2d6a9f; padding-bottom:8px; margin-bottom:16px; }
    .insight-box { background:#e8f4fd; border-left:4px solid #2d6a9f; padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; color:#1a3a5c; }
</style>
""", unsafe_allow_html=True)

st.markdown("## ⚠️ Risk Analysis")

@st.cache_data
def load_data():
    return pd.read_csv('../data/processed/sample_data.csv')

df = load_data()

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">👤 Default Rate by Gender</div>', unsafe_allow_html=True)
    gender_risk = df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
    gender_risk.columns = ['Gender', 'Default Rate']
    gender_risk['Default Rate %'] = gender_risk['Default Rate'] * 100
    fig = px.bar(gender_risk, x='Gender', y='Default Rate %',
                 color='Default Rate %', color_continuous_scale='Reds')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f8f9fa')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-title">🎂 Default Rate by Age Group</div>', unsafe_allow_html=True)
    df['Age Group'] = pd.cut(df['AGE_YEARS'],
                              bins=[0,25,35,45,55,100],
                              labels=['<25','25-35','35-45','45-55','55+'])
    age_risk = df.groupby('Age Group')['TARGET'].mean().reset_index()
    age_risk['Default Rate %'] = age_risk['TARGET'] * 100
    fig = px.bar(age_risk, x='Age Group', y='Default Rate %',
                 color='Default Rate %', color_continuous_scale='RdYlGn_r')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f8f9fa')
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">💳 Credit Amount vs Default</div>', unsafe_allow_html=True)
    df['Credit Segment'] = pd.cut(df['AMT_CREDIT'],
                                   bins=[0,200000,500000,1000000,float('inf')],
                                   labels=['<200K','200K-500K','500K-1M','>1M'])
    seg_risk = df.groupby('Credit Segment')['TARGET'].mean().reset_index()
    seg_risk['Default Rate %'] = seg_risk['TARGET'] * 100
    fig = px.bar(seg_risk, x='Credit Segment', y='Default Rate %',
                 color='Default Rate %', color_continuous_scale='Reds')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f8f9fa')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-title">💼 Employment Years vs Default</div>', unsafe_allow_html=True)
    fig = px.scatter(df.sample(3000), x='YEARS_EMPLOYED', y='AMT_CREDIT',
                     color='TARGET', color_continuous_scale='RdBu',
                     opacity=0.6, title='')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f8f9fa')
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Younger applicants (&lt;25) and those with very high credit amounts show the highest default rates. EXT_SOURCE scores are the strongest protective factors.</div>', unsafe_allow_html=True)
