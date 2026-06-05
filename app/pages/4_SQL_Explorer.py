import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

st.set_page_config(page_title="SQL Explorer", page_icon="🗄️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a3a5c 0%, #2d6a9f 100%); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .section-title { font-size:1.3rem; font-weight:600; color:#1a3a5c; border-bottom:2px solid #2d6a9f; padding-bottom:8px; margin-bottom:16px; }
    .insight-box { background:#e8f4fd; border-left:4px solid #2d6a9f; padding:14px 18px; border-radius:0 8px 8px 0; margin:12px 0; color:#1a3a5c; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🗄️ SQL Explorer")
st.markdown("Run live SQL queries directly on the loan database.")

@st.cache_resource
def get_connection():
    return sqlite3.connect('../data/processed/bankloan.db', check_same_thread=False)

conn = get_connection()

# Preset queries
preset_queries = {
    "Overall Default Rate"              : "SELECT COUNT(*) as total, SUM(TARGET) as defaults, ROUND(AVG(TARGET)*100,2) as default_rate_pct FROM loan_applications",
    "Default Rate by Age Group"         : """SELECT CASE WHEN AGE_YEARS < 25 THEN '<25' WHEN AGE_YEARS < 35 THEN '25-35' WHEN AGE_YEARS < 45 THEN '35-45' WHEN AGE_YEARS < 55 THEN '45-55' ELSE '55+' END as age_group, COUNT(*) as total, ROUND(AVG(TARGET)*100,2) as default_rate FROM loan_applications GROUP BY age_group ORDER BY default_rate DESC""",
    "Risk by Credit Segment"            : """SELECT CASE WHEN AMT_CREDIT < 200000 THEN '<200K' WHEN AMT_CREDIT < 500000 THEN '200K-500K' WHEN AMT_CREDIT < 1000000 THEN '500K-1M' ELSE '>1M' END as segment, COUNT(*) as total, ROUND(AVG(TARGET)*100,2) as default_rate FROM loan_applications GROUP BY segment""",
    "Top 10 Riskiest Profiles"          : "SELECT NAME_CONTRACT_TYPE, NAME_INCOME_TYPE, COUNT(*) as total, ROUND(AVG(TARGET)*100,2) as default_rate FROM loan_applications GROUP BY NAME_CONTRACT_TYPE, NAME_INCOME_TYPE HAVING total > 100 ORDER BY default_rate DESC LIMIT 10",
    "Portfolio Health Summary"          : "SELECT COUNT(*) as total_loans, SUM(TARGET) as defaults, ROUND(AVG(AMT_CREDIT),0) as avg_credit, ROUND(AVG(AMT_INCOME_TOTAL),0) as avg_income, ROUND(AVG(AGE_YEARS),0) as avg_age FROM loan_applications",
}

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="section-title">📋 Preset Queries</div>', unsafe_allow_html=True)
    selected = st.selectbox("Choose a query:", list(preset_queries.keys()))
    query    = st.text_area("SQL Query:", value=preset_queries[selected], height=160)
    run      = st.button("▶️ Run Query", use_container_width=True)

with col2:
    st.markdown('<div class="section-title">📊 Query Results</div>', unsafe_allow_html=True)
    if run:
        try:
            result = pd.read_sql(query, conn)
            st.dataframe(result, use_container_width=True)
            st.markdown(f'<div class="insight-box">✅ Returned <b>{len(result)}</b> rows</div>', unsafe_allow_html=True)

            # Auto chart if numeric columns exist
            num_cols = result.select_dtypes(include='number').columns.tolist()
            str_cols = result.select_dtypes(include='object').columns.tolist()
            if str_cols and num_cols:
                fig = px.bar(result, x=str_cols[0], y=num_cols[-1],
                             color_discrete_sequence=['#2d6a9f'])
                fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f8f9fa')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Query error: {e}")

st.markdown('<div class="insight-box">💡 You can also write your own custom SQL in the text box above.</div>', unsafe_allow_html=True)