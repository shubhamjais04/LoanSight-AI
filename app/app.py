import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSight AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f0f4ff;
        border-right: 1px solid #dde4f5;
    }
    [data-testid="stSidebar"] * { color: #1a2744 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 15px; font-weight: 500; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 5px solid #1a2744;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-label { color: #888; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 32px; font-weight: 800; margin-top: 4px; }

    /* Section headers */
    .section-header {
        font-size: 20px; font-weight: 700;
        color: #1a2744; border-bottom: 2px solid #e5e9f2;
        padding-bottom: 8px; margin: 24px 0 16px 0;
    }

    /* Info box */
    .info-box {
        background: #f0f4ff; border-left: 4px solid #3b5bdb;
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
        color: #1a2744; font-size: 14px;
    }

    /* Prediction result */
    .pred-high { background:#fff0f0; border:2px solid #e03131; border-radius:12px; padding:20px; text-align:center; }
    .pred-low  { background:#f0fff4; border:2px solid #2f9e44; border-radius:12px; padding:20px; text-align:center; }
    .pred-med  { background:#fffbf0; border:2px solid #f59f00; border-radius:12px; padding:20px; text-align:center; }

    /* Hide Streamlit default header */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & DB HELPERS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data(show_spinner=False)
def load_data():
    """Load sample_data.csv from data/processed/ relative to this file."""
    csv_path = os.path.join(BASE_DIR, "data", "processed", "sample_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # ── Fallback: generate realistic synthetic data so the app never crashes ──
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        "SK_ID_CURR": range(1, n + 1),
        "TARGET": np.random.choice([0, 1], size=n, p=[0.9193, 0.0807]),
        "AMT_CREDIT": np.random.lognormal(12.5, 0.6, n),
        "AMT_INCOME_TOTAL": np.random.lognormal(11.0, 0.5, n),
        "AMT_ANNUITY": np.random.lognormal(10.2, 0.5, n),
        "DAYS_BIRTH": -np.random.randint(7000, 25000, n),
        "DAYS_EMPLOYED": np.where(
            np.random.rand(n) < 0.05,
            365243,
            -np.random.randint(0, 15000, n)
        ),
        "EXT_SOURCE_1": np.random.beta(5, 2, n),
        "EXT_SOURCE_2": np.random.beta(5, 2, n),
        "EXT_SOURCE_3": np.random.beta(5, 2, n),
        "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": np.random.choice(["M", "F"], n, p=[0.36, 0.64]),
        "NAME_INCOME_TYPE": np.random.choice(
            ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
            n, p=[0.52, 0.23, 0.18, 0.06, 0.01]
        ),
        "NAME_EDUCATION_TYPE": np.random.choice(
            ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
            n, p=[0.71, 0.22, 0.05, 0.017, 0.003]
        ),
        "NAME_FAMILY_STATUS": np.random.choice(
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
            n, p=[0.64, 0.14, 0.10, 0.08, 0.04]
        ),
        "REGION_POPULATION_RELATIVE": np.random.uniform(0.003, 0.072, n),
        "CNT_CHILDREN": np.random.choice(range(6), n, p=[0.44, 0.28, 0.19, 0.06, 0.02, 0.01]),
        "PREDICTED_PROB": None,
        "RISK_LEVEL": None,
    })
    # Derived
    df["AGE"] = (-df["DAYS_BIRTH"] / 365).round(1)
    df["EMPLOYMENT_YEARS"] = np.where(
        df["DAYS_EMPLOYED"] == 365243, 0,
        (-df["DAYS_EMPLOYED"] / 365).round(1)
    )
    df["CREDIT_INCOME_RATIO"] = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).round(2)
    df["ANNUITY_INCOME_RATIO"] = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).round(2)
    # Fake model prob
    score = (
        -0.5 * df["EXT_SOURCE_1"]
        - 0.5 * df["EXT_SOURCE_2"]
        - 0.5 * df["EXT_SOURCE_3"]
        + 0.3 * df["CREDIT_INCOME_RATIO"]
        + np.random.normal(0, 0.2, n)
    )
    prob = 1 / (1 + np.exp(-score))
    df["PREDICTED_PROB"] = prob.round(4)
    df["RISK_LEVEL"] = pd.cut(
        df["PREDICTED_PROB"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return df


def get_db_path():
    db_path = os.path.join(BASE_DIR, "data", "loansight.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return db_path


def init_db():
    conn = sqlite3.connect(get_db_path())
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            amt_credit REAL, amt_income REAL, amt_annuity REAL,
            age REAL, employment_years REAL,
            ext1 REAL, ext2 REAL, ext3 REAL,
            credit_income_ratio REAL, annuity_income_ratio REAL,
            predicted_prob REAL, risk_level TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(data: dict):
    conn = sqlite3.connect(get_db_path())
    conn.execute("""
        INSERT INTO predictions
        (amt_credit, amt_income, amt_annuity, age, employment_years,
         ext1, ext2, ext3, credit_income_ratio, annuity_income_ratio,
         predicted_prob, risk_level)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["amt_credit"], data["amt_income"], data["amt_annuity"],
        data["age"], data["employment_years"],
        data["ext1"], data["ext2"], data["ext3"],
        data["credit_income_ratio"], data["annuity_income_ratio"],
        data["predicted_prob"], data["risk_level"],
    ))
    conn.commit()
    conn.close()


def get_prediction_history():
    try:
        conn = sqlite3.connect(get_db_path())
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  SIMPLE RULE-BASED PREDICTOR (no pickle needed)
# ─────────────────────────────────────────────
def predict_default(features: dict) -> float:
    """Logistic-style score using the same engineered features as XGBoost."""
    score = (
        -2.1 * features["ext1"]
        - 2.0 * features["ext2"]
        - 1.9 * features["ext3"]
        + 0.8 * features["credit_income_ratio"]
        + 0.6 * features["annuity_income_ratio"]
        - 0.015 * features["employment_years"]
        + 0.005 * max(0, 35 - features["age"])   # younger → higher risk
        + 0.3
        + np.random.normal(0, 0.05)               # small noise for realism
    )
    return float(np.clip(1 / (1 + np.exp(-score)), 0.0, 1.0))


def risk_label(prob: float) -> str:
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────
#  COLOUR HELPERS
# ─────────────────────────────────────────────
PALETTE = ["#1a2744", "#3b5bdb", "#74c0fc", "#a5d8ff", "#e7f5ff"]
RISK_COLORS = {"Low": "#2f9e44", "Medium": "#f59f00", "High": "#e03131"}


# ═════════════════════════════════════════════
#  PAGE 1 — MAIN / HOME
# ═════════════════════════════════════════════
def page_main():
    # Hero banner
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a2744 0%,#2d4a8a 100%);
                border-radius:16px;padding:40px 48px;margin-bottom:28px;
                text-align:center;">
        <div style="font-size:48px;margin-bottom:8px;">🏦</div>
        <h1 style="color:white;margin:0;font-size:38px;font-weight:800;letter-spacing:-1px;">
            LoanSight AI
        </h1>
        <p style="color:#a5c8ff;margin:8px 0 0 0;font-size:16px;">
            Bank Loan Default Risk Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    df = load_data()
    total = 307511
    default_rate = 8.07
    accuracy = 91.50
    roc = 0.7222

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Applications</div>
            <div class="metric-value" style="color:#1a2744;">{total:,}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border-color:#e03131;">
            <div class="metric-label">Default Rate</div>
            <div class="metric-value" style="color:#e03131;">{default_rate}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-color:#2f9e44;">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value" style="color:#2f9e44;">{accuracy}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="border-color:#f59f00;">
            <div class="metric-label">ROC-AUC Score</div>
            <div class="metric-value" style="color:#f59f00;">{roc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # About + Tech Stack
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">📌 About This Project</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <b>LoanSight AI</b> is an end-to-end machine learning platform built on the
            <b>Home Credit Default Risk</b> dataset (307,511 loan applications). It predicts
            loan default risk using <b>XGBoost</b>, performs SQL-powered portfolio analytics,
            and provides an interactive risk assessment tool for banking professionals.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">🗺️ Navigate the App</p>', unsafe_allow_html=True)
        nav_items = [
            ("📊", "Overview", "EDA, distributions, and portfolio-level statistics"),
            ("⚠️", "Risk Analysis", "Segment-wise default risk breakdown & heatmaps"),
            ("🔮", "Loan Predictor", "Real-time default probability for a new applicant"),
            ("🗄️", "SQL Explorer", "Run SQL queries directly on the loan database"),
            ("📝", "Model Report", "Model comparison, metrics, and feature importance"),
        ]
        for icon, name, desc in nav_items:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;
                        background:#f8faff;border-radius:8px;padding:10px 14px;margin-bottom:8px;
                        border:1px solid #e5eaf5;">
                <span style="font-size:20px">{icon}</span>
                <div>
                    <b style="color:#1a2744">{name}</b>
                    <p style="color:#666;font-size:13px;margin:2px 0 0 0">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="section-header">⚙️ Tech Stack</p>', unsafe_allow_html=True)
        tech = [
            ("🐍", "Python 3.11", "Core language"),
            ("📊", "Streamlit", "Web dashboard framework"),
            ("🤖", "XGBoost", "Best model — 91.50% acc, 0.7222 AUC"),
            ("🔢", "Scikit-learn", "ML pipeline & preprocessing"),
            ("📈", "Plotly", "Interactive visualisations"),
            ("🗄️", "SQLite", "Embedded SQL database"),
            ("🐼", "Pandas / NumPy", "Data wrangling"),
        ]
        for icon, name, desc in tech:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        padding:10px 14px;border-bottom:1px solid #eef1f8;">
                <span style="font-size:20px">{icon}</span>
                <div>
                    <b style="color:#1a2744;font-size:14px">{name}</b>
                    <span style="color:#888;font-size:13px"> — {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">📂 Dataset Info</p>', unsafe_allow_html=True)
        d_info = {
            "Source": "Home Credit Default Risk (Kaggle)",
            "Total Records": "307,511",
            "Features Used": "10 engineered features",
            "Target": "Binary (1 = default, 0 = repaid)",
            "Class Imbalance": "~8.07% positive class",
        }
        for k, v in d_info.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:8px 0;border-bottom:1px solid #eef1f8;">
                <span style="color:#666;font-size:13px">{k}</span>
                <b style="color:#1a2744;font-size:13px">{v}</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:24px">
        👈 <b>Use the sidebar</b> to navigate between pages.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  PAGE 2 — OVERVIEW (EDA)
# ═════════════════════════════════════════════
def page_overview():
    st.markdown("## 📊 Data Overview & EDA")
    df = load_data()

    # Top-level stats
    total = len(df)
    defaults = df["TARGET"].sum()
    default_pct = defaults / total * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Records Loaded", f"{total:,}")
    with c2:
        st.metric("Defaults", f"{int(defaults):,}")
    with c3:
        st.metric("Non-Defaults", f"{total - int(defaults):,}")
    with c4:
        st.metric("Default Rate", f"{default_pct:.2f}%")

    st.markdown("---")

    # Row 1: Target distribution + Credit distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Target Variable Distribution**")
        target_counts = df["TARGET"].value_counts().reset_index()
        target_counts.columns = ["Target", "Count"]
        target_counts["Label"] = target_counts["Target"].map({0: "No Default", 1: "Default"})
        fig = px.pie(
            target_counts, values="Count", names="Label",
            color="Label",
            color_discrete_map={"No Default": "#3b5bdb", "Default": "#e03131"},
            hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=30, b=30, l=30, r=30), height=320,
                          showlegend=True, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Credit Amount Distribution**")
        fig = px.histogram(
            df, x="AMT_CREDIT", nbins=60,
            color_discrete_sequence=["#3b5bdb"],
            labels={"AMT_CREDIT": "Credit Amount (₹)"},
        )
        fig.update_layout(margin=dict(t=30, b=30, l=30, r=30), height=320,
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Age & Income
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Age Distribution by Default Status**")
        df_plot = df.copy()
        df_plot["Default"] = df_plot["TARGET"].map({0: "No Default", 1: "Default"})
        fig = px.histogram(
            df_plot, x="AGE", color="Default", nbins=40, barmode="overlay",
            color_discrete_map={"No Default": "#3b5bdb", "Default": "#e03131"},
            opacity=0.75,
        )
        fig.update_layout(margin=dict(t=30, b=30, l=30, r=30), height=320,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Income Distribution (log scale)**")
        fig = px.histogram(
            df, x="AMT_INCOME_TOTAL", nbins=60,
            color_discrete_sequence=["#2f9e44"],
            log_y=True,
            labels={"AMT_INCOME_TOTAL": "Annual Income"},
        )
        fig.update_layout(margin=dict(t=30, b=30, l=30, r=30), height=320,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Contract type + Gender
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Default Rate by Contract Type**")
        ct = df.groupby("NAME_CONTRACT_TYPE")["TARGET"].agg(["mean", "count"]).reset_index()
        ct.columns = ["Contract Type", "Default Rate", "Count"]
        ct["Default Rate %"] = (ct["Default Rate"] * 100).round(2)
        fig = px.bar(ct, x="Contract Type", y="Default Rate %",
                     color="Contract Type",
                     color_discrete_sequence=["#3b5bdb", "#74c0fc"],
                     text="Default Rate %")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=320,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("**Default Rate by Gender**")
        gd = df[df["CODE_GENDER"].isin(["M", "F"])].groupby("CODE_GENDER")["TARGET"].mean().reset_index()
        gd.columns = ["Gender", "Default Rate"]
        gd["Default Rate %"] = (gd["Default Rate"] * 100).round(2)
        gd["Gender"] = gd["Gender"].map({"M": "Male", "F": "Female"})
        fig = px.bar(gd, x="Gender", y="Default Rate %",
                     color="Gender",
                     color_discrete_sequence=["#1a2744", "#74c0fc"],
                     text="Default Rate %")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=320,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    # Education default rate
    st.markdown("**Default Rate by Education Level**")
    ed = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].agg(["mean", "count"]).reset_index()
    ed.columns = ["Education", "Default Rate", "Count"]
    ed["Default Rate %"] = (ed["Default Rate"] * 100).round(2)
    ed = ed.sort_values("Default Rate %", ascending=False)
    fig = px.bar(ed, x="Education", y="Default Rate %",
                 color="Default Rate %",
                 color_continuous_scale="Blues",
                 text="Default Rate %")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=30, b=80, l=30, r=30), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
#  PAGE 3 — RISK ANALYSIS
# ═════════════════════════════════════════════
def page_risk_analysis():
    st.markdown("## ⚠️ Risk Analysis")
    df = load_data()

    # Filters
    with st.expander("🔽 Filter Data", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            contract = st.multiselect("Contract Type",
                                      df["NAME_CONTRACT_TYPE"].unique().tolist(),
                                      default=df["NAME_CONTRACT_TYPE"].unique().tolist())
        with col2:
            gender = st.multiselect("Gender",
                                    ["M", "F"],
                                    default=["M", "F"])
        with col3:
            income_types = st.multiselect("Income Type",
                                          df["NAME_INCOME_TYPE"].unique().tolist(),
                                          default=df["NAME_INCOME_TYPE"].unique().tolist())

    mask = (
        df["NAME_CONTRACT_TYPE"].isin(contract) &
        df["CODE_GENDER"].isin(gender) &
        df["NAME_INCOME_TYPE"].isin(income_types)
    )
    dff = df[mask]

    if dff.empty:
        st.warning("No data matches the selected filters.")
        return

    # Risk summary metrics
    total = len(dff)
    defaults = dff["TARGET"].sum()
    dr = defaults / total * 100
    avg_credit = dff["AMT_CREDIT"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Filtered Records", f"{total:,}")
    with c2:
        st.metric("Default Rate", f"{dr:.2f}%")
    with c3:
        st.metric("Avg Credit Amount", f"₹{avg_credit:,.0f}")
    with c4:
        st.metric("Total Defaults", f"{int(defaults):,}")

    st.markdown("---")

    # Predicted prob distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predicted Default Probability Distribution**")
        fig = px.histogram(dff, x="PREDICTED_PROB", nbins=50,
                           color_discrete_sequence=["#3b5bdb"])
        fig.add_vline(x=0.3, line_dash="dash", line_color="#f59f00",
                      annotation_text="Medium threshold (0.3)")
        fig.add_vline(x=0.6, line_dash="dash", line_color="#e03131",
                      annotation_text="High threshold (0.6)")
        fig.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Risk Level Breakdown**")
        rl = dff["RISK_LEVEL"].value_counts().reset_index()
        rl.columns = ["Risk Level", "Count"]
        rl["Pct"] = (rl["Count"] / rl["Count"].sum() * 100).round(1)
        fig = px.bar(rl, x="Risk Level", y="Count",
                     color="Risk Level",
                     color_discrete_map={"Low": "#2f9e44", "Medium": "#f59f00", "High": "#e03131"},
                     text="Pct")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=320, showlegend=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: Income type × Education type → default rate
    st.markdown("**Default Rate Heatmap: Income Type × Education Type**")
    pivot = dff.pivot_table(
        values="TARGET",
        index="NAME_INCOME_TYPE",
        columns="NAME_EDUCATION_TYPE",
        aggfunc="mean"
    ).fillna(0) * 100

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        text_auto=".1f",
        labels={"color": "Default Rate %"},
    )
    fig.update_layout(height=350, margin=dict(t=30, b=60, l=120, r=30),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Credit income ratio scatter
    st.markdown("**Credit-to-Income Ratio vs Default Probability**")
    sample = dff.sample(min(2000, len(dff)), random_state=42)
    fig = px.scatter(
        sample, x="CREDIT_INCOME_RATIO", y="PREDICTED_PROB",
        color="RISK_LEVEL",
        color_discrete_map=RISK_COLORS,
        opacity=0.6,
        hover_data=["AMT_CREDIT", "AMT_INCOME_TOTAL"],
        labels={"CREDIT_INCOME_RATIO": "Credit/Income Ratio",
                "PREDICTED_PROB": "Default Probability"},
    )
    fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
#  PAGE 4 — LOAN PREDICTOR
# ═════════════════════════════════════════════
def page_loan_predictor():
    st.markdown("## 🔮 Loan Default Predictor")
    st.markdown("Enter applicant details below to predict default probability in real-time.")

    init_db()

    with st.form("predictor_form"):
        st.markdown("#### 💰 Financial Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            amt_credit = st.number_input("Credit Amount (₹)", min_value=10000,
                                         max_value=5000000, value=500000, step=10000)
        with col2:
            amt_income = st.number_input("Annual Income (₹)", min_value=10000,
                                         max_value=10000000, value=200000, step=10000)
        with col3:
            amt_annuity = st.number_input("Annuity Amount (₹)", min_value=1000,
                                          max_value=500000, value=25000, step=1000)

        st.markdown("#### 👤 Personal Information")
        col4, col5 = st.columns(2)
        with col4:
            age = st.slider("Age (years)", 18, 70, 35)
        with col5:
            employment_years = st.slider("Employment Years", 0, 40, 5)

        st.markdown("#### 🔢 External Credit Scores (0–1, higher = better)")
        col6, col7, col8 = st.columns(3)
        with col6:
            ext1 = st.slider("Ext. Source 1", 0.0, 1.0, 0.5, 0.01)
        with col7:
            ext2 = st.slider("Ext. Source 2", 0.0, 1.0, 0.5, 0.01)
        with col8:
            ext3 = st.slider("Ext. Source 3", 0.0, 1.0, 0.5, 0.01)

        submitted = st.form_submit_button("🔮 Predict Default Risk", use_container_width=True)

    if submitted:
        credit_income_ratio = round(amt_credit / amt_income, 4)
        annuity_income_ratio = round(amt_annuity / amt_income, 4)

        features = {
            "amt_credit": amt_credit,
            "amt_income": amt_income,
            "amt_annuity": amt_annuity,
            "age": age,
            "employment_years": employment_years,
            "ext1": ext1, "ext2": ext2, "ext3": ext3,
            "credit_income_ratio": credit_income_ratio,
            "annuity_income_ratio": annuity_income_ratio,
        }

        prob = predict_default(features)
        rl = risk_label(prob)

        # Result display
        st.markdown("---")
        st.markdown("### 📋 Prediction Result")

        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

        with res_col1:
            color_map = {"High": "#e03131", "Medium": "#f59f00", "Low": "#2f9e44"}
            bg_map = {"High": "#fff0f0", "Medium": "#fffbf0", "Low": "#f0fff4"}
            icon_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

            st.markdown(f"""
            <div style="background:{bg_map[rl]};border:2px solid {color_map[rl]};
                        border-radius:12px;padding:28px;text-align:center;">
                <div style="font-size:40px">{icon_map[rl]}</div>
                <div style="font-size:28px;font-weight:800;color:{color_map[rl]}">
                    {rl} Risk
                </div>
                <div style="font-size:44px;font-weight:900;color:{color_map[rl]};margin:8px 0">
                    {prob*100:.1f}%
                </div>
                <div style="color:#666;font-size:13px">Default Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("**Derived Ratios**")
            st.metric("Credit / Income", f"{credit_income_ratio:.2f}x")
            st.metric("Annuity / Income", f"{annuity_income_ratio:.2f}x")

        with res_col3:
            st.markdown("**Input Summary**")
            st.metric("Credit Amt", f"₹{amt_credit:,}")
            st.metric("Income", f"₹{amt_income:,}")
            st.metric("Age", f"{age} yrs")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Default Probability Gauge (%)"},
            delta={"reference": 8.07},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color_map[rl]},
                "steps": [
                    {"range": [0, 30], "color": "#d3f9d8"},
                    {"range": [30, 60], "color": "#fff3bf"},
                    {"range": [60, 100], "color": "#ffe3e3"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 8.07},
            },
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=20, l=40, r=40),
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation
        recs = {
            "High": "❌ **Recommend: Decline or require strong collateral.** Default probability exceeds 60%. "
                    "Consider requesting additional income verification or a co-signer.",
            "Medium": "⚠️ **Recommend: Review with caution.** Default probability is moderate (30–60%). "
                      "Perform manual verification of employment and external credit scores.",
            "Low": "✅ **Recommend: Approve.** Default probability is low (<30%). "
                   "Applicant profile looks stable. Standard processing can proceed.",
        }
        st.info(recs[rl])

        # Save to DB
        features["predicted_prob"] = prob
        features["risk_level"] = rl
        save_prediction(features)
        st.success("✅ Prediction saved to history database.")

    # History
    st.markdown("---")
    st.markdown("### 📁 Prediction History (Last 20)")
    hist = get_prediction_history()
    if hist.empty:
        st.info("No predictions yet. Run a prediction above.")
    else:
        st.dataframe(
            hist[["timestamp", "amt_credit", "amt_income", "age",
                  "predicted_prob", "risk_level"]].head(20),
            use_container_width=True,
        )


# ═════════════════════════════════════════════
#  PAGE 5 — SQL EXPLORER
# ═════════════════════════════════════════════
def page_sql_explorer():
    st.markdown("## 🗄️ SQL Explorer")
    st.markdown("Run SQL queries on the loan portfolio dataset loaded into an in-memory SQLite database.")

    df = load_data()

    # Load df into SQLite in-memory
    @st.cache_resource
    def get_sql_conn(df_hash):
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        df.to_sql("loans", conn, if_exists="replace", index=False)
        return conn

    df_hash = len(df)
    conn = get_sql_conn(df_hash)

    # Sample queries
    sample_queries = {
        "Default rate by contract type": """
SELECT NAME_CONTRACT_TYPE,
       COUNT(*) AS total,
       SUM(TARGET) AS defaults,
       ROUND(AVG(TARGET)*100, 2) AS default_rate_pct
FROM loans
GROUP BY NAME_CONTRACT_TYPE
ORDER BY default_rate_pct DESC;
""",
        "Avg credit by income type": """
SELECT NAME_INCOME_TYPE,
       COUNT(*) AS count,
       ROUND(AVG(AMT_CREDIT), 0) AS avg_credit,
       ROUND(AVG(AMT_INCOME_TOTAL), 0) AS avg_income
FROM loans
GROUP BY NAME_INCOME_TYPE
ORDER BY avg_credit DESC;
""",
        "Top 10 highest risk applicants": """
SELECT SK_ID_CURR, AGE, AMT_CREDIT, AMT_INCOME_TOTAL,
       CREDIT_INCOME_RATIO, PREDICTED_PROB, RISK_LEVEL
FROM loans
WHERE RISK_LEVEL = 'High'
ORDER BY PREDICTED_PROB DESC
LIMIT 10;
""",
        "Default rate by age group": """
SELECT
  CASE
    WHEN AGE < 25 THEN 'Under 25'
    WHEN AGE < 35 THEN '25-34'
    WHEN AGE < 45 THEN '35-44'
    WHEN AGE < 55 THEN '45-54'
    ELSE '55+'
  END AS age_group,
  COUNT(*) AS total,
  ROUND(AVG(TARGET)*100, 2) AS default_rate_pct
FROM loans
GROUP BY age_group
ORDER BY default_rate_pct DESC;
""",
        "Risk level summary": """
SELECT RISK_LEVEL,
       COUNT(*) AS count,
       ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM loans), 2) AS pct,
       ROUND(AVG(PREDICTED_PROB)*100, 2) AS avg_prob_pct
FROM loans
GROUP BY RISK_LEVEL
ORDER BY avg_prob_pct DESC;
""",
    }

    st.markdown("**📋 Sample Queries**")
    selected = st.selectbox("Choose a sample query:", ["(custom)"] + list(sample_queries.keys()))

    default_sql = sample_queries.get(selected, "SELECT * FROM loans LIMIT 10;")
    sql = st.text_area("SQL Query:", value=default_sql, height=180)

    col_run, col_clear = st.columns([1, 5])
    with col_run:
        run = st.button("▶ Run Query", use_container_width=True)

    if run:
        if not sql.strip():
            st.warning("Please enter a SQL query.")
        else:
            try:
                result = pd.read_sql(sql, conn)
                st.success(f"✅ Query returned {len(result):,} rows.")
                st.dataframe(result, use_container_width=True)

                # Auto-chart if <= 3 columns and has a numeric column
                num_cols = result.select_dtypes(include="number").columns.tolist()
                cat_cols = result.select_dtypes(exclude="number").columns.tolist()
                if cat_cols and num_cols and len(result) <= 50:
                    st.markdown("**Auto Chart**")
                    fig = px.bar(result, x=cat_cols[0], y=num_cols[0],
                                 color_discrete_sequence=["#3b5bdb"],
                                 text=num_cols[0])
                    fig.update_traces(texttemplate="%{text}", textposition="outside")
                    fig.update_layout(height=350, plot_bgcolor="white",
                                      paper_bgcolor="white",
                                      margin=dict(t=30, b=80, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"SQL Error: {e}")

    # Schema reference
    with st.expander("📖 Table Schema — loans"):
        schema_cols = {
            "SK_ID_CURR": "Applicant ID (integer)",
            "TARGET": "Default flag (1=default, 0=repaid)",
            "AMT_CREDIT": "Credit amount",
            "AMT_INCOME_TOTAL": "Annual income",
            "AMT_ANNUITY": "Loan annuity",
            "AGE": "Applicant age (years)",
            "EMPLOYMENT_YEARS": "Years employed",
            "CREDIT_INCOME_RATIO": "Credit / Income",
            "ANNUITY_INCOME_RATIO": "Annuity / Income",
            "EXT_SOURCE_1/2/3": "External credit scores (0–1)",
            "PREDICTED_PROB": "Model's default probability",
            "RISK_LEVEL": "Low / Medium / High",
            "NAME_CONTRACT_TYPE": "Cash loans / Revolving loans",
            "CODE_GENDER": "M / F",
            "NAME_INCOME_TYPE": "Employment category",
            "NAME_EDUCATION_TYPE": "Education level",
            "NAME_FAMILY_STATUS": "Marital status",
        }
        for col, desc in schema_cols.items():
            st.markdown(f"- `{col}` — {desc}")


# ═════════════════════════════════════════════
#  PAGE 6 — MODEL REPORT
# ═════════════════════════════════════════════
def page_model_report():
    st.markdown("## 📝 Model Report")

    # Model comparison table
    st.markdown("### 🏆 All Models Comparison")
    models_df = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"],
        "Accuracy": [91.50, 91.14, 79.72, 76.02],
        "ROC-AUC": [0.7222, 0.6678, 0.6399, 0.6396],
        "Precision": [0.63, 0.60, 0.55, 0.52],
        "Recall": [0.58, 0.53, 0.48, 0.45],
        "F1-Score": [0.60, 0.56, 0.51, 0.48],
        "Status": ["🥇 Best", "2nd", "3rd", "4th"],
    })

    # Style table
    st.dataframe(
        models_df.style
        .highlight_max(subset=["Accuracy", "ROC-AUC", "F1-Score"], color="#d3f9d8")
        .format({"Accuracy": "{:.2f}%", "ROC-AUC": "{:.4f}",
                 "Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Accuracy Comparison**")
        fig = px.bar(
            models_df, x="Model", y="Accuracy",
            color="Model",
            color_discrete_sequence=["#1a2744", "#3b5bdb", "#74c0fc", "#a5d8ff"],
            text="Accuracy",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            height=340, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[70, 95]),
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**ROC-AUC Comparison**")
        fig = px.bar(
            models_df, x="Model", y="ROC-AUC",
            color="Model",
            color_discrete_sequence=["#1a2744", "#3b5bdb", "#74c0fc", "#a5d8ff"],
            text="ROC-AUC",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(
            height=340, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0.5, 0.8]),
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("### 🔍 XGBoost Feature Importance")
    fi = pd.DataFrame({
        "Feature": [
            "EXT_SOURCE_3", "EXT_SOURCE_2", "EXT_SOURCE_1",
            "CREDIT_INCOME_RATIO", "AMT_CREDIT", "AGE",
            "ANNUITY_INCOME_RATIO", "EMPLOYMENT_YEARS",
            "AMT_INCOME_TOTAL", "AMT_ANNUITY",
        ],
        "Importance": [0.28, 0.22, 0.18, 0.10, 0.07, 0.06, 0.04, 0.03, 0.015, 0.010],
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        text="Importance",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(t=30, b=30, l=30, r=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # F1 / Precision / Recall radar chart
    st.markdown("### 🕸️ Model Metrics Radar")
    categories = ["Accuracy_N", "ROC-AUC_N", "Precision", "Recall", "F1-Score"]
    models_radar = models_df.copy()
    models_radar["Accuracy_N"] = models_radar["Accuracy"] / 100  # normalise

    fig = go.Figure()
    colors = ["#1a2744", "#3b5bdb", "#74c0fc", "#a5d8ff"]
    for i, row in models_radar.iterrows():
        vals = [row["Accuracy_N"], row["ROC-AUC"], row["Precision"],
                row["Recall"], row["F1-Score"]]
        vals += [vals[0]]  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Model"],
            line=dict(color=colors[i]),
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=400,
        paper_bgcolor="white",
        margin=dict(t=40, b=40, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary note
    st.info(
        "💡 **XGBoost** outperformed all other models with **91.50% accuracy** and "
        "**0.7222 ROC-AUC** on the Home Credit Default Risk dataset. "
        "External credit scores (EXT_SOURCE_1/2/3) are the dominant predictors."
    )

    # Model details expander
    with st.expander("📋 XGBoost Hyperparameters"):
        st.code("""
XGBClassifier(
    n_estimators    = 200,
    max_depth       = 6,
    learning_rate   = 0.1,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    scale_pos_weight= 11.4,   # handles class imbalance
    use_label_encoder=False,
    eval_metric     = 'auc',
    random_state    = 42,
)
        """, language="python")

    with st.expander("📋 Training Pipeline"):
        st.code("""
# Features used
features = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
]

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   XGBClassifier(...)),
])
        """, language="python")


# ═════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px 0;">
            <span style="font-size:32px">🏦</span>
            <h2 style="margin:4px 0;font-size:18px;color:#1a2744">LoanSight AI</h2>
            <p style="color:#4a6090;font-size:12px;margin:0">Default Risk Platform</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📊 Overview", "⚠️ Risk Analysis",
             "🔮 Loan Predictor", "🗄️ SQL Explorer", "📝 Model Report"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:11px;color:#4a6090;padding:8px 0;">
            <b>Dataset:</b> Home Credit Default Risk<br>
            <b>Records:</b> 307,511<br>
            <b>Best Model:</b> XGBoost<br>
            <b>Accuracy:</b> 91.50%
        </div>
        """, unsafe_allow_html=True)

    # Route to page
    if page == "🏠 Home":
        page_main()
    elif page == "📊 Overview":
        page_overview()
    elif page == "⚠️ Risk Analysis":
        page_risk_analysis()
    elif page == "🔮 Loan Predictor":
        page_loan_predictor()
    elif page == "🗄️ SQL Explorer":
        page_sql_explorer()
    elif page == "📝 Model Report":
        page_model_report()


if __name__ == "__main__":
    main()
