import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

import plotly.express as px

# -----------------------
# PAGE CONFIG & STYLING
# -----------------------
st.set_page_config(
    page_title="AI-Driven B2B Lead Scoring & Customer Value Dashboard",
    page_icon="📈",
    layout="wide"
)

# Simple colourful card styling
st.markdown(
    """
    <style>
        .metric-card {
            background: #111827;
            padding: 16px 18px;
            border-radius: 14px;
            border: 1px solid #1f2937;
            box-shadow: 0 0 18px rgba(0,0,0,0.25);
        }
        .metric-label {
            font-size: 13px;
            font-weight: 500;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 700;
            color: #f9fafb;
        }
        .metric-sub {
            font-size: 11px;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# SAMPLE DATA (FALLBACK)
# -----------------------
def generate_sample_lead_data(n=500):
    np.random.seed(42)
    industries = ["IT", "Retail", "Manufacturing", "Healthcare", "Finance", "Education"]
    locations = ["North", "South", "East", "West"]
    sources = ["LinkedIn", "Referral", "Event", "Email Campaign", "Cold Call"]
    interests = ["High", "Medium", "Low"]

    df = pd.DataFrame({
        "Lead_ID": [f"L{i:05}" for i in range(1, n+1)],
        "Industry": np.random.choice(industries, n),
        "Company_Size": np.random.randint(10, 500, n),
        "Annual_Revenue_INR_Lakhs": np.random.randint(10, 1000, n),
        "Location": np.random.choice(locations, n),
        "Website_Visits": np.random.randint(1, 50, n),
        "Email_Clicks": np.random.randint(0, 15, n),
        "Meetings": np.random.randint(0, 5, n),
        "Lead_Source": np.random.choice(sources, n),
        "Engagement_Score": np.random.randint(20, 100, n),
        "Product_Interest": np.random.choice(interests, n, p=[0.35, 0.45, 0.20]),
        "Decision_Time_Days": np.random.randint(5, 90, n),
    })

    # synthetic conversion probability
    base = (
        0.25
        + 0.0004 * df["Company_Size"]
        + 0.0002 * df["Annual_Revenue_INR_Lakhs"]
        + 0.01 * df["Meetings"]
        + 0.006 * df["Email_Clicks"]
        + 0.005 * df["Website_Visits"]
        + 0.007 * df["Engagement_Score"]
        - 0.002 * df["Decision_Time_Days"]
        + df["Product_Interest"].map({"Low": -0.05, "Medium": 0.05, "High": 0.12})
    )
    channel_bonus = df["Lead_Source"].map({
        "Referral": 0.15,
        "LinkedIn": 0.06,
        "Event": 0.08,
        "Email Campaign": 0.03,
        "Cold Call": -0.04
    }).fillna(0.0)
    base = base + channel_bonus
    proba = 1 / (1 + np.exp(-(base - base.mean()) / (base.std() + 1e-6)))
    df["Converted"] = (np.random.rand(n) < proba).astype(int)
    return df


def generate_sample_clv_data(n=300):
    np.random.seed(24)
    industries = ["IT", "Retail", "Manufacturing", "Healthcare", "Finance", "Education"]
    upsell = ["Yes", "No"]

    df = pd.DataFrame({
        "Customer_ID": [f"C{i:05}" for i in range(1, n+1)],
        "Industry": np.random.choice(industries, n),
        "Tenure_Months": np.random.randint(3, 48, n),
        "Annual_Spend_INR_Lakhs": np.random.randint(5, 500, n),
        "Product_Usage_Percent": np.random.randint(40, 100, n),
        "Support_Tickets": np.random.randint(0, 10, n),
        "Satisfaction_Score_1_10": np.random.randint(5, 10, n),
        "Payment_Timeliness_Percent": np.random.randint(70, 100, n),
        "Upsell_Responses": np.random.choice(upsell, n, p=[0.3, 0.7]),
    })

    clv = (
        2.5 * df["Annual_Spend_INR_Lakhs"]
        + 1.2 * df["Tenure_Months"]
        + 0.9 * df["Product_Usage_Percent"]
        + 30 * (df["Upsell_Responses"] == "Yes").astype(int)
        + 2 * df["Satisfaction_Score_1_10"]
        + 0.8 * df["Payment_Timeliness_Percent"]
        - 4 * df["Support_Tickets"]
        + np.random.normal(0, 50, n)
    )
    df["CLV_Predicted_Future_Value_INR_Lakhs"] = np.clip(clv.round(0), 20, None)
    return df


# -----------------------
# DATA LOADERS
# -----------------------
@st.cache_data
def load_lead_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    # try repo file; if not present, fall back to synthetic
    if Path("lead_scoring_data.csv").exists():
        return pd.read_csv("lead_scoring_data.csv")

    return generate_sample_lead_data(500)


@st.cache_data
def load_clv_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if Path("customer_clv_data.csv").exists():
        return pd.read_csv("customer_clv_data.csv")

    return generate_sample_clv_data(300)


# -----------------------
# LEAD MODEL & SCORING
# -----------------------
def score_leads(df):
    df = df.copy()

    # if Converted missing, create synthetic 0/1 so model works
    if "Converted" not in df.columns:
        df["Converted"] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])

    target = "Converted"
    num_cols = [
        "Company_Size",
        "Annual_Revenue_INR_Lakhs",
        "Website_Visits",
        "Email_Clicks",
        "Meetings",
        "Engagement_Score",
        "Decision_Time_Days",
    ]
    cat_cols = ["Industry", "Location", "Lead_Source", "Product_Interest"]

    # keep only existing columns (for robustness)
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    X = df[num_cols + cat_cols]
    y = df[target]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
        ],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=250, random_state=42, max_depth=None, class_weight="balanced"
    )

    clf = Pipeline(steps=[("prep", pre), ("model", model)])
    clf.fit(X, y)

    # predicted probability of conversion as AI lead score
    proba = clf.predict_proba(X)[:, 1]
    df["AI_Lead_Score"] = (proba * 100).round(1)

    # score bands / priorities
    df["Score_Band"] = pd.cut(
        proba,
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    df["Priority_Level"] = df["Score_Band"]  # alias for clarity

    return df


# -----------------------
# KPI CARD RENDERER
# -----------------------
def metric_card(col, label, value, subtext=""):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------
# MAIN APP
# -----------------------
st.title("AI-Driven B2B Lead Scoring & Customer Value Dashboard")

st.write(
    "Use this dashboard to **prioritize B2B leads** and **understand customer value**. "
    "Upload your own CSVs or explore with the built-in sample data."
)

# Upload widgets
c1, c2 = st.columns(2)
with c1:
    lead_file = st.file_uploader(
        "Upload Leads CSV (optional)",
        type=["csv"],
        key="lead_csv",
        help="If left blank, the app will use a sample leads dataset.",
    )
with c2:
    clv_file = st.file_uploader(
        "Upload Customer CLV CSV (optional)",
        type=["csv"],
        key="clv_csv",
        help="If left blank, the app will use a sample customer dataset.",
    )

# Load & process data
leads_raw = load_lead_data(lead_file)
leads = score_leads(leads_raw)
customers = load_clv_data(clv_file).copy()

# Prepare CLV-related fields
if "CLV_Predicted_Future_Value_INR_Lakhs" in customers.columns:
    customers["CLV_Rupees"] = customers["CLV_Predicted_Future_Value_INR_Lakhs"] * 100000
else:
    # fallback if column renamed
    clv_col_guess = [c for c in customers.columns if "clv" in c.lower()][0]
    customers["CLV_Rupees"] = customers[clv_col_guess]

# tenure buckets
if "Tenure_Months" in customers.columns:
    customers["Tenure_Bucket"] = pd.cut(
        customers["Tenure_Months"],
        bins=[0, 6, 12, 24, 60],
        labels=["< 6 months", "6–12 months", "12–24 months", "24+ months"],
        include_lowest=True,
    )
else:
    customers["Tenure_Bucket"] = "Unknown"

# high value segment = top 1/3rd CLV
threshold_high_value = customers["CLV_Rupees"].quantile(0.67)
customers["High_Value"] = customers["CLV_Rupees"] >= threshold_high_value

# high churn risk heuristic
customers["High_Churn_Risk"] = False
if "Support_Tickets" in customers.columns:
    customers.loc[customers["Support_Tickets"] >= 6, "High_Churn_Risk"] = True
if "Payment_Timeliness_Percent" in customers.columns:
    customers.loc[customers["Payment_Timeliness_Percent"] <= 85, "High_Churn_Risk"] = True

# -----------------------
# KPIs
# -----------------------
# LEAD SCORING & FUNNEL
overall_conv = 100 * leads["Converted"].mean()

high_priority = leads[leads["Priority_Level"] == "High"]
if len(high_priority) > 0:
    conv_high_priority = 100 * high_priority["Converted"].mean()
else:
    conv_high_priority = 0.0

total_converted = leads["Converted"].sum()
high_priority_converted = high_priority["Converted"].sum()
if total_converted > 0:
    share_new_from_high = 100 * high_priority_converted / total_converted
else:
    share_new_from_high = 0.0

avg_meetings_close = leads.loc[leads["Converted"] == 1, "Meetings"].mean()

# CUSTOMER VALUE & RETENTION
avg_clv = customers["CLV_Rupees"].mean()
pct_high_value = 100 * customers["High_Value"].mean()
top_20_count = max(1, int(0.2 * len(customers)))
revenue_total = customers["CLV_Rupees"].sum()
revenue_top_20 = customers.sort_values("CLV_Rupees", ascending=False).head(top_20_count)[
    "CLV_Rupees"
].sum()
pct_revenue_top_20 = 100 * revenue_top_20 / revenue_total if revenue_total > 0 else 0.0
pct_high_churn = 100 * customers["High_Churn_Risk"].mean()

# Best-performing channel based on conversion rate
if "Lead_Source" in leads.columns:
    ch_perf = leads.groupby("Lead_Source")["Converted"].mean().sort_values(
        ascending=False
    )
    best_channel = ch_perf.index[0]
    best_channel_rate = 100 * ch_perf.iloc[0]
else:
    best_channel = "-"
    best_channel_rate = 0.0

# Average engagement score of customers (converted leads)
avg_engagement_customers = leads.loc[leads["Converted"] == 1, "Engagement_Score"].mean()

# -----------------------
# LAYOUT: TABS
# -----------------------
tab1, tab2 = st.tabs(["Lead Scoring & Funnel", "Customer Value & Retention"])

# ===== TAB 1: LEAD SCORING & FUNNEL =====
with tab1:
    st.subheader("Lead Scoring & Funnel")

    col1, col2, col3, col4 = st.columns(4)
    metric_card(
        col1,
        "Overall Lead Conversion Rate (%)",
        f"{overall_conv:.1f}%",
        "Out of all evaluated leads",
    )
    metric_card(
        col2,
        "Conversion Rate of High-Priority Leads (%)",
        f"{conv_high_priority:.1f}%",
        "Leads with High AI score band",
    )
    metric_card(
        col3,
        "Share of New Customers from High-Priority Leads (%)",
        f"{share_new_from_high:.1f}%",
        "Among all converted leads",
    )
    metric_card(
        col4,
        "Average Meetings Needed to Close a Deal",
        f"{avg_meetings_close:.2f}",
        "Only for converted leads",
    )

    st.markdown("---")

    # Charts for tab 1 (5 charts)
    # 1. Conversion rate by lead score band
    conv_by_band = (
        leads.groupby("Score_Band")["Converted"].mean().mul(100).reset_index()
    )
    fig1 = px.bar(
        conv_by_band,
        x="Score_Band",
        y="Converted",
        color="Score_Band",
        title="Conversion Rate by Lead Score Band",
        labels={"Converted": "Conversion Rate (%)", "Score_Band": "Score Band"},
    )

    # 2. Lead mix by priority level (donut)
    mix_priority = leads["Priority_Level"].value_counts().reset_index()
    mix_priority.columns = ["Priority_Level", "Count"]
    fig2 = px.pie(
        mix_priority,
        values="Count",
        names="Priority_Level",
        hole=0.5,
        title="Lead Mix by Priority Level",
    )

    # 3. Conversion rate by lead source
    if "Lead_Source" in leads.columns:
        conv_by_source = (
            leads.groupby("Lead_Source")["Converted"].mean().mul(100).reset_index()
        )
        fig3 = px.bar(
            conv_by_source,
            x="Lead_Source",
            y="Converted",
            title="Conversion Rate by Lead Source",
            labels={"Converted": "Conversion Rate (%)", "Lead_Source": "Lead Source"},
        )
    else:
        fig3 = None

    # 4. Avg engagement by lead source
    if "Lead_Source" in leads.columns:
        eng_by_source = (
            leads.groupby("Lead_Source")["Engagement_Score"].mean().reset_index()
        )
        fig4 = px.bar(
            eng_by_source,
            x="Lead_Source",
            y="Engagement_Score",
            title="Average Engagement Score by Lead Source",
            labels={"Engagement_Score": "Avg Engagement Score"},
        )
    else:
        fig4 = None

    # 5. Avg AI score by industry
    if "Industry" in leads.columns:
        score_by_industry = (
            leads.groupby("Industry")["AI_Lead_Score"].mean().reset_index()
        )
        fig5 = px.bar(
            score_by_industry,
            x="Industry",
            y="AI_Lead_Score",
            title="Average AI Lead Score by Industry",
            labels={"AI_Lead_Score": "Average AI Score"},
        )
    else:
        fig5 = None

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(fig1, use_container_width=True)
    with r1c2:
        st.plotly_chart(fig2, use_container_width=True)
    with r2c1:
        if fig3 is not None:
            st.plotly_chart(fig3, use_container_width=True)
    with r2c2:
        if fig4 is not None:
            st.plotly_chart(fig4, use_container_width=True)

    st.plotly_chart(fig5, use_container_width=True)

# ===== TAB 2: CUSTOMER VALUE & RETENTION =====
with tab2:
    st.subheader("Customer Value & Retention")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(
        c1,
        "Average Predicted Lifetime Value per Customer (₹)",
        f"₹{avg_clv:,.0f}",
        "From CLV model outputs",
    )
    metric_card(
        c2,
        "% of Customers in High-Value Segment",
        f"{pct_high_value:.1f}%",
        f"High Value = CLV ≥ ₹{threshold_high_value:,.0f} lakhs",
    )
    metric_card(
        c3,
        "% of Revenue from Top 20% Customers",
        f"{pct_revenue_top_20:.1f}%",
        "Share of total predicted revenue",
    )
    metric_card(
        c4,
        "% of Customers at High Churn Risk",
        f"{pct_high_churn:.1f}%",
        "Based on complaints & payment behaviour",
    )

    c5, c6 = st.columns(2)
    metric_card(
        c5,
        "Best-Performing Channel Conversion Rate (%)",
        f"{best_channel_rate:.1f}%",
        f"Channel: {best_channel}",
    )
    metric_card(
        c6,
        "Average Engagement Score of Customers",
        f"{avg_engagement_customers:.1f}",
        "Engagement for converted leads",
    )

    st.markdown("---")

    # Charts for tab 2 (5 charts)
    # 6. Avg CLV by industry
    clv_by_industry = (
        customers.groupby("Industry")["CLV_Rupees"].mean().reset_index()
    )
    fig6 = px.bar(
        clv_by_industry,
        x="Industry",
        y="CLV_Rupees",
        title="Average Predicted CLV by Industry",
        labels={"CLV_Rupees": "Avg CLV (₹)"},
    )

    # 7. Avg CLV by tenure bucket
    clv_by_tenure = (
        customers.groupby("Tenure_Bucket")["CLV_Rupees"].mean().reset_index()
    )
    fig7 = px.bar(
        clv_by_tenure,
        x="Tenure_Bucket",
        y="CLV_Rupees",
        title="Average CLV by Tenure Bucket",
        labels={"CLV_Rupees": "Avg CLV (₹)", "Tenure_Bucket": "Tenure"},
    )

    # 8. CLV distribution
    fig8 = px.histogram(
        customers,
        x="CLV_Rupees",
        nbins=30,
        title="Distribution of Predicted CLV",
        labels={"CLV_Rupees": "CLV (₹)"},
    )

    # 9. Avg time to decision by score band
    time_by_band = (
        leads.groupby("Score_Band")["Decision_Time_Days"].mean().reset_index()
    )
    fig9 = px.bar(
        time_by_band,
        x="Score_Band",
        y="Decision_Time_Days",
        title="Average Time to Decision by Score Band",
        labels={"Decision_Time_Days": "Avg Decision Time (days)", "Score_Band": "Score Band"},
    )

    # 10. Avg meetings by industry
    meetings_by_industry = (
        leads.groupby("Industry")["Meetings"].mean().reset_index()
    )
    fig10 = px.bar(
        meetings_by_industry,
        x="Industry",
        y="Meetings",
        title="Average Meetings Needed by Industry",
        labels={"Meetings": "Avg Meetings"},
    )

    c21, c22 = st.columns(2)
    c23, c24 = st.columns(2)

    with c21:
        st.plotly_chart(fig6, use_container_width=True)
    with c22:
        st.plotly_chart(fig7, use_container_width=True)

    with c23:
        st.plotly_chart(fig8, use_container_width=True)
    with c24:
        st.plotly_chart(fig9, use_container_width=True)

    st.plotly_chart(fig10, use_container_width=True)
