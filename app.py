import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px

# ----------------------------------------------------
# 0. PAGE SETUP
# ----------------------------------------------------
st.set_page_config(
    page_title="AI-Driven B2B Lead Scoring & CLV Dashboard",
    layout="wide",
)

st.title("AI-Driven Lead Scoring & Customer Value Dashboard")

st.markdown(
    """
    Use this dashboard to upload your B2B marketing and customer data,
    score leads with AI, and understand **which leads and customers drive value**.
    """
)

# ----------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------
@st.cache_data
def load_lead_data(uploaded_file):
    if uploaded_file is None:
        return None           # ⬅️ do NOT try to read a local CSV
    return pd.read_csv(uploaded_file)


@st.cache_data
def load_customer_data(uploaded_file):
     if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


st.sidebar.header("1. Upload Data (optional)")
lead_file = st.sidebar.file_uploader("Leads CSV", type=["csv"], key="lead_csv")
customer_file = st.sidebar.file_uploader("Customer CLV CSV", type=["csv"], key="cust_csv")

leads_raw = load_lead_data(lead_file)
customers_raw = load_customer_data(customer_file)

# ----------------------------------------------------
# 2. MODEL: BUILD LEAD SCORE & PRIORITY
# ----------------------------------------------------
@st.cache_resource
def build_and_score_model(leads_df: pd.DataFrame):
    df = leads_df.copy()

    # Basic cleaning / feature engineering
    interest_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Product_Interest_Num"] = df["Product_Interest"].map(interest_map)

    target = "Converted"
    numeric_cols = [
        "Company_Size",
        "Annual_Revenue_INR_Lakhs",
        "Website_Visits",
        "Email_Clicks",
        "Meetings",
        "Engagement_Score",
        "Decision_Time_Days",
        "Product_Interest_Num",
    ]
    cat_cols = ["Industry", "Location", "Lead_Source"]

    X = df[numeric_cols + cat_cols]
    y = df[target]

    # Preprocess + model
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[("cat", categorical_transformer, cat_cols)],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # Predict conversion probabilities for all leads
    conv_prob = pipe.predict_proba(X)[:, 1]
    df["Lead_Score"] = (conv_prob * 100).round(1)

    # Define priority: Top 10% = High Priority
    high_threshold = np.percentile(df["Lead_Score"], 90)
    df["Priority_Level"] = np.where(
        df["Lead_Score"] >= high_threshold,
        "High Priority (Top 10%)",
        "Standard Priority",
    )

    # Score bands for charts
    score_bins = [0, 20, 40, 60, 80, 100]
    score_labels = ["0–20", "21–40", "41–60", "61–80", "81–100"]
    df["Score_Band"] = pd.cut(
        df["Lead_Score"],
        bins=score_bins,
        labels=score_labels,
        include_lowest=True,
        right=True,
    )

    return df, high_threshold


scored_leads, high_score_threshold = build_and_score_model(leads_raw)

# ----------------------------------------------------
# 3. KPI CALCULATIONS
# ----------------------------------------------------
def compute_lead_kpis(df: pd.DataFrame):
    overall_conv = 100 * df["Converted"].mean()

    high_df = df[df["Priority_Level"] == "High Priority (Top 10%)"]
    if len(high_df) > 0:
        high_conv = 100 * high_df["Converted"].mean()
    else:
        high_conv = 0.0

    converted_df = df[df["Converted"] == 1]
    if len(converted_df) > 0:
        share_new_cust_from_high = (
            100
            * len(high_df[high_df["Converted"] == 1])
            / len(converted_df)
        )
        avg_meetings_to_close = converted_df["Meetings"].mean()
    else:
        share_new_cust_from_high = 0.0
        avg_meetings_to_close = 0.0

    return (
        overall_conv,
        high_conv,
        share_new_cust_from_high,
        avg_meetings_to_close,
    )


(
    overall_conv_rate,
    high_priority_conv_rate,
    share_customers_from_high,
    avg_meetings_to_close,
) = compute_lead_kpis(scored_leads)


def compute_customer_kpis(customers_df: pd.DataFrame, leads_df: pd.DataFrame):
    df = customers_df.copy()

    clv_col = "CLV_Predicted_Future_Value_INR_Lakhs"

    avg_ltv = df[clv_col].mean()

    # High-value customers (top 30% by CLV)
    high_val_threshold = np.percentile(df[clv_col], 70)
    high_value_flag = df[clv_col] >= high_val_threshold
    pct_high_value_customers = 100 * high_value_flag.mean()

    # Revenue from top 20% customers
    df_sorted = df.sort_values(clv_col, ascending=False)
    top_20_count = max(1, int(0.2 * len(df_sorted)))
    revenue_top_20 = df_sorted.iloc[:top_20_count][clv_col].sum()
    revenue_total = df_sorted[clv_col].sum()
    pct_revenue_from_top20 = 100 * revenue_top_20 / revenue_total

    # Simple churn risk rule of thumb
    churn_flag = (
        (df["Satisfaction_Score_1_10"] <= 6)
        | (df["Payment_Timeliness_Percent"] < 85)
        | (df["Support_Tickets"] >= 5)
    )
    pct_high_churn_risk = 100 * churn_flag.mean()

    # Best-performing channel by conversion
    conv_by_source = (
        leads_df.groupby("Lead_Source")["Converted"].mean().sort_values(ascending=False)
        * 100
    )
    best_channel = conv_by_source.index[0]
    best_channel_rate = conv_by_source.iloc[0]

    # Average engagement score of customers (converted leads only)
    cust_engagement = leads_df[leads_df["Converted"] == 1]["Engagement_Score"]
    avg_engagement_customers = cust_engagement.mean()

    return (
        avg_ltv,
        pct_high_value_customers,
        pct_revenue_from_top20,
        pct_high_churn_risk,
        best_channel,
        best_channel_rate,
        avg_engagement_customers,
    )


(
    avg_ltv_lakhs,
    pct_high_value_customers,
    pct_revenue_from_top20,
    pct_high_churn_risk,
    best_channel_name,
    best_channel_rate,
    avg_engagement_customers,
) = compute_customer_kpis(customers_raw, scored_leads)

# ----------------------------------------------------
# 4. TABS
# ----------------------------------------------------
tab1, tab2 = st.tabs(["Lead Scoring & Funnel", "Customer Value & Retention"])

# ====================================================
# TAB 1: LEAD SCORING & FUNNEL
# ====================================================
with tab1:
    st.subheader("Summary KPIs")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Overall Lead Conversion Rate (%)",
        f"{overall_conv_rate:0.1f} %",
    )
    c2.metric(
        "Conversion Rate of High-Priority Leads (%)",
        f"{high_priority_conv_rate:0.1f} %",
        help="High-priority = Top 10% of leads by AI lead score.",
    )
    c3.metric(
        "Share of New Customers from High-Priority Leads (%)",
        f"{share_customers_from_high:0.1f} %",
    )
    c4.metric(
        "Average Meetings Needed to Close a Deal",
        f"{avg_meetings_to_close:0.2f}",
    )

    st.markdown(
        f"High-priority leads are defined as the **Top 10%** of leads by AI lead score "
        f"(score ≥ {high_score_threshold:0.1f})."
    )

    st.markdown("---")
    st.subheader("Lead Scoring & Funnel Charts")

    # ---------- 1) Conversion rate by lead score band ----------
    conv_by_band = (
        scored_leads.groupby("Score_Band")["Converted"].mean().reset_index()
    )
    conv_by_band["Conversion_Rate"] = conv_by_band["Converted"] * 100

    fig_band = px.bar(
        conv_by_band,
        x="Score_Band",
        y="Conversion_Rate",
        labels={"Score_Band": "Lead Score Band", "Conversion_Rate": "Conversion Rate (%)"},
        title="Conversion Rate by Lead Score Band",
        color="Score_Band",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_band.update_layout(showlegend=False)

    # ---------- 2) Lead mix by priority level (donut) ----------
    mix_priority = (
        scored_leads["Priority_Level"].value_counts().reset_index()
    )
    mix_priority.columns = ["Priority_Level", "Count"]

    fig_priority = px.pie(
        mix_priority,
        names="Priority_Level",
        values="Count",
        title="Lead Mix by Priority Level",
        hole=0.5,
        color="Priority_Level",
        color_discrete_map={
            "High Priority (Top 10%)": "#EF553B",
            "Standard Priority": "#636EFA",
        },
    )

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(fig_band, use_container_width=True)
    col_b.plotly_chart(fig_priority, use_container_width=True)

    # ---------- 3) Conversion rate by lead source ----------
    conv_by_source = (
        scored_leads.groupby("Lead_Source")["Converted"].mean().reset_index()
    )
    conv_by_source["Conversion_Rate"] = conv_by_source["Converted"] * 100

    fig_conv_source = px.bar(
        conv_by_source,
        x="Lead_Source",
        y="Conversion_Rate",
        title="Conversion Rate by Lead Source",
        labels={"Conversion_Rate": "Conversion Rate (%)", "Lead_Source": "Lead Source"},
        color="Lead_Source",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_conv_source.update_layout(showlegend=False)

    # ---------- 4) Avg engagement by lead source ----------
    eng_by_source = (
        scored_leads.groupby("Lead_Source")["Engagement_Score"].mean().reset_index()
    )

    fig_eng_source = px.bar(
        eng_by_source,
        x="Lead_Source",
        y="Engagement_Score",
        title="Average Engagement Score by Lead Source",
        labels={"Engagement_Score": "Average Engagement Score"},
        color="Lead_Source",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_eng_source.update_layout(showlegend=False)

    col_c, col_d = st.columns(2)
    col_c.plotly_chart(fig_conv_source, use_container_width=True)
    col_d.plotly_chart(fig_eng_source, use_container_width=True)

    # ---------- 5) Avg AI score by industry ----------
    ai_by_industry = (
        scored_leads.groupby("Industry")["Lead_Score"].mean().reset_index()
    )

    fig_ai_industry = px.bar(
        ai_by_industry,
        x="Industry",
        y="Lead_Score",
        title="Average AI Lead Score by Industry",
        labels={"Lead_Score": "Average Lead Score (0–100)"},
        color="Industry",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_ai_industry.update_layout(showlegend=False)

    # ---------- 9) Avg time to decision by score band ----------
    time_by_band = (
        scored_leads.groupby("Score_Band")["Decision_Time_Days"].mean().reset_index()
    )

    fig_time_band = px.line(
        time_by_band,
        x="Score_Band",
        y="Decision_Time_Days",
        title="Average Time to Decision by Lead Score Band",
        markers=True,
        labels={"Score_Band": "Lead Score Band", "Decision_Time_Days": "Average Decision Time (days)"},
    )

    col_e, col_f = st.columns(2)
    col_e.plotly_chart(fig_ai_industry, use_container_width=True)
    col_f.plotly_chart(fig_time_band, use_container_width=True)

    # ---------- 10) Avg meetings by industry ----------
    meetings_by_industry = (
        scored_leads.groupby("Industry")["Meetings"].mean().reset_index()
    )

    fig_meetings_industry = px.bar(
        meetings_by_industry,
        x="Industry",
        y="Meetings",
        title="Average Meetings Needed by Industry",
        labels={"Meetings": "Average Meetings"},
        color="Industry",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_meetings_industry.update_layout(showlegend=False)

    st.plotly_chart(fig_meetings_industry, use_container_width=True)

# ====================================================
# TAB 2: CUSTOMER VALUE & RETENTION
# ====================================================
with tab2:
    st.subheader("Customer Value & Retention KPIs")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Average Predicted Lifetime Value per Customer (₹ Lakhs)",
        f"{avg_ltv_lakhs:0.2f}",
    )
    d2.metric(
        "% of Customers in High-Value Segment",
        f"{pct_high_value_customers:0.1f} %",
        help="High-value customers are the top 30% by predicted lifetime value.",
    )
    d3.metric(
        "% of Revenue from Top 20% Customers",
        f"{pct_revenue_from_top20:0.1f} %",
    )
    d4.metric(
        "% of Customers at High Churn Risk",
        f"{pct_high_churn_risk:0.1f} %",
    )

    e1, e2 = st.columns(2)
    e1.metric(
        "Best-Performing Channel Conversion Rate (%)",
        f"{best_channel_rate:0.1f} %",
        help=f"Best channel: {best_channel_name}",
    )
    e2.metric(
        "Average Engagement Score of Customers",
        f"{avg_engagement_customers:0.2f}",
    )

    st.markdown("---")
    st.subheader("Customer Value Charts")

    # ---------- 6) Avg CLV by industry ----------
    clv_col = "CLV_Predicted_Future_Value_INR_Lakhs"
    clv_by_industry = (
        customers_raw.groupby("Industry")[clv_col].mean().reset_index()
    )

    fig_clv_industry = px.bar(
        clv_by_industry,
        x="Industry",
        y=clv_col,
        title="Average CLV by Industry (₹ Lakhs)",
        labels={clv_col: "Average CLV (₹ Lakhs)"},
        color="Industry",
        color_discrete_sequence=px.colors.qualitative.D3,
    )
    fig_clv_industry.update_layout(showlegend=False)

    # ---------- 7) Avg CLV by tenure bucket ----------
    tenure_bins = [0, 6, 12, 24, 999]
    tenure_labels = ["0–6 months", "7–12 months", "13–24 months", "25+ months"]
    customers_temp = customers_raw.copy()
    customers_temp["Tenure_Bucket"] = pd.cut(
        customers_temp["Tenure_Months"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
        right=True,
    )

    clv_by_tenure = (
        customers_temp.groupby("Tenure_Bucket")[clv_col].mean().reset_index()
    )

    fig_clv_tenure = px.bar(
        clv_by_tenure,
        x="Tenure_Bucket",
        y=clv_col,
        title="Average CLV by Tenure Bucket (₹ Lakhs)",
        labels={"Tenure_Bucket": "Customer Tenure", clv_col: "Average CLV (₹ Lakhs)"},
        color="Tenure_Bucket",
        color_discrete_sequence=px.colors.sequential.Emrld,
    )
    fig_clv_tenure.update_layout(showlegend=False)

    f1, f2 = st.columns(2)
    f1.plotly_chart(fig_clv_industry, use_container_width=True)
    f2.plotly_chart(fig_clv_tenure, use_container_width=True)

    # ---------- 8) CLV distribution ----------
    fig_clv_dist = px.histogram(
        customers_raw,
        x=clv_col,
        nbins=30,
        title="Distribution of Customer Lifetime Value (₹ Lakhs)",
        labels={clv_col: "CLV (₹ Lakhs)", "count": "Number of Customers"},
        color_discrete_sequence=["#AB63FA"],
    )

    st.plotly_chart(fig_clv_dist, use_container_width=True)
