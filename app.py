# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px

# ------------------------------------------------------
# 0. PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="AI-Driven B2B Lead Scoring & CLV Dashboard",
    layout="wide",
)

st.title("AI-Driven B2B Lead Scoring & Customer Value Dashboard")
st.markdown(
    """
    Use this dashboard to **prioritize B2B leads** and **understand customer value**.  
    Upload your own CSVs (optional) or use the sample data to explore the KPIs and charts.
    """
)

# ------------------------------------------------------
# 1. DATA LOADERS
# ------------------------------------------------------
@st.cache_data
def load_lead_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("lead_scoring_data.csv")
    return df


@st.cache_data
def load_customer_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("customer_clv_data.csv")
    return df


col_u1, col_u2 = st.columns(2)
with col_u1:
    lead_file = st.file_uploader(
        "Upload Leads CSV (optional)",
        type=["csv"],
        help="If empty, the sample lead_scoring_data.csv in the repo is used.",
    )
with col_u2:
    cust_file = st.file_uploader(
        "Upload Customer CLV CSV (optional)",
        type=["csv"],
        help="If empty, the sample customer_clv_data.csv in the repo is used.",
    )

leads_raw = load_lead_data(lead_file)
customers_raw = load_customer_data(cust_file)

# ------------------------------------------------------
# 2. TRAIN SIMPLE AI MODEL & SCORE LEADS
# ------------------------------------------------------
@st.cache_data
def score_leads(leads_df: pd.DataFrame) -> pd.DataFrame:
    df = leads_df.copy()

    # Basic cleaning: drop rows without target
    df = df.dropna(subset=["Converted"])

    # Features & target
    target_col = "Converted"
    numeric_cols = [
        "Company_Size",
        "Annual_Revenue_INR_Lakhs",
        "Website_Visits",
        "Email_Clicks",
        "Meetings",
        "Engagement_Score",
        "Decision_Time_Days",
    ]
    cat_cols = ["Industry", "Location", "Lead_Source", "Product_Interest"]

    X = df[numeric_cols + cat_cols]
    y = df[target_col]

    # Categorical transformer (use sparse=False for compatibility)
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, cat_cols)],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # Predict probability of conversion for all leads
    proba = pipe.predict_proba(X)[:, 1]
    df["AI_Lead_Score"] = (proba * 100).round(2)  # 0–100

    # Priority buckets: top 20% = High, next 30% = Medium, rest = Low
    df = df.sort_values("AI_Lead_Score", ascending=False).reset_index(drop=True)
    n = len(df)
    high_cut = int(0.2 * n)
    med_cut = int(0.5 * n)

    priority = ["High"] * high_cut + ["Medium"] * (med_cut - high_cut) + ["Low"] * (
        n - med_cut
    )
    df["Priority_Level"] = priority

    # Score band for charts (0–20, 20–40, ... 80–100)
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    df["Score_Band"] = pd.cut(
        df["AI_Lead_Score"], bins=bins, labels=labels, include_lowest=True
    )

    return df


leads = score_leads(leads_raw)
customers = customers_raw.copy()

# ------------------------------------------------------
# 3. METRIC CALCULATIONS
# ------------------------------------------------------
# --- LEAD SCORING & FUNNEL KPIs ---
total_leads = len(leads)
total_converted = leads["Converted"].sum()
overall_conv_rate = (total_converted / total_leads * 100) if total_leads > 0 else 0.0

high_priority = leads[leads["Priority_Level"] == "High"]
high_total = len(high_priority)
high_converted = high_priority["Converted"].sum()
high_conv_rate = (high_converted / high_total * 100) if high_total > 0 else 0.0

share_new_from_high = (
    high_converted / total_converted * 100 if total_converted > 0 else 0.0
)

closed_deals = leads[leads["Converted"] == 1]
avg_meetings_to_close = (
    closed_deals["Meetings"].mean() if len(closed_deals) > 0 else np.nan
)

# --- CUSTOMER VALUE & RETENTION KPIs ---
# CLV is stored in Lakhs; convert to rupees for display, but keep Lakhs for logic
clv_lakhs_col = "CLV_Predicted_Future_Value_INR_Lakhs"
customers = customers.dropna(subset=[clv_lakhs_col])
customers["CLV_Rupees"] = customers[clv_lakhs_col] * 1e5

avg_clv_rupees = customers["CLV_Rupees"].mean() if len(customers) > 0 else 0.0

# High-value = top 25% CLV
if len(customers) > 0:
    hv_threshold = customers[clv_lakhs_col].quantile(0.75)
    high_value_customers = customers[customers[clv_lakhs_col] >= hv_threshold]
    pct_high_value_customers = (
        len(high_value_customers) / len(customers) * 100 if len(customers) > 0 else 0.0
    )
else:
    pct_high_value_customers = 0.0

# % of revenue from top 20% customers
if len(customers) > 0:
    customers_sorted = customers.sort_values(clv_lakhs_col, ascending=False)
    top_20_count = max(1, int(0.2 * len(customers_sorted)))
    top_20 = customers_sorted.head(top_20_count)
    pct_revenue_top_20 = (
        top_20[clv_lakhs_col].sum() / customers_sorted[clv_lakhs_col].sum() * 100
    )
else:
    pct_revenue_top_20 = 0.0

# Simple churn risk rule: low satisfaction or low payment timeliness
if len(customers) > 0:
    customers["High_Churn_Risk"] = np.where(
        (customers["Satisfaction_Score_1_10"] <= 6)
        | (customers["Payment_Timeliness_Percent"] < 85),
        1,
        0,
    )
    pct_high_churn_risk = customers["High_Churn_Risk"].mean() * 100
else:
    pct_high_churn_risk = 0.0

# Best-performing channel (by conversion rate)
channel_conv = (
    leads.groupby("Lead_Source")["Converted"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
if len(channel_conv) > 0:
    best_channel_row = channel_conv.iloc[0]
    best_channel_name = best_channel_row["Lead_Source"]
    best_channel_conv_rate = best_channel_row["Converted"] * 100
else:
    best_channel_name = "-"
    best_channel_conv_rate = 0.0

# Average engagement score of customers (converted leads)
avg_engagement_customers = (
    closed_deals["Engagement_Score"].mean() if len(closed_deals) > 0 else np.nan
)

# ------------------------------------------------------
# 4. KPI CARDS IN TABS
# ------------------------------------------------------
tab_leads, tab_value = st.tabs(
    ["Lead Scoring & Funnel", "Customer Value & Retention"]
)

with tab_leads:
    st.subheader("Summary KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Lead Conversion Rate (%)", f"{overall_conv_rate:0.1f}%")
    c2.metric(
        "Conversion Rate of High-Priority Leads (%)", f"{high_conv_rate:0.1f}%"
    )
    c3.metric(
        "Share of New Customers from High-Priority Leads (%)",
        f"{share_new_from_high:0.1f}%",
    )
    if not np.isnan(avg_meetings_to_close):
        c4.metric(
            "Average Meetings Needed to Close a Deal",
            f"{avg_meetings_to_close:0.2f}",
        )
    else:
        c4.metric("Average Meetings Needed to Close a Deal", "–")

with tab_value:
    st.subheader("Customer Value & Retention KPIs")

    c5, c6, c7 = st.columns(3)
    c8, c9, c10 = st.columns(3)

    c5.metric(
        "Average Predicted Lifetime Value per Customer (₹)",
        f"₹{avg_clv_rupees:,.0f}",
    )
    c6.metric(
        "% of Customers in High-Value Segment", f"{pct_high_value_customers:0.1f}%"
    )
    c7.metric(
        "% of Revenue from Top 20% Customers", f"{pct_revenue_top_20:0.1f}%"
    )
    c8.metric(
        "% of Customers at High Churn Risk", f"{pct_high_churn_risk:0.1f}%"
    )
    c9.metric(
        "Best-Performing Channel Conversion Rate (%)",
        f"{best_channel_name}: {best_channel_conv_rate:0.1f}%",
    )
    if not np.isnan(avg_engagement_customers):
        c10.metric(
            "Average Engagement Score of Customers",
            f"{avg_engagement_customers:0.1f}",
        )
    else:
        c10.metric("Average Engagement Score of Customers", "–")

st.markdown("---")

# ------------------------------------------------------
# 5. VISUALISATIONS (10 CHARTS, NO TABLES)
# ------------------------------------------------------
st.subheader("Lead Scoring & Funnel Visualisations")

# 1. Conversion rate by lead score band
conv_by_band = (
    leads.groupby("Score_Band")["Converted"].mean().reset_index().sort_values(
        "Score_Band"
    )
)
fig1 = px.bar(
    conv_by_band,
    x="Score_Band",
    y="Converted",
    title="Conversion Rate by Lead Score Band",
    labels={"Converted": "Conversion Rate"},
    text=conv_by_band["Converted"].apply(lambda x: f"{x*100:0.1f}%"),
)
fig1.update_traces(textposition="outside")
fig1.update_yaxes(tickformat=".0%")
st.plotly_chart(fig1, use_container_width=True)

# 2. Lead mix by priority level (donut)
priority_mix = (
    leads["Priority_Level"].value_counts().reset_index()
)
priority_mix.columns = ["Priority_Level", "Count"]
fig2 = px.pie(
    priority_mix,
    names="Priority_Level",
    values="Count",
    hole=0.5,
    title="Lead Mix by Priority Level",
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Conversion rate by lead source
conv_by_source = (
    leads.groupby("Lead_Source")["Converted"].mean().reset_index().sort_values(
        "Converted", ascending=False
    )
)
fig3 = px.bar(
    conv_by_source,
    x="Lead_Source",
    y="Converted",
    title="Conversion Rate by Lead Source",
    labels={"Converted": "Conversion Rate"},
    text=conv_by_source["Converted"].apply(lambda x: f"{x*100:0.1f}%"),
)
fig3.update_traces(textposition="outside")
fig3.update_yaxes(tickformat=".0%")
st.plotly_chart(fig3, use_container_width=True)

# 4. Average engagement by lead source
eng_by_source = (
    leads.groupby("Lead_Source")["Engagement_Score"].mean().reset_index()
)
fig4 = px.bar(
    eng_by_source,
    x="Lead_Source",
    y="Engagement_Score",
    title="Average Engagement Score by Lead Source",
)
st.plotly_chart(fig4, use_container_width=True)

# 5. Average AI score by industry
score_by_industry = (
    leads.groupby("Industry")["AI_Lead_Score"].mean().reset_index()
)
fig5 = px.bar(
    score_by_industry,
    x="Industry",
    y="AI_Lead_Score",
    title="Average AI Lead Score by Industry",
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.subheader("Customer Value & Retention Visualisations")

# 6. Average CLV by industry
clv_by_industry = (
    customers.groupby("Industry")[clv_lakhs_col].mean().reset_index()
)
fig6 = px.bar(
    clv_by_industry,
    x="Industry",
    y=clv_lakhs_col,
    title="Average Predicted CLV by Industry (₹ Lakhs)",
    labels={clv_lakhs_col: "Predicted CLV (₹ Lakhs)"},
)
st.plotly_chart(fig6, use_container_width=True)

# 7. Average CLV by tenure bucket
tenure_bins = [0, 6, 12, 24, np.inf]
tenure_labels = ["<6M", "6–12M", "12–24M", ">24M"]
customers["Tenure_Bucket"] = pd.cut(
    customers["Tenure_Months"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
)
clv_by_tenure = (
    customers.groupby("Tenure_Bucket")[clv_lakhs_col].mean().reset_index()
)
fig7 = px.bar(
    clv_by_tenure,
    x="Tenure_Bucket",
    y=clv_lakhs_col,
    title="Average Predicted CLV by Tenure Bucket (₹ Lakhs)",
    labels={clv_lakhs_col: "Predicted CLV (₹ Lakhs)", "Tenure_Bucket": "Tenure"},
)
st.plotly_chart(fig7, use_container_width=True)

# 8. CLV distribution
fig8 = px.histogram(
    customers,
    x=clv_lakhs_col,
    nbins=20,
    title="Distribution of Predicted Customer Lifetime Value (₹ Lakhs)",
    labels={clv_lakhs_col: "Predicted CLV (₹ Lakhs)"},
)
st.plotly_chart(fig8, use_container_width=True)

# 9. Average time to decision by score band
time_by_band = (
    leads.groupby("Score_Band")["Decision_Time_Days"].mean().reset_index()
)
fig9 = px.bar(
    time_by_band,
    x="Score_Band",
    y="Decision_Time_Days",
    title="Average Time to Decision by Score Band (Days)",
    labels={"Decision_Time_Days": "Avg Decision Time (Days)"},
)
st.plotly_chart(fig9, use_container_width=True)

# 10. Average meetings by chosen grouping (industry or source)
st.subheader("Average Meetings Comparison")

group_choice = st.selectbox(
    "Group by",
    options=["Industry", "Lead_Source"],
    index=0,
    help="Choose whether to view average meetings by industry or by lead source.",
)

meetings_group = (
    leads.groupby(group_choice)["Meetings"].mean().reset_index()
)
fig10 = px.bar(
    meetings_group,
    x=group_choice,
    y="Meetings",
    title=f"Average Meetings by {group_choice}",
    labels={"Meetings": "Average Meetings"},
)
st.plotly_chart(fig10, use_container_width=True)
