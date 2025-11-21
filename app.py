# app.py
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import altair as alt


# --------------------------------------------------
# 0. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Driven B2B Lead Scoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI-Driven B2B Lead Scoring Dashboard")

st.markdown(
    """
This dashboard helps a B2B marketing or sales team **prioritise leads** using AI.  
Upload a leads CSV and the app will:

1. Train an AI model on your data  
2. Score every lead (0–100) based on conversion likelihood  
3. Mark the **Top 10%** as “High-priority”  
4. Show KPIs and charts that are easy to understand: conversion %, high-priority share, etc.
"""
)

# --------------------------------------------------
# 1. FILE UPLOAD
# --------------------------------------------------
st.sidebar.header("Upload Lead Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload your leads CSV",
    type=["csv"],
    help=(
        "CSV must contain at least these columns: "
        "Lead_ID, Industry, Company_Size, Annual_Revenue_INR_Lakhs, "
        "Location, Website_Visits, Email_Clicks, Meetings, Lead_Source, "
        "Engagement_Score, Product_Interest, Decision_Time_Days, Converted"
    )
)

REQUIRED_COLUMNS = [
    "Lead_ID",
    "Industry",
    "Company_Size",
    "Annual_Revenue_INR_Lakhs",
    "Location",
    "Website_Visits",
    "Email_Clicks",
    "Meetings",
    "Lead_Source",
    "Engagement_Score",
    "Product_Interest",
    "Decision_Time_Days",
    "Converted",
]


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"These required columns are missing from your CSV: {', '.join(missing)}")
        st.stop()


# --------------------------------------------------
# 2. MODEL + SCORING FUNCTION
# --------------------------------------------------
def build_and_score_model(raw_df: pd.DataFrame):
    """
    Train a Random Forest model on the uploaded data and
    return a dataframe with predicted probabilities, lead scores and segments.
    """

    df = raw_df.copy()

    # Ensure required columns exist
    validate_columns(df)

    # ----- Prepare target variable -----
    # Convert any "Yes/No", "Y/N" style to 1/0 if needed
    if df["Converted"].dtype == "O":
        df["Converted"] = df["Converted"].str.strip().str.lower().map(
            {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
        )
    df["Converted"] = df["Converted"].astype(int)

    # ----- Feature engineering -----
    interest_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Product_Interest_Num"] = df["Product_Interest"].map(interest_map).fillna(2)

    numeric_features = [
        "Company_Size",
        "Annual_Revenue_INR_Lakhs",
        "Website_Visits",
        "Email_Clicks",
        "Meetings",
        "Engagement_Score",
        "Decision_Time_Days",
        "Product_Interest_Num",
    ]

    categorical_features = ["Industry", "Location", "Lead_Source"]

    X = df[numeric_features + categorical_features]
    y = df["Converted"]

    # Preprocessing + model pipeline
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False  # important for scikit-learn 1.2.x on Streamlit Cloud
    )

    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    # Train/test split mainly so model doesn’t overfit completely
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    # Predict conversion probability for ALL leads
    proba = clf.predict_proba(X)[:, 1]
    df["Predicted_Conversion_Prob"] = proba
    df["Lead_Score"] = (proba * 100).round(2)

    # Define priority segments based on score distribution
    top10_threshold = np.percentile(df["Lead_Score"], 90)

    median_score = df["Lead_Score"].median()
    df["Priority_Segment"] = np.where(
        df["Lead_Score"] >= top10_threshold,
        "High Priority (Top 10%)",
        np.where(
            df["Lead_Score"] >= median_score,
            "Medium Priority",
            "Low Priority",
        ),
    )

    return df, top10_threshold


# --------------------------------------------------
# 3. MAIN DASHBOARD
# --------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a leads CSV from the sidebar to see the dashboard.")
    st.stop()

# Load & clean data
data = pd.read_csv(uploaded_file)
validate_columns(data)

scored_df, high_threshold = build_and_score_model(data)

# Basic numbers
total_leads = len(scored_df)
overall_conv_rate = scored_df["Converted"].mean() * 100

high_mask = scored_df["Priority_Segment"] == "High Priority (Top 10%)"
high_priority_share = high_mask.mean() * 100
high_priority_conv_rate = (
    scored_df.loc[high_mask, "Converted"].mean() * 100 if high_mask.any() else 0.0
)

avg_predicted_prob = scored_df["Lead_Score"].mean()

# --------------------------------------------------
# 3A. KPI CARDS
# --------------------------------------------------
st.subheader("Summary KPIs (Business-Friendly)")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_col1.metric(
    "Overall Conversion Rate",
    f"{overall_conv_rate:.1f} %",
    help="What percentage of all uploaded leads actually converted."
)

kpi_col2.metric(
    "Conversion Rate of High-Priority Leads",
    f"{high_priority_conv_rate:.1f} %",
    help="Among the leads in the top 10% of AI scores, what percentage converted."
)

kpi_col3.metric(
    "Share of Leads Marked High-Priority",
    f"{high_priority_share:.1f} %",
    help="What portion of all leads fall into the top 10% score band."
)

kpi_col4.metric(
    "Average Predicted Conversion Chance",
    f"{avg_predicted_prob:.1f} %",
    help="Average AI-predicted chance of conversion across all leads."
)

st.caption("High-priority leads are defined as the **Top 10%** of leads by AI lead score.")

st.markdown("---")

# --------------------------------------------------
# 3B. VISUALISATIONS
# --------------------------------------------------

# 1) Lead score distribution
st.subheader("Lead Score Distribution")

score_hist = (
    alt.Chart(scored_df)
    .mark_bar()
    .encode(
        x=alt.X("Lead_Score:Q", bin=alt.Bin(maxbins=30), title="Lead Score (0–100)"),
        y=alt.Y("count():Q", title="Number of Leads"),
        tooltip=["count()"],
    )
    .properties(height=300)
)

st.altair_chart(score_hist, use_container_width=True)

st.markdown("---")

# 2) Conversion rate by Industry & Lead Source
st.subheader("Where Conversions Are Coming From")

conv_by_industry = (
    scored_df.groupby("Industry")["Converted"]
    .mean()
    .reset_index()
    .rename(columns={"Converted": "Conversion_Rate"})
)
conv_by_industry["Conversion_Rate"] *= 100

conv_by_source = (
    scored_df.groupby("Lead_Source")["Converted"]
    .mean()
    .reset_index()
    .rename(columns={"Converted": "Conversion_Rate"})
)
conv_by_source["Conversion_Rate"] *= 100

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Conversion Rate by Industry**")
    chart_industry = (
        alt.Chart(conv_by_industry)
        .mark_bar()
        .encode(
            x=alt.X("Industry:N", sort="-y"),
            y=alt.Y("Conversion_Rate:Q", title="Conversion Rate (%)"),
            tooltip=["Industry", alt.Tooltip("Conversion_Rate:Q", format=".1f")],
        )
        .properties(height=350)
    )
    st.altair_chart(chart_industry, use_container_width=True)

with col_b:
    st.markdown("**Conversion Rate by Lead Source**")
    chart_source = (
        alt.Chart(conv_by_source)
        .mark_bar()
        .encode(
            x=alt.X("Lead_Source:N", sort="-y"),
            y=alt.Y("Conversion_Rate:Q", title="Conversion Rate (%)"),
            tooltip=["Lead_Source", alt.Tooltip("Conversion_Rate:Q", format=".1f")],
        )
        .properties(height=350)
    )
    st.altair_chart(chart_source, use_container_width=True)

st.markdown("---")

# 3) Priority segments breakdown
st.subheader("Lead Priority Segments")

segment_counts = (
    scored_df.groupby("Priority_Segment")
    .size()
    .reset_index(name="Count")
)
segment_counts["Percent"] = segment_counts["Count"] / total_leads * 100

seg_chart = (
    alt.Chart(segment_counts)
    .mark_bar()
    .encode(
        x=alt.X("Priority_Segment:N", sort=["High Priority (Top 10%)", "Medium Priority", "Low Priority"],
                title="Priority Segment"),
        y=alt.Y("Percent:Q", title="Share of Leads (%)"),
        tooltip=[
            "Priority_Segment",
            alt.Tooltip("Percent:Q", format=".1f"),
            "Count",
        ],
        color=alt.Color("Priority_Segment:N", legend=None),
    )
    .properties(height=350)
)

st.altair_chart(seg_chart, use_container_width=True)

st.markdown("---")

# 4) Top 10 leads by AI score
st.subheader("Top 10 Leads by AI Lead Score")

top10 = scored_df.sort_values("Lead_Score", ascending=False).head(10)

top10_chart = (
    alt.Chart(top10)
    .mark_bar()
    .encode(
        x=alt.X("Lead_ID:N", sort=top10["Lead_Score"].tolist(), title="Lead ID"),
        y=alt.Y("Lead_Score:Q", title="Lead Score (0–100)"),
        color="Priority_Segment:N",
        tooltip=[
            "Lead_ID",
            "Lead_Score",
            "Industry",
            "Lead_Source",
            "Company_Size",
            "Annual_Revenue_INR_Lakhs",
            "Converted",
        ],
    )
    .properties(height=350)
)

st.altair_chart(top10_chart, use_container_width=True)

st.caption(
    "Use this chart to quickly see which specific companies the AI believes are most likely to convert."
)
