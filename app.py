import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px

# --------------------------------------------------
# 1. PAGE CONFIG & BASIC STYLING
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Driven B2B Lead Scoring & Customer Value Dashboard",
    layout="wide",
)

# Custom CSS for colourful cards & cleaner look
st.markdown(
    """
    <style>
    .big-title {
        font-size: 32px !important;
        font-weight: 800 !important;
        color: #0f4c75;
    }
    .sub-text {
        color: #555555;
        font-size: 14px;
    }
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd, #e8f5e9);
        padding: 14px 18px;
        border-radius: 14px;
        border: 1px solid #bbdefb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 13px;
        color: #455a64;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0d47a1;
    }
    .metric-caption {
        font-size: 11px;
        color: #78909c;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 2. HEADER
# --------------------------------------------------
st.markdown(
    '<div class="big-title">AI-Driven B2B Lead Scoring & Customer Value Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-text">Upload a single CSV containing both lead and customer attributes to see AI scores, funnel metrics, and value insights.</div>',
    unsafe_allow_html=True,
)

st.write("")

# --------------------------------------------------
# 3. FILE UPLOADER (ONLY ONE)
# --------------------------------------------------
uploaded = st.file_uploader(
    "Upload combined B2B lead & customer CSV",
    type=["csv"],
    help=(
        "The file should include columns: Lead_ID, Industry, Lead_Source, "
        "Engagement_Score, Meetings, Decision_Time_Days, Converted, "
        "CLV, Tenure_Months, Churn_Risk, Revenue (₹)."
    ),
)

if uploaded is None:
    st.info("⬆️ Upload your CSV to generate the AI lead scores and dashboard.")
    st.stop()

# Read data
df = pd.read_csv(uploaded)

# --------------------------------------------------
# 4. BASIC COLUMN SAFETY CHECKS
# --------------------------------------------------
required_base = [
    "Lead_ID",
    "Industry",
    "Lead_Source",
    "Engagement_Score",
    "Meetings",
    "Decision_Time_Days",
    "Converted",
]
required_clv = ["CLV", "Tenure_Months", "Churn_Risk", "Revenue"]

missing_base = [c for c in required_base if c not in df.columns]
missing_clv = [c for c in required_clv if c not in df.columns]

if missing_base:
    st.error(f"These required columns are missing from your file: {missing_base}")
    st.stop()

if missing_clv:
    st.warning(
        f"Some customer-value columns are missing ({missing_clv}). "
        "Lead scoring & funnel will still work, but a few value KPIs/charts may be blank."
    )

# --------------------------------------------------
# 5. BUILD SIMPLE AI LEAD SCORING MODEL
# --------------------------------------------------
def build_lead_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Train a simple model on this batch and return dataframe with Lead_Score and Priority_Level."""

    model_df = data.copy()

    # Features & target
    feature_cols_cat = ["Industry", "Lead_Source"]
    feature_cols_num = ["Engagement_Score", "Meetings", "Decision_Time_Days"]

    X = model_df[feature_cols_cat + feature_cols_num]
    y = model_df["Converted"].astype(int)

    # Preprocess: OneHot for categoricals (no sparse flag to avoid version issues)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ],
        remainder="passthrough",
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])

    # Robust train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
    except ValueError:
        # In case of very small or all-same data, fit on full data
        pipe.fit(X, y)

    # Get probability of conversion as AI score
    proba = pipe.predict_proba(X)[:, 1]
    model_df["Lead_Score"] = (proba * 100).round(2)

    # Priority buckets
    def to_bucket(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        else:
            return "Low"

    model_df["Priority_Level"] = model_df["Lead_Score"].apply(to_bucket)
    return model_df


df = build_lead_scores(df)

# --------------------------------------------------
# 6. DERIVED COLUMNS FOR VALUE ANALYSIS
# --------------------------------------------------
if "CLV" in df.columns:
    # High / Medium / Low value segmentation based on CLV quantiles
    q_low = df["CLV"].quantile(0.33)
    q_high = df["CLV"].quantile(0.66)

    def value_segment(clv):
        if clv >= q_high:
            return "High"
        elif clv >= q_low:
            return "Medium"
        else:
            return "Low"

    df["Value_Segment"] = df["CLV"].apply(value_segment)

if "Churn_Risk" in df.columns:
    df["High_Churn"] = (df["Churn_Risk"] >= 0.6).astype(int)

# --------------------------------------------------
# 7. KPI CALCULATIONS
# --------------------------------------------------
converted_mask = df["Converted"] == 1
high_priority_mask = df["Priority_Level"] == "High"

# 1. Overall Lead Conversion Rate
overall_conv = converted_mask.mean() * 100

# 2. Conversion Rate of High-Priority Leads
if high_priority_mask.any():
    high_priority_conv = (df[high_priority_mask]["Converted"] == 1).mean() * 100
else:
    high_priority_conv = np.nan

# 3. Share of New Customers from High-Priority Leads
if converted_mask.any():
    share_new_from_high = (
        df[high_priority_mask & converted_mask].shape[0]
        / df[converted_mask].shape[0]
        * 100
    )
else:
    share_new_from_high = np.nan

# 4. Average Meetings Needed to Close a Deal
if converted_mask.any():
    avg_meetings_close = df.loc[converted_mask, "Meetings"].mean()
else:
    avg_meetings_close = np.nan

# --- Customer value KPIs (needs CLV etc.) ---
if not missing_clv:
    # 7. Average Predicted Lifetime Value per Customer (₹)
    avg_clv = df.loc[converted_mask, "CLV"].mean()

    # 8. % of Customers in High-Value Segment
    high_value_mask = (df["Value_Segment"] == "High") & converted_mask
    if converted_mask.any():
        pct_high_value_customers = high_value_mask.mean() * 100
    else:
        pct_high_value_customers = np.nan

    # 9. % of Revenue from Top 20% Customers (by CLV)
    cust_df = df[converted_mask].copy()
    cust_df = cust_df.sort_values("CLV", ascending=False)
    if not cust_df.empty:
        top_n = max(1, int(np.ceil(0.2 * len(cust_df))))
        top_rev = cust_df.head(top_n)["Revenue"].sum()
        total_rev = cust_df["Revenue"].sum()
        pct_rev_top20 = (top_rev / total_rev * 100) if total_rev > 0 else np.nan
    else:
        pct_rev_top20 = np.nan

    # 10. % of Customers at High Churn Risk
    if "High_Churn" in df.columns and converted_mask.any():
        pct_high_churn_customers = (
            df.loc[converted_mask, "High_Churn"].mean() * 100
        )
    else:
        pct_high_churn_customers = np.nan

    # 11. Best-Performing Channel Conversion Rate (%)
    channel_conv = (
        df.groupby("Lead_Source")["Converted"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    if not channel_conv.empty:
        best_channel = channel_conv.index[0]
        best_channel_conv = channel_conv.iloc[0]
    else:
        best_channel = "N/A"
        best_channel_conv = np.nan

    # 12. Average Engagement Score of Customers
    avg_engagement_customers = df.loc[converted_mask, "Engagement_Score"].mean()
else:
    avg_clv = pct_high_value_customers = pct_rev_top20 = np.nan
    pct_high_churn_customers = best_channel_conv = avg_engagement_customers = np.nan
    best_channel = "N/A"

# Helper to format numbers
def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x:0.1f}%"

def fmt_num(x, decimals=1):
    return "-" if pd.isna(x) else f"{x:0.{decimals}f}"

def fmt_rupees(x):
    return "-" if pd.isna(x) else f"₹{x:,.0f}"


# --------------------------------------------------
# 8. LAYOUT: TABS
# --------------------------------------------------
tab_lead, tab_value = st.tabs(
    ["📊 Lead Scoring & Funnel", "💰 Customer Value & Retention"]
)

# --------------------------------------------------
# 8A. LEAD SCORING & FUNNEL TAB
# --------------------------------------------------
with tab_lead:
    st.markdown("### Lead Scoring & Funnel KPIs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Overall Lead Conversion Rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_pct(overall_conv)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Converted ÷ Total leads</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Conversion Rate of High-Priority Leads</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_pct(high_priority_conv)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Based on AI lead score ≥ 70</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Share of New Customers from High-Priority Leads</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_pct(share_new_from_high)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Among all converted customers</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Meetings Needed to Close a Deal</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_num(avg_meetings_close, 2)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Average meetings for converted leads</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### Lead Scoring & Funnel Visualisations")

    # 1. Conversion rate by lead score band
    bins = [0, 30, 50, 70, 85, 100]
    labels = ["0–30", "31–50", "51–70", "71–85", "86–100"]
    df["Score_Band"] = pd.cut(df["Lead_Score"], bins=bins, labels=labels, include_lowest=True)

    conv_by_band = (
        df.groupby("Score_Band")["Converted"]
        .mean()
        .reset_index()
    )
    conv_by_band["Conversion_Rate"] = conv_by_band["Converted"] * 100

    fig1 = px.bar(
        conv_by_band,
        x="Score_Band",
        y="Conversion_Rate",
        color="Conversion_Rate",
        color_continuous_scale="Blues",
        labels={"Score_Band": "Lead Score Band", "Conversion_Rate": "Conversion Rate (%)"},
        title="Conversion Rate by Lead Score Band",
    )

    # 2. Lead mix by priority level (donut)
    mix_priority = df["Priority_Level"].value_counts().reset_index()
    mix_priority.columns = ["Priority_Level", "Count"]

    fig2 = px.pie(
        mix_priority,
        names="Priority_Level",
        values="Count",
        hole=0.55,
        color="Priority_Level",
        color_discrete_map={"High": "#1b5e20", "Medium": "#ffb300", "Low": "#c62828"},
        title="Lead Mix by Priority Level",
    )

    # 3. Conversion rate by lead source
    conv_by_source = (
        df.groupby("Lead_Source")["Converted"]
        .mean()
        .reset_index()
    )
    conv_by_source["Conversion_Rate"] = conv_by_source["Converted"] * 100

    fig3 = px.bar(
        conv_by_source,
        x="Lead_Source",
        y="Conversion_Rate",
        color="Conversion_Rate",
        color_continuous_scale="Viridis",
        title="Conversion Rate by Lead Source",
        labels={"Conversion_Rate": "Conversion Rate (%)"},
    )

    # 4. Avg engagement by lead source
    eng_by_source = (
        df.groupby("Lead_Source")["Engagement_Score"]
        .mean()
        .reset_index()
    )

    fig4 = px.bar(
        eng_by_source,
        x="Lead_Source",
        y="Engagement_Score",
        color="Engagement_Score",
        color_continuous_scale="Plasma",
        title="Average Engagement Score by Lead Source",
    )

    # 5. Avg AI score by industry
    ai_by_industry = (
        df.groupby("Industry")["Lead_Score"]
        .mean()
        .reset_index()
    )

    fig5 = px.bar(
        ai_by_industry,
        x="Industry",
        y="Lead_Score",
        color="Lead_Score",
        color_continuous_scale="Tealgrn",
        title="Average AI Lead Score by Industry",
        labels={"Lead_Score": "Average Lead Score (0–100)"},
    )

    # Layout for lead tab charts
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(fig1, use_container_width=True)
    with r1c2:
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(fig3, use_container_width=True)
    with r2c2:
        st.plotly_chart(fig4, use_container_width=True)

    st.plotly_chart(fig5, use_container_width=True)


# --------------------------------------------------
# 8B. CUSTOMER VALUE & RETENTION TAB
# --------------------------------------------------
with tab_value:
    st.markdown("### Customer Value & Retention KPIs")

    c1, c2, c3, c4 = st.columns(4)
    c5, c6, _, _ = st.columns(4)

    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Predicted Lifetime Value per Customer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_rupees(avg_clv)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Mean CLV for converted customers</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">% of Customers in High-Value Segment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_pct(pct_high_value_customers)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Top CLV tier (High)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">% of Revenue from Top 20% Customers</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_pct(pct_rev_top20)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">By revenue among customers</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">% of Customers at High Churn Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_pct(pct_high_churn_customers)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Churn risk ≥ 0.60</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Best-Performing Channel Conversion Rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_pct(best_channel_conv)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-caption">Channel: {best_channel}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Engagement Score of Customers</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fmt_num(avg_engagement_customers,1)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Only for converted customers</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### Customer Value & Retention Visualisations")

    if missing_clv:
        st.info(
            "Customer value charts require CLV, Tenure_Months, Churn_Risk, and Revenue columns. "
            "Add them to your CSV to unlock full visuals."
        )
    else:
        # 6. Avg CLV by industry
        clv_industry = (
            df[converted_mask]
            .groupby("Industry")["CLV"]
            .mean()
            .reset_index()
        )

        fig6 = px.bar(
            clv_industry,
            x="Industry",
            y="CLV",
            color="CLV",
            color_continuous_scale="Sunset",
            labels={"CLV": "Average CLV (₹)"},
            title="Average CLV by Industry",
        )

        # 7. Avg CLV by tenure bucket
        tenure_bins = [0, 6, 12, 24, 36, 120]
        tenure_labels = ["0–6", "7–12", "13–24", "25–36", "37+"]

        df["Tenure_Bucket"] = pd.cut(
            df["Tenure_Months"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
        )

        clv_tenure = (
            df[converted_mask]
            .groupby("Tenure_Bucket")["CLV"]
            .mean()
            .reset_index()
        )

        fig7 = px.bar(
            clv_tenure,
            x="Tenure_Bucket",
            y="CLV",
            color="CLV",
            color_continuous_scale="RdPu",
            labels={"CLV": "Average CLV (₹)", "Tenure_Bucket": "Tenure (months)"},
            title="Average CLV by Tenure Bucket",
        )

        # 8. CLV distribution
        fig8 = px.histogram(
            df[converted_mask],
            x="CLV",
            nbins=30,
            color_discrete_sequence=["#1e88e5"],
            title="CLV Distribution for Customers",
            labels={"CLV": "Customer Lifetime Value (₹)"},
        )

        # 9. Avg time to decision by score band
        time_by_band = (
            df.groupby("Score_Band")["Decision_Time_Days"]
            .mean()
            .reset_index()
        )

        fig9 = px.line(
            time_by_band,
            x="Score_Band",
            y="Decision_Time_Days",
            markers=True,
            title="Average Time to Decision by Lead Score Band",
            labels={"Decision_Time_Days": "Avg Time to Decision (days)"},
        )

        # 10. Avg meetings by lead source
        meetings_by_source = (
            df.groupby("Lead_Source")["Meetings"]
            .mean()
            .reset_index()
        )

        fig10 = px.bar(
            meetings_by_source,
            x="Lead_Source",
            y="Meetings",
            color="Meetings",
            color_continuous_scale="YlGnBu",
            title="Average Meetings by Lead Source",
        )

        # Layout for value tab charts
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(fig6, use_container_width=True)
        with r1c2:
            st.plotly_chart(fig7, use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.plotly_chart(fig8, use_container_width=True)
        with r2c2:
            st.plotly_chart(fig9, use_container_width=True)

        st.plotly_chart(fig10, use_container_width=True)
