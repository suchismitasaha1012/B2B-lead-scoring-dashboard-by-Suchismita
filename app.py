import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px

# ------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI B2B Lead Scoring & Customer Value Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Give breathing space at the top so title is never cut */
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 2rem !important;
    }

    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        flex: 1 1 220px;
        background: linear-gradient(135deg, #eff6ff, #e0f2fe);
        border-radius: 0.9rem;
        padding: 0.9rem 1.05rem 0.95rem 1.05rem;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
        color: #0f172a;
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #0f172a;
    }

    .metric-caption {
        font-size: 0.73rem;
        color: #6b7280;
    }

    .chart-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown(
    '<div class="big-title">AI-Driven B2B Lead Scoring & Customer Value Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Upload a single CSV containing lead and customer attributes to explore AI scores, funnel efficiency, and value-driven insights."
    "</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload combined B2B lead & customer CSV",
    type=["csv"],
    help=(
        "Expected useful columns (any missing ones will simply disable the related KPIs): "
        "Lead_ID, Industry, Company_Size, Annual_Revenue_INR_Lakhs, Location, Website_Visits, "
        "Email_Clicks, Meetings, Lead_Source, Engagement_Score, Product_Interest, "
        "Decision_Time_Days, Converted, CLV, Revenue, Tenure_Months, Churn_Risk."
    ),
)

if uploaded_file is None:
    st.info(
        "⬆️ Upload your CSV to see AI scoring, funnel metrics, and customer value charts.\n\n"
        "Make sure you at least have **Converted** (0/1) so the AI model can learn."
    )
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------------------------------------------------
# BASIC COLUMN SANITY
# ------------------------------------------------------------
required_for_model = [
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

missing_for_model = [c for c in required_for_model if c not in df.columns]
if missing_for_model:
    st.error(
        "The following columns are required to train the AI lead-scoring model and are missing "
        f"from your file: **{', '.join(missing_for_model)}**.\n\n"
        "Please add them to your dataset and re-upload."
    )
    st.stop()

# ------------------------------------------------------------
# AI LEAD SCORING MODEL
# ------------------------------------------------------------
model_df = df.copy()

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

X = model_df[numeric_cols + cat_cols]
y = model_df["Converted"].astype(int)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)

rf = RandomForestClassifier(
    n_estimators=250,
    random_state=42,
    class_weight="balanced_subsample",
    max_depth=None,
    min_samples_leaf=2,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = rf  # we'll manually transform then fit; to keep things simple
X_train_transformed = preprocess.fit_transform(X_train)
rf.fit(X_train_transformed, y_train)

# Score all leads
X_all_transformed = preprocess.transform(X)
ai_score_proba = rf.predict_proba(X_all_transformed)[:, 1]
df["AI_Score"] = ai_score_proba * 100  # 0-100 scale

# Priority bands
def score_band(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


df["Priority_Band"] = df["AI_Score"].apply(score_band)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def pct(x):
    return float(x) * 100.0


def safe_pct(numer, denom):
    if denom == 0:
        return None
    return pct(numer / denom)


def format_value(v, fmt="{:,.1f}%"):
    if v is None or (isinstance(v, (float, int)) and np.isnan(v)):
        return "—"
    return fmt.format(v)


def metric_card(label, value, caption=None, fmt="{:,.1f}%"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{format_value(value, fmt)}</div>
            <div class="metric-caption">{caption or ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# KPI CALCULATIONS
# ------------------------------------------------------------
total_leads = len(df)
converted = df["Converted"].sum()
non_converted = total_leads - converted

overall_conv = safe_pct(converted, total_leads)

high_band = df[df["Priority_Band"] == "High"]
high_conv = safe_pct(high_band["Converted"].sum(), len(high_band)) if len(high_band) else None

converted_df = df[df["Converted"] == 1]
converted_high = converted_df[converted_df["Priority_Band"] == "High"]
share_new_from_high = safe_pct(len(converted_high), len(converted_df)) if len(converted_df) else None

avg_meetings_closed = (
    converted_df["Meetings"].mean() if len(converted_df) else None
)

# Customer-value-related columns (optional)
has_clv = "CLV" in df.columns
has_revenue = "Revenue" in df.columns
has_tenure = "Tenure_Months" in df.columns
has_churn = "Churn_Risk" in df.columns

if has_clv:
    avg_clv = df["CLV"].mean()
    high_value_threshold = df["CLV"].quantile(0.75)
    high_value_customers = df[df["CLV"] >= high_value_threshold]
    pct_high_value_customers = safe_pct(len(high_value_customers), total_leads)

    if has_revenue:
        # If you have a separate Revenue column, use that;
        # otherwise CLV is used as a value proxy for revenue contribution charts.
        value_col = "Revenue"
    else:
        value_col = "CLV"

    df_sorted_value = df.sort_values(value_col, ascending=False)
    top_20_count = max(1, int(0.2 * total_leads))
    top_20_customers = df_sorted_value.head(top_20_count)
    pct_revenue_top20 = safe_pct(
        top_20_customers[value_col].sum(), df[value_col].sum()
    )
else:
    avg_clv = None
    pct_high_value_customers = None
    pct_revenue_top20 = None

if has_churn:
    pct_high_churn_risk = safe_pct(
        (df["Churn_Risk"] > 0).sum(), total_leads
    )
else:
    pct_high_churn_risk = None

# Channel conversion metrics
channel_conv = (
    df.groupby("Lead_Source")["Converted"].mean().sort_values(ascending=False)
)
if len(channel_conv) > 0:
    best_channel = channel_conv.index[0]
    best_channel_rate = pct(channel_conv.iloc[0])
    best_channel_label = f"{best_channel} ({best_channel_rate:.1f}%)"
else:
    best_channel_label = None

avg_engagement_customers = df["Engagement_Score"].mean()

# ------------------------------------------------------------
# TABS: KPI CARDS
# ------------------------------------------------------------
tab_lead, tab_value = st.tabs(["📈 Lead Scoring & Funnel", "💰 Customer Value & Retention"])
# ---------------------------------------------------------
# 🔝 TOP 10 PRIORITY LEADS (BASED ON MODEL PREDICTION)
# ---------------------------------------------------------

st.markdown("### 🔝 Top 10 Priority Leads")

# if you named the scored dataframe differently, change `leads_scored`
df_scored = leads_scored.copy()

# make sure the AI score column name below matches your code
AI_COL = "AI_Score"          # e.g. "AI_Score", "Lead_Score", "score" etc.

if AI_COL not in df_scored.columns:
    st.info("AI score not found. Make sure the scoring step runs before this block.")
else:
    # sort by AI score (highest first) and pick top 10
    top10_cols = [
        "Lead_ID",
        "Industry",
        "Lead_Source",
        "Company_Size",
        "Annual_Revenue_INR_Lakhs",
        "Engagement_Score",
        "Meetings",
        AI_COL,
    ]

    # keep only columns that actually exist (avoids key errors)
    top10_cols = [c for c in top10_cols if c in df_scored.columns]

    top10 = (
        df_scored
        .sort_values(AI_COL, ascending=False)
        .head(10)[top10_cols]
        .rename(columns={AI_COL: "Predicted_Conversion_Probability"})
    )

    st.dataframe(
        top10.style.format({
            "Annual_Revenue_INR_Lakhs": "{:,.0f}",
            "Predicted_Conversion_Probability": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab_lead:
    st.markdown('<div class="section-header">Lead Scoring & Funnel KPIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)

    metric_card(
        "Overall Lead Conversion Rate",
        overall_conv,
        caption="Converted ÷ Total leads",
    )
    metric_card(
        "Conversion Rate of High-Priority Leads",
        high_conv,
        caption="Among AI-scored High-priority leads",
    )
    metric_card(
        "Share of New Customers from High-Priority Leads",
        share_new_from_high,
        caption="High-priority customers ÷ All new customers",
    )
    metric_card(
        "Average Meetings Needed to Close a Deal",
        avg_meetings_closed,
        caption="Mean meetings for converted leads",
        fmt="{:,.2f}",
    )

    st.markdown('</div>', unsafe_allow_html=True)

with tab_value:
    st.markdown('<div class="section-header">Customer Value & Retention KPIs</div>', unsafe_allow_html=True)

    if not (has_clv or has_revenue or has_churn):
        st.warning(
            "To fully populate value & retention KPIs, add these columns to your dataset: "
            "`CLV`, `Revenue`, `Tenure_Months`, `Churn_Risk`.\n\n"
            "Lead-funnel KPIs and non-CLV charts are still computed from your current file."
        )

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)

    metric_card(
        "Average Predicted Lifetime Value per Customer (₹)",
        avg_clv,
        caption="Mean CLV across all leads/customers",
        fmt="₹{:,.0f}",
    )
    metric_card(
        "% of Customers in High-Value Segment",
        pct_high_value_customers,
        caption="Customers with CLV ≥ 75th percentile",
    )
    metric_card(
        "% of Revenue from Top 20% Customers",
        pct_revenue_top20,
        caption="Share of value from top 20% by CLV / Revenue",
    )
    metric_card(
        "% of Customers at High Churn Risk",
        pct_high_churn_risk,
        caption="Churn_Risk > 0",
    )
    metric_card(
        "Best-Performing Channel Conversion Rate",
        best_channel_rate if best_channel_label else None,
        caption=best_channel_label or "No channel information",
    )
    metric_card(
        "Average Engagement Score of Customers",
        avg_engagement_customers,
        caption="Mean Engagement_Score across all leads",
        fmt="{:,.1f}",
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# CHARTS (10 – NO TABLES)
# ------------------------------------------------------------
st.markdown('<div class="section-header">Lead & Value Visualizations</div>', unsafe_allow_html=True)

# 1 & 2: Conversion rate by score band, lead mix by priority level
col1, col2 = st.columns(2)
with col1:
    conv_by_band = (
        df.groupby("Priority_Band")["Converted"]
        .mean()
        .reindex(["High", "Medium", "Low"])
        .dropna()
        * 100
    )
    fig1 = px.bar(
        conv_by_band,
        x=conv_by_band.index,
        y=conv_by_band.values,
        labels={"x": "Priority Band", "y": "Conversion Rate (%)"},
        color=conv_by_band.index,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig1.update_layout(title="Conversion Rate by Lead Score Band", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    band_mix = df["Priority_Band"].value_counts().reindex(["High", "Medium", "Low"]).dropna()
    fig2 = px.pie(
        values=band_mix.values,
        names=band_mix.index,
        hole=0.5,
        color=band_mix.index,
        color_discrete_sequence=px.colors.sequential.Teal_r,
    )
    fig2.update_layout(title="Lead Mix by Priority Level")
    st.plotly_chart(fig2, use_container_width=True)

# 3 & 4: Conversion rate by lead source, avg engagement by lead source
col3, col4 = st.columns(2)
with col3:
    conv_by_source = df.groupby("Lead_Source")["Converted"].mean() * 100
    conv_by_source = conv_by_source.sort_values(ascending=False)
    fig3 = px.bar(
        conv_by_source,
        x=conv_by_source.index,
        y=conv_by_source.values,
        labels={"x": "Lead Source", "y": "Conversion Rate (%)"},
        color=conv_by_source.values,
        color_continuous_scale="Blues",
    )
    fig3.update_layout(title="Conversion Rate by Lead Source", coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    eng_by_source = df.groupby("Lead_Source")["Engagement_Score"].mean()
    eng_by_source = eng_by_source.sort_values(ascending=False)
    fig4 = px.bar(
        eng_by_source,
        x=eng_by_source.index,
        y=eng_by_source.values,
        labels={"x": "Lead Source", "y": "Avg Engagement Score"},
        color=eng_by_source.values,
        color_continuous_scale="Teal",
    )
    fig4.update_layout(title="Average Engagement by Lead Source", coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

# 5 & 6: Avg AI score by industry, Avg CLV by industry
col5, col6 = st.columns(2)
with col5:
    ai_by_industry = df.groupby("Industry")["AI_Score"].mean().sort_values(ascending=False)
    fig5 = px.bar(
        ai_by_industry,
        x=ai_by_industry.index,
        y=ai_by_industry.values,
        labels={"x": "Industry", "y": "Avg AI Lead Score"},
        color=ai_by_industry.values,
        color_continuous_scale="Purples",
    )
    fig5.update_layout(title="Average AI Lead Score by Industry", coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    if has_clv:
        clv_by_industry = df.groupby("Industry")["CLV"].mean().sort_values(ascending=False)
        fig6 = px.bar(
            clv_by_industry,
            x=clv_by_industry.index,
            y=clv_by_industry.values,
            labels={"x": "Industry", "y": "Avg CLV (₹)"},
            color=clv_by_industry.values,
            color_continuous_scale="Oranges",
        )
        fig6.update_layout(title="Average CLV by Industry", coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Add a **CLV** column to see *Average CLV by Industry*.")

# 7 & 8: Avg CLV by tenure bucket, CLV distribution
col7, col8 = st.columns(2)
with col7:
    if has_clv and has_tenure:
        bins = [-1, 6, 12, 24, np.inf]
        labels = ["< 6 months", "6–12 months", "12–24 months", "24+ months"]
        df["Tenure_Bucket"] = pd.cut(df["Tenure_Months"], bins=bins, labels=labels)
        clv_by_tenure = df.groupby("Tenure_Bucket")["CLV"].mean()
        fig7 = px.bar(
            clv_by_tenure,
            x=clv_by_tenure.index.astype(str),
            y=clv_by_tenure.values,
            labels={"x": "Tenure Bucket", "y": "Avg CLV (₹)"},
            color=clv_by_tenure.values,
            color_continuous_scale="Greens",
        )
        fig7.update_layout(title="Average CLV by Tenure Bucket", coloraxis_showscale=False)
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Add **CLV** and **Tenure_Months** columns to see *CLV by Tenure Bucket*.")

with col8:
    if has_clv:
        fig8 = px.histogram(
            df,
            x="CLV",
            nbins=30,
            marginal="box",
            labels={"CLV": "Customer Lifetime Value (₹)"},
            color_discrete_sequence=["#0ea5e9"],
        )
        fig8.update_layout(title="CLV Distribution")
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Add a **CLV** column to see the *CLV Distribution* chart.")

# 9 & 10: Avg time to decision by score band, Avg meetings by industry
col9, col10 = st.columns(2)
with col9:
    time_by_band = (
        df.groupby("Priority_Band")["Decision_Time_Days"]
        .mean()
        .reindex(["High", "Medium", "Low"])
        .dropna()
    )
    fig9 = px.bar(
        time_by_band,
        x=time_by_band.index,
        y=time_by_band.values,
        labels={"x": "Priority Band", "y": "Avg Time to Decision (days)"},
        color=time_by_band.values,
        color_continuous_scale="RdPu",
    )
    fig9.update_layout(title="Average Time to Decision by Score Band", coloraxis_showscale=False)
    st.plotly_chart(fig9, use_container_width=True)

with col10:
    meetings_by_industry = df.groupby("Industry")["Meetings"].mean().sort_values(ascending=False)
    fig10 = px.bar(
        meetings_by_industry,
        x=meetings_by_industry.index,
        y=meetings_by_industry.values,
        labels={"x": "Industry", "y": "Avg Meetings per Lead"},
        color=meetings_by_industry.values,
        color_continuous_scale="YlGnBu",
    )
    fig10.update_layout(title="Average Meetings by Industry", coloraxis_showscale=False)
    st.plotly_chart(fig10, use_container_width=True)
