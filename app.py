import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# PAGE CONFIG & BASIC STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Driven B2B Lead Scoring & Customer Value Dashboard",
    layout="wide",
)

# Custom CSS for colourful cards and nicer layout
st.markdown(
    """
    <style>
        .main {
            background: #0f172a;
            color: #f9fafb;
        }
        h1, h2, h3, h4 {
            color: #f9fafb !important;
            font-weight: 700 !important;
        }
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
            background: linear-gradient(135deg, #1d4ed8, #22c55e);
            color: white;
            box-shadow: 0 10px 30px rgba(15,23,42,0.45);
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.85;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
        }
        .submetric {
            font-size: 0.8rem;
            opacity: 0.75;
        }
        .section-box {
            background: #020617;
            border-radius: 1rem;
            padding: 1.2rem 1.4rem 0.6rem 1.4rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.65);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #020617;
            border-radius: 999px;
            padding: 0.35rem 1.1rem;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #1e293b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SAMPLE DATA GENERATORS (only used if user doesn't upload)
# -----------------------------------------------------------------------------
@st.cache_data
def load_sample_leads() -> pd.DataFrame:
    return pd.read_csv("lead_scoring_data.csv")

@st.cache_data
def load_sample_customers() -> pd.DataFrame:
    return pd.read_csv("customer_clv_data.csv")


# -----------------------------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_lead_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_sample_leads()

@st.cache_data
def load_customer_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return load_sample_customers()


# -----------------------------------------------------------------------------
# LEAD SCORING MODEL
# -----------------------------------------------------------------------------
FEATURE_COLS = [
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
]

CAT_COLS = ["Industry", "Location", "Lead_Source", "Product_Interest"]
TARGET_COL = "Converted"

@st.cache_resource
def train_lead_model(leads: pd.DataFrame):
    # Guard in case user uploads incomplete data
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in leads.columns]
    if missing:
        raise ValueError(f"Missing required columns in leads CSV: {missing}")

    X = leads[FEATURE_COLS]
    y = leads[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            # NOTE: no 'sparse' / 'sparse_output' arg here → works on all sklearn versions
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    return pipe


def score_leads(leads: pd.DataFrame, model) -> pd.DataFrame:
    scored = leads.copy()
    proba = model.predict_proba(scored[FEATURE_COLS])[:, 1]
    scored["Lead_Score"] = (proba * 100).round(1)

    # Priority buckets
    conditions = [
        scored["Lead_Score"] >= 80,
        scored["Lead_Score"].between(50, 80, inclusive="left"),
        scored["Lead_Score"] < 50,
    ]
    choices = ["High", "Medium", "Low"]
    scored["Priority"] = np.select(conditions, choices, default="Medium")

    # Score bands (for charts)
    bins = [0, 20, 40, 60, 80, 101]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    scored["Lead_Score_Band"] = pd.cut(
        scored["Lead_Score"], bins=bins, labels=labels, include_lowest=True
    )
    return scored


# -----------------------------------------------------------------------------
# CUSTOMER VALUE / CLV HELPERS
# -----------------------------------------------------------------------------
def add_customer_segments(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()
    clv_col = "CLV_Predicted_Future_Value_INR_Lakhs"
    if clv_col not in df.columns:
        # if user removes column accidentally, create safe dummy
        df[clv_col] = df.get("Annual_Spend_INR_Lakhs", 100)

    # High-value customers = top 30% by predicted CLV
    threshold = df[clv_col].quantile(0.70)
    df["High_Value_Flag"] = (df[clv_col] >= threshold).astype(int)

    # Simple churn risk proxy (higher = worse)
    df["Churn_Risk_Score"] = (
        0.5 * (df.get("Support_Tickets", 0).fillna(0) / (df["Support_Tickets"].max() or 1))
        + 0.3 * (1 - (df.get("Satisfaction_Score_1_10", 7).fillna(7) / 10))
        + 0.2 * (1 - (df.get("Payment_Timeliness_Percent", 90).fillna(90) / 100))
    )
    df["High_Churn_Risk_Flag"] = (df["Churn_Risk_Score"] >= df["Churn_Risk_Score"].median()).astype(int)

    # Tenure buckets
    tenure = df.get("Tenure_Months", pd.Series([12] * len(df)))
    bins = [0, 6, 12, 24, 120]
    labels = ["0-6", "6-12", "12-24", "24+"]

    df["Tenure_Bucket"] = pd.cut(tenure, bins=bins, labels=labels, include_lowest=True)

    return df


# -----------------------------------------------------------------------------
# KPI CALCULATIONS
# -----------------------------------------------------------------------------
def compute_lead_kpis(scored_leads: pd.DataFrame):
    converted = scored_leads[scored_leads[TARGET_COL] == 1]

    overall_conv = converted.shape[0] / scored_leads.shape[0] if len(scored_leads) else 0
    high_leads = scored_leads[scored_leads["Priority"] == "High"]
    high_conv = (
        high_leads[high_leads[TARGET_COL] == 1].shape[0] / high_leads.shape[0]
        if len(high_leads)
        else 0
    )
    share_new_from_high = (
        converted[converted["Priority"] == "High"].shape[0] / converted.shape[0]
        if len(converted)
        else 0
    )
    avg_meetings_to_close = converted["Meetings"].mean() if len(converted) else 0

    return {
        "overall_conv": overall_conv,
        "high_conv": high_conv,
        "share_new_from_high": share_new_from_high,
        "avg_meetings": avg_meetings_to_close,
    }


def compute_customer_kpis(customers: pd.DataFrame, scored_leads: pd.DataFrame):
    clv_col = "CLV_Predicted_Future_Value_INR_Lakhs"
    avg_clv = customers[clv_col].mean() if clv_col in customers.columns else 0

    pct_high_value = customers["High_Value_Flag"].mean() if len(customers) else 0

    # Revenue from top 20% customers
    if len(customers):
        n_top = max(1, int(0.2 * len(customers)))
        top_customers = customers.nlargest(n_top, clv_col)
        pct_revenue_top20 = top_customers[clv_col].sum() / customers[clv_col].sum()
    else:
        pct_revenue_top20 = 0

    pct_high_churn_risk = customers["High_Churn_Risk_Flag"].mean() if len(customers) else 0

    # Best-performing channel (Lead_Source) based on conversion rate
    channel_conv = (
        scored_leads.groupby("Lead_Source")[TARGET_COL].mean().sort_values(ascending=False)
        if len(scored_leads)
        else pd.Series(dtype=float)
    )
    if len(channel_conv):
        best_channel = channel_conv.index[0]
        best_channel_rate = channel_conv.iloc[0]
    else:
        best_channel = "—"
        best_channel_rate = 0.0

    # Average engagement score of customers (converted leads)
    converted = scored_leads[scored_leads[TARGET_COL] == 1]
    avg_engagement_customers = (
        converted["Engagement_Score"].mean() if len(converted) else 0
    )

    return {
        "avg_clv": avg_clv,
        "pct_high_value": pct_high_value,
        "pct_revenue_top20": pct_revenue_top20,
        "pct_high_churn_risk": pct_high_churn_risk,
        "best_channel": best_channel,
        "best_channel_rate": best_channel_rate,
        "avg_engagement_customers": avg_engagement_customers,
    }


# -----------------------------------------------------------------------------
# UI – HEADER
# -----------------------------------------------------------------------------
st.title("AI-Driven B2B Lead Scoring & Customer Value Dashboard")
st.caption(
    "Use this dashboard to **prioritize B2B leads** and **understand customer value**. "
    "Upload your own CSVs or explore using the sample data."
)

with st.expander("Data inputs", expanded=True):
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        lead_file = st.file_uploader(
            "Upload Leads CSV (optional)",
            type=["csv"],
            key="lead_file",
            help="Must contain columns like Lead_ID, Industry, Company_Size, Annual_Revenue_INR_Lakhs, "
                 "Location, Website_Visits, Email_Clicks, Meetings, Lead_Source, Engagement_Score, "
                 "Product_Interest, Decision_Time_Days, Converted.",
        )
    with col_u2:
        cust_file = st.file_uploader(
            "Upload Customer CLV CSV (optional)",
            type=["csv"],
            key="cust_file",
            help="Must contain CLV_Predicted_Future_Value_INR_Lakhs and other customer attributes.",
        )

# -----------------------------------------------------------------------------
# LOAD DATA & TRAIN MODEL
# -----------------------------------------------------------------------------
try:
    leads_raw = load_lead_data(lead_file)
    customers_raw = load_customer_data(cust_file)

    model = train_lead_model(leads_raw)
    leads_scored = score_leads(leads_raw, model)
    customers = add_customer_segments(customers_raw)

except Exception as e:
    st.error(
        "There was a problem loading your data or training the model. "
        "Please check that your CSVs contain the required columns."
    )
    st.exception(e)
    st.stop()

lead_kpis = compute_lead_kpis(leads_scored)
customer_kpis = compute_customer_kpis(customers, leads_scored)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📈 Lead Scoring & Funnel", "💰 Customer Value & Retention"])


# -----------------------------------------------------------------------------
# TAB 1 – LEAD SCORING & FUNNEL
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Summary KPIs")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Lead Conversion Rate (%)</div>
                <div class="metric-value">{lead_kpis['overall_conv']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Conversion Rate of High-Priority Leads (%)</div>
                <div class="metric-value">{lead_kpis['high_conv']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Share of New Customers from High-Priority Leads (%)</div>
                <div class="metric-value">{lead_kpis['share_new_from_high']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Meetings Needed to Close a Deal</div>
                <div class="metric-value">{lead_kpis['avg_meetings']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Charts for leads -----------------------------------------------------
    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### Lead Quality & Conversion")

        col_a, col_b = st.columns(2)

        # 1. Conversion rate by lead score band
        with col_a:
            conv_by_band = (
                leads_scored.groupby("Lead_Score_Band")[TARGET_COL]
                .mean()
                .reset_index()
                .rename(columns={TARGET_COL: "Conversion_Rate"})
            )
            fig1 = px.bar(
                conv_by_band,
                x="Lead_Score_Band",
                y="Conversion_Rate",
                labels={"Lead_Score_Band": "Lead Score Band", "Conversion_Rate": "Conversion Rate"},
                color="Conversion_Rate",
                color_continuous_scale="Blues",
                text=conv_by_band["Conversion_Rate"].map(lambda x: f"{x*100:.1f}%"),
            )
            fig1.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig1, use_container_width=True)

        # 2. Lead mix by priority level (donut)
        with col_b:
            mix_priority = (
                leads_scored["Priority"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "Priority", "Priority": "Count"})
            )
            fig2 = px.pie(
                mix_priority,
                names="Priority",
                values="Count",
                hole=0.5,
                color="Priority",
                color_discrete_map={"High": "#ef4444", "Medium": "#eab308", "Low": "#22c55e"},
            )
            fig2.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Second row of lead charts
    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### Channel & Engagement Performance")

        col_c, col_d = st.columns(2)

        # 3. Conversion rate by lead source
        with col_c:
            conv_by_source = (
                leads_scored.groupby("Lead_Source")[TARGET_COL]
                .mean()
                .reset_index()
                .rename(columns={TARGET_COL: "Conversion_Rate"})
            )
            fig3 = px.bar(
                conv_by_source,
                x="Lead_Source",
                y="Conversion_Rate",
                color="Conversion_Rate",
                color_continuous_scale="Viridis",
                text=conv_by_source["Conversion_Rate"].map(lambda x: f"{x*100:.1f}%"),
            )
            fig3.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Lead Source",
                yaxis_title="Conversion Rate",
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 4. Avg engagement by lead source
        with col_d:
            avg_eng_by_source = (
                leads_scored.groupby("Lead_Source")["Engagement_Score"]
                .mean()
                .reset_index()
                .rename(columns={"Engagement_Score": "Avg_Engagement"})
            )
            fig4 = px.bar(
                avg_eng_by_source,
                x="Lead_Source",
                y="Avg_Engagement",
                color="Avg_Engagement",
                color_continuous_scale="Plasma",
            )
            fig4.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Lead Source",
                yaxis_title="Average Engagement Score",
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Third row
    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### Industry & Cycle Time")

        col_e, col_f = st.columns(2)

        # 5. Avg AI score by industry
        with col_e:
            avg_score_by_industry = (
                leads_scored.groupby("Industry")["Lead_Score"]
                .mean()
                .reset_index()
                .rename(columns={"Lead_Score": "Avg_Lead_Score"})
            )
            fig5 = px.bar(
                avg_score_by_industry,
                x="Industry",
                y="Avg_Lead_Score",
                color="Avg_Lead_Score",
                color_continuous_scale="Turbo",
            )
            fig5.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Industry",
                yaxis_title="Average AI Lead Score",
            )
            st.plotly_chart(fig5, use_container_width=True)

        # 9. Avg time to decision by score band
        with col_f:
            time_by_band = (
                leads_scored.groupby("Lead_Score_Band")["Decision_Time_Days"]
                .mean()
                .reset_index()
                .rename(columns={"Decision_Time_Days": "Avg_Decision_Time_Days"})
            )
            fig9 = px.line(
                time_by_band,
                x="Lead_Score_Band",
                y="Avg_Decision_Time_Days",
                markers=True,
            )
            fig9.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Lead Score Band",
                yaxis_title="Avg Time to Decision (days)",
            )
            st.plotly_chart(fig9, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Fourth row
    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### Meetings Load by Industry")

        # 10. Avg meetings by industry
        avg_meetings_by_industry = (
            leads_scored.groupby("Industry")["Meetings"]
            .mean()
            .reset_index()
            .rename(columns={"Meetings": "Avg_Meetings"})
        )
        fig10 = px.bar(
            avg_meetings_by_industry,
            x="Industry",
            y="Avg_Meetings",
            color="Avg_Meetings",
            color_continuous_scale="Cividis",
        )
        fig10.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            xaxis_title="Industry",
            yaxis_title="Average Meetings per Lead",
        )
        st.plotly_chart(fig10, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2 – CUSTOMER VALUE & RETENTION
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Customer Value & Retention KPIs")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Predicted Lifetime Value per Customer (₹ Lakhs)</div>
                <div class="metric-value">{customer_kpis['avg_clv']:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">% of Customers in High-Value Segment</div>
                <div class="metric-value">{customer_kpis['pct_high_value']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">% of Revenue from Top 20% Customers</div>
                <div class="metric-value">{customer_kpis['pct_revenue_top20']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">% of Customers at High Churn Risk</div>
                <div class="metric-value">{customer_kpis['pct_high_churn_risk']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Best-Performing Channel Conversion Rate (%)</div>
                <div class="metric-value">{customer_kpis['best_channel_rate']*100:.1f}%</div>
                <div class="submetric">Channel: {customer_kpis['best_channel']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Engagement Score of Customers</div>
                <div class="metric-value">{customer_kpis['avg_engagement_customers']:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- CLV charts (4 of the 10 overall charts) ------------------------------
    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### CLV by Industry & Tenure")

        col1, col2 = st.columns(2)

        # 6. Avg CLV by industry
        with col1:
            clv_by_industry = (
                customers.groupby("Industry")["CLV_Predicted_Future_Value_INR_Lakhs"]
                .mean()
                .reset_index()
                .rename(columns={"CLV_Predicted_Future_Value_INR_Lakhs": "Avg_CLV"})
            )
            fig6 = px.bar(
                clv_by_industry,
                x="Industry",
                y="Avg_CLV",
                color="Avg_CLV",
                color_continuous_scale="Magma",
            )
            fig6.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Industry",
                yaxis_title="Average CLV (₹ Lakhs)",
            )
            st.plotly_chart(fig6, use_container_width=True)

        # 7. Avg CLV by tenure bucket
        with col2:
            clv_by_tenure = (
                customers.groupby("Tenure_Bucket")["CLV_Predicted_Future_Value_INR_Lakhs"]
                .mean()
                .reset_index()
                .rename(columns={"CLV_Predicted_Future_Value_INR_Lakhs": "Avg_CLV"})
            )
            fig7 = px.bar(
                clv_by_tenure,
                x="Tenure_Bucket",
                y="Avg_CLV",
                color="Avg_CLV",
                color_continuous_scale="Inferno",
            )
            fig7.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Tenure Bucket (months)",
                yaxis_title="Average CLV (₹ Lakhs)",
            )
            st.plotly_chart(fig7, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown("### CLV Distribution & Risk")

        col3, col4 = st.columns(2)

        # 8. CLV distribution
        with col3:
            fig8 = px.histogram(
                customers,
                x="CLV_Predicted_Future_Value_INR_Lakhs",
                nbins=30,
                color_discrete_sequence=["#22c55e"],
            )
            fig8.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="Predicted CLV (₹ Lakhs)",
                yaxis_title="Number of Customers",
            )
            st.plotly_chart(fig8, use_container_width=True)

        # Simple chart for churn risk vs CLV (optional but insightful)
        with col4:
            risk_df = customers.copy()
            fig_risk = px.box(
                risk_df,
                x="High_Churn_Risk_Flag",
                y="CLV_Predicted_Future_Value_INR_Lakhs",
                color="High_Churn_Risk_Flag",
                color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            )
            fig_risk.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                xaxis_title="High Churn Risk (0 = No, 1 = Yes)",
                yaxis_title="Predicted CLV (₹ Lakhs)",
                showlegend=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
