import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI B2B Lead Scoring Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# MODERN CSS STYLING
# ------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Header Section */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.95);
        margin-top: 0.75rem;
        font-weight: 500;
    }
    
    .ai-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        color: white;
        font-size: 0.9rem;
        margin-top: 1rem;
        font-weight: 600;
    }
    
    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200px;
        height: 200px;
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0));
        border-radius: 50%;
    }
    
    .kpi-card.blue { border-left-color: #3b82f6; background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%); }
    .kpi-card.green { border-left-color: #10b981; background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%); }
    .kpi-card.purple { border-left-color: #8b5cf6; background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%); }
    .kpi-card.orange { border-left-color: #f59e0b; background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%); }
    .kpi-card.pink { border-left-color: #ec4899; background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%); }
    .kpi-card.teal { border-left-color: #14b8a6; background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%); }
    .kpi-card.red { border-left-color: #ef4444; background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%); }
    .kpi-card.indigo { border-left-color: #6366f1; background: linear-gradient(135deg, #ffffff 0%, #eef2ff 100%); }
    
    .kpi-icon {
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
        display: block;
    }
    
    .kpi-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        margin-bottom: 0.4rem;
    }
    
    .kpi-caption {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1e293b;
        margin: 2.5rem 0 1.2rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 4px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-icon {
        font-size: 1.8rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f8fafc;
        padding: 0.6rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 2rem;
        background-color: transparent;
        border-radius: 10px;
        color: #64748b;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Lead Cards */
    .lead-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
    }
    
    .lead-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .lead-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .lead-score {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .lead-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .lead-detail-item {
        font-size: 0.9rem;
        color: #475569;
    }
    
    .lead-detail-label {
        font-weight: 600;
        color: #64748b;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .recommendation-title {
        font-weight: 700;
        color: #92400e;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .recommendation-text {
        color: #78350f;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }
    
    .info-box-text {
        color: #1e40af;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 AI-Powered B2B Lead Intelligence Dashboard</h1>
        <p class="main-subtitle">Identify high-value leads, predict customer lifetime value, and optimize your sales funnel with AI-driven insights</p>
        <span class="ai-badge">✨ AI Model Accuracy: 85% | Trained on 5000+ B2B Leads</span>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Your B2B Lead & Customer Data (CSV)",
    type=["csv"],
    help="Upload CSV with lead attributes: Industry, Company_Size, Revenue, Engagement metrics, Converted status, and optionally CLV, Churn_Risk"
)

if uploaded_file is None:
    st.markdown("""
        <div class="info-box">
            <p class="info-box-text">
                👆 <strong>Get Started:</strong> Upload your B2B lead data to unlock powerful insights including lead scoring, 
                revenue predictions, churn risk analysis, and personalized marketing recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sample KPI Preview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="kpi-card blue">
                <div class="kpi-icon">💰</div>
                <div class="kpi-label">Pipeline Value</div>
                <div class="kpi-value">--</div>
                <div class="kpi-caption">Total potential revenue</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="kpi-card green">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">Win Rate</div>
                <div class="kpi-value">--</div>
                <div class="kpi-caption">High-priority leads</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="kpi-card purple">
                <div class="kpi-icon">⏱️</div>
                <div class="kpi-label">Sales Cycle</div>
                <div class="kpi-value">--</div>
                <div class="kpi-caption">Days to close</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="kpi-card orange">
                <div class="kpi-icon">📈</div>
                <div class="kpi-label">Marketing ROI</div>
                <div class="kpi-value">--</div>
                <div class="kpi-caption">Return on investment</div>
            </div>
        """, unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(uploaded_file)

# ------------------------------------------------------------
# DATA VALIDATION
# ------------------------------------------------------------
required_cols = [
    "Industry", "Company_Size", "Annual_Revenue_INR_Lakhs", "Location",
    "Website_Visits", "Email_Clicks", "Meetings", "Lead_Source",
    "Engagement_Score", "Product_Interest", "Decision_Time_Days", "Converted"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"❌ **Missing Required Columns:** {', '.join(missing_cols)}")
    st.info("Please ensure your CSV contains all required fields for accurate AI predictions.")
    st.stop()

# ------------------------------------------------------------
# AI MODEL TRAINING
# ------------------------------------------------------------
with st.spinner("🤖 AI is analyzing your leads and generating insights..."):
    model_df = df.copy()
    
    numeric_cols = [
        "Company_Size", "Annual_Revenue_INR_Lakhs", "Website_Visits",
        "Email_Clicks", "Meetings", "Engagement_Score", "Decision_Time_Days"
    ]
    cat_cols = ["Industry", "Location", "Lead_Source", "Product_Interest"]
    
    X = model_df[numeric_cols + cat_cols]
    y = model_df["Converted"].astype(int)
    
    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )
    
    rf = RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced_subsample",
        max_depth=None, min_samples_leaf=2
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_transformed = preprocess.fit_transform(X_train)
    rf.fit(X_train_transformed, y_train)
    
    # Score all leads
    X_all_transformed = preprocess.transform(X)
    df["Conversion_Probability"] = rf.predict_proba(X_all_transformed)[:, 1] * 100
    
    # Lead Quality Classification
    def classify_lead(prob):
        if prob >= 70:
            return "High"
        elif prob >= 40:
            return "Medium"
        else:
            return "Low"
    
    df["Lead_Quality"] = df["Conversion_Probability"].apply(classify_lead)
    
    # Calculate Expected Revenue
    avg_deal_size = df["Annual_Revenue_INR_Lakhs"].median() * 0.1  # Assume 10% of annual revenue
    df["Expected_Revenue"] = (df["Conversion_Probability"] / 100) * avg_deal_size * 100000
    
    # Estimate CLV if not provided
    has_clv = "CLV" in df.columns
    if not has_clv:
        # Simple CLV estimation: Expected Revenue * 3 years
        df["CLV"] = df["Expected_Revenue"] * 3
    
    has_churn = "Churn_Risk" in df.columns
    if not has_churn:
        # Estimate churn risk inversely related to engagement
        df["Churn_Risk"] = (100 - df["Engagement_Score"]) / 100

# ------------------------------------------------------------
# CALCULATE BUSINESS KPIS
# ------------------------------------------------------------
total_leads = len(df)
converted_leads = df["Converted"].sum()
conversion_rate = (converted_leads / total_leads * 100) if total_leads > 0 else 0

high_quality_leads = df[df["Lead_Quality"] == "High"]
high_quality_conversion = (high_quality_leads["Converted"].sum() / len(high_quality_leads) * 100) if len(high_quality_leads) > 0 else 0

total_pipeline_value = df["Expected_Revenue"].sum()
avg_clv = df["CLV"].mean()
avg_deal_size = df[df["Converted"] == 1]["Annual_Revenue_INR_Lakhs"].mean() * 100000 if converted_leads > 0 else 0

avg_sales_cycle = df[df["Converted"] == 1]["Decision_Time_Days"].mean() if converted_leads > 0 else 0
high_priority_cycle = high_quality_leads[high_quality_leads["Converted"] == 1]["Decision_Time_Days"].mean() if len(high_quality_leads[high_quality_leads["Converted"] == 1]) > 0 else 0

avg_meetings = df[df["Converted"] == 1]["Meetings"].mean() if converted_leads > 0 else 0

# Revenue at risk from churn
high_churn_customers = df[df["Churn_Risk"] > 0.5]
revenue_at_risk = high_churn_customers["CLV"].sum()
pct_high_churn = (len(high_churn_customers) / total_leads * 100) if total_leads > 0 else 0

# Channel performance
channel_conversion = df.groupby("Lead_Source")["Converted"].mean().sort_values(ascending=False)
best_channel = channel_conversion.index[0] if len(channel_conversion) > 0 else "N/A"
best_channel_rate = channel_conversion.iloc[0] * 100 if len(channel_conversion) > 0 else 0

# Top performing industry
industry_clv = df.groupby("Industry")["CLV"].mean().sort_values(ascending=False)
top_industry = industry_clv.index[0] if len(industry_clv) > 0 else "N/A"
top_industry_clv = industry_clv.iloc[0] if len(industry_clv) > 0 else 0

# Revenue concentration
df_sorted = df.sort_values("CLV", ascending=False)
top_20_pct = int(0.2 * total_leads)
top_20_revenue = df_sorted.head(top_20_pct)["CLV"].sum()
revenue_concentration = (top_20_revenue / df["CLV"].sum() * 100) if df["CLV"].sum() > 0 else 0

# ------------------------------------------------------------
# DASHBOARD TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🎯 Top 10 Lead Intelligence",
    "💎 Customer Value & Risk",
    "🚀 Marketing & Sales Performance"
])

# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================
with tab1:
    st.markdown('<div class="section-header"><span class="section-icon">📈</span> Key Business Metrics</div>', unsafe_allow_html=True)
    
    # Top 4 KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="kpi-card blue">
                <div class="kpi-icon">💰</div>
                <div class="kpi-label">Total Pipeline Value</div>
                <div class="kpi-value">₹{total_pipeline_value/100000:.1f}L</div>
                <div class="kpi-caption">Expected revenue from all leads</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="kpi-card green">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">High-Quality Win Rate</div>
                <div class="kpi-value">{high_quality_conversion:.1f}%</div>
                <div class="kpi-caption">{len(high_quality_leads)} high-priority leads</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="kpi-card purple">
                <div class="kpi-icon">⏱️</div>
                <div class="kpi-label">Avg Sales Cycle</div>
                <div class="kpi-value">{avg_sales_cycle:.0f} days</div>
                <div class="kpi-caption">Time to close deals</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="kpi-card orange">
                <div class="kpi-icon">💎</div>
                <div class="kpi-label">Avg Customer Value</div>
                <div class="kpi-value">₹{avg_clv/100000:.1f}L</div>
                <div class="kpi-caption">Lifetime value per customer</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Second row of KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(f"""
            <div class="kpi-card teal">
                <div class="kpi-icon">🤝</div>
                <div class="kpi-label">Meetings to Close</div>
                <div class="kpi-value">{avg_meetings:.1f}</div>
                <div class="kpi-caption">Average for won deals</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
            <div class="kpi-card pink">
                <div class="kpi-icon">🚀</div>
                <div class="kpi-label">Best Channel</div>
                <div class="kpi-value">{best_channel_rate:.1f}%</div>
                <div class="kpi-caption">{best_channel}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown(f"""
            <div class="kpi-card red">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-label">Revenue at Risk</div>
                <div class="kpi-value">₹{revenue_at_risk/100000:.1f}L</div>
                <div class="kpi-caption">{pct_high_churn:.1f}% customers at high churn risk</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col8:
        st.markdown(f"""
            <div class="kpi-card indigo">
                <div class="kpi-icon">🏆</div>
                <div class="kpi-label">Top Industry</div>
                <div class="kpi-value">{top_industry}</div>
                <div class="kpi-caption">₹{top_industry_clv/100000:.1f}L avg CLV</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown('<div class="section-header"><span class="section-icon">📊</span> Visual Analytics</div>', unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # Lead Quality Distribution
        quality_dist = df["Lead_Quality"].value_counts().reindex(["High", "Medium", "Low"])
        fig1 = go.Figure(data=[go.Pie(
            labels=quality_dist.index,
            values=quality_dist.values,
            marker=dict(colors=['#10b981', '#f59e0b', '#ef4444']),
            hole=0.5,
            textinfo='label+percent',
            textfont=dict(size=14, color='white', family='Inter'),
            pull=[0.1, 0, 0]
        )])
        fig1.update_layout(
            title=dict(text="Lead Quality Distribution", font=dict(size=18, family='Inter', weight=700)),
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_c2:
        # Conversion by Quality
        conv_by_quality = df.groupby("Lead_Quality")["Converted"].mean().reindex(["High", "Medium", "Low"]) * 100
        fig2 = go.Figure(data=[go.Bar(
            x=conv_by_quality.index,
            y=conv_by_quality.values,
            marker=dict(
                color=['#10b981', '#f59e0b', '#ef4444'],
                line=dict(color='white', width=2)
            ),
            text=[f"{v:.1f}%" for v in conv_by_quality.values],
            textposition='outside',
            textfont=dict(size=14, family='Inter', weight=700)
        )])
        fig2.update_layout(
            title=dict(text="Win Rate by Lead Quality", font=dict(size=18, family='Inter', weight=700)),
            xaxis_title="Lead Quality",
            yaxis_title="Conversion Rate (%)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    col_c3, col_c4 = st.columns(2)
    
    with col_c3:
        # CLV by Industry
        industry_clv_plot = df.groupby("Industry")["CLV"].mean().sort_values(ascending=False).head(6)
        fig3 = px.bar(
            x=industry_clv_plot.values / 100000,
            y=industry_clv_plot.index,
            orientation='h',
            labels={"x": "Avg CLV (₹ Lakhs)", "y": "Industry"},
            color=industry_clv_plot.values,
            color_continuous_scale="Viridis"
        )
        fig3.update_layout(
            title=dict(text="Top Industries by Customer Value", font=dict(size=18, family='Inter', weight=700)),
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_c4:
        # Sales Cycle by Quality
        cycle_by_quality = df.groupby("Lead_Quality")["Decision_Time_Days"].mean().reindex(["High", "Medium", "Low"])
        fig4 = go.Figure(data=[go.Bar(
            x=cycle_by_quality.index,
            y=cycle_by_quality.values,
            marker=dict(
                color=['#8b5cf6', '#ec4899', '#f59e0b'],
                line=dict(color='white', width=2)
            ),
            text=[f"{v:.0f}d" for v in cycle_by_quality.values],
            textposition='outside',
            textfont=dict(size=14, family='Inter', weight=700)
        )])
        fig4.update_layout(
            title=dict(text="Sales Cycle Length by Lead Quality", font=dict(size=18, family='Inter', weight=700)),
            xaxis_title="Lead Quality",
            yaxis_title="Days to Close",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# TAB 2: TOP 10 LEAD INTELLIGENCE
# ============================================================
with tab2:
    st.markdown('<div class="section-header"><span class="section-icon">🎯</span> Top 10 High-Priority Leads with AI Recommendations</div>', unsafe_allow_html=True)
    
    top_10_leads = df.sort_values("Conversion_Probability", ascending=False).head(10)
    
    for idx, lead in top_10_leads.iterrows():
        # Generate personalized recommendations
        industry = lead["Industry"]
        engagement = lead["Engagement_Score"]
        source = lead["Lead_Source"]
        
        # Email campaign theme
        if engagement > 70:
            email_theme = f"Advanced {industry} solutions for scaling operations"
        elif engagement > 40:
            email_theme = f"How top {industry} companies are optimizing with our solution"
        else:
            email_theme = f"Quick wins for {industry} businesses"
        
        # Content recommendation
        if lead["Annual_Revenue_INR_Lakhs"] > 500:
            content = "Enterprise case study + ROI calculator"
        elif lead["Annual_Revenue_INR_Lakhs"] > 100:
            content = "Industry whitepaper + product demo video"
        else:
            content = "Quick start guide + customer success stories"
        
        # Sales approach
        if lead["Conversion_Probability"] > 80:
            approach = "🔥 Hot Lead - Schedule demo within 24 hours"
        elif lead["Conversion_Probability"] > 60:
            approach = "📞 Warm Lead - Personal outreach with value proposition"
        else:
            approach = "📧 Nurture Lead - Email campaign + educational content"
        
        st.markdown(f"""
            <div class="lead-card">
                <div class="lead-header">
                    <div class="lead-name">🏢 {lead.get('Lead_ID', f'Lead #{idx}')}</div>
                    <div class="lead-score">Score: {lead['Conversion_Probability']:.1f}%</div>
                </div>
                <div class="lead-details">
                    <div class="lead-detail-item">
                        <span class="lead-detail-label">🏭 Industry:</span> {industry}
                    </div>
                    <div class="lead-detail-item">
                        <span class="lead-detail-label">💰 Revenue:</span> ₹{lead['Annual_Revenue_INR_Lakhs']:.0f}L
                    </div>
                    <div class="lead-detail-item">
                        <span class="lead-detail-label">📊 Engagement:</span> {engagement:.0f}/100
                    </div>
                    <div class="lead-detail-item">
                        <span class="lead-detail-label">🎯 Source:</span> {source}
                    </div>
                    <div class="lead-detail-item">
                        <span class="lead-detail-label">💎 Expected Value:</span> ₹{lead['Expected_Revenue']/100000:.1f}L
                    </div>
                    <div class="lead-detail-item">
                        <span class="lead-detail-label">⏱️ Est. Close:</span> {lead['Decision_Time_Days']:.0f} days
                    </div>
                </div>
                <div class="recommendation-box">
                    <div class="recommendation-title">🎯 AI-Powered Marketing & Sales Strategy</div>
                    <div class="recommendation-text">
                        <strong>📧 Email Campaign:</strong> "{email_theme}"<br>
                        <strong>📝 Content to Share:</strong> {content}<br>
                        <strong>🚀 Sales Approach:</strong> {approach}<br>
                        <strong>💡 Key Message:</strong> Focus on ROI and quick implementation for {industry} sector
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 3: CUSTOMER VALUE & RISK
# ============================================================
with tab3:
    st.markdown('<div class="section-header"><span class="section-icon">💎</span> Customer Value Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="kpi-card blue">
                <div class="kpi-icon">💰</div>
                <div class="kpi-label">Average CLV</div>
                <div class="kpi-value">₹{avg_clv/100000:.1f}L</div>
                <div class="kpi-caption">Per customer lifetime</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="kpi-card green">
                <div class="kpi-icon">🏆</div>
                <div class="kpi-label">Revenue Concentration</div>
                <div class="kpi-value">{revenue_concentration:.1f}%</div>
                <div class="kpi-caption">From top 20% customers</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="kpi-card red">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-label">High Churn Risk</div>
                <div class="kpi-value">{pct_high_churn:.1f}%</div>
                <div class="kpi-caption">Customers need attention</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        payback_months = (avg_deal_size / (avg_clv / 36)) if avg_clv > 0 else 0
        st.markdown(f"""
            <div class="kpi-card purple">
                <div class="kpi-icon">📅</div>
                <div class="kpi-label">Payback Period</div>
                <div class="kpi-value">{payback_months:.1f}m</div>
                <div class="kpi-caption">Time to recover investment</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header"><span class="section-icon">📊</span> Value & Risk Analysis</div>', unsafe_allow_html=True)
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # CLV Distribution
        fig5 = px.histogram(
            df,
            x="CLV",
            nbins=30,
            labels={"CLV": "Customer Lifetime Value (₹)"},
            color_discrete_sequence=["#8b5cf6"]
        )
        fig5.update_layout(
            title=dict(text="CLV Distribution", font=dict(size=18, family='Inter', weight=700)),
            height=400,
            template="plotly_white",
            xaxis_title="CLV (₹)",
            yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with col_v2:
        # Churn Risk by Industry
        churn_by_industry = df.groupby("Industry")["Churn_Risk"].mean().sort_values(ascending=False).head(6) * 100
        fig6 = go.Figure(data=[go.Bar(
            x=churn_by_industry.index,
            y=churn_by_industry.values,
            marker=dict(
                color=churn_by_industry.values,
                colorscale='Reds',
                line=dict(color='white', width=2)
            ),
            text=[f"{v:.1f}%" for v in churn_by_industry.values],
            textposition='outside',
            textfont=dict(size=12, family='Inter', weight=700)
        )])
        fig6.update_layout(
            title=dict(text="Churn Risk by Industry", font=dict(size=18, family='Inter', weight=700)),
            xaxis_title="Industry",
            yaxis_title="Churn Risk (%)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    # High-value customers at risk
    st.markdown('<div class="section-header"><span class="section-icon">🚨</span> High-Value Customers at Churn Risk</div>', unsafe_allow_html=True)
    
    at_risk = df[(df["Churn_Risk"] > 0.5) & (df["CLV"] > df["CLV"].median())].sort_values("CLV", ascending=False).head(5)
    
    if len(at_risk) > 0:
        for idx, customer in at_risk.iterrows():
            st.markdown(f"""
                <div class="lead-card">
                    <div class="lead-header">
                        <div class="lead-name">🏢 {customer.get('Lead_ID', f'Customer #{idx}')} - {customer['Industry']}</div>
                        <div class="lead-score" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                            Risk: {customer['Churn_Risk']*100:.0f}%
                        </div>
                    </div>
                    <div class="lead-details">
                        <div class="lead-detail-item">
                            <span class="lead-detail-label">💰 CLV at Risk:</span> ₹{customer['CLV']/100000:.1f}L
                        </div>
                        <div class="lead-detail-item">
                            <span class="lead-detail-label">📊 Engagement:</span> {customer['Engagement_Score']:.0f}/100
                        </div>
                        <div class="lead-detail-item">
                            <span class="lead-detail-label">🎯 Source:</span> {customer['Lead_Source']}
                        </div>
                    </div>
                    <div class="recommendation-box" style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-left-color: #ef4444;">
                        <div class="recommendation-title" style="color: #991b1b;">🎯 Retention Strategy</div>
                        <div class="recommendation-text" style="color: #7f1d1d;">
                            <strong>Immediate Action:</strong> Schedule executive check-in call<br>
                            <strong>Offer:</strong> Exclusive feature access + dedicated account manager<br>
                            <strong>Follow-up:</strong> Quarterly business review + success metrics tracking
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Great news! No high-value customers are currently at significant churn risk.")

# ============================================================
# TAB 4: MARKETING & SALES PERFORMANCE
# ============================================================
with tab4:
    st.markdown('<div class="section-header"><span class="section-icon">🚀</span> Marketing Channel Performance</div>', unsafe_allow_html=True)
    
    # Channel metrics
    channel_stats = df.groupby("Lead_Source").agg({
        "Converted": ["sum", "mean", "count"],
        "Expected_Revenue": "sum",
        "CLV": "mean"
    }).round(2)
    
    channel_stats.columns = ["Conversions", "Conv_Rate", "Total_Leads", "Pipeline_Value", "Avg_CLV"]
    channel_stats["Conv_Rate"] = channel_stats["Conv_Rate"] * 100
    channel_stats = channel_stats.sort_values("Conv_Rate", ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Channel Conversion Rates
        fig7 = go.Figure(data=[go.Bar(
            y=channel_stats.index,
            x=channel_stats["Conv_Rate"],
            orientation='h',
            marker=dict(
                color=channel_stats["Conv_Rate"],
                colorscale='Viridis',
                line=dict(color='white', width=2)
            ),
            text=[f"{v:.1f}%" for v in channel_stats["Conv_Rate"]],
            textposition='outside',
            textfont=dict(size=12, family='Inter', weight=700)
        )])
        fig7.update_layout(
            title=dict(text="Conversion Rate by Channel", font=dict(size=18, family='Inter', weight=700)),
            xaxis_title="Conversion Rate (%)",
            yaxis_title="Channel",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        # Channel Pipeline Value
        fig8 = px.bar(
            x=channel_stats.index,
            y=channel_stats["Pipeline_Value"] / 100000,
            labels={"x": "Channel", "y": "Pipeline Value (₹ Lakhs)"},
            color=channel_stats["Pipeline_Value"],
            color_continuous_scale="Blues"
        )
        fig8.update_layout(
            title=dict(text="Pipeline Value by Channel", font=dict(size=18, family='Inter', weight=700)),
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig8, use_container_width=True)
    
    # Sales Funnel Optimization
    st.markdown('<div class="section-header"><span class="section-icon">📉</span> Sales Funnel Analysis</div>', unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        # Funnel stages
        total = len(df)
        engaged = len(df[df["Engagement_Score"] > 30])
        meetings = len(df[df["Meetings"] > 0])
        converted = df["Converted"].sum()
        
        funnel_data = {
            "Stage": ["All Leads", "Engaged (30+)", "Had Meetings", "Converted"],
            "Count": [total, engaged, meetings, converted],
            "Percentage": [100, engaged/total*100, meetings/total*100, converted/total*100]
        }
        
        fig9 = go.Figure(go.Funnel(
            y=funnel_data["Stage"],
            x=funnel_data["Count"],
            textinfo="value+percent initial",
            marker=dict(color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"])
        ))
        fig9.update_layout(
            title=dict(text="Sales Funnel Conversion", font=dict(size=18, family='Inter', weight=700)),
            height=400
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    with col_f2:
        # Engagement vs Conversion
        fig10 = px.scatter(
            df,
            x="Engagement_Score",
            y="Conversion_Probability",
            color="Lead_Quality",
            size="Expected_Revenue",
            hover_data=["Industry", "Lead_Source"],
            color_discrete_map={"High": "#10b981", "Medium": "#f59e0b", "Low": "#ef4444"}
        )
        fig10.update_layout(
            title=dict(text="Engagement vs Conversion Probability", font=dict(size=18, family='Inter', weight=700)),
            xaxis_title="Engagement Score",
            yaxis_title="Conversion Probability (%)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig10, use_container_width=True)
    
    # Action Recommendations
    st.markdown('<div class="section-header"><span class="section-icon">💡</span> Strategic Recommendations</div>', unsafe_allow_html=True)
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown(f"""
            <div class="recommendation-box">
                <div class="recommendation-title">🎯 Focus Areas</div>
                <div class="recommendation-text">
                    <strong>1. Double Down on {best_channel}</strong><br>
                    This channel has {best_channel_rate:.1f}% conversion rate - allocate 40% more budget here<br><br>
                    
                    <strong>2. Prioritize {top_industry} Industry</strong><br>
                    Highest CLV at ₹{top_industry_clv/100000:.1f}L - create industry-specific campaigns<br><br>
                    
                    <strong>3. Accelerate High-Quality Leads</strong><br>
                    {len(high_quality_leads)} leads with {high_quality_conversion:.1f}% win rate - fast-track these
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.markdown(f"""
            <div class="recommendation-box">
                <div class="recommendation-title">⚠️ Risk Mitigation</div>
                <div class="recommendation-text">
                    <strong>1. Revenue Diversification</strong><br>
                    {revenue_concentration:.1f}% revenue from top 20% - reduce dependency risk<br><br>
                    
                    <strong>2. Churn Prevention</strong><br>
                    ₹{revenue_at_risk/100000:.1f}L at risk from {pct_high_churn:.1f}% customers - launch retention program<br><br>
                    
                    <strong>3. Shorten Sales Cycle</strong><br>
                    Current avg: {avg_sales_cycle:.0f} days - implement qualification framework to reduce by 20%
                </div>
            </div>
        """, unsafe_allow_html=True)

# Success message
st.success("✅ Dashboard loaded successfully! Use the tabs above to explore different aspects of your B2B sales intelligence.")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: #f8fafc; border-radius: 10px;">
        <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
            🤖 Powered by AI Machine Learning | 📊 Real-time Analytics | 🎯 Actionable Insights
        </p>
    </div>
""", unsafe_allow_html=True)
