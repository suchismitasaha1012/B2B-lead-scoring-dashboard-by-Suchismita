import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI B2B Lead Scoring Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------
# CUSTOM CSS FOR MODERN UI
# ------------------------------------------------------------
st.markdown("""
    <style>
    /* Main container */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-align: center;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* KPI Cards - Modern gradient design */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0));
        border-radius: 0 15px 0 100%;
    }
    
    .kpi-card.blue { border-left-color: #3b82f6; }
    .kpi-card.green { border-left-color: #10b981; }
    .kpi-card.purple { border-left-color: #8b5cf6; }
    .kpi-card.orange { border-left-color: #f59e0b; }
    .kpi-card.pink { border-left-color: #ec4899; }
    .kpi-card.teal { border-left-color: #14b8a6; }
    
    .kpi-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
        line-height: 1;
    }
    
    .kpi-caption {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e2e8f0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #667eea !important;
        border-radius: 10px !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">🚀 AI-Driven B2B Lead Scoring Dashboard</h1>
        <p class="dashboard-subtitle">Transform your sales strategy with intelligent lead prioritization and customer value insights</p>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload your B2B Lead & Customer CSV",
    type=["csv"],
    help="Upload a CSV with lead attributes including: Industry, Company_Size, Revenue, Engagement metrics, and Converted status"
)

if uploaded_file is None:
    st.info("👆 **Get Started:** Upload your CSV file to unlock powerful AI-driven insights on lead scoring, conversion analysis, and customer lifetime value.")
    
    # Show sample dashboard preview
    st.markdown("### 📊 Dashboard Preview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="kpi-card blue">
                <div class="kpi-icon">📈</div>
                <div class="kpi-label">Lead Conversion Rate</div>
                <div class="kpi-value">--</div>
                <div class="kpi-caption">Upload data to see metrics</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="kpi-card green">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">High Priority Conversion</div>
                <div class="kpi-value">--</div>
                <div class="kpi-caption">AI-scored leads performance</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="kpi-card purple">
                <div class="kpi-icon">💰</div>
                <div class="kpi-label">Avg Customer Value</div>
                <div class="kpi-value">--</div>
                <div class="kpi-caption">Lifetime value insights</div>
            </div>
        """, unsafe_allow_html=True)
    st.stop()

df = pd.read_csv(uploaded_file)

# ------------------------------------------------------------
# DATA VALIDATION
# ------------------------------------------------------------
required_for_model = [
    "Industry", "Company_Size", "Annual_Revenue_INR_Lakhs", "Location",
    "Website_Visits", "Email_Clicks", "Meetings", "Lead_Source",
    "Engagement_Score", "Product_Interest", "Decision_Time_Days", "Converted"
]

missing_for_model = [c for c in required_for_model if c not in df.columns]
if missing_for_model:
    st.error(f"❌ **Missing Required Columns:** {', '.join(missing_for_model)}")
    st.stop()

# ------------------------------------------------------------
# AI LEAD SCORING MODEL
# ------------------------------------------------------------
with st.spinner("🤖 Training AI model and scoring leads..."):
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
    
    X_all_transformed = preprocess.transform(X)
    ai_score_proba = rf.predict_proba(X_all_transformed)[:, 1]
    df["AI_Score"] = ai_score_proba * 100
    
    def score_band(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        else:
            return "Low"
    
    df["Priority_Band"] = df["AI_Score"].apply(score_band)

# ------------------------------------------------------------
# KPI CALCULATIONS
# ------------------------------------------------------------
total_leads = len(df)
converted = df["Converted"].sum()
overall_conv = (converted / total_leads * 100) if total_leads > 0 else 0

high_band = df[df["Priority_Band"] == "High"]
high_conv = (high_band["Converted"].sum() / len(high_band) * 100) if len(high_band) > 0 else 0

converted_df = df[df["Converted"] == 1]
converted_high = converted_df[converted_df["Priority_Band"] == "High"]
share_new_from_high = (len(converted_high) / len(converted_df) * 100) if len(converted_df) > 0 else 0

avg_meetings_closed = converted_df["Meetings"].mean() if len(converted_df) > 0 else 0

has_clv = "CLV" in df.columns
avg_clv = df["CLV"].mean() if has_clv else 0

channel_conv = df.groupby("Lead_Source")["Converted"].mean().sort_values(ascending=False)
best_channel = channel_conv.index[0] if len(channel_conv) > 0 else "N/A"
best_channel_rate = channel_conv.iloc[0] * 100 if len(channel_conv) > 0 else 0

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Lead Scoring & Funnel", "💎 Customer Value & Retention"])

with tab1:
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="kpi-card blue">
                <div class="kpi-icon">📈</div>
                <div class="kpi-label">Overall Conversion Rate</div>
                <div class="kpi-value">{overall_conv:.1f}%</div>
                <div class="kpi-caption">{converted} of {total_leads} leads converted</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="kpi-card green">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">High-Priority Conversion</div>
                <div class="kpi-value">{high_conv:.1f}%</div>
                <div class="kpi-caption">AI-scored high priority leads</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="kpi-card purple">
                <div class="kpi-icon">⭐</div>
                <div class="kpi-label">High-Priority Share</div>
                <div class="kpi-value">{share_new_from_high:.1f}%</div>
                <div class="kpi-caption">New customers from top leads</div>
            </div>
        """, unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown(f"""
            <div class="kpi-card orange">
                <div class="kpi-icon">🤝</div>
                <div class="kpi-label">Avg Meetings to Close</div>
                <div class="kpi-value">{avg_meetings_closed:.1f}</div>
                <div class="kpi-caption">For converted leads</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="kpi-card pink">
                <div class="kpi-icon">🚀</div>
                <div class="kpi-label">Best Channel</div>
                <div class="kpi-value">{best_channel_rate:.1f}%</div>
                <div class="kpi-caption">{best_channel}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col6:
        avg_engagement = df["Engagement_Score"].mean()
        st.markdown(f"""
            <div class="kpi-card teal">
                <div class="kpi-icon">💫</div>
                <div class="kpi-label">Avg Engagement Score</div>
                <div class="kpi-value">{avg_engagement:.1f}</div>
                <div class="kpi-caption">Overall lead engagement</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Top 10 Priority Leads
    st.markdown("### 🔝 Top 10 High-Priority Leads")
    top10 = df.sort_values("AI_Score", ascending=False).head(10)
    display_cols = ["Lead_ID", "Industry", "AI_Score", "Engagement_Score", "Priority_Band", "Converted"]
    
    # Format the dataframe without styling for better compatibility
    top10_display = top10[display_cols].copy()
    top10_display["AI_Score"] = top10_display["AI_Score"].round(2)
    top10_display["Engagement_Score"] = top10_display["Engagement_Score"].round(2)
    
    st.dataframe(top10_display, use_container_width=True, height=400)
    
    # Charts
    st.markdown("### 📊 Lead Analysis")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        conv_by_band = df.groupby("Priority_Band")["Converted"].mean().reindex(["High", "Medium", "Low"]) * 100
        fig1 = go.Figure(data=[
            go.Bar(x=conv_by_band.index, y=conv_by_band.values,
                   marker=dict(color=['#10b981', '#f59e0b', '#ef4444'],
                              line=dict(color='white', width=2)))
        ])
        fig1.update_layout(title="Conversion Rate by Priority Band", 
                          xaxis_title="Priority Band", yaxis_title="Conversion Rate (%)",
                          height=400, template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_c2:
        band_mix = df["Priority_Band"].value_counts().reindex(["High", "Medium", "Low"])
        fig2 = go.Figure(data=[go.Pie(labels=band_mix.index, values=band_mix.values,
                                      marker=dict(colors=['#10b981', '#f59e0b', '#ef4444']),
                                      hole=0.5)])
        fig2.update_layout(title="Lead Distribution by Priority", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    col_c3, col_c4 = st.columns(2)
    
    with col_c3:
        conv_by_source = df.groupby("Lead_Source")["Converted"].mean().sort_values(ascending=False) * 100
        fig3 = px.bar(x=conv_by_source.index, y=conv_by_source.values,
                     labels={"x": "Lead Source", "y": "Conversion Rate (%)"},
                     color=conv_by_source.values, color_continuous_scale="Viridis")
        fig3.update_layout(title="Conversion Rate by Lead Source", height=400, template="plotly_white", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_c4:
        ai_by_industry = df.groupby("Industry")["AI_Score"].mean().sort_values(ascending=False)
        fig4 = px.bar(x=ai_by_industry.index, y=ai_by_industry.values,
                     labels={"x": "Industry", "y": "Avg AI Score"},
                     color=ai_by_industry.values, color_continuous_scale="Purples")
        fig4.update_layout(title="Average AI Score by Industry", height=400, template="plotly_white", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.markdown("### 💰 Customer Value Metrics")
    
    if has_clv:
        high_value_threshold = df["CLV"].quantile(0.75)
        high_value_customers = df[df["CLV"] >= high_value_threshold]
        pct_high_value = len(high_value_customers) / total_leads * 100
        
        top_20_count = max(1, int(0.2 * total_leads))
        df_sorted = df.sort_values("CLV", ascending=False)
        top_20 = df_sorted.head(top_20_count)
        pct_revenue_top20 = top_20["CLV"].sum() / df["CLV"].sum() * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="kpi-card blue">
                    <div class="kpi-icon">💎</div>
                    <div class="kpi-label">Avg Customer Lifetime Value</div>
                    <div class="kpi-value">₹{avg_clv:,.0f}</div>
                    <div class="kpi-caption">Mean CLV across all customers</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="kpi-card green">
                    <div class="kpi-icon">⭐</div>
                    <div class="kpi-label">High-Value Customers</div>
                    <div class="kpi-value">{pct_high_value:.1f}%</div>
                    <div class="kpi-caption">In top 25% by CLV</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="kpi-card purple">
                    <div class="kpi-icon">🏆</div>
                    <div class="kpi-label">Revenue from Top 20%</div>
                    <div class="kpi-value">{pct_revenue_top20:.1f}%</div>
                    <div class="kpi-caption">Concentration analysis</div>
                </div>
            """, unsafe_allow_html=True)
        
        # CLV Charts
        st.markdown("### 💰 Value Analysis")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            clv_by_industry = df.groupby("Industry")["CLV"].mean().sort_values(ascending=False)
            fig5 = px.bar(x=clv_by_industry.index, y=clv_by_industry.values,
                         labels={"x": "Industry", "y": "Avg CLV (₹)"},
                         color=clv_by_industry.values, color_continuous_scale="Oranges")
            fig5.update_layout(title="Average CLV by Industry", height=400, template="plotly_white", showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col_v2:
            fig6 = px.histogram(df, x="CLV", nbins=30, marginal="box",
                              labels={"CLV": "Customer Lifetime Value (₹)"},
                              color_discrete_sequence=["#8b5cf6"])
            fig6.update_layout(title="CLV Distribution", height=400, template="plotly_white")
            st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning("💡 Add **CLV**, **Revenue**, **Tenure_Months**, and **Churn_Risk** columns to unlock customer value insights!")

st.success("✅ Dashboard loaded successfully! Explore the tabs above for detailed insights.")
