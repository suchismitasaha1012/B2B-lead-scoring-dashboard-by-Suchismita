import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------
# 0. PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI-Driven B2B Lead Scoring Dashboard",
    layout="wide",
)

st.title("AI-Driven B2B Lead Scoring Dashboard")
st.markdown(
    """
Upload your **leads CSV** below.

The file **must contain at least these columns** (case-sensitive):

- `Lead_ID`
- `Industry`
- `Company_Size`
- `Annual_Revenue_INR_Lakhs`
- `Location`
- `Website_Visits`
- `Email_Clicks`
- `Meetings`
- `Lead_Source`
- `Engagement_Score`
- `Product_Interest`  (values like **High / Medium / Low**)
- `Decision_Time_Days`
- `Converted`  (1 = became customer, 0 = did not)

The app will:

1. Train an AI model on your data (no hard row limit – 5,000+ leads is fine).
2. Score **all leads**.
3. Show **Top 10 leads** and **KPIs** linked to your project objectives.
"""
)

REQUIRED_COLS = [
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

# ------------------------------------------------
# 1. FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Lead CSV",
    type=["csv"],
    help="CSV with the required columns. 5,000+ rows supported.",
)

if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to see the dashboard.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

st.subheader("Preview of Uploaded Data")
st.dataframe(df.head())

# ------------------------------------------------
# 2. VALIDATE COLUMNS
# ------------------------------------------------
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    st.error(
        f"❌ Missing required columns in your CSV: {missing_cols}\n\n"
        "Please fix the file and upload again."
    )
    st.stop()

# ------------------------------------------------
# 3. PREPARE FEATURES / TARGET
# ------------------------------------------------
# Map Product_Interest to numeric (ordinal)
interest_map = {"Low": 1, "Medium": 2, "High": 3}
df["Product_Interest_Num"] = df["Product_Interest"].map(interest_map)

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

X = df[numeric_cols + cat_cols].copy()
y = df["Converted"].astype(int)

# Guard: if all y are 0 or 1 only, model cannot learn
if y.nunique() < 2:
    st.error(
        "Your `Converted` column has only one value "
        "(all 0s or all 1s). The model cannot learn patterns.\n\n"
        "Please provide data where some leads converted and some did not."
    )
    st.stop()

# ------------------------------------------------
# 4. BUILD PIPELINE & TRAIN MODEL
# ------------------------------------------------
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocess = ColumnTransformer(
    transformers=[("cat", categorical_transformer, cat_cols)],
    remainder="passthrough",  # keep numeric columns as-is
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
)

pipe = Pipeline(
    steps=[
        ("prep", preprocess),
        ("model", model),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(X_train, y_train)

# ------------------------------------------------
# 5. MODEL METRICS
# ------------------------------------------------
y_proba_test = pipe.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba_test)
precision = precision_score(y_test, y_pred_test, zero_division=0)
recall = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

# ------------------------------------------------
# 6. SCORE ALL LEADS & CREATE TOP-10 TABLE
# ------------------------------------------------
all_proba = pipe.predict_proba(X)[:, 1]
df_scored = df.copy()
df_scored["Lead_Score"] = (all_proba * 100).round(2)
df_scored = df_scored.sort_values("Lead_Score", ascending=False)
df_scored["Rank"] = df_scored["Lead_Score"].rank(
    ascending=False, method="first"
).astype(int)

top10 = df_scored.head(10)

# ------------------------------------------------
# 7. KPI CALCULATIONS (FOR OBJECTIVES)
# ------------------------------------------------
overall_conv_rate = y.mean() * 100

top_decile_cutoff = int(max(len(df_scored) * 0.1, 1))
top_decile = df_scored.head(top_decile_cutoff)
rest = df_scored.iloc[top_decile_cutoff:]

top_decile_conv = top_decile["Converted"].mean() * 100
rest_conv = rest["Converted"].mean() * 100 if len(rest) > 0 else np.nan

lift = (
    top_decile_conv / overall_conv_rate
    if overall_conv_rate > 0
    else np.nan
)

industry_scores = (
    df_scored.groupby("Industry")["Lead_Score"].mean().sort_values(ascending=False)
)
source_scores = (
    df_scored.groupby("Lead_Source")["Lead_Score"].mean().sort_values(ascending=False)
)

# ------------------------------------------------
# 8. DISPLAY DASHBOARD
# ------------------------------------------------
st.markdown("---")
st.subheader("Top 10 Leads by AI Score")

st.dataframe(
    top10[
        [
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
            "Lead_Score",
            "Converted",
            "Rank",
        ]
    ]
)

st.markdown("---")
st.subheader("Dashboard KPIs")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

kpi_col1.metric(
    "Overall Conversion Rate (%)",
    f"{overall_conv_rate:.1f}",
)
kpi_col2.metric(
    "Model AUC-ROC",
    f"{auc:.3f}",
)
kpi_col3.metric(
    "Model F1-Score",
    f"{f1:.3f}",
)
kpi_col4.metric(
    "Lift (Top 10% vs Overall)",
    f"{lift:.2f}" if not np.isnan(lift) else "N/A",
)

st.markdown("### Average Lead Score by Industry")
st.dataframe(industry_scores.rename("Avg_Lead_Score").reset_index())

st.markdown("### Average Lead Score by Lead Source")
st.dataframe(source_scores.rename("Avg_Lead_Score").reset_index())

st.markdown(
    """
**How this links to project objectives:**

1. **Enhancing lead qualification & prioritisation** – Top 10 + Rank + Lead Score let sales focus on the most promising B2B accounts.  
2. **Predicting CLV / churn (extension)** – the same pipeline can be applied on the customer CLV dataset.  
3. **Personalising marketing campaigns** – Industry / Lead Source wise scores show which segments respond best.  
4. **Optimising funnel efficiency** – Lift and conversion KPIs show improvement versus random targeting.
"""
)
