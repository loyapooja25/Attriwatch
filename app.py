import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Apply dark theme and aesthetic styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #58a6ff;
    }
    .metric-label, .metric-value {
        color: #FFD700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained models
attr_model = joblib.load("best_attrition_model_catboost.pkl")
perf_model = joblib.load("randomforest_best_modell-b.pkl")

st.set_page_config(page_title="AttriWatch", layout="wide")
st.title("AttriWatch: Predict Employee Attrition & Performance")

# Sidebar for thresholds
st.sidebar.header("Threshold Settings")
attr_thresh = st.sidebar.slider("Attrition Threshold", 0.0, 1.0, 0.56, 0.01)
perf_thresh = st.sidebar.slider("Performance Threshold", 0.0, 1.0, 0.5, 0.01)

# Choose input mode
mode = st.radio("Select Input Mode", ["Upload CSV", "Manual Entry"])

# Preprocessing function
def preprocess(df):
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    for col in df.select_dtypes(include='object').columns:
        df[col], _ = pd.factorize(df[col])
    df['YearsAtCompany_AgeRatio'] = df['YearsAtCompany'] / (df['Age'] + 1)
    df['MonthlyIncome_WorkingYearsRatio'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['OverTime_JobSatisfaction'] = df['OverTime'] * df['JobSatisfaction']
    df['AvgSatisfaction'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']) / 3
    df['RecentlyPromoted'] = (df['YearsSinceLastPromotion'] < 2).astype(int)
    df['TenureWithManagerRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    return df

# Upload CSV mode
if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload Employee Data CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df_proc = preprocess(df.copy())
        df_proc['AttritionProb'] = attr_model.predict_proba(df_proc)[:, 1]
        df_proc['PerformanceProb'] = perf_model.predict_proba(df_proc)[:, 1]
        df_proc['RetentionFlag'] = (df_proc['AttritionProb'] > attr_thresh) & (df_proc['PerformanceProb'] > perf_thresh)

        st.subheader("High Performers at Risk")
        high_risk = df_proc[df_proc['RetentionFlag']]
        st.dataframe(high_risk[['Age', 'JobRole', 'Department', 'MonthlyIncome', 'AttritionProb', 'PerformanceProb']])
        st.download_button("Download Priority List", high_risk.to_csv(index=False), file_name="retention_priority.csv")

# Manual entry mode
elif mode == "Manual Entry":
    st.subheader("Manual Entry Form")
    with st.form("manual_form"):
        age = st.slider("Age", 18, 65)
        monthly_income = st.slider("Monthly Income", 1000, 20000)
        total_working_years = st.slider("Total Working Years", 0, 40)
        years_at_company = st.slider("Years at Company", 0, 40)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4)
        env_satisfaction = st.slider("Environment Satisfaction", 1, 4)
        rel_satisfaction = st.slider("Relationship Satisfaction", 1, 4)
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        years_since_promo = st.slider("Years Since Last Promotion", 0, 15)
        years_with_manager = st.slider("Years With Current Manager", 0, 15)
        salary_hike = st.slider("Last Salary Hike (%)", 0, 25)
        work_life = st.slider("Work-Life Balance", 1, 4)
        years_in_role = st.slider("Years in Current Role", 0, 20)
        submitted = st.form_submit_button("Predict")

    if submitted:
        overtime_val = 1 if overtime == "Yes" else 0
        row = pd.DataFrame([{
            'Age': age,
            'MonthlyIncome': monthly_income,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'JobSatisfaction': job_satisfaction,
            'EnvironmentSatisfaction': env_satisfaction,
            'RelationshipSatisfaction': rel_satisfaction,
            'OverTime': overtime_val,
            'YearsSinceLastPromotion': years_since_promo,
            'YearsWithCurrManager': years_with_manager,
            'EmpLastSalaryHikePercent': salary_hike,
            'EmpWorkLifeBalance': work_life,
            'ExperienceYearsInCurrentRole': years_in_role,
            'YearsAtCompany_AgeRatio': years_at_company / (age + 1),
            'MonthlyIncome_WorkingYearsRatio': monthly_income / (total_working_years + 1),
            'OverTime_JobSatisfaction': overtime_val * job_satisfaction,
            'AvgSatisfaction': (env_satisfaction + job_satisfaction + rel_satisfaction) / 3,
            'RecentlyPromoted': int(years_since_promo < 2),
            'TenureWithManagerRatio': years_with_manager / (years_at_company + 1)
        }])

        attr_risk = attr_model.predict_proba(row)[0, 1]
        perf_prob = perf_model.predict_proba(row)[0, 1]

        st.metric("Attrition Risk", f"{attr_risk:.2f}")
        st.metric("Performance Probability", f"{perf_prob:.2f}")

        if attr_risk > attr_thresh and perf_prob > perf_thresh:
            st.markdown("<div style='background-color:#ff4d4d;padding:10px;border-radius:10px;'>High Performer at Risk</div>", unsafe_allow_html=True)

            explainer = shap.Explainer(attr_model)
            shap_values = explainer(row)
            top_factors = pd.DataFrame({
                'Feature': row.columns,
                'SHAP Value': shap_values.values[0]
            }).sort_values(by='SHAP Value', key=abs, ascending=False).head(5)

            st.subheader("Top Features Contributing to Attrition")
            st.table(top_factors)

            st.subheader("Recommended Retention Actions")
            for feat in top_factors['Feature']:
                if "Satisfaction" in feat:
                    st.markdown(f"- Improve {feat.replace('_', ' ')} through feedback and environment improvement.")
                elif "Promotion" in feat:
                    st.markdown("- Explore growth opportunities or potential promotions.")
                elif "OverTime" in feat:
                    st.markdown("- Review workload balance and overtime policies.")
                elif "WorkLifeBalance" in feat:
                    st.markdown("- Improve flexibility or work-from-home options.")
        else:
            st.markdown("<div style='background-color:#2ecc71;padding:10px;border-radius:10px;'>This employee is not a retention risk</div>", unsafe_allow_html=True)
