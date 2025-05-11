import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="AttriWatch", layout="wide")
st.title("AttriWatch: Predict Employee Attrition & Performance")

# Load Models
attr_model = joblib.load("best_attrition_modelaa_catboost.pkl")
perf_model = joblib.load("catboost_best_modell-b.pkl")

st.sidebar.header("Threshold Settings")
attr_thresh = st.sidebar.slider("Attrition Threshold", 0.0, 1.0, 0.56)
perf_thresh = st.sidebar.slider("Performance Threshold", 0.0, 1.0, 0.5)

mode = st.radio("Select Input Mode", ["Manual Entry"])

if mode == "Manual Entry":
    with st.form("manual_form"):
        st.subheader("Attrition-related Features")
        Age = st.slider("Age", 18, 60)
        BusinessTravel = st.selectbox("Business Travel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
        DailyRate = st.slider("Daily Rate", 100, 1500)
        Department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
        DistanceFromHome = st.slider("Distance From Home", 1, 30)
        Education = st.slider("Education Level (1-5)", 1, 5)
        EducationField = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
        EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4)
        Gender = st.selectbox("Gender", ['Male', 'Female'])
        HourlyRate = st.slider("Hourly Rate", 30, 120)
        JobInvolvement = st.slider("Job Involvement", 1, 4)
        JobLevel = st.slider("Job Level", 1, 5)
        JobRole = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                             'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
        JobSatisfaction = st.slider("Job Satisfaction", 1, 4)
        MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        MonthlyIncome = st.slider("Monthly Income", 1000, 20000)
        MonthlyRate = st.slider("Monthly Rate", 1000, 25000)
        NumCompaniesWorked = st.slider("Num Companies Worked", 0, 10)
        OverTime = st.selectbox("OverTime", ['Yes', 'No'])
        PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25)
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4)
        StockOptionLevel = st.slider("Stock Option Level", 0, 3)
        TotalWorkingYears = st.slider("Total Working Years", 1, 40)
        TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6)
        WorkLifeBalance = st.slider("Work Life Balance", 1, 4)
        YearsAtCompany = st.slider("Years at Company", 0, 40)
        YearsInCurrentRole = st.slider("Years in Current Role", 0, 20)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15)
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 20)

        # Derived features
        OverTime_val = 1 if OverTime == "Yes" else 0
        YearsAtCompany_AgeRatio = YearsAtCompany / (Age + 1)
        MonthlyIncome_WorkingYearsRatio = MonthlyIncome / (TotalWorkingYears + 1)
        OverTime_JobSatisfaction = OverTime_val * JobSatisfaction
        AvgSatisfaction = (EnvironmentSatisfaction + JobSatisfaction + RelationshipSatisfaction) / 3
        RecentlyPromoted = int(YearsSinceLastPromotion < 2)
        TenureWithManagerRatio = YearsWithCurrManager / (YearsAtCompany + 1)

        st.subheader("Performance-related Features")
        EmpEnvironmentSatisfaction = EnvironmentSatisfaction
        EmpLastSalaryHikePercent = PercentSalaryHike
        EmpWorkLifeBalance = WorkLifeBalance
        ExperienceYearsAtThisCompany = YearsAtCompany
        ExperienceYearsInCurrentRole = YearsInCurrentRole
        Perf_YearsSinceLastPromotion = YearsSinceLastPromotion
        Perf_YearsWithCurrManager = YearsWithCurrManager

        submit = st.form_submit_button("Predict")

    if submit:
        # Attrition row
        attr_row = pd.DataFrame([{
            'Age': Age, 'BusinessTravel': BusinessTravel, 'DailyRate': DailyRate, 'Department': Department,
            'DistanceFromHome': DistanceFromHome, 'Education': Education, 'EducationField': EducationField,
            'EnvironmentSatisfaction': EnvironmentSatisfaction, 'Gender': Gender, 'HourlyRate': HourlyRate,
            'JobInvolvement': JobInvolvement, 'JobLevel': JobLevel, 'JobRole': JobRole,
            'JobSatisfaction': JobSatisfaction, 'MaritalStatus': MaritalStatus, 'MonthlyIncome': MonthlyIncome,
            'MonthlyRate': MonthlyRate, 'NumCompaniesWorked': NumCompaniesWorked, 'OverTime': OverTime_val,
            'PercentSalaryHike': PercentSalaryHike, 'RelationshipSatisfaction': RelationshipSatisfaction,
            'StockOptionLevel': StockOptionLevel, 'TotalWorkingYears': TotalWorkingYears,
            'TrainingTimesLastYear': TrainingTimesLastYear, 'WorkLifeBalance': WorkLifeBalance,
            'YearsAtCompany': YearsAtCompany, 'YearsInCurrentRole': YearsInCurrentRole,
            'YearsSinceLastPromotion': YearsSinceLastPromotion, 'YearsWithCurrManager': YearsWithCurrManager,
            'YearsAtCompany_AgeRatio': YearsAtCompany_AgeRatio,
            'MonthlyIncome_WorkingYearsRatio': MonthlyIncome_WorkingYearsRatio,
            'OverTime_JobSatisfaction': OverTime_JobSatisfaction,
            'AvgSatisfaction': AvgSatisfaction,
            'RecentlyPromoted': RecentlyPromoted,
            'TenureWithManagerRatio': TenureWithManagerRatio
        }])

        # Performance row
        perf_row = pd.DataFrame([{
            'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
            'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
            'EmpWorkLifeBalance': EmpWorkLifeBalance,
            'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
            'YearsWithCurrManager': Perf_YearsWithCurrManager,
            'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
            'YearsSinceLastPromotion': Perf_YearsSinceLastPromotion
        }])

        # Predictions
        attr_risk = attr_model.predict_proba(attr_row)[0, 1]
        perf_prob = perf_model.predict_proba(perf_row)[0, 1]

        st.metric("Attrition Risk", f"{attr_risk:.2f}")
        st.metric("Performance Probability", f"{perf_prob:.2f}")

        if attr_risk > attr_thresh and perf_prob > perf_thresh:
            st.markdown("<div style='background-color:#ff4d4d;padding:10px;border-radius:10px;'>High Performer at Risk</div>", unsafe_allow_html=True)

            explainer = shap.Explainer(attr_model)
            shap_values = explainer(attr_row)
            top_factors = pd.DataFrame({
                'Feature': attr_row.columns,
                'SHAP Value': shap_values.values[0]
            }).sort_values(by='SHAP Value', key=abs, ascending=False).head(5)

            st.subheader("Top Features Contributing to Attrition")
            st.table(top_factors)

            st.subheader("Recommended Retention Actions")
            for feat in top_factors['Feature']:
                if "Satisfaction" in feat:
                    st.markdown(f"- Improve {feat.replace('_', ' ')} through feedback and support.")
                elif "Promotion" in feat or "YearsSinceLastPromotion" in feat:
                    st.markdown("- Explore internal mobility or promotion options.")
                elif "OverTime" in feat:
                    st.markdown("- Adjust workload or team support for balance.")
                elif "WorkLifeBalance" in feat:
                    st.markdown("- Consider flexible or hybrid work arrangements.")
        else:
            st.markdown("<div style='background-color:#2ecc71;padding:10px;border-radius:10px;'>This employee is not a retention risk</div>", unsafe_allow_html=True)

    
