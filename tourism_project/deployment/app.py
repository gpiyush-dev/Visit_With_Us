"""Streamlit UI for predicting tourism package purchase propensity."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"

st.set_page_config(page_title="Visit with Us | Purchase Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Visit with Us")
st.subheader("Wellness Tourism Package — Purchase Predictor")
st.write("Enter customer details to estimate the likelihood of purchasing the package.")
if not MODEL_PATH.exists():
    st.error("The trained model is not available yet. Run the GitHub Actions pipeline and ensure model.joblib is committed.")
    st.stop()
model = joblib.load(MODEL_PATH)

with st.form("customer_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
        marital = st.selectbox("Marital status", ["Married", "Single", "Divorced", "Unmarried"])
        income = st.number_input("Monthly income (₹)", min_value=0, max_value=1000000, value=25000, step=1000)
    with c2:
        visitors = st.number_input("People visiting", min_value=1, max_value=20, value=2)
        children = st.number_input("Children below age 5", min_value=0, max_value=10, value=0)
        trips = st.number_input("Trips per year", min_value=0, max_value=50, value=3)
        stars = st.selectbox("Preferred property stars", [3, 4, 5], index=0)
        passport = st.selectbox("Has passport?", ["No", "Yes"])
        own_car = st.selectbox("Owns a car?", ["No", "Yes"])
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    c3, c4 = st.columns(2)
    with c3:
        pitch_score = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)
        followups = st.number_input("Number of follow-ups", min_value=0, max_value=20, value=3)
    with c4:
        duration = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=120, value=10)
    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

if submitted:
    customer = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city_tier,
        "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": visitors, "NumberOfFollowups": followups,
        "ProductPitched": product, "PreferredPropertyStar": stars,
        "MaritalStatus": marital, "NumberOfTrips": trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_score, "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income,
    }])
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(probability >= 0.5)
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    if prediction:
        st.success("Prediction: likely to purchase. Consider prioritizing a personalized follow-up.")
    else:
        st.info("Prediction: less likely to purchase at the current threshold. Consider a lower-cost nurture campaign.")
    st.caption("This is a decision-support estimate, not a guarantee. Use consent-based outreach and monitor model performance.")
