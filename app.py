import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("xgb_model.pkl","rb"))
model_columns = pickle.load(open("model_columns.pkl","rb"))

st.title("📞 Telecom Churn Prediction (Call Usage Model)")

account_length = st.number_input("Account Length (days)", 0)
voice_mail_plan = st.selectbox("Voice Mail Plan", [0,1])
voice_mail_messages = st.number_input("Voice Mail Messages", 0)

day_mins = st.number_input("Day Minutes", 0.0)
evening_mins = st.number_input("Evening Minutes", 0.0)
night_mins = st.number_input("Night Minutes", 0.0)
international_mins = st.number_input("International Minutes", 0.0)

customer_service_calls = st.number_input("Customer Service Calls", 0)

international_plan = st.selectbox("International Plan", [0,1])

day_calls = st.number_input("Day Calls", 0)
evening_calls = st.number_input("Evening Calls", 0)
night_calls = st.number_input("Night Calls", 0)
international_calls = st.number_input("International Calls", 0)

total_charge = st.number_input("Total Charge", 0.0)

if st.button("Predict"):
    input_data = {
        "account_length": account_length,
        "voice_mail_plan": voice_mail_plan,
        "voice_mail_messages": voice_mail_messages,
        "day_mins": day_mins,
        "evening_mins": evening_mins,
        "night_mins": night_mins,
        "international_mins": international_mins,
        "customer_service_calls": customer_service_calls,
        "international_plan": international_plan,
        "day_calls": day_calls,
        "evening_calls": evening_calls,
        "night_calls": night_calls,
        "international_calls": international_calls,
        "total_charge": total_charge
    }

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    if prob > 0.3:
        st.error("❌ Customer WILL CHURN")
    else:
        st.success("✅ Customer will NOT churn")

    st.write("Churn probability:", round(prob*100,2), "%")