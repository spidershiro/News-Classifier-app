import streamlit as st
import pickle
import json

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("metrics.json", "r") as f:
    metrics = json.load(f)

st.title("News prediction model")
user_input = st.text_input("Enter news title")
if st.button("Predict"):
    X_news = vectorizer.transform([user_input])
    prediction = model.predict(X_news)[0]
    st.success("Likely True" if prediction == 1 else "Likely False")
    str = f"With {metrics['accuracy']:.2f} % accuracy"
    st.markdown(f":blue[{str}]")
