import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# App layout
st.title("🧠 Mental Health Text Classifier")
st.markdown("Enter a mental health-related sentence, and the model will classify it.")

user_input = st.text_area("Your input:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # predicted label
        st.success(f" Predicted Mental Health Status: **{prediction}**")

        # pie chart of prediction probabilities
        probs = model.predict_proba(vectorized)[0]
        labels = model.classes_

        df_probs = pd.DataFrame({'label': labels, 'probability': probs})
        fig, ax = plt.subplots()
        ax.pie(df_probs['probability'], labels=df_probs['label'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio for circle

        st.markdown("#### Prediction Confidence")
        st.pyplot(fig)
