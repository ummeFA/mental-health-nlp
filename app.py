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

        # Pie chart of prediction probabilities
        probs = model.predict_proba(vectorized)[0]
        labels = model.classes_

        df_probs = pd.DataFrame({'label': labels, 'probability': probs})
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(df_probs['label'], df_probs['probability'], color='skyblue')
        ax.set_xlabel("Prediction Probability")
        ax.set_xlim(0, 1)
        ax.set_title("Prediction Confidence by Class")

        # Add % labels next to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{width:.2%}", va='center')

        st.markdown("#### Prediction Confidence (Bar Chart)")
        st.pyplot(fig)

