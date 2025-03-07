import streamlit as st
import joblib
import os

# load the model
@st.cache_resource
def load_model():
    model_path = "rf_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

#  load the vectorizer
@st.cache_resource
def load_vectorizer():
    vectorizer_path = "vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        st.error("Model not found.")
        return None
    try:
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        return None

rf_model = load_model()
vectorizer = load_vectorizer()

st.title("📰 Fake News Detection")
st.write("Web App for predicting fake news.")

# Input 
user_input = st.text_area("Enter News Article:")

if st.button("Predict"):
    if rf_model and vectorizer:
        if user_input.strip():
            user_input_vectorized = vectorizer.transform([user_input])

            prediction = rf_model.predict(user_input_vectorized)
            label = "Fake News" if prediction[0] == 1 else "Real News"
            
            st.subheader("Prediction:")
            st.success(f"🔍 {label}")
        else:
            st.warning("Please enter some text to predict.")
    else:
        st.error("Model or vectorizer not loaded. Fix the issue and restart.")
