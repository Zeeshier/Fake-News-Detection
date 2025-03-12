import streamlit as st
import joblib
import os

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    model_path = "rf_model.pkl"
    if not os.path.exists(model_path):
        st.error("🚨 Model not found.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the vectorizer
@st.cache_resource
def load_vectorizer():
    vectorizer_path = "vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        st.error("🚨 Vectorizer not found.")
        return None
    try:
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer
    except Exception as e:
        st.error(f"❌ Error loading vectorizer: {e}")
        return None

rf_model = load_model()
vectorizer = load_vectorizer()

# Custom CSS for a modern UI
st.markdown(
    """
    <style>
        body { background-color: #f4f4f4; }
        .stTextArea textarea { border-radius: 12px; padding: 12px; font-size: 16px; }
        .stButton>button {
            border-radius: 8px;
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            transition: 0.3s;
            border: none;
        }
        .stButton>button:hover { background: linear-gradient(135deg, #0056b3, #004494); }
        .stAlert { border-radius: 12px; padding: 12px; font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.title("📰 Fake News Detection")
st.markdown("<h3 style='color: #007bff;'>Web App for predicting fake news.</h3>", unsafe_allow_html=True)

# User Input
user_input = st.text_area("📝 Enter News Article:", height=150)

# Prediction
if st.button("🔍 Predict"):
    if rf_model and vectorizer:
        if user_input.strip():
            user_input_vectorized = vectorizer.transform([user_input])
            prediction = rf_model.predict(user_input_vectorized)
            label = "❌ Fake News" if prediction[0] == 1 else "✅ Real News"

            # Styled Prediction Output
            st.subheader("Prediction:")
            if prediction[0] == 0:
                st.success(f"🎯 {label}")
            else:
                st.error(f"⚠️ {label}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    else:
        st.error("🚨 Model or vectorizer not loaded. Please check the files and restart.")

