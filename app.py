# Fake-News-Detection
import streamlit as st
import joblib
import os

# Define paths
current_dir = os.getcwd()
vectorizer_path = os.path.join(current_dir, 'vectorizer.jb')
model_path = os.path.join(current_dir, 'lr_model.jb')

# Load model with error handling
try:
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    model_loaded = False
except Exception as e:
    st.error(f"An error occurred: {e}")
    model_loaded = False

# App UI
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")
news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if not model_loaded:
        st.error("Cannot check news because model files were not found.")
    elif not news_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            transform_input = vectorizer.transform([news_input])
            prediction = model.predict(transform_input)
            if prediction[0] == 1:
                st.success("The news is real! 👍")
            else:
                st.error("The news is fake! 🚨")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Please enter some text to analyze.")
