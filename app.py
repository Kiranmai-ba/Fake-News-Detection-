import streamlit as st 
import joblib
import os

# Use relative paths to find the model files
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorizer_path = os.path.join(current_dir, 'vectorizer.jb')
model_path = os.path.join(current_dir, 'lr_model.jb')

try:
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    model_loaded = False

st.title("Fake News Detector") 
st.write("Enter a News Article below to check whether it is Fake or Real.")

news_input = st.text_area("News Article:", "")

if st.button("Check News"):
    if not model_loaded:
        st.error("Cannot check news because model files were not found.")
    elif news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        
        if prediction[0] == 1:
            st.success("The news is real!")
        else:
            st.error("The news is fake!")
    else:
        st.warning("Please enter some text to analyze.")

