import streamlit as st
import joblib

# Load the vectorizer and model
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")  # Added quotes around the file path

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

# Get user input
inputn = st.text_area("News Article:", "")

if st.button("Check News"):
    if inputn.strip():
        # Transform input and make prediction
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)
        
        # Display result
        if prediction[0] == 1:
            st.success("The News is Real!")
        else:
            st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to Analyze")

