import streamlit as st
import joblib

# Load your trained models safely
@st.cache(allow_output_mutation=True)
def load_models():
    try:
        # Load the models directly, assuming they are in the same directory
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models once when app starts
vectorizer, model = load_models()

# Streamlit interface
st.title("Fake News Detection App")

user_input = st.text_area("Enter news text for classification:")

if st.button("Predict"):
    if vectorizer is None or model is None:
        st.error("Models are not loaded properly. Please check the setup.")
    elif user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        try:
            # Transform input text
            input_features = vectorizer.transform([user_input])
            # Predict
            prediction = model.predict(input_features)
            # Display result
            if prediction[0] == 1:
                st.success("The news is likely **Real**.")
            else:
                st.warning("The news is likely **Fake**.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
