import streamlit as st
import joblib
mport os

# Load your trained models safely
@st.cache(allow_output_mutation=True)
def load_models():
    try:
        vectorizer_path = "vectorizer.jb"
        model_path = "lr_model.jb"
        
        # Check if files exist
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            st.error("Model files not found. Please ensure 'vectorizer.jb' and 'lr_model.jb' are in the correct directory.")
            return None, None
        
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models once
vectorizer, model = load_models()

# Your Streamlit app interface
st.title("Fake News Detection App")

user_input = st.text_area("Enter news text for classification:")

if st.button("Predict"):
    if vectorizer is None or model is None:
        st.error("Models are not loaded properly. Please check the setup.")
    elif user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        try:
            # Transform input
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

