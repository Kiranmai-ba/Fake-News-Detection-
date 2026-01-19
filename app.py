import os
import streamlit as st
import joblib

# Paths to your files
current_dir = os.getcwd()
vectorizer_path = os.path.join(current_dir, 'vectorizer.jb')
model_path = os.path.join(current_dir, 'lr_model.jb')

@st.cache_data
def load_artifacts():
    vectorizer = None
    model = None
    
    # Load vectorizer
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file not found at '{vectorizer_path}'. Please upload or place it in the correct directory.")
    else:
        try:
            vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            st.error(f"Error loading vectorizer: {e}")
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please upload or place it in the correct directory.")
    else:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    return vectorizer, model

def main():
    st.title("Fake News Detection")
    st.write("Enter a news article below to check whether it is Fake or Real.")
    
    # Load artifacts
    vectorizer, model = load_artifacts()
    
    if vectorizer is None or model is None:
        st.warning("Missing model or vectorizer. Please ensure they are in the correct directory.")
        return
    
    news_input = st.text_area("News Article:", "")
    
    if st.button("Check News"):
        if not news_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            try:
                features = vectorizer.transform([news_input])
                prediction = model.predict(features)
                if prediction[0] == 1:
                    st.success("The news is real! 👍")
                else:
                    st.error("The news is fake! 🚨")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
