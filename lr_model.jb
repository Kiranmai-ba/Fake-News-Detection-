import joblib
import os

# Load the saved vectorizer and the trained fake news detection model
vectorizer = joblib.load('./vectorizer.jb')
lr_model = joblib.load('./lr_model.jb')

# New article to classify
new_text = "Sample news article"

# Transform the new text into the feature vector
X_new = vectorizer.transform([new_text])

# Predict whether the news is real or fake
prediction = lr_model.predict(X_new)[0]

# Convert prediction to boolean: True for real news, False for fake news
is_real_news = bool(prediction)

print(is_real_news)
