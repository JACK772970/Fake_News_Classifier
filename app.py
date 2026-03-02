import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords', quiet = True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load model and tfidf
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))




def clean_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    words = content.split()
    content = [i for i in words if i not in stop_words]
    content = ' '.join(content)
    return content


# App UI
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it is Real or Fake!")

input_text = st.text_area("Paste news article here")

if st.button("Predict"):
    cleaned = clean_text(input_text)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("✅ This appears to be REAL NEWS!")
    else:
        st.error("🚨 This appears to be FAKE NEWS!")

