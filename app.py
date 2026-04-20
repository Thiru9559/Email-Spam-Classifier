import nltk
import random

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "msg" not in st.session_state:
    st.session_state.msg = ""

# ---------------- NLP ----------------
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    words = [w for w in text if w.isalnum()]
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

# ---------------- MODEL ----------------
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ---------------- UI ----------------
st.title("📧 Spam Classifier")
st.caption("Simple AI to detect spam messages")

# ---------------- EXAMPLES ----------------
st.subheader("Try Examples")

col1, col2, col3 = st.columns(3)

if col1.button("Win money"):
    st.session_state.msg = "Win money now!!! Click here"

if col2.button("Casual chat"):
    st.session_state.msg = "Hey, are we meeting today?"

if col3.button("Special offer"):
    st.session_state.msg = "Limited offer, claim now!"

if st.button("Generate random"):
    samples = [
        "Congratulations! You won a prize",
        "Act fast! Limited deal!",
        "Click here to claim reward!",
        "You have been selected"
    ]
    st.session_state.msg = random.choice(samples)

# ---------------- INPUT ----------------
input_sms = st.text_area("Enter your message", value=st.session_state.msg, height=150)

# ---------------- PREDICT ----------------
if st.button("Analyze Message", use_container_width=True):

    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]
        spam_prob = proba[1]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

        st.write(f"Confidence: {round(spam_prob*100,2)}%")
        st.progress(int(spam_prob * 100))

        st.code(transformed_sms)

        st.session_state.history.append((input_sms, result))

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.subheader("Recent Predictions")

    for msg, res in reversed(st.session_state.history[-5:]):
        st.write(f"{'🚨 Spam' if res else '✅ Not Spam'} → {msg}")