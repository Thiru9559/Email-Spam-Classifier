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
st.set_page_config(page_title="Spam Classifier", page_icon="💎", layout="centered")

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "msg" not in st.session_state:
    st.session_state.msg = ""

# ---------------- GLASS UI STYLE ----------------
st.markdown("""
<style>

/* 🌈 Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #020617, #1e1b4b);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* 🧊 Glass Cards */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
}

/* 💎 Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: white;
}

/* 🌟 Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* 🚀 Neon Button */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border: none;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #a855f7;
}

/* 🚨 Result */
.spam {
    background: linear-gradient(90deg, #ff4b4b, #ff7b7b);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.notspam {
    background: linear-gradient(90deg, #00c853, #69f0ae);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="title">💎 Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered message detection</div>', unsafe_allow_html=True)

# ---------------- EXAMPLES ----------------
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("### 💡 Try Examples")

    col1, col2, col3 = st.columns(3)

    if col1.button("💰 Win money"):
        st.session_state.msg = "Win money now!!! Click here"

    if col2.button("👋 Casual"):
        st.session_state.msg = "Hey, are we meeting today?"

    if col3.button("🎁 Offer"):
        st.session_state.msg = "Limited offer, claim now!"

    if st.button("🎲 Generate Random"):
        samples = [
            "Congratulations! You won a prize 🎉",
            "Act fast! Limited deal!",
            "Click here to claim reward!",
            "You have been selected 🎁"
        ]
        st.session_state.msg = random.choice(samples)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter your message", value=st.session_state.msg, height=150)

# ---------------- PREDICT ----------------
if st.button("🚀 Analyze Message", use_container_width=True):

    if input_sms.strip() == "":
        st.warning("Enter a message first!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]
        spam_prob = proba[1]

        st.markdown("---")

        if result == 1:
            st.markdown('<div class="spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="notspam">✅ SAFE MESSAGE</div>', unsafe_allow_html=True)

        st.write(f"Confidence: {round(spam_prob*100,2)}%")

        st.progress(int(spam_prob * 100))

        st.markdown("### 🧠 Keywords")
        st.code(transformed_sms)

        st.session_state.history.append((input_sms, result))

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write("### 🕘 Recent Predictions")

    for msg, res in reversed(st.session_state.history[-5:]):
        st.write(f"{'🚨' if res else '✅'} {msg}")

    st.markdown('</div>', unsafe_allow_html=True)