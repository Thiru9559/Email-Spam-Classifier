# Email Spam Classifier 🚀

A Machine Learning web app built using Streamlit that classifies messages as Spam or Not Spam.

🚀 **Live Demo:** https://email-spam-classifier-123456789.streamlit.app/

## Features

* Text preprocessing using NLTK
* TF-IDF Vectorization
* Naive Bayes model
* Simple Streamlit UI

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
or
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Tech Stack

* Python
* Scikit-learn
* NLTK
* Streamlit





email-spam-classifier-new
End to end code for the email spam classifier project
📧 Email/SMS Spam Classifier

An end-to-end Machine Learning project that classifies messages as Spam or Not Spam using Natural Language Processing (NLP) and a trained model deployed with Streamlit.

🚀 Features
Classifies SMS/Email text into Spam / Not Spam
Uses TF-IDF Vectorization
Text preprocessing with NLTK
Simple and interactive Streamlit UI
Lightweight and easy to deploy
🧠 How It Works
User enters a message
Text is preprocessed:
Lowercasing
Tokenization
Removing stopwords & punctuation
Stemming
Text is converted into vectors using TF-IDF
Pre-trained ML model predicts the result
Output is displayed as Spam / Not Spam
📂 Project Structure
├── app.py                # Main Streamlit app
├── model.pkl            # Trained ML model
├── vectorizer.pkl       # TF-IDF vectorizer
├── spam.csv             # Dataset
├── sms-spam-detection.ipynb  # Model training notebook
├── requirements.txt     # Dependencies
├── setup.sh             # Deployment setup
├── Procfile             # Deployment config
├── nltk.txt             # Required NLTK data
└── README.md            # Project documentation
⚙️ Installation
git clone https://github.com/Thiru9559/Email-Spam-Classifier
cd Email-Spam-Classifier
pip install -r requirements.txt
▶️ Run the App
streamlit run app.py
📦 Dependencies

From :

streamlit
nltk
scikit-learn
🔧 NLTK Setup

Make sure to download required datasets:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

(Also referenced in )

🌐 Deployment

This project is configured for deployment using:

Streamlit Cloud / Heroku
Uses Procfile and setup.sh for configuration
📊 Model Training
Dataset: spam.csv
Algorithm: (trained in notebook)
Notebook: sms-spam-detection.ipynb
🖥️ App Interface

From :

Text input box for message
Predict button
Displays:
Spam
Not Spam
🎯 Future Improvements
Improve UI/UX
Add probability score
Support multiple languages
Switch to Django/React for scalable frontend
