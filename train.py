import pandas as pd, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]
df.columns = ['label','text']
df['label'] = df['label'].map({'ham':0,'spam':1})

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['label']

model = MultinomialNB().fit(X, y)

pickle.dump(tfidf, open('vectorizer.pkl','wb'))
pickle.dump(model, open('model.pkl','wb'))
