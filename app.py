import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorize = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

X = vectorize.fit_transform(patterns)
y = tags
clf.fit(X, y)

x = vectorize.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorize.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
        
counter = 0

def main():
    global counter
    st.title("A Basic Chatbot Showing the functionality of Streamlit")

