import os
import json
import datetime
import csv
import random
import nltk
import ssl
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issue for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load intents.json file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Preprocessing function for better text accuracy
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words
    ]
    return " ".join(filtered_tokens)

# Extract training data
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        processed_text = preprocess_text(pattern)
        tags.append(intent["tag"])
        patterns.append(processed_text)

# Vectorization and model training
vectorize = TfidfVectorizer(ngram_range=(1, 3))
X = vectorize.fit_transform(patterns)
y = tags

clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(X, y)

# Streamlit UI Configuration
st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS for better UI styling
st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        .stTextArea>div>div>textarea {
            border-radius: 15px;
            padding: 15px;
        }
        .chat-user {
            background-color: #e3f2fd;
            padding: 10px 15px;
            border-radius: 18px 18px 0 18px;
            margin: 5px 0;
            display: inline-block;
            max-width: 80%;
            float: right;
            clear: both;
        }
        .chat-bot {
            background-color: #f1f1f1;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 0;
            margin: 5px 0;
            display: inline-block;
            max-width: 80%;
            float: left;
            clear: both;
        }
        .timestamp {
            font-size: 0.8em;
            color: #666;
            clear: both;
        }
        .divider {
            margin: 1rem 0;
            border-top: 1px solid #eee;
        }
    </style>
""", unsafe_allow_html=True)

# Chatbot response function
def chatbot(user_input):
    processed_input = preprocess_text(user_input)
    input_vector = vectorize.transform([processed_input])
    tag = clf.predict(input_vector)[0]

    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Main chatbot interface
def main():
    st.title("🤖 NLP Chatbot")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to our NLP Chatbot")
        st.write("Start a conversation by typing a message below.")

        # Ensure chat log file exists
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

        user_input = st.text_input("Type your message here and press Enter:", placeholder="Ask me anything...")

        if user_input:
            response = chatbot(user_input)

            # Display chat
            st.markdown(f"<div class='chat-user'>{user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bot'>{response}</div>", unsafe_allow_html=True)

            # Log conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ["goodbye", "bye"]:
                st.success("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("📜 Conversation History")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        try:
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.markdown(f"<div class='chat-user'>{row[0]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='chat-bot'>{row[1]}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='timestamp'>{row[2]}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("No conversation history found. Start chatting first!")

    elif choice == "About":
        st.header("ℹ️ About This Project")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.write("""
        This project was created as a part of the **Natural Language Processing** course.
        It uses **NLTK for preprocessing** and **Logistic Regression with TF-IDF** for intent classification.
        """)

if __name__ == "__main__":
    main()
