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


st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .css-1d391kg {
            padding-top: 3rem;
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
        .title {
            color: #1e88e5;
        }
        .team-member {
            padding: 8px;
            margin: 5px 0;
            background-color: #f5f5f5;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

counter = 0

def chatbot(user_input):
    input_text = vectorize.transform([user_input])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def main():
    global counter
    st.title("🤖 NLP Chatbot")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to our NLP Chatbot")
        st.write("Start a conversation by typing a message below.")
        
        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("Type your message here and press Enter:", 
                                 key=f"user_input_{counter}",
                                 placeholder="Ask me anything...")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            
            # Display chat bubbles
            st.markdown(f"<div class='chat-user'>{user_input_str}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bot'>{response}</div>", unsafe_allow_html=True)
            
            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<div class='timestamp'>{timestamp}</div>", unsafe_allow_html=True)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # Save the conversation
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.success("Thank you for chatting with me. Have a great day!")
                st.stop()
                
    elif choice == "Conversation History":
        st.header("📜 Conversation History")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
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
        This project was created by a group of 5 students studying in the 3rd Year of B.Tech in Information Technology 
        at Maulana Abul Kalam Azad University of Technology, West Bengal.
        """)

        st.subheader("👥 Group Members:")
        st.markdown("""
        <div class="team-member"><strong>Nirvik Ghosh</strong></div>
        <div class="team-member"><strong>Rudra Banerjee</strong></div>
        <div class="team-member"><strong>Snehasis Sardar</strong></div>
        <div class="team-member"><strong>Subha Sadhu</strong></div>
        <div class="team-member"><strong>Avik Mukherjee</strong></div>
        """, unsafe_allow_html=True)

        st.write("""
        This project is a part of the course 'Natural Language Processing'. Each member of the group has contributed 
        to the project in various ways, including data collection, model training, interface design, and documentation.
        """)

        st.subheader("📚 Project Overview:")
        st.write("""
        The project is divided into two parts:
        1. **NLP techniques and Logistic Regression algorithm** is used to train the chatbot on labeled intents and entities.
        2. For building the **Chatbot interface**, Streamlit web framework is used to build a web-based chatbot interface. 
           The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("📊 Dataset:")
        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - **Intents**: The intent of the user input (e.g. "greeting", "budget", "about")
        - **Entities**: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - **Text**: The user input text.
        """)

        st.subheader("💻 Streamlit Chatbot Interface:")
        st.write("""
        The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text 
        and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses 
        to user input.
        """)

        st.subheader("🎯 Conclusion:")
        st.write("""
        In this project, a chatbot is built that can understand and respond to user input based on intents. 
        The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. 
        This project can be extended by adding more data, using more sophisticated NLP techniques, or implementing 
        deep learning algorithms.
        """)

if __name__ == "__main__":
    main()