# Streamlit Chatbot - NLP Based Conversational AI

## 📌 Overview
This project is a Natural Language Processing (NLP) based chatbot developed using Streamlit, Python, and machine learning techniques. It is trained using a dataset of labeled intents and responses and leverages **TF-IDF Vectorization** and **Logistic Regression** to classify user inputs and generate appropriate responses.

## 🚀 Features
- **Machine Learning-Based Response Prediction:** Utilizes **TF-IDF vectorization** and **Logistic Regression** for intent classification.
- **Streamlit Web Interface:** Provides an interactive UI for users to engage with the chatbot.
- **Conversation History:** Stores and displays user-chatbot interactions using a CSV log.
- **Dynamic User Input Handling:** Processes user inputs and generates responses in real-time.
- **Customizable Intent Data:** Uses an external JSON file to store intents, making it easy to extend with new topics and responses.

## 🏗️ Tech Stack
- **Python** - Core language
- **Streamlit** - UI framework for building the chatbot interface
- **scikit-learn** - Machine learning library for text processing
- **nltk** - Natural Language Toolkit for text tokenization
- **JSON** - Storage format for chatbot intents
- **CSV** - Logging user-chatbot interactions

## 📂 Project Structure
```
├── app.py                 # Main script for running the chatbot
├── intents.json               # JSON file containing chatbot intents & responses
├── chat_log.csv               # CSV file storing conversation history
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
Ensure you have Python (>=3.8) installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Avik-creator/Chatbot.git
cd chatbot
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Chatbot
```bash
streamlit run chatbot.py
```

## 🛠️ How It Works
1. Loads intents from `intents.json`.
2. Uses **TF-IDF Vectorization** to process and transform text inputs.
3. Applies **Logistic Regression** to predict the best intent match.
4. Retrieves a predefined response corresponding to the detected intent.
5. Logs user-chatbot interactions in `chat_log.csv`.

## 📄 Intent JSON Structure
Example `intents.json` format:
```json
[
    {
        "tag": "greeting",
        "patterns": ["Hello", "Hi there", "Hey!"],
        "responses": ["Hello! How can I assist you today?", "Hi there! What can I do for you?"]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "Bye",
            "See you later",
            "Goodbye",
            "Take care"
        ],
        "responses": [
            "Goodbye",
            "See you later",
            "Take care"
        ]
    },
]
```

## 📜 Logging Chat History
User interactions are saved in `chat_log.csv` in the following format:
```
User Input, Chatbot Response, Timestamp
"Hello", "Hello! How can I assist you today?", "2025-03-24 10:00:00"
```

## 📌 Future Enhancements
- Integrate **Deep Learning models** (e.g., Transformers) for improved response accuracy.
- Add **speech-to-text support** for voice-based interactions.
- Implement **cloud storage** for conversation logs.
- Improve UI/UX with **React-based front-end**.

## 📢 Contributors
This project was created by a group of 5 students from **Maulana Abul Kalam Azad University of Technology, West Bengal** as part of their **Natural Language Processing course**.

- **Nirvik Ghosh**
- **Rudra Banerjee**
- **Snehasis Sardar**
- **Subha Sadhu**
- **Avik Mukherjee**

## 📜 License
This project is licensed under the MIT License.

---
_We appreciate any feedback or contributions! Feel free to fork, modify, and improve the chatbot._ 🎉

