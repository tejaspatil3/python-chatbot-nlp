import os
import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# File to save chat history
CHAT_HISTORY_FILE = "chat_history.txt"

# Load intents from the JSON file
def load_intents(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)["intents"]
    except Exception as e:
        st.error(f"Error loading intents: {e}")
        return []

# Train the chatbot model
def train_chatbot(intents):
    tags = []
    patterns = []

    for intent in intents:
        if 'patterns' in intent and 'tag' in intent:
            for pattern in intent['patterns']:
                tags.append(intent['tag'])
                patterns.append(pattern)
        else:
            st.error("Each intent must have a 'patterns' and 'tag' key.")
            st.stop()

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    clf = LogisticRegression(random_state=0, max_iter=10000)
    x = vectorizer.fit_transform(patterns)
    clf.fit(x, tags)
    return vectorizer, clf

# Chatbot response generator
def chatbot(input_text, vectorizer, clf, intents):
    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]
    
    for intent in intents:
        if intent['tag'] == predicted_tag:
            project_idea = intent.get("project_idea", " ")
            steps = intent.get("steps", [])
            response = f"{project_idea}\n\n"
            response += "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            return response
    return "I'm sorry, I didn't understand that."

# Save chat history to a file
def save_chat_to_file():
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        for message in st.session_state.chat_history:
            file.write(f'{message["sender"]}: {message["message"]}\n')

# Streamlit interface in main()
def main():
    # Set page configuration
    st.set_page_config(page_title="BOTeja", layout="centered")
    st.title("BOTeja")
    st.write("Ask for a DIY project idea, and I'll provide step-by-step instructions!")

    st.markdown(
        """
        <style>
        .chat-bubble {
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            display: inline-block;
            font-family: Arial, sans-serif;
            max-width: 80%;
        }
        .user {
            background-color: #1e90ff;
            color: white;
            text-align: right;
            float: right;
            clear: both;
        }
        .bot {
            background-color: #2ecc71;
            color: white;
            text-align: left;
            float: left;
            clear: both;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Load intents and train model
    intents_file = os.path.abspath("./intents4.json")
    intents = load_intents(intents_file)
    vectorizer, clf = train_chatbot(intents)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        class_name = "bot" if message["sender"] == "Boteja" else "user"
        st.markdown(
            f'<div class="chat-bubble {class_name}">{message["message"]}</div>',
            unsafe_allow_html=True,
        )

    # User input section
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your Message:", key="user_input")
        submitted = st.form_submit_button("Send")

    # Process user input
    if submitted and user_input:
        # Append user message to chat history
        st.session_state.chat_history.append({"sender": "User", "message": user_input})

        # Generate chatbot response
        bot_response = chatbot(user_input, vectorizer, clf, intents)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"sender": "Boteja", "message": bot_response})

        # Save chat history
        save_chat_to_file()

        # Rerun to display updated messages
        st.rerun()

# Run the chatbot
if __name__ == '__main__':
    main()
