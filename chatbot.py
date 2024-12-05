import os
import json
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents from the JSON file
def load_intents(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
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
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

# Function to save chat history to a file
def save_chat_history(chat_name):
    # Save chat history with the provided chat_name
    if st.session_state.chat_history:
        chat_filename = f"{chat_name}_chat_history.json"
        chat_data = {"conversation": st.session_state.chat_history}

        # Write to JSON file
        with open(chat_filename, 'w') as f:
            json.dump(chat_data, f, indent=4)
        st.success(f"Chat history saved as {chat_filename}")

# Function to load and display previous chats
def load_chat_history(chat_name):
    chat_filename = f"{chat_name}_chat_history.json"
    if os.path.exists(chat_filename):
        with open(chat_filename, 'r') as f:
            chat_data = json.load(f)
        return chat_data.get("conversation", [])
    return []


# Streamlit interface in main()
def main():
    # Set page configuration
    st.set_page_config(page_title="Chatbot Interface", layout="centered")

    st.markdown(
        """
        <style>
        body {
            background-color: #002b5b;
        }
        .chat-bubble {
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            display: inline-block;
            color: white;
            font-family: Arial, sans-serif;
        }
        .user {
            display: flex;
            justify-content: flex-end;
            margin-top: 20px;
            background-color: #1e90ff;
            text-align: right;
        }
        .bot {
            display: flex;
            
            margin-top: 20px;
            background-color: #2ecc71;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    #Getting the intents.json file
    intents_file = os.path.abspath("./intents.json")

    # Load intents and train model
    intents = load_intents(intents_file)
    vectorizer, clf = train_chatbot(intents)

    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input for chat name (if starting a new conversation)
    if "chat_name" not in st.session_state:
        chat_name = st.text_input("Enter chat name to start a new conversation:", key="chat_name")
        if chat_name:
            # Load chat history from a file if it exists
            st.session_state.chat_history = load_chat_history(chat_name)
    else:
        chat_name = st.session_state.chat_name
    
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

        # Generate bot response (Placeholder response for now)
        # Generate chatbot response
        # bot_response = f"Boteja: You said, '{user_input}'"
        bot_response = chatbot(user_input, vectorizer, clf, intents)
        
        # Add bot message to chat history
        st.session_state.chat_history.append({"sender": "Boteja", "message": bot_response})
    


    # Buttons for saving and starting a new conversation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save Chat"):
            if chat_name:
                save_chat_history(chat_name)
            else:
                st.warning("Please enter a chat name first.")
    with col2:
        if st.button("Start New Chat"):
            # Clear chat history and ask for new chat name
            st.session_state.chat_history = []
            st.session_state.chat_name = None
            st.experimental_user = ""


# Call the function

if __name__ == '__main__':
    main()