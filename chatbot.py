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



# Streamlit interface in main()
def main():
    # Set page configuration
    st.set_page_config(page_title="BOTeja", layout="centered")
    st.title("BOTeja")
    st.write("I can give you DIY Project Ideas")

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
        
        # Add bot message to chat history
        st.session_state.chat_history.append({"sender": "Boteja", "message": bot_response})
    
# Call the function

if __name__ == '__main__':
    main()