import streamlit as st  # For creating the chatbot web app UI
import os  # To interact with environment variables
from dotenv import load_dotenv  # To load variables from a .env file into the environment

from groq import Groq  # Used internally by ChatGroq for API integration
from langchain.chains import ConversationChain  # For handling a conversational LLM chain
from langchain.memory import ConversationBufferMemory  # Memory that stores entire chat history
from langchain_groq import ChatGroq  # LangChain wrapper for Groq LLMs

# Load environment variables from .env file (e.g., GROQ_API_KEY)
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')  # Securely access the API key from environment

def initialize_memory():
    """Create unlimited conversation memory."""
    return ConversationBufferMemory()  # Use BufferMemory to retain all previous messages

def initialize_model(model_name: str) -> ChatGroq:
    """Instantiate a ChatGroq model with the selected name."""
    try:
        return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)  # Connect to Groq LLM with chosen model
    except Exception as e:
        st.error(f"Oops! Couldn’t wake up Rhea’s brain: {e}")  # User-friendly error if model load fails
        st.stop()  # Stop app execution if model cannot be initialized

def restore_memory_from_history(memory, history: list):
    """Reconstruct memory from saved session history."""
    for message in history:
        memory.save_context(
            {'input': message['human']},  # Feed past user message
            {'output': message['AI']}    # Feed past AI reply
        )  # This helps preserve continuity across reruns

def display_chat_history(history: list):
    """Render the entire chat history."""
    for message in history:
        with st.chat_message("user"):
            st.write(message['human'])  # Show each user's message
        with st.chat_message("assistant"):
            st.write(message['AI'])  # Show each AI's response

def main():
    """Main function to launch the Streamlit chatbot UI."""
    st.title("\ud83d\udcac Meet Rhea – Your Personal AI Companion")  # Main heading
    st.markdown("Hi! I'm **Rhea**, your AI buddy. Let's talk about anything. \ud83d\ude0a")  # Friendly intro message

    st.sidebar.title('\ud83e\udde0 Model Brain Picker')  # Sidebar header
    selected_model = st.sidebar.selectbox(
        'Choose a brain for Rhea:',  # Dropdown label
        [
            'mistral-saba-24b',
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'gemma2-9b-it',
            'deepseek-r1-distill-llama-70b',
            'qwen-qwq-32b'
        ]  # Choices provided to the user
    )

    memory = initialize_memory()  # Set up persistent memory

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Initialize new session history
    else:
        restore_memory_from_history(memory, st.session_state.chat_history)  # Load memory if exists

    user_input = st.chat_input("Type your message to Rhea...")  # Chat input UI for the user

    chat_model = initialize_model(selected_model)  # Load selected LLM model
    conversation = ConversationChain(llm=chat_model, memory=memory)  # Link model and memory into conversation

    if user_input:
        response = conversation(user_input)  # Get response from AI
        ai_reply = response['response'] if isinstance(response, dict) else str(response)  # Clean format

        new_message = {'human': user_input, 'AI': ai_reply}  # Store user-AI message pair
        st.session_state.chat_history.append(new_message)  # Save to session history

        st.write("**Rhea:**", ai_reply)  # Display Rhea's reply

    display_chat_history(st.session_state.chat_history)  # Show all messages

if __name__ == "__main__":
    main()  # Run the Streamlit app
