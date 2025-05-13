import streamlit as stl
import os
from dotenv import load_dotenv

from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')

def initialize_memory(length: int) -> ConversationBufferWindowMemory:
    """Create a memory buffer with a defined conversational window."""
    return ConversationBufferWindowMemory(k=length)

def initialize_model(model_name: str) -> ChatGroq:
    """Instantiate a ChatGroq model with the selected name."""
    try:
        return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    except Exception as e:
        stl.error(f"Error initializing model: {e}")
        stl.stop()

def restore_memory_from_history(memory, history: list):
    """Reconstruct memory from saved session history."""
    for message in history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )

def display_chat_history(history: list):
    """Render the entire chat history."""
    for message in history:
        with stl.chat_message("user"):
            stl.write(message['human'])
        with stl.chat_message("assistant"):
            stl.write(message['AI'])

def main():
    """Main function to launch the Streamlit chatbot UI."""
    stl.title("🤖 Welcome to Srika – Your Personal AI Chatterbot")
    stl.sidebar.title('Model Selection')

    # Sidebar options
    selected_model = stl.sidebar.selectbox(
        'Pick your poison:',
        [
            'mistral-saba-24b',
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'gemma2-9b-it',
            'deepseek-r1-distill-llama-70b',
            'qwen-qwq-32b'
        ]
    )

    memory_length = stl.sidebar.slider(
        'Conversational memory length:', min_value=1, max_value=10, value=5
    )

    # Initialize memory and chat history
    memory = initialize_memory(memory_length)

    if 'chat_history' not in st.session_state:
        stl.session_state.chat_history = []
    else:
        restore_memory_from_history(memory, st.session_state.chat_history)

    user_input = stl.chat_input("Say anything...")

    # Set up the conversation chain
    chat_model = initialize_model(selected_model)
    conversation = ConversationChain(llm=chat_model, memory=memory)

    # Process user input
    if user_input:
        response = conversation(user_input)
        ai_reply = response['response'] if isinstance(response, dict) else str(response)

        # Store new exchange in session state
        new_message = {'human': user_input, 'AI': ai_reply}
        stl.session_state.chat_history.append(new_message)

        # Display latest response
        stl.write("Chatbot:", ai_reply)

    # Show full chat history
    display_chat_history(st.session_state.chat_history)

if __name__ == "__main__":
    main()
