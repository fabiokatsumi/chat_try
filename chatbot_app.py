import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("💬 Simple Chatbot with Gemini & LangChain")
st.caption("A basic Streamlit chatbot using Google's Gemini Pro API via LangChain.")

# --- API Key Handling ---
# Option 1: Get API key from sidebar input (more interactive)
google_api_key = st.sidebar.text_input("Google API Key", type="password", key="api_key_input")
st.sidebar.markdown(
    "Get your Google API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)"
)

# Option 2: Get API key from environment variable (more secure for deployment)
# Make sure to set the GOOGLE_API_KEY environment variable if using this.
# google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        # You can add a default system message if needed
        # SystemMessage(content="You are a helpful AI assistant.")
    ]

# --- Helper Function to Initialize LLM ---
def get_llm(api_key):
    """Initializes the ChatGoogleGenerativeAI model."""
    # Ensure convert_system_message_to_human=True for models that might not natively support SystemMessages yet
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",  # Or choose another model like "gemini-1.5-flash"
            google_api_key=api_key,
            convert_system_message_to_human=True,
            temperature=0.7 # Adjust creativity (0 = deterministic, 1 = more creative)
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize the LLM. Check your API key and configuration. Error: {e}")
        return None

# --- Display Existing Chat Messages ---
st.write("---") # Separator
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    # elif isinstance(message, SystemMessage): # Optional: Display system messages if needed
    #     with st.chat_message("system"):
    #         st.info(message.content) # Or use st.markdown

# --- Handle User Input and Chat Logic ---
if prompt := st.chat_input("Ask me anything..."):
    # 1. Check if API key is provided
    if not google_api_key:
        st.warning("Please enter your Google API Key in the sidebar to chat.")
        st.stop() # Stop execution if no API key

    # 2. Initialize LLM (only if API key is present)
    llm = get_llm(google_api_key)
    if not llm: # If LLM initialization failed
        st.stop()

    # 3. Add user message to session state and display it
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Generate and display AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Create a placeholder for the response
        full_response = ""
        try:
            with st.spinner("Thinking..."):
                 # Directly invoke the LLM with the current message history
                 # LangChain's ChatGoogleGenerativeAI takes the list of messages directly
                ai_response = llm.invoke(st.session_state.messages)
                full_response = ai_response.content

            # Display the final response
            message_placeholder.markdown(full_response)

            # 5. Add AI response to session state
            st.session_state.messages.append(AIMessage(content=full_response))

        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
            # Optional: Remove the last user message if the API call failed?
            # if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
            #     st.session_state.messages.pop()


# --- Initial message ---
if len(st.session_state.messages) == 0:
     st.info("Enter your Google API Key in the sidebar and type a message below to start chatting!")
