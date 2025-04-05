import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Helper Functions ---

@st.cache_resource(show_spinner="Loading and processing PDF...")
def load_and_process_pdfs(uploaded_files):
    """Loads PDF files, splits them into chunks, creates embeddings, and builds a FAISS vector store."""
    if not uploaded_files:
        return None

    all_docs = []
    for uploaded_file in uploaded_files:
        # Save temp file to disk (PyPDFLoader needs a file path)
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(uploaded_file.name) # Clean up temp file

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

def get_context_retriever_chain(_vector_store):
    """Creates a chain to retrieve relevant context based on chat history."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY) # Use a fast model for context retrieval

    retriever = _vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(_retriever_chain):
    """Creates the main RAG chain for answering questions based on retrieved context."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY) # Use the main model for generation

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}. If the context doesn't contain the answer, say you don't have enough information from the provided documents."),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(_retriever_chain, stuff_documents_chain)

def get_response(user_query, _chat_history, _vector_store):
    """Gets the chatbot's response, using RAG if a vector store is available."""
    if _vector_store is None:
        # Basic chat without RAG
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        chain = prompt | llm
        response = chain.invoke({
            "chat_history": _chat_history,
            "input": user_query
        })
        return response.content
    else:
        # RAG chat
        retriever_chain = get_context_retriever_chain(_vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        response = conversation_rag_chain.invoke({
            "chat_history": _chat_history,
            "input": user_query
        })
        return response['answer']


# --- Streamlit App ---

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖")
st.title("🤖 Gemini Chatbot with Document RAG")

# --- Sidebar for API Key and Document Upload ---
with st.sidebar:
    st.header("Configuration")

    # Check if API key is loaded
    api_key_configured = bool(GOOGLE_API_KEY)

    if not api_key_configured:
        GOOGLE_API_KEY = st.text_input("Enter your Google API Key:", type="password")
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            st.success("API Key Configured!")
            api_key_configured = True
        else:
            st.warning("Please enter your Google API Key to use the chatbot.")
            st.stop() # Stop execution if no API key
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        st.success("API Key loaded successfully!")


    st.header("Upload Documents (PDF)")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents here",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader" # Add a key to manage state better
    )

    if uploaded_files:
        if "vector_store" not in st.session_state or st.session_state.get("processed_files") != [f.name for f in uploaded_files]:
            # Process only if new files are uploaded or vector store doesn't exist
            st.session_state.vector_store = load_and_process_pdfs(uploaded_files)
            st.session_state.processed_files = [f.name for f in uploaded_files] # Store names of processed files
            if st.session_state.vector_store:
                st.success("Documents processed successfully! You can now ask questions about them.")
            else:
                st.error("Failed to process documents.")
        elif st.session_state.vector_store:
            st.info("Documents already processed.")
    elif "vector_store" in st.session_state:
        # Clear vector store if files are removed
         if st.button("Clear Uploaded Documents Context"):
            del st.session_state.vector_store
            if "processed_files" in st.session_state:
                del st.session_state.processed_files
            st.rerun()


# --- Chat Interface ---

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Ask me anything, or upload PDFs to ask questions about specific documents."),
    ]

# Retrieve vector store from session state (if processed)
vector_store = st.session_state.get("vector_store", None)

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query.strip() != "":
    if not api_key_configured:
         st.error("Please configure your Google API Key in the sidebar first.")
    else:
        # Add user message to history and display
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Get response (RAG or basic)
        with st.spinner("Thinking..."):
            response = get_response(user_query, st.session_state.chat_history, vector_store)

        # Add AI response to history and display
        st.session_state.chat_history.append(AIMessage(content=response))
        with st.chat_message("AI"):
            st.markdown(response)

        # Scroll to bottom (optional, might need adjustments depending on Streamlit version)
        # st.experimental_rerun() # Can sometimes help, but might clear input
