# chatbot_app.py
# Gemini RAG Chatbot with Save/Run Prompt Features (including 'run all')
# Version with corrected indentation for command handling block.
# User Context: Saturday, April 5, 2025 - São Paulo, Brazil

import streamlit as st
import google.generativeai as genai
import os
import time # Imported for potential use (e.g., small delays), currently commented out
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
api_key_configured = False # Flag to track runtime configuration

# --- Helper Functions ---

@st.cache_resource(show_spinner="Loading and processing PDFs...")
def load_and_process_pdfs(uploaded_files):
    """Loads PDF files, splits them into chunks, creates embeddings, and builds a FAISS vector store."""
    global api_key_configured
    if not GOOGLE_API_KEY or not api_key_configured:
        return None

    if not uploaded_files:
        return None

    all_docs = []
    temp_files = []
    try:
        for uploaded_file in uploaded_files:
            temp_file_path = f"temp_{uploaded_file.file_id}_{uploaded_file.name}" if hasattr(uploaded_file, 'file_id') else f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_file_path)

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)

        if not all_docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)

        if not GOOGLE_API_KEY or not api_key_configured:
             st.error("Google API Key missing or invalid. Cannot create embeddings.")
             return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        return vector_store

    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        return None
    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    st.warning(f"Could not remove temporary file {file_path}: {e}")


def get_context_retriever_chain(_vector_store):
    """Creates a chain to retrieve relevant context based on chat history."""
    if not api_key_configured: return None
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
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
    if not api_key_configured: return None
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based ONLY on the below context:\n\n{context}\n\nIf the answer is not in the context, state that clearly based on the provided documents. Do not make up information."),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(_retriever_chain, stuff_documents_chain)

def get_response(user_query, _chat_history, _vector_store):
    """Gets the chatbot's response, using RAG if a vector store is available."""
    if not api_key_configured:
        return "Error: Google API Key is not configured or is invalid. Please check in the sidebar."

    if _vector_store is None:
        # Basic chat without RAG
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ])
            chain = prompt | llm
            response = chain.invoke({"chat_history": _chat_history, "input": user_query})
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            st.error(f"Error during basic chat generation: {e}")
            return "Sorry, I encountered an error trying to generate a response."
    else:
        # RAG chat
        try:
            retriever_chain = get_context_retriever_chain(_vector_store)
            if retriever_chain is None: return "Error: Could not create retriever chain (check API key?)."
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            if conversation_rag_chain is None: return "Error: Could not create RAG chain (check API key?)."
            response = conversation_rag_chain.invoke({"chat_history": _chat_history, "input": user_query})
            return response['answer'] if isinstance(response, dict) and 'answer' in response else str(response)
        except Exception as e:
            st.error(f"Error during RAG generation: {e}")
            return "Sorry, I encountered an error trying to generate a response using the documents."

# --- Streamlit App ---

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Gemini Chatbot with Document RAG & Saved Prompts")

# --- Initialize session state variables ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! Ask me anything, upload PDFs, save/run prompts.")]
if "saved_prompts" not in st.session_state:
    st.session_state.saved_prompts = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = set()
if "run_prompt_name" not in st.session_state:
    st.session_state.run_prompt_name = None

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    # API Key Management (simplified check)
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = bool(GOOGLE_API_KEY) # Initialize from .env if present

    api_key_input_value = GOOGLE_API_KEY if st.session_state.api_key_configured else ""

    if not st.session_state.api_key_configured:
        api_key_input_value = st.text_input("Enter Google API Key:", value=api_key_input_value, type="password", key="api_key_input_widget")
        if api_key_input_value: # Check if user entered anything
             if st.button("Configure API Key", key="config_api_button"):
                 try:
                     genai.configure(api_key=api_key_input_value)
                     models = list(genai.list_models()) # Test call
                     if not models: raise ValueError("Invalid Key or Permissions")
                     GOOGLE_API_KEY = api_key_input_value # Store validated key globally
                     st.session_state.api_key_configured = True # Set session flag
                     api_key_configured = True # Update global flag as well
                     st.success("API Key Configured Successfully!")
                     st.rerun()
                 except Exception as e:
                     st.error(f"API Key Configuration Failed: {e}")
                     st.session_state.api_key_configured = False
                     api_key_configured = False # Reset global flag
                     GOOGLE_API_KEY = None
        elif not api_key_input_value and not GOOGLE_API_KEY: # Only warn if nothing entered and nothing from env
            st.warning("Please enter your Google API Key.")

    if st.session_state.api_key_configured:
        api_key_configured = True # Ensure global flag is sync with session if already set
        st.success("API Key is configured.")

    st.divider()
    # Document Upload
    st.header("Upload Documents (PDF)")
    uploaded_files = st.file_uploader("Upload PDFs for RAG", type="pdf", accept_multiple_files=True, key="file_uploader", disabled=not api_key_configured)

    if uploaded_files:
        current_file_ids = {f.file_id for f in uploaded_files if hasattr(f, 'file_id')}
        if current_file_ids != st.session_state.processed_file_ids:
            if api_key_configured:
                st.session_state.vector_store = load_and_process_pdfs(uploaded_files)
                if st.session_state.vector_store:
                     st.session_state.processed_file_ids = current_file_ids
                     st.success(f"RAG enabled using {len(st.session_state.processed_file_ids)} document(s).")
                else:
                     st.session_state.processed_file_ids = set()
                     st.session_state.vector_store = None
            else:
                 st.warning("API key must be configured to process documents.")
    elif not uploaded_files and st.session_state.processed_file_ids:
         st.session_state.vector_store = None
         st.session_state.processed_file_ids = set()
         st.info("Document context cleared.")

    if st.session_state.vector_store is not None: st.caption(f" RAG active ({len(st.session_state.processed_file_ids)} docs)")
    else: st.caption(" RAG inactive")
    st.divider()
    # Saved Prompts
    st.header("Saved Prompts")
    if not st.session_state.saved_prompts: st.caption("No prompts saved yet.")
    else:
        for name in list(st.session_state.saved_prompts.keys()):
             if name in st.session_state.saved_prompts:
                prompt_text = st.session_state.saved_prompts[name]
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1: st.text(name)
                with col2:
                    if st.button("Run", key=f"run_{name}", help=f"Run: {prompt_text[:50]}...", use_container_width=True, disabled=not api_key_configured):
                        st.session_state.run_prompt_name = name
                        st.rerun()
                with col3:
                    if st.button("Del", key=f"del_{name}", help="Delete prompt", use_container_width=True):
                        del st.session_state.saved_prompts[name]
                        if st.session_state.run_prompt_name == name: st.session_state.run_prompt_name = None
                        st.success(f"Prompt '{name}' deleted.")
                        st.rerun()

# --- Chat Interface ---
st.subheader("Chat Window")

# Handle trigger from sidebar run button
user_query_from_sidebar = None
if st.session_state.run_prompt_name:
    prompt_name_to_run = st.session_state.run_prompt_name
    st.session_state.run_prompt_name = None # Clear flag immediately
    if prompt_name_to_run in st.session_state.saved_prompts:
        user_query_from_sidebar = st.session_state.saved_prompts[prompt_name_to_run]
        st.session_state.chat_history.append(HumanMessage(content=f"Running saved prompt: '{prompt_name_to_run}'"))
    else:
        st.warning(f"Prompt '{prompt_name_to_run}' not found anymore.")

# Get input from chat box or use sidebar trigger
user_query = st.chat_input("Type message, 'save prompt as ...', 'run prompt ...', or 'run all prompts'")
if user_query_from_sidebar:
    user_query = user_query_from_sidebar # Prioritize sidebar run

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        avatar = "🤖" if isinstance(message, AIMessage) else "👤"
        with st.chat_message(avatar):
             st.markdown(str(message.content))


# Process new user input (only if query exists)
if user_query is not None and user_query.strip() != "":

    # Add the user's input/action to history if it's not already the last message
    if not st.session_state.chat_history or st.session_state.chat_history[-1].content != user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        # Rerun immediately to show the user message before processing
        # This makes the flow: type message -> see message -> see response/feedback
        st.rerun()

    # --- Define Command Prefixes ---
    save_prefix = "save prompt as "
    run_prefix = "run prompt "
    run_all_command = "run all prompts"

    # --- Process the LATEST Human message in history ---
    latest_user_message_content = None
    if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
        latest_user_message_content = st.session_state.chat_history[-1].content

    processed_action = False # Flag to track if the latest message led to an action

    if latest_user_message_content:
        normalized_query = latest_user_message_content.strip().lower()

        # --- Command Block ---
        # This block processes the latest user message if it matches a command format

        # 1. Save Command
        if normalized_query.startswith(save_prefix) and ":" in latest_user_message_content:
            processed_action = True
            try:
                command_body = latest_user_message_content[len(save_prefix):]
                name_part, prompt_text = command_body.split(":", 1)
                prompt_name = name_part.strip()
                prompt_text = prompt_text.strip()
                if prompt_name and prompt_text:
                    st.session_state.saved_prompts[prompt_name] = prompt_text
                    feedback = f"Prompt '{prompt_name}' saved successfully!"
                    st.session_state.chat_history.append(AIMessage(content=feedback))
                else:
                    err_msg = "Invalid format. Name and prompt text cannot be empty."
                    st.session_state.chat_history.append(AIMessage(content=f"Error: {err_msg}"))
            except ValueError:
                 err_msg = "Invalid format. Use 'save prompt as NAME: PROMPT TEXT' (colon required)."
                 st.session_state.chat_history.append(AIMessage(content=f"Error: {err_msg}"))
            st.rerun() # Rerun after save attempt

        # 2. Run Single Prompt Command
        elif normalized_query.startswith(run_prefix):
            processed_action = True
            prompt_name_to_run = latest_user_message_content[len(run_prefix):].strip()
            if prompt_name_to_run in st.session_state.saved_prompts:
                saved_prompt_text = st.session_state.saved_prompts[prompt_name_to_run]
                feedback = f"Okay, running prompt '{prompt_name_to_run}'."
                st.session_state.chat_history.append(AIMessage(content=feedback))
                # Add the actual prompt text as the next message to be processed
                st.session_state.chat_history.append(HumanMessage(content=saved_prompt_text))
            else:
                feedback = f"Sorry, I couldn't find a saved prompt named '{prompt_name_to_run}'."
                st.session_state.chat_history.append(AIMessage(content=feedback))
            st.rerun() # Rerun to process the added prompt or show feedback

        # 3. Run All Prompts Command
        elif normalized_query == run_all_command:
            processed_action = True
            if not st.session_state.saved_prompts:
                feedback = "No saved prompts to run."
                st.session_state.chat_history.append(AIMessage(content=feedback))
            elif not api_key_configured:
                feedback = "API Key not configured. Cannot run prompts."
                st.session_state.chat_history.append(AIMessage(content=feedback))
            else:
                # Execute Sequence
                start_message = "--- Starting sequence: Running all saved prompts ---"
                st.session_state.chat_history.append(AIMessage(content=start_message))

                vector_store = st.session_state.get("vector_store", None)
                current_step = 0
                total_steps = len(st.session_state.saved_prompts)

                with st.status(f"Running prompt sequence (0/{total_steps})...", expanded=True) as status:
                    for name, prompt_text in st.session_state.saved_prompts.items():
                        current_step += 1
                        status.update(label=f"Running sequence ({current_step}/{total_steps}): '{name}'")
                        # Add messages to history for processing
                        st.session_state.chat_history.append(HumanMessage(content=f"Running prompt: '{name}'"))
                        st.session_state.chat_history.append(HumanMessage(content=prompt_text))
                        # Display step info in status
                        st.markdown(f"**Running: {name}**\n> {prompt_text}")
                        response = get_response(prompt_text, st.session_state.chat_history, vector_store)
                        st.session_state.chat_history.append(AIMessage(content=response))
                        # Display response in status
                        st.markdown(f"**Response:**\n{response}")
                        st.divider()

                status.update(label="Prompt sequence completed!", state="complete", expanded=False)
                end_message = "--- Finished running all saved prompts ---"
                st.session_state.chat_history.append(AIMessage(content=end_message))
            # Rerun needed to display all history added during the sequence run cleanly
            st.rerun()
        # --- End of Command Block ---

    # --- Normal Chat Processing ---
    # Process the latest human message if it wasn't a command AND if API key is OK
    if latest_user_message_content and not processed_action:
        if not api_key_configured:
             # Add error message only if not already the last message
             if not st.session_state.chat_history or st.session_state.chat_history[-1].content != "Error: API Key not configured. Cannot process message.":
                 st.session_state.chat_history.append(AIMessage(content="Error: API Key not configured. Cannot process message."))
                 st.rerun()
        else:
            # Get LLM response for the latest user message
            with st.spinner("Thinking..."):
                vector_store = st.session_state.get("vector_store", None)
                response = get_response(latest_user_message_content, st.session_state.chat_history, vector_store)

            # Add AI response to history
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun() # Rerun to display the new AI response

# End of script run. Streamlit displays the state.
