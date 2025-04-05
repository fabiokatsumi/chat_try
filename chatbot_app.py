# chatbot_app.py
# Gemini RAG Chatbot with Save/Run Prompt Features (including 'run all')
# Added DEBUG print statements to diagnose "no answer" issue.
# User Context: Saturday, April 5, 2025 2:18 PM - São Paulo, Brazil

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

# --- VERSION WITH DEBUG PRINTS ---
def get_response(user_query, _chat_history, _vector_store):
    """Gets the chatbot's response, using RAG if a vector store is available."""
    print(f"\n--- Entered get_response ---") # DEBUG Start
    print(f"DEBUG: Received query: '{user_query}'") # DEBUG Query
    print(f"DEBUG: API Key Configured: {api_key_configured}") # DEBUG Key status
    print(f"DEBUG: Vector Store Provided: {True if _vector_store else False}") # DEBUG RAG status

    if not api_key_configured:
        print("DEBUG: get_response returning API key error message.") # DEBUG
        return "Error: Google API Key is not configured or is invalid. Please check in the sidebar."

    final_response = None # Initialize response variable

    if _vector_store is None:
        print("--- DEBUG: get_response - Taking Basic Chat Path ---") # DEBUG Basic Path
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ])
            chain = prompt | llm
            print("DEBUG: Invoking basic chain...") # DEBUG Invoking
            response_obj = chain.invoke({
                "chat_history": _chat_history,
                "input": user_query
            })
            print(f"DEBUG: Basic chain raw response object: {type(response_obj)} | {response_obj}") # DEBUG Raw Response

            if hasattr(response_obj, 'content'):
                final_response = response_obj.content
            else:
                final_response = str(response_obj) # Fallback
            print(f"DEBUG: Extracted basic response: '{final_response}'") # DEBUG Extracted Response

        except Exception as e:
            print(f"!!! ERROR during basic chat generation: {type(e).__name__}: {e}", flush=True) # DEBUG Print exception
            st.error(f"Error during basic chat generation: {e}")
            final_response = "Sorry, I encountered an error trying to generate a basic response."

    else:
        print("--- DEBUG: get_response - Taking RAG Path ---") # DEBUG RAG Path
        try:
            print("DEBUG: Getting retriever chain...") # DEBUG
            retriever_chain = get_context_retriever_chain(_vector_store)
            if retriever_chain is None:
                print("!!! ERROR: Retriever chain is None.") # DEBUG
                return "Error: Could not create retriever chain (check API key?)."

            print("DEBUG: Getting RAG chain...") # DEBUG
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            if conversation_rag_chain is None:
                print("!!! ERROR: RAG chain is None.") # DEBUG
                return "Error: Could not create RAG chain (check API key?)."

            print("DEBUG: Invoking RAG chain...") # DEBUG Invoking
            response_obj = conversation_rag_chain.invoke({
                "chat_history": _chat_history,
                "input": user_query
            })
            print(f"DEBUG: RAG chain raw response object: {type(response_obj)} | {response_obj}") # DEBUG Raw Response

            if isinstance(response_obj, dict) and 'answer' in response_obj:
                final_response = response_obj['answer']
            else:
                st.warning(f"Unexpected RAG response structure: {response_obj}")
                final_response = str(response_obj)
            print(f"DEBUG: Extracted RAG response: '{final_response}'") # DEBUG Extracted Response

        except Exception as e:
            print(f"!!! ERROR during RAG generation: {type(e).__name__}: {e}", flush=True) # DEBUG Print exception
            st.error(f"Error during RAG generation: {e}")
            final_response = "Sorry, I encountered an error trying to generate a RAG response."

    print(f"--- Exiting get_response. Returning: '{final_response}' ---") # DEBUG Exit
    # Ensure we always return a string, even if it's empty
    return final_response if final_response is not None else ""
# --- END OF DEBUG get_response ---


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
# Ensure api_key_configured reflects session state if available
if 'api_key_configured' in st.session_state:
    api_key_configured = st.session_state.api_key_configured
elif GOOGLE_API_KEY: # Attempt initial config check if key from env
     try:
        genai.configure(api_key=GOOGLE_API_KEY)
        list(genai.list_models()) # Test call
        api_key_configured = True
        st.session_state.api_key_configured = True
     except Exception:
        api_key_configured = False
        st.session_state.api_key_configured = False
        GOOGLE_API_KEY = None # Clear invalid key from env
else:
    st.session_state.api_key_configured = False


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    # API Key Management using session state flag
    api_key_input_value = GOOGLE_API_KEY if GOOGLE_API_KEY else ""

    if not st.session_state.api_key_configured:
        api_key_input_value = st.text_input("Enter Google API Key:", value=api_key_input_value, type="password", key="api_key_input_widget")
        if api_key_input_value:
             if st.button("Configure API Key", key="config_api_button"):
                 try:
                     genai.configure(api_key=api_key_input_value)
                     models = list(genai.list_models())
                     if not models: raise ValueError("Invalid Key or Permissions")
                     GOOGLE_API_KEY = api_key_input_value
                     st.session_state.api_key_configured = True # Set session flag
                     api_key_configured = True # Update global flag
                     st.success("API Key Configured Successfully!")
                     st.rerun()
                 except Exception as e:
                     st.error(f"API Key Configuration Failed: {e}")
                     st.session_state.api_key_configured = False
                     api_key_configured = False
                     GOOGLE_API_KEY = None
        elif not GOOGLE_API_KEY: # Only warn if no key entered and none from env
            st.warning("Please enter your Google API Key.")

    if st.session_state.api_key_configured:
        api_key_configured = True # Sync global flag just in case
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
        # Add indication message to history directly
        st.session_state.chat_history.append(HumanMessage(content=f"Running saved prompt: '{prompt_name_to_run}'"))
        # user_query_from_sidebar now holds the actual prompt text to be processed
    else:
        st.warning(f"Prompt '{prompt_name_to_run}' not found anymore.")

# Get input from chat box or use sidebar trigger
user_query = st.chat_input("Type message, 'save prompt as ...', 'run prompt ...', or 'run all prompts'", disabled=not api_key_configured)
if user_query_from_sidebar:
    user_query = user_query_from_sidebar # Prioritize sidebar run

# --- Input Processing Logic ---
# This block processes user_query if it exists (from chat input or sidebar)
if user_query is not None and user_query.strip() != "":

    # Add the user's message to history if it's new
    if not st.session_state.chat_history or st.session_state.chat_history[-1].content != user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        # Rerun to show the user's message before processing starts
        st.rerun()

    # --- Process the LATEST Human message in history ---
    # This ensures we process the message just added or the one prepared by 'run prompt' command
    latest_user_message_content = None
    if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
        latest_user_message_content = st.session_state.chat_history[-1].content

    processed_action = False # Flag to check if message was handled as a command/action

    if latest_user_message_content:
        # Define Command Prefixes/Keywords
        save_prefix = "save prompt as "
        run_prefix = "run prompt "
        run_all_command = "run all prompts"
        normalized_query = latest_user_message_content.strip().lower()

        # --- Command Block ---
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
                    err_msg = "Invalid format. Name/text empty."
                    st.session_state.chat_history.append(AIMessage(content=f"Error: {err_msg}"))
            except ValueError:
                 err_msg = "Invalid format. Use 'save prompt as NAME: TEXT'"
                 st.session_state.chat_history.append(AIMessage(content=f"Error: {err_msg}"))
            st.rerun()

        # 2. Run Single Prompt Command
        elif normalized_query.startswith(run_prefix):
            processed_action = True
            prompt_name_to_run = latest_user_message_content[len(run_prefix):].strip()
            if prompt_name_to_run in st.session_state.saved_prompts:
                saved_prompt_text = st.session_state.saved_prompts[prompt_name_to_run]
                feedback = f"Okay, running prompt '{prompt_name_to_run}'."
                st.session_state.chat_history.append(AIMessage(content=feedback))
                st.session_state.chat_history.append(HumanMessage(content=saved_prompt_text)) # Add actual prompt for next cycle
            else:
                feedback = f"Prompt '{prompt_name_to_run}' not found."
                st.session_state.chat_history.append(AIMessage(content=feedback))
            st.rerun()

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
                start_message = "--- Starting sequence: Running all saved prompts ---"
                st.session_state.chat_history.append(AIMessage(content=start_message))
                vector_store = st.session_state.get("vector_store", None)
                current_step = 0
                total_steps = len(st.session_state.saved_prompts)
                with st.status(f"Running prompt sequence (0/{total_steps})...", expanded=True) as status:
                    for name, prompt_text in st.session_state.saved_prompts.items():
                        current_step += 1
                        status.update(label=f"Running sequence ({current_step}/{total_steps}): '{name}'")
                        st.session_state.chat_history.append(HumanMessage(content=f"Running prompt: '{name}'"))
                        st.session_state.chat_history.append(HumanMessage(content=prompt_text))
                        st.markdown(f"**Running: {name}**\n> {prompt_text}")
                        response = get_response(prompt_text, st.session_state.chat_history, vector_store)
                        st.session_state.chat_history.append(AIMessage(content=response))
                        st.markdown(f"**Response:**\n{response}")
                        st.divider()
                status.update(label="Prompt sequence completed!", state="complete", expanded=False)
                end_message = "--- Finished running all saved prompts ---"
                st.session_state.chat_history.append(AIMessage(content=end_message))
            st.rerun() # Rerun needed to display results cleanly
        # --- End Command Block ---


        # --- Normal Chat Processing (if latest message wasn't a command) ---
        if not processed_action:
            print(f"--- Entering Normal Chat Processing Block for: '{latest_user_message_content}' ---") # DEBUG
            if not api_key_configured:
                 if not st.session_state.chat_history or st.session_state.chat_history[-1].content != "Error: API Key not configured. Cannot process message.":
                     print("DEBUG: Adding API key error message.") # DEBUG
                     st.session_state.chat_history.append(AIMessage(content="Error: API Key not configured. Cannot process message."))
                     st.rerun() # Rerun to display the error message
            else:
                # Get LLM response for the latest user message
                print(f"DEBUG: Calling get_response function for query: '{latest_user_message_content}'") # DEBUG
                response_content = None # Initialize
                with st.spinner("Thinking..."):
                    vector_store = st.session_state.get("vector_store", None)
                    # Pass the actual message content string
                    response_content = get_response(latest_user_message_content, st.session_state.chat_history, vector_store)

                print(f"DEBUG: Received from get_response: '{response_content}' (Type: {type(response_content)})") # DEBUG

                # Add AI response to history ONLY if it's not None or empty/whitespace
                if response_content and response_content.strip():
                    print("DEBUG: Appending AI message to history.") # DEBUG
                    st.session_state.chat_history.append(AIMessage(content=response_content))
                    print("DEBUG: Rerunning to display AI response.") # DEBUG
                    st.rerun() # Rerun ONLY after getting the response to display it
                else:
                    print("!!! WARNING: get_response returned empty/None content. No AI message added. Rerunning anyway.") # DEBUG
                    # Maybe add a generic failure message?
                    # st.session_state.chat_history.append(AIMessage(content="[I could not generate a response for that.]"))
                    st.rerun() # Rerun even if no response to clear spinner etc.
        # --- End Normal Chat Processing ---

# --- Display Chat History ---
# This loop runs regardless of whether new input was processed or not, ensuring history is always shown
st.divider() # Visual separator before chat history display
st.write("Chat History:") # Add a label for clarity
chat_display_area = st.container(height=500) # Use a container with fixed height for scroll
with chat_display_area:
    for message in st.session_state.chat_history:
        avatar = "🤖" if isinstance(message, AIMessage) else "👤"
        with st.chat_message(avatar):
             st.markdown(str(message.content))

