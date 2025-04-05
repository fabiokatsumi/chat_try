# chatbot_app.py
# Gemini RAG Chatbot with Save/Run Prompt Features (including 'run all')
# Current time context: Saturday, April 5, 2025 - São Paulo, Brazil

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

# Use st.cache_data for data processing, st.cache_resource for connections/models
# Caching the vector store creation based on the content/names of uploaded files
@st.cache_resource(show_spinner="Loading and processing PDFs...")
def load_and_process_pdfs(uploaded_files):
    """Loads PDF files, splits them into chunks, creates embeddings, and builds a FAISS vector store."""
    global api_key_configured # Allow modification of the global flag
    if not GOOGLE_API_KEY or not api_key_configured:
        # Don't explicitly show error here, handled in sidebar / main logic
        # This prevents repeated errors during processing attempts before key is valid
        return None # Need API key for embeddings

    if not uploaded_files:
        return None

    all_docs = []
    temp_files = []
    try:
        for uploaded_file in uploaded_files:
            # Create a unique temporary file name using streamlit's file_id if available
            temp_file_path = f"temp_{uploaded_file.file_id}_{uploaded_file.name}" if hasattr(uploaded_file, 'file_id') else f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_file_path) # Keep track for cleanup

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)

        if not all_docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)

        # Re-check API key just before expensive embedding call
        if not GOOGLE_API_KEY or not api_key_configured:
             st.error("Google API Key missing or invalid. Cannot create embeddings.")
             return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        # Success message now displayed in the sidebar logic after calling this function
        return vector_store

    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        return None
    finally:
        # Clean up temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    st.warning(f"Could not remove temporary file {file_path}: {e}")


def get_context_retriever_chain(_vector_store):
    """Creates a chain to retrieve relevant context based on chat history."""
    if not api_key_configured: return None # Need configured API key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1) # Lower temp for deterministic query generation

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
    if not api_key_configured: return None # Need configured API key
    # Using 1.5 Flash here for speed/cost, change to 1.5 Pro if higher quality needed
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
            # Using 1.5 Flash here for speed/cost
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
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
            # Check if response object has 'content' attribute
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response) # Fallback
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

            response = conversation_rag_chain.invoke({
                "chat_history": _chat_history,
                "input": user_query
            })

            if isinstance(response, dict) and 'answer' in response:
                return response['answer']
            else:
                st.warning(f"Unexpected RAG response structure: {response}")
                return str(response) # Return string representation as fallback
        except Exception as e:
            st.error(f"Error during RAG generation: {e}")
            return "Sorry, I encountered an error trying to generate a response using the documents."

# --- Streamlit App ---

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Gemini Chatbot with Document RAG & Saved Prompts")

# --- Initialize session state variables ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Ask me anything, upload PDFs, save prompts ('save prompt as NAME: TEXT'), run saved prompts ('run prompt NAME' or via sidebar), or run all saved prompts ('run all prompts')."),
    ]
if "saved_prompts" not in st.session_state:
    st.session_state.saved_prompts = {} # Dictionary to store {name: prompt_text}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None # Holds the FAISS index
if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = set() # Track IDs of processed files to avoid reprocessing same set

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # API Key Management
    # Use a separate variable for input to avoid overwriting GOOGLE_API_KEY from .env prematurely
    api_key_input = GOOGLE_API_KEY or ""
    if not api_key_configured: # Only show input if not yet successfully configured
        api_key_input = st.text_input("Enter your Google API Key:", value=api_key_input, type="password", key="api_key_input_widget")

    # Configure button / logic
    if not api_key_configured and api_key_input:
        if st.button("Configure API Key"):
            try:
                genai.configure(api_key=api_key_input)
                # Test configuration by listing models (optional but recommended)
                models = list(genai.list_models())
                if not models:
                     raise ValueError("No models listed, API key might be invalid or lack permissions.")
                GOOGLE_API_KEY = api_key_input # Store validated key
                api_key_configured = True
                st.success("API Key Configured Successfully!")
                st.rerun() # Rerun to update UI state
            except Exception as e:
                st.error(f"Failed to configure API Key: {e}")
                api_key_configured = False
    elif api_key_configured:
        st.success("API Key is configured.")
    elif not api_key_input:
         st.warning("Please enter your Google API Key.")


    st.divider()

    # Document Upload and Processing
    st.header("Upload Documents (PDF)")
    uploaded_files = st.file_uploader(
        "Upload PDF documents for RAG",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader",
        disabled=not api_key_configured # Disable if no valid key
    )

    if uploaded_files:
        current_file_ids = {f.file_id for f in uploaded_files if hasattr(f, 'file_id')}
        # Process only if the set of uploaded files has changed
        if current_file_ids != st.session_state.processed_file_ids:
            if api_key_configured:
                # Process PDFs is cached, will run only if files actually changed content/name
                st.session_state.vector_store = load_and_process_pdfs(uploaded_files)
                if st.session_state.vector_store:
                     st.session_state.processed_file_ids = current_file_ids # Update processed IDs on success
                     st.success(f"RAG enabled using {len(st.session_state.processed_file_ids)} document(s).")
                else:
                     # Error handled in load_and_process_pdfs or API key check
                     st.session_state.processed_file_ids = set() # Reset on failure
                     st.session_state.vector_store = None
                     # Don't show redundant error if load_and_process showed one
            else:
                 st.warning("API key must be configured to process documents.")

    elif not uploaded_files and st.session_state.processed_file_ids:
         # Clear vector store if all files are removed by the user
         st.session_state.vector_store = None
         st.session_state.processed_file_ids = set()
         st.info("Document context cleared.")

    # Display RAG status concisely
    if st.session_state.vector_store is not None:
        st.caption(f" RAG active ({len(st.session_state.processed_file_ids)} docs)")
    else:
        st.caption(" RAG inactive")

    st.divider()

    # Saved Prompts Management
    st.header("Saved Prompts")
    if not st.session_state.saved_prompts:
        st.caption("No prompts saved yet.")
        st.caption("Use chat: `save prompt as NAME: TEXT`")
    else:
        # List individual prompts with Run/Delete buttons
        for name in list(st.session_state.saved_prompts.keys()): # Iterate over copy of keys for safe deletion
             if name in st.session_state.saved_prompts: # Check if not deleted during loop
                prompt_text = st.session_state.saved_prompts[name]
                col1, col2, col3 = st.columns([4, 1, 1]) # Adjust column ratios if needed
                with col1:
                    st.text(name)
                    # Optionally show tooltip with full text on hover
                    # st.caption(prompt_text[:30] + "...", help=prompt_text)
                with col2:
                    # Unique key for each button
                    if st.button("Run", key=f"run_{name}", help=f"Run: {prompt_text[:50]}...", use_container_width=True):
                        if not api_key_configured:
                             st.warning("Please configure API key before running prompts.")
                        else:
                            # Flag to run this specific prompt in the main chat logic
                            st.session_state.run_prompt_name = name
                            st.rerun() # Rerun to process the flag
                with col3:
                    # Unique key for each button
                    if st.button("Del", key=f"del_{name}", help="Delete prompt", use_container_width=True):
                        del st.session_state.saved_prompts[name]
                        st.success(f"Prompt '{name}' deleted.")
                        # Clear the run flag if the deleted prompt was flagged
                        if st.session_state.get("run_prompt_name") == name:
                            del st.session_state.run_prompt_name
                        st.rerun() # Rerun to update the list

# --- Chat Interface ---
st.subheader("Chat Window")

# Check if we need to run a SINGLE prompt triggered from the sidebar button
if "run_prompt_name" in st.session_state and st.session_state.run_prompt_name:
    prompt_name_to_run = st.session_state.run_prompt_name
    # Ensure it's cleared BEFORE potential processing to avoid re-triggering on errors below
    del st.session_state.run_prompt_name

    if prompt_name_to_run in st.session_state.saved_prompts:
        # Prepare to run the saved prompt
        user_query = st.session_state.saved_prompts[prompt_name_to_run]
        action_message = f"Running saved prompt: '{prompt_name_to_run}'"
        # Add action message, it will be displayed in the loop below
        st.session_state.chat_history.append(HumanMessage(content=action_message))
        # Let the normal processing flow handle this user_query
        # It will be added again as Human message if needed in the input processing block
    else:
        st.warning(f"Prompt '{prompt_name_to_run}' not found anymore.")
        user_query = None # Prevent processing this cycle

# Get fresh user input only if not running a prompt from sidebar trigger
else:
    user_query = st.chat_input("Type message, 'save prompt as ...', 'run prompt ...', or 'run all prompts'")


# Display chat history RENDER LOOP (runs every time)
chat_container = st.container() # Use a container for potentially better update behavior
with chat_container:
    for message in st.session_state.chat_history:
        avatar = "🤖" if isinstance(message, AIMessage) else "👤"
        with st.chat_message(avatar):
            st.markdown(message.content) # Use markdown for better formatting


# Process new user input (or single saved prompt triggered from sidebar)
if user_query is not None and user_query.strip() != "":

    # Add the user's input/action to history immediately for display context
    # Check if this exact message is already the last one to avoid duplicates from run commands
    if not st.session_state.chat_history or st.session_state.chat_history[-1].content != user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        # Display the newly added message immediately (will be shown again by full render, but gives instant feedback)
        with chat_container: # Add to the container where history is displayed
            with st.chat_message("👤"):
                st.markdown(user_query)

    # --- Command Handling ---
    processed_as_command = False
    normalized_query = user_query.strip().lower()

    # 1. Save Command
    save_prefix = "save prompt as "
    if normalized_query.startswith(save_prefix) and ":" in user_query:
        processed_as_command = True # Assume it's a command attempt
        try:
            command_body = user_query[len(save_prefix):]
            name_part, prompt_text = command_body.split(":", 1)
            prompt_name = name_part.strip()
            prompt_text = prompt_text.strip()

            if prompt_name and prompt_text:
                st.session_state.saved_prompts[prompt_name] = prompt_text
                feedback = f"Prompt '{prompt_name}' saved successfully!"
                st.session_state.chat_history.append(AIMessage(content=feedback))
                st.rerun() # Rerun to update sidebar and display feedback cleanly
            else:
                 err_msg = "Invalid format. Name and prompt text cannot be empty."
                 st.session_state.chat_history.append(AIMessage(content=f"Error: {err_msg}"))
                 st.rerun() # Rerun to show error

        except ValueError:
             err_msg = "Invalid format. Use 'save prompt as NAME: PROMPT TEXT' (colon required)."
             st.session_state.chat_history.append(AIMessage(content=f"Error: {err_msg}"))
             st.rerun() # Rerun to show error

    # 2. Run Single Prompt Command (from chat input)
    run_prefix = "run prompt "
    elif normalized_query.startswith(run_prefix):
        processed_as_command = True # Assume it's a command attempt
        prompt_name_to_run = user_query[len(run_prefix):].strip()
        if prompt_name_to_run in st.session_state.saved_prompts:
            # Retrieve the saved prompt text and treat it as the NEXT user_query to process
            saved_prompt_text = st.session_state.saved_prompts[prompt_name_to_run]
            feedback = f"Okay, running prompt '{prompt_name_to_run}'."
            st.session_state.chat_history.append(AIMessage(content=feedback))
            # Add the actual prompt text to history for processing in the *next* block
            st.session_state.chat_history.append(HumanMessage(content=saved_prompt_text))
            # Don't set user_query here, let the next block handle the new HumanMessage
            processed_as_command = False # IMPORTANT: Allow it to fall through to LLM call
            st.rerun() # Rerun to process the newly added prompt text message

        else:
            feedback = f"Sorry, I couldn't find a saved prompt named '{prompt_name_to_run}'."
            st.session_state.chat_history.append(AIMessage(content=feedback))
            st.rerun() # Rerun to show feedback

    # 3. Run All Prompts Command
    elif normalized_query == "run all prompts":
        processed_as_command = True # Mark as command handled
        # User command "run all prompts" already added to history and displayed

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

            with st.status(f"Running prompt sequence (0/{total_steps})...", expanded=True) as status: # Expand during run
                for name, prompt_text in st.session_state.saved_prompts.items():
                    current_step += 1
                    status.update(label=f"Running prompt sequence ({current_step}/{total_steps}): '{name}'")

                    # Add indication message & prompt text to history
                    sequence_step_message = f"Running prompt: '{name}'"
                    st.session_state.chat_history.append(HumanMessage(content=sequence_step_message))
                    st.session_state.chat_history.append(HumanMessage(content=prompt_text))
                    # Display within status is good for long sequences
                    st.markdown(f"**Running: {name}**\n> {prompt_text}")

                    # Get response using the current history
                    response = get_response(prompt_text, st.session_state.chat_history, vector_store)

                    # Add AI response to history and display in status
                    st.session_state.chat_history.append(AIMessage(content=response))
                    st.markdown(f"**Response:**\n{response}")
                    st.divider()
                    # time.sleep(0.1) # Optional short pause

            # Sequence finished - update status and add final message
            status.update(label="Prompt sequence completed!", state="complete", expanded=False)
            end_message = "--- Finished running all saved prompts ---"
            st.session_state.chat_history.append(AIMessage(content=end_message))

        # Rerun needed to display all history added during the sequence run cleanly
        st.rerun()


    # --- Normal Chat / Processing after 'run prompt NAME' ---
    # This block executes if it wasn't a command OR if 'run prompt NAME' prepared a prompt
    if not processed_as_command:
        # Check if the last message is Human (it should be if we got here legitimately)
        if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
            actual_query_to_process = st.session_state.chat_history[-1].content
            if not api_key_configured:
                 # Add error message as AI response
                 st.session_state.chat_history.append(AIMessage(content="Error: Please configure your Google API Key in the sidebar first."))
                 st.rerun() # Rerun to display the error message
            else:
                # Get response (RAG or basic)
                with st.spinner("Thinking..."):
                    vector_store = st.session_state.get("vector_store", None)
                    response = get_response(actual_query_to_process, st.session_state.chat_history, vector_store)

                # Add AI response to history
                st.session_state.chat_history.append(AIMessage(content=response))
                st.rerun() # Rerun to display the new AI response
        # Else: If last message wasn't Human, something unexpected happened, do nothing this cycle.

# End of script run. Streamlit handles the display based on session state.
