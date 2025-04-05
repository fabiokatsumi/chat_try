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
api_key_configured = False # Flag to track runtime configuration

# --- Helper Functions ---

# Use st.cache_data for data processing, st.cache_resource for connections/models
# Caching the vector store creation based on the content/names of uploaded files
@st.cache_resource(show_spinner="Loading and processing PDFs...")
def load_and_process_pdfs(uploaded_files):
    """Loads PDF files, splits them into chunks, creates embeddings, and builds a FAISS vector store."""
    global api_key_configured
    if not api_key_configured:
        st.error("API Key not configured. Cannot process documents.")
        return None # Need API key for embeddings

    if not uploaded_files:
        return None

    all_docs = []
    temp_files = []
    try:
        for uploaded_file in uploaded_files:
            # Create a unique temporary file name
            temp_file_path = f"temp_{uploaded_file.file_id}_{uploaded_file.name}"
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

        # Ensure API key is available before creating embeddings
        if not GOOGLE_API_KEY:
             st.error("Google API Key is missing. Cannot create embeddings.")
             return None

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        st.success(f"Processed {len(uploaded_files)} PDF(s) into {len(split_docs)} chunks.")
        return vector_store

    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        return None
    finally:
        # Clean up temporary files
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)


def get_context_retriever_chain(_vector_store):
    """Creates a chain to retrieve relevant context based on chat history."""
    if not GOOGLE_API_KEY: return None # Need API key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

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
    if not GOOGLE_API_KEY: return None # Need API key
    # Using 1.5 Flash here for speed, change to 1.5 Pro if needed
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based ONLY on the below context:\n\n{context}\n\nIf the answer is not in the context, say you don't have that information in the provided documents."),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(_retriever_chain, stuff_documents_chain)

def get_response(user_query, _chat_history, _vector_store):
    """Gets the chatbot's response, using RAG if a vector store is available."""
    if not GOOGLE_API_KEY:
        return "Error: Google API Key is not configured."

    if _vector_store is None:
        # Basic chat without RAG
        try:
            # Using 1.5 Flash here for speed
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question concisely."),
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
                # Handle cases where response might be a string or other structure
                return str(response)
        except Exception as e:
            st.error(f"Error during basic chat generation: {e}")
            return "Sorry, I encountered an error trying to generate a response."

    else:
        # RAG chat
        try:
            retriever_chain = get_context_retriever_chain(_vector_store)
            if retriever_chain is None: return "Error: Could not create retriever chain."

            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            if conversation_rag_chain is None: return "Error: Could not create RAG chain."

            response = conversation_rag_chain.invoke({
                "chat_history": _chat_history,
                "input": user_query
            })

            # Check response structure from create_retrieval_chain
            if isinstance(response, dict) and 'answer' in response:
                return response['answer']
            else:
                # Fallback or handle unexpected structure
                st.warning(f"Unexpected RAG response structure: {response}")
                return str(response) # Return string representation as fallback
        except Exception as e:
            st.error(f"Error during RAG generation: {e}")
            return "Sorry, I encountered an error trying to generate a response using the documents."

# --- Streamlit App ---

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖")
st.title("🤖 Gemini Chatbot with Document RAG")

# --- Initialize session state variables ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Ask me anything, or upload PDFs in the sidebar to ask questions about them. You can also save prompts using 'save prompt as NAME: TEXT' and run them with 'run prompt NAME' or via the sidebar."),
    ]
if "saved_prompts" not in st.session_state:
    st.session_state.saved_prompts = {} # Dictionary to store {name: prompt_text}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None # Holds the FAISS index
if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = set() # Track IDs of processed files

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # API Key Management
    if not GOOGLE_API_KEY:
        api_key_input = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")
        if api_key_input:
            GOOGLE_API_KEY = api_key_input
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                # Test configuration (optional but recommended)
                # list(genai.list_models())
                st.success("API Key Configured!")
                api_key_configured = True
            except Exception as e:
                st.error(f"Failed to configure API Key: {e}")
                GOOGLE_API_KEY = None # Reset if invalid
                api_key_configured = False
        else:
            st.warning("Please enter your Google API Key.")
            api_key_configured = False
    else:
        # Verify existing key from .env (optional)
        try:
             genai.configure(api_key=GOOGLE_API_KEY)
             # list(genai.list_models()) # Optional test
             st.success("API Key loaded successfully!")
             api_key_configured = True
        except Exception as e:
            st.error(f"API Key loaded from .env failed configuration: {e}")
            GOOGLE_API_KEY = None # Clear invalid key
            api_key_configured = False

    st.divider()

    # Document Upload and Processing
    st.header("Upload Documents (PDF)")
    uploaded_files = st.file_uploader(
        "Upload PDF documents for RAG",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        current_file_ids = {f.file_id for f in uploaded_files}
        # Process only if the set of uploaded files has changed
        if current_file_ids != st.session_state.processed_file_ids:
            if api_key_configured:
                with st.spinner("Processing PDFs..."):
                    st.session_state.vector_store = load_and_process_pdfs(uploaded_files)
                    if st.session_state.vector_store:
                         st.session_state.processed_file_ids = current_file_ids # Update processed IDs
                    else:
                         # Error handled in load_and_process_pdfs
                         st.session_state.processed_file_ids = set() # Reset on failure
            else:
                 st.warning("Please configure API key before uploading documents.")

    elif not uploaded_files and st.session_state.processed_file_ids:
         # Clear vector store if all files are removed
         st.session_state.vector_store = None
         st.session_state.processed_file_ids = set()
         st.info("Document context cleared.")

    # Display RAG status
    if st.session_state.vector_store is not None:
        st.success(f"RAG enabled using {len(st.session_state.processed_file_ids)} document(s).")
    else:
        st.info("RAG is disabled. Upload PDFs to enable.")


    st.divider()

    # Saved Prompts Management
    st.header("Saved Prompts")
    if not st.session_state.saved_prompts:
        st.caption("No prompts saved yet.")
    else:
        # Use a temporary list for safe iteration if deleting
        for name in list(st.session_state.saved_prompts.keys()):
            if name in st.session_state.saved_prompts: # Check if not deleted in this loop run
                prompt_text = st.session_state.saved_prompts[name]
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(name)
                with col2:
                    # Use name in key for uniqueness
                    if st.button("Run", key=f"run_{name}", help=f"Run: {prompt_text[:50]}...", use_container_width=True):
                        # Flag to run this prompt in the main chat logic
                        st.session_state.run_prompt_name = name
                        st.rerun() # Rerun to process the flag
                with col3:
                    # Use name in key for uniqueness
                    if st.button("Del", key=f"del_{name}", help="Delete prompt", use_container_width=True):
                        del st.session_state.saved_prompts[name]
                        st.success(f"Prompt '{name}' deleted.")
                        # Clear the run flag if the deleted prompt was flagged
                        if st.session_state.get("run_prompt_name") == name:
                            del st.session_state.run_prompt_name
                        st.rerun() # Rerun to update the list


# --- Chat Interface ---

# Check if we need to run a prompt triggered from the sidebar
if "run_prompt_name" in st.session_state and st.session_state.run_prompt_name:
    prompt_name_to_run = st.session_state.run_prompt_name
    if prompt_name_to_run in st.session_state.saved_prompts:
        # Prepare to run the saved prompt
        user_query = st.session_state.saved_prompts[prompt_name_to_run]
        action_message = f"Running saved prompt: '{prompt_name_to_run}'"
        st.session_state.chat_history.append(HumanMessage(content=action_message))
        # Display indication in chat temporarily (will be overwritten by rerun if needed)
        # with st.chat_message("Human"):
        #    st.markdown(f"_{action_message}_")
        # Optional: Also display the prompt text being run
        # st.markdown(f"> {user_query}")

        del st.session_state.run_prompt_name # Clear the flag AFTER deciding to run
        # Let the normal processing flow handle this user_query
    else:
        st.warning(f"Prompt '{prompt_name_to_run}' not found anymore.")
        del st.session_state.run_prompt_name # Clear the flag
        user_query = None # Prevent processing this cycle

# Get fresh user input only if not running a prompt from sidebar trigger
else:
    user_query = st.chat_input("Type message, 'save prompt as NAME: TEXT', or 'run prompt NAME'")


# Display existing chat messages first
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.write(message.content)


# Process new user input (or saved prompt triggered)
if user_query is not None and user_query.strip() != "":

    # Display the user query/action that's about to be processed
    # (This handles displaying the saved prompt text when run command is used)
    if not any(isinstance(msg, HumanMessage) and msg.content == user_query for msg in st.session_state.chat_history[-2:]): # Avoid duplicate display if already shown
         st.session_state.chat_history.append(HumanMessage(content=user_query))
         with st.chat_message("Human", avatar="👤"):
                st.markdown(user_query)

    # --- Command Handling ---
    processed_as_command = False # Flag to check if input was a command

    # 1. Save Command
    save_prefix = "save prompt as "
    if user_query.lower().startswith(save_prefix) and ":" in user_query:
        try:
            # Extract name and text carefully
            command_body = user_query[len(save_prefix):]
            name_part, prompt_text = command_body.split(":", 1)
            prompt_name = name_part.strip()
            prompt_text = prompt_text.strip()

            if prompt_name and prompt_text:
                st.session_state.saved_prompts[prompt_name] = prompt_text
                feedback_message = f"Prompt '{prompt_name}' saved successfully!"
                st.session_state.chat_history.append(AIMessage(content=feedback_message)) # Add AI feedback
                with st.chat_message("AI", avatar="🤖"):
                     st.markdown(feedback_message)
                processed_as_command = True
                st.rerun() # Rerun to update sidebar immediately
            else:
                 # Handle cases with empty name or text after splitting
                 st.error("Invalid format. Name and prompt text cannot be empty. Use 'save prompt as NAME: PROMPT TEXT'")
                 # Remove the invalid command message potentially added
                 if st.session_state.chat_history[-1].content == user_query:
                     st.session_state.chat_history.pop()
                 processed_as_command = True # Prevent further processing

        except ValueError:
             st.error("Invalid format. Use 'save prompt as NAME: PROMPT TEXT' (colon is required)")
             if st.session_state.chat_history[-1].content == user_query:
                 st.session_state.chat_history.pop()
             processed_as_command = True # Prevent further processing

    # 2. Run Command (from chat input)
    run_prefix = "run prompt "
    if not processed_as_command and user_query.lower().startswith(run_prefix):
        prompt_name_to_run = user_query[len(run_prefix):].strip()
        if prompt_name_to_run in st.session_state.saved_prompts:
            # Retrieve the saved prompt text and treat it as the new user_query
            saved_prompt_text = st.session_state.saved_prompts[prompt_name_to_run]
            feedback_message = f"Okay, running prompt '{prompt_name_to_run}'."
            st.session_state.chat_history.append(AIMessage(content=feedback_message)) # Add AI feedback
            with st.chat_message("AI", avatar="🤖"):
                 st.markdown(feedback_message)

            # Replace user_query with the actual prompt text to be processed by the LLM
            user_query = saved_prompt_text
            processed_as_command = False # Ensure it proceeds to LLM call using the *saved* text

            # Add the saved prompt text itself as a human message for clarity in history
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human", avatar="👤"):
                st.markdown(user_query)

        else:
            feedback_message = f"Sorry, I couldn't find a saved prompt named '{prompt_name_to_run}'."
            st.session_state.chat_history.append(AIMessage(content=feedback_message))
            with st.chat_message("AI", avatar="🤖"):
                st.markdown(feedback_message)
            processed_as_command = True # Stop processing, command failed

    # --- Normal Chat / Processed Run Command ---
    if not processed_as_command and user_query is not None:
        if not api_key_configured:
             st.error("Please configure your Google API Key in the sidebar first.")
             # Remove the human message that couldn't be processed
             if st.session_state.chat_history[-1].content == user_query:
                 st.session_state.chat_history.pop()
        else:
            # Get response (RAG or basic)
            with st.spinner("Thinking..."):
                # Retrieve vector store from session state safely
                vector_store = st.session_state.get("vector_store", None)
                response = get_response(user_query, st.session_state.chat_history, vector_store)

            # Add AI response to history and display
            st.session_state.chat_history.append(AIMessage(content=response))
            with st.chat_message("AI", avatar="🤖"):
                st.markdown(response)

# Add a check for rerun if needed, especially after command processing that doesn't send to LLM
# This seems handled correctly by the specific reruns after save/delete now.
