import asyncio
import random
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from ragbase.chain import ask_question, create_chain
from ragbase.config import Config
from ragbase.ingestor import Ingestor
from ragbase.model import create_llm
from ragbase.retriever import create_retriever
from ragbase.uploader import upload_files

from qdrant_client import QdrantClient

load_dotenv()

LOADING_MESSAGES = [
    "Searching through our knowledge base...",
    "Analyzing relevant documents...",
    "Connecting information from multiple sources...",
    "Retrieving context-specific data...",
    "Synthesizing insights from company documents...",
    "Extracting key information from our database...",
    "Combining relevant data points...",
    "Processing organizational knowledge...",
    "Accessing secure company archives...",
    "Integrating information across departments...",
]


@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    # global files_ingested, last_trained
    
    # Ensure the directory exists
    os.makedirs(Config.Path.DATABASE_DIR, exist_ok=True)
    
    last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    files_ingested = len(files)
    
    st.session_state.last_trained = last_trained
    st.session_state.files_ingested = files_ingested
    
    with open(Config.Path.DATABASE_DIR / "training_info.txt", "w") as f:
        f.write(f"Last Trained: {last_trained}\n")
        f.write(f"Files Ingested: {files_ingested}\n")
    
    file_paths = upload_files(files)
    vector_store = Ingestor().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)


async def ask_chain(question: str, chain):
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.png")
    )
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if type(event) is str:
                full_response += event
                message_placeholder.markdown(full_response)
            if type(event) is list:
                documents.extend(event)
        if Config.Retriever.SHOW_SOURCES:
            for i, doc in enumerate(documents):
                with st.expander(f"Source #{i+1}"):
                    st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def show_upload_documents():
    holder = st.empty()
    with holder.container():
        st.title("Document Ingestion Portal 📁")
        st.header("VectorDB for VS-GPT 🆚🤖")
        st.subheader("Upload your documents to create the knowledge base")
        uploaded_files = st.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.warning("Please upload PDF documents to continue!")
        st.stop()

    with st.spinner("Analyzing your document(s)..."):
        holder.empty()
        chain = build_qa_chain(uploaded_files)

    return chain


def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.png"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.png"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])


def show_chat_input(chain):
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png"),
        ):
            st.markdown(prompt)
        asyncio.run(ask_chain(prompt, chain))

# setting up the page header here.
hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

st.set_page_config(page_title="VectoScalar-GPT", page_icon="🆚")

st.html(
    """
<style>
    .st-emotion-cache-p4micv {
        width: 2.75rem;
        height: 2.75rem;
    }
</style>
"""
)

# removing all the default streamlit configs here
st.markdown(hide_st_style, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm VS-Bot 🤖, How can I help you today?",
        }
    ]

if Config.CONVERSATION_MESSAGES_LIMIT > 0 and Config.CONVERSATION_MESSAGES_LIMIT <= len(
    st.session_state.messages
):
    st.warning(
        "You have reached the conversation limit. Refresh the page to start a new conversation."
    )
    st.stop()

# ingestion_UI.
chain = show_upload_documents()

# Chat Interface UI.
st.markdown("""
                # :rainbow[VectoScalar-GPT] 🆚🤖
            """)

col1, col2, col3 = st.columns([1,1.3,1]) # cols, rows

# Read from the text file to display on UI
# with open(Config.Path.DATABASE_DIR / "training_info.txt", "r") as f:
#     lines = f.readlines()
#     last_trained = lines[0].strip().split(": ")[1]
#     files_ingested = lines[1].strip().split(": ")[1]
    
with col1:
    st.success(f"Version : {Config.VERSION}")

with col2:
    st.info(f"Last Trained : {st.session_state.last_trained}")

with col3:
    st.warning(f"Files Ingested : {st.session_state.files_ingested}")

show_message_history()
show_chat_input(chain)
