import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title="HR RAG Chatbot", page_icon="🤖")
st.title("🤖 HR RAG Chatbot")
st.write("Ask HR-related questions from company documents"
google_api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

if not google_api_key:
    st.warning("Please enter Gemini API Key")
    st.stop()
@st.cache_resource
def load_vectorstore():

    pdf_loader = DirectoryLoader(
        path="data",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    docx_loader = DirectoryLoader(
        path="data",
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
    )

    pdf_documents = pdf_loader.load()
    docx_documents = docx_loader.load()

    documents = pdf_documents + docx_documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )

    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_api_key
)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an HR assistant. Answer only from the provided context."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

store = {}


def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)
user_question = st.text_input("Ask a question")

if st.button("Submit"):

    if user_question:

        response = rag_with_memory.invoke(
            {"question": user_question},
            config={"configurable": {"session_id": "user1"}}
        )

        st.success(response)
