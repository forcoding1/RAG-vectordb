import google.generativeai as genai
import json
import os
import sys
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
import textwrap
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader   
import tempfile

# Load credentials
with open('creds.json', 'r') as file:
    data = json.load(file)

os.environ["GEMINI_API_KEY"] = data["GEMINI_API_KEY"]

# Define model generator
def model_generator(name):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name=name,
        generation_config=generation_config,
    )
    return model

# Define function to create vector store
def get_vector(file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getbuffer())
        temp_file_path = temp_file.name

    docs = []
    loaders = [PyPDFLoader(temp_file_path)]
    for loader in loaders:
        docs.extend(loader.load())

    # Remove temporary file after loading content
    os.remove(temp_file_path)
    
    # Split text and create vector index
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    embedding_function = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_index = Chroma.from_documents(docs, embedding_function).as_retriever(search_kwargs={"k": 6})
    return vector_index

# Define function to create QA chain
def gen_prompt_chain(model_name, vector_index):
    template = """You are a helpful AI assistant who provides clear, thorough answers with complete sentences. 
        Use a friendly, conversational tone while being comprehensive.
        Be very precise while answering.
        Focus only on relevant information to answer my questions.
        {context}
        Question: {question}
        Helpful Answer:"""
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=data["GEMINI_API_KEY"], temperature=0.2)
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# Streamlit interface
st.title("Chat with PDF")

uploaded_file = st.file_uploader("Upload the pdf", type=["pdf"])

if uploaded_file is not None:
    vector_index = get_vector(uploaded_file)
    qa_chain = gen_prompt_chain("gemini-1.5-flash", vector_index)
    
    # Question input and answer display
    question = st.text_input("Ask a question about the document:")
    
    if question:
        response = qa_chain.invoke({"query": question})
        answer = response["result"]
        source_docs = response["source_documents"]

        # Display the answer
        st.write("Answer:", answer)

        # Optionally, display source documents used for the answer
        if source_docs:
            st.write("Sources:")
            for doc in source_docs:
                st.write(f"- {doc.page_content[:200]}...")  # Preview the content of source documents
