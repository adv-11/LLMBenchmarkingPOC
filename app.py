from json import load
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain import hub
load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"] 
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

print (HF_TOKEN , '\n', MISTRAL_API_KEY)


# func to create a new question-answer pair
def create_qa_pair():
    question = st.text_input("Enter Question:")
    answer = st.text_input("Enter Answer:")
    
    if st.button("Add Q&A Pair"):
        if question and answer:
            return question, answer
        else:
            st.warning("Both fields are required.")
    return None, None

# main 
def main():
    st.title("LLM Evaluation Platform")

    # LLM Selection and Document Upload
    st.subheader("Step 1: Select LLM and Upload Document ⬆️")
    
    llm_options = ["llama3.2", "qwen0.5", "mistral"]  
    selected_llms = st.multiselect("Select LLM(s):", llm_options)

    # Display the selected LLM(s)
    st.write("You selected:", selected_llms)
    
    # Document upload
    uploaded_file = st.file_uploader("Upload Document for RAG", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        st.success("Document uploaded successfully!")

    # Dataset Creation
    st.subheader("Step 2: Create Dataset of Questions and Answers 📚")
    
    # Initialize an empty dataframe
    qa_dataframe = pd.DataFrame(columns=["Question", "Answer"])
    
    # Create Q&A pairs
    question, answer = create_qa_pair()
    
    if question and answer:
        # Add the new Q&A pair to the dataframe
        new_row = pd.DataFrame({"Question": [question], "Answer": [answer]})
        qa_dataframe = pd.concat([qa_dataframe, new_row], ignore_index=True)
        st.success("Q&A Pair added successfully!")

    # Display the current dataset
    st.subheader("Current Dataset")
    st.dataframe(qa_dataframe)

    # Option to download the dataset as CSV
    if st.button("Download Dataset as CSV"):
        csv = qa_dataframe.to_csv(index=False)
        st.download_button("Download CSV", csv, "qa_dataset.csv", "text/csv")

    # Dashboard
    st.subheader("Step 3: Run Evaluation 🚀")
    
    if st.button('Run'):
        st.write("Running Evaluation...")
        # Here you would implement the RAG logic with the selected LLMs, uploaded document, and Q&A dataframe
        # For example, you could call a function to process the document and generate responses based on the Q&A pairs.
        
        # Placeholder for RAG processing logic
              

if __name__ == "__main__":
    main()