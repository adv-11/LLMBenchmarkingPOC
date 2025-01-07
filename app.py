from langchain_community.vectorstores import FAISS
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
# from langchain import hub
load_dotenv()
from langchain_huggingface import ChatHuggingFace
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from litellm import LiteLLM


callbacks = [StreamingStdOutCallbackHandler()]

HF_TOKEN = os.environ["HF_TOKEN"] 
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

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
        
        # Initialize LiteLLM with Hugging Face token
        llm = LiteLLM()

        # Load the uploaded document
        if uploaded_file is not None:

            if uploaded_file.type == "application/pdf":
                with open("temp_uploaded_file.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("temp_uploaded_file.pdf")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = TextLoader(uploaded_file)
            else:
                loader = TextLoader(uploaded_file)

            documents = loader.load()

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            # Initialize the vector store
            vector_store = FAISS.from_documents(docs, MistralAIEmbeddings(api_key=MISTRAL_API_KEY))
            create_retrieval_chain(vector_store, llm)
            # Create a retrieval chain
            retrieval_chain = create_retrieval_chain(vector_store, llm)

            # Evaluate each LLM
            results = {}
            for llm_name in selected_llms:
                if llm_name == "llama3.2":
                    model = ChatHuggingFace(model_name="llama3.2", hf_token=HF_TOKEN)
                elif llm_name == "qwen0.5":
                    model = ChatHuggingFace(model_name="qwen0.5", hf_token=HF_TOKEN)
                elif llm_name == "mistral":
                    model = ChatMistralAI(api_key=MISTRAL_API_KEY)
            for _, row in qa_dataframe.iterrows():
            # Generate responses for each question in the dataset
                for index, row in qa_dataframe.iterrows():
                    question = row["Question"]
                    expected_answer = row["Answer"]

                    # Generate response using the selected LLM
                    response = model.generate(question)

                    # Compare the response with the expected answer
                    results[question] = {
                    "expected": expected_answer,
                    "response": response,
                    "match": response.strip().lower() == expected_answer.strip().lower()
                    }

            # Display the results
            st.subheader("Evaluation Results")
            for question, result in results.items():
                st.write(f"Question: {question}")
                st.write(f"Expected Answer: {result['expected']}")
                st.write(f"LLM Response: {result['response']}")
                st.write(f"Match: {result['match']}")
                st.write("---")


if __name__ == "__main__":
    main()