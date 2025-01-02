import streamlit as st
import pandas as pd

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

    # page selection
    page = st.sidebar.selectbox("Select Page", ["Page 1: LLM Selection", "Page 2: Dataset Creation" , "Page 3: Dashboard"])

    if page == "Page 1: LLM Selection":
        st.header("Select LLM and Upload Document")
        
        llm_options = ["llama3.2", "qwen0.5", "mistral"]  
        selected_llms = st.multiselect("Select LLM(s):", llm_options)

        # Display the selected LLM(s)
        st.write("You selected:", selected_llms)
        
        # doc upload
        uploaded_file = st.file_uploader("Upload Document for RAG", type=["pdf", "docx", "txt"])
        
        if uploaded_file is not None:
            st.success("Document uploaded successfully!")
        
        

    elif page == "Page 2: Dataset Creation":
        st.header("Create Dataset of Questions and Answers")
        
        # Initialize an empty dataframe
        if 'qa_dataframe' not in st.session_state:
            st.session_state.qa_dataframe = pd.DataFrame(columns=["Question", "Answer"])
        
        # Create Q&A pairs
        question, answer = create_qa_pair()
        
        if question and answer:
            # Add the new Q&A pair to the dataframe
            new_row = pd.DataFrame({"Question": [question], "Answer": [answer]})
            st.session_state.qa_dataframe = pd.concat([st.session_state.qa_dataframe, new_row], ignore_index=True)
            st.success("Q&A Pair added successfully!")

        # Display the current dataset
        st.subheader("Current Dataset")
        st.dataframe(st.session_state.qa_dataframe)

        # Option to download the dataset as CSV
        if st.button("Download Dataset as CSV"):
            csv = st.session_state.qa_dataframe.to_csv(index=False)
            st.download_button("Download CSV", csv, "qa_dataset.csv", "text/csv")

    elif page == 'Page 3: Dashboard':
        st.header("Dashboard")
        
        st.button('Run Evaluation')
                # Here you would implement the RAG logic with the selected LLMs, uploaded document, and Q&A dataframe
                # For example, you could call a function to process the document and generate responses based on the Q&A pairs.
                
                # Placeholder for RAG processing logic
              
        

if __name__ == "__main__":
    main()