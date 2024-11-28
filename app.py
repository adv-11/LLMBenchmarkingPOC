import streamlit as st
import pandas as pd

# Function to create a new question-answer pair
def create_qa_pair():
    question = st.text_input("Enter Question:")
    answer = st.text_input("Enter Answer:")
    
    if st.button("Add Q&A Pair"):
        if question and answer:
            return question, answer
        else:
            st.warning("Both fields are required.")
    return None, None

# Main application
def main():
    st.title("LLM Evaluation Platform")

    # Page selection
    page = st.sidebar.selectbox("Select Page", ["Page 1: LLM Selection", "Page 2: Dataset Creation"])

    if page == "Page 1: LLM Selection":
        st.header("Select LLM and Upload Document")
        
        # Dropdown for LLM selection
        llm_options = ["LLM 1", "LLM 2", "LLM 3"]  # Replace with actual LLM names
        selected_llm = st.selectbox("Select an LLM:", llm_options)
        
        # Document upload
        uploaded_file = st.file_uploader("Upload Document for RAG", type=["pdf", "docx", "txt"])
        
        if uploaded_file is not None:
            st.success("Document uploaded successfully!")
        
        # Run button
        if st.button("Run"):
            if uploaded_file is not None:
                st.success("Running evaluation with selected LLM...")
                # Here you can call the backend logic
            else:
                st.warning("Please upload a document before running.")

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

if __name__ == "__main__":
    main()