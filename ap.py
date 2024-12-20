import streamlit as st
from model import RAGPDFBot

# Initialize the bot
bot = RAGPDFBot()

def initialize_model(file_path):
    # Build vector database from the provided PDF file
    chunk_size = 500
    overlap = 50
    bot.build_vectordb(chunk_size=chunk_size, overlap=overlap, file_path=file_path)
    st.write("Vector DB built successfully!")

def retrieve(input):
    # Retrieve relevant context and generate the response
    bot.retrieval(user_input=input)
    return bot.inference()

def main():
    st.title("RAG with Cohere API")

    # Upload PDF document to build vector database
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        initialize_model("uploaded_document.pdf")

    # Text input for the user query
    query = st.text_input("Ask a question:")

    if query:
        # Retrieve the context and generate a response using Cohere
        response = retrieve(query)
        st.subheader("Generated Response")
        st.write(response)

if __name__ == "__main__":
    main()