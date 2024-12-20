import cohere
import time
import logging
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import PyPDFLoader
# from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

class RAGPDFBot:
    """
    A RAG-based bot using Cohere for inference and LangChain for retrieval.
    """
    def __init__(self):
        """
        Initialize the bot, load API key, and set up the Cohere client.
        """
        load_dotenv()
        self.file_path = ""
        self.user_input = ""
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        if not self.cohere_api_key:
            raise ValueError("Cohere API Key is missing. Check your .env file.")

        self.client = cohere.Client(self.cohere_api_key)
        self.embedding_model = HuggingFaceEmbeddings()  # Specify the embedding model
        logging.info("Cohere client initialized successfully.")

    def build_vectordb(self, chunk_size, overlap, file_path):
        """
        Build the vector database from a PDF file.
        """
        logging.info("Building vector database...")
        self.file_path = file_path
        loader = PyPDFLoader(file_path=self.file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

        # Build the vectorstore index
        try:
            self.index = VectorstoreIndexCreator(
                embedding=self.embedding_model,  # Explicit embedding model
                text_splitter=text_splitter
            ).from_loaders([loader])
            logging.info("Vector database built successfully.")
        except Exception as e:
            logging.error(f"Error building vector database: {e}")
            raise

    def retrieval(self, user_input, top_k=1):
        """
        Retrieve relevant context from the vector database.
        """
        self.user_input = user_input
        logging.info("Performing similarity search for context retrieval...")

        try:
            result = self.index.vectorstore.similarity_search(self.user_input, k=top_k)
            context = "\n".join([document.page_content[:500] for document in result])  # Limit context length
            logging.info("Context retrieved successfully.")

            # Define the prompt template
            template = """
            Context: {context}

            Instructions for the LLM:
            1. Based on the provided context, answer the question in a precise and accurate manner.
            2. If the question is directly related to the context, provide a clear and concise response.
            3. Ensure the answer does not exceed 3 lines.
            4. If the question is unrelated to the context, respond with "I don't know."
            5. Avoid any irrelevant or nonsensical information in the answer.

            Question:
            {question}
            """
            self.prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            raise

    def inference(self):
        """
        Generate a response using Cohere's API with retry logic for rate limiting.
        """
        logging.info("Generating response from Cohere API...")
        prompt = self.prompt.format(question=self.user_input)
        token_count = len(prompt.split())  # Approximate token count
        logging.info(f"Request token count: {token_count}")

        while True:  # Retry logic for rate limiting
            try:
                response = self.client.generate(
                    model="command-xlarge",
                    prompt=prompt,
                    max_tokens=100,  # Reduced token limit
                    temperature=0.5,
                    stop_sequences=["I don't know."]
                )
                logging.info("Response generated successfully.")
                return response.generations[0].text.strip()
            except cohere.TooManyRequestsError:
                logging.warning("Rate limit exceeded. Retrying after 30 seconds...")
                time.sleep(30)  # Wait and retry
            except Exception as e:
                logging.error(f"Error during inference: {e}")
                raise