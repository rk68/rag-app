import os
import logging
import pandas as pd
import argparse
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain_groq import ChatGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
parse_api_key = os.getenv('PARSE_API_KEY')
azure_api_key = os.getenv('AZURE_API_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')
azure_deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')
azure_api_version = os.getenv('AZURE_API_VERSION')

# Global variables for lazy loading
llm = None
pinecone_index = None
query_engine = None

def log_and_exit(message):
    logging.error(message)
    raise SystemExit(message)

def initialize_apis():
    global llm, pinecone_index
    try:
        if llm is None:
            llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
        if pinecone_index is None:
            pinecone_client = Pinecone(pinecone_api_key)
            pinecone_index = pinecone_client.Index("demo")
        logging.info("Initialized Gemini and Pinecone.")
    except Exception as e:
        log_and_exit(f"Error initializing APIs: {e}")

def load_pdf_data():
    PDF_FILE_PATH = "policy.pdf"
    try:
        parser = LlamaParse(api_key=parse_api_key, result_type="markdown")
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=[PDF_FILE_PATH], file_extractor=file_extractor).load_data()
        logging.info(f"Loaded {len(documents)} documents from PDF.")
        return documents
    except Exception as e:
        log_and_exit(f"Error loading PDF file: {e}")

def create_index(documents):
    global llm, pinecone_index
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logging.info("Created index from documents.")
        return index
    except Exception as e:
        log_and_exit(f"Error creating index: {e}")

def setup_query_engine(index):
    try:
        retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
        query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
        return query_engine
    except Exception as e:
        log_and_exit(f"Error setting up query engine: {e}")

def process_csv(input_csv_path, output_csv_path, query_engine):
    df = pd.read_csv(input_csv_path)
    responses = []

    for _, row in df.iterrows():
        question = row['question']
        try:
            response = query_engine.query(question)
            responses.append(response)
        except Exception as e:
            responses.append("Error: " + str(e))

    df['response'] = responses

    vectorizer = TfidfVectorizer().fit_transform(df[['ideal_answer', 'response']].values.flatten())
    vectors = vectorizer.toarray()
    cosine_similarities = []

    for i in range(0, len(vectors), 2):
        cos_sim = cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0]
        cosine_similarities.append(cos_sim)

    df['cosine_similarity'] = cosine_similarities
    df.to_csv(output_csv_path, index=False)
    print(f"Output saved to {output_csv_path}")

def run_streamlit_app():
    import streamlit as st

    global query_engine

    # Lazy load the query engine
    if query_engine is None:
        initialize_apis()
        documents = load_pdf_data()
        index = create_index(documents)
        query_engine = setup_query_engine(index)

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("RAG Chat Application")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat['user'])
        with st.chat_message("bot"):
            st.markdown(chat['response'])

    # User input for new question
    if question := st.chat_input("Enter your question"):
        response = query_engine.query(question)
        st.session_state.chat_history.append({'user': question, 'response': response})
        st.experimental_rerun()  # Refresh the UI to display the new chat entry

def run_terminal_app():
    global query_engine

    # Lazy load the query engine
    if query_engine is None:
        initialize_apis()
        documents = load_pdf_data()
        index = create_index(documents)
        query_engine = setup_query_engine(index)

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = query_engine.query(question)
        print("Response:", response)

def main():
    # Check if running with Streamlit
    try:
        import streamlit as st
        run_streamlit_app()
    except ImportError:
        run_terminal_app()

if __name__ == "__main__":
    main()
