import os
import logging
import pandas as pd
import argparse
import streamlit as st
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import (
    StorageContext, VectorStoreIndex, SimpleDirectoryReader, 
    get_response_synthesizer, Settings
)
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
from llama_index.core.retrievers import (
    VectorIndexRetriever, RouterRetriever
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.query_engine import (
    RetrieverQueryEngine, FLAREInstructQueryEngine, MultiStepQueryEngine
)
from llama_index.core.indices.query.query_transform import (
    HyDEQueryTransform, StepDecomposeQueryTransform
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fetch API keys from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PARSE_API_KEY = os.getenv('PARSE_API_KEY')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')

# Global variables for lazy loading
llm = None
pinecone_index = None

def log_and_exit(message):
    logging.error(message)
    raise SystemExit(message)

def initialize_apis(api, model):
    global llm, pinecone_index
    try:
        if llm is None:
            llm = initialize_llm(api, model)
        if pinecone_index is None:
            pinecone_client = Pinecone(PINECONE_API_KEY)
            pinecone_index = pinecone_client.Index("demo")
        logging.info("Initialized LLM and Pinecone.")
    except Exception as e:
        log_and_exit(f"Error initializing APIs: {e}")

def initialize_llm(api, model):
    if api == 'groq':
        model_mappings = {
            'mixtral-8x7b': "mixtral-8x7b-32768",
            'llama3-8b': "llama3-8b-8192",
            'llama3-70b': "llama3-70b-8192",
            'gemma-7b': "gemma-7b-it"
        }
        return Groq(model=model_mappings[model], api_key=os.getenv("GROQ_API_KEY"))
    elif api == 'azure':
        if model == 'gpt35':
            return AzureOpenAI(
                deployment_name=AZURE_DEPLOYMENT_NAME,
                temperature=0,
                api_key=AZURE_API_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION
            )

def load_pdf_data():
    PDF_FILE_PATH = "policy.pdf"
    try:
        parser = LlamaParse(api_key=PARSE_API_KEY, result_type="markdown")
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=[PDF_FILE_PATH], file_extractor=file_extractor).load_data()
        logging.info(f"Loaded {len(documents)} documents from PDF.")
        return documents
    except Exception as e:
        log_and_exit(f"Error loading PDF file: {e}")

def create_index(documents, embedding_model_type="HF", embedding_model="BAAI/bge-large-en-v1.5", retriever_method="BM25"):
    global llm, pinecone_index
    try:
        embed_model = select_embedding_model(embedding_model_type, embedding_model)

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

        if retriever_method in ["BM25", "BM25+Vector"]:
            nodes = create_bm25_nodes(documents)
            logging.info("Created BM25 nodes from documents.")
            if retriever_method == "BM25+Vector":
                vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
                logging.info("Created index for BM25+Vector from documents.")
                return index, nodes
            return None, nodes
        else:
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            logging.info("Created index from documents.")
            return index, None
    except Exception as e:
        log_and_exit(f"Error creating index: {e}")

def select_embedding_model(embedding_model_type, embedding_model):
    if embedding_model_type == "HF":
        return HuggingFaceEmbedding(model_name=embedding_model)
    elif embedding_model_type == "OAI":
        return OpenAIEmbedding()  # Implement OAI Embedding if needed

def create_bm25_nodes(documents):
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def setup_query_engine(index, response_mode, nodes=None, query_engine_method=None, retriever_method=None):
    global llm
    try:
        logging.info(f"Setting up query engine with retriever_method: {retriever_method} and query_engine_method: {query_engine_method}")
        retriever = select_retriever(index, nodes, retriever_method)
        
        if retriever is None:
            log_and_exit("Failed to create retriever. Index or nodes might be None.")

        response_synthesizer = get_response_synthesizer(response_mode=response_mode)
        index_query_engine = index.as_query_engine(similarity_top_k=2) if index else None

        if query_engine_method == "FLARE":
            query_engine = FLAREInstructQueryEngine(
                query_engine=index_query_engine,
                max_iterations=4,
                verbose=False
            )
        elif query_engine_method == "MS":
            query_engine = MultiStepQueryEngine(
                query_engine=index_query_engine,
                query_transform=StepDecomposeQueryTransform(llm=llm, verbose=False),
                index_summary="Used to answer questions about the regulation"
            )
        else:
            query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
        
        if query_engine is None:
            log_and_exit("Failed to create query engine.")
        
        return query_engine
    except Exception as e:
        logging.error(f"Error setting up query engine: {e}")
        traceback.print_exc()
        log_and_exit(f"Error setting up query engine: {e}")

def select_retriever(index, nodes, retriever_method):
    logging.info(f"Selecting retriever with method: {retriever_method}")
    if nodes is not None:
        logging.info(f"Available document IDs: {list(range(len(nodes)))}")
    else:
        logging.warning("Nodes are None")
    
    if retriever_method == 'BM25':
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
    elif retriever_method == "BM25+Vector":
        if index is None:
            log_and_exit("Index must be initialized when using BM25+Vector retriever method.")
        return RouterRetriever.from_defaults(
            retriever_tools=[
                RetrieverTool.from_defaults(
                    retriever=VectorIndexRetriever(index=index),
                    description='Useful in most cases',
                ),
                RetrieverTool.from_defaults(
                    retriever=BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2),
                    description='Useful if searching about specific information',
                ),
            ],
            llm=llm,
            select_multi=True
        )
    elif retriever_method == "Vector Search":
        if index is None:
            log_and_exit("Index must be initialized when using Vector Search retriever method.")
        return VectorIndexRetriever(index=index, similarity_top_k=2)
    else:
        log_and_exit(f"Unsupported retriever method: {retriever_method}")

def retrieve_and_update_contexts(api, model, file_path):
    global query_engine

    if query_engine is None:
        initialize_apis(api, model)
        documents = load_pdf_data()
        _, nodes = create_index(documents, retriever_method='BM25')
        query_engine = setup_query_engine(None, response_mode="compact_accumulate", nodes=nodes, retriever_method='BM25')

    df = pd.read_csv(file_path)

    for idx, row in df.iterrows():
        question = row['question']
        response = query_engine.query(question)
        retrieved_nodes = response.source_nodes
        chunks = [node.text for node in retrieved_nodes]
        logging.info(f"Context response for question {idx}: {response}")
        df.at[idx, 'contexts'] = " ".join(chunks)

    df.to_csv(file_path, index=False)
    logging.info(f"Processed questions and updated the CSV file: {file_path}")

def retrieve_answers_for_modes(api, model, file_path):
    global query_engine

    df = pd.read_csv(file_path)
    initialize_apis(api, model)
    documents = load_pdf_data()
    index, _ = create_index(documents)

    response_modes = ["refine", "compact", "tree_summarize", "simple_summarize"]

    for idx, row in df.iterrows():
        question = row['question']
        for mode in response_modes:
            query_engine = setup_query_engine(index, response_mode=mode, retriever_method='Vector Search')
            response = query_engine.query(question)
            answer_column = f"{mode}_answer"
            df.at[idx, answer_column] = response.response

    df.to_csv(file_path, index=False)
    logging.info(f"Processed questions and updated the CSV file with answers: {file_path}")

def run_streamlit_app(api, model):
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None

    st.title("RAG Chat Application")

    selected_api = st.selectbox("Select API", ["azure", "ollama", "groq"])
    selected_model = st.selectbox("Select Model", ["llama3-8b", "llama3-70b", "mixtral-8x7b", "gemma-7b", "gpt35"])
    embedding_model_type = st.selectbox("Select Embedding Model Type", ["HF", "OAI"])
    embedding_model = st.selectbox("Select Embedding Model", ["BAAI/bge-large-en-v1.5", "other_model"])
    retriever_method = st.selectbox("Select Retriever Method", ["Vector Search", "BM25", "BM25+Vector"])
    #query_method = st.selectbox("Select Query Engine Method", ["Default", "FLARE", "MS"])
    query_method = None # Change this when its ready
    if st.button("Initialize"):
        initialize_apis(selected_api, selected_model)
        documents = load_pdf_data()
        index, nodes = create_index(documents, embedding_model_type=embedding_model_type, embedding_model=embedding_model, retriever_method=retriever_method)
        st.session_state.query_engine = setup_query_engine(index, response_mode="tree_summarize", nodes=nodes, query_engine_method=query_method, retriever_method=retriever_method)
        st.success("Initialization complete.")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat['user'])
        with st.chat_message("bot"):
            st.markdown(chat['response'])

    if question := st.chat_input("Enter your question"):
        if st.session_state.query_engine:
            with st.spinner('Generating response...'):
                response = st.session_state.query_engine.query(question)
            st.session_state.chat_history.append({'user': question, 'response': response.response})
            st.rerun()
        else:
            st.error("Query engine is not initialized. Please initialize it first.")

def configure_query_engine(index, nodes, query_method, retriever_method):
    global query_engine

    if query_method == "FLARE":
        query_engine = setup_query_engine(None if retriever_method in ["BM25", "BM25+Vector"] else index, response_mode="tree_summarize", nodes=nodes, query_engine_method="FLARE", retriever_method=retriever_method)
    elif query_method == "MS":
        query_engine = setup_query_engine(None if retriever_method in ["BM25", "BM25+Vector"] else index, response_mode="tree_summarize", nodes=nodes, query_engine_method="MS", retriever_method=retriever_method)
    else:
        query_engine = setup_query_engine(None if retriever_method in ["BM25", "BM25+Vector"] else index, response_mode="tree_summarize", nodes=nodes, retriever_method=retriever_method)

def run_terminal_app(api, model, query_method, retriever_method):
    global query_engine

    if query_engine is None:
        initialize_apis(api, model)
        documents = load_pdf_data()
        if retriever_method in ["BM25", "BM25+Vector"]:
            index, nodes = create_index(documents, retriever_method=retriever_method)
            query_engine = setup_query_engine(index, response_mode="compact_accumulate", nodes=nodes, query_engine_method=query_method, retriever_method=retriever_method)
        else:
            index, _ = create_index(documents, retriever_method=retriever_method)
            query_engine = setup_query_engine(index, response_mode="compact_accumulate", query_engine_method=query_method, retriever_method=retriever_method)

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = query_engine.query(question)
        print_contexts_and_answer(response, query_method, retriever_method, nodes, index)

def print_contexts_and_answer(response, query_method, retriever_method, nodes, index):
    retrieved_nodes = response.source_nodes
    chunks = [node.text for node in retrieved_nodes]
    print("Contexts:")
    for chunk in chunks:
        print(chunk)
    if retriever_method in ["BM25", "BM25+Vector"]:
        query_engine = setup_query_engine(None, response_mode="tree_summarize", nodes=nodes, query_engine_method=query_method, retriever_method=retriever_method)
    else:
        query_engine = setup_query_engine(index, response_mode="tree_summarize", query_engine_method=query_method, retriever_method=retriever_method)
    final_response = query_engine.query(question)
    print("Final Answer:", final_response.response)

def main():
    parser = argparse.ArgumentParser(description="Run the RAG app.")
    parser.add_argument('--mode', type=str, choices=['terminal', 'benchmark', 'retrieve_contexts', 'retrieve_answers'], required=False, default='terminal', help="Mode to run the application in: 'terminal', 'benchmark', 'retrieve_contexts', 'retrieve_answers'")
    parser.add_argument('--api', type=str, choices=['azure', 'ollama', 'groq'], required=False, default='azure', help='Which api to use to call LLMs: ollama, groq or azure (openai)')
    parser.add_argument('--model', type=str, choices=['llama3-8b', 'llama3-70b', 'mixtral-8x7b', 'gemma-7b',  'gpt35'], default='gpt35')
    parser.add_argument('--embedding_model_type', type=str, choices=['HF'], required=False, default="HF")
    parser.add_argument('--embedding_model', type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument('--csv_file', type=str, required=False, help='Path to the CSV file containing questions')
    parser.add_argument('--query_method', type=str, choices=['Default', 'FLARE', 'MS'], required=False, default='Default', help='Query Engine Method to use')
    parser.add_argument('--retriever_method', type=str, choices=['Vector Search', 'BM25', 'BM25+Vector'], required=False, default='Vector Search', help='Retriever Method to use')
    args = parser.parse_args()

    if args.mode == 'terminal':
        run_terminal_app(args.api, args.model, args.query_method, args.retriever_method)
    elif args.mode == 'retrieve_contexts':
        if args.csv_file:
            retrieve_and_update_contexts(args.api, args.model, args.csv_file)
        else:
            log_and_exit("CSV file path is required for retrieve_contexts mode.")
    elif args.mode == 'retrieve_answers':
        if args.csv_file:
            retrieve_answers_for_modes(args.api, args.model, args.csv_file)
        else:
            log_and_exit("CSV file path is required for retrieve_answers mode.")
    elif args.mode == 'benchmark':
        pass

if __name__ == "__main__":
    use_streamlit = True
    if use_streamlit:
        run_streamlit_app('azure', 'gpt35')
    else:
        main()
