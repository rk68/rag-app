import os
import logging
import pandas as pd
import argparse
import streamlit as st
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import RouterRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain_groq import ChatGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from llama_index.core.query_engine import FLAREInstructQueryEngine, MultiStepQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

def initialize_apis(api, model):
    global llm, pinecone_index
    try:
        if llm is None:
            if api == 'groq':
                if model == 'mixtral-8x7b':
                    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
                elif model == 'llama3-8b':
                    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
                elif model == "llama3-70b":
                    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
                elif model == "gemma-7b":
                    llm = ChatGroq(model="gemma-7b-it", temperature=0)

            elif api == 'azure':
                if model == 'gpt35':
                    llm = AzureOpenAI(
                        deployment_name="gpt35",
                        temperature=0,
                        api_key=azure_api_key,
                        azure_endpoint=azure_endpoint,
                        api_version=azure_api_version
                    )
                    
        if pinecone_index is None:
            index_name = "demo"
            pinecone_client = Pinecone(pinecone_api_key)
            pinecone_index = pinecone_client.Index(index_name)
        logging.info("Initialized LLM and Pinecone.")
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

def create_index(documents, embedding_model_type="HF", embedding_model="BAAI/bge-large-en-v1.5", retriever_method="BM25"):
    global llm, pinecone_index
    try:
        if embedding_model_type == "HF":
            embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        elif embedding_model_type == "OAI":
            # embed_model = OpenAIEmbedding() implement oai EMBEDDING
            pass

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

        if retriever_method == "BM25" or retriever_method == "BM25+Vector":
            splitter = SentenceSplitter(chunk_size=512)
            nodes = splitter.get_nodes_from_documents(documents)
            storage_context = StorageContext.from_defaults()
            return None, nodes  # Return None for index when using BM25
        else:
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            logging.info("Created index from documents.")
            return index, None  # Return None for nodes when not using BM25
    except Exception as e:
        log_and_exit(f"Error creating index: {e}")

def setup_query_engine(index, response_mode, nodes=None, query_engine_method=None, retriever_method=None):
    global llm
    try:
        logging.info(f"Setting up query engine with retriever_method: {retriever_method} and query_engine_method: {query_engine_method}")

        if retriever_method == 'BM25':
            retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
        elif retriever_method == "BM25+Vector":
            vector_retriever = VectorIndexRetriever(index=index)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

            retriever_tools = [
                RetrieverTool.from_defaults(
                    retriever=vector_retriever,
                    description='Useful in most cases',
                ),
                RetrieverTool.from_defaults(
                    retriever=bm25_retriever,
                    description='Useful if searching about specific information',
                ),                 
            ]
            retriever = RouterRetriever.from_defaults(
                retriever_tools=retriever_tools,
                llm=llm,
                select_multi=True
            )
        else:
            retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

        response_synthesizer = get_response_synthesizer(response_mode=response_mode)
        index_query_engine = index.as_query_engine(similarity_top_k=2) if index else None

        if query_engine_method == "FLARE":
            query_engine = FLAREInstructQueryEngine(
                query_engine=index_query_engine,
                max_iterations=7,
                verbose=True
            )
        elif query_engine_method == "MS":
            step_decompose_transform = StepDecomposeQueryTransform(llm=llm, verbose=True)
            index_summary = "Used to answer questions about the regulation"
            query_engine = MultiStepQueryEngine(
                query_engine=index_query_engine,
                query_transform=step_decompose_transform,
                index_summary=index_summary
            )
        else:
            query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
        return query_engine
    except Exception as e:
        log_and_exit(f"Error setting up query engine: {e}")

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
            query_engine = setup_query_engine(index, response_mode=mode, retriever_method='Default')
            response = query_engine.query(question)
            answer_column = f"{mode}_answer"
            df.at[idx, answer_column] = response.response

    df.to_csv(file_path, index=False)
    logging.info(f"Processed questions and updated the CSV file with answers: {file_path}")



def run_streamlit_app(api, model):
    global query_engine

    if query_engine is None:
        initialize_apis(api, model)
        documents = load_pdf_data()
        index, nodes = create_index(documents)
        query_engine = setup_query_engine(index, response_mode="tree_summarize", nodes=nodes, retriever_method='BM25')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("RAG Chat Application")

    query_method = st.selectbox("Select Query Engine Method", ["Default", "FLARE", "MS"])
    retriever_method = st.selectbox("Select Retriever Method", ["Default", "BM25", "BM25+Vector"])
    selected_api = st.selectbox("Select API", ["azure", "ollama", "groq"])
    selected_model = st.selectbox("Select Model", ["llama3-8b", "llama3-70b", "mixtral-8x7b", "gemma-7b", "gpt35"])
    embedding_model_type = st.selectbox("Select Embedding Model Type", ["HF", "OAI"])
    embedding_model = st.selectbox("Select Embedding Model", ["BAAI/bge-large-en-v1.5", "other_model"])  # Add your embedding models here

    if query_method == "FLARE":
        if retriever_method in ["BM25", "BM25+Vector"]:
            query_engine = setup_query_engine(None, response_mode="tree_summarize", nodes=nodes, query_engine_method="FLARE", retriever_method=retriever_method)
        else:
            query_engine = setup_query_engine(index, response_mode="tree_summarize", query_engine_method="FLARE", retriever_method=retriever_method)
    elif query_method == "MS":
        if retriever_method in ["BM25", "BM25+Vector"]:
            query_engine = setup_query_engine(None, response_mode="tree_summarize", nodes=nodes, query_engine_method="MS", retriever_method=retriever_method)
        else:
            query_engine = setup_query_engine(index, response_mode="tree_summarize", query_engine_method="MS", retriever_method=retriever_method)
    else:
        if retriever_method in ["BM25", "BM25+Vector"]:
            query_engine = setup_query_engine(None, response_mode="tree_summarize", nodes=nodes, retriever_method=retriever_method)
        else:
            query_engine = setup_query_engine(index, response_mode="tree_summarize", retriever_method=retriever_method)

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat['user'])
        with st.chat_message("bot"):
            st.markdown(chat['response'])

    if question := st.chat_input("Enter your question"):
        response = query_engine.query(question)
        st.session_state.chat_history.append({'user': question, 'response': response.response})
        st.rerun()

def run_terminal_app(api, model, query_method, retriever_method):
    global query_engine

    if query_engine is None:
        initialize_apis(api, model)
        documents = load_pdf_data()
        if retriever_method == "BM25" or retriever_method == "BM25+Vector":
            _, nodes = create_index(documents, retriever_method=retriever_method)
            query_engine = setup_query_engine(None, response_mode="compact_accumulate", nodes=nodes, query_engine_method=query_method, retriever_method=retriever_method)
        else:
            index, _ = create_index(documents, retriever_method=retriever_method)
            query_engine = setup_query_engine(index, response_mode="compact_accumulate", query_engine_method=query_method, retriever_method=retriever_method)

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = query_engine.query(question)
        retrieved_nodes = response.source_nodes
        chunks = [node.text for node in retrieved_nodes]
        print("Contexts:")
        for chunk in chunks:
            print(chunk)
        if retriever_method == "BM25" or retriever_method == "BM25+Vector":
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
    parser.add_argument('--retriever_method', type=str, choices=['Default', 'BM25', 'BM25+Vector'], required=False, default='Default', help='Retriever Method to use')
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
    import sys
    #if 'streamlit' in sys.argv[0]:
    run_streamlit_app('azure', 'gpt35')
    #else:
    #    main()
