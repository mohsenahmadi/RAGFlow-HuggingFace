import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
import os
import tempfile
import hashlib
from datetime import datetime
import json
import mimetypes
import time
from requests.exceptions import HTTPError

# Load secrets
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Debug: Print secrets to verify (remove in production)
st.write("Secrets loaded:", {
    "ASTRA_DB_API_ENDPOINT": ASTRA_DB_API_ENDPOINT,
    "ASTRA_DB_APPLICATION_TOKEN": "****" if ASTRA_DB_APPLICATION_TOKEN else None,
    "ASTRA_DB_NAMESPACE": ASTRA_DB_NAMESPACE,
    "OPENAI_API_KEY": "****" if OPENAI_API_KEY else None
})

# App title
st.title("DocVectorizer for RAG with OpenAI Embeddings")

# Input for use case to dynamically set collection name
use_case = st.text_input("Enter use case (e.g., technical, marketing)", value="default")
collection_name = f"rag_{use_case.lower().replace(' ', '_')}"

# File uploader for multiple document types
supported_formats = ["pdf", "md", "txt", "json"]
uploaded_files = st.file_uploader(f"Upload Documents ({', '.join(supported_formats)})", 
                                 type=supported_formats, 
                                 accept_multiple_files=True)

# Text splitter settings (optimized for legal documents)
chunk_size = 512
chunk_overlap = 100
st.write(f"Using optimized settings - Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")

# Helper function to generate a hash for a document
def get_document_hash(content, filename):
    combined = f"{content}{filename}"
    return hashlib.md5(combined.encode()).hexdigest()

# Helper function to check if a document already exists
def document_exists(doc_hash, vectorstore):
    try:
        # Search for the document using similarity search with metadata filter
        results = vectorstore.similarity_search(
            query="",  # Empty query
            k=1,  # Only need one result
            filter={"doc_hash": doc_hash}  # Filter by doc_hash
        )
        exists = len(results) > 0
        st.write(f"Checking duplicate for doc_hash {doc_hash}: {'Exists' if exists else 'Not found'}")
        return exists
    except Exception as e:
        st.warning(f"Error checking duplicate for doc_hash {doc_hash}: {str(e)}")
        # Continue with insertion if we can't verify
        return False

# Helper function to load and process different file types
def load_document(file_path, file_name, file_type):
    docs = []
    try:
        if file_type.endswith('pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_type.endswith(('md', 'txt')):
            loader = TextLoader(file_path)
            docs = loader.load()
        elif file_type.endswith('json'):
            def extract_content(data, metadata):
                if isinstance(data, dict):
                    if "content" in data:
                        return data["content"]
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 10:
                            return value
                return str(data)
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                content_key=None,
                text_content=True,
                json_lines=False,
                content_extractor=extract_content
            )
            docs = loader.load()
        for doc in docs:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            doc_hash = get_document_hash(content, file_name)
            doc.metadata.update({
                "filename": file_name,
                "upload_date": datetime.now().isoformat(),
                "file_type": file_type,
                "doc_hash": doc_hash
            })
        st.write(f"Loaded {len(docs)} documents from {file_name}")
        return docs
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return []

if uploaded_files:
    # Initialize embeddings with retry logic
    try:
        def initialize_embeddings_with_retry(max_retries=3, delay=5):
            for attempt in range(max_retries):
                try:
                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-large",
                        openai_api_key=OPENAI_API_KEY
                    )
                    return embeddings
                except HTTPError as e:
                    error_code = e.response.status_code if e.response else None
                    if error_code == 503 and attempt < max_retries - 1:
                        st.warning(f"503 error on attempt {attempt + 1}, retrying in {delay} seconds...")
                        time.sleep(delay)
                    elif error_code == 401:
                        raise Exception("Invalid or unauthorized OPENAI_API_KEY. Please check your key.")
                    elif error_code == 429:
                        raise Exception("Rate limit exceeded for OpenAI API.")
                    else:
                        raise e
            raise Exception("Max retries reached for initializing embeddings")

        embeddings = initialize_embeddings_with_retry()
        sample_text = "This is a test document to check embedding dimensions."
        sample_embedding = embeddings.embed_query(sample_text)
        actual_dimension = len(sample_embedding)
        st.info(f"Embedding model dimension: {actual_dimension}")
        
        # Define expected dimension for text-embedding-3-large
        EMBEDDING_DIMENSION = 3072
        if actual_dimension != EMBEDDING_DIMENSION:
            st.warning(f"Warning: Embedding dimension ({actual_dimension}) does not match expected dimension ({EMBEDDING_DIMENSION})")
    
    except Exception as e:
        st.error(f"Failed to initialize OpenAIEmbeddings: {str(e)}")
        st.stop()
    
    # Create or access vector store
    try:
        vectorstore = AstraDBVectorStore(
            collection_name=collection_name,
            embedding=embeddings,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_NAMESPACE
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDBVectorStore: {str(e)}")
        st.stop()
    
    documents = []
    skipped_docs = 0
    processed_docs = 0
    
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            file_type = mimetypes.guess_type(file.name)[0]
            if not file_type:
                ext = os.path.splitext(file.name)[1].lower()
                if ext == '.md':
                    file_type = 'text/markdown'
                elif ext == '.json':
                    file_type = 'application/json'
                elif ext == '.txt':
                    file_type = 'text/plain'
                elif ext == '.pdf':
                    file_type = 'application/pdf'
            
            with open(tmp_file_path, 'r', errors='ignore') as f:
                try:
                    content = f.read()
                    doc_hash = get_document_hash(content, file.name)
                    
                    # Initialize empty vector store first to skip this check on first document
                    if processed_docs > 0:
                        try:
                            if document_exists(doc_hash, vectorstore):
                                st.info(f"Skipping duplicate document: {file.name}")
                                skipped_docs += 1
                                continue
                        except Exception as e:
                            st.warning(f"Error in duplicate check for {file.name}: {str(e)}")
                except Exception as e:
                    st.warning(f"Could not check duplication for {file.name}: {str(e)}")
            
            try:
                docs = load_document(tmp_file_path, file.name, file_type)
                if docs:
                    documents.extend(docs)
                    processed_docs += 1
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    st.write(f"Documents processed: {processed_docs}, Duplicates skipped: {skipped_docs}")
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        st.write(f"Total chunks created: {len(chunks)}")
        
        try:
            batch_size = 10
            total_chunks = len(chunks)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                current_batch = chunks[i:end_idx]
                
                # We don't need to check chunks for duplicates if we've already filtered documents
                vectorstore.add_documents(current_batch)
                st.write(f"Stored {len(current_batch)} chunks in batch {i//batch_size + 1}")
                
                progress = (end_idx / total_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({end_idx}/{total_chunks} chunks)")
            
            # Try to count documents if this method is available (using AstraDB client directly)
            try:
                if hasattr(vectorstore, "client") and hasattr(vectorstore.client, "count_documents"):
                    total_records = vectorstore.client.count_documents(collection_name=collection_name)
                    st.success(f"Documents successfully vectorized and stored in collection {collection_name}. Total records in DB: {total_records}")
                else:
                    st.success(f"Documents successfully vectorized and stored in collection {collection_name}.")
            except Exception as e:
                st.success(f"Documents successfully vectorized and stored in collection {collection_name}.")
                st.info(f"Count error (non-critical): {str(e)}")
            
        except Exception as e:
            st.error(f"Failed to store documents in AstraDB: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
    else:
        if skipped_docs > 0:
            st.warning("All documents were duplicates, nothing new to process")
        else:
            st.warning("No documents were processed")
else:
    st.info(f"Please upload documents in the following formats: {', '.join(supported_formats)}")