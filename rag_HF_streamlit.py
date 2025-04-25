import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore  # Fixed import
from langchain.embeddings import HuggingFaceHubEmbeddings
import os
import tempfile
import hashlib
from datetime import datetime
import json
import mimetypes

# Load secrets
try:
    ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_DB_NAMESPACE = st.secrets.get("ASTRA_DB_NAMESPACE", None)
    HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Debug: Print secrets to verify (remove in production)
st.write("Secrets loaded:", {
    "ASTRA_DB_API_ENDPOINT": ASTRA_DB_API_ENDPOINT,
    "ASTRA_DB_APPLICATION_TOKEN": "****" if ASTRA_DB_APPLICATION_TOKEN else None,
    "ASTRA_DB_NAMESPACE": ASTRA_DB_NAMESPACE,
    "HUGGINGFACEHUB_API_TOKEN": "****" if HUGGINGFACEHUB_API_TOKEN else None
})

# App title
st.title("DocVectorizer for RAG with Hugging Face Embeddings")

# Input for use case to dynamically set collection name
use_case = st.text_input("Enter use case (e.g., technical, marketing)", value="default")
collection_name = f"rag_{use_case.lower().replace(' ', '_')}"

# File uploader for multiple document types
supported_formats = ["pdf", "md", "txt", "json"]
uploaded_files = st.file_uploader(f"Upload Documents ({', '.join(supported_formats)})", 
                                 type=supported_formats, 
                                 accept_multiple_files=True)

# Text splitter settings
chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=750, step=50)
chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=150, step=10)

# Helper function to generate a hash for a document
def get_document_hash(content, filename):
    combined = f"{content}{filename}"
    return hashlib.md5(combined.encode()).hexdigest()

# Helper function to check if a document already exists
def document_exists(doc_hash, vectorstore):
    try:
        results = vectorstore.similarity_search(
            "check_duplicate_placeholder",
            k=1,
            filter={"doc_hash": doc_hash}
        )
        return len(results) > 0
    except Exception:
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
        return docs
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return []

if uploaded_files:
    # Initialize embeddings
    try:
        embeddings = HuggingFaceHubEmbeddings(
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        )
        sample_text = "This is a test document to check embedding dimensions."
        sample_embedding = embeddings.embed_query(sample_text)
        actual_dimension = len(sample_embedding)
        st.info(f"Embedding model dimension: {actual_dimension}")
        
        # Define expected dimension
        EMBEDDING_DIMENSION = 384  # Known for paraphrase-multilingual-MiniLM-L12-v2
        if actual_dimension != EMBEDDING_DIMENSION:
            st.warning(f"Warning: Embedding dimension ({actual_dimension}) does not match expected dimension ({EMBEDDING_DIMENSION})")
    
    except Exception as e:
        st.error(f"Failed to initialize HuggingFaceHubEmbeddings: {str(e)}")
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
                    if document_exists(doc_hash, vectorstore):
                        st.info(f"Skipping duplicate document: {file.name}")
                        skipped_docs += 1
                        continue
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
        
        try:
            batch_size = 10
            total_chunks = len(chunks)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                current_batch = chunks[i:end_idx]
                
                filtered_batch = []
                for doc in current_batch:
                    if not document_exists(doc.metadata.get("doc_hash", ""), vectorstore):
                        filtered_batch.append(doc)
                
                if filtered_batch:
                    vectorstore.add_documents(filtered_batch)
                
                progress = (end_idx / total_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({end_idx}/{total_chunks} chunks)")
            
            st.success(f"Documents successfully vectorized and stored in collection {collection_name}")
            
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