import os
import glob
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

# Initialize ChromaDB client. Data will be saved locally to a 'chroma_db' folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# We use SentenceTransformers for local embedding generation.
# 'all-MiniLM-L6-v2' is small, fast, and excellent for RAG retrieved tasks.
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="clearpath_docs",
    embedding_function=sentence_transformer_ef
)

# Initialize the Cross-Encoder for Stage 2 Reranking
# This model is specifically trained to grade how well a document answers a query
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def extract_text_from_pdfs(docs_dir):
    """Reads all PDFs in the given directory and extracts text."""
    all_text = []
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    
    for file_path in pdf_files:
        try:
            reader = PdfReader(file_path)
            file_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    file_text += page_text
            
            # We track the source file for citation/debugging
            all_text.append({"source": os.path.basename(file_path), "text": file_text})
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return all_text

def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Splits text into chunks of `chunk_size` characters with `overlap`.
    This is an overlapping chunking strategy.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap) # Move forward, allowing for overlap
        
    return chunks

def build_vector_db():
    """Extracts text, chunks it, and adds it to ChromaDB."""
    print("Checking if database already exists...")
    if collection.count() > 0:
        print(f"Database contains {collection.count()} chunks. Ready to use.")
        return

    print("Extracting text from docs...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(os.getcwd(), 'docs')
    
    documents = extract_text_from_pdfs(docs_dir)
    
    print("Chunking text and generating embeddings...")
    chunk_docs = []
    metadatas = []
    ids = []
    
    chunk_id_counter = 0
    
    for doc in documents:
        # We process each document individually to preserve file-level metadata
        chunks = chunk_text(doc['text'])
        for i, chunk in enumerate(chunks):
            chunk_docs.append(chunk)
            metadatas.append({"source": doc['source'], "chunk_index": i})
            ids.append(f"chunk_{chunk_id_counter}")
            chunk_id_counter += 1
            
    # Add to ChromaDB in batches (Chroma handles the embedding generation automatically here)
    batch_size = 500
    for i in range(0, len(chunk_docs), batch_size):
        end_idx = min(i + batch_size, len(chunk_docs))
        print(f"Adding batch {i} to {end_idx}...")
        collection.add(
            documents=chunk_docs[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        
    print(f"Successfully added {collection.count()} chunks to the vector database.")

def retrieve_context(query, top_k=3):
    """Retrieves the most relevant chunks using a two-stage Retrieve & Rerank approach."""
    
    # --- TEMPORARY SCREENSHOT TOGGLE ---
    USE_RERANKER = True # Change this to True to see the fixed Reranker!
    # ---------------------------------
    
    if not USE_RERANKER:
        # --- BEFORE: The Basic Bi-Encoder (The Failure Case) ---
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        retrieved_chunks = []
        final_meta = []
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            retrieved_chunks.append(f"[Source: {source}]\n{doc}")
            final_meta.append(results['metadatas'][0][i])
        return "\n\n---\n\n".join(retrieved_chunks), final_meta
    
    else:
        # --- AFTER: The Cross-Encoder Reranker (The Success Case) ---
        fetch_k = 15
        results = collection.query(
            query_texts=[query],
            n_results=fetch_k
        )
        
        retrieved_docs = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]
        
        cross_input = [[query, doc] for doc in retrieved_docs]
        scores = reranker_model.predict(cross_input)
        
        scored_results = sorted(zip(scores, retrieved_docs, retrieved_metadatas), key=lambda x: x[0], reverse=True)
        
        # Filter out chunks below a minimum relevance score
        # Cross-encoders output "logits", which can be negative (e.g., -1.5) even for valid fuzzy matches (like typos).
        # We use -2.0 to allow for typos but still reject completely off-topic queries.
        RELEVANCE_THRESHOLD = -2.0
        relevant_results = [(s, d, m) for s, d, m in scored_results if s >= RELEVANCE_THRESHOLD]
        best_results = relevant_results[:top_k]
        
        formatted_chunks = []
        final_metadata = []
        
        for score, doc, meta in best_results:
            source = meta['source']
            meta['reranker_score'] = float(score)
            final_metadata.append(meta)
            formatted_chunks.append(f"[Source: {source}]\n{doc}")
            
        return "\n\n---\n\n".join(formatted_chunks), final_metadata

if __name__ == "__main__":
    # If we run this script directly, it will build the database
    build_vector_db()

