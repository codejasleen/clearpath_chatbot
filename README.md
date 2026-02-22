# Clearpath Chatbot Support System

This project is a functional, testable customer support chatbot for "Clearpath", a fictional project management SaaS. It implements a fully local RAG (Retrieval-Augmented Generation) pipeline, a deterministic model router, and an output evaluator to ensure safe and accurate responses from the LLM.

## How to Run the Project Locally

**Prerequisites:**
- Python 3.9+
- A Groq API Key

**1. Setup the Environment**
Clone this repository and navigate to the project root:
```bash
# Create and activate a Virtual Environment
python -m venv venv

# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate

# Install the dependencies
pip install fastapi uvicorn groq chromadb sentence-transformers pydantic pypdf2 python-multipart
```

**2. Set up the API Key**
You must set your Groq API key as an environment variable:
```bash
# On Windows (PowerShell):
$env:GROQ_API_KEY="your-api-key"

# On Mac/Linux:
export GROQ_API_KEY="your-api-key"
```

**3. Build the Vector Database (Layer 1)**
Run the RAG extraction script to process the 30 PDF documents in the `docs/` folder, chunk them, create local embeddings, and store them in a local ChromaDB folder.
```bash
python backend/rag.py
```
*(Note: This might take a minute on the first run as it downloads the local embedding model).*

**4. Start the Backend API Server**
Start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```
The server will be running at `http://127.0.0.1:8000`.

**5. Open the Chat Interface**
Using your standard file explorer, navigate to the `frontend/` folder and double-click the `index.html` file to open it in your web browser. 
Type a message and click "Send"!

---

## Groq Models Used

This project utilizes two models via the Groq API, selected dynamically by the `router.py` logic:

1.  **Simple Queries:** `llama-3.1-8b-instant` 
    *   *Used for greetings, short questions, and single-fact lookups.*
2.  **Complex Queries:** `llama-3.3-70b-versatile`
    *   *Used for queries containing interrogative words (why/how/explain), error reports, or multi-step questions.*

## Known Limitations & Missing Features

1.  **Streaming Answers:** Streaming was not implemented; the user must wait for the entire generation process (retrieval + LLM inference) to finish before seeing the response.

