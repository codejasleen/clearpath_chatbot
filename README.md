# Clearpath Chatbot Support System

This project is a functional, production-ready customer support chatbot for "Clearpath", a fictional project management SaaS. It implements an advanced RAG (Retrieval-Augmented Generation) pipeline, a deterministic model router, and an output evaluator to ensure safe and accurate responses from the LLM.

### 🌐 Live Deployment
The project is currently deployed and accessible at: **http://13.49.75.239**

---

## How to Run the Project Locally

**Prerequisites:**
- Python 3.9+
- A Groq API Key

**1. Setup the Environment**
Clone this repository and navigate to the project root:
```bash
# Create and activate a Virtual Environment
python3 -m venv venv

# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate

# Install the dependencies
pip install fastapi uvicorn groq chromadb sentence-transformers pydantic pypdf2 python-dotenv python-multipart
```

**2. Set up the Environment Variables**
Create a `.env` file in the root directory or export the variable directly:
```bash
# On Windows (PowerShell):
$env:GROQ_API_KEY="your-api-key"

# On Mac/Linux:
export GROQ_API_KEY="your-api-key"
```

**3. Start the Backend Server**
The backend `main.py` is configured to handle the `/query` endpoint and serve the frontend statically.
Run the startup command from the root directory:
```bash
python -m uvicorn backend.main:app --reload
```
The server will now be running at `http://localhost:8000`.

**4. Access the Chat Interface**
Open your web browser and go to `http://localhost:8000`. The frontend interface is served directly from the backend. Type a message and hit "Send"!

---

## Groq Models & Environment Config

This project utilizes two models via the Groq API. Queries are routed dynamically by the rule-based logic in `main.py` to balance cost, speed, and reasoning capabilities:

1. **Simple Queries:** `llama-3.1-8b-instant`
   * *Used for greetings, short questions, and single-fact lookups. Also used for the Query Condensation step.*
2. **Complex Queries:** `llama-3.3-70b-versatile`
   * *Used for queries containing interrogative words (why/how/explain), multi-step questions, or queries exceeding 10 words.*

**Environment Config:** The application simply requires the `GROQ_API_KEY` to be set in the environment.

---

## Bonus Challenges Attempted

1. **Conversational Memory with Query Condensation:** Implemented an active sliding-window memory (last 4 messages) managed by a backend session dictionary. Importantly, to make this work with the vector database, I implemented a **Query Condensation Layer** using the 8B model. It intercepts follow-up questions containing pronouns (e.g., "Are you sure about it?") and rewrites them into standalone search queries before hitting ChromaDB.
2. **Advanced Two-Stage Retrieval (Reranking):** Upgraded from standard dense retrieval to a two-stage pipeline. ChromaDB first fetches the top 15 chunks using `all-MiniLM-L6-v2`, and then a powerful Neural **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) reranks them mathematically, returning only the top 3 most precisely relevant chunks. I also tuned the negative logits threshold to allow for typo tolerance.
3. **Lexical Grounding Evaluator:** Built a custom algorithmic hallucination checker. Before returning the final answer, the system extracts all long nouns from the LLM's response and verifies that at least 50% of them actually exist in the retrieved PDF chunks. If an LLM hallucinates new features not found in the manuals, it perfectly flags the output with `low_grounding`.
4. **AWS Deployment:** Fully deployed the backend and frontend on an AWS EC2 instance.

---

## Known Issues & Limitations

1. **In-Memory State Management:** The conversation history is currently stored in a Python dictionary (`CONVERSATIONS`) inside the server memory. While fast, this means that if the server crashes or restarts, all active user conversational histories are lost. In a real production system, this would be moved to Redis or a database.
2. **No Streaming:** Output streaming was purposely disabled in favor of strict JSON responses (`/query`) to ensure the frontend could confidently parse the structured evaluator warnings (`metadata.flags`) without complex stream chunk parsing. This increases perceived latency for the user.
3. **Cross-Encoder Latency "Cold Start":** Because the Cross-Encoder model runs locally on CPU (especially noticeable on the EC2 instance without a GPU), the very first query takes slightly longer as PyTorch loads the model into active memory.
