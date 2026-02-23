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

According to the optional Bonus Challenges rubric, the following three challenges were successfully implemented:

1. **Conversation memory:** The chatbot maintains conversation memory across turns. This is implemented via an active sliding-window memory (last 4 messages) managed by a backend session dictionary (`CONVERSATIONS`). A *Query Condensation Layer* using the 8B model was also implemented to rewrite pronoun-heavy follow-up questions before hitting the vector database.
2. **Eval harness:** A dedicated `eval_harness.py` test suite was written. It contains a dozen test queries covering factual retrieval, out-of-bounds questions, and competitor mentions. Running `python backend/eval_harness.py` automatically evaluates the system against these expected answers and reports Pass/Fail for each.
3. **Live deploy:** The full backend and frontend stack was deployed to a business-grade cloud provider (AWS EC2) and is accessible via a public URL interface.

---

## Additional Self-Improved Features

Beyond the assignment rubric, several advanced RAG concepts were engineered to ensure production-grade reliability:

1. **Advanced Two-Stage Retrieval (Reranking):** Upgraded from standard dense retrieval to a two-stage pipeline. ChromaDB first fetches the top 15 chunks using `all-MiniLM-L6-v2`, and then a powerful Neural **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) reranks them mathematically, returning only the top 3 most precisely relevant chunks. I also tuned the negative logits threshold to allow for typo tolerance.
2. **Lexical Grounding Flagging:** Built a custom algorithmic hallucination checker. Before returning the final answer, the system extracts all long nouns from the LLM's response and verifies that at least 50% (for simple models) to 60% (for complex models) of them actually exist in the retrieved PDF chunks. If an LLM hallucinates new features not found in the manuals, it perfectly flags the output with `low_grounding`.
3. **Query Condensation Layer:** Built an LLM-powered query rewrite step using the fast 8B model. When a user asks a follow-up question containing pronouns or implicit context (e.g., "how much does it cost?"), the condensation layer rewrites the query using the entire conversation history into a standalone, search-optimized string (e.g., "what is the price of the enterprise plan?"). This dramatically improves vector semantic similarity matching for multi-turn conversations.

---

## Known Issues & Limitations

1. **In-Memory State Management:** The conversation history is currently stored in a Python dictionary (`CONVERSATIONS`) inside the server memory. While fast, this means that if the server crashes or restarts, all active user conversational histories are lost. In a real production system, this would be moved to Redis or a database.
2. **No Streaming:** Output streaming was purposely disabled in favor of strict JSON responses (`/query`) to ensure the frontend could confidently parse the structured evaluator warnings (`metadata.flags`) without complex stream chunk parsing. This increases perceived latency for the user.
3. **Cross-Encoder Latency "Cold Start":** Because the Cross-Encoder model runs locally on CPU (especially noticeable on the EC2 instance without a GPU), the very first query takes slightly longer as PyTorch loads the model into active memory.
