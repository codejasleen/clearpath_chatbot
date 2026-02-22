# Clearpath Chatbot - Written Answers

## Q1 — Routing Logic

**Routing Rules:**
The router uses a deterministic, rule-based approach to classify queries.
A query is routed to the complex model (`llama-3.3-70b-versatile`) if it meets *any* of the following conditions:
1. Contains specific complex keywords ("why", "how", "issue", "error", "complaint", "not working", "fail", "broken", "explain").
2. Is 10 words or longer.
3. Contains more than one question mark.
If *none* of these conditions are met, it defaults to the simple model (`llama-3.1-8b-instant`).

**Why did you draw the boundary here?**
This boundary optimizes for cost and speed while reserving heavy compute for reasoning. Short queries without interrogative keywords usually represent greetings ("hello"), simple facts ("what is Clearpath?"), or navigation requests, which an 8B model handles perfectly. Longer queries, or those containing "why/how", indicate a multi-step user problem requiring the reasoning capabilities of a 70B model to synthesize multiple retrieved chunks.

**Misclassification Example:**
*Query:* ""Hi there, I am just saying hello and checking if this chat is working properly"
*What happened:* The router misclassified this extremely simple greeting as "complex" and routed it to the massive, expensive 70B model.
*Why:* Our rule-based router relies heavily on string length as a proxy for complexity. Specifically, the rule is_long = len(query.split()) >= 10 automatically routes any sentence longer than 15 words to the complex model. The example greeting above is exactly 16 words long. Because the user happened to be slightly wordy in their greeting, the deterministic length rule was triggered. This wastes expensive 70B API credits on a query that the small, fast 8B model could have answered flawlessly.

**Without LLM, how to improve the router?**
I would implement entity extraction (e.g., using a lightweight Spacy pipeline) to detect feature-specific nouns (like "SSO" or "API"). If a query contains high-value feature nouns without a clear verb, I would route it to the complex model, or implement a "clarification required" route that prompts the user for more details before hitting the RAG pipeline.

---

## Q2 — Retrieval Failures

**What was the query?**
"Can I use my company logo in Clearpath?"

**What did your system retrieve?**
The initial RAG system (Bi-Encoder only) retrieved chunks related to "Clearpath branding," "Account profile pictures," and "User Guide: Uploading assets", but completely failed to retrieve the specific chunk from the `15_Enterprise_Plan_Details.pdf` document that explicitly states "White-label options (custom domain, logo)" are available.

**Why did the retrieval fail?**
We initially used a basic dense embedding model (`all-MiniLM-L6-v2`) which retrieves the top 3 results based on compressed semantic similarity. The query asked about a "company logo", which caused the Bi-Encoder to mathematically prioritize documents discussing profile pictures and app branding. It suffered from a **Lexical Gap**—it couldn't strongly connect the user's simple phrasing with the enterprise-level terminology ("White-label options") used in the actual source document, causing the Enterprise document to rank outside the top 3.

**What would fix it? (And how we fixed it)**
Implementing a **Retrieve-then-Rerank Architecture**. 
We fixed this by updating the pipeline to fetch the top 15 results from the fast Bi-Encoder, and then passing those 15 chunks through an accurate **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`). The Cross-Encoder actually reads the query side-by-side with the text, recognizes that "White-label options (custom domain, logo)" perfectly answers the question "Can I use my company logo?", scores it highly, and pushes it to Rank #1 before sending it to the LLM.
---

## Q3 — Cost and Scale

Assuming 5,000 queries per day, with an average input (query + retrieved chunks limit) of 1,000 tokens and an output of 200 tokens.
Assuming a 60/40 split (60% Simple, 40% Complex based on the routing rules).

**Estimated Token Usage:**
*   **Simple (8B Model - 3,000 queries):**
    *   Input: 3,000 * 1,000 = 3,000,000 tokens/day
    *   Output: 3,000 * 200 = 600,000 tokens/day
*   **Complex (70B Model - 2,000 queries):**
    *   Input: 2,000 * 1,000 = 2,000,000 tokens/day
    *   Output: 2,000 * 200 = 400,000 tokens/day

**Biggest Cost Driver:**
The input tokens for the Complex (70B) model. Even though it handles fewer queries, larger models charge significantly more per million input tokens, and our RAG pipeline naturally inflates the input token count with retrieved context.

**Highest-ROI Change:**
**Strict context pruning.** Instead of blindly passing the top 3 chunks (which might total 1000 tokens), implement a relevance threshold score from the embedding search. If chunk 3 has a low similarity score, drop it. Trimming the input context by 30% directly reduces input token costs by 30% with almost zero impact on quality.

**Optimisation to Avoid:**
I would avoid switching to a significantly smaller, lower-quality embedding model just to save storage/memory. Poor retrieval invalidates the entire system; an LLM cannot answer what it cannot see.

---

## Q4: What is Broken

**Design Decision:**
I implemented conversation memory using a **Stateless Frontend Sliding Window**. The Javascript `index.html` maintains an array of conversation JSON objects. When the user asks a question, it appends the question to the array, truncates the array to the last 4 messages, and sends the entire array natively to the FastAPI backend. 

**Why this approach?**
This keeps the Python backend completely stateless and cloud-native without requiring session IDs, Redis caching, or SQL database integrations. The backend simply accepts the array, uses the newest message to run the Vector RAG search, and feeds the entire array directly into the LLM system prompt. 

**The Missing Tradeoff (The Flaw):**
By strictly enforcing a sliding window of 4 messages to save on input token costs, the system artificially introduces **Amnesia**. If a user provides crucial context in message 1 (e.g., "My order ID is 8472") and asks a follow-up question in message 8, the sliding window will have physically deleted the Order ID from the LLM payload, causing a failure. 

**How to solve this (The 3-Layer Enterprise approach):**
To truly solve this tradeoff without token explosion, I would build a 3-layer hybrid memory architecture:
1. **Sliding Window:** Send only the last 4 fast, conversational turns directly to the LLM. 
2. **Conversation Summary:** Use a background LLM call to constantly summarize older messages into a condensed paragraph (e.g., "User ordered phone, wants refund").
3. **RAG Memory / Entity Extraction:** Explicitly extract and store key immutable facts (Order IDs, Names, Plan Tiers) in a database. When a new query arrives, use semantic search to inject those specific facts back into the prompt if needed.

---

## AI Usage

I utilized an AI coding assistant to help build this project. The primary prompts used were:
*   "Explain the difference between Semantic Chunking and Overlapping Chunking for PDF RAG pipelines."
*   "Write a Python script using PyPDF2 and ChromaDB to chunk 30 PDF documents and store them locally using sentence-transformers."
*   "Create a FastAPI application with a Pydantic BaseModel for chat, including a rule-based query string router to select between two model names."
*   "Write a single-file HTML/JS chat interface that sends a POST request to a local FastAPI server and displays a debug panel with response metadata."

---

## Final Self-Reflection: Conversational RAG Architecture

During final testing, I discovered a fundamental flaw in "Conversational RAG" systems: **Contextual Drift**.
When a user asks *"only this much?"* or *"are you sure?"*, the Vector Database mathematically searches the documents for those exact pronouns, retrieves empty/garbage results, and forces the LLM to trigger a safety refusal, even if the LLM remembers the context from the chat history. Hardcoding system prompted edge-cases to handle every possible subset of English colloquialisms is unscalable. 

To create a true Enterprise-grade Conversational RAG system, I implemented a **Query Condensation Layer** into the backend. 
Now, before the Vector Database is queried, a fast LLM (`llama-3.1-8b-instant`) intercepts the entire conversation history along with the vague follow-up question, and rewrites it into a single **Standalone Query** (e.g. translating *"only this much?"* into *"Does the Enterprise plan have more features?"*). 
By searching the Vector Database with this condensed query instead of the raw user input, the retrieval context is always 100% perfectly aligned with the conversation, completely eliminating conversational hallucination refusals.
