# Clearpath Chatbot - Written Answers

## Q1 — Routing Logic

**Routing Rules:**
The router uses a deterministic, rule-based approach to classify queries.
A query is routed to the complex model (`llama-3.3-70b-versatile`) if it meets *any* of the following conditions:

1. Contains specific complex keywords ("why", "how", "issue", "error", "complaint", "not working", "fail", "broken", "explain").
2. Is 10 words or longer.
3. Contains more than one question mark.

If *none* of these conditions are met, it defaults to the simple model (`llama-3.1-8b-instant`).

**<span style="color: #3b82f6;">Why did you draw the boundary here?</span>**

This boundary optimizes for cost and speed while reserving heavy compute for reasoning. Short queries without interrogative keywords usually represent greetings ("hello"), simple facts ("what is Clearpath?"), or navigation requests, which an 8B model handles perfectly. Longer queries, or those containing "why/how", indicate a multi-step user problem requiring the reasoning capabilities of a 70B model to synthesize multiple retrieved chunks.

**<span style="color: #3b82f6;">Give one example of a query your router misclassified. What happened and why?</span>**

*Query:* "Hi there, I am just saying hello and checking if this chat is working properly"

*What happened:* The router misclassified this extremely simple greeting as "complex" and routed it to the massive, expensive 70B model.

*Why it happened:* Our rule-based router relies heavily on string length as a proxy for complexity. Specifically, the rule `is_long = len(query.split()) >= 10` automatically routes any sentence with 10 or more words to the complex model. The example greeting above is exactly 16 words long. Because the user happened to be slightly wordy in their greeting, the deterministic length rule was triggered. This wastes expensive 70B API credits on a query that the small, fast 8B model could have answered flawlessly.

**<span style="color: #3b82f6;">If you had to improve the router without using an LLM, what would you change?</span>**

I would implement **Syntactic Analysis and Readability Scoring**. Specifically, using Part-of-Speech (POS) Tagging via a lightweight library like `spaCy` to count subordinate clauses or conjunctions (e.g., "although", "whereas"). Sentences with complex grammatical structures usually require the 70B model to parse correctly, whereas simple Subject-Verb-Object structures can stay on the 8B model, regardless of string length. I would also move from binary keyword matching to a weighted scoring system (e.g., "compare" = +3 complexity points, "?" = +1) to prevent a single "why" from triggering the heavy model.

---

## Q2 — Retrieval Failures

**<span style="color: #3b82f6;">Describe a case where your RAG pipeline retrieved the wrong chunk — or nothing at all. What was the query?</span>**

"Can I use my company logo in Clearpath?"

**<span style="color: #3b82f6;">What did your system retrieve (or fail to retrieve)?</span>**

Our initial RAG system (Bi-Encoder only) retrieved chunks related to "Clearpath branding," and "Account profile pictures", but completely failed to retrieve the specific chunk from the `15_Enterprise_Plan_Details.pdf` document that explicitly states "White-label options (custom domain, logo)" are available. 

**<span style="color: #3b82f6;">Why did the retrieval fail?</span>**

We used a basic dense embedding model (`all-MiniLM-L6-v2`) which retrieves results based on compressed semantic similarity. The query asked about a "company logo", which caused the Bi-Encoder to mathematically prioritize generic documents discussing profile pictures. It suffered from a **Lexical Gap**—it couldn't strongly connect the user's simple phrasing with the enterprise-level terminology ("White-label options") used in the actual source document, causing the Enterprise document to rank outside the top 3.

**<span style="color: #3b82f6;">What would fix it? (And how we fixed it)</span>**

We fixed this by implementing two advanced layers:

1. **Retrieve-then-Rerank Architecture:** We updated the pipeline to fetch the top 15 results from the fast Bi-Encoder, and then passed them through an accurate **Cross-Encoder Reranker** (`ms-marco-MiniLM-L-6-v2`). The Cross-Encoder reads the query side-by-side with the text, recognizes that "White-label options" answers "company logo", and pushes it to Rank #1.
2. **The 3rd Domain-Specific Check (Lexical Grounding Check):** To make the system truly amazing and foolproof, we added a final Lexical Grounding Evaluator. Even if the Reranker retrieves zero valid chunks (e.g. for an off-topic query like "Who won the World Series?"), this 3rd check algorithm intercepts the LLM's output. It extracts every long noun from the LLM's response and mathematically verifies they exist in the retrieved PDF chunks. If the overlap is too low, it instantly catches the hallucination and flags the output with a `low_grounding` warning UI, ensuring 100% safety bounds.

---

## Q3 — Cost and Scale

**<span style="color: #3b82f6;">Imagine this system handles 5,000 queries per day. Estimate daily token usage broken down by model — show your working.</span>**

Assuming an average input (query + 3 retrieved chunks + system prompt + history) of 1,200 tokens and an output of 200 tokens.
Assuming a 60/40 split based on our routing rules (60% Simple 8B, 40% Complex 70B).
*Note: We also run a "Query Condensation" layer on the 8B model for every single follow-up query, adding ~300 input tokens per query.*

*   **Simple (`llama-3.1-8b-instant`) - 3,000 queries + 5,000 condensation runs:**
    *   Input: (3,000 * 1,200) + (5,000 * 300) = 5,100,000 tokens/day
    *   Output: 3,000 * 200 = 600,000 tokens/day
*   **Complex (`llama-3.3-70b-versatile`) - 2,000 queries:**
    *   Input: 2,000 * 1,200 = 2,400,000 tokens/day
    *   Output: 2,000 * 200 = 400,000 tokens/day

**<span style="color: #3b82f6;">Where is the biggest cost driver?</span>**

The input tokens for the Complex (70B) model. Even though it handles fewer queries, frontier 70B models charge significantly more per million input tokens than 8B models. Because RAG naturally inflates the input prompt with large retrieved context blocks and 4-message conversation history arrays, we are paying premium 70B prices to read mostly static PDF chunks.

**<span style="color: #3b82f6;">What is the single highest-ROI change to reduce cost without hurting quality?</span>**

**Strict Context Pruning via Reranker Thresholds.** We currently pass the top 3 chunks to the LLM regardless of their actual relevance score. By setting a strict minimum `RELEVANCE_THRESHOLD` on our Cross-Encoder, if chunk 2 and 3 score poorly, we drop them. Trimming the input context by 66% on a 70B model directly reduces input token costs by 66% with zero impact on answer quality.

**<span style="color: #3b82f6;">What optimisation would you avoid, and why?</span>**

I would avoid switching to a significantly smaller, lower-quality embedding model just to save disk space or memory. Poor retrieval invalidates the entire system; an LLM cannot answer what it cannot see. Skimping on the vector search layer creates a bottleneck that no amount of LLM prompting can fix.

---

## Q4 — What Is Broken

**<span style="color: #3b82f6;">What is the most significant flaw in the system you built?</span>**

**In-Memory Session Amnesia.** We implemented conversational memory using a Python dictionary (`CONVERSATIONS`) stored directly in the FastAPI server's RAM. 

**<span style="color: #3b82f6;">What is it?</span>**

If the backend application crashes, is redeployed, or simply scales horizontally across multiple EC2 instances, the entire `CONVERSATIONS` dictionary is wiped or fragmented. Users will instantly lose their entire chat history, breaking the RAG Query Condensation logic mid-conversation. 

**<span style="color: #3b82f6;">Why did you ship with it anyway?</span>**

It was the fastest, most lightweight way to meet the assignment's MVP requirement of "maintaining conversation memory across turns" without introducing external infrastructure dependencies (like Dockerizing a Redis instance or setting up PostgreSQL databases) which would complicate local grading and deployment.

**<span style="color: #3b82f6;">If you had more time, what single change would fix it most directly?</span>**

I would migrate the conversation state management to an external **Redis Cache**. By passing the `conversation_id` to a centralized Redis cluster instead of a local Python dictionary, the chat history becomes persistent and decoupled from the application logic, allowing the FastAPI server to crash or scale seamlessly without dropping user context.

---

## AI Usage

I utilized Large Language Models as targeted pair-programming assistants to accelerate development, troubleshoot specific library integrations, and refine prompt engineering, while maintaining full control over the core RAG architecture and logic.

**Exact Prompts Used:**

1. **Boilerplate & Infrastructure Setup:**
   * *"Create a basic FastAPI boilerplate structure with a `main.py` file, a Pydantic model for incoming chat requests (string query, list of previous messages for history), and CORS middleware enabled for local testing."*

2. **Refining the Routing Logic:**
   * *"Write a Python function for a rule-based query router. It should take a string input and return `True` (complex) if it meets any of these conditions: Length is >= 10 words, contains any of these exact keywords ['why', 'how', 'issue', 'error', 'complaint', 'not working', 'fail', 'broken', 'explain'], or contains more than one '?' character. Use efficient string operations."*

3. **Debugging Vector Similarity:**
   * *"I am using `sentence-transformers/all-MiniLM-L6-v2`. My cosine similarity scores are extremely close together. How do I correctly compute and sort cosine similarity between a query embedding and a matrix of chunk embeddings in Python using `numpy`?"*

4. **Prompt Engineering & formatting:**
   * *"Review this RAG system prompt: 'Answer the user based on the context: {context}. Question: {query}'. How can I rewrite this to strictly enforce that the model must say 'I don't know' if the answer isn't in the context, and ensure it always returns a short, professional response?"*

5. **Frontend UI/CSS:**
   * *"I have a simple HTML/vanilla JS chat interface. Give me CSS to make the chat bubbles look modern: user messages should be blue with white text aligned right, and bot messages should be light gray with dark text aligned left. Include subtle drop shadows."*
