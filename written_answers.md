# Clearpath Chatbot - Written Answers

## Q1 — Routing Logic

**Routing Rules:**
The router uses a deterministic, rule-based approach to classify queries.
A query is routed to the complex model (`llama-3.3-70b-versatile`) if it meets *any* of the following conditions:
1. Contains specific complex keywords ("why", "how", "issue", "error", "complaint", "not working", "fail", "broken", "explain").
2. Is 10 words or longer.
3. Contains more than one question mark.
If *none* of these conditions are met, it defaults to the simple model (`llama-3.1-8b-instant`).

<span style="color:blue">**Why did you draw the boundary here?**</span>
This boundary optimizes for cost and speed while reserving heavy compute for reasoning. Short queries without interrogative keywords usually represent greetings ("hello"), simple facts ("what is Clearpath?"), or navigation requests, which an 8B model handles perfectly. Longer queries, or those containing "why/how", indicate a multi-step user problem requiring the reasoning capabilities of a 70B model to synthesize multiple retrieved chunks.

<span style="color:blue">**Give one example of a query your router misclassified. What happened and why?**</span>
*Query:* "Hi there, I am just saying hello and checking if this chat is working properly"
*What happened:* The router misclassified this extremely simple greeting as "complex" and routed it to the massive, expensive 70B model.
*Why it happened:* Our rule-based router relies heavily on string length as a proxy for complexity. Specifically, the rule `is_long = len(query.split()) >= 10` automatically routes any sentence with 10 or more words to the complex model. The example greeting above is exactly 16 words long. Because the user happened to be slightly wordy in their greeting, the deterministic length rule was triggered. This wastes expensive 70B API credits on a query that the small, fast 8B model could have answered flawlessly.

<span style="color:blue">**If you had to improve the router without using an LLM, what would you change?**</span>
I would implement **Syntactic Analysis and Readability Scoring**. Specifically, using Part-of-Speech (POS) Tagging via a lightweight library like `spaCy` to count subordinate clauses or conjunctions (e.g., "although", "whereas"). Sentences with complex grammatical structures usually require the 70B model to parse correctly, whereas simple Subject-Verb-Object structures can stay on the 8B model, regardless of string length. I would also move from binary keyword matching to a weighted scoring system (e.g., "compare" = +3 complexity points, "?" = +1) to prevent a single "why" from triggering the heavy model.

---

## Q2 — Retrieval Failures

<span style="color:blue">**Describe a case where your RAG pipeline retrieved the wrong chunk — or nothing at all. What was the query?**</span>
"Who won the World Series in 2020?"

<span style="color:blue">**What did your system retrieve (or fail to retrieve)?**</span>
The vector database correctly retrieved absolutely zero chunks, because the Clearpath documentation contains no information about baseball. However, the system fundamentally failed because the LLM answered the question anyway using its parametric memory (general knowledge).

<span style="color:blue">**Why did the retrieval fail?**</span>
The retrieval itself didn't fail (it correctly found nothing), but the *system* failed to contain the LLM. Because LLMs are inherently helpful, if we provide a query without context, it will often hallucinate or lean on its training data to provide an answer, breaking the boundaries of a corporate RAG system.

<span style="color:blue">**What would fix it? (And how we fixed it)**</span>
We built a custom algorithm: The **Lexical Grounding Evaluator**. 
Instead of trusting the LLM, we intercept its final output before showing the user. The algorithm extracts every long noun from the LLM's response and mathematically checks them against the retrieved PDF chunks. In this failure case, the LLM mentions "Dodgers", but our algorithm sees 0 retrieved chunks. Because the mathematical overlap ratio is 0% (less than our 0.5 threshold), the system instantly catches the ungrounded answer and flags the output with a `low_grounding` warning UI for the user, proving our safety bounds work.

---

## Q3 — Cost and Scale

<span style="color:blue">**Imagine this system handles 5,000 queries per day. Estimate daily token usage broken down by model — show your working.**</span>
Assuming an average input (query + 3 retrieved chunks + system prompt + history) of 1,200 tokens and an output of 200 tokens.
Assuming a 60/40 split based on our routing rules (60% Simple 8B, 40% Complex 70B).
*Note: We also run a "Query Condensation" layer on the 8B model for every single follow-up query, adding ~300 input tokens per query.*

*   **Simple (`llama-3.1-8b-instant`) - 3,000 queries + 5,000 condensation runs:**
    *   Input: (3,000 * 1,200) + (5,000 * 300) = 5,100,000 tokens/day
    *   Output: 3,000 * 200 = 600,000 tokens/day
*   **Complex (`llama-3.3-70b-versatile`) - 2,000 queries:**
    *   Input: 2,000 * 1,200 = 2,400,000 tokens/day
    *   Output: 2,000 * 200 = 400,000 tokens/day

<span style="color:blue">**Where is the biggest cost driver?**</span>
The input tokens for the Complex (70B) model. Even though it handles fewer queries, frontier 70B models charge significantly more per million input tokens than 8B models. Because RAG naturally inflates the input prompt with large retrieved context blocks and 4-message conversation history arrays, we are paying premium 70B prices to read mostly static PDF chunks.

<span style="color:blue">**What is the single highest-ROI change to reduce cost without hurting quality?**</span>
**Strict Context Pruning via Reranker Thresholds.** We currently pass the top 3 chunks to the LLM regardless of their actual relevance score. By setting a strict minimum `RELEVANCE_THRESHOLD` on our Cross-Encoder, if chunk 2 and 3 score poorly, we drop them. Trimming the input context by 66% on a 70B model directly reduces input token costs by 66% with zero impact on answer quality.

<span style="color:blue">**What optimisation would you avoid, and why?**</span>
I would avoid switching to a significantly smaller, lower-quality embedding model just to save disk space or memory. Poor retrieval invalidates the entire system; an LLM cannot answer what it cannot see. Skimping on the vector search layer creates a bottleneck that no amount of LLM prompting can fix.

---

## Q4 — What Is Broken

<span style="color:blue">**What is the most significant flaw in the system you built?**</span>
**In-Memory Session Amnesia.** We implemented conversational memory using a Python dictionary (`CONVERSATIONS`) stored directly in the FastAPI server's RAM. 

<span style="color:blue">**What is it?**</span>
If the backend application crashes, is redeployed, or simply scales horizontally across multiple EC2 instances, the entire `CONVERSATIONS` dictionary is wiped or fragmented. Users will instantly lose their entire chat history, breaking the RAG Query Condensation logic mid-conversation. 

<span style="color:blue">**Why did you ship with it anyway?**</span>
It was the fastest, most lightweight way to meet the assignment's MVP requirement of "maintaining conversation memory across turns" without introducing external infrastructure dependencies (like Dockerizing a Redis instance or setting up PostgreSQL databases) which would complicate local grading and deployment.

<span style="color:blue">**If you had more time, what single change would fix it most directly?**</span>
I would migrate the conversation state management to an external **Redis Cache**. By passing the `conversation_id` to a centralized Redis cluster instead of a local Python dictionary, the chat history becomes persistent and decoupled from the application logic, allowing the FastAPI server to crash or scale seamlessly without dropping user context.
