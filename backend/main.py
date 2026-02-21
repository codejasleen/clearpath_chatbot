from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import time
import re
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import json
from backend.rag import retrieve_context
from dotenv import load_dotenv
load_dotenv() # This reads the .env file and sets the variables!

app = FastAPI()

# Allow our frontend to talk to our backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(os.path.dirname(current_dir), 'frontend', 'index.html')
    return FileResponse(frontend_path)

# Make sure you set your GROQ_API_KEY environment variable!
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

# --- LAYER 2: MODEL ROUTER ---
def route_query(query: str) -> str:
    """
    Rule-based router.
    Simple queries go to llama-3.1-8b-instant.
    Complex queries go to llama-3.3-70b-versatile.
    """
    query_lower = query.lower()
    words = query_lower.split()
    
    # Rule 1: Complex keywords
    complex_keywords = ["why", "how", "issue", "error", "complaint", "not working", "fail", "broken", "explain"]
    has_complex_keyword = any(kw in query_lower for kw in complex_keywords)
    
    # Rule 2: Multi-step or long queries
    is_long = len(words) >= 10
    
    # Rule 3: Multiple questions
    multiple_questions = query.count('?') > 1

    if has_complex_keyword or is_long or multiple_questions:
        return "llama-3.3-70b-versatile"
    else:
        return "llama-3.1-8b-instant"

# --- LAYER 3: OUTPUT EVALUATOR ---
def evaluate_output(query: str, response: str, retrieved_chunks_used: int, context: str, is_complex: bool) -> list:
    """
    Evaluates the LLM output for potential issues.
    Returns a list of flags (strings).
    """
    flags = []
    resp_lower = response.lower()
    
    # Check 1: Refusals
    refusal_keywords = [
        "i cannot", "i can't", "i don't know", "i do not know", 
        "as an ai", "i'm sorry, but", "i am sorry, but",
        "i don't have enough information"
    ]
    is_refusal = any(kw in resp_lower for kw in refusal_keywords)
    if is_refusal:
        flags.append("Model refused to answer or indicated lack of knowledge.")
        
    # Check 2: No-context response for factual queries
    # If no context was retrieved AND it's not a refusal AND it's not small talk, we might want to flag it. 
    # But since we use Few-Shot to allow small talk, let's just only flag "No Context" if they asked a complex question.
    if retrieved_chunks_used == 0 and is_complex:
        flags.append("No context retrieved for this complex query.")
        
    # Check 3: Groundedness / Overlap Check 
    # ONLY run this if it's a Complex query AND it's not a known Refusal
    if is_complex and not is_refusal:
        words_in_response = set(re.findall(r'\b[a-z]{6,}\b', resp_lower))
        if words_in_response and context:
            context_lower = context.lower()
            # Count how many of these long words actually appear in the retrieved context
            overlap_count = sum(1 for word in words_in_response if word in context_lower)
            overlap_ratio = overlap_count / len(words_in_response)
            
            # If less than 50% of the significant words came from the context, it's very likely hallucinating
            if overlap_ratio < 0.5:
                flags.append(f"Warning: Low groundedness score ({overlap_ratio:.0%}). Possible hallucination detected.")
        
    return flags

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    # Extract the very last message the user sent as the actual query
    if not request.messages:
        return {"error": "No messages provided"}
    
    current_query = request.messages[-1].content
    
    # --- ADVANCED: Query Condensation Layer ---
    # Convert vague follow-ups ("only this much?") into standalone search queries using past chat history
    search_query = current_query
    if len(request.messages) > 1:
        condensation_prompt = "Given the following conversation history, rewrite the user's latest follow-up question into a single standalone search query that contains all necessary facts (like subjects or plan names) from the past messages.\nIf the latest question is already completely standalone, return it exactly as is.\nCRITICAL: If the latest question is just a conversational confirmation (like 'are you sure?', 'is this correct?', 'thanks'), DO NOT rewrite it into a factual search. Just return the exact phrase as is!\nDo NOT answer the question. ONLY return the rewritten query string.\n\nConversation History:\n"
        for msg in request.messages[:-1]:
            condensation_prompt += f"{msg.role.capitalize()}: {msg.content}\n"
            
        condensation_prompt += f"\nLatest Question: {current_query}\nStandalone Query:"
        
        try:
            condense_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": condensation_prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=60
            )
            # The LLM gives us a clean standalone query like "What features are in the Enterprise plan?"
            search_query = condense_completion.choices[0].message.content.strip().replace('"', '')
            print(f"\n[QUERY REWRITER] Translated: '{current_query}'  =>  '{search_query}'")
        except Exception:
            pass # Fallback to original query
            
    # 1. Route the query (using the original question)
    model_name = route_query(current_query)
    
    # 2. Retrieve Context (RAG) using the newly CONDENSED query!
    context, metadata = retrieve_context(search_query)
    chunks_retrieved = len(metadata) if metadata else 0
    
    # --- DEBUG: Print Retrieved Chunks to Terminal ---
    print(f"\n{'='*60}")
    print(f"FAILED RETRIEVAL TEST PROOF")
    print(f"{'='*60}")
    print(f"QUERY: '{current_query}'")
    print(f"DOCUMENTS RETRIEVED (Top {chunks_retrieved}):\n")
    
    if metadata:
        # We split the raw context string by our custom delimiter to get the individual chunks back
        raw_chunks = context.split("\n\n---\n\n")
        
        # We process the top retrieved documents to print cleanly
        for i, meta in enumerate(metadata):
            source_file = meta.get('source', 'Unknown File')
            
            # Grab the actual chunk text, strip the "[Source: x]" header from it, and clean newlines
            chunk_text = raw_chunks[i].replace(f"[Source: {source_file}]\n", "").replace("\n", " ")
            # Truncate to 200 characters for a clean terminal preview
            preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            
            print(f"  {i+1}. {source_file}")
            print(f'     => "{preview}"\n')
            
    print(f"{'='*60}\n")
    
    # 3. Build Prompt
    system_prompt = {
        "role": "system",
        "content": f"""
        You are the Clearpath Customer Support AI.
        
        Use the Context below AND the Conversation History to answer questions. 
        If a user asks a factual question about Clearpath and the answer is NOT in the Context or History, you must say exactly: "I don't have enough information to answer that."
        
        Example 1 (Missing Fact):
        User: What is the cost of the Enterprise plan?
        Context: [No pricing info]
        You: I don't have enough information to answer that.
        
        Example 2 (Conversational Ping):
        User: Are you sure this is correct?
        Context: []
        You: Yes, I am sure!
        
        Context:
        {context}
        """
    }
    
    # 4. Compile the full conversation history
    # We append the system prompt at the very beginning of the history array
    full_conversation_history = [system_prompt]
    for msg in request.messages:
        full_conversation_history.append({"role": msg.role, "content": msg.content})
    
    # 5. Call LLM with the full history and STREAM it
    try:
        def generate():
            chat_completion = client.chat.completions.create(
                messages=full_conversation_history,
                model=model_name,
                temperature=0.2, # Low temp for more factual answers
                stream=True
            )
            
            full_response = ""
            for chunk in chat_completion:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    # Send each token instantly via SSE
                    yield f"data: {json.dumps({'content': content})}\n\n"
                    
            # --- THE TRAILING PAYLOAD ---
            # The LLM is done streaming. We now have the complete sentence in memory.
            # We can safely run our Evaluator function on the full response!
            is_complex_route = (model_name == 'llama-3.3-70b-versatile')
            eval_flags = evaluate_output(current_query, full_response, chunks_retrieved, context, is_complex_route)
            
            latency = round((time.time() - start_time) * 1000, 2)
            
            # Send the secret trailing JSON payload to trigger the UI flag!
            debug_info = {
                "evaluator_flags": eval_flags,
                "model_used": model_name,
                "latency_ms": latency
            }
            yield f"data: {json.dumps({'trailing_eval': debug_info})}\n\n"
            
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        return {"error": str(e)}
