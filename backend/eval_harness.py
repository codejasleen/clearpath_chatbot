import requests
import json
import time

URL = "http://127.0.0.1:8000/chat"

# The rubric asks us to write our own set of test queries with expected answers
# We will test our specific architecture edge-cases (RAG Lookup, Missing Facts, Conversational Pings)
test_cases = [
    {
        "name": "Standard RAG Lookup",
        "messages": [{"role": "user", "content": "What integrations are supported in the Pro plan?"}],
        "should_refuse": False,
        "must_contain": ["Slack"]
    },
    {
        "name": "Out of Bounds Question (Missing Fact)",
        "messages": [{"role": "user", "content": "Who is the CEO of Clearpath?"}],
        "should_refuse": True,
        "must_contain": ["I don't have enough information"]
    },
    {
        "name": "Query Condensation (Pronoun Resolution)",
        "messages": [
            {"role": "user", "content": "What features are in the Enterprise plan?"},
            {"role": "assistant", "content": "The Enterprise plan includes SSO and dedicated support."},
            {"role": "user", "content": "Does it have custom workflows?"}
        ],
        "should_refuse": False,
        "must_contain": ["workflow"]
    },
    {
        "name": "Complex Query Routing",
        "messages": [{"role": "user", "content": "Compare the features of the Pro and Enterprise plans in detail."}],
        "should_refuse": False,
        "must_contain": [] # Just making sure it successfully completes a long query without hallucinating a refusal
    }
]

def run_eval():
    passed = 0
    print(f"{'='*50}")
    print("Starting Clearpath AI Evaluation Harness...")
    print(f"{'='*50}\n")
    
    for idx, test in enumerate(test_cases):
        print(f"Test {idx+1}/{len(test_cases)}: {test['name']}")
        
        try:
            # We must use stream=True because our endpoint is SSE
            response = requests.post(URL, json={"messages": test["messages"]}, stream=True)
            bot_text = ""
            flags = []
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line.replace("data: ", "")
                        try:
                            parsed = json.loads(data_str)
                            if "content" in parsed:
                                bot_text += parsed["content"]
                            if "trailing_eval" in parsed:
                                if parsed["trailing_eval"]["evaluator_flags"]:
                                    flags.extend(parsed["trailing_eval"]["evaluator_flags"])
                        except json.JSONDecodeError:
                            pass
                            
            refused = len(flags) > 0
            
            # --- Grading Logic ---
            success = True
            
            # 1. Did it trigger the Evaluator correctly?
            if test["should_refuse"] and not refused:
                print("  [FAIL] Expected the Evaluator to flag a refusal, but it missed it.")
                success = False
            elif not test["should_refuse"] and refused:
                print(f"  [FAIL] Got an unexpected yellow refusal flag. Flags: {flags}")
                success = False
                
            # 2. Did the text contain the expected answer substring?
            for string in test.get("must_contain", []):
                if string.lower() not in bot_text.lower():
                    print(f"  [FAIL] Missing expected string '{string}' in output.")
                    success = False
            
            if success:
                print("  [PASS] All conditions met.")
                passed += 1
            else:
                print(f"  => Bot Output Preview: {bot_text[:150]}...\n")
                
        except Exception as e:
            print(f"  [FAIL] Connection Exception: {str(e)}. Is the backend server running?")
            
    print(f"\n{'-'*50}")
    print(f"Final Score: {passed}/{len(test_cases)} Tests Passed")
    print(f"{'-'*50}\n")

if __name__ == "__main__":
    run_eval()
