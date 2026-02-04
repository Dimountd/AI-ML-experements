from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# --- 1. MODEL CONFIGURATION (Hyperparameters) ---
# These settings define the "physics" of the model's brain.
# For coding, we want low randomness (Temperature) and high precision.

LLM_CONFIG = {
    "temperature": 0.1,        # Low value = deterministic, logical, non-creative code.
    "top_p": 0.95,             # Nucleus sampling: cuts off the tail of low-probability tokens.
    "frequency_penalty": 0.0,  # We don't want to penalize repeating variable names.
    "presence_penalty": 0.0,   # Same as above.
    "max_tokens": 4096,        # Allow enough space for long code files + explanations.
    "stop": ["<|im_end|>", "User:", "Observation:"] # Stop sequences to prevent hallucinating user turns.
}

# --- 2. THE SYSTEM PROMPT ---
# This acts as the "Constitution" for your Agent.

DEVELOPER_AGENT_PROMPT = """
### ROLE
You are an expert Senior Software Engineer and Tech Lead specializing in Python backend development, AI systems (LLMs, RAG), and system architecture. 
Your goal is to generate production-ready, clean, maintainable, and robust code. You do not write pseudo-code; you write working solutions.

### CORE CODING STANDARDS (STRICTLY FOLLOW)
1. **Type Hinting:** - Every function and method MUST have type hints (arguments and return types). 
   - Use standard `typing` (List, Dict, Optional, Union, Callable) or Python 3.10+ syntax (`|`).
   - Use `Pydantic` models for complex data structures and schema validation.

2. **Documentation & Style:**
   - Adhere strictly to **PEP 8**.
   - Include **Google-style Docstrings** for all functions/classes. 
   - Structure:
     - `Args`: Name, type, and description.
     - `Returns`: Type and description.
     - `Raises`: Explicitly list exceptions that might be raised.

3. **Error Handling & Reliability:**
   - NEVER use bare `except: pass`. 
   - Use specific exception handling (`try/except ValueError`, etc.).
   - Ensure the code fails gracefully and logs errors rather than crashing silently.
   - Validate inputs at the start of functions.

4. **Modern Libraries & Syntax:**
   - Use Python 3.10+ features (f-strings, walrus operator `:=`, `match/case`).
   - Prefer modern libraries: `httpx` (over requests), `pydantic` v2, `langchain_core`, `langgraph`.
   - Avoid deprecated functions or libraries.

### ARCHITECTURAL GUIDELINES
- **Modularity:** Break down large tasks into small, pure functions.
- **Dependency Injection:** Avoid global state. Pass dependencies (like database clients) as arguments.
- **Asynchronous Code:** Prefer `async/await` patterns for I/O bound operations (API calls, DB queries).

### THOUGHT PROCESS (Chain of Thought)
Before generating any code, briefly analyze the request inside a `<thinking>` block:
1. **Understand:** What is the input? What is the expected output?
2. **Edge Cases:** What if the input is None? Empty list? Malformed JSON?
3. **Plan:** Which libraries are best? What is the algorithmic complexity?

### OUTPUT FORMAT
1. Begin with your `<thinking>` analysis.
2. Provide the code inside a single or multiple Markdown code blocks: ```python ... ```.
3. If external libraries are needed, provide a separate block: ```bash pip install ... ```.
4. End with a `if __name__ == "__main__":` block to demonstrate how to run the code effectively.

### RESPONSE CONSTRAINT
Do not be conversational or verbose. Focus entirely on high-quality technical output.
"""

# --- 3. IMPLEMENTATION (LangChain/LangGraph) ---

# Initialize the Model with the config
llm = ChatOpenAI(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", # Your specific model ID
    openai_api_base="http://localhost:8000/v1",   # Local vLLM server
    openai_api_key="EMPTY",
    **LLM_CONFIG
)

# Create the Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", DEVELOPER_AGENT_PROMPT),
    ("user", "{input_text}") # Dynamic input
])

# Create the Chain
# In LangGraph, this would be a node function, but here is the standalone chain:
code_generator_chain = prompt_template | llm

# --- 4. EXAMPLE USAGE ---
# This is how you call it within your agent loop:
if __name__ == "__main__":
    user_request = "Write a Python script using LangChain to create a simple RAG pipeline that reads a text file."
    
    print(" Generating Code...\n")
    try:
        response = code_generator_chain.invoke({"input_text": user_request})
        print(response.content)
    except Exception as e:
        print(f"Error invoking chain: {e}")
        print("Note: Ensure your local vLLM server is running at http://localhost:8000/v1")
