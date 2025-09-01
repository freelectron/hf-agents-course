from llama_index.llms.ollama import Ollama

REQUEST_TIMEOUT = 180
MODEL_NAME = "qwen2.5:14b"

# Initialize Ollama LLM
llm = Ollama(
    model=MODEL_NAME, 
    request_timeout=REQUEST_TIMEOUT
)