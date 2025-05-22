# Unit 1: ReAct Framework & SmolAgents Library

## 1. Tokenization and Message Formatting

Each LLM uses its own encoding schema to convert human-written text into tokens that the model can understand.

![Tokenization Diagram](attachment:7c0b51fb-73be-4f25-8b12-8e3a5484cfc4:image.png)

Chat templates help transform a JSON list of messages (with roles like `user`, `assistant`, etc.) into a format that the LLM understands.

---

## 2. Tools and Tool Templates

Tools are callable objects (usually Python functions) that you must describe to an agent in a specific format so it can invoke them properly. There's also a standardized template for specifying a tool.

---

## 3. Agent Reasoning Loop (ReAct)

Agents use a **Think-Act-Observe** loop to interact with their environment. The LLM output is structured (e.g., JSON) to allow external parsers to extract actions and interpret responses.

### 🧩 Stop & Parse: JSON-Based Action Handling

If the LLM decides to use a tool, it might produce an output like:

![Stop & Parse JSON Example](attachment:fdb3dc9f-a3c1-475c-97ec-aa3c7573b6f0:image.png)

In this case, an external parser:
- Reads the formatted action
- Identifies which tool to call
- Extracts required parameters  
> [Reference](https://huggingface.co/learn/agents-course/unit1/actions)

**Clarity and format consistency are essential**.

---

### 🧠 Code Agents: Python-Based Action Handling

Instead of JSON, a Code Agent outputs **executable Python code blocks**.

On the left of the following diagram: Stop & Parse approach using iterative tool calls.  
On the right: Code Agent that outputs code and executes it directly.

![Stop & Parse vs Code Agent](attachment:3eb252d2-8a7d-4bc9-829b-278eb6fe0bfe:image.png)

These agents may have internal templates for parsing outputs (e.g., expected JSON or `final_answer()` calls), which tell the system how to execute code and interpret results.

---

## 4. CodeAgent Memory & Loop Termination

A `CodeAgent` always takes in the current **memory**, which is a list of tuples:

- `Thought`: reasoning or planning text  
- `Action`: the tool the agent decided to invoke  
- `Observation`: result/output from the tool

When the agent believes it has a final answer, it outputs a `final_answer()` call, which terminates the loop.

![Memory Diagram 1](attachment:4841cf9f-7b87-4acb-afaa-81616d2f1ab9:image.png)  
![Memory Diagram 2](attachment:38d44477-7b78-4445-ba6c-4921d92a9691:image.png)

---

## 5. Common Questions

### ❓ How does the raw input to the LLM (e.g., QWEN) look?

SmolAgents sends raw **string input via HTTP** to the LLM API.  
Tokenization and special transformations (e.g., start/end tokens) happen inside the LLM backend, often using frameworks like HuggingFace Transformers.

---

### ❓ How are tool calls executed?

Look at the `_step_stream()` function in `smolagents/agents.py`.

At each step:
1. Memory is serialized via `self.write_memory_to_messages()`
2. Full context is sent to the LLM
3. LLM outputs a Python code block, often invoking a tool or `final_answer()`

Example system prompt from context:
```json
"system": {
  "content": [
    {
      "type": "text",
      "text": "You are an expert assistant who can solve any task using code blobs..."
    }
  ]
}
