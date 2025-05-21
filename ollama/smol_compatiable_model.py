import requests
from typing import List, Dict

from openai import OpenAI

from .__init__ import MODEL_NAME_QWEN_OLLAMA

class OllamaQwen7BClientModel:
    def __init__(
        self,
        model_id: str = 'Qwen/Qwen2.5-Coder-32B-Instruct',
        model_id_ollama_openai: str = MODEL_NAME_QWEN_OLLAMA,
        api_base: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        num_ctx: int = 2048,
    ):
        self.model_id_ollama_openai = model_id_ollama_openai
        self.model_id = model_id
        self.api_base = api_base
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.client = OpenAI(
            base_url=api_base,
            api_key='something-random-does-not-matter',
        )

    def generate(self, messages: List[Dict[str, str]], stream: bool = False,  **kwargs) -> str:

        allowed_kwargs = dict()
        if kwargs.get("stop_sequences"):
            allowed_kwargs["stop"] = kwargs.get("stop_sequences")

        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_id_ollama_openai,
            stream=stream,
        )

        return chat_completion.choices[0].message
