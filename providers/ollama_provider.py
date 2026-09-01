import os
from ollama import Client
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.client = Client(host=host)

    def run(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature}
        )
        return response["message"]["content"].strip()