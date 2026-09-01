from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Common interface every provider must implement. This is what makes
    the harness provider-agnostic — swapping providers never touches
    harness.py, embedder.py, or scorer.py.
    """

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def run(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the text response.
        Must be deterministic at temperature=0 where the provider supports it.
        """
        raise NotImplementedError