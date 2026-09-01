import os
from .base import LLMProvider

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.0):
        super().__init__(model, temperature)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Get a key at https://console.anthropic.com/settings/keys "
                "and set it with: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self.client = Anthropic(api_key=api_key)

    def run(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()