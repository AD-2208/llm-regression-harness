from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider


def get_provider(name: str, model: str, temperature: float = 0.0):
    """
    Factory function — returns the correct provider instance by name.
    """
    if name == "ollama":
        return OllamaProvider(model=model, temperature=temperature)
    elif name == "openai":
        return OpenAIProvider(model=model, temperature=temperature)
    elif name == "anthropic":
        return AnthropicProvider(model=model, temperature=temperature)
    else:
        raise ValueError(
            f"Unknown provider: {name}. Choose 'ollama', 'openai', or 'anthropic'."
        )