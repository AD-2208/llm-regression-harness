"""
pytest-compatible regression test suite.
Run with: pytest eval/ -v

Requires baselines to exist — run `python harness.py baseline` first.

Provider and model are configurable via environment variables so this
suite can run against any provider in CI without code changes:

    HARNESS_PROVIDER=ollama HARNESS_MODEL=mistral pytest eval/ -v
    HARNESS_PROVIDER=openai HARNESS_MODEL=gpt-4o-mini pytest eval/ -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import load_corpus, embed_text, load_baseline, baseline_exists
from scorer import score_result, DEFAULT_THRESHOLD
from providers import get_provider

PROVIDER_NAME = os.environ.get("HARNESS_PROVIDER", "ollama")
MODEL_NAME = os.environ.get("HARNESS_MODEL", "mistral")
TEMPERATURE = 0.0

corpus = load_corpus()


@pytest.fixture(scope="module")
def provider():
    return get_provider(PROVIDER_NAME, model=MODEL_NAME, temperature=TEMPERATURE)


@pytest.mark.parametrize("entry", corpus, ids=[e["id"] for e in corpus])
def test_prompt_regression(entry, provider):
    prompt_id = entry["id"]

    if not baseline_exists(prompt_id):
        pytest.skip(f"No baseline found for {prompt_id} — run `python harness.py baseline` first")

    baseline = load_baseline(prompt_id)

    current_output = provider.run(entry["prompt"])
    current_embedding = embed_text(current_output)

    # Respect per-prompt threshold override if present
    threshold = entry.get("threshold", DEFAULT_THRESHOLD)

    result = score_result(prompt_id, baseline, current_embedding, threshold=threshold)

    assert result["passed"], (
        f"Regression detected in {prompt_id}: "
        f"similarity={result['similarity']} "
        f"(threshold={threshold}, delta={result['delta']})"
    )