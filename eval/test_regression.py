"""
pytest-compatible regression test suite.
Run with: pytest eval/ -v

Requires baselines to exist — run `python harness.py baseline` first.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import load_corpus, embed_text, load_baseline, baseline_exists
from scorer import score_result, DEFAULT_THRESHOLD

corpus = load_corpus()

# Dynamically generate one test per prompt in the corpus
@pytest.mark.parametrize("entry", corpus, ids=[e["id"] for e in corpus])
def test_prompt_regression(entry):
    prompt_id = entry["id"]

    if not baseline_exists(prompt_id):
        pytest.skip(f"No baseline found for {prompt_id} — run `python harness.py baseline` first")

    baseline = load_baseline(prompt_id)
    
    # Import here to avoid Ollama connection at collection time
    from harness import run_prompt
    current_output = run_prompt(entry["prompt"])
    current_embedding = embed_text(current_output)

    result = score_result(prompt_id, baseline, current_embedding, threshold=DEFAULT_THRESHOLD)

    assert result["passed"], (
        f"Regression detected in {prompt_id}: "
        f"similarity={result['similarity']} "
        f"(threshold={DEFAULT_THRESHOLD}, delta={result['delta']})"
    )