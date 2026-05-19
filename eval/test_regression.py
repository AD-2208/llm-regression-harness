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

@pytest.mark.parametrize("entry", corpus, ids=[e["id"] for e in corpus])
def test_prompt_regression(entry):
    prompt_id = entry["id"]

    if not baseline_exists(prompt_id):
        pytest.skip(f"No baseline found for {prompt_id} — run `python harness.py baseline` first")

    baseline = load_baseline(prompt_id)

    from harness import run_prompt
    current_output = run_prompt(entry["prompt"])
    current_embedding = embed_text(current_output)

    # Respect per-prompt threshold override if present
    threshold = entry.get("threshold", DEFAULT_THRESHOLD)

    result = score_result(prompt_id, baseline, current_embedding, threshold=threshold)

    assert result["passed"], (
        f"Regression detected in {prompt_id}: "
        f"similarity={result['similarity']} "
        f"(threshold={threshold}, delta={result['delta']})"
    )