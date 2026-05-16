# llm-regression-harness

> Catches prompt regressions before they reach production — semantic drift scoring
> across model versions with CI/CD integration.

![Tests](https://github.com/AD-2208/llm-regression-harness/actions/workflows/eval.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The Problem

When you change a prompt, swap a model, or update a system instruction, how do you
know you haven't broken something? Most teams find out from users. This harness
catches regressions automatically — before they ship.

---

## How It Works

1. **Baseline capture** — Run your prompt suite against a model and embed every
   output using `sentence-transformers`. Store embeddings as the baseline.
2. **Regression detection** — After any change (prompt edit, model swap, temperature
   tweak), re-run the suite and compute cosine similarity between new outputs and
   the baseline.
3. **Threshold flagging** — Any output below the similarity threshold (default: 0.85)
   is flagged as a regression with a diff-style report.
4. **CI integration** — A GitHub Actions workflow runs the full suite on every push
   and fails the build if regressions are detected.

   prompt → model → embed → compare to baseline → pass / FAIL

---

## Results

| Model | Prompts Tested | Regressions Detected | Avg Similarity | CI Status |
|-------|---------------|---------------------|----------------|-----------|
| mistral (v0.1 → v0.3) | 30 | TBD | TBD | — |
| llama3 (temp 0.0 → 0.7) | 30 | TBD | TBD | — |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/llm-regression-harness
cd llm-regression-harness
pip install -r requirements.txt

# Pull a local model (requires Ollama)
ollama pull mistral

# Capture baseline
python harness.py baseline --model mistral

# Run regression check
python harness.py check --model mistral

# Or run via pytest
pytest eval/
```

---

## Project Structure
llm-regression-harness/
├── eval/
│   ├── test_corpus.json        # Prompt/expected-output pairs
│   └── test_regression.py      # pytest-compatible regression tests
├── harness.py                  # CLI entrypoint (baseline + check)
├── embedder.py                 # Sentence-transformer embedding logic
├── scorer.py                   # Cosine similarity + threshold logic
├── reporter.py                 # JSON + markdown report generation
├── baselines/                  # Stored baseline embeddings (git-tracked)
├── reports/                    # Generated regression reports
├── .github/
│   └── workflows/
│       └── eval.yml            # CI pipeline
├── requirements.txt
└── README.md

---

## Adding Your Own Test Cases

Edit `eval/test_corpus.json`:

```json
[
  {
    "id": "summarise_001",
    "prompt": "Summarise the following in one sentence: ...",
    "tags": ["summarisation", "brevity"],
    "notes": "Should be factual and under 30 words"
  }
]
```

No expected output string required — the harness compares against the baseline
embedding, so it detects *semantic* drift, not exact string changes.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | `0.85` | Below this = regression flagged |
| `model` | `mistral` | Any Ollama-compatible model |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `temperature` | `0.0` | LLM temperature (0 for determinism) |

---

## CI/CD Integration

On every push, GitHub Actions:
1. Pulls baseline embeddings from the repo
2. Runs the full prompt suite against the configured model
3. Computes similarity scores
4. Fails the build if any prompt scores below threshold
5. Uploads the regression report as a build artifact

---

## Motivation

Built as part of a portfolio demonstrating production-grade LLM evaluation practices.
Most LLM projects show that a model *can* do something. This one measures whether it
*keeps* doing it correctly across changes.

---

## Roadmap

- [ ] Multi-model comparison (run same suite across 3 models simultaneously)
- [ ] LangSmith integration for trace-level debugging of regressions
- [ ] Threshold auto-tuning based on historical variance
- [ ] HTML report with side-by-side output comparison

---

## License

MIT
