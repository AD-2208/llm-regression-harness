# llm-regression-harness

> Catches prompt regressions before they reach production — semantic drift scoring
> across model versions with CI/CD integration.

![Tests](https://github.com/AD-2208/llm-regression-harness/actions/workflows/eval.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Demo

**Regression detected** — two prompts were semantically changed, harness catches both:
Running regression check — model: mistral, threshold: 0.8
Corpus size: 30 prompts
CHECK rsn_001 ... ✗ FAIL  similarity=0.7672  delta=-0.0328
CHECK sty_001 ... ✗ FAIL  similarity=0.1934  delta=-0.6066
REGRESSION CHECK SUMMARY
Total prompts : 30  |  Passed: 28  |  Failed: 2  |  Pass rate: 93.3%
❌ REGRESSIONS (2):
rsn_001    similarity=0.7672   delta=-0.0328
sty_001    similarity=0.1934   delta=-0.6066

**Clean run on main** — all 30 prompts pass after restoring correct prompts:
Running regression check — model: mistral, threshold: 0.8
Corpus size: 30 prompts
[all 30 prompts: ✓ PASS  similarity=1.0]
Total prompts : 30  |  Passed: 30  |  Failed: 0  |  Pass rate: 100.0%
✅ No regressions detected.

See the [live demo PR](https://github.com/AD-2208/llm-regression-harness/pull/1)
for the full diff of what changed and the CI failure it triggered.

---

## The Problem

When you change a prompt, swap a model, or update a system instruction, how do you
know you haven't broken something? Most teams find out from users.

This harness catches regressions automatically — before they ship — the same way
unit tests catch broken code.

---

## How It Works
prompt → LLM → output → sentence-transformer → embedding vector
↓
cosine similarity
↓
compare to stored baseline
↓
pass / FAIL + delta

1. **Baseline capture** — run each prompt through the LLM, embed the output using
   `sentence-transformers/all-MiniLM-L6-v2`, store the 384-dimensional vector.
   This encodes *semantic meaning*, not exact words.

2. **Regression check** — after any change (prompt edit, model swap, temperature
   tweak), re-run the suite. Compute cosine similarity between new embeddings and
   stored baselines. Similarity of 1.0 = identical meaning. 0.0 = completely
   unrelated.

3. **Threshold flagging** — anything below threshold (default: 0.80) is flagged
   as a regression with a delta score showing how far it drifted.

4. **CI integration** — GitHub Actions runs the full suite on every push and
   exits with code 1 if regressions are detected, failing the build.

---

## Results

| Scenario | Prompts | Passed | Failed | Avg Similarity | Result |
|----------|---------|--------|--------|----------------|--------|
| Baseline vs same model (mistral) | 30 | 30 | 0 | 1.0000 | ✅ |
| Demo: 2 prompts semantically changed | 30 | 28 | 2 | 0.9654 | ❌ |
| rsn_001: question reframed entirely | 1 | 0 | 1 | 0.7672 | ❌ |
| sty_001: task changed to translation | 1 | 0 | 1 | 0.1934 | ❌ |

**Key observation:** `sty_001` scored 0.1934 — near zero — because the prompt
was changed from tone rewriting to Spanish translation. The model output is in a
completely different language and domain. The harness catches this immediately.
`rsn_001` scored 0.7672 — the task is still mathematical but asks a different
question, producing a different answer. Both are caught before reaching production.

---

## Quickstart

```bash
git clone https://github.com/AD-2208/llm-regression-harness
cd llm-regression-harness
pip install -r requirements.txt

# Requires Ollama — https://ollama.com
ollama pull mistral

# Capture baseline embeddings
python harness.py baseline --model mistral

# Run regression check
python harness.py check --model mistral

# Or via pytest
pytest eval/ -v
```

---

## Project Structure
llm-regression-harness/
├── eval/
│   ├── test_corpus.json        # 30 prompts across 5 task categories
│   └── test_regression.py      # pytest-compatible regression tests
├── harness.py                  # CLI entrypoint (baseline + check)
├── embedder.py                 # Sentence-transformer embedding logic
├── scorer.py                   # Cosine similarity + threshold logic
├── reporter.py                 # JSON + markdown report generation
├── baselines/                  # Stored baseline embeddings (git-tracked)
├── reports/                    # Generated regression reports (gitignored)
├── .github/
│   └── workflows/
│       └── eval.yml            # CI pipeline
├── requirements.txt
└── README.md

---

## Test Corpus

30 prompts across 5 categories, chosen to cover distinct semantic task types:

| Category | Count | Sensitivity | Notes |
|----------|-------|-------------|-------|
| Summarisation | 6 | Medium | Tests factual retention and conciseness |
| Classification | 6 | Low | Single-label outputs — any label change is unambiguous |
| Instruction following | 6 | High | Format-sensitive — preamble or structure changes register |
| Reasoning | 6 | Medium | Includes known cognitive bias traps |
| Style rewriting | 6 | High | Most sensitive to temperature and prompt changes |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | `0.80` | Global threshold — below this = regression |
| `model` | `mistral` | Any Ollama-compatible model |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `temperature` | `0.0` | Fixed at 0 for deterministic baselines |

Per-prompt threshold overrides are supported in `test_corpus.json` via a
`"threshold"` field — used for inherently variable prompts like creative tasks.

---

## Adding Your Own Test Cases

Edit `eval/test_corpus.json`:

```json
{
  "id": "your_001",
  "category": "your_category",
  "prompt": "Your prompt here",
  "tags": ["tag1", "tag2"],
  "notes": "What the output should semantically contain"
}
```

Then recapture baselines:

```bash
python harness.py baseline --model mistral --force
```

---

## Design Decisions

**Why semantic similarity over exact string matching?**
LLMs are non-deterministic. Exact string matching would flag every run as a
regression. Semantic embeddings capture *meaning* — a model can rephrase an
answer and still pass, but a model that gives a fundamentally different answer
will fail.

**Why commit baselines to git?**
Baselines need to be stable and reproducible across environments. Storing them
in the repo means CI always runs against the same reference point. A baseline
update is a deliberate, reviewable commit — not a silent drift.

**Why per-prompt thresholds?**
Creative tasks like haiku generation are semantically variable by nature — two
valid haiku about autumn have different words but equivalent meaning. A global
threshold would produce false positives. Per-prompt overrides let you tune
sensitivity where it matters.

---

## CI/CD Integration

On every push, GitHub Actions:
1. Installs dependencies and pulls the mistral model via Ollama
2. Checks that baselines exist in the repo
3. Runs the full 30-prompt regression suite
4. Fails the build (exit code 1) if any regressions are detected
5. Uploads the regression report as a downloadable build artifact

---

## Motivation

Most LLM portfolio projects demonstrate that a model *can* do something.
This one measures whether it *keeps* doing it correctly across changes.

Prompt regression is a real production problem that most teams handle manually
or not at all. This harness treats prompt quality as a first-class engineering
concern — the same way mature software teams treat code quality.

---

## License

MIT
