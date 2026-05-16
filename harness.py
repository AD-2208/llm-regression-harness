"""
LLM Regression Harness — CLI entrypoint

Usage:
    python harness.py baseline --model gpt-oss:20b-cloud
    python harness.py check --model gpt-oss:20b-cloud
    python harness.py check --model gpt-oss:20b-cloud --threshold 0.80
"""

import argparse
import json
import os
from ollama import Client
import numpy as np

from embedder import load_corpus, embed_text, save_baseline, load_baseline, baseline_exists
from scorer import score_result, summarise_results, DEFAULT_THRESHOLD
from reporter import save_json_report, save_markdown_report, print_summary

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "gpt-oss:20b-cloud"
TEMPERATURE = 0.0  # Always 0 for deterministic baselines

def get_client() -> Client:
    return Client(host=OLLAMA_HOST)

def run_prompt(prompt: str, model: str = DEFAULT_MODEL) -> str:
    client = get_client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE}
    )
    return response["message"]["content"].strip()

def cmd_baseline(args):
    corpus = load_corpus()
    model = args.model
    force = args.force

    print(f"\nCapturing baselines — model: {model}, temperature: {TEMPERATURE}")
    print(f"Corpus size: {len(corpus)} prompts\n")

    skipped = 0
    captured = 0

    for entry in corpus:
        prompt_id = entry["id"]

        if baseline_exists(prompt_id) and not force:
            print(f"  SKIP  {prompt_id} (baseline exists — use --force to overwrite)")
            skipped += 1
            continue

        print(f"  RUN   {prompt_id} ...", end=" ", flush=True)
        outputs = [run_prompt(entry["prompt"], model=model) for _ in range(3)]
        embeddings = [embed_text(output) for output in outputs]
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        save_baseline(prompt_id, avg_embedding)
        print("✓")
        captured += 1

    print(f"\nDone. Captured: {captured}, Skipped: {skipped}")
    print("Baselines saved to baselines/")
    print("Commit baselines/ to git so CI can use them.\n")

def cmd_check(args):
    corpus = load_corpus()
    model = args.model
    threshold = args.threshold

    print(f"\nRunning regression check — model: {model}, threshold: {threshold}")
    print(f"Corpus size: {len(corpus)} prompts\n")

    results = []
    missing = []

    for entry in corpus:
        prompt_id = entry["id"]

        if not baseline_exists(prompt_id):
            print(f"  SKIP  {prompt_id} (no baseline)")
            missing.append(prompt_id)
            continue

        print(f"  CHECK {prompt_id} ...", end=" ", flush=True)
        baseline = load_baseline(prompt_id)
        output = run_prompt(entry["prompt"], model=model)
        current_embedding = embed_text(output)
        result = score_result(prompt_id, baseline, current_embedding, threshold=threshold)
        results.append(result)

        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{status}  similarity={result['similarity']}")

    if not results:
        print("\nNo results — run baseline first: python harness.py baseline")
        return

    summary = summarise_results(results)
    print_summary(summary, results)

    save_json_report(results, summary, model)
    save_markdown_report(results, summary, model)

    if missing:
        print(f"Warning: {len(missing)} prompts skipped (no baseline): {missing}\n")

    # Exit code 1 if any regressions — important for CI
    if summary["failed"] > 0:
        exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="LLM Regression Harness — detect semantic drift across model versions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Capture baseline embeddings")
    baseline_parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    baseline_parser.add_argument("--force", action="store_true", help="Overwrite existing baselines")
    baseline_parser.set_defaults(func=cmd_baseline)

    # check command
    check_parser = subparsers.add_parser("check", help="Run regression check against baselines")
    check_parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    check_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold (default: 0.80)")
    check_parser.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()