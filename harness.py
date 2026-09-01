"""
LLM Regression Harness — CLI entrypoint

Usage:
    python harness.py baseline --model mistral
    python harness.py baseline --model gpt-4o-mini --provider openai
    python harness.py check --model mistral
    python harness.py check --model mistral --threshold 0.80
"""

import argparse
import os

from embedder import load_corpus, embed_text, save_baseline, load_baseline, baseline_exists
from scorer import score_result, summarise_results, DEFAULT_THRESHOLD
from reporter import save_json_report, save_markdown_report, print_summary
from providers import get_provider

DEFAULT_MODEL = "mistral"
DEFAULT_PROVIDER = "ollama"
TEMPERATURE = 0.0  # Always 0 for deterministic baselines


def cmd_baseline(args):
    corpus = load_corpus()
    provider = get_provider(args.provider, model=args.model, temperature=TEMPERATURE)
    force = args.force

    print(f"\nCapturing baselines — provider: {args.provider}, model: {args.model}, temperature: {TEMPERATURE}")
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
        output = provider.run(entry["prompt"])
        embedding = embed_text(output)
        save_baseline(prompt_id, embedding)
        print("✓")
        captured += 1

    print(f"\nDone. Captured: {captured}, Skipped: {skipped}")
    print("Baselines saved to baselines/")
    print("Commit baselines/ to git so CI can use them.\n")


def cmd_check(args):
    corpus = load_corpus()
    provider = get_provider(args.provider, model=args.model, temperature=TEMPERATURE)
    threshold = args.threshold

    print(f"\nRunning regression check — provider: {args.provider}, model: {args.model}, threshold: {threshold}")
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
        output = provider.run(entry["prompt"])
        current_embedding = embed_text(output)

        prompt_threshold = entry.get("threshold", threshold)
        result = score_result(prompt_id, baseline, current_embedding, threshold=prompt_threshold)
        results.append(result)

        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{status}  similarity={result['similarity']}")

    if not results:
        print("\nNo results — run baseline first: python harness.py baseline")
        return

    summary = summarise_results(results)
    print_summary(summary, results)

    save_json_report(results, summary, args.model)
    save_markdown_report(results, summary, args.model)

    if missing:
        print(f"Warning: {len(missing)} prompts skipped (no baseline): {missing}\n")

    if summary["failed"] > 0:
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Regression Harness — detect semantic drift across model versions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Capture baseline embeddings")
    baseline_parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    baseline_parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["ollama", "openai", "anthropic"], help="LLM provider")
    baseline_parser.add_argument("--force", action="store_true", help="Overwrite existing baselines")
    baseline_parser.set_defaults(func=cmd_baseline)

    check_parser = subparsers.add_parser("check", help="Run regression check against baselines")
    check_parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    check_parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["ollama", "openai", "anthropic"], help="LLM provider")
    check_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold (default: 0.80)")
    check_parser.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()