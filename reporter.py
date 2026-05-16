import json
import os
from datetime import datetime

REPORTS_DIR = "reports"

def save_json_report(results: list[dict], summary: dict, model: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"regression_report_{timestamp}.json"
    path = os.path.join(REPORTS_DIR, filename)
    report = {
        "timestamp": timestamp,
        "model": model,
        "summary": summary,
        "results": results
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved: {path}")
    return path

def save_markdown_report(results: list[dict], summary: dict, model: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"regression_report_{timestamp}.md"
    path = os.path.join(REPORTS_DIR, filename)

    lines = []
    lines.append(f"# Regression Report")
    lines.append(f"\n**Model:** `{model}`  ")
    lines.append(f"**Timestamp:** {timestamp}  ")
    lines.append(f"**Pass rate:** {summary['pass_rate']*100:.1f}%  ")
    lines.append(f"**Avg similarity:** {summary['avg_similarity']}  ")
    lines.append(f"**Passed:** {summary['passed']} / {summary['total']}  ")
    lines.append(f"\n---\n")

    # Failed prompts first
    failed = [r for r in results if not r["passed"]]
    passed = [r for r in results if r["passed"]]

    if failed:
        lines.append(f"## ❌ Regressions Detected ({len(failed)})\n")
        lines.append("| ID | Similarity | Threshold | Delta |")
        lines.append("|---|---|---|---|")
        for r in failed:
            lines.append(f"| `{r['id']}` | {r['similarity']} | {r['threshold']} | {r['delta']} |")
        lines.append("")

    lines.append(f"## ✅ Passed ({len(passed)})\n")
    lines.append("| ID | Similarity | Threshold | Delta |")
    lines.append("|---|---|---|---|")
    for r in passed:
        lines.append(f"| `{r['id']}` | {r['similarity']} | {r['threshold']} | {r['delta']} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report saved: {path}")
    return path

def print_summary(summary: dict, results: list[dict]):
    print("\n" + "="*50)
    print("REGRESSION CHECK SUMMARY")
    print("="*50)
    print(f"Total prompts : {summary['total']}")
    print(f"Passed        : {summary['passed']}")
    print(f"Failed        : {summary['failed']}")
    print(f"Pass rate     : {summary['pass_rate']*100:.1f}%")
    print(f"Avg similarity: {summary['avg_similarity']}")
    print(f"Min similarity: {summary['min_similarity']}")
    print("="*50)

    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n❌ REGRESSIONS ({len(failed)}):")
        for r in failed:
            print(f"  {r['id']:<20} similarity={r['similarity']}  delta={r['delta']}")
    else:
        print("\n✅ No regressions detected.")
    print()