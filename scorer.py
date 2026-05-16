import numpy as np

DEFAULT_THRESHOLD = 0.85

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two normalised embeddings.
    Both vectors should already be L2-normalised (embedder does this).
    For normalised vectors, dot product == cosine similarity.
    """
    return float(np.dot(a, b))

def score_result(prompt_id: str, baseline: np.ndarray, current: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    similarity = cosine_similarity(baseline, current)
    passed = similarity >= threshold
    return {
        "id": prompt_id,
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "passed": passed,
        "delta": round(similarity - threshold, 4)
    }

def summarise_results(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    similarities = [r["similarity"] for r in results]
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total > 0 else 0,
        "avg_similarity": round(float(np.mean(similarities)), 4),
        "min_similarity": round(float(np.min(similarities)), 4),
        "max_similarity": round(float(np.max(similarities)), 4)
    }