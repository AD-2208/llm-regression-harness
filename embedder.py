from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASELINES_DIR = "baselines"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_text(text: str) -> np.ndarray:
    return get_model().encode(text, normalize_embeddings=True)

def embed_text_stable(text: str, runs: int = 3) -> np.ndarray:
    """
    Run embedding multiple times and average — produces a more stable
    baseline that accounts for minor tokenisation variance.
    """
    embeddings = [get_model().encode(text, normalize_embeddings=True) for _ in range(runs)]
    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)
    return avg

def save_baseline(prompt_id: str, embedding: np.ndarray):
    os.makedirs(BASELINES_DIR, exist_ok=True)
    path = os.path.join(BASELINES_DIR, f"{prompt_id}.npy")
    np.save(path, embedding)

def load_baseline(prompt_id: str) -> np.ndarray | None:
    path = os.path.join(BASELINES_DIR, f"{prompt_id}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)

def baseline_exists(prompt_id: str) -> bool:
    return os.path.exists(os.path.join(BASELINES_DIR, f"{prompt_id}.npy"))

def load_corpus(corpus_path: str = "eval/test_corpus.json") -> list[dict]:
    with open(corpus_path, "r") as f:
        return json.load(f)