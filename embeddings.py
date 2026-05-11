"""
GloVe 6B 100d embedding utilities: download, parse, and build Keras embedding matrix.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Dict
from urllib.request import urlretrieve

import numpy as np

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = "glove"
GLOVE_ZIP = os.path.join(GLOVE_DIR, "glove.6B.zip")
GLOVE_TXT = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")


def download_glove() -> None:
    """
    Download GloVe 6B (includes 100d vectors) into glove/, unzip, skip if vectors exist.
    """
    os.makedirs(GLOVE_DIR, exist_ok=True)
    if os.path.isfile(GLOVE_TXT):
        print(f"GloVe file already present: {GLOVE_TXT}")
        return

    if not os.path.isfile(GLOVE_ZIP):
        print(f"Downloading {GLOVE_URL} ...")
        urlretrieve(GLOVE_URL, GLOVE_ZIP)
    else:
        print(f"Using existing zip: {GLOVE_ZIP}")

    print("Extracting zip ...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as zf:
        zf.extractall(GLOVE_DIR)

    if not os.path.isfile(GLOVE_TXT):
        raise FileNotFoundError(f"Expected after unzip: {GLOVE_TXT}")


def load_glove_embeddings(glove_path: str = GLOVE_TXT) -> Dict[str, np.ndarray]:
    """
    Parse a GloVe .txt file into word -> float32 vector.
    """
    embeddings: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            embeddings[word] = vec
    return embeddings


def _word_index_from_tokenizer_json(tokenizer_json_path: str) -> Dict[str, int]:
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok_json = json.load(f)
    word_index = tok_json.get("config", {}).get("word_index", {})
    if isinstance(word_index, str):
        word_index = json.loads(word_index)
    if not isinstance(word_index, dict):
        raise ValueError("tokenizer.json missing config.word_index")
    # JSON keys are strings; values may be int
    out: Dict[str, int] = {}
    for w, i in word_index.items():
        out[str(w)] = int(i)
    return out


def build_embedding_matrix(
    tokenizer_json_path: str = "processed/tokenizer.json",
    glove_path: str = GLOVE_TXT,
    vocab_size: int = 20000,
    embed_dim: int = 100,
) -> np.ndarray:
    """
    Build (vocab_size, embed_dim) matrix aligned with Keras Embedding indices.
    Row i is the GloVe vector for tokenizer index i; OOV rows stay zero.
    Saves processed/embedding_matrix.npy and returns (matrix, glove_hits, words_in_vocab_range).
    """
    glove = load_glove_embeddings(glove_path)
    word_index = _word_index_from_tokenizer_json(tokenizer_json_path)

    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    hits = 0
    in_vocab_range = 0
    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue
        in_vocab_range += 1
        vec = glove.get(word)
        if vec is None:
            continue
        if vec.shape[0] != embed_dim:
            continue
        matrix[idx] = vec
        hits += 1

    os.makedirs("processed", exist_ok=True)
    out_path = os.path.join("processed", "embedding_matrix.npy")
    np.save(out_path, matrix)
    return matrix, hits, in_vocab_range


if __name__ == "__main__":
    download_glove()
    matrix, hits, in_vocab_range = build_embedding_matrix()
    oov_assignments = in_vocab_range - hits
    covered_rows = int(np.sum(np.any(matrix != 0, axis=1)))
    print(
        f"Tokenizer words with index < vocab_size (20000): {in_vocab_range}\n"
        f"Matched to GloVe (non-OOV): {hits}\n"
        f"OOV (no GloVe vector for that word): {oov_assignments}\n"
        f"Matrix shape: {matrix.shape}; non-zero rows: {covered_rows}\n"
        f"Saved: processed/embedding_matrix.npy"
    )
