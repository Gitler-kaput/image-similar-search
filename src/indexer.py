from __future__ import annotations
import os
import numpy as np

def build_index(embeddings: np.ndarray):
    try:
        import faiss  # type: ignore
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        return "faiss", index
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric="cosine", algorithm="auto")
        nn.fit(embeddings.astype(np.float32))
        return "sklearn", nn

def save_index(kind: str, index, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if kind == "faiss":
        import faiss  # type: ignore
        faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    else:
        with open(os.path.join(out_dir, "index.sklearn.txt"), "w", encoding="utf-8") as f:
            f.write("sklearn index is rebuilt from embeddings.npy on load\n")

def load_or_build_index(embeddings: np.ndarray, artifacts_dir: str):
    faiss_path = os.path.join(artifacts_dir, "index.faiss")
    if os.path.exists(faiss_path):
        try:
            import faiss  # type: ignore
            return "faiss", faiss.read_index(faiss_path)
        except Exception:
            pass
    return build_index(embeddings)

def search(kind: str, index, query_emb: np.ndarray, top_k: int = 10):
    query_emb = query_emb.astype(np.float32)
    if query_emb.ndim == 1:
        query_emb = query_emb[None, :]
    if kind == "faiss":
        scores, idx = index.search(query_emb, top_k)
        return idx[0].tolist(), scores[0].tolist()
    else:
        dists, idx = index.kneighbors(query_emb, n_neighbors=top_k)
        sims = (1.0 - dists[0]).tolist()
        return idx[0].tolist(), sims
