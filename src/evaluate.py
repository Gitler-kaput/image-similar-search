from __future__ import annotations
import argparse
import os
import numpy as np
from .utils import load_json
from .embedder import make_embedder
from .indexer import load_or_build_index, search as index_search
from .metrics import precision_at_k, recall_at_k, average_precision

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_dir", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    pairs = load_json(os.path.join(args.pairs_dir, "pairs.json"))
    embs = np.load(os.path.join(args.artifacts_dir, "embeddings.npy"))
    paths = load_json(os.path.join(args.artifacts_dir, "paths.json"))
    meta = load_json(os.path.join(args.artifacts_dir, "meta.json"))

    encoder = meta.get("encoder", "clip")
    image_size = int(meta.get("image_size", 224))
    embedder = make_embedder(encoder, device=args.device, image_size=image_size)

    kind, index = load_or_build_index(embs, args.artifacts_dir)

    pks, rks, aps = [], [], []
    for pair in pairs:
        q = pair["augmented"]
        target = pair["original"]

        q_emb = embedder.encode_paths([q], batch_size=1)[0]
        idx, _ = index_search(kind, index, q_emb, top_k=args.k)
        ranked_paths = [paths[i] for i in idx]
        relevant = np.array([1 if rp == target else 0 for rp in ranked_paths], dtype=int)

        pks.append(precision_at_k(relevant, args.k))
        rks.append(recall_at_k(relevant, args.k))
        aps.append(average_precision(relevant, args.k))

    print(f"Index: {kind} | Encoder: {encoder}")
    print(f"Pairs: {len(pairs)} | K={args.k}")
    print(f"Precision@K: {np.mean(pks):.4f}")
    print(f"Recall@K:    {np.mean(rks):.4f}")
    print(f"mAP@K:       {np.mean(aps):.4f}")

if __name__ == "__main__":
    main()
