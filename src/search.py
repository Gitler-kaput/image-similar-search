from __future__ import annotations
import argparse
import os
import numpy as np
from PIL import Image
from .utils import load_json, save_json, safe_open_image
from .embedder import make_embedder
from .indexer import load_or_build_index, search as index_search

def make_montage(query_path: str, hits: list, out_path: str, thumb: int = 220, cols: int = 6):
    imgs = [safe_open_image(query_path)] + [safe_open_image(p) for p, _ in hits]
    thumbs = []
    for im in imgs:
        im = im.copy()
        im.thumbnail((thumb, thumb))
        thumbs.append(im)
    rows = (len(thumbs) + cols - 1) // cols
    w = cols * thumb
    h = rows * thumb
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    for i, im in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x, y = c * thumb, r * thumb
        canvas.paste(im, (x, y))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--artifacts_dir", required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--encoder", default=None, choices=[None, "clip", "resnet"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    embs = np.load(os.path.join(args.artifacts_dir, "embeddings.npy"))
    paths = load_json(os.path.join(args.artifacts_dir, "paths.json"))
    meta = load_json(os.path.join(args.artifacts_dir, "meta.json"))

    encoder = args.encoder or meta.get("encoder", "clip")
    image_size = int(meta.get("image_size", 224))

    embedder = make_embedder(encoder, device=args.device, image_size=image_size)
    q_emb = embedder.encode_paths([args.query], batch_size=1)[0]

    kind, index = load_or_build_index(embs, args.artifacts_dir)
    idx, sims = index_search(kind, index, q_emb, top_k=args.top_k)

    hits = []
    for i, s in zip(idx, sims):
        hits.append({"path": paths[i], "similarity": float(s)})

    os.makedirs(args.out_dir, exist_ok=True)
    save_json({"query": args.query, "top_k": hits, "index_kind": kind}, os.path.join(args.out_dir, "topk.json"))

    top_hits = [(h["path"], h["similarity"]) for h in hits]
    make_montage(args.query, top_hits, os.path.join(args.out_dir, "montage.jpg"))

    print("Saved:", os.path.join(args.out_dir, "topk.json"))
    print("Saved:", os.path.join(args.out_dir, "montage.jpg"))

if __name__ == "__main__":
    main()
