from __future__ import annotations
import argparse
import os
import time
import numpy as np
from .utils import list_images, save_json
from .embedder import make_embedder
from .indexer import build_index, save_index

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--encoder", default="clip", choices=["clip", "resnet"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    paths = list_images(args.images_dir)
    if not paths:
        raise SystemExit("No images found in images_dir")

    os.makedirs(args.out_dir, exist_ok=True)
    embedder = make_embedder(args.encoder, device=args.device, image_size=args.image_size)

    t0 = time.time()
    embs = embedder.encode_paths(paths, batch_size=args.batch_size)
    dt = time.time() - t0

    np.save(os.path.join(args.out_dir, "embeddings.npy"), embs)
    save_json(paths, os.path.join(args.out_dir, "paths.json"))

    kind, index = build_index(embs)
    save_index(kind, index, args.out_dir)

    meta = {
        "encoder": args.encoder,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "index_kind": kind,
        "n_images": len(paths),
        "dim": int(embs.shape[1]),
        "build_seconds": round(dt, 3),
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))
    print(f"Done: {len(paths)} images, dim={embs.shape[1]}, index={kind}, time={dt:.2f}s")

if __name__ == "__main__":
    main()
