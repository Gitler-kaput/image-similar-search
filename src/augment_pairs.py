from __future__ import annotations
import argparse
import os
import random
from PIL import Image, ImageEnhance
from .utils import list_images, safe_open_image, save_json

def augment(img: Image.Image) -> Image.Image:
    img = img.copy()
    w, h = img.size
    scale = random.uniform(0.8, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    x0 = random.randint(0, max(0, w - nw))
    y0 = random.randint(0, max(0, h - nh))
    img = img.crop((x0, y0, x0 + nw, y0 + nh)).resize((w, h))
    ang = random.uniform(-10, 10)
    img = img.rotate(ang, resample=Image.BICUBIC, expand=False)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--per_image", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    paths = list_images(args.images_dir)
    if not paths:
        raise SystemExit("No images found")

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = []
    for src in paths:
        img = safe_open_image(src)
        base = os.path.splitext(os.path.basename(src))[0]
        for j in range(args.per_image):
            aug = augment(img)
            out_path = os.path.join(args.out_dir, f"{base}__aug{j}.jpg")
            aug.save(out_path, quality=random.randint(50, 90))
            pairs.append({"original": src, "augmented": out_path})

    save_json(pairs, os.path.join(args.out_dir, "pairs.json"))
    print(f"Saved {len(pairs)} pairs")

if __name__ == "__main__":
    main()
