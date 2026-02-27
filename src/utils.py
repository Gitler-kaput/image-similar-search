import os
import json
from typing import List
from PIL import Image, ImageOps

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def list_images(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths

def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img

def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
