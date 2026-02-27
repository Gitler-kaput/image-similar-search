from __future__ import annotations
import numpy as np
from typing import List
from .utils import safe_open_image

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

class BaseEmbedder:
    def encode_paths(self, paths: List[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

class ClipEmbedder(BaseEmbedder):
    def __init__(self, device: str = "cpu", model_name: str = "openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPProcessor, CLIPModel
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def encode_paths(self, paths: List[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        embs = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = [safe_open_image(p) for p in batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            feats = feats.detach().cpu().numpy().astype(np.float32)
            embs.append(feats)
        embs = np.vstack(embs)
        return l2_normalize(embs)

class ResNetEmbedder(BaseEmbedder):
    def __init__(self, device: str = "cpu", image_size: int = 224):
        import torch
        import torchvision.transforms as T
        from torchvision.models import resnet50, ResNet50_Weights
        self.torch = torch
        self.device = device
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.to(device)
        self.model.eval()
        self.tf = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ])

    def encode_paths(self, paths: List[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        embs = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = [self.tf(safe_open_image(p)) for p in batch]
            x = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                feats = self.model(x)
            feats = feats.detach().cpu().numpy().astype(np.float32)
            embs.append(feats)
        embs = np.vstack(embs)
        return l2_normalize(embs)

def make_embedder(encoder: str, device: str = "cpu", image_size: int = 224) -> BaseEmbedder:
    encoder = (encoder or "").lower().strip()
    if encoder == "clip":
        return ClipEmbedder(device=device)
    if encoder == "resnet":
        return ResNetEmbedder(device=device, image_size=image_size)
    raise ValueError(f"Unknown encoder: {encoder}")
