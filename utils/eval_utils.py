from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@torch.no_grad()
def clip_score_image_text(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image: Image.Image,
    text: str,
    device: torch.device,
) -> float:
    """Cosine similarity between CLIP image embedding and text embedding."""

    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    image_inputs = clip_processor(images=image, return_tensors="pt").to(device)

    text_features = clip_model.get_text_features(**text_inputs)
    image_features = clip_model.get_image_features(**image_inputs)

    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)

    score = (image_features * text_features).sum(dim=-1).item()
    return float(score)
