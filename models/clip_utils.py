from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils.image_utils import PadInfo, clamp_int, pad_to_square


def load_clip(model_id: str, device: torch.device) -> Tuple[CLIPModel, CLIPProcessor]:
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval().to(device)
    return model, processor


@torch.no_grad()
def localize_with_clip_patch_similarity(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image: Image.Image,
    text: str,
    device: torch.device,
) -> Tuple[Tuple[int, int], float]:
    """Step 2: CLIP 引导目标定位。

    - 将图像 pad 成正方形后 resize 到 CLIP 视觉编码器的输入尺寸（ViT-B/32: 224）。
    - 提取 patch token（去掉 CLS）后投影到 CLIP embedding 空间。
    - 计算每个 patch 与文本 embedding 的余弦相似度。
    - 取相似度最高的 patch 中心点，映射回原图坐标，返回 (x, y)。

    Returns:
        (x, y): 原图坐标系下的点
        best_score: 该 patch 与文本的相似度
    """

    orig_w, orig_h = image.size

    padded, pad_info = pad_to_square(image)
    image_size = int(getattr(clip_model.vision_model.config, "image_size", 224))
    patch_size = int(getattr(clip_model.vision_model.config, "patch_size", 32))

    # 注意：这里 resize 会改变纵横比（因为我们先 pad 成正方形），点坐标可稳定映射回原图
    padded_resized = padded.resize((image_size, image_size), resample=Image.BICUBIC)

    image_inputs = clip_processor(images=padded_resized, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    text_features = clip_model.get_text_features(**text_inputs)
    text_features = F.normalize(text_features, dim=-1)  # (1, D)

    # vision_model 输出: last_hidden_state = (B, 1 + num_patches, hidden)
    vision_out = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
    patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # (1, N, hidden)

    # 投影到 CLIP embedding 空间
    patch_features = clip_model.visual_projection(patch_tokens)  # (1, N, D)
    patch_features = F.normalize(patch_features, dim=-1)

    # 余弦相似度（归一化后点积）
    sims = torch.matmul(patch_features, text_features[:, :, None]).squeeze(-1).squeeze(0)  # (N,)

    best_idx = int(torch.argmax(sims).item())
    best_score = float(sims[best_idx].item())

    n_patches = sims.numel()
    grid = int(math.sqrt(n_patches))
    if grid * grid != n_patches:
        raise ValueError(f"Unexpected number of patches: {n_patches}")

    row = best_idx // grid
    col = best_idx % grid

    # patch 中心点（在 224x224 坐标系）
    center_x_resized = (col + 0.5) * patch_size
    center_y_resized = (row + 0.5) * patch_size

    # 映射到 padded 原尺寸坐标系
    scale = pad_info.size / float(image_size)
    x_padded = center_x_resized * scale
    y_padded = center_y_resized * scale

    # 去除 padding 映射回原图坐标
    x = int(round(x_padded - pad_info.left))
    y = int(round(y_padded - pad_info.top))

    x = clamp_int(x, 0, orig_w - 1)
    y = clamp_int(y, 0, orig_h - 1)

    return (x, y), best_score
