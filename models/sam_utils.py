from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from utils.image_utils import pil_to_numpy_rgb


def load_sam_predictor(
    sam_ckpt_path: str,
    device: torch.device,
    model_type: str = "vit_b",
):
    """Step 1: 加载 SAM (ViT-B) predictor。

    依赖：segment-anything 官方仓库。
    """

    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=sam_ckpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


@torch.no_grad()
def predict_mask_from_point(
    predictor,
    image: Image.Image,
    point_xy: Tuple[int, int],
) -> np.ndarray:
    """Step 3: SAM 自动分割。

    输入：原图 + (x, y) 点提示
    输出：二值 mask（HxW，dtype=bool）
    """

    image_np = pil_to_numpy_rgb(image)
    predictor.set_image(image_np)

    point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best = int(np.argmax(scores))
    mask = masks[best]  # (H, W) bool

    return mask
