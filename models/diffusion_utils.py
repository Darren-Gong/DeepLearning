from __future__ import annotations

from typing import Optional

import torch
from PIL import Image


def load_inpaint_pipeline(
    model_id_or_path: str,
    device: torch.device,
):
    """Step 1: 加载 Stable Diffusion Inpainting pipeline。"""

    from diffusers import StableDiffusionInpaintPipeline

    use_fp16 = device.type == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    # 可选：如果环境装了 xformers，会显著省显存
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe.set_progress_bar_config(disable=False)

    return pipe


@torch.no_grad()
def inpaint(
    pipe,
    image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    seed: Optional[int] = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """Step 4: Stable Diffusion Inpainting。

    注意：mask_image 需为单通道 'L'，白色(255)区域表示要重绘。
    """

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    # autocast 仅在 CUDA+FP16 时启用
    use_autocast = pipe.device.type == "cuda" and getattr(pipe, "dtype", torch.float32) == torch.float16

    if use_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
    else:
        out = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    return out.images[0]
