from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from models.clip_utils import load_clip, localize_with_clip_patch_similarity
from models.diffusion_utils import inpaint, load_inpaint_pipeline
from models.sam_utils import load_sam_predictor, predict_mask_from_point
from utils.eval_utils import clip_score_image_text
from utils.image_utils import ensure_dir, load_image, mask_to_pil_l, save_image, save_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-guided Image Editing: CLIP + SAM + SD Inpainting")

    parser.add_argument("--image", type=str, required=True, help="输入 RGB 图像路径")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示，如: 'a cat wearing sunglasses'")

    parser.add_argument("--output_dir", type=str, default="outputs/demo", help="输出目录")

    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP 模型 id 或本地路径",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Stable Diffusion Inpainting 模型 id 或本地路径",
    )

    parser.add_argument(
        "--sam_ckpt",
        type=str,
        required=True,
        help="SAM vit_b checkpoint 本地路径 (如 sam_vit_b_01ec64.pth)",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_b",
        choices=["vit_b"],
        help="SAM 模型类型（本实验固定使用 vit_b）",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --------------------------
    # Step 1: 加载模型 + 自动检测 GPU
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    clip_model, clip_processor = load_clip(args.clip_model, device)
    sam_predictor = load_sam_predictor(args.sam_ckpt, device=device, model_type=args.sam_model_type)
    sd_pipe = load_inpaint_pipeline(args.sd_model, device=device)

    # --------------------------
    # Step 5: 保存原始图像
    # --------------------------
    ensure_dir(args.output_dir)
    image = load_image(args.image)
    save_image(image, os.path.join(args.output_dir, "original.png"))

    # --------------------------
    # Step 2: CLIP 引导目标定位
    # --------------------------
    (x, y), best_patch_score = localize_with_clip_patch_similarity(
        clip_model=clip_model,
        clip_processor=clip_processor,
        image=image,
        text=args.prompt,
        device=device,
    )
    print(f"[Step2] CLIP best patch center (x, y)=({x}, {y}), score={best_patch_score:.4f}")

    # --------------------------
    # Step 3: SAM 自动分割
    # --------------------------
    mask_bool = predict_mask_from_point(
        predictor=sam_predictor,
        image=image,
        point_xy=(x, y),
    )

    # 保存 mask（白色=重绘区域）
    mask_path = os.path.join(args.output_dir, "sam_mask.png")
    save_mask(mask_bool, mask_path)
    mask_pil_l = mask_to_pil_l(mask_bool)

    # --------------------------
    # Step 4: Stable Diffusion Inpainting
    # --------------------------
    edited = inpaint(
        pipe=sd_pipe,
        image=image,
        mask_image=mask_pil_l,
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    save_image(edited, os.path.join(args.output_dir, "edited.png"))

    # --------------------------
    # Step 5: 评估 (CLIP score)
    # --------------------------
    score = clip_score_image_text(
        clip_model=clip_model,
        clip_processor=clip_processor,
        image=edited,
        text=args.prompt,
        device=device,
    )

    print(f"[Step5] CLIP score (edited vs text): {score:.4f}")


if __name__ == "__main__":
    main()
