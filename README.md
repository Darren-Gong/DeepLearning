# Text-guided Image Editing (CLIP + SAM + Stable Diffusion Inpainting)

- **CLIP**：图文语义对齐 + 网格级（patch-level）相似度定位目标点
- **SAM**：用定位点作为 prompt 自动分割目标区域
- **Stable Diffusion Inpainting**：仅在 mask 区域根据文本提示重绘
- **CLIP score**：评估生成图像与文本的一致性（余弦相似度）

## 目录结构

```
project/
├── main.py
├── models/
│   ├── clip_utils.py
│   ├── sam_utils.py
│   └── diffusion_utils.py
├── utils/
│   ├── image_utils.py
│   └── eval_utils.py
├── requirements.txt
└── README.md
```

## 环境安装

建议新建虚拟环境后安装：

```bash
conda activate med
pip install -r requirements.txt
```


## 模型下载与路径说明

本项目使用 3 个预训练模型（CLIP / SAM / Stable Diffusion Inpainting）。其中：

- CLIP 与 Stable Diffusion 默认通过 Hugging Face 自动下载并缓存；
- SAM 需要手动下载 checkpoint 并通过 `--sam_ckpt` 指定本地路径。

下面给出官方/常用下载入口与本地路径说明。

### CLIP (openai/clip-vit-base-patch32)

- Hugging Face 模型页：
  - https://huggingface.co/openai/clip-vit-base-patch32
- 自动下载缓存路径（默认）：
  - Linux: `~/.cache/huggingface/hub`
- 离线/本地使用：
  - 可将模型手动下载到本地目录，并通过 `--clip_model /path/to/clip-vit-base-patch32` 指向该目录。

### SAM 权重下载（必需）

本项目使用官方 SAM `vit_b`，需要你提供本地 checkpoint 路径（`--sam_ckpt`）。

你可以从官方发布页面下载 `sam_vit_b_01ec64.pth`，或使用你已经下载到本地的 checkpoint。

- 官方仓库下载地址（Release / checkpoint）：
  - https://github.com/facebookresearch/segment-anything
- 推荐文件名：`sam_vit_b_01ec64.pth`
- 使用方式：
  - 运行时通过 `--sam_ckpt /path/to/sam_vit_b_01ec64.pth` 指定本地路径

### Stable Diffusion Inpainting (runwayml/stable-diffusion-inpainting)

- Hugging Face 模型页：
  - https://huggingface.co/runwayml/stable-diffusion-inpainting
- 自动下载缓存路径（默认）：
  - Linux: `~/.cache/huggingface/hub`
- 离线/本地使用：
  - 可将模型手动下载到本地目录，并通过 `--sd_model /path/to/stable-diffusion-inpainting` 指向该目录。

## 运行示例

```bash
python main.py \
  --image /path/to/input.jpg \
  --prompt "a cat wearing sunglasses" \
  --sam_ckpt /path/to/sam_vit_b_01ec64.pth \
  --output_dir outputs/demo \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --seed 42
```

运行后会在 `--output_dir` 下保存：

- `original.png`：原图
- `sam_mask.png`：SAM 分割 mask（白色区域为重绘区域）
- `edited.png`：局部重绘后的结果

并在终端打印：

- CLIP 定位点坐标 `(x, y)`
- CLIP score（生成图像 vs 文本）

