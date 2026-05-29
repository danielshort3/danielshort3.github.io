# ComfyUI IPAdapter Comparison

Generated with `build/run-project-starfall-comfyui-ipadapter-comparison.py`.

## Setup

- Base model: `animagine-xl-4.0-opt.safetensors`
- ControlNet: `mistoLine_rank256.safetensors`
- IPAdapter: `ip-adapter-plus_sdxl_vit-h.safetensors`
- CLIP Vision: `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
- Identity reference: `img/project-starfall/enemies/bandit-cutter-reference.png`
- Sampling: `dpmpp_2m_sde`, `karras`, 22 steps, CFG 6.5
- IPAdapter settings: weight 0.45, linear, end 0.65

## Result

ControlNet-only follows the pose guides more literally but fails character consistency. It frequently changes the enemy into unrelated shapes, symbols, animal-eared mascots, or abstract creatures.

ControlNet plus IPAdapter is substantially better for maintaining a bandit-like identity: dark hood, glowing eyes, scarf, belt, dagger, and compact body language appear across most frames.

The remaining issue is pose authority. IPAdapter resists large pose changes, especially hit/death frames, and can keep the character upright or invent extra weapons/details. This means IPAdapter should be used as an identity lock, but not at one global setting for every animation.

## Recommendation

Use this pipeline, but tune by animation group:

- Idle/walk/attack: IPAdapter weight around 0.40-0.50.
- Hit/death/collapse: lower IPAdapter weight around 0.20-0.35, stronger ControlNet, and animation-specific prompts.
- Keep using rounded pose guides; sharp guide corners produced animal-ear artifacts.
- Generate each animation group separately, then run local alignment/cropping before assembling sheets.
