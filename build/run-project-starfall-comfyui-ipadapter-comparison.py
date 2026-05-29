#!/usr/bin/env python3
"""
Generate a Project Starfall ComfyUI comparison set:

- ControlNet-only frames from deterministic pose guides.
- ControlNet + IPAdapter frames using a single reference image.
- A contact sheet and simple bounding-box metrics for visual comparison.

Run this with the ComfyUI Python environment so Pillow is available:

    C:\\Users\\clopt\\Documents\\coding\\Personal_Projects\\ComfyUI\\.venv\\Scripts\\python.exe ^
      build\\run-project-starfall-comfyui-ipadapter-comparison.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(r"C:\Users\clopt\Documents\coding\Personal_Projects\danielshort3.github.io")
COMFY_ROOT = Path(r"C:\Users\clopt\AppData\Local\Programs\ComfyUI\resources\ComfyUI")
COMFY_URL = "http://127.0.0.1:8188"

CHECKPOINT = "animagine-xl-4.0-opt.safetensors"
CONTROLNET = "mistoLine_rank256.safetensors"
IPADAPTER = "ip-adapter-plus_sdxl_vit-h.safetensors"
CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

OUT_ROOT = REPO_ROOT / "asset-sources" / "project-starfall" / "comfyui-tests" / "ipadapter-comparison"
POSE_DIR = OUT_ROOT / "pose-guides"
CONTROL_DIR = OUT_ROOT / "controlnet-only"
IPADAPTER_DIR = OUT_ROOT / "ipadapter-controlnet"
REFERENCE_SOURCE = REPO_ROOT / "img" / "project-starfall" / "enemies" / "bandit-cutter-reference.png"
REFERENCE_INPUT_NAME = "starfall_bandit_identity_reference.png"

POSITIVE_BASE = (
    "masterpiece, best quality, solo, single character, one character only, "
    "chibi hooded bandit enemy, full body, centered, feet visible, clean silhouette, "
    "simple readable shape, small dagger, dark black hood with tan trim, red scarf, brown cloak, "
    "compact mascot-like proportions, "
    "rounded cloth hood with no animal ears, maple-inspired side-scrolling RPG enemy sprite, "
    "plain solid bright green background"
)
NEGATIVE_PROMPT = (
    "low quality, blurry, realistic, 3d render, cropped, cut off feet, multiple characters, "
    "many characters, group, lineup, collage, character sheet, multiple poses, duplicate subject, "
    "duplicate limbs, extra arms, extra legs, animal, slime, text, watermark, panel borders, "
    "sprite sheet, overlapping frames, parts from another frame, animal ears, fox ears, cat ears, "
    "rabbit ears, horns, antennae"
)


@dataclass(frozen=True)
class PoseSpec:
    key: str
    prompt: str
    lean: int = 0
    bob: int = 0
    left_leg: int = 0
    right_leg: int = 0
    left_arm: tuple[int, int] = (-68, 82)
    right_arm: tuple[int, int] = (92, 82)
    dagger: tuple[int, int] = (138, 50)
    hood_width: int = 168
    hood_height: int = 190
    body_squash: int = 0
    fallen: int = 0


POSES = [
    PoseSpec("idle_1", "standing idle pose, calm breathing frame 1"),
    PoseSpec("idle_2", "standing idle pose, calm breathing frame 2", bob=-12, hood_width=174),
    PoseSpec("idle_3", "standing idle pose, calm breathing frame 3", bob=8, hood_width=162, body_squash=8),
    PoseSpec("walk_1", "walking animation frame 1, left foot forward", lean=-14, left_leg=-34, right_leg=24, left_arm=(-88, 64), right_arm=(86, 100)),
    PoseSpec("walk_2", "walking animation frame 2, passing step", bob=-8, left_leg=-4, right_leg=4, left_arm=(-74, 82), right_arm=(98, 72)),
    PoseSpec("walk_3", "walking animation frame 3, right foot forward", lean=14, left_leg=28, right_leg=-34, left_arm=(-54, 102), right_arm=(118, 54)),
    PoseSpec("attack_1", "dagger attack windup frame 1", lean=-20, left_arm=(-66, 76), right_arm=(62, -38), dagger=(112, -94)),
    PoseSpec("attack_2", "dagger attack slash frame 2, arm extended", lean=18, left_arm=(-52, 84), right_arm=(178, 22), dagger=(242, -6)),
    PoseSpec("attack_3", "dagger attack recovery frame 3", lean=8, bob=4, left_arm=(-66, 94), right_arm=(118, 76), dagger=(162, 34)),
    PoseSpec("hit_1", "damage taken recoil frame, one frame only", lean=-36, bob=4, left_arm=(-110, 44), right_arm=(112, 118), dagger=(154, 92), body_squash=-8),
    PoseSpec("death_1", "defeat animation frame 1, falling backward", lean=-52, bob=24, left_arm=(-114, 64), right_arm=(92, 128), dagger=(132, 110), body_squash=-16),
    PoseSpec("death_2", "defeat animation frame 2, collapsing low", lean=-90, bob=112, left_leg=-48, right_leg=42, left_arm=(-132, 22), right_arm=(92, 36), dagger=(154, 18), hood_width=178, hood_height=142, body_squash=22, fallen=1),
    PoseSpec("death_3", "defeat animation frame 3, flattened on ground", lean=-96, bob=170, left_leg=-58, right_leg=58, left_arm=(-132, 10), right_arm=(126, 14), dagger=(178, 0), hood_width=192, hood_height=108, body_squash=44, fallen=2),
]


def http_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{COMFY_URL}{path}"
    if payload is None:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_comfy_ready() -> None:
    try:
        stats = http_json("/system_stats")
    except urllib.error.URLError as exc:
        raise SystemExit(f"ComfyUI is not reachable at {COMFY_URL}: {exc}") from exc
    version = stats.get("system", {}).get("comfyui_version", "unknown")
    device = (stats.get("devices") or [{}])[0].get("name", "unknown device")
    print(f"ComfyUI {version} ready on {device}")


def draw_pose(spec: PoseSpec, path: Path) -> None:
    width = height = 768
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    cx = 384 + spec.lean
    top = 188 + spec.bob
    hood_w = spec.hood_width
    hood_h = spec.hood_height
    body_top = top + hood_h + 10
    ground_y = 666

    if spec.fallen:
        y = min(ground_y - 90, body_top + spec.bob // 2)
        draw.ellipse((cx - 180, y - 68, cx + 148, y + 54), outline=(0, 0, 0), width=12)
        draw.line([(cx - 160, y + 44), (cx - 215, y + 92)], fill=(0, 0, 0), width=12)
        draw.line([(cx + 132, y + 40), (cx + 224, y + 62)], fill=(0, 0, 0), width=12)
        draw.line([(cx - 70, y + 44), (cx - 112, ground_y - 18)], fill=(0, 0, 0), width=12)
        draw.line([(cx + 48, y + 44), (cx + 118, ground_y - 20)], fill=(0, 0, 0), width=12)
        draw.line([(cx + 162, y + 54), (cx + 240, y + 32)], fill=(0, 0, 0), width=8)
        draw.line([(cx + 162, y + 54), (cx + 218, y + 92)], fill=(0, 0, 0), width=8)
        draw.line([(cx - 210, ground_y), (cx + 250, ground_y)], fill=(0, 0, 0), width=4)
        image.save(path)
        return

    hood_box = (
        cx - hood_w // 2 - 62,
        top - 10,
        cx + hood_w // 2 + 62,
        top + hood_h + 88,
    )
    draw.ellipse(hood_box, outline=(0, 0, 0), width=12)
    draw.ellipse((cx - 82, top + 72, cx + 82, top + 198), outline=(0, 0, 0), width=10)

    body_bottom = ground_y - 82 + spec.body_squash
    cloak = [
        (cx - 94, body_top),
        (cx - 142, body_bottom),
        (cx - 56, body_bottom + 26),
        (cx, body_bottom - 14),
        (cx + 56, body_bottom + 26),
        (cx + 142, body_bottom),
        (cx + 94, body_top),
    ]
    draw.line(cloak, fill=(0, 0, 0), width=12, joint="curve")

    left_hip = (cx - 44, body_bottom + 8)
    right_hip = (cx + 44, body_bottom + 8)
    left_foot = (cx - 62 + spec.left_leg, ground_y - 16)
    right_foot = (cx + 62 + spec.right_leg, ground_y - 16)
    draw.line([left_hip, left_foot], fill=(0, 0, 0), width=12)
    draw.line([right_hip, right_foot], fill=(0, 0, 0), width=12)
    draw.line([(left_foot[0] - 28, left_foot[1]), (left_foot[0] + 24, left_foot[1])], fill=(0, 0, 0), width=12)
    draw.line([(right_foot[0] - 24, right_foot[1]), (right_foot[0] + 30, right_foot[1])], fill=(0, 0, 0), width=12)

    left_shoulder = (cx - 102, body_top + 22)
    right_shoulder = (cx + 102, body_top + 22)
    left_hand = (cx + spec.left_arm[0], body_top + spec.left_arm[1])
    right_hand = (cx + spec.right_arm[0], body_top + spec.right_arm[1])
    dagger_tip = (cx + spec.dagger[0], body_top + spec.dagger[1])
    draw.line([left_shoulder, left_hand], fill=(0, 0, 0), width=10)
    draw.line([right_shoulder, right_hand], fill=(0, 0, 0), width=10)
    draw.line([right_hand, dagger_tip], fill=(0, 0, 0), width=8)
    draw.line([right_hand, (dagger_tip[0] - 28, dagger_tip[1] + 60)], fill=(0, 0, 0), width=8)
    draw.line([dagger_tip, (dagger_tip[0] - 28, dagger_tip[1] + 60)], fill=(0, 0, 0), width=8)
    draw.line([(cx - 180, ground_y), (cx + 210, ground_y)], fill=(0, 0, 0), width=4)
    image.save(path)


def prepare_reference(source: Path, destination: Path) -> None:
    source_image = Image.open(source).convert("RGBA")
    source_image.thumbnail((520, 520), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (768, 768), (244, 238, 226, 255))
    x = (canvas.width - source_image.width) // 2
    y = (canvas.height - source_image.height) // 2
    canvas.alpha_composite(source_image, (x, y))
    canvas.convert("RGB").save(destination)


def prepare_inputs() -> None:
    for directory in [POSE_DIR, CONTROL_DIR, IPADAPTER_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    comfy_input = COMFY_ROOT / "input"
    comfy_input.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_SOURCE.exists():
        raise SystemExit(f"Missing identity reference: {REFERENCE_SOURCE}")
    prepare_reference(REFERENCE_SOURCE, comfy_input / REFERENCE_INPUT_NAME)
    shutil.copy2(comfy_input / REFERENCE_INPUT_NAME, OUT_ROOT / REFERENCE_INPUT_NAME)

    for spec in POSES:
        local_pose = POSE_DIR / f"{spec.key}.png"
        comfy_pose = comfy_input / f"starfall_pose_{spec.key}.png"
        draw_pose(spec, local_pose)
        shutil.copy2(local_pose, comfy_pose)


def make_prompt(spec: PoseSpec, variant: str, seed: int, steps: int, args: argparse.Namespace) -> dict[str, Any]:
    positive = f"{POSITIVE_BASE}, {spec.prompt}, matching the black line pose guide"
    use_ipadapter = variant == "ipadapter"
    ksampler_model = ["14", 0] if use_ipadapter else ["1", 0]
    prompt: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positive}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEGATIVE_PROMPT}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 768, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ksampler_model,
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 6.5,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"starfall_ipadapter_compare_{variant}_{spec.key}", "images": ["6", 0]}},
        "8": {"class_type": "LoadImage", "inputs": {"image": f"starfall_pose_{spec.key}.png"}},
        "9": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CONTROLNET}},
        "10": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "control_net": ["9", 0],
                "image": ["8", 0],
                "strength": 0.78,
                "start_percent": 0.0,
                "end_percent": 0.9,
                "vae": ["1", 2],
            },
        },
    }

    if use_ipadapter:
        prompt.update(
            {
                "11": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IPADAPTER}},
                "12": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
                "13": {"class_type": "LoadImage", "inputs": {"image": REFERENCE_INPUT_NAME}},
                "14": {
                    "class_type": "IPAdapterAdvanced",
                    "inputs": {
                        "model": ["1", 0],
                        "ipadapter": ["11", 0],
                        "image": ["13", 0],
                        "clip_vision": ["12", 0],
                        "weight": args.ip_weight,
                        "weight_type": args.ip_weight_type,
                        "combine_embeds": "concat",
                        "start_at": 0.0,
                        "end_at": args.ip_end,
                        "embeds_scaling": args.ip_embeds_scaling,
                    },
                },
            }
        )
    return prompt


def queue_prompt(prompt: dict[str, Any], client_id: str) -> str:
    payload = {"client_id": client_id, "prompt": prompt}
    response = http_json("/prompt", payload)
    errors = response.get("node_errors") or {}
    if errors:
        raise RuntimeError(json.dumps(errors, indent=2))
    return response["prompt_id"]


def wait_for_prompt(prompt_id: str, timeout: int = 600) -> dict[str, Any]:
    started = time.time()
    while time.time() - started < timeout:
        history = http_json(f"/history/{prompt_id}")
        result = history.get(prompt_id)
        if result:
            status = result.get("status", {})
            if status.get("completed"):
                return result
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(result, indent=2))
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def copy_output(history_result: dict[str, Any], destination: Path) -> None:
    images = history_result.get("outputs", {}).get("7", {}).get("images", [])
    if not images:
        raise RuntimeError(f"No SaveImage output found: {json.dumps(history_result, indent=2)[:1000]}")
    image = images[0]
    source = COMFY_ROOT / "output" / image.get("subfolder", "") / image["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def foreground_bbox(image_path: Path) -> dict[str, float | int]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    corners = [image.getpixel((0, 0)), image.getpixel((width - 1, 0)), image.getpixel((0, height - 1)), image.getpixel((width - 1, height - 1))]
    bg = tuple(int(sum(channel) / len(corners)) for channel in zip(*corners))
    pixels = image.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            if sum(abs(pixel[i] - bg[i]) for i in range(3)) > 42:
                xs.append(x)
                ys.append(y)
    if not xs:
        return {"x": 0, "y": 0, "w": 0, "h": 0, "cx": 0, "cy": 0}
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {
        "x": x0,
        "y": y0,
        "w": x1 - x0 + 1,
        "h": y1 - y0 + 1,
        "cx": round((x0 + x1) / 2, 2),
        "cy": round((y0 + y1) / 2, 2),
    }


def make_contact_sheet() -> None:
    thumb_w = 196
    thumb_h = 196
    label_h = 28
    columns = [("pose", POSE_DIR), ("controlnet", CONTROL_DIR), ("ipadapter", IPADAPTER_DIR)]
    sheet = Image.new("RGB", (thumb_w * len(columns), (thumb_h + label_h) * (len(POSES) + 1)), (245, 245, 241))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for col, (label, _) in enumerate(columns):
        draw.text((col * thumb_w + 8, 8), label, fill=(18, 28, 40), font=font)

    for row, spec in enumerate(POSES, start=1):
        y = row * (thumb_h + label_h)
        draw.text((8, y + 6), spec.key, fill=(18, 28, 40), font=font)
        for col, (_, directory) in enumerate(columns):
            if directory == POSE_DIR:
                path = directory / f"{spec.key}.png"
            else:
                path = directory / f"{spec.key}.png"
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w - 12, thumb_h - 12), Image.Resampling.LANCZOS)
            x = col * thumb_w + (thumb_w - image.width) // 2
            y_img = y + label_h + (thumb_h - image.height) // 2
            sheet.paste(image, (x, y_img))

    sheet.save(OUT_ROOT / "comparison-contact-sheet.png")


def run(args: argparse.Namespace) -> None:
    ensure_comfy_ready()
    prepare_inputs()
    selected_poses = POSES[: args.limit] if args.limit else POSES
    summary: dict[str, Any] = {"controlnet-only": {}, "ipadapter-controlnet": {}}

    workflow_dir = OUT_ROOT / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)

    for variant, out_dir in [("control", CONTROL_DIR), ("ipadapter", IPADAPTER_DIR)]:
        for index, spec in enumerate(selected_poses):
            seed = args.seed + index
            workflow = make_prompt(spec, variant, seed, args.steps, args)
            workflow_path = workflow_dir / f"{variant}_{spec.key}.json"
            workflow_path.write_text(json.dumps({"client_id": "codex-starfall-ipadapter", "prompt": workflow}, indent=2), encoding="utf-8")
            print(f"Queueing {variant} {spec.key} seed={seed}")
            prompt_id = queue_prompt(workflow, "codex-starfall-ipadapter")
            result = wait_for_prompt(prompt_id, timeout=args.timeout)
            destination = out_dir / f"{spec.key}.png"
            copy_output(result, destination)
            summary_key = "ipadapter-controlnet" if variant == "ipadapter" else "controlnet-only"
            summary[summary_key][spec.key] = foreground_bbox(destination)

    if selected_poses == POSES:
        make_contact_sheet()

    (OUT_ROOT / "comparison-metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote comparison assets to {OUT_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=52703000)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="Only generate the first N poses.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--ip-weight", type=float, default=0.45)
    parser.add_argument("--ip-weight-type", default="linear")
    parser.add_argument("--ip-end", type=float, default=0.65)
    parser.add_argument("--ip-embeds-scaling", default="V only")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
