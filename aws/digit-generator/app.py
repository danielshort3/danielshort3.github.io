import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "VAE.pth")
LATENT_DIM = int(os.getenv("LATENT_DIM", "20"))
DEFAULT_ROWS = int(os.getenv("GRID_ROWS", "8"))
DEFAULT_COLS = int(os.getenv("GRID_COLS", "8"))
VALUE_MIN = float(os.getenv("VALUE_MIN", "-2"))
VALUE_MAX = float(os.getenv("VALUE_MAX", "2"))
VALUE_STEP = float(os.getenv("VALUE_STEP", "0.2"))
DEFAULT_MODE = os.getenv("SAMPLING_MODE", "cluster")
LATENT_STATS_PATH = os.path.join(os.path.dirname(__file__), "latent_stats.json")

RESPONSE_HEADERS = {
  "Content-Type": "application/json"
}

torch.set_grad_enabled(False)
torch.set_num_threads(1)


class VAE(nn.Module):
  def __init__(self, latent_dim=LATENT_DIM):
    super().__init__()
    self.encoder_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
    self.encoder_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
    self.encoder_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
    self.fc_mu = nn.Linear(128 * 3 * 3, latent_dim)
    self.fc_logvar = nn.Linear(128 * 3 * 3, latent_dim)

    self.fc_decoder = nn.Linear(latent_dim, 128 * 3 * 3)
    self.decoder_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
    self.decoder_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0)
    self.decoder_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)

  def encode(self, x):
    h1 = F.relu(self.encoder_conv1(x))
    h2 = F.relu(self.encoder_conv2(h1))
    h3 = F.relu(self.encoder_conv3(h2))
    h3 = h3.view(-1, 128 * 3 * 3)
    return self.fc_mu(h3), self.fc_logvar(h3)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    h3 = F.relu(self.fc_decoder(z))
    h3 = h3.view(-1, 128, 3, 3)
    h4 = F.relu(self.decoder_conv1(h3))
    h5 = F.relu(self.decoder_conv2(h4))
    return torch.sigmoid(self.decoder_conv3(h5))

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar


_MODEL = None
_LATENT_STATS = None


def load_model():
  global _MODEL
  if _MODEL is not None:
    return _MODEL
  if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
  model = VAE(latent_dim=LATENT_DIM)
  state = torch.load(MODEL_PATH, map_location="cpu")
  model.load_state_dict(state, strict=True)
  model.eval()
  _MODEL = model
  return _MODEL


def load_latent_stats():
  global _LATENT_STATS
  if _LATENT_STATS is not None:
    return _LATENT_STATS
  if not os.path.exists(LATENT_STATS_PATH):
    raise FileNotFoundError(f"Latent stats not found at {LATENT_STATS_PATH}")
  with open(LATENT_STATS_PATH, "r", encoding="utf-8") as handle:
    data = json.load(handle)
  means = np.array(data.get("means", []), dtype=np.float32)
  stds = np.array(data.get("stds", []), dtype=np.float32)
  if means.shape != (10, LATENT_DIM) or stds.shape != (10, LATENT_DIM):
    raise ValueError("Latent stats shape mismatch")
  _LATENT_STATS = {
    "means": means,
    "stds": stds,
    "counts": data.get("counts", [])
  }
  return _LATENT_STATS


def parse_body(event):
  if not event:
    return {}
  body = event.get("body")
  if not body:
    return {}
  if event.get("isBase64Encoded"):
    body = base64.b64decode(body).decode("utf-8")
  if isinstance(body, (bytes, bytearray)):
    body = body.decode("utf-8")
  if isinstance(body, str):
    try:
      return json.loads(body)
    except json.JSONDecodeError:
      return {}
  if isinstance(body, dict):
    return body
  return {}


def clamp(value, min_val, max_val):
  return max(min_val, min(max_val, value))


def coerce_int(value, default, min_val=None, max_val=None):
  try:
    value = int(value)
  except (TypeError, ValueError):
    value = default
  if min_val is not None:
    value = max(min_val, value)
  if max_val is not None:
    value = min(max_val, value)
  return value


def coerce_float(value, default, min_val=None, max_val=None):
  try:
    value = float(value)
  except (TypeError, ValueError):
    value = default
  if min_val is not None:
    value = max(min_val, value)
  if max_val is not None:
    value = min(max_val, value)
  return value


def coerce_digit(value):
  try:
    digit = int(value)
  except (TypeError, ValueError):
    return None
  if 0 <= digit <= 9:
    return digit
  return None


def normalize_mode(value):
  raw = str(value or DEFAULT_MODE).strip().lower()
  if raw in ("cluster", "digit", "digit-cluster", "digit_cluster"):
    return "cluster"
  if raw in ("random", "rand"):
    return "random"
  return "cluster" if DEFAULT_MODE == "cluster" else "random"


def tensor_to_base64(tensor):
  arr = tensor.squeeze().cpu().numpy()
  arr = np.clip(arr, 0, 1)
  arr = (arr * 255).astype("uint8")
  img = Image.fromarray(arr, mode="L")
  buf = io.BytesIO()
  img.save(buf, format="PNG")
  return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_sweep_values(center, rows, cols):
  count = rows * cols
  if count <= 1:
    return torch.tensor([center], dtype=torch.float32)
  step = VALUE_STEP if VALUE_STEP > 0 else 0.2
  base = np.arange(VALUE_MIN, VALUE_MAX + (step / 2), step, dtype=np.float32)
  if base.size == 0:
    base = np.array([VALUE_MIN, VALUE_MAX], dtype=np.float32)
  if base.size == 1:
    seq = np.full(count, base[0], dtype=np.float32)
  else:
    ping = np.concatenate([base, base[-2:0:-1]])
    mid = count // 2
    center_idx = int(np.argmin(np.abs(ping - center)))
    start_idx = (center_idx - mid) % ping.size
    seq = np.array([ping[(start_idx + i) % ping.size] for i in range(count)], dtype=np.float32)
  grid = seq.reshape(rows, cols)
  for row in range(1, rows, 2):
    grid[row] = grid[row][::-1]
  return torch.from_numpy(grid.flatten())


def build_base_vector(mode, seed, cluster_digit):
  resolved_mode = normalize_mode(mode)
  if resolved_mode == "cluster":
    digit = coerce_digit(cluster_digit)
    if digit is None:
      digit = int(seed) % 10
    try:
      stats = load_latent_stats()
    except Exception:
      resolved_mode = "random"
    else:
      mean = torch.tensor(stats["means"][digit], dtype=torch.float32)
      std = torch.tensor(stats["stds"][digit], dtype=torch.float32)
      noise = torch.randn(LATENT_DIM)
      return mean + (noise * std), resolved_mode, digit
  base = torch.randn(LATENT_DIM)
  return base, "random", None


def generate_grid(seed, dim, value, rows, cols, mode, cluster_digit):
  model = load_model()
  count = rows * cols
  torch.manual_seed(seed)
  np.random.seed(seed)

  base, resolved_mode, resolved_digit = build_base_vector(mode, seed, cluster_digit)
  latents = base.unsqueeze(0).repeat(count, 1)
  sweep = build_sweep_values(value, rows, cols)
  latents[:, dim] = sweep
  with torch.no_grad():
    decoded = model.decode(latents).cpu()

  images = [tensor_to_base64(decoded[i, 0]) for i in range(count)]
  grid = [images[i * cols:(i + 1) * cols] for i in range(rows)]
  return grid, resolved_mode, resolved_digit


def handler(event, context):
  start = time.time()
  method = (
    (event.get("requestContext", {}).get("http", {}).get("method"))
    or event.get("httpMethod")
    or "GET"
  )

  if method == "OPTIONS":
    return {
      "statusCode": 204,
      "headers": RESPONSE_HEADERS
    }

  if method == "GET":
    return {
      "statusCode": 200,
      "headers": RESPONSE_HEADERS,
      "body": json.dumps({
        "status": "ok",
        "latent_dim": LATENT_DIM,
        "model_loaded": _MODEL is not None
      })
    }

  payload = parse_body(event)
  seed = payload.get("seed")
  seed = coerce_int(seed, int(time.time() * 1000) % 1000000, 0, 2_147_483_647)
  mode = payload.get("mode", payload.get("sampling_mode", DEFAULT_MODE))
  cluster_digit = payload.get("cluster_digit", payload.get("digit"))
  dim = payload.get("dim", payload.get("dimension", 0))
  dim = coerce_int(dim, 0, 0, LATENT_DIM - 1)
  value = payload.get("value", payload.get("latent_value", 0.0))
  value = coerce_float(value, 0.0, VALUE_MIN, VALUE_MAX)

  grid = payload.get("grid", {})
  rows = payload.get("rows", grid.get("rows", DEFAULT_ROWS) if isinstance(grid, dict) else DEFAULT_ROWS)
  cols = payload.get("cols", grid.get("cols", DEFAULT_COLS) if isinstance(grid, dict) else DEFAULT_COLS)
  rows = coerce_int(rows, DEFAULT_ROWS, 1, 8)
  cols = coerce_int(cols, DEFAULT_COLS, 1, 8)

  try:
    images, resolved_mode, resolved_digit = generate_grid(seed, dim, value, rows, cols, mode, cluster_digit)
    duration_ms = int((time.time() - start) * 1000)
    return {
      "statusCode": 200,
      "headers": RESPONSE_HEADERS,
      "body": json.dumps({
        "seed": seed,
        "mode": resolved_mode,
        "cluster_digit": resolved_digit,
        "dim": dim,
        "value": value,
        "rows": rows,
        "cols": cols,
        "latent_dim": LATENT_DIM,
        "value_min": VALUE_MIN,
        "value_max": VALUE_MAX,
        "images": images,
        "duration_ms": duration_ms
      })
    }
  except Exception as exc:
    return {
      "statusCode": 500,
      "headers": RESPONSE_HEADERS,
      "body": json.dumps({ "error": str(exc) })
    }
