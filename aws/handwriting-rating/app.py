import base64
import io
import json
import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_3.pth")
RESPONSE_HEADERS = {
  "Content-Type": "application/json"
}

torch.set_grad_enabled(False)
torch.set_num_threads(1)


class MNISTModelv3(nn.Module):
  def __init__(self, input_shape: int, output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.conv_block_3 = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=256 * 3 * 3, out_features=4096),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),
      nn.Dropout(),
      nn.Linear(in_features=4096, out_features=output_shape)
    )

  def forward(self, x):
    x = self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))
    return x


_MODEL = None


def load_model():
  global _MODEL
  if _MODEL is not None:
    return _MODEL
  if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
  model = MNISTModelv3(input_shape=1, output_shape=10)
  state = torch.load(MODEL_PATH, map_location="cpu")
  model.load_state_dict(state, strict=True)
  model.eval()
  _MODEL = model
  return _MODEL


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


def decode_image(b64_str):
  if not b64_str or not isinstance(b64_str, str):
    return None
  if b64_str.startswith("data:"):
    parts = b64_str.split(",", 1)
    if len(parts) == 2:
      b64_str = parts[1]
  try:
    raw = base64.b64decode(b64_str)
  except (ValueError, TypeError):
    return None
  try:
    return Image.open(io.BytesIO(raw)).convert("L")
  except Exception:
    return None


def preprocess_image(img):
  arr = np.asarray(img).astype("float32") / 255.0
  if arr.mean() > 0.5:
    arr = 1.0 - arr

  mask = arr > 0.1
  if mask.any():
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    img = Image.fromarray((arr * 255).astype("uint8")).crop((x0, y0, x1, y1))
  else:
    img = Image.fromarray((arr * 255).astype("uint8"))

  w, h = img.size
  size = max(w, h)
  canvas = Image.new("L", (size, size), color=0)
  canvas.paste(img, ((size - w) // 2, (size - h) // 2))
  img = canvas.resize((28, 28), resample=Image.BILINEAR)

  arr = np.asarray(img).astype("float32") / 255.0
  tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
  return tensor


def score_image(img):
  model = load_model()
  tensor = preprocess_image(img)
  logits = model(tensor)
  probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
  return {
    "digit_confidences": {
      str(i): round(float(p), 6)
      for i, p in enumerate(probs)
    }
  }


def handler(event, context):
  start = time.time()
  headers_out = dict(RESPONSE_HEADERS)
  method = (
    (event.get("requestContext", {}).get("http", {}).get("method"))
    or event.get("httpMethod")
    or "GET"
  )

  if method == "OPTIONS":
    return {
      "statusCode": 204,
      "headers": headers_out
    }

  if method == "GET":
    return {
      "statusCode": 200,
      "headers": headers_out,
      "body": json.dumps({
        "status": "ok",
        "model_loaded": _MODEL is not None
      })
    }

  payload = parse_body(event)
  b64 = (
    payload.get("image")
    or payload.get("b64")
    or payload.get("img")
    or payload.get("data")
  )
  if not b64:
    return {
      "statusCode": 400,
      "headers": headers_out,
      "body": json.dumps({ "error": "Missing base64 image payload." })
    }

  img = decode_image(b64)
  if img is None:
    return {
      "statusCode": 400,
      "headers": headers_out,
      "body": json.dumps({ "error": "Invalid image payload." })
    }

  try:
    result = score_image(img)
    result["duration_ms"] = int((time.time() - start) * 1000)
    return {
      "statusCode": 200,
      "headers": headers_out,
      "body": json.dumps(result)
    }
  except Exception as exc:
    return {
      "statusCode": 500,
      "headers": headers_out,
      "body": json.dumps({ "error": str(exc) })
    }
