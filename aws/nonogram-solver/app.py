import base64
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SIZE = int(os.getenv("GRID_SIZE", "5"))
DEFAULT_CLUE_MAX_LEN = int(os.getenv("CLUE_MAX_LEN", "3"))
DEFAULT_CLUE_VOCAB = int(os.getenv("CLUE_VOCAB", "5"))
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "models", "checkpoint_52000.pth")

torch.set_grad_enabled(False)
torch.set_num_threads(1)


def clues_from_line(line):
  runs = []
  count = 0
  for val in line:
    if int(val) == 1:
      count += 1
    else:
      if count:
        runs.append(count)
        count = 0
  if count:
    runs.append(count)
  return runs or [0]


def build_clues(solution):
  rows = [clues_from_line(row) for row in solution]
  cols = [clues_from_line(solution[:, idx]) for idx in range(solution.shape[1])]
  return rows, cols


def pad_clue(clue, max_len):
  return clue + [0] * (max_len - len(clue))


class ClueTransformer(nn.Module):
  def __init__(self, grid, max_len, vocab, heads, layers, dim):
    super().__init__()
    self.embed = nn.Embedding(vocab + 1, dim)
    self.pos_enc = nn.Parameter(torch.randn(1, max_len * grid, dim))
    block = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
    self.xf = nn.TransformerEncoder(block, layers)

  def forward(self, x):
    b, g, l = x.shape
    x = x.view(b, -1)
    e = self.embed(x)
    if self.pos_enc.size(1) < e.size(1):
      self.pos_enc = nn.Parameter(torch.randn(1, e.size(1), e.size(-1), device=e.device))
    e = e + self.pos_enc[:, : e.size(1)]
    return self.xf(e)


class PolicyNetwork(nn.Module):
  def __init__(self, grid, max_len, vocab):
    super().__init__()
    self.grid = grid
    self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
    self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
    self.fc1 = nn.Linear(8 * grid * grid, 16)
    self.row_trans = ClueTransformer(grid, max_len, vocab, heads=2, layers=1, dim=16)
    self.col_trans = ClueTransformer(grid, max_len, vocab, heads=2, layers=1, dim=16)
    self.fc2 = nn.Linear(16 * 3, 32)
    self.fc3 = nn.Linear(32, grid * grid * 2)

  def forward(self, s, r, c):
    x = F.relu(self.conv1(s.unsqueeze(1).float()))
    x = F.relu(self.conv2(x))
    x = x.view(-1, 8 * self.grid * self.grid)
    x = F.relu(self.fc1(x))
    r = self.row_trans(r).mean(1)
    c = self.col_trans(c).mean(1)
    x = torch.cat((x, r, c), dim=1)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x.view(-1, self.grid * self.grid, 2)


_MODEL = None
_MODEL_META = {}


def load_model():
  global _MODEL, _MODEL_META
  if _MODEL is not None:
    return _MODEL, _MODEL_META

  if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

  ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
  clue_max_len = int(ckpt.get("clue_max_len", DEFAULT_CLUE_MAX_LEN))
  vocab = int(ckpt.get("clue_dim", DEFAULT_CLUE_VOCAB))
  model = PolicyNetwork(GRID_SIZE, clue_max_len, vocab)
  state = ckpt.get("model_state_dict", ckpt)
  model.load_state_dict(state, strict=False)
  model.eval()

  _MODEL = model
  _MODEL_META = {
    "clue_max_len": clue_max_len,
    "vocab": vocab
  }
  return _MODEL, _MODEL_META


def normalize_solution(raw):
  arr = np.array(raw, dtype=np.int32)
  if arr.shape != (GRID_SIZE, GRID_SIZE):
    raise ValueError(f"solution must be {GRID_SIZE}x{GRID_SIZE}")
  return np.where(arr > 0, 1, 0)


def solve_nonogram(solution, seed=None, max_steps=None):
  model, meta = load_model()
  clue_max_len = meta["clue_max_len"]

  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

  row_clues, col_clues = build_clues(solution)
  row_pad = np.array([[pad_clue(c, clue_max_len) for c in row_clues]], dtype=np.int64)
  col_pad = np.array([[pad_clue(c, clue_max_len) for c in col_clues]], dtype=np.int64)

  state = np.full((1, GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
  chosen = set()
  steps = []
  max_steps = max_steps or GRID_SIZE * GRID_SIZE

  for _ in range(max_steps):
    logits = model(
      torch.tensor(state, dtype=torch.float32),
      torch.tensor(row_pad, dtype=torch.long),
      torch.tensor(col_pad, dtype=torch.long)
    )
    flat_logits = logits.view(-1)
    probs = torch.softmax(flat_logits, dim=0)
    order = torch.argsort(probs, descending=True)

    selected = None
    for idx in order.tolist():
      cell = idx // 2
      r = cell // GRID_SIZE
      c = cell % GRID_SIZE
      if (r, c) in chosen:
        continue
      selected = idx
      break

    if selected is None:
      break

    pred = selected % 2
    cell = selected // 2
    row = cell // GRID_SIZE
    col = cell % GRID_SIZE
    actual = int(solution[row, col])
    correct = pred == actual
    confidence = float(probs[selected].item())

    chosen.add((row, col))
    state[0, row, col] = actual
    steps.append({
      "row": row,
      "col": col,
      "predicted": pred,
      "actual": actual,
      "correct": bool(correct),
      "confidence": round(confidence, 4)
    })

    if len(chosen) >= GRID_SIZE * GRID_SIZE:
      break

  correct_count = sum(1 for s in steps if s["correct"])
  solved = np.array_equal(state[0], solution)
  return {
    "grid": GRID_SIZE,
    "row_clues": row_clues,
    "col_clues": col_clues,
    "solution": solution.tolist(),
    "steps": steps,
    "step_count": len(steps),
    "correct_count": correct_count,
    "correct_rate": round((correct_count / len(steps)) if steps else 0, 4),
    "solved": bool(solved)
  }


def parse_body(event):
  if not event:
    return {}
  body = event.get("body")
  if not body:
    return {}
  if event.get("isBase64Encoded"):
    body = base64.b64decode(body).decode("utf-8")
  try:
    return json.loads(body) if isinstance(body, str) else body
  except json.JSONDecodeError:
    return {}


def cors_headers(origin):
  allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
  if not allowed:
    allowed = ["*"]
  if "*" in allowed:
    allow_origin = "*"
  else:
    allow_origin = origin if origin in allowed else allowed[0]
  return {
    "Access-Control-Allow-Origin": allow_origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Max-Age": "86400"
  }


def handler(event, context):
  start = time.time()
  method = (
    (event.get("requestContext", {}).get("http", {}).get("method"))
    or event.get("httpMethod")
    or "GET"
  )
  headers = event.get("headers") or {}
  origin = headers.get("origin") or headers.get("Origin") or ""
  base_headers = cors_headers(origin)

  if method == "OPTIONS":
    return {
      "statusCode": 204,
      "headers": base_headers
    }

  if method == "GET":
    model_loaded = _MODEL is not None
    return {
      "statusCode": 200,
      "headers": base_headers,
      "body": json.dumps({
        "status": "ok",
        "model_loaded": model_loaded,
        "grid": GRID_SIZE
      })
    }

  payload = parse_body(event)
  seed = payload.get("seed")
  if seed is not None:
    try:
      seed = int(seed)
    except (TypeError, ValueError):
      seed = None

  if payload.get("solution") is not None:
    solution = normalize_solution(payload.get("solution"))
  else:
    if seed is not None:
      np.random.seed(seed)
    solution = np.random.randint(2, size=(GRID_SIZE, GRID_SIZE))

  try:
    result = solve_nonogram(solution, seed=seed)
    duration_ms = int((time.time() - start) * 1000)
    result["duration_ms"] = duration_ms
    return {
      "statusCode": 200,
      "headers": base_headers,
      "body": json.dumps(result)
    }
  except Exception as exc:
    return {
      "statusCode": 500,
      "headers": base_headers,
      "body": json.dumps({
        "error": str(exc)
      })
    }
