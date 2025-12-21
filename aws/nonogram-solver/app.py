import base64
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SIZE = int(os.getenv("GRID_SIZE", "5"))
CLUE_MAX_LEN = int(os.getenv("CLUE_MAX_LEN", "3"))
CLUE_VOCAB = int(os.getenv("CLUE_VOCAB", "5"))
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "models", "checkpoint_52000.pth")

torch.set_grad_enabled(False)
torch.set_num_threads(1)

_EXISTING_SOLUTIONS = set()


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


def generate_unique_nonogram(grid_size, batch_size, existing_solutions=None):
  if existing_solutions is None:
    existing_solutions = set()
  solutions = []
  while len(solutions) < batch_size:
    new = np.random.randint(2, size=(batch_size, grid_size, grid_size))
    for sol in new:
      tup = tuple(map(tuple, sol))
      if tup not in existing_solutions:
        solutions.append(sol)
        existing_solutions.add(tup)
      if len(solutions) == batch_size:
        break
  solutions = np.asarray(solutions)

  row_clues, col_clues = [], []
  for sol in solutions:
    row_clues.append([[len(s) for s in "".join(map(str, r)).split("0") if s] or [0] for r in sol])
    col_clues.append([[len(s) for s in "".join(map(str, c)).split("0") if s] or [0] for c in sol.T])
  return solutions, row_clues, col_clues, existing_solutions


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
  model = PolicyNetwork(GRID_SIZE, CLUE_MAX_LEN, CLUE_VOCAB)
  state = ckpt.get("model_state_dict", ckpt)
  model.load_state_dict(state, strict=False)
  model.train()

  _MODEL = model
  _MODEL_META = {
    "clue_max_len": CLUE_MAX_LEN,
    "vocab": CLUE_VOCAB
  }
  return _MODEL, _MODEL_META


def normalize_solution(raw):
  arr = np.array(raw, dtype=np.int32)
  if arr.shape != (GRID_SIZE, GRID_SIZE):
    raise ValueError(f"solution must be {GRID_SIZE}x{GRID_SIZE}")
  return np.where(arr > 0, 1, 0)


def solve_nonogram(solution=None, seed=None, max_steps=None):
  model, meta = load_model()
  clue_max_len = meta["clue_max_len"]

  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

  if solution is None:
    sols, row_batch, col_batch, _ = generate_unique_nonogram(GRID_SIZE, 1, _EXISTING_SOLUTIONS)
    solution = sols[0]
    row_clues = row_batch[0]
    col_clues = col_batch[0]
  else:
    solution = normalize_solution(solution)
    row_clues, col_clues = build_clues(solution)

  if len(row_clues) == 1 and isinstance(row_clues[0], list):
    if len(row_clues[0]) == GRID_SIZE and all(isinstance(x, list) for x in row_clues[0]):
      row_clues = row_clues[0]
  if len(col_clues) == 1 and isinstance(col_clues[0], list):
    if len(col_clues[0]) == GRID_SIZE and all(isinstance(x, list) for x in col_clues[0]):
      col_clues = col_clues[0]

  row_pad = np.array([[pad_clue(c, clue_max_len) for c in row_clues]], dtype=np.int64)
  col_pad = np.array([[pad_clue(c, clue_max_len) for c in col_clues]], dtype=np.int64)

  state = np.full((1, GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
  chosen = set()
  steps = []
  step_limit = max_steps or GRID_SIZE * GRID_SIZE
  step_count = 0
  done = False

  while not done:
    logits = model(
      torch.tensor(state, dtype=torch.float32),
      torch.tensor(row_pad, dtype=torch.long),
      torch.tensor(col_pad, dtype=torch.long)
    )
    probs = torch.softmax(logits.view(logits.size(0), -1), dim=-1)
    dist = torch.distributions.Categorical(probs)
    selected = int(dist.sample()[0].item())

    pos = selected // 2
    pred = selected % 2
    row = pos // GRID_SIZE
    col = pos % GRID_SIZE
    actual = int(solution[row, col])
    duplicate = (row, col) in chosen
    if not duplicate:
      chosen.add((row, col))
      state[0, row, col] = actual

    step_count += 1
    result = "Duplicate" if duplicate else "Correct"
    steps.append({
      "row": row,
      "col": col,
      "predicted": pred,
      "actual": actual,
      "correct": not duplicate,
      "duplicate": duplicate,
      "result": result
    })

    done = np.array_equal(state[0], solution) or step_count >= step_limit

  correct_count = sum(1 for s in steps if s["correct"])
  solved = bool(done)
  return {
    "grid": GRID_SIZE,
    "row_clues": row_clues,
    "col_clues": col_clues,
    "solution": solution.tolist(),
    "steps": steps,
    "step_count": step_count,
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


def handler(event, context):
  start = time.time()
  method = (
    (event.get("requestContext", {}).get("http", {}).get("method"))
    or event.get("httpMethod")
    or "GET"
  )
  response_headers = {
    "Content-Type": "application/json"
  }

  if method == "OPTIONS":
    return {
      "statusCode": 204
    }

  if method == "GET":
    model_loaded = _MODEL is not None
    return {
      "statusCode": 200,
      "headers": response_headers,
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

  try:
    result = solve_nonogram(payload.get("solution"), seed=seed)
    duration_ms = int((time.time() - start) * 1000)
    result["duration_ms"] = duration_ms
    return {
      "statusCode": 200,
      "headers": response_headers,
      "body": json.dumps(result)
    }
  except Exception as exc:
    return {
      "statusCode": 500,
      "headers": response_headers,
      "body": json.dumps({
        "error": str(exc)
      })
    }
