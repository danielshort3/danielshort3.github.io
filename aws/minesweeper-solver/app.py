import base64
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PREFIX = os.getenv("MODEL_PREFIX", "")
DEFAULT_GRID_SIZE = int(os.getenv("GRID_SIZE", "9"))
DEFAULT_MINE_RATIO = float(os.getenv("MINE_RATIO", "0.2"))

torch.set_grad_enabled(False)
torch.set_num_threads(1)


class BaseCNN(nn.Module):
  def __init__(self, input_channels=1, output_size=25, cnn_variant="adaptive_pool", dueling=False, grid_size=5):
    super().__init__()
    self.dueling = dueling
    self.cnn_variant = cnn_variant
    self.output_size = output_size

    self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    if self.cnn_variant == "max_pool":
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    elif self.cnn_variant == "adaptive_pool":
      self.pool = nn.AdaptiveAvgPool2d((5, 5))
    elif self.cnn_variant == "global_avg":
      self.pool = None
    else:
      raise ValueError(f"Unknown CNN variant: {self.cnn_variant}")

    fc_input_dim = self._compute_fc_input_dim(grid_size)

    if self.dueling:
      self.value_fc = nn.Sequential(
        nn.Linear(fc_input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1)
      )
      self.adv_fc = nn.Sequential(
        nn.Linear(fc_input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, self.output_size)
      )
    else:
      self.fc = nn.Sequential(
        nn.Linear(fc_input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, self.output_size)
      )

  def _compute_fc_input_dim(self, grid_size):
    with torch.no_grad():
      dummy = torch.zeros(1, 1, grid_size, grid_size)
      x = F.relu(self.conv1(dummy))
      x = F.relu(self.conv2(x))
      x = F.relu(self.conv3(x))
      x = F.relu(self.conv4(x))
      if self.cnn_variant == "global_avg":
        x = x.mean(dim=[2, 3])
        return x.shape[1]
      x = self.pool(x)
      return x.numel()

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))

    if self.cnn_variant == "global_avg":
      x = x.mean(dim=[2, 3])
      if self.dueling:
        value = self.value_fc(x)
        adv = self.adv_fc(x)
        return value + adv - adv.mean(dim=1, keepdim=True)
      return self.fc(x)

    x = self.pool(x)
    x = x.view(x.size(0), -1)
    if self.dueling:
      value = self.value_fc(x)
      adv = self.adv_fc(x)
      return value + adv - adv.mean(dim=1, keepdim=True)
    return self.fc(x)


class InferenceAgent:
  def __init__(self, device="cpu"):
    self.device = device
    self.model = None
    self.grid_size = 5
    self.num_mines = 3
    self.dqn_variant = "DQN"
    self.cnn_variant = "adaptive_pool"
    self.replay_type = "regular"
    self.success_rate = 0.0
    self.epsilon = 0.0

  def load_model(self, prefix):
    meta_path = f"{prefix}_metadata.json"
    weights_path = f"{prefix}_weights.pt"
    with open(meta_path, "r", encoding="utf-8") as handle:
      meta = json.load(handle)

    self.grid_size = int(meta.get("grid_size", 5))
    self.num_mines = int(meta.get("num_mines", 3))
    self.dqn_variant = meta.get("dqn_variant", "DQN")
    self.cnn_variant = meta.get("cnn_variant", "adaptive_pool")
    self.replay_type = meta.get("replay_type", "regular")
    self.success_rate = float(meta.get("success_rate", 0.0))

    dueling = (self.dqn_variant == "DuelingDQN")
    model = BaseCNN(
      input_channels=1,
      output_size=self.grid_size * self.grid_size,
      cnn_variant=self.cnn_variant,
      dueling=dueling,
      grid_size=self.grid_size
    ).to(self.device)
    state = torch.load(weights_path, map_location=self.device)
    model.load_state_dict(state)
    model.eval()
    self.model = model

  def act(self, state, board_size=None):
    if self.model is None:
      raise RuntimeError("Model not loaded")
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    with torch.no_grad():
      q_values = self.model(state_tensor).cpu().numpy().squeeze()
    board = state[0]
    board_size = board_size or board.shape[0]
    if q_values.size != board_size * board_size:
      model_grid = self.grid_size
      try:
        q_grid = q_values.reshape(model_grid, model_grid)
      except ValueError:
        q_values = q_values[:board_size * board_size]
      else:
        if board_size < model_grid:
          q_grid = q_grid[:board_size, :board_size]
        elif board_size > model_grid:
          fill = np.min(q_grid)
          padded = np.full((board_size, board_size), fill)
          padded[:model_grid, :model_grid] = q_grid
          q_grid = padded
        q_values = q_grid.flatten()
    flat_board = board.flatten()
    revealed_mask = (np.abs(flat_board + 0.125) > 1e-6)
    if revealed_mask.all():
      return int(np.argmax(q_values))
    q_values[revealed_mask] = np.min(q_values)
    return int(np.argmax(q_values))


class MinesweeperEnv:
  def __init__(self, size=5, num_mines=None, seed=None):
    self.size = size
    self.num_mines = num_mines
    self.rng = np.random.default_rng(seed)
    self.state = None
    self.mines = None
    self.revealed = None
    self.flagged = None
    self.first_click = True
    self.done = False
    self.reset(seed)

  def reset(self, seed=None):
    if seed is not None:
      self.rng = np.random.default_rng(seed)
    total_cells = self.size * self.size
    min_mines = int(np.ceil(0.1 * total_cells))
    max_mines = int(np.ceil(0.2 * total_cells))
    if self.num_mines is None:
      self.num_mines = int(self.rng.integers(min_mines, max_mines + 1))
    self.state = np.zeros((self.size, self.size), dtype=np.int32)
    self.revealed = np.zeros((self.size, self.size), dtype=bool)
    self.flagged = np.zeros((self.size, self.size), dtype=bool)
    self.mines = np.zeros((self.size, self.size), dtype=bool)
    self.first_click = True
    self.done = False
    return self.get_state_representation()

  def place_mines(self, first_action):
    y0, x0 = divmod(first_action, self.size)
    safe_zone = [
      (y, x)
      for y in range(max(0, y0 - 1), min(self.size, y0 + 2))
      for x in range(max(0, x0 - 1), min(self.size, x0 + 2))
    ]
    num_placed = 0
    while num_placed < self.num_mines:
      y, x = self.rng.integers(0, self.size), self.rng.integers(0, self.size)
      if (y, x) not in safe_zone and not self.mines[y, x]:
        self.mines[y, x] = True
        num_placed += 1

    for y in range(self.size):
      for x in range(self.size):
        if self.mines[y, x]:
          self.state[y, x] = -1
        else:
          self.state[y, x] = self.count_adjacent_mines(y, x)

  def count_adjacent_mines(self, y, x):
    count = 0
    for dy in [-1, 0, 1]:
      for dx in [-1, 0, 1]:
        ny, nx = y + dy, x + dx
        if (dy != 0 or dx != 0) and 0 <= ny < self.size and 0 <= nx < self.size:
          if self.mines[ny, nx]:
            count += 1
    return count

  def cascade(self, y, x):
    stack = [(y, x)]
    while stack:
      cy, cx = stack.pop()
      for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
          ny, nx = cy + dy, cx + dx
          if dy == 0 and dx == 0:
            continue
          if 0 <= ny < self.size and 0 <= nx < self.size:
            if not self.revealed[ny, nx] and not self.flagged[ny, nx]:
              self.revealed[ny, nx] = True
              if self.state[ny, nx] == 0:
                stack.append((ny, nx))

  def reveal_3x3_block(self, y0, x0):
    for yy in range(max(0, y0 - 1), min(self.size, y0 + 2)):
      for xx in range(max(0, x0 - 1), min(self.size, x0 + 2)):
        if not self.revealed[yy, xx] and not self.flagged[yy, xx]:
          self.revealed[yy, xx] = True
    for yy in range(max(0, y0 - 1), min(self.size, y0 + 2)):
      for xx in range(max(0, x0 - 1), min(self.size, x0 + 2)):
        if self.revealed[yy, xx] and self.state[yy, xx] == 0:
          self.cascade(yy, xx)

  def reveal_cell(self, y, x):
    if self.revealed[y, x]:
      return 0.0, False, "Cell already revealed."
    if self.flagged[y, x]:
      return 0.0, False, "Cannot reveal a flagged cell."
    if self.first_click:
      first_action = y * self.size + x
      self.place_mines(first_action)
      self.first_click = False
      self.reveal_3x3_block(y, x)
      return 0.3, False, "Opening safe zone."

    if self.mines[y, x]:
      self.revealed[y, x] = True
      return -1.0, True, "Mine triggered."

    self.revealed[y, x] = True
    if self.state[y, x] == 0:
      self.cascade(y, x)

    if np.sum(self.revealed) + self.num_mines == self.size * self.size:
      return 1.0, True, "Board cleared."
    return 0.3, False, "Safe move."

  def step(self, action, action_type="reveal"):
    y, x = divmod(action, self.size)
    if self.done:
      return self.get_state_representation(), 0.0, True, {
        "message": "Game is already over.",
        "last_action": None
      }

    if action_type == "reveal":
      reward, done, msg = self.reveal_cell(y, x)
    else:
      reward, done, msg = 0.0, False, "Invalid action type."

    self.done = done
    return self.get_state_representation(), reward, done, {
      "message": msg,
      "last_action": {
        "x": x,
        "y": y,
        "action_type": action_type
      }
    }

  def get_state_representation(self):
    cell_values = np.where(self.revealed, self.state, 9)
    mapped = np.zeros_like(cell_values, dtype=float)
    for y in range(self.size):
      for x in range(self.size):
        v = cell_values[y, x]
        if v == 9:
          mapped[y, x] = -1.0
        elif v == -1:
          mapped[y, x] = -2.0
        else:
          mapped[y, x] = float(v)
    scaled = mapped / 8.0
    return scaled[np.newaxis, :, :]


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


def find_best_model():
  best = None
  for name in os.listdir(MODEL_DIR):
    if not name.endswith("_metadata.json"):
      continue
    prefix = os.path.join(MODEL_DIR, name.replace("_metadata.json", ""))
    try:
      with open(prefix + "_metadata.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    except Exception:
      continue
    grid = int(meta.get("grid_size", 0))
    score = float(meta.get("success_rate", 0.0))
    candidate = (grid, score, prefix, meta)
    if best is None or candidate > best:
      best = candidate
  return best


def load_agent():
  global _AGENT, _MODEL_META
  if "_AGENT" in globals() and globals().get("_AGENT") is not None:
    return globals()["_AGENT"], globals()["_MODEL_META"]

  agent = InferenceAgent(device="cpu")
  meta = None
  if DEFAULT_MODEL_PREFIX:
    prefix = os.path.join(MODEL_DIR, DEFAULT_MODEL_PREFIX)
    agent.load_model(prefix)
    meta = {
      "prefix": DEFAULT_MODEL_PREFIX,
      "grid_size": agent.grid_size,
      "num_mines": agent.num_mines,
      "dqn_variant": agent.dqn_variant,
      "cnn_variant": agent.cnn_variant,
      "replay_type": agent.replay_type,
      "success_rate": agent.success_rate
    }
  else:
    best = find_best_model()
    if best is None:
      raise FileNotFoundError("No model metadata found.")
    _, _, prefix, meta = best
    agent.load_model(prefix)
    meta = {
      "prefix": os.path.basename(prefix),
      "grid_size": agent.grid_size,
      "num_mines": agent.num_mines,
      "dqn_variant": agent.dqn_variant,
      "cnn_variant": agent.cnn_variant,
      "replay_type": agent.replay_type,
      "success_rate": agent.success_rate
    }

  globals()["_AGENT"] = agent
  globals()["_MODEL_META"] = meta
  return agent, meta


def solve_game(seed=None, max_steps=None):
  agent, meta = load_agent()
  grid_size = DEFAULT_GRID_SIZE if DEFAULT_GRID_SIZE > 0 else agent.grid_size
  num_mines = agent.num_mines
  if DEFAULT_MINE_RATIO > 0:
    num_mines = max(1, int(round(grid_size * grid_size * DEFAULT_MINE_RATIO)))
  env = MinesweeperEnv(size=grid_size, num_mines=num_mines, seed=seed)
  state = env.get_state_representation()
  steps = []
  solution = None
  safe_moves = 0
  hit_mine = False
  step_limit = max_steps or (grid_size * grid_size * 2)

  for _ in range(step_limit):
    action = agent.act(state, board_size=grid_size)
    prev_revealed = env.revealed.copy()
    next_state, reward, done, info = env.step(action, "reveal")
    last = info.get("last_action") or {}
    row = int(last.get("y", action // grid_size))
    col = int(last.get("x", action % grid_size))
    if solution is None and not env.first_click:
      solution = env.state.tolist()

    newly_opened = []
    newly = np.logical_and(env.revealed, np.logical_not(prev_revealed))
    for y, x in zip(*np.where(newly)):
      newly_opened.append([int(x), int(y)])

    if env.revealed[row, col] and env.state[row, col] == -1:
      hit_mine = True

    if reward > 0:
      safe_moves += 1

    steps.append({
      "row": row,
      "col": col,
      "reward": float(reward),
      "done": bool(done),
      "message": info.get("message", ""),
      "newly_opened": newly_opened,
      "hit_mine": bool(hit_mine and done)
    })

    state = next_state
    if done:
      break

  if solution is None:
    solution = env.state.tolist()

  return {
    "grid": grid_size,
    "mines": int(num_mines),
    "solution": solution,
    "steps": steps,
    "step_count": len(steps),
    "safe_moves": safe_moves,
    "hit_mine": bool(hit_mine),
    "success": bool(not hit_mine and env.done),
    "model": meta
  }


def handler(event, context):
  start = time.time()
  method = (
    (event.get("requestContext", {}).get("http", {}).get("method"))
    or event.get("httpMethod")
    or "GET"
  )
  path = event.get("rawPath") or event.get("path") or "/"
  if path:
    path = path.rstrip("/") or "/"

  response_headers = {
    "Content-Type": "application/json"
  }

  if method == "OPTIONS":
    return {"statusCode": 204}

  if method == "GET":
    agent, meta = load_agent()
    grid_size = DEFAULT_GRID_SIZE if DEFAULT_GRID_SIZE > 0 else agent.grid_size
    num_mines = agent.num_mines
    if DEFAULT_MINE_RATIO > 0:
      num_mines = max(1, int(round(grid_size * grid_size * DEFAULT_MINE_RATIO)))
    return {
      "statusCode": 200,
      "headers": response_headers,
      "body": json.dumps({
        "status": "ok",
        "model": meta,
        "grid": grid_size,
        "mines": num_mines
      })
    }

  payload = parse_body(event)
  seed = payload.get("seed")
  if seed is not None:
    try:
      seed = int(seed)
    except (TypeError, ValueError):
      seed = None
  max_steps = payload.get("max_steps")
  if max_steps is not None:
    try:
      max_steps = int(max_steps)
    except (TypeError, ValueError):
      max_steps = None

  try:
    result = solve_game(seed=seed, max_steps=max_steps)
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
