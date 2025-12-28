import base64
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PREFIX = os.getenv("MODEL_PREFIX", "Guesser_tiny_transformer_20251228-005944_final")
DEFAULT_GRID_SIZE = int(os.getenv("GRID_SIZE", "9"))
MINE_RATIO_ENV = os.getenv("MINE_RATIO")
try:
  DEFAULT_MINE_RATIO = float(MINE_RATIO_ENV) if MINE_RATIO_ENV else 0.0
except ValueError:
  DEFAULT_MINE_RATIO = 0.0
DEFAULT_MINE_COUNT = int(os.getenv("MINE_COUNT", "0"))
MAX_GUESS_COMPONENT = int(os.getenv("GUESS_COMPONENT_MAX", "18"))
EPS = 1e-9

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


class DistributionalCNN(nn.Module):
  def __init__(self, input_channels=1, output_size=25, num_atoms=51, cnn_variant="adaptive_pool", dueling=False, grid_size=5):
    super().__init__()
    self.dueling = dueling
    self.cnn_variant = cnn_variant
    self.output_size = output_size
    self.num_atoms = num_atoms

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
        nn.Linear(512, self.num_atoms)
      )
      self.adv_fc = nn.Sequential(
        nn.Linear(fc_input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, self.output_size * self.num_atoms)
      )
    else:
      self.fc = nn.Sequential(
        nn.Linear(fc_input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, self.output_size * self.num_atoms)
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
    else:
      x = self.pool(x)
      x = x.view(x.size(0), -1)

    if self.dueling:
      value = self.value_fc(x).view(-1, 1, self.num_atoms)
      adv = self.adv_fc(x).view(-1, self.output_size, self.num_atoms)
      return value + adv - adv.mean(dim=1, keepdim=True)

    return self.fc(x).view(-1, self.output_size, self.num_atoms)


class TinyTransformer(nn.Module):
  def __init__(self, input_channels=1, output_size=25, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, dueling=False, grid_size=5):
    super().__init__()
    self.dueling = dueling
    self.output_size = output_size
    self.grid_size = grid_size
    self.input_channels = input_channels
    self.seq_len = grid_size * grid_size

    if self.output_size != self.seq_len:
      raise ValueError("TinyTransformer output_size must match grid_size * grid_size.")

    self.input_proj = nn.Linear(self.input_channels, d_model)
    self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
    self.dropout = nn.Dropout(dropout)

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      batch_first=True,
      activation="gelu"
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    if self.dueling:
      self.value_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, 1)
      )
      self.adv_head = nn.Linear(d_model, 1)
    else:
      self.head = nn.Linear(d_model, 1)

  def forward(self, x):
    b, c, h, w = x.shape
    if h * w != self.seq_len:
      raise ValueError("Input size does not match transformer grid size.")

    x = x.permute(0, 2, 3, 1).contiguous().view(b, self.seq_len, c)
    x = self.input_proj(x)
    x = self.dropout(x + self.pos_embed)
    x = self.encoder(x)

    if self.dueling:
      pooled = x.mean(dim=1)
      value = self.value_head(pooled)
      adv = self.adv_head(x).squeeze(-1)
      return value + adv - adv.mean(dim=1, keepdim=True)

    return self.head(x).squeeze(-1)


class DistributionalTinyTransformer(nn.Module):
  def __init__(self, input_channels=1, output_size=25, num_atoms=51, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, dueling=False, grid_size=5):
    super().__init__()
    self.dueling = dueling
    self.output_size = output_size
    self.grid_size = grid_size
    self.input_channels = input_channels
    self.num_atoms = num_atoms
    self.seq_len = grid_size * grid_size

    if self.output_size != self.seq_len:
      raise ValueError("DistributionalTinyTransformer output_size must match grid_size * grid_size.")

    self.input_proj = nn.Linear(self.input_channels, d_model)
    self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
    self.dropout = nn.Dropout(dropout)

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      batch_first=True,
      activation="gelu"
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    if self.dueling:
      self.value_head = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, self.num_atoms)
      )
      self.adv_head = nn.Linear(d_model, self.num_atoms)
    else:
      self.head = nn.Linear(d_model, self.num_atoms)

  def forward(self, x):
    b, c, h, w = x.shape
    if h * w != self.seq_len:
      raise ValueError("Input size does not match transformer grid size.")

    x = x.permute(0, 2, 3, 1).contiguous().view(b, self.seq_len, c)
    x = self.input_proj(x)
    x = self.dropout(x + self.pos_embed)
    x = self.encoder(x)

    if self.dueling:
      pooled = x.mean(dim=1)
      value = self.value_head(pooled).view(-1, 1, self.num_atoms)
      adv = self.adv_head(x).view(-1, self.seq_len, self.num_atoms)
      return value + adv - adv.mean(dim=1, keepdim=True)

    return self.head(x).view(-1, self.seq_len, self.num_atoms)


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
    self.state_channels = 1
    self.distributional = False
    self.num_atoms = 51
    self.v_min = None
    self.v_max = None
    self.use_dueling = False
    self.support = None
    self.delta_z = None
    self.transformer_config = None

  def _default_value_range(self, grid_size):
    max_steps = grid_size * grid_size
    step_reward = 0.3
    win_reward = 1.0
    loss_reward = -1.0
    v_max = win_reward + step_reward * (max_steps - 1)
    v_min = loss_reward - step_reward * (max_steps - 1)
    return v_min, v_max

  def _build_model(self):
    if self.cnn_variant == "tiny_transformer":
      cfg = self.transformer_config or {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1
      }
      if self.distributional:
        return DistributionalTinyTransformer(
          input_channels=self.state_channels,
          output_size=self.grid_size * self.grid_size,
          num_atoms=self.num_atoms,
          dueling=self.use_dueling,
          grid_size=self.grid_size,
          **cfg
        )
      return TinyTransformer(
        input_channels=self.state_channels,
        output_size=self.grid_size * self.grid_size,
        dueling=self.use_dueling,
        grid_size=self.grid_size,
        **cfg
      )

    if self.cnn_variant not in ["max_pool", "adaptive_pool", "global_avg"]:
      raise ValueError(f"Unknown model variant: {self.cnn_variant}")

    if self.distributional:
      return DistributionalCNN(
        input_channels=self.state_channels,
        output_size=self.grid_size * self.grid_size,
        num_atoms=self.num_atoms,
        cnn_variant=self.cnn_variant,
        dueling=self.use_dueling,
        grid_size=self.grid_size
      )

    return BaseCNN(
      input_channels=self.state_channels,
      output_size=self.grid_size * self.grid_size,
      cnn_variant=self.cnn_variant,
      dueling=self.use_dueling,
      grid_size=self.grid_size
    )

  def _get_expected_q_values(self, model, state_tensor):
    if self.distributional:
      logits = model(state_tensor)
      probs = F.softmax(logits, dim=2)
      return torch.sum(probs * self.support, dim=2)
    return model(state_tensor)

  def load_model(self, prefix):
    meta_path = f"{prefix}_metadata.json"
    weights_path = f"{prefix}_weights.pt"
    with open(meta_path, "r", encoding="utf-8") as handle:
      meta = json.load(handle)

    self.state_channels = int(meta.get("state_channels", 1))
    self.grid_size = int(meta.get("grid_size", 5))
    self.num_mines = int(meta.get("num_mines", 3))
    self.dqn_variant = meta.get("dqn_variant", "DQN")
    self.cnn_variant = meta.get("cnn_variant", "adaptive_pool")
    self.replay_type = meta.get("replay_type", "regular")
    self.success_rate = float(meta.get("success_rate", 0.0))
    self.transformer_config = meta.get("transformer_config") or {
      "d_model": 64,
      "nhead": 4,
      "num_layers": 2,
      "dim_feedforward": 128,
      "dropout": 0.1
    }

    self.use_dueling = self.dqn_variant in ["DuelingDQN", "Rainbow"]
    self.distributional = meta.get("distributional", self.dqn_variant == "Rainbow")
    self.num_atoms = int(meta.get("num_atoms") or 51)
    self.v_min = meta.get("v_min")
    self.v_max = meta.get("v_max")
    if self.distributional:
      if self.v_min is None or self.v_max is None:
        self.v_min, self.v_max = self._default_value_range(self.grid_size)
      self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
      self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
    else:
      self.support = None
      self.delta_z = None

    model = self._build_model().to(self.device)
    state = torch.load(weights_path, map_location=self.device)
    model.load_state_dict(state)
    model.eval()
    self.model = model

  def get_action_scores(self, state):
    if self.model is None:
      raise RuntimeError("Model not loaded")
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    with torch.no_grad():
      q_values = self._get_expected_q_values(self.model, state_tensor).cpu().numpy().squeeze()
    return q_values

  def act(self, state, board_size=None):
    q_values = self.get_action_scores(state)
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


def enumerate_component(num_vars, constraints):
  if num_vars == 0:
    return 1, []
  var_to_constraints = [[] for _ in range(num_vars)]
  totals = []
  for idx, (vars_in, total) in enumerate(constraints):
    totals.append(int(total))
    for v in vars_in:
      var_to_constraints[v].append(idx)
  rem_sum = totals[:]
  rem_vars = [len(vars_in) for vars_in, _ in constraints]
  order = sorted(range(num_vars), key=lambda v: len(var_to_constraints[v]), reverse=True)
  assignment = [0] * num_vars
  mine_counts = [0] * num_vars
  total_solutions = 0

  def backtrack(depth):
    nonlocal total_solutions
    if depth == num_vars:
      if all(rs == 0 for rs in rem_sum):
        total_solutions += 1
        for v in range(num_vars):
          if assignment[v]:
            mine_counts[v] += 1
      return
    var = order[depth]
    for val in (0, 1):
      ok = True
      changed = []
      for c in var_to_constraints[var]:
        rem_sum[c] -= val
        rem_vars[c] -= 1
        changed.append(c)
        if rem_sum[c] < 0 or rem_sum[c] > rem_vars[c]:
          ok = False
          break
      if ok:
        assignment[var] = val
        backtrack(depth + 1)
        assignment[var] = 0
      for c in changed:
        rem_sum[c] += val
        rem_vars[c] += 1

  backtrack(0)
  return total_solutions, mine_counts


def get_unknown_cells(env):
  unknown_mask = np.logical_not(env.revealed) & np.logical_not(env.flagged)
  coords = np.argwhere(unknown_mask)
  return [tuple(map(int, coord)) for coord in coords]


def _pick_best_with_agent(env, agent, candidates):
  if not candidates:
    return None
  try:
    scores = agent.get_action_scores(env.get_state_representation())
  except Exception:
    return None
  size = env.size
  best = max(candidates, key=lambda coord: scores[coord[0] * size + coord[1]])
  return best


def select_guess_action(env, analysis, agent=None, rng=None):
  if rng is None:
    rng = np.random.default_rng()
  unknown = get_unknown_cells(env)
  if not unknown:
    return None, "none"

  if analysis and analysis.get("mine_probs"):
    frontier = analysis.get("frontier", [])
    mine_probs = analysis.get("mine_probs", [])
    if frontier and mine_probs:
      min_prob = min(mine_probs)
      candidates = [
        frontier[idx]
        for idx, prob in enumerate(mine_probs)
        if abs(prob - min_prob) <= EPS
      ]
      if candidates:
        if agent and len(candidates) > 1:
          best = _pick_best_with_agent(env, agent, candidates)
          if best is not None:
            y, x = best
            return y * env.size + x, "nn_tiebreak"
        y, x = rng.choice(candidates)
        return y * env.size + x, "probability"

  if agent:
    best = _pick_best_with_agent(env, agent, unknown)
    if best is not None:
      y, x = best
      return y * env.size + x, "nn"

  y, x = rng.choice(unknown)
  return y * env.size + x, "random"


def build_constraints(env):
  size = env.size
  revealed = env.revealed
  flagged = env.flagged
  numbers = env.state
  frontier_set = set()
  constraints = []

  for y in range(size):
    for x in range(size):
      if not revealed[y, x]:
        continue
      if numbers[y, x] < 0:
        continue
      unknown = []
      flagged_count = 0
      for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
          if dy == 0 and dx == 0:
            continue
          ny, nx = y + dy, x + dx
          if 0 <= ny < size and 0 <= nx < size:
            if flagged[ny, nx]:
              flagged_count += 1
            elif not revealed[ny, nx]:
              unknown.append((ny, nx))
      if not unknown:
        continue
      total = int(numbers[y, x]) - flagged_count
      if total < 0 or total > len(unknown):
        return None, None
      for cell in unknown:
        frontier_set.add(cell)
      constraints.append((unknown, total))

  frontier = list(frontier_set)
  if not frontier:
    return [], []

  index_map = {coord: idx for idx, coord in enumerate(frontier)}
  indexed_constraints = []
  for neighbors, total in constraints:
    idxs = [index_map[n] for n in neighbors]
    if total < 0 or total > len(idxs):
      return None, None
    indexed_constraints.append((idxs, total))

  return frontier, indexed_constraints


def analyze_board(env, max_component):
  frontier, constraints = build_constraints(env)
  if frontier is None:
    return None
  if not frontier:
    return {"safe": [], "mines": [], "frontier": [], "mine_probs": []}
  if not constraints:
    return None

  parent = list(range(len(frontier)))

  def find(a):
    while parent[a] != a:
      parent[a] = parent[parent[a]]
      a = parent[a]
    return a

  def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
      parent[rb] = ra

  for vars_in, _ in constraints:
    if len(vars_in) < 2:
      continue
    anchor = vars_in[0]
    for v in vars_in[1:]:
      union(anchor, v)

  components = {}
  for idx in range(len(frontier)):
    root = find(idx)
    components.setdefault(root, []).append(idx)

  constraints_by_root = {root: [] for root in components}
  for vars_in, total in constraints:
    root = find(vars_in[0])
    constraints_by_root[root].append((vars_in, total))

  safe = []
  mines = []
  mine_probs = [None] * len(frontier)

  for root, vars_list in components.items():
    if len(vars_list) > max_component:
      return None
    local_index = {v: i for i, v in enumerate(vars_list)}
    local_constraints = []
    for vars_in, total in constraints_by_root[root]:
      local_constraints.append(([local_index[v] for v in vars_in], total))
    total_solutions, mine_counts = enumerate_component(len(vars_list), local_constraints)
    if total_solutions == 0:
      return None
    for v in vars_list:
      prob = mine_counts[local_index[v]] / total_solutions
      mine_probs[v] = prob
      if abs(prob) <= EPS:
        safe.append(frontier[v])
      elif abs(prob - 1.0) <= EPS:
        mines.append(frontier[v])

  if any(prob is None for prob in mine_probs):
    return None

  return {
    "safe": safe,
    "mines": mines,
    "frontier": frontier,
    "mine_probs": mine_probs
  }


def collect_newly_opened(prev_revealed, new_revealed):
  newly_opened = []
  for y in range(new_revealed.shape[0]):
    for x in range(new_revealed.shape[1]):
      if new_revealed[y, x] and not prev_revealed[y, x]:
        newly_opened.append([x, y])
  return newly_opened


def hybrid_step(env, agent, max_component=18, max_forced_steps=1000):
  total_reward = 0.0
  newly_opened = []
  last_action = {}
  message = ""
  forced_steps = 0
  guess_used = False
  guess_source = "none"

  while not env.done:
    analysis = analyze_board(env, max_component)
    if analysis is None:
      break
    safe_moves = analysis["safe"]
    mine_moves = analysis["mines"]
    if not safe_moves and not mine_moves:
      action, guess_source = select_guess_action(env, analysis, agent=agent)
      if action is None:
        break
      guess_used = True
      prev = env.revealed.copy()
      _, reward, done, msg = env.step(action, "reveal")
      total_reward += reward
      last_action = msg.get("last_action", {}) if isinstance(msg, dict) else {}
      message = msg.get("message", "") if isinstance(msg, dict) else str(msg)
      newly_opened.extend(collect_newly_opened(prev, env.revealed))
      return {
        "reward": total_reward,
        "done": bool(done),
        "message": message,
        "last_action": last_action,
        "newly_opened": newly_opened,
        "guess_used": guess_used,
        "guess_source": guess_source,
        "forced_steps": forced_steps
      }

    for y, x in mine_moves:
      if not env.flagged[y, x] and not env.revealed[y, x]:
        env.flagged[y, x] = True

    for y, x in safe_moves:
      if env.revealed[y, x] or env.flagged[y, x]:
        continue
      prev = env.revealed.copy()
      _, reward, done, msg = env.step(y * env.size + x, "reveal")
      total_reward += reward
      last_action = msg.get("last_action", {}) if isinstance(msg, dict) else {}
      message = msg.get("message", "") if isinstance(msg, dict) else str(msg)
      newly_opened.extend(collect_newly_opened(prev, env.revealed))
      if done:
        return {
          "reward": total_reward,
          "done": True,
          "message": message,
          "last_action": last_action,
          "newly_opened": newly_opened,
          "guess_used": guess_used,
          "guess_source": guess_source,
          "forced_steps": forced_steps
        }

    forced_steps += 1
    if forced_steps >= max_forced_steps:
      break

  if not env.done:
    action, guess_source = select_guess_action(env, analysis, agent=agent)
    if action is None:
      return {
        "reward": total_reward,
        "done": bool(env.done),
        "message": message,
        "last_action": last_action,
        "newly_opened": newly_opened,
        "guess_used": guess_used,
        "guess_source": guess_source,
        "forced_steps": forced_steps
      }
    guess_used = True
    prev = env.revealed.copy()
    _, reward, done, msg = env.step(action, "reveal")
    total_reward += reward
    last_action = msg.get("last_action", {}) if isinstance(msg, dict) else {}
    message = msg.get("message", "") if isinstance(msg, dict) else str(msg)
    newly_opened.extend(collect_newly_opened(prev, env.revealed))

  return {
    "reward": total_reward,
    "done": bool(env.done),
    "message": message,
    "last_action": last_action,
    "newly_opened": newly_opened,
    "guess_used": guess_used,
    "guess_source": guess_source,
    "forced_steps": forced_steps
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


def find_best_model(target_grid=None, target_mines=None):
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
    mines = int(meta.get("num_mines", 0))
    if target_grid and grid != target_grid:
      continue
    if target_mines and mines != target_mines:
      continue
    score = float(meta.get("success_rate", 0.0))
    candidate = (score, grid, mines, prefix, meta)
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
    target_grid = DEFAULT_GRID_SIZE if DEFAULT_GRID_SIZE > 0 else None
    target_mines = None
    if DEFAULT_MINE_COUNT > 0:
      target_mines = DEFAULT_MINE_COUNT
    elif DEFAULT_MINE_RATIO > 0 and target_grid:
      target_mines = max(1, int(round(target_grid * target_grid * DEFAULT_MINE_RATIO)))
    best = find_best_model(target_grid, target_mines)
    if best is None:
      best = find_best_model()
    if best is None:
      raise FileNotFoundError("No model metadata found.")
    _, _, _, prefix, meta = best
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


def resolve_grid_size(agent, override=None):
  if override is not None:
    try:
      override = int(override)
    except (TypeError, ValueError):
      override = None
  if override and override > 0:
    return override
  if DEFAULT_GRID_SIZE > 0:
    return DEFAULT_GRID_SIZE
  return agent.grid_size


def resolve_mine_count(grid_size, agent, override=None):
  if override is not None:
    try:
      override = int(override)
    except (TypeError, ValueError):
      override = None
  if override and override > 0:
    num_mines = override
  elif DEFAULT_MINE_COUNT > 0:
    num_mines = DEFAULT_MINE_COUNT
  elif DEFAULT_MINE_RATIO > 0:
    num_mines = max(1, int(round(grid_size * grid_size * DEFAULT_MINE_RATIO)))
  else:
    num_mines = agent.num_mines
  max_mines = max(1, (grid_size * grid_size) - 1)
  return min(num_mines, max_mines)


def solve_game(seed=None, max_steps=None, grid_size=None, num_mines=None):
  agent, meta = load_agent()
  grid_size = resolve_grid_size(agent, grid_size)
  num_mines = resolve_mine_count(grid_size, agent, num_mines)
  env = MinesweeperEnv(size=grid_size, num_mines=num_mines, seed=seed)
  steps = []
  solution = None
  safe_moves = 0
  hit_mine = False
  step_limit = max_steps or (grid_size * grid_size * 2)

  for _ in range(step_limit):
    result = hybrid_step(env, agent, max_component=MAX_GUESS_COMPONENT)
    last = result.get("last_action") or {}
    newly_opened = result.get("newly_opened", [])
    done = bool(result.get("done"))
    if not last and not newly_opened and not done:
      break
    row = int(last.get("y", 0))
    col = int(last.get("x", 0))
    if solution is None and not env.first_click:
      solution = env.state.tolist()

    if env.revealed[row, col] and env.state[row, col] == -1:
      hit_mine = True

    decision = "logic"
    if result.get("guess_used"):
      source = result.get("guess_source") or "unknown"
      decision = f"guess:{source}"

    reward = result.get("reward", 0.0)
    if reward > 0:
      safe_moves += 1

    steps.append({
      "row": row,
      "col": col,
      "reward": float(reward),
      "done": bool(done),
      "message": result.get("message", ""),
      "newly_opened": newly_opened,
      "hit_mine": bool(hit_mine and done),
      "decision": decision
    })

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
    grid_size = resolve_grid_size(agent)
    num_mines = resolve_mine_count(grid_size, agent)
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
  grid_size = payload.get("grid_size") or payload.get("grid") or payload.get("size")
  num_mines = payload.get("num_mines") or payload.get("mines") or payload.get("mine_count")
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
  if grid_size is not None:
    try:
      grid_size = int(grid_size)
    except (TypeError, ValueError):
      grid_size = None
  if num_mines is not None:
    try:
      num_mines = int(num_mines)
    except (TypeError, ValueError):
      num_mines = None

  try:
    result = solve_game(seed=seed, max_steps=max_steps, grid_size=grid_size, num_mines=num_mines)
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
