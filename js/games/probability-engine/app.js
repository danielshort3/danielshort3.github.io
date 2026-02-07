(() => {
  "use strict";

  const PE = window.ProbabilityEngine = window.ProbabilityEngine || {};
  const {
    SAVE_KEY,
    AUTOSAVE_MS,
    TICK_MS,
    LIVE_UI_REFRESH_MS,
    DECK_EV_BASELINE_SAMPLES,
    DECK_EV_DELTA_SAMPLES,
    MIN_DECK_SIZE,
    MAX_DECK_SIZE,
    PITY_LIMIT,
    BASE_STARTING_CASH,
    Big,
    RARITY_ORDER,
    RARITY_SCORE,
    SYMBOLS,
    PACK_RARITY_SYMBOLS,
    UPGRADE_DEFS,
    INGREDIENTS,
    BUFFS,
    RECIPES,
    SKILL_TREE
  } = PE;

  if (!Big || !SYMBOLS || !UPGRADE_DEFS || !RECIPES || !SKILL_TREE) {
    throw new Error("Probability Engine dependencies missing. Check script load order.");
  }

const dom = {
  body: document.body,
  stats: document.getElementById("stats"),
  selectedMachineTitle: document.getElementById("selected-machine-title"),
  eraText: document.getElementById("era-text"),
  spinButton: document.getElementById("spin-button"),
  autoButton: document.getElementById("auto-button"),
  buyMachineButton: document.getElementById("buy-machine-button"),
  gridWrapper: document.getElementById("grid-wrapper"),
  slotGrid: document.getElementById("slot-grid"),
  particles: document.getElementById("particles"),
  lockRow: document.getElementById("lock-row"),
  machineMeta: document.getElementById("machine-meta"),
  machineList: document.getElementById("machine-list"),
  deckList: document.getElementById("deck-list"),
  deckEvSummary: document.getElementById("deck-ev-summary"),
  inventoryList: document.getElementById("inventory-list"),
  packCost: document.getElementById("pack-cost"),
  packOdds: document.getElementById("pack-odds"),
  packPreview: document.getElementById("pack-preview"),
  buyPackButton: document.getElementById("buy-pack-button"),
  upgradeList: document.getElementById("upgrade-list"),
  recipeList: document.getElementById("recipe-list"),
  ingredientShop: document.getElementById("ingredient-shop"),
  buffList: document.getElementById("buff-list"),
  prestigeGain: document.getElementById("prestige-gain"),
  prestigeButton: document.getElementById("prestige-button"),
  skillTree: document.getElementById("skill-tree"),
  eventLog: document.getElementById("event-log"),
  offlineModal: document.getElementById("offline-modal"),
  offlineSummary: document.getElementById("offline-summary"),
  claimOfflineButton: document.getElementById("claim-offline-button")
};
let lastLiveUiRenderAt = 0;
let deckInsightCacheKey = "";
let deckInsightCacheValue = null;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function randomInt(maxExclusive) {
  return Math.floor(Math.random() * maxExclusive);
}

function chooseWeighted(items) {
  const total = items.reduce((sum, item) => sum + item.weight, 0);
  if (total <= 0) {
    return items[0].value;
  }
  let target = Math.random() * total;
  for (const item of items) {
    target -= item.weight;
    if (target <= 0) {
      return item.value;
    }
  }
  return items[items.length - 1].value;
}

function formatBig(value, decimals = 2) {
  const big = Big.from(value);
  if (big.isZero()) {
    return "0";
  }
  const abs = big.abs();
  if (abs.e < 6) {
    const number = big.toNumber();
    if (Number.isFinite(number)) {
      const rounded = number.toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
      });
      return rounded;
    }
  }
  return `${big.m.toFixed(decimals)}e${big.e}`;
}

function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSeconds(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  }
  if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}

function getSymbolById(id) {
  return SYMBOLS[id];
}

function rarityClass(rarity) {
  return `rarity-${rarity.toLowerCase()}`;
}

function rarityScore(symbolId) {
  const symbol = getSymbolById(symbolId);
  return RARITY_SCORE[symbol.rarity] || 0;
}

function createDeckPreset(machineIndex) {
  if (machineIndex === 1) {
    return [
      "sheep", "sheep", "sprout", "sprout", "farmer",
      "coin", "coin", "gear", "wolf", "battery"
    ];
  }
  if (machineIndex === 2) {
    return [
      "gear", "gear", "battery", "battery", "magnet",
      "coin", "coal", "diamond", "reactor", "wolf"
    ];
  }
  return [
    "coal", "coal", "coin", "coin", "sheep",
    "sprout", "gear", "gear", "wolf", "battery"
  ];
}

function createStartingInventory() {
  return {
    coal: 3,
    coin: 3,
    sheep: 2,
    sprout: 2,
    gear: 2,
    battery: 1,
    wolf: 1,
    magnet: 1,
    farmer: 1,
    bomb: 1,
    diamond: 0,
    reactor: 0,
    chronos: 0,
    oracle: 0,
    singularity: 0
  };
}

function createMachineState(index, owned) {
  return {
    id: index,
    name: `Machine ${index + 1}`,
    owned,
    deck: createDeckPreset(index),
    grid: [],
    spinning: false,
    spinTimer: null,
    spinInterval: null,
    lastSpinAt: 0,
    lastReward: Big.zero(),
    averageWin: Big.from(9),
    patrons: index + 1,
    badLuck: 0,
    jackpotReady: false,
    lockedCols: [false, false, false, false],
    totalEarned: Big.zero(),
    totalSynergies: 0
  };
}

function getOwnedMachineCount(state) {
  return state.machines.filter((machine) => machine.owned).length;
}

function getGridSize(state) {
  return state.skillTree.grid4 ? 4 : 3;
}

function ensureGrid(state, machine) {
  const size = getGridSize(state);
  const total = size * size;
  if (machine.grid.length !== total) {
    machine.grid = new Array(total).fill(machine.deck[0] || "coin");
  }
}

function startingCashAmount(state) {
  return Big.from(BASE_STARTING_CASH).mul(state.skillTree.startingCash ? 1.1 : 1);
}

function createInitialState(options = {}) {
  const machines = [
    createMachineState(0, true),
    createMachineState(1, false),
    createMachineState(2, false)
  ];

  const state = {
    version: 1,
    wallet: startingCashAmount({ skillTree: options.skillTree || { startingCash: false } }),
    influence: Big.zero(),
    shards: Big.zero(),
    chips: options.chips || 0,
    lifetimeCash: options.lifetimeCash ? Big.from(options.lifetimeCash) : Big.zero(),
    runCash: Big.zero(),
    totalSpins: 0,
    avgWin: Big.from(12),
    entropy: 0,
    autoSpin: false,
    selectedMachine: 0,
    upgrades: {
      spinSpeed: 0,
      luck: 0,
      autoRate: 0
    },
    ingredients: {
      copper: 0,
      herb: 0,
      neon: 0
    },
    buffs: {
      liquidLuckEnd: 0,
      overclockEnd: 0
    },
    skillTree: {
      startingCash: false,
      symbolLocking: false,
      grid4: false,
      ...(options.skillTree || {})
    },
    inventory: {
      ...createStartingInventory(),
      ...(options.legendaryInventory || {})
    },
    packPurchases: 0,
    logs: [],
    offline: {
      pendingGain: Big.zero(),
      awaySeconds: 0
    },
    lastTick: Date.now(),
    lastSave: Date.now(),
    machinePurchaseProgress: 1,
    machines
  };

  for (const machine of state.machines) {
    ensureGrid(state, machine);
    machine.grid = generateRandomGrid(state, machine, false);
  }

  return state;
}

function serializeMachine(machine) {
  return {
    id: machine.id,
    name: machine.name,
    owned: machine.owned,
    deck: machine.deck,
    grid: machine.grid,
    lastSpinAt: machine.lastSpinAt,
    lastReward: machine.lastReward.toArray(),
    averageWin: machine.averageWin.toArray(),
    patrons: machine.patrons,
    badLuck: machine.badLuck,
    jackpotReady: machine.jackpotReady,
    lockedCols: machine.lockedCols,
    totalEarned: machine.totalEarned.toArray(),
    totalSynergies: machine.totalSynergies
  };
}

function deserializeMachine(raw, fallback) {
  const machine = {
    ...fallback,
    ...raw,
    deck: Array.isArray(raw.deck) ? raw.deck.filter((id) => SYMBOLS[id]) : fallback.deck,
    grid: Array.isArray(raw.grid) ? raw.grid.filter((id) => SYMBOLS[id]) : fallback.grid,
    lastReward: Big.from(raw.lastReward || fallback.lastReward),
    averageWin: Big.from(raw.averageWin || fallback.averageWin),
    totalEarned: Big.from(raw.totalEarned || fallback.totalEarned),
    spinTimer: null,
    spinInterval: null
  };
  return machine;
}

function saveGame(state) {
  try {
    const payload = {
      version: state.version,
      wallet: state.wallet.toArray(),
      influence: state.influence.toArray(),
      shards: state.shards.toArray(),
      chips: state.chips,
      lifetimeCash: state.lifetimeCash.toArray(),
      runCash: state.runCash.toArray(),
      totalSpins: state.totalSpins,
      avgWin: state.avgWin.toArray(),
      entropy: state.entropy,
      autoSpin: state.autoSpin,
      selectedMachine: state.selectedMachine,
      upgrades: state.upgrades,
      ingredients: state.ingredients,
      buffs: state.buffs,
      skillTree: state.skillTree,
      inventory: state.inventory,
      packPurchases: state.packPurchases,
      logs: state.logs.slice(0, 60),
      machinePurchaseProgress: state.machinePurchaseProgress,
      machines: state.machines.map(serializeMachine),
      timestamp: Date.now()
    };
    localStorage.setItem(SAVE_KEY, JSON.stringify(payload));
    state.lastSave = Date.now();
  } catch (error) {
    console.error("Save failed", error);
  }
}

function loadGame() {
  const rawValue = localStorage.getItem(SAVE_KEY);
  if (!rawValue) {
    return createInitialState();
  }

  try {
    const parsed = JSON.parse(rawValue);
    const fallback = createInitialState();
    const state = {
      ...fallback,
      version: parsed.version || fallback.version,
      wallet: Big.from(parsed.wallet || fallback.wallet),
      influence: Big.from(parsed.influence || fallback.influence),
      shards: Big.from(parsed.shards || fallback.shards),
      chips: Number(parsed.chips || 0),
      lifetimeCash: Big.from(parsed.lifetimeCash || fallback.lifetimeCash),
      runCash: Big.from(parsed.runCash || fallback.runCash),
      totalSpins: Number(parsed.totalSpins || 0),
      avgWin: Big.from(parsed.avgWin || fallback.avgWin),
      entropy: Number(parsed.entropy || 0),
      autoSpin: Boolean(parsed.autoSpin),
      selectedMachine: Number(parsed.selectedMachine || 0),
      upgrades: {
        ...fallback.upgrades,
        ...(parsed.upgrades || {})
      },
      ingredients: {
        ...fallback.ingredients,
        ...(parsed.ingredients || {})
      },
      buffs: {
        ...fallback.buffs,
        ...(parsed.buffs || {})
      },
      skillTree: {
        ...fallback.skillTree,
        ...(parsed.skillTree || {})
      },
      inventory: {
        ...fallback.inventory,
        ...(parsed.inventory || {})
      },
      packPurchases: Number(parsed.packPurchases || 0),
      logs: Array.isArray(parsed.logs) ? parsed.logs.slice(0, 60) : [],
      machinePurchaseProgress: Number(parsed.machinePurchaseProgress || 1),
      machines: fallback.machines,
      offline: {
        pendingGain: Big.zero(),
        awaySeconds: 0
      }
    };

    if (Array.isArray(parsed.machines)) {
      for (let i = 0; i < state.machines.length; i += 1) {
        const fallbackMachine = state.machines[i];
        const rawMachine = parsed.machines[i];
        if (rawMachine) {
          state.machines[i] = deserializeMachine(rawMachine, fallbackMachine);
        }
      }
    }

    state.selectedMachine = clamp(state.selectedMachine, 0, state.machines.length - 1);
    for (const machine of state.machines) {
      ensureGrid(state, machine);
    }

    const awaySeconds = Math.max(0, Math.floor((Date.now() - Number(parsed.timestamp || Date.now())) / 1000));
    state.offline = {
      pendingGain: estimateOfflineGain(state, awaySeconds),
      awaySeconds
    };

    state.lastTick = Date.now();
    return state;
  } catch (error) {
    console.error("Load failed", error);
    return createInitialState();
  }
}

function getUpgradeCost(type, state) {
  const definition = UPGRADE_DEFS[type];
  const level = state.upgrades[type] || 0;
  return Big.from(definition.base).mul(Big.from(definition.growth).pow(level));
}

function getPackCost(state) {
  return Big.from(34).mul(Big.from(1.17).pow(state.packPurchases));
}

function getMachineCost(index) {
  if (index === 1) {
    return Big.from(850);
  }
  if (index === 2) {
    return Big.from(19000);
  }
  return Big.zero();
}

function getSpinCost(state, machineIndex) {
  const size = getGridSize(state);
  const base = 5 + machineIndex * 2;
  const scale = size === 4 ? 1.7 : 1;
  return Big.from(base * scale);
}

function getSpinDelayMs(state) {
  const level = state.upgrades.spinSpeed || 0;
  const overclockMult = isBuffActive(state, "overclock") ? 0.7 : 1;
  const delay = 1150 * Math.pow(0.86, level) * overclockMult;
  return Math.max(130, Math.floor(delay));
}

function getAutoSpinIntervalMs(state) {
  const level = state.upgrades.autoRate || 0;
  const overclockMult = isBuffActive(state, "overclock") ? 0.75 : 1;
  const interval = 1050 * Math.pow(0.9, level) * overclockMult;
  return Math.max(220, Math.floor(interval));
}

function getLuckBias(state) {
  const base = 0.05 + (state.upgrades.luck || 0) * 0.04;
  const buff = isBuffActive(state, "liquidLuck") ? 0.15 : 0;
  return clamp(base + buff, 0, 0.85);
}

function getSynergyChance(state) {
  const base = 0.72 + (state.upgrades.luck || 0) * 0.02;
  const buff = isBuffActive(state, "liquidLuck") ? 0.2 : 0;
  const entropyPenalty = state.entropy * 0.09;
  return clamp(base + buff - entropyPenalty, 0.35, 0.98);
}

function isBuffActive(state, buffId) {
  const endKey = `${buffId}End`;
  return Number(state.buffs[endKey] || 0) > Date.now();
}

function getBigWinThreshold(state, machine) {
  const base = Big.from(36 + machine.id * 14);
  const sizeBoost = getGridSize(state) === 4 ? 1.45 : 1;
  return base.mul(sizeBoost);
}

function getNeighbors(index, size) {
  const row = Math.floor(index / size);
  const col = index % size;
  const neighbors = [];
  if (row > 0) {
    neighbors.push(index - size);
  }
  if (row < size - 1) {
    neighbors.push(index + size);
  }
  if (col > 0) {
    neighbors.push(index - 1);
  }
  if (col < size - 1) {
    neighbors.push(index + 1);
  }
  return neighbors;
}

function drawFromDeck(state, machine) {
  if (machine.deck.length === 0) {
    return "coin";
  }
  const luckBias = getLuckBias(state);
  let selected = machine.deck[randomInt(machine.deck.length)];
  const checks = 1 + Math.floor(luckBias * 3);
  for (let i = 0; i < checks; i += 1) {
    const alternate = machine.deck[randomInt(machine.deck.length)];
    if (Math.random() < luckBias && rarityScore(alternate) > rarityScore(selected)) {
      selected = alternate;
    }
  }
  return selected;
}

function generateRandomGrid(state, machine, jackpotActive) {
  const size = getGridSize(state);
  const total = size * size;
  const grid = new Array(total);
  for (let i = 0; i < total; i += 1) {
    const col = i % size;
    const canLock = state.skillTree.symbolLocking && machine.lockedCols[col];
    if (canLock && machine.grid[i]) {
      grid[i] = machine.grid[i];
    } else {
      grid[i] = drawFromDeck(state, machine);
    }
  }

  if (jackpotActive) {
    const jackpotOptions = machine.deck.filter((id) => rarityScore(id) >= 2);
    const guaranteedSymbol = jackpotOptions.length > 0
      ? jackpotOptions[randomInt(jackpotOptions.length)]
      : "diamond";
    const replacements = Math.max(2, Math.floor(size / 2) + 1);
    for (let i = 0; i < replacements; i += 1) {
      grid[randomInt(total)] = guaranteedSymbol;
    }
  }

  return grid;
}

function shouldTrigger(state, multiplier = 1) {
  return Math.random() < clamp(getSynergyChance(state) * multiplier, 0, 1);
}

function createSynergyCounter() {
  return {
    records: {},
    total: 0,
    add(key, label, amount = 1) {
      if (!this.records[key]) {
        this.records[key] = {
          label,
          count: 0
        };
      }
      this.records[key].count += amount;
      this.total += amount;
    }
  };
}

function evaluateGrid(state, machine, grid, jackpotActive) {
  const size = getGridSize(state);
  const cells = grid.map((symbolId) => ({
    id: symbolId,
    removed: false,
    bombTriggered: false
  }));
  const synergies = createSynergyCounter();
  let bonus = 0;
  let bombExplosion = false;

  for (let i = 0; i < cells.length; i += 1) {
    const cell = cells[i];
    if (cell.removed) {
      continue;
    }
    const symbol = getSymbolById(cell.id);
    const neighbors = getNeighbors(i, size);

    if (symbol.id === "wolf") {
      for (const neighborIndex of neighbors) {
        const neighbor = cells[neighborIndex];
        if (neighbor.removed || neighbor.id !== "sheep") {
          continue;
        }
        if (shouldTrigger(state, 1.02)) {
          neighbor.removed = true;
          bonus += 12;
          synergies.add("wolf_sheep", "Wolf devoured Sheep", 1);
        }
      }
    }

    if (symbol.id === "farmer") {
      for (const neighborIndex of neighbors) {
        const neighbor = cells[neighborIndex];
        if (neighbor.removed || neighbor.id !== "sprout") {
          continue;
        }
        if (shouldTrigger(state, 1.02)) {
          neighbor.removed = true;
          bonus += 9;
          synergies.add("farmer_sprout", "Farmer harvested Sprout", 1);
        }
      }
    }

    if (symbol.id === "singularity") {
      for (const neighborIndex of neighbors) {
        const neighbor = cells[neighborIndex];
        if (neighbor.removed) {
          continue;
        }
        const target = getSymbolById(neighbor.id);
        if (target.rarity !== "Common") {
          continue;
        }
        if (shouldTrigger(state, 1.1)) {
          neighbor.removed = true;
          bonus += 20;
          synergies.add("singularity_consume", "Singularity consumed Common", 1);
        }
      }
    }

    if (symbol.id === "bomb" && !cell.bombTriggered) {
      const targets = [];
      for (const neighborIndex of neighbors) {
        const neighbor = cells[neighborIndex];
        if (neighbor.removed) {
          continue;
        }
        const targetSymbol = getSymbolById(neighbor.id);
        if (targetSymbol.rarity === "Legendary" || targetSymbol.id === "bomb") {
          continue;
        }
        targets.push(neighborIndex);
      }
      if (targets.length > 0 && shouldTrigger(state, 1.06)) {
        cell.bombTriggered = true;
        bombExplosion = true;
        let blastValue = 0;
        for (const targetIndex of targets) {
          if (!cells[targetIndex].removed) {
            cells[targetIndex].removed = true;
            blastValue += getSymbolById(cells[targetIndex].id).value * 1.5 + 3;
          }
        }
        bonus += blastValue;
        synergies.add("bomb_blast", "Bomb blast chain", targets.length);
      }
    }
  }

  let payout = 0;
  for (let i = 0; i < cells.length; i += 1) {
    const cell = cells[i];
    if (cell.removed) {
      continue;
    }

    const symbol = getSymbolById(cell.id);
    let mult = 1;
    const neighbors = getNeighbors(i, size);

    for (const neighborIndex of neighbors) {
      const neighbor = cells[neighborIndex];
      if (neighbor.removed) {
        continue;
      }
      const neighborSymbol = getSymbolById(neighbor.id);

      if (symbol.id === "coal" && neighborSymbol.id === "gear" && shouldTrigger(state, 0.98)) {
        mult += 0.35;
        synergies.add("coal_gear", "Coal + Gear", 1);
      }

      if (symbol.id === "coin" && neighborSymbol.id === "magnet" && shouldTrigger(state, 0.98)) {
        mult += 0.45;
        synergies.add("coin_magnet", "Coin + Magnet", 1);
      }

      if (symbol.id === "diamond" && neighborSymbol.id === "magnet" && shouldTrigger(state, 1.03)) {
        mult += 0.7;
        synergies.add("diamond_magnet", "Diamond resonance", 1);
      }

      if (symbol.id === "reactor" && neighborSymbol.id === "battery" && shouldTrigger(state, 1.05)) {
        mult += 0.6;
        synergies.add("reactor_battery", "Reactor charged", 1);
      }

      if (symbol.id === "sheep" && neighborSymbol.id === "farmer" && shouldTrigger(state, 0.96)) {
        mult += 0.32;
        synergies.add("sheep_farmer", "Farm loop", 1);
      }

      if (symbol.id === "oracle" && RARITY_SCORE[neighborSymbol.rarity] >= 2 && shouldTrigger(state, 0.95)) {
        mult += 0.25;
        synergies.add("oracle_future", "Oracle foresight", 1);
      }

      if (symbol.id === "chronos" && neighborSymbol.tags.includes("Multiplier") && shouldTrigger(state, 0.95)) {
        mult += 0.28;
        synergies.add("chronos_phase", "Chronos phase shift", 1);
      }
    }

    payout += symbol.value * mult;
  }

  let reward = Big.from(payout + bonus);
  const efficiency = synergies.total / Math.max(1, cells.length);
  reward = reward.mul(1 + efficiency * 0.95);

  if (jackpotActive) {
    reward = reward.mul(2.6).add(50);
  }

  if (isBuffActive(state, "liquidLuck")) {
    reward = reward.mul(1.08);
  }

  reward = reward.mul(Math.max(0.05, 1 - state.entropy));

  const influenceGain = reward.mul(machine.patrons * (0.008 + efficiency * 0.012));

  return {
    reward,
    influenceGain,
    synergies,
    bombExplosion,
    efficiency,
    jackpotActive
  };
}

function machineCanSpin(state, machineIndex) {
  const machine = state.machines[machineIndex];
  if (!machine || !machine.owned || machine.spinning) {
    return false;
  }
  if (machine.deck.length < MIN_DECK_SIZE) {
    return false;
  }
  const cost = getSpinCost(state, machineIndex);
  return state.wallet.gte(cost);
}

function runSpin(state, machineIndex, isAuto = false) {
  const machine = state.machines[machineIndex];
  if (!machine || !machine.owned || machine.spinning) {
    return;
  }

  if (machine.deck.length < MIN_DECK_SIZE) {
    if (!isAuto) {
      addLog(state, "Deck too small. Keep at least 10 symbols.", "warn");
    }
    return;
  }

  const cost = getSpinCost(state, machineIndex);
  if (!state.wallet.gte(cost)) {
    if (!isAuto) {
      addLog(state, "Insufficient Credits for spin cost.", "bad");
    }
    return;
  }

  state.wallet = state.wallet.sub(cost);
  machine.spinning = true;
  machine.lastSpinAt = Date.now();
  renderLiveUi(state, true);

  if (machine.spinInterval) {
    clearInterval(machine.spinInterval);
  }

  machine.spinInterval = setInterval(() => {
    machine.grid = generateRandomGrid(state, machine, false);
    if (state.selectedMachine === machineIndex) {
      renderGrid(state);
    }
  }, 70);

  const jackpotNow = machine.jackpotReady;
  machine.jackpotReady = false;

  machine.spinTimer = setTimeout(() => {
    if (machine.spinInterval) {
      clearInterval(machine.spinInterval);
      machine.spinInterval = null;
    }

    const finalGrid = generateRandomGrid(state, machine, jackpotNow);
    machine.grid = finalGrid;
    const result = evaluateGrid(state, machine, finalGrid, jackpotNow);
    machine.spinning = false;

    machine.lastReward = result.reward;
    machine.averageWin = machine.averageWin.mul(0.88).add(result.reward.mul(0.12));
    machine.totalEarned = machine.totalEarned.add(result.reward);
    machine.totalSynergies += result.synergies.total;

    state.wallet = state.wallet.add(result.reward);
    state.influence = state.influence.add(result.influenceGain);
    state.runCash = state.runCash.add(result.reward);
    state.lifetimeCash = state.lifetimeCash.add(result.reward);
    state.avgWin = state.avgWin.mul(0.92).add(result.reward.mul(0.08));
    state.totalSpins += 1;

    if (result.efficiency > 0.1) {
      state.entropy = Math.max(0, state.entropy - result.efficiency * 0.0012);
    }

    const bigWinThreshold = getBigWinThreshold(state, machine);
    if (result.reward.gte(bigWinThreshold)) {
      machine.badLuck = 0;
    } else {
      machine.badLuck += 1;
    }

    if (machine.badLuck >= PITY_LIMIT) {
      machine.badLuck = 0;
      machine.jackpotReady = true;
      addLog(state, `${machine.name} entered Jackpot Mode.`, "good");
    }

    const topSynergies = Object.values(result.synergies.records)
      .sort((a, b) => b.count - a.count)
      .slice(0, 2)
      .map((entry) => `${entry.label} x${entry.count}`)
      .join(", ");

    if (result.jackpotActive) {
      spawnFloatingText("JACKPOT", "#ffd166");
      spawnParticles(20, "#ffd166");
      addLog(state, `${machine.name} hit Jackpot Mode for ${formatBig(result.reward)} Credits.`, "good");
    } else {
      addLog(
        state,
        `${machine.name} paid ${formatBig(result.reward)} (${result.synergies.total} synergies${topSynergies ? `: ${topSynergies}` : ""}).`,
        result.reward.gte(cost.mul(2.2)) ? "good" : "warn"
      );
    }

    if (result.bombExplosion) {
      triggerBoardShake();
      spawnParticles(18, "#ff8a3d");
    } else if (result.synergies.total > 0) {
      spawnParticles(9, "#6efacc");
    }

    renderSpinOutcomeUi(state, machine.id);
  }, getSpinDelayMs(state));
}

function getPackWeights(state) {
  const luckLevel = state.upgrades.luck || 0;
  const common = Math.max(20, 80 - luckLevel * 3);
  const rare = 15 + luckLevel * 2;
  const epic = 5 + luckLevel * 0.9;
  const legendary = Math.max(0, (luckLevel - 6) * 0.45);
  return {
    Common: common,
    Rare: rare,
    Epic: epic,
    Legendary: legendary
  };
}

function openSymbolPack(state) {
  const cost = getPackCost(state);
  if (!state.wallet.gte(cost)) {
    addLog(state, "Not enough Credits for pack purchase.", "bad");
    return;
  }

  state.wallet = state.wallet.sub(cost);
  state.packPurchases += 1;
  const weights = getPackWeights(state);
  const gains = [];

  for (let i = 0; i < 3; i += 1) {
    const rarity = chooseWeighted([
      { value: "Common", weight: weights.Common },
      { value: "Rare", weight: weights.Rare },
      { value: "Epic", weight: weights.Epic },
      { value: "Legendary", weight: weights.Legendary }
    ]);
    const pool = PACK_RARITY_SYMBOLS[rarity];
    const symbolId = pool[randomInt(pool.length)];
    state.inventory[symbolId] = (state.inventory[symbolId] || 0) + 1;
    gains.push(symbolId);
  }

  const summary = gains.map((id) => getSymbolById(id).name).join(", ");
  dom.packPreview.textContent = `Pack opened: ${summary}`;

  if (gains.some((id) => getSymbolById(id).rarity === "Legendary")) {
    addLog(state, `Pack spike: ${summary}`, "good");
    spawnFloatingText("LEGENDARY", "#ffbe55");
    spawnParticles(16, "#ffbe55");
  } else {
    addLog(state, `Pack opened: ${summary}`, "warn");
  }

  saveGame(state);
  renderAll(state);
}

function buyUpgrade(state, upgradeId) {
  const cost = getUpgradeCost(upgradeId, state);
  if (!state.wallet.gte(cost)) {
    addLog(state, "Insufficient Credits for upgrade.", "bad");
    return;
  }

  state.wallet = state.wallet.sub(cost);
  state.upgrades[upgradeId] += 1;
  addLog(state, `${UPGRADE_DEFS[upgradeId].label} upgraded to Lv.${state.upgrades[upgradeId]}.`, "good");
  saveGame(state);
  renderAll(state);
}

function buyMachine(state, machineIndex) {
  const machine = state.machines[machineIndex];
  if (!machine || machine.owned) {
    return;
  }

  if (machineIndex > 0 && !state.machines[machineIndex - 1].owned) {
    addLog(state, "Previous machine must be purchased first.", "warn");
    return;
  }

  const cost = getMachineCost(machineIndex);
  if (!state.wallet.gte(cost)) {
    addLog(state, "Not enough Credits to purchase machine.", "bad");
    return;
  }

  state.wallet = state.wallet.sub(cost);
  machine.owned = true;
  machine.deck = createDeckPreset(machineIndex);
  ensureGrid(state, machine);
  machine.grid = generateRandomGrid(state, machine, false);
  machine.patrons = machineIndex + 1;
  state.machinePurchaseProgress = Math.max(state.machinePurchaseProgress, machineIndex + 1);
  addLog(state, `${machine.name} purchased.`, "good");
  saveGame(state);
  renderAll(state);
}

function equipSymbol(state, symbolId) {
  const machine = state.machines[state.selectedMachine];
  if (!machine || !machine.owned) {
    return;
  }

  if ((state.inventory[symbolId] || 0) <= 0) {
    return;
  }

  if (machine.deck.length >= MAX_DECK_SIZE) {
    addLog(state, `Deck cap reached (${MAX_DECK_SIZE}).`, "warn");
    return;
  }

  state.inventory[symbolId] -= 1;
  machine.deck.push(symbolId);
  addLog(state, `${getSymbolById(symbolId).name} added to ${machine.name} deck.`, "good");
  renderAll(state);
}

function unequipSymbol(state, symbolId) {
  const machine = state.machines[state.selectedMachine];
  if (!machine || !machine.owned) {
    return;
  }

  if (machine.deck.length <= MIN_DECK_SIZE) {
    addLog(state, `Deck must stay at ${MIN_DECK_SIZE} symbols minimum.`, "warn");
    return;
  }

  const index = machine.deck.indexOf(symbolId);
  if (index === -1) {
    return;
  }

  machine.deck.splice(index, 1);
  state.inventory[symbolId] = (state.inventory[symbolId] || 0) + 1;
  addLog(state, `${getSymbolById(symbolId).name} removed from deck.`, "warn");
  renderAll(state);
}

function totalOwnedSymbolCount(state, symbolId) {
  let count = state.inventory[symbolId] || 0;
  for (const machine of state.machines) {
    if (!machine.owned) {
      continue;
    }
    for (const item of machine.deck) {
      if (item === symbolId) {
        count += 1;
      }
    }
  }
  return count;
}

function getShardYield(symbolId) {
  const rarity = getSymbolById(symbolId).rarity;
  if (rarity === "Legendary") {
    return 120;
  }
  if (rarity === "Epic") {
    return 42;
  }
  if (rarity === "Rare") {
    return 16;
  }
  return 5;
}

function scrapSymbol(state, symbolId) {
  const invCount = state.inventory[symbolId] || 0;
  if (invCount <= 0) {
    return;
  }
  if (totalOwnedSymbolCount(state, symbolId) <= 1) {
    addLog(state, "Cannot scrap final copy of a symbol.", "warn");
    return;
  }

  state.inventory[symbolId] -= 1;
  const shardGain = Big.from(getShardYield(symbolId));
  state.shards = state.shards.add(shardGain);
  addLog(state, `Scrapped ${getSymbolById(symbolId).name} for ${formatBig(shardGain)} Shards.`, "good");
  renderAll(state);
}

function craftRecipe(state, recipeId) {
  const recipe = RECIPES.find((item) => item.id === recipeId);
  if (!recipe) {
    return;
  }

  if (!state.shards.gte(recipe.costShards)) {
    addLog(state, "Not enough Shards for fusion.", "bad");
    return;
  }

  for (const [symbolId, amount] of Object.entries(recipe.parts)) {
    if ((state.inventory[symbolId] || 0) < amount) {
      addLog(state, `Missing ${amount}x ${getSymbolById(symbolId).name} for fusion.`, "bad");
      return;
    }
  }

  state.shards = state.shards.sub(recipe.costShards);
  for (const [symbolId, amount] of Object.entries(recipe.parts)) {
    state.inventory[symbolId] -= amount;
  }
  state.inventory[recipe.output] = (state.inventory[recipe.output] || 0) + recipe.amount;
  addLog(state, `Fusion complete: ${getSymbolById(recipe.output).name} crafted.`, "good");
  spawnParticles(14, "#c27bff");
  renderAll(state);
}

function buyIngredient(state, ingredientId) {
  const def = INGREDIENTS[ingredientId];
  if (!def) {
    return;
  }
  const cost = Big.from(def.cost);
  if (!state.influence.gte(cost)) {
    addLog(state, "Not enough Influence for ingredient.", "bad");
    return;
  }
  state.influence = state.influence.sub(cost);
  state.ingredients[ingredientId] += 1;
  addLog(state, `Bought ${def.label}.`, "warn");
  renderAll(state);
}

function mixBuff(state, buffId) {
  const buff = BUFFS[buffId];
  if (!buff) {
    return;
  }

  for (const [ingredientId, amount] of Object.entries(buff.requirements)) {
    if ((state.ingredients[ingredientId] || 0) < amount) {
      addLog(state, `Missing ${amount}x ${INGREDIENTS[ingredientId].label}.`, "bad");
      return;
    }
  }

  for (const [ingredientId, amount] of Object.entries(buff.requirements)) {
    state.ingredients[ingredientId] -= amount;
  }

  const key = `${buffId}End`;
  const now = Date.now();
  const baseTime = Math.max(now, state.buffs[key] || 0);
  state.buffs[key] = baseTime + buff.durationMs;
  addLog(state, `${buff.label} mixed. ${buff.effect}.`, "good");
  spawnFloatingText(buff.label, "#68ffb2");
  renderAll(state);
}

function estimatePrestigeGain(state) {
  const runLog = state.runCash.log10();
  if (!Number.isFinite(runLog) || runLog < 0) {
    return 0;
  }
  const estimated = Math.pow(10, runLog / 3);
  if (!Number.isFinite(estimated)) {
    return 1000000000;
  }
  return Math.floor(estimated);
}

function collectLegendaryInventory(state) {
  const result = {};
  for (const symbolId of Object.keys(SYMBOLS)) {
    const symbol = getSymbolById(symbolId);
    if (symbol.rarity !== "Legendary") {
      continue;
    }
    const count = totalOwnedSymbolCount(state, symbolId);
    if (count > 0) {
      result[symbolId] = count;
    }
  }
  return result;
}

function prestige(state) {
  const gain = estimatePrestigeGain(state);
  if (gain < 1) {
    addLog(state, "Run needs more cash flow before prestige is worthwhile.", "warn");
    return state;
  }

  const legendaryInventory = collectLegendaryInventory(state);
  const nextState = createInitialState({
    chips: state.chips + gain,
    skillTree: { ...state.skillTree },
    legendaryInventory,
    lifetimeCash: state.lifetimeCash
  });

  addLog(nextState, `System rebooted. +${gain} Chips secured.`, "good");
  saveGame(nextState);
  return nextState;
}

function buySkillNode(state, nodeId) {
  const node = SKILL_TREE[nodeId];
  if (!node) {
    return;
  }
  if (state.skillTree[nodeId]) {
    return;
  }
  if (node.requires && !state.skillTree[node.requires]) {
    addLog(state, "Required node not unlocked yet.", "warn");
    return;
  }
  if (state.chips < node.cost) {
    addLog(state, "Not enough Chips for node unlock.", "bad");
    return;
  }

  state.chips -= node.cost;
  state.skillTree[nodeId] = true;

  if (nodeId === "grid4") {
    for (const machine of state.machines) {
      ensureGrid(state, machine);
      machine.grid = generateRandomGrid(state, machine, false);
    }
  }

  addLog(state, `${node.label} unlocked.`, "good");
  saveGame(state);
  renderAll(state);
}

function estimateOfflineGain(state, awaySeconds) {
  if (awaySeconds < 15) {
    return Big.zero();
  }

  const spinEverySec = getAutoSpinIntervalMs(state) / 1000;
  const estimatedSpins = awaySeconds / Math.max(0.2, spinEverySec);
  const machineCount = getOwnedMachineCount(state);
  const autoFactor = state.autoSpin ? 1 : 0.35;
  const entropyPenalty = Math.max(0.1, 1 - state.entropy * 0.7);

  if (estimatedSpins <= 0 || machineCount <= 0) {
    return Big.zero();
  }

  return state.avgWin
    .mul(estimatedSpins)
    .mul(machineCount)
    .mul(autoFactor)
    .mul(entropyPenalty);
}

function addLog(state, message, tone = "warn") {
  state.logs.unshift({
    message,
    tone,
    at: Date.now()
  });
  if (state.logs.length > 80) {
    state.logs.length = 80;
  }
}

function updateEra(state) {
  const log = state.lifetimeCash.log10();
  let era = "mechanical";
  let label = "Mechanical Basement Rig";
  if (log >= 9) {
    era = "quantum";
    label = "Quantum Probability Core";
  } else if (log >= 5) {
    era = "cyber";
    label = "Neon Casino Mainframe";
  }
  dom.body.setAttribute("data-era", era);
  dom.eraText.textContent = label;
}

function spawnParticles(count, color) {
  const rect = dom.gridWrapper.getBoundingClientRect();
  for (let i = 0; i < count; i += 1) {
    const particle = document.createElement("div");
    particle.className = "particle";
    particle.style.background = color;
    particle.style.left = `${Math.random() * rect.width}px`;
    particle.style.top = `${Math.random() * rect.height}px`;
    particle.style.setProperty("--dx", `${(Math.random() - 0.5) * 180}px`);
    particle.style.setProperty("--dy", `${(Math.random() - 0.5) * 180}px`);
    dom.particles.appendChild(particle);
    setTimeout(() => particle.remove(), 650);
  }
}

function spawnFloatingText(text, color) {
  const label = document.createElement("div");
  label.className = "floater";
  label.textContent = text;
  label.style.color = color;
  dom.particles.appendChild(label);
  setTimeout(() => label.remove(), 920);
}

function triggerBoardShake() {
  dom.gridWrapper.classList.remove("shake");
  void dom.gridWrapper.offsetWidth;
  dom.gridWrapper.classList.add("shake");
}

function renderStats(state) {
  const entries = [
    ["Credits", formatBig(state.wallet)],
    ["Influence", formatBig(state.influence)],
    ["Shards", formatBig(state.shards)],
    ["Chips", `${Math.floor(state.chips)}`],
    ["Entropy", formatPercent(state.entropy, 2)],
    ["Run Cash", formatBig(state.runCash)],
    ["Lifetime Cash", formatBig(state.lifetimeCash)],
    ["Total Spins", state.totalSpins.toLocaleString()]
  ];

  dom.stats.innerHTML = entries.map(([label, value]) => `
    <div class="stat">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>
  `).join("");
}

function renderGrid(state) {
  const machine = state.machines[state.selectedMachine];
  if (!machine || !machine.owned) {
    dom.slotGrid.innerHTML = "";
    return;
  }

  const size = getGridSize(state);
  dom.slotGrid.style.gridTemplateColumns = `repeat(${size}, minmax(0, 1fr))`;

  dom.slotGrid.innerHTML = machine.grid.map((symbolId) => {
    const symbol = getSymbolById(symbolId);
    const spinClass = machine.spinning ? "spinning" : "";
    return `
      <div class="symbol-cell ${rarityClass(symbol.rarity)} ${spinClass}">
        <div class="glyph">${symbol.glyph}</div>
        <div class="name">${symbol.name}</div>
        <div class="value">+${symbol.value}</div>
      </div>
    `;
  }).join("");
}

function renderLockControls(state) {
  const machine = state.machines[state.selectedMachine];
  if (!state.skillTree.symbolLocking || !machine || !machine.owned) {
    dom.lockRow.innerHTML = "";
    return;
  }

  const size = getGridSize(state);
  const buttons = [];
  for (let col = 0; col < size; col += 1) {
    const active = machine.lockedCols[col] ? "active" : "";
    buttons.push(`<button type="button" class="${active}" data-action="toggle-lock" data-col="${col}">Col ${col + 1} ${machine.lockedCols[col] ? "Locked" : "Unlocked"}</button>`);
  }
  dom.lockRow.innerHTML = buttons.join("");
}

function renderMachineMeta(state) {
  const machine = state.machines[state.selectedMachine];
  if (!machine || !machine.owned) {
    dom.machineMeta.innerHTML = "";
    return;
  }

  const spinCost = getSpinCost(state, machine.id);
  const autoInterval = getAutoSpinIntervalMs(state) / 1000;
  const delay = getSpinDelayMs(state) / 1000;
  const pity = machine.jackpotReady ? "Primed" : `${machine.badLuck}/${PITY_LIMIT}`;

  dom.machineMeta.innerHTML = `
    <div class="meta-item">Spin Cost<strong>${formatBig(spinCost)}</strong></div>
    <div class="meta-item">Last Reward<strong>${formatBig(machine.lastReward)}</strong></div>
    <div class="meta-item">Patrons<strong>${machine.patrons}</strong></div>
    <div class="meta-item">Pity Meter<strong>${pity}</strong></div>
    <div class="meta-item">Spin Delay<strong>${delay.toFixed(2)}s</strong></div>
    <div class="meta-item">Auto Interval<strong>${autoInterval.toFixed(2)}s</strong></div>
    <div class="meta-item">Deck Size<strong>${machine.deck.length}</strong></div>
    <div class="meta-item">Machine Earned<strong>${formatBig(machine.totalEarned)}</strong></div>
  `;
}

function renderMachineList(state) {
  dom.machineList.innerHTML = state.machines.map((machine, index) => {
    if (machine.owned) {
      const selectedClass = index === state.selectedMachine ? "selected" : "";
      return `
        <div class="machine-card ${selectedClass}">
          <div class="line"><strong>${machine.name}</strong><span class="pill">Owned</span></div>
          <div class="line"><span>Deck</span><span>${machine.deck.length}</span></div>
          <div class="line"><span>Patrons</span><span>${machine.patrons}</span></div>
          <div class="line"><span>Avg Win</span><span>${formatBig(machine.averageWin)}</span></div>
          <div class="line"><span>Jackpot</span><span>${machine.jackpotReady ? "Primed" : `${machine.badLuck}/${PITY_LIMIT}`}</span></div>
          <button type="button" data-action="select-machine" data-machine="${index}">Manage</button>
        </div>
      `;
    }

    const previousOwned = index === 0 || state.machines[index - 1].owned;
    const cost = getMachineCost(index);
    return `
      <div class="machine-card">
        <div class="line"><strong>${machine.name}</strong><span class="pill">Locked</span></div>
        <div class="line"><span>Purchase Cost</span><span>${formatBig(cost)}</span></div>
        <div class="line"><span>Requirement</span><span>${previousOwned ? "Ready" : "Buy prior machine"}</span></div>
        <button type="button" data-action="buy-machine" data-machine="${index}" ${previousOwned ? "" : "disabled"}>Buy</button>
      </div>
    `;
  }).join("");
}

function symbolCountMap(list) {
  const map = new Map();
  for (const symbolId of list) {
    map.set(symbolId, (map.get(symbolId) || 0) + 1);
  }
  return map;
}

function symbolSort(a, b) {
  const rarityDelta = RARITY_SCORE[getSymbolById(b).rarity] - RARITY_SCORE[getSymbolById(a).rarity];
  if (rarityDelta !== 0) {
    return rarityDelta;
  }
  return getSymbolById(a).name.localeCompare(getSymbolById(b).name);
}

function getDeltaMeta(delta, prefix) {
  if (delta == null) {
    return {
      text: `${prefix}: --`,
      className: "ev-neutral"
    };
  }
  const big = Big.from(delta);
  let sign = "±";
  let className = "ev-neutral";
  if (big.gt(0)) {
    sign = "+";
    className = "ev-good";
  } else if (big.lt(0)) {
    sign = "-";
    className = "ev-bad";
  }
  return {
    text: `${prefix}: ${sign}${formatBig(big.abs())} / spin`,
    className
  };
}

function estimateExpectedRewardForDeck(state, machine, testDeck, sampleCount) {
  if (!Array.isArray(testDeck) || testDeck.length === 0 || sampleCount <= 0) {
    return Big.zero();
  }

  const simMachine = {
    id: machine.id,
    deck: testDeck,
    grid: machine.grid.slice(),
    lockedCols: machine.lockedCols.slice()
  };
  ensureGrid(state, simMachine);

  let total = Big.zero();
  for (let i = 0; i < sampleCount; i += 1) {
    const simGrid = generateRandomGrid(state, simMachine, false);
    const outcome = evaluateGrid(state, simMachine, simGrid, false);
    total = total.add(outcome.reward);
    simMachine.grid = simGrid;
  }
  return total.div(sampleCount);
}

function buildDeckInsightCacheKey(state, machine, inventorySymbolIds, deckSymbolIds) {
  const size = getGridSize(state);
  const lockSlice = machine.lockedCols.slice(0, size).map((locked) => (locked ? "1" : "0")).join("");
  const locksActive = state.skillTree.symbolLocking && lockSlice.includes("1");
  const parts = [
    machine.id,
    size,
    state.upgrades.luck || 0,
    state.entropy.toFixed(5),
    isBuffActive(state, "liquidLuck") ? 1 : 0,
    isBuffActive(state, "overclock") ? 1 : 0,
    state.skillTree.symbolLocking ? 1 : 0,
    lockSlice,
    machine.deck.join(","),
    inventorySymbolIds.join(","),
    deckSymbolIds.join(",")
  ];
  if (locksActive) {
    parts.push(machine.grid.join(","));
  }
  return parts.join("|");
}

function buildDeckBuilderContext(state) {
  const machine = state.machines[state.selectedMachine];
  if (!machine || !machine.owned) {
    return null;
  }

  const inventorySymbolIds = Object.entries(state.inventory)
    .filter(([, count]) => count > 0)
    .map(([symbolId]) => symbolId)
    .sort(symbolSort);
  const deckSymbolIds = [...new Set(machine.deck)].sort(symbolSort);
  const key = buildDeckInsightCacheKey(state, machine, inventorySymbolIds, deckSymbolIds);
  if (deckInsightCacheKey === key && deckInsightCacheValue) {
    return deckInsightCacheValue;
  }

  const baseline = estimateExpectedRewardForDeck(state, machine, machine.deck, DECK_EV_BASELINE_SAMPLES);
  const addDeltas = {};
  const removeDeltas = {};

  for (const symbolId of inventorySymbolIds) {
    if (machine.deck.length >= MAX_DECK_SIZE) {
      addDeltas[symbolId] = null;
      continue;
    }
    const testDeck = machine.deck.concat(symbolId);
    const expected = estimateExpectedRewardForDeck(state, machine, testDeck, DECK_EV_DELTA_SAMPLES);
    addDeltas[symbolId] = expected.sub(baseline);
  }

  for (const symbolId of deckSymbolIds) {
    if (machine.deck.length <= MIN_DECK_SIZE) {
      removeDeltas[symbolId] = null;
      continue;
    }
    const index = machine.deck.indexOf(symbolId);
    if (index === -1) {
      removeDeltas[symbolId] = null;
      continue;
    }
    const testDeck = machine.deck.slice();
    testDeck.splice(index, 1);
    const expected = estimateExpectedRewardForDeck(state, machine, testDeck, DECK_EV_DELTA_SAMPLES);
    removeDeltas[symbolId] = expected.sub(baseline);
  }

  const context = {
    baseline,
    addDeltas,
    removeDeltas
  };
  deckInsightCacheKey = key;
  deckInsightCacheValue = context;
  return context;
}

function renderDeck(state, deckContext) {
  const machine = state.machines[state.selectedMachine];
  if (!machine || !machine.owned) {
    dom.deckEvSummary.textContent = "Expected return per spin: --";
    dom.deckList.innerHTML = "";
    return;
  }

  dom.deckEvSummary.textContent = `Expected return per spin: ${formatBig(deckContext ? deckContext.baseline : Big.zero())}`;
  const counts = symbolCountMap(machine.deck);
  const rows = [...counts.keys()].sort(symbolSort).map((symbolId) => {
    const symbol = getSymbolById(symbolId);
    const count = counts.get(symbolId);
    const canRemove = machine.deck.length > MIN_DECK_SIZE;
    const removeDelta = deckContext && Object.prototype.hasOwnProperty.call(deckContext.removeDeltas, symbolId)
      ? deckContext.removeDeltas[symbolId]
      : null;
    const removeDeltaMeta = canRemove
      ? getDeltaMeta(removeDelta, "Expected Δ if removed")
      : {
        text: "Expected Δ if removed: -- (minimum deck size)",
        className: "ev-neutral"
      };
    return `
      <div class="list-row ${rarityClass(symbol.rarity)}">
        <div class="glyph">${symbol.glyph}</div>
        <div class="meta">
          <strong>${symbol.name} x${count}</strong>
          <small>${symbol.rarity} | Value ${symbol.value} | ${symbol.tags.join("/")}</small>
          <small class="ev-hint ${removeDeltaMeta.className}">${removeDeltaMeta.text}</small>
        </div>
        <div class="actions">
          <button type="button" data-action="unequip" data-symbol="${symbolId}" ${canRemove ? "" : "disabled"}>- Deck</button>
        </div>
      </div>
    `;
  }).join("");

  dom.deckList.innerHTML = rows || "<p class='small'>Deck is empty.</p>";
}

function renderInventory(state, deckContext) {
  const entries = Object.entries(state.inventory)
    .filter(([, count]) => count > 0)
    .sort((a, b) => symbolSort(a[0], b[0]));

  const machine = state.machines[state.selectedMachine];
  const deckFull = machine.deck.length >= MAX_DECK_SIZE;

  dom.inventoryList.innerHTML = entries.map(([symbolId, count]) => {
    const symbol = getSymbolById(symbolId);
    const scrapAllowed = count > 0 && totalOwnedSymbolCount(state, symbolId) > 1;
    const scrapYield = getShardYield(symbolId);
    const addDelta = deckContext && Object.prototype.hasOwnProperty.call(deckContext.addDeltas, symbolId)
      ? deckContext.addDeltas[symbolId]
      : null;
    const addDeltaMeta = deckFull
      ? {
        text: "Expected Δ if added: -- (deck full)",
        className: "ev-neutral"
      }
      : getDeltaMeta(addDelta, "Expected Δ if added");
    return `
      <div class="list-row ${rarityClass(symbol.rarity)}">
        <div class="glyph">${symbol.glyph}</div>
        <div class="meta">
          <strong>${symbol.name} x${count}</strong>
          <small>${symbol.rarity} | Value ${symbol.value}</small>
          <small class="ev-hint ${addDeltaMeta.className}">${addDeltaMeta.text}</small>
        </div>
        <div class="actions">
          <button type="button" data-action="equip" data-symbol="${symbolId}" ${deckFull ? "disabled" : ""}>+ Deck</button>
          <button type="button" data-action="scrap" data-symbol="${symbolId}" ${scrapAllowed ? "" : "disabled"}>Scrap +${scrapYield}</button>
        </div>
      </div>
    `;
  }).join("") || "<p class='small'>Inventory empty.</p>";
}

function renderShop(state) {
  const packCost = getPackCost(state);
  const weights = getPackWeights(state);
  dom.packCost.textContent = `${formatBig(packCost)} Credits`;
  dom.packOdds.textContent = `Odds: Common ${weights.Common.toFixed(1)} | Rare ${weights.Rare.toFixed(1)} | Epic ${weights.Epic.toFixed(1)} | Legendary ${weights.Legendary.toFixed(1)}`;
  dom.buyPackButton.disabled = !state.wallet.gte(packCost);

  dom.upgradeList.innerHTML = Object.entries(UPGRADE_DEFS).map(([id, def]) => {
    const level = state.upgrades[id] || 0;
    const cost = getUpgradeCost(id, state);
    return `
      <div class="upgrade-row">
        <div class="line">
          <span>${def.label} Lv.${level}</span>
          <span class="cost">${formatBig(cost)}</span>
        </div>
        <p class="small">${def.description}</p>
        <button type="button" data-action="upgrade" data-upgrade="${id}" ${state.wallet.gte(cost) ? "" : "disabled"}>Upgrade</button>
      </div>
    `;
  }).join("");
}

function renderRecipes(state) {
  dom.recipeList.innerHTML = RECIPES.map((recipe) => {
    const outputSymbol = getSymbolById(recipe.output);
    const ingredients = Object.entries(recipe.parts)
      .map(([id, amount]) => `${amount}x ${getSymbolById(id).name}`)
      .join(" + ");
    const canCraft = state.shards.gte(recipe.costShards)
      && Object.entries(recipe.parts).every(([id, amount]) => (state.inventory[id] || 0) >= amount);

    return `
      <div class="recipe-row ${rarityClass(outputSymbol.rarity)}">
        <div class="line">
          <span>${outputSymbol.name}</span>
          <span class="cost">${recipe.costShards} Shards</span>
        </div>
        <p class="small">${ingredients}</p>
        <button type="button" data-action="craft" data-recipe="${recipe.id}" ${canCraft ? "" : "disabled"}>Fuse</button>
      </div>
    `;
  }).join("");
}

function renderIngredients(state) {
  dom.ingredientShop.innerHTML = Object.entries(INGREDIENTS).map(([id, ingredient]) => {
    const cost = Big.from(ingredient.cost);
    return `
      <div class="ing-row">
        <div class="line">
          <span>${ingredient.label} x${state.ingredients[id]}</span>
          <span class="cost">${ingredient.cost} Influence</span>
        </div>
        <button type="button" data-action="buy-ingredient" data-ingredient="${id}" ${state.influence.gte(cost) ? "" : "disabled"}>Buy</button>
      </div>
    `;
  }).join("");

  dom.buffList.innerHTML = Object.entries(BUFFS).map(([id, buff]) => {
    const key = `${id}End`;
    const remainingMs = Math.max(0, (state.buffs[key] || 0) - Date.now());
    const requires = Object.entries(buff.requirements)
      .map(([ingredientId, amount]) => `${amount} ${INGREDIENTS[ingredientId].label}`)
      .join(" + ");
    const canMix = Object.entries(buff.requirements)
      .every(([ingredientId, amount]) => (state.ingredients[ingredientId] || 0) >= amount);
    return `
      <div class="buff-row">
        <div class="line">
          <span>${buff.label}</span>
          <span class="pill">${remainingMs > 0 ? `${formatSeconds(remainingMs / 1000)} left` : "Inactive"}</span>
        </div>
        <p class="small">${requires}</p>
        <p class="small">${buff.effect}</p>
        <button type="button" data-action="mix" data-buff="${id}" ${canMix ? "" : "disabled"}>Mix</button>
      </div>
    `;
  }).join("");
}

function renderPrestige(state) {
  const gain = estimatePrestigeGain(state);
  dom.prestigeGain.textContent = `${gain} Chips`;
  dom.prestigeButton.disabled = gain < 1;
}

function renderSkillTree(state) {
  dom.skillTree.innerHTML = Object.entries(SKILL_TREE).map(([id, node]) => {
    const unlocked = state.skillTree[id];
    const requirementMet = !node.requires || state.skillTree[node.requires];
    const canBuy = !unlocked && requirementMet && state.chips >= node.cost;
    const classes = ["skill-node"];
    if (unlocked) {
      classes.push("active");
    }
    if (!unlocked && !requirementMet) {
      classes.push("locked");
    }
    return `
      <div class="${classes.join(" ")}">
        <div class="line">
          <strong>${node.label}</strong>
          <span class="cost">${node.cost} Chips</span>
        </div>
        <p class="small">${node.description}</p>
        <button type="button" data-action="skill" data-skill="${id}" ${canBuy ? "" : "disabled"}>${unlocked ? "Unlocked" : "Unlock"}</button>
      </div>
    `;
  }).join("");
}

function renderEventLog(state) {
  dom.eventLog.innerHTML = state.logs.slice(0, 30).map((entry) => `
    <div class="log-line ${entry.tone || "warn"}">${entry.message}</div>
  `).join("");
}

function renderButtons(state) {
  const machine = state.machines[state.selectedMachine];
  const spinCost = getSpinCost(state, state.selectedMachine);
  const canSpin = machine && machine.owned && machine.deck.length >= MIN_DECK_SIZE && state.wallet.gte(spinCost) && !machine.spinning;

  dom.spinButton.disabled = !canSpin;
  dom.spinButton.textContent = `Spin (${formatBig(spinCost)})`;
  dom.autoButton.textContent = `Auto: ${state.autoSpin ? "On" : "Off"}`;

  const nextIndex = state.machines.findIndex((item) => !item.owned);
  if (nextIndex === -1) {
    dom.buyMachineButton.disabled = true;
    dom.buyMachineButton.textContent = "All Machines Owned";
  } else {
    const nextCost = getMachineCost(nextIndex);
    dom.buyMachineButton.disabled = !state.wallet.gte(nextCost);
    dom.buyMachineButton.textContent = `Buy ${state.machines[nextIndex].name} (${formatBig(nextCost)})`;
  }
}

function renderSelectedMachineTitle(state) {
  const machine = state.machines[state.selectedMachine];
  dom.selectedMachineTitle.textContent = machine ? machine.name : "Machine";
}

function renderLiveUi(state, force = false) {
  const now = Date.now();
  if (!force && now - lastLiveUiRenderAt < LIVE_UI_REFRESH_MS) {
    return;
  }
  lastLiveUiRenderAt = now;
  updateEra(state);
  renderStats(state);
  renderSelectedMachineTitle(state);
  renderMachineMeta(state);
  renderPrestige(state);
  renderButtons(state);
}

function renderSpinOutcomeUi(state, machineIndex) {
  renderLiveUi(state, true);
  renderMachineList(state);
  renderEventLog(state);
  if (state.selectedMachine === machineIndex) {
    renderGrid(state);
    renderLockControls(state);
  }
}

function renderAll(state) {
  renderLiveUi(state, true);
  const deckContext = buildDeckBuilderContext(state);
  renderGrid(state);
  renderLockControls(state);
  renderMachineList(state);
  renderDeck(state, deckContext);
  renderInventory(state, deckContext);
  renderShop(state);
  renderRecipes(state);
  renderIngredients(state);
  renderSkillTree(state);
  renderEventLog(state);
}

function stepSimulation(state) {
  const now = Date.now();
  const dt = Math.max(0, (now - state.lastTick) / 1000);
  state.lastTick = now;

  const entropyGrowth = 0.00001 + getOwnedMachineCount(state) * 0.000002;
  state.entropy = clamp(state.entropy + entropyGrowth * dt, 0, 0.92);

  for (const machine of state.machines) {
    if (!machine.owned) {
      continue;
    }
    if (Math.random() < dt * 0.35) {
      machine.patrons = clamp(machine.patrons + (Math.random() < 0.5 ? -1 : 1), 1, 8);
    }
    const passiveInfluence = Big.from(machine.patrons * dt * 0.0026);
    state.influence = state.influence.add(passiveInfluence);
  }

  if (state.autoSpin) {
    const interval = getAutoSpinIntervalMs(state);
    for (const machine of state.machines) {
      if (!machine.owned || machine.spinning) {
        continue;
      }
      if (Date.now() - machine.lastSpinAt >= interval) {
        runSpin(state, machine.id, true);
      }
    }
  }

  renderLiveUi(state);
}

function bindEvents(stateRef) {
  dom.spinButton.addEventListener("click", () => {
    runSpin(stateRef.current, stateRef.current.selectedMachine, false);
    renderAll(stateRef.current);
  });

  dom.autoButton.addEventListener("click", () => {
    stateRef.current.autoSpin = !stateRef.current.autoSpin;
    addLog(stateRef.current, `Auto-spin ${stateRef.current.autoSpin ? "enabled" : "disabled"}.`, "warn");
    renderAll(stateRef.current);
  });

  dom.buyMachineButton.addEventListener("click", () => {
    const nextIndex = stateRef.current.machines.findIndex((machine) => !machine.owned);
    if (nextIndex >= 0) {
      buyMachine(stateRef.current, nextIndex);
    }
  });

  dom.buyPackButton.addEventListener("click", () => {
    openSymbolPack(stateRef.current);
  });

  dom.machineList.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const action = target.dataset.action;
    const machineIndex = Number(target.dataset.machine);
    if (!Number.isFinite(machineIndex)) {
      return;
    }
    if (action === "select-machine") {
      stateRef.current.selectedMachine = machineIndex;
      renderAll(stateRef.current);
      return;
    }
    if (action === "buy-machine") {
      buyMachine(stateRef.current, machineIndex);
    }
  });

  dom.deckList.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const action = target.dataset.action;
    const symbolId = target.dataset.symbol;
    if (!symbolId) {
      return;
    }
    if (action === "unequip") {
      unequipSymbol(stateRef.current, symbolId);
    }
  });

  dom.inventoryList.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const action = target.dataset.action;
    const symbolId = target.dataset.symbol;
    if (!symbolId) {
      return;
    }
    if (action === "equip") {
      equipSymbol(stateRef.current, symbolId);
    }
    if (action === "scrap") {
      scrapSymbol(stateRef.current, symbolId);
    }
  });

  dom.upgradeList.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.dataset.action !== "upgrade") {
      return;
    }
    const id = target.dataset.upgrade;
    if (!id) {
      return;
    }
    buyUpgrade(stateRef.current, id);
  });

  dom.recipeList.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.dataset.action !== "craft") {
      return;
    }
    const recipeId = target.dataset.recipe;
    if (!recipeId) {
      return;
    }
    craftRecipe(stateRef.current, recipeId);
  });

  dom.ingredientShop.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.dataset.action !== "buy-ingredient") {
      return;
    }
    const ingredientId = target.dataset.ingredient;
    if (!ingredientId) {
      return;
    }
    buyIngredient(stateRef.current, ingredientId);
  });

  dom.buffList.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.dataset.action !== "mix") {
      return;
    }
    const buffId = target.dataset.buff;
    if (!buffId) {
      return;
    }
    mixBuff(stateRef.current, buffId);
  });

  dom.skillTree.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.dataset.action !== "skill") {
      return;
    }
    const skillId = target.dataset.skill;
    if (!skillId) {
      return;
    }
    buySkillNode(stateRef.current, skillId);
  });

  dom.lockRow.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.dataset.action !== "toggle-lock") {
      return;
    }
    const col = Number(target.dataset.col);
    if (!Number.isFinite(col)) {
      return;
    }
    const machine = stateRef.current.machines[stateRef.current.selectedMachine];
    machine.lockedCols[col] = !machine.lockedCols[col];
    renderAll(stateRef.current);
  });

  dom.prestigeButton.addEventListener("click", () => {
    stateRef.current = prestige(stateRef.current);
    renderAll(stateRef.current);
  });

  dom.claimOfflineButton.addEventListener("click", () => {
    const state = stateRef.current;
    if (state.offline.pendingGain.gt(0)) {
      state.wallet = state.wallet.add(state.offline.pendingGain);
      state.runCash = state.runCash.add(state.offline.pendingGain);
      state.lifetimeCash = state.lifetimeCash.add(state.offline.pendingGain);
      addLog(state, `Offline haul claimed: ${formatBig(state.offline.pendingGain)} Credits.`, "good");
    }
    state.offline.pendingGain = Big.zero();
    state.offline.awaySeconds = 0;
    dom.offlineModal.classList.add("hidden");
    renderAll(state);
  });

  window.addEventListener("beforeunload", () => {
    saveGame(stateRef.current);
  });
}

function initialize() {
  const stateRef = {
    current: loadGame()
  };

  if (stateRef.current.offline.pendingGain.gt(0)) {
    dom.offlineSummary.textContent = `You were away for ${formatSeconds(stateRef.current.offline.awaySeconds)}. Estimated offline gain: ${formatBig(stateRef.current.offline.pendingGain)} Credits.`;
    dom.offlineModal.classList.remove("hidden");
  }

  if (stateRef.current.logs.length === 0) {
    addLog(stateRef.current, "Probability Engine online.", "good");
  }

  bindEvents(stateRef);
  renderAll(stateRef.current);

  setInterval(() => {
    stepSimulation(stateRef.current);
  }, TICK_MS);

  setInterval(() => {
    saveGame(stateRef.current);
  }, AUTOSAVE_MS);
}

initialize();
})();
