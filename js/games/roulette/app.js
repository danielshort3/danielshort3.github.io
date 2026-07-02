(() => {
  "use strict";

  const WHEEL_ORDER = [
    "0", "28", "9", "26", "30", "11", "7", "20", "32", "17",
    "5", "22", "34", "15", "3", "24", "36", "13", "1", "00",
    "27", "10", "25", "29", "12", "8", "19", "31", "18", "6",
    "21", "33", "16", "4", "23", "35", "14", "2"
  ];

  const RED_NUMBERS = new Set([
    1, 3, 5, 7, 9, 12, 14, 16, 18,
    19, 21, 23, 25, 27, 30, 32, 34, 36
  ]);

  const CHIP_VALUES = [1, 5, 25, 100];
  const ALL_POCKETS = ["0", "00", ...Array.from({ length: 36 }, (_, idx) => String(idx + 1))];
  const ALL_POCKET_SET = new Set(ALL_POCKETS);
  const STARTING_BANKROLL = 2000;
  const MAX_HISTORY = 200;
  const SPIN_DURATION_MS = 5400;
  const STORAGE_KEY = "roulette-double-zero-session-v1";

  const refs = {
    bankroll: document.getElementById("roulette-bankroll"),
    totalBet: document.getElementById("roulette-total-bet"),
    spinCount: document.getElementById("roulette-spin-count"),
    status: document.getElementById("roulette-spin-status"),
    spinButton: document.getElementById("roulette-spin"),
    undoButton: document.getElementById("roulette-undo"),
    clearButton: document.getElementById("roulette-clear"),
    rebetButton: document.getElementById("roulette-rebet"),
    topZone: document.getElementById("roulette-top-zone"),
    numberGrid: document.getElementById("roulette-number-grid"),
    columnRow: document.getElementById("roulette-column-row"),
    dozenRow: document.getElementById("roulette-dozen-row"),
    outsideRow: document.getElementById("roulette-outside-row"),
    wheel: document.getElementById("roulette-wheel"),
    wheelSurface: document.getElementById("roulette-wheel-surface"),
    wheelLabels: document.getElementById("roulette-wheel-labels"),
    ball: document.getElementById("roulette-ball"),
    lastPocket: document.getElementById("roulette-last-pocket"),
    historyMeta: document.getElementById("roulette-history-meta"),
    hotList: document.getElementById("roulette-hot-list"),
    recentList: document.getElementById("roulette-recent-list")
  };

  if (!refs.spinButton || !refs.numberGrid || !refs.wheel || !refs.ball) {
    return;
  }

  const chipButtons = Array.from(document.querySelectorAll("[data-chip]"));
  const betDefinitions = new Map();
  const betButtons = new Map();

  const state = {
    bankroll: STARTING_BANKROLL,
    selectedChip: CHIP_VALUES[0],
    activeBets: new Map(),
    betOperations: [],
    lastBetSnapshot: new Map(),
    history: [],
    lastPocket: "",
    spins: 0,
    spinning: false,
    wheelRotationDeg: 0,
    ballRotationDeg: 0,
    highlightedButtons: new Set()
  };

  const storageSupported = hasLocalStorageSupport();
  let persistTimer = 0;

  function hasLocalStorageSupport() {
    try {
      const testKey = "__roulette00_storage_test__";
      window.localStorage.setItem(testKey, "1");
      window.localStorage.removeItem(testKey);
      return true;
    } catch {
      return false;
    }
  }

  function registerBet(definition) {
    const normalizedNumbers = new Set((definition.numbers || []).map((value) => String(value)));
    betDefinitions.set(definition.id, {
      ...definition,
      numbers: normalizedNumbers
    });
  }

  function buildBetDefinitions() {
    registerBet({ id: "straight-0", label: "0", numbers: ["0"], payout: 35 });
    registerBet({ id: "straight-00", label: "00", numbers: ["00"], payout: 35 });

    for (let value = 1; value <= 36; value += 1) {
      registerBet({
        id: `straight-${value}`,
        label: String(value),
        numbers: [String(value)],
        payout: 35
      });
    }

    registerBet({
      id: "basket-first-five",
      label: "0-00-1-2-3",
      numbers: ["0", "00", "1", "2", "3"],
      payout: 6
    });

    const columnBets = [
      { id: "column-1", label: "Column 1", values: [] },
      { id: "column-2", label: "Column 2", values: [] },
      { id: "column-3", label: "Column 3", values: [] }
    ];

    for (let value = 1; value <= 36; value += 1) {
      const columnIndex = (value - 1) % 3;
      columnBets[columnIndex].values.push(String(value));
    }

    columnBets.forEach((column) => {
      registerBet({
        id: column.id,
        label: column.label,
        numbers: column.values,
        payout: 2
      });
    });

    registerBet({
      id: "dozen-1",
      label: "1st 12",
      numbers: Array.from({ length: 12 }, (_, idx) => String(idx + 1)),
      payout: 2
    });

    registerBet({
      id: "dozen-2",
      label: "2nd 12",
      numbers: Array.from({ length: 12 }, (_, idx) => String(idx + 13)),
      payout: 2
    });

    registerBet({
      id: "dozen-3",
      label: "3rd 12",
      numbers: Array.from({ length: 12 }, (_, idx) => String(idx + 25)),
      payout: 2
    });

    const redNumbers = Array.from(RED_NUMBERS).map((value) => String(value));
    const blackNumbers = Array.from({ length: 36 }, (_, idx) => idx + 1)
      .filter((value) => !RED_NUMBERS.has(value))
      .map((value) => String(value));

    registerBet({
      id: "outside-low",
      label: "1 to 18",
      numbers: Array.from({ length: 18 }, (_, idx) => String(idx + 1)),
      payout: 1
    });

    registerBet({
      id: "outside-even",
      label: "Even",
      numbers: Array.from({ length: 18 }, (_, idx) => String((idx + 1) * 2)),
      payout: 1
    });

    registerBet({ id: "outside-red", label: "Red", numbers: redNumbers, payout: 1 });
    registerBet({ id: "outside-black", label: "Black", numbers: blackNumbers, payout: 1 });

    registerBet({
      id: "outside-odd",
      label: "Odd",
      numbers: Array.from({ length: 18 }, (_, idx) => String((idx * 2) + 1)),
      payout: 1
    });

    registerBet({
      id: "outside-high",
      label: "19 to 36",
      numbers: Array.from({ length: 18 }, (_, idx) => String(idx + 19)),
      payout: 1
    });
  }

  function pocketColor(pocket) {
    if (pocket === "0" || pocket === "00") {
      return "green";
    }

    return RED_NUMBERS.has(Number(pocket)) ? "red" : "black";
  }

  function pocketSortValue(pocket) {
    if (pocket === "00") {
      return 37;
    }

    return Number(pocket);
  }

  function normalizeDeg(value) {
    const normalized = value % 360;
    return normalized < 0 ? normalized + 360 : normalized;
  }

  function shortestSignedDelta(fromDeg, toDeg) {
    return ((toDeg - fromDeg + 540) % 360) - 180;
  }

  function sumMap(map) {
    let total = 0;
    map.forEach((amount) => {
      total += Number(amount || 0);
    });
    return total;
  }

  function formatCurrency(amount) {
    const safe = Math.max(0, Math.round(Number(amount || 0)));
    return `$${safe.toLocaleString("en-US")}`;
  }

  function formatSignedCurrency(amount) {
    const value = Math.round(Number(amount || 0));
    const abs = `$${Math.abs(value).toLocaleString("en-US")}`;
    return value >= 0 ? `+${abs}` : `-${abs}`;
  }

  function setStatus(message, tone = "neutral") {
    refs.status.textContent = message;
    refs.status.dataset.tone = tone;
  }

  function queueAutoSave() {
    if (!storageSupported) {
      return;
    }

    window.clearTimeout(persistTimer);
    persistTimer = window.setTimeout(() => {
      persistTimer = 0;
      saveSession({ silent: true });
    }, 120);
  }

  function serializeBetMap(map) {
    return Array.from(map.entries())
      .map(([betId, amount]) => [String(betId || ""), Math.round(Number(amount || 0))])
      .filter(([betId, amount]) => betDefinitions.has(betId) && Number.isFinite(amount) && amount > 0);
  }

  function parseBetEntries(entries) {
    const parsed = new Map();

    if (!Array.isArray(entries)) {
      return parsed;
    }

    entries.forEach((entry) => {
      if (!Array.isArray(entry) || entry.length < 2) {
        return;
      }

      const betId = String(entry[0] || "").trim();
      const amount = Math.round(Number(entry[1] || 0));

      if (!betDefinitions.has(betId) || !Number.isFinite(amount) || amount <= 0) {
        return;
      }

      parsed.set(betId, amount);
    });

    return parsed;
  }

  function parseHistory(history) {
    if (!Array.isArray(history)) {
      return [];
    }

    const parsed = [];
    history.forEach((value) => {
      const pocket = String(value || "").trim();
      if (!ALL_POCKET_SET.has(pocket)) {
        return;
      }
      parsed.push(pocket);
    });

    return parsed.slice(0, MAX_HISTORY);
  }

  function sessionPayload() {
    return {
      version: 1,
      savedAt: Date.now(),
      bankroll: Math.max(0, Math.round(Number(state.bankroll || 0))),
      selectedChip: CHIP_VALUES.includes(state.selectedChip) ? state.selectedChip : CHIP_VALUES[0],
      activeBets: serializeBetMap(state.activeBets),
      lastBetSnapshot: serializeBetMap(state.lastBetSnapshot),
      history: state.history.slice(0, MAX_HISTORY),
      lastPocket: state.lastPocket,
      spins: Math.max(0, Math.round(Number(state.spins || 0))),
      wheelMod: normalizeDeg(state.wheelRotationDeg),
      ballMod: normalizeDeg(state.ballRotationDeg)
    };
  }

  function saveSession(options = {}) {
    const settings = {
      silent: false,
      ...options
    };

    if (!storageSupported) {
      if (!settings.silent) {
        setStatus("Local session storage is unavailable in this browser.", "warn");
      }
      return false;
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessionPayload()));
      if (!settings.silent) {
        setStatus("Session saved locally.", "neutral");
      }
      return true;
    } catch {
      if (!settings.silent) {
        setStatus("Failed to save session in local storage.", "warn");
      }
      return false;
    }
  }

  function applySessionSnapshot(payload) {
    const bankroll = Number(payload && payload.bankroll);
    const selectedChip = Number(payload && payload.selectedChip);
    const spins = Number(payload && payload.spins);
    const activeBets = parseBetEntries(payload && payload.activeBets);
    const lastBetSnapshot = parseBetEntries(payload && payload.lastBetSnapshot);
    const history = parseHistory(payload && payload.history);
    const lastPocket = String(payload && payload.lastPocket || "").trim();
    const wheelMod = normalizeDeg(Number(payload && payload.wheelMod));
    const ballMod = normalizeDeg(Number(payload && payload.ballMod));

    state.bankroll = Number.isFinite(bankroll) && bankroll >= 0 ? Math.round(bankroll) : STARTING_BANKROLL;
    state.selectedChip = CHIP_VALUES.includes(selectedChip) ? selectedChip : CHIP_VALUES[0];
    state.activeBets = activeBets;
    state.lastBetSnapshot = lastBetSnapshot;
    state.history = history;
    state.lastPocket = ALL_POCKET_SET.has(lastPocket) ? lastPocket : (history[0] || "");
    state.spins = Number.isFinite(spins) && spins >= 0 ? Math.max(Math.round(spins), history.length) : history.length;
    state.spinning = false;
    state.betOperations.length = 0;

    state.wheelRotationDeg = Number.isFinite(wheelMod) ? wheelMod : 0;
    refs.wheel.style.transform = `rotate(${state.wheelRotationDeg}deg)`;

    state.ballRotationDeg = Number.isFinite(ballMod) ? ballMod : 0;
    clearHighlights();
    refreshUiFromState();
  }

  function loadSession(options = {}) {
    const settings = {
      announce: false,
      ...options
    };

    if (!storageSupported) {
      return false;
    }

    let raw = "";
    try {
      raw = String(window.localStorage.getItem(STORAGE_KEY) || "");
    } catch {
      return false;
    }

    if (!raw) {
      return false;
    }

    let payload;
    try {
      payload = JSON.parse(raw);
    } catch {
      return false;
    }

    applySessionSnapshot(payload);

    if (settings.announce) {
      const savedAt = Number(payload && payload.savedAt);
      const stamp = Number.isFinite(savedAt)
        ? new Date(savedAt).toLocaleString()
        : "local storage";
      setStatus(`Restored your previous session from ${stamp}.`, "neutral");
    }

    return true;
  }

  function createBetButton(betId, options) {
    const definition = betDefinitions.get(betId);
    if (!definition) {
      throw new Error(`Missing roulette bet definition: ${betId}`);
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = `roulette00-bet ${options.extraClass || ""}`.trim();
    button.dataset.betId = betId;
    button.setAttribute("aria-label", `${definition.label} bet. Pays ${definition.payout} to 1.`);

    const label = document.createElement("span");
    label.className = "roulette00-bet-text";
    label.textContent = options.text || definition.label;

    const odds = document.createElement("span");
    odds.className = "roulette00-bet-odds";
    odds.textContent = `${definition.payout}:1`;

    const chipTotal = document.createElement("span");
    chipTotal.className = "roulette00-chip-total";

    button.append(label, odds, chipTotal);

    if (options.hideOdds) {
      button.classList.add("hide-odds");
    }

    button.addEventListener("click", (event) => {
      if (event.shiftKey) {
        event.preventDefault();
        applyWager(betId, -state.selectedChip);
        return;
      }
      applyWager(betId, state.selectedChip);
    });

    button.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      applyWager(betId, -state.selectedChip);
    });

    betButtons.set(betId, button);
    return button;
  }

  function renderTableLayout() {
    const zeroRow = document.createElement("div");
    zeroRow.className = "roulette00-zero-row";
    zeroRow.append(
      createBetButton("straight-0", { text: "0", extraClass: "is-green", hideOdds: true }),
      createBetButton("straight-00", { text: "00", extraClass: "is-green", hideOdds: true })
    );

    refs.topZone.append(
      zeroRow,
      createBetButton("basket-first-five", { text: "0 00 1 2 3", extraClass: "is-top" })
    );

    for (let row = 0; row < 12; row += 1) {
      const rowElement = document.createElement("div");
      rowElement.className = "roulette00-number-row";

      for (let column = 1; column <= 3; column += 1) {
        const pocket = String((row * 3) + column);
        rowElement.append(
          createBetButton(`straight-${pocket}`, {
            text: pocket,
            extraClass: `is-${pocketColor(pocket)}`,
            hideOdds: true
          })
        );
      }

      refs.numberGrid.append(rowElement);
    }

    refs.columnRow.append(
      createBetButton("column-1", { text: "2 to 1", extraClass: "is-column" }),
      createBetButton("column-2", { text: "2 to 1", extraClass: "is-column" }),
      createBetButton("column-3", { text: "2 to 1", extraClass: "is-column" })
    );

    refs.dozenRow.append(
      createBetButton("dozen-1", { text: "1st 12", extraClass: "is-dozen" }),
      createBetButton("dozen-2", { text: "2nd 12", extraClass: "is-dozen" }),
      createBetButton("dozen-3", { text: "3rd 12", extraClass: "is-dozen" })
    );

    refs.outsideRow.append(
      createBetButton("outside-low", { text: "1 to 18", extraClass: "is-outside" }),
      createBetButton("outside-even", { text: "Even", extraClass: "is-outside" }),
      createBetButton("outside-red", { text: "Red", extraClass: "is-red is-outside" }),
      createBetButton("outside-black", { text: "Black", extraClass: "is-black is-outside" }),
      createBetButton("outside-odd", { text: "Odd", extraClass: "is-outside" }),
      createBetButton("outside-high", { text: "19 to 36", extraClass: "is-outside" })
    );
  }

  function renderWheelSurface() {
    const slice = 360 / WHEEL_ORDER.length;

    const gradientStops = WHEEL_ORDER.map((pocket, index) => {
      const start = (index * slice).toFixed(4);
      const end = ((index + 1) * slice).toFixed(4);
      const color = pocketColor(pocket);
      const fill = color === "red"
        ? "#a62a34"
        : (color === "green" ? "#1f8b53" : "#171a1f");
      return `${fill} ${start}deg ${end}deg`;
    });

    refs.wheelSurface.style.background = `conic-gradient(from 0deg, ${gradientStops.join(",")})`;
  }

  function renderWheelLabels() {
    refs.wheelLabels.innerHTML = "";

    const radius = (refs.wheel.clientWidth / 2) - 32;
    const slice = 360 / WHEEL_ORDER.length;

    WHEEL_ORDER.forEach((pocket, index) => {
      const angle = (index * slice) + (slice / 2);
      const label = document.createElement("span");
      label.className = `roulette00-wheel-label is-${pocketColor(pocket)}`;
      label.textContent = pocket;
      label.style.transform = `translate(-50%, -50%) rotate(${angle}deg) translateY(-${radius}px) rotate(${-angle}deg)`;
      refs.wheelLabels.append(label);
    });
  }

  function renderBallPosition() {
    const radius = (refs.wheel.clientWidth / 2) - 8;
    refs.ball.style.transform = `translate(-50%, -50%) rotate(${state.ballRotationDeg}deg) translateY(-${radius}px)`;
  }

  function updateBetChipDisplay(betId) {
    const button = betButtons.get(betId);
    if (!button) {
      return;
    }

    const chipTotal = button.querySelector(".roulette00-chip-total");
    const amount = state.activeBets.get(betId) || 0;

    if (amount > 0) {
      chipTotal.textContent = formatCurrency(amount);
      button.classList.add("has-chip");
    } else {
      chipTotal.textContent = "";
      button.classList.remove("has-chip");
    }
  }

  function updateAllBetChipDisplays() {
    betButtons.forEach((_, betId) => {
      updateBetChipDisplay(betId);
    });
  }

  function updateLastPocketDisplay() {
    if (!state.lastPocket) {
      refs.lastPocket.textContent = "--";
      refs.lastPocket.classList.remove("is-red", "is-black", "is-green");
      return;
    }

    refs.lastPocket.textContent = state.lastPocket;
    refs.lastPocket.classList.remove("is-red", "is-black", "is-green");
    refs.lastPocket.classList.add(`is-${pocketColor(state.lastPocket)}`);
  }

  function updatePrimaryMetrics() {
    refs.bankroll.textContent = formatCurrency(state.bankroll);
    refs.totalBet.textContent = formatCurrency(sumMap(state.activeBets));
    refs.spinCount.textContent = String(state.spins);
  }

  function updateControlAvailability() {
    const hasCurrentBets = sumMap(state.activeBets) > 0;
    const hasUndo = state.betOperations.length > 0;
    const hasLastBet = state.lastBetSnapshot.size > 0;

    refs.spinButton.disabled = state.spinning;
    refs.undoButton.disabled = state.spinning || !hasUndo;
    refs.clearButton.disabled = state.spinning || !hasCurrentBets;
    refs.rebetButton.disabled = state.spinning || !hasLastBet;

    chipButtons.forEach((button) => {
      button.disabled = state.spinning;
    });

    betButtons.forEach((button) => {
      button.disabled = state.spinning;
    });
  }

  function clearHighlights() {
    state.highlightedButtons.forEach((button) => {
      button.classList.remove("is-winning");
    });
    state.highlightedButtons.clear();
  }

  function highlightWinningBets(winningBetIds, winningPocket) {
    clearHighlights();

    winningBetIds.forEach((betId) => {
      const button = betButtons.get(betId);
      if (!button) {
        return;
      }
      button.classList.add("is-winning");
      state.highlightedButtons.add(button);
    });

    const straightButton = betButtons.get(`straight-${winningPocket}`);
    if (straightButton) {
      straightButton.classList.add("is-winning");
      state.highlightedButtons.add(straightButton);
    }

    window.setTimeout(clearHighlights, 2200);
  }

  function updateRecentList() {
    refs.recentList.innerHTML = "";

    if (!state.history.length) {
      const placeholder = document.createElement("span");
      placeholder.className = "roulette00-card-note";
      placeholder.textContent = "No spins yet.";
      refs.recentList.append(placeholder);
      return;
    }

    state.history.slice(0, 16).forEach((pocket) => {
      const pill = document.createElement("span");
      pill.className = `roulette00-recent-pill is-${pocketColor(pocket)}`;
      pill.textContent = pocket;
      refs.recentList.append(pill);
    });
  }

  function updateHotNumbers() {
    refs.hotList.innerHTML = "";
    refs.historyMeta.textContent = `Tracking ${state.history.length} of ${MAX_HISTORY} spins.`;

    if (!state.history.length) {
      const placeholder = document.createElement("li");
      placeholder.className = "roulette00-card-note";
      placeholder.textContent = "Spin to begin collecting hot-number data.";
      refs.hotList.append(placeholder);
      return;
    }

    const counts = new Map(ALL_POCKETS.map((pocket) => [pocket, 0]));
    state.history.forEach((pocket) => {
      counts.set(pocket, (counts.get(pocket) || 0) + 1);
    });

    const ranked = Array.from(counts.entries())
      .map(([pocket, count]) => ({ pocket, count }))
      .sort((a, b) => {
        if (b.count !== a.count) {
          return b.count - a.count;
        }
        return pocketSortValue(a.pocket) - pocketSortValue(b.pocket);
      })
      .slice(0, 10);

    ranked.forEach((entry, index) => {
      const line = document.createElement("li");
      line.className = "roulette00-hot-item";

      const rank = document.createElement("span");
      rank.className = "roulette00-hot-rank";
      rank.textContent = `#${index + 1}`;

      const pocket = document.createElement("span");
      pocket.className = `roulette00-pocket-pill is-${pocketColor(entry.pocket)}`;
      pocket.textContent = entry.pocket;

      const count = document.createElement("span");
      count.className = "roulette00-hot-count";
      count.textContent = String(entry.count);

      const bar = document.createElement("span");
      bar.className = "roulette00-hot-bar";

      const barFill = document.createElement("span");
      const percent = (entry.count / state.history.length) * 100;
      barFill.style.width = `${Math.max(entry.count > 0 ? 6 : 0, percent)}%`;
      bar.append(barFill);

      line.append(rank, pocket, count, bar);
      refs.hotList.append(line);
    });
  }

  function refreshUiFromState() {
    setSelectedChip(state.selectedChip, { skipSave: true });
    updateAllBetChipDisplays();
    updateLastPocketDisplay();
    updatePrimaryMetrics();
    updateRecentList();
    updateHotNumbers();
    renderBallPosition();
    updateControlAvailability();
  }

  function applyWager(betId, delta, options = {}) {
    if (state.spinning) {
      return false;
    }

    if (!betDefinitions.has(betId)) {
      return false;
    }

    const settings = {
      record: true,
      silent: false,
      ...options
    };

    const current = state.activeBets.get(betId) || 0;

    if (delta > 0 && state.bankroll < delta) {
      if (!settings.silent) {
        setStatus("Not enough bankroll for that chip value.", "warn");
      }
      return false;
    }

    if (delta < 0 && current < Math.abs(delta)) {
      if (!settings.silent) {
        setStatus("No chip of that value is placed on this bet.", "warn");
      }
      return false;
    }

    const next = current + delta;
    if (next > 0) {
      state.activeBets.set(betId, next);
    } else {
      state.activeBets.delete(betId);
    }

    state.bankroll -= delta;

    if (settings.record) {
      state.betOperations.push({ betId, delta });
    }

    updateBetChipDisplay(betId);
    updatePrimaryMetrics();
    updateControlAvailability();
    queueAutoSave();

    return true;
  }

  function chooseWinningPocket() {
    const index = Math.floor(Math.random() * WHEEL_ORDER.length);
    return WHEEL_ORDER[index];
  }

  function computeSpinTargets(winningPocket) {
    const winningIndex = WHEEL_ORDER.indexOf(winningPocket);
    const slice = 360 / WHEEL_ORDER.length;

    // Keep a fixed resting wheel orientation so label orientation stays consistent.
    const finalWheelMod = 0;
    const pocketLocalAngle = (winningIndex + 0.5) * slice;
    const finalBallMod = normalizeDeg(pocketLocalAngle + finalWheelMod);

    const currentWheelMod = normalizeDeg(state.wheelRotationDeg);
    const wheelDelta = normalizeDeg(finalWheelMod - currentWheelMod);
    const wheelTurns = (4 + Math.floor(Math.random() * 3)) * 360;
    const wheelTarget = state.wheelRotationDeg + wheelDelta + wheelTurns;

    const currentBallMod = normalizeDeg(state.ballRotationDeg);
    const settleDelta = shortestSignedDelta(currentBallMod, finalBallMod);
    const ballTurns = -((7 + Math.floor(Math.random() * 3)) * 360);
    const ballTarget = state.ballRotationDeg + ballTurns + settleDelta;

    return { wheelTarget, ballTarget, finalWheelMod, finalBallMod };
  }

  function animateSpin(targets) {
    return new Promise((resolve) => {
      refs.wheel.style.transition = `transform ${SPIN_DURATION_MS}ms cubic-bezier(0.12, 0.84, 0.2, 1)`;
      refs.ball.style.transition = `transform ${Math.round(SPIN_DURATION_MS * 0.92)}ms cubic-bezier(0.1, 0.84, 0.24, 1)`;

      window.requestAnimationFrame(() => {
        state.wheelRotationDeg = targets.wheelTarget;
        refs.wheel.style.transform = `rotate(${state.wheelRotationDeg}deg)`;

        state.ballRotationDeg = targets.ballTarget;
        renderBallPosition();
      });

      window.setTimeout(() => {
        refs.wheel.style.transition = "";
        refs.ball.style.transition = "";

        state.wheelRotationDeg = normalizeDeg(targets.finalWheelMod);
        refs.wheel.style.transform = `rotate(${state.wheelRotationDeg}deg)`;

        state.ballRotationDeg = normalizeDeg(targets.finalBallMod);
        renderBallPosition();

        resolve();
      }, SPIN_DURATION_MS + 40);
    });
  }

  function settleSpin(winningPocket, totalWager) {
    state.lastBetSnapshot = new Map(state.activeBets);

    let returned = 0;
    const winningBetIds = [];

    state.activeBets.forEach((amount, betId) => {
      const definition = betDefinitions.get(betId);
      if (!definition || !definition.numbers.has(winningPocket)) {
        return;
      }

      winningBetIds.push(betId);
      returned += amount * (definition.payout + 1);
    });

    state.bankroll += returned;
    state.activeBets.clear();
    state.betOperations.length = 0;

    state.lastPocket = winningPocket;
    state.spins += 1;

    state.history.unshift(winningPocket);
    if (state.history.length > MAX_HISTORY) {
      state.history.length = MAX_HISTORY;
    }

    updateAllBetChipDisplays();
    updateLastPocketDisplay();
    updatePrimaryMetrics();
    updateRecentList();
    updateHotNumbers();
    highlightWinningBets(winningBetIds, winningPocket);

    const net = returned - totalWager;
    const color = pocketColor(winningPocket);

    if (returned > 0) {
      setStatus(
        `Pocket ${winningPocket} (${color}). Returned ${formatCurrency(returned)}. Round result ${formatSignedCurrency(net)}.`,
        "good"
      );
    } else {
      setStatus(
        `Pocket ${winningPocket} (${color}). No payout this spin. Round result ${formatSignedCurrency(net)}.`,
        "warn"
      );
    }

    queueAutoSave();
  }

  async function handleSpin() {
    if (state.spinning) {
      return;
    }

    const totalWager = sumMap(state.activeBets);
    if (!totalWager) {
      setStatus("Click the table to place at least one chip before spinning.", "warn");
      return;
    }

    state.spinning = true;
    updateControlAvailability();
    setStatus("Wheel spinning...", "neutral");

    const winningPocket = chooseWinningPocket();
    const targets = computeSpinTargets(winningPocket);

    await animateSpin(targets);
    settleSpin(winningPocket, totalWager);

    state.spinning = false;
    updateControlAvailability();
  }

  function handleUndo() {
    if (state.spinning) {
      return;
    }

    const lastOperation = state.betOperations.pop();
    if (!lastOperation) {
      setStatus("No chip placement to undo.", "warn");
      return;
    }

    applyWager(lastOperation.betId, -lastOperation.delta, { record: false, silent: true });
    setStatus("Last chip placement undone.", "neutral");
  }

  function clearCurrentBets() {
    if (state.spinning) {
      return;
    }

    const total = sumMap(state.activeBets);
    if (!total) {
      setStatus("No active bets to clear.", "warn");
      return;
    }

    state.bankroll += total;
    state.activeBets.clear();
    state.betOperations.length = 0;

    updateAllBetChipDisplays();
    updatePrimaryMetrics();
    updateControlAvailability();
    queueAutoSave();

    setStatus("All chips returned to bankroll.", "neutral");
  }

  function reapplyLastBet() {
    if (state.spinning) {
      return;
    }

    if (!state.lastBetSnapshot.size) {
      setStatus("No previous bet pattern to reapply yet.", "warn");
      return;
    }

    const currentTotal = sumMap(state.activeBets);
    if (currentTotal > 0) {
      state.bankroll += currentTotal;
      state.activeBets.clear();
      state.betOperations.length = 0;
    }

    const required = sumMap(state.lastBetSnapshot);
    if (required > state.bankroll) {
      updateAllBetChipDisplays();
      updatePrimaryMetrics();
      updateControlAvailability();
      queueAutoSave();
      setStatus("Insufficient bankroll to rebet the previous pattern.", "warn");
      return;
    }

    state.lastBetSnapshot.forEach((amount, betId) => {
      state.activeBets.set(betId, amount);
      state.bankroll -= amount;
    });

    updateAllBetChipDisplays();
    updatePrimaryMetrics();
    updateControlAvailability();
    queueAutoSave();

    setStatus("Previous bet pattern reapplied.", "neutral");
  }

  function setSelectedChip(nextValue, options = {}) {
    const settings = {
      skipSave: false,
      ...options
    };

    const chipValue = Number(nextValue);
    if (!CHIP_VALUES.includes(chipValue)) {
      return;
    }

    state.selectedChip = chipValue;
    chipButtons.forEach((button) => {
      const value = Number(button.dataset.chip || 0);
      button.classList.toggle("is-selected", value === chipValue);
    });

    if (!settings.skipSave) {
      queueAutoSave();
    }
  }

  function bindEvents() {
    refs.spinButton.addEventListener("click", () => {
      handleSpin().catch(() => {
        state.spinning = false;
        updateControlAvailability();
        setStatus("Spin failed. Try again.", "warn");
      });
    });

    refs.undoButton.addEventListener("click", handleUndo);
    refs.clearButton.addEventListener("click", clearCurrentBets);
    refs.rebetButton.addEventListener("click", reapplyLastBet);

    chipButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const chipValue = Number(button.dataset.chip || 0);
        if (!chipValue || state.spinning) {
          return;
        }

        setSelectedChip(chipValue);
      });
    });

    let resizeTimer = 0;
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        renderWheelLabels();
        renderBallPosition();
      }, 140);
    });

    window.addEventListener("beforeunload", () => {
      saveSession({ silent: true });
    });

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") {
        saveSession({ silent: true });
      }
    });
  }

  function init() {
    buildBetDefinitions();
    renderTableLayout();

    renderWheelSurface();
    renderWheelLabels();
    renderBallPosition();

    const restored = loadSession({ announce: true });
    if (!restored) {
      refreshUiFromState();
      if (!storageSupported) {
        setStatus("Click the table to place chips, then spin. Local save/load is unavailable in this browser.", "warn");
      } else {
        setStatus("Click the table to place chips, then spin.", "neutral");
      }
    }

    bindEvents();
    updateControlAvailability();
  }

  init();
})();
