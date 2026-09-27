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
  const MIN_SPIN_DURATION_MS = 8100;
  const STORAGE_KEY = "roulette-double-zero-session-v1";
  const betDefinitions = new Map();

  buildBetDefinitions();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      WHEEL_ORDER, betDefinitions, calculateSettlement, drawUniformIndex,
      createSpinPlan, sampleSpinMotion, getWheelGeometry
    };
  }
  if (typeof document === "undefined") {
    return;
  }

  const refs = {
    bankroll: document.getElementById("roulette-bankroll"),
    totalBet: document.getElementById("roulette-total-bet"),
    spinCount: document.getElementById("roulette-spin-count"),
    status: document.getElementById("roulette-spin-status"),
    roundPhase: document.getElementById("roulette-round-phase"),
    spinButton: document.getElementById("roulette-spin"),
    undoButton: document.getElementById("roulette-undo"),
    clearButton: document.getElementById("roulette-clear"),
    rebetButton: document.getElementById("roulette-rebet"),
    doubleButton: document.getElementById("roulette-double"),
    newSessionButton: document.getElementById("roulette-new-session"),
    mobileSpinButton: document.getElementById("roulette-mobile-spin"),
    mobileBar: document.querySelector(".roulette00-mobile-bar"),
    mobileChip: document.getElementById("roulette-mobile-chip"),
    mobileTotalBet: document.getElementById("roulette-mobile-total-bet"),
    mobileBankroll: document.getElementById("roulette-mobile-bankroll"),
    topZone: document.getElementById("roulette-top-zone"),
    numberGrid: document.getElementById("roulette-number-grid"),
    columnRow: document.getElementById("roulette-column-row"),
    dozenRow: document.getElementById("roulette-dozen-row"),
    outsideRow: document.getElementById("roulette-outside-row"),
    wheel: document.getElementById("roulette-wheel"),
    wheelWrap: document.querySelector(".roulette00-wheel-wrap"),
    wheelSurface: document.getElementById("roulette-wheel-surface"),
    wheelLabels: document.getElementById("roulette-wheel-labels"),
    ball: document.getElementById("roulette-ball"),
    lastPocket: document.getElementById("roulette-last-pocket"),
    payoutBreakdown: document.getElementById("roulette-payout-breakdown"),
    historyMeta: document.getElementById("roulette-history-meta"),
    hotList: document.getElementById("roulette-hot-list"),
    recentList: document.getElementById("roulette-recent-list")
  };

  if (!refs.spinButton || !refs.numberGrid || !refs.wheel || !refs.ball) {
    return;
  }

  const chipButtons = Array.from(document.querySelectorAll("[data-chip]"));
  const modeButtons = Array.from(document.querySelectorAll("[data-wager-mode]"));
  const betButtons = new Map();
  const insideTabs = [];
  const previewButtons = new Set();
  let hoveredInsideBetId = "";
  let focusedInsideBetId = "";

  const state = {
    bankroll: STARTING_BANKROLL,
    selectedChip: CHIP_VALUES[0],
    wagerMode: "add",
    activeBets: new Map(),
    betOperations: [],
    lastBetSnapshot: new Map(),
    history: [],
    lastPocket: "",
    spins: 0,
    spinning: false,
    wheelRotationDeg: 0,
    ballRotationDeg: 0,
    ballTrackFraction: 0,
    ballBouncePx: 0,
    highlightedButtons: new Set()
  };

  const storageSupported = hasLocalStorageSupport();
  const reducedMotionQuery = typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;
  const mobileLayoutQuery = typeof window.matchMedia === "function"
    ? window.matchMedia("(max-width: 640px)")
    : null;
  let persistTimer = 0;
  let newSessionArmed = false;
  let newSessionTimer = 0;
  let highlightTimer = 0;
  let numberGridMode = "";
  let mobileBarFrame = 0;
  let activeSpin = null;

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

    const inside = (kind, numbers, payout, shortLabel) => {
      const values = numbers.map(String);
      registerBet({
        id: `${kind}-${values.join("-")}`,
        label: `${kind === "six-line" ? "Six line" : kind[0].toUpperCase() + kind.slice(1)} ${values.join(", ")}`,
        shortLabel,
        group: kind,
        numbers: values,
        payout
      });
    };

    [["0", "00"], ["0", "1"], ["0", "2"], ["00", "2"], ["00", "3"]]
      .forEach((numbers) => inside("split", numbers, 17, numbers.join(" / ")));

    for (let row = 0; row < 12; row += 1) {
      const first = (row * 3) + 1;
      for (let column = 0; column < 2; column += 1) {
        const left = first + column;
        inside("split", [left, left + 1], 17, `${left} / ${left + 1}`);
      }
      inside("street", [first, first + 1, first + 2], 11, `${first}–${first + 2}`);

      if (row === 11) {
        continue;
      }
      for (let column = 0; column < 3; column += 1) {
        const upper = first + column;
        inside("split", [upper, upper + 3], 17, `${upper} / ${upper + 3}`);
      }
      for (let column = 0; column < 2; column += 1) {
        const left = first + column;
        inside("corner", [left, left + 1, left + 3, left + 4], 8,
          `${left}·${left + 1} / ${left + 3}·${left + 4}`);
      }
      inside("six-line", Array.from({ length: 6 }, (_, index) => first + index), 5,
        `${first}–${first + 5}`);
    }

    [["0", "1", "2"], ["0", "2", "00"], ["00", "2", "3"]]
      .forEach((numbers) => inside("trio", numbers, 11, numbers.join(" / ")));
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

  function clamp01(value) {
    return Math.min(1, Math.max(0, value));
  }

  function smoothstep(value) {
    const fraction = clamp01(value);
    return fraction * fraction * (3 - (2 * fraction));
  }

  function getWheelGeometry(diameter) {
    const size = Number.isFinite(diameter) ? Math.max(0, diameter) : 0;
    const outerRadius = Math.max(0, (size / 2) - Math.max(8, size * 0.033));
    const pocketRadius = Math.max(0, (size / 2) - Math.max(45, size * 0.148));
    const labelRadius = Math.min(outerRadius, pocketRadius + 8);
    const deflectorRadius = pocketRadius + ((outerRadius - pocketRadius) * 0.45);
    return { outerRadius, pocketRadius, labelRadius, deflectorRadius };
  }

  function createSpinPlan(winningPocket, startWheelDeg, startBallDeg, options = {}) {
    const winningIndex = WHEEL_ORDER.indexOf(winningPocket);
    if (winningIndex < 0) {
      throw new RangeError("Winning pocket must be on the wheel.");
    }
    const sliceDeg = 360 / WHEEL_ORDER.length;
    const pocketAngleDeg = (winningIndex + 0.5) * sliceDeg;
    const durationMs = Math.max(1000, Number(options.durationMs) || MIN_SPIN_DURATION_MS);
    const lockFraction = 0.88;
    const lockMs = durationMs * lockFraction;
    const tailSeconds = (durationMs - lockMs) / 1000;
    const wheelTurns = Math.max(1, Math.round(Number(options.wheelTurns) || 5));
    const ballTurns = Math.max(1, Math.round(Number(options.ballTurns) || 8));
    const wheelLockSpeed = 72;
    const ballLockSpeed = -64;
    const landingOffsetDeg = Math.max(-180,
      Math.min(180, Number(options.landingOffsetDeg) || 0));
    const lockBallDeg = startBallDeg - (ballTurns * 360) + landingOffsetDeg;
    const alignedWheelMod = normalizeDeg(lockBallDeg - pocketAngleDeg);
    const wheelAlignment = normalizeDeg(alignedWheelMod - normalizeDeg(startWheelDeg));
    const lockWheelDeg = startWheelDeg + (wheelTurns * 360) + wheelAlignment;
    const endWheelDeg = lockWheelDeg + (wheelLockSpeed * tailSeconds / 2);

    return {
      winningPocket, pocketAngleDeg, durationMs, lockMs, lockFraction,
      startWheelDeg, startBallDeg, lockWheelDeg, lockBallDeg,
      endWheelDeg, endBallDeg: lockBallDeg + (endWheelDeg - lockWheelDeg),
      wheelLockSpeed, ballLockSpeed
    };
  }

  function deceleratingAngle(startDeg, endDeg, endSpeedDegPerSec, durationSeconds, fraction) {
    const p = clamp01(fraction);
    const distance = endDeg - startDeg;
    const terminalTravel = endSpeedDegPerSec * durationSeconds;
    return startDeg + (distance * ((2 * p) - (p * p))) +
      (terminalTravel * ((p * p) - p));
  }

  function sampleSpinMotion(plan, elapsedMs) {
    const elapsed = Math.max(0, Math.min(plan.durationMs, Number(elapsedMs) || 0));
    if (elapsed >= plan.lockMs) {
      const tailElapsed = (elapsed - plan.lockMs) / 1000;
      const tailDuration = (plan.durationMs - plan.lockMs) / 1000;
      const wheelDeg = plan.lockWheelDeg +
        (plan.wheelLockSpeed * tailElapsed) -
        ((plan.wheelLockSpeed * tailElapsed * tailElapsed) / (2 * tailDuration));
      return {
        phase: "locked",
        wheelDeg,
        ballDeg: plan.lockBallDeg + (wheelDeg - plan.lockWheelDeg),
        trackFraction: 0,
        bouncePx: 0
      };
    }

    const fraction = elapsed / plan.lockMs;
    const wheelDeg = deceleratingAngle(plan.startWheelDeg, plan.lockWheelDeg,
      plan.wheelLockSpeed, plan.lockMs / 1000, fraction);
    let ballDeg = deceleratingAngle(plan.startBallDeg, plan.lockBallDeg,
      plan.ballLockSpeed, plan.lockMs / 1000, fraction);
    let trackFraction = 1;
    let bouncePx = 0;
    let phase = "orbit";

    if (fraction < 0.11) {
      phase = "launch";
      trackFraction = smoothstep(fraction / 0.11);
    } else if (fraction >= 0.68) {
      phase = "drop";
      const dropFraction = (fraction - 0.68) / 0.32;
      trackFraction = 1 - smoothstep(dropFraction);
      // Eight fixed deflectors sit halfway down the bowl, 45 degrees apart.
      // The ball kicks outward only as it passes one at the matching radius.
      const deflectorDistance = ((normalizeDeg(ballDeg) + 22.5) % 45) - 22.5;
      const angularContact = Math.exp(-Math.pow(deflectorDistance / 9, 2));
      const radialContact = Math.exp(-Math.pow((trackFraction - 0.45) / 0.3, 2));
      const impact = angularContact * radialContact * (1 - dropFraction);
      bouncePx = 11 * impact;
      ballDeg -= 3 * (deflectorDistance / 9) * impact;
    }

    return { phase, wheelDeg, ballDeg, trackFraction, bouncePx };
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

  function setRoundPhase(label) {
    if (refs.roundPhase && refs.roundPhase.textContent !== label) {
      refs.roundPhase.textContent = label;
    }
  }

  function removeInsidePreviewClasses() {
    previewButtons.forEach((button) => button.classList.remove("is-preview-covered"));
    previewButtons.clear();
  }

  function refreshInsidePreview() {
    removeInsidePreviewClasses();
    if (state.spinning) {
      return;
    }
    const definition = betDefinitions.get(hoveredInsideBetId || focusedInsideBetId);
    if (!definition || !definition.group) {
      return;
    }
    definition.numbers.forEach((pocket) => {
      const straight = betButtons.get(`straight-${pocket}`);
      if (straight) {
        straight.classList.add("is-preview-covered");
        previewButtons.add(straight);
      }
    });
  }

  function clearInsidePreview() {
    hoveredInsideBetId = "";
    focusedInsideBetId = "";
    removeInsidePreviewClasses();
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
      wagerMode: state.wagerMode === "remove" ? "remove" : "add",
      activeBets: serializeBetMap(state.activeBets),
      lastBetSnapshot: serializeBetMap(state.lastBetSnapshot),
      history: state.history.slice(0, MAX_HISTORY),
      lastPocket: state.lastPocket,
      spins: Math.max(0, Math.round(Number(state.spins || 0))),
      wheelMod: normalizeDeg(state.wheelRotationDeg),
      ballMod: normalizeDeg(state.ballRotationDeg),
      pendingSpin: activeSpin ? {
        winningPocket: activeSpin.winningPocket,
        totalWager: activeSpin.totalWager,
        endWheelMod: normalizeDeg(activeSpin.plan.endWheelDeg)
      } : null
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
    const wagerMode = String(payload && payload.wagerMode || "add");
    const spins = Number(payload && payload.spins);
    const activeBets = parseBetEntries(payload && payload.activeBets);
    const lastBetSnapshot = parseBetEntries(payload && payload.lastBetSnapshot);
    const history = parseHistory(payload && payload.history);
    const lastPocket = String(payload && payload.lastPocket || "").trim();
    const wheelMod = normalizeDeg(Number(payload && payload.wheelMod));
    const ballMod = normalizeDeg(Number(payload && payload.ballMod));

    state.bankroll = Number.isFinite(bankroll) && bankroll >= 0 ? Math.round(bankroll) : STARTING_BANKROLL;
    state.selectedChip = CHIP_VALUES.includes(selectedChip) ? selectedChip : CHIP_VALUES[0];
    state.wagerMode = wagerMode === "remove" ? "remove" : "add";
    state.activeBets = activeBets;
    state.lastBetSnapshot = lastBetSnapshot;
    state.history = history;
    state.lastPocket = ALL_POCKET_SET.has(lastPocket) ? lastPocket : (history[0] || "");
    state.spins = Number.isFinite(spins) && spins >= 0 ? Math.max(Math.round(spins), history.length) : history.length;
    state.spinning = false;
    state.betOperations.length = 0;

    state.wheelRotationDeg = Number.isFinite(wheelMod) ? wheelMod : 0;
    refs.wheel.style.transform = `rotate(${state.wheelRotationDeg}deg)`;

    const lastIndex = WHEEL_ORDER.indexOf(state.lastPocket);
    state.ballRotationDeg = lastIndex >= 0
      ? normalizeDeg(state.wheelRotationDeg + ((lastIndex + 0.5) * 360 / WHEEL_ORDER.length))
      : (Number.isFinite(ballMod) ? ballMod : 0);
    state.ballTrackFraction = 0;
    state.ballBouncePx = 0;
    if (refs.wheelWrap) {
      refs.wheelWrap.dataset.phase = state.lastPocket ? "locked" : "idle";
    }
    markWinningWheelLabel(state.lastPocket);
    setRoundPhase(state.lastPocket ? "Result" : "Ready");
    clearHighlights();
    refreshUiFromState();

    const pending = payload && payload.pendingSpin;
    const pendingPocket = String(pending && pending.winningPocket || "");
    const pendingWager = Number(pending && pending.totalWager);
    const pendingWheel = Number(pending && pending.endWheelMod);
    if (!ALL_POCKET_SET.has(pendingPocket) ||
      !Number.isSafeInteger(pendingWager) || pendingWager <= 0 ||
      pendingWager !== sumMap(state.activeBets) || !Number.isFinite(pendingWheel)) {
      return false;
    }

    state.wheelRotationDeg = normalizeDeg(pendingWheel);
    state.ballRotationDeg = normalizeDeg(state.wheelRotationDeg +
      ((WHEEL_ORDER.indexOf(pendingPocket) + 0.5) * 360 / WHEEL_ORDER.length));
    refs.wheel.style.transform = `rotate(${state.wheelRotationDeg}deg)`;
    renderBallPosition();
    if (refs.wheelWrap) {
      refs.wheelWrap.dataset.phase = "locked";
    }
    settleSpin(pendingPocket, pendingWager);
    setRoundPhase("Result");
    saveSession({ silent: true });
    return true;
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

    const recoveredSpin = applySessionSnapshot(payload);

    if (settings.announce && recoveredSpin) {
      setStatus(`Interrupted spin completed. ${refs.status.textContent}`, refs.status.dataset.tone);
    } else if (settings.announce) {
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
    button.setAttribute("aria-label", `${definition.label} bet. Pays ${definition.payout} to 1. No current wager.`);

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
      if (event.shiftKey || state.wagerMode === "remove") {
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

    if (definition.group) {
      button.addEventListener("pointerenter", () => {
        hoveredInsideBetId = betId;
        refreshInsidePreview();
      });
      button.addEventListener("pointerleave", () => {
        if (hoveredInsideBetId === betId) {
          hoveredInsideBetId = "";
          refreshInsidePreview();
        }
      });
      button.addEventListener("focus", () => {
        focusedInsideBetId = betId;
        refreshInsidePreview();
      });
      button.addEventListener("blur", () => {
        if (focusedInsideBetId === betId) {
          focusedInsideBetId = "";
          refreshInsidePreview();
        }
      });
    }

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

    for (let number = 1; number <= 36; number += 1) {
      const pocket = String(number);
      const button = createBetButton(`straight-${pocket}`, {
        text: pocket,
        extraClass: `is-${pocketColor(pocket)}`,
        hideOdds: true
      });
      button.style.setProperty("--mobile-order", number);
    }
    for (let column = 1; column <= 3; column += 1) {
      const columnButton = createBetButton(`column-${column}`, {
        text: "2 to 1",
        extraClass: "is-column"
      });
      columnButton.style.setProperty("--mobile-order", 36 + column);
    }
    renderNumberGridLayout();

    refs.columnRow.hidden = true;

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

  function isMobileLayout() {
    return mobileLayoutQuery ? mobileLayoutQuery.matches : window.innerWidth <= 640;
  }

  function renderNumberGridLayout() {
    const mode = isMobileLayout() ? "mobile" : "desktop";
    if (mode === numberGridMode) {
      return;
    }
    clearInsidePreview();
    const focusedBet = refs.numberGrid.contains(document.activeElement)
      ? document.activeElement
      : null;
    const rows = [];
    if (mode === "mobile") {
      for (let first = 1; first <= 36; first += 3) {
        const row = document.createElement("div");
        row.className = "roulette00-number-row";
        for (let value = first; value < first + 3; value += 1) {
          row.append(betButtons.get(`straight-${value}`));
        }
        rows.push(row);
      }
      const columnRow = document.createElement("div");
      columnRow.className = "roulette00-number-row";
      for (let column = 1; column <= 3; column += 1) {
        columnRow.append(betButtons.get(`column-${column}`));
      }
      rows.push(columnRow);
    } else {
      for (let rowNumber = 2; rowNumber >= 0; rowNumber -= 1) {
        const row = document.createElement("div");
        row.className = "roulette00-number-row";
        for (let column = 0; column < 12; column += 1) {
          row.append(betButtons.get(`straight-${(column * 3) + rowNumber + 1}`));
        }
        row.append(betButtons.get(`column-${rowNumber + 1}`));
        rows.push(row);
      }
    }
    refs.numberGrid.replaceChildren(...rows);
    numberGridMode = mode;
    if (focusedBet) {
      focusedBet.focus({ preventScroll: true });
    }
  }

  function activateInsideTab(index, focus = false) {
    clearInsidePreview();
    insideTabs.forEach((entry, entryIndex) => {
      const active = entryIndex === index;
      entry.button.classList.toggle("is-active", active);
      entry.button.setAttribute("aria-selected", String(active));
      entry.button.tabIndex = active ? 0 : -1;
      entry.pane.hidden = !active;
    });
    if (focus && insideTabs[index]) {
      insideTabs[index].button.focus();
    }
  }

  function renderInsideBets() {
    const layout = refs.numberGrid.closest(".roulette00-layout");
    if (!layout) {
      return;
    }

    const section = document.createElement("section");
    section.className = "roulette00-linebets";
    section.setAttribute("aria-labelledby", "roulette-inside-heading");

    const head = document.createElement("div");
    head.className = "roulette00-linebets-head";
    const heading = document.createElement("h3");
    heading.id = "roulette-inside-heading";
    heading.textContent = "Inside combinations";
    head.append(heading);

    const note = document.createElement("p");
    note.className = "roulette00-card-note roulette00-linebets-note";
    note.textContent = "Cover adjacent positions on the betting layout. Pick a category, then place chips on a combination.";

    const tabs = document.createElement("div");
    tabs.className = "roulette00-line-tabs";
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Inside bet types");

    const list = document.createElement("div");
    list.className = "roulette00-linebet-list";
    const groups = [
      { label: "Splits", kinds: ["split"] },
      { label: "Streets & trios", kinds: ["street", "trio"] },
      { label: "Corners", kinds: ["corner"] },
      { label: "Six lines", kinds: ["six-line"] }
    ];

    groups.forEach((group, index) => {
      const button = document.createElement("button");
      button.id = `roulette-inside-tab-${index}`;
      button.type = "button";
      button.className = "roulette00-line-tab";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", `roulette-inside-pane-${index}`);

      const pane = document.createElement("div");
      pane.id = `roulette-inside-pane-${index}`;
      pane.className = "roulette00-linebet-pane";
      pane.setAttribute("role", "tabpanel");
      pane.setAttribute("aria-labelledby", button.id);
      pane.tabIndex = 0;

      const grid = document.createElement("div");
      grid.className = "roulette00-linebet-grid";
      const betIds = [];
      betDefinitions.forEach((definition, betId) => {
        if (!group.kinds.includes(definition.group)) {
          return;
        }
        betIds.push(betId);
        grid.append(createBetButton(betId, {
          text: definition.shortLabel,
          extraClass: "is-line"
        }));
      });
      pane.append(grid);
      button.addEventListener("click", () => activateInsideTab(index));
      button.addEventListener("keydown", (event) => {
        let next = index;
        if (event.key === "ArrowRight") {
          next = (index + 1) % insideTabs.length;
        } else if (event.key === "ArrowLeft") {
          next = (index - 1 + insideTabs.length) % insideTabs.length;
        } else if (event.key === "Home") {
          next = 0;
        } else if (event.key === "End") {
          next = insideTabs.length - 1;
        } else {
          return;
        }
        event.preventDefault();
        activateInsideTab(next, true);
      });

      insideTabs.push({ button, pane, betIds, label: group.label });
      tabs.append(button);
      list.append(pane);
    });

    section.append(head, note, tabs, list);
    layout.after(section);
    activateInsideTab(0);
    updateInsideTabTotals();
  }

  function updateInsideTabTotals() {
    insideTabs.forEach((entry) => {
      const amount = entry.betIds.reduce((total, betId) => total + (state.activeBets.get(betId) || 0), 0);
      entry.button.textContent = amount ? `${entry.label} · ${formatCurrency(amount)}` : entry.label;
      entry.button.setAttribute("aria-label",
        amount ? `${entry.label}, ${formatCurrency(amount)} wagered` : `${entry.label}, no wagers`);
    });
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

    const geometry = getWheelGeometry(refs.wheel.clientWidth);
    const radius = geometry.labelRadius;
    if (refs.wheelWrap) {
      refs.wheelWrap.style.setProperty("--roulette-pocket-radius", `${geometry.pocketRadius}px`);
      refs.wheelWrap.style.setProperty("--roulette-label-radius", `${geometry.labelRadius}px`);
      refs.wheelWrap.style.setProperty("--roulette-outer-radius", `${geometry.outerRadius}px`);
      refs.wheelWrap.style.setProperty("--roulette-deflector-radius", `${geometry.deflectorRadius}px`);
    }
    const slice = 360 / WHEEL_ORDER.length;

    WHEEL_ORDER.forEach((pocket, index) => {
      const angle = (index * slice) + (slice / 2);
      const label = document.createElement("span");
      label.className = `roulette00-wheel-label is-${pocketColor(pocket)}`;
      label.textContent = pocket;
      label.style.transform = `translate(-50%, -50%) rotate(${angle}deg) translateY(-${radius}px) rotate(${-angle}deg)`;
      refs.wheelLabels.append(label);
    });
    if (!state.spinning) {
      markWinningWheelLabel(state.lastPocket);
    }
  }

  function markWinningWheelLabel(pocket) {
    refs.wheelLabels.querySelectorAll(".is-winning").forEach((label) => {
      label.classList.remove("is-winning");
    });
    const index = WHEEL_ORDER.indexOf(pocket);
    if (index >= 0 && refs.wheelLabels.children[index]) {
      refs.wheelLabels.children[index].classList.add("is-winning");
    }
  }

  function renderBallPosition() {
    const geometry = getWheelGeometry(refs.wheel.clientWidth);
    const radius = geometry.pocketRadius +
      ((geometry.outerRadius - geometry.pocketRadius) * state.ballTrackFraction) +
      state.ballBouncePx;
    refs.ball.style.transform = `translate(-50%, -50%) rotate(${state.ballRotationDeg}deg) translateY(-${radius}px)`;
  }

  function updateBetChipDisplay(betId) {
    const button = betButtons.get(betId);
    if (!button) {
      return;
    }

    const chipTotal = button.querySelector(".roulette00-chip-total");
    const definition = betDefinitions.get(betId);
    const amount = state.activeBets.get(betId) || 0;

    if (amount > 0) {
      chipTotal.textContent = formatCurrency(amount);
      button.classList.add("has-chip");
    } else {
      chipTotal.textContent = "";
      button.classList.remove("has-chip");
    }

    if (definition) {
      const wagerText = amount > 0
        ? `Current wager ${formatCurrency(amount)}.`
        : "No current wager.";
      const actionText = state.wagerMode === "remove"
        ? (amount > 0
          ? `Activate to remove ${formatCurrency(Math.min(state.selectedChip, amount))}.`
          : "No wager to remove.")
        : `Activate to add ${formatCurrency(state.selectedChip)}.`;
      button.setAttribute(
        "aria-label",
        `${definition.label} bet. Pays ${definition.payout} to 1. ${wagerText} ${actionText}`
      );
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
    const bankroll = formatCurrency(state.bankroll);
    const totalBet = formatCurrency(sumMap(state.activeBets));

    refs.bankroll.textContent = bankroll;
    refs.totalBet.textContent = totalBet;
    refs.spinCount.textContent = String(state.spins);

    if (refs.mobileBankroll) {
      refs.mobileBankroll.textContent = bankroll;
    }
    if (refs.mobileTotalBet) {
      refs.mobileTotalBet.textContent = totalBet;
    }
    if (refs.mobileChip) {
      refs.mobileChip.textContent = formatCurrency(state.selectedChip);
    }
    updateInsideTabTotals();
  }

  function updateControlAvailability() {
    const hasCurrentBets = sumMap(state.activeBets) > 0;
    const hasUndo = state.betOperations.length > 0;
    const hasLastBet = state.lastBetSnapshot.size > 0;

    refs.spinButton.disabled = state.spinning;
    refs.undoButton.disabled = state.spinning || !hasUndo;
    refs.clearButton.disabled = state.spinning || !hasCurrentBets;
    refs.rebetButton.disabled = state.spinning || !hasLastBet;
    if (refs.doubleButton) {
      refs.doubleButton.disabled = state.spinning || !hasCurrentBets;
    }
    if (refs.newSessionButton) {
      refs.newSessionButton.disabled = state.spinning;
    }
    if (refs.mobileSpinButton) {
      refs.mobileSpinButton.disabled = state.spinning;
    }

    chipButtons.forEach((button) => {
      button.disabled = state.spinning;
    });

    modeButtons.forEach((button) => {
      button.disabled = state.spinning;
    });

    betButtons.forEach((button) => {
      button.disabled = state.spinning;
    });
  }

  function clearHighlights() {
    window.clearTimeout(highlightTimer);
    highlightTimer = 0;
    state.highlightedButtons.forEach((button) => {
      button.classList.remove("is-winning-pocket", "is-paid");
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
      button.classList.add("is-paid");
      state.highlightedButtons.add(button);
    });

    const straightButton = betButtons.get(`straight-${winningPocket}`);
    if (straightButton) {
      straightButton.classList.add("is-winning-pocket");
      state.highlightedButtons.add(straightButton);
    }

    highlightTimer = window.setTimeout(() => {
      highlightTimer = 0;
      clearHighlights();
    }, 2200);
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
      .filter((entry) => entry.count > 0)
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
    setWagerMode(state.wagerMode, { skipSave: true });
    updateAllBetChipDisplays();
    updateLastPocketDisplay();
    updatePrimaryMetrics();
    updateRecentList();
    updateHotNumbers();
    if (state.lastPocket && state.lastBetSnapshot.size) {
      renderPayoutBreakdown(state.lastPocket, sumMap(state.lastBetSnapshot),
        calculateSettlement(state.lastBetSnapshot, state.lastPocket));
    } else if (refs.payoutBreakdown) {
      refs.payoutBreakdown.replaceChildren();
    }
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

    if (delta < 0 && current <= 0) {
      if (!settings.silent) {
        setStatus("No wager is placed on this bet yet.", "warn");
      }
      return false;
    }

    const appliedDelta = delta < 0 ? -Math.min(current, Math.abs(delta)) : delta;
    const next = current + appliedDelta;
    if (next > 0) {
      state.activeBets.set(betId, next);
    } else {
      state.activeBets.delete(betId);
    }

    state.bankroll -= appliedDelta;

    if (settings.record) {
      state.betOperations.push({ type: "wager", betId, delta: appliedDelta });
    }

    updateBetChipDisplay(betId);
    updatePrimaryMetrics();
    updateControlAvailability();
    queueAutoSave();

    if (!settings.silent) {
      const definition = betDefinitions.get(betId);
      const verb = appliedDelta > 0 ? "Placed" : "Returned";
      setStatus(
        `${verb} ${formatCurrency(Math.abs(appliedDelta))} ${appliedDelta > 0 ? "on" : "from"} ${definition.label}. ` +
        `${formatCurrency(next)} on this bet; pays ${definition.payout}:1.`,
        "neutral"
      );
    }

    return true;
  }

  function drawUniformIndex(length, nextUint32) {
    const range = 0x100000000;
    if (!Number.isInteger(length) || length < 1 || length > range || typeof nextUint32 !== "function") {
      throw new RangeError("A valid pocket count and random source are required.");
    }
    const limit = Math.floor(range / length) * length;
    let sample;
    do {
      sample = nextUint32();
      if (!Number.isInteger(sample) || sample < 0 || sample >= range) {
        throw new RangeError("Random source must return an unsigned 32-bit integer.");
      }
    } while (sample >= limit);
    return sample % length;
  }

  function chooseWinningPocket() {
    if (window.crypto && typeof window.crypto.getRandomValues === "function") {
      const value = new Uint32Array(1);
      const index = drawUniformIndex(WHEEL_ORDER.length, () => {
        window.crypto.getRandomValues(value);
        return value[0];
      });
      return WHEEL_ORDER[index];
    }
    const index = Math.floor(Math.random() * WHEEL_ORDER.length);
    return WHEEL_ORDER[index];
  }

  function applySpinMotion(sample) {
    state.wheelRotationDeg = sample.wheelDeg;
    state.ballRotationDeg = sample.ballDeg;
    state.ballTrackFraction = sample.trackFraction;
    state.ballBouncePx = sample.bouncePx;
    refs.wheel.style.transform = `rotate(${sample.wheelDeg}deg)`;
    renderBallPosition();
    if (refs.wheelWrap) {
      refs.wheelWrap.dataset.phase = sample.phase;
    }
    setRoundPhase(sample.phase === "launch" ? "No more bets" :
      (sample.phase === "orbit" ? "Ball circling" : "Ball dropping"));
  }

  function animateSpin(pending) {
    return new Promise((resolve) => {
      pending.resolve = resolve;
      const frame = (now) => {
        if (activeSpin !== pending) {
          return;
        }
        if (pending.startedAt === null) {
          pending.startedAt = now;
        }
        const elapsedMs = now - pending.startedAt;
        applySpinMotion(sampleSpinMotion(pending.plan, elapsedMs));
        if (elapsedMs >= pending.plan.durationMs) {
          pending.frameId = 0;
          resolve();
          return;
        }
        pending.frameId = window.requestAnimationFrame(frame);
      };
      pending.frameId = window.requestAnimationFrame(frame);
    });
  }

  function calculateSettlement(bets, winningPocket) {
    const winningBets = [];
    let returned = 0;
    bets.forEach((amount, betId) => {
      const definition = betDefinitions.get(betId);
      if (!definition || !definition.numbers.has(winningPocket)) {
        return;
      }
      const payout = amount * (definition.payout + 1);
      winningBets.push({ betId, label: definition.label, amount, odds: definition.payout, returned: payout });
      returned += payout;
    });
    return { returned, winningBets };
  }

  function renderPayoutBreakdown(winningPocket, totalWager, settlement) {
    if (!refs.payoutBreakdown) {
      return;
    }
    refs.payoutBreakdown.replaceChildren();
    const summary = document.createElement("p");
    summary.className = "roulette00-payout-summary";
    summary.textContent = `${formatCurrency(totalWager)} staked · ${formatCurrency(settlement.returned)} returned · ` +
      `${formatSignedCurrency(settlement.returned - totalWager)} net`;
    refs.payoutBreakdown.append(summary);

    if (!settlement.winningBets.length) {
      const empty = document.createElement("p");
      empty.className = "roulette00-payout-empty";
      empty.textContent = `No placed bet covered ${winningPocket}.`;
      refs.payoutBreakdown.append(empty);
      return;
    }

    const list = document.createElement("ul");
    list.className = "roulette00-payout-list";
    settlement.winningBets
      .slice()
      .sort((first, second) => second.returned - first.returned)
      .forEach((bet) => {
        const item = document.createElement("li");
        item.textContent = `${bet.label}: ${formatCurrency(bet.amount)} × ${bet.odds + 1} = ` +
          formatCurrency(bet.returned);
        list.append(item);
      });
    refs.payoutBreakdown.append(list);
  }

  function settleSpin(winningPocket, totalWager) {
    state.lastBetSnapshot = new Map(state.activeBets);
    const settlement = calculateSettlement(state.activeBets, winningPocket);
    const returned = settlement.returned;

    state.bankroll += returned;
    state.activeBets.clear();
    state.betOperations.length = 0;

    state.lastPocket = winningPocket;
    state.spins += 1;
    markWinningWheelLabel(winningPocket);

    state.history.unshift(winningPocket);
    if (state.history.length > MAX_HISTORY) {
      state.history.length = MAX_HISTORY;
    }

    updateAllBetChipDisplays();
    updateLastPocketDisplay();
    updatePrimaryMetrics();
    updateRecentList();
    updateHotNumbers();
    highlightWinningBets(settlement.winningBets.map((bet) => bet.betId), winningPocket);
    renderPayoutBreakdown(winningPocket, totalWager, settlement);

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

  function finishActiveSpin(pending, options = {}) {
    if (!pending || pending.settled || activeSpin !== pending) {
      return;
    }
    pending.settled = true;
    if (pending.frameId) {
      window.cancelAnimationFrame(pending.frameId);
      pending.frameId = 0;
    }
    applySpinMotion(sampleSpinMotion(pending.plan, pending.plan.durationMs));
    state.wheelRotationDeg = normalizeDeg(state.wheelRotationDeg);
    state.ballRotationDeg = normalizeDeg(state.ballRotationDeg);
    refs.wheel.style.transform = `rotate(${state.wheelRotationDeg}deg)`;
    renderBallPosition();
    activeSpin = null;
    settleSpin(pending.winningPocket, pending.totalWager);
    setRoundPhase("Result");

    if (!options.skipScroll) {
      if (pending.options.scrollResult && isMobileLayout()) {
        refs.spinButton.closest(".roulette00-wheel-card").scrollIntoView({
          behavior: "auto",
          block: "start"
        });
      } else if (!isMobileLayout()) {
        refs.payoutBreakdown.scrollIntoView({ behavior: "auto", block: "end" });
      }
    }

    state.spinning = false;
    updateControlAvailability();
    saveSession({ silent: true });
    if (pending.resolve) {
      pending.resolve();
      pending.resolve = null;
    }
  }

  async function handleSpin(options = {}) {
    if (state.spinning) {
      return;
    }

    const totalWager = sumMap(state.activeBets);
    if (!totalWager) {
      setStatus("Click the table to place at least one chip before spinning.", "warn");
      return;
    }

    state.spinning = true;
    clearInsidePreview();
    updateControlAvailability();
    setRoundPhase("No more bets");
    setStatus("No more bets. Spin underway.", "neutral");
    markWinningWheelLabel("");

    const winningPocket = chooseWinningPocket();
    const plan = createSpinPlan(winningPocket, state.wheelRotationDeg, state.ballRotationDeg, {
      durationMs: MIN_SPIN_DURATION_MS + Math.round(Math.random() * 1200),
      wheelTurns: 4 + Math.floor(Math.random() * 3),
      ballTurns: 7 + Math.floor(Math.random() * 3),
      landingOffsetDeg: (Math.random() - 0.5) * 360
    });
    const pending = {
      winningPocket, totalWager, options, plan,
      frameId: 0, startedAt: null, resolve: null, settled: false
    };
    activeSpin = pending;
    saveSession({ silent: true });

    if (reducedMotionQuery && reducedMotionQuery.matches) {
      finishActiveSpin(pending);
      return;
    }

    await animateSpin(pending);
    finishActiveSpin(pending);
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

    if (lastOperation.type === "snapshot") {
      state.activeBets = new Map(lastOperation.activeBets);
      state.bankroll = lastOperation.bankroll;
      updateAllBetChipDisplays();
      updatePrimaryMetrics();
      updateControlAvailability();
      queueAutoSave();
      setStatus(`${lastOperation.reason || "Table action"} undone. Your previous bets were restored.`, "neutral");
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
    const required = sumMap(state.lastBetSnapshot);
    const availableAfterReturningCurrentBets = state.bankroll + currentTotal;

    if (required > availableAfterReturningCurrentBets) {
      setStatus("Insufficient bankroll to rebet the previous pattern.", "warn");
      return;
    }

    state.betOperations.push({
      type: "snapshot",
      reason: "Rebet",
      activeBets: new Map(state.activeBets),
      bankroll: state.bankroll
    });
    state.activeBets = new Map(state.lastBetSnapshot);
    state.bankroll = availableAfterReturningCurrentBets - required;

    updateAllBetChipDisplays();
    updatePrimaryMetrics();
    updateControlAvailability();
    queueAutoSave();

    setStatus("Previous bet pattern reapplied.", "neutral");
  }

  function doubleActiveBets() {
    if (state.spinning) {
      return;
    }
    const additionalStake = sumMap(state.activeBets);
    if (!additionalStake) {
      setStatus("Place a bet before doubling.", "warn");
      return;
    }
    if (!Number.isSafeInteger(additionalStake * 2) || additionalStake > state.bankroll) {
      setStatus("Insufficient bankroll to double all current bets.", "warn");
      return;
    }

    state.betOperations.push({
      type: "snapshot",
      reason: "Double",
      activeBets: new Map(state.activeBets),
      bankroll: state.bankroll
    });
    state.activeBets.forEach((amount, betId) => {
      state.activeBets.set(betId, amount * 2);
    });
    state.bankroll -= additionalStake;
    updateAllBetChipDisplays();
    updatePrimaryMetrics();
    updateControlAvailability();
    queueAutoSave();
    setStatus(`Doubled every active bet. ${formatCurrency(additionalStake * 2)} on the table.`, "neutral");
  }

  function resetNewSessionControl() {
    newSessionArmed = false;
    window.clearTimeout(newSessionTimer);
    newSessionTimer = 0;

    if (!refs.newSessionButton) {
      return;
    }

    refs.newSessionButton.textContent = "New session";
    refs.newSessionButton.classList.remove("is-confirming");
    refs.newSessionButton.setAttribute("aria-label", "Start a new session and reset bankroll");
  }

  function handleNewSession() {
    if (state.spinning) {
      return;
    }

    if (!newSessionArmed) {
      newSessionArmed = true;
      refs.newSessionButton.textContent = "Confirm reset";
      refs.newSessionButton.classList.add("is-confirming");
      refs.newSessionButton.setAttribute(
        "aria-label",
        "Confirm new session. This resets bankroll, bets, spin history, and statistics."
      );
      setStatus("Select Confirm reset within 5 seconds to start a fresh session.", "warn");
      newSessionTimer = window.setTimeout(() => {
        resetNewSessionControl();
        setStatus("New session reset canceled.", "neutral");
      }, 5000);
      return;
    }

    resetNewSessionControl();
    state.bankroll = STARTING_BANKROLL;
    state.selectedChip = CHIP_VALUES[0];
    state.wagerMode = "add";
    state.activeBets = new Map();
    state.betOperations.length = 0;
    state.lastBetSnapshot = new Map();
    state.history = [];
    state.lastPocket = "";
    state.spins = 0;
    clearInsidePreview();
    markWinningWheelLabel("");
    state.wheelRotationDeg = 0;
    state.ballRotationDeg = 0;
    state.ballTrackFraction = 0;
    state.ballBouncePx = 0;
    refs.wheel.style.transition = "";
    refs.ball.style.transition = "";
    refs.wheel.style.transform = "rotate(0deg)";
    if (refs.wheelWrap) {
      refs.wheelWrap.dataset.phase = "idle";
    }
    setRoundPhase("Ready");
    clearHighlights();
    refreshUiFromState();
    saveSession({ silent: true });
    setStatus("New session started with 2,000 virtual credits.", "good");
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
      const selected = value === chipValue;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });

    if (refs.mobileChip) {
      refs.mobileChip.textContent = formatCurrency(chipValue);
    }

    updateAllBetChipDisplays();

    if (!settings.skipSave) {
      queueAutoSave();
    }
  }

  function setWagerMode(nextMode, options = {}) {
    const settings = {
      skipSave: false,
      ...options
    };
    const mode = nextMode === "remove" ? "remove" : "add";

    state.wagerMode = mode;
    modeButtons.forEach((button) => {
      const selected = button.dataset.wagerMode === mode;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    updateAllBetChipDisplays();

    if (!settings.skipSave) {
      queueAutoSave();
    }
  }

  function updateMobileBarVisibility() {
    if (!refs.mobileBar) {
      return;
    }
    const wheelCard = refs.spinButton.closest(".roulette00-wheel-card");
    if (!wheelCard) {
      return;
    }
    const bounds = wheelCard.getBoundingClientRect();
    const wheelVisible = bounds.bottom > 0 && bounds.top < window.innerHeight;
    refs.mobileBar.classList.toggle("is-visible", isMobileLayout() && !wheelVisible);
  }

  function scheduleMobileBarVisibility() {
    if (mobileBarFrame) {
      return;
    }
    mobileBarFrame = window.requestAnimationFrame(() => {
      mobileBarFrame = 0;
      updateMobileBarVisibility();
    });
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
    if (refs.doubleButton) {
      refs.doubleButton.addEventListener("click", doubleActiveBets);
    }
    if (refs.newSessionButton) {
      refs.newSessionButton.addEventListener("click", handleNewSession);
    }
    if (refs.mobileSpinButton) {
      refs.mobileSpinButton.addEventListener("click", () => {
        const wheelCard = refs.spinButton.closest(".roulette00-wheel-card");
        if (wheelCard && isMobileLayout()) {
          wheelCard.scrollIntoView({
            behavior: reducedMotionQuery && reducedMotionQuery.matches ? "auto" : "smooth",
            block: "start"
          });
          wheelCard.tabIndex = -1;
          wheelCard.focus({ preventScroll: true });
          scheduleMobileBarVisibility();
        }
        handleSpin({ scrollResult: true }).catch(() => {
          state.spinning = false;
          updateControlAvailability();
          setStatus("Spin failed. Try again.", "warn");
        });
      });
    }

    chipButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const chipValue = Number(button.dataset.chip || 0);
        if (!chipValue || state.spinning) {
          return;
        }

        setSelectedChip(chipValue);
      });
    });

    modeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        if (state.spinning) {
          return;
        }
        setWagerMode(button.dataset.wagerMode);
        setStatus(
          state.wagerMode === "remove"
            ? "Remove mode active. Select a wager to return up to the selected chip value."
            : "Add mode active. Select a wager to place one chip.",
          "neutral"
        );
      });
    });

    let resizeTimer = 0;
    window.addEventListener("resize", () => {
      renderNumberGridLayout();
      scheduleMobileBarVisibility();
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        renderWheelLabels();
        renderBallPosition();
      }, 140);
    });
    window.addEventListener("scroll", scheduleMobileBarVisibility, { passive: true });

    window.addEventListener("beforeunload", () => {
      finishActiveSpin(activeSpin, { skipScroll: true });
      saveSession({ silent: true });
    });

    window.addEventListener("pagehide", () => {
      finishActiveSpin(activeSpin, { skipScroll: true });
      saveSession({ silent: true });
    });

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") {
        finishActiveSpin(activeSpin, { skipScroll: true });
        saveSession({ silent: true });
      }
    });

    if (reducedMotionQuery) {
      const onReducedMotionChange = (event) => {
        if (event.matches) {
          finishActiveSpin(activeSpin, { skipScroll: true });
        }
      };
      if (typeof reducedMotionQuery.addEventListener === "function") {
        reducedMotionQuery.addEventListener("change", onReducedMotionChange);
      } else if (typeof reducedMotionQuery.addListener === "function") {
        reducedMotionQuery.addListener(onReducedMotionChange);
      }
    }
  }

  function init() {
    renderTableLayout();
    renderInsideBets();

    renderWheelSurface();
    renderWheelLabels();
    renderBallPosition();
    setRoundPhase("Ready");

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
    document.body.classList.add("roulette00-js-ready");
    updateMobileBarVisibility();
    updateControlAvailability();
  }

  init();
})();
