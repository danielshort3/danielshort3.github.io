const crypto = require('crypto');
const path = require('path');
const configPath = path.resolve(__dirname, '../../slot-config/classic.json');
let slotConfig;
try {
  slotConfig = require(configPath);
} catch {
  slotConfig = require('./classic-config');
}

const BASE_ROWS = slotConfig.baseRows || slotConfig.rows || 3;
const BASE_REELS = slotConfig.baseReels || slotConfig.reels || 3;
const MAX_ROWS = slotConfig.maxRows || slotConfig.rows || BASE_ROWS;
const MAX_REELS = slotConfig.maxReels || slotConfig.reels || BASE_REELS;
const ROWS = MAX_ROWS;
const REELS = MAX_REELS;
const UPGRADE_COSTS = slotConfig.upgradeCosts || { rows: [500, 1500], reels: [750, 2000], lines: [300, 900, 1800] };
const SYMBOLS = Array.isArray(slotConfig.symbols)
  ? slotConfig.symbols.map(entry => (typeof entry === 'string' ? { key: entry, label: entry } : entry))
  : Object.keys(slotConfig.payouts).map(key => ({ key, label: key }));
const PAYOUTS = slotConfig.payouts || {};
const MAX_PATTERN_TIER = 3;
const MAX_PAYOUT = Math.max(
  ...Object.entries(PAYOUTS)
    .filter(([key]) => key !== 'bonus' && key !== 'wild')
    .map(([, value]) => value)
);

function randomFloat() {
  const buf = crypto.randomBytes(4);
  const value = buf.readUInt32BE(0);
  return value / 0xffffffff;
}

function activeSymbolKeys() {
  return SYMBOLS.map(s => s.key);
}

function pickSymbol() {
  const keys = activeSymbolKeys();
  const weights = keys.map(key => {
    const payout = PAYOUTS[key] || 1;
    return payout > 0 ? 1 / payout : 1;
  });
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let target = randomFloat() * total;
  for (let i = 0; i < keys.length; i += 1) {
    if (target <= weights[i]) return keys[i];
    target -= weights[i];
  }
  return keys[keys.length - 1];
}

function clampRows(value) {
  const target = Number.isFinite(value) ? value : BASE_ROWS;
  return Math.max(BASE_ROWS, Math.min(target, MAX_ROWS));
}

function clampReels(value) {
  const target = Number.isFinite(value) ? value : BASE_REELS;
  return Math.max(BASE_REELS, Math.min(target, MAX_REELS));
}

function buildOutcome(rows = ROWS, reels = REELS) {
  const outcome = Array.from({ length: rows }, () => Array(reels).fill(''));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < reels; col += 1) {
      outcome[row][col] = pickSymbol();
    }
  }
  return outcome;
}

const PATTERN_RULES = [
  {
    tier: 0,
    generate(rows, reels) {
      return Array.from({ length: rows }, (_, row) => Array(reels).fill(row));
    }
  },
  {
    tier: 1,
    generate(rows, reels) {
      const down = Array.from({ length: reels }, (_, i) => Math.min(rows - 1, i));
      const up = Array.from({ length: reels }, (_, i) => Math.max(0, rows - 1 - i));
      return [down, up];
    }
  },
  {
    tier: 2,
    generate(rows, reels) {
      const zig = [];
      const zag = [];
      let r1 = 0;
      let d1 = 1;
      let r2 = rows - 1;
      let d2 = -1;
      for (let i = 0; i < reels; i += 1) {
        zig.push(r1);
        zag.push(r2);
        r1 += d1;
        r2 += d2;
        if (rows > 1) {
          if (r1 === rows - 1 || r1 === 0) d1 *= -1;
          if (r2 === rows - 1 || r2 === 0) d2 *= -1;
        }
      }
      return [zig, zag];
    }
  },
  {
    tier: 3,
    generate(rows, reels) {
      const out = [];
      const path = Array(reels).fill(0);
      function step(col, row) {
        path[col] = row;
        if (col === reels - 1) {
          out.push([...path]);
          return;
        }
        for (const delta of [-1, 0, 1]) {
          const nextRow = row + delta;
          if (nextRow >= 0 && nextRow < rows) step(col + 1, nextRow);
        }
      }
      for (let row = 0; row < rows; row += 1) step(0, row);
      return out;
    }
  }
];

function linePatternDefs(rows, reels) {
  const defs = [];
  PATTERN_RULES.forEach(rule => {
    const patterns = rule.generate(rows, reels);
    patterns.forEach(pattern => defs.push({ pattern, tier: rule.tier }));
  });
  return defs;
}

function unlockedPatterns(rows, reels, maxTier = MAX_PATTERN_TIER) {
  return linePatternDefs(rows, reels)
    .filter(def => def.tier <= maxTier)
    .map(def => def.pattern);
}

function matchFromLeft(outcome, pattern) {
  if (!outcome.length) return { symbol: null, count: 0, indexes: [] };
  const rows = outcome.length;
  const indexes = [];
  let base = null;
  for (let col = 0; col < pattern.length; col += 1) {
    const row = Math.max(0, Math.min(rows - 1, pattern[col]));
    const value = outcome[row]?.[col];
    if (col === 0) {
      if (value !== 'wild') base = value;
      indexes.push(col);
      continue;
    }
    if (value === base || value === 'wild' || base === null) {
      if (base === null && value !== 'wild') base = value;
      indexes.push(col);
    } else {
      break;
    }
  }
  if (base === null) base = 'wild';
  return { symbol: base, count: indexes.length, indexes };
}

function evaluateOutcome(outcome, bet, rows = ROWS, reels = REELS, tier = MAX_PATTERN_TIER) {
  const patterns = unlockedPatterns(rows, reels, tier);
  const groups = [];
  patterns.forEach(pattern => {
    const result = matchFromLeft(outcome, pattern);
    if (result.count >= 3) {
      groups.push({ ...result, pattern });
    }
  });
  const bonusCount = outcome.reduce(
    (total, row) => total + row.filter(symbol => symbol === 'bonus').length,
    0
  );
  if (bonusCount >= 3) {
    groups.push({ pattern: null, indexes: [], symbol: 'bonus', count: bonusCount });
  }
  let payout = 0;
  groups.forEach(group => {
    let base = PAYOUTS[group.symbol] || 0;
    if (group.symbol === 'wild') base = MAX_PAYOUT * 2;
    const amount = bet * base * Math.max(1, group.count - 2);
    group.payout = amount;
    payout += amount;
  });
  return { payout, groups };
}

function spin(bet, opts = {}) {
  const rows = clampRows(opts.rows);
  const reels = clampReels(opts.reels);
  const lineTier = Number.isFinite(opts.lineTier) ? Math.max(0, Math.min(opts.lineTier, MAX_PATTERN_TIER)) : MAX_PATTERN_TIER;
  const outcome = buildOutcome(rows, reels);
  const { payout, groups } = evaluateOutcome(outcome, bet, rows, reels, lineTier);
  const metadata = {
    rows,
    reels,
    lineTier,
    lines: linePatternDefs(rows, reels).filter(def => def.tier <= lineTier),
    upgrades: {
      baseRows: BASE_ROWS,
      baseReels: BASE_REELS,
      maxRows: MAX_ROWS,
      maxReels: MAX_REELS,
      costs: UPGRADE_COSTS
    }
  };
  return {
    outcome,
    payout,
    groups,
    metadata
  };
}

function machineMetadata() {
  return {
    id: slotConfig.id,
    name: slotConfig.name,
    tier: slotConfig.tier,
    rows: ROWS,
    reels: REELS,
    lineTier: MAX_PATTERN_TIER,
    lines: linePatternDefs(ROWS, REELS).filter(def => def.tier <= MAX_PATTERN_TIER),
    payouts: PAYOUTS,
    symbols: SYMBOLS.map(entry => ({ key: entry.key, label: entry.label })),
    upgrades: {
      baseRows: BASE_ROWS,
      maxRows: MAX_ROWS,
      baseReels: BASE_REELS,
      maxReels: MAX_REELS,
      costs: UPGRADE_COSTS
    }
  };
}

module.exports = {
  spin,
  machineMetadata,
  evaluateOutcome
};
