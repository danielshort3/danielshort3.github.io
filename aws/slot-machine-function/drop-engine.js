const crypto = require('crypto');

let dropConfig;
try {
  // Prefer the project-level config if present
  dropConfig = require('../../slot-config/drop-tables.json');
} catch {
  dropConfig = require('./slot-config/drop-tables.json');
}

const DROP_TABLES = dropConfig.tables || dropConfig;
const DROP_TIER_WEIGHTS = dropConfig.tierWeights || { common: 0.8, rare: 0.15, epic: 0.05 };
const DROP_BOOST_SCALE = 0.05;
const BET_DROP_SCALE = 0.1;
const WIN_DROP_MULT = 2;
const LOSE_DROP_MULT = 1;

const DEFAULT_NAMES = {
  goldChip: 'Golden Chip',
  scratchCard: 'Scratch Card',
  circuit: 'Circuit',
  inventorySlotCoupon: 'Inventory Slot Coupon',
  gearSlotCoupon: 'Gear Slot Coupon',
  storageSlotCoupon: 'Storage Slot Coupon',
  deckSlotCoupon: 'Deck Slot Coupon',
  cardSlotCoupon: 'Card Slot Coupon',
  vipMarks: 'VIP Mark',
  reelMod: 'Reel Mod',
  spinBooster: 'Spin Booster',
  card: 'Card',
  gear: 'Gear'
};

const DEFAULT_ICONS = {
  goldChip: 'img/slot/items/gold_chip.png',
  scratchCard: 'img/slot/items/scratch_card.png',
  circuit: 'img/slot/items/circuit.png',
  inventorySlotCoupon: 'img/slot/items/inventory_slot_coupon.png',
  gearSlotCoupon: 'img/slot/items/gear_slot_coupon.png',
  storageSlotCoupon: 'img/slot/items/storage_slot_coupon.png',
  deckSlotCoupon: 'img/slot/items/deck_slot_coupon.png',
  cardSlotCoupon: 'img/slot/items/card_slot_coupon.png',
  vipMarks: 'img/slot/items/vip_mark.png',
  reelMod: 'img/slot/items/reel_mod_t1.png',
  spinBooster: 'img/slot/items/spinBoosterT1.png',
  card: 'img/slot/items/card.png',
  gear: 'img/slot/items/gear.png'
};

const CATEGORY_MAP = {
  goldChip: 'Materials',
  scratchCard: 'Materials',
  circuit: 'Materials',
  vipMarks: 'Materials',
  reelMod: 'Mods',
  spinBooster: 'Mods',
  card: 'Cards',
  gear: 'Equipment',
  inventorySlotCoupon: 'Upgrades',
  gearSlotCoupon: 'Upgrades',
  storageSlotCoupon: 'Upgrades',
  deckSlotCoupon: 'Upgrades',
  cardSlotCoupon: 'Upgrades'
};

const randomFloat = () => {
  const buf = crypto.randomBytes(4);
  return buf.readUInt32BE(0) / 0xffffffff;
};

const normalizeTable = (key = 'classic') => {
  const table = DROP_TABLES?.[key] || DROP_TABLES?.classic || {};
  return {
    common: Array.isArray(table.common) ? table.common : [],
    rare: Array.isArray(table.rare) ? table.rare : [],
    epic: Array.isArray(table.epic) ? table.epic : []
  };
};

const normalizeDrop = (entry = {}) => {
  const amount = Number.isFinite(entry.amount) ? entry.amount : 1;
  const name = entry.name || DEFAULT_NAMES[entry.type] || entry.type || 'Item';
  const icon = entry.icon || DEFAULT_ICONS[entry.type] || null;
  const drop = {
    type: entry.type || 'item',
    amount,
    name,
    icon
  };
  if (entry.rarity) drop.rarity = entry.rarity;
  if (entry.tier) drop.tier = entry.tier;
  return drop;
};

const dropKey = (drop = {}) => {
  if (drop.type === 'gear') {
    return `gear:${drop.rarity || drop.tier || 'Basic'}`;
  }
  if (drop.type === 'card') {
    return `card:${drop.tier || drop.rarity || 'Basic'}`;
  }
  if (drop.type === 'reelMod' || drop.type === 'spinBooster') {
    return `${drop.type}:${drop.tier || 1}`;
  }
  return drop.type || 'item';
};

const applyDrops = (inventory = {}, drops = []) => {
  if (!drops?.length) return inventory || {};
  const next = { ...(inventory || {}) };
  drops.forEach(drop => {
    const key = dropKey(drop);
    const amt = Number.isFinite(drop.amount) ? drop.amount : 1;
    next[key] = (next[key] || 0) + amt;
  });
  return next;
};

const computeDropMultiplier = ({ bet = 1, payout = 0, upgrades = {}, skillActive = false } = {}) => {
  const safeBet = Math.max(0, bet);
  const dropBoost = Math.max(0, upgrades?.dropBoost || 0);
  const base = 1 + DROP_BOOST_SCALE * dropBoost;
  const betBonus = 1 + BET_DROP_SCALE * Math.log(1 + safeBet);
  const skillUnlocked = (upgrades?.dropBoostUnlock || 0) > 0;
  const effectLevel = Math.max(0, upgrades?.dropRateEffect || 0);
  const skillMult = skillActive && skillUnlocked ? (1 + DROP_BOOST_SCALE * (1 + effectLevel)) : 1;
  const outcome = payout > 0 ? WIN_DROP_MULT : LOSE_DROP_MULT;
  const total = base * betBonus * skillMult * outcome;
  return {
    total,
    components: { base, bet: betBonus, skill: skillMult, outcome }
  };
};

const rollFromTable = (tableKey, mult = 1) => {
  const table = normalizeTable(tableKey);
  const tiers = Object.keys(table).filter(tier => (table[tier] || []).length > 0);
  const weights = tiers.map(tier => DROP_TIER_WEIGHTS[tier] || 0);
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0) || 1;
  const results = [];
  tiers.forEach((tier, idx) => {
    const tierProb = (weights[idx] || 0) / totalWeight;
    (table[tier] || []).forEach(entry => {
      const baseChance = Math.max(0, Number(entry.chance) || 0);
      const chance = Math.min(1, baseChance * mult) * tierProb;
      if (chance > 0 && randomFloat() < chance) {
        results.push(normalizeDrop(entry));
      }
    });
  });
  return results;
};

const rollDrops = ({ bet = 1, payout = 0, upgrades = {}, skillActive = false, tableKey = 'classic' } = {}) => {
  const multiplier = computeDropMultiplier({ bet, payout, upgrades, skillActive });
  const drops = rollFromTable(tableKey, multiplier.total);
  return { drops, multiplier };
};

const rateDetails = (tableKey = 'classic', mult = 1) => {
  const table = normalizeTable(tableKey);
  const tiers = Object.keys(table).filter(tier => (table[tier] || []).length > 0);
  const weights = tiers.map(tier => DROP_TIER_WEIGHTS[tier] || 0);
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0) || 1;
  const out = [];
  tiers.forEach((tier, idx) => {
    const tierProb = (weights[idx] || 0) / totalWeight;
    (table[tier] || []).forEach(entry => {
      const baseChance = Math.max(0, Number(entry.chance) || 0);
      const chance = Math.min(1, baseChance * mult) * tierProb;
      const drop = normalizeDrop(entry);
      out.push({
        ...drop,
        chance,
        tier,
        category: CATEGORY_MAP[drop.type] || 'Materials'
      });
    });
  });
  return out.sort((a, b) => b.chance - a.chance);
};

const getDropTable = (tableKey = 'classic') => normalizeTable(tableKey);
const getTierWeights = () => ({ ...DROP_TIER_WEIGHTS });

module.exports = {
  rollDrops,
  rollFromTable,
  rateDetails,
  computeDropMultiplier,
  applyDrops,
  getDropTable,
  getTierWeights,
  dropKey,
  constants: {
    BET_DROP_SCALE,
    WIN_DROP_MULT,
    LOSE_DROP_MULT,
    DROP_BOOST_SCALE
  }
};
