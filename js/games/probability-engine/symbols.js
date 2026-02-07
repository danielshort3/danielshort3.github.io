(() => {
  "use strict";

  const PE = window.ProbabilityEngine = window.ProbabilityEngine || {};

const RARITY_ORDER = ["Common", "Rare", "Epic", "Legendary"];
const RARITY_SCORE = {
  Common: 0,
  Rare: 1,
  Epic: 2,
  Legendary: 3
};

const SYMBOLS = {
  coal: {
    id: "coal",
    name: "Coal",
    glyph: "CL",
    value: 2,
    rarity: "Common",
    tags: ["Miner", "Ore"]
  },
  coin: {
    id: "coin",
    name: "Coin",
    glyph: "CN",
    value: 3,
    rarity: "Common",
    tags: ["Currency"]
  },
  sheep: {
    id: "sheep",
    name: "Sheep",
    glyph: "SP",
    value: 4,
    rarity: "Common",
    tags: ["Consumable", "Animal"]
  },
  sprout: {
    id: "sprout",
    name: "Sprout",
    glyph: "SR",
    value: 2,
    rarity: "Common",
    tags: ["Consumable", "Plant"]
  },
  gear: {
    id: "gear",
    name: "Gear",
    glyph: "GR",
    value: 5,
    rarity: "Common",
    tags: ["Machine", "Metal"]
  },
  battery: {
    id: "battery",
    name: "Battery",
    glyph: "BT",
    value: 6,
    rarity: "Rare",
    tags: ["Energy"]
  },
  wolf: {
    id: "wolf",
    name: "Wolf",
    glyph: "WF",
    value: 7,
    rarity: "Rare",
    tags: ["Predator"]
  },
  magnet: {
    id: "magnet",
    name: "Magnet",
    glyph: "MG",
    value: 8,
    rarity: "Rare",
    tags: ["Multiplier", "Metal"]
  },
  farmer: {
    id: "farmer",
    name: "Farmer",
    glyph: "FM",
    value: 7,
    rarity: "Rare",
    tags: ["Support"]
  },
  bomb: {
    id: "bomb",
    name: "Bomb",
    glyph: "BM",
    value: 5,
    rarity: "Rare",
    tags: ["Consumable", "Blast"]
  },
  diamond: {
    id: "diamond",
    name: "Diamond",
    glyph: "DM",
    value: 18,
    rarity: "Epic",
    tags: ["Gem"]
  },
  reactor: {
    id: "reactor",
    name: "Reactor",
    glyph: "RC",
    value: 16,
    rarity: "Epic",
    tags: ["Multiplier", "Energy"]
  },
  chronos: {
    id: "chronos",
    name: "Chronos",
    glyph: "CH",
    value: 14,
    rarity: "Epic",
    tags: ["Time"]
  },
  oracle: {
    id: "oracle",
    name: "Oracle",
    glyph: "OR",
    value: 24,
    rarity: "Legendary",
    tags: ["Legendary", "Multiplier"]
  },
  singularity: {
    id: "singularity",
    name: "Singularity",
    glyph: "SG",
    value: 30,
    rarity: "Legendary",
    tags: ["Legendary", "Predator"]
  }
};

const PACK_RARITY_SYMBOLS = {
  Common: ["coal", "coin", "sheep", "sprout", "gear"],
  Rare: ["battery", "wolf", "magnet", "farmer", "bomb"],
  Epic: ["diamond", "reactor", "chronos"],
  Legendary: ["oracle", "singularity"]
};

  PE.RARITY_ORDER = RARITY_ORDER;
  PE.RARITY_SCORE = RARITY_SCORE;
  PE.SYMBOLS = SYMBOLS;
  PE.PACK_RARITY_SYMBOLS = PACK_RARITY_SYMBOLS;
})();
