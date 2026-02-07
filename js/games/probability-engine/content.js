(() => {
  "use strict";

  const PE = window.ProbabilityEngine = window.ProbabilityEngine || {};

const UPGRADE_DEFS = {
  spinSpeed: {
    label: "Spin Motor",
    description: "Reduces spin delay.",
    base: 40,
    growth: 1.8
  },
  luck: {
    label: "Luck Injector",
    description: "Biases deck draws and synergy odds.",
    base: 60,
    growth: 2.0
  },
  autoRate: {
    label: "Auto Governor",
    description: "Increases auto-spin rate.",
    base: 120,
    growth: 2.18
  }
};

const INGREDIENTS = {
  copper: {
    label: "Copper Filings",
    cost: 18
  },
  herb: {
    label: "Dry Herbs",
    cost: 22
  },
  neon: {
    label: "Neon Dust",
    cost: 40
  }
};

const BUFFS = {
  liquidLuck: {
    label: "Liquid Luck",
    durationMs: 10 * 60 * 1000,
    requirements: {
      herb: 2,
      neon: 1
    },
    effect: "+20% synergy trigger chance"
  },
  overclock: {
    label: "Overclock Tonic",
    durationMs: 5 * 60 * 1000,
    requirements: {
      copper: 2,
      neon: 1
    },
    effect: "+30% spin speed"
  }
};

const RECIPES = [
  {
    id: "diamond",
    output: "diamond",
    amount: 1,
    costShards: 50,
    parts: {
      coal: 3
    }
  },
  {
    id: "reactor",
    output: "reactor",
    amount: 1,
    costShards: 90,
    parts: {
      battery: 2,
      gear: 2
    }
  },
  {
    id: "singularity",
    output: "singularity",
    amount: 1,
    costShards: 320,
    parts: {
      diamond: 2,
      reactor: 1
    }
  }
];

const SKILL_TREE = {
  startingCash: {
    label: "Bankroll Primer",
    description: "+10% starting Credits after prestige.",
    cost: 25,
    requires: null
  },
  symbolLocking: {
    label: "Symbol Locking",
    description: "Lock reel columns so they do not spin.",
    cost: 60,
    requires: "startingCash"
  },
  grid4: {
    label: "4x4 Matrix",
    description: "Expands all machines from 3x3 to 4x4.",
    cost: 180,
    requires: "symbolLocking"
  }
};

  PE.UPGRADE_DEFS = UPGRADE_DEFS;
  PE.INGREDIENTS = INGREDIENTS;
  PE.BUFFS = BUFFS;
  PE.RECIPES = RECIPES;
  PE.SKILL_TREE = SKILL_TREE;
})();
