(() => {
  "use strict";

  const getHelpers = () => window.STELLAR_DOGFIGHT_HELPERS || {};

  function log(message) {
    const { logEvent } = getHelpers();
    if (logEvent) {
      logEvent(message);
    }
  }

  function emitPulse(x, y, color, radius) {
    const { spawnPulse } = getHelpers();
    if (spawnPulse) {
      spawnPulse(x, y, color, radius);
    }
  }

  function wrap(entity) {
    const { wrapEntity } = getHelpers();
    if (wrapEntity) {
      wrapEntity(entity);
    }
  }

  function distanceBetween(a, b) {
    const { distanceBetween: dist } = getHelpers();
    if (dist) {
      return dist(a, b);
    }
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function listEnemies() {
    const { getEnemies } = getHelpers();
    return getEnemies ? getEnemies() : [];
  }

  function safeRandInt(min, max) {
    const { randInt } = getHelpers();
    if (randInt) {
      return randInt(min, max);
    }
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function safeGeneratePart(rarity) {
    const { generatePart } = getHelpers();
    return generatePart ? generatePart(rarity) : null;
  }

  const TIER_META = {
    common: { label: "Common", weight: 6 },
    uncommon: { label: "Uncommon", weight: 4 },
    rare: { label: "Rare", weight: 2.4 },
    epic: { label: "Epic", weight: 1.4 },
    legendary: { label: "Legendary", weight: 0.7 }
  };

  const TIER_ORDER = ["common", "uncommon", "rare", "epic", "legendary"];

  const ABILITIES = {
    overdrive: {
      id: "overdrive",
      name: "Overdrive",
      desc: "Boosts fire rate and damage for 6 seconds.",
      cooldown: 16,
      duration: 6,
      onStart: (ship) => {
        ship.damage *= 1.2;
        ship.fireRate *= 1.3;
        log("Overdrive engaged. Weapons hot.");
      },
      onEnd: (ship) => {
        ship.damage /= 1.2;
        ship.fireRate /= 1.3;
      }
    },
    barrier: {
      id: "barrier",
      name: "Barrier Field",
      desc: "Reduces incoming damage and restores shields.",
      cooldown: 18,
      duration: 4,
      onStart: (ship) => {
        ship.damageReduction = Math.min(0.6, ship.damageReduction + 0.35);
        ship.shield = Math.min(ship.maxShield, ship.shield + ship.maxShield * 0.35);
        log("Barrier field online.");
      },
      onEnd: (ship) => {
        ship.damageReduction = Math.max(0, ship.damageReduction - 0.35);
      }
    },
    blink: {
      id: "blink",
      name: "Blink Drive",
      desc: "Short-range teleport in the aim direction.",
      cooldown: 12,
      duration: 0.6,
      onStart: (ship) => {
        const distance = 160;
        ship.x += Math.cos(ship.angle) * distance;
        ship.y += Math.sin(ship.angle) * distance;
        ship.invulnerable = 0.5;
        wrap(ship);
        emitPulse(ship.x, ship.y, "#44d2c2");
        log("Blink drive executed.");
      },
      onEnd: () => {}
    },
    cloak: {
      id: "cloak",
      name: "Phase Cloak",
      desc: "Scrambles enemy targeting for a short time.",
      cooldown: 20,
      duration: 6,
      onStart: (ship, state) => {
        state.enemyAccuracyMod = 0.75;
        log("Cloak active. Enemy accuracy reduced.");
      },
      onEnd: (ship, state) => {
        state.enemyAccuracyMod = 1;
      }
    }
  };

  const SHIPS = [
    {
      id: "scout",
      name: "Scout",
      tier: 1,
      desc: "Light interceptor that stays mobile and charges quickly.",
      passive: "+15% speed, +10% energy regen, -12% shield.",
      unlock: { rank: 1 },
      mods: { mult: { maxSpeed: 1.15, accel: 1.15, energyRegen: 1.1, maxShield: 0.88 } },
      abilityId: "overdrive",
      signatureWeapon: "repeater",
      nextIds: ["rift", "corsair"]
    },
    {
      id: "vanguard",
      name: "Vanguard",
      tier: 1,
      desc: "Balanced hull with steady energy throughput.",
      passive: "+10% energy regen.",
      unlock: { rank: 1 },
      mods: { mult: { energyRegen: 1.1 } },
      abilityId: "overdrive",
      signatureWeapon: "pulse",
      nextIds: ["warden", "striker"]
    },
    {
      id: "warden",
      name: "Warden",
      tier: 2,
      desc: "Reinforced shield plating with a slower top speed.",
      passive: "+40 shield, +20 hull, -10% speed.",
      unlock: { rank: 3, credits: 240 },
      mods: { add: { maxShield: 40, maxHealth: 20 }, mult: { maxSpeed: 0.9 } },
      abilityId: "barrier",
      signatureWeapon: "plasma",
      nextIds: ["titan", "specter"]
    },
    {
      id: "striker",
      name: "Striker",
      tier: 2,
      desc: "Aggressive assault frame built for close-range fights.",
      passive: "+12% damage, +8% fire rate, -10% shield.",
      unlock: { rank: 2, credits: 180 },
      mods: { mult: { damage: 1.12, fireRate: 1.08, maxShield: 0.9 } },
      abilityId: "overdrive",
      signatureWeapon: "scatter",
      nextIds: ["tempest", "specter"]
    },
    {
      id: "rift",
      name: "Rift",
      tier: 2,
      desc: "High-velocity interceptor with blink mobility.",
      passive: "+20% speed, -10% hull.",
      unlock: { rank: 4, credits: 320, blueprints: 2 },
      mods: { mult: { maxSpeed: 1.2, accel: 1.2, maxHealth: 0.9 } },
      abilityId: "blink",
      signatureWeapon: "rail",
      nextIds: ["specter", "tempest"]
    },
    {
      id: "corsair",
      name: "Corsair",
      tier: 2,
      desc: "Crit-focused raider that darts between volleys.",
      passive: "+12% crit chance, +6% speed, -10% energy.",
      unlock: { rank: 3, credits: 220, blueprints: 1 },
      mods: { add: { critChance: 0.12 }, mult: { maxSpeed: 1.06, maxEnergy: 0.9 } },
      abilityId: "cloak",
      signatureWeapon: "volley",
      nextIds: ["specter", "tempest"]
    },
    {
      id: "specter",
      name: "Specter",
      tier: 3,
      desc: "Stealth striker tuned for critical hits.",
      passive: "+10% crit chance, -8% shield.",
      unlock: { rank: 5, credits: 420, faction: { id: "nova", rep: 18 } },
      mods: { add: { critChance: 0.1 }, mult: { maxShield: 0.92 } },
      abilityId: "cloak",
      signatureWeapon: "ion",
      nextIds: []
    },
    {
      id: "titan",
      name: "Titan",
      tier: 3,
      desc: "Heavy siege platform built to anchor the line.",
      passive: "+80 shield, +40 hull, -18% speed.",
      unlock: { rank: 5, credits: 480, blueprints: 2 },
      mods: { add: { maxShield: 80, maxHealth: 40 }, mult: { maxSpeed: 0.82, accel: 0.85 } },
      abilityId: "barrier",
      signatureWeapon: "siege",
      nextIds: []
    },
    {
      id: "tempest",
      name: "Tempest",
      tier: 3,
      desc: "Rapid-strike frame that chews through lighter targets.",
      passive: "+20% fire rate, +15% energy regen, -10% damage.",
      unlock: { rank: 5, credits: 460, faction: { id: "vortex", rep: 12 } },
      mods: { mult: { fireRate: 1.2, energyRegen: 1.15, damage: 0.9 } },
      abilityId: "overdrive",
      signatureWeapon: "repeater",
      nextIds: []
    }
  ];

  const WEAPONS = [
    {
      id: "basic",
      tier: "common",
      name: "Basic Blaster",
      desc: "Starter cannon with steady output.",
      tags: ["Starter", "Reliable"],
      unlock: { rank: 1 },
      stats: { damage: 9, fireRate: 3.6, energyCost: 18, bulletSpeed: 500, projectiles: 1, spread: 0.16 }
    },
    {
      id: "pulse",
      tier: "common",
      name: "Pulse Blaster",
      desc: "Reliable bolts with steady energy costs.",
      tags: ["Balanced", "Energy"],
      unlock: { rank: 1 },
      stats: { damage: 12, fireRate: 4, energyCost: 16, bulletSpeed: 520, projectiles: 1, spread: 0.14 },
      upgrades: [
        { tier: "common", name: "Phase Capacitors I", cost: 120, desc: "+4 damage.", apply: (stats) => { stats.damage += 4; } },
        { tier: "uncommon", name: "Phase Capacitors II", cost: 180, desc: "+4 damage.", apply: (stats) => { stats.damage += 4; } },
        { tier: "rare", name: "Harmonic Cycling", cost: 250, desc: "+25% fire rate.", apply: (stats) => { stats.fireRate *= 1.25; } },
        { tier: "rare", name: "Flux Recycling", cost: 320, desc: "-18% energy cost.", apply: (stats) => { stats.energyCost *= 0.82; } },
        {
          tier: "epic",
          name: "Overcharge Burst",
          cost: 420,
          desc: "Every 4th shot fires an extra bolt with +60% damage.",
          apply: (stats) => {
            stats.barrageEvery = 4;
            stats.barrageProjectiles = (stats.barrageProjectiles || 0) + 1;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.6);
          }
        }
      ]
    },
    {
      id: "repeater",
      tier: "common",
      name: "Repeater Array",
      desc: "Rapid-fire pulses for sustained pressure.",
      tags: ["Rapid", "Sustain"],
      unlock: { rank: 1, credits: 80 },
      stats: { damage: 7, fireRate: 6.2, energyCost: 12, bulletSpeed: 520, projectiles: 1, spread: 0.12 },
      upgrades: [
        { tier: "common", name: "Servo Feed I", cost: 110, desc: "+18% fire rate.", apply: (stats) => { stats.fireRate *= 1.18; } },
        { tier: "uncommon", name: "Servo Feed II", cost: 170, desc: "+18% fire rate.", apply: (stats) => { stats.fireRate *= 1.18; } },
        { tier: "uncommon", name: "Heat Sinks", cost: 230, desc: "-20% energy cost.", apply: (stats) => { stats.energyCost *= 0.8; } },
        { tier: "rare", name: "Staccato Rails", cost: 300, desc: "+20% bullet speed, -spread.", apply: (stats) => { stats.bulletSpeed *= 1.2; stats.spread = Math.max(0.08, stats.spread - 0.04); } },
        {
          tier: "epic",
          name: "Saturation Burst",
          cost: 420,
          desc: "Every 6th shot fires 2 extra bolts.",
          apply: (stats) => {
            stats.barrageEvery = 6;
            stats.barrageProjectiles = (stats.barrageProjectiles || 0) + 2;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.35);
          }
        }
      ]
    },
    {
      id: "rail",
      tier: "epic",
      name: "Rail Lance",
      desc: "High-velocity slugs for precision bursts.",
      tags: ["Precision", "High Damage"],
      unlock: { rank: 4, credits: 260 },
      stats: { damage: 28, fireRate: 1.5, energyCost: 28, bulletSpeed: 760, projectiles: 1, spread: 0.045 },
      upgrades: [
        { tier: "uncommon", name: "Mag-Rail Focus I", cost: 180, desc: "+25% bullet speed, +10% damage.", apply: (stats) => { stats.bulletSpeed *= 1.25; stats.damage *= 1.1; } },
        { tier: "rare", name: "Mag-Rail Focus II", cost: 250, desc: "+25% bullet speed, +10% damage.", apply: (stats) => { stats.bulletSpeed *= 1.25; stats.damage *= 1.1; } },
        { tier: "rare", name: "Recoil Dampers", cost: 320, desc: "+25% fire rate, -spread.", apply: (stats) => { stats.fireRate *= 1.25; stats.spread = Math.max(0.02, stats.spread - 0.02); } },
        { tier: "epic", name: "Penetrator Core", cost: 420, desc: "Shots pierce two targets.", apply: (stats) => { stats.pierce += 2; } },
        {
          tier: "legendary",
          name: "Singularity Charge",
          cost: 520,
          desc: "Every 3rd shot overcharges for +120% damage and +1 pierce.",
          apply: (stats) => {
            stats.barrageEvery = 3;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 2.2);
            stats.barragePierce = Math.max(stats.barragePierce || 0, 1);
          }
        }
      ]
    },
    {
      id: "scatter",
      tier: "uncommon",
      name: "Scattershot",
      desc: "Wide cone for close-range pressure.",
      tags: ["Spread", "Close"],
      unlock: { rank: 2, credits: 140 },
      stats: { damage: 7, fireRate: 3.2, energyCost: 20, bulletSpeed: 460, projectiles: 4, spread: 0.3 },
      upgrades: [
        { tier: "common", name: "Choke Ring I", cost: 130, desc: "-spread, +10% bullet speed.", apply: (stats) => { stats.spread = Math.max(0.18, stats.spread - 0.08); stats.bulletSpeed *= 1.1; } },
        { tier: "uncommon", name: "Choke Ring II", cost: 190, desc: "-spread.", apply: (stats) => { stats.spread = Math.max(0.14, stats.spread - 0.06); } },
        { tier: "rare", name: "Heavy Shot", cost: 260, desc: "+25% damage.", apply: (stats) => { stats.damage *= 1.25; } },
        {
          tier: "rare",
          name: "Concussive Pellets",
          cost: 320,
          desc: "Pellets slow targets.",
          apply: (stats) => {
            stats.slowChance = (stats.slowChance || 0) + 0.18;
            stats.slowDuration = Math.max(stats.slowDuration || 0, 1.4);
          }
        },
        {
          tier: "epic",
          name: "Overpressure Barrage",
          cost: 420,
          desc: "Every 4th shot overloads pellets for +90% damage and splash.",
          apply: (stats) => {
            stats.barrageEvery = 4;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.9);
            stats.barrageSplashRadius = Math.max(stats.barrageSplashRadius || 0, 18);
            stats.barrageSplashDamage = Math.max(stats.barrageSplashDamage || 0, 0.55);
          }
        }
      ]
    },
    {
      id: "volley",
      tier: "uncommon",
      name: "Volley Cannon",
      desc: "Triple-bolt volleys with high burst damage.",
      tags: ["Burst", "Mid-range"],
      unlock: { rank: 2, credits: 160 },
      stats: { damage: 10, fireRate: 3.1, energyCost: 19, bulletSpeed: 500, projectiles: 3, spread: 0.22 },
      upgrades: [
        { tier: "common", name: "Ballistic Tuning I", cost: 130, desc: "+12% bullet speed, -spread.", apply: (stats) => { stats.bulletSpeed *= 1.12; stats.spread = Math.max(0.18, stats.spread - 0.04); } },
        { tier: "uncommon", name: "Ballistic Tuning II", cost: 190, desc: "+12% bullet speed, -spread.", apply: (stats) => { stats.bulletSpeed *= 1.12; stats.spread = Math.max(0.14, stats.spread - 0.04); } },
        { tier: "rare", name: "Capacitor Rounds", cost: 260, desc: "+20% damage.", apply: (stats) => { stats.damage *= 1.2; } },
        { tier: "rare", name: "Volley Rack", cost: 320, desc: "+1 projectile.", apply: (stats) => { stats.projectiles += 1; } },
        {
          tier: "epic",
          name: "Backline Salvo",
          cost: 420,
          desc: "Every 5th shot fires 2 extra bolts with +50% damage and pierce.",
          apply: (stats) => {
            stats.barrageEvery = 5;
            stats.barrageProjectiles = (stats.barrageProjectiles || 0) + 2;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.5);
            stats.barragePierce = Math.max(stats.barragePierce || 0, 1);
          }
        }
      ]
    },
    {
      id: "plasma",
      tier: "epic",
      name: "Plasma Burst",
      desc: "Slow shots with splash damage.",
      tags: ["Splash", "Area"],
      unlock: { rank: 4, credits: 220 },
      stats: { damage: 18, fireRate: 2.2, energyCost: 22, bulletSpeed: 440, projectiles: 1, spread: 0.12, splashRadius: 26 },
      upgrades: [
        { tier: "uncommon", name: "Volatile Mix I", cost: 170, desc: "+30% splash radius.", apply: (stats) => { stats.splashRadius *= 1.3; } },
        { tier: "rare", name: "Volatile Mix II", cost: 240, desc: "+25% splash radius.", apply: (stats) => { stats.splashRadius *= 1.25; } },
        { tier: "rare", name: "Pressure Injectors", cost: 320, desc: "+25% damage.", apply: (stats) => { stats.damage *= 1.25; stats.splashDamage = Math.max(stats.splashDamage || 0.6, 0.75); } },
        { tier: "epic", name: "Thermal Focus", cost: 400, desc: "+20% fire rate.", apply: (stats) => { stats.fireRate *= 1.2; } },
        {
          tier: "legendary",
          name: "Plasma Superheat",
          cost: 520,
          desc: "Every 3rd shot detonates in a larger blast with +40% damage.",
          apply: (stats) => {
            stats.barrageEvery = 3;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.4);
            stats.barrageSplashRadius = Math.max(stats.barrageSplashRadius || 0, 52);
            stats.barrageSplashDamage = Math.max(stats.barrageSplashDamage || 0, 0.85);
          }
        }
      ]
    },
    {
      id: "siege",
      tier: "epic",
      name: "Siege Mortar",
      desc: "Slow siege shells that detonate in a wide radius.",
      tags: ["Splash", "Heavy"],
      unlock: { rank: 4, credits: 240 },
      stats: { damage: 28, fireRate: 1.2, energyCost: 28, bulletSpeed: 360, projectiles: 1, spread: 0.18, splashRadius: 38, splashDamage: 0.75 },
      upgrades: [
        { tier: "uncommon", name: "Reinforced Payloads I", cost: 170, desc: "+20% splash radius.", apply: (stats) => { stats.splashRadius *= 1.2; } },
        { tier: "rare", name: "Reinforced Payloads II", cost: 240, desc: "+20% splash radius.", apply: (stats) => { stats.splashRadius *= 1.2; } },
        { tier: "rare", name: "Siege Core", cost: 320, desc: "+30% damage.", apply: (stats) => { stats.damage *= 1.3; stats.splashDamage = Math.max(stats.splashDamage || 0.6, 0.85); } },
        { tier: "epic", name: "Cycler Assembly", cost: 400, desc: "+20% fire rate.", apply: (stats) => { stats.fireRate *= 1.2; } },
        {
          tier: "legendary",
          name: "Cataclysm Charge",
          cost: 520,
          desc: "Every 4th shot detonates with massive splash and +50% damage.",
          apply: (stats) => {
            stats.barrageEvery = 4;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.5);
            stats.barrageSplashRadius = Math.max(stats.barrageSplashRadius || 0, 64);
            stats.barrageSplashDamage = Math.max(stats.barrageSplashDamage || 0, 0.9);
          }
        }
      ]
    },
    {
      id: "ion",
      tier: "legendary",
      name: "Ion Stream",
      desc: "Rapid fire that slows targets and arcs to nearby foes.",
      tags: ["Rapid", "Debuff", "Arc"],
      unlock: { rank: 5, credits: 260, faction: { id: "nova", rep: 12 } },
      stats: {
        damage: 6,
        fireRate: 7.5,
        energyCost: 11,
        bulletSpeed: 420,
        projectiles: 1,
        spread: 0.16,
        slowChance: 0.25,
        slowDuration: 1.1,
        arcDamage: 0.55,
        arcRadius: 140,
        arcChains: 1,
        arcRequiresSlow: true
      },
      upgrades: [
        {
          tier: "uncommon",
          name: "Stasis Filament I",
          cost: 170,
          desc: "+15% slow chance.",
          apply: (stats) => {
            stats.slowChance = (stats.slowChance || 0) + 0.15;
            stats.slowDuration = Math.max(stats.slowDuration || 0, 1.3);
          }
        },
        {
          tier: "rare",
          name: "Stasis Filament II",
          cost: 240,
          desc: "+15% slow chance.",
          apply: (stats) => {
            stats.slowChance = (stats.slowChance || 0) + 0.15;
            stats.slowDuration = Math.max(stats.slowDuration || 0, 1.5);
          }
        },
        { tier: "rare", name: "Ion Conduits I", cost: 320, desc: "-18% energy cost.", apply: (stats) => { stats.energyCost *= 0.82; } },
        { tier: "epic", name: "Ion Conduits II", cost: 400, desc: "-18% energy cost.", apply: (stats) => { stats.energyCost *= 0.82; } },
        {
          tier: "legendary",
          name: "Arc Cascade",
          cost: 520,
          desc: "+1 arc chain, +25% arc damage, +15% arc radius.",
          apply: (stats) => {
            stats.arcChains = (stats.arcChains || 0) + 1;
            stats.arcDamage *= 1.25;
            stats.arcRadius *= 1.15;
          }
        }
      ]
    }
  ];

  const SECONDARIES = [
    {
      id: "emp",
      tier: "common",
      name: "EMP Pulse",
      desc: "Disables enemies in a radius.",
      cooldown: 18,
      unlock: { rank: 2 },
      activate: (ship, state) => {
        const radius = 160;
        const enemies = listEnemies();
        enemies.forEach((enemy) => {
          if (distanceBetween(ship, enemy) <= radius) {
            enemy.slowTimer = Math.max(enemy.slowTimer, 2.2);
            enemy.fireCooldown = Math.max(enemy.fireCooldown, 2);
          }
        });
        emitPulse(ship.x, ship.y, "#57e0ff", radius);
        log("EMP pulse detonated.");
      }
    },
    {
      id: "decoy",
      tier: "rare",
      name: "Decoy Drone",
      desc: "Deploys a decoy that draws fire.",
      cooldown: 22,
      unlock: { rank: 3, credits: 160 },
      activate: (ship, state) => {
        state.decoy = {
          x: ship.x + Math.cos(ship.angle) * 40,
          y: ship.y + Math.sin(ship.angle) * 40,
          timer: 6
        };
        emitPulse(state.decoy.x, state.decoy.y, "#f6c65f", 60);
        log("Decoy drone deployed.");
      }
    },
    {
      id: "mine",
      tier: "epic",
      name: "Mine Layer",
      desc: "Drops proximity mines behind you.",
      cooldown: 16,
      unlock: { rank: 4, credits: 200 },
      activate: (ship, state) => {
        state.mines.push({
          x: ship.x - Math.cos(ship.angle) * 40,
          y: ship.y - Math.sin(ship.angle) * 40,
          radius: 18,
          timer: 8
        });
        log("Mine deployed.");
      }
    },
    {
      id: "repair",
      tier: "uncommon",
      name: "Repair Burst",
      desc: "Restores hull and shield instantly.",
      cooldown: 20,
      unlock: { rank: 2, credits: 140 },
      activate: (ship, state) => {
        ship.health = Math.min(ship.maxHealth, ship.health + ship.maxHealth * 0.25);
        ship.shield = Math.min(ship.maxShield, ship.shield + ship.maxShield * 0.3);
        emitPulse(ship.x, ship.y, "#6ee7b7", 120);
        log("Repair burst triggered.");
      }
    }
  ];

  const PART_RARITIES = {
    common: { label: "Common", mult: 1 },
    uncommon: { label: "Uncommon", mult: 1.2 },
    rare: { label: "Rare", mult: 1.4 },
    epic: { label: "Epic", mult: 1.7 },
    legendary: { label: "Legendary", mult: 2 }
  };

  const PART_SLOTS = ["barrel", "core", "targeting", "thruster"];

  const PART_TEMPLATES = {
    barrel: { name: "Barrel", stats: { damage: [2, 6], bulletSpeed: [20, 60] } },
    core: { name: "Core", stats: { fireRate: [0.3, 0.9], energyCost: [-2, -0.5] } },
    targeting: { name: "Targeting", stats: { critChance: [0.02, 0.06], spread: [-0.05, -0.02] } },
    thruster: { name: "Thruster", stats: { maxSpeed: [8, 22], accel: [18, 50] } }
  };

  const FACTIONS = [
    { id: "nova", name: "Nova Forge", desc: "Weapons research guild." },
    { id: "aegis", name: "Aegis Union", desc: "Defensive fleet coalition." },
    { id: "vortex", name: "Vortex Syndicate", desc: "Speed and mobility cartel." }
  ];

  const SECTOR_MODIFIERS = [
    {
      id: "nebula",
      name: "Nebula Drift",
      desc: "Weapon spread +20%, energy regen -10%.",
      player: { spreadMult: 1.2, energyRegen: 0.9 },
      enemy: { fireRate: 0.95 }
    },
    {
      id: "meteor",
      name: "Meteor Wake",
      desc: "Speed -10%, enemy speed +10%.",
      player: { speed: 0.9 },
      enemy: { speed: 1.1 }
    },
    {
      id: "flare",
      name: "Solar Flare",
      desc: "Shield regen -20%, enemy fire rate -10%.",
      player: { shieldRegen: 0.8 },
      enemy: { fireRate: 0.9 }
    },
    {
      id: "surge",
      name: "Power Surge",
      desc: "Energy regen +15%, energy costs +10%.",
      player: { energyRegen: 1.15, energyCost: 1.1 },
      enemy: { damage: 1 }
    }
  ];

  const CONTRACT_DEFS = [
    {
      id: "elimination",
      title: "Elimination Order",
      desc: "Destroy 12 hostile ships.",
      target: 12,
      type: "kills",
      reward: { credits: 120, xp: 50, rep: 6 }
    },
    {
      id: "ace",
      title: "Ace Intercept",
      desc: "Defeat an elite or boss.",
      target: 1,
      type: "elite",
      reward: { credits: 160, xp: 70, rep: 8 }
    },
    {
      id: "flawless",
      title: "Flawless Drift",
      desc: "Avoid hull damage for 40 seconds.",
      target: 40,
      type: "noDamage",
      reward: { credits: 140, xp: 60, rep: 7 }
    }
  ];

  const SALVAGE_CACHE = {
    name: "Vortex Salvage Cache",
    pityThreshold: 4,
    rareIds: ["part-rare", "part-epic"],
    table: [
      { id: "credits-small", label: "Credit Payout", weight: 34, roll: () => ({ type: "credits", amount: safeRandInt(120, 200) }) },
      { id: "credits-jackpot", label: "Credit Jackpot", weight: 14, roll: () => ({ type: "credits", amount: safeRandInt(260, 420) }) },
      { id: "blueprints", label: "Blueprint Cache", weight: 16, roll: () => ({ type: "blueprints", amount: safeRandInt(1, 2) }) },
      { id: "part-uncommon", label: "Enhanced Attachment", weight: 18, roll: () => ({ type: "part", part: safeGeneratePart("uncommon") }) },
      { id: "part-rare", label: "Rare Attachment", weight: 10, roll: () => ({ type: "part", part: safeGeneratePart("rare") }) },
      { id: "part-epic", label: "Epic Attachment", weight: 6, roll: () => ({ type: "part", part: safeGeneratePart("epic") }) }
    ]
  };

  const ELITE_MODS = [
    { id: "shielded", name: "Shielded", color: "#7ca8ff", mods: { maxShield: 0.6 } },
    { id: "agile", name: "Agile", color: "#6ee7b7", mods: { speed: 1.2, accel: 1.2 } },
    { id: "berserk", name: "Berserk", color: "#f48b7f", mods: { damage: 1.25, fireRate: 1.2 } }
  ];

  const HANGAR_UPGRADES = [
    {
      id: "hull",
      name: "Hull Plating",
      tier: "common",
      category: "Defense",
      desc: "+10 max hull per level.",
      maxLevel: 12,
      baseCost: 1,
      costScale: 1.18,
      apply: (stats, level) => {
        stats.maxHealth += level * 10;
      }
    },
    {
      id: "shield",
      name: "Shield Array",
      tier: "common",
      category: "Defense",
      desc: "+8 max shield per level.",
      maxLevel: 12,
      baseCost: 1,
      costScale: 1.18,
      apply: (stats, level) => {
        stats.maxShield += level * 8;
      }
    },
    {
      id: "shield-regenerator",
      name: "Shield Regenerator",
      tier: "uncommon",
      category: "Defense",
      desc: "+1.5 shield regen per level.",
      maxLevel: 10,
      baseCost: 1,
      costScale: 1.22,
      apply: (stats, level) => {
        stats.shieldRegen += level * 1.5;
      }
    },
    {
      id: "damage-dampers",
      name: "Damage Dampers",
      tier: "rare",
      category: "Defense",
      desc: "Reduce damage taken by 1.8% per level.",
      maxLevel: 8,
      baseCost: 2,
      costScale: 1.25,
      apply: (stats, level) => {
        const current = stats.damageReduction || 0;
        const next = current + level * 0.018;
        stats.damageReduction = current > 0.3 ? current : Math.min(0.3, next);
      }
    },
    {
      id: "reactive-repair",
      name: "Reactive Repair",
      tier: "epic",
      category: "Defense",
      desc: "Restore 2.5% hull on kill per level.",
      maxLevel: 6,
      baseCost: 2,
      costScale: 1.3,
      apply: (stats, level) => {
        const current = stats.healOnKill || 0;
        const next = current + level * 0.025;
        stats.healOnKill = current > 0.2 ? current : Math.min(0.2, next);
      }
    },
    {
      id: "aegis-relay",
      name: "Aegis Relay",
      tier: "legendary",
      category: "Defense",
      desc: "Periodic aegis pulse restores shields and slows nearby foes.",
      maxLevel: 3,
      baseCost: 3,
      costScale: 1.35,
      apply: (stats, level) => {
        if (level <= 0) return;
        const cooldown = Math.max(14, 24 - level * 2);
        stats.aegisCooldown = stats.aegisCooldown
          ? Math.min(stats.aegisCooldown, cooldown)
          : cooldown;
        stats.aegisShieldRestore = Math.max(stats.aegisShieldRestore || 0, 0.18 + level * 0.06);
        stats.aegisPulseRadius = Math.max(stats.aegisPulseRadius || 0, 120 + level * 25);
        stats.aegisPulseDamage = Math.max(stats.aegisPulseDamage || 0, 12 + level * 6);
        stats.aegisPulseSlow = Math.max(stats.aegisPulseSlow || 0, 0.6 + level * 0.25);
      }
    },
    {
      id: "weapon-calibration",
      name: "Weapon Calibration",
      tier: "common",
      category: "Offense",
      desc: "+5% weapon damage per level.",
      maxLevel: 12,
      baseCost: 1,
      costScale: 1.18,
      apply: (stats, level) => {
        stats.damage *= 1 + level * 0.05;
      }
    },
    {
      id: "fire-control",
      name: "Fire Control",
      tier: "uncommon",
      category: "Offense",
      desc: "+5% fire rate per level.",
      maxLevel: 10,
      baseCost: 1,
      costScale: 1.22,
      apply: (stats, level) => {
        stats.fireRate *= 1 + level * 0.05;
      }
    },
    {
      id: "targeting",
      name: "Targeting Suite",
      tier: "rare",
      category: "Offense",
      desc: "+2% crit chance per level.",
      maxLevel: 8,
      baseCost: 2,
      costScale: 1.25,
      apply: (stats, level) => {
        const current = stats.critChance || 0;
        const next = current + level * 0.02;
        stats.critChance = current > 0.4 ? current : Math.min(0.4, next);
      }
    },
    {
      id: "munitions-loader",
      name: "Munitions Loader",
      tier: "rare",
      category: "Offense",
      desc: "+30 projectile speed per level.",
      maxLevel: 8,
      baseCost: 2,
      costScale: 1.25,
      apply: (stats, level) => {
        stats.bulletSpeed += level * 30;
      }
    },
    {
      id: "amplifier-core",
      name: "Amplifier Core",
      tier: "epic",
      category: "Offense",
      desc: "+0.15 crit multiplier per level.",
      maxLevel: 6,
      baseCost: 2,
      costScale: 1.3,
      apply: (stats, level) => {
        stats.critMultiplier += level * 0.15;
      }
    },
    {
      id: "barrage-sync",
      name: "Barrage Sync",
      tier: "legendary",
      category: "Offense",
      desc: "Every few shots, unleash a barrage with extra projectiles.",
      maxLevel: 3,
      baseCost: 3,
      costScale: 1.35,
      apply: (stats, level) => {
        if (level <= 0) return;
        const cadence = Math.max(4, 7 - level);
        stats.barrageEvery = stats.barrageEvery
          ? Math.min(stats.barrageEvery, cadence)
          : cadence;
        stats.barrageProjectiles = Math.max(stats.barrageProjectiles || 0, 1 + Math.floor(level / 2));
        stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.25 + level * 0.1);
        if (level >= 2) {
          stats.barragePierce = Math.max(stats.barragePierce || 0, 1);
        }
      }
    },
    {
      id: "thrusters",
      name: "Vector Thrusters",
      tier: "common",
      category: "Mobility",
      desc: "+8 max speed and +14 accel per level.",
      maxLevel: 10,
      baseCost: 1,
      costScale: 1.18,
      apply: (stats, level) => {
        stats.maxSpeed += level * 8;
        stats.accel += level * 14;
      }
    },
    {
      id: "attitude-control",
      name: "Attitude Control",
      tier: "uncommon",
      category: "Mobility",
      desc: "+0.25 turn rate per level (caps at 6.5).",
      maxLevel: 8,
      baseCost: 1,
      costScale: 1.22,
      apply: (stats, level) => {
        const target = stats.turnRate + level * 0.25;
        stats.turnRate = stats.turnRate > 6.5 ? stats.turnRate : Math.min(6.5, target);
      }
    },
    {
      id: "inertial-dampers",
      name: "Inertial Dampers",
      tier: "rare",
      category: "Mobility",
      desc: "Reduce drift loss by 0.1 per level (caps at 1.4).",
      maxLevel: 8,
      baseCost: 2,
      costScale: 1.25,
      apply: (stats, level) => {
        const target = stats.damping - level * 0.1;
        stats.damping = stats.damping < 1.4 ? stats.damping : Math.max(1.4, target);
      }
    },
    {
      id: "boost-couplers",
      name: "Boost Couplers",
      tier: "epic",
      category: "Mobility",
      desc: "+6% boost speed and -1 boost cost per level.",
      maxLevel: 6,
      baseCost: 2,
      costScale: 1.3,
      apply: (stats, level) => {
        stats.boostMultiplier += level * 0.06;
        stats.boostCost = Math.max(6, stats.boostCost - level);
      }
    },
    {
      id: "reactor",
      name: "Reactor Coils",
      tier: "common",
      category: "Utility",
      desc: "+5 energy regen per level.",
      maxLevel: 10,
      baseCost: 1,
      costScale: 1.18,
      apply: (stats, level) => {
        stats.energyRegen += level * 5;
      }
    },
    {
      id: "capacitor-banks",
      name: "Capacitor Banks",
      tier: "common",
      category: "Utility",
      desc: "+14 max energy per level.",
      maxLevel: 10,
      baseCost: 1,
      costScale: 1.18,
      apply: (stats, level) => {
        stats.maxEnergy += level * 14;
      }
    },
    {
      id: "efficiency-tuning",
      name: "Efficiency Tuning",
      tier: "uncommon",
      category: "Utility",
      desc: "-0.6 energy cost per level (caps at 8).",
      maxLevel: 8,
      baseCost: 1,
      costScale: 1.22,
      apply: (stats, level) => {
        stats.energyCost = Math.max(8, stats.energyCost - level * 0.6);
      }
    },
    {
      id: "salvage-magnet",
      name: "Salvage Magnetics",
      tier: "rare",
      category: "Utility",
      desc: "+4% salvage bonus per level.",
      maxLevel: 6,
      baseCost: 2,
      costScale: 1.25,
      apply: (stats, level) => {
        const current = stats.salvageBonus || 0;
        const next = current + level * 0.04;
        stats.salvageBonus = current > 0.5 ? current : Math.min(0.5, next);
      }
    },
    {
      id: "tactical-scanner",
      name: "Tactical Scanner",
      tier: "rare",
      category: "Utility",
      desc: "+4% XP gain per level.",
      maxLevel: 6,
      baseCost: 2,
      costScale: 1.25,
      apply: (stats, level) => {
        const current = stats.xpBonus || 0;
        const next = current + level * 0.04;
        stats.xpBonus = current > 0.5 ? current : Math.min(0.5, next);
      }
    },
    {
      id: "upgrade-forecast",
      name: "Upgrade Forecast",
      tier: "epic",
      category: "Utility",
      desc: "+0.06 upgrade luck per level (caps at 0.4).",
      maxLevel: 5,
      baseCost: 2,
      costScale: 1.3,
      apply: (stats, level) => {
        const current = stats.upgradeLuck || 0;
        const next = current + level * 0.06;
        stats.upgradeLuck = current > 0.4 ? current : Math.min(0.4, next);
      }
    },
    {
      id: "energy-siphon",
      name: "Energy Siphon",
      tier: "legendary",
      category: "Utility",
      desc: "Restore 6 energy on kill per level.",
      maxLevel: 3,
      baseCost: 3,
      costScale: 1.35,
      apply: (stats, level) => {
        stats.energyOnKill += level * 6;
      }
    }
  ];

  const FRONTIER_STARTERS = ["vanguard", "scout"];

  const FRONTIER_UPGRADES = [
    {
      id: "hull-plating",
      name: "Hull Plating",
      tier: "common",
      category: "Defense",
      desc: "+18 max hull.",
      maxLevel: 6,
      baseCost: 90,
      apply: (stats) => {
        stats.maxHealth += 18;
      }
    },
    {
      id: "shield-capacitor",
      name: "Shield Capacitor",
      tier: "common",
      category: "Defense",
      desc: "+20 max shield.",
      maxLevel: 6,
      baseCost: 100,
      apply: (stats) => {
        stats.maxShield += 20;
      }
    },
    {
      id: "shield-regenerator",
      name: "Shield Regenerator",
      tier: "uncommon",
      category: "Defense",
      desc: "+2 shield regen.",
      maxLevel: 5,
      baseCost: 120,
      apply: (stats) => {
        stats.shieldRegen += 2;
      }
    },
    {
      id: "energy-cell",
      name: "Energy Cells",
      tier: "common",
      category: "Utility",
      desc: "+18 max energy.",
      maxLevel: 6,
      baseCost: 90,
      apply: (stats) => {
        stats.maxEnergy += 18;
      }
    },
    {
      id: "reactor-tuning",
      name: "Reactor Tuning",
      tier: "uncommon",
      category: "Utility",
      desc: "+3 energy regen.",
      maxLevel: 5,
      baseCost: 120,
      apply: (stats) => {
        stats.energyRegen += 3;
      }
    },
    {
      id: "weapon-calibration",
      name: "Weapon Calibration",
      tier: "uncommon",
      category: "Offense",
      desc: "+12% weapon damage.",
      maxLevel: 6,
      baseCost: 130,
      apply: (stats) => {
        stats.damage *= 1.12;
      }
    },
    {
      id: "rate-accelerator",
      name: "Rate Accelerator",
      tier: "rare",
      category: "Offense",
      desc: "+12% fire rate.",
      maxLevel: 5,
      baseCost: 140,
      apply: (stats) => {
        stats.fireRate *= 1.12;
      }
    },
    {
      id: "thruster-rigging",
      name: "Thruster Rigging",
      tier: "uncommon",
      category: "Mobility",
      desc: "+10% speed, +14 accel.",
      maxLevel: 5,
      baseCost: 120,
      apply: (stats) => {
        stats.maxSpeed *= 1.1;
        stats.accel += 14;
      }
    },
    {
      id: "gimbal-array",
      name: "Gimbal Array",
      tier: "rare",
      category: "Mobility",
      desc: "+0.35 turn rate, +8 accel.",
      maxLevel: 4,
      baseCost: 140,
      apply: (stats) => {
        stats.turnRate += 0.35;
        stats.accel += 8;
      }
    }
  ];

  const FIELD_UPGRADES = [
    {
      id: "halo-emitter",
      name: "Halo Emitter",
      tier: "common",
      category: "Control",
      kind: "skill",
      skillId: "halo",
      desc: "Skill: damaging halo scales in radius, power, and pulse rate.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        const radiusBoost = 16 + level * 6;
        const damageBoost = 4 + level * 3;
        const intervalDrop = 0.02 + level * 0.01;
        ship.auraRadius = (ship.auraRadius || 60) + radiusBoost;
        ship.auraDamage = (ship.auraDamage || 8) + damageBoost;
        ship.auraInterval = Math.max(0.22, (ship.auraInterval || 0.48) - intervalDrop);
      }
    },
    {
      id: "minefield-protocol",
      name: "Minefield Protocol",
      tier: "common",
      category: "Control",
      kind: "skill",
      skillId: "minefield",
      desc: "Skill: automated mines grow in damage, radius, and drop rate.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        const dropBoost = 0.12 + level * 0.05;
        const radiusBoost = 8 + level * 6;
        const damageBoost = 12 + level * 8;
        const durationBoost = 0.4 + level * 0.35;
        const intervalDrop = 0.05 + level * 0.03;
        ship.mineDropChance = Math.min(0.85, (ship.mineDropChance || 0) + dropBoost);
        ship.mineRadius = (ship.mineRadius || 30) + radiusBoost;
        ship.mineDamage = (ship.mineDamage || 28) + damageBoost;
        ship.mineDuration = Math.min(10, (ship.mineDuration || 5) + durationBoost);
        ship.mineInterval = Math.max(0.6, (ship.mineInterval || 1.2) - intervalDrop);
      }
    },
    {
      id: "escort-wing",
      name: "Escort Wing",
      tier: "common",
      category: "Offense",
      kind: "skill",
      skillId: "escort",
      desc: "Skill: helper ships scale in count, damage, and fire rate.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        const damageBoost = 0.08 + level * 0.04;
        ship.helperCount = Math.min(5, (ship.helperCount || 0) + 1);
        ship.helperDamageRatio = Math.min(0.9, (ship.helperDamageRatio || 0.25) + damageBoost);
        ship.helperFireRate = Math.max(ship.helperFireRate || 1.1, 0.9 + level * 0.35);
        ship.helperRange = Math.max(ship.helperRange || 320, 320 + level * 80);
        ship.helperOrbitRadius = Math.max(ship.helperOrbitRadius || 26, 26 + level * 5);
        ship.helperOrbitSpeed = Math.max(ship.helperOrbitSpeed || 1.4, 1.4 + level * 0.2);
      }
    },
    {
      id: "shockwave-core",
      name: "Shockwave Core",
      tier: "common",
      category: "Control",
      kind: "skill",
      skillId: "shockwave",
      desc: "Skill: periodic shockwaves hit harder, wider, and more often.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        const intervalDrop = 0.3 + level * 0.12;
        const radiusBoost = 14 + level * 8;
        const damageBoost = 10 + level * 7;
        ship.shockwaveInterval = Math.max(1.6, (ship.shockwaveInterval || 4.4) - intervalDrop);
        ship.shockwaveRadius = (ship.shockwaveRadius || 100) + radiusBoost;
        ship.shockwaveDamage = (ship.shockwaveDamage || 20) + damageBoost;
        ship.shockwaveSlow = Math.min(3, (ship.shockwaveSlow || 0.7) + 0.2 + level * 0.15);
      }
    },
    {
      id: "harrier-missiles",
      name: "Harrier Missiles",
      tier: "common",
      category: "Offense",
      kind: "skill",
      skillId: "harrier",
      desc: "Skill: seeker missiles ramp up in count, damage, and cadence.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        const intervalDrop = 0.28 + level * 0.12;
        const damageBoost = 12 + level * 6;
        ship.missileInterval = Math.max(1.1, (ship.missileInterval || 3.8) - intervalDrop);
        ship.missileDamage = (ship.missileDamage || 22) + damageBoost;
        ship.missileCount = Math.min(5, (ship.missileCount || 0) + 1);
        ship.missileSpeed = Math.max(ship.missileSpeed || 520, 520 + level * 70);
      }
    },
    {
      id: "barrage-matrix",
      name: "Barrage Matrix",
      tier: "common",
      category: "Offense",
      kind: "skill",
      skillId: "barrage",
      desc: "Skill: rhythmic barrages add projectiles, pierce, and splash.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        const cadence = Math.max(4, 8 - level);
        const projectileBonus = 1 + Math.floor(level / 2);
        ship.barrageEvery = ship.barrageEvery ? Math.min(ship.barrageEvery, cadence) : cadence;
        ship.barrageProjectiles = Math.max(ship.barrageProjectiles || 0, projectileBonus);
        ship.barragePierce = Math.max(ship.barragePierce || 0, Math.floor((level + 1) / 2));
        ship.barrageBonusDamage = Math.max(ship.barrageBonusDamage || 1, 1 + 0.15 + level * 0.08);
        ship.barrageSplashRadius = Math.max(ship.barrageSplashRadius || 0, 18 + level * 8);
        ship.barrageSplashDamage = Math.max(ship.barrageSplashDamage || 0.6, 0.6 + level * 0.05);
      }
    },
    {
      id: "arc-lattice",
      name: "Arc Lattice",
      tier: "common",
      category: "Control",
      kind: "skill",
      skillId: "arc-lattice",
      desc: "Skill: shots chain lightning with growing range and damage.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        ship.arcDamage = Math.max(ship.arcDamage || 0, 0.25 + level * 0.1);
        ship.arcRadius = Math.max(ship.arcRadius || 0, 120 + level * 25);
        ship.arcChains = Math.max(ship.arcChains || 0, 1 + Math.floor((level - 1) / 2));
        ship.arcRequiresSlow = false;
        ship.slowChance = Math.min(0.6, (ship.slowChance || 0) + 0.04 + level * 0.02);
        ship.slowDuration = Math.max(ship.slowDuration || 0, 1.2 + level * 0.2);
      }
    },
    {
      id: "bulwark-resonance",
      name: "Bulwark Resonance",
      tier: "common",
      category: "Defense",
      kind: "skill",
      skillId: "bulwark",
      desc: "Skill: shield breaks trigger restoring aegis pulses.",
      maxStacks: 5,
      apply: (ship, level = 1) => {
        const cooldown = Math.max(10, 22 - level * 2.2);
        ship.aegisCooldown = ship.aegisCooldown
          ? Math.min(ship.aegisCooldown, cooldown)
          : cooldown;
        ship.aegisShieldRestore = Math.max(ship.aegisShieldRestore || 0, 0.16 + level * 0.06);
        ship.aegisPulseRadius = Math.max(ship.aegisPulseRadius || 0, 110 + level * 24);
        ship.aegisPulseDamage = Math.max(ship.aegisPulseDamage || 0, 12 + level * 8);
        ship.aegisPulseSlow = Math.max(ship.aegisPulseSlow || 0, 0.7 + level * 0.28);
      }
    },
    {
      id: "calibrated-coils",
      name: "Calibrated Coils",
      tier: "common",
      category: "Offense",
      desc: "+12% fire rate.",
      maxStacks: 3,
      apply: (ship) => {
        ship.fireRate *= 1.12;
      }
    },
    {
      id: "coil-optimizers",
      name: "Coil Optimizers",
      tier: "common",
      category: "Utility",
      desc: "-8% energy cost.",
      maxStacks: 3,
      apply: (ship) => {
        ship.energyCost *= 0.92;
      }
    },
    {
      id: "kinetic-fins",
      name: "Kinetic Fins",
      tier: "common",
      category: "Offense",
      desc: "+12% projectile speed.",
      maxStacks: 3,
      apply: (ship) => {
        ship.bulletSpeed *= 1.12;
      }
    },
    {
      id: "hull-patches",
      name: "Hull Patches",
      tier: "common",
      category: "Defense",
      desc: "+20 max hull and heal 12 now.",
      maxStacks: 3,
      apply: (ship) => {
        ship.maxHealth += 20;
        ship.health = Math.min(ship.maxHealth, ship.health + 12);
      }
    },
    {
      id: "shield-patches",
      name: "Shield Patches",
      tier: "common",
      category: "Defense",
      desc: "+15 max shield and +10% regen.",
      maxStacks: 3,
      apply: (ship) => {
        ship.maxShield += 15;
        ship.shield = Math.min(ship.maxShield, ship.shield + 10);
        ship.shieldRegen *= 1.1;
      }
    },
    {
      id: "stability-gyros",
      name: "Stability Gyros",
      tier: "common",
      category: "Utility",
      desc: "-0.03 weapon spread.",
      maxStacks: 3,
      apply: (ship) => {
        ship.spread = Math.max(0.05, ship.spread - 0.03);
      }
    },
    {
      id: "core-overclock",
      name: "Core Overclock",
      tier: "common",
      category: "Offense",
      desc: "+10% weapon damage.",
      maxStacks: 3,
      apply: (ship) => {
        ship.damage *= 1.1;
      }
    },
    {
      id: "targeting-lenses",
      name: "Targeting Lenses",
      tier: "common",
      category: "Offense",
      desc: "+4% crit chance and +0.15 crit damage.",
      maxStacks: 2,
      apply: (ship) => {
        ship.critChance += 0.04;
        ship.critMultiplier += 0.15;
      }
    },
    {
      id: "power-routing",
      name: "Power Routing",
      tier: "common",
      category: "Utility",
      desc: "+20 max energy and +15% regen.",
      maxStacks: 3,
      apply: (ship) => {
        ship.maxEnergy += 20;
        ship.energy = Math.min(ship.maxEnergy, ship.energy + 20);
        ship.energyRegen *= 1.15;
      }
    },
    {
      id: "boost-servos",
      name: "Boost Servos",
      tier: "common",
      category: "Mobility",
      desc: "+12% boost power and -8% boost cost.",
      maxStacks: 3,
      apply: (ship) => {
        ship.boostMultiplier *= 1.12;
        ship.boostCost *= 0.92;
      }
    },
    {
      id: "ablative-lattice",
      name: "Ablative Lattice",
      tier: "common",
      category: "Defense",
      desc: "+10 max hull and 5% damage reduction.",
      maxStacks: 3,
      apply: (ship) => {
        ship.maxHealth += 10;
        ship.damageReduction = Math.min(0.45, ship.damageReduction + 0.05);
      }
    },
    {
      id: "vector-jets",
      name: "Vector Jets",
      tier: "common",
      category: "Mobility",
      desc: "+0.25 turn rate and +10% accel.",
      maxStacks: 3,
      apply: (ship) => {
        ship.turnRate += 0.25;
        ship.accel *= 1.1;
      }
    },
    {
      id: "signal-forecast",
      name: "Signal Forecast",
      tier: "uncommon",
      category: "Strategy",
      desc: "Higher chance for rare upgrades later.",
      maxStacks: 2,
      apply: (ship) => {
        ship.upgradeLuck = Math.min(0.9, (ship.upgradeLuck || 0) + 0.2);
      }
    },
    {
      id: "longrange-intel",
      name: "Longrange Intel",
      tier: "rare",
      category: "Strategy",
      desc: "Higher chance for epic upgrades later.",
      maxStacks: 2,
      apply: (ship) => {
        ship.upgradeLuck = Math.min(0.9, (ship.upgradeLuck || 0) + 0.35);
      }
    },
    {
      id: "oracle-array",
      name: "Oracle Array",
      tier: "epic",
      category: "Strategy",
      desc: "Greatly increases chances for epic and legendary upgrades.",
      maxStacks: 1,
      apply: (ship) => {
        ship.upgradeLuck = Math.min(0.9, (ship.upgradeLuck || 0) + 0.5);
      }
    },
    {
      id: "overcharged-blasters",
      name: "Overcharged Blasters",
      tier: "uncommon",
      category: "Offense",
      desc: "+20% weapon damage.",
      maxStacks: 3,
      apply: (ship) => {
        ship.damage *= 1.2;
      }
    },
    {
      id: "rapid-cyclers",
      name: "Rapid Cyclers",
      tier: "uncommon",
      category: "Offense",
      desc: "+25% fire rate and -10% energy cost.",
      maxStacks: 2,
      apply: (ship) => {
        ship.fireRate *= 1.25;
        ship.energyCost *= 0.9;
      }
    },
    {
      id: "afterburners",
      name: "Afterburners",
      tier: "uncommon",
      category: "Mobility",
      desc: "+15% max speed and +20% accel.",
      maxStacks: 2,
      apply: (ship) => {
        ship.maxSpeed *= 1.15;
        ship.accel *= 1.2;
      }
    },
    {
      id: "attitude-jets",
      name: "Attitude Jets",
      tier: "uncommon",
      category: "Mobility",
      desc: "+25% turn rate.",
      maxStacks: 2,
      apply: (ship) => {
        ship.turnRate *= 1.25;
      }
    },
    {
      id: "reinforced-hull",
      name: "Reinforced Hull",
      tier: "uncommon",
      category: "Defense",
      desc: "+30 max hull and heal 20 now.",
      maxStacks: 2,
      apply: (ship) => {
        ship.maxHealth += 30;
        ship.health = Math.min(ship.maxHealth, ship.health + 20);
      }
    },
    {
      id: "shield-harmonizer",
      name: "Shield Harmonizer",
      tier: "uncommon",
      category: "Defense",
      desc: "+25 max shield and +20% regen.",
      maxStacks: 2,
      apply: (ship) => {
        ship.maxShield += 25;
        ship.shield = Math.min(ship.maxShield, ship.shield + 25);
        ship.shieldRegen *= 1.2;
      }
    },
    {
      id: "reactor-surge",
      name: "Reactor Surge",
      tier: "uncommon",
      category: "Utility",
      desc: "+30 max energy and +20% regen.",
      maxStacks: 2,
      apply: (ship) => {
        ship.maxEnergy += 30;
        ship.energy = Math.min(ship.maxEnergy, ship.energy + 30);
        ship.energyRegen *= 1.2;
      }
    },
    {
      id: "boost-reserves",
      name: "Boost Reserves",
      tier: "uncommon",
      category: "Mobility",
      desc: "+15% boost power and -10% boost cost.",
      maxStacks: 2,
      apply: (ship) => {
        ship.boostMultiplier *= 1.15;
        ship.boostCost *= 0.9;
      }
    },
    {
      id: "strike-accelerators",
      name: "Strike Accelerators",
      tier: "uncommon",
      category: "Offense",
      desc: "+15% fire rate and +10% projectile speed.",
      maxStacks: 2,
      apply: (ship) => {
        ship.fireRate *= 1.15;
        ship.bulletSpeed *= 1.1;
      }
    },
    {
      id: "high-output-cells",
      name: "High Output Cells",
      tier: "uncommon",
      category: "Offense",
      desc: "+18% weapon damage and +10 max energy.",
      maxStacks: 2,
      apply: (ship) => {
        ship.damage *= 1.18;
        ship.maxEnergy += 10;
        ship.energy = Math.min(ship.maxEnergy, ship.energy + 10);
      }
    },
    {
      id: "target-stabilizers",
      name: "Target Stabilizers",
      tier: "uncommon",
      category: "Utility",
      desc: "-0.05 weapon spread and +12% projectile speed.",
      maxStacks: 2,
      apply: (ship) => {
        ship.spread = Math.max(0.04, ship.spread - 0.05);
        ship.bulletSpeed *= 1.12;
      }
    },
    {
      id: "siphon-cells",
      name: "Siphon Cells",
      tier: "uncommon",
      category: "Defense",
      desc: "Restore 3% hull and 6 energy on kill.",
      maxStacks: 2,
      apply: (ship) => {
        ship.healOnKill = Math.max(ship.healOnKill, 0.03);
        ship.energyOnKill = Math.max(ship.energyOnKill, 6);
      }
    },
    {
      id: "shield-overclock",
      name: "Shield Overclock",
      tier: "uncommon",
      category: "Defense",
      desc: "+35 max shield and +15% regen.",
      maxStacks: 2,
      apply: (ship) => {
        ship.maxShield += 35;
        ship.shield = Math.min(ship.maxShield, ship.shield + 20);
        ship.shieldRegen *= 1.15;
      }
    },
    {
      id: "afterimage-thrusters",
      name: "Afterimage Thrusters",
      tier: "uncommon",
      category: "Mobility",
      desc: "+20% accel and +12% max speed.",
      maxStacks: 2,
      apply: (ship) => {
        ship.accel *= 1.2;
        ship.maxSpeed *= 1.12;
      }
    },
    {
      id: "evasive-thrusters",
      name: "Evasive Thrusters",
      tier: "rare",
      category: "Mobility",
      desc: "+10% speed and 8% damage reduction.",
      maxStacks: 2,
      apply: (ship) => {
        ship.maxSpeed *= 1.1;
        ship.damageReduction = Math.min(0.4, ship.damageReduction + 0.08);
      }
    },
    {
      id: "gimbal-core",
      name: "Gimbal Core",
      tier: "rare",
      category: "Mobility",
      desc: "+40% turn rate and +6% max speed.",
      maxStacks: 1,
      apply: (ship) => {
        ship.turnRate *= 1.4;
        ship.maxSpeed *= 1.06;
      }
    },
    {
      id: "precision-targeting",
      name: "Precision Targeting",
      tier: "rare",
      category: "Offense",
      desc: "+8% crit chance and +0.2 crit multiplier.",
      maxStacks: 2,
      apply: (ship) => {
        ship.critChance += 0.08;
        ship.critMultiplier += 0.2;
      }
    },
    {
      id: "ion-rounds",
      name: "Ion Rounds",
      tier: "rare",
      category: "Control",
      desc: "Shots have a 30% chance to slow targets.",
      maxStacks: 1,
      apply: (ship) => {
        ship.slowChance = Math.max(ship.slowChance, 0.3);
        ship.slowDuration = Math.max(ship.slowDuration, 1.4);
      }
    },
    {
      id: "salvage-drones",
      name: "Salvage Drones",
      tier: "rare",
      category: "Utility",
      desc: "+25% credits and +15% XP per kill.",
      maxStacks: 2,
      apply: (ship) => {
        ship.salvageBonus += 0.25;
        ship.xpBonus += 0.15;
      }
    },
    {
      id: "twin-cannons",
      name: "Twin Cannons",
      tier: "rare",
      category: "Offense",
      desc: "+1 projectile per shot with a wider spread.",
      maxStacks: 1,
      apply: (ship) => {
        ship.projectiles += 1;
        ship.spread += 0.08;
      }
    },
    {
      id: "overload-capacitors",
      name: "Overload Capacitors",
      tier: "rare",
      category: "Utility",
      desc: "+40 max energy and +25% regen.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxEnergy += 40;
        ship.energy = Math.min(ship.maxEnergy, ship.energy + 20);
        ship.energyRegen *= 1.25;
      }
    },
    {
      id: "precision-matrix",
      name: "Precision Matrix",
      tier: "rare",
      category: "Offense",
      desc: "+12% crit chance and +0.3 crit damage.",
      maxStacks: 1,
      apply: (ship) => {
        ship.critChance += 0.12;
        ship.critMultiplier += 0.3;
      }
    },
    {
      id: "phase-rounds",
      name: "Phase Rounds",
      tier: "rare",
      category: "Offense",
      desc: "+1 pierce and +18% projectile speed.",
      maxStacks: 1,
      apply: (ship) => {
        ship.pierce += 1;
        ship.bulletSpeed *= 1.18;
      }
    },
    {
      id: "arc-conduit",
      name: "Arc Conduit",
      tier: "rare",
      category: "Control",
      desc: "Shots arc to 1 target for 35% damage.",
      maxStacks: 1,
      apply: (ship) => {
        ship.arcDamage = Math.max(ship.arcDamage || 0, 0.35);
        ship.arcRadius = Math.max(ship.arcRadius || 0, 120);
        ship.arcChains = Math.max(ship.arcChains || 0, 1);
      }
    },
    {
      id: "cryo-core",
      name: "Cryo Core",
      tier: "rare",
      category: "Control",
      desc: "Shots have a 45% chance to slow for 1.8s.",
      maxStacks: 1,
      apply: (ship) => {
        ship.slowChance = Math.max(ship.slowChance, 0.45);
        ship.slowDuration = Math.max(ship.slowDuration, 1.8);
      }
    },
    {
      id: "overdrive-reactor",
      name: "Overdrive Reactor",
      tier: "rare",
      category: "Utility",
      desc: "+60 max energy and +30% regen.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxEnergy += 60;
        ship.energy = Math.min(ship.maxEnergy, ship.energy + 30);
        ship.energyRegen *= 1.3;
      }
    },
    {
      id: "hunter-gyros",
      name: "Hunter Gyros",
      tier: "rare",
      category: "Mobility",
      desc: "+20% max speed and +0.5 turn rate.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxSpeed *= 1.2;
        ship.turnRate += 0.5;
      }
    },
    {
      id: "nanobot-rig",
      name: "Nanobot Rig",
      tier: "epic",
      category: "Defense",
      desc: "Restore 5% hull and 10 energy on every kill.",
      maxStacks: 1,
      apply: (ship) => {
        ship.healOnKill = Math.max(ship.healOnKill, 0.05);
        ship.energyOnKill = Math.max(ship.energyOnKill, 10);
      }
    },
    {
      id: "phase-pierce",
      name: "Phase Pierce",
      tier: "epic",
      category: "Offense",
      desc: "+2 pierce and +10% projectile speed.",
      maxStacks: 1,
      apply: (ship) => {
        ship.pierce += 2;
        ship.bulletSpeed *= 1.1;
      }
    },
    {
      id: "shockwave-shells",
      name: "Shockwave Shells",
      tier: "epic",
      category: "Offense",
      desc: "+40% splash radius and +15% splash damage.",
      maxStacks: 1,
      apply: (ship) => {
        ship.splashRadius = ship.splashRadius > 0 ? ship.splashRadius * 1.4 : 24;
        ship.splashDamage = Math.max(0.7, ship.splashDamage || 0.6) * 1.15;
      }
    },
    {
      id: "vampiric-cells",
      name: "Vampiric Cells",
      tier: "epic",
      category: "Defense",
      desc: "Restore 8% hull and 14 energy on kill.",
      maxStacks: 1,
      apply: (ship) => {
        ship.healOnKill = Math.max(ship.healOnKill, 0.08);
        ship.energyOnKill = Math.max(ship.energyOnKill, 14);
      }
    },
    {
      id: "barrage-protocol",
      name: "Barrage Protocol",
      tier: "epic",
      category: "Offense",
      desc: "Every 5th shot fires +2 projectiles with +50% damage.",
      maxStacks: 1,
      apply: (ship) => {
        ship.barrageEvery = ship.barrageEvery ? Math.min(ship.barrageEvery, 5) : 5;
        ship.barrageProjectiles = Math.max(ship.barrageProjectiles || 0, 2);
        ship.barrageBonusDamage = Math.max(ship.barrageBonusDamage || 1, 1.5);
      }
    },
    {
      id: "phase-cyclone",
      name: "Phase Cyclone",
      tier: "epic",
      category: "Offense",
      desc: "+3 pierce and +20% damage.",
      maxStacks: 1,
      apply: (ship) => {
        ship.pierce += 3;
        ship.damage *= 1.2;
      }
    },
    {
      id: "fortress-plating",
      name: "Fortress Plating",
      tier: "epic",
      category: "Defense",
      desc: "+120 max hull, +20% shield regen, +15% damage reduction.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxHealth += 120;
        ship.shieldRegen *= 1.2;
        ship.damageReduction = Math.min(0.6, ship.damageReduction + 0.15);
      }
    },
    {
      id: "nova-reactor",
      name: "Nova Reactor",
      tier: "epic",
      category: "Utility",
      desc: "+80 max energy, +50% regen, +15% fire rate.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxEnergy += 80;
        ship.energy = Math.min(ship.maxEnergy, ship.energy + 40);
        ship.energyRegen *= 1.5;
        ship.fireRate *= 1.15;
      }
    },
    {
      id: "phantom-thrusters",
      name: "Phantom Thrusters",
      tier: "epic",
      category: "Mobility",
      desc: "+30% speed, +30% accel, +0.8 turn rate.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxSpeed *= 1.3;
        ship.accel *= 1.3;
        ship.turnRate += 0.8;
      }
    },
    {
      id: "entropy-shells",
      name: "Entropy Shells",
      tier: "epic",
      category: "Offense",
      desc: "Shots gain splash (radius 28) and +20% damage.",
      maxStacks: 1,
      apply: (ship) => {
        ship.splashRadius = ship.splashRadius > 0 ? ship.splashRadius * 1.3 : 28;
        ship.splashDamage = Math.max(ship.splashDamage || 0.6, 0.7);
        ship.damage *= 1.2;
      }
    },
    {
      id: "quantum-barrage",
      name: "Quantum Barrage",
      tier: "legendary",
      category: "Offense",
      desc: "Every 4th volley becomes a quantum burst with piercing splash.",
      maxStacks: 1,
      apply: (ship) => {
        ship.projectiles += 1;
        ship.spread += 0.06;
        ship.barrageEvery = ship.barrageEvery || 4;
        ship.barrageProjectiles = Math.max(ship.barrageProjectiles || 0, 2);
        ship.barragePierce = Math.max(ship.barragePierce || 0, 2);
        ship.barrageBonusDamage = Math.max(ship.barrageBonusDamage || 1, 1.35);
        ship.barrageSplashRadius = Math.max(ship.barrageSplashRadius || 0, 44);
        ship.barrageSplashDamage = Math.max(ship.barrageSplashDamage || 0, 0.8);
      }
    },
    {
      id: "aegis-matrix",
      name: "Aegis Matrix",
      tier: "legendary",
      category: "Defense",
      desc: "Shield break triggers a surge that restores shields and blasts nearby enemies.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxShield += 50;
        ship.shield = Math.min(ship.maxShield, ship.shield + 30);
        ship.damageReduction = Math.min(0.5, ship.damageReduction + 0.12);
        ship.aegisCooldown = ship.aegisCooldown || 18;
        ship.aegisShieldRestore = Math.max(ship.aegisShieldRestore || 0, 0.35);
        ship.aegisPulseRadius = Math.max(ship.aegisPulseRadius || 0, 170);
        ship.aegisPulseDamage = Math.max(ship.aegisPulseDamage || 0, 28);
        ship.aegisPulseSlow = Math.max(ship.aegisPulseSlow || 0, 1.6);
      }
    },
    {
      id: "starforged-core",
      name: "Starforged Core",
      tier: "legendary",
      category: "Offense",
      desc: "+35% damage, +25% fire rate, -20% energy cost.",
      maxStacks: 1,
      apply: (ship) => {
        ship.damage *= 1.35;
        ship.fireRate *= 1.25;
        ship.energyCost *= 0.8;
      }
    },
    {
      id: "tempest-drive",
      name: "Tempest Drive",
      tier: "legendary",
      category: "Mobility",
      desc: "+40% speed, +35% accel, +1.2 turn rate, -30% boost cost.",
      maxStacks: 1,
      apply: (ship) => {
        ship.maxSpeed *= 1.4;
        ship.accel *= 1.35;
        ship.turnRate += 1.2;
        ship.boostCost *= 0.7;
      }
    },
    {
      id: "event-horizon",
      name: "Event Horizon",
      tier: "legendary",
      category: "Control",
      desc: "Shots arc to 2 targets for 70% damage and slow.",
      maxStacks: 1,
      apply: (ship) => {
        ship.arcDamage = Math.max(ship.arcDamage || 0, 0.7);
        ship.arcRadius = Math.max(ship.arcRadius || 0, 180);
        ship.arcChains = Math.max(ship.arcChains || 0, 2);
        ship.slowChance = Math.max(ship.slowChance, 0.35);
        ship.slowDuration = Math.max(ship.slowDuration, 1.8);
      }
    },
    {
      id: "phoenix-swarm",
      name: "Phoenix Swarm",
      tier: "legendary",
      category: "Defense",
      desc: "Restore 12% hull and 22 energy on every kill.",
      maxStacks: 1,
      apply: (ship) => {
        ship.healOnKill = Math.max(ship.healOnKill, 0.12);
        ship.energyOnKill = Math.max(ship.energyOnKill, 22);
      }
    }
  ];

  const ENEMY_TYPES = [
    {
      id: "scout",
      name: "Scout",
      health: 30,
      shield: 0,
      speed: 240,
      accel: 440,
      turnRate: 4.8,
      fireRate: 0.8,
      bulletSpeed: 330,
      damage: 6,
      radius: 10,
      credits: 14,
      score: 60,
      color: "#4dd1c5",
      preferredRange: 180,
      weight: 1.3
    },
    {
      id: "fighter",
      name: "Fighter",
      health: 60,
      shield: 18,
      speed: 190,
      accel: 330,
      turnRate: 3.6,
      fireRate: 0.85,
      bulletSpeed: 340,
      damage: 9,
      radius: 13,
      credits: 22,
      score: 80,
      color: "#f6c65f",
      preferredRange: 220,
      weight: 1.1
    },
    {
      id: "interceptor",
      name: "Interceptor",
      health: 42,
      shield: 10,
      speed: 270,
      accel: 470,
      turnRate: 5.4,
      fireRate: 1.15,
      bulletSpeed: 360,
      damage: 8,
      radius: 12,
      credits: 24,
      score: 90,
      color: "#6ee7b7",
      preferredRange: 210,
      weight: 1.0
    },
    {
      id: "skirmisher",
      name: "Skirmisher",
      health: 34,
      shield: 0,
      speed: 300,
      accel: 520,
      turnRate: 4.6,
      fireRate: 1.1,
      bulletSpeed: 360,
      damage: 6,
      radius: 10,
      credits: 20,
      score: 85,
      color: "#8fd3ff",
      burstCount: 3,
      burstInterval: 0.12,
      burstCooldown: 1.4,
      preferredRange: 200,
      minWave: 2,
      weight: 1.0
    },
    {
      id: "disruptor",
      name: "Disruptor",
      health: 52,
      shield: 16,
      speed: 210,
      accel: 330,
      turnRate: 3.2,
      fireRate: 0.95,
      bulletSpeed: 320,
      damage: 9,
      radius: 12,
      credits: 26,
      score: 110,
      color: "#43e0c0",
      bulletSlowPlayer: true,
      bulletSlowDuration: 1.6,
      bulletTint: "#43e0c0",
      preferredRange: 230,
      minWave: 3,
      weight: 0.9
    },
    {
      id: "sniper",
      name: "Sniper",
      health: 38,
      shield: 10,
      speed: 170,
      accel: 240,
      turnRate: 2.4,
      fireRate: 0.55,
      bulletSpeed: 640,
      damage: 18,
      radius: 11,
      credits: 32,
      score: 130,
      color: "#b98cff",
      preferredRange: 360,
      minWave: 4,
      weight: 0.7
    },
    {
      id: "bomber",
      name: "Bomber",
      health: 95,
      shield: 28,
      speed: 150,
      accel: 230,
      turnRate: 2.1,
      fireRate: 0.6,
      bulletSpeed: 260,
      damage: 15,
      radius: 17,
      credits: 32,
      score: 140,
      color: "#f48b7f",
      pattern: "spread",
      spreadCount: 3,
      spreadAngle: 0.24,
      bulletRadius: 4,
      bulletLife: 2.1,
      preferredRange: 260,
      minWave: 3,
      weight: 0.8
    },
    {
      id: "rusher",
      name: "Rusher",
      health: 68,
      shield: 8,
      speed: 260,
      accel: 430,
      turnRate: 3.4,
      fireRate: 0.7,
      bulletSpeed: 330,
      damage: 8,
      radius: 14,
      credits: 28,
      score: 120,
      color: "#ff8a7a",
      ramDamage: 16,
      ramCooldown: 5,
      ramRange: 28,
      preferredRange: 150,
      minWave: 4,
      weight: 0.7
    },
    {
      id: "bulwark",
      name: "Bulwark",
      health: 170,
      shield: 70,
      speed: 120,
      accel: 170,
      turnRate: 1.6,
      fireRate: 0.45,
      bulletSpeed: 240,
      damage: 22,
      radius: 24,
      credits: 54,
      score: 210,
      color: "#f6c65f",
      pattern: "spread",
      spreadCount: 4,
      spreadAngle: 0.2,
      bulletRadius: 4,
      bulletLife: 2.2,
      preferredRange: 280,
      shieldPulseCooldown: 8,
      shieldPulseRadius: 220,
      shieldPulseAmount: 14,
      shieldPulseSelf: true,
      shieldPulseColor: "#7ca8ff",
      minWave: 5,
      weight: 0.5
    }
  ];

  const ACE_TYPE = {
    id: "ace",
    name: "Ace",
    health: 170,
    shield: 60,
    speed: 210,
    accel: 360,
    turnRate: 3.8,
    fireRate: 1.4,
    bulletSpeed: 420,
    damage: 16,
    radius: 19,
    credits: 90,
    score: 320,
    color: "#7ca8ff"
  };

  const BOSS_TYPE = {
    id: "dreadnought",
    name: "Dreadnought",
    health: 380,
    shield: 140,
    speed: 150,
    accel: 220,
    turnRate: 1.4,
    fireRate: 0.9,
    bulletSpeed: 320,
    damage: 20,
    radius: 26,
    credits: 220,
    score: 620,
    color: "#b98cff",
    pattern: "spread"
  };

  window.STELLAR_DOGFIGHT_DB = {
    TIER_META,
    TIER_ORDER,
    ABILITIES,
    SHIPS,
    WEAPONS,
    SECONDARIES,
    PART_RARITIES,
    PART_SLOTS,
    PART_TEMPLATES,
    FACTIONS,
    SECTOR_MODIFIERS,
    CONTRACT_DEFS,
    SALVAGE_CACHE,
    ELITE_MODS,
    HANGAR_UPGRADES,
    FRONTIER_STARTERS,
    FRONTIER_UPGRADES,
    FIELD_UPGRADES,
    ENEMY_TYPES,
    ACE_TYPE,
    BOSS_TYPE
  };
})();
