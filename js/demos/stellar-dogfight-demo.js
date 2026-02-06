    (() => {
      "use strict";

      const STORAGE_KEY = "stellarDogfightProgress";
      const UPGRADE_REROLL_COST = 45;
      const UPGRADE_REROLL_SCALE = 1.62;
      const WORLD_WIDTH = 3200;
      const WORLD_HEIGHT = 1800;
      const LEVEL_WAVES = 10;
      const SKILL_LIMIT = 3;
      const HUD_UPDATE_INTERVAL_MS = 50;
      const DEFERRED_UI_FLUSH_MS = 220;
      const DEFERRED_SAVE_FLUSH_MS = 180;
      const PREMIUM_CURRENCY_LABEL = "Astralite";
      const PREMIUM_DROP_RUN_CAP = 4;
      const DROP_PITY_THRESHOLDS = {
        key: 4,
        blueprint: 5
      };
      const BALANCE_TUNING = {
        playerEnergyCostScale: 0.9,
        helperRatePenaltyPerExtra: 0.18,
        helperDamagePenaltyPerExtra: 0.24,
        barrageBonusSoftCap: 2,
        barrageBonusSoftScale: 0.6,
        barrageDamageSoftScale: 0.72,
        arcChainFalloffPerJump: 0.22,
        waveDensitySoftScale: [
          { wave: 10, scale: 1 },
          { wave: 18, scale: 0.9 },
          { wave: Infinity, scale: 0.8 }
        ]
      };
      const PREMIUM_DROP_PITY_THRESHOLD = 4;
      const MILESTONE_WAVES = [3, 6, 9];
      const THREAT_TIERS = [
        { id: "normal", label: "Normal", minWave: 1, enemyScale: 1, fireRate: 1, eliteBonus: 0 },
        { id: "veteran", label: "Veteran", minWave: 5, enemyScale: 1.08, fireRate: 1.05, eliteBonus: 0.06 },
        { id: "elite", label: "Elite", minWave: 10, enemyScale: 1.16, fireRate: 1.11, eliteBonus: 0.12 },
        { id: "mythic", label: "Mythic", minWave: 16, enemyScale: 1.25, fireRate: 1.17, eliteBonus: 0.18 }
      ];
      const BUILD_PATHS = [
        {
          id: "balanced",
          name: "Balanced Command",
          desc: "No specialization penalties. Stable output in all situations.",
          apply: () => {}
        },
        {
          id: "crit",
          name: "Critical Overwatch",
          desc: "Keystone: high crit burst and precision pressure.",
          apply: (stats) => {
            stats.critChance = Math.min(0.95, stats.critChance + 0.14);
            stats.critMultiplier += 0.45;
            stats.damage *= 1.12;
          }
        },
        {
          id: "drone",
          name: "Drone Command",
          desc: "Keystone: autonomous support wing focus.",
          apply: (stats) => {
            stats.helperCount = Math.max(stats.helperCount || 0, 2);
            stats.helperDamageRatio = Math.max(stats.helperDamageRatio || 0, 0.52);
            stats.helperFireRate = Math.max(stats.helperFireRate || 0, 1.95);
            stats.helperRange = Math.max(stats.helperRange || 0, 480);
            stats.damage *= 0.95;
          }
        },
        {
          id: "singularity",
          name: "Singularity Doctrine",
          desc: "Keystone: black-hole and arc-control dominance.",
          apply: (stats) => {
            stats.blackHoleChance = Math.max(stats.blackHoleChance || 0, 0.22);
            stats.blackHoleRadius = Math.max(stats.blackHoleRadius || 0, 120);
            stats.blackHoleDuration = Math.max(stats.blackHoleDuration || 0, 1.7);
            stats.blackHoleForce = Math.max(stats.blackHoleForce || 0, 380);
            stats.blackHoleDamage = Math.max(stats.blackHoleDamage || 0, 16);
            stats.arcDamage = Math.max(stats.arcDamage || 0, 0.34);
            stats.arcRadius = Math.max(stats.arcRadius || 0, 135);
            stats.arcChains = Math.max(stats.arcChains || 0, 1);
            stats.fireRate *= 0.94;
          }
        },
        {
          id: "brawler",
          name: "Brawler Core",
          desc: "Keystone: close-range pressure and heavy survivability.",
          apply: (stats) => {
            stats.maxHealth += 80;
            stats.maxShield += 60;
            stats.damageReduction = Math.min(0.7, (stats.damageReduction || 0) + 0.12);
            stats.auraRadius = Math.max(stats.auraRadius || 0, 96);
            stats.auraDamage = Math.max(stats.auraDamage || 0, 18);
            stats.maxSpeed *= 0.96;
          }
        }
      ];
      const WEEKLY_MUTATORS = [
        {
          id: "solar-flare",
          label: "Solar Flare",
          desc: "Hazard zones intensify and appear more often.",
          applyPlayer: () => {},
          applyWave: (waveConfig) => { waveConfig.hazardBoost += 0.5; }
        },
        {
          id: "glass-cannon",
          label: "Glass Cannon",
          desc: "Both sides hit harder.",
          applyPlayer: (stats) => { stats.damage *= 1.18; stats.maxShield *= 0.9; },
          applyWave: (waveConfig) => { waveConfig.enemyDamage *= 1.15; }
        },
        {
          id: "shattered-shields",
          label: "Shattered Shields",
          desc: "Shield values are lower but hull pressure is higher.",
          applyPlayer: (stats) => { stats.maxShield *= 0.78; stats.maxHealth *= 1.14; },
          applyWave: (waveConfig) => { waveConfig.enemyShieldScale *= 0.75; waveConfig.enemyHealthScale *= 1.1; }
        },
        {
          id: "overclocked-sector",
          label: "Overclocked Sector",
          desc: "Projectile cadence is accelerated for all ships.",
          applyPlayer: (stats) => { stats.fireRate *= 1.1; },
          applyWave: (waveConfig) => { waveConfig.enemyFireRate *= 1.12; }
        }
      ];
      const STAT_GLOSSARY = [
        {
          term: "DPS",
          detail: "Estimated sustained damage per second, including crit and projectile count."
        },
        {
          term: "EHP",
          detail: "Effective hit points from hull + shield adjusted by damage reduction."
        },
        {
          term: "Energy Sustain",
          detail: "Energy regen minus weapon drain at current fire rate. Negative means eventual dry fire."
        },
        {
          term: "Threat Tier",
          detail: "Wave pressure class. Higher tiers increase enemy output and elite pressure."
        },
        {
          term: "Objective",
          detail: "Current wave requirement. Completing it ends the wave once hostiles are cleared."
        },
        {
          term: "Weekly Mutator",
          detail: "Rotating global modifier affecting player and/or enemy wave behavior."
        },
        {
          term: "Challenge Seed",
          detail: "Weekly challenge identifier for consistent comparisons and runs."
        }
      ];
      const TUTORIAL_STEPS = [
        {
          title: "Core Flight",
          text: "Use WASD to move, mouse to aim, and Space or mouse press to fire. Boost to dodge pressure and reposition."
        },
        {
          title: "Energy Discipline",
          text: "Weapons and boost spend energy. Keep Energy Sustain near neutral so you can keep firing under pressure."
        },
        {
          title: "Wave Objectives",
          text: "Each wave can require elimination, survival, or elite hunts. Watch the objective strip and finish it clean."
        },
        {
          title: "Build Identity",
          text: "Pick one Build Identity in the Hangar and stack upgrades around it for stronger synergies."
        },
        {
          title: "Premium + Milestones",
          text: "Elite and boss kills can drop Astralite. Milestone waves grant bonus loot and route choices."
        }
      ];
      const INVENTORY_LIMIT = 42;
      const ITEM_UPGRADE_SLOTS = {
        common: 2,
        uncommon: 3,
        rare: 4,
        epic: 5,
        legendary: 6
      };
      const ITEM_UPGRADE_CHANCE = {
        common: 0.8,
        uncommon: 0.65,
        rare: 0.5,
        epic: 0.36,
        legendary: 0.24
      };
      const FIELD_DROP_LIMIT = 10;
      const FIELD_DROP_INTERVAL = 12;
      const FIELD_DROP_LIFETIME = 26;
      const FIELD_DROP_MIN_DISTANCE = 180;
      const FIELD_DROP_TYPES = [
        {
          id: "nova",
          name: "Nova Strike",
          tier: "epic",
          weight: 0.5,
          symbol: "NOVA",
          damageScale: 4.2
        },
        {
          id: "hull",
          name: "Hull Patch",
          tier: "uncommon",
          weight: 1.6,
          symbol: "HULL",
          healRatio: 0.35
        },
        {
          id: "shield",
          name: "Shield Battery",
          tier: "uncommon",
          weight: 1.4,
          symbol: "SHLD",
          healRatio: 0.4
        },
        {
          id: "energy",
          name: "Energy Cell",
          tier: "common",
          weight: 1.6,
          symbol: "ENG",
          healRatio: 0.45
        },
        {
          id: "invuln",
          name: "Phase Ward",
          tier: "rare",
          weight: 0.9,
          symbol: "WARD",
          duration: 5.5
        },
        {
          id: "damage",
          name: "Overcharge",
          tier: "rare",
          weight: 0.95,
          symbol: "DMG",
          duration: 7,
          multiplier: 2
        },
        {
          id: "speed",
          name: "Afterburner",
          tier: "uncommon",
          weight: 1.1,
          symbol: "SPD",
          duration: 7,
          multiplier: 1.35
        }
      ];
      const WEAPON_TIER_SCALING = {
        damage: 0.12,
        fireRate: 0.08,
        bulletSpeed: 0.06,
        energyCost: 0.05,
        spread: 0.04
      };
      const WEAPON_AFFIX_POOL = [
        {
          id: "accelerated",
          label: "Accelerated",
          desc: "+18% bullet speed.",
          minTier: "common",
          apply: (stats) => { stats.bulletSpeed *= 1.18; }
        },
        {
          id: "charged",
          label: "Charged Coils",
          desc: "+15% damage.",
          minTier: "uncommon",
          apply: (stats) => { stats.damage *= 1.15; }
        },
        {
          id: "overclocked",
          label: "Overclocked",
          desc: "+15% fire rate.",
          minTier: "uncommon",
          apply: (stats) => { stats.fireRate *= 1.15; }
        },
        {
          id: "stabilized",
          label: "Stabilized",
          desc: "-20% spread.",
          minTier: "rare",
          apply: (stats) => { stats.spread = Math.max(0.02, stats.spread * 0.8); }
        },
        {
          id: "piercing",
          label: "Piercing",
          desc: "+1 pierce.",
          minTier: "rare",
          apply: (stats) => { stats.pierce = (stats.pierce || 0) + 1; }
        },
        {
          id: "cluster",
          label: "Cluster Rounds",
          desc: "+1 projectile, +spread.",
          minTier: "rare",
          apply: (stats) => {
            stats.projectiles = (stats.projectiles || 1) + 1;
            stats.spread = Math.min(0.36, stats.spread + 0.06);
          }
        },
        {
          id: "detonator",
          label: "Detonator",
          desc: "Adds splash damage.",
          minTier: "epic",
          apply: (stats) => {
            stats.splashRadius = Math.max(stats.splashRadius || 0, 24);
            stats.splashDamage = Math.max(stats.splashDamage || 0.6, 0.65);
          }
        },
        {
          id: "arc-conduit",
          label: "Arc Conduit",
          desc: "Bolts arc to nearby foes.",
          minTier: "epic",
          apply: (stats) => {
            stats.arcDamage = Math.max(stats.arcDamage || 0, 0.45);
            stats.arcRadius = Math.max(stats.arcRadius || 0, 120);
            stats.arcChains = Math.max(stats.arcChains || 0, 1);
          }
        },
        {
          id: "barrage",
          label: "Barrage Sync",
          desc: "Every 5th shot fires extra bolts.",
          minTier: "epic",
          apply: (stats) => {
            stats.barrageEvery = Math.min(stats.barrageEvery || 5, 5);
            stats.barrageProjectiles = (stats.barrageProjectiles || 0) + 2;
            stats.barrageBonusDamage = Math.max(stats.barrageBonusDamage || 1, 1.4);
          }
        },
        {
          id: "singularity",
          label: "Void Singularity",
          desc: "Each shot spawns a black hole.",
          minTier: "legendary",
          apply: (stats) => {
            stats.blackHoleChance = Math.max(stats.blackHoleChance || 0, 1);
            stats.blackHoleRadius = Math.max(stats.blackHoleRadius || 0, 120);
            stats.blackHoleDuration = Math.max(stats.blackHoleDuration || 0, 1.6);
            stats.blackHoleForce = Math.max(stats.blackHoleForce || 0, 380);
            stats.blackHoleDamage = Math.max(stats.blackHoleDamage || 0, 18);
          }
        },
        {
          id: "rift-echo",
          label: "Rift Echo",
          desc: "22% chance to echo the volley.",
          minTier: "legendary",
          apply: (stats) => {
            stats.echoChance = Math.max(stats.echoChance || 0, 0.22);
            stats.echoDamage = Math.max(stats.echoDamage || 0, 0.55);
          }
        }
      ];
      const ATTACHMENT_AFFIX_POOL = [
        {
          id: "reinforced",
          label: "Reinforced",
          desc: "+20 max shield.",
          minTier: "uncommon",
          apply: (stats) => { stats.maxShield = (stats.maxShield || 0) + 20; }
        },
        {
          id: "reactive",
          label: "Reactive",
          desc: "+10 max hull.",
          minTier: "uncommon",
          apply: (stats) => { stats.maxHealth = (stats.maxHealth || 0) + 10; }
        },
        {
          id: "tuned",
          label: "Tuned",
          desc: "+8 energy regen.",
          minTier: "rare",
          apply: (stats) => { stats.energyRegen = (stats.energyRegen || 0) + 8; }
        },
        {
          id: "precision",
          label: "Precision",
          desc: "+2% crit chance.",
          minTier: "rare",
          apply: (stats) => { stats.critChance = (stats.critChance || 0) + 0.02; }
        },
        {
          id: "overdrive",
          label: "Overdrive",
          desc: "+8% max speed.",
          minTier: "epic",
          apply: (stats) => { stats.maxSpeed = (stats.maxSpeed || 0) * 1.08; }
        },
        {
          id: "tactician",
          label: "Tactician",
          desc: "+0.2 upgrade luck.",
          minTier: "epic",
          apply: (stats) => { stats.upgradeLuck = (stats.upgradeLuck || 0) + 0.2; }
        }
      ];
      const dom = {
        canvas: document.querySelector("[data-role='battlefield']"),
        overlay: document.querySelector("[data-role='overlay']"),
        overlayContent: document.querySelector("[data-role='overlay-content']"),
        tips: document.querySelector("[data-role='tips']"),
        log: document.querySelector("[data-role='log']"),
        activeUpgrades: document.querySelector("[data-role='active-upgrades']"),
        hangar: document.querySelector("[data-role='hangar']"),
        shipyard: document.querySelector("[data-role='shipyard']"),
        premiumShop: document.querySelector("[data-role='premium-shop']"),
        inventory: document.querySelector("[data-role='inventory']"),
        equippedItems: document.querySelector("[data-role='equipped-items']"),
        secondaries: document.querySelector("[data-role='secondaries']"),
        salvage: document.querySelector("[data-role='salvage']"),
        contracts: document.querySelector("[data-role='contracts']"),
        factions: document.querySelector("[data-role='factions']"),
        history: document.querySelector("[data-role='history']"),
        keybinds: document.querySelector("[data-role='keybinds']"),
        tierPill: document.querySelector("[data-role='tier-pill']"),
        minimap: document.querySelector("[data-role='minimap']"),
        settingsOverlay: document.querySelector("[data-role='settings-overlay']"),
        settingsBody: document.querySelector("[data-role='settings-body']"),
        perfOverlay: document.querySelector("[data-role='perf-overlay']"),
        perfDetailOverlay: document.querySelector("[data-role='perf-detail-overlay']"),
        perfSettingsCard: document.querySelector("[data-role='perf-settings-card']"),
        perfSettingsBasic: document.querySelector("[data-role='perf-settings-basic']"),
        perfSettingsDetail: document.querySelector("[data-role='perf-settings-detail']"),
        perfLogBtn: document.querySelector("[data-role='perf-log-btn']"),
        perfLogStatus: document.querySelector("[data-role='perf-log-status']"),
        statusIcons: document.querySelector("[data-role='status-icons']"),
        runAnalytics: document.querySelector("[data-role='run-analytics']")
      };
      const settingsPanel = document.querySelector("[data-tab-panel='settings']");
      const settingsHome = settingsPanel ? settingsPanel.parentElement : null;
      const settingsAnchor = settingsPanel ? settingsPanel.nextElementSibling : null;
      let settingsOpen = false;
      let settingsReturnTab = "systems";
      let settingsResumeMode = null;

      const ctx = dom.canvas.getContext("2d", { alpha: false });
      const stats = {
        hullText: document.querySelector("[data-stat='hull-text']"),
        shieldText: document.querySelector("[data-stat='shield-text']"),
        energyText: document.querySelector("[data-stat='energy-text']"),
        wave: document.querySelector("[data-stat='wave']"),
        tier: document.querySelector("[data-stat='tier']"),
        enemyCount: document.querySelector("[data-stat='enemy-count']"),
        score: document.querySelector("[data-stat='score']"),
        credits: document.querySelector("[data-stat='credits']"),
        rank: document.querySelector("[data-stat='rank']"),
        xp: document.querySelector("[data-stat='xp']"),
        techPoints: document.querySelector("[data-stat='tech-points']"),
        bestWave: document.querySelector("[data-stat='best-wave']"),
        totalKills: document.querySelector("[data-stat='total-kills']"),
        shipName: document.querySelector("[data-stat='ship-name']"),
        weaponName: document.querySelector("[data-stat='weapon-name']"),
        secondaryName: document.querySelector("[data-stat='secondary-name']"),
        abilityStatus: document.querySelector("[data-stat='ability-status']"),
        secondaryStatus: document.querySelector("[data-stat='secondary-status']"),
        sectorMod: document.querySelector("[data-stat='sector-mod']"),
        status: document.querySelector("[data-stat='status']"),
        modeLabel: document.querySelector("[data-stat='mode-label']"),
        sector: document.querySelector("[data-stat='sector']"),
        blueprints: document.querySelector("[data-stat='blueprints']"),
        bankedCredits: document.querySelector("[data-stat='banked-credits']"),
        bankedCreditsTotal: document.querySelector("[data-stat='banked-credits-total']"),
        premiumCurrency: document.querySelector("[data-stat='premium-currency']"),
        premiumCurrencyTotal: document.querySelector("[data-stat='premium-currency-total']"),
        salvageKeys: document.querySelector("[data-stat='salvage-keys']"),
        contractStatus: document.querySelector("[data-stat='contract-status']"),
        controlMode: document.querySelector("[data-stat='control-mode']"),
        damage: document.querySelector("[data-stat='damage']"),
        fireRate: document.querySelector("[data-stat='fire-rate']"),
        speed: document.querySelector("[data-stat='speed']"),
        shieldRegen: document.querySelector("[data-stat='shield-regen']"),
        energyRegen: document.querySelector("[data-stat='energy-regen']"),
        crit: document.querySelector("[data-stat='crit']"),
        dps: document.querySelector("[data-stat='dps']"),
        ehp: document.querySelector("[data-stat='ehp']"),
        energySustain: document.querySelector("[data-stat='energy-sustain']"),
        threatTier: document.querySelector("[data-stat='threat-tier']"),
        threatTierHud: document.querySelector("[data-stat='threat-tier-hud']"),
        objective: document.querySelector("[data-stat='objective']"),
        objectiveHud: document.querySelector("[data-stat='objective-hud']"),
        weeklyMutator: document.querySelector("[data-stat='weekly-mutator']"),
        challengeSeed: document.querySelector("[data-stat='challenge-seed']")
      };
      const meters = {
        hull: document.querySelector("[data-meter='hull']"),
        shield: document.querySelector("[data-meter='shield']"),
        energy: document.querySelector("[data-meter='energy']")
      };
      const hudMeters = {
        hull: document.querySelector("[data-hud-meter='hull']"),
        shield: document.querySelector("[data-hud-meter='shield']"),
        energy: document.querySelector("[data-hud-meter='energy']")
      };
      const hudStats = {
        hull: document.querySelector("[data-hud-stat='hull']"),
        shield: document.querySelector("[data-hud-stat='shield']"),
        energy: document.querySelector("[data-hud-stat='energy']")
      };

      const input = {
        keys: new Set(),
        pointer: { x: 0, y: 0, screenX: 0, screenY: 0, active: false, moved: false },
        firing: false,
        padFiring: false,
        boost: false,
        aimAngle: 0,
        aimSource: "mouse",
        aimMode: "hybrid",
        capture: null,
        padMoveX: 0,
        padMoveY: 0,
        padAimX: 0,
        padAimY: 0
      };

      const KEYBIND_LABELS = {
        forward: "Move Forward",
        back: "Move Back",
        left: "Strafe Left",
        right: "Strafe Right",
        aimUp: "Aim Up",
        aimDown: "Aim Down",
        aimLeft: "Aim Left",
        aimRight: "Aim Right",
        fire: "Fire",
        boost: "Boost",
        ability: "Ability",
        secondary: "Secondary",
        dock: "Upgrade Dock",
        pause: "Pause",
        help: "Help Overlay"
      };

      const state = {
        mode: "hangar",
        overlayMode: null,
        gameMode: "arcade",
        level: 1,
        wave: 1,
        score: 0,
        credits: 0,
        kills: 0,
        lastTime: 0,
        runStart: 0,
        waveStart: 0,
        width: 0,
        height: 0,
        sector: 1,
        sectorMod: null,
        contracts: [],
        training: false,
        resumeMode: "flight",
        runActive: false,
        runBanked: false,
        runEndedByAbort: false,
        runLoadout: null,
        runHighlights: [],
        runPremiumDrops: 0,
        lastContractRender: 0,
        lastSidebarRender: 0,
        lastSaveFlushAt: 0,
        contractsDirty: false,
        sidebarDirty: false,
        savePending: false,
        inventorySelectionId: null,
        onboardingTimer: 0,
        onboardingSave: 0,
        difficulty: "normal",
        enemyAccuracyMod: 1,
        decoy: null,
        mines: [],
        blackHoles: [],
        skillSlots: [],
        lastHudUpdate: 0,
        activeUpgradeKey: "",
        lastHullHitAt: 0,
        upgradeStacks: {},
        upgradeOptions: [],
        upgradeRerolls: 0,
        worldWidth: WORLD_WIDTH,
        worldHeight: WORLD_HEIGHT,
        renderScale: 1,
        camera: { x: 0, y: 0 },
        fieldDropTimer: 0,
        frontier: null,
        lossRewards: null,
        levelRewards: null,
        salvageResults: null,
        hazards: [],
        threatTier: THREAT_TIERS[0].id,
        waveObjective: null,
        objectiveClearPending: false,
        choiceEvent: null,
        routeBonus: null,
        milestoneRewardsClaimed: {},
        tutorialMode: false,
        tutorialStep: 0,
        tutorialComplete: false,
        overlayReturnMode: null,
        weekly: null,
        challengeSeed: "",
        controllerConnected: false,
        controllerPrevButtons: [],
        telemetryRun: null
      };

      const BASE_PLAYER = {
        maxHealth: 130,
        maxShield: 96,
        shieldRegen: 6,
        maxEnergy: 130,
        energyRegen: 28,
        energyCost: 16,
        damage: 12,
        fireRate: 4,
        bulletSpeed: 520,
        maxSpeed: 240,
        accel: 620,
        damping: 2.4,
        critChance: 0.06,
        critMultiplier: 1.6,
        projectiles: 1,
        spread: 0.14,
        splashRadius: 0,
        splashDamage: 0.6,
        pierce: 0,
        turnRate: 4.2,
        boostMultiplier: 1.35,
        boostCost: 19,
        damageReduction: 0,
        slowChance: 0,
        slowDuration: 0,
        salvageBonus: 0,
        xpBonus: 0,
        healOnKill: 0,
        energyOnKill: 0,
        barrageEvery: 0,
        barrageProjectiles: 0,
        barragePierce: 0,
        barrageBonusDamage: 1,
        barrageSplashRadius: 0,
        barrageSplashDamage: 0,
        barrageCounter: 0,
        upgradeLuck: 0,
        damageBoostTimer: 0,
        damageBoostMultiplier: 1,
        speedBoostTimer: 0,
        speedBoostMultiplier: 1,
        auraRadius: 0,
        auraDamage: 0,
        auraInterval: 0.45,
        auraTimer: 0,
        mineDropChance: 0,
        mineInterval: 1.2,
        mineTimer: 0,
        mineRadius: 0,
        mineDamage: 0,
        mineDuration: 5,
        helperCount: 0,
        helperDamageRatio: 0,
        helperFireRate: 0,
        helperRange: 0,
        helperOrbitRadius: 26,
        helperOrbitSpeed: 1.4,
        shockwaveInterval: 0,
        shockwaveTimer: 0,
        shockwaveRadius: 0,
        shockwaveDamage: 0,
        shockwaveSlow: 0,
        missileInterval: 0,
        missileTimer: 0,
        missileDamage: 0,
        missileCount: 0,
        missileSpeed: 0,
        aegisCooldown: 0,
        aegisReadyAt: 0,
        aegisShieldRestore: 0,
        aegisPulseRadius: 0,
        aegisPulseDamage: 0,
        aegisPulseSlow: 0,
        arcDamage: 0,
        arcRadius: 0,
        arcChains: 0,
        arcRequiresSlow: false,
        blackHoleChance: 0,
        blackHoleRadius: 0,
        blackHoleDuration: 0,
        blackHoleForce: 0,
        blackHoleDamage: 0,
        echoChance: 0,
        echoDamage: 0,
        slowTimer: 0,
        slowFactor: 1
      };

      const DIFFICULTY_SETTINGS = {
        easy: { label: "Easy", enemyScale: 0.82, enemyDamage: 0.86, reward: 0.88 },
        normal: { label: "Normal", enemyScale: 1, enemyDamage: 1, reward: 1 },
        hard: { label: "Hard", enemyScale: 1.14, enemyDamage: 1.1, reward: 1.12 },
        adaptive: { label: "Adaptive", enemyScale: 1, enemyDamage: 1, reward: 1 }
      };

      const GAME_DB = window.STELLAR_DOGFIGHT_DB || {};
      const {
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
        PREMIUM_SHOP_ITEMS,
        FIELD_UPGRADES,
        ENEMY_TYPES,
        ACE_TYPE,
        BOSS_TYPE
      } = GAME_DB;

      let progress = loadProgress();
      ensurePremiumState();
      setPerfMode(normalizePerfMode(progress.settings));
      syncProgressSelections();
      state.weekly = getWeeklyChallengeInfo();
      state.challengeSeed = state.weekly.seed;
      const audioController = window.STELLAR_DOGFIGHT_AUDIO && window.STELLAR_DOGFIGHT_AUDIO.createAudioController
        ? window.STELLAR_DOGFIGHT_AUDIO.createAudioController({
          enabled: (progress.settings.audio || "on") !== "off"
        })
        : null;
      let player = null;
      let enemies = [];
      let bullets = [];
      let helpers = [];
      let particles = [];
      let pulses = [];
      let lootBursts = [];
      let damageNumbers = [];
      let stars = [];
      let obstacles = [];
      let fieldDrops = [];
      let backgroundGradient = null;
      let minimapCtx = null;
      const perfMetrics = {
        fps: 0,
        frameMs: 0,
        avgFrameMs: 0,
        updateMs: 0,
        avgUpdateMs: 0,
        renderMs: 0,
        avgRenderMs: 0,
        hudMs: 0,
        avgHudMs: 0,
        maxFrameMs: 0,
        rafIntervalMs: 0,
        avgRafIntervalMs: 0,
        maxRafIntervalMs: 0,
        idleMs: 0,
        avgIdleMs: 0,
        busyRatio: 0,
        avgBusyRatio: 0,
        deltaMs: 0,
        avgDeltaMs: 0,
        slowFrames: 0,
        hitchFrames: 0,
        slowRate: 0,
        hitchRate: 0,
        frames: 0,
        lastFpsAt: performance.now(),
        lastOverlayUpdate: 0,
        lastFrameEnd: 0
      };
      const longTaskMetrics = {
        count: 0,
        totalDuration: 0,
        supported: false
      };
      let longTaskObserver = null;
      const perfLog = {
        active: false,
        entries: [],
        startAt: 0,
        lastSampleAt: 0,
        samples: 0,
        lastLongTaskCount: 0,
        lastLongTaskDuration: 0,
        sampleIntervalMs: 250
      };

      function isAudioEnabled() {
        return (progress.settings.audio || "on") === "on";
      }

      function applyAudioFromProgress() {
        if (!audioController) return;
        audioController.setEnabled(isAudioEnabled());
        audioController.setMode(state.mode);
      }

      function resumeAudio() {
        if (audioController) {
          audioController.resume();
        }
      }

      function playAudioCue(eventName, payload) {
        if (audioController) {
          audioController.play(eventName, payload);
        }
      }

      window.STELLAR_DOGFIGHT_HELPERS = {
        logEvent,
        spawnPulse,
        spawnBlackHole,
        damageEnemy,
        wrapEntity,
        distanceBetween,
        rand,
        randInt,
        pick,
        pickWeighted,
        formatStat,
        getEnemies: () => enemies,
        generatePart: (rarity) => generatePart(rarity)
      };

      function hashString(inputValue) {
        const inputText = String(inputValue || "");
        let hash = 0;
        for (let i = 0; i < inputText.length; i += 1) {
          hash = (hash * 31 + inputText.charCodeAt(i)) >>> 0;
        }
        return hash >>> 0;
      }

      function getIsoWeekStamp(date) {
        const now = date ? new Date(date.getTime()) : new Date();
        const utc = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
        const dayNum = utc.getUTCDay() || 7;
        utc.setUTCDate(utc.getUTCDate() + 4 - dayNum);
        const yearStart = new Date(Date.UTC(utc.getUTCFullYear(), 0, 1));
        const weekNo = Math.ceil((((utc - yearStart) / 86400000) + 1) / 7);
        return `${utc.getUTCFullYear()}-W${String(weekNo).padStart(2, "0")}`;
      }

      function getWeeklyChallengeInfo() {
        const weekKey = getIsoWeekStamp(new Date());
        const mutators = Array.isArray(WEEKLY_MUTATORS) ? WEEKLY_MUTATORS : [];
        const mutator = mutators.length
          ? mutators[hashString(weekKey) % mutators.length]
          : { id: "none", label: "None", desc: "No mutator this week." };
        const seedHash = hashString(`${weekKey}:${mutator.id}`);
        const seed = seedHash.toString(16).toUpperCase().padStart(8, "0");
        return {
          weekKey,
          mutator,
          seed
        };
      }

      function getActiveBuildPath() {
        const pathId = progress.buildPath || "balanced";
        return BUILD_PATHS.find((item) => item.id === pathId) || BUILD_PATHS[0];
      }

      function resolveThreatTier(globalWave) {
        const value = Math.max(1, globalWave || 1);
        let tier = THREAT_TIERS[0];
        THREAT_TIERS.forEach((entry) => {
          if (value >= entry.minWave) tier = entry;
        });
        return tier;
      }

      function getThreatTierMeta(threatId) {
        return THREAT_TIERS.find((entry) => entry.id === threatId) || THREAT_TIERS[0];
      }

      function formatThreatTierLabel(threatId) {
        return getThreatTierMeta(threatId).label;
      }

      function formatRouteBonusLabel(routeId) {
        if (!routeId) return "";
        return routeId.split("-").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
      }

      function createWaveObjective(globalWave) {
        if (state.training) {
          return {
            id: "eliminate",
            label: "Eliminate hostiles",
            progressLabel: "Clear all enemies",
            complete: false
          };
        }
        const roll = Math.random();
        if (globalWave >= 6 && roll > 0.74) {
          const target = globalWave >= 14 ? 3 : 2;
          return {
            id: "elite-hunt",
            label: `Hunt priority targets (${target})`,
            progress: 0,
            target,
            progressLabel: `0 / ${target}`,
            complete: false
          };
        }
        if (globalWave >= 4 && roll > 0.48) {
          const timer = clamp(14 + Math.floor(globalWave * 0.45), 14, 26);
          return {
            id: "survive",
            label: `Survive ${timer}s`,
            timer,
            progressLabel: `${timer}s remaining`,
            complete: false
          };
        }
        return {
          id: "eliminate",
          label: "Eliminate hostiles",
          progressLabel: "Clear all enemies",
          complete: false
        };
      }

      function updateObjectiveProgressLabel() {
        if (!state.waveObjective) return;
        const objective = state.waveObjective;
        if (objective.id === "survive") {
          objective.progressLabel = `${Math.max(0, Math.ceil(objective.timer || 0))}s remaining`;
          return;
        }
        if (objective.id === "elite-hunt") {
          objective.progressLabel = `${objective.progress || 0} / ${objective.target || 0}`;
          return;
        }
        objective.progressLabel = "Clear all enemies";
      }

      function initRunTelemetry() {
        state.telemetryRun = {
          startedAt: Date.now(),
          mode: state.training ? "training" : (isFrontierMode() ? "frontier" : "arcade"),
          seed: state.challengeSeed || "",
          mutator: state.weekly?.mutator?.id || "none",
          ship: player?.ship?.id || progress.selectedShip,
          weapon: player?.weapon?.templateId || player?.weapon?.id || "unknown",
          shotsFired: 0,
          shotsHit: 0,
          damageDealt: 0,
          damageTaken: 0,
          kills: 0,
          eliteKills: 0,
          abilityUses: 0,
          secondaryUses: 0,
          objectiveCompletions: 0,
          hazardTicks: 0
        };
        pushTelemetryEvent("run_start", {
          mode: state.telemetryRun.mode,
          ship: state.telemetryRun.ship,
          weapon: state.telemetryRun.weapon,
          seed: state.telemetryRun.seed,
          mutator: state.telemetryRun.mutator
        });
      }

      function pushTelemetryEvent(eventName, payload = {}) {
        progress.telemetry = progress.telemetry || { recent: [] };
        const event = {
          ts: Date.now(),
          event: eventName,
          ...payload
        };
        progress.telemetry.recent = [event, ...(progress.telemetry.recent || [])].slice(0, 80);
        try {
          window.dispatchEvent(new CustomEvent("stellar:telemetry", { detail: event }));
        } catch (error) {
          // Ignore custom event failures.
        }
      }

      function finalizeRunTelemetry(summary) {
        if (!state.telemetryRun) return null;
        const telemetry = {
          ...state.telemetryRun,
          endedAt: Date.now(),
          wave: summary?.wave || state.wave,
          globalWave: summary?.globalWave || getGlobalWave(state.wave || 1),
          score: summary?.score || Math.round(state.score),
          credits: summary?.credits || Math.round(state.credits),
          durationSec: Math.round(summary?.durationSec || 0)
        };
        telemetry.accuracy = telemetry.shotsFired > 0
          ? telemetry.shotsHit / telemetry.shotsFired
          : 0;
        state.telemetryRun = null;
        pushTelemetryEvent("run_end", telemetry);
        return telemetry;
      }

      function getDerivedCombatMetrics(statsObj) {
        const statsValue = statsObj || player || BASE_PLAYER;
        const damage = Math.max(1, statsValue.damage || 1);
        const fireRate = Math.max(0.1, statsValue.fireRate || 0.1);
        const critChance = clamp(statsValue.critChance || 0, 0, 0.95);
        const critMult = Math.max(1, statsValue.critMultiplier || 1);
        const projectiles = Math.max(1, statsValue.projectiles || 1);
        const avgCritFactor = 1 + critChance * (critMult - 1);
        const dps = damage * fireRate * projectiles * avgCritFactor;
        const dr = clamp(statsValue.damageReduction || 0, 0, 0.85);
        const rawHp = Math.max(1, (statsValue.maxHealth || 0) + (statsValue.maxShield || 0));
        const ehp = rawHp / Math.max(0.15, 1 - dr);
        const sustain = (statsValue.energyRegen || 0) - ((statsValue.energyCost || 0) * fireRate);
        return { dps, ehp, sustain };
      }

      function describeDelta(before, after, key, digits = 1, asPercent = false) {
        const beforeValue = before[key] || 0;
        const afterValue = after[key] || 0;
        const rawDiff = afterValue - beforeValue;
        if (Math.abs(rawDiff) < 0.0001) return null;
        const factor = asPercent ? 100 : 1;
        const diff = rawDiff * factor;
        const value = Number.isInteger(diff) ? diff.toString() : diff.toFixed(digits);
        const sign = diff > 0 ? "+" : "";
        const suffix = asPercent ? "%" : "";
        return `${sign}${value}${suffix}`;
      }

      function getHangarPreviewText(upgrade, level) {
        if (!upgrade || level >= (upgrade.maxLevel || 1)) return "Maxed";
        const base = buildBaseStats();
        const preview = { ...base };
        if (typeof upgrade.apply === "function") {
          upgrade.apply(preview, level + 1);
        }
        const beforeMetrics = getDerivedCombatMetrics(base);
        const afterMetrics = getDerivedCombatMetrics(preview);
        const dpsDelta = describeDelta(beforeMetrics, afterMetrics, "dps", 0);
        const ehpDelta = describeDelta(beforeMetrics, afterMetrics, "ehp", 0);
        const sustainDelta = describeDelta(beforeMetrics, afterMetrics, "sustain", 1);
        return [
          dpsDelta ? `DPS ${dpsDelta}` : "",
          ehpDelta ? `EHP ${ehpDelta}` : "",
          sustainDelta ? `Sustain ${sustainDelta}` : ""
        ].filter(Boolean).join(" • ") || "No immediate delta";
      }

      function getPremiumPreviewText(item) {
        if (!item) return "No preview";
        const base = buildBaseStats();
        const preview = { ...base };
        const currentLevel = getPremiumItemLevel(item);
        if (typeof item.apply === "function") {
          item.apply(preview, item.kind === "one-time" ? 1 : currentLevel + 1);
        }
        const beforeMetrics = getDerivedCombatMetrics(base);
        const afterMetrics = getDerivedCombatMetrics(preview);
        const dpsDelta = describeDelta(beforeMetrics, afterMetrics, "dps", 0);
        const ehpDelta = describeDelta(beforeMetrics, afterMetrics, "ehp", 0);
        const sustainDelta = describeDelta(beforeMetrics, afterMetrics, "sustain", 1);
        return [
          dpsDelta ? `DPS ${dpsDelta}` : "",
          ehpDelta ? `EHP ${ehpDelta}` : "",
          sustainDelta ? `Sustain ${sustainDelta}` : ""
        ].filter(Boolean).join(" • ") || "No immediate delta";
      }

      function init() {
        setupTabs();
        setupArmoryNav();
        setupCanvas();
        setupLongTaskObserver();
        attachEvents();
        applySettingsFromProgress();
        checkProgressionUnlocks();
        setupWorld();
        player = createPlayer();
        renderHangar();
        renderShipyard();
        renderPremiumShop();
        renderArmory();
        renderContracts();
        renderSettings();
        renderHistory();
        renderRunAnalytics();
        setOverlay("start");
        updateLayout();
        logEvent("Hangar systems online. Launch when ready.");
        state.lastTime = performance.now();
        requestAnimationFrame(loop);
      }

      function setupCanvas() {
        resizeCanvas();
        resizeMinimap();
        window.addEventListener("resize", () => {
          resizeCanvas();
          resizeMinimap();
          if (!input.pointer.active) {
            input.pointer.screenX = state.width * 0.5;
            input.pointer.screenY = state.height * 0.5;
            updatePointerWorld();
          }
        });
      }

      function resizeCanvas() {
        const rect = dom.canvas.getBoundingClientRect();
        const dpr = getRenderScaleDpr();
        if (!rect.width || !rect.height) return;
        dom.canvas.width = Math.floor(rect.width * dpr);
        dom.canvas.height = Math.floor(rect.height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        state.width = rect.width;
        state.height = rect.height;
        backgroundGradient = ctx.createRadialGradient(
          state.width * 0.5,
          state.height * -0.2,
          120,
          state.width * 0.5,
          state.height * 0.4,
          state.width * 1.1
        );
        backgroundGradient.addColorStop(0, "rgba(68, 210, 194, 0.15)");
        backgroundGradient.addColorStop(0.45, "rgba(16, 28, 46, 0.9)");
        backgroundGradient.addColorStop(1, "rgba(5, 8, 14, 1)");
        updateStarfield();
      }

      function resizeMinimap() {
        if (!dom.minimap) return;
        const rect = dom.minimap.getBoundingClientRect();
        const dpr = getRenderScaleDpr();
        if (!rect.width || !rect.height) return;
        dom.minimap.width = Math.floor(rect.width * dpr);
        dom.minimap.height = Math.floor(rect.height * dpr);
        if (!minimapCtx) {
          minimapCtx = dom.minimap.getContext("2d");
        }
        minimapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      function setupLongTaskObserver() {
        if (!("PerformanceObserver" in window)) return;
        const supported = PerformanceObserver.supportedEntryTypes;
        if (Array.isArray(supported) && !supported.includes("longtask")) return;
        longTaskMetrics.supported = true;
        try {
          longTaskObserver = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
              longTaskMetrics.count += 1;
              longTaskMetrics.totalDuration += entry.duration || 0;
            });
          });
          longTaskObserver.observe({ entryTypes: ["longtask"] });
        } catch (error) {
          longTaskMetrics.supported = false;
          longTaskObserver = null;
        }
      }

      function attachEvents() {
        document.addEventListener("click", (event) => {
          resumeAudio();
          const action = event.target.closest("[data-action]");
          if (!action) return;
          const actionName = action.dataset.action;
          if (actionName === "launch") startMission();
          if (actionName === "training") startTraining();
          if (actionName === "tutorial") startTutorial();
          if (actionName === "replay-last-loadout") replayLastLoadout();
          if (actionName === "glossary") openGlossary();
          if (actionName === "dock") toggleFrontierDock();
          if (actionName === "pause") togglePause();
          if (actionName === "help") toggleHelpOverlay();
          if (actionName === "reset") resetMission();
          if (actionName === "settings") openSettings();
          if (actionName === "settings-close") closeSettings();
          if (actionName === "perf-log") togglePerfLog();
          if (actionName === "reset-progress") resetProgress();
          if (actionName === "preset-save") saveLoadoutPreset(action.dataset.slot || "");
          if (actionName === "preset-load") loadLoadoutPreset(action.dataset.slot || "");
        });

        if (dom.settingsOverlay) {
          dom.settingsOverlay.addEventListener("click", (event) => {
            if (event.target === dom.settingsOverlay) {
              closeSettings();
            }
          });
        }

        if (dom.tips) {
          dom.tips.addEventListener("click", (event) => {
            const card = event.target.closest("[data-tip-id]");
            if (card) {
              card.remove();
            }
          });
        }

        dom.overlay.addEventListener("click", (event) => {
          const overlayAction = event.target.closest("[data-overlay-action]");
          if (overlayAction) {
            const actionName = overlayAction.dataset.overlayAction;
            if (actionName === "launch") startMission();
            if (actionName === "resume") togglePause();
            if (actionName === "restart") {
              if (state.training) {
                startTraining();
              } else {
                startMission();
              }
            }
            if (actionName === "reset") resetMission();
            if (actionName === "reroll") rerollUpgrades();
            if (actionName === "skip") acceptUpgrade(null);
            if (actionName === "dock-close") closeFrontierDock();
            if (actionName === "dock") openFrontierDock();
            if (actionName === "tier-up") openFrontierTierSelect();
            if (actionName === "next-level") startMission();
            if (actionName === "choice-continue") {
              hideOverlay();
              setMode("flight");
            }
            if (actionName === "tutorial-prev") {
              state.tutorialStep = Math.max(0, (state.tutorialStep || 0) - 1);
              setOverlay("tutorial");
            }
            if (actionName === "tutorial-next") {
              state.tutorialStep = Math.min(TUTORIAL_STEPS.length - 1, (state.tutorialStep || 0) + 1);
              setOverlay("tutorial");
            }
            if (actionName === "tutorial-close") {
              state.tutorialMode = false;
              state.tutorialComplete = true;
              restoreOverlayReturn();
            }
            if (actionName === "tutorial-training") {
              state.tutorialMode = false;
              state.tutorialComplete = true;
              if (state.mode !== "hangar" && state.mode !== "gameover" && state.mode !== "victory") {
                resetMission();
              }
              startTraining();
            }
            if (actionName === "glossary-close") {
              restoreOverlayReturn();
            }
            if (actionName === "salvage-close") {
              state.salvageResults = null;
              if (state.mode === "hangar" && !state.runActive) {
                setOverlay("start");
              } else {
                hideOverlay();
              }
            }
            return;
          }

          const choiceButton = event.target.closest("[data-choice-id]");
          if (choiceButton) {
            applyChoiceEvent(choiceButton.dataset.choiceId);
            return;
          }

          const upgradeButton = event.target.closest("[data-upgrade-id]");
          if (upgradeButton) {
            acceptUpgrade(upgradeButton.dataset.upgradeId);
            return;
          }

          const frontierUpgrade = event.target.closest("[data-frontier-upgrade]");
          if (frontierUpgrade) {
            purchaseFrontierUpgrade(frontierUpgrade.dataset.frontierUpgrade);
            return;
          }

          const frontierShip = event.target.closest("[data-frontier-ship]");
          if (frontierShip) {
            selectFrontierShip(frontierShip.dataset.frontierShip);
          }
        });

        dom.hangar.addEventListener("click", (event) => {
          if (!isFeatureUnlocked("hangar")) {
            showTip("locked-hangar", "Hangar locked", getFeatureHint("hangar"), { kind: "lock" });
            return;
          }
          const buildPathButton = event.target.closest("[data-build-path]");
          if (buildPathButton) {
            const pathId = buildPathButton.dataset.buildPath || "";
            if (!BUILD_PATHS.some((item) => item.id === pathId)) return;
            if (progress.buildPath === pathId) return;
            progress.buildPath = pathId;
            applyLoadoutChange("Build identity updated. Applies next sortie.");
            showTip(null, "Build identity", `${getActiveBuildPath().name} selected.`, {
              kind: "info",
              repeatable: true,
              duration: 3600
            });
            renderHangar();
            return;
          }
          const button = event.target.closest("[data-hangar-id]");
          if (!button) return;
          const upgrade = HANGAR_UPGRADES.find((item) => item.id === button.dataset.hangarId);
          if (!upgrade) return;
          const currentLevel = progress.hangar[upgrade.id] || 0;
          const maxLevel = upgrade.maxLevel || 1;
          const cost = getHangarUpgradeCost(upgrade, currentLevel);
          if (progress.techPoints < cost || currentLevel >= maxLevel) return;
          progress.techPoints = Math.max(0, progress.techPoints - cost);
          progress.hangar[upgrade.id] = currentLevel + 1;
          saveProgress();
          renderHangar();
          if (state.mode === "flight" || state.mode === "upgrade" || state.mode === "paused") {
            logEvent("Hangar upgrade installed for next sortie.");
          } else {
            player = createPlayer();
          }
        });

        dom.shipyard.addEventListener("click", (event) => {
          if (!isFeatureUnlocked("shipyard")) {
            showTip("locked-shipyard", "Shipyard locked", getFeatureHint("shipyard"), { kind: "lock" });
            return;
          }
          const button = event.target.closest("[data-ship-id]");
          if (!button) return;
          const ship = SHIPS.find((item) => item.id === button.dataset.shipId);
          if (!ship) return;
          const unlocked = !!progress.shipUnlocks[ship.id];
          if (!unlocked) {
            if (!unlockItem(ship, progress.shipUnlocks)) return;
            logEvent(`Ship unlocked: ${ship.name}.`);
          }
          if (progress.selectedShip !== ship.id) {
            progress.selectedShip = ship.id;
            applyLoadoutChange("Ship selection updated. Applies next sortie.");
          } else {
            saveProgress();
            renderShipyard();
          }
        });

        if (dom.inventory) {
          dom.inventory.addEventListener("click", (event) => {
            if (!isFeatureUnlocked("armory")) {
              showTip("locked-armory", "Gear locked", getFeatureHint("armory"), { kind: "lock" });
              return;
            }
            const selection = event.target.closest("[data-inventory-select]");
            if (selection) {
              state.inventorySelectionId = selection.dataset.inventorySelect;
              renderInventory();
              return;
            }
            const action = event.target.closest("[data-item-action]");
            const card = event.target.closest("[data-item-id]");
            if (!card) return;
            const item = getItemById(card.dataset.itemId);
            if (!item) return;
            if (!action) return;
            const actionName = action.dataset.itemAction;
            if (actionName === "equip") {
              equipItem(item);
            } else if (actionName === "upgrade") {
              upgradeItem(item);
            } else if (actionName === "sell") {
              sellItem(item);
            }
          });
        }

        dom.secondaries.addEventListener("click", (event) => {
          if (!isFeatureUnlocked("armory")) {
            showTip("locked-armory-secondary", "Secondaries locked", getFeatureHint("armory"), { kind: "lock" });
            return;
          }
          const button = event.target.closest("[data-secondary-id]");
          if (!button) return;
          const secondary = SECONDARIES.find((item) => item.id === button.dataset.secondaryId);
          if (!secondary) return;
          const unlocked = !!progress.secondaryUnlocks[secondary.id];
          if (!unlocked) {
            if (!unlockItem(secondary, progress.secondaryUnlocks)) return;
            logEvent(`Secondary unlocked: ${secondary.name}.`);
          }
          if (progress.selectedSecondary !== secondary.id) {
            progress.selectedSecondary = secondary.id;
            applyLoadoutChange("Secondary selection updated. Applies next sortie.");
          } else {
            saveProgress();
            renderArmory();
          }
        });


        dom.salvage.addEventListener("click", (event) => {
          if (!isFeatureUnlocked("salvage")) {
            showTip("locked-salvage", "Salvage locked", getFeatureHint("salvage"), { kind: "lock" });
            return;
          }
          const button = event.target.closest("[data-salvage-action='open']");
          if (!button) return;
          const count = button.dataset.salvageCount || "1";
          openSalvageCaches(count);
        });

        if (dom.premiumShop) {
          dom.premiumShop.addEventListener("click", (event) => {
            const button = event.target.closest("[data-premium-buy]");
            if (!button) return;
            buyPremiumItem(button.dataset.premiumBuy);
          });
        }

        dom.keybinds.addEventListener("click", (event) => {
          const button = event.target.closest("[data-bind]");
          if (!button) return;
          input.capture = button.dataset.bind;
          renderKeybinds();
        });

        document.addEventListener("click", (event) => {
          const optionBtn = event.target.closest(".option-btn");
          if (!optionBtn) return;
          const group = optionBtn.closest("[data-setting]");
          if (!group) return;
          const setting = group.dataset.setting;
          const option = optionBtn.dataset.option;
          if (!option) return;
          if (setting === "difficulty") {
            progress.settings.difficulty = option;
          } else if (setting === "game-mode") {
            progress.settings.gameMode = option;
            if (state.runActive) {
              logEvent("Mode change applies on the next launch.");
            }
          } else if (setting === "shipyard-tier") {
            progress.settings.shipyardTier = option;
          } else if (setting === "inventory-filter") {
            progress.settings.inventoryFilter = option;
          } else if (setting === "input-mode") {
            progress.settings.inputMode = option;
          } else if (setting === "particles") {
            progress.settings.particles = option;
          } else if (setting === "hit-flash") {
            progress.settings.hitFlash = option === "on";
          } else if (setting === "hud-layout") {
            progress.settings.hudLayout = option;
          } else if (setting === "hud-scale") {
            progress.settings.hudScale = option;
          } else if (setting === "audio") {
            progress.settings.audio = option;
          } else if (setting === "palette") {
            progress.settings.palette = option;
          } else if (setting === "render-scale") {
            progress.settings.renderScale = option;
          } else if (setting === "perf-mode") {
            setPerfMode(option);
          }
          saveProgress();
          applySettingsFromProgress();
          renderSettings();
          renderShipyard();
          renderPremiumShop();
          renderArmory();
          logEvent(`Settings updated: ${setting.replace("-", " ")}.`);
        });

        dom.canvas.addEventListener("pointermove", (event) => {
          const rect = dom.canvas.getBoundingClientRect();
          input.pointer.screenX = event.clientX - rect.left;
          input.pointer.screenY = event.clientY - rect.top;
          input.pointer.active = true;
          input.pointer.moved = true;
          updatePointerWorld();
        });

        dom.canvas.addEventListener("pointerdown", () => {
          resumeAudio();
          input.firing = true;
        });

        window.addEventListener("pointerup", () => {
          input.firing = false;
        });

        window.addEventListener("keydown", (event) => {
          const key = normalizeKey(event);
          if (!key) return;
          if (key === "escape" && settingsOpen) {
            event.preventDefault();
            closeSettings();
            return;
          }
          if (input.capture) {
            event.preventDefault();
            if (key === "escape") {
              input.capture = null;
              renderKeybinds();
              return;
            }
            if (!isBindableKey(key)) {
              logEvent("Key not supported for binding.");
              return;
            }
            const conflicts = getKeybindConflicts(input.capture, key);
            if (conflicts.length) {
              const labels = conflicts.map((action) => KEYBIND_LABELS[action] || action).join(", ");
              showTip("keybind-conflict", "Key already used", `${formatKeybind(key)} is bound to ${labels}.`, {
                kind: "info",
                repeatable: true,
                duration: 4500
              });
              return;
            }
            progress.keybinds[input.capture] = key;
            input.capture = null;
            saveProgress();
            renderSettings();
            logEvent("Keybind updated.");
            return;
          }
          if (key === "escape") {
            if (state.mode === "dock" || state.mode === "tier-select") {
              event.preventDefault();
              closeFrontierDock();
              return;
            }
            if (state.mode === "flight" || state.mode === "training" || state.mode === "paused") {
              event.preventDefault();
              togglePause();
            }
            return;
          }
          if (key === " " || key.startsWith("arrow")) {
            event.preventDefault();
          }
          if (key === (progress.keybinds.pause || "p")) {
            if (state.mode === "dock" || state.mode === "tier-select") {
              closeFrontierDock();
            } else {
              togglePause();
            }
            return;
          }
          if (key === "r") {
            resetMission();
            return;
          }
          if (key === (progress.keybinds.help || "h")) {
            event.preventDefault();
            toggleHelpOverlay();
            return;
          }
          if (!event.repeat) {
            if (key === progress.keybinds.ability) {
              activateAbility();
            }
            if (key === progress.keybinds.secondary) {
              activateSecondary();
            }
            if (key === progress.keybinds.dock) {
              toggleFrontierDock();
            }
          }
          input.keys.add(key);
        });

        window.addEventListener("keyup", (event) => {
          const key = normalizeKey(event);
          if (!key) return;
          input.keys.delete(key);
        });

        window.addEventListener("blur", () => {
          if (state.mode === "flight") {
            setPaused(true);
          }
        });

        document.addEventListener("visibilitychange", () => {
          if (document.hidden && state.mode === "flight") {
            setPaused(true);
          }
        });
      }

      function setupTabs() {
        const groups = document.querySelectorAll("[data-tab-group]");
        groups.forEach((group) => {
          const buttons = Array.from(group.querySelectorAll("[data-tab-target]"));
          const panels = Array.from(group.querySelectorAll("[data-tab-panel]"));
          if (!buttons.length || !panels.length) return;
          const defaultTarget = group.dataset.tabDefault || buttons[0].dataset.tabTarget;
          const missionGrid = document.querySelector(".mission-grid");
          const missionShell = document.querySelector(".mission-shell");

          const activate = (target, focusButton) => {
            buttons.forEach((button) => {
              const isActive = button.dataset.tabTarget === target;
              button.classList.toggle("is-active", isActive);
              button.setAttribute("aria-selected", isActive ? "true" : "false");
              button.tabIndex = isActive ? 0 : -1;
              if (isActive && focusButton) {
                button.focus();
              }
            });

            panels.forEach((panel) => {
              const isActive = panel.dataset.tabPanel === target;
              panel.classList.toggle("is-active", isActive);
              panel.hidden = !isActive;
            });

            if (group.dataset.tabGroup === "sidebar") {
              const isWide = target !== "systems" && target !== "upgrades";
              if (missionGrid) {
                missionGrid.classList.toggle("is-wide", isWide);
              }
              if (missionShell) {
                missionShell.classList.toggle("is-wide", isWide);
              }
            }
          };

          activate(defaultTarget, false);

          group.addEventListener("click", (event) => {
            const button = event.target.closest("[data-tab-target]");
            if (!button || !group.contains(button)) return;
            activate(button.dataset.tabTarget, true);
          });

          group.addEventListener("keydown", (event) => {
            const currentIndex = buttons.findIndex((button) => button.classList.contains("is-active"));
            if (currentIndex === -1) return;
            let nextIndex = currentIndex;
            if (event.key === "ArrowRight" || event.key === "ArrowDown") {
              nextIndex = (currentIndex + 1) % buttons.length;
            } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
              nextIndex = (currentIndex - 1 + buttons.length) % buttons.length;
            } else if (event.key === "Home") {
              nextIndex = 0;
            } else if (event.key === "End") {
              nextIndex = buttons.length - 1;
            } else {
              return;
            }
            event.preventDefault();
            activate(buttons[nextIndex].dataset.tabTarget, true);
          });
        });
      }

      function getActiveTabTarget() {
        const activeButton = document.querySelector("[data-tab-group='sidebar'] [data-tab-target].is-active");
        return activeButton ? activeButton.dataset.tabTarget : "systems";
      }

      function selectTab(target) {
        const button = document.querySelector(`[data-tab-group='sidebar'] [data-tab-target="${target}"]`);
        if (button) {
          button.click();
        }
      }

      function isPlayingMode(mode) {
        return ["flight", "training", "paused", "upgrade", "dock", "tier-select", "gameover", "victory"].includes(mode);
      }

      function updateLayout() {
        const playing = isPlayingMode(state.mode);
        document.body.classList.toggle("is-playing", playing);
        document.body.classList.toggle("is-hangar", !playing);
        resizeCanvas();
        resizeMinimap();
      }

      function openSettings() {
        if (settingsOpen) return;
        settingsOpen = true;
        settingsReturnTab = getActiveTabTarget();
        settingsResumeMode = state.mode;
        if (state.mode === "flight" || state.mode === "training") {
          setPaused(true);
        }
        if (settingsPanel && dom.settingsBody) {
          selectTab("settings");
          settingsPanel.hidden = false;
          settingsPanel.classList.add("is-active");
          dom.settingsBody.appendChild(settingsPanel);
        }
        if (dom.settingsOverlay) {
          dom.settingsOverlay.hidden = false;
        }
        updateLayout();
        renderSettings();
      }

      function closeSettings(shouldResume = true) {
        if (!settingsOpen) return;
        settingsOpen = false;
        if (settingsPanel && settingsHome) {
          if (settingsAnchor && settingsAnchor.parentElement === settingsHome) {
            settingsHome.insertBefore(settingsPanel, settingsAnchor);
          } else {
            settingsHome.appendChild(settingsPanel);
          }
        }
        if (settingsReturnTab) {
          selectTab(settingsReturnTab);
        }
        if (dom.settingsOverlay) {
          dom.settingsOverlay.hidden = true;
        }
        if (shouldResume && (settingsResumeMode === "flight" || settingsResumeMode === "training")) {
          setPaused(false);
        }
        updateLayout();
      }

      function setupArmoryNav() {
        const nav = document.querySelector("[data-armory-nav]");
        if (!nav) return;
        const buttons = Array.from(nav.querySelectorAll("[data-armory-target]"));
        const sections = Array.from(document.querySelectorAll("[data-armory-section]"));
        if (!buttons.length || !sections.length) return;

        const activate = (target, shouldSave) => {
          let nextTarget = target;
          if (!buttons.some((button) => button.dataset.armoryTarget === nextTarget)) {
            nextTarget = buttons[0].dataset.armoryTarget;
          }
          buttons.forEach((button) => {
            button.classList.toggle("is-active", button.dataset.armoryTarget === nextTarget);
          });
          sections.forEach((section) => {
            const isActive = section.dataset.armorySection === nextTarget;
            section.hidden = !isActive;
          });
          if (shouldSave) {
            progress.settings.armorySection = nextTarget;
            saveProgress();
          }
        };

        const storedTarget = progress.settings.armorySection || buttons[0].dataset.armoryTarget;
        activate(storedTarget, false);

        nav.addEventListener("click", (event) => {
          const button = event.target.closest("[data-armory-target]");
          if (!button) return;
          activate(button.dataset.armoryTarget, true);
        });
      }

      function loadProgress() {
        const fallback = {
          rank: 1,
          xp: 0,
          techPoints: 0,
          bestWave: 1,
          bestLevel: 1,
          campaignLevel: 1,
          totalKills: 0,
          bankedCredits: 0,
          blueprints: 0,
          salvageKeys: 0,
          salvagePity: 0,
          salvageHistory: [],
          dropPity: {
            key: 0,
            blueprint: 0
          },
          records: {
            bestScore: 0,
            bestKills: 0,
            bestSurvivalSec: 0
          },
          runAnalytics: [],
          telemetry: {
            recent: []
          },
          premiumCurrency: 0,
          premiumDropPity: 0,
          premiumShop: {
            oneTime: {},
            levels: {}
          },
          buildPath: "balanced",
          buildHistory: [],
          lastLoadout: null,
          featureUnlocks: {
            upgrades: false,
            ability: false,
            secondary: false,
            hangar: false,
            armory: false,
            shipyard: false,
            contracts: false,
            salvage: false
          },
          onboarding: {
            flightSeconds: 0
          },
          tipsSeen: {},
          selectedShip: "vanguard",
          selectedWeapon: "basic",
          selectedSecondary: "emp",
          shipUnlocks: { vanguard: true, scout: true },
          weaponUnlocks: { basic: true },
          secondaryUnlocks: { emp: true },
          weaponLevels: {},
          inventory: [],
          equipped: {
            weaponId: null,
            attachments: {
              barrel: null,
              core: null,
              targeting: null,
              thruster: null
            }
          },
          partsInventory: [],
          equippedParts: {
            barrel: null,
            core: null,
            targeting: null,
            thruster: null
          },
          factions: {
            nova: 0,
            aegis: 0,
            vortex: 0
          },
          settings: {
            difficulty: "normal",
            gameMode: "arcade",
            inputMode: "hybrid",
            particles: "medium",
            hitFlash: true,
            hudLayout: "standard",
            hudScale: "md",
            audio: "on",
            palette: "default",
            armorySection: "inventory",
            partsMode: "equip",
            shipyardTier: "all",
            inventoryFilter: "all",
            renderScale: "auto",
            perfMode: "off",
            perfOverlay: false,
            perfDetail: false
          },
          keybinds: {
            forward: "w",
            back: "s",
            left: "a",
            right: "d",
            aimUp: "arrowup",
            aimDown: "arrowdown",
            aimLeft: "arrowleft",
            aimRight: "arrowright",
            fire: " ",
            boost: "shift",
            ability: "e",
            secondary: "q",
            dock: "u",
            pause: "p",
            help: "h"
          },
          loadoutPresets: {
            a: null,
            b: null,
            c: null
          },
          runHistory: [],
          hangar: {
            hull: 0,
            shield: 0,
            "shield-regenerator": 0,
            "damage-dampers": 0,
            "reactive-repair": 0,
            "aegis-relay": 0,
            "weapon-calibration": 0,
            "fire-control": 0,
            targeting: 0,
            "munitions-loader": 0,
            "amplifier-core": 0,
            "barrage-sync": 0,
            thrusters: 0,
            "attitude-control": 0,
            "inertial-dampers": 0,
            "boost-couplers": 0,
            reactor: 0,
            "capacitor-banks": 0,
            "efficiency-tuning": 0,
            "salvage-magnet": 0,
            "tactical-scanner": 0,
            "upgrade-forecast": 0,
            "energy-siphon": 0
          }
        };
        try {
          const stored = JSON.parse(localStorage.getItem(STORAGE_KEY));
          if (!stored) return fallback;
          const merged = {
            ...fallback,
            ...stored,
            hangar: {
              ...fallback.hangar,
              ...(stored.hangar || {})
            },
            shipUnlocks: {
              ...fallback.shipUnlocks,
              ...(stored.shipUnlocks || {})
            },
            weaponUnlocks: {
              ...fallback.weaponUnlocks,
              ...(stored.weaponUnlocks || {})
            },
            secondaryUnlocks: {
              ...fallback.secondaryUnlocks,
              ...(stored.secondaryUnlocks || {})
            },
            weaponLevels: {
              ...fallback.weaponLevels,
              ...(stored.weaponLevels || {})
            },
            inventory: Array.isArray(stored.inventory) ? stored.inventory : fallback.inventory,
            equipped: {
              ...fallback.equipped,
              ...(stored.equipped || {}),
              attachments: {
                ...fallback.equipped.attachments,
                ...((stored.equipped && stored.equipped.attachments) || {})
              }
            },
            equippedParts: {
              ...fallback.equippedParts,
              ...(stored.equippedParts || {})
            },
            factions: {
              ...fallback.factions,
              ...(stored.factions || {})
            },
            settings: {
              ...fallback.settings,
              ...(stored.settings || {})
            },
            dropPity: {
              ...fallback.dropPity,
              ...(stored.dropPity || {})
            },
            records: {
              ...fallback.records,
              ...(stored.records || {})
            },
            premiumShop: {
              ...fallback.premiumShop,
              ...(stored.premiumShop || {}),
              oneTime: {
                ...fallback.premiumShop.oneTime,
                ...((stored.premiumShop && stored.premiumShop.oneTime) || {})
              },
              levels: {
                ...fallback.premiumShop.levels,
                ...((stored.premiumShop && stored.premiumShop.levels) || {})
              }
            },
            keybinds: {
              ...fallback.keybinds,
              ...(stored.keybinds || {})
            },
            loadoutPresets: {
              ...fallback.loadoutPresets,
              ...(stored.loadoutPresets || {})
            },
            runHistory: Array.isArray(stored.runHistory) ? stored.runHistory.slice(0, 8) : fallback.runHistory,
            runAnalytics: Array.isArray(stored.runAnalytics) ? stored.runAnalytics.slice(0, 10) : fallback.runAnalytics,
            salvageHistory: Array.isArray(stored.salvageHistory) ? stored.salvageHistory.slice(0, 6) : fallback.salvageHistory,
            buildHistory: Array.isArray(stored.buildHistory) ? stored.buildHistory.slice(0, 10) : fallback.buildHistory,
            telemetry: {
              recent: Array.isArray(stored.telemetry?.recent) ? stored.telemetry.recent.slice(0, 80) : []
            },
            featureUnlocks: {
              ...fallback.featureUnlocks,
              ...(stored.featureUnlocks || {})
            },
            onboarding: {
              ...fallback.onboarding,
              ...(stored.onboarding || {})
            },
            tipsSeen: {
              ...fallback.tipsSeen,
              ...(stored.tipsSeen || {})
            },
            lastLoadout: sanitizeLoadoutPreset(stored.lastLoadout || null)
          };
          merged.bestWave = Number.isFinite(merged.bestWave) ? Math.max(1, merged.bestWave) : 1;
          const inferredCampaign = Math.max(1, Math.floor((merged.bestWave || 1) / LEVEL_WAVES) + 1);
          const storedCampaign = Number.isFinite(stored.campaignLevel) ? stored.campaignLevel : null;
          merged.campaignLevel = Number.isFinite(storedCampaign)
            ? Math.max(1, storedCampaign, inferredCampaign)
            : inferredCampaign;
          const storedBestLevel = Number.isFinite(stored.bestLevel) ? stored.bestLevel : null;
          merged.bestLevel = Number.isFinite(storedBestLevel)
            ? Math.max(1, storedBestLevel, merged.campaignLevel - 1)
            : Math.max(1, merged.campaignLevel - 1);
          merged.dropPity = {
            key: Number.isFinite(merged.dropPity?.key) ? Math.max(0, merged.dropPity.key) : 0,
            blueprint: Number.isFinite(merged.dropPity?.blueprint) ? Math.max(0, merged.dropPity.blueprint) : 0
          };
          merged.records = {
            bestScore: Number.isFinite(merged.records?.bestScore) ? Math.max(0, merged.records.bestScore) : 0,
            bestKills: Number.isFinite(merged.records?.bestKills) ? Math.max(0, merged.records.bestKills) : 0,
            bestSurvivalSec: Number.isFinite(merged.records?.bestSurvivalSec)
              ? Math.max(0, merged.records.bestSurvivalSec)
              : 0
          };
          merged.premiumCurrency = Number.isFinite(merged.premiumCurrency)
            ? Math.max(0, Math.floor(merged.premiumCurrency))
            : 0;
          merged.premiumDropPity = Number.isFinite(merged.premiumDropPity)
            ? Math.max(0, Math.floor(merged.premiumDropPity))
            : 0;
          merged.buildPath = BUILD_PATHS.some((item) => item.id === merged.buildPath) ? merged.buildPath : "balanced";
          merged.loadoutPresets = {
            a: sanitizeLoadoutPreset(merged.loadoutPresets?.a),
            b: sanitizeLoadoutPreset(merged.loadoutPresets?.b),
            c: sanitizeLoadoutPreset(merged.loadoutPresets?.c)
          };
          const legacyAim = {
            aimUp: "i",
            aimDown: "k",
            aimLeft: "j",
            aimRight: "l"
          };
          const hasLegacyAim = stored.keybinds && Object.keys(legacyAim)
            .every((key) => stored.keybinds[key] === legacyAim[key]);
          if (hasLegacyAim) {
            merged.keybinds = {
              ...merged.keybinds,
              aimUp: "arrowup",
              aimDown: "arrowdown",
              aimLeft: "arrowleft",
              aimRight: "arrowright"
            };
          }
          return merged;
        } catch (error) {
          return fallback;
        }
      }

      function sanitizeLoadoutPreset(preset) {
        if (!preset || typeof preset !== "object") return null;
        const attachments = {
          barrel: null,
          core: null,
          targeting: null,
          thruster: null,
          ...((preset.equippedAttachments && typeof preset.equippedAttachments === "object")
            ? preset.equippedAttachments
            : {})
        };
        return {
          shipId: typeof preset.shipId === "string" ? preset.shipId : null,
          secondaryId: typeof preset.secondaryId === "string" ? preset.secondaryId : null,
          weaponId: typeof preset.weaponId === "string" ? preset.weaponId : null,
          equippedAttachments: attachments,
          label: typeof preset.label === "string" ? preset.label : ""
        };
      }

      function ensurePremiumState() {
        progress.premiumCurrency = Number.isFinite(progress.premiumCurrency)
          ? Math.max(0, Math.floor(progress.premiumCurrency))
          : 0;
        progress.premiumShop = progress.premiumShop || {};
        progress.premiumShop.oneTime = progress.premiumShop.oneTime || {};
        progress.premiumShop.levels = progress.premiumShop.levels || {};
        const shopItems = Array.isArray(PREMIUM_SHOP_ITEMS) ? PREMIUM_SHOP_ITEMS : [];
        shopItems.forEach((item) => {
          if (!item || !item.id) return;
          if (item.kind === "one-time") {
            progress.premiumShop.oneTime[item.id] = !!progress.premiumShop.oneTime[item.id];
            return;
          }
          const maxLevel = Number.isFinite(item.maxLevel) ? Math.max(1, item.maxLevel) : 1;
          const raw = progress.premiumShop.levels[item.id];
          const level = Number.isFinite(raw) ? Math.floor(raw) : 0;
          progress.premiumShop.levels[item.id] = clamp(level, 0, maxLevel);
        });
      }

      function saveProgress() {
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
        } catch (error) {
          // Storage can be unavailable in some browser modes.
        }
      }

      function queueProgressSave() {
        state.savePending = true;
      }

      function queueSidebarRefresh() {
        state.sidebarDirty = true;
      }

      function queueContractsRefresh() {
        state.contractsDirty = true;
      }

      function flushDeferredState(force = false) {
        const now = performance.now();
        if (state.savePending && (force || now - state.lastSaveFlushAt >= DEFERRED_SAVE_FLUSH_MS)) {
          saveProgress();
          state.savePending = false;
          state.lastSaveFlushAt = now;
        }
        if (state.sidebarDirty && (force || now - state.lastSidebarRender >= DEFERRED_UI_FLUSH_MS)) {
          renderShipyard();
          renderPremiumShop();
          renderArmory();
          state.sidebarDirty = false;
          state.lastSidebarRender = now;
        }
        if (state.contractsDirty && (force || now - state.lastContractRender >= DEFERRED_UI_FLUSH_MS)) {
          renderContracts();
          state.contractsDirty = false;
          state.lastContractRender = now;
        }
      }

      function resetProgress() {
        const confirmed = window.confirm("Reset all progress? This cannot be undone.");
        if (!confirmed) return;
        try {
          localStorage.removeItem(STORAGE_KEY);
        } catch (error) {
          // Ignore storage failures.
        }
        window.location.reload();
      }

      function syncProgressSelections() {
        const shipIds = SHIPS.map((ship) => ship.id);
        if (!shipIds.includes(progress.selectedShip)) {
          progress.selectedShip = SHIPS[0].id;
        }
        if (!progress.shipUnlocks[progress.selectedShip]) {
          const unlockedShip = SHIPS.find((ship) => progress.shipUnlocks[ship.id]);
          progress.selectedShip = unlockedShip ? unlockedShip.id : SHIPS[0].id;
          progress.shipUnlocks[progress.selectedShip] = true;
        }

        const secondaryIds = SECONDARIES.map((secondary) => secondary.id);
        if (!secondaryIds.includes(progress.selectedSecondary)) {
          progress.selectedSecondary = SECONDARIES[0].id;
        }
        if (!progress.secondaryUnlocks[progress.selectedSecondary]) {
          const unlockedSecondary = SECONDARIES.find((secondary) => progress.secondaryUnlocks[secondary.id]);
          progress.selectedSecondary = unlockedSecondary ? unlockedSecondary.id : SECONDARIES[0].id;
          progress.secondaryUnlocks[progress.selectedSecondary] = true;
        }
        ensurePremiumState();
        ensureInventory();
        saveProgress();
      }

      function ensureInventory() {
        progress.inventory = Array.isArray(progress.inventory) ? progress.inventory : [];
        progress.equipped = progress.equipped || {
          weaponId: null,
          attachments: {
            barrel: null,
            core: null,
            targeting: null,
            thruster: null
          }
        };
        progress.equipped.attachments = {
          barrel: null,
          core: null,
          targeting: null,
          thruster: null,
          ...(progress.equipped.attachments || {})
        };

        const existingIds = new Set(progress.inventory.map((item) => item.id));
        if (Array.isArray(progress.partsInventory) && progress.partsInventory.length) {
          progress.partsInventory.forEach((part) => {
            if (!part || existingIds.has(part.id)) return;
            const attachment = convertPartToAttachment(part);
            if (!attachment) return;
            progress.inventory.push(attachment);
            existingIds.add(attachment.id);
          });
        }

        const legacyWeaponId = progress.selectedWeapon || "basic";
        const hasWeaponItem = progress.inventory.some((item) => item.type === "weapon");
        if (!hasWeaponItem) {
          const starter = createWeaponItem({ templateId: legacyWeaponId, tier: "common", isStarter: true });
          progress.inventory.push(starter);
          progress.equipped.weaponId = starter.id;
        }
        if (!progress.equipped.weaponId || !getItemById(progress.equipped.weaponId)) {
          const fallbackWeapon = progress.inventory.find((item) => item.type === "weapon");
          progress.equipped.weaponId = fallbackWeapon ? fallbackWeapon.id : null;
        }

        PART_SLOTS.forEach((slot) => {
          const legacyEquipped = progress.equippedParts ? progress.equippedParts[slot] : null;
          const preferredId = progress.equipped.attachments[slot] || legacyEquipped;
          const attachment = preferredId ? getItemById(preferredId) : null;
          progress.equipped.attachments[slot] = attachment ? attachment.id : null;
        });

        if (progress.inventory.length > INVENTORY_LIMIT) {
          progress.inventory = progress.inventory.slice(0, INVENTORY_LIMIT);
        }

        progress.inventory.forEach((item) => {
          if (!item) return;
          if (!item.type) {
            if (item.templateId) {
              item.type = "weapon";
            } else if (item.slot) {
              item.type = "attachment";
            }
          }
          if (!item.tier) {
            item.tier = getPartTier(item);
          }
          if (!item.upgradeSlots) {
            item.upgradeSlots = ITEM_UPGRADE_SLOTS[item.tier] || 2;
          }
          if (!Number.isFinite(item.upgradeSuccesses)) {
            item.upgradeSuccesses = 0;
          }
        });
      }

      function normalizePerfMode(settings) {
        if (!settings) return "off";
        if (settings.perfMode === "basic" || settings.perfMode === "detail") {
          return settings.perfMode;
        }
        if (settings.perfMode === "off") {
          if (settings.perfDetail) return "detail";
          if (settings.perfOverlay) return "basic";
          return "off";
        }
        if (settings.perfDetail) return "detail";
        if (settings.perfOverlay) return "basic";
        return "off";
      }

      function setPerfMode(mode) {
        const normalized = mode === "basic" || mode === "detail" || mode === "off" ? mode : "off";
        progress.settings.perfMode = normalized;
        progress.settings.perfOverlay = normalized === "basic";
        progress.settings.perfDetail = normalized === "detail";
      }

      function getPerfMode() {
        return normalizePerfMode(progress.settings);
      }

      function getRenderScaleCap() {
        const value = progress.settings.renderScale || "auto";
        if (value === "auto") return null;
        const cap = parseFloat(value);
        if (!Number.isFinite(cap) || cap <= 0) return null;
        return cap;
      }

      function getRenderScaleDpr() {
        const baseDpr = Math.max(1, window.devicePixelRatio || 1);
        const cap = getRenderScaleCap();
        const dpr = cap ? Math.min(baseDpr, cap) : baseDpr;
        state.renderScale = dpr;
        return dpr;
      }

      function applySettingsFromProgress() {
        state.difficulty = progress.settings.difficulty || "normal";
        if (!state.runActive || state.mode === "hangar" || state.mode === "gameover" || state.mode === "victory") {
          state.gameMode = progress.settings.gameMode || "arcade";
        }
        input.aimMode = progress.settings.inputMode || "hybrid";
        if (input.aimMode === "keyboard" || input.aimMode === "controller") {
          input.aimSource = "keyboard";
        } else {
          input.aimSource = "mouse";
        }
        syncHudPresentation();
        updateParticleSettings();
        resizeCanvas();
        resizeMinimap();
        applyAudioFromProgress();
        updatePerformanceOverlayVisibility();
      }

      function syncHudPresentation() {
        const layout = progress.settings.hudLayout || "standard";
        const scale = progress.settings.hudScale || "md";
        const palette = progress.settings.palette || "default";
        document.body.classList.toggle("is-hud-compact", layout === "compact");
        document.body.classList.toggle("is-hud-scale-sm", scale === "sm");
        document.body.classList.toggle("is-hud-scale-lg", scale === "lg");
        document.body.classList.toggle("is-colorblind", palette === "colorblind");
      }

      function isFeatureUnlocked(feature) {
        return !!(progress.featureUnlocks && progress.featureUnlocks[feature]);
      }

      function getFeatureHint(feature) {
        const hints = {
          upgrades: "Win a wave to unlock.",
          ability: "Play a little longer to unlock.",
          secondary: "Unlocks after abilities.",
          hangar: "Reach Rank 2 to open.",
          armory: "Reach Rank 3 to open.",
          shipyard: "Reach Rank 4 or find a blueprint.",
          contracts: "Reach Wave 3 to open.",
          salvage: "Find a salvage key to open."
        };
        return hints[feature] || "Unlocks later.";
      }

      function getFeatureTip(feature) {
        const abilityKey = formatKeybind(progress.keybinds.ability);
        const secondaryKey = formatKeybind(progress.keybinds.secondary);
        const tips = {
          upgrades: {
            title: "Boosts unlocked!",
            message: "After each wave, pick one boost."
          },
          ability: {
            title: "Ability ready!",
            message: `Press ${abilityKey} to use it.`
          },
          secondary: {
            title: "Secondary ready!",
            message: `Press ${secondaryKey} to deploy it.`
          },
          hangar: {
            title: "Hangar open!",
            message: "Spend tech points in Boosts."
          },
          armory: {
            title: "Gear locker open!",
            message: "Swap weapons + parts in Gear."
          },
          shipyard: {
            title: "Shipyard open!",
            message: "Unlock new ships in Ships."
          },
          contracts: {
            title: "Tasks unlocked!",
            message: "Grab a task for bonus loot."
          },
          salvage: {
            title: "Caches unlocked!",
            message: "Open salvage in Gear."
          }
        };
        return tips[feature] || null;
      }

      function getTipIcon(kind) {
        const icons = {
          unlock: "🎉",
          reward: "🎁",
          lock: "🔒",
          info: "💡"
        };
        return icons[kind] || icons.info;
      }

      function showTip(id, title, message, options = {}) {
        if (!dom.tips || !title || !message) return;
        progress.tipsSeen = progress.tipsSeen || {};
        const repeatable = options.repeatable === true;
        const kind = options.kind || "info";
        const icon = options.icon || getTipIcon(kind);
        const duration = Number.isFinite(options.duration) ? options.duration : 9000;
        if (id && !repeatable) {
          if (progress.tipsSeen[id]) return;
          progress.tipsSeen[id] = true;
          saveProgress();
        }
        const tipId = id || `notice-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        const card = document.createElement("div");
        card.className = "tip-card";
        card.dataset.tipId = tipId;
        card.dataset.kind = kind;
        card.innerHTML = `
          <span class="tip-icon" aria-hidden="true">${icon}</span>
          <div class="tip-body">
            <strong>${title}</strong>
            <span>${message}</span>
          </div>
        `;
        dom.tips.prepend(card);
        while (dom.tips.children.length > 3) {
          dom.tips.removeChild(dom.tips.lastChild);
        }
        setTimeout(() => {
          if (card.parentElement) {
            card.remove();
          }
        }, duration);
      }

      function unlockFeature(feature) {
        progress.featureUnlocks = progress.featureUnlocks || {};
        if (isFeatureUnlocked(feature)) return false;
        progress.featureUnlocks[feature] = true;
        const tip = getFeatureTip(feature);
        if (tip) {
          showTip(`unlock-${feature}`, tip.title, tip.message, { kind: "unlock" });
          logEvent(tip.title);
        }
        saveProgress();
        renderHangar();
        renderShipyard();
        renderPremiumShop();
        renderArmory();
        renderContracts();
        renderSettings();
        return true;
      }

      function checkProgressionUnlocks() {
        const flightSeconds = progress.onboarding?.flightSeconds || 0;
        const activeWave = isCampaignMode()
          ? getGlobalWave(state.wave || 1)
          : (state.wave || 1);
        const waveProgress = Math.max(progress.bestWave || 1, activeWave);

        if (!isFeatureUnlocked("upgrades") && (flightSeconds >= 45 || waveProgress >= 2)) {
          unlockFeature("upgrades");
        }
        if (isFeatureUnlocked("upgrades") && !isFeatureUnlocked("ability")
          && (flightSeconds >= 90 || waveProgress >= 3 || progress.rank >= 2)) {
          unlockFeature("ability");
        }
        if (isFeatureUnlocked("ability") && !isFeatureUnlocked("secondary")
          && (flightSeconds >= 140 || waveProgress >= 4 || progress.rank >= 3)) {
          unlockFeature("secondary");
        }
        if (!isFeatureUnlocked("hangar") && progress.rank >= 2) {
          unlockFeature("hangar");
        }
        if (isFeatureUnlocked("hangar") && !isFeatureUnlocked("armory") && progress.rank >= 3) {
          unlockFeature("armory");
        }
        if (!isFeatureUnlocked("shipyard") && (progress.rank >= 4 || progress.blueprints >= 1)) {
          unlockFeature("shipyard");
        }
        if (!isFeatureUnlocked("contracts") && (progress.bestWave >= 3 || progress.runHistory.length > 0)) {
          unlockFeature("contracts");
        }
        if (isFeatureUnlocked("armory") && !isFeatureUnlocked("salvage")
          && ((progress.salvageKeys || 0) >= 1 || progress.totalKills >= 12)) {
          unlockFeature("salvage");
        }
      }

      function updateParticleSettings() {
        if (!state.width || !state.height) return;
        updateStarfield();
      }

      function getParticleScale() {
        const setting = progress.settings.particles || "medium";
        if (setting === "low") return 0.6;
        if (setting === "high") return 1.4;
        return 1;
      }

      function getParticleCount(base) {
        return Math.max(1, Math.round(base * getParticleScale()));
      }

      function getShipById(id) {
        return SHIPS.find((ship) => ship.id === id) || SHIPS[0];
      }

      function getWeaponById(id) {
        return WEAPONS.find((weapon) => weapon.id === id) || WEAPONS[0];
      }

      function getSecondaryById(id) {
        return SECONDARIES.find((secondary) => secondary.id === id) || SECONDARIES[0];
      }

      function getItemById(id) {
        if (!id || !Array.isArray(progress.inventory)) return null;
        return progress.inventory.find((item) => item.id === id) || null;
      }

      function getEquippedWeapon() {
        const weaponId = progress.equipped?.weaponId;
        return weaponId ? getItemById(weaponId) : null;
      }

      function getEquippedAttachment(slot) {
        if (!slot) return null;
        const attachmentId = progress.equipped?.attachments?.[slot];
        return attachmentId ? getItemById(attachmentId) : null;
      }

      function getEquippedAttachments() {
        return PART_SLOTS.map((slot) => getEquippedAttachment(slot)).filter(Boolean);
      }

      function getFieldDropDef(id) {
        return FIELD_DROP_TYPES.find((entry) => entry.id === id) || FIELD_DROP_TYPES[0];
      }

      function rollFieldDropDef() {
        return pickWeighted(FIELD_DROP_TYPES, FIELD_DROP_TYPES.map((entry) => entry.weight));
      }

      function getTierIndex(tier) {
        const index = TIER_ORDER.indexOf(tier);
        return index >= 0 ? index : 0;
      }

      function rollItemTier(preferred, weights) {
        if (preferred) return preferred;
        const tiers = TIER_ORDER.slice();
        const baseWeights = tiers.map((tier) => (TIER_META[tier] ? TIER_META[tier].weight : 1));
        const finalWeights = Array.isArray(weights) && weights.length === tiers.length ? weights : baseWeights;
        return pickWeighted(tiers, finalWeights);
      }

      function getAffixCount(tier, type) {
        const base = {
          common: 1,
          uncommon: 1,
          rare: 2,
          epic: 2,
          legendary: 3
        };
        const baseCount = base[tier] || 1;
        const allowBonus = tier === "epic" || tier === "legendary";
        const bonus = allowBonus && Math.random() < 0.45 ? 1 : 0;
        const cap = type === "attachment" ? 2 : 3;
        return Math.min(baseCount + bonus, cap);
      }

      function pickAffixes(pool, tier, count) {
        if (!count) return [];
        const tierIndex = getTierIndex(tier);
        const eligible = pool.filter((affix) => getTierIndex(affix.minTier || "common") <= tierIndex);
        if (!eligible.length) return [];
        const picks = shuffle([...eligible]).slice(0, count);
        if (tierIndex >= getTierIndex("epic")) {
          const tierAffixes = eligible.filter((affix) => getTierIndex(affix.minTier || "common") >= tierIndex);
          if (tierAffixes.length && !picks.some((affix) => tierAffixes.includes(affix))) {
            picks[0] = pick(tierAffixes);
          }
        }
        return picks;
      }

      function applyWeaponTierScaling(stats, tier) {
        const tierIndex = getTierIndex(tier);
        if (!tierIndex) return;
        stats.damage *= 1 + WEAPON_TIER_SCALING.damage * tierIndex;
        stats.fireRate *= 1 + WEAPON_TIER_SCALING.fireRate * tierIndex;
        stats.bulletSpeed *= 1 + WEAPON_TIER_SCALING.bulletSpeed * tierIndex;
        if (Number.isFinite(stats.energyCost)) {
          stats.energyCost = Math.max(4, stats.energyCost * (1 - WEAPON_TIER_SCALING.energyCost * tierIndex));
        }
        if (Number.isFinite(stats.spread)) {
          stats.spread = Math.max(0.02, stats.spread * (1 - WEAPON_TIER_SCALING.spread * tierIndex));
        }
      }

      function applyWeaponUpgradeScaling(stats, item) {
        const level = item.upgradeSuccesses || 0;
        if (!level) return;
        stats.damage *= 1 + level * 0.08;
        stats.fireRate *= 1 + level * 0.06;
        stats.bulletSpeed *= 1 + level * 0.05;
        if (Number.isFinite(stats.energyCost)) {
          stats.energyCost = Math.max(3, stats.energyCost * (1 - level * 0.05));
        }
        if (Number.isFinite(stats.spread)) {
          stats.spread = Math.max(0.02, stats.spread * (1 - level * 0.04));
        }
        if (Number.isFinite(stats.critChance)) {
          stats.critChance += level * 0.01;
        }
      }

      function applyAttachmentUpgradeScaling(stats, item) {
        const level = item.upgradeSuccesses || 0;
        if (!level) return;
        const scale = 1 + level * 0.12;
        Object.keys(stats).forEach((key) => {
          if (!Number.isFinite(stats[key])) return;
          stats[key] *= scale;
        });
      }

      function getWeaponItemStats(item) {
        if (!item || !item.stats) return null;
        const stats = { ...item.stats };
        applyWeaponUpgradeScaling(stats, item);
        return stats;
      }

      function getAttachmentStats(item) {
        if (!item || !item.stats) return null;
        const stats = { ...item.stats };
        applyAttachmentUpgradeScaling(stats, item);
        return stats;
      }

      function buildStatSummary(stats) {
        if (!stats) return "";
        return Object.keys(stats)
          .filter((key) => Number.isFinite(stats[key]) && stats[key] !== 0 && !BARRAGE_STAT_KEYS.has(key))
          .map((key) => formatStat(key, stats[key]))
          .join(" · ");
      }

      function createWeaponItem({ templateId, tier, isStarter } = {}) {
        const pool = isStarter ? WEAPONS : WEAPONS.filter((weapon) => weapon.id !== "basic");
        const template = templateId ? getWeaponById(templateId) : pick(pool.length ? pool : WEAPONS);
        const finalTier = rollItemTier(tier);
        const stats = { ...template.stats };
        applyWeaponTierScaling(stats, finalTier);
        const affixCount = getAffixCount(finalTier, "weapon");
        const affixes = pickAffixes(WEAPON_AFFIX_POOL, finalTier, affixCount);
        affixes.forEach((affix) => affix.apply(stats));
        return {
          id: `weapon_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          type: "weapon",
          templateId: template.id,
          tier: finalTier,
          name: template.name,
          stats,
          affixes: affixes.map((affix) => ({ id: affix.id, label: affix.label, desc: affix.desc })),
          upgradeSlots: ITEM_UPGRADE_SLOTS[finalTier] || 2,
          upgradeSuccesses: 0,
          createdAt: Date.now(),
          isStarter: !!isStarter
        };
      }

      function createAttachmentItem({ tier, slot } = {}) {
        const finalTier = rollItemTier(tier);
        const templateSlot = slot || pick(PART_SLOTS);
        const template = PART_TEMPLATES[templateSlot];
        if (!template) return null;
        const rarity = PART_RARITIES[finalTier] || PART_RARITIES.common;
        const stats = {};
        Object.keys(template.stats).forEach((key) => {
          const range = template.stats[key];
          const value = rand(range[0], range[1]) * rarity.mult;
          const rounded = Math.abs(value) < 1 ? parseFloat(value.toFixed(2)) : Math.round(value);
          stats[key] = rounded;
        });
        const affixCount = getAffixCount(finalTier, "attachment");
        const affixes = pickAffixes(ATTACHMENT_AFFIX_POOL, finalTier, affixCount);
        affixes.forEach((affix) => affix.apply(stats));
        const tierLabel = formatTierLabel(finalTier);
        return {
          id: `att_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          type: "attachment",
          slot: templateSlot,
          tier: finalTier,
          name: `${tierLabel} ${template.name}`,
          stats,
          affixes: affixes.map((affix) => ({ id: affix.id, label: affix.label, desc: affix.desc })),
          upgradeSlots: ITEM_UPGRADE_SLOTS[finalTier] || 2,
          upgradeSuccesses: 0,
          createdAt: Date.now()
        };
      }

      function convertPartToAttachment(part) {
        if (!part || !part.stats || !part.slot) return null;
        const tier = getPartTier(part);
        return {
          id: part.id || `att_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          type: "attachment",
          slot: part.slot,
          tier,
          name: part.name || part.slot,
          stats: { ...part.stats },
          affixes: Array.isArray(part.affixes) ? part.affixes : [],
          upgradeSlots: ITEM_UPGRADE_SLOTS[tier] || 2,
          upgradeSuccesses: part.upgradeSuccesses || 0,
          createdAt: part.createdAt || Date.now()
        };
      }

      function addInventoryItem(item, options = {}) {
        if (!item) return false;
        progress.inventory = Array.isArray(progress.inventory) ? progress.inventory : [];
        if (progress.inventory.length >= INVENTORY_LIMIT) {
          const value = getItemSellValue(item);
          progress.bankedCredits += value;
          if (options.notify) {
            logEvent(`Inventory full. Auto-sold ${item.name} for ${value} credits.`);
          }
          return false;
        }
        progress.inventory.unshift(item);
        return true;
      }

      function applyMods(stats, mods) {
        if (!mods) return;
        const add = mods.add || {};
        const mult = mods.mult || {};
        Object.keys(add).forEach((key) => {
          stats[key] = (stats[key] || 0) + add[key];
        });
        Object.keys(mult).forEach((key) => {
          stats[key] = (stats[key] || 0) * mult[key];
        });
      }

      function applyAttachmentStats(stats, item) {
        const attachmentStats = getAttachmentStats(item);
        if (!attachmentStats) return;
        Object.keys(attachmentStats).forEach((key) => {
          stats[key] = (stats[key] || 0) + attachmentStats[key];
        });
      }

      function applyFrontierUpgrades(stats, upgrades) {
        if (!upgrades) return;
        FRONTIER_UPGRADES.forEach((upgrade) => {
          const level = upgrades[upgrade.id] || 0;
          for (let i = 0; i < level; i += 1) {
            if (upgrade.apply) upgrade.apply(stats, i + 1);
          }
        });
      }

      function applyPremiumBonuses(stats) {
        const shopItems = Array.isArray(PREMIUM_SHOP_ITEMS) ? PREMIUM_SHOP_ITEMS : [];
        if (!shopItems.length) return;
        ensurePremiumState();
        shopItems.forEach((item) => {
          if (!item || typeof item.apply !== "function") return;
          if (item.kind === "one-time") {
            if (progress.premiumShop.oneTime[item.id]) {
              item.apply(stats, 1);
            }
            return;
          }
          const level = progress.premiumShop.levels[item.id] || 0;
          if (level > 0) {
            item.apply(stats, level);
          }
        });
      }

      function applyBuildPathBonuses(stats) {
        const path = getActiveBuildPath();
        if (path && typeof path.apply === "function") {
          path.apply(stats);
        }
      }

      function applyAttachmentSetBonuses(stats) {
        const attachments = getEquippedAttachments();
        if (!attachments.length) return;
        const tierCounts = attachments.reduce((acc, item) => {
          const tier = item?.tier || "common";
          acc[tier] = (acc[tier] || 0) + 1;
          return acc;
        }, {});
        const rarePlus = (tierCounts.rare || 0) + (tierCounts.epic || 0) + (tierCounts.legendary || 0);
        const epicPlus = (tierCounts.epic || 0) + (tierCounts.legendary || 0);
        if (rarePlus >= 2) {
          stats.damage *= 1.08;
          stats.energyRegen += 6;
        }
        if (epicPlus >= 3) {
          stats.projectiles += 1;
          stats.critChance = Math.min(0.95, stats.critChance + 0.05);
        }
        if ((tierCounts.legendary || 0) >= 2) {
          stats.blackHoleChance = Math.max(stats.blackHoleChance || 0, 0.16);
          stats.damageReduction = Math.min(0.72, (stats.damageReduction || 0) + 0.08);
        }
      }

      function applyConditionalSynergies(stats) {
        if ((stats.slowChance || 0) >= 0.3 && (stats.arcDamage || 0) > 0) {
          stats.arcDamage *= 1.15;
          stats.arcChains = Math.max(stats.arcChains || 0, 2);
        }
        if ((stats.helperCount || 0) > 0 && (stats.auraRadius || 0) > 0) {
          stats.helperDamageRatio = Math.min(0.95, (stats.helperDamageRatio || 0) + 0.12);
        }
        if ((stats.blackHoleChance || 0) > 0 && (stats.echoChance || 0) > 0) {
          stats.echoDamage = Math.max(stats.echoDamage || 0, 0.42);
        }
      }

      function applyWeeklyMutatorToPlayer(stats) {
        const mutator = state.weekly && state.weekly.mutator;
        if (mutator && typeof mutator.applyPlayer === "function") {
          mutator.applyPlayer(stats);
        }
      }

      function applyGlobalSoftCaps(stats) {
        const softMult = (value, cap, softness = 0.45) => {
          if (!Number.isFinite(value)) return value;
          if (value <= cap) return value;
          return cap + (value - cap) * softness;
        };
        stats.fireRate = softMult(stats.fireRate, 12.5, 0.4);
        stats.damage = softMult(stats.damage, 380, 0.48);
        stats.maxSpeed = softMult(stats.maxSpeed, 540, 0.52);
        stats.energyRegen = softMult(stats.energyRegen, 160, 0.46);
        stats.damageReduction = clamp(stats.damageReduction || 0, 0, 0.78);
        stats.critChance = clamp(stats.critChance || 0, 0, 0.85);
        stats.barrageBonusDamage = softMult(stats.barrageBonusDamage || 1, 3, 0.52);
      }

      function buildBaseStats(loadout = {}) {
        const stats = { ...BASE_PLAYER };
        const ship = getShipById(loadout.shipId || progress.selectedShip);
        const weaponItem = loadout.weaponItem || getEquippedWeapon();
        const secondary = getSecondaryById(loadout.secondaryId || progress.selectedSecondary);
        applyMods(stats, ship.mods);
        if (weaponItem) {
          const weaponStats = getWeaponItemStats(weaponItem);
          if (weaponStats) {
            Object.assign(stats, weaponStats);
          }
        }
        HANGAR_UPGRADES.forEach((upgrade) => {
          const level = progress.hangar[upgrade.id] || 0;
          upgrade.apply(stats, level);
        });
        applyPremiumBonuses(stats);
        applyBuildPathBonuses(stats);
        getEquippedAttachments().forEach((attachment) => applyAttachmentStats(stats, attachment));
        applyAttachmentSetBonuses(stats);
        applyConditionalSynergies(stats);
        applyWeeklyMutatorToPlayer(stats);
        applyGlobalSoftCaps(stats);
        stats.ship = ship;
        stats.weapon = weaponItem || null;
        stats.secondary = secondary;
        return stats;
      }

      function createPlayer(loadout = {}) {
        const base = buildBaseStats(loadout);
        if (state.gameMode === "frontier" && state.frontier && state.frontier.active) {
          applyFrontierUpgrades(base, state.frontier.upgrades);
        }
        const ability = ABILITIES[base.ship.abilityId];
        const spawn = getSpawnPoint();
        return {
          ...base,
          ability,
          x: spawn.x,
          y: spawn.y,
          vx: 0,
          vy: 0,
          angle: 0,
          health: base.maxHealth,
          shield: base.maxShield,
          energy: base.maxEnergy,
          radius: 14,
          shieldCooldown: 0,
          fireCooldown: 0,
          hitFlash: 0,
          abilityTimer: 0,
          abilityCooldown: 0,
          secondaryCooldown: 0,
          invulnerable: 0
        };
      }

      function createStars(count) {
        const width = state.worldWidth || state.width;
        const height = state.worldHeight || state.height;
        return Array.from({ length: count }, () => ({
          x: Math.random() * width,
          y: Math.random() * height,
          radius: rand(0.6, 1.6),
          speed: rand(6, 26),
          alpha: rand(0.3, 0.9)
        }));
      }

      function updateStarfield() {
        stars = createStars(getStarCount());
      }

      function getStarCount() {
        const base = getParticleCount(160);
        const viewArea = state.width * state.height;
        const worldArea = state.worldWidth * state.worldHeight;
        if (!viewArea || !worldArea) return base;
        const density = clamp(worldArea / viewArea, 1, 4);
        return Math.round(base * density);
      }

      function setupWorld() {
        state.worldWidth = WORLD_WIDTH;
        state.worldHeight = WORLD_HEIGHT;
        obstacles = generateObstacles();
        updateStarfield();
        updateCamera();
        updatePointerWorld();
      }

      function generateObstacles() {
        const width = state.worldWidth;
        const height = state.worldHeight;
        const padding = 160;
        const safeRadius = 200;
        const obstacles = [];
        const rockSeeds = [
          { x: width * 0.28, y: height * 0.3 },
          { x: width * 0.7, y: height * 0.3 },
          { x: width * 0.32, y: height * 0.7 },
          { x: width * 0.68, y: height * 0.66 }
        ];
        rockSeeds.forEach((seed) => {
          const radius = rand(110, 170);
          const x = clamp(seed.x + rand(-130, 130), radius + padding, width - radius - padding);
          const y = clamp(seed.y + rand(-120, 120), radius + padding, height - radius - padding);
          obstacles.push({
            kind: "rock",
            x,
            y,
            radius,
            shade: rand(0.3, 0.7)
          });
        });
        const debrisSeeds = [
          { x: width * 0.5, y: height * 0.18, width: 520, height: 90 },
          { x: width * 0.5, y: height * 0.82, width: 520, height: 90 },
          { x: width * 0.18, y: height * 0.5, width: 90, height: 420 },
          { x: width * 0.82, y: height * 0.5, width: 90, height: 420 }
        ];
        debrisSeeds.forEach((seed) => {
          const x = clamp(seed.x + rand(-90, 90), seed.width / 2 + padding, width - seed.width / 2 - padding);
          const y = clamp(seed.y + rand(-90, 90), seed.height / 2 + padding, height - seed.height / 2 - padding);
          obstacles.push({
            kind: "debris",
            x,
            y,
            width: seed.width,
            height: seed.height,
            shade: rand(0.2, 0.6)
          });
        });
        for (let i = 0; i < 6; i += 1) {
          const radius = rand(60, 100);
          const x = rand(radius + padding, width - radius - padding);
          const y = rand(radius + padding, height - radius - padding);
          obstacles.push({
            kind: "rock",
            x,
            y,
            radius,
            shade: rand(0.25, 0.55)
          });
        }
        return obstacles.filter((obstacle) => !isObstacleNearPoint(obstacle, width * 0.5, height * 0.5, safeRadius));
      }

      function isObstacleNearPoint(obstacle, x, y, buffer) {
        if (!obstacle) return false;
        if (obstacle.kind === "rock") {
          return distanceBetween(obstacle, { x, y }) <= obstacle.radius + buffer;
        }
        const halfWidth = obstacle.width * 0.5;
        const halfHeight = obstacle.height * 0.5;
        const closestX = clamp(x, obstacle.x - halfWidth, obstacle.x + halfWidth);
        const closestY = clamp(y, obstacle.y - halfHeight, obstacle.y + halfHeight);
        return Math.hypot(x - closestX, y - closestY) <= buffer;
      }

      function getSpawnPoint() {
        const center = { x: state.worldWidth * 0.5, y: state.worldHeight * 0.5 };
        if (!isPointInObstacle(center, 140)) return center;
        for (let i = 0; i < 12; i += 1) {
          const angle = rand(0, Math.PI * 2);
          const distance = rand(140, 320);
          const candidate = {
            x: clamp(center.x + Math.cos(angle) * distance, 80, state.worldWidth - 80),
            y: clamp(center.y + Math.sin(angle) * distance, 80, state.worldHeight - 80)
          };
          if (!isPointInObstacle(candidate, 140)) {
            return candidate;
          }
        }
        return center;
      }

      function getFieldDropSpawnPoint() {
        const padding = 70;
        const buffer = 30;
        for (let i = 0; i < 20; i += 1) {
          const candidate = {
            x: rand(padding, state.worldWidth - padding),
            y: rand(padding, state.worldHeight - padding)
          };
          if (isPointInObstacle(candidate, buffer)) continue;
          if (player && distanceBetween(candidate, player) < FIELD_DROP_MIN_DISTANCE) continue;
          const tooClose = fieldDrops.some((drop) => distanceBetween(candidate, drop) < (drop.radius || 16) + 32);
          if (tooClose) continue;
          return candidate;
        }
        return {
          x: rand(padding, state.worldWidth - padding),
          y: rand(padding, state.worldHeight - padding)
        };
      }

      function isPointInObstacle(point, buffer = 0) {
        return obstacles.some((obstacle) => {
          if (obstacle.kind === "rock") {
            return distanceBetween(point, obstacle) <= obstacle.radius + buffer;
          }
          const halfWidth = obstacle.width * 0.5;
          const halfHeight = obstacle.height * 0.5;
          const insideX = point.x >= obstacle.x - halfWidth - buffer && point.x <= obstacle.x + halfWidth + buffer;
          const insideY = point.y >= obstacle.y - halfHeight - buffer && point.y <= obstacle.y + halfHeight + buffer;
          return insideX && insideY;
        });
      }

      function getViewportBounds() {
        const left = state.camera.x;
        const top = state.camera.y;
        return {
          left,
          top,
          right: left + state.width,
          bottom: top + state.height
        };
      }

      function updateCamera() {
        if (!player) return;
        const maxX = Math.max(0, state.worldWidth - state.width);
        const maxY = Math.max(0, state.worldHeight - state.height);
        state.camera.x = clamp(player.x - state.width * 0.5, 0, maxX);
        state.camera.y = clamp(player.y - state.height * 0.5, 0, maxY);
      }

      function screenToWorld(x, y) {
        return {
          x: clamp(x + state.camera.x, 0, state.worldWidth),
          y: clamp(y + state.camera.y, 0, state.worldHeight)
        };
      }

      function updatePointerWorld() {
        if (!input.pointer.active || !player) {
          input.pointer.x = player ? player.x : state.worldWidth * 0.5;
          input.pointer.y = player ? player.y : state.worldHeight * 0.5;
          return;
        }
        const world = screenToWorld(input.pointer.screenX, input.pointer.screenY);
        input.pointer.x = world.x;
        input.pointer.y = world.y;
      }

      function rollSectorModifier() {
        return pick(SECTOR_MODIFIERS);
      }

      function updateSector() {
        const waveProgress = getGlobalWave(state.wave);
        const nextSector = Math.max(1, Math.ceil(waveProgress / 3));
        if (nextSector !== state.sector) {
          state.sector = nextSector;
          state.sectorMod = rollSectorModifier();
          if (state.sectorMod) {
            logEvent(`Sector ${state.sector.toString().padStart(2, "0")} anomaly: ${state.sectorMod.name}.`);
          }
        }
      }

      function rollContracts() {
        if (state.training || !isFeatureUnlocked("contracts")) return [];
        const pool = shuffle([...CONTRACT_DEFS]);
        const count = 2 + Math.round(Math.random());
        return pool.slice(0, count).map((contract) => ({
          ...contract,
          progress: 0,
          complete: false,
          factionId: pick(FACTIONS).id
        }));
      }

      function getDifficultySettings() {
        const base = DIFFICULTY_SETTINGS[state.difficulty] || DIFFICULTY_SETTINGS.normal;
        if (state.difficulty !== "adaptive" || !player) {
          return base;
        }
        const healthRatio = player.health / player.maxHealth;
        const performance = clamp((healthRatio - 0.5) * 0.35, -0.08, 0.12);
        return {
          ...base,
          enemyScale: base.enemyScale + performance,
          enemyDamage: base.enemyDamage + performance * 0.6,
          reward: base.reward + performance * 0.3
        };
      }

      function maybeShowRunQuickTip() {
        const helpKey = formatKeybind(progress.keybinds.help || "h");
        const abilityKey = formatKeybind(progress.keybinds.ability);
        const secondaryKey = formatKeybind(progress.keybinds.secondary);
        showTip("quick-run-hint", "Flight controls", `Ability ${abilityKey}, secondary ${secondaryKey}, help ${helpKey}.`, {
          kind: "info",
          duration: 6500
        });
      }

      function startMission() {
        if (state.mode !== "hangar" && state.mode !== "gameover" && state.mode !== "victory") return;
        applySettingsFromProgress();
        state.level = isFrontierMode() ? 1 : Math.max(1, progress.campaignLevel || 1);
        input.keys.clear();
        input.firing = false;
        input.padFiring = false;
        input.boost = false;
        checkProgressionUnlocks();
        enemies = [];
        bullets = [];
        helpers = [];
        particles = [];
        lootBursts = [];
        damageNumbers = [];
        fieldDrops = [];
        state.mines = [];
        state.decoy = null;
        state.enemyAccuracyMod = 1;
        state.upgradeStacks = {};
        state.upgradeOptions = [];
        state.upgradeRerolls = 0;
        state.skillSlots = [];
        state.lossRewards = null;
        state.levelRewards = null;
        state.salvageResults = null;
        state.blackHoles = [];
        state.hazards = [];
        state.waveObjective = null;
        state.threatTier = THREAT_TIERS[0].id;
        state.choiceEvent = null;
        state.routeBonus = null;
        state.milestoneRewardsClaimed = {};
        state.overlayReturnMode = null;
        state.fieldDropTimer = 0;
        state.wave = 1;
        state.score = 0;
        state.credits = 0;
        state.kills = 0;
        state.sector = 1;
        state.sectorMod = rollSectorModifier();
        state.training = false;
        state.resumeMode = "flight";
        state.runActive = true;
        state.runBanked = false;
        state.runEndedByAbort = false;
        state.runHighlights = [];
        state.runPremiumDrops = 0;
        state.telemetryRun = null;
        state.controllerPrevButtons = [];
        state.lastContractRender = 0;
        state.contracts = rollContracts();
        state.runStart = performance.now();
        state.waveStart = performance.now();
        setupWorld();
        if (isFrontierMode()) {
          initFrontierRun();
        } else {
          state.frontier = null;
          player = createPlayer();
        }
        updateCamera();
        updatePointerWorld();
        state.runLoadout = { ship: player.ship?.name || "Unknown", weapon: player.weapon?.name || "Unknown" };
        captureCurrentLoadoutAsLastRun();
        initRunTelemetry();
        spawnWave(state.wave);
        hideOverlay();
        setMode("flight");
        playAudioCue("launch");
        maybeShowRunQuickTip();
        if (state.sectorMod) {
          logEvent(`Sector 01 anomaly: ${state.sectorMod.name}.`);
        }
        if (isFrontierMode()) {
          logEvent("Frontier patrol launched. Upgrade at the dock.");
        } else {
          logEvent(`Level ${state.level} launched. Wave 1 inbound.`);
        }
        renderContracts();
      }

      function startTraining() {
        if (state.mode !== "hangar" && state.mode !== "gameover" && state.mode !== "victory") return;
        applySettingsFromProgress();
        state.gameMode = "arcade";
        input.keys.clear();
        input.firing = false;
        input.padFiring = false;
        input.boost = false;
        checkProgressionUnlocks();
        state.frontier = null;
        enemies = [];
        bullets = [];
        helpers = [];
        particles = [];
        lootBursts = [];
        damageNumbers = [];
        fieldDrops = [];
        state.mines = [];
        state.decoy = null;
        state.enemyAccuracyMod = 1;
        state.upgradeStacks = {};
        state.upgradeOptions = [];
        state.upgradeRerolls = 0;
        state.skillSlots = [];
        state.lossRewards = null;
        state.levelRewards = null;
        state.salvageResults = null;
        state.blackHoles = [];
        state.hazards = [];
        state.waveObjective = null;
        state.threatTier = THREAT_TIERS[0].id;
        state.choiceEvent = null;
        state.routeBonus = null;
        state.milestoneRewardsClaimed = {};
        state.overlayReturnMode = null;
        state.fieldDropTimer = 0;
        state.level = 1;
        state.wave = 1;
        state.score = 0;
        state.credits = 0;
        state.kills = 0;
        state.sector = 1;
        state.sectorMod = rollSectorModifier();
        state.training = true;
        state.resumeMode = "training";
        state.runActive = true;
        state.runBanked = false;
        state.runEndedByAbort = false;
        state.runHighlights = [];
        state.runPremiumDrops = 0;
        state.telemetryRun = null;
        state.controllerPrevButtons = [];
        state.contracts = [];
        state.runStart = performance.now();
        state.waveStart = performance.now();
        state.lastContractRender = 0;
        setupWorld();
        player = createPlayer();
        state.runLoadout = { ship: player.ship?.name || "Unknown", weapon: player.weapon?.name || "Unknown" };
        captureCurrentLoadoutAsLastRun();
        initRunTelemetry();
        updateCamera();
        updatePointerWorld();
        spawnWave(state.wave);
        hideOverlay();
        setMode("training");
        playAudioCue("launch");
        maybeShowRunQuickTip();
        if (state.sectorMod) {
          logEvent(`Training sector: ${state.sectorMod.name}.`);
        }
        logEvent("Training session initiated. No rewards earned.");
        renderContracts();
      }

      function resetMission() {
        if (state.mode !== "hangar" && state.mode !== "gameover" && state.mode !== "victory") {
          triggerAbortRewards();
          return;
        }
        if (state.mode !== "hangar") {
          endRun("abort");
        }
        setMode("hangar");
        enemies = [];
        bullets = [];
        helpers = [];
        particles = [];
        lootBursts = [];
        damageNumbers = [];
        fieldDrops = [];
        state.mines = [];
        state.decoy = null;
        state.enemyAccuracyMod = 1;
        state.upgradeStacks = {};
        state.upgradeOptions = [];
        state.upgradeRerolls = 0;
        state.skillSlots = [];
        state.lossRewards = null;
        state.levelRewards = null;
        state.salvageResults = null;
        state.blackHoles = [];
        state.hazards = [];
        state.waveObjective = null;
        state.threatTier = THREAT_TIERS[0].id;
        state.choiceEvent = null;
        state.routeBonus = null;
        state.milestoneRewardsClaimed = {};
        state.overlayReturnMode = null;
        state.fieldDropTimer = 0;
        state.level = isFrontierMode() ? 1 : Math.max(1, progress.campaignLevel || 1);
        state.wave = 1;
        state.score = 0;
        state.credits = 0;
        state.kills = 0;
        state.contracts = [];
        state.sector = 1;
        state.sectorMod = null;
        state.training = false;
        state.runActive = false;
        state.runBanked = false;
        state.runEndedByAbort = false;
        state.runLoadout = null;
        state.runHighlights = [];
        state.runPremiumDrops = 0;
        state.telemetryRun = null;
        state.controllerPrevButtons = [];
        state.lastContractRender = 0;
        state.frontier = null;
        input.keys.clear();
        input.firing = false;
        input.padFiring = false;
        input.boost = false;
        setupWorld();
        player = createPlayer();
        setOverlay("start");
        renderContracts();
        renderShipyard();
        renderPremiumShop();
        renderArmory();
        renderHistory();
        logEvent("Run reset. Hangar ready.");
      }

      function setMode(mode) {
        state.mode = mode;
        if (mode === "hangar" && settingsOpen) {
          closeSettings(false);
        }
        const labels = {
          hangar: "Hangar",
          flight: "In Flight",
          upgrade: "Upgrade Bay",
          dock: "Docked",
          "tier-select": "Tier Uplink",
          paused: "Paused",
          gameover: "Wrecked",
          training: "Training",
          victory: "Victory"
        };
        stats.status.textContent = labels[mode] || "Hangar";
        updateLayout();
        if (audioController) {
          audioController.setMode(mode);
        }
      }

      function togglePause() {
        if (state.mode === "dock" || state.mode === "tier-select") {
          closeFrontierDock();
        } else if (state.mode === "flight" || state.mode === "training") {
          setPaused(true);
        } else if (state.mode === "paused") {
          setPaused(false);
        }
      }

      function toggleHelpOverlay() {
        if (state.mode === "flight" || state.mode === "training") {
          setPaused(true);
          setOverlay("help");
          return;
        }
        if (state.mode === "paused") {
          if (state.overlayMode === "help") {
            setPaused(false);
          } else {
            setOverlay("help");
          }
        }
      }

      function restoreOverlayReturn() {
        const nextMode = state.overlayReturnMode || (state.mode === "hangar" ? "start" : null);
        state.overlayReturnMode = null;
        if (nextMode) {
          setOverlay(nextMode);
        } else {
          hideOverlay();
        }
      }

      function startTutorial() {
        if (state.mode === "flight" || state.mode === "training") {
          setPaused(true);
          state.overlayReturnMode = "paused";
        } else {
          state.overlayReturnMode = state.overlayMode || (state.mode === "hangar" ? "start" : null);
        }
        state.tutorialMode = true;
        state.tutorialStep = 0;
        setOverlay("tutorial");
      }

      function openGlossary() {
        if (state.mode === "flight" || state.mode === "training") {
          setPaused(true);
          state.overlayReturnMode = "paused";
        } else {
          state.overlayReturnMode = state.overlayMode || (state.mode === "hangar" ? "start" : null);
        }
        setOverlay("glossary");
      }

      function setPaused(shouldPause) {
        if (shouldPause) {
          state.resumeMode = state.mode === "paused" ? state.resumeMode : state.mode;
          setMode("paused");
          setOverlay("paused");
        } else {
          setMode(state.resumeMode === "training" ? "training" : "flight");
          hideOverlay();
        }
      }

      function isFrontierMode() {
        return state.gameMode === "frontier";
      }

      function isCampaignMode() {
        return !isFrontierMode() && !state.training;
      }

      function getGlobalWave(wave = state.wave) {
        if (!isCampaignMode()) return wave;
        const level = Math.max(1, state.level || progress.campaignLevel || 1);
        return (level - 1) * LEVEL_WAVES + wave;
      }

      function getWaveDisplay(wave = state.wave) {
        if (!isCampaignMode()) return `${wave}`;
        return `${wave}/${LEVEL_WAVES}`;
      }

      function getActiveMutator() {
        return state.weekly && state.weekly.mutator ? state.weekly.mutator : null;
      }

      function getMutatorWaveConfig(globalWave) {
        const config = {
          enemyDamage: 1,
          enemyShieldScale: 1,
          enemyHealthScale: 1,
          enemyFireRate: 1,
          hazardBoost: 0
        };
        const mutator = getActiveMutator();
        if (mutator && typeof mutator.applyWave === "function") {
          mutator.applyWave(config, globalWave);
        }
        if (state.routeBonus && state.routeBonus.wavesRemaining > 0) {
          if (state.routeBonus.id === "aggression") {
            config.enemyHealthScale *= 1.08;
            config.enemyDamage *= 1.08;
          } else if (state.routeBonus.id === "fortify") {
            config.enemyDamage *= 0.9;
          } else if (state.routeBonus.id === "prospector") {
            config.enemyShieldScale *= 1.05;
          }
        }
        return config;
      }

      function getFrontierStarterId() {
        const starters = (FRONTIER_STARTERS || []).filter((id) => progress.shipUnlocks[id]);
        if (progress.selectedShip && starters.includes(progress.selectedShip)) {
          return progress.selectedShip;
        }
        if (starters.length) return starters[0];
        return (FRONTIER_STARTERS && FRONTIER_STARTERS[0]) || progress.selectedShip;
      }

      function initFrontierRun() {
        const starterId = getFrontierStarterId();
        const starterShip = getShipById(starterId);
        const weaponItem = getEquippedWeapon();
        state.frontier = {
          active: true,
          tier: starterShip.tier || 1,
          shipId: starterShip.id,
          weaponItem,
          upgrades: {},
          spawnTimer: 0
        };
        player = createPlayer({ shipId: starterShip.id, weaponItem });
      }

      function getFrontierUpgradeCost(upgrade, level, tier) {
        const tierScale = 1 + Math.max(0, tier - 1) * 0.28;
        const levelScale = 1 + level * 0.48;
        return Math.round((upgrade.baseCost || 100) * tierScale * levelScale);
      }

      function getFrontierTierCost(tier) {
        return Math.round(250 * Math.pow(Math.max(1, tier), 1.28));
      }

      function getHangarUpgradeCost(upgrade, level) {
        const baseCost = Number.isFinite(upgrade.baseCost) ? upgrade.baseCost : 1;
        const scale = Number.isFinite(upgrade.costScale) ? upgrade.costScale : 1.2;
        return Math.max(1, Math.ceil(baseCost * Math.pow(scale, level)));
      }

      function getFrontierNextShips() {
        if (!state.frontier || !state.frontier.shipId) return [];
        const ship = getShipById(state.frontier.shipId);
        const nextIds = ship.nextIds || [];
        return nextIds.map((id) => getShipById(id)).filter(Boolean);
      }

      function purchaseFrontierUpgrade(id) {
        if (!isFrontierMode() || !state.frontier || !player) return;
        const upgrade = FRONTIER_UPGRADES.find((item) => item.id === id);
        if (!upgrade) return;
        const level = state.frontier.upgrades[upgrade.id] || 0;
        const maxLevel = upgrade.maxLevel || 1;
        if (level >= maxLevel) return;
        const cost = getFrontierUpgradeCost(upgrade, level, state.frontier.tier);
        if (state.credits < cost) return;
        const healthRatio = player.maxHealth ? player.health / player.maxHealth : 1;
        const shieldRatio = player.maxShield ? player.shield / player.maxShield : 1;
        const energyRatio = player.maxEnergy ? player.energy / player.maxEnergy : 1;
        state.credits -= cost;
        state.frontier.upgrades[upgrade.id] = level + 1;
        if (upgrade.apply) {
          upgrade.apply(player, level + 1);
        }
        player.health = Math.min(player.maxHealth, player.maxHealth * healthRatio);
        player.shield = Math.min(player.maxShield, player.maxShield * shieldRatio);
        player.energy = Math.min(player.maxEnergy, player.maxEnergy * energyRatio);
        logEvent(`Frontier upgrade installed: ${upgrade.name}.`);
        renderActiveUpgrades();
        setOverlay("dock");
      }

      function toggleFrontierDock() {
        if (!isFrontierMode()) {
          logEvent("Frontier dock available in Frontier mode.");
          return;
        }
        if (state.mode === "dock" || state.mode === "tier-select") {
          closeFrontierDock();
        } else if (state.mode === "flight") {
          openFrontierDock();
        }
      }

      function openFrontierDock() {
        if (!isFrontierMode() || !state.frontier || state.mode !== "flight") return;
        setMode("dock");
        setOverlay("dock");
      }

      function closeFrontierDock() {
        if (state.mode !== "dock" && state.mode !== "tier-select") return;
        setMode("flight");
        hideOverlay();
      }

      function openFrontierTierSelect() {
        if (!isFrontierMode() || !state.frontier) return;
        const nextShips = getFrontierNextShips();
        if (!nextShips.length) return;
        const cost = getFrontierTierCost(state.frontier.tier);
        if (state.credits < cost) {
          logEvent("Not enough credits to upgrade tiers.");
          return;
        }
        setMode("tier-select");
        setOverlay("tier-select");
      }

      function selectFrontierShip(id) {
        if (!isFrontierMode() || !state.frontier) return;
        const nextShips = getFrontierNextShips().map((ship) => ship.id);
        if (!nextShips.includes(id)) return;
        const cost = getFrontierTierCost(state.frontier.tier);
        if (state.credits < cost) return;
        const ship = getShipById(id);
        state.credits -= cost;
        state.frontier.tier = ship.tier || state.frontier.tier + 1;
        state.frontier.shipId = ship.id;
        state.frontier.weaponItem = getEquippedWeapon();
        state.frontier.upgrades = {};
        state.frontier.spawnTimer = 0;
        player = createPlayer({ shipId: ship.id, weaponItem: state.frontier.weaponItem });
        state.runLoadout = { ship: ship.name, weapon: player.weapon?.name || "Unknown" };
        logEvent(`Tier ${state.frontier.tier} frame acquired: ${ship.name}.`);
        closeFrontierDock();
        renderActiveUpgrades();
      }

      function spawnWave(wave) {
        const globalWave = getGlobalWave(wave);
        const isHardWave = wave % 5 === 0;
        const isBossWave = isCampaignMode() ? wave === LEVEL_WAVES : wave % 10 === 0;
        const baseCount = Math.min(4 + Math.floor(globalWave * 0.95), 17);
        const difficulty = getDifficultySettings();
        const threatTier = resolveThreatTier(globalWave);
        state.threatTier = threatTier.id;
        const difficultyAimBonus = state.difficulty === "hard"
          ? 0.03
          : state.difficulty === "easy"
            ? -0.04
            : 0;
        state.enemyAccuracyMod = clamp(0.82 + globalWave * 0.009 + difficultyAimBonus, 0.78, 0.94);
        state.waveObjective = createWaveObjective(globalWave);
        state.objectiveClearPending = false;
        const frontierTier = isFrontierMode() && state.frontier ? state.frontier.tier : 1;
        const mutatorWave = getMutatorWaveConfig(globalWave);
        const waveRamp = 1
          + globalWave * 0.056
          + Math.max(0, globalWave - 12) * 0.009
          + Math.max(0, globalWave - 20) * 0.012;
        let enemyScale = waveRamp * difficulty.enemyScale * (1 + (frontierTier - 1) * 0.18) * threatTier.enemyScale;
        if (isBossWave) {
          enemyScale *= 1.1;
        } else if (isHardWave) {
          enemyScale *= 1.05;
        }
        const densityScale = getWaveDensityScale(globalWave);
        const hardBonus = isHardWave && !isBossWave ? 2 : 0;
        const bossBonus = isBossWave ? 3 : 0;
        const rawSpawnCount = baseCount
          + (wave % 3 === 0 ? 1 : 0)
          + hardBonus
          + bossBonus
          + (isFrontierMode() ? Math.max(0, Math.floor((frontierTier - 1) * 1.2)) : 0);
        const spawnCap = isBossWave ? 20 : 18;
        const spawnCount = clamp(Math.round(rawSpawnCount * densityScale), 3, spawnCap);
        const availableTypes = ENEMY_TYPES.filter((type) => {
          if (type.minWave && globalWave < type.minWave) return false;
          if (type.maxWave && globalWave > type.maxWave) return false;
          return true;
        });
        const selectionPool = availableTypes.length ? availableTypes : ENEMY_TYPES;
        const weights = selectionPool.map((type) => type.weight || 1);
        const forceEliteRate = Math.max(0, threatTier.eliteBonus || 0);
        for (let i = 0; i < spawnCount; i += 1) {
          const type = pickWeighted(selectionPool, weights);
          const enemy = createEnemy(type, enemyScale, difficulty);
          enemy.threatTier = threatTier.id;
          enemy.baseDamage *= mutatorWave.enemyDamage;
          enemy.maxShield *= mutatorWave.enemyShieldScale;
          enemy.shield = enemy.maxShield;
          enemy.maxHealth *= mutatorWave.enemyHealthScale;
          enemy.health = enemy.maxHealth;
          enemy.baseFireRate *= (mutatorWave.enemyFireRate * (threatTier.fireRate || 1));
          if (!enemy.elite && type.id !== "dreadnought" && Math.random() < forceEliteRate) {
            applyEliteMod(enemy);
          }
          if (threatTier.id === "mythic") {
            enemy.dashCooldown = rand(3.4, 5.4);
            enemy.dashTimer = rand(1, enemy.dashCooldown);
            enemy.dashStrength = 1.8;
          }
          enemies.push(enemy);
        }
        if (isHardWave && !isBossWave) {
          enemies.push(createEnemy(ACE_TYPE, enemyScale + 0.25, difficulty, true));
          logEvent("Hard wave incoming. Enemy ace spotted.");
        }
        if (isBossWave) {
          enemies.push(createEnemy(BOSS_TYPE, enemyScale + 0.5, difficulty, true));
          logEvent("Boss wave: capital dreadnought detected.");
        }
        if (!isBossWave && globalWave >= 6 && (isHardWave || Math.random() < 0.22)) {
          enemies.push(createMiniBoss(enemyScale, difficulty, threatTier));
          logEvent("Miniboss contact: command-class hostile entering combat.");
        }
        spawnWaveHazards(globalWave, threatTier, mutatorWave);
        if (state.routeBonus && state.routeBonus.wavesRemaining > 0) {
          state.routeBonus.wavesRemaining -= 1;
          if (state.routeBonus.wavesRemaining <= 0) {
            state.routeBonus = null;
            logEvent("Route bonus expired.");
          }
        }
        updateObjectiveProgressLabel();
      }

      function createMiniBoss(scale, difficulty, threatTier) {
        const baseType = pick([
          ENEMY_TYPES.find((item) => item.id === "bulwark"),
          ENEMY_TYPES.find((item) => item.id === "bomber"),
          ENEMY_TYPES.find((item) => item.id === "disruptor")
        ].filter(Boolean));
        const type = baseType || BOSS_TYPE;
        const enemy = createEnemy(type, scale * 1.45, difficulty, true);
        enemy.id = `miniboss-${type.id}`;
        enemy.name = `Command ${type.name}`;
        enemy.radius = Math.max(enemy.radius, 20);
        enemy.maxHealth *= 1.35;
        enemy.health = enemy.maxHealth;
        enemy.maxShield *= 1.3;
        enemy.shield = enemy.maxShield;
        enemy.baseDamage *= 1.22;
        enemy.baseFireRate *= 1.1;
        enemy.score *= 2.1;
        enemy.credits = Math.round(enemy.credits * 2.2);
        enemy.color = "#ff9f6b";
        enemy.miniboss = true;
        enemy.threatTier = threatTier.id;
        enemy.phaseThreshold = enemy.maxHealth * 0.55;
        enemy.phaseActive = false;
        enemy.phaseTimer = 0;
        enemy.summonTimer = rand(5, 8);
        enemy.summonCooldown = rand(8, 11);
        enemy.dashCooldown = rand(3.2, 4.8);
        enemy.dashTimer = rand(1, enemy.dashCooldown);
        enemy.dashStrength = 2.2;
        return enemy;
      }

      function spawnWaveHazards(globalWave, threatTier, waveConfig) {
        state.hazards = [];
        if (state.training) return;
        const baseCount = globalWave >= 8 ? 1 : 0;
        const tierBonus = threatTier.id === "elite" ? 1 : threatTier.id === "mythic" ? 2 : 0;
        const mutatorBonus = Math.floor((waveConfig?.hazardBoost || 0) * 2);
        const count = clamp(baseCount + tierBonus + mutatorBonus, 0, 4);
        for (let i = 0; i < count; i += 1) {
          const point = randomWorldPosition();
          state.hazards.push({
            id: `hazard_${Date.now()}_${i}`,
            x: point.x,
            y: point.y,
            radius: rand(90, 140),
            damagePerSec: rand(10, 18) * (1 + globalWave * 0.02),
            slowFactor: 0.85,
            life: rand(14, 22)
          });
        }
      }

      function getWaveDensityScale(globalWave) {
        const tiers = BALANCE_TUNING.waveDensitySoftScale || [];
        for (let i = 0; i < tiers.length; i += 1) {
          const tier = tiers[i];
          if (globalWave <= tier.wave) return tier.scale;
        }
        return 1;
      }

      function createEnemy(type, scale, difficulty, forceElite) {
        const position = randomWorldPosition();
        const baseHealth = type.health * scale;
        const baseShield = type.shield * scale;
        const baseSpeed = type.speed + scale * 12;
        const baseAccel = type.accel + scale * 16;
        const baseFireRate = type.fireRate + scale * 0.04;
        const baseBulletSpeed = type.bulletSpeed + scale * 10;
        const baseDamage = (type.damage + scale * 1.2) * (difficulty ? difficulty.enemyDamage : 1);
        const baseTurnRate = type.turnRate || 2.6;
        const preferredRange = Number.isFinite(type.preferredRange)
          ? type.preferredRange + rand(-18, 18)
          : rand(180, 260);
        const strafeBias = Number.isFinite(type.strafeBias)
          ? type.strafeBias
          : (Math.random() > 0.5 ? 1 : -1);
        const burstCount = type.burstCount || 0;
        const burstInterval = type.burstInterval || 0.12;
        const burstCooldown = type.burstCooldown || 1.2;
        const shieldPulseCooldown = type.shieldPulseCooldown || 0;
        const ramCooldown = type.ramCooldown || 0;
        const enemy = {
          id: type.id,
          name: type.name,
          x: position.x,
          y: position.y,
          vx: 0,
          vy: 0,
          angle: rand(0, Math.PI * 2),
          radius: type.radius,
          maxHealth: baseHealth,
          health: baseHealth,
          maxShield: baseShield,
          shield: baseShield,
          baseSpeed,
          baseAccel,
          baseFireRate,
          baseBulletSpeed,
          baseDamage,
          baseTurnRate,
          fireCooldown: rand(0.2, 0.6),
          color: type.color,
          credits: type.credits,
          score: type.score,
          hitFlash: 0,
          slowTimer: 0,
          slowFactor: 1,
          preferredRange,
          strafeBias,
          pattern: type.pattern || (type.spreadCount ? "spread" : "single"),
          spreadCount: type.spreadCount || 0,
          spreadAngle: Number.isFinite(type.spreadAngle) ? type.spreadAngle : null,
          bulletRadius: type.bulletRadius || 3,
          bulletLife: type.bulletLife || 1.6,
          bulletTint: type.bulletTint || null,
          bulletSlowPlayer: !!type.bulletSlowPlayer,
          bulletSlowDuration: type.bulletSlowDuration || 1.4,
          burstCount,
          burstInterval,
          burstCooldown,
          burstRemaining: 0,
          burstTimer: 0,
          shieldPulseCooldown,
          shieldPulseTimer: shieldPulseCooldown ? rand(0.3, shieldPulseCooldown) : 0,
          shieldPulseRadius: type.shieldPulseRadius || 0,
          shieldPulseAmount: type.shieldPulseAmount || 0,
          shieldPulseColor: type.shieldPulseColor || "#7ca8ff",
          shieldPulseSelf: !!type.shieldPulseSelf,
          ramDamage: type.ramDamage || 0,
          ramCooldown,
          ramTimer: ramCooldown ? rand(0.3, ramCooldown) : 0,
          ramRange: type.ramRange || 0,
          ramKnockback: type.ramKnockback || 0,
          navTarget: null,
          navTimer: 0,
          navCooldown: 0,
          stuckTimer: 0,
          lastTargetDist: null
        };
        const eliteChance = forceElite ? 1 : (scale > 1.1 && Math.random() < 0.18);
        if (eliteChance && type.id !== "dreadnought") {
          applyEliteMod(enemy);
        }
        return enemy;
      }

      function applyEliteMod(enemy) {
        const mod = pick(ELITE_MODS);
        enemy.elite = mod;
        if (mod.mods.maxShield) {
          enemy.maxShield *= 1 + mod.mods.maxShield;
          enemy.shield = enemy.maxShield;
        }
        if (mod.mods.speed) {
          enemy.baseSpeed *= mod.mods.speed;
        }
        if (mod.mods.accel) {
          enemy.baseAccel *= mod.mods.accel;
        }
        if (mod.mods.damage) {
          enemy.baseDamage *= mod.mods.damage;
        }
        if (mod.mods.fireRate) {
          enemy.baseFireRate *= mod.mods.fireRate;
        }
        enemy.color = mod.color || enemy.color;
      }

      function randomWorldPosition() {
        const padding = 60;
        const minDistance = player ? 240 : 0;
        for (let i = 0; i < 12; i += 1) {
          const position = {
            x: rand(padding, state.worldWidth - padding),
            y: rand(padding, state.worldHeight - padding)
          };
          if (isPointInObstacle(position, 50)) continue;
          if (player && distanceBetween(position, player) < minDistance) continue;
          return position;
        }
        return {
          x: clamp(state.worldWidth * 0.5 + rand(-120, 120), padding, state.worldWidth - padding),
          y: clamp(state.worldHeight * 0.5 + rand(-120, 120), padding, state.worldHeight - padding)
        };
      }

      function loop(timestamp) {
        const perfMode = getPerfMode();
        const shouldMeasure = perfMode !== "off" || perfLog.active;
        const rawDeltaMs = Math.max(0, timestamp - state.lastTime || 0);
        const delta = Math.min(0.033, rawDeltaMs / 1000 || 0);
        state.lastTime = timestamp;
        if (shouldMeasure) {
          const frameStart = performance.now();
          const updateStart = performance.now();
          update(delta);
          const updateEnd = performance.now();
          render();
          const renderEnd = performance.now();
          updateHud();
          const hudEnd = performance.now();
          recordPerformance(
            frameStart,
            updateEnd - updateStart,
            renderEnd - updateEnd,
            hudEnd - renderEnd,
            rawDeltaMs,
            delta * 1000
          );
        } else {
          update(delta);
          render();
          updateHud();
        }
        requestAnimationFrame(loop);
      }

      function update(delta) {
        updateStars(delta);
        updateParticles(delta);
        updatePulses(delta);
        updateLootBursts(delta);
        updateDamageNumbers(delta);
        trackOnboarding(delta);
        if (state.mode !== "flight" && state.mode !== "training") {
          return;
        }
        updateCamera();
        updatePointerWorld();
        pollGamepadInput(delta);
        updatePlayer(delta);
        updateDecoy(delta);
        updateFieldDrops(delta);
        updateHazards(delta);
        updateEnemies(delta);
        updateBlackHoles(delta);
        updateSkillSystems(delta);
        updateHelpers(delta);
        updateBullets(delta);
        updateMines(delta);
        updateContracts(delta);
        handleCollisions();
        updateWaveObjective(delta);
        if (isFrontierMode()) {
          updateFrontierSpawner(delta);
        } else {
          checkWaveStatus();
        }
      }

      function trackOnboarding(delta) {
        if (state.mode !== "flight" && state.mode !== "training") return;
        state.onboardingTimer += delta;
        if (state.onboardingTimer < 1) return;
        const elapsed = Math.floor(state.onboardingTimer);
        state.onboardingTimer -= elapsed;
        progress.onboarding = progress.onboarding || { flightSeconds: 0 };
        progress.onboarding.flightSeconds = (progress.onboarding.flightSeconds || 0) + elapsed;
        state.onboardingSave += elapsed;
        checkProgressionUnlocks();
        if (state.onboardingSave >= 12) {
          state.onboardingSave = 0;
          saveProgress();
        }
      }

      function pollGamepadInput() {
        if (!navigator.getGamepads) return;
        const pads = navigator.getGamepads();
        const pad = Array.from(pads || []).find((item) => item && item.connected);
        state.controllerConnected = !!pad;
        if (!pad) {
          input.padMoveX = 0;
          input.padMoveY = 0;
          input.padAimX = 0;
          input.padAimY = 0;
          input.padFiring = false;
          return;
        }
        input.padMoveX = Math.abs(pad.axes[0] || 0) > 0.15 ? (pad.axes[0] || 0) : 0;
        input.padMoveY = Math.abs(pad.axes[1] || 0) > 0.15 ? (pad.axes[1] || 0) : 0;
        input.padAimX = Math.abs(pad.axes[2] || 0) > 0.2 ? (pad.axes[2] || 0) : 0;
        input.padAimY = Math.abs(pad.axes[3] || 0) > 0.2 ? (pad.axes[3] || 0) : 0;
        const readButton = (index) => {
          const btn = pad.buttons[index];
          return !!(btn && btn.pressed);
        };
        const firePressed = readButton(7) || readButton(5);
        const boostPressed = readButton(0);
        const abilityPressed = readButton(1);
        const secondaryPressed = readButton(2);
        const dockPressed = readButton(3);
        const pausePressed = readButton(9);
        const helpPressed = readButton(8);
        const previous = state.controllerPrevButtons || [];
        const wasPressed = (index) => !!previous[index];
        if (abilityPressed && !wasPressed(1)) {
          activateAbility();
        }
        if (secondaryPressed && !wasPressed(2)) {
          activateSecondary();
        }
        if (dockPressed && !wasPressed(3)) {
          toggleFrontierDock();
        }
        if (pausePressed && !wasPressed(9)) {
          togglePause();
        }
        if (helpPressed && !wasPressed(8)) {
          toggleHelpOverlay();
        }
        state.controllerPrevButtons = [false, abilityPressed, secondaryPressed, dockPressed, false, firePressed, false, firePressed, helpPressed, pausePressed];
        input.padFiring = firePressed;
        input.boost = boostPressed;
        if (input.padAimX || input.padAimY) {
          input.aimAngle = Math.atan2(input.padAimY, input.padAimX);
          input.aimSource = "keyboard";
        }
      }

      function updateHazards(delta) {
        if (!state.hazards || !state.hazards.length) return;
        for (let i = state.hazards.length - 1; i >= 0; i -= 1) {
          const hazard = state.hazards[i];
          hazard.life -= delta;
          if (hazard.life <= 0) {
            state.hazards.splice(i, 1);
            continue;
          }
          if (player && distanceBetween(player, hazard) <= hazard.radius + player.radius) {
            applyDamage(player, hazard.damagePerSec * delta, { owner: "enemy" });
            player.slowTimer = Math.max(player.slowTimer || 0, 0.18);
            if (state.telemetryRun) {
              state.telemetryRun.hazardTicks += 1;
            }
          }
          for (let j = enemies.length - 1; j >= 0; j -= 1) {
            const enemy = enemies[j];
            if (distanceBetween(enemy, hazard) > hazard.radius + enemy.radius) continue;
            applyDamage(enemy, hazard.damagePerSec * delta * 0.75, { owner: "player" });
            enemy.slowTimer = Math.max(enemy.slowTimer || 0, 0.18);
            if (enemy.health <= 0) {
              destroyEnemy(enemy, j);
            }
          }
        }
      }

      function updateWaveObjective(delta) {
        if (!state.waveObjective || state.mode !== "flight") return;
        const objective = state.waveObjective;
        if (objective.complete) return;
        if (objective.id === "survive") {
          objective.timer = Math.max(0, (objective.timer || 0) - delta);
          if (objective.timer <= 0) {
            objective.complete = true;
            enemies = [];
            logEvent("Objective complete: survival timer held.");
            if (state.telemetryRun) state.telemetryRun.objectiveCompletions += 1;
          }
        } else if (objective.id === "elite-hunt") {
          if ((objective.progress || 0) >= (objective.target || 0)) {
            objective.complete = true;
            enemies = [];
            logEvent("Objective complete: priority targets neutralized.");
            if (state.telemetryRun) state.telemetryRun.objectiveCompletions += 1;
          }
        } else if (objective.id === "eliminate" && enemies.length === 0) {
          objective.complete = true;
          if (state.telemetryRun) state.telemetryRun.objectiveCompletions += 1;
        }
        updateObjectiveProgressLabel();
      }

      function updateStars(delta) {
        const width = state.worldWidth || state.width;
        const height = state.worldHeight || state.height;
        stars.forEach((star) => {
          star.x += star.speed * delta * 0.03;
          star.y += star.speed * delta * 0.18;
          if (star.x < 0) star.x += width;
          if (star.x > width) star.x -= width;
          if (star.y < 0) star.y += height;
          if (star.y > height) star.y -= height;
        });
      }

      function updateParticles(delta) {
        for (let i = particles.length - 1; i >= 0; i -= 1) {
          const particle = particles[i];
          particle.x += particle.vx * delta;
          particle.y += particle.vy * delta;
          particle.life -= delta;
          if (particle.life <= 0) {
            particles.splice(i, 1);
          }
        }
      }

      function updatePulses(delta) {
        for (let i = pulses.length - 1; i >= 0; i -= 1) {
          const pulse = pulses[i];
          pulse.life -= delta;
          pulse.radius = Math.min(pulse.maxRadius, pulse.radius + pulse.speed * delta);
          if (pulse.life <= 0) {
            pulses.splice(i, 1);
          }
        }
      }

      function updateBlackHoles(delta) {
        if (!state.blackHoles || !state.blackHoles.length) return;
        for (let i = state.blackHoles.length - 1; i >= 0; i -= 1) {
          const hole = state.blackHoles[i];
          hole.life -= delta;
          if (hole.life <= 0) {
            state.blackHoles.splice(i, 1);
            continue;
          }
          for (let j = enemies.length - 1; j >= 0; j -= 1) {
            const enemy = enemies[j];
            const dx = hole.x - enemy.x;
            const dy = hole.y - enemy.y;
            const dist = Math.hypot(dx, dy) || 1;
            if (dist > hole.radius) continue;
            const pull = (1 - dist / hole.radius) * hole.force;
            enemy.vx += (dx / dist) * pull * delta;
            enemy.vy += (dy / dist) * pull * delta;
            if (hole.damage > 0) {
              applyDamage(enemy, hole.damage * delta, { owner: "player" });
              if (enemy.health <= 0) {
                destroyEnemy(enemy, j);
              }
            }
          }
        }
      }

      function updateLootBursts(delta) {
        for (let i = lootBursts.length - 1; i >= 0; i -= 1) {
          const burst = lootBursts[i];
          burst.life -= delta;
          burst.y += burst.vy * delta;
          if (burst.life <= 0) {
            lootBursts.splice(i, 1);
          }
        }
      }

      function updateDamageNumbers(delta) {
        for (let i = damageNumbers.length - 1; i >= 0; i -= 1) {
          const burst = damageNumbers[i];
          burst.life -= delta;
          burst.x += burst.vx * delta;
          burst.y += burst.vy * delta;
          if (burst.life <= 0) {
            damageNumbers.splice(i, 1);
          }
        }
      }

      function updatePlayer(delta) {
        if (!player) return;
        if (player.slowTimer > 0) {
          player.slowTimer = Math.max(0, player.slowTimer - delta);
        }
        player.slowFactor = player.slowTimer > 0 ? 0.7 : 1;
        const keyboardAim = getKeyboardAimVector();
        if (keyboardAim.active) {
          input.aimAngle = Math.atan2(keyboardAim.y, keyboardAim.x);
          input.aimSource = "keyboard";
        } else if (input.aimMode === "hybrid" && input.pointer.active) {
          input.aimAngle = Math.atan2(input.pointer.y - player.y, input.pointer.x - player.x);
          input.aimSource = "mouse";
        }
        const turnRate = (player.turnRate || 0) * (player.slowFactor || 1);
        player.angle = rotateTowards(player.angle, input.aimAngle, turnRate * delta);
        input.pointer.moved = false;

        if (player.abilityCooldown > 0) {
          player.abilityCooldown = Math.max(0, player.abilityCooldown - delta);
        }
        if (player.abilityTimer > 0) {
          player.abilityTimer = Math.max(0, player.abilityTimer - delta);
          if (player.abilityTimer <= 0 && player.ability && player.ability.onEnd) {
            player.ability.onEnd(player, state);
          }
        }
        if (player.secondaryCooldown > 0) {
          player.secondaryCooldown = Math.max(0, player.secondaryCooldown - delta);
        }
        if (player.invulnerable > 0) {
          player.invulnerable = Math.max(0, player.invulnerable - delta);
        }
        if (player.damageBoostTimer > 0) {
          player.damageBoostTimer = Math.max(0, player.damageBoostTimer - delta);
          if (player.damageBoostTimer <= 0) {
            player.damageBoostMultiplier = 1;
          }
        }
        if (player.speedBoostTimer > 0) {
          player.speedBoostTimer = Math.max(0, player.speedBoostTimer - delta);
          if (player.speedBoostTimer <= 0) {
            player.speedBoostMultiplier = 1;
          }
        }

        const allowArrowThrust = input.aimMode !== "keyboard";
        const thrustForward = isActionActive("forward") || (allowArrowThrust && hasKey("arrowup"));
        const thrustBack = isActionActive("back") || (allowArrowThrust && hasKey("arrowdown"));
        const strafeLeft = isActionActive("left") || (allowArrowThrust && hasKey("arrowleft"));
        const strafeRight = isActionActive("right") || (allowArrowThrust && hasKey("arrowright"));
        const padX = input.padMoveX || 0;
        const padY = input.padMoveY || 0;
        const boosting = (isActionActive("boost") || input.boost) && player.energy > player.boostCost * delta;

        if (boosting) {
          player.energy = Math.max(0, player.energy - player.boostCost * delta);
        }

        const playerMod = state.sectorMod ? state.sectorMod.player : {};
        const slowFactor = player.slowFactor || 1;
        const speedBoost = player.speedBoostTimer > 0 ? (player.speedBoostMultiplier || 1) : 1;
        const accel = player.accel * (boosting ? player.boostMultiplier : 1) * (playerMod.speed || 1) * slowFactor * speedBoost;
        let ax = 0;
        let ay = 0;
        player.thrusting = false;

        if (thrustForward) {
          ax += Math.cos(player.angle) * accel;
          ay += Math.sin(player.angle) * accel;
          player.thrusting = true;
        }
        if (thrustBack) {
          ax -= Math.cos(player.angle) * accel * 0.6;
          ay -= Math.sin(player.angle) * accel * 0.6;
          player.thrusting = true;
        }
        if (strafeLeft) {
          ax += Math.cos(player.angle - Math.PI / 2) * accel * 0.7;
          ay += Math.sin(player.angle - Math.PI / 2) * accel * 0.7;
          player.thrusting = true;
        }
        if (strafeRight) {
          ax += Math.cos(player.angle + Math.PI / 2) * accel * 0.7;
          ay += Math.sin(player.angle + Math.PI / 2) * accel * 0.7;
          player.thrusting = true;
        }
        if (padX || padY) {
          ax += padX * accel * 0.9;
          ay += padY * accel * 0.9;
          player.thrusting = true;
        }

        player.vx += ax * delta;
        player.vy += ay * delta;

        const damping = Math.max(0, 1 - player.damping * delta);
        player.vx *= damping;
        player.vy *= damping;

        const speedLimit = player.maxSpeed * (boosting ? player.boostMultiplier : 1) * (playerMod.speed || 1) * slowFactor * speedBoost;
        const speed = Math.hypot(player.vx, player.vy);
        if (speed > speedLimit) {
          const ratio = speedLimit / speed;
          player.vx *= ratio;
          player.vy *= ratio;
        }

        player.x += player.vx * delta;
        player.y += player.vy * delta;
        wrapEntity(player);
        resolveObstacleCollisions(player);
        wrapEntity(player);

        player.fireCooldown = Math.max(0, player.fireCooldown - delta);
        player.shieldCooldown = Math.max(0, player.shieldCooldown - delta);

        const energyRegen = player.energyRegen * (playerMod.energyRegen || 1);
        const shieldRegen = player.shieldRegen * (playerMod.shieldRegen || 1);
        player.energy = Math.min(player.maxEnergy, player.energy + energyRegen * delta);
        if (player.shieldCooldown <= 0) {
          player.shield = Math.min(player.maxShield, player.shield + shieldRegen * delta);
        }

        if (player.hitFlash > 0) {
          player.hitFlash = Math.max(0, player.hitFlash - delta);
        }

        const firing = input.firing || input.padFiring || isActionActive("fire");
        const energyCost = player.energyCost * (playerMod.energyCost || 1) * BALANCE_TUNING.playerEnergyCostScale;
        if (firing && player.fireCooldown <= 0 && player.energy >= energyCost) {
          firePlayer();
        }
      }

      function updateSkillSystems(delta) {
        if (!player) return;
        updateAura(delta);
        updateMineDeployment(delta);
        updateShockwave(delta);
        updateMissiles(delta);
      }

      function updateAura(delta) {
        if (player.auraRadius <= 0 || player.auraDamage <= 0) return;
        player.auraTimer = (player.auraTimer || 0) - delta;
        if (player.auraTimer > 0) return;
        const interval = player.auraInterval || 0.45;
        player.auraTimer = interval;
        const damage = player.auraDamage * interval;
        for (let i = enemies.length - 1; i >= 0; i -= 1) {
          const enemy = enemies[i];
          if (distanceBetween(player, enemy) <= player.auraRadius + enemy.radius) {
            applyDamage(enemy, damage, { owner: "player" });
            if (enemy.health <= 0) {
              destroyEnemy(enemy, i);
            }
          }
        }
      }

      function updateMineDeployment(delta) {
        if (player.mineDropChance <= 0 || player.mineRadius <= 0 || player.mineDamage <= 0) return;
        player.mineTimer = (player.mineTimer || 0) - delta;
        if (player.mineTimer > 0) return;
        player.mineTimer = player.mineInterval || 1.2;
        if (Math.random() < player.mineDropChance) {
          state.mines.push({
            x: player.x + rand(-10, 10),
            y: player.y + rand(-10, 10),
            radius: player.mineRadius,
            damage: player.mineDamage,
            timer: player.mineDuration || 5
          });
        }
      }

      function updateShockwave(delta) {
        if (player.shockwaveInterval <= 0 || player.shockwaveRadius <= 0 || player.shockwaveDamage <= 0) return;
        player.shockwaveTimer = (player.shockwaveTimer || 0) - delta;
        if (player.shockwaveTimer > 0) return;
        player.shockwaveTimer = player.shockwaveInterval;
        spawnPulse(player.x, player.y, "#7ca8ff", player.shockwaveRadius);
        for (let i = enemies.length - 1; i >= 0; i -= 1) {
          const enemy = enemies[i];
          if (distanceBetween(player, enemy) <= player.shockwaveRadius + enemy.radius) {
            applyDamage(enemy, player.shockwaveDamage, { owner: "player" });
            if (player.shockwaveSlow > 0) {
              enemy.slowTimer = Math.max(enemy.slowTimer || 0, player.shockwaveSlow);
            }
            if (enemy.health <= 0) {
              destroyEnemy(enemy, i);
            }
          }
        }
      }

      function updateMissiles(delta) {
        if (player.missileInterval <= 0 || player.missileDamage <= 0 || player.missileCount <= 0) return;
        player.missileTimer = (player.missileTimer || 0) - delta;
        if (player.missileTimer > 0) return;
        const target = getNearestEnemy(player.x, player.y, 520);
        if (!target) return;
        player.missileTimer = player.missileInterval;
        const count = Math.max(1, player.missileCount);
        for (let i = 0; i < count; i += 1) {
          const angle = Math.atan2(target.y - player.y, target.x - player.x) + rand(-0.2, 0.2);
          const speed = player.missileSpeed || 520;
          bullets.push({
            owner: "player",
            x: player.x + Math.cos(angle) * (player.radius + 6),
            y: player.y + Math.sin(angle) * (player.radius + 6),
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            radius: 3,
            life: 1.6,
            damage: player.missileDamage,
            tint: "#f6c65f"
          });
        }
      }

      function updateHelpers(delta) {
        if (!player || player.helperCount <= 0) {
          helpers = [];
          return;
        }
        const rawCount = Math.min(6, Math.max(1, player.helperCount));
        const count = rawCount <= 3
          ? rawCount
          : Math.min(5, 3 + Math.floor((rawCount - 3) * 0.5));
        const orbitRadius = player.helperOrbitRadius || 26;
        const orbitSpeed = player.helperOrbitSpeed || 1.4;
        while (helpers.length < count) {
          helpers.push({
            angle: Math.random() * Math.PI * 2,
            fireCooldown: rand(0.1, 0.6)
          });
        }
        if (helpers.length > count) {
          helpers = helpers.slice(0, count);
        }
        const baseAngle = player.angle;
        helpers.forEach((helper, index) => {
          helper.angle += orbitSpeed * delta;
          const offset = helper.angle + (index * Math.PI * 2) / count;
          helper.x = player.x + Math.cos(offset) * orbitRadius;
          helper.y = player.y + Math.sin(offset) * orbitRadius;
          helper.fireCooldown = Math.max(0, helper.fireCooldown - delta);
        });
        const target = getNearestEnemy(player.x, player.y, player.helperRange || 360);
        if (!target) return;
        const baseRate = Math.max(0.4, player.helperFireRate || 1.2);
        const helperPenalty = 1 + Math.max(0, helpers.length - 2) * BALANCE_TUNING.helperRatePenaltyPerExtra;
        const rate = Math.max(0.35, baseRate / helperPenalty);
        helpers.forEach((helper) => {
          if (helper.fireCooldown > 0) return;
          fireHelper(helper, target);
          helper.fireCooldown = 1 / rate;
        });
      }

      function fireHelper(helper, target) {
        if (!helper || !target) return;
        const angle = Math.atan2(target.y - helper.y, target.x - helper.x) + rand(-0.1, 0.1);
        const speed = player.bulletSpeed * 0.9;
        const helperPenalty = 1 / (1 + Math.max(0, helpers.length - 2) * BALANCE_TUNING.helperDamagePenaltyPerExtra);
        const damage = player.damage * (player.helperDamageRatio || 0.3) * helperPenalty;
        bullets.push({
          owner: "player",
          x: helper.x + Math.cos(angle) * 6,
          y: helper.y + Math.sin(angle) * 6,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          radius: 2.6,
          life: 1.4,
          damage,
          tint: "#6ee7b7"
        });
      }

      function getNearestEnemy(x, y, range) {
        let best = null;
        let bestDist = range || Infinity;
        for (let i = 0; i < enemies.length; i += 1) {
          const enemy = enemies[i];
          const distance = Math.hypot(enemy.x - x, enemy.y - y);
          if (distance < bestDist) {
            best = enemy;
            bestDist = distance;
          }
        }
        return best;
      }

      function activateAbility() {
        if (!player || !player.ability) return;
        if (state.mode !== "flight" && state.mode !== "training") return;
        if (!isFeatureUnlocked("ability")) {
          showTip("locked-ability", "Ability locked", getFeatureHint("ability"), { kind: "lock" });
          return;
        }
        if (player.abilityCooldown > 0 || player.abilityTimer > 0) return;
        player.abilityTimer = player.ability.duration || 0;
        player.abilityCooldown = player.ability.cooldown || 0;
        if (player.ability.onStart) {
          player.ability.onStart(player, state);
        }
        if (state.telemetryRun) {
          state.telemetryRun.abilityUses += 1;
        }
        playAudioCue("ability");
      }

      function activateSecondary() {
        if (!player || !player.secondary) return;
        if (state.mode !== "flight" && state.mode !== "training") return;
        if (!isFeatureUnlocked("secondary")) {
          showTip("locked-secondary", "Secondary locked", getFeatureHint("secondary"), { kind: "lock" });
          return;
        }
        if (player.secondaryCooldown > 0) return;
        player.secondaryCooldown = player.secondary.cooldown || 0;
        if (player.secondary.activate) {
          player.secondary.activate(player, state);
        }
        if (state.telemetryRun) {
          state.telemetryRun.secondaryUses += 1;
        }
        playAudioCue("secondary");
      }

      function updateEnemies(delta) {
        const difficulty = getDifficultySettings();
        enemies.forEach((enemy) => {
          enemy.fireCooldown = Math.max(0, enemy.fireCooldown - delta);
          if (enemy.hitFlash > 0) {
            enemy.hitFlash = Math.max(0, enemy.hitFlash - delta);
          }
          if (enemy.invulnerable > 0) {
            enemy.invulnerable = Math.max(0, enemy.invulnerable - delta);
          }
          if (enemy.slowTimer > 0) {
            enemy.slowTimer = Math.max(0, enemy.slowTimer - delta);
            enemy.slowFactor = enemy.slowTimer > 0 ? 0.6 : 1;
          }
          if (enemy.shieldPulseCooldown > 0) {
            enemy.shieldPulseTimer = Math.max(0, enemy.shieldPulseTimer - delta);
          }
          if (enemy.ramCooldown > 0) {
            enemy.ramTimer = Math.max(0, enemy.ramTimer - delta);
          }
          if (enemy.dashCooldown > 0) {
            enemy.dashTimer = Math.max(0, (enemy.dashTimer || 0) - delta);
          }
          if (enemy.miniboss) {
            if (!enemy.phaseActive && enemy.health <= (enemy.phaseThreshold || 0)) {
              enemy.phaseActive = true;
              enemy.phaseTimer = 2.6;
              enemy.invulnerable = 2.2;
              spawnPulse(enemy.x, enemy.y, "#ff9f6b", 210);
              logEvent("Miniboss phase shift detected.");
            }
            if (enemy.phaseActive) {
              enemy.phaseTimer = Math.max(0, (enemy.phaseTimer || 0) - delta);
              if (enemy.phaseTimer <= 0) {
                enemy.phaseActive = false;
              }
            }
            enemy.summonTimer = Math.max(0, (enemy.summonTimer || 0) - delta);
            if (enemy.summonTimer <= 0 && enemies.length < 36) {
              const interceptor = ENEMY_TYPES.find((item) => item.id === "interceptor") || ENEMY_TYPES[0];
              const addCount = 2;
              for (let s = 0; s < addCount; s += 1) {
                const summoned = createEnemy(interceptor, 1.1 + (state.wave || 1) * 0.04, difficulty, true);
                summoned.x = clamp(enemy.x + rand(-70, 70), 30, state.worldWidth - 30);
                summoned.y = clamp(enemy.y + rand(-70, 70), 30, state.worldHeight - 30);
                enemies.push(summoned);
              }
              enemy.summonTimer = enemy.summonCooldown || 9;
              spawnPulse(enemy.x, enemy.y, "#b98cff", 150);
            }
          }
          if (enemy.burstRemaining > 0) {
            enemy.burstTimer = Math.max(0, enemy.burstTimer - delta);
          }

          const target = state.decoy && state.decoy.timer > 0 ? state.decoy : player;
          const dx = target.x - enemy.x;
          const dy = target.y - enemy.y;
          const distance = Math.hypot(dx, dy) || 1;
          const bulletSpeed = enemy.baseBulletSpeed * (state.sectorMod?.enemy?.bulletSpeed || 1);
          const leadTime = Math.min(0.8, distance / bulletSpeed);
          const targetX = target.x + (target.vx || 0) * leadTime;
          const targetY = target.y + (target.vy || 0) * leadTime;
          const aimAngle = Math.atan2(targetY - enemy.y, targetX - enemy.x);
          const turnRate = (enemy.baseTurnRate || 0) * enemy.slowFactor;
          enemy.angle = rotateTowards(enemy.angle, aimAngle, turnRate * delta);

          const accel = enemy.baseAccel * (state.sectorMod?.enemy?.accel || 1) * enemy.slowFactor;
          let ax = 0;
          let ay = 0;
          const moveTarget = getEnemyNavigationTarget(enemy, target, delta, distance);
          const moveDx = moveTarget.x - enemy.x;
          const moveDy = moveTarget.y - enemy.y;
          const moveDistance = Math.hypot(moveDx, moveDy) || 1;
          const toTargetX = moveDx / moveDistance;
          const toTargetY = moveDy / moveDistance;
          const currentSpeed = Math.hypot(enemy.vx, enemy.vy);
          const isDetouring = moveTarget !== target;

          if (isDetouring || distance > enemy.preferredRange) {
            ax += toTargetX * accel;
            ay += toTargetY * accel;
          } else {
            const strafeAngle = aimAngle + Math.PI / 2 * enemy.strafeBias;
            ax += Math.cos(strafeAngle) * accel * 0.85;
            ay += Math.sin(strafeAngle) * accel * 0.85;
          }

          const avoidance = getObstacleAvoidance(enemy, toTargetX, toTargetY);
          ax += avoidance.x * accel * 0.65;
          ay += avoidance.y * accel * 0.65;
          const edgeAvoidance = getBoundaryAvoidance(enemy, toTargetX, toTargetY);
          ax += edgeAvoidance.x * accel * 0.85;
          ay += edgeAvoidance.y * accel * 0.85;
          if ((isDetouring || distance > enemy.preferredRange * 0.7) && currentSpeed < 34) {
            ax += toTargetX * accel * 0.45;
            ay += toTargetY * accel * 0.45;
          }

          enemy.vx += ax * delta;
          enemy.vy += ay * delta;

          const damping = Math.max(0, 1 - 2 * delta);
          enemy.vx *= damping;
          enemy.vy *= damping;

          const maxSpeed = enemy.baseSpeed * (state.sectorMod?.enemy?.speed || 1) * enemy.slowFactor;
          const speed = Math.hypot(enemy.vx, enemy.vy);
          if (speed > maxSpeed) {
            const ratio = maxSpeed / speed;
            enemy.vx *= ratio;
            enemy.vy *= ratio;
          }

          enemy.x += enemy.vx * delta;
          enemy.y += enemy.vy * delta;
          wrapEntity(enemy);
          resolveObstacleCollisions(enemy);
          wrapEntity(enemy);

          if (enemy.dashCooldown > 0 && enemy.dashTimer <= 0 && player) {
            const angleToPlayer = Math.atan2(player.y - enemy.y, player.x - enemy.x);
            const dashPower = (enemy.baseSpeed || 200) * (enemy.dashStrength || 1.8);
            enemy.vx += Math.cos(angleToPlayer) * dashPower;
            enemy.vy += Math.sin(angleToPlayer) * dashPower;
            enemy.dashTimer = enemy.dashCooldown;
            spawnPulse(enemy.x, enemy.y, enemy.miniboss ? "#ff9f6b" : "#b98cff", enemy.miniboss ? 150 : 90);
          }

          if (enemy.shieldPulseCooldown > 0 && enemy.shieldPulseTimer <= 0) {
            const radius = enemy.shieldPulseRadius || 0;
            const amount = enemy.shieldPulseAmount || 0;
            if (amount > 0) {
              enemies.forEach((ally) => {
                if (radius <= 0) {
                  if (!enemy.shieldPulseSelf || ally !== enemy) return;
                } else if (distanceBetween(enemy, ally) > radius) {
                  return;
                }
                ally.shield = Math.min(ally.maxShield, ally.shield + amount);
              });
            }
            spawnPulse(enemy.x, enemy.y, enemy.shieldPulseColor || "#7ca8ff", radius || 120);
            enemy.shieldPulseTimer = enemy.shieldPulseCooldown;
          }

          if (enemy.ramDamage > 0 && player && enemy.ramTimer <= 0) {
            const playerDistance = distanceBetween(enemy, player);
            const ramRange = enemy.ramRange || (enemy.radius + player.radius + 8);
            if (playerDistance <= ramRange) {
              applyDamage(player, enemy.ramDamage, { owner: "enemy" });
              enemy.ramTimer = enemy.ramCooldown || 0;
              enemy.hitFlash = 0.2;
              spawnPulse(enemy.x, enemy.y, "#f06969", 90);
              if (enemy.ramKnockback) {
                const pushAngle = Math.atan2(player.y - enemy.y, player.x - enemy.x);
                player.vx += Math.cos(pushAngle) * enemy.ramKnockback;
                player.vy += Math.sin(pushAngle) * enemy.ramKnockback;
              }
            }
          }

          const fireRate = enemy.baseFireRate * (state.sectorMod?.enemy?.fireRate || 1);
          const inRange = distance < 520;
          if (enemy.burstRemaining > 0) {
            if (enemy.burstTimer <= 0 && inRange) {
              fireEnemy(enemy, enemy.angle);
              enemy.burstRemaining -= 1;
              enemy.burstTimer = enemy.burstInterval || 0.12;
            }
          } else if (enemy.fireCooldown <= 0 && inRange) {
            if (enemy.burstCount > 1) {
              fireEnemy(enemy, enemy.angle);
              enemy.burstRemaining = enemy.burstCount - 1;
              enemy.burstTimer = enemy.burstInterval || 0.12;
              enemy.fireCooldown = enemy.burstCooldown || (1 / fireRate);
            } else {
              fireEnemy(enemy, enemy.angle);
              enemy.fireCooldown = 1 / fireRate;
            }
          }
        });
      }

      function updateBullets(delta) {
        for (let i = bullets.length - 1; i >= 0; i -= 1) {
          const bullet = bullets[i];
          bullet.x += bullet.vx * delta;
          bullet.y += bullet.vy * delta;
          bullet.life -= delta;
          if (isPointInObstacle(bullet, bullet.radius + 2)) {
            bullets.splice(i, 1);
            continue;
          }
          if (bullet.life <= 0 || isOutOfBounds(bullet)) {
            bullets.splice(i, 1);
          }
        }
      }

      function updateDecoy(delta) {
        if (!state.decoy) return;
        state.decoy.timer -= delta;
        if (state.decoy.timer <= 0) {
          state.decoy = null;
        }
      }

      function updateMines(delta) {
        if (!state.mines.length) return;
        for (let i = state.mines.length - 1; i >= 0; i -= 1) {
          const mine = state.mines[i];
          mine.timer -= delta;
          if (mine.timer <= 0) {
            state.mines.splice(i, 1);
            continue;
          }
          for (let j = enemies.length - 1; j >= 0; j -= 1) {
            const enemy = enemies[j];
            if (distanceBetween(mine, enemy) <= mine.radius + enemy.radius) {
              applyDamage(enemy, mine.damage || 40, { owner: "player" });
              spawnExplosion(mine.x, mine.y, "#f6c65f");
              state.mines.splice(i, 1);
              if (enemy.health <= 0) {
                destroyEnemy(enemy, j);
              }
              break;
            }
          }
        }
      }

      function getFieldDropInterval() {
        const wave = Math.max(1, getGlobalWave(state.wave || 1));
        return Math.max(7, FIELD_DROP_INTERVAL - wave * 0.25);
      }

      function spawnFieldDrop() {
        if (fieldDrops.length >= FIELD_DROP_LIMIT) return;
        const def = rollFieldDropDef();
        const point = getFieldDropSpawnPoint();
        fieldDrops.push({
          id: `drop_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          typeId: def.id,
          x: point.x,
          y: point.y,
          radius: 16,
          life: FIELD_DROP_LIFETIME,
          maxLife: FIELD_DROP_LIFETIME,
          phase: rand(0, Math.PI * 2)
        });
      }

      function triggerNovaStrike(def) {
        if (!player || !enemies.length) return;
        const wave = Math.max(1, getGlobalWave(state.wave || 1));
        const damage = Math.round(30 + wave * (def.damageScale || 4));
        const pulseRadius = Math.max(state.worldWidth, state.worldHeight);
        spawnPulse(player.x, player.y, "#f6c65f", pulseRadius);
        for (let i = enemies.length - 1; i >= 0; i -= 1) {
          const enemy = enemies[i];
          applyDamage(enemy, damage, { owner: "player" });
          if (enemy.health <= 0) {
            destroyEnemy(enemy, i);
          }
        }
        logEvent("Nova strike detonated across the sector.");
      }

      function applyFieldDropEffect(drop) {
        if (!drop || !player) return;
        const def = getFieldDropDef(drop.typeId);
        if (!def) return;
        if (def.id === "nova") {
          triggerNovaStrike(def);
          return;
        }
        if (def.id === "hull") {
          const amount = Math.round(player.maxHealth * (def.healRatio || 0.3));
          const before = player.health;
          player.health = Math.min(player.maxHealth, player.health + amount);
          const restored = Math.round(player.health - before);
          spawnPulse(player.x, player.y, "#6ee7b7", 140);
          logEvent(`Hull patch applied: +${restored} hull.`);
          return;
        }
        if (def.id === "shield") {
          const amount = Math.round(player.maxShield * (def.healRatio || 0.35));
          const before = player.shield;
          player.shield = Math.min(player.maxShield, player.shield + amount);
          const restored = Math.round(player.shield - before);
          spawnPulse(player.x, player.y, "#7ca8ff", 140);
          logEvent(`Shield battery charged: +${restored} shield.`);
          return;
        }
        if (def.id === "energy") {
          const amount = Math.round(player.maxEnergy * (def.healRatio || 0.35));
          const before = player.energy;
          player.energy = Math.min(player.maxEnergy, player.energy + amount);
          const restored = Math.round(player.energy - before);
          spawnPulse(player.x, player.y, "#44d2c2", 140);
          logEvent(`Energy cell absorbed: +${restored} energy.`);
          return;
        }
        if (def.id === "invuln") {
          player.invulnerable = Math.max(player.invulnerable, def.duration || 4);
          spawnPulse(player.x, player.y, "#f6c65f", 160);
          logEvent("Phase ward online. Invulnerable.");
          return;
        }
        if (def.id === "damage") {
          player.damageBoostTimer = Math.max(player.damageBoostTimer || 0, def.duration || 6);
          player.damageBoostMultiplier = Math.max(player.damageBoostMultiplier || 1, def.multiplier || 2);
          spawnPulse(player.x, player.y, "#f48b7f", 160);
          logEvent("Overcharge active. Damage doubled.");
          return;
        }
        if (def.id === "speed") {
          player.speedBoostTimer = Math.max(player.speedBoostTimer || 0, def.duration || 6);
          player.speedBoostMultiplier = Math.max(player.speedBoostMultiplier || 1, def.multiplier || 1.3);
          spawnPulse(player.x, player.y, "#6ee7b7", 160);
          logEvent("Afterburners engaged. Speed boosted.");
        }
      }

      function updateFieldDrops(delta) {
        if (state.mode !== "flight" && state.mode !== "training") return;
        state.fieldDropTimer += delta;
        const interval = getFieldDropInterval();
        if (state.fieldDropTimer >= interval) {
          state.fieldDropTimer = state.fieldDropTimer % interval;
          spawnFieldDrop();
        }
        for (let i = fieldDrops.length - 1; i >= 0; i -= 1) {
          const drop = fieldDrops[i];
          drop.life -= delta;
          if (drop.life <= 0) {
            fieldDrops.splice(i, 1);
            continue;
          }
          if (player && distanceBetween(player, drop) <= (drop.radius || 16) + player.radius) {
            applyFieldDropEffect(drop);
            fieldDrops.splice(i, 1);
          }
        }
      }

      function handleCollisions() {
        for (let i = bullets.length - 1; i >= 0; i -= 1) {
          const bullet = bullets[i];
          if (bullet.owner === "player") {
            for (let j = enemies.length - 1; j >= 0; j -= 1) {
              const enemy = enemies[j];
              if (distanceBetween(bullet, enemy) <= bullet.radius + enemy.radius) {
                if (state.telemetryRun) {
                  state.telemetryRun.shotsHit += 1;
                }
                applyDamage(enemy, bullet.damage, bullet);
                if (bullet.splashRadius && bullet.splashRadius > 0) {
                  applySplashDamage(enemy, bullet);
                }
                triggerArcChain(enemy, bullet);
                if (enemy.health <= 0) {
                  destroyEnemy(enemy, j);
                }
                if (bullet.pierce && bullet.pierce > 0) {
                  bullet.pierce -= 1;
                } else {
                  bullets.splice(i, 1);
                }
                break;
              }
            }
          } else if (bullet.owner === "enemy") {
            if (player.invulnerable <= 0 && distanceBetween(bullet, player) <= bullet.radius + player.radius) {
              applyDamage(player, bullet.damage, bullet);
              bullets.splice(i, 1);
              if (player.health <= 0) {
                triggerGameOver();
              }
            }
          }
        }
      }

      function applySplashDamage(primary, bullet) {
        const radius = bullet.splashRadius || 0;
        if (!radius) return;
        const splashDamage = bullet.damage * (bullet.splashDamage || 0.6);
        for (let j = enemies.length - 1; j >= 0; j -= 1) {
          const enemy = enemies[j];
          if (enemy === primary) continue;
          if (distanceBetween(primary, enemy) <= radius) {
            applyDamage(enemy, splashDamage, bullet);
            if (enemy.health <= 0) {
              destroyEnemy(enemy, j);
            }
          }
        }
      }

      function findArcTarget(primary, radius) {
        let target = null;
        let closest = radius;
        for (let i = 0; i < enemies.length; i += 1) {
          const enemy = enemies[i];
          if (enemy === primary) continue;
          const distance = distanceBetween(primary, enemy);
          if (distance <= closest) {
            closest = distance;
            target = enemy;
          }
        }
        return target;
      }

      function triggerArcChain(primary, bullet) {
        if (!bullet || !bullet.arcDamage || !bullet.arcRadius || !bullet.arcChains) return;
        if (bullet.arcRequiresSlow && (!primary.slowTimer || primary.slowTimer <= 0)) return;
        const target = findArcTarget(primary, bullet.arcRadius);
        if (!target) return;
        bullet.arcChains -= 1;
        const chainIndex = bullet.arcChainIndex || 0;
        bullet.arcChainIndex = chainIndex + 1;
        const arcFalloff = 1 / (1 + chainIndex * BALANCE_TUNING.arcChainFalloffPerJump);
        const arcDamage = bullet.damage * bullet.arcDamage * arcFalloff;
        applyDamage(target, arcDamage, { owner: "player" });
        spawnPulse(target.x, target.y, "#7ca8ff", Math.min(160, bullet.arcRadius));
        const targetIndex = enemies.indexOf(target);
        if (target.health <= 0 && targetIndex >= 0) {
          destroyEnemy(target, targetIndex);
        }
      }

      function updateContracts(delta) {
        if (!state.contracts.length || state.training) return;
        let changed = false;
        const now = performance.now();
        state.contracts.forEach((contract) => {
          if (contract.complete) return;
          if (contract.type === "noDamage") {
            const prev = contract.progress;
            contract.progress = Math.min(contract.target, contract.progress + delta);
            if (Math.floor(contract.progress) !== Math.floor(prev)) {
              changed = true;
            }
            if (contract.progress >= contract.target) {
              completeContract(contract);
            }
          }
        });
        if (changed && now - state.lastContractRender > 250) {
          state.lastContractRender = now;
          renderContracts();
        }
      }

      function resetNoDamageContracts() {
        if (!state.contracts.length) return;
        let updated = false;
        state.contracts.forEach((contract) => {
          if (contract.type === "noDamage" && !contract.complete) {
            contract.progress = 0;
            updated = true;
          }
        });
        if (updated) {
          renderContracts();
        }
      }

      function completeContract(contract) {
        contract.complete = true;
        contract.progress = contract.target;
        state.credits += contract.reward.credits;
        awardXp(contract.reward.xp, { deferPersistence: true });
        progress.factions[contract.factionId] = (progress.factions[contract.factionId] || 0) + contract.reward.rep;
        queueProgressSave();
        renderContracts();
        queueSidebarRefresh();
        logEvent(`Contract complete: ${contract.title}.`);
      }

      function updateKillContracts(enemy) {
        if (!state.contracts.length) return;
        let updated = false;
        state.contracts.forEach((contract) => {
          if (contract.complete) return;
          if (contract.type === "kills") {
            contract.progress += 1;
            updated = true;
            if (contract.progress >= contract.target) {
              completeContract(contract);
            }
          }
          if (contract.type === "elite" && (enemy.elite || enemy.id === "dreadnought" || enemy.id === "ace")) {
            contract.progress = contract.target;
            updated = true;
            completeContract(contract);
          }
        });
        if (updated) {
          queueContractsRefresh();
        }
      }

      function maybeDropLoot(enemy) {
        const isBoss = enemy.id === "dreadnought";
        const isElite = enemy.elite || enemy.id === "ace";
        const dropPity = ensureDropPityState();
        ensurePremiumState();
        progress.premiumDropPity = Number.isFinite(progress.premiumDropPity)
          ? Math.max(0, Math.floor(progress.premiumDropPity))
          : 0;
        const drops = [];
        const keyChance = isBoss ? 1 : isElite ? 0.38 : 0.14;
        const pityKeyReady = isElite && !isBoss && dropPity.key >= DROP_PITY_THRESHOLDS.key;
        if (pityKeyReady || Math.random() < keyChance) {
          progress.salvageKeys += 1;
          dropPity.key = 0;
          if (pityKeyReady) {
            logEvent("Pity drop: salvage key recovered from elite wreckage.");
          } else {
            logEvent("Salvage key recovered from the wreckage.");
          }
          drops.push({
            label: "Salvage Key +1",
            tier: isBoss ? "epic" : isElite ? "rare" : "uncommon"
          });
        } else if (isElite) {
          dropPity.key += 1;
        }
        const blueprintChance = isBoss ? 1 : isElite ? 0.32 : 0.1;
        const pityBlueprintReady = isElite && !isBoss && dropPity.blueprint >= DROP_PITY_THRESHOLDS.blueprint;
        if (pityBlueprintReady || Math.random() < blueprintChance) {
          progress.blueprints += 1;
          dropPity.blueprint = 0;
          if (pityBlueprintReady) {
            logEvent("Pity drop: blueprint recovered from elite debris.");
          } else {
            logEvent("Blueprint recovered from debris.");
          }
          drops.push({
            label: "Blueprint +1",
            tier: isBoss ? "epic" : isElite ? "rare" : "uncommon"
          });
        } else if (isElite) {
          dropPity.blueprint += 1;
        }

        if ((isElite || isBoss) && state.runPremiumDrops < PREMIUM_DROP_RUN_CAP) {
          const pityReady = !isBoss && progress.premiumDropPity >= PREMIUM_DROP_PITY_THRESHOLD;
          const routeBonus = state.routeBonus && state.routeBonus.id === "prospector" ? 0.1 : 0;
          const dropChance = isBoss ? 1 : Math.max(0.18, 0.34 - state.runPremiumDrops * 0.05 + routeBonus);
          if (pityReady || Math.random() < dropChance) {
            const premiumAmount = isBoss ? randInt(4, 6) : randInt(1, 2);
            progress.premiumCurrency += premiumAmount;
            progress.premiumDropPity = 0;
            state.runPremiumDrops += 1;
            const capReached = state.runPremiumDrops >= PREMIUM_DROP_RUN_CAP;
            if (pityReady) {
              logEvent(`${PREMIUM_CURRENCY_LABEL} pity cache recovered (+${premiumAmount}).`);
            } else {
              logEvent(capReached
                ? `${PREMIUM_CURRENCY_LABEL} cache recovered (+${premiumAmount}). Run drop limit reached.`
                : `${PREMIUM_CURRENCY_LABEL} cache recovered (+${premiumAmount}).`);
            }
            drops.push({
              label: `${PREMIUM_CURRENCY_LABEL} +${premiumAmount}`,
              tier: isBoss ? "legendary" : "epic"
            });
          } else if (isElite) {
            progress.premiumDropPity += 1;
          }
        }

        const attachmentChance = isBoss ? 0.8 : isElite ? 0.4 : 0.14;
        if (Math.random() < attachmentChance) {
          const attachmentTier = rollItemTier(null, isBoss
            ? [1.6, 2.4, 2.1, 1.5, 0.9]
            : isElite ? [3.8, 3, 2.4, 1.4, 0.7] : null);
          const attachment = createAttachmentItem({ tier: attachmentTier });
          if (attachment) {
            addInventoryItem(attachment, { notify: true });
            logEvent(`Recovered attachment: ${attachment.name}.`);
            drops.push({
              label: `Attachment: ${attachment.name}`,
              tier: attachment.tier
            });
          }
        }

        const weaponChance = isBoss ? 0.6 : isElite ? 0.28 : 0.09;
        if (Math.random() < weaponChance) {
          const weaponTier = rollItemTier(null, isBoss
            ? [1.2, 2.2, 2.2, 1.6, 1]
            : isElite ? [3.5, 3, 2.4, 1.4, 0.8] : null);
          const weaponDrop = createWeaponItem({ tier: weaponTier });
          if (weaponDrop) {
            addInventoryItem(weaponDrop, { notify: true });
            logEvent(`Recovered weapon: ${weaponDrop.name}.`);
            drops.push({
              label: `Weapon: ${weaponDrop.name}`,
              tier: weaponDrop.tier
            });
          }
        }
        if (drops.length) {
          spawnLootBursts(enemy.x, enemy.y, drops);
        }
        checkProgressionUnlocks();
      }

      function ensureDropPityState() {
        progress.dropPity = progress.dropPity || { key: 0, blueprint: 0 };
        progress.dropPity.key = Number.isFinite(progress.dropPity.key) ? Math.max(0, progress.dropPity.key) : 0;
        progress.dropPity.blueprint = Number.isFinite(progress.dropPity.blueprint) ? Math.max(0, progress.dropPity.blueprint) : 0;
        return progress.dropPity;
      }

      function generatePart(preferred) {
        return createAttachmentItem({ tier: preferred });
      }

      function checkWaveStatus() {
        if (state.mode !== "flight" && state.mode !== "training") return;
        const objective = state.waveObjective;
        const objectiveComplete = objective
          ? (objective.complete || (objective.id === "eliminate" && enemies.length === 0))
          : enemies.length === 0;
        if (!objectiveComplete) return;
        if (objective && objective.id !== "eliminate" && enemies.length > 0) {
          enemies = [];
        }
        if (enemies.length > 0) return;
        if (state.training) {
          state.wave += 1;
          state.waveStart = performance.now();
          updateSector();
          restoreBetweenWaves();
          checkProgressionUnlocks();
          spawnWave(state.wave);
          logEvent(`Training wave ${state.wave} online.`);
          return;
        }
        const globalWave = getGlobalWave(state.wave);
        progress.bestWave = Math.max(progress.bestWave, globalWave);
        if (MILESTONE_WAVES.includes(state.wave) && !state.milestoneRewardsClaimed[state.wave]) {
          state.milestoneRewardsClaimed[state.wave] = true;
          grantMilestoneRewards(state.wave);
          state.choiceEvent = buildChoiceEvent(state.wave);
          setMode("paused");
          setOverlay("choice-event");
          saveProgress();
          return;
        }
        saveProgress();
        if (isCampaignMode() && state.wave >= LEVEL_WAVES) {
          completeLevel();
          return;
        }
        state.wave += 1;
        state.waveStart = performance.now();
        updateSector();
        restoreBetweenWaves();
        checkProgressionUnlocks();
        if (isFeatureUnlocked("upgrades")) {
          state.upgradeRerolls = 0;
          state.upgradeOptions = rollUpgrades();
          setMode("upgrade");
          setOverlay("upgrade");
          logEvent("Wave cleared. Choose a field upgrade.");
        } else {
          spawnWave(state.wave);
        }
      }

      function grantMilestoneRewards(wave) {
        const credits = 160 + wave * 42;
        const blueprints = wave >= 9 ? 2 : wave >= 6 ? 1 : 0;
        const salvage = wave >= 9 ? 2 : 1;
        const premium = wave >= 9 ? 3 : wave >= 6 ? 2 : 1;
        progress.bankedCredits += credits;
        progress.salvageKeys += salvage;
        progress.blueprints += blueprints;
        progress.premiumCurrency += premium;
        const blueprintText = blueprints > 0 ? `, +${blueprints} blueprint${blueprints === 1 ? "" : "s"}` : "";
        showTip(null, `Milestone ${wave}`, `+${credits} credits, +${salvage} key${salvage === 1 ? "" : "s"}${blueprintText}, +${premium} ${PREMIUM_CURRENCY_LABEL}.`, {
          kind: "reward",
          repeatable: true,
          duration: 6200
        });
        logEvent(`Milestone chest secured at wave ${wave}.`);
      }

      function buildChoiceEvent(wave) {
        const options = [
          {
            id: "aggression",
            title: "Aggression Route",
            desc: "Harder waves for three sectors, larger payouts.",
            apply: () => {
              state.routeBonus = { id: "aggression", wavesRemaining: 3 };
              state.credits += 120 + wave * 30;
            }
          },
          {
            id: "fortify",
            title: "Fortify Route",
            desc: "Safer waves for three sectors, stronger defenses now.",
            apply: () => {
              state.routeBonus = { id: "fortify", wavesRemaining: 3 };
              if (player) {
                applyHullHeal(player, player.maxHealth * 0.35);
                applyShieldHeal(player, player.maxShield * 0.45);
                player.damageReduction = Math.min(0.82, (player.damageReduction || 0) + 0.08);
              }
            }
          },
          {
            id: "prospector",
            title: "Prospector Route",
            desc: "Higher resource focus and premium pity acceleration.",
            apply: () => {
              state.routeBonus = { id: "prospector", wavesRemaining: 3 };
              progress.salvageKeys += 1;
              progress.premiumDropPity = Math.min(PREMIUM_DROP_PITY_THRESHOLD - 1, (progress.premiumDropPity || 0) + 2);
            }
          }
        ];
        return {
          wave,
          options
        };
      }

      function applyChoiceEvent(choiceId) {
        const event = state.choiceEvent;
        if (!event || !Array.isArray(event.options)) return;
        const option = event.options.find((item) => item.id === choiceId);
        if (!option) return;
        option.apply();
        state.choiceEvent = null;
        saveProgress();
        renderPremiumShop();
        queueSidebarRefresh();
        setMode("flight");
        hideOverlay();
        if (state.routeBonus) {
          logEvent(`Route selected: ${option.title}.`);
        }
        pushTelemetryEvent("route_selected", {
          route: option.id,
          wave: state.wave
        });
        state.wave += 1;
        state.waveStart = performance.now();
        updateSector();
        restoreBetweenWaves();
        spawnWave(state.wave);
      }

      function updateFrontierSpawner(delta) {
        if (!state.frontier || !state.frontier.active) return;
        state.frontier.spawnTimer += delta;
        const tier = state.frontier.tier || 1;
        const interval = Math.max(10, 18 - tier * 1.2);
        const shouldForce = enemies.length === 0 && state.frontier.spawnTimer >= 2.5;
        if (state.frontier.spawnTimer >= interval || shouldForce) {
          state.frontier.spawnTimer = 0;
          state.wave += 1;
          state.waveStart = performance.now();
          updateSector();
          spawnWave(state.wave);
          if (state.wave % 3 === 0) {
            logEvent(`Frontier spike: threat level ${state.wave}.`);
          }
        }
      }

      function restoreBetweenWaves() {
        applyHullHeal(player, player.maxHealth * 0.25);
        applyShieldHeal(player, player.maxShield - player.shield);
        player.energy = player.maxEnergy;
      }

      function getUpgradeWeight(tier, luck) {
        const meta = getTierMeta(tier);
        if (!luck) return meta.weight;
        const tierIndex = Math.max(0, TIER_ORDER.indexOf(tier));
        const normalized = tierIndex / Math.max(1, TIER_ORDER.length - 1);
        const bias = 1 + luck * (normalized * 1.6 - 0.4);
        return Math.max(0.12, meta.weight * bias);
      }

      function getUpgradeRerollCost() {
        const rerolls = Math.max(0, state.upgradeRerolls || 0);
        return Math.round(UPGRADE_REROLL_COST * Math.pow(UPGRADE_REROLL_SCALE, rerolls));
      }

      function getSkillId(upgrade) {
        return upgrade.skillId || upgrade.id;
      }

      function isSkillUpgrade(upgrade) {
        return upgrade && upgrade.kind === "skill";
      }

      function getSkillTier(level) {
        const index = clamp((level || 1) - 1, 0, TIER_ORDER.length - 1);
        return TIER_ORDER[index] || "common";
      }

      function getUpgradeTier(upgrade, level) {
        if (!upgrade) return "common";
        if (isSkillUpgrade(upgrade)) {
          const resolvedLevel = Number.isFinite(level) ? level : 1;
          return getSkillTier(resolvedLevel);
        }
        return upgrade.tier || "common";
      }

      function canSelectSkillUpgrade(upgrade) {
        if (!isSkillUpgrade(upgrade)) return true;
        const skillId = getSkillId(upgrade);
        return state.skillSlots.includes(skillId) || state.skillSlots.length < SKILL_LIMIT;
      }

      function rollUpgrades() {
        const available = FIELD_UPGRADES.filter((upgrade) => {
          const stack = state.upgradeStacks[upgrade.id] || 0;
          if (stack >= (upgrade.maxStacks || 99)) return false;
          return canSelectSkillUpgrade(upgrade);
        });
        const picks = [];
        const pool = [...available];
        const luck = player ? player.upgradeLuck || 0 : 0;
        while (picks.length < Math.min(3, pool.length)) {
          const weights = pool.map((upgrade) => {
            const stack = state.upgradeStacks[upgrade.id] || 0;
            const tier = getUpgradeTier(upgrade, stack + 1);
            return getUpgradeWeight(tier, luck);
          });
          const selected = pickWeighted(pool, weights);
          picks.push(selected);
          pool.splice(pool.indexOf(selected), 1);
        }
        return picks;
      }

      function acceptUpgrade(id) {
        if (state.mode !== "upgrade") return;
        if (!isFeatureUnlocked("upgrades")) return;
        if (id) {
          const upgrade = FIELD_UPGRADES.find((item) => item.id === id);
          if (upgrade) {
            const wasSkill = isSkillUpgrade(upgrade);
            const skillId = wasSkill ? getSkillId(upgrade) : null;
            const isNewSkill = wasSkill && !state.skillSlots.includes(skillId);
            if (isNewSkill && state.skillSlots.length >= SKILL_LIMIT) {
              logEvent("Skill slots are full. Choose a different upgrade.");
              return;
            }
            const previousHealth = player.health;
            const previousShield = player.shield;
            const stack = state.upgradeStacks[upgrade.id] || 0;
            const nextLevel = stack + 1;
            upgrade.apply(player, nextLevel);
            state.upgradeStacks[upgrade.id] = nextLevel;
            if (isNewSkill) {
              state.skillSlots.push(skillId);
              logEvent(`Skill system online: ${upgrade.name}.`);
            } else if (wasSkill) {
              logEvent(`Skill upgraded: ${upgrade.name}.`);
            } else {
              logEvent(`Upgrade acquired: ${upgrade.name}.`);
            }
            const healthGain = player.health - previousHealth;
            const shieldGain = player.shield - previousShield;
            if (healthGain > 0) {
              spawnDamageNumber(player.x, player.y - player.radius - 12, healthGain, { color: "#6ee7b7", prefix: "+" });
            }
            if (shieldGain > 0) {
              spawnDamageNumber(player.x, player.y - player.radius - 22, shieldGain, { color: "#57e0ff", prefix: "+" });
            }
            playAudioCue("upgrade");
          }
        } else {
          logEvent("Upgrade skipped. Launching next wave.");
        }
        hideOverlay();
        setMode("flight");
        spawnWave(state.wave);
      }

      function rerollUpgrades() {
        const rerollCost = getUpgradeRerollCost();
        if (state.credits < rerollCost) {
          logEvent("Not enough credits to reroll upgrades.");
          return;
        }
        state.credits -= rerollCost;
        state.upgradeRerolls += 1;
        state.upgradeOptions = rollUpgrades();
        setOverlay("upgrade");
      }

      function firePlayer() {
        const playerMod = state.sectorMod ? state.sectorMod.player : {};
        const energyCost = player.energyCost * (playerMod.energyCost || 1);
        player.fireCooldown = 1 / player.fireRate;
        player.energy = Math.max(0, player.energy - energyCost);
        if (state.telemetryRun) {
          state.telemetryRun.shotsFired += 1;
        }
        const barrageEvery = player.barrageEvery || 0;
        if (barrageEvery > 0) {
          player.barrageCounter = (player.barrageCounter || 0) + 1;
        }
        const isBarrage = barrageEvery > 0 && player.barrageCounter % barrageEvery === 0;
        const rawBonusProjectiles = isBarrage ? (player.barrageProjectiles || 0) : 0;
        const softCap = BALANCE_TUNING.barrageBonusSoftCap;
        const bonusProjectiles = rawBonusProjectiles <= softCap
          ? rawBonusProjectiles
          : softCap + Math.floor((rawBonusProjectiles - softCap) * BALANCE_TUNING.barrageBonusSoftScale);
        const count = Math.max(1, player.projectiles + bonusProjectiles);
        const spread = (count > 1 ? player.spread : 0) * (playerMod.spreadMult || 1);
        const baseAngle = player.angle - spread * (count - 1) * 0.5;
        const rawBarrageMultiplier = isBarrage ? (player.barrageBonusDamage || 1) : 1;
        const barrageMultiplier = 1 + (rawBarrageMultiplier - 1) * BALANCE_TUNING.barrageDamageSoftScale;
        const damageMultiplier = player.damageBoostTimer > 0 ? (player.damageBoostMultiplier || 1) : 1;
        const splashRadius = isBarrage
          ? Math.max(player.splashRadius || 0, player.barrageSplashRadius || 0)
          : (player.splashRadius || 0);
        const splashDamage = isBarrage
          ? Math.max(player.splashDamage || 0.6, player.barrageSplashDamage || 0.6)
          : (player.splashDamage || 0.6);
        const pierce = (player.pierce || 0) + (isBarrage ? (player.barragePierce || 0) : 0);
        const tint = isBarrage ? "#f6c65f" : (player.arcDamage ? "#7ca8ff" : null);
        const arcChains = Math.max(0, Math.min(4, player.arcChains || 0));
        for (let i = 0; i < count; i += 1) {
          const angle = baseAngle + spread * i;
          const crit = Math.random() < player.critChance;
          const damage = player.damage * barrageMultiplier * damageMultiplier * (crit ? player.critMultiplier : 1);
          const speed = player.bulletSpeed;
          bullets.push({
            owner: "player",
            x: player.x + Math.cos(angle) * (player.radius + 4),
            y: player.y + Math.sin(angle) * (player.radius + 4),
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            radius: 3,
            life: 1.4,
            damage,
            crit,
            slow: player.slowChance > 0 && Math.random() < player.slowChance,
            splashRadius,
            splashDamage,
            pierce,
            arcDamage: player.arcDamage || 0,
            arcRadius: player.arcRadius || 0,
            arcChains,
            arcChainIndex: 0,
            arcRequiresSlow: !!player.arcRequiresSlow,
            isBarrage,
            tint
          });
        }
        if (player.echoChance > 0 && Math.random() < player.echoChance) {
          const echoTint = "#9aa7ff";
          const echoDamage = player.echoDamage || 0.55;
          for (let i = 0; i < count; i += 1) {
            const angle = baseAngle + spread * i;
            const damage = player.damage * barrageMultiplier * damageMultiplier * echoDamage;
            const speed = player.bulletSpeed * 0.95;
            bullets.push({
              owner: "player",
              x: player.x + Math.cos(angle) * (player.radius + 4),
              y: player.y + Math.sin(angle) * (player.radius + 4),
              vx: Math.cos(angle) * speed,
              vy: Math.sin(angle) * speed,
              radius: 3,
              life: 1.2,
              damage,
              crit: false,
              slow: false,
              splashRadius,
              splashDamage,
              pierce,
              arcDamage: player.arcDamage || 0,
              arcRadius: player.arcRadius || 0,
              arcChains,
              arcChainIndex: 0,
              arcRequiresSlow: !!player.arcRequiresSlow,
              isBarrage: false,
              tint: echoTint
            });
          }
        }
        if (player.blackHoleChance > 0 && Math.random() < player.blackHoleChance) {
          const distance = 120 + (player.blackHoleRadius || 0) * 0.35;
          const holeX = clamp(player.x + Math.cos(player.angle) * distance, 0, state.worldWidth);
          const holeY = clamp(player.y + Math.sin(player.angle) * distance, 0, state.worldHeight);
          spawnBlackHole(holeX, holeY, {
            radius: player.blackHoleRadius || 120,
            duration: player.blackHoleDuration || 1.6,
            force: player.blackHoleForce || 360,
            damage: player.blackHoleDamage || 0
          });
        }
        playAudioCue("shot");
      }

      function fireEnemy(enemy, angle) {
        const accuracy = state.enemyAccuracyMod || 1;
        const jitter = (1 - accuracy) * 0.6;
        const bulletSpeed = enemy.baseBulletSpeed * (state.sectorMod?.enemy?.bulletSpeed || 1);
        const damage = enemy.baseDamage * (state.sectorMod?.enemy?.damage || 1);
        const pattern = enemy.pattern || "single";
        const spreadCount = enemy.spreadCount || (pattern === "spread" ? 5 : 1);
        const spreadAngle = Number.isFinite(enemy.spreadAngle)
          ? enemy.spreadAngle
          : (pattern === "spread" ? 0.18 : 0);
        const spread = spreadCount > 1 ? spreadAngle : 0;
        const bulletRadius = enemy.bulletRadius || 3;
        const bulletLife = enemy.bulletLife || 1.6;
        const bulletTint = enemy.bulletTint || null;
        const baseAngle = angle - spread * (spreadCount - 1) * 0.5;
        for (let i = 0; i < spreadCount; i += 1) {
          const shotAngle = baseAngle + spread * i + rand(-jitter, jitter);
          bullets.push({
            owner: "enemy",
            x: enemy.x + Math.cos(shotAngle) * (enemy.radius + 4),
            y: enemy.y + Math.sin(shotAngle) * (enemy.radius + 4),
            vx: Math.cos(shotAngle) * bulletSpeed,
            vy: Math.sin(shotAngle) * bulletSpeed,
            radius: bulletRadius,
            life: bulletLife,
            damage,
            slowPlayer: !!enemy.bulletSlowPlayer,
            slowDuration: enemy.bulletSlowDuration || 1.4,
            tint: bulletTint
          });
        }
      }

      function applyDamage(target, amount, bullet) {
        if (target === player && player.invulnerable > 0) {
          return;
        }
        if (target !== player && target && target.invulnerable > 0) {
          return;
        }
        const hadShield = target.shield > 0;
        const mitigation = target.damageReduction || 0;
        let finalDamage = amount * (1 - mitigation);
        let shieldDamage = 0;
        let hullDamage = 0;
        if (target.shield > 0) {
          const absorbed = Math.min(target.shield, finalDamage);
          target.shield -= absorbed;
          finalDamage -= absorbed;
          shieldDamage = absorbed;
          if (progress.settings.hitFlash) {
            target.hitFlash = 0.15;
          }
        }
        if (finalDamage > 0) {
          target.health -= finalDamage;
          hullDamage = finalDamage;
          if (progress.settings.hitFlash) {
            target.hitFlash = 0.2;
          }
          if (target === player) {
            state.lastHullHitAt = performance.now();
            resetNoDamageContracts();
          }
        }
        if (target.health < 0) {
          target.health = 0;
        }
        if (shieldDamage > 0) {
          spawnDamageNumber(target.x, target.y - target.radius - 12, shieldDamage, { color: "#57e0ff", prefix: "-" });
        }
        if (hullDamage > 0) {
          spawnDamageNumber(target.x, target.y - target.radius - 22, hullDamage, { color: "#f06969", prefix: "-" });
        }
        if (state.telemetryRun) {
          if (target === player) {
            state.telemetryRun.damageTaken += shieldDamage + hullDamage;
          } else if (bullet && bullet.owner === "player") {
            state.telemetryRun.damageDealt += shieldDamage + hullDamage;
          }
        }
        if (target === player) {
          player.shieldCooldown = 1.6;
          if (shieldDamage > 0 || hullDamage > 0) {
            playAudioCue("player-hit");
          }
          if (hadShield && target.shield <= 0) {
            triggerAegisMatrix();
          }
          if (bullet && bullet.slowPlayer) {
            const slowDuration = bullet.slowDuration || 1.4;
            player.slowTimer = Math.max(player.slowTimer || 0, slowDuration);
          }
        }
        if (bullet && bullet.slow && target !== player) {
          target.slowTimer = player.slowDuration;
        }
      }

      function damageEnemy(enemy, amount, options = {}) {
        if (!enemy || amount <= 0) return false;
        applyDamage(enemy, amount, { owner: "player" });
        if (options.slowDuration) {
          enemy.slowTimer = Math.max(enemy.slowTimer || 0, options.slowDuration);
        }
        const index = enemies.indexOf(enemy);
        if (enemy.health <= 0 && index >= 0) {
          destroyEnemy(enemy, index);
          return true;
        }
        return false;
      }

      function applyHullHeal(target, amount) {
        if (!target || amount <= 0) return;
        const before = target.health;
        target.health = Math.min(target.maxHealth, target.health + amount);
        const delta = target.health - before;
        if (delta > 0) {
          spawnDamageNumber(target.x, target.y - target.radius - 12, delta, { color: "#6ee7b7", prefix: "+" });
        }
      }

      function applyShieldHeal(target, amount) {
        if (!target || amount <= 0) return;
        const before = target.shield;
        target.shield = Math.min(target.maxShield, target.shield + amount);
        const delta = target.shield - before;
        if (delta > 0) {
          spawnDamageNumber(target.x, target.y - target.radius - 18, delta, { color: "#57e0ff", prefix: "+" });
        }
      }

      function triggerAegisMatrix() {
        if (!player || !player.aegisCooldown) return;
        const now = performance.now();
        if (player.aegisReadyAt && now < player.aegisReadyAt) return;
        player.aegisReadyAt = now + player.aegisCooldown * 1000;
        if (player.aegisShieldRestore > 0) {
          const targetShield = player.maxShield * player.aegisShieldRestore;
          applyShieldHeal(player, targetShield - player.shield);
        }
        player.invulnerable = Math.max(player.invulnerable, 1.4);
        if (player.aegisPulseRadius > 0) {
          spawnPulse(player.x, player.y, "#f6c65f", player.aegisPulseRadius);
        }
        if (player.aegisPulseDamage > 0 || player.aegisPulseSlow > 0) {
          for (let i = enemies.length - 1; i >= 0; i -= 1) {
            const enemy = enemies[i];
            if (distanceBetween(player, enemy) <= player.aegisPulseRadius) {
              if (player.aegisPulseDamage > 0) {
                applyDamage(enemy, player.aegisPulseDamage, { owner: "player" });
              }
              if (player.aegisPulseSlow > 0) {
                enemy.slowTimer = Math.max(enemy.slowTimer, player.aegisPulseSlow);
              }
              if (enemy.health <= 0) {
                destroyEnemy(enemy, i);
              }
            }
          }
        }
        logEvent("Aegis Matrix surge activated.");
      }

      function destroyEnemy(enemy, index) {
        enemies.splice(index, 1);
        state.score += enemy.score;
        state.kills += 1;
        if (state.waveObjective && state.waveObjective.id === "elite-hunt" && (enemy.elite || enemy.miniboss || enemy.id === "dreadnought" || enemy.id === "ace")) {
          state.waveObjective.progress = Math.min(
            state.waveObjective.target || 0,
            (state.waveObjective.progress || 0) + 1
          );
        }
        if (state.telemetryRun) {
          state.telemetryRun.kills += 1;
          if (enemy.elite || enemy.id === "dreadnought" || enemy.id === "ace" || enemy.miniboss) {
            state.telemetryRun.eliteKills += 1;
          }
        }
        spawnExplosion(enemy.x, enemy.y, enemy.color);
        if (!state.training) {
          const difficulty = getDifficultySettings();
          const waveScale = Math.max(1, getGlobalWave(state.wave || 1));
          const waveCreditScale = 1 + Math.max(0, waveScale - 1) * 0.025;
          const creditsGain = Math.round(enemy.credits * (1 + player.salvageBonus) * difficulty.reward * waveCreditScale);
          const xpGain = Math.round((10 + waveScale * 2) * (1 + player.xpBonus) * difficulty.reward);
          state.credits += creditsGain;
          progress.totalKills += 1;
          awardXp(xpGain, { deferPersistence: true });
          updateKillContracts(enemy);
          maybeDropLoot(enemy);
          queueProgressSave();
          queueSidebarRefresh();
        }
        playAudioCue("enemy-down", { elite: !!enemy.elite || enemy.id === "dreadnought" || enemy.id === "ace" });
        if (player.healOnKill > 0) {
          applyHullHeal(player, player.maxHealth * player.healOnKill);
        }
        if (player.energyOnKill > 0) {
          player.energy = Math.min(player.maxEnergy, player.energy + player.energyOnKill);
        }
      }

      function awardXp(amount, options = {}) {
        progress.xp += amount;
        let leveled = false;
        while (progress.xp >= xpToNext(progress.rank)) {
          progress.xp -= xpToNext(progress.rank);
          progress.rank += 1;
          progress.techPoints += 1;
          leveled = true;
        }
        if (leveled) {
          logEvent(`Rank up. Pilot rank is now ${progress.rank}.`);
          renderHangar();
        }
        checkProgressionUnlocks();
        if (options.deferPersistence) {
          queueProgressSave();
        } else {
          saveProgress();
        }
      }

      function getRunSummary() {
        const durationSec = state.runStart ? Math.max(1, (performance.now() - state.runStart) / 1000) : 0;
        const difficulty = DIFFICULTY_SETTINGS[state.difficulty] || DIFFICULTY_SETTINGS.normal;
        const rewardMultiplier = getDifficultySettings().reward;
        const globalWave = getGlobalWave(state.wave);
        const level = isCampaignMode() ? Math.max(1, state.level || progress.campaignLevel || 1) : 1;
        return {
          wave: Math.max(1, state.wave),
          globalWave,
          waveDisplay: getWaveDisplay(state.wave),
          level,
          kills: state.kills,
          score: Math.round(state.score),
          credits: Math.round(state.credits),
          durationSec,
          difficultyLabel: difficulty.label,
          rewardMultiplier
        };
      }

      function buildRunAnalyticsEntry(summary, telemetry, reason, difficultyLabel) {
        const safeSummary = summary || getRunSummary();
        const safeTelemetry = telemetry || {};
        return {
          ts: Date.now(),
          reason: reason || "unknown",
          mode: isFrontierMode() ? "Frontier" : "Arcade",
          difficulty: difficultyLabel || (DIFFICULTY_SETTINGS[state.difficulty] || DIFFICULTY_SETTINGS.normal).label,
          ship: state.runLoadout?.ship || player?.ship?.name || "Unknown",
          weapon: state.runLoadout?.weapon || player?.weapon?.name || "Unknown",
          waveDisplay: safeSummary.waveDisplay || getWaveDisplay(safeSummary.wave || state.wave),
          globalWave: safeSummary.globalWave || getGlobalWave(safeSummary.wave || state.wave),
          kills: Math.round(safeSummary.kills || 0),
          score: Math.round(safeSummary.score || 0),
          credits: Math.round(safeSummary.credits || 0),
          durationSec: Math.round(safeSummary.durationSec || 0),
          accuracy: safeTelemetry.accuracy || 0,
          damageDealt: Math.round(safeTelemetry.damageDealt || 0),
          damageTaken: Math.round(safeTelemetry.damageTaken || 0),
          abilityUses: Math.round(safeTelemetry.abilityUses || 0),
          secondaryUses: Math.round(safeTelemetry.secondaryUses || 0),
          objectiveCompletions: Math.round(safeTelemetry.objectiveCompletions || 0),
          hazardTicks: Math.round(safeTelemetry.hazardTicks || 0),
          mutator: state.weekly?.mutator?.label || "None",
          seed: state.challengeSeed || "",
          threatTier: formatThreatTierLabel(state.threatTier),
          route: state.routeBonus?.id || ""
        };
      }

      function updateRunRecords(summary) {
        if (!summary) return [];
        progress.records = progress.records || {
          bestScore: 0,
          bestKills: 0,
          bestSurvivalSec: 0
        };
        const highlights = [];
        const score = Math.max(0, Math.round(summary.score || 0));
        const kills = Math.max(0, Math.round(summary.kills || 0));
        const survivalSec = Math.max(0, Math.round(summary.durationSec || 0));
        if (score > (progress.records.bestScore || 0)) {
          progress.records.bestScore = score;
          highlights.push("New best score");
        }
        if (kills > (progress.records.bestKills || 0)) {
          progress.records.bestKills = kills;
          highlights.push("Most kills");
        }
        if (survivalSec > (progress.records.bestSurvivalSec || 0)) {
          progress.records.bestSurvivalSec = survivalSec;
          highlights.push("Longest survival");
        }
        return highlights;
      }

      function getPerformanceScore(summary) {
        const minutes = summary.durationSec / 60;
        const waveProgress = summary.globalWave || summary.wave;
        const waveScore = Math.max(0, waveProgress - 1) * 1.1;
        const killScore = summary.kills * 0.08;
        const scoreScore = summary.score / 1200;
        const timeScore = minutes * 0.8;
        return (waveScore + killScore + scoreScore + timeScore) * summary.rewardMultiplier;
      }

      function getPerformanceTier(score) {
        if (score >= 24) return "legendary";
        if (score >= 18) return "epic";
        if (score >= 12) return "rare";
        if (score >= 6) return "uncommon";
        return "common";
      }

      function buildLossRewards() {
        const summary = getRunSummary();
        const performanceScore = getPerformanceScore(summary);
        const performanceTier = getPerformanceTier(performanceScore);
        const tierIndex = Math.max(0, TIER_ORDER.indexOf(performanceTier));
        const waveProgress = summary.globalWave || summary.wave;
        const minutes = summary.durationSec / 60;
        const creditBase = 40
          + waveProgress * 12
          + summary.kills * 2
          + summary.score / 140
          + minutes * 12;
        const creditBonus = Math.round(creditBase * (1 + tierIndex * 0.18) * summary.rewardMultiplier);
        const blueprintCount = clamp(
          Math.floor(Math.max(0, waveProgress - 1) / 3) + (tierIndex >= 2 ? 1 : 0),
          0,
          tierIndex >= 4 ? 3 : 2
        );
        const salvageCount = clamp(
          Math.floor(summary.kills / 25) + (tierIndex >= 1 ? 1 : 0),
          0,
          tierIndex >= 3 ? 3 : 2
        );
        const partCount = (tierIndex >= 4 && waveProgress >= 8) ? 2 : (tierIndex >= 2 ? 1 : 0);
        const weaponCount = tierIndex >= 3 && waveProgress >= 6 ? 1 : 0;
        const blueprintTier = tierIndex >= 3 ? "rare" : tierIndex >= 1 ? "uncommon" : "common";
        const salvageTier = tierIndex >= 3 ? "rare" : tierIndex >= 1 ? "uncommon" : "common";
        const partTier = tierIndex >= 4 ? "legendary" : tierIndex >= 3 ? "epic" : tierIndex >= 2 ? "rare" : "uncommon";
        const weaponTier = tierIndex >= 4 ? "legendary" : "epic";
        const rewards = [];
        if (creditBonus > 0) {
          rewards.push({
            type: "credits",
            amount: creditBonus,
            tier: performanceTier,
            title: "Recovery credits",
            desc: `${creditBonus.toLocaleString()} credits secured`
          });
        }
        if (blueprintCount > 0) {
          rewards.push({
            type: "blueprints",
            amount: blueprintCount,
            tier: blueprintTier,
            title: "Blueprint cache",
            desc: `+${blueprintCount} blueprint${blueprintCount === 1 ? "" : "s"}`
          });
        }
        if (salvageCount > 0) {
          rewards.push({
            type: "salvage",
            amount: salvageCount,
            tier: salvageTier,
            title: "Salvage keys",
            desc: `+${salvageCount} key${salvageCount === 1 ? "" : "s"}`
          });
        }
        const parts = [];
        for (let i = 0; i < partCount; i += 1) {
          const part = generatePart(partTier);
          if (part) {
            parts.push(part);
          }
        }
        parts.forEach((part) => {
          const summaryText = buildStatSummary(getAttachmentStats(part) || part.stats);
          rewards.push({
            type: "part",
            amount: 1,
            tier: part.tier || getPartTier(part),
            title: part.name,
            desc: summaryText,
            part
          });
        });
        if (weaponCount > 0) {
          const weapon = createWeaponItem({ tier: weaponTier });
          if (weapon) {
            rewards.push({
              type: "weapon",
              amount: 1,
              tier: weapon.tier,
              title: weapon.name,
              desc: buildStatSummary(getWeaponItemStats(weapon) || weapon.stats),
              item: weapon
            });
          }
        }
        return {
          tier: performanceTier,
          score: performanceScore,
          summary,
          rewards
        };
      }

      function getLevelRewardTierWeights(level) {
        if (level <= 2) return [6, 3.5, 1.4, 0.4, 0.1];
        if (level <= 4) return [4.6, 3.3, 1.8, 0.7, 0.2];
        if (level <= 6) return [3.1, 3, 2.4, 1.2, 0.4];
        if (level <= 8) return [2.1, 2.6, 2.4, 1.6, 0.8];
        return [1.1, 1.8, 2.3, 2.2, 1.2];
      }

      function buildLevelRewards(level, summary) {
        const safeLevel = Math.max(1, level);
        const rewards = [];
        const tierIndex = clamp(Math.floor((safeLevel - 1) / 2), 0, TIER_ORDER.length - 1);
        const rewardTier = TIER_ORDER[tierIndex] || "common";
        const creditBase = 140
          + safeLevel * 42
          + summary.kills * 2
          + summary.score / 180;
        const creditBonus = Math.round(creditBase * summary.rewardMultiplier);
        if (creditBonus > 0) {
          rewards.push({
            type: "credits",
            amount: creditBonus,
            tier: rewardTier,
            title: "Mission credits",
            desc: `${creditBonus.toLocaleString()} credits secured`
          });
        }
        const salvageCount = safeLevel <= 3
          ? 2
          : safeLevel <= 6
            ? 1
            : (Math.random() < 0.35 ? 1 : 0);
        const blueprintCount = safeLevel <= 3
          ? 2
          : safeLevel <= 5
            ? 1
            : (Math.random() < 0.25 ? 1 : 0);
        const supportTier = tierIndex >= 4
          ? "legendary"
          : tierIndex >= 3
            ? "epic"
            : tierIndex >= 2
              ? "rare"
              : tierIndex >= 1
                ? "uncommon"
                : "common";
        if (salvageCount > 0) {
          rewards.push({
            type: "salvage",
            amount: salvageCount,
            tier: supportTier,
            title: "Salvage keys",
            desc: `+${salvageCount} key${salvageCount === 1 ? "" : "s"}`
          });
        }
        if (blueprintCount > 0) {
          rewards.push({
            type: "blueprints",
            amount: blueprintCount,
            tier: supportTier,
            title: "Blueprint cache",
            desc: `+${blueprintCount} blueprint${blueprintCount === 1 ? "" : "s"}`
          });
        }
        const baseItemCount = safeLevel <= 2 ? 2 : 1;
        const bonusChance = safeLevel <= 3 ? 0.45 : safeLevel <= 6 ? 0.25 : safeLevel <= 9 ? 0.16 : 0.12;
        const itemCount = baseItemCount + (Math.random() < bonusChance ? 1 : 0);
        const tierWeights = getLevelRewardTierWeights(safeLevel);
        for (let i = 0; i < itemCount; i += 1) {
          const rollWeapon = Math.random() < 0.45;
          if (rollWeapon) {
            const weapon = createWeaponItem({ tier: rollItemTier(null, tierWeights) });
            if (weapon) {
              rewards.push({
                type: "weapon",
                amount: 1,
                tier: weapon.tier,
                title: weapon.name,
                desc: buildStatSummary(getWeaponItemStats(weapon) || weapon.stats),
                item: weapon
              });
            }
          } else {
            const attachment = createAttachmentItem({ tier: rollItemTier(null, tierWeights) });
            if (attachment) {
              rewards.push({
                type: "part",
                amount: 1,
                tier: attachment.tier || getPartTier(attachment),
                title: attachment.name,
                desc: buildStatSummary(getAttachmentStats(attachment) || attachment.stats),
                part: attachment
              });
            }
          }
        }
        return {
          level: safeLevel,
          tier: rewardTier,
          summary,
          rewards
        };
      }

      function applyRewardBundle(result, logLabel) {
        if (!result || !result.rewards) return;
        result.rewards.forEach((reward) => {
          if (reward.type === "credits") {
            progress.bankedCredits += reward.amount;
          }
          if (reward.type === "blueprints") {
            progress.blueprints += reward.amount;
          }
          if (reward.type === "salvage") {
            progress.salvageKeys += reward.amount;
          }
          if (reward.type === "part" && reward.part) {
            addInventoryItem(reward.part, { notify: true });
          }
          if (reward.type === "weapon" && reward.item) {
            addInventoryItem(reward.item, { notify: true });
          }
        });
        saveProgress();
        checkProgressionUnlocks();
        renderShipyard();
        renderPremiumShop();
        renderArmory();
        renderContracts();
        if (result.rewards.length) {
          const notice = result.rewards.map((reward) => {
            if (reward.type === "credits") return `+${reward.amount} coins`;
            if (reward.type === "blueprints") return `+${reward.amount} plans`;
            if (reward.type === "salvage") return `+${reward.amount} keys`;
            if (reward.type === "part") return reward.title || "Attachment";
            if (reward.type === "weapon") return reward.title || "Weapon";
            return reward.title || reward.type || "Reward";
          }).join(" · ");
          const title = logLabel ? logLabel.replace(/\s*secured$/i, "") : "Rewards";
          showTip(null, title || "Rewards", notice, { kind: "reward", repeatable: true, duration: 7000 });
        }
        if (result.rewards.length && logLabel) {
          const summary = result.rewards.map((reward) => {
            if (reward.type === "part") return reward.title;
            if (reward.type === "weapon") return reward.title;
            if (reward.type === "credits") return `${reward.amount}c`;
            if (reward.type === "blueprints") {
              return `${reward.amount} blueprint${reward.amount === 1 ? "" : "s"}`;
            }
            if (reward.type === "salvage") {
              return `${reward.amount} key${reward.amount === 1 ? "" : "s"}`;
            }
            return `${reward.amount} ${reward.type}`;
          }).join(", ");
          logEvent(`${logLabel}: ${summary}.`);
        }
      }

      function applyLossRewards(result) {
        applyRewardBundle(result, "Recovery rewards secured");
      }

      function applyLevelRewards(result) {
        applyRewardBundle(result, "Level rewards secured");
      }

      function completeLevel() {
        if (!isCampaignMode()) return;
        const completedLevel = Math.max(1, state.level || progress.campaignLevel || 1);
        const summary = getRunSummary();
        const globalWave = summary.globalWave || getGlobalWave(state.wave);
        state.levelRewards = buildLevelRewards(completedLevel, summary);
        applyLevelRewards(state.levelRewards);
        progress.bestWave = Math.max(progress.bestWave, globalWave);
        progress.bestLevel = Math.max(progress.bestLevel || 1, completedLevel);
        progress.campaignLevel = Math.max(progress.campaignLevel || 1, completedLevel + 1);
        saveProgress();
        endRun("victory");
        setMode("victory");
        setOverlay("victory");
        playAudioCue("victory");
        logEvent(`Level ${completedLevel} cleared. Rewards secured.`);
      }

      function endRun(reason) {
        if (!state.runActive || state.runBanked) return;
        state.runActive = false;
        state.runBanked = true;
        state.blackHoles = [];
        if (state.frontier) {
          state.frontier.active = false;
        }
        const summary = getRunSummary();
        const telemetry = finalizeRunTelemetry(summary);
        if (state.training) {
          state.runHighlights = [];
          return;
        }
        const bankedCredits = Math.round(state.credits);
        progress.bankedCredits += bankedCredits;
        const difficultyLabel = (DIFFICULTY_SETTINGS[state.difficulty] || DIFFICULTY_SETTINGS.normal).label;
        state.runHighlights = updateRunRecords(summary);
        const entry = {
          wave: state.wave,
          level: summary.level,
          globalWave: summary.globalWave,
          score: Math.round(state.score),
          credits: bankedCredits,
          kills: state.kills,
          ship: state.runLoadout?.ship || player?.ship?.name || "Unknown",
          weapon: state.runLoadout?.weapon || player?.weapon?.name || "Unknown",
          difficulty: difficultyLabel,
          mode: isFrontierMode() ? "Frontier" : "Arcade",
          threatTier: formatThreatTierLabel(state.threatTier),
          mutator: state.weekly?.mutator?.label || "None",
          seed: state.challengeSeed || "",
          route: state.routeBonus?.id || ""
        };
        progress.runHistory = [entry, ...progress.runHistory].slice(0, 8);
        if (telemetry) {
          const analyticsEntry = buildRunAnalyticsEntry(summary, telemetry, reason, difficultyLabel);
          progress.runAnalytics = [analyticsEntry, ...(progress.runAnalytics || [])].slice(0, 10);
        }
        saveProgress();
        checkProgressionUnlocks();
        renderHistory();
        renderRunAnalytics();
        renderShipyard();
        renderPremiumShop();
        renderArmory();
        renderContracts();
        if (bankedCredits > 0) {
          logEvent(`Run banked: ${bankedCredits} credits secured.`);
        }
      }

      function xpToNext(rank) {
        return Math.floor(120 + (rank - 1) * 80);
      }

      function triggerGameOver() {
        state.runEndedByAbort = false;
        setMode("gameover");
        if (!state.training) {
          progress.bestWave = Math.max(progress.bestWave, getGlobalWave(state.wave));
          saveProgress();
        }
        endRun("wrecked");
        if (!state.training) {
          state.lossRewards = buildLossRewards();
          applyLossRewards(state.lossRewards);
        } else {
          state.lossRewards = null;
        }
        spawnExplosion(player.x, player.y, "#ffffff", 26);
        setOverlay("gameover");
        playAudioCue("gameover");
        logEvent("Ship destroyed. Returning to hangar.");
      }

      function triggerAbortRewards() {
        state.runEndedByAbort = true;
        setMode("gameover");
        if (!state.training) {
          progress.bestWave = Math.max(progress.bestWave, getGlobalWave(state.wave));
          saveProgress();
        }
        endRun("abort");
        state.lossRewards = null;
        setOverlay("gameover");
        logEvent("Run aborted. Returning to hangar.");
      }

      function renderLootBursts() {
        if (!lootBursts.length) return;
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "600 12px \"Chakra Petch\", sans-serif";
        lootBursts.forEach((burst) => {
          const alpha = clamp(burst.life / burst.maxLife, 0, 1);
          const color = TIER_COLORS[burst.tier] || "#e5f1ff";
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = "rgba(4, 8, 16, 0.8)";
          ctx.lineWidth = 3;
          ctx.strokeText(burst.label, burst.x, burst.y);
          ctx.fillStyle = color;
          ctx.fillText(burst.label, burst.x, burst.y);
        });
        ctx.restore();
        ctx.globalAlpha = 1;
      }

      function renderDamageNumbers() {
        if (!damageNumbers.length) return;
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "700 13px \"Chakra Petch\", sans-serif";
        const useStroke = damageNumbers.length < 80;
        damageNumbers.forEach((burst) => {
          const alpha = clamp(burst.life / burst.maxLife, 0, 1);
          ctx.globalAlpha = alpha;
          if (useStroke) {
            ctx.strokeStyle = "rgba(4, 8, 16, 0.85)";
            ctx.lineWidth = 3;
            ctx.strokeText(burst.text, burst.x, burst.y);
          }
          ctx.fillStyle = burst.color;
          ctx.fillText(burst.text, burst.x, burst.y);
        });
        ctx.restore();
        ctx.globalAlpha = 1;
      }

      function renderObstacles() {
        if (!obstacles.length) return;
        const bounds = getViewportBounds();
        obstacles.forEach((obstacle) => {
          if (obstacle.kind === "rock") {
            const radius = obstacle.radius;
            if (obstacle.x + radius < bounds.left - 80 || obstacle.x - radius > bounds.right + 80
              || obstacle.y + radius < bounds.top - 80 || obstacle.y - radius > bounds.bottom + 80) {
              return;
            }
            const gradient = ctx.createRadialGradient(
              obstacle.x - radius * 0.3,
              obstacle.y - radius * 0.3,
              radius * 0.2,
              obstacle.x,
              obstacle.y,
              radius
            );
            const shade = obstacle.shade || 0.5;
            gradient.addColorStop(0, `rgba(140, 170, 195, ${0.4 + shade * 0.2})`);
            gradient.addColorStop(1, `rgba(30, 42, 60, ${0.8 + shade * 0.15})`);
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(obstacle.x, obstacle.y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
            ctx.lineWidth = 1.5;
            ctx.stroke();
            return;
          }
          const halfWidth = obstacle.width * 0.5;
          const halfHeight = obstacle.height * 0.5;
          if (obstacle.x + halfWidth < bounds.left - 80 || obstacle.x - halfWidth > bounds.right + 80
            || obstacle.y + halfHeight < bounds.top - 80 || obstacle.y - halfHeight > bounds.bottom + 80) {
            return;
          }
          const x = obstacle.x - halfWidth;
          const y = obstacle.y - halfHeight;
          ctx.fillStyle = `rgba(22, 34, 48, ${0.7 + (obstacle.shade || 0.3) * 0.2})`;
          ctx.fillRect(x, y, obstacle.width, obstacle.height);
          ctx.strokeStyle = "rgba(160, 190, 220, 0.12)";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, y, obstacle.width, obstacle.height);
        });
      }

      function renderFieldDrops() {
        if (!fieldDrops.length) return;
        const bounds = getViewportBounds();
        const now = performance.now() * 0.001;
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "600 9px \"Space Grotesk\", \"Chakra Petch\", sans-serif";
        fieldDrops.forEach((drop) => {
          const def = getFieldDropDef(drop.typeId);
          if (!def) return;
          const radius = drop.radius || 16;
          if (drop.x + radius < bounds.left - 40 || drop.x - radius > bounds.right + 40
            || drop.y + radius < bounds.top - 40 || drop.y - radius > bounds.bottom + 40) {
            return;
          }
          const color = TIER_COLORS[def.tier] || "#ffffff";
          const bob = Math.sin(now * 2 + drop.phase) * 4;
          const x = drop.x;
          const y = drop.y + bob;
          const alpha = clamp(drop.life / drop.maxLife, 0, 1);
          ctx.globalAlpha = alpha * 0.9;
          const glow = ctx.createRadialGradient(x, y, 4, x, y, radius + 14);
          glow.addColorStop(0, color);
          glow.addColorStop(1, "rgba(0, 0, 0, 0)");
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(x, y, radius + 14, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = alpha;
          ctx.fillStyle = "rgba(8, 14, 24, 0.85)";
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.fillText(def.symbol || "DROP", x, y);
        });
        ctx.restore();
        ctx.globalAlpha = 1;
      }

      function renderHazards() {
        if (!state.hazards || !state.hazards.length) return;
        const bounds = getViewportBounds();
        state.hazards.forEach((hazard) => {
          const radius = hazard.radius || 100;
          if (hazard.x + radius < bounds.left - 30 || hazard.x - radius > bounds.right + 30
            || hazard.y + radius < bounds.top - 30 || hazard.y - radius > bounds.bottom + 30) {
            return;
          }
          const lifeRatio = clamp((hazard.life || 0) / 22, 0.15, 1);
          ctx.save();
          ctx.globalAlpha = 0.2 + lifeRatio * 0.35;
          const gradient = ctx.createRadialGradient(hazard.x, hazard.y, radius * 0.2, hazard.x, hazard.y, radius);
          gradient.addColorStop(0, "rgba(240, 105, 105, 0.28)");
          gradient.addColorStop(1, "rgba(240, 105, 105, 0.02)");
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(hazard.x, hazard.y, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 0.5 + Math.sin(performance.now() * 0.008) * 0.15;
          ctx.strokeStyle = "rgba(255, 159, 107, 0.8)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(hazard.x, hazard.y, radius * 0.92, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        });
      }

      function drawWorldBounds() {
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = 2;
        ctx.strokeRect(4, 4, state.worldWidth - 8, state.worldHeight - 8);
        ctx.restore();
      }

      function renderMinimap() {
        if (!minimapCtx || !dom.minimap) return;
        const dpr = Math.max(1, state.renderScale || window.devicePixelRatio || 1);
        const width = dom.minimap.width / dpr;
        const height = dom.minimap.height / dpr;
        minimapCtx.clearRect(0, 0, width, height);
        if (!player || (state.mode !== "flight" && state.mode !== "training")) return;
        minimapCtx.fillStyle = "rgba(6, 10, 18, 0.9)";
        minimapCtx.fillRect(0, 0, width, height);

        const padding = 6;
        const mapWidth = width - padding * 2;
        const mapHeight = height - padding * 2;
        const scaleX = mapWidth / state.worldWidth;
        const scaleY = mapHeight / state.worldHeight;
        const scale = Math.min(scaleX, scaleY);
        const offsetX = (width - state.worldWidth * scale) * 0.5;
        const offsetY = (height - state.worldHeight * scale) * 0.5;

        minimapCtx.strokeStyle = "rgba(255, 255, 255, 0.18)";
        minimapCtx.lineWidth = 1;
        minimapCtx.strokeRect(offsetX, offsetY, state.worldWidth * scale, state.worldHeight * scale);

        minimapCtx.fillStyle = "rgba(140, 170, 195, 0.35)";
        obstacles.forEach((obstacle) => {
          if (obstacle.kind === "rock") {
            minimapCtx.beginPath();
            minimapCtx.arc(
              offsetX + obstacle.x * scale,
              offsetY + obstacle.y * scale,
              Math.max(2, obstacle.radius * scale),
              0,
              Math.PI * 2
            );
            minimapCtx.fill();
            return;
          }
          minimapCtx.fillRect(
            offsetX + (obstacle.x - obstacle.width * 0.5) * scale,
            offsetY + (obstacle.y - obstacle.height * 0.5) * scale,
            obstacle.width * scale,
            obstacle.height * scale
          );
        });

        minimapCtx.fillStyle = "#f06969";
        enemies.forEach((enemy) => {
          const x = offsetX + enemy.x * scale;
          const y = offsetY + enemy.y * scale;
          minimapCtx.fillRect(x - 1, y - 1, 2, 2);
        });

        fieldDrops.forEach((drop) => {
          const def = getFieldDropDef(drop.typeId);
          const color = TIER_COLORS[def.tier] || "#ffffff";
          minimapCtx.fillStyle = color;
          minimapCtx.fillRect(
            offsetX + drop.x * scale - 1,
            offsetY + drop.y * scale - 1,
            2,
            2
          );
        });

        minimapCtx.strokeStyle = "rgba(255, 255, 255, 0.35)";
        minimapCtx.strokeRect(
          offsetX + state.camera.x * scale,
          offsetY + state.camera.y * scale,
          state.width * scale,
          state.height * scale
        );

        minimapCtx.fillStyle = "#44d2c2";
        minimapCtx.beginPath();
        minimapCtx.arc(
          offsetX + player.x * scale,
          offsetY + player.y * scale,
          3,
          0,
          Math.PI * 2
        );
        minimapCtx.fill();
      }

      function render() {
        ctx.clearRect(0, 0, state.width, state.height);
        ctx.fillStyle = backgroundGradient || "#05090f";
        ctx.fillRect(0, 0, state.width, state.height);

        updateCamera();
        const bounds = getViewportBounds();
        ctx.save();
        ctx.translate(-state.camera.x, -state.camera.y);

        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        stars.forEach((star) => {
          if (star.x < bounds.left - 40 || star.x > bounds.right + 40
            || star.y < bounds.top - 40 || star.y > bounds.bottom + 40) {
            return;
          }
          ctx.globalAlpha = star.alpha;
          ctx.beginPath();
          ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.globalAlpha = 1;

        renderObstacles();
        renderFieldDrops();
        renderHazards();

        particles.forEach((particle) => {
          ctx.globalAlpha = Math.max(0, particle.life / particle.maxLife);
          ctx.fillStyle = particle.color;
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.globalAlpha = 1;

        pulses.forEach((pulse) => {
          ctx.globalAlpha = Math.max(0, pulse.life / pulse.maxLife);
          ctx.strokeStyle = pulse.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(pulse.x, pulse.y, pulse.radius, 0, Math.PI * 2);
          ctx.stroke();
        });
        ctx.globalAlpha = 1;

        renderBlackHoles();

        if (player) {
          drawAura();
        }

        if (state.decoy) {
          ctx.strokeStyle = "rgba(246, 198, 95, 0.6)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(state.decoy.x, state.decoy.y, 12, 0, Math.PI * 2);
          ctx.stroke();
        }

        state.mines.forEach((mine) => {
          ctx.fillStyle = "rgba(246, 198, 95, 0.8)";
          ctx.beginPath();
          ctx.arc(mine.x, mine.y, mine.radius * 0.5, 0, Math.PI * 2);
          ctx.fill();
        });

        bullets.forEach((bullet) => {
          let color = bullet.owner === "player" ? (bullet.crit ? "#f6c65f" : "#6ee7b7") : "#f06969";
          if (bullet.tint) {
            color = bullet.tint;
          }
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(bullet.x, bullet.y, bullet.radius, 0, Math.PI * 2);
          ctx.fill();
        });

        enemies.forEach((enemy) => {
          drawEnemyThreatHalo(enemy);
          drawShip(enemy.x, enemy.y, enemy.angle, enemy.radius, enemy.color, enemy.hitFlash > 0);
          drawShield(enemy);
          drawEnemyVitals(enemy);
        });

        if (player) {
          drawShip(player.x, player.y, player.angle, player.radius, "#44d2c2", player.hitFlash > 0, player.thrusting);
          drawShield(player);
        }

        renderHelpers();
        renderLootBursts();
        renderDamageNumbers();
        drawWorldBounds();
        drawCrosshair();
        ctx.restore();

        drawThreatIndicators();
        renderMinimap();
      }

      function isPriorityEnemy(enemy) {
        return !!(enemy && (enemy.id === "dreadnought" || enemy.id === "ace" || enemy.elite || enemy.miniboss));
      }

      function getPriorityEnemyColor(enemy) {
        if (!enemy) return "#f6c65f";
        if (enemy.id === "dreadnought") return "#f6c65f";
        if (enemy.miniboss) return "#ff9f6b";
        if (enemy.id === "ace") return "#ff9f6b";
        return "#b98cff";
      }

      function drawEnemyThreatHalo(enemy) {
        if (!isPriorityEnemy(enemy)) return;
        const now = performance.now();
        const pulse = 0.45 + Math.sin(now * 0.01) * 0.18;
        const color = getPriorityEnemyColor(enemy);
        const radius = enemy.radius + (enemy.id === "dreadnought" ? 16 : 11);
        ctx.save();
        ctx.globalAlpha = 0.45 + pulse * 0.3;
        ctx.strokeStyle = color;
        ctx.lineWidth = enemy.id === "dreadnought" ? 3 : 2;
        ctx.beginPath();
        ctx.arc(enemy.x, enemy.y, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 0.22 + pulse * 0.2;
        ctx.beginPath();
        ctx.arc(enemy.x, enemy.y, radius + 8, 0, Math.PI * 2);
        ctx.stroke();
        if (enemy.dashCooldown > 0 && enemy.dashTimer <= 0.75) {
          ctx.globalAlpha = 0.55;
          ctx.strokeStyle = "#ff9f6b";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.arc(enemy.x, enemy.y, radius + 16, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.restore();
      }

      function drawThreatIndicators() {
        if (!player || !enemies.length) return;
        if (state.mode !== "flight" && state.mode !== "training") return;
        const margin = 34;
        const centerX = state.width * 0.5;
        const centerY = state.height * 0.5;
        const priorities = enemies
          .filter((enemy) => isPriorityEnemy(enemy))
          .sort((a, b) => distanceBetween(player, a) - distanceBetween(player, b))
          .slice(0, 6);
        if (!priorities.length) return;
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "700 10px \"Space Grotesk\", \"Chakra Petch\", sans-serif";
        priorities.forEach((enemy) => {
          const screenX = enemy.x - state.camera.x;
          const screenY = enemy.y - state.camera.y;
          const onScreen = screenX >= margin && screenX <= state.width - margin
            && screenY >= margin && screenY <= state.height - margin;
          if (onScreen) return;
          const indicatorX = clamp(screenX, margin, state.width - margin);
          const indicatorY = clamp(screenY, margin, state.height - margin);
          const angle = Math.atan2(screenY - centerY, screenX - centerX);
          const color = getPriorityEnemyColor(enemy);
          const distance = Math.round(distanceBetween(player, enemy));
          ctx.save();
          ctx.translate(indicatorX, indicatorY);
          ctx.rotate(angle);
          ctx.fillStyle = color;
          ctx.strokeStyle = "rgba(8, 14, 24, 0.9)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(14, 0);
          ctx.lineTo(-8, 7);
          ctx.lineTo(-8, -7);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          ctx.restore();
          ctx.fillStyle = color;
          ctx.strokeStyle = "rgba(8, 14, 24, 0.9)";
          ctx.lineWidth = 3;
          const distanceY = indicatorY - 13;
          ctx.strokeText(`${distance}m`, indicatorX, distanceY);
          ctx.fillText(`${distance}m`, indicatorX, distanceY);
        });
        ctx.restore();
      }

      function drawAura() {
        if (!player || player.auraRadius <= 0 || player.auraDamage <= 0) return;
        if (state.mode !== "flight" && state.mode !== "training") return;
        ctx.save();
        ctx.strokeStyle = "rgba(124, 168, 255, 0.4)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(player.x, player.y, player.auraRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      function renderBlackHoles() {
        if (!state.blackHoles || !state.blackHoles.length) return;
        state.blackHoles.forEach((hole) => {
          const lifeRatio = hole.maxLife ? hole.life / hole.maxLife : 1;
          const alpha = clamp(0.2 + lifeRatio * 0.4, 0, 0.7);
          ctx.save();
          ctx.globalAlpha = alpha;
          const gradient = ctx.createRadialGradient(hole.x, hole.y, hole.radius * 0.2, hole.x, hole.y, hole.radius);
          gradient.addColorStop(0, "rgba(16, 28, 46, 0.9)");
          gradient.addColorStop(0.45, "rgba(88, 122, 208, 0.45)");
          gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(hole.x, hole.y, hole.radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        });
      }

      function renderHelpers() {
        if (!helpers.length || (state.mode !== "flight" && state.mode !== "training")) return;
        helpers.forEach((helper) => {
          ctx.save();
          ctx.translate(helper.x, helper.y);
          ctx.fillStyle = "#6ee7b7";
          ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(6, 0);
          ctx.lineTo(-4, 4);
          ctx.lineTo(-4, -4);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
          ctx.restore();
        });
      }

      function drawShip(x, y, angle, size, color, hitFlash, thrusting) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);

        if (thrusting) {
          ctx.fillStyle = "rgba(246, 198, 95, 0.8)";
          ctx.beginPath();
          ctx.moveTo(-size * 0.8, 0);
          ctx.lineTo(-size * 1.4, size * 0.5);
          ctx.lineTo(-size * 1.6, 0);
          ctx.lineTo(-size * 1.4, -size * 0.5);
          ctx.closePath();
          ctx.fill();
        }

        ctx.fillStyle = hitFlash ? "#ffffff" : color;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(size, 0);
        ctx.lineTo(-size * 0.7, size * 0.7);
        ctx.lineTo(-size * 0.4, 0);
        ctx.lineTo(-size * 0.7, -size * 0.7);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        ctx.restore();
      }

      function drawShield(entity) {
        if (entity.maxShield <= 0 || entity.shield <= 0) return;
        const ratio = entity.shield / entity.maxShield;
        ctx.save();
        ctx.globalAlpha = 0.2 + ratio * 0.4;
        ctx.strokeStyle = entity === player ? "#57e0ff" : "#f6c65f";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(entity.x, entity.y, entity.radius + 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      function drawEnemyVitals(enemy) {
        if (!enemy) return;
        const barWidth = Math.max(26, enemy.radius * 3.2);
        const barHeight = 4;
        const gap = 2;
        const hasShield = enemy.maxShield > 0;
        const totalHeight = barHeight + (hasShield ? barHeight + gap : 0);
        const x = enemy.x - barWidth / 2;
        let y = enemy.y - enemy.radius - 12 - totalHeight;
        y = Math.max(6, y);
        ctx.save();
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = "rgba(5, 8, 14, 0.7)";
        ctx.fillRect(x - 1, y - 1, barWidth + 2, totalHeight + 2);
        if (hasShield) {
          const shieldRatio = clamp(enemy.shield / enemy.maxShield, 0, 1);
          ctx.fillStyle = "#57e0ff";
          ctx.fillRect(x, y, barWidth * shieldRatio, barHeight);
          y += barHeight + gap;
        }
        const hullRatio = clamp(enemy.health / enemy.maxHealth, 0, 1);
        ctx.fillStyle = "#f06969";
        ctx.fillRect(x, y, barWidth * hullRatio, barHeight);
        ctx.restore();
      }

      function drawCrosshair() {
        const target = getAimTarget();
        const aimX = target.x;
        const aimY = target.y;
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(aimX, aimY, 10, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      function updateHud() {
        flushDeferredState();
        const now = performance.now();
        if (now - state.lastHudUpdate < HUD_UPDATE_INTERVAL_MS) {
          return;
        }
        state.lastHudUpdate = now;
        if (!player) return;
        const hullPct = clamp(player.health / player.maxHealth, 0, 1) * 100;
        const shieldPct = player.maxShield > 0 ? clamp(player.shield / player.maxShield, 0, 1) * 100 : 0;
        const energyPct = clamp(player.energy / player.maxEnergy, 0, 1) * 100;
        meters.hull.style.width = `${hullPct}%`;
        meters.shield.style.width = `${shieldPct}%`;
        meters.energy.style.width = `${energyPct}%`;
        if (hudMeters.hull) hudMeters.hull.style.width = `${hullPct}%`;
        if (hudMeters.shield) hudMeters.shield.style.width = `${shieldPct}%`;
        if (hudMeters.energy) hudMeters.energy.style.width = `${energyPct}%`;

        stats.hullText.textContent = `${Math.round(player.health)} / ${Math.round(player.maxHealth)}`;
        stats.shieldText.textContent = player.maxShield > 0
          ? `${Math.round(player.shield)} / ${Math.round(player.maxShield)}`
          : "0 / 0";
        stats.energyText.textContent = `${Math.round(player.energy)} / ${Math.round(player.maxEnergy)}`;
        if (hudStats.hull) {
          hudStats.hull.textContent = `${Math.round(player.health)}/${Math.round(player.maxHealth)}`;
        }
        if (hudStats.shield) {
          hudStats.shield.textContent = player.maxShield > 0
            ? `${Math.round(player.shield)}/${Math.round(player.maxShield)}`
            : "0/0";
        }
        if (hudStats.energy) {
          hudStats.energy.textContent = `${Math.round(player.energy)}/${Math.round(player.maxEnergy)}`;
        }

        stats.wave.textContent = getWaveDisplay(state.wave);
        stats.enemyCount.textContent = enemies.length;
        stats.score.textContent = Math.round(state.score).toLocaleString();
        stats.credits.textContent = Math.round(state.credits).toLocaleString();
        const threatLabel = formatThreatTierLabel(state.threatTier);
        if (stats.threatTier) {
          stats.threatTier.textContent = threatLabel;
        }
        if (stats.threatTierHud) {
          stats.threatTierHud.textContent = threatLabel;
        }
        const objectiveText = state.waveObjective
          ? `${state.waveObjective.label}${state.waveObjective.complete ? " (Complete)" : ` • ${state.waveObjective.progressLabel || ""}`}`
          : "Eliminate hostiles";
        if (stats.objective) {
          stats.objective.textContent = objectiveText;
        }
        if (stats.objectiveHud) {
          stats.objectiveHud.textContent = objectiveText;
        }
        if (stats.modeLabel) {
          if (state.training) {
            stats.modeLabel.textContent = "Training";
          } else if (isFrontierMode()) {
            stats.modeLabel.textContent = "Frontier";
          } else {
            stats.modeLabel.textContent = `Arcade L${state.level || 1}`;
          }
        }
        if (dom.tierPill && stats.tier) {
          const showTier = isFrontierMode() && state.frontier && state.frontier.active;
          dom.tierPill.hidden = !showTier;
          if (showTier) {
            stats.tier.textContent = state.frontier.tier || 1;
          }
        }

        stats.rank.textContent = progress.rank;
        stats.xp.textContent = `${Math.round(progress.xp)} / ${xpToNext(progress.rank)}`;
        stats.techPoints.textContent = progress.techPoints;
        stats.bestWave.textContent = progress.campaignLevel || 1;
        stats.totalKills.textContent = progress.totalKills;
        stats.bankedCredits.textContent = `Credits ${Math.round(progress.bankedCredits)}`;
        stats.bankedCreditsTotal.textContent = Math.round(progress.bankedCredits).toLocaleString();
        if (stats.premiumCurrency) {
          stats.premiumCurrency.textContent = `${PREMIUM_CURRENCY_LABEL} ${progress.premiumCurrency || 0}`;
        }
        if (stats.premiumCurrencyTotal) {
          stats.premiumCurrencyTotal.textContent = Math.round(progress.premiumCurrency || 0).toLocaleString();
        }
        stats.blueprints.textContent = `Blueprints ${progress.blueprints}`;
        stats.salvageKeys.textContent = `Keys ${progress.salvageKeys || 0}`;

        stats.damage.textContent = Math.round(player.damage);
        stats.fireRate.textContent = `${player.fireRate.toFixed(1)} / sec`;
        stats.speed.textContent = Math.round(player.maxSpeed);
        stats.shieldRegen.textContent = `${player.shieldRegen.toFixed(1)} / sec`;
        stats.energyRegen.textContent = `${player.energyRegen.toFixed(1)} / sec`;
        stats.crit.textContent = `${Math.round(player.critChance * 100)}%`;
        const derived = getDerivedCombatMetrics(player);
        if (stats.dps) {
          stats.dps.textContent = Math.round(derived.dps).toLocaleString();
        }
        if (stats.ehp) {
          stats.ehp.textContent = Math.round(derived.ehp).toLocaleString();
        }
        if (stats.energySustain) {
          const sustain = Number.isFinite(derived.sustain) ? derived.sustain : 0;
          stats.energySustain.textContent = `${sustain >= 0 ? "+" : ""}${sustain.toFixed(1)} / sec`;
        }

        stats.shipName.textContent = player.ship ? player.ship.name : "-";
        stats.weaponName.textContent = player.weapon ? player.weapon.name : "-";
        stats.secondaryName.textContent = isFeatureUnlocked("secondary")
          ? (player.secondary ? player.secondary.name : "-")
          : "Locked";
        stats.sector.textContent = `Sector ${state.sector.toString().padStart(2, "0")}`;
        stats.sectorMod.textContent = state.sectorMod ? state.sectorMod.name : "Clear";
        if (!isFeatureUnlocked("contracts")) {
          stats.contractStatus.textContent = "Locked";
        } else if (!state.contracts.length) {
          stats.contractStatus.textContent = "Offline";
        } else {
          const activeCount = state.contracts.filter((contract) => !contract.complete).length;
          stats.contractStatus.textContent = activeCount ? `${activeCount} Active` : "Complete";
        }
        if (stats.weeklyMutator) {
          stats.weeklyMutator.textContent = state.weekly?.mutator?.label || "None";
        }
        if (stats.challengeSeed) {
          stats.challengeSeed.textContent = state.challengeSeed || "-";
        }
        if (progress.settings.inputMode === "keyboard") {
          stats.controlMode.textContent = "Keyboard";
        } else if (progress.settings.inputMode === "controller") {
          stats.controlMode.textContent = state.controllerConnected ? "Controller" : "Controller (No Pad)";
        } else {
          stats.controlMode.textContent = "Hybrid";
        }

        if (!isFeatureUnlocked("ability")) {
          stats.abilityStatus.textContent = "Locked";
        } else {
          const abilityReady = player.abilityCooldown <= 0 && player.abilityTimer <= 0;
          stats.abilityStatus.textContent = player.abilityTimer > 0
            ? "Active"
            : abilityReady
              ? "Ready"
              : `CD ${Math.ceil(player.abilityCooldown)}s`;
        }

        if (!isFeatureUnlocked("secondary")) {
          stats.secondaryStatus.textContent = "Locked";
        } else {
          const secondaryReady = player.secondaryCooldown <= 0;
          stats.secondaryStatus.textContent = secondaryReady ? "Ready" : `CD ${Math.ceil(player.secondaryCooldown)}s`;
        }

        renderStatusIcons();
        renderActiveUpgrades();
      }

      function renderStatusIcons() {
        if (!dom.statusIcons || !player) return;
        const chips = [];
        if (player.abilityTimer > 0) {
          chips.push({ kind: "buff", label: `Ability ${Math.ceil(player.abilityTimer)}s` });
        }
        if (player.damageBoostTimer > 0) {
          chips.push({ kind: "buff", label: `Damage x${(player.damageBoostMultiplier || 1).toFixed(1)}` });
        }
        if (player.speedBoostTimer > 0) {
          chips.push({ kind: "buff", label: `Boost x${(player.speedBoostMultiplier || 1).toFixed(1)}` });
        }
        if (state.routeBonus && state.routeBonus.wavesRemaining > 0) {
          chips.push({
            kind: "buff",
            label: `${formatRouteBonusLabel(state.routeBonus.id)} (${state.routeBonus.wavesRemaining})`
          });
        }
        if (player.slowTimer > 0) {
          chips.push({ kind: "debuff", label: "Slowed" });
        }
        if (state.hazards && state.hazards.length > 0) {
          chips.push({ kind: "debuff", label: `Hazards ${state.hazards.length}` });
        }
        if (state.waveObjective) {
          const objectiveLabel = state.waveObjective.complete
            ? "Objective complete"
            : `${state.waveObjective.label}: ${state.waveObjective.progressLabel || ""}`;
          chips.push({ kind: "objective", label: objectiveLabel.trim() });
        }
        const mutator = state.weekly?.mutator;
        if (mutator && mutator.id && mutator.id !== "none") {
          chips.push({ kind: "system", label: mutator.label });
        }
        if (progress.settings.inputMode === "controller") {
          chips.push({
            kind: "system",
            label: state.controllerConnected ? "Pad connected" : "No controller"
          });
        }
        const display = chips.slice(0, 5);
        dom.statusIcons.innerHTML = display.map((chip) => `
          <span class="status-chip" data-kind="${chip.kind}">${chip.label}</span>
        `).join("");
      }

      function updatePerformanceOverlayVisibility() {
        const perfMode = getPerfMode();
        const showBasic = perfMode === "basic";
        const showDetail = perfMode === "detail";
        if (dom.perfOverlay) {
          dom.perfOverlay.hidden = !showBasic;
        }
        if (dom.perfDetailOverlay) {
          dom.perfDetailOverlay.hidden = !showDetail;
        }
        renderPerformanceOverlays(performance.now(), true);
      }

      function formatPerfValue(value, digits = 1) {
        if (!Number.isFinite(value)) return "--";
        return value.toFixed(digits);
      }

      function formatPerfInt(value) {
        if (!Number.isFinite(value)) return "--";
        return Math.round(value).toString();
      }

      function formatPerfPercent(value, digits = 0) {
        if (!Number.isFinite(value)) return "--";
        return `${value.toFixed(digits)}%`;
      }

      function formatPerfLogNumber(value, digits = 2) {
        if (!Number.isFinite(value)) return "";
        return value.toFixed(digits);
      }

      function buildPerformanceBasicHtml() {
        return `
          <div class="perf-row"><span class="perf-label">FPS</span><span class="perf-value">${formatPerfInt(perfMetrics.fps)}</span></div>
          <div class="perf-row"><span class="perf-label">Frame</span><span class="perf-value">${formatPerfValue(perfMetrics.avgFrameMs)} ms</span></div>
          <div class="perf-row"><span class="perf-label">Update</span><span class="perf-value">${formatPerfValue(perfMetrics.avgUpdateMs)} ms</span></div>
          <div class="perf-row"><span class="perf-label">Render</span><span class="perf-value">${formatPerfValue(perfMetrics.avgRenderMs)} ms</span></div>
        `;
      }

      function buildPerformanceDetailHtml() {
        const memory = performance && performance.memory
          ? (performance.memory.usedJSHeapSize / (1024 * 1024))
          : null;
        const rafHz = perfMetrics.avgRafIntervalMs
          ? 1000 / perfMetrics.avgRafIntervalMs
          : NaN;
        const busyPercent = perfMetrics.avgBusyRatio
          ? perfMetrics.avgBusyRatio * 100
          : NaN;
        const focusState = typeof document.hasFocus === "function"
          ? (document.hasFocus() ? "yes" : "no")
          : "n/a";
        const visibility = document.visibilityState || "unknown";
        const dpr = window.devicePixelRatio || 1;
        const renderScale = state.renderScale || dpr;
        const canvasPx = dom.canvas
          ? `${dom.canvas.width}x${dom.canvas.height}`
          : "--";
        const longTaskSummary = longTaskMetrics.supported
          ? `${longTaskMetrics.count} • ${formatPerfValue(longTaskMetrics.totalDuration)} ms`
          : "n/a";
        return `
          <div class="perf-row"><span class="perf-label">FPS</span><span class="perf-value">${formatPerfInt(perfMetrics.fps)}</span></div>
          <div class="perf-grid">
            <div class="perf-row"><span class="perf-label">Frame Avg</span><span class="perf-value">${formatPerfValue(perfMetrics.avgFrameMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">Frame Max</span><span class="perf-value">${formatPerfValue(perfMetrics.maxFrameMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">Update Avg</span><span class="perf-value">${formatPerfValue(perfMetrics.avgUpdateMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">Render Avg</span><span class="perf-value">${formatPerfValue(perfMetrics.avgRenderMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">HUD Avg</span><span class="perf-value">${formatPerfValue(perfMetrics.avgHudMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">RAF Avg</span><span class="perf-value">${formatPerfValue(perfMetrics.avgRafIntervalMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">RAF Max</span><span class="perf-value">${formatPerfValue(perfMetrics.maxRafIntervalMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">RAF Hz</span><span class="perf-value">${formatPerfValue(rafHz, 1)}</span></div>
            <div class="perf-row"><span class="perf-label">Idle Avg</span><span class="perf-value">${formatPerfValue(perfMetrics.avgIdleMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">Util Avg</span><span class="perf-value">${formatPerfPercent(busyPercent, 0)}</span></div>
            <div class="perf-row"><span class="perf-label">Delta Avg</span><span class="perf-value">${formatPerfValue(perfMetrics.avgDeltaMs)} ms</span></div>
            <div class="perf-row"><span class="perf-label">33ms+ /s</span><span class="perf-value">${formatPerfValue(perfMetrics.slowRate, 1)}</span></div>
            <div class="perf-row"><span class="perf-label">50ms+ /s</span><span class="perf-value">${formatPerfValue(perfMetrics.hitchRate, 1)}</span></div>
            <div class="perf-row"><span class="perf-label">Mode</span><span class="perf-value">${state.mode}</span></div>
            <div class="perf-row"><span class="perf-label">Visibility</span><span class="perf-value">${visibility}</span></div>
            <div class="perf-row"><span class="perf-label">Focus</span><span class="perf-value">${focusState}</span></div>
            <div class="perf-row"><span class="perf-label">DPR</span><span class="perf-value">${formatPerfValue(dpr, 2)}</span></div>
            <div class="perf-row"><span class="perf-label">Render Scale</span><span class="perf-value">${formatPerfValue(renderScale, 2)}</span></div>
            <div class="perf-row"><span class="perf-label">Long Tasks</span><span class="perf-value">${longTaskSummary}</span></div>
            <div class="perf-row"><span class="perf-label">Enemies</span><span class="perf-value">${enemies.length}</span></div>
            <div class="perf-row"><span class="perf-label">Bullets</span><span class="perf-value">${bullets.length}</span></div>
            <div class="perf-row"><span class="perf-label">Particles</span><span class="perf-value">${particles.length}</span></div>
            <div class="perf-row"><span class="perf-label">Pulses</span><span class="perf-value">${pulses.length}</span></div>
            <div class="perf-row"><span class="perf-label">Black Holes</span><span class="perf-value">${state.blackHoles ? state.blackHoles.length : 0}</span></div>
            <div class="perf-row"><span class="perf-label">Mines</span><span class="perf-value">${state.mines.length}</span></div>
            <div class="perf-row"><span class="perf-label">Helpers</span><span class="perf-value">${helpers.length}</span></div>
            <div class="perf-row"><span class="perf-label">Obstacles</span><span class="perf-value">${obstacles.length}</span></div>
            <div class="perf-row"><span class="perf-label">Stars</span><span class="perf-value">${stars.length}</span></div>
            <div class="perf-row"><span class="perf-label">Canvas</span><span class="perf-value">${Math.round(state.width)}x${Math.round(state.height)}</span></div>
            <div class="perf-row"><span class="perf-label">Canvas Px</span><span class="perf-value">${canvasPx}</span></div>
            <div class="perf-row"><span class="perf-label">World</span><span class="perf-value">${state.worldWidth}x${state.worldHeight}</span></div>
            ${memory !== null ? `<div class="perf-row"><span class="perf-label">Memory</span><span class="perf-value">${memory.toFixed(1)} MB</span></div>` : ""}
          </div>
        `;
      }

      function renderPerformanceOverlays(now, force = false) {
        const perfMode = getPerfMode();
        const showBasic = perfMode === "basic";
        const showDetail = perfMode === "detail";
        const showSettings = !!(settingsPanel && !settingsPanel.hidden);
        const allowSettingsMetrics = perfMode !== "off";
        if (!showBasic && !showDetail && !showSettings) return;
        if (!force && now - perfMetrics.lastOverlayUpdate < 250) return;
        perfMetrics.lastOverlayUpdate = now;
        const basicHtml = buildPerformanceBasicHtml();
        const detailHtml = (showDetail || (showSettings && perfMode === "detail"))
          ? buildPerformanceDetailHtml()
          : "";

        if (dom.perfOverlay) {
          if (showBasic) {
            dom.perfOverlay.hidden = false;
            dom.perfOverlay.innerHTML = basicHtml;
          } else {
            dom.perfOverlay.hidden = true;
          }
        }

        if (dom.perfDetailOverlay) {
          if (showDetail) {
            dom.perfDetailOverlay.hidden = false;
            dom.perfDetailOverlay.innerHTML = detailHtml;
          } else {
            dom.perfDetailOverlay.hidden = true;
          }
        }

        if (showSettings) {
          if (dom.perfSettingsCard) {
            dom.perfSettingsCard.hidden = !allowSettingsMetrics;
          }
          if (dom.perfSettingsBasic) {
            dom.perfSettingsBasic.hidden = perfMode !== "basic";
            dom.perfSettingsBasic.innerHTML = perfMode === "basic" ? basicHtml : "";
          }
          if (dom.perfSettingsDetail) {
            dom.perfSettingsDetail.hidden = perfMode !== "detail";
            dom.perfSettingsDetail.innerHTML = perfMode === "detail" ? detailHtml : "";
          }
        }
      }

      function recordPerformance(frameStart, updateMs, renderMs, hudMs, rafDeltaMs, deltaMs) {
        const perfMode = getPerfMode();
        const shouldSample = perfMode !== "off" || perfLog.active;
        if (!shouldSample) return;
        const now = performance.now();
        const idleMs = perfMetrics.lastFrameEnd ? Math.max(0, frameStart - perfMetrics.lastFrameEnd) : 0;
        const frameMs = now - frameStart;
        const rafIntervalMs = Math.max(0, rafDeltaMs || 0);
        const busyRatio = rafIntervalMs ? frameMs / rafIntervalMs : 0;
        const smoothing = 0.12;
        perfMetrics.frameMs = frameMs;
        perfMetrics.updateMs = updateMs;
        perfMetrics.renderMs = renderMs;
        perfMetrics.hudMs = hudMs;
        perfMetrics.rafIntervalMs = rafIntervalMs;
        perfMetrics.idleMs = idleMs;
        perfMetrics.busyRatio = busyRatio;
        perfMetrics.deltaMs = deltaMs;
        perfMetrics.avgFrameMs = perfMetrics.avgFrameMs
          ? perfMetrics.avgFrameMs * (1 - smoothing) + frameMs * smoothing
          : frameMs;
        perfMetrics.avgUpdateMs = perfMetrics.avgUpdateMs
          ? perfMetrics.avgUpdateMs * (1 - smoothing) + updateMs * smoothing
          : updateMs;
        perfMetrics.avgRenderMs = perfMetrics.avgRenderMs
          ? perfMetrics.avgRenderMs * (1 - smoothing) + renderMs * smoothing
          : renderMs;
        perfMetrics.avgHudMs = perfMetrics.avgHudMs
          ? perfMetrics.avgHudMs * (1 - smoothing) + hudMs * smoothing
          : hudMs;
        perfMetrics.avgRafIntervalMs = perfMetrics.avgRafIntervalMs
          ? perfMetrics.avgRafIntervalMs * (1 - smoothing) + rafIntervalMs * smoothing
          : rafIntervalMs;
        perfMetrics.avgIdleMs = perfMetrics.avgIdleMs
          ? perfMetrics.avgIdleMs * (1 - smoothing) + idleMs * smoothing
          : idleMs;
        perfMetrics.avgBusyRatio = perfMetrics.avgBusyRatio
          ? perfMetrics.avgBusyRatio * (1 - smoothing) + busyRatio * smoothing
          : busyRatio;
        perfMetrics.avgDeltaMs = perfMetrics.avgDeltaMs
          ? perfMetrics.avgDeltaMs * (1 - smoothing) + deltaMs * smoothing
          : deltaMs;
        perfMetrics.maxFrameMs = Math.max(perfMetrics.maxFrameMs, frameMs);
        perfMetrics.maxRafIntervalMs = Math.max(perfMetrics.maxRafIntervalMs, rafIntervalMs);
        perfMetrics.frames += 1;
        if (rafIntervalMs >= 33) perfMetrics.slowFrames += 1;
        if (rafIntervalMs >= 50) perfMetrics.hitchFrames += 1;
        if (now - perfMetrics.lastFpsAt >= 500) {
          const windowSeconds = (now - perfMetrics.lastFpsAt) / 1000;
          perfMetrics.fps = perfMetrics.frames / windowSeconds;
          perfMetrics.slowRate = perfMetrics.slowFrames / windowSeconds;
          perfMetrics.hitchRate = perfMetrics.hitchFrames / windowSeconds;
          perfMetrics.frames = 0;
          perfMetrics.lastFpsAt = now;
          perfMetrics.maxFrameMs = 0;
          perfMetrics.maxRafIntervalMs = 0;
          perfMetrics.slowFrames = 0;
          perfMetrics.hitchFrames = 0;
        }
        perfMetrics.lastFrameEnd = now;
        appendPerfLogSample(now);
        renderPerformanceOverlays(now);
      }

      function updatePerfLogUi() {
        if (dom.perfLogBtn) {
          dom.perfLogBtn.textContent = perfLog.active ? "Stop & Download Log" : "Start Perf Log";
          dom.perfLogBtn.setAttribute("aria-pressed", perfLog.active ? "true" : "false");
        }
        if (dom.perfLogStatus) {
          if (perfLog.active) {
            const elapsed = Math.max(0, performance.now() - perfLog.startAt);
            const seconds = Math.floor(elapsed / 1000);
            dom.perfLogStatus.textContent = `Recording ${seconds}s • ${perfLog.samples} samples`;
          } else {
            dom.perfLogStatus.textContent = "Idle";
          }
        }
      }

      function togglePerfLog() {
        if (perfLog.active) {
          stopPerfLog();
        } else {
          startPerfLog();
        }
      }

      function startPerfLog() {
        perfLog.active = true;
        perfLog.entries = [];
        perfLog.samples = 0;
        perfLog.startAt = performance.now();
        perfLog.lastSampleAt = 0;
        const startedAt = new Date().toISOString();
        perfLog.entries.push("# Stellar Dogfight Performance Log");
        perfLog.entries.push(`# Started: ${startedAt}`);
        perfLog.entries.push(`# Perf Mode: ${getPerfMode()}`);
        perfLog.entries.push(`# Sample Interval: ${perfLog.sampleIntervalMs} ms`);
        perfLog.lastLongTaskCount = longTaskMetrics.count;
        perfLog.lastLongTaskDuration = longTaskMetrics.totalDuration;
        perfLog.entries.push("t_ms,fps,frame_ms,update_ms,render_ms,hud_ms,raf_ms,idle_ms,busy_pct,delta_ms,enemies,bullets,particles,pulses,blackholes,mines,helpers,obstacles,stars,longtasks_total,longtasks_delta,longtask_ms_total,longtask_ms_delta,mode,visibility,focus,dpr,render_scale,canvas_w,canvas_h,canvas_px_w,canvas_px_h,world_w,world_h,memory_mb");
        updatePerfLogUi();
        logEvent("Performance log recording started.");
      }

      function stopPerfLog() {
        perfLog.active = false;
        updatePerfLogUi();
        if (!perfLog.entries.length) return;
        downloadPerfLog();
        logEvent("Performance log saved.");
      }

      function appendPerfLogSample(now) {
        if (!perfLog.active) return;
        if (now - perfLog.lastSampleAt < perfLog.sampleIntervalMs) return;
        perfLog.lastSampleAt = now;
        perfLog.samples += 1;
        const elapsedMs = Math.max(0, now - perfLog.startAt);
        const memory = performance && performance.memory
          ? (performance.memory.usedJSHeapSize / (1024 * 1024))
          : null;
        const visibility = document.visibilityState || "unknown";
        const focusState = typeof document.hasFocus === "function"
          ? (document.hasFocus() ? "yes" : "no")
          : "n/a";
        const dpr = window.devicePixelRatio || 1;
        const renderScale = state.renderScale || dpr;
        const canvasPxW = dom.canvas ? dom.canvas.width : 0;
        const canvasPxH = dom.canvas ? dom.canvas.height : 0;
        const longTaskCount = longTaskMetrics.count;
        const longTaskDuration = longTaskMetrics.totalDuration;
        const longTaskDelta = Math.max(0, longTaskCount - perfLog.lastLongTaskCount);
        const longTaskDurationDelta = Math.max(0, longTaskDuration - perfLog.lastLongTaskDuration);
        const line = [
          formatPerfLogNumber(elapsedMs, 0),
          formatPerfLogNumber(perfMetrics.fps, 1),
          formatPerfLogNumber(perfMetrics.frameMs, 2),
          formatPerfLogNumber(perfMetrics.updateMs, 2),
          formatPerfLogNumber(perfMetrics.renderMs, 2),
          formatPerfLogNumber(perfMetrics.hudMs, 2),
          formatPerfLogNumber(perfMetrics.rafIntervalMs, 2),
          formatPerfLogNumber(perfMetrics.idleMs, 2),
          formatPerfLogNumber(perfMetrics.busyRatio * 100, 1),
          formatPerfLogNumber(perfMetrics.deltaMs, 2),
          enemies.length,
          bullets.length,
          particles.length,
          pulses.length,
          state.blackHoles ? state.blackHoles.length : 0,
          state.mines.length,
          helpers.length,
          obstacles.length,
          stars.length,
          longTaskCount,
          longTaskDelta,
          formatPerfLogNumber(longTaskDuration, 2),
          formatPerfLogNumber(longTaskDurationDelta, 2),
          state.mode,
          visibility,
          focusState,
          formatPerfLogNumber(dpr, 2),
          formatPerfLogNumber(renderScale, 2),
          Math.round(state.width),
          Math.round(state.height),
          canvasPxW,
          canvasPxH,
          state.worldWidth,
          state.worldHeight,
          memory !== null ? memory.toFixed(1) : ""
        ].join(",");
        perfLog.lastLongTaskCount = longTaskCount;
        perfLog.lastLongTaskDuration = longTaskDuration;
        perfLog.entries.push(line);
        if (settingsPanel && !settingsPanel.hidden) {
          updatePerfLogUi();
        }
      }

      function downloadPerfLog() {
        const content = perfLog.entries.join("\n");
        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `stellar-dogfight-perf-${formatPerfLogTimestamp(new Date())}.txt`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }

      function formatPerfLogTimestamp(date) {
        const pad = (value) => value.toString().padStart(2, "0");
        return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
      }

      function renderActiveUpgrades() {
        const upgradeKey = getActiveUpgradeKey();
        if (upgradeKey === state.activeUpgradeKey) {
          return;
        }
        state.activeUpgradeKey = upgradeKey;
        if (isFrontierMode()) {
          if (!state.frontier || !state.frontier.active) {
            dom.activeUpgrades.innerHTML = "<span class=\"chip\">Frontier upgrades available at the dock</span>";
            return;
          }
          const entries = FRONTIER_UPGRADES.filter((upgrade) => state.frontier.upgrades[upgrade.id]);
          if (!entries.length) {
            dom.activeUpgrades.innerHTML = "<span class=\"chip\">No frontier upgrades yet</span>";
            return;
          }
          dom.activeUpgrades.innerHTML = entries.map((upgrade) => {
            const level = state.frontier.upgrades[upgrade.id] || 0;
            const tier = upgrade.tier || "common";
            return `<span class=\"chip\" data-tier="${tier}">${upgrade.name} Lv ${level}</span>`;
          }).join("");
          return;
        }
        if (!isFeatureUnlocked("upgrades")) {
          dom.activeUpgrades.innerHTML = "<span class=\"chip\">Field upgrades locked</span>";
          return;
        }
        const entries = FIELD_UPGRADES.filter((upgrade) => state.upgradeStacks[upgrade.id]);
        const skillEntries = entries.filter((upgrade) => isSkillUpgrade(upgrade));
        const statEntries = entries.filter((upgrade) => !isSkillUpgrade(upgrade));
        const renderChips = (list) => list.map((upgrade) => {
          const stack = state.upgradeStacks[upgrade.id] || 0;
          const isSkill = isSkillUpgrade(upgrade);
          const label = isSkill ? `${upgrade.name} Lv ${stack}` : (stack > 1 ? `${upgrade.name} x${stack}` : upgrade.name);
          const tier = getUpgradeTier(upgrade, stack);
          return `<span class=\"chip\" data-tier="${tier}">${label}</span>`;
        }).join("");
        const skillHtml = `
          <div class="upgrade-group">
            <span class="upgrade-group-title">Skills (${state.skillSlots.length}/${SKILL_LIMIT})</span>
            <div class="chip-list">
              ${skillEntries.length ? renderChips(skillEntries) : "<span class=\"chip\">No skills yet</span>"}
            </div>
          </div>
        `;
        const statHtml = `
          <div class="upgrade-group">
            <span class="upgrade-group-title">Boosts</span>
            <div class="chip-list">
              ${statEntries.length ? renderChips(statEntries) : "<span class=\"chip\">No boosts yet</span>"}
            </div>
          </div>
        `;
        dom.activeUpgrades.innerHTML = `${skillHtml}${statHtml}`;
      }

      function getActiveUpgradeKey() {
        if (isFrontierMode()) {
          if (!state.frontier || !state.frontier.upgrades) return "frontier:none";
          const entries = Object.entries(state.frontier.upgrades)
            .filter(([, level]) => level > 0)
            .sort((a, b) => a[0].localeCompare(b[0]))
            .map(([id, level]) => `${id}:${level}`)
            .join("|");
          return `frontier:${entries}`;
        }
        const stacks = Object.entries(state.upgradeStacks || {})
          .filter(([, count]) => count > 0)
          .sort((a, b) => a[0].localeCompare(b[0]))
          .map(([id, count]) => `${id}:${count}`)
          .join("|");
        return `arcade:${state.skillSlots.join(",")}:${stacks}`;
      }

      function renderHangar() {
        if (!isFeatureUnlocked("hangar")) {
          dom.hangar.innerHTML = renderLockedCard("Hangar locked", getFeatureHint("hangar"));
          return;
        }
        const grouped = {};
        HANGAR_UPGRADES.forEach((upgrade) => {
          const category = upgrade.category || "Utility";
          if (!grouped[category]) grouped[category] = [];
          grouped[category].push(upgrade);
        });
        const categoryOrder = ["Defense", "Offense", "Mobility", "Utility", "Strategy", "Systems", "Control"];
        const sortedCategories = Object.keys(grouped).sort((a, b) => {
          const aIndex = categoryOrder.indexOf(a);
          const bIndex = categoryOrder.indexOf(b);
          if (aIndex === -1 && bIndex === -1) return a.localeCompare(b);
          if (aIndex === -1) return 1;
          if (bIndex === -1) return -1;
          return aIndex - bIndex;
        });
        const activePath = getActiveBuildPath();
        const buildPathSection = `
          <div class="tier-section">
            <div class="tier-header">
              <span class="tier-title">Build Identity</span>
              <span class="tier-count">${activePath ? activePath.name : "Balanced Command"}</span>
            </div>
            <div class="card-grid">
              ${BUILD_PATHS.map((path) => `
                <button class="select-card ${progress.buildPath === path.id ? "is-active" : ""}" data-build-path="${path.id}">
                  <span class="select-title">${path.name}</span>
                  <span class="select-meta">${path.desc}</span>
                  <div class="select-pills">
                    <span class="select-pill">${progress.buildPath === path.id ? "Active" : "Select"}</span>
                    <span class="select-pill">Exclusive Keystone</span>
                  </div>
                </button>
              `).join("")}
            </div>
          </div>
        `;
        dom.hangar.innerHTML = buildPathSection + sortedCategories.map((category) => {
          const items = grouped[category] || [];
          items.sort((a, b) => {
            const tierA = a.tier || "common";
            const tierB = b.tier || "common";
            const tierIndexA = TIER_ORDER.indexOf(tierA);
            const tierIndexB = TIER_ORDER.indexOf(tierB);
            if (tierIndexA !== tierIndexB) return tierIndexA - tierIndexB;
            return a.name.localeCompare(b.name);
          });
          const categoryIcon = getUpgradeCategoryIcon(category);
          const categoryKey = normalizeCategory(category);
          return `
            <div class="tier-section">
              <div class="tier-header">
                <span class="tier-title">
                  <span class="select-pill category-pill" data-category="${categoryKey}" aria-label="${category}" title="${category}">${categoryIcon}</span>
                  ${category}
                </span>
                <span class="tier-count">${items.length} upgrade${items.length === 1 ? "" : "s"}</span>
              </div>
              <div class="card-grid">
                ${items.map((upgrade) => {
                  const level = progress.hangar[upgrade.id] || 0;
                  const maxLevel = upgrade.maxLevel || 1;
                  const tier = upgrade.tier || "common";
                  const tierLabel = formatTierLabel(tier);
                  const cost = getHangarUpgradeCost(upgrade, level);
                  const canBuy = level < maxLevel && progress.techPoints >= cost;
                  const costLabel = level >= maxLevel ? "Maxed" : `Cost ${cost} tech`;
                  const glyphs = renderUpgradeGlyphs(getUpgradeGlyphs(upgrade, Math.min(level + 1, maxLevel)));
                  const preview = getHangarPreviewText(upgrade, level);
                  return `
                    <button class="hangar-card" data-tier="${tier}" data-hangar-id="${upgrade.id}" ${canBuy ? "" : "disabled"}>
                      <span class="hangar-title">${upgrade.name}</span>
                      <span class="hangar-meta">${preview}</span>
                      <div class="upgrade-glyphs">${glyphs}</div>
                      <div class="select-pills">
                        <span class="select-pill tier-pill" data-tier="${tier}">${tierLabel}</span>
                        <span class="select-pill">Level ${level}/${maxLevel}</span>
                        <span class="select-pill">${costLabel}</span>
                      </div>
                    </button>
                  `;
                }).join("")}
              </div>
            </div>
          `;
        }).join("");
      }

      function renderShipyard() {
        if (!dom.shipyard) return;
        if (!isFeatureUnlocked("shipyard")) {
          dom.shipyard.innerHTML = renderLockedCard("Shipyard locked", getFeatureHint("shipyard"));
          return;
        }
        const tierGroup = document.querySelector("[data-setting='shipyard-tier']");
        let tierFilter = progress.settings.shipyardTier || "all";
        if (tierGroup && !tierGroup.querySelector(`[data-option="${tierFilter}"]`)) {
          tierFilter = "all";
          progress.settings.shipyardTier = "all";
          saveProgress();
        }
        if (tierGroup) {
          tierGroup.querySelectorAll(".option-btn").forEach((btn) => {
            btn.classList.toggle("is-active", btn.dataset.option === tierFilter);
          });
        }
        const filteredShips = tierFilter === "all"
          ? SHIPS
          : SHIPS.filter((ship) => String(Number(ship.tier) || 1) === tierFilter);
        const renderShipCard = (ship) => {
          const unlocked = !!progress.shipUnlocks[ship.id];
          const selected = progress.selectedShip === ship.id;
          const unlockable = canUnlock(ship, progress.shipUnlocks);
          const disabled = !unlocked && !unlockable ? "disabled" : "";
          const statusLabel = unlocked ? (selected ? "Active" : "Select") : (unlockable ? "Unlock" : "Locked");
          const unlockText = formatUnlockText(ship.unlock);
          const signatureWeapon = ship.signatureWeapon ? getWeaponById(ship.signatureWeapon).name : "";
          const shipTierValue = Number(ship.tier) || 1;
          const tierStyle = getShipTierStyle(shipTierValue);
          const tierPill = ship.tier ? `<span class="select-pill tier-pill" data-tier="${tierStyle}">Tier ${shipTierValue}</span>` : "";
          const tierAttr = ship.tier ? `data-tier="${tierStyle}"` : "";
          return `
            <button class="select-card ${selected ? "is-active" : ""}" ${tierAttr} data-ship-id="${ship.id}" ${disabled}>
              <span class="select-title">${ship.name}</span>
              <span class="select-meta">${ship.desc}</span>
              ${signatureWeapon ? `<span class="select-meta">Signature: ${signatureWeapon}</span>` : ""}
              <span class="select-meta">Passive: ${ship.passive}</span>
              <div class="select-pills">
                ${tierPill}
                <span class="select-pill">${ABILITIES[ship.abilityId].name}</span>
                <span class="select-pill">${statusLabel}</span>
                ${unlockText ? `<span class="select-pill">${unlockText}</span>` : ""}
              </div>
            </button>
          `;
        };
        if (!filteredShips.length) {
          dom.shipyard.innerHTML = "<span class=\"select-meta\">No ships available for this tier yet.</span>";
          return;
        }
        if (tierFilter === "all") {
          const shipsByTier = {};
          filteredShips.forEach((ship) => {
            const shipTierValue = Number(ship.tier) || 1;
            if (!shipsByTier[shipTierValue]) shipsByTier[shipTierValue] = [];
            shipsByTier[shipTierValue].push(ship);
          });
          const tiers = Object.keys(shipsByTier).map(Number).sort((a, b) => a - b);
          dom.shipyard.innerHTML = tiers.map((tier) => {
            const items = shipsByTier[tier] || [];
            return `
              <div class="tier-section">
                <div class="tier-header">
                  <span class="tier-title">Tier ${tier}</span>
                  <span class="tier-count">${items.length} ship${items.length === 1 ? "" : "s"}</span>
                </div>
                <div class="card-grid">
                  ${items.map(renderShipCard).join("")}
                </div>
              </div>
            `;
          }).join("");
          return;
        }
        dom.shipyard.innerHTML = filteredShips.map(renderShipCard).join("");
      }

      function renderArmory() {
        if (!dom.inventory) return;
        if (!isFeatureUnlocked("armory")) {
          dom.inventory.innerHTML = renderLockedCard("Gear locked", getFeatureHint("armory"));
          if (dom.equippedItems) {
            dom.equippedItems.innerHTML = "";
          }
          dom.secondaries.innerHTML = renderLockedCard("Secondaries locked", getFeatureHint("secondary"));
          renderSalvage();
          return;
        }
        renderInventory();
        renderSecondaries();
        renderSalvage();
      }

      function renderInventory() {
        if (!dom.inventory) return;
        const filterGroup = document.querySelector("[data-setting='inventory-filter']");
        const filter = progress.settings.inventoryFilter || "all";
        if (filterGroup) {
          filterGroup.querySelectorAll(".option-btn").forEach((btn) => {
            btn.classList.toggle("is-active", btn.dataset.option === filter);
          });
        }

        const equippedWeapon = getEquippedWeapon();
        const equippedCards = [
          renderEquippedItemCard(equippedWeapon, "Equipped Weapon")
        ];
        PART_SLOTS.forEach((slot) => {
          const attachment = getEquippedAttachment(slot);
          const label = `${slot.charAt(0).toUpperCase() + slot.slice(1)} Slot`;
          equippedCards.push(renderEquippedItemCard(attachment, label, slot));
        });
        if (dom.equippedItems) {
          dom.equippedItems.innerHTML = equippedCards.join("");
        }

        if (!Array.isArray(progress.inventory) || !progress.inventory.length) {
          state.inventorySelectionId = null;
          dom.inventory.innerHTML = "<span class=\"select-meta\">No gear collected yet.</span>";
          return;
        }
        const filteredItems = progress.inventory.filter((item) => {
          if (!item) return false;
          if (!item.type) return false;
          if (filter === "weapon") return item.type === "weapon";
          if (filter === "attachment") return item.type === "attachment";
          return true;
        });
        if (!filteredItems.length) {
          state.inventorySelectionId = null;
          dom.inventory.innerHTML = "<span class=\"select-meta\">No matching items.</span>";
          return;
        }
        const sortedItems = sortInventoryItems(filteredItems);
        let selectedItem = sortedItems.find((item) => item.id === state.inventorySelectionId) || null;
        if (!selectedItem) {
          selectedItem = sortedItems[0];
          state.inventorySelectionId = selectedItem ? selectedItem.id : null;
        }
        const grid = sortedItems.map((item) => renderInventoryCell(item)).join("");
        const detail = selectedItem
          ? renderInventoryDetail(selectedItem)
          : "<div class=\"inventory-detail empty\">Select a piece of gear to inspect and equip.</div>";
        dom.inventory.innerHTML = `
          <div class="inventory-grid">
            ${grid}
          </div>
          ${detail}
        `;
      }

      function getInventoryCellIcon(item) {
        if (!item) return "G";
        if (item.type === "weapon") return "WPN";
        if (item.type === "attachment") {
          const slotIcons = {
            barrel: "BAR",
            core: "CORE",
            targeting: "TRG",
            thruster: "THR"
          };
          return slotIcons[item.slot] || "MOD";
        }
        return "GEAR";
      }

      function getInventoryCellLabel(item) {
        if (!item) return "Gear";
        if (item.type === "weapon") return "Weapon";
        if (item.type === "attachment") {
          const slotLabels = {
            barrel: "Barrel",
            core: "Core",
            targeting: "Target",
            thruster: "Thruster"
          };
          return slotLabels[item.slot] || "Module";
        }
        return "Gear";
      }

      function getInventoryTypeLabel(item) {
        if (!item) return "Item";
        if (item.type === "weapon") return "Weapon";
        if (item.type === "attachment") {
          return `${item.slot ? item.slot.charAt(0).toUpperCase() + item.slot.slice(1) : "Attachment"} Attachment`;
        }
        return "Item";
      }

      function sortInventoryItems(items) {
        return items.slice().sort((a, b) => {
          const tierDiff = getTierIndex(b.tier || "common") - getTierIndex(a.tier || "common");
          if (tierDiff !== 0) return tierDiff;
          const typeDiff = (a.type || "").localeCompare(b.type || "");
          if (typeDiff !== 0) return typeDiff;
          return (a.name || "").localeCompare(b.name || "");
        });
      }

      function renderInventoryCell(item) {
        const tier = item.tier || "common";
        const equipped = isItemEquipped(item);
        const isSelected = item.id === state.inventorySelectionId;
        const icon = getInventoryCellIcon(item);
        const label = getInventoryCellLabel(item);
        const tooltip = buildItemTooltip(item);
        const upgrades = `${item.upgradeSuccesses || 0}/${item.upgradeSlots || 0}`;
        return `
          <button class="inventory-cell ${isSelected ? "is-selected" : ""} ${equipped ? "is-equipped" : ""}" data-tier="${tier}" data-item-id="${item.id}" data-inventory-select="${item.id}" ${tooltip ? `title="${escapeAttribute(tooltip)}"` : ""}>
            <span class="inventory-icon">${icon}</span>
            <span class="inventory-slot">${label}</span>
            <span class="inventory-upgrades">Upg ${upgrades}</span>
          </button>
        `;
      }

      function renderInventoryDetail(item) {
        const tier = item.tier || "common";
        const tierLabel = formatTierLabel(tier);
        const summary = getItemSummary(item);
        const typeLabel = getInventoryTypeLabel(item);
        const equipped = isItemEquipped(item);
        const comparison = item.type === "weapon"
          ? buildWeaponComparison(getEquippedWeapon(), item)
          : buildAttachmentComparison(item, getEquippedAttachment(item.slot));
        const compareBlock = equipped
          ? "<div class=\"compare-list\"><span class=\"stat-diff is-neutral\">Equipped</span></div>"
          : (comparison ? `<div class="compare-list">${comparison}</div>` : "");
        const upgradeChance = getUpgradeSuccessChance(item);
        const upgradeLabel = item.upgradeSuccesses >= item.upgradeSlots
          ? "Maxed"
          : `Upgrade ${Math.round(upgradeChance * 100)}%`;
        const canUpgrade = canUpgradeItem(item);
        const canSell = canSellItem(item);
        const affixPills = (item.affixes || []).map((affix) => `<span class="select-pill">${affix.label}</span>`).join("");
        return `
          <div class="inventory-detail" data-tier="${tier}" data-item-id="${item.id}">
            <div class="inventory-detail-header">
              <div>
                <div class="inventory-detail-title">${item.name}</div>
                <div class="inventory-detail-meta">${typeLabel} · ${summary}</div>
              </div>
              <span class="select-pill tier-pill" data-tier="${tier}">${tierLabel}</span>
            </div>
            ${compareBlock}
            ${affixPills ? `<div class="select-pills">${affixPills}</div>` : ""}
            <div class="inventory-actions">
              <button class="btn ghost btn-mini" type="button" data-item-action="equip" ${equipped ? "disabled" : ""}>
                ${equipped ? "Equipped" : "Equip"}
              </button>
              <button class="btn ghost btn-mini" type="button" data-item-action="upgrade" ${canUpgrade ? "" : "disabled"}>
                ${upgradeLabel}
              </button>
              <button class="btn ghost btn-mini" type="button" data-item-action="sell" ${canSell ? "" : "disabled"}>
                Sell ${getItemSellValue(item)}c
              </button>
            </div>
          </div>
        `;
      }

      function renderSecondaries() {
        if (!dom.secondaries) return;
        if (!isFeatureUnlocked("secondary")) {
          dom.secondaries.innerHTML = renderLockedCard("Secondaries locked", getFeatureHint("secondary"));
          return;
        }
        const currentSecondary = getSecondaryById(progress.selectedSecondary);
        const secondaryGroups = groupByTier(SECONDARIES);
        dom.secondaries.innerHTML = secondaryGroups.map((group) => {
          return renderTierSection(group.tier, group.items, (item) => {
            const unlocked = !!progress.secondaryUnlocks[item.id];
            const selected = progress.selectedSecondary === item.id;
            const unlockable = canUnlock(item, progress.secondaryUnlocks);
            const disabled = !unlocked && !unlockable ? "disabled" : "";
            const statusLabel = unlocked ? (selected ? "Equipped" : "Select") : (unlockable ? "Unlock" : "Locked");
            const unlockText = formatUnlockText(item.unlock);
            const tier = item.tier || "common";
            const tierLabel = formatTierLabel(tier);
            const comparison = selected ? "" : buildSecondaryComparison(currentSecondary, item);
            const compareBlock = comparison ? `<div class="compare-list">${comparison}</div>` : "";
            return `
              <button class="select-card ${selected ? "is-active" : ""}" data-tier="${tier}" data-secondary-id="${item.id}" ${disabled}>
                <span class="select-title">${item.name}</span>
                <span class="select-meta">${item.desc}</span>
                ${compareBlock}
                <div class="select-pills">
                  <span class="select-pill tier-pill" data-tier="${tier}">${tierLabel}</span>
                  <span class="select-pill">Cooldown ${item.cooldown}s</span>
                  <span class="select-pill">${statusLabel}</span>
                  ${unlockText ? `<span class="select-pill">${unlockText}</span>` : ""}
                </div>
              </button>
            `;
          });
        }).join("");
      }

      function renderEquippedItemCard(item, label, slot) {
        const hasItem = !!item;
        const tier = hasItem ? (item.tier || "common") : "common";
        const tierLabel = hasItem ? formatTierLabel(tier) : "";
        const summary = hasItem ? getItemSummary(item) : "Empty slot";
        const tag = hasItem
          ? (item.type === "attachment" ? `${slot.charAt(0).toUpperCase() + slot.slice(1)} Attachment` : "Weapon")
          : "Empty";
        const tooltip = hasItem ? buildItemTooltip(item) : "";
        return `
          <div class="select-card is-static" ${hasItem ? `data-tier="${tier}"` : ""} ${tooltip ? `title="${escapeAttribute(tooltip)}"` : ""}>
            <span class="select-title">${label}</span>
            <span class="select-meta">${hasItem ? item.name : "No item equipped"}</span>
            <span class="select-meta">${summary}</span>
            <div class="select-pills">
              ${hasItem ? `<span class="select-pill tier-pill" data-tier="${tier}">${tierLabel}</span>` : ""}
              <span class="select-pill">${tag}</span>
              ${hasItem ? `<span class="select-pill">Upgrades ${item.upgradeSuccesses || 0}/${item.upgradeSlots || 0}</span>` : ""}
              ${hasItem ? (item.affixes || []).slice(0, 2).map((affix) => `<span class="select-pill">${affix.label}</span>`).join("") : ""}
            </div>
          </div>
        `;
      }

      function getItemSummary(item) {
        if (!item) return "";
        const stats = item.type === "weapon" ? getWeaponItemStats(item) : getAttachmentStats(item);
        if (!stats) return "";
        if (item.type === "weapon") {
          const parts = [
            `DMG ${Math.round(stats.damage)}`,
            `FR ${stats.fireRate.toFixed(1)}`,
            `EN ${stats.energyCost.toFixed(1)}`
          ];
          if (stats.projectiles > 1) parts.push(`Proj ${stats.projectiles}`);
          if (stats.pierce > 0) parts.push(`Pierce ${stats.pierce}`);
          if (stats.splashRadius > 0) parts.push(`Splash ${Math.round(stats.splashRadius)}`);
          if (stats.arcChains > 0) parts.push(`Arc ${stats.arcChains}`);
          return parts.join(" · ");
        }
        const entries = Object.keys(stats)
          .filter((key) => Number.isFinite(stats[key]) && stats[key] !== 0)
          .map((key) => formatStat(key, stats[key]));
        return entries.slice(0, 3).join(" · ");
      }

      function buildItemTooltip(item) {
        if (!item) return "";
        const stats = item.type === "weapon" ? getWeaponItemStats(item) : getAttachmentStats(item);
        const lines = [];
        const tierLabel = formatTierLabel(item.tier || "common");
        lines.push(`${tierLabel} ${item.name}`);
        if (item.type === "attachment" && item.slot) {
          lines.push(`Slot: ${item.slot.charAt(0).toUpperCase() + item.slot.slice(1)}`);
        }
        if (stats) {
          lines.push(buildStatSummary(stats));
        }
        if (item.affixes && item.affixes.length) {
          item.affixes.forEach((affix) => {
            if (affix.desc) {
              lines.push(`${affix.label}: ${affix.desc}`);
            } else {
              lines.push(affix.label);
            }
          });
        }
        lines.push(`Upgrades: ${item.upgradeSuccesses || 0}/${item.upgradeSlots || 0}`);
        return lines.filter(Boolean).join("\n");
      }

      function escapeAttribute(value) {
        return String(value)
          .replace(/&/g, "&amp;")
          .replace(/"/g, "&quot;");
      }

      function isItemEquipped(item) {
        if (!item) return false;
        if (item.type === "weapon") {
          return progress.equipped?.weaponId === item.id;
        }
        if (item.type === "attachment") {
          return progress.equipped?.attachments?.[item.slot] === item.id;
        }
        return false;
      }

      function getUpgradeMaterial(item) {
        if (!item || !Array.isArray(progress.inventory)) return null;
        let pool = progress.inventory.filter((candidate) => {
          if (!candidate || candidate.id === item.id) return false;
          if (candidate.type !== item.type) return false;
          if (isItemEquipped(candidate)) return false;
          if (item.type === "attachment" && candidate.slot !== item.slot) return false;
          return true;
        });
        if (item.type === "weapon") {
          const sameTemplate = pool.filter((candidate) => candidate.templateId === item.templateId);
          if (sameTemplate.length) {
            pool = sameTemplate;
          }
        }
        if (!pool.length) return null;
        pool.sort((a, b) => {
          const tierDiff = getTierIndex(a.tier) - getTierIndex(b.tier);
          if (tierDiff !== 0) return tierDiff;
          return (a.upgradeSuccesses || 0) - (b.upgradeSuccesses || 0);
        });
        return pool[0];
      }

      function getUpgradeSuccessChance(item) {
        if (!item) return 0;
        const base = ITEM_UPGRADE_CHANCE[item.tier] || 0.5;
        const penalty = (item.upgradeSuccesses || 0) * 0.08;
        return clamp(base - penalty, 0.1, 0.95);
      }

      function canUpgradeItem(item) {
        if (!item) return false;
        if ((item.upgradeSuccesses || 0) >= (item.upgradeSlots || 0)) return false;
        return !!getUpgradeMaterial(item);
      }

      function canSellItem(item) {
        if (!item) return false;
        if (isItemEquipped(item)) return false;
        if (item.type === "weapon") {
          const weaponCount = progress.inventory.filter((entry) => entry.type === "weapon").length;
          return weaponCount > 1;
        }
        return true;
      }

      function equipItem(item) {
        if (!item) return;
        if (item.type === "weapon") {
          progress.equipped.weaponId = item.id;
          applyLoadoutChange("Weapon equipped. Applies next sortie.");
          return;
        }
        if (item.type === "attachment") {
          progress.equipped.attachments[item.slot] = item.id;
          applyLoadoutChange("Attachment equipped. Applies next sortie.");
        }
      }

      function upgradeItem(item) {
        if (!item) return;
        if ((item.upgradeSuccesses || 0) >= (item.upgradeSlots || 0)) {
          logEvent("Upgrade slots full.");
          return;
        }
        const material = getUpgradeMaterial(item);
        if (!material) {
          logEvent("No compatible item to consume.");
          return;
        }
        const chance = getUpgradeSuccessChance(item);
        progress.inventory = progress.inventory.filter((entry) => entry.id !== material.id);
        const success = Math.random() < chance;
        if (success) {
          item.upgradeSuccesses = (item.upgradeSuccesses || 0) + 1;
          applyLoadoutChange(`Upgrade succeeded: ${item.name} +1.`);
        } else {
          applyLoadoutChange(`Upgrade failed: ${item.name} resisted the fusion.`);
        }
      }

      function sellItem(item) {
        if (!item || !canSellItem(item)) return;
        const value = getItemSellValue(item);
        progress.inventory = progress.inventory.filter((entry) => entry.id !== item.id);
        progress.bankedCredits += value;
        applyLoadoutChange(`Sold ${item.name} for ${value} credits.`);
      }

      function getItemSellValue(item) {
        if (!item) return 0;
        const values = {
          common: 80,
          uncommon: 140,
          rare: 220,
          epic: 340,
          legendary: 520
        };
        const base = values[item.tier] || 80;
        const typeBonus = item.type === "weapon" ? 1.1 : 0.9;
        const upgradeBonus = 1 + (item.upgradeSuccesses || 0) * 0.2;
        return Math.round(base * typeBonus * upgradeBonus);
      }

      function renderLockedCard(title, message) {
        return `
          <div class="select-card is-static">
            <span class="select-title">${title}</span>
            <span class="select-meta">${message}</span>
          </div>
        `;
      }

      function renderSalvage() {
        if (!dom.salvage) return;
        if (!isFeatureUnlocked("salvage")) {
          dom.salvage.innerHTML = renderLockedCard("Salvage locked", getFeatureHint("salvage"));
          return;
        }
        const keys = progress.salvageKeys || 0;
        const pityThreshold = SALVAGE_CACHE.pityThreshold;
        const pity = Math.min(progress.salvagePity || 0, pityThreshold);
        const pityText = pity >= pityThreshold ? "Rare guarantee ready" : `Rare guarantee ${pity} / ${pityThreshold}`;
        const odds = getSalvageOdds();
        const oddsPills = odds.map((item) => {
          return `<span class="select-pill">${item.label} ${item.percent}%</span>`;
        }).join("");
        const openOptions = [
          { label: "Open 1", count: 1 },
          { label: "Open 10", count: 10 },
          { label: "Open 100", count: 100 },
          { label: `Open Max (${keys})`, count: "max" }
        ];
        const openButtons = openOptions.map((option) => {
          const isMax = option.count === "max";
          const requiredKeys = isMax ? keys : option.count;
          const canOpen = requiredKeys > 0 && keys >= requiredKeys;
          const label = isMax ? `Open Max (${keys})` : option.label;
          return `
            <button class="btn ghost btn-mini" type="button" data-salvage-action="open" data-salvage-count="${option.count}" ${canOpen ? "" : "disabled"}>
              ${label}
            </button>
          `;
        }).join("");
        const historyHtml = progress.salvageHistory && progress.salvageHistory.length
          ? progress.salvageHistory.map((entry) => `
            <div class="history-entry">
              <strong>${entry.title}</strong>
              <span>${entry.detail}</span>
            </div>
          `).join("")
          : "<span class=\"select-meta\">No caches opened yet.</span>";
        dom.salvage.innerHTML = `
          <div class="select-card is-static">
            <span class="select-title">${SALVAGE_CACHE.name}</span>
            <span class="select-meta">Keys ${keys} • ${pityText}</span>
            <span class="select-meta">Rewards roll on open. No real currency involved.</span>
            <div class="select-pills">
              ${oddsPills}
            </div>
            <div class="salvage-actions">
              ${openButtons}
            </div>
          </div>
          <div class="select-card is-static">
            <span class="select-title">Recent Hauls</span>
            <div class="history-list">
              ${historyHtml}
            </div>
          </div>
        `;
      }

      function getPremiumItemById(id) {
        const items = Array.isArray(PREMIUM_SHOP_ITEMS) ? PREMIUM_SHOP_ITEMS : [];
        return items.find((item) => item && item.id === id) || null;
      }

      function getPremiumItemLevel(item) {
        if (!item) return 0;
        ensurePremiumState();
        if (item.kind === "one-time") {
          return progress.premiumShop.oneTime[item.id] ? 1 : 0;
        }
        return progress.premiumShop.levels[item.id] || 0;
      }

      function getPremiumItemCost(item, level = 0) {
        if (!item) return 0;
        if (item.kind === "one-time") {
          return Math.max(1, Math.round(item.cost || 1));
        }
        const baseCost = Number.isFinite(item.baseCost) ? Math.max(1, item.baseCost) : 1;
        const costScale = Number.isFinite(item.costScale) ? Math.max(1, item.costScale) : 1;
        return Math.max(1, Math.round(baseCost * Math.pow(costScale, level)));
      }

      function buyPremiumItem(itemId) {
        const item = getPremiumItemById(itemId);
        if (!item) return;
        ensurePremiumState();
        const currentLevel = getPremiumItemLevel(item);
        const maxLevel = item.kind === "one-time"
          ? 1
          : (Number.isFinite(item.maxLevel) ? Math.max(1, item.maxLevel) : 1);
        if (currentLevel >= maxLevel) {
          showTip(null, "Premium Shop", "Item already maxed.", { kind: "info", repeatable: true, duration: 3000 });
          return;
        }
        const cost = getPremiumItemCost(item, currentLevel);
        if (progress.premiumCurrency < cost) {
          showTip(null, "Not enough Astralite", `Need ${cost} ${PREMIUM_CURRENCY_LABEL}.`, {
            kind: "lock",
            repeatable: true,
            duration: 3600
          });
          return;
        }

        progress.premiumCurrency = Math.max(0, progress.premiumCurrency - cost);
        if (item.kind === "one-time") {
          progress.premiumShop.oneTime[item.id] = true;
        } else {
          progress.premiumShop.levels[item.id] = Math.min(maxLevel, currentLevel + 1);
        }

        saveProgress();
        renderPremiumShop();
        queueSidebarRefresh();
        renderSettings();
        if (state.mode === "hangar" || state.mode === "gameover" || state.mode === "victory") {
          player = createPlayer();
          updateHud();
        }

        const nextLevel = getPremiumItemLevel(item);
        const rankLabel = item.kind === "one-time" ? "unlocked" : `level ${nextLevel}/${maxLevel}`;
        showTip(null, "Premium upgrade", `${item.name} ${rankLabel}.`, {
          kind: "reward",
          repeatable: true,
          duration: 5200
        });
        logEvent(`Premium shop: ${item.name} ${rankLabel}.`);
      }

      function renderPremiumShop() {
        if (!dom.premiumShop) return;
        const items = Array.isArray(PREMIUM_SHOP_ITEMS) ? PREMIUM_SHOP_ITEMS : [];
        if (!items.length) {
          dom.premiumShop.innerHTML = "<span class=\"select-meta\">Premium shop inventory unavailable.</span>";
          return;
        }
        ensurePremiumState();
        const balance = progress.premiumCurrency || 0;
        const oneTime = items.filter((item) => item.kind === "one-time");
        const scalable = items.filter((item) => item.kind !== "one-time");

        const renderPremiumCard = (item) => {
          const tier = item.tier || "legendary";
          const tierLabel = formatTierLabel(tier);
          const currentLevel = getPremiumItemLevel(item);
          const maxLevel = item.kind === "one-time"
            ? 1
            : (Number.isFinite(item.maxLevel) ? Math.max(1, item.maxLevel) : 1);
          const isMaxed = currentLevel >= maxLevel;
          const cost = getPremiumItemCost(item, currentLevel);
          const canBuy = !isMaxed && balance >= cost;
          const levelPill = item.kind === "one-time"
            ? (isMaxed ? "Owned" : "One-time")
            : `Level ${currentLevel}/${maxLevel}`;
          const actionLabel = isMaxed
            ? "Maxed"
            : item.kind === "one-time"
              ? `Unlock (${cost})`
              : `Upgrade (${cost})`;
          const preview = isMaxed ? "Maxed" : getPremiumPreviewText(item);
          return `
            <div class="select-card premium-card" data-tier="${tier}">
              <span class="select-title">${item.name}</span>
              <span class="select-meta">${item.desc}</span>
              ${item.impact ? `<span class="select-meta">${item.impact}</span>` : ""}
              <span class="select-meta">${preview}</span>
              <div class="select-pills">
                <span class="select-pill tier-pill" data-tier="${tier}">${tierLabel}</span>
                <span class="select-pill">${levelPill}</span>
                <span class="select-pill">${PREMIUM_CURRENCY_LABEL}</span>
              </div>
              <button class="btn ghost btn-mini" type="button" data-premium-buy="${item.id}" ${canBuy ? "" : "disabled"}>
                ${actionLabel}
              </button>
            </div>
          `;
        };

        dom.premiumShop.innerHTML = `
          <div class="select-card is-static">
            <span class="select-title">${PREMIUM_CURRENCY_LABEL}: ${balance.toLocaleString()}</span>
            <span class="select-meta">Premium upgrades are permanent and account-wide.</span>
            <span class="select-meta">Limited combat drops from elite and boss enemies, up to ${PREMIUM_DROP_RUN_CAP} per run.</span>
            <span class="select-meta">Premium pity: ${Math.min(PREMIUM_DROP_PITY_THRESHOLD, progress.premiumDropPity || 0)} / ${PREMIUM_DROP_PITY_THRESHOLD}</span>
          </div>
          <div class="tier-section">
            <div class="tier-header">
              <span class="tier-title">One-time unlocks</span>
              <span class="tier-count">${oneTime.length} unlock${oneTime.length === 1 ? "" : "s"}</span>
            </div>
            <div class="card-grid">
              ${oneTime.map(renderPremiumCard).join("")}
            </div>
          </div>
          <div class="tier-section">
            <div class="tier-header">
              <span class="tier-title">Scalable upgrades</span>
              <span class="tier-count">${scalable.length} track${scalable.length === 1 ? "" : "s"}</span>
            </div>
            <div class="card-grid">
              ${scalable.map(renderPremiumCard).join("")}
            </div>
          </div>
        `;
      }

      function getSalvageOdds() {
        const total = SALVAGE_CACHE.table.reduce((sum, item) => sum + item.weight, 0);
        return SALVAGE_CACHE.table.map((item) => ({
          id: item.id,
          label: item.label,
          percent: Math.round((item.weight / total) * 100)
        }));
      }

      function resolveSalvageOpenCount(token, keys) {
        if (!keys) return 0;
        if (token === "max") return keys;
        const count = parseInt(token, 10);
        if (!Number.isFinite(count) || count <= 0) return 0;
        return Math.min(count, keys);
      }

      function getSalvageRewardTier(reward, rewardDef) {
        if (reward.type === "part" && reward.part) {
          return reward.part.tier || getPartTier(reward.part);
        }
        if (rewardDef && rewardDef.id === "credits-jackpot") return "rare";
        if (rewardDef && rewardDef.id === "blueprints") return "uncommon";
        return "common";
      }

      function buildSalvageRewardResult(reward, rewardDef) {
        let title = rewardDef ? rewardDef.label : "Salvage";
        let amount = Number.isFinite(reward.amount) ? reward.amount : 1;
        let tier = getSalvageRewardTier(reward, rewardDef);
        let detail = "";
        let note = "";
        if (reward.type === "credits") {
          progress.bankedCredits += reward.amount;
          title = "Credits";
          detail = `${reward.amount.toLocaleString()} credits`;
        }
        if (reward.type === "blueprints") {
          progress.blueprints += reward.amount;
          title = "Blueprints";
          detail = `${reward.amount} blueprint${reward.amount === 1 ? "" : "s"}`;
        }
        if (reward.type === "part") {
          const part = reward.part;
          if (part) {
            const stored = addInventoryItem(part, { notify: false });
            title = part.name;
            amount = 1;
            tier = part.tier || getPartTier(part);
            detail = part.name;
            if (!stored) {
              const value = getItemSellValue(part);
              note = `Auto-sold ${value}c`;
            }
          } else {
            title = "Attachment";
            amount = 1;
            detail = "Component fragment";
          }
        }
        const isRare = rewardDef && SALVAGE_CACHE.rareIds.includes(rewardDef.id);
        progress.salvagePity = isRare ? 0 : (progress.salvagePity || 0) + 1;
        recordSalvageHistory({ title: rewardDef ? rewardDef.label : "Salvage", detail });
        return {
          type: reward.type,
          amount,
          tier,
          title,
          desc: detail,
          note
        };
      }

      function openSalvageCaches(token) {
        const keys = progress.salvageKeys || 0;
        const count = resolveSalvageOpenCount(token, keys);
        if (!count) return;
        progress.salvageKeys = Math.max(0, keys - count);
        const rewards = [];
        for (let i = 0; i < count; i += 1) {
          let rewardDef = rollSalvageReward();
          const rareReady = progress.salvagePity >= SALVAGE_CACHE.pityThreshold;
          if (rareReady && !SALVAGE_CACHE.rareIds.includes(rewardDef.id)) {
            const rareTable = SALVAGE_CACHE.table.filter((item) => SALVAGE_CACHE.rareIds.includes(item.id));
            rewardDef = pickWeighted(rareTable, rareTable.map((item) => item.weight));
          }
          const reward = rewardDef.roll();
          const result = buildSalvageRewardResult(reward, rewardDef);
          rewards.push(result);
        }
        saveProgress();
        checkProgressionUnlocks();
        renderShipyard();
        renderPremiumShop();
        renderArmory();
        if (rewards.length) {
          state.salvageResults = {
            count,
            rewards
          };
          setOverlay("salvage");
        }
        logEvent(`Salvage caches opened: ${count}.`);
      }

      function rollSalvageReward() {
        const table = SALVAGE_CACHE.table;
        return pickWeighted(table, table.map((item) => item.weight));
      }

      function recordSalvageHistory(entry) {
        progress.salvageHistory = [entry, ...(progress.salvageHistory || [])].slice(0, 6);
      }

      function normalizePresetSlot(slot) {
        const key = String(slot || "").toLowerCase();
        return ["a", "b", "c"].includes(key) ? key : "";
      }

      function getPresetDisplayLabel(slot) {
        const key = normalizePresetSlot(slot).toUpperCase();
        return key ? `Preset ${key}` : "Preset";
      }

      function captureCurrentLoadoutAsLastRun() {
        const ship = player?.ship || getShipById(progress.selectedShip);
        const secondary = player?.secondary || getSecondaryById(progress.selectedSecondary);
        const weapon = getEquippedWeapon();
        progress.lastLoadout = sanitizeLoadoutPreset({
          shipId: ship ? ship.id : null,
          secondaryId: secondary ? secondary.id : null,
          weaponId: weapon ? weapon.id : null,
          equippedAttachments: {
            ...(progress.equipped?.attachments || {})
          },
          label: `${ship ? ship.name : "Ship"} / ${player?.weapon?.name || weapon?.name || "Weapon"}`
        });
        queueProgressSave();
      }

      function applyLoadoutPreset(preset, title) {
        if (!preset) return false;
        if (preset.shipId && progress.shipUnlocks[preset.shipId]) {
          progress.selectedShip = preset.shipId;
        }
        if (preset.secondaryId && progress.secondaryUnlocks[preset.secondaryId]) {
          progress.selectedSecondary = preset.secondaryId;
        }
        if (preset.weaponId) {
          const weapon = getItemById(preset.weaponId);
          if (weapon && weapon.type === "weapon") {
            progress.equipped.weaponId = weapon.id;
          }
        }
        if (preset.equippedAttachments) {
          PART_SLOTS.forEach((slotKey) => {
            const attachmentId = preset.equippedAttachments[slotKey];
            const attachment = attachmentId ? getItemById(attachmentId) : null;
            progress.equipped.attachments[slotKey] = (attachment && attachment.type === "attachment" && attachment.slot === slotKey)
              ? attachment.id
              : null;
          });
        }
        syncProgressSelections();
        applyLoadoutChange(`${title} loaded. Applies next sortie.`);
        return true;
      }

      function saveLoadoutPreset(slot) {
        const key = normalizePresetSlot(slot);
        if (!key) return;
        const ship = getShipById(progress.selectedShip);
        const secondary = getSecondaryById(progress.selectedSecondary);
        const weapon = getEquippedWeapon();
        progress.loadoutPresets = progress.loadoutPresets || { a: null, b: null, c: null };
        progress.loadoutPresets[key] = sanitizeLoadoutPreset({
          shipId: ship ? ship.id : null,
          secondaryId: secondary ? secondary.id : null,
          weaponId: weapon ? weapon.id : null,
          equippedAttachments: {
            ...(progress.equipped?.attachments || {})
          },
          label: `${ship ? ship.name : "Ship"} / ${weapon ? weapon.name : "Weapon"}`
        });
        saveProgress();
        const title = getPresetDisplayLabel(key);
        showTip(null, title, "Loadout saved.", { kind: "info", repeatable: true, duration: 3600 });
        logEvent(`${title} saved.`);
      }

      function loadLoadoutPreset(slot) {
        const key = normalizePresetSlot(slot);
        if (!key) return;
        progress.loadoutPresets = progress.loadoutPresets || { a: null, b: null, c: null };
        const preset = sanitizeLoadoutPreset(progress.loadoutPresets[key]);
        const title = getPresetDisplayLabel(key);
        if (!preset) {
          showTip(null, title, "No saved loadout in this slot.", { kind: "info", repeatable: true, duration: 3600 });
          return;
        }
        applyLoadoutPreset(preset, title);
      }

      function replayLastLoadout() {
        const preset = sanitizeLoadoutPreset(progress.lastLoadout || null);
        if (!preset) {
          showTip(null, "Replay build", "No previous run build available yet.", {
            kind: "info",
            repeatable: true,
            duration: 3600
          });
          return;
        }
        applyLoadoutPreset(preset, "Last sortie build");
      }

      function applyLoadoutChange(message) {
        saveProgress();
        renderShipyard();
        renderPremiumShop();
        renderArmory();
        renderSettings();
        if (state.mode === "hangar" || state.mode === "gameover") {
          player = createPlayer();
        } else if (message) {
          logEvent(message);
        }
      }

      function renderContracts() {
        if (!dom.contracts) return;
        if (!isFeatureUnlocked("contracts")) {
          dom.contracts.innerHTML = "<span class=\"select-meta\">Tasks locked. " + getFeatureHint("contracts") + "</span>";
          dom.factions.innerHTML = renderLockedCard("Faction intel locked", getFeatureHint("contracts"));
          return;
        }
        const active = state.contracts || [];
        if (!active.length) {
          dom.contracts.innerHTML = "<span class=\"select-meta\">No tasks yet. Launch a mission.</span>";
        } else {
          dom.contracts.innerHTML = active.map((contract) => {
            const pct = contract.complete ? 100 : (contract.target ? Math.min(100, (contract.progress / contract.target) * 100) : 0);
            const progressText = contract.type === "noDamage"
              ? `${Math.floor(contract.progress)}s / ${contract.target}s`
              : `${contract.progress} / ${contract.target}`;
            const statusText = contract.complete ? "Complete" : progressText;
            const faction = FACTIONS.find((item) => item.id === contract.factionId);
            return `
              <div class="contract-card">
                <div class="contract-title">${contract.title}</div>
                <div class="contract-meta">
                  <span>${contract.desc}</span>
                  <span>${statusText}</span>
                  <span>Reward: ${contract.reward.credits}c, ${contract.reward.xp}xp, +${contract.reward.rep} rep</span>
                  <span>${faction ? faction.name : contract.factionId}</span>
                </div>
                <div class="contract-progress"><span style="width:${pct}%"></span></div>
              </div>
            `;
          }).join("");
        }

        dom.factions.innerHTML = FACTIONS.map((faction) => {
          const rep = progress.factions[faction.id] || 0;
          return `
            <div class="faction-card">
              <span>${faction.desc}</span>
              <strong>${faction.name}</strong>
              <div class="contract-meta">Rep ${rep}</div>
            </div>
          `;
        }).join("");
      }

      function renderSettings() {
        const modeGroup = document.querySelector("[data-setting='game-mode']");
        const difficultyGroup = document.querySelector("[data-setting='difficulty']");
        const inputGroup = document.querySelector("[data-setting='input-mode']");
        const particleGroup = document.querySelector("[data-setting='particles']");
        const hitFlashGroup = document.querySelector("[data-setting='hit-flash']");
        const hudLayoutGroup = document.querySelector("[data-setting='hud-layout']");
        const hudScaleGroup = document.querySelector("[data-setting='hud-scale']");
        const audioGroup = document.querySelector("[data-setting='audio']");
        const paletteGroup = document.querySelector("[data-setting='palette']");
        const perfModeGroup = document.querySelector("[data-setting='perf-mode']");
        const renderScaleGroup = document.querySelector("[data-setting='render-scale']");
        [modeGroup, difficultyGroup, inputGroup, particleGroup, hitFlashGroup, hudLayoutGroup, hudScaleGroup, audioGroup, paletteGroup, perfModeGroup, renderScaleGroup].forEach((group) => {
          if (!group) return;
          const setting = group.dataset.setting;
          const value = setting === "game-mode" ? progress.settings.gameMode
            : setting === "difficulty" ? progress.settings.difficulty
            : setting === "input-mode" ? progress.settings.inputMode
            : setting === "particles" ? progress.settings.particles
            : setting === "hud-layout" ? (progress.settings.hudLayout || "standard")
            : setting === "hud-scale" ? (progress.settings.hudScale || "md")
            : setting === "audio" ? (progress.settings.audio || "on")
            : setting === "palette" ? (progress.settings.palette || "default")
            : setting === "perf-mode" ? getPerfMode()
            : setting === "render-scale" ? (progress.settings.renderScale || "auto")
            : progress.settings.hitFlash ? "on" : "off";
          group.querySelectorAll(".option-btn").forEach((btn) => {
            btn.classList.toggle("is-active", btn.dataset.option === value);
          });
        });

        renderKeybinds();
        renderPerformanceOverlays(performance.now(), true);
        updatePerfLogUi();
      }

      function renderKeybinds() {
        dom.keybinds.innerHTML = Object.keys(KEYBIND_LABELS).map((key) => {
          const value = input.capture === key ? "Press key" : formatKeybind(progress.keybinds[key]);
          return `
            <div class="keybind-row">
              <span>${KEYBIND_LABELS[key]}</span>
              <button class="keybind-btn" type="button" data-bind="${key}">${value}</button>
            </div>
          `;
        }).join("");
      }

      function renderHistory() {
        if (!dom.history) return;
        if (!progress.runHistory.length) {
          dom.history.innerHTML = "<span class=\"select-meta\">No runs logged yet.</span>";
          return;
        }
        dom.history.innerHTML = progress.runHistory.map((run) => {
          const difficultyLabel = run.difficulty || "Normal";
          const modeLabel = run.mode ? ` • ${run.mode}` : "";
          const score = Number.isFinite(run.score) ? Math.round(run.score).toLocaleString() : run.score;
          const credits = Number.isFinite(run.credits) ? Math.round(run.credits).toLocaleString() : run.credits;
          const usesLevel = run.mode !== "Frontier" && Number.isFinite(run.level);
          const progressText = usesLevel
            ? `Level ${run.level} • Wave ${run.wave}/${LEVEL_WAVES}`
            : `Wave ${run.wave}`;
          const intel = [
            run.threatTier ? `Threat ${run.threatTier}` : "",
            run.mutator ? `Mutator ${run.mutator}` : "",
            run.seed ? `Seed ${run.seed}` : "",
            run.route ? `Route ${formatRouteBonusLabel(run.route)}` : ""
          ].filter(Boolean).join(" • ");
          return `
            <div class="history-entry">
              <strong>${run.ship} / ${run.weapon}</strong>
              <span>${progressText} • Score ${score} • ${difficultyLabel}${modeLabel}</span>
              <span>${run.kills} kills • ${credits} credits</span>
              ${intel ? `<span class="muted">${intel}</span>` : ""}
            </div>
          `;
        }).join("");
      }

      function renderRunAnalytics() {
        if (!dom.runAnalytics) return;
        const entries = Array.isArray(progress.runAnalytics) ? progress.runAnalytics : [];
        if (!entries.length) {
          dom.runAnalytics.innerHTML = "<span class=\"select-meta\">No analytics captured yet.</span>";
          return;
        }
        dom.runAnalytics.innerHTML = entries.map((entry) => {
          const timestamp = new Date(entry.ts || Date.now()).toLocaleString();
          const score = Number.isFinite(entry.score) ? Math.round(entry.score).toLocaleString() : "0";
          const dealt = Number.isFinite(entry.damageDealt) ? Math.round(entry.damageDealt).toLocaleString() : "0";
          const taken = Number.isFinite(entry.damageTaken) ? Math.round(entry.damageTaken).toLocaleString() : "0";
          const accuracyPct = Number.isFinite(entry.accuracy) ? `${Math.round(entry.accuracy * 100)}%` : "0%";
          const systems = [
            entry.threatTier ? `Threat ${entry.threatTier}` : "",
            entry.mutator ? entry.mutator : "",
            entry.seed ? `Seed ${entry.seed}` : ""
          ].filter(Boolean).join(" • ");
          return `
            <div class="history-entry">
              <strong>${entry.ship || "Unknown"} / ${entry.weapon || "Unknown"}</strong>
              <span>${entry.waveDisplay || "Wave 1"} • ${entry.kills || 0} kills • Score ${score} • ${entry.mode || "Arcade"}</span>
              <span>ACC ${accuracyPct} • DMG ${dealt}/${taken} • A${entry.abilityUses || 0} S${entry.secondaryUses || 0} O${entry.objectiveCompletions || 0}</span>
              <span class="muted">${systems}${systems ? " • " : ""}${timestamp}</span>
            </div>
          `;
        }).join("");
      }

      function formatKeybind(value) {
        if (!value) return "-";
        if (value === " ") return "Space";
        if (value.startsWith("arrow")) {
          return `Arrow ${value.replace("arrow", "").toUpperCase()}`;
        }
        if (value === "shift") return "Shift";
        if (value === "control") return "Ctrl";
        if (value === "alt") return "Alt";
        if (value === "meta") return "Meta";
        return value.length > 1 ? value.toUpperCase() : value.toUpperCase();
      }

      function formatStat(key, value) {
        const labels = {
          damage: "Damage",
          bulletSpeed: "Velocity",
          fireRate: "Fire Rate",
          energyCost: "Energy Cost",
          critChance: "Crit Chance",
          spread: "Spread",
          turnRate: "Turn Rate",
          maxSpeed: "Max Speed",
          accel: "Acceleration",
          projectiles: "Projectiles",
          pierce: "Pierce",
          splashRadius: "Splash Radius",
          splashDamage: "Splash Damage",
          arcDamage: "Arc Damage",
          arcRadius: "Arc Radius",
          arcChains: "Arc Chains",
          slowChance: "Slow Chance",
          slowDuration: "Slow Duration",
          maxShield: "Max Shield",
          maxHealth: "Max Hull",
          maxEnergy: "Max Energy",
          energyRegen: "Energy Regen",
          boostMultiplier: "Boost Mult",
          boostCost: "Boost Cost",
          damageReduction: "Damage Reduction",
          upgradeLuck: "Upgrade Luck",
          blackHoleChance: "Black Hole Chance",
          blackHoleRadius: "Black Hole Radius",
          blackHoleDuration: "Black Hole Duration",
          blackHoleForce: "Black Hole Force",
          blackHoleDamage: "Black Hole Damage",
          echoChance: "Echo Chance",
          echoDamage: "Echo Damage"
        };
        const label = labels[key] || key;
        const isPercent = key === "critChance"
          || key === "damageReduction"
          || key === "slowChance"
          || key === "blackHoleChance"
          || key === "echoChance"
          || key === "echoDamage";
        const displayValue = isPercent ? value * 100 : value;
        const formatted = Number.isInteger(displayValue) ? displayValue.toString() : displayValue.toFixed(2);
        const signed = displayValue > 0 ? `+${formatted}` : formatted;
        const suffix = isPercent ? "%" : "";
        return `${signed}${suffix} ${label}`;
      }

      function formatDuration(seconds) {
        const total = Math.max(0, Math.round(seconds));
        const minutes = Math.floor(total / 60);
        const secs = total % 60;
        return `${minutes}:${secs.toString().padStart(2, "0")}`;
      }

      const TIER_COLORS = {
        common: "#ffffff",
        uncommon: "#6ee7b7",
        rare: "#7ca8ff",
        epic: "#b98cff",
        legendary: "#f6c65f"
      };

      const BARRAGE_STAT_KEYS = new Set([
        "barrageEvery",
        "barrageProjectiles",
        "barrageBonusDamage",
        "barrageSplashRadius",
        "barrageSplashDamage",
        "barragePierce",
        "barrageCounter"
      ]);

      const LOWER_BETTER_STATS = new Set(["energyCost", "spread", "cooldown", "boostCost"]);

      function getTierMeta(tier) {
        return TIER_META[tier] || TIER_META.common;
      }

      function formatTierLabel(tier) {
        return getTierMeta(tier).label;
      }

      function normalizeCategory(category) {
        return (category || "upgrade").toLowerCase().replace(/\s+/g, "-");
      }

      function getUpgradeCategoryIcon(category) {
        const key = normalizeCategory(category);
        const icons = {
          offense: "⚔",
          defense: "🛡",
          mobility: "➜",
          utility: "🔧",
          strategy: "🧭",
          control: "❄",
          systems: "⚙",
          upgrade: "★"
        };
        return icons[key] || icons.upgrade;
      }

      function getUpgradeGlyphs(upgrade, level = 1) {
        if (!upgrade) return [];
        const labels = [];
        const add = (label) => {
          if (!label || labels.length >= 3 || labels.includes(label)) return;
          labels.push(label);
        };
        if (upgrade.kind === "skill" && upgrade.skillId) {
          const skillMap = {
            halo: ["HALO ↑", "RANGE ↑", "DMG ↑"],
            minefield: ["MINE ↑", "BLAST ↑", "RATE ↑"],
            escort: ["HELPER ↑", "DMG ↑", "RATE ↑"],
            shockwave: ["WAVE ↑", "RANGE ↑", "DMG ↑"],
            harrier: ["MISSILE ↑", "DMG ↑", "RATE ↑"],
            barrage: ["BARRAGE ↑", "SHOT ↑", "PIERCE ↑"],
            "arc-lattice": ["CHAIN ↑", "SLOW ↑", "RANGE ↑"],
            bulwark: ["SHIELD ↑", "HULL ↑", "AURA ↑"]
          };
          const mapped = skillMap[upgrade.skillId];
          if (mapped) {
            mapped.forEach(add);
            return labels;
          }
        }
        const id = `${upgrade.id || ""} ${upgrade.name || ""} ${upgrade.category || ""}`.toLowerCase();
        const keywordMap = [
          { test: /(damage|calibrated|coil|amplifier|fire-control|munitions|barrage|weapon-calibration)/, label: "DMG ↑" },
          { test: /(fire|rate|overclock|loader|cadence|rapid)/, label: "RATE ↑" },
          { test: /(crit|target|lens|scope)/, label: "CRIT ↑" },
          { test: /(shield|aegis|bulwark|barrier|reflect)/, label: "SHIELD ↑" },
          { test: /(hull|repair|armor|plating|patch)/, label: "HULL ↑" },
          { test: /(energy|reactor|capacitor|battery|efficiency)/, label: "ENERGY ↑" },
          { test: /(speed|boost|thruster|jet|vector|inertial)/, label: "SPEED ↑" },
          { test: /(cooldown|charge|cycle)/, label: "COOLDOWN ↓" },
          { test: /(salvage|scanner|luck|blueprint|credit)/, label: "LOOT ↑" },
          { test: /(slow|chill|freeze|snare)/, label: "SLOW ↑" },
          { test: /(range|radius|aura)/, label: "RANGE ↑" },
          { test: /(mine|shockwave|nova|pulse)/, label: "BLAST ↑" },
          { test: /(missile|rocket|torpedo)/, label: "MISSILE ↑" },
          { test: /(helper|escort|drone|wing)/, label: "HELPER ↑" }
        ];
        keywordMap.forEach((entry) => {
          if (entry.test.test(id)) add(entry.label);
        });
        if (!labels.length) {
          const category = normalizeCategory(upgrade.category || "upgrade");
          const categoryMap = {
            offense: ["DMG ↑", "RATE ↑", "CRIT ↑"],
            defense: ["SHIELD ↑", "HULL ↑", "GUARD ↑"],
            mobility: ["SPEED ↑", "BOOST ↑", "TURN ↑"],
            utility: ["ENERGY ↑", "COOLDOWN ↓", "CONTROL ↑"],
            strategy: ["LOOT ↑", "XP ↑", "BONUS ↑"],
            control: ["SLOW ↑", "RANGE ↑", "BLAST ↑"],
            systems: ["SYSTEM ↑", "ENERGY ↑", "POWER ↑"],
            upgrade: ["POWER ↑", "BOOST ↑", "READY ↑"]
          };
          (categoryMap[category] || categoryMap.upgrade).forEach(add);
        }
        return labels;
      }

      function renderUpgradeGlyphs(glyphs) {
        if (!glyphs || !glyphs.length) return "";
        return glyphs.map((glyph) => `<span class="upgrade-glyph">${glyph}</span>`).join("");
      }

      function getRewardIcon(reward) {
        const icons = {
          credits: "🪙",
          blueprints: "📘",
          salvage: "🔑",
          part: "⚙",
          weapon: "⚡"
        };
        return icons[reward.type] || "★";
      }

      function getRewardLabel(reward) {
        if (!reward) return "";
        if (reward.type === "credits") return "Credits";
        if (reward.type === "blueprints") return "Blueprints";
        if (reward.type === "salvage") return "Salvage keys";
        if (reward.type === "part") return reward.title || "Attachment";
        if (reward.type === "weapon") return reward.title || "Weapon";
        return reward.type;
      }

      function formatRewardValue(reward) {
        if (!reward) return "0";
        const amount = Number.isFinite(reward.amount) ? reward.amount : 1;
        return amount.toLocaleString();
      }

      function sortRewardsByTier(rewards) {
        if (!Array.isArray(rewards)) return [];
        return rewards.slice().sort((a, b) => {
          const tierA = a && a.tier ? a.tier : "common";
          const tierB = b && b.tier ? b.tier : "common";
          const diff = getTierIndex(tierA) - getTierIndex(tierB);
          if (diff !== 0) return diff;
          return (a && a.title ? a.title : "").localeCompare(b && b.title ? b.title : "");
        });
      }

      function getSalvageRevealDelay(tier, index) {
        const tierIndex = getTierIndex(tier || "common");
        const baseDelay = index * 0.08;
        const tierBonus = tierIndex >= 4
          ? 0.3
          : tierIndex >= 3
            ? 0.22
            : tierIndex >= 2
              ? 0.14
              : tierIndex >= 1
                ? 0.08
                : 0;
        return (baseDelay + tierBonus).toFixed(2);
      }

      function groupByTier(items) {
        const grouped = {};
        items.forEach((item) => {
          const tier = item.tier || "common";
          if (!grouped[tier]) grouped[tier] = [];
          grouped[tier].push(item);
        });
        return TIER_ORDER.filter((tier) => grouped[tier] && grouped[tier].length)
          .map((tier) => ({ tier, items: grouped[tier] }));
      }

      function formatDelta(value, digits) {
        if (!Number.isFinite(value)) return "0";
        const rounded = Number.isInteger(value) ? value.toString() : value.toFixed(digits);
        const sign = value > 0 ? "+" : "";
        return `${sign}${rounded}`;
      }

      function isBetterStat(key, diff) {
        return LOWER_BETTER_STATS.has(key) ? diff < 0 : diff > 0;
      }

      function getWeaponStatsSnapshot(item) {
        return getWeaponItemStats(item) || {};
      }

      function buildWeaponComparison(currentWeapon, candidateWeapon) {
        if (!currentWeapon || !candidateWeapon) return "";
        const currentStats = getWeaponStatsSnapshot(currentWeapon);
        const nextStats = getWeaponStatsSnapshot(candidateWeapon);
        const rules = [
          { key: "damage", label: "Damage", digits: 0 },
          { key: "fireRate", label: "Fire Rate", digits: 1 },
          { key: "energyCost", label: "Energy Cost", digits: 1 },
          { key: "bulletSpeed", label: "Velocity", digits: 0 },
          { key: "projectiles", label: "Projectiles", digits: 0 },
          { key: "spread", label: "Spread", digits: 2 },
          { key: "splashRadius", label: "Splash", digits: 0 },
          { key: "pierce", label: "Pierce", digits: 0 }
        ];
        const chips = [];
        rules.forEach((rule) => {
          const base = currentStats[rule.key] || 0;
          const next = nextStats[rule.key] || 0;
          const diff = next - base;
          if (Math.abs(diff) < 0.01) return;
          const better = isBetterStat(rule.key, diff);
          chips.push(`<span class="stat-diff ${better ? "is-positive" : "is-negative"}">${formatDelta(diff, rule.digits)} ${rule.label}</span>`);
        });
        if (!chips.length) {
          return "<span class=\"stat-diff is-neutral\">Matches equipped stats</span>";
        }
        return chips.slice(0, 4).join("");
      }

      function buildSecondaryComparison(currentSecondary, candidateSecondary) {
        if (!currentSecondary || !candidateSecondary) return "";
        const diff = (candidateSecondary.cooldown || 0) - (currentSecondary.cooldown || 0);
        if (Math.abs(diff) < 0.1) {
          return "<span class=\"stat-diff is-neutral\">Matches equipped cooldown</span>";
        }
        const better = isBetterStat("cooldown", diff);
        return `<span class="stat-diff ${better ? "is-positive" : "is-negative"}">${formatDelta(diff, 0)}s Cooldown</span>`;
      }

      function buildAttachmentComparison(item, equippedItem) {
        if (!item || !item.stats) return "";
        const currentStats = equippedItem ? (getAttachmentStats(equippedItem) || {}) : {};
        const candidateStats = getAttachmentStats(item) || {};
        const chips = Object.keys(candidateStats).map((key) => {
          const diff = (candidateStats[key] || 0) - (currentStats[key] || 0);
          if (Math.abs(diff) < 0.01) return "";
          const better = isBetterStat(key, diff);
          return `<span class="stat-diff ${better ? "is-positive" : "is-negative"}">${formatStat(key, diff)}</span>`;
        }).filter(Boolean);
        if (!chips.length) {
          return "<span class=\"stat-diff is-neutral\">Matches equipped attachment</span>";
        }
        return chips.slice(0, 4).join("");
      }

      function getPartTier(part) {
        if (!part) return "common";
        if (part.tier) return part.tier;
        if (part.rarityId) return part.rarityId;
        return (part.rarity || "Common").toLowerCase();
      }

      function getPartScrapValue(part) {
        const tier = getPartTier(part);
        const values = {
          common: 60,
          uncommon: 120,
          rare: 200,
          epic: 320,
          legendary: 520
        };
        return values[tier] || 80;
      }

      function getShipTierStyle(tier) {
        if (tier >= 4) return "epic";
        if (tier === 3) return "rare";
        if (tier === 2) return "uncommon";
        return "common";
      }

      function renderTierSection(tier, items, renderItem) {
        const meta = getTierMeta(tier);
        return `
          <div class="tier-section" data-tier="${tier}">
            <div class="tier-header">
              <span class="tier-title">${meta.label} Tier</span>
              <span class="tier-count">${items.length} ${items.length === 1 ? "option" : "options"}</span>
            </div>
            <div class="card-grid">
              ${items.map(renderItem).join("")}
            </div>
          </div>
        `;
      }

      function formatUnlockText(unlock) {
        if (!unlock) return "";
        const parts = [];
        if (unlock.rank) parts.push(`Rank ${unlock.rank}`);
        if (unlock.credits) parts.push(`${unlock.credits}c`);
        if (unlock.blueprints) parts.push(`${unlock.blueprints} blueprints`);
        if (unlock.faction) parts.push(`${unlock.faction.id.toUpperCase()} ${unlock.faction.rep}`);
        return parts.join(" · ");
      }

      function canUnlock(item, unlocks) {
        if (unlocks[item.id]) return true;
        const unlock = item.unlock || {};
        if (unlock.rank && progress.rank < unlock.rank) return false;
        if (unlock.faction) {
          const rep = progress.factions[unlock.faction.id] || 0;
          if (rep < unlock.faction.rep) return false;
        }
        if (unlock.credits && progress.bankedCredits < unlock.credits) return false;
        if (unlock.blueprints && progress.blueprints < unlock.blueprints) return false;
        return true;
      }

      function spendUnlockCost(unlock) {
        if (!unlock) return;
        if (unlock.credits) {
          progress.bankedCredits = Math.max(0, progress.bankedCredits - unlock.credits);
        }
        if (unlock.blueprints) {
          progress.blueprints = Math.max(0, progress.blueprints - unlock.blueprints);
        }
      }

      function unlockItem(item, unlocks) {
        if (unlocks[item.id]) return true;
        if (!canUnlock(item, unlocks)) return false;
        spendUnlockCost(item.unlock);
        unlocks[item.id] = true;
        saveProgress();
        return true;
      }

      function setOverlay(mode) {
        state.overlayMode = mode;
        if (mode === "start") {
          const fireKey = formatKeybind(progress.keybinds.fire);
          const boostKey = formatKeybind(progress.keybinds.boost);
          const abilityKey = formatKeybind(progress.keybinds.ability);
          const secondaryKey = formatKeybind(progress.keybinds.secondary);
          const dockKey = formatKeybind(progress.keybinds.dock);
          const helpKey = formatKeybind(progress.keybinds.help || "h");
          const level = Math.max(1, progress.campaignLevel || 1);
          const modeLabel = isFrontierMode() ? "Frontier patrol" : `Level ${level}`;
          const modeCopy = isFrontierMode()
            ? "Survive as long as you can. Dock mid-run to boost."
            : `Clear ${LEVEL_WAVES} waves. Boss on wave ${LEVEL_WAVES}.`;
          const modeTip = isFrontierMode()
            ? `<li>Dock: ${dockKey} for quick boosts.</li>`
            : `<li>Beat wave ${LEVEL_WAVES} to unlock the next level.</li>`;
          const aimKeys = [
            progress.keybinds.aimUp,
            progress.keybinds.aimLeft,
            progress.keybinds.aimDown,
            progress.keybinds.aimRight
          ].map(formatKeybind).join("/");
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Ready to fly</p>
              <h3>${modeLabel}</h3>
              <p>${modeCopy}</p>
            </div>
            <ul class="overlay-list">
              <li>Move: WASD. Aim: mouse or ${aimKeys}.</li>
              <li>Shoot: ${fireKey}. Boost: ${boostKey}.</li>
              <li>Ability: ${abilityKey}. Secondary: ${secondaryKey}.</li>
              <li>Pause: P or Esc.</li>
              <li>Help: ${helpKey} opens quick controls.</li>
              ${modeTip}
              <li>Practice = no loot.</li>
            </ul>
            <div class="overlay-actions">
              <button class="btn primary" data-overlay-action="launch">Play</button>
            </div>
          `;
        }
        if (mode === "paused") {
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Paused</p>
              <h3>Take a breather</h3>
              <p>Press resume to keep flying.</p>
            </div>
            <div class="overlay-actions">
              <button class="btn primary" data-overlay-action="resume">Resume</button>
              <button class="btn ghost" data-action="help">Help</button>
              <button class="btn ghost" data-overlay-action="reset">Exit</button>
            </div>
          `;
        }
        if (mode === "help") {
          const fireKey = formatKeybind(progress.keybinds.fire);
          const boostKey = formatKeybind(progress.keybinds.boost);
          const abilityKey = formatKeybind(progress.keybinds.ability);
          const secondaryKey = formatKeybind(progress.keybinds.secondary);
          const dockKey = formatKeybind(progress.keybinds.dock);
          const helpKey = formatKeybind(progress.keybinds.help || "h");
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Quick guide</p>
              <h3>Combat controls</h3>
              <p>Use this panel as a fast in-run reminder.</p>
            </div>
            <ul class="overlay-list">
              <li>Move with WASD and aim with mouse or arrow keys.</li>
              <li>Shoot ${fireKey}. Boost ${boostKey}. Ability ${abilityKey}.</li>
              <li>Secondary ${secondaryKey}. Frontier dock ${dockKey}.</li>
              <li>Press ${helpKey} again to close help.</li>
            </ul>
            <div class="overlay-actions">
              <button class="btn primary" data-overlay-action="resume">Resume</button>
              <button class="btn ghost" data-overlay-action="reset">Exit</button>
            </div>
          `;
        }
        if (mode === "tutorial") {
          const stepIndex = clamp(state.tutorialStep || 0, 0, TUTORIAL_STEPS.length - 1);
          state.tutorialStep = stepIndex;
          const step = TUTORIAL_STEPS[stepIndex];
          const isFirst = stepIndex === 0;
          const isLast = stepIndex === TUTORIAL_STEPS.length - 1;
          const launchTraining = state.mode === "hangar"
            ? "<button class=\"btn ghost\" data-overlay-action=\"tutorial-training\">Launch Practice</button>"
            : "";
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Tutorial ${stepIndex + 1}/${TUTORIAL_STEPS.length}</p>
              <h3>${step.title}</h3>
              <p>${step.text}</p>
            </div>
            <ul class="overlay-list">
              <li>Keep objective, threat tier, and status chips in view before committing to fights.</li>
              <li>Use route events and premium upgrades to shape long-run consistency, not just burst.</li>
            </ul>
            <div class="overlay-actions">
              <button class="btn ghost" data-overlay-action="tutorial-prev" ${isFirst ? "disabled" : ""}>Back</button>
              <button class="btn primary" data-overlay-action="${isLast ? "tutorial-close" : "tutorial-next"}">${isLast ? "Done" : "Next"}</button>
              ${launchTraining}
            </div>
          `;
        }
        if (mode === "glossary") {
          const entries = STAT_GLOSSARY.map((item) => `
            <div class="history-entry">
              <strong>${item.term}</strong>
              <span>${item.detail}</span>
            </div>
          `).join("");
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Stat glossary</p>
              <h3>Combat metrics</h3>
              <p>Use these to tune loadouts, upgrades, and route choices.</p>
            </div>
            <div class="history-list">
              ${entries}
            </div>
            <div class="overlay-actions">
              <button class="btn primary" data-overlay-action="glossary-close">Close</button>
            </div>
          `;
        }
        if (mode === "choice-event") {
          const choiceEvent = state.choiceEvent;
          const options = choiceEvent && Array.isArray(choiceEvent.options) ? choiceEvent.options : [];
          const cards = options.map((option) => `
            <button class="select-card" data-choice-id="${option.id}">
              <span class="select-title">${option.title}</span>
              <span class="select-meta">${option.desc}</span>
            </button>
          `).join("");
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Milestone wave ${choiceEvent ? choiceEvent.wave : state.wave}</p>
              <h3>Route decision</h3>
              <p>Select one strategic route for the next three waves.</p>
            </div>
            <div class="card-grid">
              ${cards || "<div class=\"select-meta\">No route options available.</div>"}
            </div>
          `;
        }
        if (mode === "upgrade") {
          const availableUpgrades = state.upgradeOptions.filter((upgrade) => {
            const stack = state.upgradeStacks[upgrade.id] || 0;
            const maxStacks = Number.isFinite(upgrade.maxStacks) ? upgrade.maxStacks : 1;
            return stack < maxStacks && canSelectSkillUpgrade(upgrade);
          });
          const hasUpgrades = availableUpgrades.length > 0;
          const creditsLabel = Math.round(state.credits).toLocaleString();
          const skillSlots = `${state.skillSlots.length}/${SKILL_LIMIT}`;
          const upgradeCards = hasUpgrades
            ? availableUpgrades.map((upgrade) => {
              const stack = state.upgradeStacks[upgrade.id] || 0;
              const maxStacks = Number.isFinite(upgrade.maxStacks) ? upgrade.maxStacks : 1;
              const nextLevel = stack + 1;
              const tier = getUpgradeTier(upgrade, nextLevel);
              const tierLabel = formatTierLabel(tier);
              const category = upgrade.category || "Upgrade";
              const categoryIcon = getUpgradeCategoryIcon(category);
              const isSkill = isSkillUpgrade(upgrade);
              const stackLabel = `Lv ${nextLevel}/${maxStacks}`;
              const kindLabel = isSkill ? "Skill" : "Boost";
              const glyphs = renderUpgradeGlyphs(getUpgradeGlyphs(upgrade, nextLevel));
              return `
                <button class="upgrade-card is-choice" data-tier="${tier}" data-kind="${isSkill ? "skill" : "stat"}" data-upgrade-id="${upgrade.id}">
                  <span class="upgrade-ribbon" data-tier="${tier}">${tierLabel}</span>
                  <span class="upgrade-stack">${stackLabel}</span>
                  <span class="upgrade-icon" aria-hidden="true">${categoryIcon}</span>
                  <span class="upgrade-title">${upgrade.name}</span>
                  <div class="upgrade-glyphs">${glyphs}</div>
                  <span class="upgrade-kind">${kindLabel}</span>
                </button>
              `;
            }).join("")
            : `
              <div class="upgrade-card is-choice is-static" data-tier="common" aria-disabled="true">
                <span class="upgrade-ribbon" data-tier="common">Complete</span>
                <span class="upgrade-icon" aria-hidden="true">${getUpgradeCategoryIcon("upgrade")}</span>
                <span class="upgrade-title">All upgrades installed</span>
                <span class="upgrade-desc">All boosts maxed. Launch next wave.</span>
                <span class="upgrade-kind">Ready</span>
              </div>
            `;
          const rerollCost = getUpgradeRerollCost();
          const canReroll = state.credits >= rerollCost;
          const rerollCopy = `Reroll grows each wave: ${rerollCost} coins.`;
          const upgradeActions = hasUpgrades
            ? `
              <button class="btn ghost" data-overlay-action="reroll" ${canReroll ? "" : "disabled"}>Reroll (${rerollCost})</button>
              <button class="btn ghost" data-overlay-action="skip">Skip</button>
            `
            : `
              <button class="btn primary" data-overlay-action="skip">Launch next wave</button>
            `;
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Wave ${getWaveDisplay(state.wave - 1)} cleared</p>
              <h3>Pick a boost</h3>
              <p>Pick one boost. Skills max ${SKILL_LIMIT}. ${rerollCopy}</p>
            </div>
            <div class="badge-group overlay-badges">
              <span class="badge">Credits ${creditsLabel}</span>
              <span class="badge">Skills ${skillSlots}</span>
            </div>
            <div class="upgrade-grid upgrade-choice-grid">
              ${upgradeCards}
            </div>
            <div class="overlay-actions">
              ${upgradeActions}
            </div>
          `;
        }
        if (mode === "dock") {
          const tier = state.frontier ? state.frontier.tier : 1;
          const upgradesHtml = FRONTIER_UPGRADES.map((upgrade) => {
            const level = state.frontier?.upgrades[upgrade.id] || 0;
            const maxLevel = upgrade.maxLevel || 1;
            const cost = getFrontierUpgradeCost(upgrade, level, tier);
            const canBuy = level < maxLevel && state.credits >= cost;
            const status = level >= maxLevel ? "Maxed" : `Cost ${cost} credits`;
            const tierId = upgrade.tier || "common";
            const tierLabel = formatTierLabel(tierId);
            const levelLabel = `Lv ${level} / ${maxLevel}`;
            const category = upgrade.category || "Upgrade";
            const categoryIcon = getUpgradeCategoryIcon(category);
            const categoryKey = normalizeCategory(category);
            const glyphs = renderUpgradeGlyphs(getUpgradeGlyphs(upgrade, Math.min(level + 1, maxLevel)));
            return `
              <button class="upgrade-card" data-tier="${tierId}" data-frontier-upgrade="${upgrade.id}" ${canBuy ? "" : "disabled"}>
                <div class="upgrade-meta">
                  <span class="select-pill tier-pill" data-tier="${tierId}">${tierLabel}</span>
                  <span class="select-pill category-pill" data-category="${categoryKey}" aria-label="${category}" title="${category}">${categoryIcon}</span>
                  <span class="select-pill">${levelLabel}</span>
                </div>
                <span class="upgrade-title">${upgrade.name}</span>
                <div class="upgrade-glyphs">${glyphs}</div>
                <span class="upgrade-desc">${status}</span>
              </button>
            `;
          }).join("");
          const nextShips = getFrontierNextShips();
          const tierCost = getFrontierTierCost(tier);
          const canTierUp = nextShips.length && state.credits >= tierCost;
          const tierButton = nextShips.length
            ? `<button class="btn primary" data-overlay-action="tier-up" ${canTierUp ? "" : "disabled"}>Tier up (${tierCost} credits)</button>`
            : "<button class=\"btn ghost\" disabled>Max tier reached</button>";
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Frontier dock</p>
              <h3>Dock upgrades</h3>
              <p>Spend credits to boost your ship. Tier up to unlock new hulls.</p>
            </div>
            <div class="upgrade-grid">
              ${upgradesHtml}
            </div>
            <div class="overlay-actions">
              <button class="btn ghost" data-overlay-action="dock-close">Resume patrol</button>
              ${tierButton}
            </div>
          `;
        }
        if (mode === "tier-select") {
          const nextShips = getFrontierNextShips();
          const tierCost = getFrontierTierCost(state.frontier?.tier || 1);
          const shipCards = nextShips.map((ship) => {
            const weaponName = ship.signatureWeapon ? getWeaponById(ship.signatureWeapon).name : "Standard";
            const tierText = ship.tier ? `Tier ${ship.tier}` : "Tier upgrade";
            const abilityName = ABILITIES[ship.abilityId]?.name || "Ability";
            return `
              <button class="select-card" data-frontier-ship="${ship.id}">
                <span class="select-title">${ship.name}</span>
                <span class="select-meta">${ship.desc}</span>
                <div class="select-pills">
                  <span class="select-pill">${tierText}</span>
                  <span class="select-pill">Signature: ${weaponName}</span>
                  <span class="select-pill">${abilityName}</span>
                </div>
              </button>
            `;
          }).join("");
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Tier upgrade</p>
              <h3>Pick your next ship</h3>
              <p>Cost ${tierCost} credits. Dock boosts reset.</p>
            </div>
            <div class="card-grid">
              ${shipCards || "<div class=\"select-meta\">No upgrades available.</div>"}
            </div>
            <div class="overlay-actions">
              <button class="btn ghost" data-overlay-action="dock">Back to dock</button>
            </div>
          `;
        }
        if (mode === "salvage") {
          const results = state.salvageResults || { rewards: [], count: 0 };
          const rewards = sortRewardsByTier(results.rewards || []);
          const rewardCards = rewards.length
            ? rewards.map((reward, index) => {
              const tier = reward.tier || "common";
              const tierIndex = getTierIndex(tier);
              const glow = tierIndex >= 2;
              const icon = getRewardIcon(reward);
              const label = reward.title || getRewardLabel(reward);
              const value = formatRewardValue(reward);
              const delay = getSalvageRevealDelay(tier, index);
              return `
                <div class="reward-item salvage-item ${glow ? "is-glow" : ""}" data-tier="${tier}" style="--reveal-delay:${delay}s;">
                  <span class="reward-icon" role="img" aria-label="${label} x${value}" title="${label} x${value}">${icon}</span>
                  <span class="reward-value">${value}</span>
                  <span class="reward-label">${label}</span>
                  ${reward.note ? `<span class="reward-note">${reward.note}</span>` : ""}
                </div>
              `;
            }).join("")
            : "<div class=\"select-meta\">No salvage recovered.</div>";
          const countLabel = results.count === 1 ? "1 cache" : `${results.count} caches`;
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Salvage recovered</p>
              <h3>${countLabel} opened</h3>
              <p>Loot pops in from common to legendary. Big drops glow.</p>
            </div>
            <div class="salvage-grid">
              ${rewardCards}
            </div>
            <div class="overlay-actions">
              <button class="btn primary" data-overlay-action="salvage-close">Back to gear</button>
            </div>
          `;
        }
        if (mode === "gameover") {
          const aborted = !!state.runEndedByAbort;
          const headerEyebrow = state.training ? "Practice done" : aborted ? "Run aborted" : "Crashed";
          const headerTitle = state.training ? "Practice wrap" : aborted ? "Mission aborted" : "Ship down";
          const headerCopy = state.training
            ? "Practice gives no loot. Try again anytime."
            : aborted
              ? "Run ended early. Banked credits are secured."
              : "Grab your rewards and refit in the hangar.";
          const lossRewards = state.lossRewards;
          const summary = lossRewards ? lossRewards.summary : getRunSummary();
          const performanceTier = lossRewards ? lossRewards.tier : "common";
          const performanceLabel = formatTierLabel(performanceTier);
          const levelChip = isCampaignMode() ? `<span class="chip">Level ${summary.level}</span>` : "";
          const highlightSection = state.runHighlights && state.runHighlights.length
            ? `
              <div class="chip-list">
                ${state.runHighlights.map((label) => `<span class="chip" data-tier="legendary">${label}</span>`).join("")}
              </div>
            `
            : "";
          const rewardCards = lossRewards && lossRewards.rewards.length
            ? lossRewards.rewards.map((reward) => {
              const tier = reward.tier || "common";
              const icon = getRewardIcon(reward);
              const label = getRewardLabel(reward);
              const value = formatRewardValue(reward);
              return `
                <div class="reward-item" data-tier="${tier}">
                  <span class="reward-icon" role="img" aria-label="${label} x${value}" title="${label} x${value}">${icon}</span>
                  <span class="reward-value">${value}</span>
                </div>
              `;
            }).join("")
            : "<div class=\"select-meta\">No salvage recovered.</div>";
          const rewardSection = (state.training || aborted) ? "" : `
            <div class="panel-subsection">
              <div class="reward-banner">
                <span class="reward-line"></span>
                <span class="reward-title">Rewards</span>
                <span class="reward-line"></span>
              </div>
              <div class="chip-list">
                <span class="chip" data-tier="${performanceTier}">${performanceLabel} performance</span>
                <span class="chip">Survival ${formatDuration(summary.durationSec)}</span>
                <span class="chip">${summary.difficultyLabel} difficulty</span>
                ${levelChip}
              </div>
              <div class="reward-grid">
                ${rewardCards}
              </div>
            </div>
          `;
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">${headerEyebrow}</p>
              <h3>${headerTitle}</h3>
              <p>${headerCopy}</p>
            </div>
            <div class="progress-grid">
              <div class="stat-tile"><span>Wave</span><strong>${summary.waveDisplay}</strong></div>
              <div class="stat-tile"><span>Kills</span><strong>${summary.kills}</strong></div>
              <div class="stat-tile"><span>Score</span><strong>${summary.score.toLocaleString()}</strong></div>
              <div class="stat-tile"><span>Survival</span><strong>${formatDuration(summary.durationSec)}</strong></div>
              <div class="stat-tile"><span>Credits</span><strong>${summary.credits.toLocaleString()}</strong></div>
            </div>
            ${highlightSection}
            ${rewardSection}
            <div class="overlay-actions">
              <button class="btn ghost" data-overlay-action="restart">Play again</button>
              <button class="btn primary" data-overlay-action="reset">Return to hangar</button>
            </div>
          `;
        }
        if (mode === "victory") {
          const result = state.levelRewards;
          const summary = result ? result.summary : getRunSummary();
          const level = result ? result.level : summary.level;
          const nextLevel = Math.max(progress.campaignLevel || level + 1, level + 1);
          const rewardTier = result ? result.tier : "common";
          const rewardLabel = formatTierLabel(rewardTier);
          const highlightSection = state.runHighlights && state.runHighlights.length
            ? `
              <div class="chip-list">
                ${state.runHighlights.map((label) => `<span class="chip" data-tier="legendary">${label}</span>`).join("")}
              </div>
            `
            : "";
          const rewardCards = result && result.rewards.length
            ? result.rewards.map((reward) => {
              const tier = reward.tier || "common";
              const icon = getRewardIcon(reward);
              const label = getRewardLabel(reward);
              const value = formatRewardValue(reward);
              return `
                <div class="reward-item" data-tier="${tier}">
                  <span class="reward-icon" role="img" aria-label="${label} x${value}" title="${label} x${value}">${icon}</span>
                  <span class="reward-value">${value}</span>
                </div>
              `;
            }).join("")
            : "<div class=\"select-meta\">No rewards recovered.</div>";
          dom.overlayContent.innerHTML = `
            <div class="overlay-header">
              <p class="eyebrow">Level ${level} cleared</p>
              <h3>You did it!</h3>
              <p>Refit now or jump to Level ${nextLevel}.</p>
            </div>
            <div class="progress-grid">
              <div class="stat-tile"><span>Wave</span><strong>${summary.waveDisplay}</strong></div>
              <div class="stat-tile"><span>Kills</span><strong>${summary.kills}</strong></div>
              <div class="stat-tile"><span>Score</span><strong>${summary.score.toLocaleString()}</strong></div>
              <div class="stat-tile"><span>Survival</span><strong>${formatDuration(summary.durationSec)}</strong></div>
              <div class="stat-tile"><span>Credits</span><strong>${summary.credits.toLocaleString()}</strong></div>
            </div>
            ${highlightSection}
            <div class="panel-subsection">
              <div class="reward-banner">
                <span class="reward-line"></span>
                <span class="reward-title">Rewards</span>
                <span class="reward-line"></span>
              </div>
              <div class="chip-list">
                <span class="chip">Level ${level}</span>
                <span class="chip" data-tier="${rewardTier}">${rewardLabel} rewards</span>
                <span class="chip">${summary.difficultyLabel} difficulty</span>
              </div>
              <div class="reward-grid">
                ${rewardCards}
              </div>
            </div>
            <div class="overlay-actions">
              <button class="btn primary" data-overlay-action="next-level">Launch Level ${nextLevel}</button>
              <button class="btn ghost" data-overlay-action="reset">Return to hangar</button>
            </div>
          `;
        }

        dom.overlay.classList.add("is-visible");
      }

      function hideOverlay() {
        state.overlayMode = null;
        dom.overlay.classList.remove("is-visible");
      }

      function logEvent(message) {
        const entry = document.createElement("div");
        entry.className = "log-entry";
        entry.textContent = message;
        dom.log.prepend(entry);
        while (dom.log.children.length > 6) {
          dom.log.removeChild(dom.log.lastChild);
        }
      }

      function spawnExplosion(x, y, color, size = 18) {
        const count = getParticleCount(14);
        for (let i = 0; i < count; i += 1) {
          const angle = Math.random() * Math.PI * 2;
          const speed = rand(60, 220);
          particles.push({
            x,
            y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: rand(0.4, 0.9),
            maxLife: 0.9,
            size: rand(2, size / 6),
            color
          });
        }
      }

      function spawnPulse(x, y, color, maxRadius = 120) {
        pulses.push({
          x,
          y,
          color,
          radius: 10,
          maxRadius,
          life: 0.6,
          maxLife: 0.6,
          speed: (maxRadius - 10) / 0.6
        });
      }

      function spawnBlackHole(x, y, options = {}) {
        const duration = options.duration || 1.6;
        const hole = {
          x,
          y,
          radius: options.radius || 120,
          force: options.force || 360,
          damage: options.damage || 0,
          life: duration,
          maxLife: duration
        };
        state.blackHoles = state.blackHoles || [];
        state.blackHoles.push(hole);
        if (state.blackHoles.length > 6) {
          state.blackHoles.splice(0, state.blackHoles.length - 6);
        }
      }

      function spawnLootBursts(x, y, drops) {
        if (!drops || !drops.length) return;
        const baseX = clamp(x, 24, state.worldWidth - 24);
        const baseY = clamp(y, 24, state.worldHeight - 24);
        drops.forEach((drop, index) => {
          lootBursts.push({
            x: clamp(baseX + rand(-12, 12), 16, state.worldWidth - 16),
            y: clamp(baseY - index * 18, 16, state.worldHeight - 16),
            vy: -28 - index * 4,
            life: 1.4,
            maxLife: 1.4,
            label: drop.label,
            tier: drop.tier || "common"
          });
        });
      }

      function spawnDamageNumber(x, y, amount, options = {}) {
        const value = Math.round(amount);
        if (!Number.isFinite(value) || value === 0) return;
        const prefix = options.prefix || "";
        const safeX = clamp(x, 16, state.worldWidth - 16);
        const safeY = clamp(y, 16, state.worldHeight - 16);
        damageNumbers.push({
          x: safeX,
          y: safeY,
          vx: rand(-10, 10),
          vy: -28,
          life: 0.9,
          maxLife: 0.9,
          text: `${prefix}${Math.abs(value)}`,
          color: options.color || "#f06969"
        });
      }

      function wrapEntity(entity) {
        const radius = entity.radius || 0;
        const minX = radius + 4;
        const minY = radius + 4;
        const maxX = state.worldWidth - radius - 4;
        const maxY = state.worldHeight - radius - 4;
        if (entity.x < minX) {
          entity.x = minX;
          if (entity.vx) entity.vx = Math.max(0, entity.vx) * 0.6;
        }
        if (entity.x > maxX) {
          entity.x = maxX;
          if (entity.vx) entity.vx = Math.min(0, entity.vx) * 0.6;
        }
        if (entity.y < minY) {
          entity.y = minY;
          if (entity.vy) entity.vy = Math.max(0, entity.vy) * 0.6;
        }
        if (entity.y > maxY) {
          entity.y = maxY;
          if (entity.vy) entity.vy = Math.min(0, entity.vy) * 0.6;
        }
      }

      function resolveObstacleCollisions(entity) {
        if (!entity || !obstacles.length) return;
        const applyBounce = (nx, ny) => {
          if (!Number.isFinite(entity.vx) || !Number.isFinite(entity.vy)) return;
          const speed = Math.hypot(entity.vx, entity.vy);
          const bounce = clamp(speed / 240, 0.12, 0.45);
          const dot = entity.vx * nx + entity.vy * ny;
          if (dot < 0) {
            entity.vx -= (1 + bounce) * dot * nx;
            entity.vy -= (1 + bounce) * dot * ny;
          } else if (speed < 10) {
            entity.vx += nx * 28 * bounce;
            entity.vy += ny * 28 * bounce;
          }
        };
        obstacles.forEach((obstacle) => {
          if (obstacle.kind === "rock") {
            const dx = entity.x - obstacle.x;
            const dy = entity.y - obstacle.y;
            const distance = Math.hypot(dx, dy);
            const minDistance = (entity.radius || 0) + obstacle.radius + 2;
            if (distance < minDistance) {
              const push = minDistance - distance;
              let nx = 0;
              let ny = 0;
              if (distance < 0.01) {
                const angle = rand(0, Math.PI * 2);
                nx = Math.cos(angle);
                ny = Math.sin(angle);
              } else {
                nx = dx / distance;
                ny = dy / distance;
              }
              entity.x += nx * push;
              entity.y += ny * push;
              applyBounce(nx, ny);
            }
            return;
          }
          const halfWidth = obstacle.width * 0.5;
          const halfHeight = obstacle.height * 0.5;
          const closestX = clamp(entity.x, obstacle.x - halfWidth, obstacle.x + halfWidth);
          const closestY = clamp(entity.y, obstacle.y - halfHeight, obstacle.y + halfHeight);
          const dx = entity.x - closestX;
          const dy = entity.y - closestY;
          const distance = Math.hypot(dx, dy) || 0;
          const minDistance = (entity.radius || 0) + 2;
          if (distance < minDistance) {
            let push = minDistance - distance;
            let nx = 0;
            let ny = 0;
            if (distance === 0) {
              const offsetX = entity.x - obstacle.x;
              const offsetY = entity.y - obstacle.y;
              const overlapX = halfWidth + minDistance - Math.abs(offsetX);
              const overlapY = halfHeight + minDistance - Math.abs(offsetY);
              if (overlapX < overlapY) {
                nx = offsetX >= 0 ? 1 : -1;
                push = overlapX;
              } else {
                ny = offsetY >= 0 ? 1 : -1;
                push = overlapY;
              }
            } else {
              nx = dx / distance;
              ny = dy / distance;
            }
            entity.x += nx * push;
            entity.y += ny * push;
            applyBounce(nx, ny);
          }
        });
      }

      function getObstacleAvoidance(entity, forwardX, forwardY) {
        if (!entity || !obstacles.length) {
          return { x: 0, y: 0 };
        }
        let steerX = 0;
        let steerY = 0;
        let bestStrength = 0;
        let bestNX = 0;
        let bestNY = 0;
        const bias = Number.isFinite(entity.strafeBias) ? entity.strafeBias : 1;
        const forwardValid = Number.isFinite(forwardX) && Number.isFinite(forwardY);
        const getTangentSign = (nx, ny) => {
          if (!forwardValid) return bias;
          const cross = forwardX * ny - forwardY * nx;
          return cross === 0 ? bias : Math.sign(cross);
        };
        obstacles.forEach((obstacle) => {
          if (obstacle.kind === "rock") {
            const dx = entity.x - obstacle.x;
            const dy = entity.y - obstacle.y;
            const distance = Math.hypot(dx, dy) || 1;
            const range = obstacle.radius + (entity.radius || 0) + 160;
            if (distance < range) {
              const strength = (range - distance) / range;
              const nx = dx / distance;
              const ny = dy / distance;
              steerX += nx * strength;
              steerY += ny * strength;
              const tangentSign = getTangentSign(nx, ny);
              const lateral = strength * 0.38 * tangentSign;
              steerX += -ny * lateral;
              steerY += nx * lateral;
              if (strength > bestStrength) {
                bestStrength = strength;
                bestNX = nx;
                bestNY = ny;
              }
            }
            return;
          }
          const halfWidth = obstacle.width * 0.5;
          const halfHeight = obstacle.height * 0.5;
          const closestX = clamp(entity.x, obstacle.x - halfWidth, obstacle.x + halfWidth);
          const closestY = clamp(entity.y, obstacle.y - halfHeight, obstacle.y + halfHeight);
          const dx = entity.x - closestX;
          const dy = entity.y - closestY;
          const distance = Math.hypot(dx, dy) || 1;
          const range = Math.max(halfWidth, halfHeight) + (entity.radius || 0) + 140;
          if (distance < range) {
            const strength = (range - distance) / range;
            const nx = distance ? dx / distance : (entity.x >= obstacle.x ? 1 : -1);
            const ny = distance ? dy / distance : (entity.y >= obstacle.y ? 1 : -1);
            steerX += nx * strength;
            steerY += ny * strength;
            const tangentSign = getTangentSign(nx, ny);
            const lateral = strength * 0.32 * tangentSign;
            steerX += -ny * lateral;
            steerY += nx * lateral;
            if (strength > bestStrength) {
              bestStrength = strength;
              bestNX = nx;
              bestNY = ny;
            }
          }
        });
        let length = Math.hypot(steerX, steerY);
        if (length < 0.08 && bestStrength > 0.1) {
          const tangentSign = getTangentSign(bestNX, bestNY);
          const tangent = bestStrength * 0.45 * tangentSign;
          steerX = bestNX * bestStrength + -bestNY * tangent;
          steerY = bestNY * bestStrength + bestNX * tangent;
        }
        if (forwardValid) {
          const backward = steerX * forwardX + steerY * forwardY;
          if (backward < 0) {
            steerX -= backward * forwardX;
            steerY -= backward * forwardY;
          }
        }
        length = Math.hypot(steerX, steerY);
        if (!length) {
          return { x: 0, y: 0 };
        }
        const scale = length > 1 ? 1 / length : 1;
        return { x: steerX * scale, y: steerY * scale };
      }

      function getBoundaryAvoidance(entity, forwardX, forwardY) {
        if (!entity) {
          return { x: 0, y: 0 };
        }
        const buffer = 140 + (entity.radius || 0);
        const maxX = state.worldWidth - buffer;
        const maxY = state.worldHeight - buffer;
        let steerX = 0;
        let steerY = 0;
        if (entity.x < buffer) {
          steerX += (buffer - entity.x) / buffer;
        } else if (entity.x > maxX) {
          steerX -= (entity.x - maxX) / buffer;
        }
        if (entity.y < buffer) {
          steerY += (buffer - entity.y) / buffer;
        } else if (entity.y > maxY) {
          steerY -= (entity.y - maxY) / buffer;
        }
        if (Number.isFinite(forwardX) && Number.isFinite(forwardY)) {
          const backward = steerX * forwardX + steerY * forwardY;
          if (backward < 0) {
            steerX -= backward * forwardX;
            steerY -= backward * forwardY;
          }
        }
        const length = Math.hypot(steerX, steerY);
        if (!length) {
          return { x: 0, y: 0 };
        }
        const scale = length > 1 ? 1 / length : 1;
        return { x: steerX * scale, y: steerY * scale };
      }

      function segmentIntersectsCircle(ax, ay, bx, by, cx, cy, radius) {
        const dx = bx - ax;
        const dy = by - ay;
        const fx = ax - cx;
        const fy = ay - cy;
        const a = dx * dx + dy * dy;
        if (a <= 0) return false;
        const b = 2 * (fx * dx + fy * dy);
        const c = fx * fx + fy * fy - radius * radius;
        let discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return false;
        discriminant = Math.sqrt(discriminant);
        const t1 = (-b - discriminant) / (2 * a);
        const t2 = (-b + discriminant) / (2 * a);
        return (t1 >= 0 && t1 <= 1) || (t2 >= 0 && t2 <= 1);
      }

      function segmentIntersectsAabb(ax, ay, bx, by, minX, minY, maxX, maxY) {
        let t0 = 0;
        let t1 = 1;
        const dx = bx - ax;
        const dy = by - ay;
        const p = [-dx, dx, -dy, dy];
        const q = [ax - minX, maxX - ax, ay - minY, maxY - ay];
        for (let i = 0; i < 4; i += 1) {
          if (p[i] === 0) {
            if (q[i] < 0) return false;
          } else {
            const r = q[i] / p[i];
            if (p[i] < 0) {
              if (r > t1) return false;
              if (r > t0) t0 = r;
            } else {
              if (r < t0) return false;
              if (r < t1) t1 = r;
            }
          }
        }
        return true;
      }

      function findBlockingObstacle(entity, target, buffer) {
        if (!entity || !target || !obstacles.length) return null;
        const ax = entity.x;
        const ay = entity.y;
        const bx = target.x;
        const by = target.y;
        let closest = null;
        let bestDist = Infinity;
        obstacles.forEach((obstacle) => {
          let intersects = false;
          if (obstacle.kind === "rock") {
            intersects = segmentIntersectsCircle(
              ax,
              ay,
              bx,
              by,
              obstacle.x,
              obstacle.y,
              obstacle.radius + buffer
            );
          } else {
            const halfWidth = obstacle.width * 0.5 + buffer;
            const halfHeight = obstacle.height * 0.5 + buffer;
            intersects = segmentIntersectsAabb(
              ax,
              ay,
              bx,
              by,
              obstacle.x - halfWidth,
              obstacle.y - halfHeight,
              obstacle.x + halfWidth,
              obstacle.y + halfHeight
            );
          }
          if (intersects) {
            const dist = distanceBetween(entity, obstacle);
            if (dist < bestDist) {
              bestDist = dist;
              closest = obstacle;
            }
          }
        });
        return closest;
      }

      function getEdgePenalty(point) {
        const margin = 170;
        const edgeDistance = Math.min(
          point.x,
          point.y,
          state.worldWidth - point.x,
          state.worldHeight - point.y
        );
        if (edgeDistance >= margin) return 0;
        return (margin - edgeDistance) / margin;
      }

      function scoreNavPoint(point, target) {
        const edgePenalty = getEdgePenalty(point);
        const obstaclePenalty = isPointInObstacle(point, 40) ? 1 : 0;
        const distance = Math.hypot(point.x - target.x, point.y - target.y);
        return distance + edgePenalty * 520 + obstaclePenalty * 900;
      }

      function clampNavPoint(point, margin) {
        return {
          x: clamp(point.x, margin, state.worldWidth - margin),
          y: clamp(point.y, margin, state.worldHeight - margin)
        };
      }

      function getDetourPoint(entity, target, obstacle, buffer) {
        const dx = target.x - entity.x;
        const dy = target.y - entity.y;
        const length = Math.hypot(dx, dy);
        if (!length) return target;
        const dirX = dx / length;
        const dirY = dy / length;
        const perpX = -dirY;
        const perpY = dirX;
        const clearance = obstacle.kind === "rock"
          ? obstacle.radius
          : Math.max(obstacle.width, obstacle.height) * 0.5;
        const offset = clearance + buffer + 90;
        const margin = 120;
        const left = clampNavPoint({ x: obstacle.x + perpX * offset, y: obstacle.y + perpY * offset }, margin);
        const right = clampNavPoint({ x: obstacle.x - perpX * offset, y: obstacle.y - perpY * offset }, margin);
        const leftScore = scoreNavPoint(left, target);
        const rightScore = scoreNavPoint(right, target);
        if (Math.abs(leftScore - rightScore) < 18) {
          return entity.strafeBias >= 0 ? left : right;
        }
        return leftScore < rightScore ? left : right;
      }

      function getNavigationTarget(entity, target) {
        if (!entity || !target) return target;
        const buffer = (entity.radius || 0) + 28;
        const blocking = findBlockingObstacle(entity, target, buffer);
        if (!blocking) return target;
        return getDetourPoint(entity, target, blocking, buffer);
      }

      function getEnemyNavigationTarget(enemy, target, delta, distanceToTarget) {
        if (!enemy || !target) return target;
        const buffer = (enemy.radius || 0) + 26;
        enemy.navTimer = Math.max(0, (enemy.navTimer || 0) - delta);
        enemy.navCooldown = Math.max(0, (enemy.navCooldown || 0) - delta);
        const distance = Number.isFinite(distanceToTarget) ? distanceToTarget : distanceBetween(enemy, target);
        const directBlock = findBlockingObstacle(enemy, target, buffer);
        const detouring = enemy.navTarget && enemy.navTimer > 0;
        if (!detouring) {
          if (!Number.isFinite(enemy.lastTargetDist)) {
            enemy.lastTargetDist = distance;
          } else {
            const progress = enemy.lastTargetDist - distance;
            if (progress < 1.2 && distance > enemy.preferredRange * 0.7) {
              enemy.stuckTimer = (enemy.stuckTimer || 0) + delta;
            } else {
              enemy.stuckTimer = Math.max(0, (enemy.stuckTimer || 0) - delta * 0.6);
            }
            enemy.lastTargetDist = distance;
          }
        } else {
          enemy.stuckTimer = Math.max(0, (enemy.stuckTimer || 0) - delta * 0.8);
          enemy.lastTargetDist = distance;
        }
        if (directBlock && (enemy.stuckTimer > 0.45 || !detouring) && enemy.navCooldown <= 0) {
          enemy.navTarget = getDetourPoint(enemy, target, directBlock, buffer);
          enemy.navTimer = 1.25;
          enemy.navCooldown = 0.4;
          enemy.stuckTimer = 0;
        }
        if (enemy.navTarget && enemy.navTimer > 0) {
          const navDistance = Math.hypot(enemy.navTarget.x - enemy.x, enemy.navTarget.y - enemy.y);
          if (navDistance < 60 || (!directBlock && navDistance < 120)) {
            enemy.navTarget = null;
            enemy.navTimer = 0;
          }
        } else {
          enemy.navTarget = null;
        }
        return enemy.navTarget || target;
      }

      function isOutOfBounds(entity) {
        const margin = 60;
        return entity.x < -margin
          || entity.x > state.worldWidth + margin
          || entity.y < -margin
          || entity.y > state.worldHeight + margin;
      }

      function normalizeKey(event) {
        if (!event) return "";
        if (event.code === "Space") return " ";
        const key = event.key;
        if (!key || key === "Unidentified") return "";
        if (key === " " || key === "Spacebar") return " ";
        return key.toLowerCase();
      }

      function isBindableKey(key) {
        if (!key) return false;
        const blocked = ["meta", "alt", "control", "capslock", "tab"];
        return !blocked.includes(key);
      }

      function getKeybindConflicts(targetAction, key) {
        if (!targetAction || !key) return [];
        return Object.keys(progress.keybinds || {}).filter((action) => {
          if (action === targetAction) return false;
          return progress.keybinds[action] === key;
        });
      }

      function isActionActive(action) {
        const key = progress.keybinds[action];
        if (!key) return false;
        return input.keys.has(key);
      }

      function hasKey(key) {
        return input.keys.has(key);
      }

      function getKeyboardAimVector() {
        if (input.aimMode === "controller") {
          return { active: false, x: 0, y: 0 };
        }
        if (input.aimMode !== "keyboard") {
          const aimKeys = [
            progress.keybinds.aimUp,
            progress.keybinds.aimDown,
            progress.keybinds.aimLeft,
            progress.keybinds.aimRight
          ].filter(Boolean);
          const usesArrowAim = aimKeys.some((key) => key.startsWith("arrow"));
          if (usesArrowAim) {
            return { x: 0, y: 0, active: false };
          }
        }
        let x = 0;
        let y = 0;
        const aimUp = progress.keybinds.aimUp;
        const aimDown = progress.keybinds.aimDown;
        const aimLeft = progress.keybinds.aimLeft;
        const aimRight = progress.keybinds.aimRight;
        if (aimUp && hasKey(aimUp)) y -= 1;
        if (aimDown && hasKey(aimDown)) y += 1;
        if (aimLeft && hasKey(aimLeft)) x -= 1;
        if (aimRight && hasKey(aimRight)) x += 1;
        const length = Math.hypot(x, y);
        if (length === 0) {
          return { active: false, x: 0, y: 0 };
        }
        return { active: true, x: x / length, y: y / length };
      }

      function getAimTarget() {
        if (input.aimSource === "keyboard" && player) {
          return {
            x: player.x + Math.cos(player.angle) * 70,
            y: player.y + Math.sin(player.angle) * 70
          };
        }
        if (input.pointer.active && Number.isFinite(input.pointer.x) && Number.isFinite(input.pointer.y)) {
          return { x: input.pointer.x, y: input.pointer.y };
        }
        if (player) {
          return { x: player.x, y: player.y };
        }
        return { x: state.worldWidth * 0.5, y: state.worldHeight * 0.5 };
      }

      function distanceBetween(a, b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
      }

      function normalizeAngle(angle) {
        return Math.atan2(Math.sin(angle), Math.cos(angle));
      }

      function rotateTowards(current, target, maxDelta) {
        if (!Number.isFinite(maxDelta) || maxDelta <= 0) {
          return normalizeAngle(current);
        }
        const diff = normalizeAngle(target - current);
        if (Math.abs(diff) <= maxDelta) {
          return normalizeAngle(target);
        }
        return normalizeAngle(current + Math.sign(diff) * maxDelta);
      }

      function rand(min, max) {
        return Math.random() * (max - min) + min;
      }

      function randInt(min, max) {
        return Math.floor(rand(min, max + 1));
      }

      function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
      }

      function shuffle(array) {
        for (let i = array.length - 1; i > 0; i -= 1) {
          const j = Math.floor(Math.random() * (i + 1));
          [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
      }

      function pickWeighted(items, weights) {
        const total = weights.reduce((sum, weight) => sum + weight, 0);
        let roll = Math.random() * total;
        for (let i = 0; i < items.length; i += 1) {
          roll -= weights[i];
          if (roll <= 0) {
            return items[i];
          }
        }
        return items[items.length - 1];
      }

      function pick(array) {
        return array[Math.floor(Math.random() * array.length)];
      }

      init();
    })();
