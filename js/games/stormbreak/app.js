(() => {
  'use strict';

  const SAVE_KEY = 'stormbreak-idle-olympus:v1';
  const SAVE_SCHEMA_VERSION = 1;
  const OFFLINE_CAP_MS = 4 * 60 * 60 * 1000;
  const OFFLINE_MIN_MS = 60 * 1000;
  const AUTOSAVE_INTERVAL_MS = 10 * 1000;
  const FIXED_STEP_SECONDS = 0.1;
  const MAX_FRAME_SECONDS = 0.5;
  const LOGICAL_WIDTH = 960;
  const LOGICAL_HEIGHT = 540;
  const MAX_DAMAGE_SPLATS = 48;
  const MAX_EFFECTS = 32;
  const MAX_PARTICLES = 72;
  const GAME_ANALYTICS_ID = 'stormbreak_idle_olympus';
  const scriptSource = document.currentScript ? document.currentScript.src : '';
  const GENERATED_ASSETS = Object.freeze({
    'temple-of-ash': Object.freeze([
      'img/games/stormbreak/temple-of-ash.webp',
      'img/games/stormbreak/temple-of-ash.png'
    ]),
    'zeus-sprite-strip': Object.freeze([
      'img/games/stormbreak/zeus-sprite-strip.webp',
      'img/games/stormbreak/zeus-sprite-strip.png'
    ]),
    'mythic-atlas': Object.freeze([
      'img/games/stormbreak/mythic-atlas.webp',
      'img/games/stormbreak/mythic-atlas.png'
    ])
  });

  const ZONES = Object.freeze({
    temple: Object.freeze({
      id: 'temple',
      name: 'Temple of Ash',
      unlockLevel: 1,
      difficulty: 1,
      goldMultiplier: 1.2,
      ambrosiaMultiplier: 1.1,
      enemySprites: Object.freeze([0, 1, 0, 3]),
      bossSprite: 2,
      tint: 'rgba(218, 105, 32, 0.055)'
    }),
    cyclops: Object.freeze({
      id: 'cyclops',
      name: 'Cliffs of the Cyclops',
      unlockLevel: 6,
      difficulty: 2.2,
      goldMultiplier: 1.8,
      ambrosiaMultiplier: 1.3,
      enemySprites: Object.freeze([1, 0, 2, 1]),
      bossSprite: 2,
      tint: 'rgba(109, 94, 190, 0.09)'
    }),
    elysium: Object.freeze({
      id: 'elysium',
      name: 'Fields of Elysium',
      unlockLevel: 14,
      difficulty: 4.4,
      goldMultiplier: 2.5,
      ambrosiaMultiplier: 1.8,
      enemySprites: Object.freeze([3, 2, 0, 3]),
      bossSprite: 3,
      tint: 'rgba(71, 164, 218, 0.1)'
    })
  });

  const ENEMY_ARCHETYPES = Object.freeze([
    Object.freeze({ name: 'Ashhorn Raider', hp: 34, attack: 5, interval: 2.1 }),
    Object.freeze({ name: 'Cyclopean Mauler', hp: 52, attack: 7, interval: 2.5 }),
    Object.freeze({ name: 'Obsidian Minotaur', hp: 78, attack: 9, interval: 2.35 }),
    Object.freeze({ name: 'Elysian Sentinel', hp: 66, attack: 8, interval: 1.9 })
  ]);

  const BOSS_NAMES = Object.freeze({
    temple: Object.freeze(['Asterion, Ash Tyrant', 'Bronzehoof the Unbroken']),
    cyclops: Object.freeze(['Brontes, Mountain Breaker', 'Steropes the Relentless']),
    elysium: Object.freeze(['Talos, Warden of Elysium', 'The Gilded Colossus'])
  });

  const ABILITIES = Object.freeze({
    bolt: Object.freeze({ cooldown: 5, damage: 3.25 }),
    storm: Object.freeze({ cooldown: 14, damage: 8 }),
    aegis: Object.freeze({ cooldown: 18, duration: 6 })
  });

  const UPGRADE_CONFIG = Object.freeze({
    lightning: Object.freeze({ baseCost: 50, growth: 1.62, maxLevel: 50 }),
    harvest: Object.freeze({ baseCost: 75, growth: 1.66, maxLevel: 50 }),
    aegis: Object.freeze({ baseCost: 100, growth: 1.7, maxLevel: 50 })
  });

  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));

  function finiteNumber(value, fallback, minimum, maximum) {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return clamp(number, minimum, maximum);
  }

  function finiteInteger(value, fallback, minimum, maximum) {
    return Math.floor(finiteNumber(value, fallback, minimum, maximum));
  }

  function formatNumber(value) {
    const safeValue = Math.max(0, Number.isFinite(value) ? value : 0);
    if (safeValue < 1000) return Math.floor(safeValue).toLocaleString();

    const units = [
      { threshold: 1e12, suffix: 'T' },
      { threshold: 1e9, suffix: 'B' },
      { threshold: 1e6, suffix: 'M' },
      { threshold: 1e3, suffix: 'K' }
    ];
    const unit = units.find((entry) => safeValue >= entry.threshold);
    const scaled = safeValue / unit.threshold;
    const digits = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
    return `${scaled.toFixed(digits).replace(/\.0+$|(?<=\.[0-9])0+$/, '')}${unit.suffix}`;
  }

  function formatRate(value) {
    if (!Number.isFinite(value) || value <= 0) return '0';
    if (value < 0.01) return '<0.01';
    if (value < 10) return value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
    return formatNumber(value);
  }

  function formatDuration(milliseconds) {
    const totalMinutes = Math.max(1, Math.floor(milliseconds / 60000));
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    if (!hours) return `${totalMinutes} minute${totalMinutes === 1 ? '' : 's'}`;
    if (!minutes) return `${hours} hour${hours === 1 ? '' : 's'}`;
    return `${hours}h ${minutes}m`;
  }

  function pushBounded(collection, value, limit) {
    collection.push(value);
    if (collection.length > limit) collection.splice(0, collection.length - limit);
  }

  function nextXpForLevel(level) {
    return Math.min(2e9, Math.floor(100 * Math.pow(1.28, Math.max(0, level - 1))));
  }

  function createDefaultState() {
    return {
      schemaVersion: SAVE_SCHEMA_VERSION,
      savedAt: Date.now(),
      gold: 0,
      ambrosia: 0,
      level: 1,
      xp: 0,
      hp: 100,
      autoAttack: true,
      activeZone: 'temple',
      upgrades: {
        lightning: 1,
        harvest: 1,
        aegis: 1
      },
      zoneWaves: {
        temple: 1,
        cyclops: 1,
        elysium: 1
      },
      totalEnemiesDefeated: 0,
      milestones: {}
    };
  }

  function maxHpForState(candidate) {
    return 100 + Math.max(0, candidate.upgrades.aegis - 1) * 25;
  }

  function normalizeState(value) {
    const fallback = createDefaultState();
    if (!value || typeof value !== 'object' || value.schemaVersion !== SAVE_SCHEMA_VERSION) {
      return fallback;
    }

    const upgrades = value.upgrades && typeof value.upgrades === 'object' ? value.upgrades : {};
    const zoneWaves = value.zoneWaves && typeof value.zoneWaves === 'object' ? value.zoneWaves : {};
    const milestones = value.milestones && typeof value.milestones === 'object' ? value.milestones : {};
    const state = {
      schemaVersion: SAVE_SCHEMA_VERSION,
      savedAt: finiteInteger(value.savedAt, Date.now(), 0, Date.now() + 60000),
      gold: finiteNumber(value.gold, 0, 0, 1e15),
      ambrosia: finiteNumber(value.ambrosia, 0, 0, 1e12),
      level: finiteInteger(value.level, 1, 1, 250),
      xp: finiteNumber(value.xp, 0, 0, 2e9),
      hp: 100,
      autoAttack: value.autoAttack !== false,
      activeZone: Object.prototype.hasOwnProperty.call(ZONES, value.activeZone)
        ? value.activeZone
        : 'temple',
      upgrades: {
        lightning: finiteInteger(upgrades.lightning, 1, 1, UPGRADE_CONFIG.lightning.maxLevel),
        harvest: finiteInteger(upgrades.harvest, 1, 1, UPGRADE_CONFIG.harvest.maxLevel),
        aegis: finiteInteger(upgrades.aegis, 1, 1, UPGRADE_CONFIG.aegis.maxLevel)
      },
      zoneWaves: {
        temple: finiteInteger(zoneWaves.temple, 1, 1, 10000),
        cyclops: finiteInteger(zoneWaves.cyclops, 1, 1, 10000),
        elysium: finiteInteger(zoneWaves.elysium, 1, 1, 10000)
      },
      totalEnemiesDefeated: finiteInteger(value.totalEnemiesDefeated, 0, 0, 1e9),
      milestones: {}
    };

    Object.keys(milestones).slice(0, 64).forEach((key) => {
      if (/^[a-z0-9_-]{1,64}$/i.test(key) && milestones[key] === true) state.milestones[key] = true;
    });
    state.hp = finiteNumber(value.hp, maxHpForState(state), 0, maxHpForState(state));

    if (state.level < ZONES[state.activeZone].unlockLevel) state.activeZone = 'temple';
    const neededXp = nextXpForLevel(state.level);
    if (state.xp >= neededXp) state.xp = neededXp - 1;
    return state;
  }

  function init() {
    const root = document.querySelector('[data-stormbreak-root]');
    const canvas = document.querySelector('#stormbreak-canvas');
    if (!root || !(canvas instanceof HTMLCanvasElement) || root.dataset.stormbreakReady === 'true') return;

    root.dataset.stormbreakReady = 'true';
    canvas.width = LOGICAL_WIDTH;
    canvas.height = LOGICAL_HEIGHT;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) return;
    context.imageSmoothingEnabled = false;

    const loading = root.querySelector('[data-stormbreak-loading]');
    const status = document.querySelector('#stormbreak-status');
    const offlineDialog = document.querySelector('#stormbreak-offline-dialog');
    const resetDialog = document.querySelector('#stormbreak-reset-dialog');
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const bindNodes = new Map();

    root.querySelectorAll('[data-bind]').forEach((node) => {
      const name = node.dataset.bind;
      if (!bindNodes.has(name)) bindNodes.set(name, []);
      bindNodes.get(name).push(node);
    });

    let storageAvailable = probeStorage();
    let state = loadState();
    let audioContext = null;
    let soundEnabled = false;
    let assetsReady = false;
    let reducedMotion = reducedMotionQuery.matches;
    let pendingOfflineReward = null;
    let lastStatusMessage = '';
    let lastFrameTime = performance.now();
    let accumulator = 0;
    let hiddenStartedAt = document.hidden ? Date.now() : 0;
    let hiddenRewardEligible = state.autoAttack;
    let autosaveTimer = 0;

    const images = {
      background: null,
      zeus: null,
      atlas: null
    };

    const runtime = {
      elapsed: 0,
      enemy: null,
      defeatedThisWave: 0,
      autoTimer: 0.35,
      manualCooldown: 0,
      userPaused: false,
      hiddenPaused: document.hidden,
      dialogPaused: false,
      shieldRemaining: 0,
      heroAnimation: 'idle',
      heroAnimationRemaining: 0,
      enemyFlashRemaining: 0,
      enemyAttackFlash: 0,
      screenShake: 0,
      cooldowns: {
        bolt: 0,
        storm: 0,
        aegis: 0
      },
      damageSplats: [],
      effects: [],
      particles: [],
      goldEvents: [],
      ambrosiaEvents: [],
      uiDirty: true,
      bossAnnouncementKey: ''
    };

    function probeStorage() {
      try {
        const probeKey = `${SAVE_KEY}:probe`;
        window.localStorage.setItem(probeKey, '1');
        window.localStorage.removeItem(probeKey);
        return true;
      } catch (error) {
        return false;
      }
    }

    function loadState() {
      if (!storageAvailable) return createDefaultState();
      try {
        const rawValue = window.localStorage.getItem(SAVE_KEY);
        if (!rawValue) return createDefaultState();
        return normalizeState(JSON.parse(rawValue));
      } catch (error) {
        return createDefaultState();
      }
    }

    function saveGame() {
      if (!storageAvailable) return false;
      try {
        state.savedAt = Date.now();
        window.localStorage.setItem(SAVE_KEY, JSON.stringify(state));
        return true;
      } catch (error) {
        storageAvailable = false;
        announce('Saving is unavailable. Progress will remain in this tab only.');
        return false;
      }
    }

    function announce(message) {
      if (!status || !message || message === lastStatusMessage) return;
      lastStatusMessage = message;
      status.textContent = message;
    }

    function trackMilestone(key, milestone, details = {}) {
      if (state.milestones[key]) return;
      state.milestones[key] = true;
      if (typeof window.gaEvent === 'function') {
        window.gaEvent('game_milestone', {
          game_id: GAME_ANALYTICS_ID,
          milestone_id: milestone,
          outcome: details.outcome || details.zone || details.upgrade || 'completed',
          duration_bucket: details.duration_bucket,
          score_bucket: details.score_bucket || (details.wave ? `wave_${Math.min(100, details.wave)}` : undefined)
        });
      }
    }

    function setBindText(name, value) {
      const nodes = bindNodes.get(name) || [];
      nodes.forEach((node) => {
        if (node.classList.contains('meter-fill')) return;
        node.textContent = String(value);
      });
    }

    function setProgress(name, ratio) {
      const percent = clamp(ratio, 0, 1) * 100;
      const nodes = bindNodes.get(name) || [];
      nodes.forEach((node) => {
        if (node.classList.contains('meter-fill')) {
          node.style.setProperty('--progress', percent.toFixed(2));
          const meter = node.closest('[role="progressbar"]');
          if (meter) meter.setAttribute('aria-valuenow', String(Math.round(percent)));
        }
      });
    }

    function isPaused() {
      return !assetsReady || runtime.userPaused || runtime.hiddenPaused || runtime.dialogPaused;
    }

    function currentZone() {
      return ZONES[state.activeZone];
    }

    function currentWave() {
      return state.zoneWaves[state.activeZone];
    }

    function isBossWave(wave = currentWave()) {
      return wave % 5 === 0;
    }

    function enemiesInWave(wave = currentWave()) {
      return isBossWave(wave) ? 1 : 10;
    }

    function lightningMultiplier() {
      return 1 + state.upgrades.lightning * 0.12;
    }

    function harvestMultiplier() {
      return 1 + state.upgrades.harvest * 0.07;
    }

    function aegisMitigation() {
      return Math.min(0.45, state.upgrades.aegis * 0.03);
    }

    function playerDamage() {
      return 13 * lightningMultiplier() * (1 + Math.max(0, state.level - 1) * 0.035);
    }

    function autoAttackInterval() {
      return Math.max(0.52, 0.92 - Math.min(0.4, state.level * 0.006));
    }

    function upgradeCost(name) {
      const config = UPGRADE_CONFIG[name];
      const level = state.upgrades[name];
      if (level >= config.maxLevel) return Infinity;
      return Math.min(1e15, Math.floor(config.baseCost * Math.pow(config.growth, level - 1)));
    }

    function upgradeEffect(name) {
      if (name === 'lightning') return `+${state.upgrades.lightning * 12}% damage`;
      if (name === 'harvest') return `+${state.upgrades.harvest * 7}% gold`;
      const hpBonus = Math.max(0, state.upgrades.aegis - 1) * 25;
      return `-${Math.round(aegisMitigation() * 100)}% damage · +${hpBonus} HP`;
    }

    function spawnEnemy() {
      const zone = currentZone();
      const wave = currentWave();
      const boss = isBossWave(wave);
      const poolIndex = (wave + runtime.defeatedThisWave - 1) % zone.enemySprites.length;
      const spriteIndex = boss ? zone.bossSprite : zone.enemySprites[poolIndex];
      const archetype = ENEMY_ARCHETYPES[spriteIndex];
      const waveScale = Math.pow(1.155, Math.max(0, wave - 1));
      const veteranScale = 1 + (runtime.defeatedThisWave % 4) * 0.045;
      const bossScale = boss ? 5.4 : 1;
      const maxHp = Math.max(1, Math.floor(archetype.hp * zone.difficulty * waveScale * veteranScale * bossScale));
      const bossNames = BOSS_NAMES[zone.id];

      runtime.enemy = {
        name: boss ? bossNames[Math.floor(wave / 5) % bossNames.length] : archetype.name,
        spriteIndex,
        boss,
        maxHp,
        hp: maxHp,
        attackDamage: Math.max(1, archetype.attack * zone.difficulty * Math.pow(1.095, wave - 1) * (boss ? 1.45 : 1)),
        attackInterval: Math.max(1.05, archetype.interval - Math.min(0.45, wave * 0.012)),
        attackTimer: 0.9 + Math.random() * 0.7,
        goldReward: Math.max(1, Math.floor((9 + archetype.hp * 0.065) * zone.goldMultiplier * harvestMultiplier() * Math.pow(1.12, wave - 1) * (boss ? 5.8 : 1))),
        xpReward: Math.max(1, Math.floor((11 + archetype.hp * 0.05) * zone.difficulty * Math.pow(1.06, wave - 1) * (boss ? 4.2 : 1)))
      };
      runtime.enemyFlashRemaining = 0;
      runtime.uiDirty = true;

      if (boss) {
        const bossKey = `${zone.id}:${wave}`;
        if (runtime.bossAnnouncementKey !== bossKey) {
          runtime.bossAnnouncementKey = bossKey;
          announce(`${runtime.enemy.name} has entered the battlefield.`);
        }
      }
    }

    function addDamageSplat(x, y, text, color, critical = false) {
      pushBounded(runtime.damageSplats, {
        x,
        y,
        text,
        color,
        critical,
        age: 0,
        duration: reducedMotion ? 0.55 : 0.9
      }, MAX_DAMAGE_SPLATS);
    }

    function addEffect(type, options = {}) {
      pushBounded(runtime.effects, {
        type,
        age: 0,
        duration: options.duration || 0.45,
        x: options.x === undefined ? 708 : options.x,
        y: options.y === undefined ? 292 : options.y,
        seed: options.seed === undefined ? Math.random() * 1000 : options.seed,
        spriteIndex: options.spriteIndex,
        color: options.color || '#f9dd64'
      }, MAX_EFFECTS);
    }

    function burstParticles(x, y, color, count) {
      const particleLimit = reducedMotion ? Math.min(16, MAX_PARTICLES) : MAX_PARTICLES;
      const safeCount = reducedMotion ? Math.min(3, count) : count;
      for (let index = 0; index < safeCount; index += 1) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 25 + Math.random() * 75;
        pushBounded(runtime.particles, {
          x,
          y,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed - 20,
          size: 2 + Math.random() * 4,
          color,
          age: 0,
          duration: 0.35 + Math.random() * 0.55
        }, particleLimit);
      }
    }

    function recordResourceEvent(collection, amount) {
      collection.push({ amount, time: runtime.elapsed });
      if (collection.length > 40) collection.splice(0, collection.length - 40);
    }

    function dealEnemyDamage(baseAmount, options = {}) {
      const enemy = runtime.enemy;
      if (!enemy || enemy.hp <= 0) return 0;
      const criticalChance = options.criticalChance === undefined ? 0.11 : options.criticalChance;
      const critical = Math.random() < criticalChance;
      const amount = Math.max(1, baseAmount * (critical ? 1.75 : 1));
      const applied = Math.min(enemy.hp, amount);
      enemy.hp = Math.max(0, enemy.hp - amount);
      runtime.enemyFlashRemaining = reducedMotion ? 0.04 : 0.12;
      runtime.screenShake = reducedMotion ? 0 : Math.min(8, (options.shake || 2) + (critical ? 3 : 0));
      runtime.heroAnimation = options.heroAnimation || 'attack';
      runtime.heroAnimationRemaining = options.animationDuration || 0.28;
      addDamageSplat(
        700 + Math.random() * 95,
        205 + Math.random() * 45,
        `${critical ? 'CRIT ' : ''}${formatNumber(Math.ceil(applied))}`,
        options.color || '#fff0a6',
        critical
      );
      if (options.effect !== false) addEffect(options.effect || 'bolt', options);
      burstParticles(options.x || 716, options.y || 300, options.color || '#f9dd64', critical ? 11 : 6);
      playSound(critical ? 'critical' : 'hit');
      runtime.uiDirty = true;

      if (enemy.hp <= 0) defeatEnemy(enemy);
      return applied;
    }

    function awardExperience(amount) {
      state.xp += amount;
      const previousLevel = state.level;
      while (state.level < 250 && state.xp >= nextXpForLevel(state.level)) {
        state.xp -= nextXpForLevel(state.level);
        state.level += 1;
      }

      if (state.level > previousLevel) {
        const oldMaxHp = 100 + Math.max(0, state.upgrades.aegis - 1) * 25;
        state.hp = Math.min(oldMaxHp, state.hp + 20 + (state.level - previousLevel) * 5);
        addDamageSplat(258, 160, `LEVEL ${state.level}`, '#86d9ff', true);
        addEffect('level', { x: 246, y: 292, duration: 1, color: '#72c8ff' });
        playSound('level');
        const newlyUnlocked = Object.values(ZONES).filter((zone) => (
          zone.unlockLevel > previousLevel && zone.unlockLevel <= state.level
        ));
        if (newlyUnlocked.length) {
          announce(`${newlyUnlocked.map((zone) => zone.name).join(' and ')} unlocked at level ${state.level}.`);
        } else {
          announce(`Zeus reached level ${state.level}.`);
        }
      }
    }

    function defeatEnemy(enemy) {
      state.gold = Math.min(1e15, state.gold + enemy.goldReward);
      recordResourceEvent(runtime.goldEvents, enemy.goldReward);
      awardExperience(enemy.xpReward);
      state.totalEnemiesDefeated += 1;
      trackMilestone('first_enemy', 'first_enemy', { zone: state.activeZone, wave: currentWave() });
      addEffect('loot', { x: 720, y: 285, duration: 1.05, spriteIndex: 4, color: '#ffc452' });

      if (enemy.boss) {
        const ambrosiaReward = Math.max(1, Math.floor(currentZone().ambrosiaMultiplier * (1 + currentWave() / 20)));
        state.ambrosia = Math.min(1e12, state.ambrosia + ambrosiaReward);
        recordResourceEvent(runtime.ambrosiaEvents, ambrosiaReward);
        addEffect('loot', { x: 770, y: 275, duration: 1.2, spriteIndex: 5, color: '#ff74cf' });
        trackMilestone(`boss_clear_${state.activeZone}`, 'boss_clear', {
          zone: state.activeZone,
          wave: currentWave()
        });
        announce(`${enemy.name} fell. Zeus claimed ${formatNumber(enemy.goldReward)} gold and ${ambrosiaReward} ambrosia.`);
      }

      runtime.defeatedThisWave += 1;
      const waveTarget = enemiesInWave();
      if (runtime.defeatedThisWave >= waveTarget) {
        const clearedWave = currentWave();
        state.zoneWaves[state.activeZone] = Math.min(10000, clearedWave + 1);
        runtime.defeatedThisWave = 0;
        state.hp = Math.min(maxHpForState(state), state.hp + maxHpForState(state) * 0.08);
        if (clearedWave % 10 === 0) {
          trackMilestone(`zone_clear_${state.activeZone}`, 'zone_clear', {
            zone: state.activeZone,
            wave: clearedWave
          });
          announce(`${currentZone().name} wave ${clearedWave} cleared. The farming route is secure.`);
        } else if (!enemy.boss) {
          announce(`${currentZone().name} wave ${clearedWave} cleared.`);
        }
      }

      runtime.enemy = null;
      spawnEnemy();
      runtime.uiDirty = true;
    }

    function enemyAttack() {
      const enemy = runtime.enemy;
      if (!enemy) return;
      const shieldMultiplier = runtime.shieldRemaining > 0 ? 0.28 : 1;
      const damage = Math.max(1, enemy.attackDamage * (1 - aegisMitigation()) * shieldMultiplier);
      state.hp = Math.max(0, state.hp - damage);
      runtime.enemyAttackFlash = reducedMotion ? 0.04 : 0.16;
      addDamageSplat(245 + Math.random() * 55, 225, `-${formatNumber(Math.ceil(damage))}`, '#ff7d70');
      addEffect('enemy-hit', { x: 270, y: 315, duration: 0.28, color: '#ff6b5f' });
      burstParticles(270, 315, '#ff796d', 6);
      playSound('hurt');

      if (state.hp <= 0) {
        const lostGold = Math.min(state.gold, Math.floor(state.gold * 0.03));
        state.gold -= lostGold;
        state.hp = maxHpForState(state);
        enemy.hp = Math.max(enemy.hp, enemy.maxHp * 0.35);
        runtime.shieldRemaining = 2;
        announce(lostGold
          ? `Zeus rallied, but the retreat cost ${formatNumber(lostGold)} gold.`
          : 'Zeus rallied beneath the Aegis and returned to battle.');
      }
      runtime.uiDirty = true;
    }

    function performBasicAttack(manual = false, point = null) {
      if (isPaused() || !runtime.enemy) return false;
      const targetX = point ? point.x : 720;
      const targetY = point ? point.y : 295;
      const manualMultiplier = manual ? 1.18 : 1;
      dealEnemyDamage(playerDamage() * manualMultiplier, {
        x: targetX,
        y: targetY,
        effect: 'bolt',
        color: manual ? '#f9ee8d' : '#bdeeff',
        criticalChance: manual ? 0.16 : 0.1,
        shake: manual ? 3 : 1.5
      });
      return true;
    }

    function activateAbility(name) {
      if (!Object.prototype.hasOwnProperty.call(ABILITIES, name) || isPaused()) return false;
      if (runtime.cooldowns[name] > 0) return false;
      const ability = ABILITIES[name];
      runtime.cooldowns[name] = ability.cooldown;

      if (name === 'bolt') {
        dealEnemyDamage(playerDamage() * ability.damage, {
          effect: 'bolt-heavy',
          color: '#fff36b',
          criticalChance: 0.2,
          heroAnimation: 'attack',
          animationDuration: 0.42,
          shake: 5
        });
        announce('Bolt struck with the force of Olympus.');
      } else if (name === 'storm') {
        for (let index = 0; index < (reducedMotion ? 2 : 5); index += 1) {
          addEffect('storm', {
            x: 625 + Math.random() * 190,
            y: 245 + Math.random() * 100,
            duration: 0.55 + Math.random() * 0.25,
            seed: Math.random() * 1000,
            color: index % 2 ? '#62d8ff' : '#fff96b'
          });
        }
        dealEnemyDamage(playerDamage() * ability.damage, {
          effect: false,
          color: '#7ee5ff',
          criticalChance: 0.1,
          heroAnimation: 'storm',
          animationDuration: 0.68,
          shake: 8
        });
        announce('The storm answered Zeus.');
        playSound('storm');
      } else {
        runtime.shieldRemaining = ability.duration;
        state.hp = Math.min(maxHpForState(state), state.hp + maxHpForState(state) * 0.18);
        runtime.heroAnimation = 'aegis';
        runtime.heroAnimationRemaining = 0.55;
        addEffect('shield', { x: 264, y: 310, duration: ability.duration, color: '#83d6ff' });
        addDamageSplat(255, 206, 'AEGIS', '#90e6ff', true);
        announce('Aegis raised: incoming damage is heavily reduced for 6 seconds.');
        playSound('shield');
      }

      runtime.uiDirty = true;
      return true;
    }

    function buyUpgrade(name) {
      if (!Object.prototype.hasOwnProperty.call(UPGRADE_CONFIG, name) || isPaused()) return false;
      const cost = upgradeCost(name);
      if (!Number.isFinite(cost) || state.gold < cost) return false;
      const previousMaxHp = maxHpForState(state);
      state.gold -= cost;
      state.upgrades[name] += 1;
      if (name === 'aegis') state.hp += maxHpForState(state) - previousMaxHp;
      trackMilestone('first_upgrade', 'first_upgrade', { upgrade: name, level: state.upgrades[name] });
      announce(`${name.charAt(0).toUpperCase()}${name.slice(1)} upgraded to level ${state.upgrades[name]}.`);
      playSound('upgrade');
      runtime.uiDirty = true;
      saveGame();
      return true;
    }

    function selectZone(zoneId) {
      const zone = ZONES[zoneId];
      if (!zone || state.level < zone.unlockLevel || state.activeZone === zoneId) return false;
      state.activeZone = zoneId;
      runtime.defeatedThisWave = 0;
      runtime.enemy = null;
      runtime.bossAnnouncementKey = '';
      spawnEnemy();
      announce(`${zone.name} selected. Rewards and danger have increased.`);
      runtime.uiDirty = true;
      saveGame();
      canvas.setAttribute('aria-label', `Stormbreak battle at ${zone.name}`);
      return true;
    }

    function updateTransientObjects(step) {
      runtime.damageSplats.forEach((splat) => {
        splat.age += step;
      });
      runtime.damageSplats = runtime.damageSplats.filter((splat) => splat.age < splat.duration);

      runtime.effects.forEach((effect) => {
        effect.age += step;
      });
      runtime.effects = runtime.effects.filter((effect) => effect.age < effect.duration);

      runtime.particles.forEach((particle) => {
        particle.age += step;
        particle.x += particle.vx * step;
        particle.y += particle.vy * step;
        particle.vy += 95 * step;
      });
      runtime.particles = runtime.particles.filter((particle) => particle.age < particle.duration);
    }

    function simulate(step) {
      runtime.elapsed += step;
      runtime.manualCooldown = Math.max(0, runtime.manualCooldown - step);
      runtime.shieldRemaining = Math.max(0, runtime.shieldRemaining - step);
      runtime.heroAnimationRemaining = Math.max(0, runtime.heroAnimationRemaining - step);
      runtime.enemyFlashRemaining = Math.max(0, runtime.enemyFlashRemaining - step);
      runtime.enemyAttackFlash = Math.max(0, runtime.enemyAttackFlash - step);
      runtime.screenShake = Math.max(0, runtime.screenShake - step * 25);
      Object.keys(runtime.cooldowns).forEach((name) => {
        runtime.cooldowns[name] = Math.max(0, runtime.cooldowns[name] - step);
      });

      if (!runtime.enemy) spawnEnemy();
      if (state.autoAttack) {
        runtime.autoTimer -= step;
        if (runtime.autoTimer <= 0) {
          performBasicAttack(false);
          runtime.autoTimer += autoAttackInterval();
        }
      }

      if (runtime.enemy) {
        runtime.enemy.attackTimer -= step;
        if (runtime.enemy.attackTimer <= 0) {
          enemyAttack();
          runtime.enemy.attackTimer += runtime.enemy.attackInterval;
        }
      }

      if (runtime.shieldRemaining <= 0 && state.hp < maxHpForState(state)) {
        state.hp = Math.min(maxHpForState(state), state.hp + maxHpForState(state) * 0.0025 * step);
      }
      updateTransientObjects(step);
      runtime.uiDirty = true;
    }

    function recentRate(collection, windowSeconds) {
      const oldestTime = runtime.elapsed - windowSeconds;
      while (collection.length && collection[0].time < oldestTime) collection.shift();
      if (!collection.length) return 0;
      const span = Math.max(3, Math.min(windowSeconds, runtime.elapsed - collection[0].time + 1));
      return collection.reduce((sum, entry) => sum + entry.amount, 0) / span;
    }

    function estimatedGoldRate() {
      const zone = currentZone();
      const wave = currentWave();
      const averageHp = 52 * zone.difficulty * Math.pow(1.155, wave - 1);
      const averageReward = 12 * zone.goldMultiplier * harvestMultiplier() * Math.pow(1.12, wave - 1);
      return Math.max(0, playerDamage() / autoAttackInterval() / Math.max(1, averageHp) * averageReward * 0.78);
    }

    function updateUI(force = false) {
      if (!force && !runtime.uiDirty) return;
      runtime.uiDirty = false;
      const zone = currentZone();
      const wave = currentWave();
      const waveTarget = enemiesInWave(wave);
      const xpNeeded = nextXpForLevel(state.level);
      const hpMax = maxHpForState(state);
      const boss = isBossWave(wave);
      const liveGoldRate = recentRate(runtime.goldEvents, 30) || estimatedGoldRate();
      const liveAmbrosiaRate = recentRate(runtime.ambrosiaEvents, 300);

      setBindText('zone-name', zone.name);
      setBindText('wave', wave);
      setBindText('objective', boss
        ? `Defeat ${runtime.enemy ? runtime.enemy.name : 'the Olympian guardian'}`
        : `Defeat the ${zone.id === 'elysium' ? 'sentinels' : 'horde'}`);
      setBindText('boss-progress', `${runtime.defeatedThisWave} / ${waveTarget}`);
      setBindText('level', state.level);
      setBindText('gold', formatNumber(state.gold));
      setBindText('gold-rate', formatRate(liveGoldRate));
      setBindText('ambrosia', formatNumber(state.ambrosia));
      setBindText('ambrosia-rate', formatRate(liveAmbrosiaRate));
      setBindText('hp', formatNumber(Math.ceil(state.hp)));
      setBindText('hp-max', formatNumber(hpMax));
      setBindText('xp', formatNumber(Math.floor(state.xp)));
      setBindText('xp-next', formatNumber(xpNeeded));
      setBindText('xp-progress', Math.floor(state.xp / xpNeeded * 100));
      setBindText('auto-label', state.autoAttack ? 'ON' : 'OFF');
      setProgress('hp-progress', state.hp / hpMax);
      setProgress('xp-progress', state.xp / xpNeeded);
      root.style.setProperty('--boss-progress', `${clamp(runtime.defeatedThisWave / waveTarget, 0, 1) * 100}%`);

      root.querySelectorAll('[data-upgrade]').forEach((button) => {
        const name = button.dataset.upgrade;
        if (!Object.prototype.hasOwnProperty.call(UPGRADE_CONFIG, name)) return;
        const level = state.upgrades[name];
        const cost = upgradeCost(name);
        const levelNode = button.querySelector('[data-upgrade-level]');
        const costNode = button.querySelector('[data-upgrade-cost]');
        const effectNode = button.querySelector('[data-upgrade-effect]');
        if (levelNode) levelNode.textContent = String(level);
        if (costNode) costNode.textContent = Number.isFinite(cost) ? formatNumber(cost) : 'MAX';
        if (effectNode) effectNode.textContent = upgradeEffect(name);
        button.disabled = isPaused() || !Number.isFinite(cost) || state.gold < cost;
        button.setAttribute('aria-label', Number.isFinite(cost)
          ? `Upgrade ${name} to level ${level + 1} for ${formatNumber(cost)} gold`
          : `${name} is at maximum level`);
      });

      root.querySelectorAll('[data-zone]').forEach((button) => {
        const zoneId = button.dataset.zone;
        const candidate = ZONES[zoneId];
        if (!candidate) return;
        const locked = state.level < candidate.unlockLevel;
        const selected = state.activeZone === zoneId;
        const statusNode = button.querySelector('[data-zone-status]');
        button.disabled = locked;
        button.setAttribute('aria-disabled', String(locked));
        button.setAttribute('aria-pressed', String(selected));
        button.classList.toggle('is-selected', selected);
        button.classList.toggle('is-locked', locked);
        if (statusNode) {
          statusNode.classList.toggle('zone-status--locked', locked);
          statusNode.textContent = locked ? `Level ${candidate.unlockLevel}` : selected ? 'Active' : 'Select';
        }
      });

      root.querySelectorAll('[data-ability]').forEach((button) => {
        const name = button.dataset.ability;
        const ability = ABILITIES[name];
        if (!ability) return;
        const remaining = runtime.cooldowns[name];
        const cooldownNode = button.querySelector('[data-ability-cooldown]');
        const readyRatio = 1 - clamp(remaining / ability.cooldown, 0, 1);
        button.style.setProperty('--cooldown-progress', `${(readyRatio * 100).toFixed(1)}%`);
        button.disabled = isPaused() || remaining > 0;
        button.classList.toggle('is-cooling-down', remaining > 0);
        if (cooldownNode) cooldownNode.textContent = isPaused()
          ? 'Paused'
          : remaining > 0
            ? `${remaining.toFixed(1)}s`
            : 'Ready';
      });

      root.querySelectorAll('[data-action="toggle-auto"]').forEach((button) => {
        button.setAttribute('aria-pressed', String(state.autoAttack));
      });
      root.querySelectorAll('[data-action="pause"]').forEach((button) => {
        button.setAttribute('aria-pressed', String(runtime.userPaused));
        const label = button.querySelector('span');
        if (label) label.textContent = runtime.userPaused ? 'Resume' : 'Pause';
      });
      root.querySelectorAll('[data-action="sound"]').forEach((button) => {
        button.setAttribute('aria-pressed', String(soundEnabled));
        const label = button.querySelector('span');
        if (label) label.textContent = soundEnabled ? 'Sound on' : 'Sound off';
      });
    }

    function siteBaseUrl() {
      if (scriptSource) return new URL('../../../', scriptSource);
      return new URL('/', window.location.href);
    }

    function loadGeneratedImage(name) {
      const assetPaths = GENERATED_ASSETS[name];
      if (!assetPaths) return Promise.reject(new Error(`Unknown generated asset: ${name}`));
      const baseUrl = siteBaseUrl();
      const candidates = assetPaths.map((assetPath) => new URL(assetPath, baseUrl).href);

      return new Promise((resolve, reject) => {
        const image = new Image();
        let candidateIndex = 0;
        image.decoding = 'async';
        image.onload = () => resolve(image);
        image.onerror = () => {
          candidateIndex += 1;
          if (candidateIndex >= candidates.length) {
            reject(new Error(`Unable to load ${name}`));
            return;
          }
          image.src = candidates[candidateIndex];
        };
        image.src = candidates[candidateIndex];
      });
    }

    function drawCover(image) {
      const sourceRatio = image.naturalWidth / image.naturalHeight;
      const targetRatio = LOGICAL_WIDTH / LOGICAL_HEIGHT;
      let sourceX = 0;
      let sourceY = 0;
      let sourceWidth = image.naturalWidth;
      let sourceHeight = image.naturalHeight;
      if (sourceRatio > targetRatio) {
        sourceWidth = image.naturalHeight * targetRatio;
        sourceX = (image.naturalWidth - sourceWidth) / 2;
      } else {
        sourceHeight = image.naturalWidth / targetRatio;
        sourceY = (image.naturalHeight - sourceHeight) / 2;
      }
      context.drawImage(
        image,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        0,
        0,
        LOGICAL_WIDTH,
        LOGICAL_HEIGHT
      );
    }

    function drawZeusCell(frameIndex, x, y, width, height) {
      const image = images.zeus;
      if (!image) return;
      const cellWidth = image.naturalWidth / 4;
      const cellHeight = image.naturalHeight;
      const safeFrame = clamp(Math.floor(frameIndex), 0, 3);
      context.drawImage(
        image,
        safeFrame * cellWidth,
        0,
        cellWidth,
        cellHeight,
        x,
        y,
        width,
        height
      );
    }

    function drawAtlasCell(cellIndex, x, y, width, height) {
      const image = images.atlas;
      if (!image) return;
      const cellWidth = image.naturalWidth / 4;
      const cellHeight = image.naturalHeight / 2;
      const safeIndex = clamp(Math.floor(cellIndex), 0, 7);
      const column = safeIndex % 4;
      const row = Math.floor(safeIndex / 4);
      context.drawImage(
        image,
        column * cellWidth,
        row * cellHeight,
        cellWidth,
        cellHeight,
        x,
        y,
        width,
        height
      );
    }

    function drawLightning(effect, heavy = false) {
      const progress = effect.age / effect.duration;
      const alpha = Math.sin(Math.PI * clamp(progress, 0, 1));
      const startX = heavy ? 330 : 320;
      const startY = heavy ? 120 : 260;
      const segments = reducedMotion ? 4 : heavy ? 10 : 7;
      context.save();
      context.globalAlpha = alpha;
      context.lineCap = 'square';
      context.lineJoin = 'miter';
      context.shadowColor = effect.color;
      context.shadowBlur = heavy ? 18 : 10;
      context.strokeStyle = '#f7fdff';
      context.lineWidth = heavy ? 6 : 3;
      context.beginPath();
      context.moveTo(startX, startY);
      for (let index = 1; index < segments; index += 1) {
        const amount = index / segments;
        const offset = Math.sin(effect.seed + index * 12.91) * (heavy ? 28 : 18) * (1 - Math.abs(amount - 0.5));
        context.lineTo(
          startX + (effect.x - startX) * amount + offset,
          startY + (effect.y - startY) * amount
        );
      }
      context.lineTo(effect.x, effect.y);
      context.stroke();
      context.strokeStyle = effect.color;
      context.lineWidth = heavy ? 2.5 : 1.25;
      context.stroke();
      context.restore();
    }

    function drawEffects() {
      runtime.effects.forEach((effect) => {
        const progress = effect.age / effect.duration;
        const alpha = 1 - clamp(progress, 0, 1);
        if (effect.type === 'bolt' || effect.type === 'bolt-heavy') {
          drawLightning(effect, effect.type === 'bolt-heavy');
          return;
        }
        if (effect.type === 'storm') {
          const stormEffect = {
            ...effect,
            x: effect.x,
            y: effect.y
          };
          drawLightning(stormEffect, true);
          return;
        }
        if (effect.type === 'shield') {
          const pulse = 1 + Math.sin(effect.age * 8) * (reducedMotion ? 0 : 0.035);
          context.save();
          context.globalAlpha = 0.22 + Math.min(0.45, alpha * 0.4);
          context.strokeStyle = effect.color;
          context.shadowColor = effect.color;
          context.shadowBlur = 16;
          context.lineWidth = 5;
          context.beginPath();
          context.ellipse(effect.x, effect.y, 128 * pulse, 178 * pulse, 0, 0, Math.PI * 2);
          context.stroke();
          context.restore();
          return;
        }
        if (effect.type === 'loot') {
          const rise = reducedMotion ? 28 : progress * 72;
          const size = effect.spriteIndex === 5 ? 62 : 54;
          context.save();
          context.globalAlpha = Math.min(1, alpha * 1.8);
          drawAtlasCell(effect.spriteIndex, effect.x - size / 2, effect.y - rise, size, size);
          context.restore();
          return;
        }
        if (effect.type === 'level') {
          context.save();
          context.globalAlpha = alpha;
          context.strokeStyle = effect.color;
          context.shadowColor = effect.color;
          context.shadowBlur = 14;
          context.lineWidth = 4;
          context.beginPath();
          context.arc(effect.x, effect.y, 45 + progress * 110, 0, Math.PI * 2);
          context.stroke();
          context.restore();
          return;
        }
        if (effect.type === 'enemy-hit') {
          context.save();
          context.globalAlpha = alpha;
          context.strokeStyle = effect.color;
          context.lineWidth = 6;
          context.beginPath();
          context.moveTo(effect.x - 45, effect.y - 35);
          context.lineTo(effect.x + 35, effect.y + 25);
          context.moveTo(effect.x + 35, effect.y - 32);
          context.lineTo(effect.x - 35, effect.y + 30);
          context.stroke();
          context.restore();
        }
      });
    }

    function drawParticles() {
      runtime.particles.forEach((particle) => {
        const alpha = 1 - particle.age / particle.duration;
        context.save();
        context.globalAlpha = clamp(alpha, 0, 1);
        context.fillStyle = particle.color;
        context.shadowColor = particle.color;
        context.shadowBlur = 7;
        context.fillRect(
          Math.round(particle.x - particle.size / 2),
          Math.round(particle.y - particle.size / 2),
          Math.max(1, Math.round(particle.size)),
          Math.max(1, Math.round(particle.size))
        );
        context.restore();
      });
    }

    function drawDamageSplats() {
      runtime.damageSplats.forEach((splat) => {
        const progress = splat.age / splat.duration;
        const lift = reducedMotion ? progress * 18 : progress * 48;
        const scale = splat.critical ? 1.12 + Math.sin(Math.min(1, progress * 2) * Math.PI) * 0.22 : 1;
        context.save();
        context.globalAlpha = clamp(1 - progress, 0, 1);
        context.translate(Math.round(splat.x), Math.round(splat.y - lift));
        context.scale(scale, scale);
        context.font = `${splat.critical ? 800 : 700} ${splat.critical ? 24 : 18}px ui-monospace, monospace`;
        context.textAlign = 'center';
        context.lineWidth = 5;
        context.strokeStyle = '#07111d';
        context.strokeText(splat.text, 0, 0);
        context.fillStyle = splat.color;
        context.fillText(splat.text, 0, 0);
        context.restore();
      });
    }

    function heroFrame() {
      if (runtime.heroAnimationRemaining <= 0) return 0;
      if (runtime.heroAnimation === 'storm') return 2;
      if (runtime.heroAnimation === 'aegis') return 3;
      return runtime.heroAnimationRemaining > 0.13 ? 1 : 2;
    }

    function drawEnemyHealth(enemy, x, y, width) {
      const ratio = clamp(enemy.hp / enemy.maxHp, 0, 1);
      context.save();
      context.fillStyle = 'rgba(3, 10, 17, 0.86)';
      context.fillRect(x, y, width, 16);
      context.fillStyle = enemy.boss ? '#ef445d' : '#d96b48';
      context.fillRect(x + 2, y + 2, (width - 4) * ratio, 12);
      context.strokeStyle = enemy.boss ? '#ffbf61' : 'rgba(218, 232, 241, 0.72)';
      context.lineWidth = enemy.boss ? 2 : 1;
      context.strokeRect(x + 0.5, y + 0.5, width - 1, 15);
      context.font = `700 ${enemy.boss ? 15 : 13}px ui-monospace, monospace`;
      context.textAlign = 'center';
      context.lineWidth = 4;
      context.strokeStyle = '#06101a';
      context.strokeText(enemy.name, x + width / 2, y - 8);
      context.fillStyle = enemy.boss ? '#ffd37a' : '#eef8ff';
      context.fillText(enemy.name, x + width / 2, y - 8);
      context.restore();
    }

    function render() {
      context.save();
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);
      context.fillStyle = '#06111c';
      context.fillRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);
      if (!assetsReady) {
        context.restore();
        return;
      }

      const shakeX = reducedMotion ? 0 : Math.sin(runtime.elapsed * 71) * runtime.screenShake;
      const shakeY = reducedMotion ? 0 : Math.cos(runtime.elapsed * 83) * runtime.screenShake * 0.45;
      context.translate(shakeX, shakeY);
      drawCover(images.background);
      context.fillStyle = currentZone().tint;
      context.fillRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);

      const groundShade = context.createLinearGradient(0, 270, 0, LOGICAL_HEIGHT);
      groundShade.addColorStop(0, 'rgba(3, 10, 19, 0)');
      groundShade.addColorStop(1, 'rgba(3, 10, 19, 0.38)');
      context.fillStyle = groundShade;
      context.fillRect(0, 250, LOGICAL_WIDTH, LOGICAL_HEIGHT - 250);

      const heroBob = reducedMotion ? 0 : Math.sin(runtime.elapsed * 2.2) * 2;
      drawZeusCell(heroFrame(), 82, 32 + heroBob, 290, 580);

      const enemy = runtime.enemy;
      if (enemy) {
        const enemySize = enemy.boss ? 355 : 310;
        const enemyX = enemy.boss ? 553 : 582;
        const enemyBob = reducedMotion ? 0 : Math.sin(runtime.elapsed * 2.8 + 1) * 4;
        const enemyY = (enemy.boss ? 147 : 185) + enemyBob;
        context.save();
        if (runtime.enemyFlashRemaining > 0) {
          context.globalAlpha = 0.72 + Math.sin(runtime.enemyFlashRemaining * 90) * 0.18;
          context.shadowColor = '#fff6a1';
          context.shadowBlur = 22;
        }
        drawAtlasCell(enemy.spriteIndex, enemyX, enemyY, enemySize, enemySize);
        context.restore();
        drawEnemyHealth(enemy, enemyX + 30, enemyY - 4, enemySize - 60);
      }

      drawEffects();
      drawParticles();
      drawDamageSplats();

      if (isPaused() && assetsReady && !runtime.hiddenPaused) {
        context.save();
        context.fillStyle = 'rgba(3, 10, 18, 0.54)';
        context.fillRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);
        context.fillStyle = '#f3f8fb';
        context.font = '800 28px ui-monospace, monospace';
        context.textAlign = 'center';
        context.fillText(runtime.dialogPaused ? 'REWARD AWAITS' : 'BATTLE PAUSED', LOGICAL_WIDTH / 2, LOGICAL_HEIGHT / 2);
        context.restore();
      }
      context.restore();
    }

    function frame(timestamp) {
      const frameSeconds = Math.min(MAX_FRAME_SECONDS, Math.max(0, (timestamp - lastFrameTime) / 1000));
      lastFrameTime = timestamp;
      if (!isPaused()) {
        accumulator += frameSeconds;
        while (accumulator >= FIXED_STEP_SECONDS) {
          simulate(FIXED_STEP_SECONDS);
          accumulator -= FIXED_STEP_SECONDS;
        }
      } else {
        accumulator = 0;
      }
      updateUI();
      render();
      window.requestAnimationFrame(frame);
    }

    function canvasPoint(event) {
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return { x: LOGICAL_WIDTH / 2, y: LOGICAL_HEIGHT / 2 };
      return {
        x: clamp((event.clientX - bounds.left) * LOGICAL_WIDTH / bounds.width, 0, LOGICAL_WIDTH),
        y: clamp((event.clientY - bounds.top) * LOGICAL_HEIGHT / bounds.height, 0, LOGICAL_HEIGHT)
      };
    }

    function tryManualAttack(point = null) {
      if (runtime.manualCooldown > 0 || isPaused()) return;
      runtime.manualCooldown = 0.14;
      performBasicAttack(true, point);
    }

    function handleCanvasAttack(event) {
      if (event.button !== undefined && event.button !== 0) return;
      tryManualAttack(canvasPoint(event));
      canvas.focus({ preventScroll: true });
      event.preventDefault();
    }

    function setPaused(paused, shouldAnnounce = true) {
      runtime.userPaused = Boolean(paused);
      accumulator = 0;
      lastFrameTime = performance.now();
      runtime.uiDirty = true;
      if (shouldAnnounce) announce(runtime.userPaused ? 'Battle paused.' : 'Battle resumed.');
      if (runtime.userPaused) saveGame();
    }

    function toggleAutoAttack() {
      state.autoAttack = !state.autoAttack;
      runtime.autoTimer = Math.min(runtime.autoTimer, 0.25);
      runtime.uiDirty = true;
      announce(`Auto battle ${state.autoAttack ? 'enabled' : 'disabled'}.`);
      saveGame();
    }

    function ensureAudioContext() {
      if (audioContext) return audioContext;
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return null;
      try {
        audioContext = new AudioContextClass();
      } catch (error) {
        audioContext = null;
      }
      return audioContext;
    }

    function playSound(type) {
      if (!soundEnabled || !audioContext) return;
      if (audioContext.state === 'suspended') audioContext.resume().catch(() => {});
      const frequencies = {
        hit: 260,
        critical: 520,
        hurt: 120,
        storm: 90,
        shield: 410,
        upgrade: 660,
        level: 780
      };
      const frequency = frequencies[type] || 320;
      const now = audioContext.currentTime;
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      oscillator.type = type === 'hurt' || type === 'storm' ? 'sawtooth' : 'square';
      oscillator.frequency.setValueAtTime(frequency, now);
      oscillator.frequency.exponentialRampToValueAtTime(Math.max(40, frequency * 0.72), now + 0.09);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(type === 'storm' ? 0.055 : 0.035, now + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.11);
      oscillator.connect(gain);
      gain.connect(audioContext.destination);
      oscillator.start(now);
      oscillator.stop(now + 0.12);
    }

    function toggleSound() {
      if (!soundEnabled) {
        const nextContext = ensureAudioContext();
        if (!nextContext) {
          announce('Sound is not supported in this browser.');
          return;
        }
        soundEnabled = true;
        if (nextContext.state === 'suspended') nextContext.resume().catch(() => {});
        playSound('upgrade');
        announce('Sound enabled.');
      } else {
        soundEnabled = false;
        announce('Sound muted.');
      }
      runtime.uiDirty = true;
    }

    function openDialog(dialog) {
      if (!dialog) return false;
      runtime.dialogPaused = true;
      runtime.uiDirty = true;
      if (typeof dialog.showModal === 'function') {
        if (!dialog.open) dialog.showModal();
      } else {
        dialog.hidden = false;
        dialog.setAttribute('open', '');
        dialog.setAttribute('aria-hidden', 'false');
      }
      return true;
    }

    function closeDialog(dialog) {
      if (dialog) {
        if (typeof dialog.close === 'function' && dialog.open) dialog.close();
        else {
          dialog.hidden = true;
          dialog.removeAttribute('open');
          dialog.setAttribute('aria-hidden', 'true');
        }
      }
      const anyOpen = [offlineDialog, resetDialog].some((candidate) => candidate && candidate.open);
      runtime.dialogPaused = anyOpen;
      runtime.uiDirty = true;
      accumulator = 0;
      lastFrameTime = performance.now();
    }

    function estimateOfflineGold(milliseconds) {
      const seconds = Math.min(OFFLINE_CAP_MS, Math.max(0, milliseconds)) / 1000;
      return Math.max(1, Math.floor(estimatedGoldRate() * seconds * 0.72));
    }

    function offerOfflineReward(milliseconds) {
      if (!state.autoAttack || milliseconds < OFFLINE_MIN_MS) return false;
      const cappedDuration = Math.min(milliseconds, OFFLINE_CAP_MS);
      const gold = estimateOfflineGold(cappedDuration);
      if (gold <= 0) return false;
      pendingOfflineReward = { durationMs: cappedDuration, gold };
      if (offlineDialog) {
        const goldNode = offlineDialog.querySelector('[data-offline-gold]');
        const timeNode = offlineDialog.querySelector('[data-offline-time]');
        if (goldNode) goldNode.textContent = formatNumber(gold);
        if (timeNode) timeNode.textContent = formatDuration(cappedDuration);
        openDialog(offlineDialog);
      } else {
        claimOfflineReward();
      }
      return true;
    }

    function claimOfflineReward() {
      if (!pendingOfflineReward) {
        closeDialog(offlineDialog);
        return false;
      }
      const reward = pendingOfflineReward;
      pendingOfflineReward = null;
      state.gold = Math.min(1e15, state.gold + reward.gold);
      trackMilestone('offline_claim', 'offline_claim', {
        outcome: 'claimed',
        duration_bucket: reward.durationMs < 60 * 60 * 1000 ? 'under_1h' : '1_to_4h',
        score_bucket: reward.gold < 1000 ? 'under_1k' : reward.gold < 100000 ? '1k_to_100k' : '100k_plus'
      });
      closeDialog(offlineDialog);
      announce(`Offline reward claimed: ${formatNumber(reward.gold)} gold.`);
      runtime.uiDirty = true;
      saveGame();
      return true;
    }

    function dismissOfflineReward() {
      pendingOfflineReward = null;
      closeDialog(offlineDialog);
      announce('Offline reward dismissed.');
      saveGame();
    }

    function performReset() {
      if (storageAvailable) {
        try {
          window.localStorage.removeItem(SAVE_KEY);
        } catch (error) {
          storageAvailable = false;
        }
      }
      state = createDefaultState();
      pendingOfflineReward = null;
      runtime.enemy = null;
      runtime.defeatedThisWave = 0;
      runtime.autoTimer = 0.25;
      runtime.manualCooldown = 0;
      runtime.userPaused = false;
      runtime.dialogPaused = false;
      runtime.shieldRemaining = 0;
      runtime.heroAnimation = 'idle';
      runtime.heroAnimationRemaining = 0;
      runtime.damageSplats = [];
      runtime.effects = [];
      runtime.particles = [];
      runtime.goldEvents = [];
      runtime.ambrosiaEvents = [];
      runtime.bossAnnouncementKey = '';
      Object.keys(runtime.cooldowns).forEach((name) => {
        runtime.cooldowns[name] = 0;
      });
      closeDialog(resetDialog);
      closeDialog(offlineDialog);
      spawnEnemy();
      announce('Campaign reset. The Temple of Ash awaits.');
      runtime.uiDirty = true;
      saveGame();
      return true;
    }

    function handleControlClick(event) {
      const upgradeButton = event.target.closest('[data-upgrade]');
      if (upgradeButton && root.contains(upgradeButton)) {
        buyUpgrade(upgradeButton.dataset.upgrade);
        return;
      }
      const zoneButton = event.target.closest('[data-zone]');
      if (zoneButton && root.contains(zoneButton)) {
        selectZone(zoneButton.dataset.zone);
        return;
      }
      const abilityButton = event.target.closest('[data-ability]');
      if (abilityButton && root.contains(abilityButton)) {
        activateAbility(abilityButton.dataset.ability);
        return;
      }
      const actionButton = event.target.closest('[data-action]');
      if (!actionButton) return;
      const action = actionButton.dataset.action;
      if (action === 'toggle-auto') toggleAutoAttack();
      else if (action === 'pause') setPaused(!runtime.userPaused);
      else if (action === 'sound') toggleSound();
      else if (action === 'reset') openDialog(resetDialog);
      else if (action === 'claim-offline') claimOfflineReward();
      else if (action === 'dismiss-offline') dismissOfflineReward();
      else if (action === 'confirm-reset') performReset();
      else if (action === 'cancel-reset') closeDialog(resetDialog);
    }

    function isTypingTarget(target) {
      if (!(target instanceof Element)) return false;
      return Boolean(target.closest('input, textarea, select, [contenteditable="true"]'));
    }

    function handleKeydown(event) {
      if (event.repeat || event.altKey || event.ctrlKey || event.metaKey || isTypingTarget(event.target)) return;
      const key = event.key.toLowerCase();
      if ((key === ' ' || key === 'spacebar') && event.target instanceof Element && event.target.closest('button, a')) return;
      if (key === 'enter' && event.target === canvas) tryManualAttack();
      else if (key === 'q' || key === '1') activateAbility('bolt');
      else if (key === 'w' || key === '2') activateAbility('storm');
      else if (key === 'e' || key === '3') activateAbility('aegis');
      else if (key === ' ' || key === 'spacebar') toggleAutoAttack();
      else if (key === 'p') setPaused(!runtime.userPaused);
      else if (key === 'm') toggleSound();
      else if (key === 'r') openDialog(resetDialog);
      else return;
      event.preventDefault();
    }

    function handleVisibilityChange() {
      if (document.hidden) {
        hiddenStartedAt = Date.now();
        hiddenRewardEligible = state.autoAttack && !runtime.userPaused;
        runtime.hiddenPaused = true;
        saveGame();
      } else {
        const hiddenDuration = hiddenStartedAt ? Date.now() - hiddenStartedAt : 0;
        hiddenStartedAt = 0;
        runtime.hiddenPaused = false;
        accumulator = 0;
        lastFrameTime = performance.now();
        if (hiddenRewardEligible) offerOfflineReward(hiddenDuration);
        runtime.uiDirty = true;
      }
    }

    function getState() {
      return {
        schemaVersion: SAVE_SCHEMA_VERSION,
        savedAt: state.savedAt,
        level: state.level,
        xp: state.xp,
        xpNext: nextXpForLevel(state.level),
        hp: state.hp,
        hpMax: maxHpForState(state),
        gold: state.gold,
        ambrosia: state.ambrosia,
        autoAttack: state.autoAttack,
        activeZone: state.activeZone,
        wave: currentWave(),
        defeatedThisWave: runtime.defeatedThisWave,
        enemiesInWave: enemiesInWave(),
        upgrades: { ...state.upgrades },
        zoneWaves: { ...state.zoneWaves },
        cooldowns: { ...runtime.cooldowns },
        paused: isPaused(),
        userPaused: runtime.userPaused,
        soundEnabled,
        reducedMotion,
        enemy: runtime.enemy ? {
          name: runtime.enemy.name,
          hp: runtime.enemy.hp,
          maxHp: runtime.enemy.maxHp,
          boss: runtime.enemy.boss,
          spriteIndex: runtime.enemy.spriteIndex
        } : null
      };
    }

    root.addEventListener('click', handleControlClick);
    canvas.addEventListener('pointerdown', handleCanvasAttack);
    window.addEventListener('keydown', handleKeydown);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('beforeunload', saveGame);
    if (offlineDialog) {
      offlineDialog.addEventListener('cancel', (event) => {
        event.preventDefault();
        dismissOfflineReward();
      });
      offlineDialog.addEventListener('click', handleControlClick);
    }
    if (resetDialog) {
      resetDialog.addEventListener('cancel', (event) => {
        event.preventDefault();
        closeDialog(resetDialog);
      });
      resetDialog.addEventListener('click', handleControlClick);
    }
    const handleReducedMotionChange = (event) => {
      reducedMotion = event.matches;
      if (reducedMotion && runtime.particles.length > 16) runtime.particles.splice(0, runtime.particles.length - 16);
    };
    if (typeof reducedMotionQuery.addEventListener === 'function') {
      reducedMotionQuery.addEventListener('change', handleReducedMotionChange);
    } else if (typeof reducedMotionQuery.addListener === 'function') {
      reducedMotionQuery.addListener(handleReducedMotionChange);
    }

    window.StormbreakGame = Object.freeze({
      getState,
      pause: () => setPaused(true),
      resume: () => setPaused(false),
      activateAbility,
      reset: performReset
    });

    const loadedSavedAt = state.savedAt;
    const offlineDuration = loadedSavedAt > 0 ? Date.now() - loadedSavedAt : 0;
    state.savedAt = Date.now();
    spawnEnemy();
    updateUI(true);
    if (!storageAvailable) announce('Saving is unavailable. Progress will remain in this tab only.');

    Promise.all([
      loadGeneratedImage('temple-of-ash'),
      loadGeneratedImage('zeus-sprite-strip'),
      loadGeneratedImage('mythic-atlas')
    ]).then(([background, zeus, atlas]) => {
      images.background = background;
      images.zeus = zeus;
      images.atlas = atlas;
      assetsReady = true;
      if (loading) loading.hidden = true;
      root.classList.add('is-ready');
      runtime.uiDirty = true;
      offerOfflineReward(offlineDuration);
      if (!pendingOfflineReward && storageAvailable) {
        announce(`Wave ${currentWave()} is ready. Auto battle is ${state.autoAttack ? 'on' : 'off'}.`);
      }
      saveGame();
    }).catch(() => {
      assetsReady = false;
      if (loading) {
        const detail = loading.querySelector('span:last-child');
        if (detail) detail.textContent = 'The generated battlefield art could not be loaded.';
      }
      announce('Stormbreak could not load its generated art. Refresh to try again.');
    });

    autosaveTimer = window.setInterval(saveGame, AUTOSAVE_INTERVAL_MS);
    if (!autosaveTimer) saveGame();
    window.requestAnimationFrame(frame);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
