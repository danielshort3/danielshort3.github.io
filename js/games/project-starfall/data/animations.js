(function initProjectStarfallDataAnimations(global) {
  'use strict';

  function createAnimationData(options) {
    const settings = options || {};
    const ASSET_ROOT = settings.ASSET_ROOT || 'img/project-starfall';
    const EQUIPMENT_ATLAS_ROOT = settings.EQUIPMENT_ATLAS_ROOT || `${ASSET_ROOT}/equipment-atlases`;
    const CLASS_FILE_IDS = settings.CLASS_FILE_IDS || {};
    const createEquipmentVisualData = settings.createEquipmentVisualData;

    const ANIMATION_ROOT = `${ASSET_ROOT}/animations`;
    const COMBAT_FX_ANIMATION_ROOT = `${ANIMATION_ROOT}/combat-fx`;
    const ANIMATION_FRAME_SIZE = 160;
    const COMPACT_ENEMY_FRAME_SIZE = 128;
    const ENEMY_PROJECTILE_FRAME_SIZE = 64;

    const PLAYER_ANIMATION_ROWS = Object.freeze(['idle', 'run', 'jump', 'fall', 'climb', 'basic', 'skill', 'party', 'hit', 'defeat']);
    const ENEMY_ANIMATION_ROWS = Object.freeze(['idle', 'move', 'telegraph', 'attack', 'projectile', 'buff', 'hit', 'defeat']);
    const PET_ANIMATION_ROWS = Object.freeze(['idle', 'run', 'jump', 'fall', 'loot', 'teleport']);
    const SKILL_FX_ANIMATION_ROWS = Object.freeze(['cast', 'projectile', 'impact', 'area']);
    const BASIC_ATTACK_FX_ANIMATION_ROWS = Object.freeze(['cast', 'projectile', 'impact', 'trail']);
    const ENEMY_COMBAT_FX_ANIMATION_ROWS = Object.freeze(['telegraph', 'melee', 'projectile', 'buff', 'impact']);

    const PLAYER_ANIMATION_CONFIG = Object.freeze({
      idle: { frames: 6, fps: 6, loop: true },
      run: { frames: 6, fps: 12, loop: true },
      jump: { frames: 6, fps: 10, loop: false, holds: [1, 1, 2, 2, 3, 3] },
      fall: { frames: 6, fps: 8, loop: false, holds: [1, 1, 2, 2, 3, 4] },
      climb: { frames: 6, fps: 8, loop: true },
      basic: { frames: 6, fps: 14, loop: false },
      skill: { frames: 6, fps: 12, loop: false },
      party: { frames: 6, fps: 12, loop: false },
      hit: { frames: 6, fps: 16, loop: false },
      defeat: { frames: 6, fps: 9, loop: false }
    });

    const ENEMY_ANIMATION_CONFIG = Object.freeze({
      idle: { frames: 6, fps: 6, loop: true },
      move: { frames: 6, fps: 10, loop: true },
      telegraph: { frames: 6, fps: 12, loop: false },
      attack: { frames: 6, fps: 16, loop: false },
      projectile: { frames: 6, fps: 14, loop: false },
      buff: { frames: 6, fps: 12, loop: false },
      hit: { frames: 6, fps: 16, loop: false },
      defeat: { frames: 6, fps: 9, loop: false }
    });

    const PET_ANIMATION_CONFIG = Object.freeze({
      idle: { frames: 6, fps: 6, loop: true },
      run: { frames: 6, fps: 12, loop: true },
      jump: { frames: 6, fps: 10, loop: false },
      fall: { frames: 6, fps: 9, loop: false },
      loot: { frames: 6, fps: 10, loop: true, loopDelay: 0.12 },
      teleport: { frames: 6, fps: 12, loop: false }
    });

    const FX_ANIMATION_CONFIG = Object.freeze({
      slash: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
      cast: { frames: 6, fps: 14, loop: true, loopDelay: 0.18 },
      arrowRelease: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
      partyBuff: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 },
      impact: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
      defeatBurst: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 }
    });

    const COMBAT_FX_ANIMATION_CONFIG = Object.freeze({
      cast: { frames: 6, fps: 16, loop: true, loopDelay: 0.18 },
      projectile: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
      impact: { frames: 6, fps: 20, loop: true, loopDelay: 0.18 },
      area: { frames: 6, fps: 14, loop: true, loopDelay: 0.18 },
      trail: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
      telegraph: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 },
      melee: { frames: 6, fps: 18, loop: true, loopDelay: 0.18 },
      buff: { frames: 6, fps: 12, loop: true, loopDelay: 0.18 }
    });

    const PORTAL_ANIMATION_CONFIG = Object.freeze({
      idle: { frames: 6, fps: 7, loop: true, loopDelay: 0.1, sequence: [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0] }
    });

    function normalizeFrameHolds(holds, frames) {
      if (!Array.isArray(holds)) return null;
      const frameCount = Math.max(1, Number(frames) || holds.length || 1);
      const normalized = [];
      for (let index = 0; index < frameCount; index += 1) {
        normalized.push(Math.max(1, Math.round(Number(holds[index]) || 1)));
      }
      return Object.freeze(normalized);
    }

    function normalizeFrameSequence(sequence, frames) {
      if (!Array.isArray(sequence)) return null;
      const frameCount = Math.max(1, Number(frames) || 1);
      const normalized = sequence
        .map((frame) => Math.floor(Number(frame) || 0))
        .filter((frame) => frame >= 0 && frame < frameCount);
      return normalized.length ? Object.freeze(normalized) : null;
    }

    function freezeAnimationStateConfig(config) {
      const state = Object.assign({}, config || {});
      if (Array.isArray(state.holds)) state.holds = normalizeFrameHolds(state.holds, state.frames);
      if (Array.isArray(state.sequence)) state.sequence = normalizeFrameSequence(state.sequence, state.frames);
      if (Object.prototype.hasOwnProperty.call(state, 'loopDelay')) state.loopDelay = Math.max(0, Number(state.loopDelay) || 0);
      return Object.freeze(state);
    }

    function makeAnimationStates(rows, config, overrides) {
      return Object.freeze(rows.reduce((states, state, row) => {
        states[state] = freezeAnimationStateConfig(Object.assign({ row }, config[state], overrides && overrides[state]));
        return states;
      }, {}));
    }

    function makeSheetAnimation(sheet, rows, config, overrides, options) {
      const settings = options || {};
      return Object.freeze({
        sheet,
        frameWidth: Number(settings.frameWidth || ANIMATION_FRAME_SIZE),
        frameHeight: Number(settings.frameHeight || settings.frameWidth || ANIMATION_FRAME_SIZE),
        states: makeAnimationStates(rows, config, overrides)
      });
    }

    function makePlayerAnimationAsset(fileId) {
      return makeSheetAnimation(`${ANIMATION_ROOT}/players/${fileId}-sheet.png`, PLAYER_ANIMATION_ROWS, PLAYER_ANIMATION_CONFIG);
    }

    const EQUIPMENT_ATLAS_ANGLE_SETS = Object.freeze({
      weapon: Object.freeze([-62, -34, -7, 8, 25, 42, 62, 78]),
      body: Object.freeze([-42, -18, -7, 0, 16, 42, 72, 90]),
      limb: Object.freeze([-15, -7, 0, 5, 10, 15, 20, 30])
    });
    const EQUIPMENT_ATLAS_WEAPON_KINDS = Object.freeze(['sword', 'axe', 'wand', 'staff', 'bow']);
    const EQUIPMENT_ATLAS_LIMB_KINDS = Object.freeze(['boots', 'gloves', 'grip', 'ring', 'focus']);

    function getEquipmentAtlasAngles(kind) {
      const itemKind = String(kind || '').toLowerCase();
      if (EQUIPMENT_ATLAS_WEAPON_KINDS.includes(itemKind)) return EQUIPMENT_ATLAS_ANGLE_SETS.weapon;
      if (EQUIPMENT_ATLAS_LIMB_KINDS.includes(itemKind)) return EQUIPMENT_ATLAS_ANGLE_SETS.limb;
      return EQUIPMENT_ATLAS_ANGLE_SETS.body;
    }

    function makeEquipmentVisualAtlas(fileId, kind) {
      const itemKind = String(kind || '').toLowerCase();
      const variants = itemKind === 'bow'
        ? Object.freeze(['rest', 'draw', 'release'])
        : Object.freeze(['default']);
      return Object.freeze({
        sheet: `${EQUIPMENT_ATLAS_ROOT}/${fileId}-atlas.png`,
        frameWidth: 128,
        frameHeight: 128,
        pivotX: 64,
        pivotY: 64,
        angles: getEquipmentAtlasAngles(itemKind),
        variants,
        kind: itemKind
      });
    }

    function mergeAnimationOverrides(base, specific) {
      return Object.freeze(ENEMY_ANIMATION_ROWS.reduce((merged, stateId) => {
        const next = Object.assign({}, base && base[stateId], specific && specific[stateId]);
        if (Object.keys(next).length) merged[stateId] = freezeAnimationStateConfig(next);
        return merged;
      }, {}));
    }

    function makeEnemyAnimationAsset(fileId, enemyId) {
      return makeSheetAnimation(
        `${ANIMATION_ROOT}/enemies/${fileId}-sheet.png`,
        ENEMY_ANIMATION_ROWS,
        ENEMY_ANIMATION_CONFIG,
        mergeAnimationOverrides(ENEMY_ANIMATION_ROW_HOLDS, ENEMY_ANIMATION_TIMING_OVERRIDES[enemyId])
      );
    }

    function compactFrameHolds(holds, fallback) {
      const source = Array.isArray(holds) && holds.length ? holds : fallback;
      if (!Array.isArray(source) || !source.length) return [1, 1, 1];
      if (source.length === 3) return source.map((hold) => Math.max(1, Math.round(Number(hold) || 1)));
      return [source[0], source[Math.floor(source.length / 2)], source[source.length - 1]]
        .map((hold) => Math.max(1, Math.round(Number(hold) || 1)));
    }

    function makeCompactEnemyAnimationAsset(fileId, enemyId) {
      const compactConfig = Object.freeze(ENEMY_ANIMATION_ROWS.reduce((config, stateId) => {
        const base = ENEMY_ANIMATION_CONFIG[stateId] || {};
        const timingOverride = ENEMY_ANIMATION_TIMING_OVERRIDES[enemyId] && ENEMY_ANIMATION_TIMING_OVERRIDES[enemyId][stateId] || {};
        const fallbackHolds = stateId === 'idle' ? [4, 2, 4] : stateId === 'defeat' ? [1, 2, 5] : [1, 1, 2];
        const fallbackFps = stateId === 'idle' ? 5 : stateId === 'move' ? 9 : stateId === 'defeat' ? 7 : base.fps || 10;
        config[stateId] = Object.assign({}, base, timingOverride, {
          frames: 3,
          fps: Number(timingOverride.fps || fallbackFps),
          holds: compactFrameHolds(timingOverride.holds, fallbackHolds)
        });
        return config;
      }, {}));
      return makeSheetAnimation(
        `${ANIMATION_ROOT}/enemies/${fileId}-compact-sheet.png`,
        ENEMY_ANIMATION_ROWS,
        compactConfig,
        null,
        { frameWidth: COMPACT_ENEMY_FRAME_SIZE, frameHeight: COMPACT_ENEMY_FRAME_SIZE }
      );
    }

    function makeEnemyProjectileAnimationAsset(fileId) {
      return makeSheetAnimation(
        `${ANIMATION_ROOT}/enemy-projectiles/${fileId}-sheet.png`,
        ['projectile'],
        Object.freeze({ projectile: { frames: 3, fps: 12, loop: true, loopDelay: 0.18, holds: [1, 1, 1] } }),
        null,
        { frameWidth: ENEMY_PROJECTILE_FRAME_SIZE, frameHeight: ENEMY_PROJECTILE_FRAME_SIZE }
      );
    }

    function makePetAnimationAsset(fileId) {
      return makeSheetAnimation(`${ANIMATION_ROOT}/pets/${fileId}-sheet.png`, PET_ANIMATION_ROWS, PET_ANIMATION_CONFIG);
    }

    function makeFxAnimationAsset(fileId, stateId) {
      return makeSheetAnimation(`${ANIMATION_ROOT}/fx/${fileId}-sheet.png`, [stateId], Object.freeze({ [stateId]: FX_ANIMATION_CONFIG[stateId] }));
    }

    function makeCombatFxAnimationAsset(fileId, folder, rows) {
      const rowIds = Object.freeze((rows || SKILL_FX_ANIMATION_ROWS).slice());
      return makeSheetAnimation(
        `${COMBAT_FX_ANIMATION_ROOT}/${folder}/${fileId}-sheet.png`,
        rowIds,
        Object.freeze(rowIds.reduce((config, rowId) => {
          config[rowId] = COMBAT_FX_ANIMATION_CONFIG[rowId] || COMBAT_FX_ANIMATION_CONFIG.impact;
          return config;
        }, {}))
      );
    }

    function makePortalAnimationAsset(fileId) {
      return makeSheetAnimation(`${ANIMATION_ROOT}/portals/${fileId}-sheet.png`, ['idle'], PORTAL_ANIMATION_CONFIG);
    }

    const GENERIC_PLAYER_ANIMATION_ASSET = makePlayerAnimationAsset('generic-player');

    const PLAYER_ANIMATION_ASSETS = Object.freeze(Object.keys(CLASS_FILE_IDS).reduce((assets, classId) => {
      // Class sheets currently contain the same pixels. Reuse the generic
      // definition so one decoded sheet serves every class until unique art
      // replaces the generated duplicates.
      assets[classId] = GENERIC_PLAYER_ANIMATION_ASSET;
      return assets;
    }, {}));

    const equipmentVisualData = typeof createEquipmentVisualData === 'function'
      ? createEquipmentVisualData({
          makeEquipmentVisualAtlas
        })
      : Object.freeze({});
    const EQUIPMENT_VISUALS = equipmentVisualData.EQUIPMENT_VISUALS;
    const FIGHTER_RIG_ANIMATION_STATES = equipmentVisualData.FIGHTER_RIG_ANIMATION_STATES;
    const FIGHTER_RIG_EQUIPMENT_VISUALS = equipmentVisualData.FIGHTER_RIG_EQUIPMENT_VISUALS;

    const GENERIC_PLAYER_RIG = Object.freeze({
        id: 'genericPlayer',
        renderer: 'blockyLayeredCanvas',
        width: 78,
        height: 92,
        scale: 0.74,
        groundY: 47,
        style: 'blocky-pixel-runtime-v1',
        palette: Object.freeze({
          outline: '#13222f',
          shadow: '#0e2636',
          skin: '#d99a6c',
          skinLight: '#f0bd88',
          hair: '#3b2725',
          shirt: '#526f86',
          shirtLight: '#8caabd',
          pants: '#33495b',
          pantsDark: '#24313f',
          belt: '#7d5132',
          boot: '#2a1f21',
          hand: '#e4aa78'
        }),
        drawOrder: Object.freeze([
          'backAura',
          'backArm',
          'backLeg',
          'frontLeg',
          'torso',
          'chest',
          'head',
          'offhand',
          'weapon',
          'frontArm',
          'frontHand',
          'frontAura'
        ]),
        anchors: Object.freeze({
          torso: Object.freeze({ x: 0, y: -18 }),
          head: Object.freeze({ x: 3, y: -51 }),
          face: Object.freeze({ x: 14, y: -50 }),
          chest: Object.freeze({ x: 3, y: -24 }),
          back: Object.freeze({ x: -13, y: -27 }),
          backShoulder: Object.freeze({ x: -8, y: -28 }),
          frontShoulder: Object.freeze({ x: 10, y: -27 }),
          backHip: Object.freeze({ x: -5, y: 3 }),
          frontHip: Object.freeze({ x: 7, y: 3 }),
          mainHand: Object.freeze({ x: 22, y: -10 }),
          weaponHand: Object.freeze({ x: 22, y: -10 }),
          offHand: Object.freeze({ x: -15, y: -8 }),
          offhand: Object.freeze({ x: -15, y: -8 }),
          backFoot: Object.freeze({ x: -7, y: 39 }),
          frontFoot: Object.freeze({ x: 11, y: 39 }),
          boots: Object.freeze({ x: 2, y: 39 }),
          gloves: Object.freeze({ x: 22, y: -10 }),
          ringCharm: Object.freeze({ x: 0, y: -18 }),
          weaponTip: Object.freeze({ x: 59, y: -10 }),
          aura: Object.freeze({ x: 0, y: -18 })
        }),
        attachmentAnchors: Object.freeze({
          head: Object.freeze({ anchor: 'head', layer: 'head' }),
          face: Object.freeze({ anchor: 'face', layer: 'head' }),
          chest: Object.freeze({ anchor: 'chest', layer: 'torso' }),
          back: Object.freeze({ anchor: 'back', layer: 'backAura' }),
          mainHand: Object.freeze({ anchor: 'mainHand', layer: 'frontHand' }),
          offHand: Object.freeze({ anchor: 'offHand', layer: 'offhand' }),
          gloves: Object.freeze({ anchor: 'gloves', layer: 'hands' }),
          boots: Object.freeze({ anchor: 'boots', layer: 'feet' }),
          ringCharm: Object.freeze({ anchor: 'ringCharm', layer: 'aura' }),
          weaponTip: Object.freeze({ anchor: 'weaponTip', layer: 'weapon' })
        }),
        equipmentSlots: Object.freeze({
          weapon: 'weaponHand',
          offhand: 'offhand',
          head: 'head',
          chest: 'torso',
          gloves: 'weaponHand',
          boots: 'feet',
          ring: 'weaponHand',
          amulet: 'aura'
        }),
        attachments: Object.freeze({
          weapon: Object.freeze({ slot: 'weapon', anchor: 'weaponHand', layer: 'weapon' }),
          offhand: Object.freeze({ slot: 'offhand', anchor: 'offhand', layer: 'offhand' }),
          chest: Object.freeze({ slot: 'chest', anchor: 'torso', layer: 'torso' }),
          boots: Object.freeze({ slot: 'boots', anchor: 'feet', layer: 'feet' }),
          ring: Object.freeze({ slot: 'ring', anchor: 'weaponHand', layer: 'aura' }),
          amulet: Object.freeze({ slot: 'amulet', anchor: 'aura', layer: 'aura' }),
          head: Object.freeze({ slot: 'head', anchor: 'head', layer: 'head' }),
          gloves: Object.freeze({ slot: 'gloves', anchor: 'weaponHand', layer: 'hands' })
        }),
        animationStates: FIGHTER_RIG_ANIMATION_STATES,
        equipmentVisuals: FIGHTER_RIG_EQUIPMENT_VISUALS
      });

    const PLAYER_RIGS = Object.freeze(Object.keys(CLASS_FILE_IDS).reduce((rigs, classId) => {
      rigs[classId] = GENERIC_PLAYER_RIG;
      return rigs;
    }, {}));

    function freezeAnimationOverrideMap(map) {
      return Object.freeze(Object.keys(map || {}).reduce((frozen, stateId) => {
        frozen[stateId] = freezeAnimationStateConfig(map[stateId]);
        return frozen;
      }, {}));
    }

    function freezeEnemyTimingOverrides(map) {
      return Object.freeze(Object.keys(map || {}).reduce((frozen, enemyId) => {
        frozen[enemyId] = freezeAnimationOverrideMap(map[enemyId]);
        return frozen;
      }, {}));
    }

    const ENEMY_ANIMATION_ROW_HOLDS = freezeAnimationOverrideMap({
      idle: { holds: [5, 2, 3, 2, 3, 5] },
      move: { holds: [1, 1, 2, 1, 1, 2] },
      telegraph: { holds: [3, 2, 1, 1, 1, 2] },
      attack: { holds: [2, 1, 1, 1, 2, 3] },
      projectile: { holds: [2, 1, 1, 1, 2, 2] },
      buff: { holds: [3, 1, 2, 1, 2, 3] },
      hit: { holds: [1, 1, 2, 2, 2, 3] },
      defeat: { holds: [1, 1, 2, 2, 4, 6] }
    });

    const ENEMY_ANIMATION_TIMING_OVERRIDES = freezeEnemyTimingOverrides({
      briarStag: {
        idle: { fps: 5, holds: [6, 2, 3, 2, 3, 6] },
        move: { fps: 9, holds: [2, 1, 2, 1, 2, 2] },
        telegraph: { fps: 10, holds: [4, 2, 1, 1, 2, 3] },
        attack: { fps: 13, holds: [3, 1, 1, 2, 2, 4] },
        projectile: { fps: 11, holds: [2, 1, 1, 2, 2, 3] },
        buff: { fps: 9, holds: [4, 1, 2, 1, 3, 4] },
        hit: { fps: 12, holds: [1, 1, 2, 2, 3, 4] },
        defeat: { fps: 7, holds: [1, 1, 2, 3, 5, 8] }
      },
      vineSnapper: {
        idle: { fps: 5, holds: [5, 2, 4, 2, 4, 5] },
        attack: { fps: 14, holds: [3, 1, 1, 1, 2, 4] },
        defeat: { fps: 7, holds: [1, 1, 2, 3, 5, 8] }
      },
      banditCutter: {
        idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
        move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
        attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
      },
      banditCutterDirect: {
        idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
        move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
        attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
      },
      banditCutterReference: {
        idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
        move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
        attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
      },
      banditCutterHybrid: {
        idle: { fps: 6, holds: [4, 2, 2, 2, 3, 4] },
        move: { fps: 11, holds: [1, 1, 1, 2, 1, 2] },
        attack: { fps: 17, holds: [2, 1, 1, 1, 2, 3] }
      },
      banditCutterPuppet: {
        idle: { fps: 6, holds: [5, 2, 3, 2, 3, 5] },
        move: { fps: 10, holds: [2, 1, 2, 1, 2, 2] },
        attack: { fps: 16, holds: [2, 1, 1, 2, 2, 3] }
      },
      banditThrower: {
        idle: { fps: 6, holds: [4, 2, 3, 2, 3, 4] },
        telegraph: { fps: 11, holds: [3, 2, 1, 1, 2, 3] },
        projectile: { fps: 15, holds: [2, 1, 1, 1, 2, 3] }
      },
      thunderRam: {
        idle: { fps: 5, holds: [6, 2, 3, 2, 3, 6] },
        move: { fps: 12, holds: [1, 1, 1, 1, 2, 2] },
        attack: { fps: 16, holds: [3, 1, 1, 1, 2, 3] }
      },
      glacierSentinel: {
        idle: { fps: 4, holds: [7, 2, 4, 2, 4, 7] },
        attack: { fps: 12, holds: [4, 2, 1, 1, 2, 5] },
        defeat: { fps: 6, holds: [1, 2, 3, 4, 6, 9] }
      }
    });

    const ENEMY_ANIMATION_FILE_IDS = Object.freeze({
      slimelet: 'slimelet',
      dewSlime: 'dew-slime',
      mossback: 'mossback',
      thornSprout: 'thorn-sprout',
      vineSnapper: 'vine-snapper',
      bristleBoar: 'bristle-boar',
      briarStag: 'briar-stag',
      dustImp: 'dust-imp',
      clockbug: 'clockbug',
      rustRatchet: 'rust-ratchet',
      coilSentry: 'coil-sentry',
      scrapWarden: 'scrap-warden',
      emberWisp: 'ember-wisp',
      ashCrawler: 'ash-crawler',
      lavaTick: 'lava-tick',
      cinderSpitter: 'cinder-spitter',
      banditCutter: 'bandit-cutter',
      banditCutterDirect: 'bandit-cutter-direct',
      banditCutterReference: 'bandit-cutter-reference',
      banditCutterHybrid: 'bandit-cutter-hybrid',
      banditCutterPuppet: 'bandit-cutter-puppet',
      banditThrower: 'bandit-thrower',
      orebackBeetle: 'oreback-beetle',
      glowcapHealer: 'glowcap-healer',
      crackedMimic: 'cracked-mimic',
      brambleking: 'brambleking',
      clockworkTitan: 'clockwork-titan',
      quarryColossus: 'quarry-colossus',
      emberjawGolem: 'emberjaw-golem',
      frostlingScout: 'frostling-scout',
      shardling: 'shardling',
      rimebackBrute: 'rimeback-brute',
      glacierSentinel: 'glacier-sentinel',
      snowglareWisp: 'snowglare-wisp',
      icebloomOracle: 'icebloom-oracle',
      galeHarrier: 'gale-harrier',
      stormboundArcher: 'stormbound-archer',
      thunderRam: 'thunder-ram',
      cloudcallAcolyte: 'cloudcall-acolyte',
      indexScribe: 'index-scribe',
      lumenSentinel: 'lumen-sentinel',
      voidMote: 'void-mote',
      eclipseDuelist: 'eclipse-duelist',
      riftAberration: 'rift-aberration',
      stormbreakRoc: 'stormbreak-roc',
      astralArchivist: 'astral-archivist',
      eclipseSovereign: 'eclipse-sovereign',
      rimewarden: 'rimewarden'
    });

    const COMPACT_ENEMY_ANIMATION_FILE_IDS = ENEMY_ANIMATION_FILE_IDS;

    const ENEMY_ANIMATION_ASSETS = Object.freeze(Object.keys(ENEMY_ANIMATION_FILE_IDS).reduce((assets, enemyId) => {
      assets[enemyId] = COMPACT_ENEMY_ANIMATION_FILE_IDS[enemyId]
        ? makeCompactEnemyAnimationAsset(COMPACT_ENEMY_ANIMATION_FILE_IDS[enemyId], enemyId)
        : makeEnemyAnimationAsset(ENEMY_ANIMATION_FILE_IDS[enemyId], enemyId);
      return assets;
    }, {}));

    const ENEMY_PROJECTILE_ANIMATION_ASSETS = Object.freeze({
      banditThrower: makeEnemyProjectileAnimationAsset('bandit-knife')
    });

    const ENEMY_ANIMATION_BEHAVIORS = Object.freeze({
      melee: Object.freeze({ id: 'melee', states: ENEMY_ANIMATION_ROWS }),
      ranged: Object.freeze({ id: 'ranged', states: ENEMY_ANIMATION_ROWS }),
      charger: Object.freeze({ id: 'charger', states: ENEMY_ANIMATION_ROWS }),
      flyer: Object.freeze({ id: 'flyer', states: ENEMY_ANIMATION_ROWS }),
      healer: Object.freeze({ id: 'healer', states: ENEMY_ANIMATION_ROWS }),
      elite: Object.freeze({ id: 'elite', states: ENEMY_ANIMATION_ROWS }),
      boss: Object.freeze({ id: 'boss', states: ENEMY_ANIMATION_ROWS })
    });

    const FX_ANIMATION_ASSETS = Object.freeze({
      slash: makeFxAnimationAsset('slash', 'slash'),
      cast: makeFxAnimationAsset('cast', 'cast'),
      arrowRelease: makeFxAnimationAsset('arrow-release', 'arrowRelease'),
      partyBuff: makeFxAnimationAsset('party-buff', 'partyBuff'),
      impact: makeFxAnimationAsset('impact', 'impact'),
      defeatBurst: makeFxAnimationAsset('defeat-burst', 'defeatBurst')
    });

    const PORTAL_ANIMATION_ASSETS = Object.freeze({
      standard: makePortalAnimationAsset('standard'),
      boss: makePortalAnimationAsset('boss'),
      locked: makePortalAnimationAsset('locked')
    });

    const PET_ANIMATION_ASSET = makePetAnimationAsset('starfall-fox');

    const BUFF_CAST_VISUALS = Object.freeze({
      fighter_guard: Object.freeze({ id: 'fighter_guard', style: 'barrier', color: '#68a9ff', accent: '#d5ecff' }),
      mage_mana_shield: Object.freeze({ id: 'mage_mana_shield', style: 'barrier', color: '#8bd7ff', accent: '#e7fbff' }),
      guardian_impact_guard: Object.freeze({ id: 'guardian_impact_guard', style: 'barrier', color: '#6fa8d9', accent: '#d5ecff' }),
      guardian_oath_barrier: Object.freeze({ id: 'guardian_oath_barrier', style: 'barrier', color: '#7fc4ff', accent: '#e3f6ff' }),
      guardian_hold_the_line: Object.freeze({ id: 'guardian_hold_the_line', style: 'barrier', color: '#466d91', accent: '#d5ecff' }),
      shieldWall: Object.freeze({ id: 'shieldWall', skillId: 'guardian_shield_wall', style: 'barrier', color: '#6fa8d9', accent: '#d5ecff' }),
      warCry: Object.freeze({ id: 'warCry', skillId: 'berserker_war_cry', style: 'cry', color: '#ef3d55', accent: '#ffbe55' }),
      rallyingFlourish: Object.freeze({ id: 'rallyingFlourish', skillId: 'duelist_rallying_flourish', style: 'flourish', color: '#5fd6c6', accent: '#ffe16a' }),
      ignitionAura: Object.freeze({ id: 'ignitionAura', skillId: 'fire_mage_ignition_aura', style: 'flame', color: '#ff7b3a', accent: '#ffd166' }),
      runeCircle: Object.freeze({ id: 'runeCircle', skillId: 'rune_mage_rune_circle', style: 'rune', color: '#28c7b7', accent: '#b8fff2' }),
      stormfront: Object.freeze({ id: 'stormfront', skillId: 'storm_mage_stormfront', style: 'storm', color: '#7aa7ff', accent: '#f0f7ff' }),
      eagleEye: Object.freeze({ id: 'eagleEye', skillId: 'sniper_eagle_eye', style: 'focus', color: '#d8c25f', accent: '#fff5b8' }),
      tacticalField: Object.freeze({ id: 'tacticalField', skillId: 'trapper_tactical_field', style: 'tactical', color: '#66d79a', accent: '#dbffe6' }),
      packCall: Object.freeze({ id: 'packCall', skillId: 'beast_archer_pack_call', style: 'pack', color: '#8ed174', accent: '#fff0a6' })
    });

    const CORE_ANIMATION_ASSETS = Object.freeze({
      players: PLAYER_ANIMATION_ASSETS,
      enemies: ENEMY_ANIMATION_ASSETS,
      enemyBehaviors: ENEMY_ANIMATION_BEHAVIORS,
      pet: PET_ANIMATION_ASSET,
      fx: FX_ANIMATION_ASSETS,
      portals: PORTAL_ANIMATION_ASSETS
    });

    function getEnemyAnimationBehavior(enemy) {
      if (!enemy) return 'melee';
      if (enemy.id === 'emberjawGolem' || enemy.behavior === 'boss') return 'boss';
      if (enemy.id === 'crackedMimic' || enemy.behavior === 'elite') return 'elite';
      if (enemy.behavior === 'flyer') return 'flyer';
      if (enemy.behavior === 'healer') return 'healer';
      if (enemy.behavior === 'charger') return 'charger';
      if (enemy.behavior === 'thrower' || enemy.behavior === 'turret') return 'ranged';
      return 'melee';
    }

    return Object.freeze({
      ANIMATION_ROOT,
      COMBAT_FX_ANIMATION_ROOT,
      EQUIPMENT_ATLAS_ROOT,
      PLAYER_ANIMATION_ROWS,
      ENEMY_ANIMATION_ROWS,
      PET_ANIMATION_ROWS,
      SKILL_FX_ANIMATION_ROWS,
      BASIC_ATTACK_FX_ANIMATION_ROWS,
      ENEMY_COMBAT_FX_ANIMATION_ROWS,
      PLAYER_ANIMATION_CONFIG,
      ENEMY_ANIMATION_CONFIG,
      PET_ANIMATION_CONFIG,
      FX_ANIMATION_CONFIG,
      COMBAT_FX_ANIMATION_CONFIG,
      PORTAL_ANIMATION_CONFIG,
      GENERIC_PLAYER_ANIMATION_ASSET,
      PLAYER_ANIMATION_ASSETS,
      EQUIPMENT_VISUALS,
      FIGHTER_RIG_ANIMATION_STATES,
      FIGHTER_RIG_EQUIPMENT_VISUALS,
      GENERIC_PLAYER_RIG,
      PLAYER_RIGS,
      ENEMY_ANIMATION_ROW_HOLDS,
      ENEMY_ANIMATION_TIMING_OVERRIDES,
      ENEMY_ANIMATION_FILE_IDS,
      ENEMY_ANIMATION_ASSETS,
      ENEMY_PROJECTILE_ANIMATION_ASSETS,
      ENEMY_ANIMATION_BEHAVIORS,
      FX_ANIMATION_ASSETS,
      PORTAL_ANIMATION_ASSETS,
      PET_ANIMATION_ASSET,
      BUFF_CAST_VISUALS,
      CORE_ANIMATION_ASSETS,
      makeEquipmentVisualAtlas,
      EQUIPMENT_ATLAS_ANGLE_SETS,
      getEquipmentAtlasAngles,
      makeCombatFxAnimationAsset,
      getEnemyAnimationBehavior
    });
  }

  const api = Object.freeze({
    createAnimationData
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.animations = api;
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
