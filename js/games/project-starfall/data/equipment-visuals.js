(function initProjectStarfallDataEquipmentVisuals(global) {
  'use strict';

  function equipmentVisualFileId(id) {
    return String(id || '').trim().replace(/_/g, '-');
  }

  function inferEquipmentVisualKind(id, slot) {
    const text = String(id || '').toLowerCase();
    if (slot === 'weapon') {
      if (text.includes('bow') || text.includes('string') || text.includes('repeater') || text.includes('recurve')) return 'bow';
      if (text.includes('axe') || text.includes('cleaver') || text.includes('maul')) return 'axe';
      if (text.includes('wand')) return 'wand';
      if (text.includes('staff') || text.includes('scepter') || text.includes('rod') || text.includes('codex') || text.includes('focus')) return 'staff';
      return 'sword';
    }
    if (slot === 'offhand') {
      if (text.includes('shield')) return 'shield';
      if (text.includes('grip')) return 'grip';
      if (text.includes('core')) return 'core';
      if (text.includes('focus')) return 'focus';
      if (text.includes('scope')) return 'scope';
      if (text.includes('kit')) return 'kit';
    }
    if (slot === 'amulet') return 'amulet';
    if (slot === 'ring') return 'ring';
    return slot || 'chest';
  }

  const EQUIPMENT_VISUAL_SLOT_META = Object.freeze({
    weapon: Object.freeze({ layer: 'weapon', order: 40 }),
    offhand: Object.freeze({ layer: 'offhand', order: 55 }),
    head: Object.freeze({ layer: 'head', order: 35 }),
    chest: Object.freeze({ layer: 'chest', order: 20 }),
    gloves: Object.freeze({ layer: 'gloves', order: 45 }),
    boots: Object.freeze({ layer: 'boots', order: 30 }),
    ring: Object.freeze({ layer: 'accessory', order: 65 }),
    amulet: Object.freeze({ layer: 'accessory', order: 47 })
  });

  function inferEquipmentVisualSlot(config) {
    const slot = String(config && config.slot || '').trim();
    if (slot) return slot;
    const layer = String(config && config.layer || '').trim();
    if (layer === 'weapon') return 'weapon';
    if (layer === 'offhand') return 'offhand';
    if (layer === 'head') return 'head';
    if (layer === 'chest') return 'chest';
    if (layer === 'gloves') return 'gloves';
    if (layer === 'boots') return 'boots';
    if (layer === 'aura') return 'ring';
    return 'chest';
  }

  function makeEquipmentVisualDefinition(config, makeEquipmentVisualAtlas) {
    const id = String(config && config.id || '').trim();
    const slot = inferEquipmentVisualSlot(config);
    const meta = EQUIPMENT_VISUAL_SLOT_META[slot] || EQUIPMENT_VISUAL_SLOT_META.chest;
    const fileId = String(config && config.fileId || equipmentVisualFileId(id));
    const kind = config && config.kind || inferEquipmentVisualKind(id, slot);
    const atlas = typeof makeEquipmentVisualAtlas === 'function'
      ? makeEquipmentVisualAtlas(fileId, kind, id)
      : null;
    const visual = {
      id,
      fileId,
      slot,
      kind,
      layer: config && config.layer || meta.layer,
      order: config && config.order || meta.order,
      assetId: config && config.assetId || ''
    };
    if (atlas) {
      visual.renderMode = 'atlas';
      visual.atlas = atlas;
    }
    return Object.freeze(visual);
  }

  const EXTRA_EQUIPMENT_VISUAL_CONFIGS = Object.freeze([
    Object.freeze({ id: 'adventurer_cutlass', slot: 'weapon', kind: 'sword' }),
    Object.freeze({ id: 'balanced_focus', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'wanderer_charm', slot: 'amulet', kind: 'amulet' }),
    Object.freeze({ id: 'vanguard_blade', slot: 'weapon', kind: 'sword' }),
    Object.freeze({ id: 'bulwark_plate', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'breaker_gauntlets', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'sentinel_greaves', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'starglass_staff', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'runewoven_robes', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'channeler_gloves', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'aetherstep_boots', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'ranger_recurve', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'pathfinder_leathers', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'deadeye_wraps', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'windrunner_boots', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'thorncrown_greatsword', slot: 'weapon', kind: 'sword' }),
    Object.freeze({ id: 'thornroot_staff', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'briarstring_longbow', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'briar_crown', slot: 'head', kind: 'head' }),
    Object.freeze({ id: 'barkplate_harness', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'grasping_thorn_gloves', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'rootstep_greaves', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'emberjaw_cleaver', slot: 'weapon', kind: 'axe' }),
    Object.freeze({ id: 'magma_scepter', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'cindercoil_bow', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'ashen_jaw_helm', slot: 'head', kind: 'head' }),
    Object.freeze({ id: 'furnaceplate', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'lavaforged_gauntlets', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'scorchtrail_boots', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'gearcleaver', slot: 'weapon', kind: 'axe' }),
    Object.freeze({ id: 'chrono_staff', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'ratchet_repeater', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'titan_visor', slot: 'head', kind: 'head' }),
    Object.freeze({ id: 'clockplate_harness', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'gyro_gauntlets', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'springstep_boots', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'colossus_maul', slot: 'weapon', kind: 'axe' }),
    Object.freeze({ id: 'geode_scepter', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'oreline_greatbow', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'deepcore_helm', slot: 'head', kind: 'head' }),
    Object.freeze({ id: 'bedrock_plate', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'quarry_fists', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'stonewake_boots', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'stormtalon_saber', slot: 'weapon', kind: 'sword' }),
    Object.freeze({ id: 'cloudspine_rod', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'skybreaker_bow', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'rocfeather_mask', slot: 'head', kind: 'head' }),
    Object.freeze({ id: 'tempest_mantle', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'lightning_grip_gloves', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'gale_boots', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'index_blade', slot: 'weapon', kind: 'sword' }),
    Object.freeze({ id: 'starbound_codex', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'cometstring_bow', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'archivist_crown', slot: 'head', kind: 'head' }),
    Object.freeze({ id: 'astral_robes', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'scribe_gloves', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'orbit_boots', slot: 'boots', kind: 'boots' }),
    Object.freeze({ id: 'eclipse_edge', slot: 'weapon', kind: 'sword' }),
    Object.freeze({ id: 'umbral_starstaff', slot: 'weapon', kind: 'staff' }),
    Object.freeze({ id: 'corona_longbow', slot: 'weapon', kind: 'bow' }),
    Object.freeze({ id: 'sovereign_crown', slot: 'head', kind: 'head' }),
    Object.freeze({ id: 'eclipse_plate', slot: 'chest', kind: 'chest' }),
    Object.freeze({ id: 'penumbra_gloves', slot: 'gloves', kind: 'gloves' }),
    Object.freeze({ id: 'sunfall_boots', slot: 'boots', kind: 'boots' })
  ]);

  const BASE_EQUIPMENT_VISUAL_CONFIGS = Object.freeze([
    Object.freeze({ id: 'training_sword', fileId: 'training-sword', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'training_wand', fileId: 'training-wand', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'training_bow', fileId: 'training-bow', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'copper_sword', fileId: 'copper-sword', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'birch_wand', fileId: 'birch-wand', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'simple_bow', fileId: 'simple-bow', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'stitched_vest', fileId: 'stitched-vest', layer: 'chest', order: 20 }),
    Object.freeze({ id: 'traveler_boots', fileId: 'traveler-boots', layer: 'boots', order: 30 }),
    Object.freeze({ id: 'fieldguard_helm', fileId: 'fieldguard-helm', layer: 'head', order: 35, assetId: 'fieldguard_helm' }),
    Object.freeze({ id: 'trailwoven_gloves', fileId: 'trailwoven-gloves', layer: 'gloves', order: 45, assetId: 'trailwoven_gloves' }),
    Object.freeze({ id: 'plain_ring', fileId: 'plain-ring', layer: 'aura', order: 70 }),
    Object.freeze({ id: 'iron_sword', fileId: 'iron-sword', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'iron_axe', fileId: 'iron-axe', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'apprentice_staff', fileId: 'apprentice-staff', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'oak_longbow', fileId: 'oak-longbow', layer: 'weapon', order: 40 }),
    Object.freeze({ id: 'guardian_tower_shield', fileId: 'guardian-tower-shield', layer: 'offhand', order: 50 }),
    Object.freeze({ id: 'berserker_war_grip', fileId: 'berserker-war-grip', layer: 'offhand', order: 55 }),
    Object.freeze({ id: 'ember_core', fileId: 'ember-core', layer: 'offhand', order: 55 }),
    Object.freeze({ id: 'rune_etched_focus', fileId: 'rune-etched-focus', layer: 'offhand', order: 55 }),
    Object.freeze({ id: 'deadeye_scope', fileId: 'deadeye-scope', layer: 'offhand', order: 55 }),
    Object.freeze({ id: 'trap_kit', fileId: 'trap-kit', layer: 'offhand', order: 55 })
  ]);

  function createBaseEquipmentVisuals(makeEquipmentVisualAtlas) {
    return Object.freeze(BASE_EQUIPMENT_VISUAL_CONFIGS.reduce((visuals, config) => {
      visuals[config.id] = makeEquipmentVisualDefinition(config, makeEquipmentVisualAtlas);
      return visuals;
    }, {}));
  }

  const FIGHTER_RIG_ANIMATION_STATES = Object.freeze({
    idle: Object.freeze({ frames: 4, fps: 5, loop: true, timeline: Object.freeze(['settle', 'breath', 'settle', 'breath']) }),
    run: Object.freeze({ frames: 4, fps: 9, loop: true, timeline: Object.freeze(['stepA', 'passA', 'stepB', 'passB']) }),
    jump: Object.freeze({ frames: 2, fps: 10, loop: false, timeline: Object.freeze(['launch', 'tuck']) }),
    fall: Object.freeze({ frames: 2, fps: 8, loop: false, timeline: Object.freeze(['hang', 'brace']) }),
    climb: Object.freeze({ frames: 4, fps: 8, loop: true, timeline: Object.freeze(['reachA', 'pullA', 'reachB', 'pullB']) }),
    basic: Object.freeze({ frames: 4, fps: 13, loop: false, timeline: Object.freeze(['windup', 'lunge', 'follow', 'recover']) }),
    skill: Object.freeze({ frames: 4, fps: 11, loop: false, timeline: Object.freeze(['charge', 'cleave', 'impact', 'recover']) }),
    party: Object.freeze({ frames: 6, fps: 12, loop: false, timeline: Object.freeze(['ready', 'raise', 'flare', 'flare', 'settle', 'settle']) }),
    hit: Object.freeze({ frames: 3, fps: 12, loop: false, timeline: Object.freeze(['recoil', 'recoil', 'settle']) }),
    defeat: Object.freeze({ frames: 4, fps: 8, loop: false, timeline: Object.freeze(['drop', 'down', 'down', 'down']) })
  });

  function getEquipmentVisualTheme(id) {
    const text = String(id || '').toLowerCase();
    if (text.includes('thorn') || text.includes('briar') || text.includes('bark') || text.includes('root')) {
      return Object.freeze({ dark: '#31452e', main: '#5f8b54', light: '#a9d46f', accent: '#f0d36a', leather: '#6b4a2f' });
    }
    if (text.includes('ember') || text.includes('magma') || text.includes('cinder') || text.includes('ashen') || text.includes('furnace') || text.includes('lava') || text.includes('scorch')) {
      return Object.freeze({ dark: '#4d1f24', main: '#b94735', light: '#ff8a3d', accent: '#ffe16a', leather: '#6b3f2c' });
    }
    if (text.includes('gear') || text.includes('chrono') || text.includes('ratchet') || text.includes('titan') || text.includes('clock') || text.includes('gyro') || text.includes('spring')) {
      return Object.freeze({ dark: '#374451', main: '#75828f', light: '#d6e1e8', accent: '#f3d86d', leather: '#5b4a35' });
    }
    if (text.includes('colossus') || text.includes('geode') || text.includes('ore') || text.includes('deepcore') || text.includes('bedrock') || text.includes('quarry') || text.includes('stone')) {
      return Object.freeze({ dark: '#30343d', main: '#6d7480', light: '#b7c3ca', accent: '#7bdff2', leather: '#4a4039' });
    }
    if (text.includes('storm') || text.includes('cloud') || text.includes('sky') || text.includes('roc') || text.includes('tempest') || text.includes('lightning') || text.includes('gale')) {
      return Object.freeze({ dark: '#243e62', main: '#4f8cff', light: '#9bdfff', accent: '#fff0a6', leather: '#33495b' });
    }
    if (text.includes('astral') || text.includes('star') || text.includes('comet') || text.includes('archive') || text.includes('scribe') || text.includes('orbit') || text.includes('index')) {
      return Object.freeze({ dark: '#31275e', main: '#7f68d9', light: '#c6b8ff', accent: '#ffe16a', leather: '#4b3d74' });
    }
    if (text.includes('eclipse') || text.includes('umbral') || text.includes('corona') || text.includes('sovereign') || text.includes('penumbra') || text.includes('sunfall')) {
      return Object.freeze({ dark: '#1f2333', main: '#4b4b78', light: '#d8c25f', accent: '#fff0a6', leather: '#2e2a38' });
    }
    if (text.includes('ranger') || text.includes('pathfinder') || text.includes('deadeye') || text.includes('windrunner')) {
      return Object.freeze({ dark: '#2d4939', main: '#4f7b58', light: '#8ed174', accent: '#ffe16a', leather: '#5a3f2a' });
    }
    if (text.includes('starglass') || text.includes('runewoven') || text.includes('channeler') || text.includes('aether')) {
      return Object.freeze({ dark: '#253963', main: '#526fbd', light: '#9bdfff', accent: '#b8fff2', leather: '#473b64' });
    }
    if (text.includes('vanguard') || text.includes('bulwark') || text.includes('breaker') || text.includes('sentinel')) {
      return Object.freeze({ dark: '#3a4756', main: '#6f7f90', light: '#d8e5ec', accent: '#f0c36a', leather: '#5a4434' });
    }
    return Object.freeze({ dark: '#3d3441', main: '#8a6d4e', light: '#d6c18f', accent: '#ffe16a', leather: '#6f412b' });
  }

  function makeFighterRigEquipmentVisual(config) {
    const kind = config && config.kind || inferEquipmentVisualKind(config && config.id, config && config.slot);
    const theme = getEquipmentVisualTheme(config && config.id);
    if (kind === 'sword') return Object.freeze({ kind, blade: theme.light, shine: '#ffffff', grip: theme.leather });
    if (kind === 'axe') return Object.freeze({ kind, blade: theme.light, shine: '#ffffff', grip: theme.leather });
    if (kind === 'staff' || kind === 'wand') return Object.freeze({ kind, rod: theme.leather, glow: theme.accent, gem: theme.light });
    if (kind === 'bow') return Object.freeze({ kind, wood: theme.leather, string: '#fff5d0', arrow: theme.accent, long: true });
    if (kind === 'chest') return Object.freeze({ kind, cloth: theme.main, trim: theme.light, stitch: theme.accent });
    if (kind === 'boots') return Object.freeze({ kind, leather: theme.leather, sole: theme.dark, buckle: theme.accent });
    if (kind === 'head') return Object.freeze({ kind, trim: theme.light, metal: theme.main, dark: theme.dark });
    if (kind === 'gloves') return Object.freeze({ kind, dark: theme.leather, metal: theme.light, edge: theme.accent });
    if (kind === 'amulet') return Object.freeze({ kind, metal: theme.light, glow: theme.accent, gem: theme.main });
    if (kind === 'ring') return Object.freeze({ kind, metal: theme.light, glow: theme.accent });
    return Object.freeze({ kind: 'chest', cloth: theme.main, trim: theme.light, stitch: theme.accent });
  }

  const BASE_FIGHTER_RIG_EQUIPMENT_VISUALS = Object.freeze({
    training_sword: Object.freeze({ kind: 'sword', blade: '#aebec9', shine: '#eaf5ff', grip: '#8f5f39' }),
    training_wand: Object.freeze({ kind: 'wand', rod: '#8b5f35', glow: '#8bd7ff', gem: '#c6f4ff' }),
    training_bow: Object.freeze({ kind: 'bow', wood: '#9b6a35', string: '#f1e6ca', arrow: '#efe2a4' }),
    copper_sword: Object.freeze({ kind: 'sword', blade: '#c8753d', shine: '#ffc179', grip: '#6b3f2c' }),
    birch_wand: Object.freeze({ kind: 'wand', rod: '#d7bd7a', glow: '#9fffd1', gem: '#ecfff4' }),
    simple_bow: Object.freeze({ kind: 'bow', wood: '#8a5c2f', string: '#f5efd6', arrow: '#ffe16a' }),
    iron_sword: Object.freeze({ kind: 'sword', blade: '#dce6ee', shine: '#ffffff', grip: '#565f6c' }),
    iron_axe: Object.freeze({ kind: 'axe', blade: '#d6e1e8', shine: '#ffffff', grip: '#815334' }),
    apprentice_staff: Object.freeze({ kind: 'staff', rod: '#724c2f', glow: '#5fa8ff', gem: '#cbe8ff' }),
    oak_longbow: Object.freeze({ kind: 'bow', wood: '#704826', string: '#fff5d0', arrow: '#ffe16a', long: true }),
    stitched_vest: Object.freeze({ kind: 'chest', cloth: '#8c5a3a', trim: '#d39a5c', stitch: '#f3d5a0' }),
    party_plate: Object.freeze({ kind: 'chest', cloth: '#526f86', trim: '#b7c3ca', stitch: '#68a9ff' }),
    party_robes: Object.freeze({ kind: 'chest', cloth: '#5b62a8', trim: '#b8e6ff', stitch: '#7bdff2' }),
    party_leathers: Object.freeze({ kind: 'chest', cloth: '#4f7b58', trim: '#d5c66a', stitch: '#ffe16a' }),
    traveler_boots: Object.freeze({ kind: 'boots', leather: '#6f412b', sole: '#2b1d1b', buckle: '#d6a14a' }),
    fieldguard_helm: Object.freeze({ kind: 'head', trim: '#8da2af', metal: '#b7c3ca', dark: '#445766' }),
    trailwoven_gloves: Object.freeze({ kind: 'gloves', dark: '#6f412b', metal: '#b7c3ca', edge: '#d6a14a' }),
    plain_ring: Object.freeze({ kind: 'ring', metal: '#f7d879', glow: '#ffe16a' }),
    guardian_tower_shield: Object.freeze({ kind: 'shield', face: '#466d91', trim: '#d5ecff', metal: '#6fa8d9' }),
    berserker_war_grip: Object.freeze({ kind: 'grip', metal: '#a22d36', edge: '#ff6b5e', dark: '#4d1f24' }),
    ember_core: Object.freeze({ kind: 'core', core: '#ff6b35', glow: '#ffc15e', dark: '#8b2635' }),
    rune_etched_focus: Object.freeze({ kind: 'focus', core: '#28c7b7', glow: '#b8fff2', dark: '#146b72' }),
    deadeye_scope: Object.freeze({ kind: 'scope', metal: '#4b5663', lens: '#ffe16a', trim: '#d8c25f' }),
    trap_kit: Object.freeze({ kind: 'kit', leather: '#8a5a36', metal: '#b7c3ca', cord: '#3f2c24' })
  });

  function createEquipmentVisualData(options) {
    const settings = options || {};
    const makeEquipmentVisualAtlas = settings.makeEquipmentVisualAtlas;
    if (typeof makeEquipmentVisualAtlas !== 'function') {
      throw new TypeError('createEquipmentVisualData requires makeEquipmentVisualAtlas');
    }

    const BASE_EQUIPMENT_VISUALS = createBaseEquipmentVisuals(makeEquipmentVisualAtlas);
    const EQUIPMENT_VISUALS = Object.freeze(Object.assign({},
      BASE_EQUIPMENT_VISUALS,
      EXTRA_EQUIPMENT_VISUAL_CONFIGS.reduce((visuals, config) => {
        visuals[config.id] = makeEquipmentVisualDefinition(config, makeEquipmentVisualAtlas);
        return visuals;
      }, {})
    ));
    const FIGHTER_RIG_EQUIPMENT_VISUALS = Object.freeze(Object.assign({},
      BASE_FIGHTER_RIG_EQUIPMENT_VISUALS,
      EXTRA_EQUIPMENT_VISUAL_CONFIGS.reduce((visuals, config) => {
        visuals[config.id] = makeFighterRigEquipmentVisual(config);
        return visuals;
      }, {})
    ));

    return Object.freeze({
      EQUIPMENT_VISUALS,
      FIGHTER_RIG_ANIMATION_STATES,
      FIGHTER_RIG_EQUIPMENT_VISUALS
    });
  }

  const api = {
    equipmentVisualFileId,
    inferEquipmentVisualKind,
    inferEquipmentVisualSlot,
    createEquipmentVisualData,
    EQUIPMENT_VISUAL_SLOT_META,
    EXTRA_EQUIPMENT_VISUAL_CONFIGS,
    FIGHTER_RIG_ANIMATION_STATES
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.equipmentVisuals = Object.assign({}, modules.equipmentVisuals || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
