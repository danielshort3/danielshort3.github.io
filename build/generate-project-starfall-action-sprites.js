#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const { getEquipmentAttachment } = require('./project-starfall-player-equipment-rig.js');

const ROOT = path.resolve(__dirname, '..');
const FRAME = 160;
const COLS = 6;
const TRANSPARENT_PIXEL = Object.freeze({ r: 0, g: 0, b: 0, alpha: 0 });
const ROWS = ['idle', 'run', 'jump', 'fall', 'climb', 'basic', 'skill', 'party', 'hit', 'defeat'];
const ENEMY_ROWS = ['idle', 'move', 'telegraph', 'attack', 'projectile', 'buff', 'hit', 'defeat'];
const PLAYER_DIR = path.join(ROOT, 'img/project-starfall/animations/players');
const CHARACTER_DIR = path.join(ROOT, 'img/project-starfall/characters');
const ENEMY_DIR = path.join(ROOT, 'img/project-starfall/enemies');
const ENEMY_SHEET_DIR = path.join(ROOT, 'img/project-starfall/animations/enemies');
const MAP_DIR = path.join(ROOT, 'img/project-starfall/maps');
const PORTAL_DIR = path.join(ROOT, 'img/project-starfall/animations/portals');
const PROCEDURAL_BACKUP_DIR = path.join(ROOT, 'img/project-starfall/backups/procedural');
const GENERIC_CHARACTER_BACKUP_PATH = path.join(PROCEDURAL_BACKUP_DIR, 'characters/generic-player-v4.png');
const GENERIC_SHEET_BACKUP_PATH = path.join(PROCEDURAL_BACKUP_DIR, 'animations/players/generic-player-sheet-v4.png');

const COLORS = Object.freeze({
  outline: '#13222f',
  shadow: '#193344',
  skin: '#d99a6c',
  skinLight: '#f0bd88',
  hair: '#3b2725',
  shirt: '#5b7180',
  shirtLight: '#8da3af',
  pants: '#344657',
  pantsDark: '#24313f',
  belt: '#7d5132',
  boot: '#2a1f21',
  hand: '#e4aa78'
});

const CLASS_VARIANTS = Object.freeze([
  {
    id: 'fighter',
    fileId: 'fighter',
    shirt: '#526f86',
    shirtLight: '#8caabd',
    pants: '#33495b',
    accent: '#f0c36a',
    motif: 'slash',
    equipment: { kind: 'sword', blade: '#dce6ee', bright: '#ffffff', grip: '#8f5f39' }
  },
  {
    id: 'mage',
    fileId: 'mage',
    shirt: '#486db5',
    shirtLight: '#86a8ff',
    pants: '#2d3b72',
    accent: '#8bd7ff',
    motif: 'spark',
    equipment: { kind: 'wand', rod: '#724c2f', glow: '#8bd7ff', gem: '#cbe8ff' }
  },
  {
    id: 'archer',
    fileId: 'archer',
    shirt: '#3f8a66',
    shirtLight: '#7bcf90',
    pants: '#314f44',
    accent: '#ffe16a',
    motif: 'leaf',
    equipment: { kind: 'bow', wood: '#8a5c2f', string: '#f5efd6', arrow: '#ffe16a' }
  },
  {
    id: 'guardian',
    fileId: 'guardian',
    shirt: '#416d91',
    shirtLight: '#84bde8',
    pants: '#293f73',
    accent: '#d5ecff',
    motif: 'shield',
    equipment: { kind: 'shield', metal: '#6fa8d9', face: '#466d91', trim: '#d5ecff' }
  },
  {
    id: 'berserker',
    fileId: 'berserker',
    shirt: '#8d2f3d',
    shirtLight: '#e05b75',
    pants: '#4d1f24',
    accent: '#ffb35c',
    motif: 'rage',
    equipment: { kind: 'axe', blade: '#d6e1e8', bright: '#ffffff', haft: '#815334' }
  },
  {
    id: 'duelist',
    fileId: 'duelist',
    shirt: '#4f6a9b',
    shirtLight: '#b7c8f0',
    pants: '#2d344e',
    accent: '#f0c36a',
    motif: 'tempo',
    equipment: { kind: 'sword', blade: '#d8e6f0', bright: '#ffffff', grip: '#6d5a9b' }
  },
  {
    id: 'fireMage',
    fileId: 'fire-mage',
    shirt: '#8b3335',
    shirtLight: '#ff8a3d',
    pants: '#4a2d35',
    accent: '#ffd36b',
    motif: 'flame',
    equipment: { kind: 'staff', rod: '#724c2f', glow: '#ff8a3d', gem: '#ffd36b' }
  },
  {
    id: 'runeMage',
    fileId: 'rune-mage',
    shirt: '#146b72',
    shirtLight: '#28c7b7',
    pants: '#243f4a',
    accent: '#b8fff2',
    motif: 'rune',
    equipment: { kind: 'focus', core: '#28c7b7', glow: '#b8fff2', dark: '#146b72' }
  },
  {
    id: 'stormMage',
    fileId: 'storm-mage',
    shirt: '#395d83',
    shirtLight: '#7bdff2',
    pants: '#27384f',
    accent: '#f4fbff',
    motif: 'storm',
    equipment: { kind: 'wand', rod: '#56657a', glow: '#7bdff2', gem: '#f4fbff' }
  },
  {
    id: 'sniper',
    fileId: 'sniper',
    shirt: '#5f6545',
    shirtLight: '#c9b35c',
    pants: '#333826',
    accent: '#ffe16a',
    motif: 'scope',
    equipment: { kind: 'bow', wood: '#704826', string: '#fff5d0', arrow: '#ffe16a', long: true }
  },
  {
    id: 'trapper',
    fileId: 'trapper',
    shirt: '#7a5134',
    shirtLight: '#c48b57',
    pants: '#3f2c24',
    accent: '#b7c3ca',
    motif: 'trap',
    equipment: { kind: 'kit', leather: '#8a5a36', metal: '#b7c3ca', cord: '#3f2c24' }
  },
  {
    id: 'beastArcher',
    fileId: 'beast-archer',
    shirt: '#55794d',
    shirtLight: '#9bc776',
    pants: '#354832',
    accent: '#f0c36a',
    motif: 'paw',
    equipment: { kind: 'bow', wood: '#6d4a2f', string: '#f3e9ce', arrow: '#f0c36a', long: true }
  }
]);

const BASE_EQUIPMENT = Object.freeze([
  { id: 'training_sword', fileId: 'training-sword', kind: 'sword', blade: '#b8c7d0', bright: '#eef6ff', grip: '#8f5f39' },
  { id: 'training_wand', fileId: 'training-wand', kind: 'wand', rod: '#8b5f35', glow: '#8bd7ff', gem: '#c6f4ff' },
  { id: 'training_bow', fileId: 'training-bow', kind: 'bow', wood: '#9b6a35', string: '#f1e6ca', arrow: '#efe2a4' },
  { id: 'copper_sword', fileId: 'copper-sword', kind: 'sword', blade: '#c8753d', bright: '#ffc179', grip: '#6b3f2c' },
  { id: 'birch_wand', fileId: 'birch-wand', kind: 'wand', rod: '#d7bd7a', glow: '#9fffd1', gem: '#ecfff4' },
  { id: 'simple_bow', fileId: 'simple-bow', kind: 'bow', wood: '#8a5c2f', string: '#f5efd6', arrow: '#ffe16a' },
  { id: 'stitched_vest', fileId: 'stitched-vest', kind: 'chest', cloth: '#8c5a3a', trim: '#d39a5c', stitch: '#f3d5a0' },
  { id: 'traveler_boots', fileId: 'traveler-boots', kind: 'boots', leather: '#6f412b', sole: '#2b1d1b', buckle: '#d6a14a' },
  { id: 'plain_ring', fileId: 'plain-ring', kind: 'ring', metal: '#f7d879', glow: '#ffe16a' },
  { id: 'iron_sword', fileId: 'iron-sword', kind: 'sword', blade: '#dce6ee', bright: '#ffffff', grip: '#565f6c' },
  { id: 'iron_axe', fileId: 'iron-axe', kind: 'axe', blade: '#d6e1e8', bright: '#ffffff', haft: '#815334' },
  { id: 'apprentice_staff', fileId: 'apprentice-staff', kind: 'staff', rod: '#724c2f', glow: '#5fa8ff', gem: '#cbe8ff' },
  { id: 'oak_longbow', fileId: 'oak-longbow', kind: 'bow', wood: '#704826', string: '#fff5d0', arrow: '#ffe16a', long: true },
  { id: 'guardian_tower_shield', fileId: 'guardian-tower-shield', kind: 'shield', metal: '#6fa8d9', face: '#466d91', trim: '#d5ecff' },
  { id: 'berserker_war_grip', fileId: 'berserker-war-grip', kind: 'grip', metal: '#a22d36', edge: '#ff6b5e', dark: '#4d1f24' },
  { id: 'ember_core', fileId: 'ember-core', kind: 'core', core: '#ff6b35', glow: '#ffc15e', dark: '#8b2635' },
  { id: 'rune_etched_focus', fileId: 'rune-etched-focus', kind: 'focus', core: '#28c7b7', glow: '#b8fff2', dark: '#146b72' },
  { id: 'deadeye_scope', fileId: 'deadeye-scope', kind: 'scope', metal: '#4b5663', lens: '#ffe16a', trim: '#d8c25f' },
  { id: 'trap_kit', fileId: 'trap-kit', kind: 'kit', leather: '#8a5a36', metal: '#b7c3ca', cord: '#3f2c24' }
]);
const EQUIPMENT_ICON_KINDS = Object.freeze(['sword', 'axe', 'wand', 'staff', 'bow', 'chest', 'boots', 'head', 'gloves', 'ring', 'amulet', 'shield', 'grip', 'core', 'focus', 'scope', 'kit']);

function normalizeEquipmentStyle(style) {
  const item = Object.assign({}, style || {});
  if (item.shine && !item.bright) item.bright = item.shine;
  if (item.grip && !item.haft && item.kind === 'axe') item.haft = item.grip;
  return item;
}

function buildEquipmentDefinitions() {
  const baseById = BASE_EQUIPMENT.reduce((items, item) => {
    items[item.id] = item;
    return items;
  }, {});
  const rigVisuals = Data.PLAYER_RIGS && Data.PLAYER_RIGS.fighter && Data.PLAYER_RIGS.fighter.equipmentVisuals || {};
  return Object.values(Data.EQUIPMENT_VISUALS || baseById).map((visual) => {
    const style = normalizeEquipmentStyle(rigVisuals[visual.id] || baseById[visual.id] || {});
    return Object.assign({}, style, {
      id: visual.id || style.id,
      fileId: visual.fileId || style.fileId,
      kind: visual.kind || style.kind || 'chest'
    });
  });
}

const EQUIPMENT = Object.freeze(buildEquipmentDefinitions());

const ENEMIES = Object.freeze([
  { id: 'slimelet', fileId: 'slimelet', kind: 'slime', main: '#78c86d', dark: '#2c7b58', light: '#b8f0a4', accent: '#efffba' },
  { id: 'dewSlime', fileId: 'dew-slime', kind: 'slime', main: '#7fd6ad', dark: '#2b806a', light: '#c7ffe8', accent: '#f5fff0' },
  { id: 'mossback', fileId: 'mossback', kind: 'beast', main: '#6e8d4c', dark: '#31452e', light: '#9fbd70', accent: '#4d6a3f' },
  { id: 'thornSprout', fileId: 'thorn-sprout', kind: 'plant', main: '#4fa86b', dark: '#245b36', light: '#8bd47a', accent: '#c94d63' },
  { id: 'vineSnapper', fileId: 'vine-snapper', kind: 'plant', main: '#397c45', dark: '#183c2a', light: '#82d66f', accent: '#d34d72' },
  { id: 'bristleBoar', fileId: 'bristle-boar', kind: 'boar', main: '#8d6348', dark: '#443027', light: '#c99566', accent: '#e8d2a2' },
  { id: 'briarStag', fileId: 'briar-stag', kind: 'boar', main: '#73563e', dark: '#2f2a20', light: '#b88e5b', accent: '#77c463' },
  { id: 'dustImp', fileId: 'dust-imp', kind: 'imp', main: '#9a6d4e', dark: '#3f2d32', light: '#d09a68', accent: '#e4c37b' },
  { id: 'clockbug', fileId: 'clockbug', kind: 'bug', main: '#8b8d85', dark: '#3a4145', light: '#d6c681', accent: '#30b5aa' },
  { id: 'rustRatchet', fileId: 'rust-ratchet', kind: 'bug', main: '#9b6b3d', dark: '#3a2d26', light: '#d49a5a', accent: '#34b7ad' },
  { id: 'coilSentry', fileId: 'coil-sentry', kind: 'bug', main: '#777f89', dark: '#27313a', light: '#d6c681', accent: '#29d6c4' },
  { id: 'scrapWarden', fileId: 'scrap-warden', kind: 'banditMelee', main: '#7b7164', dark: '#303842', light: '#c0aa78', accent: '#32b8aa' },
  { id: 'emberWisp', fileId: 'ember-wisp', kind: 'wisp', main: '#ff7b3a', dark: '#7d2934', light: '#ffd36b', accent: '#ffeec2' },
  { id: 'ashCrawler', fileId: 'ash-crawler', kind: 'beast', main: '#6e5f5a', dark: '#2b2529', light: '#b88a6a', accent: '#ff6b35' },
  { id: 'lavaTick', fileId: 'lava-tick', kind: 'bug', main: '#c55233', dark: '#4a2326', light: '#ffb35a', accent: '#fff08a' },
  { id: 'cinderSpitter', fileId: 'cinder-spitter', kind: 'imp', main: '#8b3f34', dark: '#2e2026', light: '#df7549', accent: '#ffd166' },
  { id: 'banditCutter', fileId: 'bandit-cutter', kind: 'banditMelee', main: '#8b5d3f', dark: '#273141', light: '#d59a64', accent: '#d6e1e8' },
  { id: 'banditThrower', fileId: 'bandit-thrower', kind: 'banditRanged', main: '#7b6048', dark: '#2c3742', light: '#d59a64', accent: '#ffcf63' },
  { id: 'orebackBeetle', fileId: 'oreback-beetle', kind: 'beetle', main: '#736d65', dark: '#32383a', light: '#b78d55', accent: '#6fd5a6' },
  { id: 'glowcapHealer', fileId: 'glowcap-healer', kind: 'healer', main: '#62b86c', dark: '#275640', light: '#b8f2b7', accent: '#8cf0d4' },
  { id: 'crackedMimic', fileId: 'cracked-mimic', kind: 'mimic', main: '#7b4b35', dark: '#2f2324', light: '#d5a65f', accent: '#ef5b4c' },
  { id: 'brambleking', fileId: 'brambleking', kind: 'brambleBoss', main: '#3f8f58', dark: '#1f3f2f', light: '#8bd47a', accent: '#e05b75' },
  { id: 'clockworkTitan', fileId: 'clockwork-titan', kind: 'clockTitan', main: '#7a8592', dark: '#303842', light: '#d8b74a', accent: '#29b3ad' },
  { id: 'quarryColossus', fileId: 'quarry-colossus', kind: 'quarryBoss', main: '#81796f', dark: '#34383b', light: '#c3b48f', accent: '#69d1a6' },
  { id: 'emberjawGolem', fileId: 'emberjaw-golem', kind: 'golem', main: '#4a3d42', dark: '#201d22', light: '#8b7370', accent: '#ff6b35' },
  { id: 'frostlingScout', fileId: 'frostling-scout', kind: 'imp', main: '#9fd7f2', dark: '#2f5878', light: '#f7fbff', accent: '#5ca8e8' },
  { id: 'shardling', fileId: 'shardling', kind: 'slime', main: '#bdeeff', dark: '#4e86a7', light: '#ffffff', accent: '#72d6ff' },
  { id: 'rimebackBrute', fileId: 'rimeback-brute', kind: 'beast', main: '#7fb7c8', dark: '#2f5878', light: '#d7f3ff', accent: '#f7fbff' },
  { id: 'glacierSentinel', fileId: 'glacier-sentinel', kind: 'golem', main: '#8bc7db', dark: '#2a4c6c', light: '#effdff', accent: '#79e7ff' },
  { id: 'snowglareWisp', fileId: 'snowglare-wisp', kind: 'wisp', main: '#c7efff', dark: '#477da3', light: '#ffffff', accent: '#8fd7ff' },
  { id: 'icebloomOracle', fileId: 'icebloom-oracle', kind: 'healer', main: '#a6e5ef', dark: '#3a7190', light: '#f7fbff', accent: '#a7fff2' },
  { id: 'galeHarrier', fileId: 'gale-harrier', kind: 'wisp', main: '#9fd8e7', dark: '#445f79', light: '#f4fdff', accent: '#ffe16a' },
  { id: 'stormboundArcher', fileId: 'stormbound-archer', kind: 'banditRanged', main: '#657b95', dark: '#26364f', light: '#c7eafd', accent: '#ffe16a' },
  { id: 'thunderRam', fileId: 'thunder-ram', kind: 'boar', main: '#6d7282', dark: '#263140', light: '#cdd7e8', accent: '#ffe16a' },
  { id: 'cloudcallAcolyte', fileId: 'cloudcall-acolyte', kind: 'healer', main: '#8fb4ca', dark: '#34495f', light: '#e7f8ff', accent: '#ffe16a' },
  { id: 'indexScribe', fileId: 'index-scribe', kind: 'banditRanged', main: '#6450a3', dark: '#27254a', light: '#d7fff7', accent: '#64d9c5' },
  { id: 'lumenSentinel', fileId: 'lumen-sentinel', kind: 'golem', main: '#725bd1', dark: '#2f255d', light: '#eadcff', accent: '#64d9c5' },
  { id: 'voidMote', fileId: 'void-mote', kind: 'wisp', main: '#503a8d', dark: '#16182e', light: '#c794ff', accent: '#7bdff2' },
  { id: 'eclipseDuelist', fileId: 'eclipse-duelist', kind: 'banditMelee', main: '#25222e', dark: '#0f121a', light: '#ffd36b', accent: '#64d9c5' },
  { id: 'riftAberration', fileId: 'rift-aberration', kind: 'golem', main: '#35295f', dark: '#101326', light: '#c794ff', accent: '#f06bff' },
  { id: 'stormbreakRoc', fileId: 'stormbreak-roc', kind: 'wisp', main: '#6d7282', dark: '#263140', light: '#f4fdff', accent: '#ffe16a' },
  { id: 'astralArchivist', fileId: 'astral-archivist', kind: 'banditRanged', main: '#6450a3', dark: '#27254a', light: '#d7fff7', accent: '#64d9c5' },
  { id: 'eclipseSovereign', fileId: 'eclipse-sovereign', kind: 'banditMelee', main: '#25222e', dark: '#0f121a', light: '#ffd36b', accent: '#64d9c5' },
  { id: 'rimewarden', fileId: 'rimewarden', kind: 'golem', main: '#7fb7c8', dark: '#223d5a', light: '#d7f3ff', accent: '#79e7ff' }
]);

const BOSS_ENEMY_IDS = new Set(['brambleking', 'clockworkTitan', 'quarryColossus', 'emberjawGolem', 'rimewarden', 'stormbreakRoc', 'astralArchivist', 'eclipseSovereign']);

const MAP_BACKGROUNDS = Object.freeze([
  { id: 'brambleDepths', fileId: 'bramble-depths', kind: 'bramble', sky: '#203a31', far: '#294d38', mid: '#513325', ground: '#252f29', light: '#6aa86b', accent: '#e05b75' },
  { id: 'gearworksVault', fileId: 'gearworks-vault', kind: 'gearworks', sky: '#30343a', far: '#665b48', mid: '#7a8592', ground: '#3a3430', light: '#d8b74a', accent: '#29b3ad' },
  { id: 'emberjawLair', fileId: 'emberjaw-lair', kind: 'emberjaw', sky: '#2c2632', far: '#4a2d35', mid: '#643336', ground: '#2d2c30', light: '#8b7370', accent: '#ff6b35' },
  { id: 'frostfenOutskirts', fileId: 'frostfen-outskirts', kind: 'frostfen', sky: '#d9f4ff', far: '#8fb9d8', mid: '#6d9cc1', ground: '#d8eef7', light: '#f7fdff', accent: '#72d6ff' },
  { id: 'glacierSpine', fileId: 'glacier-spine', kind: 'glacier', sky: '#ccecff', far: '#7397c2', mid: '#527caf', ground: '#c7e4f2', light: '#f5fbff', accent: '#5ed1ff' },
  { id: 'rimewardenSanctum', fileId: 'rimewarden-sanctum', kind: 'rimewarden', sky: '#24314a', far: '#365373', mid: '#5f88a9', ground: '#b9d7e8', light: '#ecfbff', accent: '#79e7ff' },
  { id: 'bramblekingCourt', fileId: 'brambleking-court', kind: 'bossRoom', motif: 'bramble', sky: '#1f2f28', far: '#294d38', mid: '#513325', ground: '#252f29', light: '#6aa86b', accent: '#e05b75' },
  { id: 'titanFoundry', fileId: 'titan-foundry', kind: 'bossRoom', motif: 'gear', sky: '#2d3138', far: '#665b48', mid: '#7a8592', ground: '#343a42', light: '#d8b74a', accent: '#29b3ad' },
  { id: 'deepcoreCore', fileId: 'deepcore-core', kind: 'bossRoom', motif: 'core', sky: '#1e2530', far: '#3a4145', mid: '#6b6960', ground: '#2a2f34', light: '#c3b48f', accent: '#69d1a6' },
  { id: 'emberjawFurnace', fileId: 'emberjaw-furnace', kind: 'bossRoom', motif: 'furnace', sky: '#211c22', far: '#4a2d35', mid: '#643336', ground: '#2d2428', light: '#ffd166', accent: '#ff6b35' },
  { id: 'rimewardenVault', fileId: 'rimewarden-vault', kind: 'bossRoom', motif: 'rime', sky: '#223d5a', far: '#365373', mid: '#5f88a9', ground: '#b9d7e8', light: '#ecfbff', accent: '#79e7ff' },
  { id: 'stormbreakAerie', fileId: 'stormbreak-aerie', kind: 'bossRoom', motif: 'storm', sky: '#26364f', far: '#4f6073', mid: '#6d7282', ground: '#354861', light: '#91dbe8', accent: '#ffe16a' },
  { id: 'astralStacks', fileId: 'astral-stacks', kind: 'bossRoom', motif: 'astral', sky: '#1c243f', far: '#29365f', mid: '#59438b', ground: '#202943', light: '#7bdff2', accent: '#c794ff' },
  { id: 'eclipseThrone', fileId: 'eclipse-throne', kind: 'bossRoom', motif: 'eclipse', sky: '#151926', far: '#1f2330', mid: '#4a5c70', ground: '#20212b', light: '#ffe16a', accent: '#7bdff2' }
]);

const BOSS_ROOM_BACKGROUNDS = MAP_BACKGROUNDS.filter((map) => map.kind === 'bossRoom');

const TRIAL_MAP_BACKGROUNDS = Object.freeze([
  { id: 'guardian_trial', fileId: 'guardian-trial', kind: 'guardianTrial', sky: '#263a50', far: '#4d5968', mid: '#6fa8d9', ground: '#344252', light: '#d5ecff', accent: '#ffd166', motif: 'shield' },
  { id: 'berserker_trial', fileId: 'berserker-trial', kind: 'berserkerTrial', sky: '#2b2028', far: '#55323a', mid: '#a94a3c', ground: '#2a2025', light: '#ff8a5f', accent: '#ffd166', motif: 'chains' },
  { id: 'duelist_trial', fileId: 'duelist-trial', kind: 'duelistTrial', sky: '#d8cfa1', far: '#a98755', mid: '#7b6048', ground: '#5b4430', light: '#ffe0a6', accent: '#68a9ff', motif: 'banners' },
  { id: 'fire_mage_trial', fileId: 'fire-mage-trial', kind: 'fireMageTrial', sky: '#2c2632', far: '#4a2d35', mid: '#7b2930', ground: '#2d2428', light: '#ffd166', accent: '#ff3d2e', motif: 'flame' },
  { id: 'rune_mage_trial', fileId: 'rune-mage-trial', kind: 'runeMageTrial', sky: '#1c243f', far: '#29365f', mid: '#226b78', ground: '#202943', light: '#b8fff2', accent: '#28c7b7', motif: 'runes' },
  { id: 'storm_mage_trial', fileId: 'storm-mage-trial', kind: 'stormMageTrial', sky: '#26364f', far: '#354861', mid: '#5f7895', ground: '#263140', light: '#d8f6ff', accent: '#ffe16a', motif: 'lightning' },
  { id: 'sniper_trial', fileId: 'sniper-trial', kind: 'sniperTrial', sky: '#bed9e4', far: '#8ca981', mid: '#7b6048', ground: '#4e3b2d', light: '#ffe0a6', accent: '#ffd166', motif: 'targets' },
  { id: 'trapper_trial', fileId: 'trapper-trial', kind: 'trapperTrial', sky: '#203a31', far: '#294d38', mid: '#513325', ground: '#252f29', light: '#81c95f', accent: '#c4475d', motif: 'traps' },
  { id: 'beast_archer_trial', fileId: 'beast-archer-trial', kind: 'beastArcherTrial', sky: '#d7efd0', far: '#6b8a5d', mid: '#6b6960', ground: '#3d4d3c', light: '#b6e37c', accent: '#69d1a6', motif: 'tracks' }
]);

const PORTAL_VARIANTS = Object.freeze([
  { id: 'standard', fileId: 'standard', core: '#36c5ff', light: '#c7f8ff', dark: '#0d5c92', accent: '#62f0ff' },
  { id: 'boss', fileId: 'boss', core: '#b073ff', light: '#f0d7ff', dark: '#56308f', accent: '#ff8fd8' },
  { id: 'locked', fileId: 'locked', core: '#8a97a5', light: '#d5dde5', dark: '#45515f', accent: '#aeb8c3', locked: true }
]);

function svg(width, height, body) {
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="crispEdges">${body}</svg>`);
}

function px(x, y, width, height, color, extra) {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${color}"${extra || ''}/>`;
}

function ellipse(cx, cy, rx, ry, color, extra) {
  return `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="${color}"${extra || ''}/>`;
}

function polygon(points, color, extra) {
  return `<polygon points="${points}" fill="${color}"${extra || ''}/>`;
}

function svgPath(d, fill, stroke, extra) {
  return `<path d="${d}" fill="${fill || 'none'}" stroke="${stroke || 'none'}"${extra || ''}/>`;
}

function group(dx, dy, body) {
  return `<g transform="translate(${dx} ${dy})">${body}</g>`;
}

function box(parts) {
  return parts.filter(Boolean).join('');
}

function shiftedRect(rect, dx, dy, colorOverride) {
  return px(rect.x + dx, rect.y + dy, rect.w, rect.h, colorOverride || rect.c, rect.extra);
}

function rects(items, dx, dy) {
  return items.map((rect) => shiftedRect(rect, dx, dy)).join('');
}

function pose(row, frame) {
  const two = frame % 2;
  const six = frame % 6;
  const p = {
    bodyX: 0,
    bodyY: 0,
    headX: 0,
    headY: 0,
    frontArm: 'rest',
    backArm: 'rest',
    frontLeg: 'stand',
    backLeg: 'stand',
    gear: 'rest'
  };

  if (row === 'idle') {
    p.bodyY = six === 1 || six === 2 ? 1 : 0;
    p.headY = p.bodyY;
  } else if (row === 'run') {
    p.bodyX = two ? -4 : 4;
    p.bodyY = two ? 1 : -1;
    p.headY = p.bodyY;
    p.frontArm = two ? 'back' : 'forward';
    p.backArm = two ? 'forward' : 'back';
    p.frontLeg = two ? 'backStep' : 'frontStep';
    p.backLeg = two ? 'frontStep' : 'backStep';
    p.gear = 'run';
  } else if (row === 'jump') {
    p.bodyY = two ? -12 : 5;
    p.headY = p.bodyY;
    p.frontArm = two ? 'airForward' : 'brace';
    p.backArm = two ? 'airBack' : 'braceBack';
    p.frontLeg = two ? 'tuckedFront' : 'crouchFront';
    p.backLeg = two ? 'tuckedBack' : 'crouchBack';
    p.gear = two ? 'air' : 'ready';
  } else if (row === 'fall') {
    p.bodyY = two ? -3 : -8;
    p.headY = p.bodyY;
    p.frontArm = two ? 'fallForward' : 'airForward';
    p.backArm = two ? 'fallBack' : 'airBack';
    p.frontLeg = two ? 'dropFront' : 'tuckedFront';
    p.backLeg = two ? 'dropBack' : 'tuckedBack';
    p.gear = 'air';
  } else if (row === 'climb') {
    p.bodyY = two ? -3 : 3;
    p.headY = p.bodyY;
    p.frontArm = two ? 'climbLow' : 'climbHigh';
    p.backArm = two ? 'climbHigh' : 'climbLow';
    p.frontLeg = two ? 'climbHigh' : 'climbLow';
    p.backLeg = two ? 'climbLow' : 'climbHigh';
    p.gear = 'stowed';
  } else if (row === 'basic') {
    p.bodyX = two ? 12 : -8;
    p.bodyY = two ? 0 : 2;
    p.headX = two ? 1 : -1;
    p.headY = p.bodyY;
    p.frontArm = two ? 'strike' : 'windup';
    p.backArm = two ? 'counter' : 'braceBack';
    p.frontLeg = two ? 'lungeFront' : 'braceFront';
    p.backLeg = two ? 'lungeBack' : 'braceBackLeg';
    p.gear = two ? 'strike' : 'windup';
  } else if (row === 'skill') {
    p.bodyX = two ? 8 : -6;
    p.bodyY = two ? -2 : 3;
    p.headY = p.bodyY;
    p.frontArm = two ? 'castForward' : 'castUp';
    p.backArm = two ? 'castBack' : 'castSupport';
    p.frontLeg = two ? 'lungeFront' : 'braceFront';
    p.backLeg = two ? 'lungeBack' : 'braceBackLeg';
    p.gear = two ? 'cast' : 'charge';
  } else if (row === 'party') {
    p.bodyY = [1, 0, -2, -3, -1, 0][six];
    p.headY = p.bodyY;
    p.frontArm = six < 3 ? 'buffUp' : 'buffWide';
    p.backArm = six < 3 ? 'buffWide' : 'buffUp';
    p.gear = 'buff';
  } else if (row === 'hit') {
    p.bodyX = [-8, -10, -7, -3, 1, 0][six];
    p.bodyY = [1, 0, 1, 0, 0, 0][six];
    p.headX = -2;
    p.headY = p.bodyY;
    p.frontArm = 'hitFront';
    p.backArm = 'hitBack';
    p.frontLeg = six < 3 ? 'braceFront' : 'stand';
    p.backLeg = six < 3 ? 'braceBackLeg' : 'stand';
    p.gear = 'hit';
  } else if (row === 'defeat') {
    p.bodyX = two ? -2 : 1;
    p.bodyY = 24 + (two ? 3 : 0);
    p.headY = p.bodyY + 3;
    p.frontArm = 'downFront';
    p.backArm = 'downBack';
    p.frontLeg = 'downFront';
    p.backLeg = 'downBack';
    p.gear = 'down';
  }
  return p;
}

function legRects(kind, front, p, palette) {
  const colors = palette || COLORS;
  const x = p.bodyX;
  const y = p.bodyY;
  const color = front ? colors.pants : colors.pantsDark;
  const boot = colors.boot;
  const table = {
    stand: [
      { x: front ? 82 : 68, y: 104, w: 9, h: 22, c: color },
      { x: front ? 80 : 66, y: 126, w: 14, h: 7, c: boot }
    ],
    frontStep: [
      { x: front ? 88 : 62, y: 104, w: 9, h: 18, c: color },
      { x: front ? 91 : 56, y: 121, w: 17, h: 7, c: boot }
    ],
    backStep: [
      { x: front ? 74 : 76, y: 104, w: 9, h: 19, c: color },
      { x: front ? 64 : 76, y: 122, w: 17, h: 7, c: boot }
    ],
    crouchFront: [
      { x: front ? 86 : 65, y: 108, w: 10, h: 15, c: color },
      { x: front ? 92 : 61, y: 121, w: 14, h: 7, c: boot }
    ],
    crouchBack: [
      { x: front ? 78 : 69, y: 107, w: 10, h: 16, c: color },
      { x: front ? 74 : 66, y: 121, w: 14, h: 7, c: boot }
    ],
    tuckedFront: [
      { x: front ? 83 : 68, y: 103, w: 12, h: 12, c: color },
      { x: front ? 89 : 65, y: 113, w: 13, h: 7, c: boot }
    ],
    tuckedBack: [
      { x: front ? 76 : 68, y: 104, w: 10, h: 14, c: color },
      { x: front ? 70 : 65, y: 116, w: 13, h: 7, c: boot }
    ],
    dropFront: [
      { x: front ? 84 : 67, y: 105, w: 9, h: 23, c: color },
      { x: front ? 84 : 66, y: 128, w: 15, h: 7, c: boot }
    ],
    dropBack: [
      { x: front ? 72 : 69, y: 105, w: 9, h: 22, c: color },
      { x: front ? 69 : 68, y: 127, w: 15, h: 7, c: boot }
    ],
    climbHigh: [
      { x: front ? 85 : 68, y: 100, w: 9, h: 15, c: color },
      { x: front ? 88 : 65, y: 113, w: 12, h: 7, c: boot }
    ],
    climbLow: [
      { x: front ? 80 : 71, y: 110, w: 9, h: 18, c: color },
      { x: front ? 79 : 72, y: 128, w: 13, h: 7, c: boot }
    ],
    braceFront: [
      { x: front ? 81 : 66, y: 104, w: 10, h: 18, c: color },
      { x: front ? 86 : 61, y: 121, w: 15, h: 7, c: boot }
    ],
    braceBackLeg: [
      { x: front ? 73 : 70, y: 104, w: 10, h: 19, c: color },
      { x: front ? 65 : 69, y: 122, w: 15, h: 7, c: boot }
    ],
    lungeFront: [
      { x: front ? 89 : 66, y: 103, w: 9, h: 18, c: color },
      { x: front ? 97 : 59, y: 120, w: 19, h: 8, c: boot }
    ],
    lungeBack: [
      { x: front ? 74 : 64, y: 104, w: 10, h: 18, c: color },
      { x: front ? 57 : 62, y: 122, w: 18, h: 7, c: boot }
    ],
    downFront: [
      { x: front ? 88 : 67, y: 122, w: 26, h: 8, c: color },
      { x: front ? 111 : 91, y: 122, w: 14, h: 8, c: boot }
    ],
    downBack: [
      { x: front ? 71 : 60, y: 127, w: 26, h: 8, c: color },
      { x: front ? 94 : 82, y: 127, w: 14, h: 8, c: boot }
    ]
  };
  return (table[kind] || table.stand).map((rect) => Object.assign({}, rect, { x: rect.x + x, y: rect.y + y }));
}

function armRects(kind, front, p, palette) {
  const colors = palette || COLORS;
  const x = p.bodyX;
  const y = p.bodyY;
  const sleeve = front ? colors.shirtLight : colors.shirt;
  const hand = colors.hand;
  const table = {
    rest: [
      { x: front ? 91 : 58, y: 78, w: 8, h: 20, c: sleeve },
      { x: front ? 92 : 58, y: 96, w: 8, h: 8, c: hand }
    ],
    forward: [
      { x: front ? 92 : 57, y: 78, w: 16, h: 8, c: sleeve },
      { x: front ? 106 : 53, y: 79, w: 8, h: 8, c: hand }
    ],
    back: [
      { x: front ? 54 : 92, y: 81, w: 18, h: 8, c: sleeve },
      { x: front ? 48 : 108, y: 82, w: 8, h: 8, c: hand }
    ],
    brace: [
      { x: front ? 91 : 58, y: 80, w: 10, h: 14, c: sleeve },
      { x: front ? 97 : 57, y: 93, w: 8, h: 8, c: hand }
    ],
    braceBack: [
      { x: front ? 54 : 91, y: 79, w: 17, h: 8, c: sleeve },
      { x: front ? 49 : 107, y: 80, w: 8, h: 8, c: hand }
    ],
    airForward: [
      { x: front ? 92 : 56, y: 67, w: 10, h: 20, c: sleeve },
      { x: front ? 98 : 54, y: 64, w: 8, h: 8, c: hand }
    ],
    airBack: [
      { x: front ? 55 : 91, y: 70, w: 10, h: 18, c: sleeve },
      { x: front ? 51 : 96, y: 65, w: 8, h: 8, c: hand }
    ],
    fallForward: [
      { x: front ? 93 : 56, y: 72, w: 18, h: 8, c: sleeve },
      { x: front ? 108 : 51, y: 74, w: 8, h: 8, c: hand }
    ],
    fallBack: [
      { x: front ? 52 : 90, y: 73, w: 18, h: 8, c: sleeve },
      { x: front ? 47 : 107, y: 75, w: 8, h: 8, c: hand }
    ],
    climbHigh: [
      { x: front ? 91 : 58, y: 58, w: 8, h: 24, c: sleeve },
      { x: front ? 91 : 58, y: 53, w: 8, h: 8, c: hand }
    ],
    climbLow: [
      { x: front ? 94 : 55, y: 83, w: 8, h: 22, c: sleeve },
      { x: front ? 94 : 55, y: 103, w: 8, h: 8, c: hand }
    ],
    windup: [
      { x: front ? 56 : 92, y: 72, w: 24, h: 8, c: sleeve },
      { x: front ? 51 : 112, y: 72, w: 8, h: 8, c: hand }
    ],
    strike: [
      { x: front ? 93 : 56, y: 82, w: 28, h: 8, c: sleeve },
      { x: front ? 119 : 51, y: 82, w: 8, h: 8, c: hand }
    ],
    counter: [
      { x: front ? 58 : 91, y: 91, w: 20, h: 8, c: sleeve },
      { x: front ? 53 : 108, y: 91, w: 8, h: 8, c: hand }
    ],
    castUp: [
      { x: front ? 90 : 57, y: 58, w: 8, h: 25, c: sleeve },
      { x: front ? 90 : 57, y: 52, w: 8, h: 8, c: hand }
    ],
    castSupport: [
      { x: front ? 57 : 91, y: 83, w: 20, h: 8, c: sleeve },
      { x: front ? 52 : 108, y: 83, w: 8, h: 8, c: hand }
    ],
    castForward: [
      { x: front ? 92 : 57, y: 73, w: 25, h: 8, c: sleeve },
      { x: front ? 115 : 52, y: 73, w: 8, h: 8, c: hand }
    ],
    castBack: [
      { x: front ? 56 : 91, y: 88, w: 20, h: 8, c: sleeve },
      { x: front ? 51 : 108, y: 88, w: 8, h: 8, c: hand }
    ],
    buffUp: [
      { x: front ? 90 : 58, y: 58, w: 8, h: 26, c: sleeve },
      { x: front ? 90 : 58, y: 52, w: 8, h: 8, c: hand }
    ],
    buffWide: [
      { x: front ? 94 : 50, y: 70, w: 22, h: 8, c: sleeve },
      { x: front ? 114 : 45, y: 69, w: 8, h: 8, c: hand }
    ],
    hitFront: [
      { x: front ? 86 : 56, y: 82, w: 20, h: 8, c: sleeve },
      { x: front ? 103 : 51, y: 84, w: 8, h: 8, c: hand }
    ],
    hitBack: [
      { x: front ? 53 : 90, y: 78, w: 18, h: 8, c: sleeve },
      { x: front ? 48 : 106, y: 79, w: 8, h: 8, c: hand }
    ],
    downFront: [
      { x: front ? 86 : 55, y: 117, w: 23, h: 8, c: sleeve },
      { x: front ? 106 : 51, y: 117, w: 8, h: 8, c: hand }
    ],
    downBack: [
      { x: front ? 62 : 76, y: 130, w: 23, h: 8, c: sleeve },
      { x: front ? 57 : 96, y: 130, w: 8, h: 8, c: hand }
    ]
  };
  return (table[kind] || table.rest).map((rect) => Object.assign({}, rect, { x: rect.x + x, y: rect.y + y }));
}

function drawGenericPlayer(row, frame, palette) {
  const colors = palette || COLORS;
  const p = pose(row, frame);
  if (row === 'defeat') {
    return box([
      px(45 + p.bodyX, 144, 70, 7, '#0e2636', ' opacity="0.24"'),
      rects(legRects(p.backLeg, false, p, colors), 0, 0),
      rects(armRects(p.backArm, false, p, colors), 0, 0),
      px(54 + p.bodyX, 117 + p.bodyY, 42, 18, colors.outline),
      px(58 + p.bodyX, 119 + p.bodyY, 36, 14, colors.shirt),
      px(95 + p.bodyX, 112 + p.headY, 22, 18, colors.outline),
      px(97 + p.bodyX, 114 + p.headY, 18, 14, colors.skin),
      px(98 + p.bodyX, 112 + p.headY, 17, 6, colors.hair),
      rects(legRects(p.frontLeg, true, p, colors), 0, 0),
      rects(armRects(p.frontArm, true, p, colors), 0, 0)
    ]);
  }
  return box([
    px(48, 141, 64, 8, '#0e2636', ' opacity="0.22"'),
    rects(legRects(p.backLeg, false, p, colors), 0, 0),
    rects(armRects(p.backArm, false, p, colors), 0, 0),
    px(64 + p.bodyX, 70 + p.bodyY, 32, 39, colors.outline),
    px(67 + p.bodyX, 73 + p.bodyY, 26, 32, colors.shirt),
    px(67 + p.bodyX, 101 + p.bodyY, 26, 7, colors.belt),
    px(74 + p.bodyX + p.headX, 48 + p.headY, 22, 24, colors.outline),
    px(76 + p.bodyX + p.headX, 53 + p.headY, 18, 16, colors.skin),
    px(76 + p.bodyX + p.headX, 48 + p.headY, 19, 8, colors.hair),
    px(92 + p.bodyX + p.headX, 58 + p.headY, 4, 7, colors.hair),
    px(90 + p.bodyX + p.headX, 59 + p.headY, 3, 3, colors.outline),
    px(82 + p.bodyX + p.headX, 68 + p.headY, 8, 4, colors.skinLight),
    rects(legRects(p.frontLeg, true, p, colors), 0, 0),
    rects(armRects(p.frontArm, true, p, colors), 0, 0)
  ]);
}

function handPoint(p) {
  if (p.gear === 'strike') return { x: 124 + p.bodyX, y: 86 + p.bodyY };
  if (p.gear === 'windup') return { x: 54 + p.bodyX, y: 76 + p.bodyY };
  if (p.gear === 'cast') return { x: 119 + p.bodyX, y: 76 + p.bodyY };
  if (p.gear === 'charge') return { x: 94 + p.bodyX, y: 56 + p.bodyY };
  if (p.gear === 'air') return { x: 101 + p.bodyX, y: 69 + p.bodyY };
  if (p.gear === 'ready') return { x: 100 + p.bodyX, y: 97 + p.bodyY };
  if (p.gear === 'run') return { x: 106 + p.bodyX, y: 82 + p.bodyY };
  if (p.gear === 'hit') return { x: 104 + p.bodyX, y: 88 + p.bodyY };
  if (p.gear === 'buff') return { x: 95 + p.bodyX, y: 56 + p.bodyY };
  return { x: 95 + p.bodyX, y: 100 + p.bodyY };
}

function swordLayer(item, row, frame) {
  const p = pose(row, frame);
  const h = handPoint(p);
  const blade = item.blade;
  const bright = item.bright;
  const grip = item.grip;
  if (p.gear === 'down') {
    return box([px(58, 136, 40, 5, blade), px(94, 135, 8, 5, bright), px(50, 137, 10, 4, grip)]);
  }
  if (p.gear === 'stowed' || p.gear === 'buff') {
    return box([px(55 + p.bodyX, 57 + p.bodyY, 5, 45, blade), px(56 + p.bodyX, 54 + p.bodyY, 3, 6, bright), px(53 + p.bodyX, 99 + p.bodyY, 9, 5, grip)]);
  }
  if (p.gear === 'windup') {
    return box([
      px(h.x - 31, h.y - 16, 8, 5, bright),
      px(h.x - 25, h.y - 12, 9, 5, blade),
      px(h.x - 17, h.y - 8, 10, 5, blade),
      px(h.x - 8, h.y - 4, 11, 5, blade),
      px(h.x - 2, h.y + 1, 10, 5, grip)
    ]);
  }
  if (p.gear === 'strike' || p.gear === 'cast') {
    return box([
      px(h.x - 3, h.y - 4, 13, 5, grip),
      px(h.x + 9, h.y - 6, 28, 6, blade),
      px(h.x + 34, h.y - 7, 10, 5, bright),
      px(h.x + 14, h.y - 13, 34, 3, '#ffcf63', ' opacity="0.62"')
    ]);
  }
  return box([
    px(h.x - 2, h.y - 1, 12, 4, grip),
    px(h.x + 8, h.y - 3, 31, 5, blade),
    px(h.x + 35, h.y - 4, 7, 5, bright)
  ]);
}

function axeLayer(item, row, frame) {
  const p = pose(row, frame);
  const h = handPoint(p);
  if (p.gear === 'down') return box([px(60, 136, 43, 5, item.haft), px(100, 130, 14, 13, item.blade), px(104, 133, 10, 4, item.bright)]);
  if (p.gear === 'stowed' || p.gear === 'buff') return box([px(55 + p.bodyX, 58 + p.bodyY, 6, 45, item.haft), px(50 + p.bodyX, 55 + p.bodyY, 16, 14, item.blade), px(55 + p.bodyX, 58 + p.bodyY, 12, 4, item.bright)]);
  if (p.gear === 'windup') return box([px(h.x - 24, h.y - 14, 31, 5, item.haft), px(h.x - 31, h.y - 23, 18, 18, item.blade), px(h.x - 26, h.y - 20, 13, 4, item.bright)]);
  return box([px(h.x - 2, h.y - 1, 34, 5, item.haft), px(h.x + 29, h.y - 12, 18, 19, item.blade), px(h.x + 31, h.y - 8, 15, 4, item.bright)]);
}

function wandLayer(item, row, frame, staff) {
  const p = pose(row, frame);
  const h = handPoint(p);
  const length = staff ? 54 : 36;
  if (p.gear === 'down') {
    return box([px(59, 135, length, 5, item.rod), px(59 + length, 132, 9, 9, item.gem)]);
  }
  if (p.gear === 'strike' || p.gear === 'cast') {
    return box([
      px(h.x, h.y - 2, length, 4, item.rod),
      px(h.x + length - 1, h.y - 5, 8, 8, item.gem),
      px(h.x + length + 8, h.y - 8, 9, 9, item.glow, ' opacity="0.62"')
    ]);
  }
  return box([
    px(h.x - 2, h.y - length + 4, 5, length, item.rod),
    px(h.x - 5, h.y - length, 11, 10, item.gem),
    px(h.x - 10, h.y - length - 5, 20, 20, item.glow, ' opacity="0.2"')
  ]);
}

function bowLayer(item, row, frame) {
  const p = pose(row, frame);
  const h = handPoint(p);
  const height = item.long ? 61 : 50;
  const x = h.x + (p.gear === 'windup' ? -9 : 3);
  const y = h.y - Math.round(height / 2);
  if (p.gear === 'down') {
    return box([px(62, 130, 5, 34, item.wood), px(64, 132, 2, 30, item.string), px(58, 146, 40, 3, item.arrow)]);
  }
  return box([
    px(x, y, 5, height, item.wood),
    px(x + 4, y + 6, 5, 8, item.wood),
    px(x + 4, y + height - 14, 5, 8, item.wood),
    px(x + 1, y + 4, 2, height - 8, item.string),
    p.gear === 'strike' || p.gear === 'cast'
      ? px(x - 26, h.y - 2, 50, 3, item.arrow)
      : px(x - 9, h.y - 1, 29, 3, item.arrow, ' opacity="0.82"')
  ]);
}

function chestLayer(item, row, frame) {
  const p = pose(row, frame);
  if (p.gear === 'down') return box([px(57 + p.bodyX, 120 + p.bodyY, 37, 10, item.cloth), px(60 + p.bodyX, 121 + p.bodyY, 31, 3, item.trim)]);
  return box([
    px(68 + p.bodyX, 75 + p.bodyY, 24, 30, item.cloth),
    px(70 + p.bodyX, 77 + p.bodyY, 20, 5, item.trim),
    px(79 + p.bodyX, 82 + p.bodyY, 3, 20, item.stitch),
    px(70 + p.bodyX, 101 + p.bodyY, 20, 4, item.trim)
  ]);
}

function bootsLayer(item, row, frame) {
  const p = pose(row, frame);
  const front = legRects(p.frontLeg, true, p).slice(-1)[0];
  const back = legRects(p.backLeg, false, p).slice(-1)[0];
  return box([
    px(back.x, back.y, back.w, back.h, item.leather),
    px(back.x, back.y + back.h - 2, back.w + 2, 3, item.sole),
    px(front.x, front.y, front.w, front.h, item.leather),
    px(front.x + Math.max(2, front.w - 6), front.y + 1, 4, 3, item.buckle),
    px(front.x, front.y + front.h - 2, front.w + 2, 3, item.sole)
  ]);
}

function ringLayer(item, row, frame) {
  const p = pose(row, frame);
  const h = handPoint(p);
  const pulse = frame % 3;
  if (p.gear === 'down') return box([px(103, 129, 5, 5, item.metal), px(108, 128, 2, 2, item.glow, ' opacity="0.8"')]);
  return box([
    px(h.x - 2, h.y + 1, 7, 5, item.metal),
    px(h.x, h.y + 2, 3, 2, 'transparent'),
    px(h.x + 4, h.y - 1 - pulse, 3, 3, item.glow, ' opacity="0.75"'),
    px(h.x - 5 - pulse, h.y + 6, 4, 2, item.glow, ' opacity="0.25"')
  ]);
}

function amuletLayer(item, row, frame) {
  const p = pose(row, frame);
  const bob = frame % 2 ? -1 : 1;
  if (p.gear === 'down') return box([px(78, 126, 8, 4, item.metal), px(86, 126, 5, 5, item.gem || item.glow)]);
  const x = 79 + p.bodyX;
  const y = 82 + p.bodyY + bob;
  return box([
    px(x - 9, y - 6, 5, 3, item.metal),
    px(x + 10, y - 6, 5, 3, item.metal),
    px(x - 4, y - 3, 4, 4, item.metal),
    px(x + 3, y - 3, 4, 4, item.metal),
    px(x - 2, y + 1, 9, 9, item.gem || item.glow || item.metal),
    px(x, y + 3, 5, 5, item.glow || item.accent || item.metal, ' opacity="0.78"')
  ]);
}

function headLayer(item, row, frame) {
  const p = pose(row, frame);
  if (p.gear === 'down') {
    return box([
      px(96 + p.bodyX, 108 + p.headY, 24, 7, item.dark || item.metal),
      px(99 + p.bodyX, 105 + p.headY, 18, 5, item.trim || item.metal)
    ]);
  }
  const x = 74 + p.bodyX + p.headX;
  const y = 43 + p.headY;
  return box([
    px(x - 2, y + 6, 26, 8, item.dark || item.metal),
    px(x, y + 3, 22, 8, item.metal || item.trim),
    px(x + 3, y, 4, 5, item.trim || item.metal),
    px(x + 10, y - 2, 4, 7, item.trim || item.metal),
    px(x + 17, y, 4, 5, item.trim || item.metal)
  ]);
}

function glovesLayer(item, row, frame) {
  const p = pose(row, frame);
  const front = armRects(p.frontArm, true, p).slice(-1)[0];
  const back = armRects(p.backArm, false, p).slice(-1)[0];
  return box([
    px(back.x - 1, back.y - 1, back.w + 2, back.h + 1, item.dark || item.metal),
    px(back.x + Math.max(1, back.w - 5), back.y, 4, 3, item.edge || item.metal),
    px(front.x - 1, front.y - 1, front.w + 2, front.h + 1, item.dark || item.metal),
    px(front.x + Math.max(1, front.w - 5), front.y, 4, 3, item.edge || item.metal)
  ]);
}

function shieldLayer(item, row, frame) {
  const p = pose(row, frame);
  const x = p.gear === 'strike' ? 67 : 55;
  const y = p.gear === 'down' ? 125 : 76 + p.bodyY;
  if (p.gear === 'down') return box([px(56, 127, 29, 16, item.face), px(53, 124, 35, 5, item.trim), px(53, 142, 35, 4, item.trim)]);
  return box([
    px(x + p.bodyX, y, 27, 35, item.trim),
    px(x + 3 + p.bodyX, y + 4, 21, 27, item.face),
    px(x + 8 + p.bodyX, y + 7, 11, 20, item.metal),
    px(x + 5 + p.bodyX, y + 30, 17, 4, item.trim)
  ]);
}

function gripLayer(item, row, frame) {
  const p = pose(row, frame);
  const h = handPoint(p);
  if (p.gear === 'down') return box([px(103, 116, 11, 9, item.metal), px(106, 115, 8, 4, item.edge)]);
  return box([
    px(h.x - 4, h.y - 5, 13, 12, item.dark),
    px(h.x - 2, h.y - 3, 11, 9, item.metal),
    px(h.x + 5, h.y - 4, 8, 4, item.edge)
  ]);
}

function coreLayer(item, row, frame) {
  const p = pose(row, frame);
  const bob = frame % 2 ? -3 : 2;
  const x = 55 + p.bodyX;
  const y = 67 + p.bodyY + bob;
  return box([
    px(x - 8, y - 8, 26, 26, item.glow, ' opacity="0.18"'),
    px(x, y, 11, 11, item.core),
    px(x + 3, y - 3, 8, 8, item.glow),
    px(x + 1, y + 12, 9, 4, item.dark)
  ]);
}

function focusLayer(item, row, frame) {
  const p = pose(row, frame);
  const bob = frame % 2 ? -3 : 2;
  const x = 108 + p.bodyX;
  const y = 69 + p.bodyY + bob;
  return box([
    px(x - 9, y - 9, 26, 26, item.glow, ' opacity="0.16"'),
    px(x, y - 6, 9, 6, item.glow),
    px(x - 5, y, 19, 8, item.core),
    px(x, y + 8, 9, 6, item.dark)
  ]);
}

function scopeLayer(item, row, frame) {
  const p = pose(row, frame);
  const x = 96 + p.bodyX + (p.gear === 'strike' ? 13 : 0);
  const y = 58 + p.bodyY;
  return box([
    px(x, y, 18, 7, item.metal),
    px(x + 14, y - 1, 7, 9, item.trim),
    px(x + 16, y + 1, 4, 5, item.lens),
    px(x - 3, y + 2, 5, 3, item.metal)
  ]);
}

function kitLayer(item, row, frame) {
  const p = pose(row, frame);
  const x = 56 + p.bodyX;
  const y = p.gear === 'down' ? 124 : 98 + p.bodyY;
  return box([
    px(x, y, 18, 15, item.leather),
    px(x + 2, y + 2, 14, 4, item.metal),
    px(x + 14, y + 8, 14, 4, item.cord),
    px(x + 25, y + 6, 5, 9, item.metal)
  ]);
}

function equipmentColor(item, keys, fallback) {
  for (const key of keys) {
    if (item && item[key]) return item[key];
  }
  return fallback;
}

function attachmentGroup(anchor, body) {
  return `<g transform="translate(${anchor.x} ${anchor.y}) rotate(${anchor.angle || 0})">${body}</g>`;
}

function safeWeaponAnchor(source, length) {
  const anchor = Object.assign({}, source || { x: 80, y: 90, angle: 0 });
  const radians = Number(anchor.angle || 0) * Math.PI / 180;
  const endX = anchor.x + Math.cos(radians) * length;
  const endY = anchor.y + Math.sin(radians) * length;
  const minX = Math.min(anchor.x - 8, endX - 8);
  const maxX = Math.max(anchor.x + 8, endX + 8);
  const minY = Math.min(anchor.y - 12, endY - 12);
  const maxY = Math.max(anchor.y + 12, endY + 12);
  if (minX < 5) anchor.x += 5 - minX;
  if (maxX > FRAME - 5) anchor.x -= maxX - (FRAME - 5);
  if (minY < 5) anchor.y += 5 - minY;
  if (maxY > FRAME - 5) anchor.y -= maxY - (FRAME - 5);
  return anchor;
}

function swordLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const blade = equipmentColor(item, ['blade', 'metal'], '#c9d5dd');
  const bright = equipmentColor(item, ['bright', 'shine', 'edge'], '#f4fbff');
  const grip = equipmentColor(item, ['grip', 'haft', 'dark'], '#765039');
  const anchor = safeWeaponAnchor(rig.weapon, 48);
  return attachmentGroup(anchor, box([
    px(-6, -3, 15, 6, grip),
    px(5, -7, 5, 14, bright),
    px(9, -4, 38, 8, blade),
    px(13, -3, 31, 2, bright),
    polygon('47,-4 55,0 47,4', bright)
  ]));
}

function axeLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const haft = equipmentColor(item, ['haft', 'grip', 'dark'], '#765039');
  const blade = equipmentColor(item, ['blade', 'metal'], '#c9d5dd');
  const bright = equipmentColor(item, ['bright', 'shine', 'edge'], '#f4fbff');
  const anchor = safeWeaponAnchor(rig.weapon, 45);
  return attachmentGroup(anchor, box([
    px(-6, -3, 45, 6, haft),
    px(33, -14, 16, 28, blade),
    polygon('38,-14 54,-8 49,0 38,-2', bright),
    px(5, -5, 7, 10, haft)
  ]));
}

function wandLayerV2(item, row, frame, staff) {
  const rig = getEquipmentAttachment(row, frame);
  const length = staff ? 52 : 36;
  const rod = equipmentColor(item, ['rod', 'haft', 'grip'], '#765039');
  const gem = equipmentColor(item, ['gem', 'core', 'bright'], '#c6f4ff');
  const glow = equipmentColor(item, ['glow', 'accent', 'bright'], '#8bd7ff');
  const anchor = safeWeaponAnchor(rig.weapon, length + 12);
  return attachmentGroup(anchor, box([
    px(-5, -3, length + 5, 6, rod),
    px(length - 1, -7, 12, 14, gem),
    px(length + 2, -4, 6, 5, '#ffffff', ' opacity="0.72"'),
    px(length - 6, -12, 23, 24, glow, ' opacity="0.18"')
  ]));
}

function bowLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const height = item.long ? 58 : 50;
  const wood = equipmentColor(item, ['wood', 'leather', 'haft'], '#8a5c2f');
  const string = equipmentColor(item, ['string', 'bright', 'trim'], '#f5efd6');
  const arrow = equipmentColor(item, ['arrow', 'accent', 'bright'], '#ffe16a');
  const anchor = Object.assign({}, rig.weapon, { angle: rig.bowAngle || 0 });
  anchor.x = Math.max(35, Math.min(125, anchor.x));
  anchor.y = Math.max(height / 2 + 6, Math.min(FRAME - height / 2 - 6, anchor.y));
  const firing = row === 'basic' || row === 'skill';
  return attachmentGroup(anchor, box([
    px(-2, -height / 2, 5, height, wood),
    px(2, -height / 2 + 4, 6, 8, wood),
    px(2, height / 2 - 12, 6, 8, wood),
    svgPath(`M 1 ${-height / 2 + 2} L ${firing ? -12 : 1} 0 L 1 ${height / 2 - 2}`, 'none', string, ' stroke-width="2"'),
    firing ? px(-18, -2, 51, 4, arrow) : px(-8, -2, 30, 4, arrow, ' opacity="0.82"'),
    firing ? polygon('33,-4 41,0 33,4', arrow) : ''
  ]));
}

function chestLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const torso = rig.torso;
  const cloth = equipmentColor(item, ['cloth', 'metal', 'leather', 'core'], '#6d7682');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright', 'accent'], '#d6a86d');
  const stitch = equipmentColor(item, ['stitch', 'dark', 'sole'], '#392b2b');
  const width = Math.max(22, torso.width);
  const height = Math.max(27, torso.height - 4);
  return attachmentGroup(torso, box([
    polygon(`${-width / 2},${-height / 2 + 5} ${-width / 2 + 4},${height / 2} ${width / 2 - 4},${height / 2} ${width / 2},${-height / 2 + 5} 0,${-height / 2}`, cloth),
    px(-width / 2 + 3, -height / 2 + 6, width - 6, 5, trim),
    px(-2, -height / 2 + 7, 4, height - 11, stitch),
    px(-width / 2 + 4, height / 2 - 5, width - 8, 5, trim)
  ]));
}

function bootsLayerV2(item, row, frame) {
  const leather = equipmentColor(item, ['leather', 'cloth', 'metal'], '#6f412b');
  const sole = equipmentColor(item, ['sole', 'dark', 'edge'], '#2b1d1b');
  const buckle = equipmentColor(item, ['buckle', 'trim', 'bright'], '#d6a14a');
  const rig = getEquipmentAttachment(row, frame);
  return box(rig.feet.map((foot, index) => attachmentGroup(foot, box([
    px(-8, -8, 17, 10, leather),
    px(-9, 0, 20, 4, sole),
    index === 0 ? px(3, -7, 4, 4, buckle) : ''
  ]))));
}

function headLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const metal = equipmentColor(item, ['metal', 'cloth', 'leather'], '#667481');
  const dark = equipmentColor(item, ['dark', 'sole', 'stitch'], '#29313a');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright', 'accent'], '#d3dde5');
  return attachmentGroup(rig.head, box([
    px(-17, -16, 34, 9, dark),
    px(-14, -23, 28, 12, metal),
    px(-10, -27, 5, 7, trim),
    px(-2, -29, 5, 9, trim),
    px(7, -27, 5, 7, trim),
    px(10, -11, 9, 5, trim)
  ]));
}

function glovesLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const dark = equipmentColor(item, ['dark', 'leather', 'cloth'], '#4b3429');
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#8b6a52');
  const edge = equipmentColor(item, ['edge', 'bright', 'buckle'], '#d3dde5');
  return box([rig.offHand, rig.mainHand].map((hand) => attachmentGroup(hand, box([
    px(-6, -5, 13, 11, dark),
    px(-4, -3, 11, 8, metal),
    px(3, -4, 5, 4, edge)
  ]))));
}

function ringLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#f7d879');
  const glow = equipmentColor(item, ['glow', 'accent', 'bright'], '#ffe16a');
  return attachmentGroup(rig.mainHand, box([
    px(1, -2, 6, 5, metal),
    px(4, -4, 4, 4, glow, ' opacity="0.78"')
  ]));
}

function amuletLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#d9b967');
  const gem = equipmentColor(item, ['gem', 'glow', 'accent'], '#8bd7ff');
  return attachmentGroup(rig.torso, box([
    svgPath('M -10 -12 L 0 -2 L 10 -12', 'none', metal, ' stroke-width="3"'),
    px(-5, -3, 10, 10, gem),
    px(-2, -1, 4, 4, '#ffffff', ' opacity="0.68"')
  ]));
}

function shieldLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const face = equipmentColor(item, ['face', 'cloth', 'metal'], '#466d91');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright'], '#d5ecff');
  const metal = equipmentColor(item, ['metal', 'dark'], '#6fa8d9');
  return attachmentGroup(rig.shield, box([
    polygon('-16,-17 16,-17 14,11 0,20 -14,11', trim),
    polygon('-12,-13 12,-13 10,8 0,15 -10,8', face),
    px(-3, -11, 6, 23, metal),
    px(-10, -3, 20, 6, metal)
  ]));
}

function gripLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const dark = equipmentColor(item, ['dark', 'leather'], '#4d1f24');
  const metal = equipmentColor(item, ['metal', 'cloth'], '#a22d36');
  const edge = equipmentColor(item, ['edge', 'bright', 'trim'], '#ff6b5e');
  return box([rig.offHand, rig.mainHand].map((hand) => attachmentGroup(hand, box([
    px(-7, -6, 15, 13, dark),
    px(-5, -4, 13, 10, metal),
    polygon('5,-5 12,-2 6,1', edge)
  ]))));
}

function coreLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const core = equipmentColor(item, ['core', 'gem', 'metal'], '#ff6b35');
  const glow = equipmentColor(item, ['glow', 'bright', 'accent'], '#ffc15e');
  const anchor = { x: rig.torso.x - 19, y: rig.torso.y - 7, angle: rig.torso.angle };
  return attachmentGroup(anchor, box([
    px(-12, -12, 24, 24, glow, ' opacity="0.18"'),
    polygon('0,-8 8,0 0,8 -8,0', core),
    px(-2, -4, 5, 5, glow)
  ]));
}

function focusLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const core = equipmentColor(item, ['core', 'gem', 'metal'], '#28c7b7');
  const glow = equipmentColor(item, ['glow', 'bright', 'accent'], '#b8fff2');
  const anchor = {
    x: Math.max(16, Math.min(FRAME - 16, rig.mainHand.x + 8)),
    y: Math.max(16, Math.min(FRAME - 16, rig.mainHand.y - 13)),
    angle: 0
  };
  return attachmentGroup(anchor, box([
    px(-13, -13, 26, 26, glow, ' opacity="0.16"'),
    polygon('0,-9 9,0 0,9 -9,0', core),
    px(-2, -5, 5, 5, glow)
  ]));
}

function scopeLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const metal = equipmentColor(item, ['metal', 'dark'], '#4b5663');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright'], '#d8c25f');
  const lens = equipmentColor(item, ['lens', 'glow', 'accent'], '#ffe16a');
  const anchor = { x: rig.head.x + 12, y: rig.head.y + 2, angle: rig.head.angle };
  return attachmentGroup(anchor, box([
    px(-5, -4, 20, 8, metal),
    px(11, -5, 7, 10, trim),
    px(13, -3, 4, 6, lens)
  ]));
}

function kitLayerV2(item, row, frame) {
  const rig = getEquipmentAttachment(row, frame);
  const leather = equipmentColor(item, ['leather', 'cloth', 'dark'], '#8a5a36');
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#b7c3ca');
  const cord = equipmentColor(item, ['cord', 'edge', 'stitch'], '#3f2c24');
  const anchor = { x: rig.torso.x - 18, y: rig.torso.y + 13, angle: rig.torso.angle };
  return attachmentGroup(anchor, box([
    px(-9, -7, 19, 15, leather),
    px(-7, -5, 15, 4, metal),
    px(7, 1, 15, 4, cord),
    px(18, -1, 5, 9, metal)
  ]));
}

function drawEquipmentLayer(item, row, frame) {
  if (item.kind === 'sword') return swordLayerV2(item, row, frame);
  if (item.kind === 'axe') return axeLayerV2(item, row, frame);
  if (item.kind === 'wand') return wandLayerV2(item, row, frame, false);
  if (item.kind === 'staff') return wandLayerV2(item, row, frame, true);
  if (item.kind === 'bow') return bowLayerV2(item, row, frame);
  if (item.kind === 'chest') return chestLayerV2(item, row, frame);
  if (item.kind === 'boots') return bootsLayerV2(item, row, frame);
  if (item.kind === 'head') return headLayerV2(item, row, frame);
  if (item.kind === 'gloves') return glovesLayerV2(item, row, frame);
  if (item.kind === 'ring') return ringLayerV2(item, row, frame);
  if (item.kind === 'amulet') return amuletLayerV2(item, row, frame);
  if (item.kind === 'shield') return shieldLayerV2(item, row, frame);
  if (item.kind === 'grip') return gripLayerV2(item, row, frame);
  if (item.kind === 'core') return coreLayerV2(item, row, frame);
  if (item.kind === 'focus') return focusLayerV2(item, row, frame);
  if (item.kind === 'scope') return scopeLayerV2(item, row, frame);
  if (item.kind === 'kit') return kitLayerV2(item, row, frame);
  return '';
}

function sparkle(x, y, color) {
  return box([
    px(x + 3, y, 4, 10, color),
    px(x, y + 3, 10, 4, color),
    px(x + 4, y + 4, 2, 2, '#ffffff')
  ]);
}

function classPalette(variant) {
  return Object.assign({}, COLORS, {
    shirt: variant.shirt,
    shirtLight: variant.shirtLight,
    pants: variant.pants,
    pantsDark: variant.pantsDark || '#222f3c'
  });
}

function drawClassMotif(variant, row, frame) {
  const p = pose(row, frame);
  const x = p.bodyX;
  const y = p.bodyY;
  const accent = variant.accent;
  const pulse = frame % 2 ? -2 : 2;
  const badge = row === 'defeat'
    ? px(82 + x, 121 + y, 6, 5, accent)
    : px(77 + x, 82 + y, 7, 7, accent);
  if (variant.motif === 'slash') {
    return box([
      badge,
      row === 'basic' || row === 'skill' ? polygon(`${103 + x},70 ${133 + x},61 ${127 + x},70 ${96 + x},80`, accent, ' opacity="0.5"') : ''
    ]);
  }
  if (variant.motif === 'spark') {
    return box([
      badge,
      px(110 + x, 64 + y + pulse, 18, 18, accent, ' opacity="0.18"'),
      sparkle(118 + x, 58 + y + pulse, accent)
    ]);
  }
  if (variant.motif === 'leaf') {
    return box([
      badge,
      polygon(`${54 + x},83 ${64 + x},73 ${73 + x},83 ${64 + x},92`, '#7bcf90', ' opacity="0.78"'),
      px(63 + x, 78, 3, 15, accent)
    ]);
  }
  if (variant.motif === 'shield') {
    return box([
      badge,
      px(50 + x, 74 + y, 12, 39, accent, ' opacity="0.2"'),
      px(118 + x, 90 + y, 20, 7, accent, row === 'skill' ? ' opacity="0.58"' : ' opacity="0.2"')
    ]);
  }
  if (variant.motif === 'rage') {
    return box([
      badge,
      polygon(`${50 + x},65 ${62 + x},54 ${58 + x},75`, '#ff6b5e', ' opacity="0.78"'),
      row === 'skill' || row === 'basic' ? px(35 + x, 92 + y, 33, 6, '#ff6b5e', ' opacity="0.42"') : ''
    ]);
  }
  if (variant.motif === 'tempo') {
    return box([
      badge,
      px(48 + x + pulse, 68 + y, 31, 5, accent, ' opacity="0.28"'),
      px(44 + x + pulse, 80 + y, 24, 4, '#ffffff', ' opacity="0.2"')
    ]);
  }
  if (variant.motif === 'flame') {
    return box([
      badge,
      polygon(`${114 + x},57 ${125 + x},78 ${111 + x},78`, '#ff6b35', ' opacity="0.82"'),
      polygon(`${119 + x},62 ${125 + x},77 ${116 + x},76`, '#ffd36b', ' opacity="0.9"')
    ]);
  }
  if (variant.motif === 'rune') {
    return box([
      badge,
      px(108 + x, 63 + y + pulse, 18, 5, accent, ' opacity="0.8"'),
      px(114 + x, 57 + y + pulse, 5, 18, accent, ' opacity="0.72"'),
      px(52 + x, 101 + y, 17, 4, '#b8fff2', ' opacity="0.64"')
    ]);
  }
  if (variant.motif === 'storm') {
    return box([
      badge,
      polygon(`${111 + x},52 ${123 + x},52 ${116 + x},69 ${126 + x},69 ${109 + x},91 ${114 + x},73 ${106 + x},73`, accent, ' opacity="0.78"')
    ]);
  }
  if (variant.motif === 'scope') {
    return box([
      badge,
      px(95 + x, 58 + y, 21, 6, '#4b5663'),
      px(111 + x, 57 + y, 7, 8, accent),
      px(116 + x, 79 + y, 24, 3, accent, row === 'basic' || row === 'skill' ? ' opacity="0.76"' : ' opacity="0.28"')
    ]);
  }
  if (variant.motif === 'trap') {
    return box([
      badge,
      px(43 + x, 122 + y, 35, 7, '#b7c3ca', ' opacity="0.75"'),
      px(47 + x, 116 + y, 5, 8, accent),
      px(59 + x, 116 + y, 5, 8, accent),
      px(71 + x, 116 + y, 5, 8, accent)
    ]);
  }
  if (variant.motif === 'paw') {
    return box([
      badge,
      px(45 + x, 116 + y, 13, 11, '#9bc776', ' opacity="0.82"'),
      px(42 + x, 110 + y, 5, 5, '#9bc776', ' opacity="0.82"'),
      px(51 + x, 108 + y, 5, 5, '#9bc776', ' opacity="0.82"'),
      px(60 + x, 111 + y, 5, 5, '#9bc776', ' opacity="0.82"')
    ]);
  }
  return badge;
}

function drawClassPlayer(variant, row, frame) {
  return box([
    drawGenericPlayer(row, frame, classPalette(variant)),
    drawEquipmentLayer(variant.equipment, row, frame),
    drawClassMotif(variant, row, frame)
  ]);
}

function drawCoinIcon(icon) {
  return box([
    ellipse(88, 126, 55, 16, icon.dark, ' opacity="0.24"'),
    px(46, 103, 74, 19, icon.dark),
    ellipse(83, 103, 37, 13, icon.main),
    ellipse(83, 102, 28, 8, icon.light),
    px(56, 84, 74, 19, icon.accent),
    ellipse(93, 84, 37, 13, icon.main),
    ellipse(93, 83, 28, 8, icon.light),
    px(48, 66, 74, 19, icon.dark),
    ellipse(85, 66, 37, 13, icon.main),
    ellipse(85, 65, 28, 8, icon.light),
    px(112, 89, 20, 31, icon.dark),
    px(118, 85, 25, 33, icon.main),
    ellipse(130, 85, 13, 8, icon.light),
    px(125, 90, 6, 23, icon.accent),
    px(70, 58, 9, 9, icon.light),
    sparkle(126, 51, icon.light),
    sparkle(40, 90, icon.light)
  ]);
}

function drawPotionIcon(icon) {
  return box([
    ellipse(91, 132, 40, 10, icon.dark, ' opacity="0.22"'),
    px(76, 29, 28, 14, '#8a5531'),
    px(73, 42, 34, 14, '#5c3a2a'),
    px(68, 54, 44, 16, icon.glass, ' opacity="0.62"'),
    px(57, 69, 68, 54, icon.dark),
    px(62, 72, 58, 50, icon.glass, ' opacity="0.52"'),
    px(66, 86, 50, 35, icon.main),
    px(66, 80, 50, 11, icon.light),
    px(78, 60, 11, 52, '#ffffff', ' opacity="0.32"'),
    px(94, 91, 11, 10, '#ffffff', ' opacity="0.26"'),
    px(55, 120, 72, 8, icon.dark),
    sparkle(117, 53, icon.light),
    sparkle(44, 81, icon.light)
  ]);
}

function drawCouponIcon(icon) {
  const mark = icon.mark === 'usable'
    ? [
      px(58, 85, 13, 16, icon.accent),
      px(61, 78, 7, 8, icon.light),
      px(55, 100, 19, 5, icon.dark),
      px(82, 86, 14, 14, icon.accent),
      px(86, 80, 6, 8, icon.light),
      px(80, 99, 18, 5, icon.dark),
      px(106, 84, 13, 19, icon.accent),
      px(109, 77, 7, 8, icon.light)
    ]
    : icon.mark === 'etc'
      ? [
        px(55, 86, 20, 16, icon.dark),
        px(59, 82, 20, 16, icon.accent),
        px(63, 86, 12, 5, icon.light),
        px(82, 86, 18, 18, icon.dark),
        px(86, 82, 18, 18, icon.accent),
        px(91, 87, 8, 8, icon.light),
        px(108, 88, 15, 15, icon.dark),
        px(111, 85, 15, 15, icon.accent)
      ]
      : [
        px(56, 86, 21, 16, icon.accent),
        px(61, 80, 11, 9, icon.light),
        px(84, 84, 16, 20, icon.dark),
        px(88, 78, 8, 8, icon.light),
        px(106, 86, 16, 16, icon.accent),
        px(109, 80, 10, 8, icon.light)
      ];
  return box([
    ellipse(90, 132, 55, 9, icon.dark, ' opacity="0.2"'),
    px(41, 55, 98, 64, icon.dark),
    px(47, 61, 86, 52, icon.main),
    px(53, 67, 74, 10, icon.light),
    mark.join(''),
    px(39, 72, 10, 12, 'transparent'),
    px(131, 72, 10, 12, 'transparent'),
    px(61, 111, 58, 5, icon.dark),
    sparkle(125, 49, icon.light)
  ]);
}

function drawDustIcon(icon) {
  return box([
    ellipse(88, 133, 46, 10, icon.dark, ' opacity="0.22"'),
    px(64, 84, 48, 38, icon.dark),
    px(68, 77, 40, 45, icon.main),
    px(73, 72, 29, 12, icon.light),
    px(75, 61, 25, 16, '#9a632a'),
    px(80, 58, 16, 7, icon.accent),
    px(60, 118, 56, 10, icon.dark),
    px(46, 118, 16, 10, icon.main),
    px(112, 119, 18, 9, icon.light),
    sparkle(122, 75, icon.accent),
    sparkle(47, 79, icon.light),
    px(135, 106, 6, 6, icon.light),
    px(37, 109, 5, 5, icon.accent)
  ]);
}

function drawCatalystIcon(icon) {
  return box([
    ellipse(89, 135, 48, 10, icon.dark, ' opacity="0.22"'),
    polygon('78,34 105,34 113,108 91,132 69,108', icon.dark),
    polygon('83,42 101,42 106,104 91,121 76,104', icon.main),
    polygon('91,42 101,42 106,104 92,116', icon.light, ' opacity="0.72"'),
    polygon('44,74 67,55 82,122 58,130', icon.dark),
    polygon('50,77 65,64 75,118 60,124', icon.accent),
    polygon('112,61 137,80 122,130 102,116', icon.dark),
    polygon('115,69 131,82 119,122 108,112', icon.main),
    px(84, 95, 17, 7, '#ffffff', ' opacity="0.38"'),
    sparkle(127, 47, icon.light),
    sparkle(43, 54, icon.light)
  ]);
}

function drawFractureIcon(icon) {
  return box([
    ellipse(90, 134, 50, 10, icon.dark, ' opacity="0.22"'),
    polygon('82,37 119,57 101,124 63,101', icon.dark),
    polygon('85,46 111,59 96,114 70,98', icon.main),
    polygon('94,50 111,59 98,106 91,83', icon.light, ' opacity="0.64"'),
    px(85, 75, 26, 5, icon.dark),
    px(72, 92, 28, 5, icon.dark),
    polygon('42,91 61,81 72,126 48,124', icon.dark),
    polygon('47,94 58,88 65,119 52,118', icon.accent),
    px(120, 104, 15, 14, icon.main),
    px(135, 119, 8, 8, icon.light),
    sparkle(123, 45, icon.accent),
    sparkle(47, 58, icon.light)
  ]);
}

function drawGelIcon(icon) {
  return box([
    ellipse(89, 134, 49, 10, icon.dark, ' opacity="0.22"'),
    px(64, 81, 52, 42, icon.dark),
    px(58, 93, 64, 31, icon.dark),
    px(69, 74, 42, 47, icon.main),
    px(62, 94, 56, 27, icon.main),
    px(78, 59, 25, 20, icon.main),
    px(82, 51, 17, 12, icon.light),
    px(76, 82, 12, 11, '#ffffff', ' opacity="0.38"'),
    px(99, 99, 8, 7, icon.accent, ' opacity="0.86"'),
    px(61, 118, 58, 8, icon.dark),
    sparkle(119, 68, icon.light)
  ]);
}

function drawOreIcon(icon) {
  return box([
    ellipse(90, 134, 52, 10, icon.dark, ' opacity="0.22"'),
    polygon('48,96 72,70 100,87 89,124 58,126', icon.dark),
    polygon('57,96 75,78 94,90 84,116 62,118', icon.main),
    polygon('74,79 94,90 84,116 76,98', icon.light, ' opacity="0.62"'),
    polygon('93,86 126,72 145,103 127,127 97,119', icon.dark),
    polygon('101,91 124,80 137,104 124,120 102,113', icon.main),
    polygon('118,80 137,104 124,120 121,98', icon.accent, ' opacity="0.7"'),
    polygon('62,55 88,47 101,70 82,88 58,80', icon.dark),
    polygon('69,60 86,54 94,70 81,81 65,76', icon.main),
    px(78, 59, 14, 5, icon.light),
    px(111, 91, 15, 5, icon.light),
    sparkle(135, 63, icon.accent)
  ]);
}

function drawScrollIcon(icon) {
  return box([
    ellipse(90, 135, 50, 10, icon.dark, ' opacity="0.2"'),
    px(48, 47, 83, 88, icon.dark),
    px(54, 51, 70, 76, icon.main),
    px(61, 59, 56, 6, icon.light),
    px(62, 78, 48, 5, icon.dark, ' opacity="0.54"'),
    px(62, 93, 36, 5, icon.dark, ' opacity="0.54"'),
    px(70, 108, 42, 5, icon.dark, ' opacity="0.48"'),
    px(43, 39, 29, 18, icon.dark),
    px(44, 42, 24, 12, icon.light),
    px(113, 124, 30, 16, icon.dark),
    px(116, 126, 24, 10, icon.light),
    sparkle(130, 75, icon.accent),
    sparkle(47, 101, icon.accent)
  ]);
}

function drawRationIcon(icon) {
  return box([
    ellipse(90, 136, 50, 10, icon.dark, ' opacity="0.22"'),
    px(55, 82, 74, 43, icon.dark),
    px(60, 75, 64, 43, icon.main),
    px(70, 67, 41, 19, icon.light),
    px(66, 93, 52, 7, icon.dark, ' opacity="0.38"'),
    px(67, 107, 31, 7, icon.dark, ' opacity="0.38"'),
    px(117, 63, 18, 53, icon.dark),
    px(121, 67, 10, 45, icon.accent),
    px(40, 92, 18, 34, icon.accent),
    px(44, 87, 11, 36, icon.dark, ' opacity="0.28"')
  ]);
}

function drawGuardTonicIcon(icon) {
  return box([
    ellipse(90, 136, 45, 9, icon.dark, ' opacity="0.2"'),
    drawPotionIcon(icon),
    polygon('90,63 121,75 115,112 90,130 65,112 59,75', icon.dark, ' opacity="0.72"'),
    polygon('90,71 111,79 107,107 90,120 73,107 69,79', icon.main),
    px(86, 81, 9, 30, icon.light, ' opacity="0.72"'),
    px(78, 91, 24, 8, icon.light, ' opacity="0.72"')
  ]);
}

function drawOilIcon(icon) {
  return box([
    ellipse(90, 136, 44, 9, icon.dark, ' opacity="0.2"'),
    drawPotionIcon(icon),
    polygon('45,111 84,102 77,126 37,136', icon.dark),
    polygon('50,113 77,107 72,121 44,128', icon.light),
    px(104, 65, 31, 8, icon.accent),
    px(112, 80, 25, 7, icon.accent, ' opacity="0.8"'),
    sparkle(136, 59, icon.light)
  ]);
}

function drawMagnetIcon(icon) {
  return box([
    ellipse(90, 137, 50, 9, icon.dark, ' opacity="0.2"'),
    px(53, 58, 27, 66, icon.dark),
    px(100, 58, 27, 66, icon.dark),
    px(66, 96, 16, 28, icon.dark),
    px(98, 96, 16, 28, icon.dark),
    px(59, 64, 16, 52, icon.main),
    px(105, 64, 16, 52, icon.main),
    px(59, 64, 16, 15, icon.light),
    px(105, 64, 16, 15, icon.light),
    px(75, 111, 30, 12, icon.dark),
    px(81, 84, 18, 8, icon.accent),
    sparkle(43, 74, icon.accent),
    sparkle(136, 98, icon.accent)
  ]);
}

function drawWhistleIcon(icon) {
  return box([
    ellipse(90, 136, 48, 9, icon.dark, ' opacity="0.2"'),
    px(53, 83, 79, 28, icon.dark),
    px(59, 78, 63, 27, icon.main),
    px(68, 72, 38, 12, icon.light),
    px(117, 89, 22, 14, icon.dark),
    px(122, 91, 16, 8, icon.main),
    px(68, 88, 17, 10, icon.dark, ' opacity="0.48"'),
    px(92, 87, 18, 7, '#ffffff', ' opacity="0.38"'),
    ellipse(61, 116, 17, 15, icon.dark),
    ellipse(61, 116, 10, 8, 'transparent'),
    px(47, 113, 18, 5, icon.accent),
    px(42, 105, 7, 16, icon.accent),
    sparkle(128, 66, icon.light)
  ]);
}

function drawAttunementPrismIcon(icon) {
  return box([
    ellipse(90, 137, 48, 9, icon.dark, ' opacity="0.2"'),
    polygon('90,28 129,72 112,128 68,128 51,72', icon.dark),
    polygon('90,39 117,73 105,117 75,117 63,73', icon.main),
    polygon('90,39 78,74 90,88 102,74', icon.light, ' opacity="0.86"'),
    polygon('63,73 78,74 75,117', icon.dark, ' opacity="0.28"'),
    polygon('117,73 102,74 105,117', icon.dark, ' opacity="0.18"'),
    polygon('78,74 90,88 75,117', icon.main, ' opacity="0.86"'),
    polygon('102,74 90,88 105,117', icon.accent, ' opacity="0.44"'),
    polygon('90,88 105,117 75,117', icon.dark, ' opacity="0.2"'),
    px(83, 55, 13, 15, '#ffffff', ' opacity="0.35"'),
    sparkle(128, 45, icon.accent),
    sparkle(48, 58, icon.light),
    sparkle(134, 114, icon.light)
  ]);
}

function drawPrismShardIcon(icon) {
  return box([
    ellipse(88, 137, 42, 8, icon.dark, ' opacity="0.2"'),
    polygon('75,37 116,70 97,122 54,95', icon.dark),
    polygon('78,49 105,72 91,109 63,92', icon.main),
    polygon('78,49 88,75 105,72', icon.light, ' opacity="0.76"'),
    polygon('88,75 91,109 63,92', icon.dark, ' opacity="0.2"'),
    polygon('88,75 105,72 91,109', icon.main),
    px(83, 66, 10, 12, '#ffffff', ' opacity="0.32"'),
    polygon('117,99 139,113 128,135 106,124', icon.dark),
    polygon('118,106 130,114 124,127 112,120', icon.accent),
    px(123, 112, 5, 5, icon.light, ' opacity="0.62"'),
    sparkle(124, 48, icon.accent),
    sparkle(49, 64, icon.light),
    sparkle(138, 126, icon.light)
  ]);
}

function drawEchoPrismIcon(icon) {
  return box([
    ellipse(90, 137, 50, 9, icon.dark, ' opacity="0.2"'),
    polygon('90,27 132,72 112,130 68,130 48,72', icon.dark),
    polygon('90,39 119,74 105,118 75,118 61,74', icon.main),
    polygon('90,39 78,75 90,90 102,75', icon.light, ' opacity="0.86"'),
    polygon('61,74 78,75 75,118', icon.dark, ' opacity="0.3"'),
    polygon('119,74 102,75 105,118', icon.main),
    polygon('90,90 105,118 75,118', icon.dark, ' opacity="0.2"'),
    svgPath('M73 89 C76 71 93 63 108 72 C121 80 120 99 108 109 C99 117 84 115 77 104', 'none', icon.accent, ' stroke-width="7" stroke-linecap="round" stroke-linejoin="round" opacity="0.92"'),
    svgPath('M74 89 L68 76 L84 79', 'none', icon.accent, ' stroke-width="7" stroke-linecap="round" stroke-linejoin="round" opacity="0.92"'),
    px(84, 57, 12, 14, '#ffffff', ' opacity="0.36"'),
    sparkle(130, 45, icon.accent),
    sparkle(48, 54, icon.light),
    sparkle(135, 116, icon.light)
  ]);
}

function drawManualIcon(icon) {
  return box([
    ellipse(90, 137, 50, 9, icon.dark, ' opacity="0.22"'),
    px(54, 52, 76, 82, icon.dark),
    px(62, 46, 61, 82, icon.main),
    px(70, 53, 45, 10, icon.light),
    px(70, 75, 45, 6, icon.dark, ' opacity="0.46"'),
    px(70, 91, 32, 6, icon.dark, ' opacity="0.46"'),
    px(84, 104, 28, 6, icon.dark, ' opacity="0.38"'),
    px(57, 50, 11, 80, icon.accent),
    px(75, 66, 28, 25, icon.dark, ' opacity="0.35"'),
    px(82, 62, 14, 33, icon.accent, ' opacity="0.88"'),
    px(73, 72, 32, 10, icon.accent, ' opacity="0.88"'),
    px(109, 119, 24, 12, icon.dark),
    px(112, 121, 18, 7, icon.light),
    sparkle(125, 55, icon.accent)
  ]);
}

function drawResetScrollIcon(icon) {
  return box([
    drawScrollIcon(icon),
    px(73, 72, 42, 42, icon.dark, ' opacity="0.68"'),
    px(79, 78, 30, 30, icon.accent, ' opacity="0.88"'),
    px(86, 72, 21, 8, icon.light),
    px(101, 77, 12, 16, icon.light),
    px(88, 103, 22, 7, icon.light),
    px(79, 91, 12, 16, icon.light),
    px(106, 67, 8, 8, icon.accent),
    px(75, 106, 8, 8, icon.accent)
  ]);
}

function drawWardingScrollIcon(icon) {
  return box([
    drawScrollIcon(icon),
    polygon('90,69 116,79 110,111 90,126 70,111 64,79', icon.dark, ' opacity="0.76"'),
    polygon('90,77 107,84 103,106 90,116 77,106 73,84', icon.accent),
    px(86, 86, 8, 23, icon.light, ' opacity="0.75"'),
    px(79, 95, 22, 7, icon.light, ' opacity="0.75"'),
    sparkle(122, 58, icon.accent)
  ]);
}

function drawRefinementCoreIcon(icon) {
  return box([
    ellipse(90, 136, 52, 10, icon.dark, ' opacity="0.22"'),
    polygon('90,36 118,55 126,96 90,132 54,96 62,55', icon.dark),
    polygon('90,45 110,61 116,93 90,120 64,93 70,61', icon.main),
    polygon('90,45 110,61 90,77 70,61', icon.light, ' opacity="0.76"'),
    polygon('90,77 116,93 90,120', icon.accent, ' opacity="0.78"'),
    polygon('90,77 64,93 90,120', icon.dark, ' opacity="0.26"'),
    px(83, 64, 14, 9, '#ffffff', ' opacity="0.35"'),
    px(75, 102, 12, 8, icon.light, ' opacity="0.58"'),
    px(101, 91, 10, 10, icon.accent, ' opacity="0.86"'),
    sparkle(125, 49, icon.accent),
    sparkle(51, 69, icon.light),
    sparkle(133, 111, icon.light)
  ]);
}

function drawSealIcon(icon) {
  return box([
    ellipse(91, 136, 48, 9, icon.dark, ' opacity="0.22"'),
    polygon('89,40 122,60 117,111 89,132 61,111 56,60', icon.dark),
    polygon('89,49 113,64 109,106 89,121 69,106 65,64', icon.main),
    px(84, 64, 10, 42, icon.light),
    px(74, 81, 30, 10, icon.light),
    px(119, 104, 22, 19, icon.accent),
    px(124, 96, 11, 34, icon.dark, ' opacity="0.36"'),
    sparkle(47, 62, icon.light),
    sparkle(132, 62, icon.light)
  ]);
}

function drawBossWeaponIcon(icon) {
  const hilt = [
    px(79, 113, 35, 9, icon.metal),
    px(86, 121, 10, 15, icon.dark),
    px(82, 131, 18, 7, icon.metal)
  ];
  const motifs = [
    sparkle(129, 45, icon.accent),
    sparkle(45, 68, icon.light)
  ];
  if (icon.form === 'bow') {
    return box([
      ellipse(90, 136, 54, 10, icon.dark, ' opacity="0.22"'),
      svgPath('M116 39 C62 51 49 103 75 132', 'none', icon.dark, ' stroke-width="11" stroke-linecap="round"'),
      svgPath('M111 48 C71 58 63 101 80 122', 'none', icon.main, ' stroke-width="7" stroke-linecap="round"'),
      svgPath('M116 39 L75 132', 'none', icon.light, ' stroke-width="3" opacity="0.75"'),
      px(72, 83, 48, 7, icon.metal),
      polygon('121,78 142,86 121,95', icon.accent),
      px(80, 78, 13, 17, icon.dark, ' opacity="0.52"'),
      polygon('55,62 72,71 57,76', icon.accent, ' opacity="0.9"'),
      polygon('69,118 84,108 82,126', icon.accent, ' opacity="0.9"'),
      motifs.join('')
    ]);
  }
  if (icon.form === 'staff' || icon.form === 'scepter') {
    const orb = icon.form === 'scepter'
      ? [
        ellipse(100, 46, 22, 22, icon.accent),
        ellipse(100, 46, 13, 13, icon.light, ' opacity="0.82"'),
        px(89, 66, 22, 10, icon.metal),
        polygon('78,51 100,25 122,51 100,43', icon.light, ' opacity="0.45"')
      ]
      : [
        polygon('97,26 121,50 98,74 75,50', icon.accent),
        polygon('98,36 111,50 98,64 85,50', icon.light, ' opacity="0.82"'),
        px(88, 68, 21, 11, icon.metal)
      ];
    return box([
      ellipse(91, 136, 48, 9, icon.dark, ' opacity="0.22"'),
      px(86, 64, 13, 70, icon.dark),
      px(92, 58, 10, 70, icon.main),
      px(97, 64, 5, 58, icon.light, ' opacity="0.62"'),
      orb.join(''),
      px(78, 121, 26, 8, icon.metal),
      motifs.join('')
    ]);
  }
  if (icon.form === 'axe') {
    return box([
      ellipse(90, 136, 54, 10, icon.dark, ' opacity="0.22"'),
      px(84, 55, 13, 80, icon.dark),
      px(91, 58, 10, 72, icon.main),
      polygon('88,42 132,58 112,87 87,82', icon.metal),
      polygon('94,48 122,61 109,78 94,76', icon.light, ' opacity="0.68"'),
      polygon('86,50 51,64 70,88 90,79', icon.dark),
      polygon('84,56 61,67 74,80 87,75', icon.main),
      px(79, 118, 29, 9, icon.accent),
      motifs.join('')
    ]);
  }
  if (icon.form === 'maul') {
    return box([
      ellipse(90, 136, 56, 10, icon.dark, ' opacity="0.22"'),
      px(84, 64, 14, 70, icon.dark),
      px(91, 67, 10, 62, icon.main),
      px(55, 42, 69, 42, icon.dark),
      px(62, 49, 55, 28, icon.main),
      px(72, 40, 9, 47, icon.metal),
      px(97, 40, 9, 47, icon.metal),
      px(67, 56, 44, 8, icon.light, ' opacity="0.65"'),
      polygon('47,51 62,40 62,84 47,73', icon.accent),
      polygon('132,51 117,40 117,84 132,73', icon.accent),
      motifs.join('')
    ]);
  }
  if (icon.form === 'repeater') {
    return box([
      ellipse(91, 136, 54, 10, icon.dark, ' opacity="0.22"'),
      px(50, 80, 78, 20, icon.dark),
      px(59, 74, 58, 18, icon.main),
      px(115, 83, 30, 7, icon.metal),
      px(64, 98, 22, 29, icon.dark),
      px(69, 102, 15, 19, icon.main),
      ellipse(91, 83, 17, 17, icon.metal),
      ellipse(91, 83, 7, 7, icon.dark),
      px(61, 66, 45, 7, icon.accent),
      motifs.join('')
    ]);
  }
  if (icon.form === 'codex') {
    return box([
      ellipse(90, 136, 52, 10, icon.dark, ' opacity="0.22"'),
      polygon('48,50 86,40 86,122 48,131', icon.dark),
      polygon('88,40 132,51 132,132 88,122', icon.dark),
      polygon('57,55 84,49 84,112 57,119', icon.main),
      polygon('91,49 123,57 123,119 91,112', icon.main),
      px(84, 45, 8, 77, icon.metal),
      px(66, 72, 12, 5, icon.accent),
      px(101, 72, 14, 5, icon.accent),
      sparkle(90, 80, icon.light),
      motifs.join('')
    ]);
  }
  return box([
    ellipse(90, 136, 52, 10, icon.dark, ' opacity="0.22"'),
    polygon('92,28 121,108 91,124 63,108', icon.dark),
    polygon('91,38 109,101 91,112 73,101', icon.main),
    polygon('91,38 101,98 91,112', icon.light, ' opacity="0.68"'),
    px(84, 91, 15, 24, icon.accent, ' opacity="0.65"'),
    hilt.join(''),
    motifs.join('')
  ]);
}

function drawBossChestIcon(icon) {
  const torso = [
    polygon('60,50 120,50 136,122 45,122', icon.dark),
    polygon('67,59 113,59 125,113 57,113', icon.main),
    polygon('67,59 89,70 89,113 57,113', icon.light, ' opacity="0.28"'),
    px(86, 57, 9, 57, icon.metal),
    px(63, 91, 54, 8, icon.dark, ' opacity="0.38"')
  ];
  const motif = icon.form === 'gear'
    ? [
      ellipse(91, 77, 18, 18, icon.metal),
      ellipse(91, 77, 7, 7, icon.dark),
      px(72, 104, 36, 7, icon.accent)
    ]
    : icon.form === 'mantle'
      ? [
        polygon('50,50 26,109 54,118 69,62', icon.light, ' opacity="0.68"'),
        polygon('130,50 153,109 126,118 111,62', icon.light, ' opacity="0.68"'),
        polygon('78,61 90,88 102,61', icon.accent)
      ]
      : icon.form === 'robe'
        ? [
          polygon('70,62 90,131 110,62 126,132 54,132', icon.dark, ' opacity="0.72"'),
          px(74, 84, 33, 6, icon.accent),
          sparkle(91, 102, icon.light)
        ]
        : icon.form === 'eclipse'
          ? [
            ellipse(91, 80, 21, 21, icon.light),
            ellipse(99, 75, 20, 20, icon.dark),
            px(70, 104, 40, 7, icon.accent)
          ]
          : icon.form === 'stone'
            ? [
              px(64, 64, 22, 19, icon.metal),
              px(95, 63, 20, 23, icon.light, ' opacity="0.5"'),
              px(74, 99, 32, 8, icon.accent)
            ]
            : icon.form === 'plate'
              ? [
                polygon('72,63 91,44 110,63 91,77', icon.accent),
                px(67, 101, 48, 8, icon.light, ' opacity="0.7"')
              ]
              : [
                px(73, 66, 9, 45, icon.accent),
                px(98, 66, 9, 45, icon.accent),
                polygon('69,58 90,43 111,58 90,72', icon.light, ' opacity="0.48"')
              ];
  return box([
    ellipse(90, 136, 56, 10, icon.dark, ' opacity="0.22"'),
    torso.join(''),
    motif.join(''),
    px(59, 52, 18, 12, '#ffffff', ' opacity="0.18"'),
    sparkle(131, 49, icon.accent),
    sparkle(46, 67, icon.light)
  ]);
}

function drawBossBootsIcon(icon) {
  const left = [
    px(53, 73, 28, 45, icon.dark),
    px(59, 66, 21, 46, icon.main),
    polygon('52,110 86,110 94,128 49,128', icon.dark),
    polygon('59,110 86,110 89,121 55,121', icon.main)
  ];
  const right = [
    px(99, 73, 28, 45, icon.dark),
    px(100, 66, 21, 46, icon.main),
    polygon('94,110 128,110 132,128 87,128', icon.dark),
    polygon('94,110 121,110 125,121 91,121', icon.main)
  ];
  const motif = icon.form === 'gear'
    ? [
      ellipse(76, 91, 10, 10, icon.metal),
      ellipse(106, 91, 10, 10, icon.metal),
      px(76, 52, 29, 8, icon.accent)
    ]
    : icon.form === 'wing'
      ? [
        polygon('48,72 22,85 48,94', icon.light, ' opacity="0.75"'),
        polygon('132,72 158,85 132,94', icon.light, ' opacity="0.75"'),
        polygon('80,51 69,82 84,77 75,105 101,66 88,72 96,51', icon.accent)
      ]
      : icon.form === 'eclipse'
        ? [
          ellipse(90, 59, 17, 17, icon.light),
          ellipse(97, 55, 16, 16, icon.dark),
          px(66, 95, 16, 6, icon.accent),
          px(99, 95, 16, 6, icon.accent)
        ]
        : icon.form === 'flame'
          ? [
            polygon('71,47 61,78 78,70 70,97 93,61 81,65 88,43', icon.accent),
            polygon('107,47 98,78 114,70 106,97 129,61 117,65 124,43', icon.light, ' opacity="0.78"')
          ]
          : icon.form === 'rune'
            ? [
              px(67, 90, 15, 5, icon.accent),
              px(72, 84, 5, 17, icon.accent),
              px(99, 90, 15, 5, icon.accent),
              px(104, 84, 5, 17, icon.accent),
              sparkle(90, 61, icon.light)
            ]
            : icon.form === 'stone'
              ? [
                px(61, 80, 20, 16, icon.metal),
                px(100, 81, 18, 15, icon.light, ' opacity="0.55"'),
                px(62, 108, 20, 7, icon.accent),
                px(99, 108, 20, 7, icon.accent)
              ]
              : [
                polygon('62,53 74,73 56,74', icon.accent),
                polygon('107,53 123,74 104,73', icon.accent),
                px(66, 96, 15, 6, icon.light, ' opacity="0.62"'),
                px(99, 96, 15, 6, icon.light, ' opacity="0.62"')
              ];
  return box([
    ellipse(90, 136, 56, 10, icon.dark, ' opacity="0.22"'),
    left.join(''),
    right.join(''),
    motif.join(''),
    sparkle(130, 57, icon.accent),
    sparkle(47, 63, icon.light)
  ]);
}

function drawBossHeadIcon(icon) {
  const motif = icon.form === 'mask'
    ? [
      polygon('89,39 132,70 118,115 89,135 60,115 46,70', icon.dark),
      polygon('89,51 119,75 109,107 89,122 69,107 59,75', icon.main),
      polygon('65,78 85,87 78,98 63,96', icon.light, ' opacity="0.82"'),
      polygon('113,78 93,87 100,98 115,96', icon.light, ' opacity="0.82"'),
      polygon('89,53 99,90 89,113 79,90', icon.accent, ' opacity="0.55"'),
      px(70, 65, 12, 7, icon.accent),
      px(98, 65, 12, 7, icon.accent)
    ]
    : icon.form === 'visor'
      ? [
        px(51, 66, 78, 45, icon.dark),
        px(58, 59, 64, 47, icon.main),
        px(66, 69, 48, 13, icon.accent),
        px(72, 73, 36, 5, icon.light),
        px(62, 51, 13, 13, icon.metal),
        px(83, 47, 14, 14, icon.metal),
        px(105, 51, 13, 13, icon.metal),
        px(72, 98, 36, 11, icon.dark, ' opacity="0.45"')
      ]
      : [
        polygon('47,76 68,48 89,71 111,48 133,76 124,119 55,119', icon.dark),
        polygon('57,77 71,59 89,80 107,59 122,77 115,110 64,110', icon.main),
        polygon('71,59 83,80 63,88', icon.light, ' opacity="0.54"'),
        polygon('107,59 97,80 117,88', icon.light, ' opacity="0.54"'),
        px(62, 108, 54, 13, icon.metal),
        px(68, 102, 42, 8, icon.light),
        px(51, 72, 12, 11, icon.accent),
        px(84, 54, 11, 14, icon.accent),
        px(118, 72, 12, 11, icon.accent)
      ];
  return box([
    ellipse(90, 136, 52, 10, icon.dark, ' opacity="0.22"'),
    motif.join(''),
    px(77, 92, 10, 7, icon.dark, ' opacity="0.5"'),
    px(94, 92, 10, 7, icon.dark, ' opacity="0.5"'),
    sparkle(128, 48, icon.accent),
    sparkle(48, 62, icon.light)
  ]);
}

function drawBossGlovesIcon(icon) {
  const left = [
    px(43, 82, 35, 36, icon.dark),
    px(49, 73, 29, 40, icon.main),
    px(57, 65, 17, 14, icon.light),
    px(47, 112, 35, 11, icon.dark),
    px(53, 116, 11, 13, icon.main),
    px(67, 115, 11, 12, icon.main)
  ];
  const right = [
    px(101, 82, 35, 36, icon.dark),
    px(102, 73, 29, 40, icon.main),
    px(106, 65, 17, 14, icon.light),
    px(98, 112, 35, 11, icon.dark),
    px(102, 115, 11, 12, icon.main),
    px(116, 116, 11, 13, icon.main)
  ];
  const motif = icon.form === 'gear'
    ? [
      px(62, 86, 13, 13, icon.metal),
      px(105, 86, 13, 13, icon.metal),
      px(66, 90, 5, 5, icon.dark),
      px(109, 90, 5, 5, icon.dark),
      px(79, 76, 22, 8, icon.accent)
    ]
    : icon.form === 'spark'
      ? [
        polygon('80,54 69,85 84,80 74,112 101,71 87,76 96,54', icon.accent),
        px(61, 91, 18, 6, icon.light),
        px(103, 91, 18, 6, icon.light)
      ]
      : icon.form === 'eclipse'
        ? [
          ellipse(90, 75, 19, 19, icon.light),
          ellipse(96, 70, 18, 18, icon.dark),
          px(63, 89, 13, 6, icon.accent),
          px(104, 89, 13, 6, icon.accent)
        ]
        : icon.form === 'rune'
          ? [
            px(62, 88, 15, 5, icon.accent),
            px(67, 82, 5, 17, icon.accent),
            px(104, 88, 15, 5, icon.accent),
            px(109, 82, 5, 17, icon.accent),
            sparkle(90, 63, icon.light)
          ]
          : icon.form === 'claw'
            ? [
              polygon('53,61 61,80 45,80', icon.accent),
              polygon('66,58 73,80 58,79', icon.accent),
              polygon('108,58 115,79 100,80', icon.accent),
              polygon('121,61 135,80 119,80', icon.accent)
            ]
            : [
              px(61, 87, 15, 8, icon.metal),
              px(104, 87, 15, 8, icon.metal),
              px(81, 69, 18, 18, icon.accent, ' opacity="0.85"')
            ];
  return box([
    ellipse(90, 136, 54, 10, icon.dark, ' opacity="0.22"'),
    left.join(''),
    right.join(''),
    motif.join(''),
    px(56, 78, 16, 8, '#ffffff', ' opacity="0.24"'),
    px(108, 78, 16, 8, '#ffffff', ' opacity="0.24"'),
    sparkle(130, 55, icon.accent),
    sparkle(46, 60, icon.light)
  ]);
}

function drawDropIcon(icon) {
  if (EQUIPMENT_ICON_KINDS.includes(icon.kind)) return drawEquipmentLayer(icon, 'skill', 2);
  if (icon.kind === 'coins') return drawCoinIcon(icon);
  if (icon.kind === 'potion' || icon.kind === 'tonic') return drawPotionIcon(icon);
  if (icon.kind === 'scroll') return drawScrollIcon(icon);
  if (icon.kind === 'ration') return drawRationIcon(icon);
  if (icon.kind === 'guardTonic') return drawGuardTonicIcon(icon);
  if (icon.kind === 'oil') return drawOilIcon(icon);
  if (icon.kind === 'magnet') return drawMagnetIcon(icon);
  if (icon.kind === 'whistle') return drawWhistleIcon(icon);
  if (icon.kind === 'attunementPrism') return drawAttunementPrismIcon(icon);
  if (icon.kind === 'echoPrism') return drawEchoPrismIcon(icon);
  if (icon.kind === 'prismShard') return drawPrismShardIcon(icon);
  if (icon.kind === 'manual') return drawManualIcon(icon);
  if (icon.kind === 'resetScroll') return drawResetScrollIcon(icon);
  if (icon.kind === 'wardingScroll') return drawWardingScrollIcon(icon);
  if (icon.kind === 'refinementCore') return drawRefinementCoreIcon(icon);
  if (icon.kind === 'seal') return drawSealIcon(icon);
  if (icon.kind === 'coupon') return drawCouponIcon(icon);
  if (icon.kind === 'dust') return drawDustIcon(icon);
  if (icon.kind === 'catalyst') return drawCatalystIcon(icon);
  if (icon.kind === 'fracture') return drawFractureIcon(icon);
  if (icon.kind === 'gel') return drawGelIcon(icon);
  if (icon.kind === 'ore') return drawOreIcon(icon);
  if (icon.kind === 'bossWeapon') return drawBossWeaponIcon(icon);
  if (icon.kind === 'bossChest') return drawBossChestIcon(icon);
  if (icon.kind === 'bossHead') return drawBossHeadIcon(icon);
  if (icon.kind === 'bossGloves') return drawBossGlovesIcon(icon);
  if (icon.kind === 'bossBoots') return drawBossBootsIcon(icon);
  return px(64, 64, 52, 52, icon.main || '#eef4f7');
}

function enemyPose(enemy, row, frame) {
  const two = frame % 2;
  const wave = [-1, 0, 1, 0, -1, 0][frame % 6];
  const poseData = {
    x: 0,
    y: 0,
    leg: two ? 5 : -5,
    reach: 0,
    cast: 0,
    pulse: frame % 6,
    collapse: 0,
    alpha: 1
  };
  if (row === 'idle') {
    poseData.y = wave;
  } else if (row === 'move') {
    poseData.x = two ? 5 : -4;
    poseData.y = enemy.kind === 'slime' || enemy.kind === 'wisp' ? (two ? -7 : 3) : (two ? 1 : -1);
    poseData.leg = two ? 8 : -8;
  } else if (row === 'telegraph') {
    poseData.x = two ? -3 : 2;
    poseData.y = two ? 2 : -1;
    poseData.cast = frame + 1;
  } else if (row === 'attack') {
    poseData.x = two ? 12 : -6;
    poseData.y = two ? -1 : 2;
    poseData.reach = two ? 18 : -6;
    poseData.leg = two ? 10 : -6;
  } else if (row === 'projectile') {
    poseData.x = two ? 4 : -3;
    poseData.y = two ? -3 : 1;
    poseData.cast = 3 + frame;
    poseData.reach = two ? 12 : 2;
  } else if (row === 'buff') {
    poseData.y = [2, 0, -3, -5, -2, 0][frame % 6];
    poseData.cast = 6 + frame;
  } else if (row === 'hit') {
    poseData.x = [-8, -10, -6, -2, 2, 0][frame % 6];
    poseData.y = two ? 1 : 0;
  } else if (row === 'defeat') {
    poseData.x = two ? -2 : 1;
    poseData.y = 12 + frame * 3;
    poseData.collapse = frame + 1;
    poseData.alpha = Math.max(0.18, 1 - frame * 0.12);
  }
  return poseData;
}

function enemyShadow(width, y, opacity) {
  return px(Math.round((160 - width) / 2), y, width, 8, '#0e2636', ` opacity="${opacity || 0.2}"`);
}

function enemyAura(enemy, p, size) {
  const pulse = Number(size || 48) + p.pulse * 3;
  const x = Math.round(80 - pulse / 2 + p.x);
  const y = Math.round(76 - pulse / 2 + p.y);
  return box([
    px(x, y, pulse, pulse, enemy.accent, ' opacity="0.12"'),
    px(x + 8, y + 8, Math.max(8, pulse - 16), Math.max(8, pulse - 16), enemy.light, ' opacity="0.12"')
  ]);
}

function enemyCastSpark(enemy, p, x, y) {
  return box([
    px(x + p.x, y + p.y - p.cast, 8, 8, enemy.accent, ' opacity="0.82"'),
    px(x + 11 + p.x, y + 4 + p.y - Math.floor(p.cast / 2), 5, 5, enemy.light, ' opacity="0.78"')
  ]);
}

function enemyDefeatBits(enemy, p) {
  if (!p.collapse) return '';
  const offset = p.collapse * 2;
  return box([
    px(48 - offset, 134, 8, 7, enemy.dark, ` opacity="${p.alpha}"`),
    px(101 + offset, 132, 9, 8, enemy.main, ` opacity="${p.alpha}"`),
    px(72, 142 - offset, 10, 6, enemy.light, ` opacity="${p.alpha}"`),
    px(88, 146 - offset, 7, 5, enemy.accent, ` opacity="${p.alpha}"`)
  ]);
}

function drawSlimeEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y + (p.collapse ? p.collapse * 3 : 0);
  const low = p.collapse ? p.collapse * 3 : 0;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 64) : '',
    enemyShadow(68, 139, 0.22),
    px(56 + x, 108 + y + low, 48, Math.max(12, 28 - low), enemy.dark, ` opacity="${p.alpha}"`),
    px(51 + x, 116 + y + low, 60, Math.max(10, 24 - low), enemy.main, ` opacity="${p.alpha}"`),
    px(64 + x, 99 + y + low, 30, Math.max(8, 19 - low), enemy.light, ` opacity="${p.alpha}"`),
    px(75 + x + p.reach, 112 + y, 8, 6, enemy.accent, ` opacity="${p.alpha}"`),
    px(89 + x + p.reach, 112 + y, 6, 6, enemy.dark, ` opacity="${p.alpha}"`),
    row === 'attack' ? px(104 + x, 118 + y, 24, 8, enemy.light, ' opacity="0.78"') : '',
    row === 'projectile' || row === 'telegraph' ? enemyCastSpark(enemy, p, 108, 111) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawBeastEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  const front = p.leg;
  const back = -p.leg;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 70) : '',
    enemyShadow(82, 142, 0.2),
    px(48 + x, 92 + y, 60, 32, enemy.dark, ` opacity="${p.alpha}"`),
    px(52 + x, 86 + y, 54, 31, enemy.main, ` opacity="${p.alpha}"`),
    px(63 + x, 80 + y, 38, 13, enemy.light, ` opacity="${p.alpha}"`),
    px(102 + x + p.reach, 83 + y, 27, 24, enemy.main, ` opacity="${p.alpha}"`),
    px(122 + x + p.reach, 93 + y, 5, 5, enemy.dark, ` opacity="${p.alpha}"`),
    px(48 + x + back, 117 + y, 9, 20, enemy.dark, ` opacity="${p.alpha}"`),
    px(66 + x + front, 116 + y, 9, 21, enemy.main, ` opacity="${p.alpha}"`),
    px(91 + x + back, 116 + y, 9, 21, enemy.dark, ` opacity="${p.alpha}"`),
    px(111 + x + front, 111 + y, 8, 25, enemy.main, ` opacity="${p.alpha}"`),
    row === 'attack' ? px(127 + x + p.reach, 98 + y, 20, 7, enemy.accent) : '',
    row === 'projectile' || row === 'telegraph' ? enemyCastSpark(enemy, p, 122, 78) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawPlantEnemy(enemy, row, frame, p) {
  const x = p.x + (frame % 2 ? 2 : -1);
  const y = p.y;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 78) : '',
    enemyShadow(58, 143, 0.18),
    px(74 + x, 94 + y, 14, 45, enemy.dark, ` opacity="${p.alpha}"`),
    px(78 + x, 89 + y, 11, 48, enemy.main, ` opacity="${p.alpha}"`),
    px(58 + x, 72 + y, 48, 24, enemy.dark, ` opacity="${p.alpha}"`),
    px(62 + x, 66 + y, 41, 26, enemy.main, ` opacity="${p.alpha}"`),
    px(70 + x, 61 + y, 22, 14, enemy.light, ` opacity="${p.alpha}"`),
    px(54 + x, 76 + y, 8, 8, enemy.accent, ` opacity="${p.alpha}"`),
    px(102 + x, 76 + y, 8, 8, enemy.accent, ` opacity="${p.alpha}"`),
    px(60 + x, 115 + y, 22, 8, enemy.light, ` opacity="${p.alpha}"`),
    px(87 + x, 118 + y, 24, 8, enemy.light, ` opacity="${p.alpha}"`),
    row === 'attack' || row === 'projectile' ? px(109 + x + p.reach, 79 + y, 29, 5, enemy.accent) : '',
    row === 'telegraph' ? enemyCastSpark(enemy, p, 80, 55) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawBrambleBossEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  const crownLift = row === 'telegraph' || row === 'buff' ? -5 : 0;
  return box([
    row === 'buff' || row === 'telegraph' ? enemyAura(enemy, p, 116) : '',
    enemyShadow(112, 146, 0.25),
    px(40 + x - p.leg / 2, 123 + y, 21, 18, enemy.dark, ` opacity="${p.alpha}"`),
    px(96 + x + p.leg / 2, 122 + y, 23, 19, enemy.dark, ` opacity="${p.alpha}"`),
    px(30 + x - p.reach / 3, 116 + y, 35, 10, enemy.main, ` opacity="${p.alpha}"`),
    px(95 + x + p.reach, 112 + y, 42, 10, enemy.main, ` opacity="${p.alpha}"`),
    px(53 + x, 76 + y, 58, 57, enemy.dark, ` opacity="${p.alpha}"`),
    px(60 + x, 71 + y, 46, 55, enemy.main, ` opacity="${p.alpha}"`),
    px(69 + x, 61 + y + crownLift, 30, 22, enemy.light, ` opacity="${p.alpha}"`),
    polygon(`${57 + x},70 ${72 + x},45 ${83 + x},72`, enemy.dark, ` opacity="${p.alpha}"`),
    polygon(`${80 + x},66 ${94 + x},39 ${105 + x},73`, enemy.dark, ` opacity="${p.alpha}"`),
    px(61 + x, 53 + y + crownLift, 13, 8, enemy.accent, ` opacity="${p.alpha}"`),
    px(94 + x, 50 + y + crownLift, 12, 8, enemy.accent, ` opacity="${p.alpha}"`),
    px(70 + x, 91 + y, 8, 8, enemy.dark, ` opacity="${p.alpha}"`),
    px(91 + x, 91 + y, 8, 8, enemy.dark, ` opacity="${p.alpha}"`),
    px(57 + x, 130 + y, 55, 8, enemy.light, ` opacity="${p.alpha}"`),
    row === 'attack' || row === 'projectile' ? px(122 + x + p.reach, 90 + y, 31, 6, enemy.accent, ' opacity="0.88"') : '',
    row === 'projectile' ? px(139 + x + p.reach, 83 + y, 9, 18, enemy.accent, ' opacity="0.88"') : '',
    row === 'telegraph' || row === 'buff' ? enemyCastSpark(enemy, p, 82, 48) + enemyCastSpark(enemy, p, 118, 76) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawBoarEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 74) : '',
    enemyShadow(88, 143, 0.22),
    px(45 + x, 91 + y, 65, 34, enemy.dark, ` opacity="${p.alpha}"`),
    px(49 + x, 85 + y, 59, 34, enemy.main, ` opacity="${p.alpha}"`),
    px(62 + x, 78 + y, 38, 9, enemy.light, ` opacity="${p.alpha}"`),
    px(102 + x + p.reach, 88 + y, 31, 26, enemy.main, ` opacity="${p.alpha}"`),
    px(127 + x + p.reach, 100 + y, 17, 5, enemy.accent, ` opacity="${p.alpha}"`),
    px(121 + x + p.reach, 92 + y, 5, 5, enemy.dark, ` opacity="${p.alpha}"`),
    px(53 + x - p.leg, 116 + y, 9, 22, enemy.dark, ` opacity="${p.alpha}"`),
    px(74 + x + p.leg, 116 + y, 9, 22, enemy.main, ` opacity="${p.alpha}"`),
    px(99 + x - p.leg, 115 + y, 9, 23, enemy.dark, ` opacity="${p.alpha}"`),
    px(115 + x + p.leg, 113 + y, 9, 25, enemy.main, ` opacity="${p.alpha}"`),
    row === 'attack' ? px(136 + x, 100 + y, 17, 5, '#fff0cf') : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawImpEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  const arm = p.reach;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 66) : '',
    enemyShadow(62, 143, 0.2),
    px(66 + x, 73 + y, 30, 43, enemy.dark, ` opacity="${p.alpha}"`),
    px(69 + x, 76 + y, 24, 36, enemy.main, ` opacity="${p.alpha}"`),
    px(69 + x, 51 + y, 26, 24, enemy.dark, ` opacity="${p.alpha}"`),
    px(72 + x, 56 + y, 20, 16, enemy.light, ` opacity="${p.alpha}"`),
    px(63 + x, 49 + y, 8, 8, enemy.dark, ` opacity="${p.alpha}"`),
    px(91 + x, 49 + y, 8, 8, enemy.dark, ` opacity="${p.alpha}"`),
    px(94 + x + arm, 82 + y, 24, 8, enemy.main, ` opacity="${p.alpha}"`),
    px(112 + x + arm, 77 + y, 24, 5, enemy.accent, ` opacity="${p.alpha}"`),
    px(61 + x - arm / 2, 83 + y, 15, 8, enemy.dark, ` opacity="${p.alpha}"`),
    px(68 + x - p.leg, 111 + y, 9, 28, enemy.dark, ` opacity="${p.alpha}"`),
    px(84 + x + p.leg, 111 + y, 9, 28, enemy.main, ` opacity="${p.alpha}"`),
    row === 'projectile' ? enemyCastSpark(enemy, p, 118, 75) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawBugEnemy(enemy, row, frame, p, beetle) {
  const x = p.x;
  const y = p.y;
  const shell = beetle ? 72 : 60;
  const shellY = beetle ? 85 : 91;
  return box([
    row === 'buff' ? enemyAura(enemy, p, beetle ? 78 : 66) : '',
    enemyShadow(beetle ? 90 : 74, 143, 0.22),
    px(44 + x, shellY + y, shell, 32, enemy.dark, ` opacity="${p.alpha}"`),
    px(49 + x, shellY - 6 + y, shell - 8, 33, enemy.main, ` opacity="${p.alpha}"`),
    px(58 + x, shellY - 12 + y, shell - 26, 14, enemy.light, ` opacity="${p.alpha}"`),
    px(102 + x + p.reach, shellY + y, 22, 18, enemy.main, ` opacity="${p.alpha}"`),
    px(116 + x + p.reach, shellY + 6 + y, 5, 5, enemy.accent, ` opacity="${p.alpha}"`),
    px(51 + x - p.leg, 117 + y, 16, 5, enemy.dark, ` opacity="${p.alpha}"`),
    px(68 + x + p.leg, 120 + y, 16, 5, enemy.dark, ` opacity="${p.alpha}"`),
    px(86 + x - p.leg, 117 + y, 16, 5, enemy.dark, ` opacity="${p.alpha}"`),
    px(104 + x + p.leg, 120 + y, 16, 5, enemy.dark, ` opacity="${p.alpha}"`),
    row === 'attack' ? px(123 + x + p.reach, shellY + 8 + y, 24, 6, enemy.accent) : '',
    row === 'projectile' || row === 'telegraph' ? enemyCastSpark(enemy, p, 122, shellY - 6) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawWispEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y - 8;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 84) : '',
    enemyShadow(52, 143, 0.14),
    px(69 + x, 70 + y, 25, 55, enemy.dark, ` opacity="${p.alpha}"`),
    px(62 + x, 86 + y, 39, 38, enemy.main, ` opacity="${p.alpha}"`),
    px(70 + x, 75 + y, 23, 45, enemy.light, ` opacity="${p.alpha}"`),
    px(78 + x, 62 + y, 11, 21, enemy.accent, ` opacity="${p.alpha}"`),
    px(82 + x, 92 + y, 5, 5, enemy.dark, ` opacity="${p.alpha}"`),
    row === 'attack' || row === 'projectile' ? px(103 + x + p.reach, 91 + y, 25, 9, enemy.accent) : '',
    row === 'telegraph' ? enemyCastSpark(enemy, p, 87, 63) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawBanditEnemy(enemy, row, frame, p, ranged) {
  const x = p.x;
  const y = p.y;
  const arm = p.reach;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 66) : '',
    enemyShadow(64, 143, 0.2),
    px(66 + x, 74 + y, 31, 42, enemy.dark, ` opacity="${p.alpha}"`),
    px(69 + x, 77 + y, 25, 34, enemy.main, ` opacity="${p.alpha}"`),
    px(70 + x, 51 + y, 25, 25, enemy.dark, ` opacity="${p.alpha}"`),
    px(72 + x, 57 + y, 20, 15, enemy.light, ` opacity="${p.alpha}"`),
    px(73 + x, 52 + y, 19, 6, '#3c2a25', ` opacity="${p.alpha}"`),
    px(93 + x + arm, 82 + y, 23, 8, enemy.main, ` opacity="${p.alpha}"`),
    ranged
      ? px(111 + x + arm, 79 + y, 28, 4, enemy.accent, ` opacity="${p.alpha}"`)
      : px(111 + x + arm, 77 + y, 32, 5, enemy.accent, ` opacity="${p.alpha}"`),
    px(61 + x - arm / 2, 83 + y, 15, 8, enemy.dark, ` opacity="${p.alpha}"`),
    px(68 + x - p.leg, 111 + y, 9, 28, enemy.dark, ` opacity="${p.alpha}"`),
    px(84 + x + p.leg, 111 + y, 9, 28, enemy.main, ` opacity="${p.alpha}"`),
    row === 'projectile' ? px(138 + x + arm, 78 + y, 12, 4, enemy.light) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawHealerEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 92) : '',
    enemyShadow(66, 143, 0.16),
    px(66 + x, 87 + y, 32, 49, enemy.dark, ` opacity="${p.alpha}"`),
    px(70 + x, 91 + y, 24, 43, enemy.main, ` opacity="${p.alpha}"`),
    px(50 + x, 62 + y, 62, 25, enemy.dark, ` opacity="${p.alpha}"`),
    px(55 + x, 56 + y, 52, 27, enemy.light, ` opacity="${p.alpha}"`),
    px(65 + x, 51 + y, 32, 13, enemy.main, ` opacity="${p.alpha}"`),
    px(60 + x, 64 + y, 8, 7, enemy.accent, ` opacity="${p.alpha}"`),
    px(88 + x, 58 + y, 8, 7, enemy.accent, ` opacity="${p.alpha}"`),
    px(80 + x, 101 + y, 6, 6, enemy.dark, ` opacity="${p.alpha}"`),
    row === 'buff' || row === 'projectile' ? enemyCastSpark(enemy, p, 104, 64) + enemyCastSpark(enemy, p, 52, 75) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawMimicEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  const open = row === 'attack' || row === 'projectile' || row === 'telegraph' || row === 'buff';
  return box([
    row === 'buff' ? enemyAura(enemy, p, 72) : '',
    enemyShadow(82, 143, 0.22),
    px(50 + x, 91 + y, 68, 43, enemy.dark, ` opacity="${p.alpha}"`),
    px(55 + x, 96 + y, 58, 33, enemy.main, ` opacity="${p.alpha}"`),
    px(51 + x, (open ? 75 : 84) + y, 66, 16, enemy.light, ` opacity="${p.alpha}"`),
    px(59 + x, (open ? 89 : 96) + y, 10, 9, '#fff2d0', ` opacity="${p.alpha}"`),
    px(77 + x, (open ? 89 : 96) + y, 10, 9, '#fff2d0', ` opacity="${p.alpha}"`),
    px(95 + x, (open ? 89 : 96) + y, 10, 9, '#fff2d0', ` opacity="${p.alpha}"`),
    px(63 + x, 107 + y, 38, 8, enemy.accent, open ? '' : ' opacity="0"'),
    px(59 + x - p.leg, 128 + y, 12, 12, enemy.dark, ` opacity="${p.alpha}"`),
    px(99 + x + p.leg, 128 + y, 12, 12, enemy.dark, ` opacity="${p.alpha}"`),
    row === 'attack' ? px(116 + x + p.reach, 101 + y, 28, 8, enemy.accent) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawGolemEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  return box([
    row === 'buff' ? enemyAura(enemy, p, 104) : '',
    enemyShadow(104, 146, 0.26),
    px(57 + x, 68 + y, 55, 59, enemy.dark, ` opacity="${p.alpha}"`),
    px(63 + x, 74 + y, 43, 47, enemy.main, ` opacity="${p.alpha}"`),
    px(68 + x, 48 + y, 34, 25, enemy.dark, ` opacity="${p.alpha}"`),
    px(72 + x, 54 + y, 26, 16, enemy.light, ` opacity="${p.alpha}"`),
    px(80 + x, 61 + y, 11, 6, enemy.accent, ` opacity="${p.alpha}"`),
    px(35 + x - p.reach / 2, 82 + y, 25, 33, enemy.dark, ` opacity="${p.alpha}"`),
    px(108 + x + p.reach, 80 + y, 27, 35, enemy.dark, ` opacity="${p.alpha}"`),
    px(39 + x - p.reach / 2, 112 + y, 21, 13, enemy.light, ` opacity="${p.alpha}"`),
    px(112 + x + p.reach, 112 + y, 22, 13, enemy.light, ` opacity="${p.alpha}"`),
    px(65 + x - p.leg, 124 + y, 17, 20, enemy.dark, ` opacity="${p.alpha}"`),
    px(90 + x + p.leg, 124 + y, 17, 20, enemy.dark, ` opacity="${p.alpha}"`),
    row === 'attack' ? px(103 + x + p.reach, 132 + y, 42, 8, enemy.accent, ' opacity="0.7"') : '',
    row === 'projectile' || row === 'telegraph' ? enemyCastSpark(enemy, p, 111, 73) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawClockworkTitanEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  const plateShift = row === 'telegraph' || row === 'buff' ? 3 : 0;
  return box([
    row === 'buff' || row === 'telegraph' ? enemyAura(enemy, p, 112) : '',
    enemyShadow(112, 146, 0.25),
    px(56 + x, 63 + y, 58, 63, enemy.dark, ` opacity="${p.alpha}"`),
    px(63 + x, 70 + y, 44, 49, enemy.main, ` opacity="${p.alpha}"`),
    px(68 + x, 43 + y, 35, 28, enemy.dark, ` opacity="${p.alpha}"`),
    px(73 + x, 49 + y, 25, 16, enemy.light, ` opacity="${p.alpha}"`),
    px(80 + x, 81 + y, 12, 12, enemy.accent, ` opacity="${p.alpha}"`),
    px(39 + x - p.reach / 2, 82 + y + plateShift, 23, 33, enemy.dark, ` opacity="${p.alpha}"`),
    px(109 + x + p.reach, 81 + y - plateShift, 25, 34, enemy.dark, ` opacity="${p.alpha}"`),
    px(43 + x - p.reach / 2, 111 + y, 19, 14, enemy.light, ` opacity="${p.alpha}"`),
    px(113 + x + p.reach, 111 + y, 20, 14, enemy.light, ` opacity="${p.alpha}"`),
    px(64 + x - p.leg, 123 + y, 17, 21, enemy.dark, ` opacity="${p.alpha}"`),
    px(92 + x + p.leg, 123 + y, 17, 21, enemy.dark, ` opacity="${p.alpha}"`),
    px(49 + x, 60 + y, 13, 13, enemy.light, ` opacity="${p.alpha}"`),
    px(111 + x, 58 + y, 15, 15, enemy.light, ` opacity="${p.alpha}"`),
    px(52 + x, 64 + y, 7, 7, enemy.dark, ` opacity="${p.alpha}"`),
    px(115 + x, 62 + y, 7, 7, enemy.dark, ` opacity="${p.alpha}"`),
    row === 'attack' ? px(103 + x + p.reach, 132 + y, 44, 8, enemy.accent, ' opacity="0.75"') : '',
    row === 'projectile' || row === 'telegraph' ? enemyCastSpark(enemy, p, 119, 73) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

function drawQuarryColossusEnemy(enemy, row, frame, p) {
  const x = p.x;
  const y = p.y;
  return box([
    row === 'buff' || row === 'telegraph' ? enemyAura(enemy, p, 116) : '',
    enemyShadow(116, 146, 0.27),
    polygon(`${55 + x},69 ${83 + x},51 ${113 + x},72 ${106 + x},126 ${60 + x},127`, enemy.dark, ` opacity="${p.alpha}"`),
    polygon(`${65 + x},75 ${84 + x},61 ${104 + x},78 ${99 + x},117 ${66 + x},119`, enemy.main, ` opacity="${p.alpha}"`),
    polygon(`${78 + x},55 ${94 + x},45 ${108 + x},61 ${98 + x},75 ${75 + x},70`, enemy.light, ` opacity="${p.alpha}"`),
    px(78 + x, 86 + y, 11, 9, enemy.accent, ` opacity="${p.alpha}"`),
    px(39 + x - p.reach / 2, 84 + y, 25, 35, enemy.dark, ` opacity="${p.alpha}"`),
    px(108 + x + p.reach, 83 + y, 28, 36, enemy.dark, ` opacity="${p.alpha}"`),
    polygon(`${38 + x - p.reach / 2},116 ${63 + x - p.reach / 2},112 ${57 + x - p.reach / 2},131 ${35 + x - p.reach / 2},130`, enemy.light, ` opacity="${p.alpha}"`),
    polygon(`${111 + x + p.reach},112 ${137 + x + p.reach},116 ${137 + x + p.reach},132 ${112 + x + p.reach},129`, enemy.light, ` opacity="${p.alpha}"`),
    px(61 + x - p.leg, 124 + y, 18, 21, enemy.dark, ` opacity="${p.alpha}"`),
    px(92 + x + p.leg, 124 + y, 18, 21, enemy.dark, ` opacity="${p.alpha}"`),
    px(66 + x, 101 + y, 36, 6, enemy.accent, ' opacity="0.58"'),
    row === 'attack' ? px(104 + x + p.reach, 134 + y, 45, 8, enemy.accent, ' opacity="0.74"') : '',
    row === 'projectile' || row === 'telegraph' ? enemyCastSpark(enemy, p, 118, 73) : '',
    enemyDefeatBits(enemy, p)
  ]);
}

const ENEMY_SPECIAL_PROFILES = Object.freeze({
  slimelet: Object.freeze({ base: 'slime', motif: 'gel', xWave: 1, yWave: 2, attackReach: 3, defeatScatter: 1 }),
  dewSlime: Object.freeze({ base: 'slime', motif: 'droplet', xWave: -1, yWave: 4, attackReach: 8, defeatScatter: 3 }),
  mossback: Object.freeze({ base: 'beast', motif: 'moss', xWave: 1, yWave: 1, attackReach: 2, defeatScatter: 2 }),
  thornSprout: Object.freeze({ base: 'plant', motif: 'thorn', xWave: -1, yWave: 1, attackReach: 6, defeatScatter: 3 }),
  vineSnapper: Object.freeze({ base: 'plant', motif: 'snapper', xWave: 2, yWave: 2, attackReach: 14, defeatScatter: 4 }),
  bristleBoar: Object.freeze({ base: 'boar', motif: 'bristles', xWave: 1, yWave: 1, attackReach: 7, legScale: 1.05, defeatScatter: 2 }),
  briarStag: Object.freeze({ base: 'boar', motif: 'antlers', xWave: 2, yWave: 2, attackReach: 13, legScale: 1.16, defeatScatter: 4 }),
  dustImp: Object.freeze({ base: 'imp', motif: 'dust', xWave: -2, yWave: 1, attackReach: 5, legScale: 1.12, defeatScatter: 3 }),
  clockbug: Object.freeze({ base: 'bug', motif: 'clock', xWave: 1, yWave: 1, attackReach: 3, defeatScatter: 2 }),
  rustRatchet: Object.freeze({ base: 'bug', motif: 'ratchet', xWave: 3, yWave: 1, attackReach: 7, legScale: 1.25, defeatScatter: 4 }),
  coilSentry: Object.freeze({ base: 'bug', motif: 'coil', xWave: 0, yWave: 0, attackReach: 11, legScale: 0.4, defeatScatter: 2 }),
  scrapWarden: Object.freeze({ base: 'banditMelee', motif: 'shield', xWave: 1, yWave: 1, attackReach: 4, legScale: 0.8, defeatScatter: 3 }),
  emberWisp: Object.freeze({ base: 'wisp', motif: 'ember', xWave: 2, yWave: 4, attackReach: 7, defeatScatter: 4 }),
  ashCrawler: Object.freeze({ base: 'beast', motif: 'ash', xWave: -1, yWave: 1, attackReach: 4, legScale: 0.75, defeatScatter: 3 }),
  lavaTick: Object.freeze({ base: 'bug', motif: 'lava', xWave: 4, yWave: 2, attackReach: 10, legScale: 1.35, defeatScatter: 5 }),
  cinderSpitter: Object.freeze({ base: 'imp', motif: 'cinder', xWave: 2, yWave: 2, attackReach: 12, legScale: 1.05, defeatScatter: 4 }),
  banditCutter: Object.freeze({ base: 'banditMelee', motif: 'blade', xWave: 1, yWave: 1, attackReach: 8, legScale: 1, defeatScatter: 2 }),
  banditThrower: Object.freeze({ base: 'banditRanged', motif: 'knife', xWave: -1, yWave: 1, attackReach: 10, legScale: 1, defeatScatter: 2 }),
  orebackBeetle: Object.freeze({ base: 'beetle', motif: 'ore', xWave: 0, yWave: 1, attackReach: 3, legScale: 0.65, defeatScatter: 2 }),
  glowcapHealer: Object.freeze({ base: 'healer', motif: 'glowcap', xWave: 1, yWave: 3, attackReach: 4, defeatScatter: 3 }),
  crackedMimic: Object.freeze({ base: 'mimic', motif: 'mimic', xWave: 2, yWave: 2, attackReach: 9, legScale: 1.15, defeatScatter: 5 }),
  brambleking: Object.freeze({ base: 'brambleBoss', motif: 'brambleCrown', xWave: 1, yWave: 1, attackReach: 8, legScale: 0.85, defeatScatter: 5 }),
  clockworkTitan: Object.freeze({ base: 'clockTitan', motif: 'titanGear', xWave: 1, yWave: 0, attackReach: 8, legScale: 0.75, defeatScatter: 4 }),
  quarryColossus: Object.freeze({ base: 'quarryBoss', motif: 'quarryCore', xWave: 0, yWave: 1, attackReach: 6, legScale: 0.7, defeatScatter: 5 }),
  emberjawGolem: Object.freeze({ base: 'golem', motif: 'jawCore', xWave: 1, yWave: 1, attackReach: 9, legScale: 0.85, defeatScatter: 5 }),
  frostlingScout: Object.freeze({ base: 'imp', motif: 'frostScout', xWave: 3, yWave: 2, attackReach: 7, legScale: 1.25, defeatScatter: 4 }),
  shardling: Object.freeze({ base: 'slime', motif: 'shard', xWave: -2, yWave: 5, attackReach: 5, defeatScatter: 5 }),
  rimebackBrute: Object.freeze({ base: 'beast', motif: 'rimeShell', xWave: 0, yWave: 1, attackReach: 4, legScale: 0.8, defeatScatter: 3 }),
  glacierSentinel: Object.freeze({ base: 'golem', motif: 'glacier', xWave: 0, yWave: 0, attackReach: 8, legScale: 0.55, defeatScatter: 4 }),
  snowglareWisp: Object.freeze({ base: 'wisp', motif: 'snowglare', xWave: -2, yWave: 5, attackReach: 6, defeatScatter: 4 }),
  icebloomOracle: Object.freeze({ base: 'healer', motif: 'icebloom', xWave: 1, yWave: 4, attackReach: 5, defeatScatter: 4 }),
  galeHarrier: Object.freeze({ base: 'wisp', motif: 'gale', xWave: 5, yWave: 4, attackReach: 10, defeatScatter: 4 }),
  stormboundArcher: Object.freeze({ base: 'banditRanged', motif: 'stormBow', xWave: 2, yWave: 1, attackReach: 13, legScale: 1.05, defeatScatter: 3 }),
  thunderRam: Object.freeze({ base: 'boar', motif: 'thunderHorn', xWave: 3, yWave: 2, attackReach: 15, legScale: 1.25, defeatScatter: 5 }),
  cloudcallAcolyte: Object.freeze({ base: 'healer', motif: 'cloudcall', xWave: -1, yWave: 4, attackReach: 5, defeatScatter: 4 }),
  indexScribe: Object.freeze({ base: 'banditRanged', motif: 'pages', xWave: 1, yWave: 2, attackReach: 12, legScale: 0.95, defeatScatter: 4 }),
  lumenSentinel: Object.freeze({ base: 'golem', motif: 'lumen', xWave: 0, yWave: 1, attackReach: 7, legScale: 0.65, defeatScatter: 4 }),
  voidMote: Object.freeze({ base: 'wisp', motif: 'void', xWave: 4, yWave: 5, attackReach: 11, defeatScatter: 5 }),
  eclipseDuelist: Object.freeze({ base: 'banditMelee', motif: 'eclipseBlade', xWave: 2, yWave: 1, attackReach: 12, legScale: 1.18, defeatScatter: 4 }),
  riftAberration: Object.freeze({ base: 'golem', motif: 'rift', xWave: 3, yWave: 2, attackReach: 14, legScale: 0.95, defeatScatter: 6 }),
  stormbreakRoc: Object.freeze({ base: 'wisp', motif: 'stormRoc', xWave: 5, yWave: 5, attackReach: 16, defeatScatter: 6 }),
  astralArchivist: Object.freeze({ base: 'banditRanged', motif: 'astralArchive', xWave: 1, yWave: 3, attackReach: 14, legScale: 0.95, defeatScatter: 5 }),
  eclipseSovereign: Object.freeze({ base: 'banditMelee', motif: 'eclipseCrown', xWave: 2, yWave: 2, attackReach: 16, legScale: 0.9, defeatScatter: 6 }),
  rimewarden: Object.freeze({ base: 'golem', motif: 'rimewarden', xWave: 1, yWave: 1, attackReach: 10, legScale: 0.7, defeatScatter: 5 })
});

function frameWave(frame, values) {
  return values[frame % values.length];
}

function specialEnemyPose(enemy, row, frame, profile) {
  const p = Object.assign({}, enemyPose(enemy, row, frame));
  const xWave = Number(profile.xWave || 0);
  const yWave = Number(profile.yWave || 0);
  p.x += frameWave(frame, [0, xWave, Math.round(xWave / 2), -xWave, 0, Math.round(-xWave / 2)]);
  p.y += frameWave(frame, [0, -yWave, Math.round(-yWave / 2), yWave, 0, Math.round(yWave / 2)]);
  if (row === 'move') {
    p.leg = Math.round(p.leg * Number(profile.legScale || 1));
    p.x += frameWave(frame, [0, 2, 4, -2, -4, 1]);
  }
  if (row === 'attack' || row === 'projectile') {
    p.reach += Number(profile.attackReach || 0);
    p.y += frameWave(frame, [0, -2, -4, -1, 2, 0]);
  }
  if (row === 'telegraph') {
    p.cast += Number(profile.attackReach || 0) / 2;
    p.x -= Math.round(Number(profile.attackReach || 0) / 4);
  }
  if (row === 'defeat') {
    p.collapse += Number(profile.defeatScatter || 0);
    p.x += frameWave(frame, [0, -3, 4, -6, 6, -2]);
  }
  return p;
}

function drawEnemyBaseKind(enemy, row, frame, p, baseKind) {
  if (baseKind === 'slime') return drawSlimeEnemy(enemy, row, frame, p);
  if (baseKind === 'beast') return drawBeastEnemy(enemy, row, frame, p);
  if (baseKind === 'plant') return drawPlantEnemy(enemy, row, frame, p);
  if (baseKind === 'boar') return drawBoarEnemy(enemy, row, frame, p);
  if (baseKind === 'imp') return drawImpEnemy(enemy, row, frame, p);
  if (baseKind === 'bug') return drawBugEnemy(enemy, row, frame, p, false);
  if (baseKind === 'beetle') return drawBugEnemy(enemy, row, frame, p, true);
  if (baseKind === 'wisp') return drawWispEnemy(enemy, row, frame, p);
  if (baseKind === 'banditMelee') return drawBanditEnemy(enemy, row, frame, p, false);
  if (baseKind === 'banditRanged') return drawBanditEnemy(enemy, row, frame, p, true);
  if (baseKind === 'healer') return drawHealerEnemy(enemy, row, frame, p);
  if (baseKind === 'mimic') return drawMimicEnemy(enemy, row, frame, p);
  if (baseKind === 'brambleBoss') return drawBrambleBossEnemy(enemy, row, frame, p);
  if (baseKind === 'clockTitan') return drawClockworkTitanEnemy(enemy, row, frame, p);
  if (baseKind === 'quarryBoss') return drawQuarryColossusEnemy(enemy, row, frame, p);
  if (baseKind === 'golem') return drawGolemEnemy(enemy, row, frame, p);
  throw new Error(`Missing enemy base renderer for ${baseKind}`);
}

function drawEnemyMotif(enemy, row, frame, p, profile) {
  const x = p.x;
  const y = p.y;
  const phase = frameWave(frame, [0, 2, 4, 2, 0, -2]);
  const attack = row === 'attack' || row === 'projectile' || row === 'telegraph';
  const defeat = row === 'defeat';
  const opacity = ` opacity="${p.alpha}"`;
  const spark = attack ? enemyCastSpark(enemy, p, 110 + p.reach, 72 + phase) : '';
  const scatter = defeat ? box([
    px(38 - phase, 132 - phase, 6, 6, enemy.accent, opacity),
    px(118 + phase, 135 - phase, 7, 5, enemy.light, opacity)
  ]) : '';
  if (profile.motif === 'gel') {
    return box([px(70 + x, 104 + y + phase, 23, 5, enemy.accent, opacity), attack ? px(103 + x + p.reach, 111 + y, 18, 6, enemy.light, ' opacity="0.78"') : '', scatter]);
  }
  if (profile.motif === 'droplet') {
    return box([px(76 + x, 90 + y - phase, 7, 13, enemy.accent, opacity), px(84 + x, 95 + y + phase, 5, 8, enemy.light, opacity), attack ? px(112 + x + p.reach, 104 + y, 9, 18, enemy.accent, ' opacity="0.8"') : '', scatter]);
  }
  if (profile.motif === 'moss') {
    return box([px(56 + x, 79 + y + phase, 18, 5, enemy.accent, opacity), px(82 + x, 77 + y - phase, 20, 5, enemy.accent, opacity), scatter]);
  }
  if (profile.motif === 'thorn') {
    return box([polygon(`${59 + x},73 ${70 + x},55 ${76 + x},77`, enemy.accent, opacity), polygon(`${99 + x},73 ${107 + x},57 ${113 + x},78`, enemy.accent, opacity), attack ? px(132 + x + p.reach, 78 + y, 18, 4, enemy.light, ' opacity="0.9"') : '', scatter]);
  }
  if (profile.motif === 'snapper') {
    return box([px(51 + x - phase, 84 + y, 13, 5, enemy.accent, opacity), px(101 + x + phase, 84 + y, 13, 5, enemy.accent, opacity), attack ? px(123 + x + p.reach, 72 + y, 26, 8, enemy.dark, ' opacity="0.9"') : '', scatter]);
  }
  if (profile.motif === 'bristles') {
    return box([px(60 + x, 75 + y, 5, 12, enemy.accent, opacity), px(75 + x, 72 + y - phase, 5, 14, enemy.accent, opacity), px(91 + x, 75 + y, 5, 12, enemy.accent, opacity), scatter]);
  }
  if (profile.motif === 'antlers') {
    return box([px(111 + x + p.reach, 76 + y - phase, 5, 17, enemy.accent, opacity), px(123 + x + p.reach, 77 + y + phase, 5, 16, enemy.accent, opacity), px(107 + x + p.reach, 75 + y, 25, 4, enemy.light, opacity), scatter]);
  }
  if (profile.motif === 'dust') {
    return box([px(43 + x - phase, 124 + y, 7, 7, enemy.accent, ' opacity="0.45"'), px(111 + x + phase, 128 + y, 5, 5, enemy.light, ' opacity="0.5"'), scatter]);
  }
  if (profile.motif === 'clock') {
    return box([px(72 + x, 85 + y, 21, 21, enemy.accent, ' opacity="0.45"'), px(82 + x, 89 + y, 4, 13, enemy.dark, opacity), px(82 + x, 95 + y, 12 + phase, 4, enemy.dark, opacity), scatter]);
  }
  if (profile.motif === 'ratchet') {
    return box([px(56 + x - phase, 82 + y, 9, 9, enemy.accent, opacity), px(88 + x + phase, 80 + y, 10, 10, enemy.light, opacity), px(113 + x + p.reach, 96 + y, 15, 6, enemy.accent, attack ? ' opacity="0.9"' : ' opacity="0.35"'), scatter]);
  }
  if (profile.motif === 'coil') {
    return box([px(70 + x, 69 + y, 28, 5, enemy.accent, opacity), px(73 + x, 79 + y + phase, 24, 5, enemy.light, opacity), px(76 + x, 89 + y - phase, 20, 5, enemy.accent, opacity), spark, scatter]);
  }
  if (profile.motif === 'shield') {
    return box([px(51 + x - phase, 77 + y, 16, 38, enemy.light, opacity), px(55 + x - phase, 82 + y, 8, 27, enemy.accent, opacity), attack ? px(123 + x + p.reach, 84 + y, 22, 6, enemy.accent, ' opacity="0.85"') : '', scatter]);
  }
  if (profile.motif === 'ember') {
    return box([px(75 + x, 55 + y - phase, 8, 20, enemy.accent, opacity), px(86 + x, 63 + y + phase, 6, 15, enemy.light, opacity), attack ? px(118 + x + p.reach, 88 + y, 22, 8, enemy.accent, ' opacity="0.75"') : '', scatter]);
  }
  if (profile.motif === 'ash') {
    return box([px(50 + x, 83 + y, 14, 6, enemy.accent, ' opacity="0.55"'), px(92 + x, 80 + y + phase, 16, 6, enemy.accent, ' opacity="0.5"'), scatter]);
  }
  if (profile.motif === 'lava') {
    return box([px(62 + x, 80 + y + phase, 38, 6, enemy.accent, opacity), px(75 + x, 91 + y - phase, 20, 5, enemy.light, opacity), attack ? px(127 + x + p.reach, 97 + y, 18, 8, enemy.light, ' opacity="0.92"') : '', scatter]);
  }
  if (profile.motif === 'cinder') {
    return box([px(70 + x, 45 + y - phase, 19, 9, enemy.accent, opacity), px(105 + x + p.reach, 74 + y, 30, 5, enemy.light, attack ? ' opacity="0.85"' : ' opacity="0.25"'), scatter]);
  }
  if (profile.motif === 'blade' || profile.motif === 'eclipseBlade') {
    const bladeColor = profile.motif === 'eclipseBlade' ? enemy.light : enemy.accent;
    return box([px(112 + x + p.reach, 72 + y - phase, 34, 5, bladeColor, opacity), px(108 + x + p.reach, 77 + y, 15, 4, enemy.dark, opacity), scatter]);
  }
  if (profile.motif === 'knife') {
    return box([px(117 + x + p.reach, 69 + y - phase, 22, 4, enemy.accent, opacity), px(134 + x + p.reach, 65 + y - phase, 8, 8, enemy.light, attack ? opacity : ' opacity="0.4"'), scatter]);
  }
  if (profile.motif === 'ore') {
    return box([polygon(`${70 + x},77 ${86 + x},66 ${102 + x},78 ${92 + x},94 ${76 + x},94`, enemy.accent, ' opacity="0.7"'), px(81 + x, 73 + y + phase, 10, 8, enemy.light, opacity), scatter]);
  }
  if (profile.motif === 'glowcap' || profile.motif === 'icebloom' || profile.motif === 'cloudcall') {
    return box([px(53 + x, 51 + y - phase, 54, 7, enemy.accent, opacity), px(74 + x, 38 + y - phase, 15, 15, enemy.light, ' opacity="0.68"'), spark, scatter]);
  }
  if (profile.motif === 'mimic') {
    return box([px(61 + x, 73 + y - phase, 43, 6, enemy.accent, opacity), px(74 + x + phase, 105 + y, 16, 8, enemy.light, attack ? opacity : ' opacity="0.45"'), scatter]);
  }
  if (profile.motif === 'brambleCrown') {
    return box([px(58 + x, 48 + y - phase, 54, 5, enemy.accent, opacity), polygon(`${70 + x},49 ${82 + x},31 ${91 + x},50`, enemy.light, opacity), scatter]);
  }
  if (profile.motif === 'titanGear') {
    return box([px(45 + x, 56 + y + phase, 16, 16, enemy.accent, opacity), px(112 + x, 55 + y - phase, 18, 18, enemy.light, opacity), spark, scatter]);
  }
  if (profile.motif === 'quarryCore' || profile.motif === 'jawCore' || profile.motif === 'lumen' || profile.motif === 'rimewarden') {
    return box([px(76 + x, 80 + y, 18, 18, enemy.accent, ' opacity="0.8"'), px(81 + x, 85 + y, 8 + phase, 8 + phase, enemy.light, opacity), spark, scatter]);
  }
  if (profile.motif === 'frostScout') {
    return box([px(67 + x, 48 + y - phase, 28, 5, enemy.accent, opacity), px(114 + x + p.reach, 77 + y, 19, 5, enemy.light, attack ? opacity : ' opacity="0.4"'), scatter]);
  }
  if (profile.motif === 'shard') {
    return box([polygon(`${77 + x},78 ${91 + x},54 ${101 + x},88 ${88 + x},101`, enemy.accent, opacity), polygon(`${61 + x},98 ${75 + x},78 ${82 + x},108`, enemy.light, opacity), scatter]);
  }
  if (profile.motif === 'rimeShell' || profile.motif === 'glacier') {
    return box([px(58 + x, 75 + y - phase, 45, 7, enemy.light, opacity), px(67 + x, 61 + y, 25, 11, enemy.accent, ' opacity="0.68"'), spark, scatter]);
  }
  if (profile.motif === 'snowglare') {
    return box([px(68 + x, 58 + y - phase, 26, 5, enemy.accent, opacity), px(78 + x, 48 + y + phase, 6, 27, enemy.light, opacity), scatter]);
  }
  if (profile.motif === 'gale') {
    return box([px(52 + x - phase, 80 + y, 38, 5, enemy.light, ' opacity="0.62"'), px(84 + x + phase, 69 + y, 42, 5, enemy.accent, ' opacity="0.56"'), scatter]);
  }
  if (profile.motif === 'stormRoc') {
    return box([
      polygon(`${48 + x - phase},78 ${22 + x - phase},58 ${60 + x},70`, enemy.light, ' opacity="0.62"'),
      polygon(`${108 + x + phase},78 ${142 + x + phase},58 ${96 + x},70`, enemy.light, ' opacity="0.62"'),
      px(70 + x, 52 + y - phase, 26, 5, enemy.accent, opacity),
      px(116 + x + p.reach, 75 + y, 28, 5, enemy.accent, attack ? opacity : ' opacity="0.42"'),
      spark,
      scatter
    ]);
  }
  if (profile.motif === 'stormBow') {
    return box([px(108 + x + p.reach, 65 + y - phase, 5, 34, enemy.accent, opacity), px(114 + x + p.reach, 80 + y, 28, 4, enemy.light, attack ? opacity : ' opacity="0.45"'), scatter]);
  }
  if (profile.motif === 'thunderHorn') {
    return box([px(116 + x + p.reach, 76 + y - phase, 24, 5, enemy.accent, opacity), px(132 + x + p.reach, 82 + y, 8, 15, enemy.light, attack ? opacity : ' opacity="0.55"'), spark, scatter]);
  }
  if (profile.motif === 'pages') {
    return box([px(52 + x - phase, 67 + y, 13, 18, enemy.light, ' opacity="0.78"'), px(116 + x + p.reach, 68 + y - phase, 15, 19, enemy.accent, attack ? opacity : ' opacity="0.48"'), scatter]);
  }
  if (profile.motif === 'astralArchive') {
    return box([
      px(45 + x - phase, 58 + y, 16, 22, enemy.light, ' opacity="0.74"'),
      px(64 + x, 50 + y - phase, 14, 20, enemy.accent, ' opacity="0.68"'),
      px(111 + x + p.reach, 62 + y + phase, 18, 23, enemy.light, attack ? opacity : ' opacity="0.5"'),
      px(133 + x + p.reach, 76 + y - phase, 14, 5, enemy.accent, attack ? ' opacity="0.9"' : ' opacity="0.34"'),
      spark,
      scatter
    ]);
  }
  if (profile.motif === 'void') {
    return box([px(68 + x, 62 + y - phase, 30, 30, enemy.dark, ' opacity="0.56"'), px(77 + x + phase, 71 + y, 12, 12, enemy.accent, opacity), spark, scatter]);
  }
  if (profile.motif === 'rift') {
    return box([px(48 + x - phase, 62 + y, 9, 70, enemy.accent, ' opacity="0.62"'), px(121 + x + phase, 58 + y, 10, 74, enemy.light, ' opacity="0.5"'), px(108 + x + p.reach, 83 + y, 35, 7, enemy.accent, attack ? opacity : ' opacity="0.25"'), scatter]);
  }
  if (profile.motif === 'eclipseCrown') {
    return box([
      px(58 + x, 44 + y - phase, 52, 5, enemy.light, opacity),
      polygon(`${70 + x},45 ${80 + x},28 ${90 + x},45`, enemy.accent, opacity),
      polygon(`${92 + x},45 ${104 + x},31 ${112 + x},47`, enemy.light, ' opacity="0.78"'),
      px(112 + x + p.reach, 72 + y - phase, 36, 6, enemy.light, attack ? opacity : ' opacity="0.5"'),
      px(114 + x + p.reach, 82 + y + phase, 34, 5, enemy.accent, attack ? ' opacity="0.88"' : ' opacity="0.34"'),
      scatter
    ]);
  }
  return box([spark, scatter]);
}

function drawSpecialEnemy(enemy, row, frame) {
  const profile = ENEMY_SPECIAL_PROFILES[enemy.id];
  if (!profile) throw new Error(`Missing custom enemy profile for ${enemy.id}`);
  const p = specialEnemyPose(enemy, row, frame, profile);
  return box([
    drawEnemyBaseKind(enemy, row, frame, p, profile.base),
    drawEnemyMotif(enemy, row, frame, p, profile)
  ]);
}

function drawSlimeletSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawDewSlimeSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawMossbackSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawThornSproutSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawVineSnapperSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawBristleBoarSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawBriarStagSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawDustImpSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawClockbugSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawRustRatchetSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawCoilSentrySpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawScrapWardenSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawEmberWispSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawAshCrawlerSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawLavaTickSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawCinderSpitterSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawBanditCutterSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawBanditThrowerSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawOrebackBeetleSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawGlowcapHealerSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawCrackedMimicSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawBramblekingSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawClockworkTitanSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawQuarryColossusSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawEmberjawGolemSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawFrostlingScoutSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawShardlingSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawRimebackBruteSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawGlacierSentinelSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawSnowglareWispSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawIcebloomOracleSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawGaleHarrierSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawStormboundArcherSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawThunderRamSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawCloudcallAcolyteSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawIndexScribeSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawLumenSentinelSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawVoidMoteSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawEclipseDuelistSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawRiftAberrationSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawStormbreakRocSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawAstralArchivistSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawEclipseSovereignSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }
function drawRimewardenSpecialEnemy(enemy, row, frame) { return drawSpecialEnemy(enemy, row, frame); }

const ENEMY_RENDERERS = Object.freeze({
  slimelet: drawSlimeletSpecialEnemy,
  dewSlime: drawDewSlimeSpecialEnemy,
  mossback: drawMossbackSpecialEnemy,
  thornSprout: drawThornSproutSpecialEnemy,
  vineSnapper: drawVineSnapperSpecialEnemy,
  bristleBoar: drawBristleBoarSpecialEnemy,
  briarStag: drawBriarStagSpecialEnemy,
  dustImp: drawDustImpSpecialEnemy,
  clockbug: drawClockbugSpecialEnemy,
  rustRatchet: drawRustRatchetSpecialEnemy,
  coilSentry: drawCoilSentrySpecialEnemy,
  scrapWarden: drawScrapWardenSpecialEnemy,
  emberWisp: drawEmberWispSpecialEnemy,
  ashCrawler: drawAshCrawlerSpecialEnemy,
  lavaTick: drawLavaTickSpecialEnemy,
  cinderSpitter: drawCinderSpitterSpecialEnemy,
  banditCutter: drawBanditCutterSpecialEnemy,
  banditThrower: drawBanditThrowerSpecialEnemy,
  orebackBeetle: drawOrebackBeetleSpecialEnemy,
  glowcapHealer: drawGlowcapHealerSpecialEnemy,
  crackedMimic: drawCrackedMimicSpecialEnemy,
  brambleking: drawBramblekingSpecialEnemy,
  clockworkTitan: drawClockworkTitanSpecialEnemy,
  quarryColossus: drawQuarryColossusSpecialEnemy,
  emberjawGolem: drawEmberjawGolemSpecialEnemy,
  frostlingScout: drawFrostlingScoutSpecialEnemy,
  shardling: drawShardlingSpecialEnemy,
  rimebackBrute: drawRimebackBruteSpecialEnemy,
  glacierSentinel: drawGlacierSentinelSpecialEnemy,
  snowglareWisp: drawSnowglareWispSpecialEnemy,
  icebloomOracle: drawIcebloomOracleSpecialEnemy,
  galeHarrier: drawGaleHarrierSpecialEnemy,
  stormboundArcher: drawStormboundArcherSpecialEnemy,
  thunderRam: drawThunderRamSpecialEnemy,
  cloudcallAcolyte: drawCloudcallAcolyteSpecialEnemy,
  indexScribe: drawIndexScribeSpecialEnemy,
  lumenSentinel: drawLumenSentinelSpecialEnemy,
  voidMote: drawVoidMoteSpecialEnemy,
  eclipseDuelist: drawEclipseDuelistSpecialEnemy,
  riftAberration: drawRiftAberrationSpecialEnemy,
  stormbreakRoc: drawStormbreakRocSpecialEnemy,
  astralArchivist: drawAstralArchivistSpecialEnemy,
  eclipseSovereign: drawEclipseSovereignSpecialEnemy,
  rimewarden: drawRimewardenSpecialEnemy
});

function drawEnemySprite(enemy, row, frame) {
  const renderer = ENEMY_RENDERERS[enemy.id];
  if (!renderer) throw new Error(`Missing custom enemy renderer for ${enemy.id}`);
  return renderer(enemy, row, frame);
}

function mapPlatform(x, y, width, height, topColor, sideColor) {
  return box([
    px(x, y, width, height, sideColor),
    px(x, y, width, 8, topColor),
    px(x, y + height - 5, width, 5, '#102033', ' opacity="0.24"')
  ]);
}

function drawBrambleDepthsMap(map) {
  const vines = Array.from({ length: 9 }, (_, index) => {
    const x = 80 + index * 142;
    const height = 170 + index % 3 * 36;
    return box([
      px(x, 0, 12, height, map.mid, ' opacity="0.82"'),
      px(x + 16, 18, 7, height - 24, map.far, ' opacity="0.72"'),
      px(x - 10, height - 18, 36, 8, map.light, ' opacity="0.46"')
    ]);
  }).join('');
  const thorns = Array.from({ length: 13 }, (_, index) => {
    const x = 18 + index * 99;
    return polygon(`${x},514 ${x + 16},480 ${x + 29},514`, index % 2 ? map.accent : map.mid, ' opacity="0.82"');
  }).join('');
  return box([
    px(0, 0, 1280, 640, map.sky),
    polygon('0,236 150,164 315,234 500,142 685,220 880,150 1075,230 1280,170 1280,420 0,420', map.far),
    polygon('0,324 140,260 290,318 460,238 620,322 780,250 980,320 1160,252 1280,320 1280,472 0,472', map.mid),
    px(0, 0, 1280, 82, '#142920', ' opacity="0.52"'),
    vines,
    mapPlatform(0, 520, 1280, 120, map.light, map.ground),
    mapPlatform(112, 442, 270, 28, map.light, '#283b2f'),
    mapPlatform(446, 368, 250, 28, map.light, '#283b2f'),
    mapPlatform(770, 296, 232, 28, map.light, '#283b2f'),
    mapPlatform(1040, 422, 230, 28, map.light, '#283b2f'),
    thorns,
    px(84, 468, 82, 10, map.accent, ' opacity="0.64"'),
    px(936, 338, 74, 10, map.accent, ' opacity="0.55"'),
    px(1128, 482, 70, 10, map.accent, ' opacity="0.58"')
  ]);
}

function drawGearworksVaultMap(map) {
  const columns = Array.from({ length: 7 }, (_, index) => {
    const x = 60 + index * 190;
    return box([
      px(x, 112, 34, 438, map.dark || '#272b31', ' opacity="0.44"'),
      px(x + 6, 128, 22, 390, map.mid, ' opacity="0.46"')
    ]);
  }).join('');
  const gears = Array.from({ length: 6 }, (_, index) => {
    const x = 130 + index * 202;
    const y = index % 2 ? 190 : 120;
    return box([
      px(x - 29, y - 5, 58, 10, map.light, ' opacity="0.56"'),
      px(x - 5, y - 29, 10, 58, map.light, ' opacity="0.56"'),
      ellipse(x, y, 31, 31, map.far, ' opacity="0.76"'),
      ellipse(x, y, 15, 15, map.sky, ' opacity="0.9"'),
      px(x - 7, y - 7, 14, 14, map.accent, ' opacity="0.58"')
    ]);
  }).join('');
  return box([
    px(0, 0, 1280, 640, map.sky),
    px(0, 92, 1280, 78, map.far, ' opacity="0.62"'),
    px(0, 170, 1280, 274, '#252a30', ' opacity="0.82"'),
    columns,
    gears,
    px(0, 454, 1280, 186, map.ground),
    mapPlatform(0, 520, 1280, 120, map.light, map.ground),
    mapPlatform(122, 446, 300, 28, map.light, '#4a4338'),
    mapPlatform(504, 374, 282, 28, map.light, '#4a4338'),
    mapPlatform(852, 302, 272, 28, map.light, '#4a4338'),
    mapPlatform(980, 448, 252, 28, map.light, '#4a4338'),
    px(0, 512, 1280, 6, map.accent, ' opacity="0.54"'),
    px(540, 252, 162, 10, map.accent, ' opacity="0.45"'),
    px(1040, 214, 116, 10, map.accent, ' opacity="0.42"')
  ]);
}

function drawEmberjawLairMap(map) {
  const vents = Array.from({ length: 8 }, (_, index) => {
    const x = 70 + index * 152;
    const top = 454 + index % 3 * 12;
    return box([
      polygon(`${x},520 ${x + 28},${top} ${x + 56},520`, map.accent, ' opacity="0.44"'),
      px(x + 22, top + 22, 12, 46, '#ffd36b', ' opacity="0.42"')
    ]);
  }).join('');
  return box([
    px(0, 0, 1280, 640, map.sky),
    polygon('0,256 158,148 324,258 512,130 704,252 880,154 1052,260 1280,134 1280,430 0,430', map.far),
    polygon('0,340 160,278 320,340 478,248 640,346 800,264 986,346 1135,262 1280,330 1280,470 0,470', map.mid),
    px(0, 382, 1280, 258, map.ground),
    vents,
    mapPlatform(0, 520, 1280, 120, map.light, map.ground),
    mapPlatform(156, 448, 300, 28, map.light, '#403033'),
    mapPlatform(540, 376, 262, 28, map.light, '#403033'),
    mapPlatform(882, 306, 248, 28, map.light, '#403033'),
    mapPlatform(952, 450, 260, 28, map.light, '#403033'),
    px(0, 512, 1280, 8, map.accent, ' opacity="0.62"'),
    px(220, 476, 110, 8, '#ffd36b', ' opacity="0.44"'),
    px(688, 406, 94, 8, '#ffd36b', ' opacity="0.42"'),
    px(1050, 334, 72, 8, '#ffd36b', ' opacity="0.4"')
  ]);
}

function drawFrostMap(map) {
  const peaks = Array.from({ length: 9 }, (_, index) => {
    const x = index * 160 - 60;
    const top = 132 + (index % 3) * 28;
    return polygon(`${x},372 ${x + 116},${top} ${x + 248},372`, index % 2 ? map.far : map.mid, ' opacity="0.68"');
  });
  const crystals = Array.from({ length: 8 }, (_, index) => {
    const x = 90 + index * 154;
    const y = 478 - (index % 2) * 74;
    return polygon(`${x},${y} ${x + 12},${y - 54} ${x + 28},${y} ${x + 12},${y + 22}`, index % 2 ? map.accent : map.light, ' opacity="0.62"');
  });
  return box([
    px(0, 0, 1280, 640, map.sky),
    polygon('0,254 135,174 320,254 510,156 700,250 890,174 1074,260 1280,156 1280,420 0,420', map.far),
    ...peaks,
    polygon('0,372 120,328 286,374 450,304 632,378 820,310 1000,376 1180,318 1280,362 1280,520 0,520', map.mid, ' opacity="0.72"'),
    px(0, 420, 1280, 220, map.ground),
    mapPlatform(0, 520, 1280, 120, map.light, map.ground),
    mapPlatform(126, 452, 300, 28, map.light, '#b7d8e7'),
    mapPlatform(510, 380, 270, 28, map.light, '#b7d8e7'),
    mapPlatform(870, 308, 250, 28, map.light, '#b7d8e7'),
    mapPlatform(974, 452, 244, 28, map.light, '#b7d8e7'),
    ...crystals,
    px(0, 512, 1280, 8, map.accent, ' opacity="0.34"')
  ]);
}

function drawBossRoomFocal(map) {
  const motif = map.motif || 'boss';
  if (motif === 'bramble') {
    return box([
      polygon('598,350 624,214 652,350', map.ground, ' opacity="0.72"'),
      polygon('564,354 638,170 718,354', map.mid, ' opacity="0.34"'),
      ellipse(640, 170, 68, 26, map.accent, ' opacity="0.38"'),
      px(606, 282, 72, 8, map.light, ' opacity="0.42"')
    ]);
  }
  if (motif === 'gear') {
    const teeth = Array.from({ length: 16 }, (_, index) => {
      const angle = index / 16 * Math.PI * 2;
      const x1 = 640 + Math.cos(angle) * 72;
      const y1 = 228 + Math.sin(angle) * 72;
      const x2 = 640 + Math.cos(angle) * 96;
      const y2 = 228 + Math.sin(angle) * 96;
      return px(Math.round((x1 + x2) / 2) - 5, Math.round((y1 + y2) / 2) - 5, 10, 10, map.light, ' opacity="0.52"');
    }).join('');
    return box([
      teeth,
      ellipse(640, 228, 82, 82, map.mid, ' opacity="0.42"'),
      ellipse(640, 228, 38, 38, map.ground, ' opacity="0.7"'),
      px(508, 254, 264, 12, map.accent, ' opacity="0.3"')
    ]);
  }
  if (motif === 'core') {
    return box([
      polygon('640,118 566,306 640,384 716,306', map.accent, ' opacity="0.34"'),
      polygon('640,160 598,298 640,350 684,298', map.light, ' opacity="0.32"'),
      ellipse(640, 340, 148, 34, map.accent, ' opacity="0.18"')
    ]);
  }
  if (motif === 'furnace') {
    return box([
      px(542, 142, 196, 206, map.ground, ' opacity="0.58"'),
      px(570, 178, 140, 132, map.mid, ' opacity="0.54"'),
      polygon('586,310 640,214 696,310', map.accent, ' opacity="0.42"'),
      polygon('610,310 646,246 682,310', map.light, ' opacity="0.34"')
    ]);
  }
  if (motif === 'rime') {
    return box([
      polygon('640,104 570,344 710,344', map.light, ' opacity="0.34"'),
      polygon('612,178 548,356 642,326', map.accent, ' opacity="0.2"'),
      polygon('668,178 638,326 738,356', map.accent, ' opacity="0.2"')
    ]);
  }
  if (motif === 'storm') {
    return box([
      polygon('644,86 586,230 648,206 608,356 738,170 666,194', map.accent, ' opacity="0.42"'),
      ellipse(640, 356, 168, 30, map.light, ' opacity="0.18"'),
      px(526, 300, 228, 8, map.light, ' opacity="0.28"')
    ]);
  }
  if (motif === 'astral') {
    return box([
      px(556, 146, 168, 210, map.ground, ' opacity="0.52"'),
      px(580, 172, 120, 18, map.accent, ' opacity="0.36"'),
      px(580, 218, 120, 18, map.light, ' opacity="0.28"'),
      px(580, 264, 120, 18, map.accent, ' opacity="0.3"'),
      ellipse(640, 202, 118, 32, map.accent, ' opacity="0.2"')
    ]);
  }
  if (motif === 'eclipse') {
    return box([
      polygon('604,352 640,146 676,352', map.ground, ' opacity="0.64"'),
      ellipse(640, 136, 78, 78, map.light, ' opacity="0.34"'),
      ellipse(666, 134, 76, 76, map.sky, ' opacity="0.7"'),
      px(568, 348, 144, 12, map.accent, ' opacity="0.36"')
    ]);
  }
  return ellipse(640, 234, 98, 42, map.accent, ' opacity="0.24"');
}

function drawBossRoomMap(map) {
  const motif = map.motif || 'boss';
  const pillars = Array.from({ length: 6 }, (_, index) => {
    const x = 80 + index * 218;
    const h = 270 + index % 2 * 58;
    return box([
      px(x, 512 - h, 42, h, map.ground, ' opacity="0.46"'),
      px(x + 6, 512 - h + 18, 30, h - 34, map.mid, ' opacity="0.38"'),
      px(x - 12, 512 - h, 66, 10, map.light, ' opacity="0.48"')
    ]);
  }).join('');
  const arenaMarks = Array.from({ length: 5 }, (_, index) => {
    const x = 170 + index * 238;
    if (motif === 'storm') return polygon(`${x},466 ${x + 28},400 ${x + 8},448 ${x + 44},448 ${x + 2},516 ${x + 18},466`, map.accent, ' opacity="0.46"');
    if (motif === 'astral') return box([ellipse(x + 24, 444, 42, 18, map.accent, ' opacity="0.32"'), px(x + 3, 441, 42, 5, map.light, ' opacity="0.5"')]);
    if (motif === 'eclipse') return box([ellipse(x + 24, 434, 36, 36, map.light, ' opacity="0.36"'), ellipse(x + 34, 432, 34, 34, map.ground, ' opacity="0.76"')]);
    if (motif === 'furnace') return polygon(`${x},520 ${x + 26},426 ${x + 52},520`, map.accent, ' opacity="0.42"');
    if (motif === 'rime') return polygon(`${x},516 ${x + 20},414 ${x + 42},516`, map.light, ' opacity="0.44"');
    if (motif === 'gear') return box([px(x - 24, 430, 96, 12, map.light, ' opacity="0.42"'), px(x + 18, 388, 12, 96, map.light, ' opacity="0.42"'), ellipse(x + 24, 436, 42, 42, map.far, ' opacity="0.42"')]);
    if (motif === 'core') return box([polygon(`${x},498 ${x + 26},392 ${x + 54},498`, map.accent, ' opacity="0.42"'), ellipse(x + 27, 456, 38, 16, map.light, ' opacity="0.3"')]);
    return polygon(`${x},514 ${x + 18},460 ${x + 36},514`, map.accent, ' opacity="0.42"');
  }).join('');
  const centerSigil = motif === 'eclipse'
    ? box([ellipse(640, 226, 92, 92, map.light, ' opacity="0.38"'), ellipse(668, 224, 92, 92, map.sky, ' opacity="0.74"')])
    : motif === 'storm'
      ? polygon('642,104 588,230 644,210 608,326 714,180 652,196', map.accent, ' opacity="0.36"')
      : motif === 'astral'
        ? box([ellipse(640, 226, 104, 38, map.accent, ' opacity="0.28"'), px(596, 222, 88, 8, map.light, ' opacity="0.42"'), px(636, 186, 8, 82, map.light, ' opacity="0.32"')])
        : ellipse(640, 234, 98, 42, map.accent, ' opacity="0.24"');
  return box([
    px(0, 0, 1280, 640, map.sky),
    polygon('0,238 145,172 322,236 500,144 688,238 860,164 1045,238 1280,154 1280,424 0,424', map.far, ' opacity="0.82"'),
    polygon('0,340 155,280 312,344 470,260 646,348 820,270 1000,346 1160,276 1280,336 1280,492 0,492', map.mid, ' opacity="0.74"'),
    px(0, 84, 1280, 76, map.ground, ' opacity="0.3"'),
    centerSigil,
    drawBossRoomFocal(map),
    pillars,
    arenaMarks,
    mapPlatform(0, 520, 1280, 120, map.light, map.ground),
    mapPlatform(120, 452, 320, 28, map.light, map.ground),
    mapPlatform(520, 378, 280, 28, map.light, map.ground),
    mapPlatform(878, 306, 264, 28, map.light, map.ground),
    mapPlatform(948, 452, 272, 28, map.light, map.ground),
    px(0, 512, 1280, 8, map.accent, ' opacity="0.58"'),
    px(586, 406, 108, 8, map.accent, ' opacity="0.5"'),
    px(966, 334, 92, 8, map.light, ' opacity="0.38"')
  ]);
}

function drawTrialMotif(map, x, y, index) {
  if (map.motif === 'shield') {
    return box([
      polygon(`${x},${y} ${x + 42},${y + 12} ${x + 35},${y + 62} ${x + 21},${y + 76} ${x + 7},${y + 62} ${x},${y + 12}`, map.mid, ' opacity="0.58"'),
      px(x + 18, y + 18, 6, 42, map.light, ' opacity="0.45"')
    ]);
  }
  if (map.motif === 'chains') {
    return Array.from({ length: 5 }, (_, link) => {
      const lx = x + link * 24;
      const ly = y + (link % 2) * 9;
      return ellipse(lx, ly, 16, 7, link % 2 ? map.light : map.accent, ' opacity="0.42"');
    }).join('');
  }
  if (map.motif === 'banners') {
    return box([
      px(x, y, 5, 82, map.ground),
      polygon(`${x + 5},${y + 8} ${x + 64},${y + 20} ${x + 5},${y + 36}`, index % 2 ? map.accent : map.mid, ' opacity="0.62"')
    ]);
  }
  if (map.motif === 'flame') {
    return box([
      polygon(`${x},${y + 74} ${x + 20},${y + 18} ${x + 42},${y + 74}`, map.accent, ' opacity="0.48"'),
      polygon(`${x + 12},${y + 70} ${x + 26},${y + 34} ${x + 36},${y + 70}`, map.light, ' opacity="0.38"')
    ]);
  }
  if (map.motif === 'runes') {
    return box([
      ellipse(x + 26, y + 28, 34, 18, map.accent, ' opacity="0.34"'),
      px(x + 5, y + 26, 42, 5, map.light, ' opacity="0.54"'),
      px(x + 23, y + 8, 5, 42, map.light, ' opacity="0.42"')
    ]);
  }
  if (map.motif === 'lightning') {
    return polygon(`${x + 24},${y} ${x + 2},${y + 44} ${x + 26},${y + 38} ${x + 12},${y + 84} ${x + 52},${y + 28} ${x + 28},${y + 34}`, map.accent, ' opacity="0.5"');
  }
  if (map.motif === 'targets') {
    return box([
      ellipse(x + 30, y + 30, 28, 28, map.light, ' opacity="0.44"'),
      ellipse(x + 30, y + 30, 18, 18, map.ground, ' opacity="0.78"'),
      ellipse(x + 30, y + 30, 7, 7, map.accent, ' opacity="0.8"')
    ]);
  }
  if (map.motif === 'traps') {
    return box([
      px(x, y + 58, 72, 7, map.dark || '#1f2f28', ' opacity="0.58"'),
      polygon(`${x + 8},${y + 58} ${x + 18},${y + 30} ${x + 28},${y + 58}`, map.accent, ' opacity="0.48"'),
      polygon(`${x + 34},${y + 58} ${x + 44},${y + 24} ${x + 54},${y + 58}`, map.light, ' opacity="0.36"')
    ]);
  }
  return box([
    ellipse(x + 34, y + 54, 30, 8, map.accent, ' opacity="0.24"'),
    polygon(`${x + 8},${y + 48} ${x + 28},${y + 28} ${x + 44},${y + 48}`, map.light, ' opacity="0.42"'),
    px(x + 45, y + 38, 34, 6, map.accent, ' opacity="0.46"')
  ]);
}

function drawTrialMap(map) {
  const farPeaks = polygon('0,250 150,176 314,248 492,158 684,246 872,170 1060,252 1280,158 1280,430 0,430', map.far, ' opacity="0.76"');
  const midPeaks = polygon('0,348 130,286 306,346 480,270 640,352 806,280 988,350 1160,282 1280,340 1280,488 0,488', map.mid, ' opacity="0.72"');
  const motifs = Array.from({ length: 7 }, (_, index) => drawTrialMotif(map, 86 + index * 180, 142 + (index % 3) * 52, index)).join('');
  const lightBeams = Array.from({ length: 5 }, (_, index) => {
    const x = 120 + index * 260;
    return polygon(`${x},0 ${x + 70},0 ${x + 220},640 ${x + 110},640`, map.light, ' opacity="0.08"');
  }).join('');
  return box([
    px(0, 0, 1280, 640, map.sky),
    lightBeams,
    farPeaks,
    midPeaks,
    motifs,
    px(0, 416, 1280, 224, map.ground),
    mapPlatform(0, 520, 1280, 120, map.light, map.ground),
    mapPlatform(136, 452, 300, 28, map.light, map.mid),
    mapPlatform(508, 378, 282, 28, map.light, map.mid),
    mapPlatform(858, 306, 260, 28, map.light, map.mid),
    mapPlatform(964, 450, 252, 28, map.light, map.mid),
    px(0, 512, 1280, 8, map.accent, ' opacity="0.42"'),
    px(190, 478, 118, 8, map.light, ' opacity="0.3"'),
    px(590, 406, 96, 8, map.accent, ' opacity="0.35"'),
    px(958, 334, 92, 8, map.light, ' opacity="0.3"')
  ]);
}

function drawMapBackground(map) {
  if (String(map.kind || '').endsWith('Trial')) return drawTrialMap(map);
  if (map.kind === 'bossRoom') return drawBossRoomMap(map);
  if (map.kind === 'bramble') return drawBrambleDepthsMap(map);
  if (map.kind === 'gearworks') return drawGearworksVaultMap(map);
  if (map.kind === 'emberjaw') return drawEmberjawLairMap(map);
  if (map.kind === 'frostfen' || map.kind === 'glacier' || map.kind === 'rimewarden') return drawFrostMap(map);
  return px(0, 0, 1280, 640, map.sky || '#203a31');
}

function drawPortalSprite(portal, frame) {
  const phase = frame % COLS;
  const open = portal.locked ? 0.72 : 1;
  const squeeze = phase === 0 || phase === 5 ? 0 : phase === 1 || phase === 4 ? 4 : 7;
  const glow = 0.4 + phase * 0.055;
  const ringInset = portal.locked ? 12 : 0;
  const centerX = 80;
  const centerY = 79;
  const ry = 54 + (phase % 3 === 1 ? 2 : 0);
  const rx = 31 + squeeze - ringInset * 0.25;
  const shardOffset = (phase % 2) * 3;
  const interior = portal.locked ? 'rgba(23,34,47,0.62)' : 'rgba(10,29,56,0.44)';
	  const sparkOpacity = portal.locked ? 0.25 : 0.68;
	  const runeOpacity = portal.locked ? 0.24 : 0.72;
	  const bossCrown = portal.id === 'boss' ? box([
	    polygon(`${centerX},22 ${centerX + 11},35 ${centerX + 25},29 ${centerX + 17},44 ${centerX - 17},44 ${centerX - 25},29 ${centerX - 11},35`, portal.accent, ` opacity="${(0.56 + phase * 0.04).toFixed(2)}"`),
	    px(centerX - 3, 24, 6, 6, portal.light, ' opacity="0.9"')
	  ]) : '';
	  const lockBars = portal.locked ? box([
    px(51, 74, 58, 10, portal.dark, ' opacity="0.9"'),
    px(58, 56, 8, 42, portal.dark, ' opacity="0.88"'),
    px(94, 56, 8, 42, portal.dark, ' opacity="0.88"'),
    px(58, 54, 44, 9, portal.light, ' opacity="0.58"'),
    px(58, 94, 44, 9, portal.light, ' opacity="0.38"')
  ]) : '';

  return box([
    ellipse(centerX, 134, 46, 10, '#102033', ' opacity="0.3"'),
    ellipse(centerX, centerY, rx + 18, ry + 12, portal.core, ` opacity="${glow.toFixed(2)}"`),
    ellipse(centerX, centerY, rx + 11, ry + 6, portal.dark, ` opacity="${(0.5 * open).toFixed(2)}"`),
    ellipse(centerX, centerY, rx, ry, portal.core, ` opacity="${(0.92 * open).toFixed(2)}"`),
    ellipse(centerX, centerY, Math.max(12, rx - 14), Math.max(28, ry - 18), interior),
	    ellipse(centerX, centerY, Math.max(8, rx - 23 + phase % 2 * 3), Math.max(20, ry - 27), portal.light, ` opacity="${(0.5 * open).toFixed(2)}"`),
	    px(73 - squeeze * 0.35, 35, 12 + squeeze * 0.7, 88, portal.accent, ` opacity="${(0.28 * open).toFixed(2)}"`),
	    polygon(`${centerX},${centerY - ry - 5} ${centerX + 10},${centerY - ry + 11} ${centerX - 10},${centerY - ry + 11}`, portal.light, ` opacity="${runeOpacity}"`),
	    polygon(`${centerX},${centerY + ry + 5} ${centerX + 10},${centerY + ry - 11} ${centerX - 10},${centerY + ry - 11}`, portal.light, ` opacity="${(runeOpacity * 0.82).toFixed(2)}"`),
	    polygon(`${centerX - rx - 12},${centerY} ${centerX - rx + 1},${centerY - 8} ${centerX - rx + 1},${centerY + 8}`, portal.accent, ` opacity="${runeOpacity}"`),
	    polygon(`${centerX + rx + 12},${centerY} ${centerX + rx - 1},${centerY - 8} ${centerX + rx - 1},${centerY + 8}`, portal.accent, ` opacity="${runeOpacity}"`),
	    px(61 - shardOffset, 48, 8, 20, portal.light, ` opacity="${sparkOpacity}"`),
    px(98 + shardOffset, 90, 7, 17, portal.light, ` opacity="${sparkOpacity}"`),
    px(49 + phase * 2, 116, 10, 5, portal.accent, ` opacity="${sparkOpacity}"`),
    px(104 - phase, 39, 8, 5, portal.accent, ` opacity="${sparkOpacity}"`),
	    portal.locked ? '' : px(79 + (phase - 2), 24, 5, 12, portal.light, ' opacity="0.72"'),
	    portal.locked ? '' : px(84 - (phase % 3), 120, 5, 14, portal.light, ' opacity="0.55"'),
	    bossCrown,
	    lockBars
	  ]);
}

function makeSheet(renderer, rows) {
  const sheetRows = rows || ROWS;
  const body = sheetRows.map((row, rowIndex) => Array.from({ length: COLS }, (_, frame) => {
    return `<svg x="${frame * FRAME}" y="${rowIndex * FRAME}" width="${FRAME}" height="${FRAME}" viewBox="0 0 ${FRAME} ${FRAME}" overflow="hidden">${renderer(row, frame)}</svg>`;
  }).join('')).join('');
  return svg(COLS * FRAME, sheetRows.length * FRAME, body);
}

function makePortalSheet(portal) {
  const body = Array.from({ length: COLS }, (_, frame) => {
    return group(frame * FRAME, 0, drawPortalSprite(portal, frame));
  }).join('');
  return svg(COLS * FRAME, FRAME, body);
}

async function writePng(buffer, destination, options) {
  const settings = options || {};
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  await sharp(buffer).png({ compressionLevel: Number.isFinite(settings.compressionLevel) ? settings.compressionLevel : 9 }).toFile(destination);
  return path.relative(ROOT, destination).replace(/\\/g, '/');
}

async function writeWebp(buffer, destination, options) {
  const settings = options || {};
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  await sharp(buffer).webp({
    quality: Number.isFinite(settings.quality) ? settings.quality : 86,
    effort: Number.isFinite(settings.effort) ? settings.effort : 4
  }).toFile(destination);
  return path.relative(ROOT, destination).replace(/\\/g, '/');
}

async function validatePngDimensions(filePath, width, height) {
  const metadata = await sharp(filePath).metadata();
  if (metadata.width !== width || metadata.height !== height) {
    throw new Error(`${path.relative(ROOT, filePath)} is ${metadata.width}x${metadata.height}; expected ${width}x${height}`);
  }
}

async function makeGenericPlayerCharacter() {
  const portrait = svg(320, 320, `<g transform="scale(2)">${drawGenericPlayer('idle', 0)}</g>`);
  return writePng(portrait, GENERIC_CHARACTER_BACKUP_PATH);
}

async function makeGenericPlayerSheet() {
  return writePng(makeSheet(drawGenericPlayer), GENERIC_SHEET_BACKUP_PATH);
}

async function makeClassCharacter(variant) {
  const portrait = svg(320, 320, `<g transform="scale(2)">${drawClassPlayer(variant, 'idle', 0)}</g>`);
  return writePng(portrait, path.join(CHARACTER_DIR, `${variant.fileId}.png`), { compressionLevel: 0 });
}

async function makeClassSheet(variant) {
  return writePng(makeSheet((row, frame) => drawClassPlayer(variant, row, frame)), path.join(PLAYER_DIR, `${variant.fileId}-sheet.png`));
}

async function makeMapBackground(map) {
  const targetDir = String(map.id || '').endsWith('_trial') ? path.join(MAP_DIR, 'trials') : MAP_DIR;
  return writeWebp(svg(1280, 640, drawMapBackground(map)), path.join(targetDir, `${map.fileId}.webp`));
}

async function makePortalAnimationSheet(portal) {
  return writePng(makePortalSheet(portal), path.join(PORTAL_DIR, `${portal.fileId}-sheet.png`));
}

async function generatePortalAssets(generated) {
  for (const portal of PORTAL_VARIANTS) {
    generated.push(await makePortalAnimationSheet(portal));
  }
}

async function generateTrialMapAssets(generated) {
  for (const map of TRIAL_MAP_BACKGROUNDS) {
    generated.push(await makeMapBackground(map));
  }
}

async function generateBossRoomAssets(generated) {
  for (const map of BOSS_ROOM_BACKGROUNDS) {
    generated.push(await makeMapBackground(map));
  }
}

async function generatePlayerAssets(generated) {
  generated.push(await makeGenericPlayerCharacter());
  generated.push(await makeGenericPlayerSheet());
  for (const variant of CLASS_VARIANTS) {
    generated.push(await makeClassCharacter(variant));
    generated.push(await makeClassSheet(variant));
  }
}

async function generateGenericPlayerBackups(generated) {
  generated.push(await makeGenericPlayerCharacter());
  generated.push(await makeGenericPlayerSheet());
}

async function validateGenericPlayerBackups() {
  await validatePngDimensions(GENERIC_CHARACTER_BACKUP_PATH, 320, 320);
  await validatePngDimensions(GENERIC_SHEET_BACKUP_PATH, COLS * FRAME, ROWS.length * FRAME);
}

async function validatePlayerAssets() {
  await validateGenericPlayerBackups();
  for (const variant of CLASS_VARIANTS) {
    await validatePngDimensions(path.join(CHARACTER_DIR, `${variant.fileId}.png`), 320, 320);
    await validatePngDimensions(path.join(PLAYER_DIR, `${variant.fileId}-sheet.png`), COLS * FRAME, ROWS.length * FRAME);
  }
  console.log(`Validated ${CLASS_VARIANTS.length} procedural class portraits/player sheets and the generic player sheet`);
}

async function generateAllAssets(generated) {
  await generateGenericPlayerBackups(generated);
  const playerAssetProcessor = require('./process-project-starfall-player-ai-assets.js');
  await playerAssetProcessor.generateAll();
  for (const map of MAP_BACKGROUNDS) {
    generated.push(await makeMapBackground(map));
  }
  await generateTrialMapAssets(generated);
  await generatePortalAssets(generated);
}

async function main() {
  const onlyIndex = process.argv.indexOf('--only');
  const onlyTarget = onlyIndex >= 0 ? String(process.argv[onlyIndex + 1] || '').trim().toLowerCase() : '';
  const validateOnly = process.argv.includes('--validate');
  const generated = [];
  const playerTargets = ['players', 'player', 'actors', 'characters', 'player-equipment'];
  if (onlyTarget && !['portals', 'maps', 'trial-maps', 'boss-rooms', 'equipment'].concat(playerTargets).includes(onlyTarget)) {
    throw new Error(`Unsupported --only target: ${onlyTarget}`);
  }
  if (playerTargets.includes(onlyTarget)) {
    const playerAssetProcessor = require('./process-project-starfall-player-ai-assets.js');
    if (validateOnly) {
      await validateGenericPlayerBackups();
      await playerAssetProcessor.main(['--validate']);
    } else {
      await generateGenericPlayerBackups(generated);
      await playerAssetProcessor.main([]);
      generated.forEach((file) => process.stdout.write(`Generated ${file}\n`));
    }
    return;
  }
  if (validateOnly) {
    if (onlyTarget === 'equipment') {
      const equipmentAtlasGenerator = require('./generate-project-starfall-equipment-atlases.js');
      await equipmentAtlasGenerator.main(['--all', '--validate']);
      return;
    }
    if (onlyTarget && !playerTargets.includes(onlyTarget)) {
      throw new Error(`Targeted validation is only supported for player or equipment assets, received: ${onlyTarget}`);
    }
    await validateGenericPlayerBackups();
    const playerAssetProcessor = require('./process-project-starfall-player-ai-assets.js');
    await playerAssetProcessor.validateAll();
    return;
  }
  if (onlyTarget === 'portals') {
    await generatePortalAssets(generated);
  } else if (onlyTarget === 'equipment') {
    const equipmentAtlasGenerator = require('./generate-project-starfall-equipment-atlases.js');
    await equipmentAtlasGenerator.main(['--all']);
    return;
  } else if (onlyTarget === 'maps') {
    for (const map of MAP_BACKGROUNDS) {
      generated.push(await makeMapBackground(map));
    }
  } else if (onlyTarget === 'trial-maps') {
    await generateTrialMapAssets(generated);
  } else if (onlyTarget === 'boss-rooms') {
    await generateBossRoomAssets(generated);
  } else {
    await generateAllAssets(generated);
  }
  generated.forEach((file) => process.stdout.write(`Generated ${file}\n`));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
