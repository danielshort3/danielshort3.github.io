(function initProjectStarfallDataBossEncounters(global) {
  'use strict';

  const BOSS_ENCOUNTERS = Object.freeze([
    Object.freeze({
      id: 'brambleking',
      bossId: 'brambleking',
      name: 'Brambleking Court',
      mapId: 'bramblekingCourt',
      setId: 'thorncrown_regalia',
      hpScale: 64,
      color: '#e05b75',
      accent: '#8bd47a',
      roomAmbient: 'bramble',
      mechanic: 'Swap lanes when roots grow, destroy thorn pods when adds spawn, and burst during Crowned Root.',
      intro: 'The court roots itself across the arena. Move between lanes before the roots close.',
      clearText: 'The crown breaks and the roots pull back from the court.',
      summary: 'Root waves force lane swaps, thorn volleys punish stacking, and Crowned Root exposes a short damage window.',
      adds: Object.freeze(['thornSprout', 'vineSnapper', 'glowcapHealer']),
      phases: Object.freeze([
        Object.freeze({ id: 'rootCourt', name: 'Root Court', threshold: 1, description: 'Root lanes telegraph where the court will split.', actions: Object.freeze(['rootWave', 'thornVolley']) }),
        Object.freeze({ id: 'thornCanopy', name: 'Thorn Canopy', threshold: 0.7, description: 'Thorn pods call in sprouts while volleys punish stacked players.', actions: Object.freeze(['thornVolley', 'addWave', 'rootWave']) }),
        Object.freeze({ id: 'crownedRoot', name: 'Crowned Root', threshold: 0.38, description: 'The crown opens a short burst window after the vine cage.', actions: Object.freeze(['vineCage', 'rootWave', 'crownExpose']) })
      ])
    }),
    Object.freeze({
      id: 'clockworkTitan',
      bossId: 'clockworkTitan',
      name: 'Titan Foundry',
      mapId: 'titanFoundry',
      setId: 'titanwork_aegis',
      hpScale: 59,
      color: '#29b3ad',
      accent: '#d8b74a',
      roomAmbient: 'gear',
      mechanic: 'Use pressure-plate gaps between gear lanes, then punish exposed plates before overclock ends.',
      intro: 'The foundry wakes one gear at a time. Watch the floor before the Titan commits.',
      clearText: 'The Titan locks up and the foundry gears grind to a halt.',
      summary: 'Gear lanes sweep the arena while exposed plates create burst windows between overclock cycles.',
      adds: Object.freeze(['clockbug', 'coilSentry', 'rustRatchet']),
      phases: Object.freeze([
        Object.freeze({ id: 'gearStart', name: 'Gear Start', threshold: 1, description: 'Heavy gear slams mark safe pressure-plate gaps.', actions: Object.freeze(['gearSlam', 'plateExpose']) }),
        Object.freeze({ id: 'foundryShift', name: 'Foundry Shift', threshold: 0.7, description: 'Gear lanes sweep longer sections while foundry minions enter.', actions: Object.freeze(['gearLane', 'addWave', 'gearSlam']) }),
        Object.freeze({ id: 'overclocked', name: 'Overclocked', threshold: 0.38, description: 'Overclock pulses accelerate the pattern before plates open.', actions: Object.freeze(['overclock', 'gearLane', 'plateExpose']) })
      ])
    }),
    Object.freeze({
      id: 'quarryColossus',
      bossId: 'quarryColossus',
      name: 'Deepcore Core',
      mapId: 'deepcoreCore',
      setId: 'deepcore_colossus',
      hpScale: 57,
      color: '#69d1a6',
      accent: '#c3b48f',
      roomAmbient: 'core',
      mechanic: 'Spread away from quake anchors, dodge rockfall shadows, and collapse the cracked core pulse.',
      intro: 'The Deepcore awakens under the platforms. Quake anchors will split the party if ignored.',
      clearText: 'The core fractures cleanly and the quarry quiets.',
      summary: 'Orefalls and quake anchors split the party across the mining terraces before the core cracks open.',
      adds: Object.freeze(['orebackBeetle', 'scrapWarden', 'glowcapHealer']),
      phases: Object.freeze([
        Object.freeze({ id: 'stoneSkin', name: 'Stone Skin', threshold: 1, description: 'Rockfall shadows and quake anchors define the first safe paths.', actions: Object.freeze(['rockfall', 'quakeAnchor']) }),
        Object.freeze({ id: 'deepSeam', name: 'Deep Core', threshold: 0.7, description: 'Adds arrive from mining seams while anchors force movement.', actions: Object.freeze(['quakeAnchor', 'addWave', 'rockfall']) }),
        Object.freeze({ id: 'coreBreak', name: 'Core Break', threshold: 0.38, description: 'The exposed core pulses around the Colossus before the next cave-in.', actions: Object.freeze(['corePulse', 'rockfall', 'quakeAnchor']) })
      ])
    }),
    Object.freeze({
      id: 'emberjawGolem',
      bossId: 'emberjawGolem',
      name: 'Emberjaw Furnace',
      mapId: 'emberjawFurnace',
      setId: 'furnaceheart_arsenal',
      hpScale: 54,
      color: '#ff7842',
      accent: '#ffd166',
      roomAmbient: 'furnace',
      mechanic: 'Cross lava seams before they flare, avoid charge lanes, and burn the core during overheat.',
      intro: 'Emberjaw floods the furnace floor with seams. Keep moving before the lava breathes.',
      clearText: 'The furnace cools and Emberjaw collapses into slag.',
      summary: 'Furnace cracks mark burning lanes, lava charges reposition Emberjaw, and overheat briefly weakens the core.',
      adds: Object.freeze(['emberWisp', 'lavaTick', 'cinderSpitter']),
      phases: Object.freeze([
        Object.freeze({ id: 'heatedStone', name: 'Heated Stone', threshold: 1, description: 'Fire cracks mark lanes before Emberjaw charges.', actions: Object.freeze(['fireCrack', 'lavaCharge']) }),
        Object.freeze({ id: 'furnaceJaw', name: 'Furnace Jaw', threshold: 0.7, description: 'Cinder adds pressure the furnace while charge lanes shift.', actions: Object.freeze(['lavaCharge', 'addWave', 'fireCrack']) }),
        Object.freeze({ id: 'meltdownCore', name: 'Meltdown Core', threshold: 0.38, description: 'Overheat exposes the core before the floor erupts again.', actions: Object.freeze(['overheat', 'fireCrack', 'lavaCharge']) })
      ])
    }),
    Object.freeze({
      id: 'rimewarden',
      bossId: 'rimewarden',
      name: 'Rimewarden Vault',
      mapId: 'rimewardenVault',
      setId: '',
      hpScale: 55,
      color: '#79e7ff',
      accent: '#f7fbff',
      roomAmbient: 'rime',
      mechanic: 'Rotate around ice walls, clear whiteout lanes, and jump frost rings before they close.',
      intro: 'The vault seals behind you. Ice walls will narrow the room before the whiteout lands.',
      clearText: 'The vault thaw cracks and the Rimewarden fades.',
      summary: 'Ice walls close training lanes, whiteout blasts sweep one platform tier, and frost rings punish slow rotations.',
      adds: Object.freeze(['frostlingScout', 'snowglareWisp', 'icebloomOracle']),
      phases: Object.freeze([
        Object.freeze({ id: 'coldSeal', name: 'Cold Seal', threshold: 1, description: 'Frost rings expand while walls mark blocked space.', actions: Object.freeze(['iceShockwave', 'iceWall']) }),
        Object.freeze({ id: 'whiteVault', name: 'White Vault', threshold: 0.7, description: 'Whiteout sweeps one tier while frost adds close in.', actions: Object.freeze(['whiteout', 'addWave', 'iceShockwave']) }),
        Object.freeze({ id: 'absoluteZero', name: 'Absolute Zero', threshold: 0.38, description: 'Walls and whiteout overlap, forcing clean rotations.', actions: Object.freeze(['iceWall', 'whiteout', 'iceShockwave']) })
      ])
    }),
    Object.freeze({
      id: 'stormbreakRoc',
      bossId: 'stormbreakRoc',
      name: 'Stormbreak Aerie',
      mapId: 'stormbreakAerie',
      setId: 'stormcaller_tempest',
      hpScale: 52,
      color: '#ffe16a',
      accent: '#91dbe8',
      roomAmbient: 'storm',
      mechanic: 'Keep out of rod circles, cross wind lanes early, and move when the divebomb shadow appears.',
      intro: 'Aurelion circles above the aerie. Rods will ground the lightning before the sky falls.',
      clearText: 'The storm breaks open and Aurelion drops from the clouds.',
      summary: 'Aurelion drops lightning rods, pushes wind lanes across platforms, and divebombs isolated targets.',
      adds: Object.freeze(['galeHarrier', 'stormboundArcher', 'cloudcallAcolyte']),
      phases: Object.freeze([
        Object.freeze({ id: 'highWinds', name: 'High Winds', threshold: 1, description: 'Wind bolts and rod circles establish the storm pattern.', actions: Object.freeze(['windBolt', 'lightningRod']) }),
        Object.freeze({ id: 'stormPerch', name: 'Storm Perch', threshold: 0.7, description: 'Wind lanes push across platforms while stormbound adds arrive.', actions: Object.freeze(['windLane', 'addWave', 'lightningRod']) }),
        Object.freeze({ id: 'skyfall', name: 'Skyfall', threshold: 0.38, description: 'Divebomb shadows force immediate movement before the next lane.', actions: Object.freeze(['divebomb', 'windLane', 'lightningRod']) })
      ])
    }),
    Object.freeze({
      id: 'astralArchivist',
      bossId: 'astralArchivist',
      name: 'Astral Stacks',
      mapId: 'astralStacks',
      setId: 'astral_index',
      hpScale: 50,
      color: '#c794ff',
      accent: '#64d9c5',
      roomAmbient: 'astral',
      mechanic: 'Step around rune pages, break memory seals, and dodge mirrored echo lanes.',
      intro: 'The shelves reorder themselves. The Archivist records every repeated mistake.',
      clearText: 'The forbidden appendix snaps shut and the stacks realign.',
      summary: 'Rune pages travel between shelves while the Archivist seals repeated actions and mirrors delayed casts.',
      adds: Object.freeze(['indexScribe', 'lumenSentinel', 'voidMote']),
      phases: Object.freeze([
        Object.freeze({ id: 'openedIndex', name: 'Opened Index', threshold: 1, description: 'Rune pages travel between shelves and memory seals mark danger.', actions: Object.freeze(['runePages', 'memorySeal']) }),
        Object.freeze({ id: 'mirroredStacks', name: 'Mirrored Stacks', threshold: 0.7, description: 'Mirrored echo lanes replay delayed casts as adds enter.', actions: Object.freeze(['mirrorEcho', 'addWave', 'runePages']) }),
        Object.freeze({ id: 'forbiddenAppendix', name: 'Forbidden Appendix', threshold: 0.38, description: 'Seals and echoes overlap until the pages settle.', actions: Object.freeze(['memorySeal', 'mirrorEcho', 'runePages']) })
      ])
    }),
    Object.freeze({
      id: 'eclipseSovereign',
      bossId: 'eclipseSovereign',
      name: 'Eclipse Throne',
      mapId: 'eclipseThrone',
      setId: 'eclipse_paragon',
      hpScale: 49,
      color: '#ffbe55',
      accent: '#7bdff2',
      roomAmbient: 'eclipse',
      introCombatDelay: 6.1,
      phaseTransitionDelay: 2.2,
      resetActionCycleOnPhase: true,
      mechanic: 'Clear solar and lunar danger zones, then regroup inside the Eclipse Dais when Totality collapses.',
      intro: 'Solar and lunar marks call danger zones. When Totality begins, regroup on the central Eclipse Dais.',
      clearText: 'Totality fades and the throne releases its light.',
      summary: 'Solar and lunar danger calls alternate until Totality demands a coordinated central regroup.',
      adds: Object.freeze(['eclipseDuelist', 'voidMote', 'lumenSentinel']),
      phases: Object.freeze([
        Object.freeze({ id: 'solarCourt', name: 'Solar Court', threshold: 1, description: 'Solar flares expand from the throne while lunar marks choose targets.', actions: Object.freeze(['solarFlare', 'lunarMark']) }),
        Object.freeze({ id: 'lunarCourt', name: 'Lunar Court', threshold: 0.7, description: 'Lunar danger marks tighten as eclipse duelists arrive.', actions: Object.freeze(['lunarMark', 'addWave', 'solarFlare']) }),
        Object.freeze({ id: 'totality', name: 'Totality', threshold: 0.38, description: 'Regroup inside the central safe dais before Totality collapses.', actions: Object.freeze(['eclipseSigils', 'solarFlare', 'lunarMark']) })
      ])
    })
  ]);

  const api = {
    BOSS_ENCOUNTERS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.bossEncounters = Object.assign({}, modules.bossEncounters || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
