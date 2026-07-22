(function initProjectStarfallDataBossMechanics(global) {
  'use strict';

  const BOSS_SPATIAL_RESPONSE_CHECKS = Object.freeze({
    rootWave: Object.freeze({ type: 'avoidHazard', label: 'Clear the marked root lane before impact.' }),
    thornVolley: Object.freeze({ type: 'dodgeProjectiles', label: 'Avoid every projectile in the thorn volley.' }),
    addWave: Object.freeze({ type: 'clearAdds', label: 'Defeat the full summoned wave.' }),
    vineCage: Object.freeze({ type: 'avoidHazard', label: 'Escape the cage impact area.' }),
    crownExpose: Object.freeze({ type: 'damageWindow', label: 'Reach the called crown section and damage the boss during Crowned Root.', requireSection: true, windowSeconds: 2.5 }),
    gearSlam: Object.freeze({ type: 'avoidHazard', label: 'Clear the marked slam before impact.' }),
    gearLane: Object.freeze({ type: 'reachSection', label: 'Reach the called gear-switch section without being hit.', requireNoHit: true }),
    plateExpose: Object.freeze({ type: 'damageWindow', label: 'Reach the called switch section and damage the boss while its plates are open.', requireSection: true, windowSeconds: 2.5 }),
    overclock: Object.freeze({ type: 'reachSection', label: 'Reach the overclock control section without being hit.', requireNoHit: true }),
    rockfall: Object.freeze({ type: 'avoidHazard', label: 'Clear the rockfall impact area.' }),
    quakeAnchor: Object.freeze({ type: 'avoidHazard', label: 'Avoid the quake-anchor lane.' }),
    corePulse: Object.freeze({ type: 'reachSection', label: 'Reach the called core-control section without being hit.', requireNoHit: true }),
    fireCrack: Object.freeze({ type: 'avoidHazard', label: 'Cross clear of the fire-crack lane before eruption.' }),
    lavaCharge: Object.freeze({ type: 'reachSection', label: 'Reach the safe-pocket section without being hit.', requireNoHit: true }),
    overheat: Object.freeze({ type: 'damageWindow', label: 'Reach the called vent section and damage the boss during overheat.', requireSection: true, windowSeconds: 2.5 }),
    iceShockwave: Object.freeze({ type: 'avoidHazard', label: 'Jump or move clear of the frost ring.' }),
    iceWall: Object.freeze({ type: 'avoidHazard', label: 'Clear the ice-wall impact line.' }),
    whiteout: Object.freeze({ type: 'avoidHazard', label: 'Leave the whiteout lane before impact.' }),
    windBolt: Object.freeze({ type: 'dodgeProjectiles', label: 'Avoid every projectile in the wind-bolt volley.' }),
    lightningRod: Object.freeze({ type: 'avoidHazard', label: 'Clear the lightning-rod strike area.' }),
    windLane: Object.freeze({ type: 'avoidHazard', label: 'Cross clear of the wind lane before impact.' }),
    divebomb: Object.freeze({ type: 'avoidHazard', label: 'Clear the divebomb lane before impact.' }),
    runePages: Object.freeze({ type: 'dodgeProjectiles', label: 'Avoid every projectile in the rune-page volley.' }),
    memorySeal: Object.freeze({ type: 'avoidHazard', label: 'Rotate off the sealed archive section.' }),
    mirrorEcho: Object.freeze({ type: 'avoidHazard', label: 'Clear the mirrored echo lane.' }),
    solarFlare: Object.freeze({ type: 'avoidHazard', label: 'Move clear before the solar zone blooms.' }),
    lunarMark: Object.freeze({ type: 'avoidHazard', label: 'Split clear of the lunar mark.' }),
    eclipseSigils: Object.freeze({ type: 'reachSection', label: 'Regroup on the eclipse dais without being hit.', requireNoHit: true })
  });

  function freezeBossSpatialResponseCheck(actionId, value) {
    const source = value && typeof value === 'object'
      ? value
      : BOSS_SPATIAL_RESPONSE_CHECKS[actionId] || null;
    if (!source || !source.type) return null;
    return Object.freeze({
      type: source.type,
      label: source.label || '',
      requireNoHit: source.requireNoHit !== false,
      requireSection: source.requireSection !== false,
      windowSeconds: Math.max(0, Number(source.windowSeconds || 0))
    });
  }

  function freezeBossSpatialActionHooks(hooks) {
    return Object.freeze(Object.keys(hooks || {}).reduce((entries, actionId) => {
      const hook = hooks[actionId] || {};
      entries[actionId] = Object.freeze({
        actionId,
        sectionId: hook.sectionId || '',
        role: hook.role || '',
        targetTier: hook.targetTier || 'ground',
        targetAnchor: hook.targetAnchor || 'sectionCenter',
        label: hook.label || '',
        response: hook.response || '',
        objective: hook.objective || '',
        responseCheck: freezeBossSpatialResponseCheck(actionId, hook.responseCheck)
      });
      return entries;
    }, {}));
  }

  function createBossSpatialMechanicDefinition(config) {
    return Object.freeze({
      id: config.id,
      mapId: config.mapId,
      label: config.label,
      summary: config.summary,
      rewardAbuseControl: config.rewardAbuseControl || '',
      partyRoleHook: config.partyRoleHook || '',
      hooks: freezeBossSpatialActionHooks(config.hooks)
    });
  }

  const BOSS_SPATIAL_MECHANICS = Object.freeze({
    brambleDepths: createBossSpatialMechanicDefinition({
      id: 'bramble_depths_root_pod_control',
      mapId: 'brambleDepths',
      label: 'Root Pod Control',
      summary: 'Root lanes, thorn root pods, and the court-gate shelf turn the dungeon into a lane-swap check before Brambleking.',
      rewardAbuseControl: 'Spatial progress comes from answering root pod and lane-lock calls instead of camping the safest shelf.',
      partyRoleHook: 'Frontline clears root lanes, ranged breaks thorn pods, and support calls the court-gate regroup.',
      hooks: {
        rootWave: { sectionId: 'brambleDepths_root_lanes', role: 'lane-lock', targetTier: 'ground', label: 'Root Lane Lock', response: 'Roots lock the root lanes; rotate before the floor closes.', objective: 'Swap lanes through the root lane call.' },
        thornVolley: { sectionId: 'brambleDepths_ridge_return', role: 'safe-return', targetTier: 'ground', label: 'Ridge Return', response: 'Thorns punish stacking on the return ridge.', objective: 'Reset through the ridge return instead of camping the shelf.' },
        addWave: { sectionId: 'brambleDepths_court_gate', role: 'root-pod-clear', targetTier: 'high', label: 'Court-Gate Root Pods', response: 'Root pod adds gather at the court gate.', objective: 'Clear the root pod shelf before the boss gate.' },
        vineCage: { sectionId: 'brambleDepths_root_lanes', role: 'cage-break', targetTier: 'ground', label: 'Vine Cage Lane', response: 'Vine cage closes the root lane and opens a brief escape route.', objective: 'Break the cage lane before adds collapse.' },
        crownExpose: { sectionId: 'brambleDepths_court_gate', role: 'burst-window', targetTier: 'high', label: 'Court Gate Burst', response: 'The court gate opens the Crowned Root burst window.', objective: 'Regroup at the court gate for burst damage.' }
      }
    }),
    bramblekingCourt: createBossSpatialMechanicDefinition({
      id: 'brambleking_court_root_pod_control',
      mapId: 'bramblekingCourt',
      label: 'Crowned Root Pod Control',
      summary: 'Root waves, thorn pod shelves, and crown-platform exposes force players to move instead of standing under the boss.',
      rewardAbuseControl: 'Thorn pod and crown calls alternate sections so one safe platform cannot solve the whole fight.',
      partyRoleHook: 'Melee owns the root lane, ranged clears thorn pods, and burst classes climb for crown exposes.',
      hooks: {
        rootWave: { sectionId: 'bramblekingCourt_root_lane', role: 'lane-lock', targetTier: 'ground', label: 'Root Lane Lock', response: 'Roots close the root lane and force a lane swap.', objective: 'Leave the locked lane before the wave lands.' },
        thornVolley: { sectionId: 'bramblekingCourt_thorn_pod_shelf', role: 'pod-pressure', targetTier: 'high', label: 'Thorn Pod Shelf', response: 'Thorn pods seed the upper shelf before volley impact.', objective: 'Clear or abandon the pod shelf before thorns land.' },
        addWave: { sectionId: 'bramblekingCourt_thorn_pod_shelf', role: 'root-pod-clear', targetTier: 'high', label: 'Root Pod Adds', response: 'Root pod adds spawn on the thorn shelf.', objective: 'Assign a player to pod clear instead of full-stack boss damage.' },
        vineCage: { sectionId: 'bramblekingCourt_root_lane', role: 'cage-break', targetTier: 'ground', label: 'Vine Cage Lane', response: 'Vines cage the root lane and punish slow rotations.', objective: 'Break the lane cage and move out.' },
        crownExpose: { sectionId: 'bramblekingCourt_crown_platform', role: 'burst-window', targetTier: 'high', label: 'Crown Platform', response: 'The crown platform opens for a short burst window.', objective: 'Climb for Crowned Root damage.' }
      }
    }),
    gearworksVault: createBossSpatialMechanicDefinition({
      id: 'gearworks_vault_switch_control',
      mapId: 'gearworksVault',
      label: 'Gear Switch Control',
      summary: 'Tank lane, sentry catwalk, and gear switch shelf calls make the factory dungeon a party-lane assignment.',
      rewardAbuseControl: 'Switch calls alternate with sentry and tank pressure so AoE camping the floor loses objective value.',
      partyRoleHook: 'Tank lower armor, ranged clears catwalk sentries, and support rotates gear switches.',
      hooks: {
        gearSlam: { sectionId: 'gearworksVault_tank_lane', role: 'frontline-check', targetTier: 'ground', label: 'Tank Lane Slam', response: 'The lower tank lane takes the gear slam.', objective: 'Hold the tank lane without dragging sentries down.' },
        gearLane: { sectionId: 'gearworksVault_gear_switch_shelf', role: 'switch-call', targetTier: 'high', label: 'Gear Switch Lane', response: 'The gear switch shelf becomes the safe-response lane.', objective: 'Climb and hit the gear switch before the lane sweeps.' },
        plateExpose: { sectionId: 'gearworksVault_gear_switch_shelf', role: 'switch-burst', targetTier: 'high', label: 'Open Gear Switch', response: 'A gear switch opens the boss plate window.', objective: 'Regroup on the switch shelf for plate damage.' },
        overclock: { sectionId: 'gearworksVault_sentry_catwalk', role: 'catwalk-control', targetTier: 'high', label: 'Overclock Catwalk', response: 'Overclock powers the sentry catwalk.', objective: 'Clear catwalk pressure before overclock ends.' },
        rockfall: { sectionId: 'gearworksVault_sentry_catwalk', role: 'catwalk-shadow', targetTier: 'high', label: 'Falling Gear Shadows', response: 'Loose gears fall across the sentry catwalk.', objective: 'Move ranged players off the catwalk shadow.' },
        quakeAnchor: { sectionId: 'gearworksVault_tank_lane', role: 'anchor-hold', targetTier: 'ground', label: 'Tank-Lane Anchor', response: 'The tank lane receives the anchor shock.', objective: 'Keep the anchor out of the switch shelf.' },
        corePulse: { sectionId: 'gearworksVault_gear_switch_shelf', role: 'switch-core', targetTier: 'high', label: 'Gear Switch Pulse', response: 'Gear switches vent the core pulse.', objective: 'Trigger the switch shelf during the pulse window.' },
        addWave: { sectionId: 'gearworksVault_sentry_catwalk', role: 'sentry-clear', targetTier: 'high', label: 'Sentry Catwalk Adds', response: 'Factory adds climb onto the sentry catwalk.', objective: 'Assign ranged clear before they flood the switch shelf.' }
      }
    }),
    titanFoundry: createBossSpatialMechanicDefinition({
      id: 'titan_foundry_switch_control',
      mapId: 'titanFoundry',
      label: 'Armor Switch Control',
      summary: 'Gear floor slams, armor switches, and sentry catwalks turn Titan Foundry into a switch-rotation boss room.',
      rewardAbuseControl: 'Armor switch calls prevent the party from solving Titan with one floor-stack AoE pattern.',
      partyRoleHook: 'Tank the gear floor, ranged controls sentries, and support calls armor switch timing.',
      hooks: {
        gearSlam: { sectionId: 'titanFoundry_gear_floor', role: 'frontline-check', targetTier: 'ground', label: 'Gear Floor Slam', response: 'The gear floor slams before the switch opens.', objective: 'Hold the boss floor and leave the marked plate.' },
        gearLane: { sectionId: 'titanFoundry_armor_switches', role: 'switch-call', targetTier: 'mid', label: 'Armor Switch Lane', response: 'Armor switches redirect the gear lane.', objective: 'Rotate to the switch lane before the sweep.' },
        plateExpose: { sectionId: 'titanFoundry_armor_switches', role: 'switch-burst', targetTier: 'mid', label: 'Open Armor Switches', response: 'Armor switches expose Titan plates.', objective: 'Burst after the armor switch call.' },
        overclock: { sectionId: 'titanFoundry_sentry_catwalk', role: 'catwalk-control', targetTier: 'high', label: 'Overclock Sentries', response: 'Overclock wakes the upper sentry catwalk.', objective: 'Clear upper sentries before the next lane.' },
        addWave: { sectionId: 'titanFoundry_sentry_catwalk', role: 'sentry-clear', targetTier: 'high', label: 'Sentry Catwalk Adds', response: 'Sentries reinforce the upper catwalk.', objective: 'Keep ranged pressure on the catwalk.' }
      }
    }),
    deepcoreCore: createBossSpatialMechanicDefinition({
      id: 'deepcore_four_chamber_control',
      mapId: 'deepcoreCore',
      label: 'Four-Chamber Core Control',
      summary: 'Tank, healer, turret, and ore-core chambers split the Deepcore fight into full-party spatial jobs.',
      rewardAbuseControl: 'Core pulses favor parties that rotate chambers instead of farming one safe ore ledge.',
      partyRoleHook: 'Tank anchors the lower chamber, support manages healer lane, ranged interrupts turrets, and burst hits the ore core.',
      hooks: {
        rockfall: { sectionId: 'deepcoreCore_turret_lane', role: 'turret-shadow', targetTier: 'high', label: 'Turret-Lane Rockfall', response: 'Rockfall shadows the turret lane.', objective: 'Move ranged off the turret lane before impact.' },
        quakeAnchor: { sectionId: 'deepcoreCore_tank_chamber', role: 'anchor-hold', targetTier: 'ground', label: 'Tank Chamber Anchor', response: 'Quake anchor pins the tank chamber.', objective: 'Hold the anchor away from healer and turret lanes.' },
        corePulse: { sectionId: 'deepcoreCore_ore_core', role: 'core-burst', targetTier: 'mid', label: 'Ore Core Pulse', response: 'The ore core opens during the pulse.', objective: 'Collapse to the ore core burst section.' },
        addWave: { sectionId: 'deepcoreCore_healer_lane', role: 'support-clear', targetTier: 'mid', label: 'Healer Lane Adds', response: 'Adds pressure the healer lane.', objective: 'Protect support while adds are active.' }
      }
    }),
    emberjawLair: createBossSpatialMechanicDefinition({
      id: 'emberjaw_lair_furnace_vents',
      mapId: 'emberjawLair',
      label: 'Furnace Vent Control',
      summary: 'West vent, safe pocket, and overheat shelf calls make Emberjaw Lair a hazard-rotation dungeon.',
      rewardAbuseControl: 'Vent pressure and overheat calls prevent players from farming one static safe pocket.',
      partyRoleHook: 'Frontline controls the vent lane, ranged clears overheat shelf adds, and support calls safe-pocket moves.',
      hooks: {
        fireCrack: { sectionId: 'emberjawLair_west_vent', role: 'vent-hazard', targetTier: 'ground', label: 'West Furnace Vent', response: 'The furnace vent cracks before fire erupts.', objective: 'Leave the vent lane before the flare.' },
        lavaCharge: { sectionId: 'emberjawLair_safe_pockets', role: 'safe-pocket', targetTier: 'mid', label: 'Safe Pocket Charge', response: 'Safe pockets define where the charge can be crossed.', objective: 'Cross through the safe pocket after the charge tell.' },
        overheat: { sectionId: 'emberjawLair_overheat_shelf', role: 'overheat-burst', targetTier: 'high', label: 'Overheat Shelf', response: 'The overheat shelf opens for burst damage.', objective: 'Climb for overheat pressure without camping it early.' },
        addWave: { sectionId: 'emberjawLair_west_vent', role: 'vent-adds', targetTier: 'ground', label: 'Vent Adds', response: 'Furnace vent adds spill from the side lane.', objective: 'Clear vent adds before the safe pocket collapses.' }
      }
    }),
    emberjawFurnace: createBossSpatialMechanicDefinition({
      id: 'emberjaw_furnace_valve_control',
      mapId: 'emberjawFurnace',
      label: 'Furnace Valve Control',
      summary: 'Lava cracks, furnace vents, valve shelves, and safe pockets rotate Emberjaw through readable hazard lanes.',
      rewardAbuseControl: 'Valve and safe-pocket calls alternate so one lava-crack gap cannot carry the fight.',
      partyRoleHook: 'Melee baits lava cracks, ranged reaches valve shelf, and support calls the safe pocket.',
      hooks: {
        fireCrack: { sectionId: 'emberjawFurnace_lava_cracks', role: 'lava-lane', targetTier: 'ground', label: 'Lava Crack Lane', response: 'Lava cracks split the furnace floor.', objective: 'Cross the crack lane before it erupts.' },
        lavaCharge: { sectionId: 'emberjawFurnace_safe_pocket', role: 'safe-pocket', targetTier: 'mid', label: 'Safe Pocket Charge', response: 'The safe pocket is the charge response.', objective: 'Move through the safe pocket after the charge.' },
        overheat: { sectionId: 'emberjawFurnace_valve_shelf', role: 'furnace-vent', targetTier: 'high', label: 'Furnace Vent Valve', response: 'The furnace vent valve opens during overheat.', objective: 'Hit the valve shelf to earn the burst window.' },
        addWave: { sectionId: 'emberjawFurnace_valve_shelf', role: 'valve-adds', targetTier: 'high', label: 'Valve Shelf Adds', response: 'Adds climb toward the furnace vent valve.', objective: 'Clear valve shelf adds before overheat.' }
      }
    }),
    rimewardenSanctum: createBossSpatialMechanicDefinition({
      id: 'rimewarden_sanctum_ice_wall_control',
      mapId: 'rimewardenSanctum',
      label: 'Ice Wall Lane Control',
      summary: 'Brute lane, oracle shelf, sentinel shelf, and ice wall lane locks split the frost dungeon into clean rotations.',
      rewardAbuseControl: 'Ice wall and whiteout calls alternate shelves so the party cannot hide on one low-risk tier.',
      partyRoleHook: 'Tank holds brute lane, ranged clears oracle and sentinel shelves, and support calls wall-safe resets.',
      hooks: {
        iceShockwave: { sectionId: 'rimewardenSanctum_brute_lane', role: 'frontline-ring', targetTier: 'ground', label: 'Brute Lane Ring', response: 'The brute lane receives the frost ring.', objective: 'Jump or leave the brute lane before the ring closes.' },
        iceWall: { sectionId: 'rimewardenSanctum_sentinel_shelf', role: 'ice-wall-lock', targetTier: 'high', label: 'Sentinel Ice Wall', response: 'An ice wall locks the sentinel shelf lane.', objective: 'Rotate around the ice wall lane lock.' },
        whiteout: { sectionId: 'rimewardenSanctum_oracle_shelf', role: 'whiteout-shelf', targetTier: 'mid', label: 'Oracle Whiteout', response: 'Whiteout sweeps the oracle shelf.', objective: 'Move oracle players before the shelf whites out.' },
        addWave: { sectionId: 'rimewardenSanctum_sentinel_shelf', role: 'sentinel-clear', targetTier: 'high', label: 'Sentinel Adds', response: 'Sentinel shelf adds arrive behind the wall.', objective: 'Clear sentinel adds before the next whiteout.' }
      }
    }),
    rimewardenVault: createBossSpatialMechanicDefinition({
      id: 'rimewarden_vault_ice_wall_control',
      mapId: 'rimewardenVault',
      label: 'Whiteout Ice Wall Control',
      summary: 'Ice wall lane locks, oracle shelf whiteouts, and sentinel calls force Rimewarden parties to keep moving.',
      rewardAbuseControl: 'Wall, whiteout, and sentinel calls rotate across all vault sections instead of rewarding one shelf.',
      partyRoleHook: 'Frontline owns the brute lane, ranged handles oracle and sentinel shelves, and support calls safe resets.',
      hooks: {
        iceShockwave: { sectionId: 'rimewardenVault_brute_lane', role: 'frontline-ring', targetTier: 'ground', label: 'Brute Lane Ring', response: 'The frost ring expands through the brute lane.', objective: 'Jump or leave the brute lane before impact.' },
        iceWall: { sectionId: 'rimewardenVault_sentinel_shelf', role: 'ice-wall-lock', targetTier: 'high', label: 'Sentinel Ice Wall', response: 'An ice wall locks the sentinel shelf lane.', objective: 'Rotate around the ice wall lane lock.' },
        whiteout: { sectionId: 'rimewardenVault_oracle_shelf', role: 'whiteout-shelf', targetTier: 'mid', label: 'Oracle Whiteout', response: 'Whiteout sweeps the oracle shelf.', objective: 'Move oracle players before the shelf whites out.' },
        addWave: { sectionId: 'rimewardenVault_sentinel_shelf', role: 'sentinel-clear', targetTier: 'high', label: 'Sentinel Adds', response: 'Adds reinforce the sentinel shelf.', objective: 'Clear sentinel adds before walls overlap.' }
      }
    }),
    stormbreakAerie: createBossSpatialMechanicDefinition({
      id: 'stormbreak_aerie_rod_perch_control',
      mapId: 'stormbreakAerie',
      label: 'Storm Rod Perch Control',
      summary: 'Ram lane pressure, storm rod perches, and harrier airspace make Aurelion a vertical anti-air boss.',
      rewardAbuseControl: 'Rod perch and harrier calls prevent full parties from stacking on the floor during every sky pattern.',
      partyRoleHook: 'Melee clears ram lane and rod perch, ranged owns harrier airspace, and support calls wind lane crossings.',
      hooks: {
        windBolt: { sectionId: 'stormbreakAerie_harrier_airspace', role: 'anti-air', targetTier: 'high', label: 'Harrier Airspace', response: 'Wind bolts cut through the harrier airspace.', objective: 'Assign anti-air players to the upper lane.' },
        lightningRod: { sectionId: 'stormbreakAerie_rod_perch', role: 'rod-perch', targetTier: 'high', label: 'Storm Rod Perch', response: 'A storm rod perch grounds the lightning.', objective: 'Move to or away from the rod perch before the strike.' },
        windLane: { sectionId: 'stormbreakAerie_ram_lane', role: 'wind-crossing', targetTier: 'ground', label: 'Ram Lane Wind', response: 'Wind lanes sweep the ram floor.', objective: 'Cross the floor lane before wind pins the party.' },
        divebomb: { sectionId: 'stormbreakAerie_harrier_airspace', role: 'skyfall', targetTier: 'high', label: 'Skyfall Airspace', response: 'Divebomb shadows the harrier airspace.', objective: 'Drop or dodge before skyfall lands.' },
        addWave: { sectionId: 'stormbreakAerie_rod_perch', role: 'perch-clear', targetTier: 'high', label: 'Rod Perch Adds', response: 'Storm adds contest the rod perch.', objective: 'Clear the perch before the next lightning call.' }
      }
    }),
    astralStacks: createBossSpatialMechanicDefinition({
      id: 'astral_stacks_mirrored_archive_control',
      mapId: 'astralStacks',
      label: 'Mirrored Archive Control',
      summary: 'Mirrored archive shelves, rune calls, and memory seals create a readable left-center-right boss route.',
      rewardAbuseControl: 'Mirrored archive calls alternate stacks and rune shelf so repeated one-side camping loses control value.',
      partyRoleHook: 'One player calls left stacks, one handles right mirrored echoes, and support anchors the center rune shelf.',
      hooks: {
        runePages: { sectionId: 'astralStacks_center_rune_shelf', role: 'rune-call', targetTier: 'mid', label: 'Center Rune Shelf', response: 'Rune pages gather on the center shelf.', objective: 'Read and clear the center rune call.' },
        memorySeal: { sectionId: 'astralStacks_left_stacks', role: 'memory-seal', targetTier: 'high', label: 'Left Archive Seal', response: 'A memory seal closes the left mirrored archive.', objective: 'Rotate off the sealed stack before repeat damage.' },
        mirrorEcho: { sectionId: 'astralStacks_right_stacks', role: 'mirrored-echo', targetTier: 'high', label: 'Right Mirror Echo', response: 'Mirrored archive echoes replay on the right stacks.', objective: 'Call the echo lane and dodge the replay.' },
        addWave: { sectionId: 'astralStacks_left_stacks', role: 'scribe-clear', targetTier: 'high', label: 'Archive Scribe Adds', response: 'Scribes climb the mirrored archive stacks.', objective: 'Clear scribe adds before the next rune call.' }
      }
    }),
    eclipseThrone: createBossSpatialMechanicDefinition({
      id: 'eclipse_throne_solar_lunar_control',
      mapId: 'eclipseThrone',
      label: 'Solar Lunar Zone Rotation',
      summary: 'Solar lane, lunar lane, eclipse dais, and mote shelf calls turn the capstone into a zone-rotation fight.',
      rewardAbuseControl: 'Solar and lunar rotations alternate with totality so the safest lane cannot solve the whole room.',
      partyRoleHook: 'Full parties split solar and lunar duties, then regroup on the eclipse dais for totality.',
      hooks: {
        solarFlare: { sectionId: 'eclipseThrone_solar_lane', role: 'solar-zone', targetTier: 'ground', label: 'Solar Lane', response: 'Solar flare marks the solar lane as dangerous.', objective: 'Clear the Solar Lane before the flare blooms.' },
        lunarMark: { sectionId: 'eclipseThrone_lunar_lane', role: 'lunar-zone', targetTier: 'ground', label: 'Lunar Lane', response: 'Lunar mark calls a focused danger zone.', objective: 'Split clear of the Lunar Lane before the mark resolves.' },
        eclipseSigils: { sectionId: 'eclipseThrone_eclipse_dais', role: 'totality-regroup', targetTier: 'high', targetAnchor: 'platformCenter', label: 'Eclipse Dais', response: 'Totality leaves the central eclipse dais safe.', objective: 'Regroup inside the Eclipse Dais safe zone.' },
        addWave: { sectionId: 'eclipseThrone_mote_shelf', role: 'mote-clear', targetTier: 'high', label: 'Mote Shelf Adds', response: 'Motes reinforce the upper shelf during zone rotation.', objective: 'Clear mote shelf adds before totality.' }
      }
    })
  });

  const api = {
    BOSS_SPATIAL_MECHANICS,
    BOSS_SPATIAL_RESPONSE_CHECKS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.bossMechanics = Object.assign({}, modules.bossMechanics || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
