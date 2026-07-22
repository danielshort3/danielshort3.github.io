(function initProjectStarfallDataMapBuilders(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataMapGeometry = (typeof require === 'function' ? require('./map-geometry.js') : null) || DataModules.mapGeometry || {};
  const DataMapLayouts = (typeof require === 'function' ? require('./map-layouts.js') : null) || DataModules.mapLayouts || {};

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  const getPlatformDefX = DataMapGeometry.getPlatformDefX;
  const getPlatformDefY = DataMapGeometry.getPlatformDefY;
  const getPlatformDefW = DataMapGeometry.getPlatformDefW;
  const getPlatformDefShape = DataMapGeometry.getPlatformDefShape;
  const getPlatformDefVisualKind = DataMapGeometry.getPlatformDefVisualKind;
  const getPlatformDefRight = DataMapGeometry.getPlatformDefRight;
  const getPlatformDefSurfaceY = DataMapGeometry.getPlatformDefSurfaceY;
  const makePlatformDef = DataMapGeometry.makePlatformDef;
  const makeSlopePlatformDef = DataMapGeometry.makeSlopePlatformDef;

  const TRAINING_LANE_Y = DataMapLayouts.TRAINING_LANE_Y;
  const TOWN_LANE_Y = DataMapLayouts.TOWN_LANE_Y;
  const getMapGeometrySeed = DataMapLayouts.getMapGeometrySeed;
  const getPartyPlayZoneAnchors = DataMapLayouts.getPartyPlayZoneAnchors;
  const isVerticalFieldLayout = DataMapLayouts.isVerticalFieldLayout;
  const getFieldLaneY = DataMapLayouts.getFieldLaneY;
  const getFieldZoneAnchors = DataMapLayouts.getFieldZoneAnchors;
  const getFieldClimbableKind = DataMapLayouts.getFieldClimbableKind;
  const getVerticalFieldGeometry = DataMapLayouts.getVerticalFieldGeometry;
  const PRIORITY_FIELD_LAYOUT_IDS = DataMapLayouts.PRIORITY_FIELD_LAYOUT_IDS;
  const getDungeonArenaSkeleton = DataMapLayouts.getDungeonArenaSkeleton;
  const MIN_PARTY_TIER_GAP = 128;

  function shouldUseSlopePhase(settings, phase, clusterIndex, cluster) {
    const plan = settings && settings.slopePlan;
    if (!plan || !Object.prototype.hasOwnProperty.call(plan, phase)) return true;
    const rule = plan[phase];
    if (typeof rule === 'function') return !!rule(clusterIndex, cluster);
    if (Array.isArray(rule)) return rule.includes(clusterIndex);
    return !!rule;
  }

  function makeClimbableBetweenPlatforms(prefix, platforms, topIndex, bottomIndex, key, kind) {
    const top = platforms[topIndex];
    const bottom = platforms[bottomIndex];
    if (!top || !bottom) return null;
    const trueOverlapLeft = Math.max(getPlatformDefX(top), getPlatformDefX(bottom));
    const trueOverlapRight = Math.min(getPlatformDefRight(top), getPlatformDefRight(bottom));
    const paddedOverlapLeft = Math.max(getPlatformDefX(top) + 54, getPlatformDefX(bottom) + 54);
    const paddedOverlapRight = Math.min(getPlatformDefRight(top) - 54, getPlatformDefRight(bottom) - 54);
    const rawX = paddedOverlapLeft <= paddedOverlapRight
      ? (paddedOverlapLeft + paddedOverlapRight) / 2
      : trueOverlapLeft <= trueOverlapRight
        ? (trueOverlapLeft + trueOverlapRight) / 2
        : getPlatformDefX(top) + getPlatformDefW(top) / 2;
    const topY = getPlatformDefSurfaceY(top, rawX);
    const bottomY = getPlatformDefSurfaceY(bottom, rawX);
    if (bottomY <= topY) return null;
    const width = kind === 'stair' ? 46 : 30;
    return {
      id: `${prefix}_${kind}_${key}`,
      x: Math.round(rawX - width / 2),
      y: Math.round(topY),
      w: width,
      h: Math.max(1, Math.round(bottomY - topY))
    };
  }

  function makePartyPlayPlatforms(width, options) {
    const worldWidth = Math.max(3600, Math.ceil(Number(width || 0) / 100) * 100);
    const settings = options || {};
    const variantSeed = getMapGeometrySeed(settings.variantKey || '');
    const laneProfile = settings.dungeon ? variantSeed % 4 : 0;
    const lanes = Object.freeze(Object.assign({}, TRAINING_LANE_Y, {
      low: TRAINING_LANE_Y.low + [0, -26, 22, -12][laneProfile],
      mid: TRAINING_LANE_Y.mid + [0, -18, 22, -12][laneProfile],
      high: TRAINING_LANE_Y.high + [0, -24, 24, -8][laneProfile],
      lowConnector: TRAINING_LANE_Y.lowConnector + [0, -20, 32, -12][laneProfile],
      highConnector: TRAINING_LANE_Y.highConnector + [0, -21, 23, -11][laneProfile]
    }));
    const anchors = getPartyPlayZoneAnchors(worldWidth, options);
    const platforms = [makePlatformDef(0, 520, worldWidth, 80, { kind: 'ground' })];
    const addPlatform = (x, y, w, visualKind) => {
      const widthLimit = Math.min(w, worldWidth - x - 220);
      if (widthLimit >= 120) platforms.push(makePlatformDef(x, y, widthLimit, visualKind === 'hop' ? 20 : 22, { kind: visualKind || 'solidLane' }));
    };
    const addSlope = (x, y, y2, w, visualKind) => {
      const widthLimit = Math.min(w, worldWidth - x - 220);
      if (widthLimit >= 180) platforms.push(makeSlopePlatformDef(x, y, y2, widthLimit, 24, { kind: visualKind || 'slope' }));
    };
    anchors.forEach((anchor, index) => {
      const drift = index % 2 ? 80 : 0;
      const lowShift = variantSeed ? ((variantSeed + index * 17) % 5 - 2) * 18 : 0;
      const midShift = variantSeed ? ((Math.floor(variantSeed / 5) + index * 13) % 5 - 2) * 18 : 0;
      const highShift = variantSeed ? ((Math.floor(variantSeed / 25) + index * 11) % 5 - 2) * 16 : 0;
      const hopShift = variantSeed ? ((Math.floor(variantSeed / 125) + index * 7) % 5 - 2) * 18 : 0;
      const zoneShift = settings.dungeon && variantSeed ? ((variantSeed + index * 29) % 7 - 3) * 16 : 0;
      const lowW = clamp(720 - lowShift + (settings.dungeon ? 36 : 0), 660, settings.dungeon ? 820 : 780);
      const midW = clamp(720 - midShift * 0.6, 660, 780);
      const highW = clamp(680 - highShift * 0.6, 640, 740);
      addPlatform(anchor + drift + lowShift + zoneShift, lanes.low, lowW, 'solidLane');
      addSlope(anchor + 740 + drift * 0.4 + zoneShift * 0.35, lanes.low, lanes.lowConnector, 280, 'slope');
      addPlatform(anchor + 1040 - drift * 0.25 + zoneShift * 0.3, lanes.lowConnector, 220, 'connector');
      if (!settings.dungeon) addSlope(anchor + 1040 - drift * 0.25 + zoneShift * 0.3, lanes.lowConnector, lanes.mid, 200, 'slope');
      addPlatform(anchor + 900 - drift * 0.35 + midShift - zoneShift * 0.25, lanes.mid, midW, 'solidLane');
      addPlatform(anchor + 600 + drift * 0.2 + zoneShift * 0.2, lanes.highConnector, 220, 'connector');
      if (!settings.dungeon) addSlope(anchor + 680 + drift * 0.18 + zoneShift * 0.2, lanes.high, lanes.mid, 280, 'slope');
      addPlatform(anchor + 320 + drift * 0.15 + highShift + zoneShift * 0.45, lanes.high, highW, 'solidLane');
      addPlatform(anchor + 1500 - drift * 0.4 + hopShift - zoneShift * 0.25, lanes.mid - 52, 240, 'hop');
    });
    return platforms;
  }

  function makePartyPlayClimbables(prefix, widthOrPlatforms, options) {
    if (Array.isArray(widthOrPlatforms)) {
      const platforms = widthOrPlatforms;
      const climbables = [];
      for (let zoneStart = 1; zoneStart < platforms.length; zoneStart += 9) {
        [
          [zoneStart, 0, 'low'],
          [zoneStart + 7, zoneStart + 4, 'high']
        ].forEach((pair) => {
          const climbable = makeClimbableBetweenPlatforms(prefix, platforms, pair[0], pair[1], `party_${Math.floor(zoneStart / 9) + 1}_${pair[2]}`, 'lift');
          if (climbable) climbables.push(climbable);
        });
      }
      return climbables;
    }
    const worldWidth = Math.max(3600, Number(widthOrPlatforms || 0));
    return makePartyPlayClimbables(prefix, makePartyPlayPlatforms(worldWidth, options), options);
  }

  function makePartyPlaySpawnPoints(platforms) {
    return platforms
      .map((platform, index) => ({ platform, index }))
      .filter((entry) => entry.index > 0 && getPlatformDefW(entry.platform) >= 640)
      .map((entry) => ({
        x: Math.round(getPlatformDefX(entry.platform) + getPlatformDefW(entry.platform) / 2),
        platformIndex: entry.index,
        weight: getPlatformDefY(entry.platform) >= 430 ? 3 : getPlatformDefY(entry.platform) >= 300 ? 2 : 1
      }));
  }

  function makeDungeonArenaPlatforms(width, mapId) {
    const skeleton = getDungeonArenaSkeleton(mapId);
    if (!skeleton) return null;
    const worldWidth = Math.max(4600, Math.ceil(Number(width || 0) / 100) * 100);
    const lowY = TRAINING_LANE_Y.low + Number(skeleton.lowShift || 0);
    let midY = TRAINING_LANE_Y.mid + Number(skeleton.midShift || 0);
    let highY = TRAINING_LANE_Y.high + Number(skeleton.highShift || 0);
    midY = Math.min(midY, lowY - MIN_PARTY_TIER_GAP);
    highY = Math.min(highY, midY - MIN_PARTY_TIER_GAP);
    const platforms = [makePlatformDef(0, TRAINING_LANE_Y.ground, worldWidth, 80, { kind: 'ground' })];
    const addPlatform = (x, y, w, visualKind) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW < 120) return;
      platforms.push(makePlatformDef(safeX, Math.round(y), safeW, visualKind === 'hop' ? 20 : 22, { kind: visualKind || 'solidLane' }));
    };
    const addSlope = (x, y, y2, w) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW < 180) return;
      platforms.push(makeSlopePlatformDef(safeX, Math.round(y), Math.round(y2), safeW, 24, { kind: 'slope' }));
    };
    const addTransitionSlope = (fromX, fromY, fromW, toX, toY, toW, rampW) => {
      const fromLeft = Number(fromX || 0);
      const fromRight = fromLeft + Number(fromW || 0);
      const toLeft = Number(toX || 0);
      const toRight = toLeft + Number(toW || 0);
      const width = Number(rampW || 0);
      if (width < 180) return;
      if (toLeft >= fromLeft) {
        const startMin = Math.max(fromLeft, toLeft - width);
        const startMax = Math.min(fromRight, toRight - width);
        addSlope(clamp(toLeft - width, startMin, startMax), fromY, toY, width);
      } else {
        const startMin = Math.max(toLeft, fromLeft - width);
        const startMax = Math.min(toRight, fromRight - width);
        addSlope(clamp(toRight, startMin, startMax), toY, fromY, width);
      }
    };
    const buildBranch = (lowX, midX, highX, widths, side) => {
      const rampW = side === 'right' ? 280 : 260;
      addTransitionSlope(0, TRAINING_LANE_Y.ground, worldWidth, lowX, lowY, widths.low, rampW);
      addPlatform(lowX, lowY, widths.low, 'solidLane');
      addTransitionSlope(lowX, lowY, widths.low, midX, midY, widths.mid, rampW);
      addPlatform(midX, midY, widths.mid, 'solidLane');
      addPlatform(highX, highY, widths.high, 'solidLane');
    };
    const leftLow = Number(skeleton.left || 300);
    const rightLow = Number(skeleton.right || 2400);
    buildBranch(
      leftLow,
      leftLow + Number(skeleton.midInset || 280),
      leftLow + Number(skeleton.highInset || 600),
      { low: Number(skeleton.lowW || 940), mid: Number(skeleton.midW || 800), high: Number(skeleton.highW || 700) },
      'left'
    );
    buildBranch(
      rightLow,
      rightLow + Number(skeleton.rightMidInset || 280),
      rightLow + Number(skeleton.rightHighInset || 120),
      { low: Number(skeleton.lowW || 940), mid: Number(skeleton.midW || 800), high: Number(skeleton.highW || 700) },
      'right'
    );
    addPlatform(1880, TRAINING_LANE_Y.lowConnector + Number(skeleton.lowShift || 0) * 0.3, 260, 'connector');
    addPlatform(1740, TRAINING_LANE_Y.highConnector + Number(skeleton.midShift || 0) * 0.3, 260, 'connector');
    addPlatform(2240, TRAINING_LANE_Y.highConnector + Number(skeleton.highShift || 0) * 0.3, 260, 'connector');
    addPlatform(2040, highY - 62, 280, 'hop');
    return platforms;
  }

  function makeBanditRidgeCampPlatforms(width) {
    const worldWidth = Math.max(5400, Math.ceil(Number(width || 0) / 100) * 100);
    const sectionIds = Object.freeze({
      lower: 'banditRidgeCamp_lower_cutter_lane',
      middle: 'banditRidgeCamp_middle_thrower_camp',
      high: 'banditRidgeCamp_high_rope_bridge',
      regroup: 'banditRidgeCamp_campfire_regroup'
    });
    const makeNamedFlat = (id, sectionId, x, y, w, routeRole, visualKind, variant) => Object.assign(
      makePlatformDef(x, y, w, visualKind === 'hop' ? 20 : 22, {
        kind: visualKind || 'solidLane',
        variant: variant || routeRole
      }),
      { id, sectionId, routeRole }
    );
    const makeNamedSlope = (id, sectionId, x, y, y2, w, routeRole) => Object.assign(
      makeSlopePlatformDef(x, y, y2, w, 24, { kind: 'slope', variant: routeRole }),
      { id, sectionId, routeRole }
    );

    return [
      makeNamedFlat('bandit_ridge_camp_ground', '', 0, 520, worldWidth, 'ground-throughline', 'ground'),

      // Lower Cutter Approach: a safe arrival, a broad melee lane, and one elevated bypass.
      makeNamedSlope('bandit_ridge_camp_lower_approach_slope', sectionIds.lower, 60, 520, 456, 260, 'entry-approach'),
      makeNamedFlat('bandit_ridge_camp_lower_barricade_lane', sectionIds.lower, 320, 456, 820, 'frontline-barricade', 'solidLane', 'dirt-terrace'),
      makeNamedFlat('bandit_ridge_camp_lower_flank', sectionIds.lower, 440, 324, 720, 'frontline-flank', 'solidLane', 'wood-scaffold'),
      makeNamedFlat('bandit_ridge_camp_lower_barricade_step', sectionIds.lower, 170, 390, 180, 'barricade-step', 'connector', 'wood-step'),
      makeNamedFlat('bandit_ridge_camp_lower_bypass_step', sectionIds.lower, 1050, 390, 190, 'flank-step', 'connector', 'wood-step'),

      // Middle Thrower Camp: two staggered perches with a main deck and alternate rope access.
      makeNamedFlat('bandit_ridge_camp_thrower_deck', sectionIds.middle, 1370, 456, 1010, 'thrower-camp-deck', 'solidLane', 'camp-deck'),
      makeNamedFlat('bandit_ridge_camp_thrower_perch_west', sectionIds.middle, 1450, 324, 720, 'thrower-perch-west', 'solidLane', 'lookout-deck'),
      makeNamedFlat('bandit_ridge_camp_thrower_perch_east', sectionIds.middle, 1990, 194, 650, 'thrower-perch-east', 'solidLane', 'lookout-deck'),
      makeNamedSlope('bandit_ridge_camp_thrower_camp_slope', sectionIds.middle, 2120, 324, 456, 250, 'camp-deck-ramp'),
      makeNamedFlat('bandit_ridge_camp_thrower_crate_step', sectionIds.middle, 1260, 390, 180, 'camp-flank-step', 'connector', 'crate-step'),
      makeNamedFlat('bandit_ridge_camp_thrower_watch_step', sectionIds.middle, 2430, 259, 220, 'lookout-step', 'connector', 'wood-step'),

      // High Rope Bridge: a distinct bridge line with two approach ropes and a drop return below it.
      makeNamedFlat('bandit_ridge_camp_bridge_approach', sectionIds.high, 2780, 324, 720, 'bridge-approach', 'solidLane', 'watch-terrace'),
      Object.assign(
        makeNamedFlat('bandit_ridge_camp_rope_bridge', sectionIds.high, 2920, 194, 960, 'rope-bridge-drop-loop', 'solidLane', 'rope-bridge'),
        { dropShortcut: true, dropTargetPlatformId: 'bandit_ridge_camp_bridge_return_lane' }
      ),
      makeNamedFlat('bandit_ridge_camp_bridge_return_lane', sectionIds.high, 3200, 456, 760, 'bridge-drop-return', 'solidLane', 'dirt-return'),
      makeNamedSlope('bandit_ridge_camp_bridge_return_slope', sectionIds.high, 2920, 520, 456, 280, 'return-lane-approach'),
      makeNamedFlat('bandit_ridge_camp_bridge_west_tower_step', sectionIds.high, 2700, 259, 200, 'bridge-west-tower', 'connector', 'tower-step'),
      makeNamedFlat('bandit_ridge_camp_bridge_drop_step', sectionIds.high, 3540, 390, 200, 'bridge-drop-marker', 'connector', 'barricade-step'),
      makeNamedFlat('bandit_ridge_camp_bridge_east_tower_step', sectionIds.high, 3830, 259, 180, 'bridge-east-tower', 'connector', 'tower-step'),

      // Campfire Regroup: broad safe footing, a gate lookout, and no routine spawn ownership.
      makeNamedFlat('bandit_ridge_camp_regroup_plateau', sectionIds.regroup, 4170, 456, 900, 'campfire-safe-plateau', 'solidLane', 'campfire-terrace'),
      makeNamedFlat('bandit_ridge_camp_exit_lookout', sectionIds.regroup, 4440, 324, 600, 'exit-lookout', 'solidLane', 'lookout-deck'),
      makeNamedFlat('bandit_ridge_camp_exit_gate_perch', sectionIds.regroup, 4760, 194, 480, 'dungeon-gate-perch', 'solidLane', 'gate-perch'),
      makeNamedSlope('bandit_ridge_camp_regroup_slope', sectionIds.regroup, 4050, 520, 456, 260, 'regroup-approach'),
      makeNamedFlat('bandit_ridge_camp_regroup_gate_step', sectionIds.regroup, 5060, 390, 200, 'dungeon-gate-step', 'connector', 'gate-step')
    ];
  }

  function makeBanditRidgeCampClimbables(platforms) {
    const source = Array.isArray(platforms) ? platforms : [];
    const platformIndexById = new Map(source.map((platform, index) => [String(platform && platform.id || ''), index]));
    const connect = (topId, bottomId, key) => makeClimbableBetweenPlatforms(
      'banditRidgeCamp',
      source,
      platformIndexById.get(topId),
      platformIndexById.get(bottomId),
      key,
      'rope'
    );
    return [
      connect('bandit_ridge_camp_lower_barricade_lane', 'bandit_ridge_camp_ground', 'lower_entry'),
      connect('bandit_ridge_camp_lower_flank', 'bandit_ridge_camp_lower_barricade_lane', 'lower_flank'),
      connect('bandit_ridge_camp_thrower_deck', 'bandit_ridge_camp_ground', 'thrower_deck'),
      connect('bandit_ridge_camp_thrower_perch_west', 'bandit_ridge_camp_thrower_deck', 'thrower_west'),
      connect('bandit_ridge_camp_thrower_perch_east', 'bandit_ridge_camp_thrower_perch_west', 'thrower_east_main'),
      connect('bandit_ridge_camp_thrower_perch_east', 'bandit_ridge_camp_thrower_deck', 'thrower_east_flank'),
      connect('bandit_ridge_camp_bridge_approach', 'bandit_ridge_camp_ground', 'bridge_approach'),
      connect('bandit_ridge_camp_rope_bridge', 'bandit_ridge_camp_bridge_approach', 'bridge_west'),
      connect('bandit_ridge_camp_rope_bridge', 'bandit_ridge_camp_bridge_return_lane', 'bridge_east'),
      connect('bandit_ridge_camp_bridge_return_lane', 'bandit_ridge_camp_ground', 'bridge_return'),
      connect('bandit_ridge_camp_regroup_plateau', 'bandit_ridge_camp_ground', 'regroup'),
      connect('bandit_ridge_camp_exit_lookout', 'bandit_ridge_camp_regroup_plateau', 'exit_lookout'),
      connect('bandit_ridge_camp_exit_gate_perch', 'bandit_ridge_camp_exit_lookout', 'exit_gate')
    ].filter(Boolean);
  }

  function makeThornpathFractureCanopyPlatforms(width) {
    const worldWidth = Math.max(5200, Math.ceil(Number(width || 0) / 100) * 100);
    const lanes = getFieldLaneY('verticalCanopy');
    const sectionIds = Object.freeze({
      return: 'thornpathThicket_meadow_return',
      canopy: 'thornpathThicket_fracture_canopy',
      fork: 'thornpathThicket_observatory_fork'
    });
    const makeNamedFlat = (id, sectionId, x, y, w, routeRole, visualKind, variant) => Object.assign(
      makePlatformDef(x, y, w, visualKind === 'hop' ? 20 : 22, {
        kind: visualKind || 'solidLane',
        variant: variant || routeRole
      }),
      { id, sectionId, routeRole }
    );
    const makeNamedSlope = (id, sectionId, x, y, y2, w, routeRole) => Object.assign(
      makeSlopePlatformDef(x, y, y2, w, 24, { kind: 'slope', variant: routeRole }),
      { id, sectionId, routeRole }
    );

    return [
      makeNamedFlat('thornpath_fracture_canopy_ground', '', 0, lanes.ground, worldWidth, 'ground-throughline', 'ground'),

      // Meadow Return: introduce the broken starstone roots before the route commits upward.
      makeNamedSlope('thornpath_rootfall_entry_slope', sectionIds.return, 80, lanes.ground, lanes.low, 260, 'rootfall-entry'),
      makeNamedFlat('thornpath_rootfall_lane', sectionIds.return, 340, lanes.low, 820, 'rootfall-frontline', 'solidLane', 'starstone-root'),
      makeNamedFlat('thornpath_rootfall_relay_shelf', sectionIds.return, 180, lanes.mid, 650, 'rootfall-relay', 'solidLane', 'fractured-bough'),
      makeNamedFlat('thornpath_rootfall_overlook', sectionIds.return, 470, lanes.high, 680, 'rootfall-overlook', 'solidLane', 'relay-canopy'),
      makeNamedFlat('thornpath_rootfall_step', sectionIds.return, 760, lanes.lowConnector, 180, 'rootfall-step', 'connector', 'root-step'),
      makeNamedFlat('thornpath_rootfall_shard_step', sectionIds.return, 930, lanes.highConnector, 170, 'rootfall-shard-step', 'connector', 'starstone-step'),

      // Fracture Canopy: four offset tiers create a readable ascent and a deliberate drop reset.
      makeNamedSlope('thornpath_relay_entry_slope', sectionIds.canopy, 1260, lanes.ground, lanes.low, 260, 'relay-entry'),
      makeNamedFlat('thornpath_relay_lower_walk', sectionIds.canopy, 1520, lanes.low, 930, 'relay-lower-control', 'solidLane', 'rootwalk'),
      makeNamedFlat('thornpath_relay_mid_deck', sectionIds.canopy, 1780, lanes.mid, 1050, 'relay-mid-control', 'solidLane', 'suspended-relay'),
      makeNamedFlat('thornpath_relay_high_bough', sectionIds.canopy, 2240, lanes.high, 980, 'relay-high-control', 'solidLane', 'fractured-bough'),
      Object.assign(
        makeNamedFlat('thornpath_relay_shard_perch', sectionIds.canopy, 2600, lanes.peak, 740, 'relay-drop-reset', 'solidLane', 'starstone-perch'),
        { dropShortcut: true, dropTargetPlatformId: 'thornpath_relay_mid_deck' }
      ),
      makeNamedFlat('thornpath_relay_lower_step', sectionIds.canopy, 2230, lanes.lowConnector, 180, 'relay-lower-step', 'connector', 'root-step'),
      makeNamedFlat('thornpath_relay_shard_step', sectionIds.canopy, 2890, lanes.highConnector, 180, 'relay-shard-step', 'connector', 'starstone-step'),

      // Observatory Fork: the right branch advances to Bandit Ridge; the upper-left relay reaches Rustcoil.
      makeNamedSlope('thornpath_fork_entry_slope', sectionIds.fork, 3460, lanes.ground, lanes.low, 260, 'fork-entry'),
      makeNamedFlat('thornpath_fork_lower_lane', sectionIds.fork, 3720, lanes.low, 920, 'fork-regroup', 'solidLane', 'fork-rootwalk'),
      makeNamedFlat('thornpath_fork_ridge_branch', sectionIds.fork, 4300, lanes.mid, 760, 'ridge-fracture-branch', 'solidLane', 'ridge-beacon'),
      makeNamedFlat('thornpath_fork_observatory_branch', sectionIds.fork, 3560, lanes.high, 720, 'observatory-relay-branch', 'solidLane', 'observatory-relay'),
      makeNamedFlat('thornpath_fork_beacon_perch', sectionIds.fork, 4400, lanes.peak, 600, 'fork-beacon-overlook', 'solidLane', 'starstone-beacon'),
      makeNamedFlat('thornpath_fork_root_step', sectionIds.fork, 4000, lanes.lowConnector, 180, 'fork-root-step', 'connector', 'root-step'),
      makeNamedFlat('thornpath_fork_shard_step', sectionIds.fork, 4210, lanes.highConnector, 170, 'fork-shard-step', 'connector', 'starstone-step')
    ];
  }

  function makeThornpathFractureCanopyClimbables(platforms) {
    const source = Array.isArray(platforms) ? platforms : [];
    const platformIndexById = new Map(source.map((platform, index) => [String(platform && platform.id || ''), index]));
    const connect = (topId, bottomId, key) => makeClimbableBetweenPlatforms(
      'thornpathThicket',
      source,
      platformIndexById.get(topId),
      platformIndexById.get(bottomId),
      key,
      'vine'
    );
    return [
      connect('thornpath_rootfall_relay_shelf', 'thornpath_rootfall_lane', 'rootfall_relay'),
      connect('thornpath_rootfall_overlook', 'thornpath_rootfall_relay_shelf', 'rootfall_overlook'),
      connect('thornpath_relay_mid_deck', 'thornpath_relay_lower_walk', 'relay_mid'),
      connect('thornpath_relay_high_bough', 'thornpath_relay_mid_deck', 'relay_high'),
      connect('thornpath_relay_shard_perch', 'thornpath_relay_high_bough', 'relay_perch'),
      connect('thornpath_relay_shard_perch', 'thornpath_relay_mid_deck', 'relay_perch_flank'),
      connect('thornpath_fork_ridge_branch', 'thornpath_fork_lower_lane', 'ridge_branch'),
      connect('thornpath_fork_observatory_branch', 'thornpath_fork_lower_lane', 'observatory_branch')
    ].filter(Boolean);
  }

  function makeFrostfenMarshRunPlatforms(width) {
    const worldWidth = Math.max(5600, Math.ceil(Number(width || 0) / 100) * 100);
    const lanes = getFieldLaneY('switchbackTerraces');
    const sectionIds = Object.freeze({
      marsh: 'frostfenOutskirts_frozen_marsh',
      shelf: 'frostfenOutskirts_rimeglass_shelf',
      grove: 'frostfenOutskirts_oracle_grove'
    });
    const makeNamedFlat = (id, sectionId, x, y, w, routeRole, visualKind, variant) => Object.assign(
      makePlatformDef(x, y, w, visualKind === 'hop' ? 20 : 22, {
        kind: visualKind || 'solidLane',
        variant: variant || routeRole
      }),
      { id, sectionId, routeRole }
    );
    const makeNamedSlope = (id, sectionId, x, y, y2, w, routeRole) => Object.assign(
      makeSlopePlatformDef(x, y, y2, w, 24, { kind: 'slope', variant: routeRole }),
      { id, sectionId, routeRole }
    );

    return [
      makeNamedFlat('frostfen_marsh_run_ground', '', 0, lanes.ground, worldWidth, 'ground-throughline', 'ground'),

      // Frozen Marsh: a long low runway makes the ice profile readable before the route climbs.
      makeNamedSlope('frostfen_marsh_entry_slope', sectionIds.marsh, 80, lanes.ground, lanes.low, 300, 'marsh-entry'),
      makeNamedFlat('frostfen_marsh_runway', sectionIds.marsh, 380, lanes.low, 920, 'marsh-slide-runway', 'solidLane', 'rime-runway'),
      makeNamedFlat('frostfen_marsh_windbreak', sectionIds.marsh, 180, lanes.mid, 720, 'marsh-windbreak', 'solidLane', 'snowbreak-shelf'),
      makeNamedFlat('frostfen_marsh_signal_overlook', sectionIds.marsh, 520, lanes.high, 700, 'signal-overlook', 'solidLane', 'signal-wreck'),
      makeNamedFlat('frostfen_marsh_runout_step', sectionIds.marsh, 230, lanes.lowConnector, 180, 'marsh-runout-step', 'connector', 'ice-step'),
      makeNamedFlat('frostfen_marsh_signal_step', sectionIds.marsh, 980, lanes.highConnector, 180, 'signal-step', 'connector', 'rime-step'),

      // Rimeglass Shelf: offset shelves create a parallel upper route and a sheltered recovery pocket.
      makeNamedSlope('frostfen_shelf_entry_slope', sectionIds.shelf, 1520, lanes.ground, lanes.low, 320, 'shelf-entry'),
      makeNamedFlat('frostfen_shelf_lower_run', sectionIds.shelf, 1840, lanes.low, 1120, 'shelf-lower-run', 'solidLane', 'packed-snow-run'),
      makeNamedFlat('frostfen_rimeglass_shelf', sectionIds.shelf, 1660, lanes.mid, 1040, 'rimeglass-main-shelf', 'solidLane', 'rimeglass-shelf'),
      makeNamedFlat('frostfen_shelf_upper_drift', sectionIds.shelf, 2300, lanes.high, 1040, 'shelf-upper-route', 'solidLane', 'wind-carved-drift'),
      makeNamedFlat('frostfen_shelf_shelter_pocket', sectionIds.shelf, 2860, lanes.mid, 720, 'shelf-shelter-pocket', 'solidLane', 'snowbreak-pocket'),
      makeNamedFlat('frostfen_shelf_lower_step', sectionIds.shelf, 2780, lanes.lowConnector, 180, 'shelf-lower-step', 'connector', 'ice-step'),
      makeNamedFlat('frostfen_shelf_upper_step', sectionIds.shelf, 2160, lanes.highConnector, 180, 'shelf-upper-step', 'connector', 'rime-step'),

      // Oracle Grove: the bloom perch overlaps a broad lower recovery lane for a deliberate drop reset.
      makeNamedSlope('frostfen_grove_entry_slope', sectionIds.grove, 3720, lanes.ground, lanes.low, 300, 'grove-entry'),
      makeNamedFlat('frostfen_oracle_recovery_run', sectionIds.grove, 4020, lanes.low, 1180, 'oracle-drop-recovery', 'solidLane', 'grove-recovery-run'),
      makeNamedFlat('frostfen_oracle_grove_shelf', sectionIds.grove, 3820, lanes.mid, 920, 'oracle-control-shelf', 'solidLane', 'oracle-grove'),
      Object.assign(
        makeNamedFlat('frostfen_oracle_bloom_perch', sectionIds.grove, 4440, lanes.high, 760, 'oracle-drop-reset', 'solidLane', 'icebloom-perch'),
        { dropShortcut: true, dropTargetPlatformId: 'frostfen_oracle_recovery_run' }
      ),
      makeNamedFlat('frostfen_oracle_exit_shelf', sectionIds.grove, 4780, lanes.mid, 660, 'glacier-exit-shelf', 'solidLane', 'rime-relay-exit'),
      makeNamedFlat('frostfen_grove_lower_step', sectionIds.grove, 4280, lanes.lowConnector, 180, 'grove-lower-step', 'connector', 'ice-step'),
      makeNamedFlat('frostfen_grove_bloom_step', sectionIds.grove, 4760, lanes.highConnector, 180, 'grove-bloom-step', 'connector', 'rime-step')
    ];
  }

  function makeFrostfenMarshRunClimbables(platforms) {
    const source = Array.isArray(platforms) ? platforms : [];
    const platformIndexById = new Map(source.map((platform, index) => [String(platform && platform.id || ''), index]));
    const connect = (topId, bottomId, key) => makeClimbableBetweenPlatforms(
      'frostfenOutskirts',
      source,
      platformIndexById.get(topId),
      platformIndexById.get(bottomId),
      key,
      'frost_ladder'
    );
    return [
      connect('frostfen_marsh_windbreak', 'frostfen_marsh_runway', 'marsh_windbreak'),
      connect('frostfen_marsh_signal_overlook', 'frostfen_marsh_windbreak', 'marsh_signal'),
      connect('frostfen_rimeglass_shelf', 'frostfen_shelf_lower_run', 'rimeglass_shelf'),
      connect('frostfen_shelf_upper_drift', 'frostfen_rimeglass_shelf', 'upper_drift'),
      connect('frostfen_oracle_grove_shelf', 'frostfen_oracle_recovery_run', 'oracle_shelf'),
      connect('frostfen_oracle_bloom_perch', 'frostfen_oracle_grove_shelf', 'oracle_bloom')
    ].filter(Boolean);
  }

  function makeRustcoilOrreryCircuitPlatforms(width) {
    const worldWidth = Math.max(5200, Math.ceil(Number(width || 0) / 100) * 100);
    const lanes = getFieldLaneY('industrialStack');
    const sectionIds = Object.freeze({
      yard: 'rustcoilRuins_surveyor_yard',
      switchworks: 'rustcoilRuins_coil_switchworks',
      gearwell: 'rustcoilRuins_warden_gearwell'
    });
    const makeNamedFlat = (id, sectionId, x, y, w, routeRole, visualKind, variant) => Object.assign(
      makePlatformDef(x, y, w, visualKind === 'hop' ? 20 : 22, {
        kind: visualKind || 'solidLane',
        variant: variant || routeRole
      }),
      { id, sectionId, routeRole }
    );
    const makeNamedSlope = (id, sectionId, x, y, y2, w, routeRole) => Object.assign(
      makeSlopePlatformDef(x, y, y2, w, 24, { kind: 'slope', variant: routeRole }),
      { id, sectionId, routeRole }
    );

    return [
      makeNamedFlat('rustcoil_orrery_circuit_ground', '', 0, lanes.ground, worldWidth, 'ground-throughline', 'ground'),

      // Surveyor Yard: a clear quest arrival flows into the first broken orrery loop.
      makeNamedSlope('rustcoil_yard_entry_slope', sectionIds.yard, 680, lanes.ground, lanes.low, 260, 'surveyor-yard-entry'),
      makeNamedFlat('rustcoil_yard_ratchet_lane', sectionIds.yard, 700, lanes.low, 760, 'yard-ratchet-lane', 'solidLane', 'ratchet-track'),
      makeNamedFlat('rustcoil_yard_service_gantry', sectionIds.yard, 420, lanes.mid, 800, 'yard-service-gantry', 'solidLane', 'service-gantry'),
      makeNamedFlat('rustcoil_yard_broken_orrery', sectionIds.yard, 720, lanes.high, 720, 'yard-orrery-overlook', 'solidLane', 'broken-orrery'),
      makeNamedFlat('rustcoil_yard_ratchet_step', sectionIds.yard, 1170, lanes.lowConnector, 180, 'yard-ratchet-step', 'connector', 'gear-step'),
      makeNamedFlat('rustcoil_yard_orrery_step', sectionIds.yard, 1210, lanes.highConnector, 180, 'yard-orrery-step', 'connector', 'orrery-step'),

      // Coil Switchworks: parallel catwalks make a readable loop around the central conveyor.
      makeNamedSlope('rustcoil_switchworks_entry_slope', sectionIds.switchworks, 1510, lanes.ground, lanes.low, 260, 'switchworks-entry'),
      makeNamedFlat('rustcoil_switchworks_conveyor', sectionIds.switchworks, 1770, lanes.low, 1000, 'switchworks-conveyor', 'solidLane', 'coil-conveyor'),
      makeNamedFlat('rustcoil_switchworks_west_catwalk', sectionIds.switchworks, 1530, lanes.mid, 940, 'switchworks-west-catwalk', 'solidLane', 'switch-catwalk'),
      makeNamedFlat('rustcoil_switchworks_return_deck', sectionIds.switchworks, 2300, lanes.mid, 1000, 'switchworks-return-deck', 'solidLane', 'return-deck'),
      makeNamedFlat('rustcoil_switchworks_east_catwalk', sectionIds.switchworks, 2440, lanes.high, 800, 'switchworks-east-catwalk', 'solidLane', 'sentry-catwalk'),
      makeNamedFlat('rustcoil_switchworks_conveyor_step', sectionIds.switchworks, 2270, lanes.lowConnector, 180, 'switchworks-conveyor-step', 'connector', 'coil-step'),
      makeNamedFlat('rustcoil_switchworks_catwalk_step', sectionIds.switchworks, 2260, lanes.highConnector, 180, 'switchworks-catwalk-step', 'connector', 'relay-step'),

      // Warden Gearwell: climb the gear ring and relay dais, then drop from the starcoil to reset.
      makeNamedSlope('rustcoil_warden_entry_slope', sectionIds.gearwell, 3410, lanes.ground, lanes.low, 260, 'warden-gearwell-entry'),
      makeNamedFlat('rustcoil_warden_return_belt', sectionIds.gearwell, 3670, lanes.low, 1040, 'warden-drop-return', 'solidLane', 'return-belt'),
      makeNamedFlat('rustcoil_warden_gear_ring', sectionIds.gearwell, 3440, lanes.mid, 820, 'warden-gear-ring', 'solidLane', 'gear-ring'),
      makeNamedFlat('rustcoil_warden_relay_dais', sectionIds.gearwell, 4200, lanes.high, 800, 'warden-relay-dais', 'solidLane', 'relay-dais'),
      Object.assign(
        makeNamedFlat('rustcoil_warden_starcoil_perch', sectionIds.gearwell, 3940, lanes.peak, 780, 'warden-starcoil-drop', 'solidLane', 'starcoil-perch'),
        { dropShortcut: true, dropTargetPlatformId: 'rustcoil_warden_return_belt' }
      ),
      makeNamedFlat('rustcoil_warden_belt_step', sectionIds.gearwell, 4700, lanes.lowConnector, 180, 'warden-belt-step', 'connector', 'gear-step'),
      makeNamedFlat('rustcoil_warden_relay_step', sectionIds.gearwell, 4760, lanes.highConnector, 180, 'warden-relay-step', 'connector', 'relay-step'),
      Object.assign(
        makeNamedFlat('rustcoil_warden_starcoil_service_bridge', sectionIds.gearwell, 4300, 490, 600, 'warden-starcoil-service-bridge', 'connector', 'starcoil-service-bridge'),
        {
          terrainVisual: { kind: 'connector', variant: 'starcoil-service-bridge', longSpan: true }
        }
      )
    ];
  }

  function makeRustcoilOrreryCircuitClimbables(platforms) {
    const source = Array.isArray(platforms) ? platforms : [];
    const platformIndexById = new Map(source.map((platform, index) => [String(platform && platform.id || ''), index]));
    const connect = (topId, bottomId, key) => makeClimbableBetweenPlatforms(
      'rustcoilRuins',
      source,
      platformIndexById.get(topId),
      platformIndexById.get(bottomId),
      key,
      'lift'
    );
    return [
      connect('rustcoil_yard_service_gantry', 'rustcoil_yard_ratchet_lane', 'yard_service'),
      connect('rustcoil_yard_broken_orrery', 'rustcoil_yard_service_gantry', 'yard_orrery'),
      connect('rustcoil_switchworks_west_catwalk', 'rustcoil_switchworks_conveyor', 'switchworks_west'),
      connect('rustcoil_switchworks_return_deck', 'rustcoil_switchworks_conveyor', 'switchworks_return'),
      connect('rustcoil_switchworks_east_catwalk', 'rustcoil_switchworks_return_deck', 'switchworks_east'),
      connect('rustcoil_warden_gear_ring', 'rustcoil_warden_return_belt', 'warden_ring'),
      connect('rustcoil_warden_relay_dais', 'rustcoil_warden_gear_ring', 'warden_relay'),
      connect('rustcoil_warden_starcoil_perch', 'rustcoil_warden_relay_dais', 'warden_starcoil')
    ].filter(Boolean);
  }

  function makePriorityFieldPlatforms(width, layoutStyle, variantKey) {
    const mapId = String(variantKey || '');
    if (!PRIORITY_FIELD_LAYOUT_IDS.includes(mapId)) return null;
    const vertical = isVerticalFieldLayout(layoutStyle);
    const lanes = getFieldLaneY(layoutStyle);
    const worldWidth = Math.max(vertical ? 4600 : 4000, Math.ceil(Number(width || 0) / 100) * 100);
    const platforms = [makePlatformDef(0, lanes.ground, worldWidth, 80, { kind: 'ground' })];
    const addPlatform = (x, y, w, visualKind) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW < 120) return;
      platforms.push(makePlatformDef(safeX, Math.round(y), safeW, visualKind === 'hop' ? 20 : 22, { kind: visualKind || 'solidLane' }));
    };
    const addSlope = (x, y, y2, w) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW < 180) return;
      platforms.push(makeSlopePlatformDef(safeX, Math.round(y), Math.round(y2), safeW, 24, { kind: 'slope' }));
    };
    const addFlatConnector = (x, y, w) => addPlatform(x, y, w || 240, 'connector');
    const buildCluster = (cluster, options, clusterIndex) => {
      const settings = options || {};
      const lowY = Number(cluster.lowY || settings.lowY || lanes.low);
      const midY = Number(cluster.midY || settings.midY || lanes.mid);
      const highY = Number(cluster.highY || settings.highY || lanes.high);
      const lowW = Number(cluster.lowW || settings.lowW || (vertical ? 760 : 1320));
      const midW = Number(cluster.midW || settings.midW || (vertical ? 720 : 1240));
      const highW = Number(cluster.highW || settings.highW || (vertical ? 680 : 1180));
      const rampW = Number(cluster.rampW || settings.rampW || 260);
      if (shouldUseSlopePhase(settings, 'groundToLow', clusterIndex, cluster)) addSlope(cluster.lowX - rampW, lanes.ground, lowY, rampW);
      addPlatform(cluster.lowX, lowY, lowW, 'solidLane');
      addFlatConnector(cluster.lowX + lowW - 280, (lowY + midY) / 2, 230);
      if (shouldUseSlopePhase(settings, 'lowToMid', clusterIndex, cluster)) addSlope(cluster.midX - rampW, lowY, midY, rampW);
      addPlatform(cluster.midX, midY, midW, 'solidLane');
      addFlatConnector(cluster.midX + Math.min(midW - 280, 320), (midY + highY) / 2, 230);
      if (shouldUseSlopePhase(settings, 'midToHigh', clusterIndex, cluster)) addSlope(cluster.highX - rampW, midY, highY, rampW);
      addPlatform(cluster.highX, highY, highW, 'solidLane');
    };

    if (mapId === 'greenrootMeadow') {
      // The opening field is an authored journey, not a repeated lane pack:
      // arrival shelf -> combat basin -> reward overlook -> broken bridge -> beacon.
      addPlatform(80, lanes.low, 920, 'solidLane');
      addSlope(1000, lanes.ground, lanes.low, 260);
      addPlatform(1260, lanes.low, 720, 'solidLane');
      addPlatform(600, lanes.mid, 640, 'solidLane');
      addPlatform(1780, lanes.lowConnector, 180, 'connector');
      addSlope(1960, lanes.ground, lanes.lowConnector, 300);
      addPlatform(2260, lanes.lowConnector, 860, 'solidLane');
      addPlatform(2480, lanes.highConnector, 520, 'solidLane');
      addSlope(3120, lanes.lowConnector, lanes.low, 260);
      addPlatform(3380, lanes.low, 660, 'solidLane');
      addPlatform(3720, lanes.mid, 320, 'solidLane');
      return platforms;
    }

    if (mapId === 'thornpathThicket') {
      return makeThornpathFractureCanopyPlatforms(worldWidth);
    }

    if (mapId === 'banditRidgeCamp') {
      return makeBanditRidgeCampPlatforms(worldWidth);
    }

    if (mapId === 'frostfenOutskirts') {
      return makeFrostfenMarshRunPlatforms(worldWidth);
    }

    if (mapId === 'rustcoilRuins') {
      return makeRustcoilOrreryCircuitPlatforms(worldWidth);
    }

    if (mapId === 'orebackQuarry') {
      [
        { lowX: 400, midX: 700, highX: 1000, lowW: 900, midW: 820, highW: 740 },
        { lowX: 1780, midX: 2080, highX: 2380, lowW: 920, midW: 820, highW: 740 },
        { lowX: 3160, midX: 3460, highX: 3760, lowW: 920, midW: 820, highW: 740 }
      ].forEach((cluster, index) => buildCluster(cluster, {
        rampW: 300,
        slopePlan: { lowToMid: [0, 2], midToHigh: [1] }
      }, index));
      addPlatform(4140, lanes.mid - 52, 440, 'solidLane');
      return platforms;
    }

    if (mapId === 'cinderHollow') {
      [
        { lowX: 320, midX: 600, highX: 880, lowW: 860, midW: 760, highW: 700 },
        { lowX: 1880, midX: 2160, highX: 2460, lowW: 920, midW: 780, highW: 700 },
        { lowX: 3440, midX: 3740, highX: 4020, lowW: 920, midW: 780, highW: 700 }
      ].forEach((cluster, index) => buildCluster(cluster, {
        rampW: 260,
        slopePlan: { lowToMid: [1], midToHigh: [2] }
      }, index));
      addPlatform(1500, lanes.low - 40, 600, 'solidLane');
      addPlatform(2920, lanes.mid - 54, 620, 'solidLane');
      addFlatConnector(4680, lanes.highConnector, 230);
      return platforms;
    }

    if (mapId === 'ashglassPass') {
      [
        { lowX: 260, midX: 720, highX: 1120, lowW: 1380, midW: 880, highW: 720, rampW: 300 },
        { lowX: 1900, midX: 2300, highX: 2700, lowW: 1380, midW: 880, highW: 720, rampW: 300 },
        { lowX: 3520, midX: 3860, highX: 4200, lowW: 1240, midW: 840, highW: 680, rampW: 280 }
      ].forEach((cluster, index) => buildCluster(cluster, {
        rampW: Number(cluster.rampW || 280),
        slopePlan: { lowToMid: [0, 2], midToHigh: [] }
      }, index));
      addPlatform(3000, lanes.peak, 720, 'solidLane');
      addSlope(2720, lanes.high, lanes.peak, 280);
      addPlatform(4520, lanes.sky, 500, 'hop');
      return platforms;
    }

    if (mapId === 'stormbreakCliffs') {
      [
        { lowX: 260, midX: 580, highX: 900, lowW: 880, midW: 800, highW: 720 },
        { lowX: 1680, midX: 2020, highX: 2380, lowW: 900, midW: 820, highW: 740 },
        { lowX: 3100, midX: 3460, highX: 3820, lowW: 900, midW: 820, highW: 740 }
      ].forEach((cluster, index) => buildCluster(cluster, {
        rampW: 280,
        slopePlan: { lowToMid: [1], midToHigh: [0] }
      }, index));
      addSlope(2380, lanes.high, lanes.peak, 280);
      addPlatform(2660, lanes.peak, 860, 'solidLane');
      addPlatform(4300, lanes.sky, 620, 'hop');
      addFlatConnector(4320, lanes.highConnector, 240);
      return platforms;
    }

    if (mapId === 'endlessRift') {
      [
        { lowX: 260, midX: 520, highX: 780, lowW: 800, midW: 760, highW: 700 },
        { lowX: 1460, midX: 1740, highX: 2020, lowW: 820, midW: 760, highW: 700 },
        { lowX: 2660, midX: 2940, highX: 3220, lowW: 820, midW: 760, highW: 700 },
        { lowX: 3860, midX: 4100, highX: 4340, lowW: 740, midW: 700, highW: 640 }
      ].forEach((cluster, index) => buildCluster(cluster, {
        rampW: 300,
        slopePlan: { lowToMid: [1], midToHigh: [] }
      }, index));
      addPlatform(2620, lanes.peak, 900, 'solidLane');
      addPlatform(2380, lanes.sky, 520, 'hop');
      return platforms;
    }

    return null;
  }

  function makeFieldPlatforms(width, layoutStyle, variantKey) {
    const vertical = isVerticalFieldLayout(layoutStyle);
    const worldWidth = Math.max(vertical ? 4600 : 4000, Math.ceil(Number(width || 0) / 100) * 100);
    const priorityPlatforms = makePriorityFieldPlatforms(worldWidth, layoutStyle, variantKey);
    if (priorityPlatforms) return priorityPlatforms;
    const anchors = getFieldZoneAnchors(worldWidth, layoutStyle);
    const lanes = getFieldLaneY(layoutStyle);
    const platforms = [makePlatformDef(0, lanes.ground, worldWidth, 80, { kind: 'ground' })];
    const addPlatform = (x, y, w, visualKind) => {
      const safeX = Math.max(120, Math.round(x));
      const widthLimit = Math.min(Math.round(w), worldWidth - safeX - 180);
      if (widthLimit >= 120) platforms.push(makePlatformDef(safeX, y, widthLimit, visualKind === 'hop' ? 20 : 22, { kind: visualKind || 'solidLane' }));
    };
    const addSlope = (x, y, y2, w, visualKind) => {
      const safeX = Math.max(120, Math.round(x));
      const widthLimit = Math.min(Math.round(w), worldWidth - safeX - 180);
      if (widthLimit >= 180) platforms.push(makeSlopePlatformDef(safeX, y, y2, widthLimit, 24, { kind: visualKind || 'slope' }));
    };
    if (vertical) {
      const geometry = getVerticalFieldGeometry(layoutStyle, variantKey);
      anchors.forEach((anchor, index) => {
        const flip = index % 2 === 1;
        const zoneDrift = (index % 3 - 1) * 18 + geometry.mapShift;
        const lift = Number(geometry.lift || 0);
        const lowX = anchor + (flip ? geometry.lowFlip : geometry.low) + zoneDrift;
        const midX = anchor + (flip ? geometry.midFlip : geometry.mid) - zoneDrift * 0.35;
        const highX = anchor + (flip ? geometry.highFlip : geometry.high) + zoneDrift * 0.25;
        const peakX = anchor + (flip ? geometry.peakFlip : geometry.peak) + geometry.peakShift;
        const skyX = anchor + (flip ? geometry.skyFlip : geometry.sky) + geometry.skyShift;
        addSlope(lowX - 120, lanes.ground, lanes.low - lift, geometry.groundRampW, 'slope');
        addPlatform(lowX, lanes.low - lift, geometry.lowW, 'solidLane');
        addPlatform(lowX + (flip ? -120 : geometry.lowW + 20), lanes.lowConnector - lift, 240, 'connector');
        if (index % 2 === 0) {
          if (flip) addSlope(midX + 40, lanes.mid - lift, lanes.low - lift, geometry.midRampW, 'slope');
          else addSlope(midX, lanes.low - lift, lanes.mid - lift, geometry.midRampW, 'slope');
        }
        addPlatform(midX, lanes.mid - lift, geometry.midW, 'solidLane');
        addPlatform(highX + (flip ? geometry.highW - 20 : -40), lanes.highConnector - lift, 240, 'connector');
        if (index === Math.floor(anchors.length / 2)) {
          if (flip) addSlope(highX + 40, lanes.mid - lift, lanes.high - lift, geometry.highRampW, 'slope');
          else addSlope(highX + 140, lanes.high - lift, lanes.mid - lift, geometry.highRampW, 'slope');
        }
        addPlatform(highX, lanes.high - lift, geometry.highW, 'solidLane');
        addPlatform(peakX, lanes.peak - lift, geometry.peakW, layoutStyle === 'astralStack' || layoutStyle === 'riftStack' ? 'island' : 'solidLane');
        addPlatform(skyX, lanes.sky - lift, geometry.skyW, 'hop');
      });
      return platforms;
    }
    const variantSeed = getMapGeometrySeed(variantKey || '');
    anchors.forEach((anchor, index) => {
      const lowShift = variantSeed ? ((variantSeed + index * 11) % 5 - 2) * 18 : 0;
      const midShift = variantSeed ? ((Math.floor(variantSeed / 5) + index * 7) % 5 - 2) * 18 : 0;
      const highShift = variantSeed ? ((Math.floor(variantSeed / 25) + index * 5) % 5 - 2) * 16 : 0;
      const hopShift = variantSeed ? ((Math.floor(variantSeed / 125) + index * 3) % 5 - 2) * 18 : 0;
      if (layoutStyle === 'switchbackTerraces') {
        const drift = index % 2 ? 140 : 0;
        addSlope(anchor + drift - 120, lanes.ground, lanes.low, 300, 'slope');
        addPlatform(anchor + drift + lowShift, lanes.low, clamp(960 - lowShift, 880, 1020), 'solidLane');
        addPlatform(anchor + 1120 - drift * 0.2, lanes.lowConnector, 240, 'connector');
        if (index % 2 === 0) addSlope(anchor + 760 - drift * 0.2, lanes.low, lanes.mid, 320, 'slope');
        addPlatform(anchor + 450 - drift * 0.6 + midShift, lanes.mid, clamp(920 - midShift, 840, 980), 'solidLane');
        addPlatform(anchor + 180 + drift * 0.55, lanes.highConnector, 240, 'connector');
        addPlatform(anchor + 40 + drift + highShift, lanes.high, clamp(820 - highShift, 760, 880), 'solidLane');
        addPlatform(anchor + 1360 - drift * 0.4 + hopShift, lanes.mid - 52, 240, 'hop');
        return;
      }
      if (layoutStyle === 'verticalCanopy') {
        const drift = index % 2 ? 170 : 0;
        addSlope(anchor + drift - 140, lanes.ground - 8, lanes.low, 420, 'slope');
        addPlatform(anchor + drift, lanes.low, 780, 'solidLane');
        addPlatform(anchor + 920 - drift * 0.25, lanes.lowConnector, 220, 'connector');
        if (index !== Math.floor(anchors.length / 2)) addSlope(anchor + 650 - drift * 0.18, lanes.low, lanes.mid, 440, 'slope');
        addPlatform(anchor + 360 - drift * 0.25, lanes.mid, 780, 'solidLane');
        addPlatform(anchor + 1040 + drift * 0.1, lanes.highConnector, 220, 'connector');
        if (index === Math.floor(anchors.length / 2)) addSlope(anchor + 580 + drift * 0.1, lanes.mid, lanes.high, 380, 'slope');
        addPlatform(anchor + 120 + drift * 0.55, lanes.high, 720, 'solidLane');
        addPlatform(anchor + 1180 - drift * 0.25, lanes.high - 72, 260, 'hop');
        return;
      }
      const laneOffset = index % 2 ? 90 : 0;
      addSlope(anchor + laneOffset - 120, lanes.ground, lanes.low, 300, 'slope');
      addPlatform(anchor + laneOffset + lowShift, lanes.low, clamp(1320 - lowShift, 1240, 1380), 'solidLane');
      addPlatform(anchor + 1260, lanes.lowConnector, 220, 'connector');
      if (index % 2 === 0) addSlope(anchor + 860 - laneOffset * 0.15, lanes.low, lanes.mid, 320, 'slope');
      addPlatform(anchor + 260 - laneOffset * 0.4 + midShift, lanes.mid, clamp(1240 - midShift, 1200, 1300), 'solidLane');
      addPlatform(anchor + 390, lanes.highConnector, 220, 'connector');
      addPlatform(anchor + 560 + laneOffset * 0.35 + highShift, lanes.high, clamp(1200 - highShift, 1200, 1260), 'solidLane');
      addPlatform(anchor + 1680 - laneOffset * 0.2 + hopShift, lanes.mid - 54, 250, 'hop');
    });
    return platforms;
  }

  function makeTerrainIslandSegments(platform, index, layoutStyle) {
    const width = Math.max(0, getPlatformDefW(platform));
    const count = width >= 1400 ? 3 : width >= 900 ? 2 : 1;
    const baseWidth = count === 3 ? 300 : count === 2 ? 340 : Math.min(420, Math.max(260, width - 120));
    const styleDrift = layoutStyle === 'switchbackTerraces' ? 36 : layoutStyle === 'verticalCanopy' ? -28 : 0;
    return Object.freeze(Array.from({ length: count }, (_, segmentIndex) => {
      const drift = ((index + segmentIndex) % 2 ? 1 : -1) * (28 + segmentIndex * 8) + styleDrift;
      const rawCenter = width * (segmentIndex + 1) / (count + 1) + drift;
      const segmentWidth = Math.min(baseWidth + (segmentIndex % 2 ? 32 : 0), Math.max(180, width - 96));
      const x = clamp(Math.round(rawCenter - segmentWidth / 2), 36, Math.max(36, width - segmentWidth - 36));
      return Object.freeze({
        x,
        w: Math.round(segmentWidth),
        depth: 28 + (index + segmentIndex) % 3 * 4
      });
    }));
  }

  function makeFieldTerrainVisuals(platforms, layoutStyle) {
    return Object.freeze(platforms.map((platform, index) => {
      const authoredVisual = platform && !Array.isArray(platform) && platform.terrainVisual;
      if (authoredVisual) {
        return Object.freeze(Object.assign({ segments: Object.freeze([]) }, authoredVisual));
      }
      const width = Math.max(0, getPlatformDefW(platform));
      if (index === 0) {
        return Object.freeze({ kind: 'ground', segments: Object.freeze([]) });
      }
      if (width <= 320) {
        return Object.freeze({ kind: 'connector', segments: Object.freeze([]) });
      }
      return Object.freeze({
        kind: 'solidLane',
        segments: Object.freeze([])
      });
    }));
  }

  function makeVerticalFieldClimbables(prefix, platforms, layoutStyle) {
    const kind = getFieldClimbableKind(layoutStyle);
    return platforms
      .map((platform, topIndex) => ({ platform, topIndex }))
      .filter((entry) => {
        const visualKind = getPlatformDefVisualKind(entry.platform);
        return entry.topIndex > 0 &&
          getPlatformDefShape(entry.platform) !== 'slope' &&
          getPlatformDefW(entry.platform) >= 500 &&
          visualKind !== 'connector' &&
          visualKind !== 'hop';
      })
      .map((entry, localIndex) => {
        const top = entry.platform;
        const bottomEntry = platforms
          .map((platform, bottomIndex) => ({ platform, bottomIndex }))
          .filter((candidate) => {
            const visualKind = getPlatformDefVisualKind(candidate.platform);
            if (candidate.bottomIndex === entry.topIndex || getPlatformDefY(candidate.platform) <= getPlatformDefY(top)) return false;
            if (getPlatformDefShape(candidate.platform) === 'slope' || visualKind === 'connector' || visualKind === 'hop') return false;
            const overlap = Math.min(getPlatformDefRight(top), getPlatformDefRight(candidate.platform)) -
              Math.max(getPlatformDefX(top), getPlatformDefX(candidate.platform));
            return overlap > 80;
          })
          .sort((a, b) => {
            const aOverlap = Math.min(getPlatformDefRight(top), getPlatformDefRight(a.platform)) - Math.max(getPlatformDefX(top), getPlatformDefX(a.platform));
            const bOverlap = Math.min(getPlatformDefRight(top), getPlatformDefRight(b.platform)) - Math.max(getPlatformDefX(top), getPlatformDefX(b.platform));
            return Math.abs(getPlatformDefY(a.platform) - getPlatformDefY(top)) - Math.abs(getPlatformDefY(b.platform) - getPlatformDefY(top)) || bOverlap - aOverlap;
          })[0];
        return bottomEntry
          ? makeClimbableBetweenPlatforms(prefix, platforms, entry.topIndex, bottomEntry.bottomIndex, `${localIndex + 1}`, kind)
          : null;
      })
      .filter(Boolean);
  }

  function makeFieldClimbables(prefix, widthOrPlatforms, layoutStyle) {
    const platforms = Array.isArray(widthOrPlatforms) ? widthOrPlatforms : null;
    if (platforms) return makeVerticalFieldClimbables(prefix, platforms, layoutStyle);
    const width = platforms
      ? Math.max(6200, platforms.reduce((maxWidth, platform) => Math.max(maxWidth, getPlatformDefRight(platform)), 0))
      : widthOrPlatforms;
    const lanes = getFieldLaneY(layoutStyle);
    const anchors = getFieldZoneAnchors(width, layoutStyle);
    const climbables = [];
    anchors.forEach((anchor, index) => {
      if (layoutStyle === 'switchbackTerraces') {
        const drift = index % 2 ? 140 : 0;
        climbables.push({ id: `${prefix}_terrace_ladder_${index + 1}_low`, x: anchor + 240 + drift, y: lanes.low, w: 30, h: lanes.ground - lanes.low });
        climbables.push({ id: `${prefix}_terrace_ladder_${index + 1}_mid`, x: anchor + 760 - drift * 0.45, y: lanes.mid, w: 30, h: lanes.low - lanes.mid });
        climbables.push({ id: `${prefix}_terrace_ladder_${index + 1}_high`, x: anchor + 560 + drift * 0.35, y: lanes.high, w: 30, h: lanes.mid - lanes.high });
        return;
      }
      if (layoutStyle === 'verticalCanopy') {
        const drift = index % 2 ? 170 : 0;
        climbables.push({ id: `${prefix}_canopy_vine_${index + 1}_low`, x: anchor + 180 + drift, y: lanes.low, w: 28, h: lanes.ground - lanes.low });
        climbables.push({ id: `${prefix}_canopy_vine_${index + 1}_mid`, x: anchor + 720 - drift * 0.2, y: lanes.mid, w: 28, h: lanes.low - lanes.mid });
        climbables.push({ id: `${prefix}_canopy_vine_${index + 1}_high`, x: anchor + 640 + drift * 0.15, y: lanes.high, w: 28, h: lanes.mid - lanes.high });
        return;
      }
      const laneOffset = index % 2 ? 90 : 0;
      climbables.push({ id: `${prefix}_lane_rope_${index + 1}_low`, x: anchor + 180 + laneOffset, y: lanes.low, w: 28, h: lanes.ground - lanes.low });
      climbables.push({ id: `${prefix}_lane_rope_${index + 1}_mid`, x: anchor + 1060, y: lanes.mid, w: 28, h: lanes.low - lanes.mid });
      climbables.push({ id: `${prefix}_lane_rope_${index + 1}_high`, x: anchor + 820 + laneOffset * 0.2, y: lanes.high, w: 28, h: lanes.mid - lanes.high });
    });
    return climbables;
  }

  function makeFieldSpawnPoints(platforms) {
    return platforms
      .map((platform, index) => ({ platform, index }))
      .filter((entry) => entry.index > 0 && getPlatformDefW(entry.platform) >= 640)
      .reduce((points, entry) => {
        const platform = entry.platform;
        const x = getPlatformDefX(platform);
        const w = getPlatformDefW(platform);
        const y = getPlatformDefY(platform);
        const weight = y >= 430 ? 3 : y >= 320 ? 2 : 1;
        if (w >= 1600) {
          points.push({ x: Math.round(x + w * 0.22), platformIndex: entry.index, weight });
          points.push({ x: Math.round(x + w * 0.5), platformIndex: entry.index, weight });
          points.push({ x: Math.round(x + w * 0.78), platformIndex: entry.index, weight });
        } else if (w >= 900) {
          points.push({ x: Math.round(x + w * 0.27), platformIndex: entry.index, weight });
          points.push({ x: Math.round(x + w * 0.73), platformIndex: entry.index, weight });
        } else {
          points.push({ x: Math.round(x + w / 2), platformIndex: entry.index, weight });
        }
        return points;
      }, []);
  }

  function makeTownPlatforms(width, variantKey) {
    const worldWidth = Math.max(3600, Math.ceil(Number(width || 0) / 100) * 100);
    const variantSeed = getMapGeometrySeed(variantKey || '');
    const profile = variantSeed % 4;
    const lanes = Object.freeze(Object.assign({}, TOWN_LANE_Y, {
      low: TOWN_LANE_Y.low + [0, -28, 22, -14][profile],
      mid: TOWN_LANE_Y.mid + [0, 32, -24, 16][profile],
      high: TOWN_LANE_Y.high + [0, -34, 30, -20][profile],
      roof: TOWN_LANE_Y.roof + [0, 40, -30, 22][profile]
    }));
    const profileShift = (values) => values[profile] || 0;
    const shift = (salt, scale) => variantSeed ? ((Math.floor(variantSeed / salt) % 5) - 2) * scale : 0;
    const platforms = [makePlatformDef(0, TOWN_LANE_Y.ground, worldWidth, 80, { kind: 'ground' })];
    const add = (x, y, w, visualKind) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW >= 160) platforms.push(makePlatformDef(safeX, y, safeW, 24, { kind: visualKind || 'solidLane' }));
    };
    const addSlope = (x, y, y2, w) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW >= 180) platforms.push(makeSlopePlatformDef(safeX, y, y2, safeW, 24, { kind: 'slope' }));
    };
    const lowLeftShift = shift(1, 26);
    const lowMarketShift = shift(5, 30);
    const lowGateShift = shift(25, 28);
    const midLeftShift = shift(3, 24);
    const midMarketShift = shift(7, 28);
    const midGateShift = shift(11, 24);
    const highLeftShift = shift(13, 26);
    const highMarketShift = shift(17, 24);
    const highGateShift = shift(19, 18);
    const roofShift = shift(23, 30);
    const lowLeftX = 220 + lowLeftShift + profileShift([0, 70, -42, 36]);
    const lowLeftW = 740 - lowLeftShift * 0.5 + profileShift([0, -46, 62, -24]);
    const lowMarketX = 1180 + lowMarketShift + profileShift([0, -62, 86, -38]);
    const lowMarketW = 720 - lowMarketShift * 0.35 + profileShift([0, 72, -54, 38]);
    const lowGateX = 2220 + lowGateShift + profileShift([0, 88, -96, 54]);
    const lowGateW = 780 - lowGateShift * 0.35 + profileShift([0, -64, 46, -34]);
    const midLeftX = 680 + midLeftShift + profileShift([0, -72, 54, -48]);
    const midLeftW = 860 - midLeftShift * 0.4 + profileShift([0, 58, -42, 68]);
    const midMarketX = 1720 + midMarketShift + profileShift([0, 64, -76, 42]);
    const midMarketW = 780 - midMarketShift * 0.35 + profileShift([0, -48, 74, -36]);
    const midGateX = 2860 + midGateShift + profileShift([0, -58, 72, -40]);
    const midGateW = 520 - midGateShift * 0.4 + profileShift([0, 66, -34, 52]);
    const highLeftX = 360 + highLeftShift + profileShift([0, 92, -68, 54]);
    const highLeftW = 700 - highLeftShift * 0.35 + profileShift([0, -52, 82, -38]);
    const highMarketX = 1440 + highMarketShift + profileShift([0, -76, 64, -52]);
    const highMarketW = 780 - highMarketShift * 0.35 + profileShift([0, 74, -48, 58]);
    const highGateX = 2440 + highGateShift + profileShift([0, 56, -82, 40]);
    const highGateW = 640 - highGateShift * 0.35 + profileShift([0, -42, 70, -28]);
    const roofX = 1040 + roofShift + profileShift([0, 74, -62, 42]);
    const roofW = 640 - roofShift * 0.35 + profileShift([0, 52, -38, 64]);
    add(lowLeftX, lanes.low, lowLeftW);
    add(lowMarketX, lanes.low, lowMarketW);
    add(lowGateX, lanes.low, lowGateW);
    add(midLeftX, lanes.mid, midLeftW);
    add(midMarketX, lanes.mid, midMarketW);
    add(midGateX, lanes.mid, midGateW);
    add(highLeftX, lanes.high, highLeftW);
    add(highMarketX, lanes.high, highMarketW);
    add(highGateX, lanes.high, highGateW);
    add(roofX, lanes.roof, roofW, 'island');
    addSlope(lowMarketX - 220, TOWN_LANE_Y.ground - 8, lanes.low, 300);
    addSlope(midMarketX - 160, lanes.low, lanes.mid, 300);
    addSlope(highLeftX + highLeftW, lanes.high, lanes.mid, 240);
    addSlope(highGateX + highGateW, lanes.high, lanes.mid, 240);
    add(1960 + shift(29, 26) + profileShift([0, 68, -48, 36]), lanes.roof + 46, 260, 'hop');
    add(3160 + shift(31, 24) + profileShift([0, -62, 54, -40]), lanes.high + 58, 240, 'hop');
    return platforms;
  }

  function makeTownClimbables(prefix, platforms) {
    return [
      makeClimbableBetweenPlatforms(prefix, platforms, 1, 0, 'left_plaza', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 4, 1, 'left_roofwalk', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 7, 4, 'left_balcony', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 10, 8, 'guild_roof', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 2, 0, 'market_plaza', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 5, 2, 'market_roofwalk', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 8, 5, 'artisan_balcony', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 3, 0, 'gate_plaza', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 6, 3, 'gate_watch', 'stair'),
      makeClimbableBetweenPlatforms(prefix, platforms, 9, 6, 'gate_roof', 'stair')
    ].filter(Boolean);
  }

  const api = Object.freeze({
    makeClimbableBetweenPlatforms,
    makePartyPlayPlatforms,
    makePartyPlayClimbables,
    makePartyPlaySpawnPoints,
    makeDungeonArenaPlatforms,
    makeBanditRidgeCampPlatforms,
    makeBanditRidgeCampClimbables,
    makeThornpathFractureCanopyPlatforms,
    makeThornpathFractureCanopyClimbables,
    makeFrostfenMarshRunPlatforms,
    makeFrostfenMarshRunClimbables,
    makeRustcoilOrreryCircuitPlatforms,
    makeRustcoilOrreryCircuitClimbables,
    makePriorityFieldPlatforms,
    makeFieldPlatforms,
    makeTerrainIslandSegments,
    makeFieldTerrainVisuals,
    makeVerticalFieldClimbables,
    makeFieldClimbables,
    makeFieldSpawnPoints,
    makeTownPlatforms,
    makeTownClimbables
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapBuilders = Object.assign({}, modules.mapBuilders || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
