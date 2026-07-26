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
    const addPlatform = (x, y, w, visualKind, options) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW < 120) return;
      const platform = makePlatformDef(safeX, Math.round(y), safeW, visualKind === 'hop' ? 20 : 22, { kind: visualKind || 'solidLane' });
      const settings = options && typeof options === 'object' ? options : {};
      if (settings.id) platform.id = String(settings.id);
      if (settings.spawnDisabled) platform.spawnDisabled = true;
      platforms.push(platform);
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
    const entryClearance = Math.max(0, Number(skeleton.entryClearance || 0));
    const leftLow = Math.max(Number(skeleton.left || 300), entryClearance ? entryClearance + 260 : 0);
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
    if (mapId === 'astralStacks') {
      // Keep the mirrored stacks silhouette, but turn the decorative center
      // pieces into one reachable shelf that can carry the rune mechanic.
      addPlatform(1740, highY, 760, 'island', {
        id: 'astralStacks_center_rune_shelf'
      });
    } else {
      addPlatform(1740, TRAINING_LANE_Y.highConnector + Number(skeleton.midShift || 0) * 0.3, 260, 'connector');
      addPlatform(2240, TRAINING_LANE_Y.highConnector + Number(skeleton.highShift || 0) * 0.3, 260, 'connector');
      addPlatform(2040, highY - 62, 280, 'hop');
    }
    return platforms;
  }

  function makePriorityFieldPlatforms(width, layoutStyle, variantKey) {
    const mapId = String(variantKey || '');
    if (!PRIORITY_FIELD_LAYOUT_IDS.includes(mapId)) return null;
    const vertical = isVerticalFieldLayout(layoutStyle);
    const lanes = getFieldLaneY(layoutStyle);
    const worldWidth = Math.max(vertical ? 4600 : 4000, Math.ceil(Number(width || 0) / 100) * 100);
    const platforms = [makePlatformDef(0, lanes.ground, worldWidth, 80, { kind: 'ground' })];
    const addPlatform = (x, y, w, visualKind, options) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW < 120) return;
      const platform = makePlatformDef(safeX, Math.round(y), safeW, visualKind === 'hop' ? 20 : 22, { kind: visualKind || 'solidLane' });
      const settings = options && typeof options === 'object' ? options : {};
      if (settings.id) platform.id = String(settings.id);
      if (settings.spawnDisabled) platform.spawnDisabled = true;
      if (settings.climbableDisabled) platform.climbableDisabled = true;
      platforms.push(platform);
    };
    const addSlope = (x, y, y2, w, options) => {
      const safeX = Math.max(120, Math.round(x));
      const safeW = Math.min(Math.round(w), worldWidth - safeX - 160);
      if (safeW < 180) return;
      const platform = makeSlopePlatformDef(safeX, Math.round(y), Math.round(y2), safeW, 24, { kind: 'slope' });
      const settings = options && typeof options === 'object' ? options : {};
      if (settings.id) platform.id = String(settings.id);
      platforms.push(platform);
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
      // Preserve the compact, playful meadow while giving its starter route
      // four authored beats instead of repeating the same three-lane cluster.
      addSlope(520, lanes.ground, 456, 280);
      addPlatform(760, 456, 900, 'solidLane');
      addPlatform(500, 328, 900, 'solidLane');
      addFlatConnector(1420, 392, 220);

      addSlope(1660, lanes.ground, 456, 300);
      addPlatform(1960, 456, 720, 'solidLane');
      addSlope(2560, 456, 328, 300);
      addPlatform(1960, 328, 720, 'solidLane');
      addSlope(1960, 328, 200, 300);
      addPlatform(1500, 200, 880, 'solidLane');

      addPlatform(2380, 200, 120, 'connector');
      addPlatform(2500, 200, 840, 'solidLane');
      addPlatform(2700, 456, 640, 'solidLane');
      addPlatform(3360, 456, 640, 'solidLane');
      addPlatform(2700, 328, 640, 'solidLane');
      addPlatform(3360, 328, 640, 'solidLane');
      addSlope(3300, 456, 328, 300);
      addSlope(3400, 328, 200, 300);
      addPlatform(3360, 200, 640, 'solidLane');
      addFlatConnector(2600, 392, 220);
      return platforms;
    }

    if (mapId === 'thornpathThicket') {
      // Keep the playful vertical-canopy identity, but stage the climb as one
      // readable route: a calm scout apron, two teaching loops, then a safe
      // fork approach. Overlapping tiers create forgiving vine/drop resets.
      addSlope(600, lanes.ground, lanes.low, 300);
      addPlatform(760, lanes.low, 700, 'solidLane');
      addPlatform(760, lanes.mid, 840, 'solidLane');
      addSlope(1300, lanes.low, lanes.mid, 300);

      addPlatform(1530, lanes.low, 850, 'solidLane');
      addPlatform(1660, lanes.mid, 820, 'solidLane');
      addPlatform(1880, lanes.high, 740, 'solidLane');
      addSlope(2180, lanes.mid, lanes.high, 300);

      addSlope(2460, lanes.ground, lanes.low, 300);
      addPlatform(2500, lanes.low, 1000, 'solidLane');
      addPlatform(2700, lanes.mid, 900, 'solidLane');
      addPlatform(2920, lanes.high, 800, 'solidLane');
      addPlatform(3200, lanes.peak, 720, 'solidLane');
      addSlope(3300, lanes.high, lanes.peak, 300);

      addFlatConnector(3500, lanes.low, 200);
      addFlatConnector(3680, lanes.mid, 180);
      addFlatConnector(2680, lanes.high, 160);
      addFlatConnector(3780, lanes.high, 160);
      addFlatConnector(3960, lanes.peak, 140);

      addSlope(3580, lanes.ground, lanes.low, 300);
      addPlatform(3820, lanes.low, 820, 'solidLane');
      addPlatform(3960, lanes.mid, 760, 'solidLane');
      addPlatform(4020, lanes.high, 700, 'solidLane');
      addPlatform(4160, lanes.peak, 620, 'solidLane');
      return platforms;
    }

    if (mapId === 'banditRidgeCamp') {
      const lowY = lanes.low;
      const midY = lanes.mid + 6;
      const highY = lanes.high + 14;
      [
        { lowX: 400, midX: 700, highX: 1000, lowW: 1320, midW: 1180, highW: 1040 },
        { lowX: 2000, midX: 2300, highX: 2600, lowW: 1320, midW: 1180, highW: 1040 },
        { lowX: 3600, midX: 3900, highX: 4200, lowW: 1320, midW: 1180, highW: 1040 }
      ].forEach((cluster, index) => buildCluster(Object.assign({}, cluster, { lowY, midY, highY }), {
        rampW: 300,
        slopePlan: { lowToMid: [0, 2], midToHigh: [1] }
      }, index));
      addPlatform(3500, midY - 58, 280, 'hop');
      return platforms;
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
      // Keep the playful lava-shaft silhouette, but make each section teach a
      // distinct combat lesson. The upper vent shelf is an optional, spawn-free
      // bypass; the final 940px stays calm for the Pathfinder and Emberjaw gate.
      addSlope(420, lanes.ground, lanes.low, 300, { id: 'cinderHollow_entry_ramp' });
      addPlatform(620, lanes.low, 780, 'solidLane', {
        id: 'cinderHollow_ash_floor_low',
        climbableDisabled: true
      });
      addSlope(900, lanes.low, lanes.mid, 300, { id: 'cinderHollow_ash_ramp' });
      addPlatform(720, lanes.mid, 680, 'solidLane', {
        id: 'cinderHollow_ash_overlook',
        climbableDisabled: true
      });
      addFlatConnector(1400, lanes.lowConnector, 180);

      addPlatform(1500, lanes.low, 900, 'solidLane', { id: 'cinderHollow_vent_floor' });
      addSlope(1740, lanes.low, lanes.mid, 300, { id: 'cinderHollow_vent_ramp' });
      addPlatform(1740, lanes.mid, 900, 'solidLane', {
        id: 'cinderHollow_vent_shelf',
        climbableDisabled: true
      });
      addSlope(2040, lanes.mid, lanes.high, 300, { id: 'cinderHollow_vent_bypass_ramp' });
      addPlatform(2040, lanes.high, 760, 'solidLane', {
        id: 'cinderHollow_vent_bypass',
        spawnDisabled: true,
        climbableDisabled: true
      });
      addFlatConnector(2700, lanes.highConnector, 180);

      addSlope(2900, lanes.ground, lanes.low, 300, { id: 'cinderHollow_wisp_entry_ramp' });
      addPlatform(2900, lanes.low, 900, 'solidLane', {
        id: 'cinderHollow_wisp_recovery',
        climbableDisabled: true
      });
      addSlope(3300, lanes.low, lanes.mid, 300, { id: 'cinderHollow_wisp_mid_ramp' });
      addPlatform(3120, lanes.mid, 900, 'solidLane', {
        id: 'cinderHollow_wisp_turn_mid',
        climbableDisabled: true
      });
      addPlatform(3400, lanes.high, 860, 'solidLane', { id: 'cinderHollow_wisp_turn_high' });
      addFlatConnector(4140, lanes.highConnector, 180);
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
      // The Aerie gate should feel like a playful storm lookout, not a portal
      // floating above an impossible jump. A solid, spawn-free perch lets the
      // regular storm-stair builder connect it to the final combat lane.
      addPlatform(4300, lanes.sky, 620, 'solidLane', {
        id: 'stormbreakCliffs_aerie_perch',
        spawnDisabled: true
      });
      addFlatConnector(4320, lanes.highConnector, 240);
      return platforms;
    }

    if (mapId === 'endlessRift') {
      // Keep the existing playful Rift painting and rune-stair language, but
      // make the promised four-quadrant loop physically true. The broad lanes
      // form a ring around a calm central dais; short bridges and forgiving
      // slopes let every quadrant return to the core without a ground reset.
      addPlatform(920, lanes.low, 640, 'solidLane', { id: 'endlessRift_sw_outer_low' });
      addPlatform(1600, lanes.low, 640, 'solidLane', { id: 'endlessRift_sw_inner_low' });
      addPlatform(1240, lanes.mid, 820, 'solidLane', { id: 'endlessRift_sw_mid' });

      addPlatform(920, lanes.high, 640, 'solidLane', { id: 'endlessRift_nw_outer_high' });
      addPlatform(1600, lanes.high, 640, 'solidLane', { id: 'endlessRift_nw_inner_high' });
      addPlatform(1240, lanes.peak, 820, 'solidLane', { id: 'endlessRift_nw_peak' });

      addPlatform(2960, lanes.high, 640, 'solidLane', { id: 'endlessRift_ne_inner_high' });
      addPlatform(3640, lanes.high, 640, 'solidLane', { id: 'endlessRift_ne_outer_high' });
      addPlatform(3140, lanes.peak, 820, 'solidLane', { id: 'endlessRift_ne_peak' });

      addPlatform(2960, lanes.low, 640, 'solidLane', { id: 'endlessRift_se_inner_low' });
      addPlatform(3640, lanes.low, 640, 'solidLane', { id: 'endlessRift_se_outer_low' });
      addPlatform(3140, lanes.mid, 820, 'solidLane', { id: 'endlessRift_se_mid' });

      addPlatform(2200, lanes.mid, 800, 'island', {
        id: 'endlessRift_core_dais',
        spawnDisabled: true
      });
      addPlatform(2240, lanes.high, 720, 'solidLane', {
        id: 'endlessRift_north_ring_bridge',
        spawnDisabled: true
      });
      addPlatform(2240, lanes.low, 720, 'solidLane', {
        id: 'endlessRift_south_ring_bridge',
        spawnDisabled: true
      });
      addPlatform(2060, lanes.mid, 140, 'connector', { id: 'endlessRift_west_core_spoke' });
      addPlatform(3000, lanes.mid, 140, 'connector', { id: 'endlessRift_east_core_spoke' });

      addSlope(620, lanes.ground, lanes.low, 300, { id: 'endlessRift_entry_ramp' });
      addSlope(940, lanes.low, lanes.mid, 300, { id: 'endlessRift_sw_ramp' });
      addSlope(920, lanes.high, lanes.mid, 320, { id: 'endlessRift_nw_ramp' });
      addSlope(3660, lanes.mid, lanes.high, 300, { id: 'endlessRift_ne_ramp' });
      addSlope(3360, lanes.mid, lanes.low, 300, { id: 'endlessRift_se_ramp' });
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
    const addPlatform = (x, y, w, visualKind, options) => {
      const safeX = Math.max(120, Math.round(x));
      const widthLimit = Math.min(Math.round(w), worldWidth - safeX - 180);
      if (widthLimit < 120) return;
      const platform = makePlatformDef(safeX, y, widthLimit, visualKind === 'hop' ? 20 : 22, { kind: visualKind || 'solidLane' });
      const settings = options && typeof options === 'object' ? options : {};
      if (settings.id) platform.id = String(settings.id);
      if (settings.spawnDisabled) platform.spawnDisabled = true;
      platforms.push(platform);
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
      if (variantKey === 'astralArchive') {
        // The archive keeps its three playful tower stacks, while two quiet
        // rune bridges let players loop between them without resetting to
        // the ground after every room.
        addPlatform(1340, 846, 302, 'connector', {
          id: 'astralArchive_west_rune_bridge_01',
          spawnDisabled: true
        });
        addPlatform(1642, 846, 301, 'connector', {
          id: 'astralArchive_west_rune_bridge_02',
          spawnDisabled: true
        });
        addPlatform(1943, 846, 301, 'connector', {
          id: 'astralArchive_west_rune_bridge_03',
          spawnDisabled: true
        });
        addPlatform(3389, 666, 254, 'connector', {
          id: 'astralArchive_east_rune_bridge_01',
          spawnDisabled: true
        });
        addPlatform(3643, 666, 253, 'connector', {
          id: 'astralArchive_east_rune_bridge_02',
          spawnDisabled: true
        });
      }
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
    const climbables = platforms
      .map((platform, topIndex) => ({ platform, topIndex }))
      .filter((entry) => {
        const visualKind = getPlatformDefVisualKind(entry.platform);
        return entry.topIndex > 0 &&
          getPlatformDefShape(entry.platform) !== 'slope' &&
          getPlatformDefW(entry.platform) >= 500 &&
          !entry.platform.climbableDisabled &&
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
    return climbables;
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
      .filter((entry) => {
        const visualKind = getPlatformDefVisualKind(entry.platform);
        return entry.index > 0 &&
          getPlatformDefW(entry.platform) >= 640 &&
          !entry.platform.spawnDisabled &&
          visualKind !== 'connector' &&
          visualKind !== 'hop';
      })
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
