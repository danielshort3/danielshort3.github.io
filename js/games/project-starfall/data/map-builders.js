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
      addTransitionSlope(midX, midY, widths.mid, highX, highY, widths.high, rampW);
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

  function makePriorityFieldPlatforms(width, layoutStyle, variantKey) {
    const mapId = String(variantKey || '');
    if (!PRIORITY_FIELD_LAYOUT_IDS.includes(mapId)) return null;
    const vertical = isVerticalFieldLayout(layoutStyle);
    const lanes = getFieldLaneY(layoutStyle);
    const worldWidth = Math.max(vertical ? 5200 : 8400, Math.ceil(Number(width || 0) / 100) * 100);
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
      if (shouldUseSlopePhase(settings, 'lowToMid', clusterIndex, cluster)) {
        addSlope(Math.max(cluster.lowX, cluster.midX - rampW), lowY, midY, rampW);
      }
      addPlatform(cluster.midX, midY, midW, 'solidLane');
      addFlatConnector(cluster.midX + Math.min(midW - 280, 320), (midY + highY) / 2, 230);
      if (shouldUseSlopePhase(settings, 'midToHigh', clusterIndex, cluster)) {
        addSlope(Math.max(cluster.midX, cluster.highX - rampW), midY, highY, rampW);
      }
      addPlatform(cluster.highX, highY, highW, 'solidLane');
    };

    if (mapId === 'greenrootMeadow') {
      const lowY = lanes.low;
      const midY = lanes.mid + 10;
      const highY = lanes.high + 20;
      [
        { lowX: 300, midX: 560, highX: 820, lowW: 1420, midW: 1320, highW: 1220 },
        { lowX: 2360, midX: 2600, highX: 2860, lowW: 1400, midW: 1320, highW: 1220 },
        { lowX: 4440, midX: 4700, highX: 4960, lowW: 1420, midW: 1320, highW: 1220 },
        { lowX: 6320, midX: 6560, highX: 6820, lowW: 1260, midW: 1220, highW: 1120 }
      ].forEach((cluster, index) => buildCluster(Object.assign({}, cluster, { lowY, midY, highY }), {
        rampW: 260
      }, index));
      addPlatform(1940, lanes.mid - 52, 260, 'hop');
      addPlatform(5980, lanes.mid - 52, 260, 'hop');
      return platforms;
    }

    if (mapId === 'banditRidgeCamp') {
      const lowY = lanes.low;
      const midY = lanes.mid + 6;
      const highY = lanes.high + 14;
      [
        { lowX: 360, midX: 660, highX: 960, lowW: 1320, midW: 1180, highW: 1060 },
        { lowX: 2300, midX: 2560, highX: 2880, lowW: 1340, midW: 1200, highW: 1080 },
        { lowX: 4300, midX: 4580, highX: 4880, lowW: 1340, midW: 1200, highW: 1080 },
        { lowX: 6260, midX: 6540, highX: 6840, lowW: 1260, midW: 1160, highW: 980 }
      ].forEach((cluster, index) => buildCluster(Object.assign({}, cluster, { lowY, midY, highY }), {
        rampW: 280
      }, index));
      addPlatform(6900, midY - 58, 620, 'solidLane');
      addPlatform(7240, highY - 58, 320, 'hop');
      return platforms;
    }

    if (mapId === 'orebackQuarry') {
      [
        { lowX: 300, midX: 620, highX: 940, lowW: 880, midW: 800, highW: 720 },
        { lowX: 1700, midX: 2020, highX: 2360, lowW: 900, midW: 800, highW: 720 },
        { lowX: 3100, midX: 3420, highX: 3760, lowW: 900, midW: 800, highW: 720 }
      ].forEach((cluster, index) => buildCluster(cluster, {
        rampW: 280
      }, index));
      addSlope(2440, lanes.high, lanes.peak, 280);
      addPlatform(2720, lanes.peak, 820, 'solidLane');
      addPlatform(4300, lanes.mid - 52, 620, 'solidLane');
      addFlatConnector(4040, lanes.highConnector, 110);
      return platforms;
    }

    if (mapId === 'cinderHollow') {
      [
        { lowX: 320, midX: 600, highX: 880, lowW: 860, midW: 760, highW: 700 },
        { lowX: 1880, midX: 2160, highX: 2460, lowW: 920, midW: 780, highW: 700 },
        { lowX: 3440, midX: 3740, highX: 4020, lowW: 920, midW: 780, highW: 700 }
      ].forEach((cluster, index) => buildCluster(cluster, {
        rampW: 260
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
        rampW: Number(cluster.rampW || 280)
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
        rampW: 280
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
        rampW: 300
      }, index));
      addPlatform(2620, lanes.peak, 900, 'solidLane');
      addPlatform(2380, lanes.sky, 520, 'hop');
      return platforms;
    }

    return null;
  }

  function makeFieldPlatforms(width, layoutStyle, variantKey) {
    const vertical = isVerticalFieldLayout(layoutStyle);
    const worldWidth = Math.max(vertical ? 4600 : 6200, Math.ceil(Number(width || 0) / 100) * 100);
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
    // Low-rise upper-town profiles such as Rustcoil need compact ramps so the
    // collision surface does not run underneath both walkable platforms.
    // Taller profiles keep the longer, gentler ramp grade.
    const upperRampWidth = Math.abs(lanes.mid - lanes.high) <= 80 ? 200 : 280;
    addSlope(highLeftX + highLeftW - upperRampWidth, lanes.mid, lanes.high, upperRampWidth);
    addSlope(highGateX - upperRampWidth, lanes.mid, lanes.high, upperRampWidth);
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
