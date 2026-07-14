(function initProjectStarfallUiHud(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const colorWithAlpha = CoreMath.colorWithAlpha || function colorWithAlphaFallback(color, alpha) {
    const value = String(color || '#ffffff');
    if (!/^#[0-9a-f]{6}$/i.test(value)) return value;
    const channel = clamp(Math.round((Number(alpha) || 0) * 255), 0, 255);
    return `${value}${channel.toString(16).padStart(2, '0')}`;
  };
  const UI_CHANGE_HUD_SNAPSHOT_DOMAINS = Object.freeze(['hud', 'equipment', 'cards', 'skills', 'party', 'pet']);
  const HUD_SNAPSHOT_MERGE_KEYS = Object.freeze([
    'stats',
    'statUpgrades',
    'map',
    'channel',
    'classData',
    'advancedData',
    'nextLevelXp',
    'activeBuffs',
    'activeCooldowns',
    'bossEncounter',
    'mapModifiers',
    'onboarding',
    'audio',
    'settings',
    'performanceBenchmark'
  ]);
  const HUD_WIDGET_DRAG_ENTRIES = Object.freeze([
    Object.freeze({ dragKey: 'minimapDrag', stateKey: 'minimapState', boxType: 'minimap', bottomInset: 0 }),
    Object.freeze({ dragKey: 'questTrackerDrag', stateKey: 'questTrackerState', boxType: 'questTracker', bottomInset: 0 }),
    Object.freeze({ dragKey: 'combatMetricsDrag', stateKey: 'combatMetricsPanelState', boxType: 'combatMetrics', bottomInset: 10 })
  ]);
  const HUD_WIDGET_DRAG_KEYS = Object.freeze(HUD_WIDGET_DRAG_ENTRIES.map((entry) => entry.dragKey));
  const HUD_METER_ANIMATION_MS = 520;
  const CANVAS_OVERLAY_CACHE_MS = 50;
  const CANVAS_OVERLAY_CACHE_WITH_WINDOWS_MS = 75;
  const CANVAS_OVERLAY_CACHE_ADAPTIVE_MAX_MS = 125;
  const CANVAS_OVERLAY_CACHE_WITH_WINDOWS_ADAPTIVE_MAX_MS = 160;
  const CANVAS_OVERLAY_CACHE_ADAPTIVE_FRAME_MS = 24;
  const CANVAS_OVERLAY_CACHE_FRAME_MULTIPLIER = 3.25;
  const REWARD_TOAST_PLACEMENT = 'bottom-left';
  const REWARD_TOAST_HOLD_SECONDS = 1;
  const REWARD_TOAST_FADE_SECONDS = 0.45;
  const REWARD_TOAST_DURATION_SECONDS = REWARD_TOAST_HOLD_SECONDS + REWARD_TOAST_FADE_SECONDS;
  const REWARD_TOAST_SLIDE_UP_PX = 18;
  const REWARD_TOAST_BOTTOM_PANEL_GAP = 12;
  const REWARD_TOAST_BACKGROUND = 'rgba(16,32,51,0.68)';
  const REWARD_TOAST_BORDER = 'rgba(255,255,255,0.12)';

  function shouldRefreshUiChangeHudSnapshot(domains) {
    const set = new Set(domains || []);
    return UI_CHANGE_HUD_SNAPSHOT_DOMAINS.some((domain) => set.has(domain));
  }

  function mergeHudSnapshot(targetSnapshot, hudSnapshot) {
    const target = targetSnapshot || {};
    const source = hudSnapshot || null;
    if (!source) return target;
    target.cacheRevision = Number(source.cacheRevision || target.cacheRevision || 0);
    target.domainRevisions = source.domainRevisions || target.domainRevisions || {};
    HUD_SNAPSHOT_MERGE_KEYS.forEach((key) => {
      target[key] = source[key] || target[key];
    });
    return target;
  }

  function getDismissGuideDomAction(target) {
    const source = target || null;
    if (source && typeof source.hasAttribute === 'function' && source.hasAttribute('data-starfall-dismiss-guide')) {
      return { handled: true, type: 'dismissGuide' };
    }
    return { handled: false, type: '' };
  }

  function getHudRegionAction(region) {
    const source = region || {};
    if (source.type === 'station-prompt') return { handled: true, type: 'stationPrompt', action: source.action };
    if (source.type === 'minimap-toggle') return { handled: true, type: 'toggleMinimap' };
    if (source.type === 'quest-tracker-toggle') return { handled: true, type: 'toggleQuestTracker' };
    return { handled: false, type: '' };
  }

  function getWorldDerivedSnapshotUpdate(engine, data, options) {
    const source = engine || {};
    const settings = options || {};
    const update = {};
    if (typeof source.getRuntimeOverlaySnapshot === 'function') update.runtime = source.getRuntimeOverlaySnapshot();
    if (typeof source.getQuestNpcSnapshot === 'function') update.questNpcs = source.getQuestNpcSnapshot();
    if (source.runtime && Array.isArray(source.runtime.portals) && typeof source.getPortalSummary === 'function') {
      update.portals = source.runtime.portals.map((portal) => source.getPortalSummary(portal));
    }
    if (typeof source.getMapChannelSnapshot === 'function') update.channel = source.getMapChannelSnapshot();
    if (source.state) {
      const maps = data && Array.isArray(data.MAPS) ? data.MAPS : [];
      const player = source.state.player;
      update.map = maps.find((map) => map && map.id === source.state.mapId) || settings.currentMap;
      update.station = player && player.activeStation || '';
      update.portal = player && player.activePortalId || '';
      update.questNpc = player && player.activeQuestNpcId || '';
    }
    if (typeof source.currentMapLootDrops === 'function') update.lootDrops = source.currentMapLootDrops().slice();
    if (typeof source.findReachableLoot === 'function') update.nearbyLoot = source.findReachableLoot(100);
    if (typeof source.getWaveState === 'function' && source.state) {
      const clone = typeof settings.clonePlain === 'function' ? settings.clonePlain : (value) => value;
      update.wave = clone(source.getWaveState(source.state.mapId));
    }
    return update;
  }

  function createHudRuntimeUiHelpers() {
    return Object.freeze({
      shouldRefreshUiChangeHudSnapshot,
      mergeHudSnapshot,
      getDismissGuideDomAction,
      getHudRegionAction,
      getWorldDerivedSnapshotUpdate
    });
  }

  function getCanvasStatusHudBox(width, height, options) {
    const settings = options && options.settings || {};
    const video = settings.video || {};
    const runtime = options && options.runtime || {};
    const statusHudHeight = Number(options && options.statusHudHeight || 84);
    const statusHudY = Number(options && options.statusHudY || 688);
    const runtimeHudTop = runtime && Number(runtime.hudTop);
    const reservedHudHeight = Number.isFinite(runtimeHudTop) && runtimeHudTop > 0
      ? Math.max(96, height - runtimeHudTop)
      : Math.max(96, height - statusHudY);
    const h = Math.min(Math.round(statusHudHeight * Number(video.hudScale || 1)), reservedHudHeight);
    return {
      x: 0,
      y: Math.max(0, height - h),
      w: Math.max(260, width),
      h
    };
  }

  function getCanvasHudTop(width, height, options) {
    const player = options && options.player;
    if (!player || !player.classId) return height - 8;
    return getCanvasStatusHudBox(width, height, options).y;
  }

  function getCanvasUiBottom(width, height, padding, options) {
    return Math.max(88, getCanvasHudTop(width, height, options) - (Number(padding) || 8));
  }

  function getCanvasHudTheme(accentColor, uiTheme) {
    const accent = /^#[0-9a-f]{6}$/i.test(String(accentColor || '')) ? accentColor : '#7bdff2';
    const ui = uiTheme || {};
    return {
      accent,
      frame: ui.frame,
      frameDeep: ui.frameDeep,
      frameSoft: ui.frameSoft,
      panel: ui.parchment,
      panelShade: ui.parchmentDim,
      panelLine: ui.line,
      ink: ui.ink,
      gold: ui.gold,
      goldSoft: 'rgba(245,207,114,0.34)',
      goldDim: ui.lineSoft,
      cyanSoft: ui.cyanSoft
    };
  }

  function getCanvasHudDividerMetadata(x, box) {
    if (!Number.isFinite(x) || x <= box.x + 20 || x >= box.x + box.w - 20) return null;
    const top = box.y + 10;
    const bottom = box.y + box.h - 10;
    const px = Math.round(x);
    return {
      rects: [
        { x: px - 2, y: top + 3, w: 4, h: bottom - top - 6, fill: 'rgba(4,12,22,0.78)' },
        { x: px - 3, y: top + 8, w: 1, h: bottom - top - 16, fill: 'rgba(245,207,114,0.46)' },
        { x: px + 2, y: top + 8, w: 1, h: bottom - top - 16, fill: 'rgba(245,207,114,0.46)' },
        { x: px - 1, y: top + 6, w: 1, h: bottom - top - 12, fill: 'rgba(255,255,255,0.12)' }
      ],
      diamonds: [
        { cx: px, cy: top + 6, radius: 3, fill: 'rgba(216,170,85,0.7)' },
        { cx: px, cy: bottom - 6, radius: 3, fill: 'rgba(216,170,85,0.7)' }
      ]
    };
  }

  function getCanvasStatusHudFrameMetadata(box, layout, accentColor, uiTheme) {
    const theme = getCanvasHudTheme(accentColor, uiTheme);
    const frameLayout = layout || {};
    const panels = [];
    if (frameLayout.infoRight) {
      panels.push({
        x: box.x + 14,
        y: box.y + 14,
        w: Math.max(80, frameLayout.infoRight - box.x - 20),
        h: box.h - 28,
        radius: 5,
        fill: 'rgba(10,28,43,0.48)',
        stroke: 'rgba(245,207,114,0.14)'
      });
    }
    if (frameLayout.commandLeft) {
      panels.push({
        x: Math.max(box.x + 16, frameLayout.commandLeft + 8),
        y: box.y + 14,
        w: Math.max(72, box.x + box.w - frameLayout.commandLeft - 24),
        h: box.h - 28,
        radius: 5,
        fill: 'rgba(10,28,43,0.38)',
        stroke: 'rgba(245,207,114,0.12)'
      });
    }
    return {
      theme,
      shadow: {
        color: 'rgba(0,0,0,0.42)',
        blur: 16,
        offsetY: -3
      },
      shadowRect: {
        x: box.x + 2,
        y: box.y + 3,
        w: box.w - 4,
        h: box.h - 5,
        radius: 7,
        fill: theme.frameDeep,
        stroke: theme.goldSoft
      },
      rects: [
        { x: box.x + 6, y: box.y + 7, w: box.w - 12, h: box.h - 13, radius: 6, fill: theme.frame, stroke: 'rgba(245,207,114,0.36)' },
        { x: box.x + 11, y: box.y + 12, w: box.w - 22, h: box.h - 23, radius: 5, fill: 'rgba(6,15,26,0.92)', stroke: 'rgba(245,207,114,0.26)' }
      ],
      lines: [
        { x: box.x + 16, y: box.y + 12, w: box.w - 32, h: 1, fill: 'rgba(255,255,255,0.1)' },
        { x: box.x + 14, y: box.y + 7, w: box.w - 28, h: 1, fill: 'rgba(245,207,114,0.48)' },
        { x: box.x + 14, y: box.y + box.h - 10, w: box.w - 28, h: 1, fill: 'rgba(245,207,114,0.38)' }
      ],
      panels,
      dividers: [
        { x: frameLayout.infoRight },
        { x: frameLayout.commandLeft }
      ],
      accents: {
        x: box.x + 2,
        y: box.y + 3,
        w: box.w - 4,
        h: box.h - 5,
        options: {
          gems: box.w >= 560,
          inset: 10,
          centerTop: false,
          centerBottom: box.w >= 500
        }
      }
    };
  }

  function getCanvasHudChipMetadata(x, y, w, h, text, options, uiTheme) {
    const settings = options || {};
    const theme = getCanvasHudTheme(settings.accentColor, uiTheme);
    const active = !!settings.active;
    const fill = settings.fill || (active ? 'rgba(16,39,59,0.92)' : 'rgba(242,223,189,0.9)');
    const stroke = settings.stroke || (active ? 'rgba(245,207,114,0.34)' : theme.goldDim);
    return {
      rect: {
        x,
        y,
        w,
        h,
        radius: 7,
        fill,
        stroke
      },
      text: {
        value: text,
        x: x + w / 2,
        y: y + h / 2,
        color: settings.color || (active ? '#fff7df' : theme.ink),
        font: settings.font || '900 10px system-ui',
        align: 'center',
        baseline: 'middle',
        maxWidth: w - 10,
        maxLines: 1,
        lineHeight: 11
      }
    };
  }

  function getCanvasHudSocketMetadata(x, y, w, h, active, accentColor, uiTheme) {
    const theme = getCanvasHudTheme(accentColor, uiTheme);
    const enabled = !!active;
    const accentStroke = colorWithAlpha(theme.accent, 0.58);
    const accentFill = colorWithAlpha(theme.accent, 0.5);
    const accentCorner = colorWithAlpha(theme.accent, 0.72);
    return {
      shadow: enabled ? {
        color: theme.accent,
        blur: 8
      } : null,
      outer: {
        x,
        y,
        w,
        h,
        radius: 7,
        fill: enabled ? '#2b5872' : '#102033',
        stroke: enabled ? accentStroke : 'rgba(245,207,114,0.42)'
      },
      inner: {
        x: x + 4,
        y: y + 4,
        w: w - 8,
        h: h - 8,
        radius: 5,
        fill: enabled ? 'rgba(89,216,255,0.18)' : 'rgba(0,0,0,0.28)',
        stroke: 'rgba(255,255,255,0.1)'
      },
      topLine: {
        x: x + 8,
        y: y + 6,
        w: Math.max(8, w - 16),
        h: 1,
        fill: enabled ? accentFill : 'rgba(255,255,255,0.14)'
      },
      bottomLine: {
        x: x + 6,
        y: y + h - 7,
        w: Math.max(8, w - 12),
        h: 1,
        fill: 'rgba(0,0,0,0.32)'
      },
      corners: {
        x,
        y,
        w,
        h,
        options: {
          stroke: enabled ? accentCorner : 'rgba(245,207,114,0.5)',
          fill: 'rgba(3,10,19,0.86)',
          pinFill: enabled ? accentCorner : 'rgba(245,207,114,0.46)',
          pins: w >= 42 && h >= 42
        }
      }
    };
  }

  function drawCanvasFillRectEntry(ctx, entry) {
    if (!ctx || !entry || typeof ctx.fillRect !== 'function') return;
    ctx.fillStyle = entry.fill;
    ctx.fillRect(entry.x, entry.y, entry.w, entry.h);
  }

  function drawCanvasDiamondEntry(ctx, entry) {
    if (
      !ctx ||
      !entry ||
      typeof ctx.beginPath !== 'function' ||
      typeof ctx.moveTo !== 'function' ||
      typeof ctx.lineTo !== 'function' ||
      typeof ctx.closePath !== 'function' ||
      typeof ctx.fill !== 'function'
    ) return;
    ctx.fillStyle = entry.fill;
    ctx.beginPath();
    ctx.moveTo(entry.cx, entry.cy - entry.radius);
    ctx.lineTo(entry.cx + entry.radius, entry.cy);
    ctx.lineTo(entry.cx, entry.cy + entry.radius);
    ctx.lineTo(entry.cx - entry.radius, entry.cy);
    ctx.closePath();
    ctx.fill();
  }

  function drawCanvasHudDivider(ctx, x, box) {
    const divider = getCanvasHudDividerMetadata(x, box);
    if (!ctx || !divider) return false;
    ctx.save();
    divider.rects.forEach((rect) => drawCanvasFillRectEntry(ctx, rect));
    divider.diamonds.forEach((diamond) => drawCanvasDiamondEntry(ctx, diamond));
    ctx.restore();
    return true;
  }

  function drawCanvasStatusHudFrame(ctx, box, layout, accentColor, options) {
    const settings = options || {};
    const drawRoundRect = settings.drawRoundRect;
    const drawCanvasUiFrameAccents = settings.drawCanvasUiFrameAccents;
    const frame = getCanvasStatusHudFrameMetadata(box, layout, accentColor, settings.uiTheme);
    if (!ctx || !frame || typeof drawRoundRect !== 'function' || typeof drawCanvasUiFrameAccents !== 'function') return false;
    ctx.save();
    ctx.shadowColor = frame.shadow.color;
    ctx.shadowBlur = frame.shadow.blur;
    ctx.shadowOffsetY = frame.shadow.offsetY;
    drawRoundRectEntry(drawRoundRect, frame.shadowRect);
    ctx.shadowColor = 'transparent';
    frame.rects.forEach((rect) => drawRoundRectEntry(drawRoundRect, rect));
    frame.lines.forEach((line) => drawCanvasFillRectEntry(ctx, line));
    frame.panels.forEach((panel) => drawRoundRectEntry(drawRoundRect, panel));
    frame.dividers.forEach((divider) => {
      drawCanvasHudDivider(ctx, divider.x, box);
    });
    drawCanvasUiFrameAccents(frame.accents);
    ctx.restore();
    return true;
  }

  function drawCanvasHudChip(ctx, x, y, w, h, text, options) {
    const settings = options || {};
    const drawRoundRect = settings.drawRoundRect;
    const drawCanvasText = settings.drawCanvasText;
    const chip = getCanvasHudChipMetadata(x, y, w, h, text, settings.metadataOptions || settings, settings.uiTheme);
    if (!ctx || !chip || typeof drawRoundRect !== 'function' || typeof drawCanvasText !== 'function') return false;
    drawRoundRectEntry(drawRoundRect, chip.rect);
    drawCanvasTextEntry(drawCanvasText, chip.text);
    return true;
  }

  function drawCanvasHudSocket(ctx, x, y, w, h, active, accentColor, options) {
    const settings = options || {};
    const drawRoundRect = settings.drawRoundRect;
    const drawCanvasUiSlotCorners = settings.drawCanvasUiSlotCorners;
    const socket = getCanvasHudSocketMetadata(x, y, w, h, active, accentColor, settings.uiTheme);
    if (!ctx || !socket || typeof drawRoundRect !== 'function' || typeof drawCanvasUiSlotCorners !== 'function') return false;
    ctx.save();
    if (socket.shadow) {
      ctx.shadowColor = socket.shadow.color;
      ctx.shadowBlur = socket.shadow.blur;
    }
    drawRoundRectEntry(drawRoundRect, socket.outer);
    ctx.shadowBlur = 0;
    drawRoundRectEntry(drawRoundRect, socket.inner);
    drawCanvasFillRectEntry(ctx, socket.topLine);
    drawCanvasFillRectEntry(ctx, socket.bottomLine);
    drawCanvasUiSlotCorners(socket.corners);
    ctx.restore();
    return true;
  }

  function getCanvasHudMeterMetadata(label, value, max, color, x, y, w, h, options) {
    const settings = options || {};
    const animationKey = label === 'HP' ? 'hp' : label === 'MP' ? 'mp' : label === 'XP' ? 'xp' : '';
    const animated = settings.animated || null;
    const amount = Math.max(0, Number(animated ? animated.value : value) || 0);
    const total = Math.max(1, Number(animated ? animated.max : max) || 1);
    const ratio = clamp(amount / total, 0, 1);
    const formatNumber = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (number) => String(Math.round(Number(number) || 0));
    const formatMeter = typeof settings.formatMeterPercent === 'function'
      ? settings.formatMeterPercent
      : (number, maximum) => `${Math.round(clamp(number / maximum, 0, 1) * 100)}%`;
    const formatXpMeter = typeof settings.formatXpMeterPercent === 'function'
      ? settings.formatXpMeterPercent
      : formatMeter;
    const valueText = `${formatNumber(amount)}/${formatNumber(total)}`;
    const percentText = label === 'XP' ? formatXpMeter(amount, total) : formatMeter(amount, total);
    const fillW = Math.max(4, Math.round(w * ratio));
    return {
      animationKey,
      amount,
      total,
      ratio,
      valueText,
      percentText,
      outer: {
        x: x - 1,
        y: y - 1,
        w: w + 2,
        h: h + 2,
        radius: 6,
        fill: 'rgba(7,14,24,0.94)',
        stroke: 'rgba(245,207,114,0.42)'
      },
      inner: {
        x: x + 1,
        y: y + 1,
        w: w - 2,
        h: h - 2,
        radius: 5,
        fill: 'rgba(0,0,0,0.48)',
        stroke: 'rgba(255,255,255,0.1)'
      },
      clip: {
        x: x + 1,
        y: y + 1,
        w: w - 2,
        h: h - 2,
        radius: 7
      },
      fill: {
        x: x + 1,
        y: y + 1,
        w: Math.max(0, fillW - 2),
        h: h - 2,
        fill: color
      },
      shine: {
        x: x + 1,
        y: y + 1,
        w: Math.max(0, fillW - 2),
        h: Math.max(2, Math.floor(h * 0.32)),
        fill: 'rgba(255,255,255,0.16)'
      },
      edge: {
        x: x + Math.max(1, fillW - 3),
        y: y + 1,
        w: 2,
        h: h - 2,
        fill: 'rgba(16,32,51,0.18)'
      },
      labelText: {
        value: label,
        x: x + 9,
        y: y + h / 2,
        color: '#ffffff',
        font: '900 10px system-ui',
        baseline: 'middle',
        maxWidth: Math.max(40, w * 0.35),
        maxLines: 1,
        lineHeight: 11
      },
      percentTextMeta: label === 'XP' ? {
        value: percentText,
        x: x + w / 2,
        y: y + h / 2,
        color: '#ffffff',
        font: '950 12px system-ui',
        align: 'center',
        baseline: 'middle',
        maxWidth: Math.max(42, w * 0.22),
        maxLines: 1,
        lineHeight: 13
      } : null,
      valueTextMeta: {
        value: valueText,
        x: x + w - 9,
        y: y + h / 2,
        color: '#ffffff',
        font: label === 'XP' ? '900 10px system-ui' : '900 11px system-ui',
        align: 'right',
        baseline: 'middle',
        maxWidth: Math.max(56, label === 'XP' ? w * 0.34 : w * 0.45),
        maxLines: 1,
        lineHeight: 12
      }
    };
  }

  function drawCanvasHudMeter(ctx, label, value, max, color, x, y, w, h, options) {
    const settings = options || {};
    const drawRoundRect = settings.drawRoundRect;
    const drawCanvasText = settings.drawCanvasText;
    const meter = getCanvasHudMeterMetadata(label, value, max, color, x, y, w, h, {
      animated: settings.animated,
      formatAbbreviatedInteger: settings.formatAbbreviatedInteger,
      formatMeterPercent: settings.formatMeterPercent,
      formatXpMeterPercent: settings.formatXpMeterPercent
    });
    if (!ctx || !meter || typeof drawRoundRect !== 'function' || typeof drawCanvasText !== 'function') return false;
    drawRoundRectEntry(drawRoundRect, meter.outer);
    drawRoundRectEntry(drawRoundRect, meter.inner);
    ctx.save();
    if (typeof ctx.roundRect === 'function' && typeof ctx.clip === 'function') {
      ctx.beginPath();
      ctx.roundRect(meter.clip.x, meter.clip.y, meter.clip.w, meter.clip.h, meter.clip.radius);
      ctx.clip();
    }
    ctx.fillStyle = meter.fill.fill;
    ctx.fillRect(meter.fill.x, meter.fill.y, meter.fill.w, meter.fill.h);
    ctx.fillStyle = meter.shine.fill;
    ctx.fillRect(meter.shine.x, meter.shine.y, meter.shine.w, meter.shine.h);
    ctx.fillStyle = meter.edge.fill;
    ctx.fillRect(meter.edge.x, meter.edge.y, meter.edge.w, meter.edge.h);
    ctx.restore();
    drawCanvasTextEntry(drawCanvasText, meter.labelText);
    drawCanvasTextEntry(drawCanvasText, meter.percentTextMeta);
    drawCanvasTextEntry(drawCanvasText, meter.valueTextMeta);
    return true;
  }

  function createHudChromeUiHelpers() {
    return Object.freeze({
      getCanvasStatusHudBox,
      getCanvasHudTop,
      getCanvasUiBottom,
      getCanvasHudTheme,
      getCanvasHudDividerMetadata,
      getCanvasStatusHudFrameMetadata,
      getCanvasHudChipMetadata,
      getCanvasHudSocketMetadata,
      drawCanvasHudDivider,
      drawCanvasStatusHudFrame,
      drawCanvasHudChip,
      drawCanvasHudSocket,
      getCanvasHudMeterMetadata,
      drawCanvasHudMeter
    });
  }

  function getCanvasBossEncounterHudMetadata(boss, width, options) {
    if (!boss || !boss.active) return null;
    const settings = options || {};
    const w = clamp(Math.round(width * 0.44), 360, 560);
    const x = Math.round((width - w) / 2);
    const y = settings.y != null ? Number(settings.y) : 14;
    const h = settings.h != null ? Number(settings.h) : 72;
    const ratio = clamp(Number(boss.hpRatio || 0), 0, 1);
    const color = boss.color || '#ffbe55';
    const accent = boss.accentColor || '#ffffff';
    const phaseCount = Math.max(1, Number(boss.phaseCount || 1));
    const phaseIndex = clamp(Math.floor(Number(boss.phaseIndex || 0)), 0, phaseCount - 1);
    const fillW = Math.max(5, Math.round((w - 22) * ratio));
    const phaseText = `${boss.phaseName || 'Phase'} ${phaseIndex + 1}/${phaseCount}`;
    const actionText = boss.pendingActionLabel ? `${boss.pendingActionLabel} ${(boss.pendingActionProgress * 100).toFixed(0)}%` : boss.mechanic || '';
    const tickW = Math.max(28, Math.min(62, (w - 30) / phaseCount - 4));
    const tickY = y + 49;
    const ticks = [];
    for (let index = 0; index < phaseCount; index += 1) {
      ticks.push({
        x: x + 12 + index * (tickW + 4),
        y: tickY,
        w: tickW,
        h: 3,
        fill: index <= phaseIndex ? colorWithAlpha(accent, 0.84) : 'rgba(255,255,255,0.18)'
      });
    }
    return {
      box: { x, y, w, h },
      ratio,
      color,
      accent,
      phaseCount,
      phaseIndex,
      frame: {
        x,
        y,
        w,
        h,
        radius: 10,
        fill: 'rgba(9,31,59,0.78)',
        stroke: colorWithAlpha(accent, 0.42)
      },
      barShadow: {
        color,
        blur: 14
      },
      barFrame: {
        x: x + 11,
        y: y + 31,
        w: w - 22,
        h: 16,
        radius: 8,
        fill: 'rgba(255,255,255,0.12)',
        stroke: colorWithAlpha(color, 0.34)
      },
      fill: {
        x: x + 11,
        y: y + 31,
        w: fillW,
        h: 16,
        fill: colorWithAlpha(color, 0.86)
      },
      accentFill: {
        x: x + 11,
        y: y + 31,
        w: Math.max(3, Math.round(fillW * 0.18)),
        h: 16,
        fill: colorWithAlpha(accent, 0.72)
      },
      titleText: {
        value: boss.bossName || boss.name || 'Boss',
        x: x + 14,
        y: y + 15,
        color: '#ffffff',
        font: '950 14px system-ui',
        baseline: 'middle',
        maxWidth: w * 0.48,
        maxLines: 1,
        lineHeight: 15
      },
      hpText: {
        value: `${Math.round(ratio * 100)}%`,
        x: x + w - 14,
        y: y + 15,
        color: '#ffffff',
        font: '950 13px system-ui',
        align: 'right',
        baseline: 'middle',
        maxWidth: 58,
        maxLines: 1,
        lineHeight: 14
      },
      phaseText: {
        value: phaseText,
        x: x + 14,
        y: y + 58,
        color: accent,
        font: '900 10px system-ui',
        baseline: 'middle',
        maxWidth: w * 0.42,
        maxLines: 1,
        lineHeight: 11
      },
      actionText: {
        value: actionText,
        x: x + w - 14,
        y: y + 58,
        color: '#f7fbff',
        font: '850 10px system-ui',
        align: 'right',
        baseline: 'middle',
        maxWidth: w * 0.48,
        maxLines: 1,
        lineHeight: 11
      },
      ticks
    };
  }

  function getCanvasBossEncounterOverlayMetadata(boss, width, height, options) {
    if (!boss) return null;
    const panel = boss.intro || boss.clear;
    if (!panel) return null;
    const settings = options || {};
    const formatNumber = typeof settings.formatIntegerWithCommas === 'function'
      ? settings.formatIntegerWithCommas
      : (value) => String(Math.round(Number(value) || 0));
    const clear = !!boss.clear;
    const w = clamp(Math.round(width * 0.42), 340, 520);
    const h = clear ? 126 : 112;
    const x = Math.round((width - w) / 2);
    const y = clear ? Math.max(92, Math.round(height * 0.18)) : 98;
    const color = panel.color || boss.color || '#ffbe55';
    const accent = panel.accentColor || boss.accentColor || '#ffffff';
    const title = clear ? 'Boss Cleared' : (boss.bossName || boss.name || 'Boss Encounter');
    const body = clear ? panel.text || 'Encounter cleared.' : boss.intro && (boss.intro.text || boss.intro.intro) || boss.summary || '';
    const detail = clear
      ? `+${formatNumber(panel.xp || 0)} XP  +${formatNumber(panel.coins || 0)} coins`
      : boss.mechanic || '';
    const dropText = clear && Array.isArray(panel.drops) && panel.drops.length
      ? panel.drops.map((drop) => `${drop.quantity > 1 ? `${drop.quantity}x ` : ''}${drop.name}`).join(', ')
      : '';
    return {
      clear,
      box: { x, y, w, h },
      color,
      accent,
      frame: {
        x,
        y,
        w,
        h,
        radius: 12,
        fill: 'rgba(9,31,59,0.88)',
        stroke: colorWithAlpha(color, 0.68)
      },
      stripe: {
        x: x + 10,
        y: y + 10,
        w: w - 20,
        h: 5,
        fill: colorWithAlpha(color, 0.22)
      },
      titleText: {
        value: title,
        x: x + 18,
        y: y + 28,
        color: '#ffffff',
        font: '950 16px system-ui',
        baseline: 'middle',
        maxWidth: w - 36,
        maxLines: 1,
        lineHeight: 17
      },
      bodyText: {
        value: body,
        x: x + 18,
        y: y + 52,
        color: '#f7fbff',
        font: '800 11px system-ui',
        maxWidth: w - 36,
        maxLines: 2,
        lineHeight: 14
      },
      detailText: {
        value: detail,
        x: x + 18,
        y: y + h - 24,
        color: accent,
        font: '900 11px system-ui',
        maxWidth: w - 36,
        maxLines: 1,
        lineHeight: 12
      },
      dropText: dropText ? {
        value: dropText,
        x: x + 18,
        y: y + h - 42,
        color: '#ffffff',
        font: '800 10px system-ui',
        maxWidth: w - 36,
        maxLines: 1,
        lineHeight: 11
      } : null
    };
  }

  function drawCanvasTextEntry(drawCanvasText, entry) {
    if (!entry || typeof drawCanvasText !== 'function') return;
    drawCanvasText(entry);
  }

  function drawRoundRectEntry(drawRoundRect, entry) {
    if (!entry || typeof drawRoundRect !== 'function') return;
    drawRoundRect(entry);
  }

  function drawCanvasBossEncounterHud(ctx, boss, width, options) {
    const settings = options || {};
    const drawRoundRect = settings.drawRoundRect;
    const drawCanvasText = settings.drawCanvasText;
    const hud = getCanvasBossEncounterHudMetadata(boss, width, settings.metadataOptions);
    if (!ctx || !hud || typeof drawRoundRect !== 'function' || typeof drawCanvasText !== 'function') return false;
    ctx.save();
    drawRoundRectEntry(drawRoundRect, hud.frame);
    ctx.shadowColor = hud.barShadow.color;
    ctx.shadowBlur = hud.barShadow.blur;
    drawRoundRectEntry(drawRoundRect, hud.barFrame);
    ctx.shadowBlur = 0;
    ctx.fillStyle = hud.fill.fill;
    ctx.fillRect(hud.fill.x, hud.fill.y, hud.fill.w, hud.fill.h);
    ctx.fillStyle = hud.accentFill.fill;
    ctx.fillRect(hud.accentFill.x, hud.accentFill.y, hud.accentFill.w, hud.accentFill.h);
    drawCanvasTextEntry(drawCanvasText, hud.titleText);
    drawCanvasTextEntry(drawCanvasText, hud.hpText);
    drawCanvasTextEntry(drawCanvasText, hud.phaseText);
    drawCanvasTextEntry(drawCanvasText, hud.actionText);
    hud.ticks.forEach((tick) => {
      ctx.fillStyle = tick.fill;
      ctx.fillRect(tick.x, tick.y, tick.w, tick.h);
    });
    ctx.restore();
    return true;
  }

  function drawCanvasBossEncounterOverlays(ctx, boss, width, height, options) {
    const settings = options || {};
    const drawRoundRect = settings.drawRoundRect;
    const drawCanvasText = settings.drawCanvasText;
    const overlay = getCanvasBossEncounterOverlayMetadata(boss, width, height, settings.metadataOptions);
    if (!ctx || !overlay || typeof drawRoundRect !== 'function' || typeof drawCanvasText !== 'function') return false;
    ctx.save();
    drawRoundRectEntry(drawRoundRect, overlay.frame);
    ctx.fillStyle = overlay.stripe.fill;
    ctx.fillRect(overlay.stripe.x, overlay.stripe.y, overlay.stripe.w, overlay.stripe.h);
    drawCanvasTextEntry(drawCanvasText, overlay.titleText);
    drawCanvasTextEntry(drawCanvasText, overlay.bodyText);
    drawCanvasTextEntry(drawCanvasText, overlay.detailText);
    drawCanvasTextEntry(drawCanvasText, overlay.dropText);
    ctx.restore();
    return true;
  }

  function createHudBossEncounterUiHelpers() {
    return Object.freeze({
      getCanvasBossEncounterHudMetadata,
      getCanvasBossEncounterOverlayMetadata,
      drawCanvasBossEncounterHud,
      drawCanvasBossEncounterOverlays
    });
  }

  function getCombatMetricsPanelBox(width, bottomY, state, options) {
    const settings = options || {};
    const w = 228;
    const h = 154;
    const maxBottom = Number(bottomY || settings.canvasViewHeight || 806);
    const panelState = state || {};
    const defaultX = 14;
    const defaultY = 54;
    const rawX = panelState.userPlaced ? Number(panelState.x || defaultX) : defaultX;
    const rawY = panelState.userPlaced ? Number(panelState.y || defaultY) : defaultY;
    return {
      x: clamp(rawX, 8, Math.max(8, Number(width || settings.canvasViewWidth || 1280) - w - 8)),
      y: clamp(rawY, 8, Math.max(8, maxBottom - h - 10)),
      w,
      h
    };
  }

  function getMinimapBox(width, height, bottomY, state, options) {
    const settings = options || {};
    const minimapState = state || {};
    const compact = !!(minimapState && minimapState.compact);
    const boxW = compact ? 144 : width < 700 ? 182 : 220;
    const boxH = compact ? 64 : width < 700 ? 112 : 132;
    const defaultX = width - boxW - 16;
    const defaultY = 16;
    const rawX = minimapState.userPlaced ? Number(minimapState.x || defaultX) : defaultX;
    const rawY = minimapState.userPlaced ? Number(minimapState.y || defaultY) : defaultY;
    const uiBottom = typeof settings.getCanvasUiBottom === 'function'
      ? settings.getCanvasUiBottom(width, height, 8)
      : height;
    const maxBottom = Number(bottomY || uiBottom);
    return {
      x: clamp(rawX, 8, Math.max(8, width - boxW - 8)),
      y: clamp(rawY, 8, Math.max(8, maxBottom - boxH)),
      w: boxW,
      h: boxH,
      compact
    };
  }

  function getHudWidgetPointerAction(region) {
    const source = region || {};
    if (source.type === 'minimap-drag') {
      return {
        handled: true,
        type: 'startHudWidgetDrag',
        dragKey: 'minimapDrag',
        boxType: 'minimap',
        shouldPreventDefault: true
      };
    }
    if (source.type === 'quest-tracker-drag') {
      return {
        handled: true,
        type: 'startHudWidgetDrag',
        dragKey: 'questTrackerDrag',
        boxType: 'questTracker',
        shouldPreventDefault: true
      };
    }
    if (source.type === 'combat-metrics-drag') {
      return {
        handled: true,
        type: 'startHudWidgetDrag',
        dragKey: 'combatMetricsDrag',
        boxType: 'combatMetrics',
        shouldPreventDefault: true
      };
    }
    return {
      handled: false,
      type: '',
      dragKey: '',
      boxType: '',
      shouldPreventDefault: false
    };
  }

  function getHudWidgetReleaseAction(state) {
    const source = state || {};
    const dragKey = HUD_WIDGET_DRAG_KEYS.find((key) => !!source[key]) || '';
    if (!dragKey) return { handled: false, dragKey: '', shouldDraw: false };
    return { handled: true, dragKey, shouldDraw: true };
  }

  function getHudWidgetCancelAction() {
    return {
      handled: true,
      dragKeys: HUD_WIDGET_DRAG_KEYS.slice(),
      shouldDraw: true
    };
  }

  function getHudWidgetMoveAction(state) {
    const source = state || {};
    const entry = HUD_WIDGET_DRAG_ENTRIES.find((item) => !!source[item.dragKey]) || null;
    if (!entry) {
      return {
        handled: false,
        dragKey: '',
        drag: null,
        stateKey: '',
        boxType: '',
        bottomInset: 0,
        shouldDraw: false,
        shouldPreventDefault: false
      };
    }
    return {
      handled: true,
      dragKey: entry.dragKey,
      drag: source[entry.dragKey],
      stateKey: entry.stateKey,
      boxType: entry.boxType,
      bottomInset: entry.bottomInset,
      shouldDraw: true,
      shouldPreventDefault: true
    };
  }

  function getHudWidgetDragUpdate(point, drag, box, bounds) {
    const sourcePoint = point || {};
    const sourceDrag = drag || {};
    const sourceBox = box || {};
    const settings = bounds || {};
    const width = Number(settings.width || 0);
    const bottomLimit = Number(settings.bottomLimit || settings.height || 0);
    const bottomInset = Number(settings.bottomInset || 0);
    return {
      x: clamp(Number(sourcePoint.x || 0) - Number(sourceDrag.dx || 0), 8, Math.max(8, width - Number(sourceBox.w || 0) - 8)),
      y: clamp(Number(sourcePoint.y || 0) - Number(sourceDrag.dy || 0), 8, Math.max(8, bottomLimit - Number(sourceBox.h || 0) - bottomInset)),
      userPlaced: true
    };
  }

  function createHudWidgetUiHelpers() {
    return Object.freeze({
      getCombatMetricsPanelBox,
      getMinimapBox,
      getHudWidgetPointerAction,
      getHudWidgetMoveAction,
      getHudWidgetDragUpdate,
      getHudWidgetReleaseAction,
      getHudWidgetCancelAction,
      getQuestTrackerEntries,
      getQuestTrackerNaturalHeight,
      getQuestTrackerBox
    });
  }

  function getRewardPopupDisplayState(popup, now, width) {
    if (!popup) return null;
    const age = now - Number(popup.createdAt || now);
    const remaining = Number(popup.expiresAt || now) - now;
    const alpha = clamp(Math.min(1, age / 0.18, remaining / 0.28), 0, 1);
    const rankUp = popup.kind === 'attunementRankUp';
    const boxW = rankUp ? Math.min(470, Math.max(310, width - 32)) : Math.min(360, Math.max(250, width - 32));
    const boxH = rankUp ? 104 : 72;
    const x = Math.round((width - boxW) / 2);
    const y = rankUp ? 54 : 64;
    return {
      age,
      remaining,
      alpha,
      rankUp,
      x,
      y,
      w: boxW,
      h: boxH,
      shadowColor: rankUp ? 'rgba(136,86,197,0.5)' : 'rgba(9,31,59,0.28)',
      shadowBlur: rankUp ? 24 : 16,
      shadowOffsetY: rankUp ? 8 : 6,
      fill: rankUp ? 'rgba(34,20,56,0.96)' : 'rgba(255,250,240,0.96)',
      stroke: rankUp ? 'rgba(255,209,102,0.94)' : 'rgba(255,209,102,0.86)',
      pulse: rankUp ? Math.sin(clamp(age / 0.5, 0, 1) * Math.PI) : 0,
      textX: rankUp ? x + boxW / 2 : x + 16,
      title: popup.title || 'Rewards',
      titleColor: rankUp ? '#ffe16a' : '#102033',
      titleFont: rankUp ? '950 18px system-ui' : '900 13px system-ui',
      titleLineHeight: rankUp ? 20 : 14,
      titleAlign: rankUp ? 'center' : 'left',
      bodyText: popup.text || (popup.parts || []).join(', '),
      bodyColor: rankUp ? '#ffffff' : '#5f6f7a',
      bodyFont: rankUp ? '900 12px system-ui' : '800 11px system-ui',
      bodyY: y + (rankUp ? 42 : 34),
      bodyAlign: rankUp ? 'center' : 'left',
      partsText: rankUp && (popup.parts || []).length ? (popup.parts || []).slice(0, 4).join('  |  ') : ''
    };
  }

  function rectsOverlap(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  }

  function getTopToastPosition(width, height, boxW, boxH, toastY, blockers, options) {
    const settings = options || {};
    const gap = Number(settings.gap || 8);
    const toastBlockers = Array.isArray(blockers) ? blockers : [];
    const preferred = { x: Math.round((width - boxW) / 2), y: toastY, w: boxW, h: boxH };
    if (!toastBlockers.some((region) => rectsOverlap(preferred, region))) return preferred;
    const rowBlockers = toastBlockers
      .filter((region) => region.y < toastY + boxH + gap && region.y + region.h > toastY - gap)
      .map((region) => ({
        start: clamp(region.x - gap, 16, width - 16),
        end: clamp(region.x + region.w + gap, 16, width - 16)
      }))
      .sort((a, b) => a.start - b.start);
    let cursor = 16;
    let bestGap = null;
    rowBlockers.forEach((interval) => {
      if (interval.start - cursor >= boxW) {
        const candidate = { start: cursor, end: interval.start };
        if (!bestGap || candidate.end - candidate.start > bestGap.end - bestGap.start) bestGap = candidate;
      }
      cursor = Math.max(cursor, interval.end);
    });
    if (width - 16 - cursor >= boxW) {
      const candidate = { start: cursor, end: width - 16 };
      if (!bestGap || candidate.end - candidate.start > bestGap.end - bestGap.start) bestGap = candidate;
    }
    if (bestGap) {
      return {
        x: Math.round(bestGap.start + Math.max(0, bestGap.end - bestGap.start - boxW) / 2),
        y: toastY,
        w: boxW,
        h: boxH
      };
    }
    const clearY = toastBlockers
      .filter((region) => preferred.x < region.x + region.w && preferred.x + preferred.w > region.x)
      .reduce((nextY, region) => Math.max(nextY, region.y + region.h + gap), toastY);
    preferred.y = clamp(clearY, 12, Math.max(12, height - boxH - 96));
    return preferred;
  }

  function getCanvasToastDisplayEntries(toasts, now, width, height, options) {
    const settings = options || {};
    const source = Array.isArray(toasts) ? toasts : [];
    const rewardPlacement = settings.rewardPlacement || REWARD_TOAST_PLACEMENT;
    const maxW = Math.min(430, width - 32);
    const boxH = Number(settings.boxHeight || 38);
    const gap = Number(settings.gap || 8);
    const measureText = typeof settings.measureText === 'function'
      ? settings.measureText
      : (message) => String(message || '').length * 7;
    const blockers = (settings.hitRegions || []).filter((region) =>
      region && (
        region.type === 'window-shell' ||
        region.type === 'menu-shell' ||
        region.type === 'minimap-drag' ||
        region.type === 'quest-tracker-drag'
      )
    );
    const entries = [];
    const addToastStack = (stack, placement) => {
      stack.forEach((toast, index) => {
        const message = String(toast.message || '');
        const age = now - Number(toast.createdAt || now);
        const remaining = Number(toast.expiresAt || now) - now;
        const alpha = clamp(Math.min(1, age / 0.16, remaining / 0.24), 0, 1);
        const boxW = Math.min(maxW, Math.max(240, measureText(message) + 34));
        let x = Math.round((width - boxW) / 2);
        let y = placement === 'center'
          ? Math.round((height - boxH) / 2) + (index - (stack.length - 1) / 2) * (boxH + gap)
          : 18 + index * (boxH + gap) + Math.max(0, 3 - stack.length) * 2;
        if (placement === 'top') {
          const topPosition = getTopToastPosition(width, height, boxW, boxH, y, blockers, { gap });
          x = topPosition.x;
          y = topPosition.y;
        }
        entries.push({ message, alpha, x, y, w: boxW, h: boxH, placement });
      });
    };
    addToastStack(source.filter((toast) => toast.placement === 'center').slice(-2), 'center');
    addToastStack(source.filter((toast) => toast.placement !== 'center' && toast.placement !== rewardPlacement).slice(-3), 'top');
    return entries;
  }

  function getCanvasRewardToastDisplayEntries(toasts, now, width, height, options) {
    const settings = options || {};
    const source = Array.isArray(toasts) ? toasts : [];
    const rewardPlacement = settings.rewardPlacement || REWARD_TOAST_PLACEMENT;
    const rewardBoxH = Number(settings.boxHeight || 34);
    const rewardGap = Number(settings.gap || 7);
    const availableTop = Number(settings.availableTop || 16);
    const bottomPanelGap = Number(settings.bottomPanelGap || REWARD_TOAST_BOTTOM_PANEL_GAP);
    const holdSeconds = Number(settings.holdSeconds || REWARD_TOAST_HOLD_SECONDS);
    const fadeSeconds = Number(settings.fadeSeconds || REWARD_TOAST_FADE_SECONDS);
    const slideUpPx = Number(settings.slideUpPx || REWARD_TOAST_SLIDE_UP_PX);
    const statusHudTop = Number.isFinite(Number(settings.statusHudTop))
      ? Number(settings.statusHudTop)
      : getCanvasStatusHudBox(width, height, settings.statusHudOptions).y;
    const bottom = Math.max(availableTop + rewardBoxH, statusHudTop - bottomPanelGap);
    const maxRewardToasts = Math.max(1, Math.floor((bottom - availableTop + rewardGap) / (rewardBoxH + rewardGap)));
    const visible = source.filter((toast) => toast.placement === rewardPlacement).slice(-maxRewardToasts);
    const rewardMaxW = Math.min(360, width - 32);
    const measureText = typeof settings.measureText === 'function'
      ? settings.measureText
      : (message) => String(message || '').length * 7;
    return visible.map((toast, index) => {
      const message = String(toast.message || '');
      const age = now - Number(toast.createdAt || now);
      const fadeProgress = clamp((age - holdSeconds) / fadeSeconds, 0, 1);
      const alpha = clamp(Math.min(1, age / 0.12, 1 - fadeProgress), 0, 1);
      if (alpha <= 0) return null;
      const slideY = fadeProgress * slideUpPx;
      const stackFromBottom = visible.length - 1 - index;
      const boxW = Math.min(rewardMaxW, Math.max(210, measureText(message) + 36));
      const xp = String(toast.kind || '') === 'xp';
      return {
        message,
        alpha,
        x: 16,
        y: Math.round(bottom - rewardBoxH - stackFromBottom * (rewardBoxH + rewardGap) - slideY),
        w: boxW,
        h: rewardBoxH,
        xp,
        accent: xp ? 'rgba(123,223,242,0.84)' : 'rgba(255,209,102,0.84)',
        glow: xp ? 'rgba(123,223,242,0.16)' : 'rgba(255,209,102,0.14)',
        textColor: xp ? 'rgba(216,248,255,0.94)' : 'rgba(255,243,194,0.94)'
      };
    }).filter(Boolean);
  }

  function getCanvasToastQueueUpdate(existingToasts, message, now, options) {
    const settings = options || {};
    const rewardPlacement = settings.rewardPlacement || REWARD_TOAST_PLACEMENT;
    const topDuration = Number(settings.topDuration || 2.6);
    const rewardDuration = Number(settings.rewardDuration || REWARD_TOAST_DURATION_SECONDS);
    const maxNormalToasts = Math.max(1, Math.floor(Number(settings.maxNormalToasts || 3) || 3));
    const maxRewardToasts = Math.max(1, Math.floor(Number(settings.maxRewardToasts || 40) || 40));
    const payload = message && typeof message === 'object' ? message : { message };
    const text = String(payload.message || payload.text || '').trim();
    if (!text) return null;
    const requestedPlacement = String(payload.placement || '').toLowerCase();
    const placement = requestedPlacement === 'center'
      ? 'center'
      : requestedPlacement === rewardPlacement ? rewardPlacement : 'top';
    const expiresAt = now + (placement === rewardPlacement ? rewardDuration : topDuration);
    const nextToasts = (Array.isArray(existingToasts) ? existingToasts : [])
      .filter((toast) => Number(toast.expiresAt || 0) > now)
      .concat({
        message: text,
        placement,
        kind: String(payload.kind || ''),
        createdAt: now,
        expiresAt
      });
    return {
      text,
      placement,
      expiresAt,
      toasts: nextToasts
        .filter((toast) => toast.placement !== rewardPlacement)
        .slice(-maxNormalToasts)
        .concat(nextToasts.filter((toast) => toast.placement === rewardPlacement).slice(-maxRewardToasts))
    };
  }

  function getCanvasRewardPopupQueueUpdate(existingPopups, reward, now, options) {
    const settings = options || {};
    const payload = reward || {};
    const title = String(payload.title || 'Rewards').trim();
    const text = String(payload.text || '').trim();
    if (!title && !text) return null;
    const kind = String(payload.kind || '');
    const duration = kind === 'attunementRankUp'
      ? Number(settings.rankUpDuration || 4.2)
      : Number(settings.duration || 3.4);
    const maxPopups = Math.max(1, Math.floor(Number(settings.maxPopups || 2) || 2));
    return {
      popup: {
        title,
        text,
        kind,
        parts: Array.isArray(payload.parts) ? payload.parts.slice(0, 4) : [],
        beforeTier: payload.beforeTier || '',
        afterTier: payload.afterTier || '',
        createdAt: now,
        expiresAt: now + duration
      },
      popups: (Array.isArray(existingPopups) ? existingPopups : [])
        .filter((item) => Number(item.expiresAt || 0) > now)
        .concat({
          title,
          text,
          kind,
          parts: Array.isArray(payload.parts) ? payload.parts.slice(0, 4) : [],
          beforeTier: payload.beforeTier || '',
          afterTier: payload.afterTier || '',
          createdAt: now,
          expiresAt: now + duration
        })
        .slice(-maxPopups)
    };
  }

  function createHudToastRewardUiHelpers() {
    return Object.freeze({
      getRewardPopupDisplayState,
      getCanvasToastDisplayEntries,
      getCanvasRewardToastDisplayEntries,
      getCanvasToastQueueUpdate,
      getCanvasRewardPopupQueueUpdate
    });
  }

  function getCanvasChannelMenuItems(channelSnapshot) {
    const source = channelSnapshot || {};
    const channels = Array.isArray(source.channels) && source.channels.length
      ? source.channels
      : Array.from({ length: 8 }, (_, index) => ({
        id: `ch${index + 1}`,
        label: `Ch. ${index + 1}`,
        name: `Channel ${index + 1}`,
        current: index === 0
      }));
    const currentId = source.currentId || (channels.find((channel) => channel.current) || channels[0] || {}).id;
    return channels.map((channel) => ({
      label: channel.label || channel.name || channel.id,
      action: 'changeChannel',
      channelId: channel.id,
      iconId: 'worldmap',
      selected: channel.current || channel.id === currentId
    }));
  }

  function getCanvasMenuGroups(snapshot, options) {
    const source = snapshot || {};
    const settings = options || {};
    const getChannelItems = typeof settings.getCanvasChannelMenuItems === 'function'
      ? settings.getCanvasChannelMenuItems
      : () => getCanvasChannelMenuItems(source.channel || {});
    return [
      {
        title: 'Character',
        items: [
          { label: 'Character', panel: 'character', iconId: 'character' },
          { label: 'Equipment', panel: 'equipment', iconId: 'equipment' },
          { label: 'Party', panel: 'partyPanel', iconId: 'partyPanel' },
          { label: 'Inventory', panel: 'inventory', iconId: 'inventory' },
          { label: 'Skills', panel: 'skills', iconId: 'skills' },
          { label: 'Quests', panel: 'quests', iconId: 'quests' }
        ]
      },
      {
        title: 'World',
        items: [
          { label: 'World Map', panel: 'worldmap', iconId: 'worldmap' },
          { label: 'Monster Guide', panel: 'monsters', iconId: 'monsters' },
          { label: 'Shop', panel: 'shop', iconId: 'shop' },
          { label: 'Upgrade', panel: 'upgrade', iconId: 'upgrade' },
          { label: 'Plinko', panel: 'plinko', iconId: 'plinko' },
          { label: source.dailyLogin && source.dailyLogin.claimable ? 'Daily Reward!' : 'Daily Rewards', panel: 'daily', iconId: 'daily' },
          { label: 'Cash Shop', panel: 'cashShop', iconId: 'cashShop' },
          { label: 'Beta Systems', panel: 'beta', iconId: 'beta' },
          { label: 'Guide', panel: 'guide', iconId: 'guide' },
          { label: 'Log', panel: 'log', iconId: 'log' }
        ]
      },
      {
        title: 'Channels',
        items: getChannelItems()
      },
      {
        title: 'Settings',
        items: [
          { label: 'Focus / Fullscreen', action: 'fullscreen', iconId: 'fullscreen' },
          { label: 'Settings', panel: 'settings', iconId: 'settings' },
          { label: 'Keybinds', panel: 'keybinds', iconId: 'keybinds' },
          { label: 'Admin Settings', panel: 'admin', iconId: 'admin' }
        ]
      }
    ];
  }

  function getCanvasMenuFooterAction() {
    return { label: 'Logout', action: 'load', iconId: 'logout', danger: true };
  }

  function getCanvasHudQuickActions(panelIds, options) {
    const settings = options || {};
    const sourceIds = Array.isArray(panelIds) ? panelIds : [];
    const getBindableAction = typeof settings.getBindableAction === 'function'
      ? settings.getBindableAction
      : () => null;
    return sourceIds
      .map((panelId) => {
        const action = getBindableAction(panelId);
        if (!action || !action.panel) return null;
        return {
          panel: action.panel,
          label: String(action.label || '').replace(/\s+Popup$/, ''),
          iconId: action.panel
        };
      })
      .filter(Boolean);
  }

  function getCanvasStatusHudLayout(box, quickActions, options) {
    const settings = options || {};
    const hudBox = box || { x: 0, y: 0, w: 0, h: 0 };
    const actions = Array.isArray(quickActions) ? quickActions : [];
    const pad = Number(settings.pad || 10);
    const menuW = Number(settings.menuWidth || 62);
    const menuH = Number(settings.menuHeight || 52);
    const quickGap = Number(settings.quickGap || 5);
    const quickSize = Number(settings.quickSize || 52);
    const halfGap = Number(settings.halfGap || 8);
    const menuX = hudBox.x + hudBox.w - pad - menuW;
    const menuY = hudBox.y + Math.max(10, Math.round((hudBox.h - menuH) / 2));
    const quickTotalW = actions.length * quickSize + Math.max(0, actions.length - 1) * quickGap;
    const quickX = menuX - (quickTotalW ? quickTotalW + 8 : 0);
    const contentLeft = hudBox.x + pad;
    const contentRight = quickTotalW ? quickX : menuX;
    const contentW = Math.max(220, contentRight - contentLeft - 8);
    const infoW = Math.min(214, Math.max(154, Math.floor(contentW * 0.25)));
    const meterX = contentLeft + infoW + 12;
    const meterW = Math.min(660, Math.max(160, contentRight - meterX - 8));
    const halfW = Math.max(58, Math.floor((meterW - halfGap) / 2));
    const meterTopY = hudBox.y + 14;
    const xpY = hudBox.y + 44;
    const chipY = hudBox.y + 30;
    const levelW = 58;
    return {
      pad,
      menuW,
      menuH,
      menuX,
      menuY,
      quickGap,
      quickSize,
      quickTotalW,
      quickX,
      contentLeft,
      contentRight,
      contentW,
      infoW,
      meterX,
      meterW,
      halfGap,
      halfW,
      meterTopY,
      xpY,
      frame: {
        infoRight: contentLeft + infoW + 5,
        commandLeft: quickTotalW ? quickX - 6 : menuX - 6
      },
      chipY,
      levelW,
      coinsX: contentLeft + levelW + 6,
      coinsW: Math.max(72, infoW - levelW - 6),
      resource: { x: contentLeft, y: hudBox.y + 56, w: infoW, h: 18 },
      meters: {
        hp: { x: meterX, y: meterTopY, w: halfW, h: 18 },
        mp: { x: meterX + halfW + halfGap, y: meterTopY, w: halfW, h: 18 },
        xp: { x: meterX, y: xpY, w: meterW, h: 16 }
      },
      quickButtons: actions.map((action, index) => ({
        action,
        x: quickX + index * (quickSize + quickGap),
        y: menuY,
        size: quickSize
      })),
      menu: { x: menuX, y: menuY, w: menuW, h: menuH }
    };
  }

  function createHudMenuUiHelpers() {
    return Object.freeze({
      getCanvasChannelMenuItems,
      getCanvasMenuGroups,
      getCanvasMenuFooterAction,
      getCanvasHudQuickActions,
      getCanvasStatusHudLayout,
      getCanvasMenuIconId,
      getCanvasMenuLayout
    });
  }

  function hasActiveCanvasOverlayGesture(state) {
    const source = state || {};
    return !!(
      source.canvasDrag ||
      source.minimapDrag ||
      source.questTrackerDrag ||
      source.combatMetricsDrag ||
      source.upgradePromptDrag ||
      source.potentialPromptDrag ||
      source.shardCraftPromptDrag ||
      source.questPromptDrag ||
      source.dropQuantityPromptDrag ||
      source.adminNumberPromptDrag ||
      source.confirmPromptDrag ||
      source.canvasBindDrag ||
      source.canvasInventoryDrag ||
      source.canvasGearDrag ||
      source.canvasSliderDrag ||
      source.plinkoDropHold && source.plinkoDropHold.active
    );
  }

  function getAdaptiveCanvasOverlayCacheMaxAge(baseMs, maxMs, frameMs, options) {
    const settings = options || {};
    const base = Math.max(0, Number(baseMs || 0) || 0);
    const max = Math.max(base, Number(maxMs || base) || base);
    const threshold = Number(settings.adaptiveFrameMs || 24);
    const multiplier = Number(settings.frameMultiplier || 3.25);
    const recentFrameMs = Number(frameMs || 0);
    if (!recentFrameMs || recentFrameMs < threshold) return base;
    const adaptiveMs = Math.ceil(recentFrameMs * multiplier);
    return Math.max(base, Math.min(max, adaptiveMs));
  }

  function getRecentCanvasOverlayFrameMs(debug) {
    if (!debug) return 0;
    const samples = Array.isArray(debug.samples) ? debug.samples : [];
    const frame = debug.currentFrame || samples[samples.length - 1] || null;
    const frameMs = Number(frame && (frame.frameMs || frame.deltaMs) || 0);
    return Number.isFinite(frameMs) && frameMs > 0 ? frameMs : 0;
  }

  function getCanvasOverlayCacheMaxAge(state, options) {
    const source = state || {};
    const settings = options || {};
    if (!source.engineRunning || !source.hasPlayerClass) return 0;
    if (source.hasActiveGesture) return 0;
    if (source.mapTransitionActive) return 0;
    const openWindows = Array.isArray(source.openWindows) ? source.openWindows : [];
    const hasWindowOverlay = !!(openWindows.length || source.commandOpen || source.overlayModalKey);
    const bypassPanelIds = source.bypassPanelIds;
    const hasBypassPanel = hasWindowOverlay && openWindows.some((panelId) => {
      if (!bypassPanelIds) return false;
      if (typeof bypassPanelIds.has === 'function') return bypassPanelIds.has(panelId);
      return Array.isArray(bypassPanelIds) && bypassPanelIds.includes(panelId);
    });
    if (hasBypassPanel) return 0;
    return getAdaptiveCanvasOverlayCacheMaxAge(
      hasWindowOverlay ? settings.windowBaseMs : settings.baseMs,
      hasWindowOverlay ? settings.windowAdaptiveMaxMs : settings.adaptiveMaxMs,
      source.recentFrameMs,
      settings
    );
  }

  function getCanvasOverlayCacheKey(width, height, snapshot, options) {
    const source = options || {};
    const state = snapshot && snapshot.state || {};
    const player = state.player || {};
    const openWindows = Array.isArray(source.openWindows) ? source.openWindows : [];
    const windowState = source.windowState || {};
    const windows = openWindows.map((panelId) => {
      const win = windowState && windowState[panelId];
      return win ? [
        panelId,
        Math.round(Number(win.x || 0)),
        Math.round(Number(win.y || 0)),
        Math.round(Number(win.w || 0)),
        Math.round(Number(win.h || 0)),
        Math.round(Number(win.scroll || 0)),
        Number(win.z || 0)
      ].join(':') : panelId;
    }).join('|');
    const minimap = source.minimapState || {};
    const questTracker = source.questTrackerState || {};
    const combatMetrics = source.combatMetricsPanelState || {};
    const hoverTarget = source.canvasHoverTarget || null;
    const itemContextMenu = source.itemContextMenu || null;
    return [
      Math.round(Number(width || 0)),
      Math.round(Number(height || 0)),
      Number(snapshot && snapshot.cacheRevision || 0),
      Number(source.canvasDrawRevision || 0),
      source.assetReadyKey == null ? '' : source.assetReadyKey,
      state.mapId || '',
      player.classId || '',
      player.advancedClassId || '',
      source.commandOpen ? 1 : 0,
      windows,
      source.overlayModalKey == null ? '' : source.overlayModalKey,
      hoverTarget && hoverTarget.key || '',
      itemContextMenu && itemContextMenu.active ? `${itemContextMenu.mode}:${itemContextMenu.x}:${itemContextMenu.y}:${itemContextMenu.id || ''}` : '',
      Array.isArray(source.canvasToasts) ? source.canvasToasts.length : Number(source.canvasToastCount || 0),
      Array.isArray(source.canvasRewardPopups) ? source.canvasRewardPopups.length : Number(source.canvasRewardPopupCount || 0),
      `${Math.round(Number(minimap.x || 0))}:${Math.round(Number(minimap.y || 0))}:${minimap.compact ? 1 : 0}`,
      `${Math.round(Number(questTracker.x || 0))}:${Math.round(Number(questTracker.y || 0))}:${questTracker.compact ? 1 : 0}`,
      `${Math.round(Number(combatMetrics.x || 0))}:${Math.round(Number(combatMetrics.y || 0))}`,
      source.inventorySellSettingsOpen ? 1 : 0,
      source.storageTab || '',
      source.petPotionPickerKind || ''
    ].join('::');
  }

  function getCanvasOverlayLayerCacheState(cache, w, h, key, now, maxAgeMs) {
    const source = cache || {};
    const stale = !source.canvas || source.w !== w || source.h !== h || source.key !== key || Number(now || 0) - Number(source.updatedAt || 0) > Number(maxAgeMs || 0);
    return {
      stale,
      reusableCanvas: source.canvas && source.w === w && source.h === h ? source.canvas : null
    };
  }

  function cloneCanvasRegions(regions) {
    return (regions || []).map((region) => Object.assign({}, region));
  }

  function getCanvasOverlayStatsPayload(state) {
    const source = state || {};
    const openWindows = Array.isArray(source.openWindows) ? source.openWindows : [];
    const canvasHitRegions = Array.isArray(source.canvasHitRegions) ? source.canvasHitRegions : [];
    const canvasToasts = Array.isArray(source.canvasToasts) ? source.canvasToasts : [];
    const canvasRewardPopups = Array.isArray(source.canvasRewardPopups) ? source.canvasRewardPopups : [];
    const hoverDebugStats = source.hoverDebugStats || {};
    const canvasDrawStats = source.canvasDrawStats || {};
    const overlayCacheStats = source.canvasOverlayCacheStats || {};
    const panelCacheStats = source.panelCacheStats || {};
    const windowDragStats = source.windowDragStats || {};
    const hoverTarget = source.canvasHoverTarget || null;
    return {
      openWindows: openWindows.length,
      openPanelIds: openWindows.join(',') || 'none',
      canvasHitRegions: canvasHitRegions.length,
      commandOpen: source.commandOpen,
      toasts: canvasToasts.length,
      rewardPopups: canvasRewardPopups.length,
      hoverPointerMoves: hoverDebugStats.pointerMoves,
      hoverTargetChanges: hoverDebugStats.targetChanges,
      hoverForcedDraws: hoverDebugStats.forcedDraws,
      hoverCoalescedSkips: hoverDebugStats.coalescedSkips,
      activeHoverType: hoverTarget && hoverTarget.type || '',
      activeHoverKey: hoverTarget && hoverTarget.key || '',
      canvasDrawRequests: canvasDrawStats.requests,
      canvasImmediateDraws: canvasDrawStats.immediate,
      canvasDeferredDraws: canvasDrawStats.deferred,
      canvasSkippedRunningDraws: canvasDrawStats.skippedWhileRunning,
      overlayCacheHits: overlayCacheStats.hits || 0,
      overlayCacheMisses: overlayCacheStats.misses || 0,
      panelCacheHits: panelCacheStats.hits || 0,
      panelCacheMisses: panelCacheStats.misses || 0,
      panelCachePrewarmQueued: panelCacheStats.prewarmQueued || 0,
      panelCachePrewarmCompleted: panelCacheStats.prewarmCompleted || 0,
      panelCachePrewarmSkipped: panelCacheStats.prewarmSkipped || 0,
      windowDragMoves: windowDragStats.moves,
      activeDragPanel: windowDragStats.activePanel || ''
    };
  }

  function createHudOverlayCacheUiHelpers() {
    return Object.freeze({
      hasActiveCanvasOverlayGesture,
      getAdaptiveCanvasOverlayCacheMaxAge,
      getRecentCanvasOverlayFrameMs,
      getCanvasOverlayCacheMaxAge,
      getCanvasOverlayCacheKey,
      getCanvasOverlayLayerCacheState,
      cloneCanvasRegions,
      getCanvasOverlayStatsPayload
    });
  }

  function getCanvasMenuIconId(item) {
    if (!item) return '';
    return item.iconId || item.panel || item.action || '';
  }

  function getCanvasMenuLayout(width, height, bottomY, groups, options) {
    const settings = options || {};
    const compactMenu = Number(width || 0) < 700;
    const w = compactMenu ? Math.min(430, Math.max(352, width - 32)) : 352;
    const columns = 2;
    const rowH = Number(settings.rowHeight || 24);
    const rowGap = Number(settings.rowGap || 4);
    const columnGap = Number(settings.columnGap || 6);
    const sourceGroups = Array.isArray(groups) ? groups : [];
    const footer = settings.footer || getCanvasMenuFooterAction();
    const cellW = Math.floor((w - 24 - columnGap) / columns);
    const groupRowCount = (group) => Math.max(1, Math.ceil((group.items || []).length / columns));
    const contentH = sourceGroups.reduce((sum, group) => sum + 18 + groupRowCount(group) * (rowH + rowGap) + 3, 14) + 12 + rowH + 12;
    const bottomLimit = Math.max(118, Number(bottomY || height - 18));
    const h = Math.min(height - 34, Math.max(90, bottomLimit - 24), contentH);
    const x = compactMenu ? Math.round((width - w) / 2) : width - w - 18;
    const y = Math.max(14, bottomLimit - h - 10);
    let cy = y + 12;
    const groupLayouts = sourceGroups.map((group) => {
      const titleY = cy;
      cy += 15;
      const rows = (group.items || []).map((item, index) => {
        const col = index % columns;
        const row = Math.floor(index / columns);
        return {
          item,
          x: x + 12 + col * (cellW + columnGap),
          y: cy + row * (rowH + rowGap),
          w: cellW,
          h: rowH
        };
      });
      cy += groupRowCount(group) * (rowH + rowGap);
      cy += 3;
      return {
        title: group.title,
        titleX: x + 12,
        titleY,
        rows
      };
    });
    const footerY = y + h - rowH - 12;
    const separatorY = footerY - 7;
    return {
      compactMenu,
      x,
      y,
      w,
      h,
      columns,
      rowH,
      rowGap,
      columnGap,
      cellW,
      contentH,
      bottomLimit,
      groups: groupLayouts,
      footer,
      footerBox: { x: x + 12, y: footerY, w: w - 24, h: rowH },
      separator: { x1: x + 12, x2: x + w - 12, y: separatorY + 0.5 }
    };
  }

  function getStationPromptContext(snapshot, options) {
    const source = snapshot || {};
    const settings = options || {};
    const keyLabels = settings.keyLabels || {};
    const player = source.state ? source.state.player : null;
    if (!player) return null;
    const stationId = player.activeStation;
    const portalId = player.activePortalId;
    const questNpcId = player.activeQuestNpcId;
    if (!stationId && !portalId && !questNpcId) return null;
    const station = stationId ? ((source.map && source.map.stations) || []).find((candidate) => candidate.id === stationId) : null;
    const portal = portalId ? (source.portals || []).find((candidate) => candidate.id === portalId) : null;
    const questNpc = questNpcId && source.questNpcs
      ? (source.questNpcs.npcs || []).find((candidate) => candidate.id === questNpcId && ((candidate.iconStates || []).length || candidate.servicePanelId))
      : null;
    const openWindows = settings.openWindows || [];
    if (openWindows.includes('plinko') && ((stationId === 'plinko') || (questNpc && questNpc.servicePanelId === 'plinko'))) return null;
    const title = questNpc ? questNpc.name : station ? station.name : portal ? portal.label : '';
    if (!title) return null;
    return {
      title,
      promptAction: portal ? 'portal' : questNpc ? 'npcTalk' : 'interact',
      hint: portal
        ? `${keyLabels.moveUp || 'Up'} Travel`
        : questNpc
          ? `${keyLabels.npcTalk || 'Y'} Talk`
          : `${keyLabels.interact || 'F'} Use`,
      hintColor: portal ? '#2f7dd6' : '#177645',
      kindLabel: questNpc && questNpc.servicePanelId ? 'Service NPC' : questNpc ? 'Quest NPC' : portal ? 'Portal' : 'Station',
      target: questNpc || station || portal || null,
      station,
      portal,
      questNpc
    };
  }

  function getStationPromptLayout(width, height, bottomY, target, options) {
    const settings = options || {};
    const camera = settings.camera || { x: 0, y: 0, zoom: 1 };
    const boxW = Math.min(320, Math.max(220, width - 32));
    const boxH = 42;
    const zoom = Math.max(1, Number(camera.zoom || 1));
    let x = Math.round((width - boxW) / 2);
    let y = Math.max(18, Math.min(height - boxH - 18, Number(bottomY || height) - boxH - 14));
    if (target) {
      const centerX = (Number(target.x || 0) + Number(target.w || 0) / 2 - Number(camera.x || 0)) * zoom;
      const topY = (Number(target.y || 0) - Number(camera.y || 0)) * zoom;
      const bottomTargetY = (Number(target.y || 0) + Number(target.h || 0) - Number(camera.y || 0)) * zoom;
      x = Math.round(clamp(centerX - boxW / 2, 12, width - boxW - 12));
      y = topY - boxH - 12 >= 12
        ? Math.round(topY - boxH - 12)
        : Math.round(clamp(bottomTargetY + 10, 18, height - boxH - 18));
    }
    return { x, y, w: boxW, h: boxH };
  }

  function getStationPromptRenderMetadata(prompt, box, options) {
    if (!prompt || !box) return null;
    const settings = options || {};
    return {
      region: {
        type: 'station-prompt',
        action: prompt.promptAction,
        x: box.x,
        y: box.y,
        w: box.w,
        h: box.h
      },
      panel: {
        x: box.x,
        y: box.y,
        w: box.w,
        h: box.h,
        options: {
          fill: settings.fill || 'rgba(255,244,216,0.88)',
          stroke: settings.stroke || 'rgba(245,207,114,0.42)',
          radius: 8
        }
      },
      titleText: {
        value: prompt.title,
        x: box.x + 12,
        y: box.y + 7,
        color: '#102033',
        font: '900 12px system-ui',
        maxWidth: box.w - 112,
        lineHeight: 13,
        maxLines: 1
      },
      hintText: {
        value: prompt.hint,
        x: box.x + box.w - 12,
        y: box.y + 8,
        color: prompt.hintColor,
        font: '900 11px system-ui',
        align: 'right',
        maxWidth: 98,
        lineHeight: 12
      },
      kindText: {
        value: prompt.kindLabel,
        x: box.x + 12,
        y: box.y + 25,
        color: '#5f6f7a',
        font: '10px system-ui',
        maxWidth: box.w - 24,
        lineHeight: 11
      }
    };
  }

  function getQuestNpcIconEntries(snapshot, width, options) {
    const source = snapshot || {};
    const settings = options || {};
    const questNpcs = source.questNpcs && Array.isArray(source.questNpcs.npcs) ? source.questNpcs.npcs : [];
    const camera = source.camera || { x: 0, y: 0 };
    const zoom = Math.max(1, Number(camera.zoom || 1));
    const size = Number(settings.size || 28);
    const gap = Number(settings.gap || 7);
    const topOffset = Number(settings.topOffset || 47);
    const entries = [];
    questNpcs.forEach((npc) => {
      const iconStates = Array.isArray(npc.iconStates) ? npc.iconStates : [];
      if (!iconStates.length) return;
      const totalW = iconStates.length * size + Math.max(0, iconStates.length - 1) * gap;
      const baseX = Math.round((npc.x + npc.w / 2 - Number(camera.x || 0)) * zoom - totalW / 2);
      const y = Math.round((npc.y - Number(camera.y || 0)) * zoom - topOffset);
      iconStates.forEach((icon, index) => {
        const x = baseX + index * (size + gap);
        if (x + size < 0 || x > width || y + size < 0) return;
        const reward = icon.action === 'claim';
        const talk = icon.action === 'talk';
        const active = icon.action === 'active';
        entries.push({
          npcId: npc.id,
          questId: icon.questId,
          x,
          y,
          w: size,
          h: size,
          fill: reward ? '#eff9ef' : talk ? '#eef7ff' : active ? '#edf3ff' : '#fff7d8',
          stroke: reward ? 'rgba(23,118,69,0.72)' : talk ? 'rgba(47,125,214,0.72)' : active ? 'rgba(47,95,176,0.74)' : 'rgba(216,183,74,0.78)',
          radius: 9,
          textColor: reward ? '#177645' : talk ? '#2f7dd6' : active ? '#2f5fb0' : '#8a6b00',
          font: talk || active ? '950 15px system-ui' : '950 21px system-ui',
          iconText: icon.icon || (reward ? '?' : talk ? '...' : active ? '*' : '!'),
          regionType: reward ? 'quest-npc-reward-icon' : talk ? 'quest-npc-talk-icon' : active ? 'quest-npc-active-icon' : 'quest-npc-accept-icon',
          action: reward ? 'claim' : talk ? 'talk' : active ? 'active' : 'accept'
        });
      });
    });
    return entries;
  }

  function getQuestNpcIconRenderMetadata(entry) {
    if (!entry) return null;
    return {
      slot: {
        x: entry.x,
        y: entry.y,
        w: entry.w,
        h: entry.h,
        options: {
          fill: entry.fill,
          stroke: entry.stroke,
          radius: entry.radius
        }
      },
      text: {
        value: entry.iconText,
        x: entry.x + entry.w / 2,
        y: entry.y + entry.h / 2 + 1,
        fillStyle: entry.textColor,
        font: entry.font,
        textAlign: 'center',
        textBaseline: 'middle'
      },
      region: {
        type: entry.regionType,
        action: entry.action,
        npcId: entry.npcId,
        questId: entry.questId,
        x: entry.x,
        y: entry.y,
        w: entry.w,
        h: entry.h
      }
    };
  }

  function createHudWorldPromptUiHelpers() {
    return Object.freeze({
      getStationPromptContext,
      getStationPromptLayout,
      getStationPromptRenderMetadata,
      getQuestNpcIconEntries,
      getQuestNpcIconRenderMetadata
    });
  }

  function getQuestTrackerEntries(snapshot, options) {
    const source = snapshot || {};
    const settings = options || {};
    const formatDungeonRespawnLabel = typeof settings.formatDungeonRespawnLabel === 'function'
      ? settings.formatDungeonRespawnLabel
      : function formatDungeonRespawnLabelFallback() { return ''; };
    const progress = source.progress;
    const activeDungeon = source.dungeon && source.dungeon.activeDungeon;
    const mapKillQuest = source.mapKillQuest;
    const onboarding = source.onboarding || {};
    const nextStep = onboarding.hidden ? null : onboarding.nextStep;
    const claimableQuests = progress && Array.isArray(progress.claimableQuests) ? progress.claimableQuests.slice(0, 2) : [];
    const showMapKillQuest = mapKillQuest && (mapKillQuest.active || mapKillQuest.claimable);
    const guideEntry = nextStep ? {
      title: `Guide ${Number(onboarding.completeCount || 0) + 1}/${Number(onboarding.total || 0)}: ${nextStep.title}`,
      guideType: 'guide',
      guideId: nextStep.id || nextStep.panelId || nextStep.title,
      objectives: [{ label: nextStep.summary || 'Continue the guide.', value: 0, goal: 1, complete: false, status: '' }]
    } : null;
    if (!guideEntry && (!progress || (!progress.activeQuest && !progress.activeTrial && !activeDungeon && !showMapKillQuest && !claimableQuests.length))) return [];
    const dungeonRespawnLabel = formatDungeonRespawnLabel(activeDungeon);
    const dungeonEntry = activeDungeon ? {
      title: activeDungeon.name,
      guideType: 'dungeon',
      guideId: activeDungeon.id,
      objectives: [Object.assign({
        label: dungeonRespawnLabel || `Defeat ${activeDungeon.bossName}`,
        value: 0,
        goal: 1,
        complete: false
      }, dungeonRespawnLabel ? { status: '' } : {})]
    } : null;
    const mapKillEntry = showMapKillQuest ? {
      title: `${mapKillQuest.title} · ${mapKillQuest.completions} done`,
      guideType: 'mapKill',
      guideId: mapKillQuest.mapId,
      objectives: [{ label: mapKillQuest.claimable ? `Claim reward from ${mapKillQuest.npcName || 'map NPC'}` : mapKillQuest.label, value: mapKillQuest.value, goal: mapKillQuest.goal, complete: !!mapKillQuest.claimable }]
    } : null;
    const claimEntries = claimableQuests.map((quest) => ({
      title: `${quest.title} complete`,
      guideType: 'quest',
      guideId: quest.id,
      objectives: [{ label: `Claim from ${quest.npcName || 'quest NPC'}: ${quest.rewardSummary}`, value: 1, goal: 1, complete: true }]
    }));
    const activeQuest = progress.activeQuest ? Object.assign({}, progress.activeQuest, {
      guideType: 'quest',
      guideId: progress.activeQuest.id
    }) : null;
    const activeTrial = progress.activeTrial ? Object.assign({}, progress.activeTrial, {
      guideType: 'trial',
      guideId: progress.activeTrial.id
    }) : null;
    return [guideEntry, ...claimEntries, mapKillEntry, activeQuest, activeTrial, dungeonEntry].filter(Boolean);
  }

  function getQuestTrackerNaturalHeight(entries, guidance) {
    const rowH = 16;
    const guide = guidance || {};
    const introH = guide.active ? 0 : 18;
    return 30 + introH + (entries || []).reduce((sum, entry) => {
      const selected = guide.active && guide.targetType === entry.guideType && guide.targetId === entry.guideId;
      return sum + 10 + 19 + Math.min(2, (entry.objectives || []).length) * rowH + (selected ? 34 : 0);
    }, 0);
  }

  function getQuestTrackerBox(snapshot, width, height, bottomY, state, options) {
    const source = snapshot || {};
    const trackerState = state || {};
    const settings = options || {};
    const entries = Array.isArray(settings.entries) ? settings.entries : getQuestTrackerEntries(source, settings);
    const compact = !!(trackerState && trackerState.compact);
    const boxW = compact ? 154 : Math.min(312, Math.max(238, width * 0.31));
    const guidance = settings.guidance || source.questGuidance;
    const naturalH = compact ? 44 : getQuestTrackerNaturalHeight(entries, guidance);
    const uiBottom = typeof settings.getCanvasUiBottom === 'function'
      ? settings.getCanvasUiBottom(width, height, 8)
      : height;
    const maxBottom = Number(bottomY || uiBottom);
    const boxH = compact ? 44 : Math.min(naturalH, Math.max(82, maxBottom - 16 - 8));
    const defaultX = 16;
    const defaultY = 16;
    const rawX = trackerState.userPlaced ? Number(trackerState.x || defaultX) : defaultX;
    const rawY = trackerState.userPlaced ? Number(trackerState.y || defaultY) : defaultY;
    return {
      x: clamp(rawX, 8, Math.max(8, width - boxW - 8)),
      y: clamp(rawY, 8, Math.max(8, maxBottom - boxH)),
      w: boxW,
      h: boxH,
      compact,
      entries
    };
  }

  function getStatusTileProgress(entry) {
    const status = entry || {};
    if (Number.isFinite(Number(status.progress))) return clamp(Number(status.progress), 0, 1);
    const remaining = Number(status.remaining || 0);
    const duration = Number(status.duration || status.baseCooldown || 0);
    return duration > 0 ? clamp(remaining / duration, 0, 1) : remaining > 0 ? 1 : 0;
  }

  function formatStatusTileTimer(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    if (value >= 60) return `${Math.ceil(value / 60)}m`;
    if (value >= 10) return `${Math.ceil(value)}s`;
    return `${value.toFixed(1).replace(/\.0$/, '')}s`;
  }

  function getCanvasSkillCooldownOverlayMetadata(x, y, size, cooldown) {
    const progress = getStatusTileProgress(cooldown);
    const textSize = Math.max(8, Math.min(12, Math.floor(size * 0.28)));
    const lineHeight = Math.max(9, Math.min(13, Math.floor(size * 0.3)));
    return {
      progress,
      clip: { x, y, w: size, h: size },
      shade: { x, y, w: size, h: size, fill: 'rgba(5,12,22,0.38)' },
      progressShade: { x, y, w: size, h: Math.round(size * progress), fill: 'rgba(3,8,15,0.62)' },
      text: size >= 24 ? {
        value: formatStatusTileTimer(cooldown && cooldown.remaining),
        x: x + size / 2,
        y: y + size / 2 - 5,
        color: '#e3f8ff',
        font: `900 ${textSize}px system-ui`,
        align: 'center',
        baseline: 'middle',
        maxWidth: size - 4,
        maxLines: 1,
        lineHeight
      } : null
    };
  }

  function drawCanvasSkillCooldownOverlay(ctx, x, y, size, cooldown, options) {
    const settings = options || {};
    const drawCanvasText = settings.drawCanvasText;
    const overlay = getCanvasSkillCooldownOverlayMetadata(x, y, size, cooldown);
    if (
      !ctx ||
      !overlay ||
      typeof ctx.save !== 'function' ||
      typeof ctx.beginPath !== 'function' ||
      typeof ctx.rect !== 'function' ||
      typeof ctx.clip !== 'function' ||
      typeof ctx.restore !== 'function'
    ) return false;
    if (overlay.text && typeof drawCanvasText !== 'function') return false;
    ctx.save();
    ctx.beginPath();
    ctx.rect(overlay.clip.x, overlay.clip.y, overlay.clip.w, overlay.clip.h);
    ctx.clip();
    drawCanvasFillRectEntry(ctx, overlay.shade);
    drawCanvasFillRectEntry(ctx, overlay.progressShade);
    ctx.restore();
    drawCanvasTextEntry(drawCanvasText, overlay.text);
    return true;
  }

  function getCanvasStatusIconTileMetadata(entry, x, y, size, options) {
    const status = entry || {};
    const settings = options || {};
    const kind = settings.kind || 'buff';
    const skill = settings.skill || null;
    const progress = getStatusTileProgress(status);
    const timer = formatStatusTileTimer(status.remaining);
    const cooldown = kind === 'cooldown';
    const formatCooldown = typeof settings.formatCooldownLabel === 'function'
      ? settings.formatCooldownLabel
      : formatStatusTileTimer;
    const barW = Math.max(0, Math.round((size - 8) * progress));
    return {
      status,
      kind,
      skill,
      progress,
      fill: cooldown ? 'rgba(13,22,35,0.92)' : 'rgba(16,32,51,0.88)',
      stroke: cooldown ? 'rgba(123,223,242,0.48)' : 'rgba(255,209,102,0.5)',
      badgeText: String(status.label || status.id || '').slice(0, 3).toUpperCase(),
      badgeFill: cooldown ? '#e3f8ff' : '#fff3c4',
      bar: cooldown ? null : {
        trackX: x + 4,
        trackY: y + size - 7,
        trackW: size - 8,
        trackH: 4,
        trackRadius: 2,
        fillX: x + 5,
        fillY: y + size - 6,
        fillW: Math.max(0, barW - 2),
        fillH: 2,
        fill: settings.barColor || '#ffd166'
      },
      timer,
      timerBox: {
        x: x + 4,
        y: y + 4,
        w: Math.min(size - 8, Math.max(20, timer.length * 7)),
        h: 13,
        radius: 4,
        fill: cooldown ? 'rgba(4,12,22,0.74)' : 'rgba(16,32,51,0.68)',
        stroke: 'rgba(255,255,255,0.12)'
      },
      timerText: {
        x: x + 7,
        y: y + 6,
        color: cooldown ? '#c9f6ff' : '#fff7df'
      },
      region: {
        type: cooldown ? 'hud-cooldown' : 'hud-buff',
        tooltipKey: `${cooldown ? 'hud-cooldown' : 'hud-buff'}:${status.skillId || status.id || status.label}`,
        tooltipTitle: status.label || status.id || (cooldown ? 'Skill Cooldown' : 'Active Buff'),
        tooltipSubtitle: cooldown ? `Ready in ${formatCooldown(status.remaining)}` : `${formatCooldown(status.remaining)} remaining`,
        tooltipLines: [
          skill ? `Source: ${skill.name}` : '',
          cooldown && status.baseCooldown ? `Base cooldown: ${formatCooldown(status.baseCooldown)}` : '',
          !cooldown && status.duration ? `Duration: ${formatCooldown(status.duration)}` : '',
          !cooldown && status.positional ? 'Active while standing in its aura.' : '',
          skill && skill.description ? skill.description : ''
        ].filter(Boolean),
        tooltipAccent: cooldown ? '#2f7dd6' : '#d8a531',
        x,
        y,
        w: size,
        h: size
      }
    };
  }

  function drawCanvasStatusIconTile(ctx, entry, x, y, size, options) {
    const settings = options || {};
    const kind = settings.kind || 'buff';
    const skill = settings.skill || null;
    const drawRoundRect = settings.drawRoundRect;
    const drawSkillIcon = settings.drawSkillIcon;
    const drawIconBadge = settings.drawIconBadge;
    const drawCanvasText = settings.drawCanvasText;
    const addCanvasRegion = settings.addCanvasRegion;
    const tile = getCanvasStatusIconTileMetadata(entry, x, y, size, {
      kind,
      skill,
      barColor: settings.barColor,
      formatCooldownLabel: settings.formatCooldownLabel
    });
    if (
      !ctx ||
      !tile ||
      typeof drawRoundRect !== 'function' ||
      typeof drawCanvasText !== 'function' ||
      typeof addCanvasRegion !== 'function' ||
      (skill && typeof drawSkillIcon !== 'function') ||
      (!skill && typeof drawIconBadge !== 'function')
    ) return false;
    drawRoundRect({
      x,
      y,
      w: size,
      h: size,
      radius: 8,
      fill: tile.fill,
      stroke: tile.stroke
    });
    if (skill) {
      drawSkillIcon({ skill, x: x + 3, y: y + 3, size: size - 6 });
    } else {
      drawIconBadge({
        text: tile.badgeText,
        x: x + 4,
        y: y + 4,
        size: size - 8,
        fill: tile.badgeFill
      });
    }
    if (kind === 'cooldown') {
      drawCanvasSkillCooldownOverlay(ctx, x + 3, y + 3, size - 6, tile.status, {
        drawCanvasText
      });
    } else if (tile.bar) {
      drawRoundRect({
        x: tile.bar.trackX,
        y: tile.bar.trackY,
        w: tile.bar.trackW,
        h: tile.bar.trackH,
        radius: tile.bar.trackRadius,
        fill: 'rgba(9,17,27,0.7)',
        stroke: 'rgba(255,255,255,0.12)'
      });
      drawCanvasFillRectEntry(ctx, {
        x: tile.bar.fillX,
        y: tile.bar.fillY,
        w: tile.bar.fillW,
        h: tile.bar.fillH,
        fill: tile.bar.fill
      });
    }
    drawRoundRect({
      x: tile.timerBox.x,
      y: tile.timerBox.y,
      w: tile.timerBox.w,
      h: tile.timerBox.h,
      radius: tile.timerBox.radius,
      fill: tile.timerBox.fill,
      stroke: tile.timerBox.stroke
    });
    drawCanvasText({
      value: tile.timer,
      x: tile.timerText.x,
      y: tile.timerText.y,
      color: tile.timerText.color,
      font: '900 8px system-ui',
      maxWidth: size - 10,
      maxLines: 1,
      lineHeight: 9
    });
    addCanvasRegion(tile.region);
    return true;
  }

  function getCanvasStatusIconGridLayout(entries, x, bottomY, options) {
    const settings = options || {};
    const statuses = (entries || []).slice(0, Number(settings.limit || 8));
    if (!statuses.length) return { statuses, tiles: [], header: null, nextBottom: bottomY, groupH: 0 };
    const tile = Number(settings.tileSize || 38);
    const gap = Number(settings.gap || 6);
    const label = settings.label || '';
    const boxW = Number(settings.width || 204);
    const headerH = label ? 18 : 0;
    const columns = Math.max(1, Math.floor((boxW + gap) / (tile + gap)));
    const rows = Math.ceil(statuses.length / columns);
    const gridW = columns * tile + Math.max(0, columns - 1) * gap;
    const groupH = headerH + (headerH ? 4 : 0) + rows * tile + Math.max(0, rows - 1) * gap;
    let tileY = bottomY - groupH;
    const header = label ? {
      label,
      x: x + 4,
      y: tileY,
      maxWidth: boxW - 8,
      height: headerH
    } : null;
    if (header) tileY += headerH + 4;
    const gridX = x + Math.max(0, Math.floor((boxW - gridW) / 2));
    return {
      statuses,
      tiles: statuses.map((status, index) => {
        const col = index % columns;
        const row = Math.floor(index / columns);
        return {
          status,
          x: gridX + col * (tile + gap),
          y: tileY + row * (tile + gap),
          size: tile
        };
      }),
      header,
      tile,
      gap,
      boxW,
      columns,
      rows,
      gridW,
      groupH,
      nextBottom: bottomY - groupH
    };
  }

  function getCanvasStatusIconPanelGridLayout(entries, x, y, w, options) {
    const settings = options || {};
    const statuses = (entries || []).slice(0, Number(settings.limit || 6));
    if (!statuses.length) return { statuses, tiles: [], height: 0 };
    const tile = 38;
    const gap = 8;
    const labelH = 13;
    const cellW = 58;
    const columns = Math.max(1, Math.min(statuses.length, Math.floor((w + gap) / (cellW + gap))));
    const rows = Math.ceil(statuses.length / columns);
    const cellH = tile + labelH + 5;
    return {
      statuses,
      tiles: statuses.map((status, index) => {
        const col = index % columns;
        const row = Math.floor(index / columns);
        const cx = x + col * (cellW + gap);
        const cy = y + row * cellH;
        return {
          status,
          tileX: cx + Math.floor((cellW - tile) / 2),
          tileY: cy,
          labelX: cx + cellW / 2,
          labelY: cy + tile + 4,
          cellW,
          size: tile
        };
      }),
      tile,
      gap,
      labelH,
      cellW,
      columns,
      rows,
      cellH,
      height: rows * cellH
    };
  }

  function drawCanvasStatusIconGrid(ctx, entries, x, bottomY, options) {
    const settings = options || {};
    const drawCanvasText = settings.drawCanvasText;
    const drawStatusIconTile = settings.drawStatusIconTile;
    const layout = getCanvasStatusIconGridLayout(entries, x, bottomY, settings);
    if (!layout.tiles.length) return layout.nextBottom;
    if (layout.header) {
      if (typeof drawCanvasText !== 'function') return layout.nextBottom;
      drawCanvasText({
        value: layout.header.label,
        x: layout.header.x,
        y: layout.header.y,
        color: settings.headerColor || '#fff7df',
        font: '950 11px system-ui',
        maxWidth: layout.header.maxWidth,
        lineHeight: 12
      });
    }
    if (typeof drawStatusIconTile !== 'function') return layout.nextBottom;
    layout.tiles.forEach((tile) => {
      drawStatusIconTile({
        status: tile.status,
        x: tile.x,
        y: tile.y,
        size: tile.size,
        options: settings
      });
    });
    return layout.nextBottom;
  }

  function drawCanvasStatusIconPanelGrid(ctx, entries, x, y, w, options) {
    const settings = options || {};
    const drawCanvasText = settings.drawCanvasText;
    const drawStatusIconTile = settings.drawStatusIconTile;
    const layout = getCanvasStatusIconPanelGridLayout(entries, x, y, w, settings);
    if (!layout.tiles.length) return 0;
    if (typeof drawStatusIconTile !== 'function' || typeof drawCanvasText !== 'function') return layout.height;
    layout.tiles.forEach((tile) => {
      const status = tile.status || {};
      drawStatusIconTile({
        status,
        x: tile.tileX,
        y: tile.tileY,
        size: tile.size,
        options: settings
      });
      drawCanvasText({
        value: status.label || status.id || '',
        x: tile.labelX,
        y: tile.labelY,
        color: '#5f6f7a',
        font: '800 8px system-ui',
        align: 'center',
        maxWidth: tile.cellW,
        lineHeight: 9,
        maxLines: 1
      });
    });
    return layout.height;
  }

  function createHudStatusIconUiHelpers() {
    return Object.freeze({
      getStatusTileProgress,
      formatStatusTileTimer,
      getCanvasSkillCooldownOverlayMetadata,
      drawCanvasSkillCooldownOverlay,
      getCanvasStatusIconTileMetadata,
      drawCanvasStatusIconTile,
      getCanvasStatusIconGridLayout,
      drawCanvasStatusIconGrid,
      getCanvasStatusIconPanelGridLayout,
      drawCanvasStatusIconPanelGrid
    });
  }

  const api = {
    HUD_METER_ANIMATION_MS,
    CANVAS_OVERLAY_CACHE_MS,
    CANVAS_OVERLAY_CACHE_WITH_WINDOWS_MS,
    CANVAS_OVERLAY_CACHE_ADAPTIVE_MAX_MS,
    CANVAS_OVERLAY_CACHE_WITH_WINDOWS_ADAPTIVE_MAX_MS,
    CANVAS_OVERLAY_CACHE_ADAPTIVE_FRAME_MS,
    CANVAS_OVERLAY_CACHE_FRAME_MULTIPLIER,
    REWARD_TOAST_PLACEMENT,
    REWARD_TOAST_HOLD_SECONDS,
    REWARD_TOAST_FADE_SECONDS,
    REWARD_TOAST_DURATION_SECONDS,
    REWARD_TOAST_SLIDE_UP_PX,
    REWARD_TOAST_BOTTOM_PANEL_GAP,
    REWARD_TOAST_BACKGROUND,
    REWARD_TOAST_BORDER,
    shouldRefreshUiChangeHudSnapshot,
    mergeHudSnapshot,
    getDismissGuideDomAction,
    getHudRegionAction,
    getWorldDerivedSnapshotUpdate,
    createHudRuntimeUiHelpers,
    getCanvasStatusHudBox,
    getCanvasHudTop,
    getCanvasUiBottom,
    getCanvasHudTheme,
    getCanvasHudDividerMetadata,
    getCanvasStatusHudFrameMetadata,
    getCanvasHudChipMetadata,
    getCanvasHudSocketMetadata,
    drawCanvasHudDivider,
    drawCanvasStatusHudFrame,
    drawCanvasHudChip,
    drawCanvasHudSocket,
    getCanvasHudMeterMetadata,
    drawCanvasHudMeter,
    createHudChromeUiHelpers,
    getCanvasBossEncounterHudMetadata,
    getCanvasBossEncounterOverlayMetadata,
    drawCanvasBossEncounterHud,
    drawCanvasBossEncounterOverlays,
    createHudBossEncounterUiHelpers,
    getCombatMetricsPanelBox,
    getMinimapBox,
    getHudWidgetPointerAction,
    getHudWidgetMoveAction,
    getHudWidgetDragUpdate,
    getHudWidgetReleaseAction,
    getHudWidgetCancelAction,
    createHudWidgetUiHelpers,
    getRewardPopupDisplayState,
    getCanvasToastDisplayEntries,
    getCanvasRewardToastDisplayEntries,
    getCanvasToastQueueUpdate,
    getCanvasRewardPopupQueueUpdate,
    createHudToastRewardUiHelpers,
    getCanvasChannelMenuItems,
    getCanvasMenuGroups,
    getCanvasMenuFooterAction,
    getCanvasHudQuickActions,
    getCanvasStatusHudLayout,
    createHudMenuUiHelpers,
    hasActiveCanvasOverlayGesture,
    getAdaptiveCanvasOverlayCacheMaxAge,
    getRecentCanvasOverlayFrameMs,
    getCanvasOverlayCacheMaxAge,
    getCanvasOverlayCacheKey,
    getCanvasOverlayLayerCacheState,
    cloneCanvasRegions,
    getCanvasOverlayStatsPayload,
    createHudOverlayCacheUiHelpers,
    getCanvasMenuIconId,
    getCanvasMenuLayout,
    getStationPromptContext,
    getStationPromptLayout,
    getStationPromptRenderMetadata,
    getQuestNpcIconEntries,
    getQuestNpcIconRenderMetadata,
    createHudWorldPromptUiHelpers,
    getQuestTrackerEntries,
    getQuestTrackerNaturalHeight,
    getQuestTrackerBox,
    getStatusTileProgress,
    formatStatusTileTimer,
    getCanvasSkillCooldownOverlayMetadata,
    drawCanvasSkillCooldownOverlay,
    getCanvasStatusIconTileMetadata,
    drawCanvasStatusIconTile,
    getCanvasStatusIconGridLayout,
    drawCanvasStatusIconGrid,
    getCanvasStatusIconPanelGridLayout,
    drawCanvasStatusIconPanelGrid,
    createHudStatusIconUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.hud = Object.assign({}, modules.hud || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
