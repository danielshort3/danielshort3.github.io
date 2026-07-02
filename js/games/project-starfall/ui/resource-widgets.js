(function initProjectStarfallUiResourceWidgets(global) {
  'use strict';

  const SKILL_PATH_HINTS = Object.freeze({
    fighter: Object.freeze([
      Object.freeze({ label: 'Leveling', summary: 'Heavy Strike -> Ground Slam -> Momentum Burst' }),
      Object.freeze({ label: 'Bossing', summary: 'Heavy Strike -> Power Break -> Guard' })
    ]),
    mage: Object.freeze([
      Object.freeze({ label: 'Leveling', summary: 'Magic Bolt -> Arcane Burst -> Energy Release' }),
      Object.freeze({ label: 'Control', summary: 'Spell Mark -> Mana Shield -> Energy Release' })
    ]),
    archer: Object.freeze([
      Object.freeze({ label: 'Leveling', summary: 'Quick Shot -> Piercing Arrow -> Focused Volley' }),
      Object.freeze({ label: 'Bossing', summary: 'Marked Shot -> Eagle Stance -> Focused Volley' })
    ]),
    guardian: Object.freeze([
      Object.freeze({ label: 'Survival', summary: 'Shield Bash -> Impact Guard -> Shield Wall' }),
      Object.freeze({ label: 'Bossing', summary: 'Impact Guard -> Retaliation Wave -> Verdict' })
    ]),
    berserker: Object.freeze([
      Object.freeze({ label: 'Leveling', summary: 'Blood Cleave -> Rage Surge -> Crimson Recovery' }),
      Object.freeze({ label: 'Burst', summary: 'Pain to Power -> Last Stand -> War Cry' })
    ]),
    duelist: Object.freeze([
      Object.freeze({ label: 'Tempo', summary: 'Quick Cut -> Flash Step -> Rallying Flourish' }),
      Object.freeze({ label: 'Mobility', summary: 'Flash Step keeps uptime while training' })
    ]),
    fireMage: Object.freeze([
      Object.freeze({ label: 'Mobbing', summary: 'Fireball -> Burning Mark -> Wildfire' }),
      Object.freeze({ label: 'Burst', summary: 'Heat Vent -> Inferno Burst at high Heat' })
    ]),
    runeMage: Object.freeze([
      Object.freeze({ label: 'Setup', summary: 'Rune Mark -> Ground Glyph -> Detonation' }),
      Object.freeze({ label: 'Support', summary: 'Mana Seal -> Rune Circle for safer fights' })
    ]),
    stormMage: Object.freeze([
      Object.freeze({ label: 'Mobbing', summary: 'Chain Bolt clears clusters; watch Charge' }),
      Object.freeze({ label: 'Mobility', summary: 'Static Shift keeps spell flow moving' })
    ]),
    sniper: Object.freeze([
      Object.freeze({ label: 'Bossing', summary: 'Aimed Shot -> Weak Point -> Execution Shot' }),
      Object.freeze({ label: 'Precision', summary: 'Steady Breath improves range and precision payoff' })
    ]),
    trapper: Object.freeze([
      Object.freeze({ label: 'Control', summary: 'Snare Trap -> Spike Trap -> Detonate' }),
      Object.freeze({ label: 'Mobbing', summary: 'Tripwire -> Kill Zone for prepared routes' })
    ]),
    beastArcher: Object.freeze([
      Object.freeze({ label: 'Hybrid', summary: 'Companion Strike -> Pounce Roll -> Pack Call' }),
      Object.freeze({ label: 'Sustain', summary: 'Bond generation keeps longer fights stable' })
    ])
  });

  const RESOURCE_WIDGET_META = Object.freeze({
    fighter: Object.freeze({ type: 'bar', label: 'Momentum', detail: 'Impact resource', segments: 5 }),
    mage: Object.freeze({ type: 'orb', label: 'Energy', detail: 'Spell resource', segments: 5 }),
    archer: Object.freeze({ type: 'pips', label: 'Focus', detail: 'Mark payoff', segments: 5 }),
    guardian: Object.freeze({ type: 'segments', label: 'Stored Impact', detail: 'Shield segments', segments: 5 }),
    berserker: Object.freeze({ type: 'danger', label: 'Rage', detail: 'Low HP increases payoff', segments: 5 }),
    duelist: Object.freeze({ type: 'pips', label: 'Tempo', detail: 'Rhythm windows', segments: 4 }),
    fireMage: Object.freeze({ type: 'heat', label: 'Heat', detail: 'Vent before overheat', segments: 5 }),
    runeMage: Object.freeze({ type: 'runes', label: 'Runes', detail: 'Active setup count', segments: 3 }),
    stormMage: Object.freeze({ type: 'charge', label: 'Charge', detail: 'Chain readiness', segments: 5 }),
    sniper: Object.freeze({ type: 'aim', label: 'Aim', detail: 'Weak-point pressure', segments: 3 }),
    trapper: Object.freeze({ type: 'traps', label: 'Traps', detail: 'Armed triggers', segments: 4 }),
    beastArcher: Object.freeze({ type: 'bond', label: 'Bond', detail: 'Companion readiness', segments: 4 })
  });

  function clampValue(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function getSkillPathHints(owner) {
    return SKILL_PATH_HINTS[owner] || [];
  }

  function getResourceWidgetData(snapshot, options) {
    const settings = options || {};
    const clamp = settings.clamp || clampValue;
    const metaByClass = settings.resourceWidgetMeta || RESOURCE_WIDGET_META;
    const player = snapshot && snapshot.state && snapshot.state.player || {};
    if (!player.classId) return null;
    const stats = snapshot.stats || {};
    const classId = player.advancedClassId || player.classId;
    const classData = snapshot.advancedData || snapshot.classData || {};
    const meta = metaByClass[classId] || metaByClass[player.classId] || metaByClass.fighter;
    const max = Math.max(1, Number(stats.secondaryResourceMax || 1));
    const value = clamp(Number(player.resource || 0), 0, max);
    const ratio = clamp(value / max, 0, 1);
    const objects = Array.isArray(player.activeSkillObjects) ? player.activeSkillObjects : [];
    const mechanics = player.classMechanics || {};
    const time = Number(settings.nowSeconds == null ? Date.now() / 1000 : settings.nowSeconds);
    const traps = objects.filter((object) => object && object.type === 'trap');
    const runes = objects.filter((object) => object && /rune|glyph|circle/i.test(String(object.skillId || '')));
    const marked = Number(snapshot.markedEnemyCount || 0) || (snapshot.enemies || []).filter((enemy) => enemy && enemy.hp > 0 && (Number(enemy.marked || 0) > 0 || Number(enemy.weakPoint || 0) > 0)).length;
    const activeBuffIds = new Set((snapshot.activeBuffs || []).map((buff) => buff.id || buff.skillId));
    const danger = Number(player.hp || 0) / Math.max(1, Number(stats.maxHp || 1)) <= 0.35;
    const segments = Math.max(1, Number(meta.segments || 5));
    const filled = classId === 'guardian'
      ? Math.min(segments, Math.round(clamp(Number(mechanics.guardianImpact || 0) / 120, 0, 1) * segments))
      : classId === 'duelist'
        ? Math.min(segments, Math.round(Number(mechanics.duelistTempo || 0)))
        : classId === 'trapper'
      ? Math.min(segments, traps.filter((object) => Number(object.armedAt || 0) <= time).length)
      : classId === 'runeMage'
        ? Math.min(segments, runes.length + (activeBuffIds.has('runeCircle') ? 1 : 0))
        : classId === 'sniper'
          ? Math.min(segments, marked || Math.round(ratio * segments))
          : Math.round(ratio * segments);
    const detail = classId === 'guardian'
      ? `${Math.round(Number(mechanics.guardianImpact || 0))}/120 impact stored`
      : classId === 'duelist'
        ? `${Math.round(Number(mechanics.duelistTempo || 0))}/${segments} tempo${Number(mechanics.duelistTempoExpiresAt || 0) > time ? ' active' : ''}`
        : classId === 'trapper'
      ? `${traps.filter((object) => Number(object.armedAt || 0) <= time).length}/${segments} armed`
      : classId === 'runeMage'
        ? `${filled}/${segments} rune setup`
        : classId === 'sniper'
          ? `${marked} weak-point target${marked === 1 ? '' : 's'}`
          : classId === 'fireMage' && ratio >= 0.82
            ? 'High heat: vent soon'
            : classId === 'berserker' && danger
              ? 'Danger bonus active'
              : meta.detail;
    return {
      id: classId,
      type: meta.type,
      label: classData.resourceName || meta.label,
      detail,
      value,
      max,
      ratio,
      segments,
      filled,
      color: classData.resourceColor || '#7bdff2',
      danger,
      warning: classId === 'fireMage' && ratio >= 0.82
    };
  }

  function getCanvasResourceWidgetMetadata(widget, x, y, w, h, options) {
    if (!widget) return null;
    const settings = options || {};
    const formatNumber = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (value) => String(Math.round(Number(value) || 0));
    const urgent = !!(widget.warning || widget.danger);
    const labelW = Math.min(72, Math.max(42, w * 0.38));
    const segmentGap = 3;
    const segmentX = x + labelW + 10;
    const segmentY = y + 6;
    const segmentW = Math.max(5, (w - labelW - 24 - segmentGap * Math.max(0, widget.segments - 1)) / widget.segments);
    const segments = [];
    for (let index = 0; index < widget.segments; index += 1) {
      const filled = index < widget.filled;
      const sx = segmentX + index * (segmentW + segmentGap);
      const heatOffset = widget.type === 'heat' ? (widget.segments - index - 1) * 1.2 : 0;
      segments.push({
        x: sx,
        y: segmentY + heatOffset,
        w: segmentW,
        h: h - 12 - heatOffset,
        radius: 4,
        fill: filled ? widget.color : 'rgba(238,246,255,0.16)',
        stroke: filled ? 'rgba(255,255,255,0.28)' : ''
      });
    }
    return {
      urgent,
      accent: urgent ? '#ffbe55' : widget.color,
      frame: {
        x,
        y,
        w,
        h,
        radius: 7,
        fill: urgent ? 'rgba(255,244,216,0.88)' : 'rgba(31,59,82,0.78)',
        stroke: urgent ? 'rgba(216,106,32,0.42)' : 'rgba(255,255,255,0.2)'
      },
      highlight: {
        x: x + 4,
        y: y + 2,
        w: w - 8,
        h: 1,
        fill: urgent ? 'rgba(255,255,255,0.42)' : 'rgba(255,255,255,0.14)'
      },
      labelText: {
        value: widget.label,
        x: x + 7,
        y: y + 5,
        color: urgent ? '#9a5b36' : '#d8f4ff',
        font: '900 8px system-ui',
        maxWidth: labelW,
        maxLines: 1,
        lineHeight: 9
      },
      segments,
      valueText: {
        value: `${formatNumber(widget.value)}/${formatNumber(widget.max)}`,
        x: x + w - 6,
        y: y + 6,
        color: urgent ? '#102033' : '#f7fbff',
        font: '900 8px system-ui',
        align: 'right',
        maxWidth: 42,
        maxLines: 1,
        lineHeight: 9
      }
    };
  }

  function drawCanvasResourceWidget(ctx, widget, x, y, w, h, options) {
    const settings = options || {};
    const drawRoundRect = settings.drawRoundRect;
    const drawFillRect = settings.drawFillRect;
    const drawCanvasText = settings.drawCanvasText;
    const meta = getCanvasResourceWidgetMetadata(widget, x, y, w, h, {
      formatAbbreviatedInteger: settings.formatAbbreviatedInteger
    });
    if (!meta) return false;
    if (
      typeof drawRoundRect !== 'function' ||
      typeof drawFillRect !== 'function' ||
      typeof drawCanvasText !== 'function'
    ) {
      return false;
    }
    drawRoundRect(meta.frame);
    drawFillRect(meta.highlight);
    drawCanvasText(meta.labelText);
    meta.segments.forEach((segment) => {
      drawRoundRect(segment);
    });
    drawCanvasText(meta.valueText);
    return true;
  }

  function createResourceWidgetUiHelpers(options) {
    const settings = options || {};
    return Object.freeze({
      getSkillPathHints,
      getResourceWidgetData(snapshot) {
        const helperOptions = {
          clamp: settings.clamp,
          resourceWidgetMeta: settings.resourceWidgetMeta
        };
        if (Object.prototype.hasOwnProperty.call(settings, 'nowSeconds')) {
          helperOptions.nowSeconds = settings.nowSeconds;
        } else if (typeof settings.getNowSeconds === 'function') {
          helperOptions.nowSeconds = settings.getNowSeconds();
        }
        return getResourceWidgetData(snapshot, helperOptions);
      },
      getCanvasResourceWidgetMetadata(widget, x, y, w, h) {
        return getCanvasResourceWidgetMetadata(widget, x, y, w, h, {
          formatAbbreviatedInteger: settings.formatAbbreviatedInteger
        });
      },
      drawCanvasResourceWidget(ctx, widget, x, y, w, h, drawOptions) {
        return drawCanvasResourceWidget(ctx, widget, x, y, w, h, Object.assign({}, drawOptions || {}, {
          formatAbbreviatedInteger: settings.formatAbbreviatedInteger
        }));
      }
    });
  }

  const api = {
    SKILL_PATH_HINTS,
    RESOURCE_WIDGET_META,
    createResourceWidgetUiHelpers,
    getSkillPathHints,
    getResourceWidgetData,
    getCanvasResourceWidgetMetadata,
    drawCanvasResourceWidget
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.resourceWidgets = Object.assign({}, modules.resourceWidgets || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
