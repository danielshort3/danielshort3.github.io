(function initProjectStarfallUiSkillDisplay(global) {
  'use strict';

  const CoreFormat = (typeof require === 'function' ? require('../core/format.js') : null) || global.ProjectStarfallCore || {};
  const UiModules = global.ProjectStarfallUiModules || {};
  const UiSkillMetadata = (typeof require === 'function' ? require('./skill-metadata.js') : null) || UiModules.skillMetadata || {};

  const formatCooldownLabel = CoreFormat.formatCooldownLabel || function formatCooldownLabelFallback(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    if (value < 10) return `${value.toFixed(1).replace(/\.0$/, '')}s`;
    return `${Math.ceil(value)}s`;
  };

  function clampValue(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatIntegerFallback(value) {
    return String(Math.round(Number(value) || 0));
  }

  function getDamageFloorRatio(snapshot, options) {
    const clamp = options && options.clamp || clampValue;
    const stats = snapshot && snapshot.stats || {};
    const floorPercent = Number(stats.damageRange && stats.damageRange.floorPercent || stats.damageFloor || 50);
    return clamp(floorPercent / 100, 0.5, 0.9);
  }

  function formatDamageRange(range, options) {
    if (!range) return '';
    const formatAbbreviatedInteger = options && options.formatAbbreviatedInteger || formatIntegerFallback;
    const min = Math.max(1, Math.round(Number(range.min || 1)));
    const max = Math.max(1, Math.round(Number(range.max || 1)));
    return `${formatAbbreviatedInteger(min)}-${formatAbbreviatedInteger(max)}`;
  }

  function formatSkillDamageRange(maxValue, floorRatio, options) {
    const formatSkillPercent = options && options.formatSkillPercent || UiSkillMetadata.formatSkillPercent || ((value) => `${Math.max(1, Math.round(Number(value || 0) * 100))}%`);
    const maxLabel = formatSkillPercent(maxValue);
    const minLabel = formatSkillPercent(Number(maxValue || 0) * floorRatio);
    return minLabel === maxLabel ? maxLabel : `${minLabel}-${maxLabel}`;
  }

  function getSkillDamageEstimate(snapshot, skill, nextRank, options) {
    const settings = options || {};
    const isPassiveSkill = settings.isPassiveSkill || UiSkillMetadata.isPassiveSkill || (() => false);
    const isDefensiveSkill = settings.isDefensiveSkill || UiSkillMetadata.isDefensiveSkill || (() => false);
    const isMobilitySkill = settings.isMobilitySkill || UiSkillMetadata.isMobilitySkill || (() => false);
    const isBuffSkill = settings.isBuffSkill || UiSkillMetadata.isBuffSkill || (() => false);
    const getSkillLineCount = settings.getSkillLineCount || UiSkillMetadata.getSkillLineCount || (() => 1);
    const formatSkillPercent = settings.formatSkillPercent || UiSkillMetadata.formatSkillPercent || ((value) => `${Math.max(1, Math.round(Number(value || 0) * 100))}%`);
    if (!skill || isPassiveSkill(skill)) return 'Passive';
    const rank = Math.max(1, Number(nextRank) || 1);
    if (isDefensiveSkill(skill)) {
      return `Shield ${formatSkillPercent(0.12 + rank * 0.01)} max HP`;
    }
    if (skill.id.includes('trap') || skill.id.includes('field') || skill.id.includes('glyph') || skill.id.includes('circle')) {
      return `Field ${formatSkillPercent(0.7 + rank * 0.04)}/tick`;
    }
    if (skill.movementEffect && skill.movementEffect.trail === 'flame') {
      return `Trail ${formatSkillPercent(0.55 + rank * 0.035)}/field`;
    }
    if (isMobilitySkill(skill)) return 'Mobility';
    if (isBuffSkill(skill) || skill.id.includes('stance') || skill.id.includes('surge') || skill.id.includes('breath')) return `Buff ${Math.round(8 + rank * 0.4)}s`;
    const finisher = String(skill.type || '').includes('Finisher') ? 0.65 : 0;
    const total = (1.25 + rank * 0.11 + finisher) * (Number(skill.lineDamageScale || 1) || 1);
    const lines = getSkillLineCount(skill) || 1;
    const range = formatSkillDamageRange(total / lines, getDamageFloorRatio(snapshot, settings), settings);
    return lines > 1 ? `Damage ${range} x${lines}` : `Damage ${range}`;
  }

  function limitSkillSummaryText(value, limit) {
    const max = Math.max(18, Number(limit || 58) || 58);
    const text = String(value || '').replace(/\s+/g, ' ').replace(/^Self:\s*/i, '').trim();
    return text.length > max ? `${text.slice(0, Math.max(0, max - 3)).trim()}...` : text;
  }

  function getSkillPassiveBonusSummary(skill, rank, options) {
    const compactStats = options && options.compactStats || (() => '');
    if (!skill || !skill.passiveStats) return 'Passive bonus';
    const currentRank = Math.max(0, Number(rank || 0) || 0);
    const stats = Object.entries(skill.passiveStats).reduce((result, [key, value]) => {
      result[key] = Number(value || 0) * Math.max(1, currentRank || 1);
      return result;
    }, {});
    const summary = compactStats(stats, 2);
    if (!summary) return 'Passive bonus';
    return `Passive ${currentRank > 0 ? 'Bonus' : 'Per Lv'} ${summary}`;
  }

  function getSkillPrimaryEffectSummary(snapshot, skill, rank, options) {
    const settings = options || {};
    const isPassiveSkill = settings.isPassiveSkill || UiSkillMetadata.isPassiveSkill || (() => false);
    if (!skill) return '';
    if (isPassiveSkill(skill)) return getSkillPassiveBonusSummary(skill, rank, settings);
    if (skill.partyEffect) return limitSkillSummaryText(skill.partyEffect, 58);
    return getSkillDamageEstimate(snapshot, skill, rank || 1, settings);
  }

  function formatSkillMissingRequirementSummary(labels, limit) {
    const entries = (labels || []).filter(Boolean);
    if (!entries.length) return '';
    const visible = entries.slice(0, Math.max(1, Number(limit || 1) || 1));
    const suffix = entries.length > visible.length ? ` +${entries.length - visible.length} more` : '';
    return limitSkillSummaryText(`${visible.join(' or ')}${suffix}`, 54);
  }

  function getSkillBreakpointLabels(skill, options) {
    const isPassiveSkill = options && options.isPassiveSkill || UiSkillMetadata.isPassiveSkill || (() => false);
    if (!skill) return [];
    const ranks = [3, 5, 10, 20].filter((rank) => rank <= Number(skill.maxRank || 0));
    if (skill.primaryTraining) return ranks.filter((rank) => rank <= 10).map((rank) => `Level ${rank} smoother training`);
    if (skill.roleTags && skill.roleTags.includes('Party')) return ranks.filter((rank) => rank >= 5).map((rank) => `Level ${rank} stronger self-buff`);
    if (isPassiveSkill(skill)) return ranks.map((rank) => `Level ${rank} passive scaling`);
    return ranks.slice(0, 2).map((rank) => `Level ${rank} payoff`);
  }

  function getSkillNextLevelSummary(snapshot, skill, options) {
    const settings = options || {};
    if (!skill) return '';
    const rank = Number(snapshot.state.skills[skill.id] || 0);
    if (rank >= Number(skill.maxRank || 0)) return 'At max level';
    const nextRank = Math.min(Number(skill.maxRank || 1), rank + 1);
    const getSkillMpCost = settings.getSkillMpCost || UiSkillMetadata.getSkillMpCost || (() => 0);
    const getVisibleSkillCooldown = settings.getVisibleSkillCooldown || UiSkillMetadata.getVisibleSkillCooldown || (() => 0);
    const formatAbbreviatedInteger = settings.formatAbbreviatedInteger || formatIntegerFallback;
    const currentCost = getSkillMpCost(skill, rank);
    const nextCost = getSkillMpCost(skill, nextRank);
    const nextDamage = getSkillDamageEstimate(snapshot, skill, nextRank, settings);
    const pieces = [`Next level ${nextRank}: ${nextDamage}`];
    if (nextCost !== currentCost) pieces.push(`MP ${formatAbbreviatedInteger(currentCost)}->${formatAbbreviatedInteger(nextCost)}`);
    const cooldown = getVisibleSkillCooldown(skill);
    if (cooldown) pieces.push(`CD ${(settings.formatCooldownLabel || formatCooldownLabel)(cooldown)}`);
    return pieces.join(' | ');
  }

  function getCanvasSkillDependencyCueMetadata(requiredRank, x, y, size, state) {
    const config = state || {};
    const stroke = config.future ? '#2f7dd6' : config.met ? '#3a9b5f' : '#c44949';
    const fill = config.future ? '#eef6ff' : config.met ? '#eaf8ee' : '#fff0ee';
    const label = String(Math.max(1, Number(requiredRank || 1)));
    const badgeW = Math.max(8, label.length * 5 + 4);
    return {
      stroke,
      fill,
      label,
      frame: {
        x: x - 1,
        y: y - 1,
        w: size + 2,
        h: size + 2,
        radius: Math.min(5, size / 3),
        fill: null,
        stroke,
        lineWidth: 1.4
      },
      badge: {
        x: x + size - badgeW + 2,
        y: y + size - 7,
        w: badgeW,
        h: 8,
        radius: 3,
        fill,
        stroke
      },
      text: {
        value: label,
        x: x + size - badgeW / 2 + 2,
        y: y + size - 3,
        color: '#102033',
        font: '800 6px system-ui',
        align: 'center',
        baseline: 'middle'
      }
    };
  }

  function drawCanvasSkillDependencyCue(ctx, skill, requiredRank, x, y, size, state, options) {
    const settings = options || {};
    const drawSkillIcon = settings.drawSkillIcon;
    const drawRoundRect = settings.drawRoundRect;
    const drawCanvasText = settings.drawCanvasText;
    const cue = getCanvasSkillDependencyCueMetadata(requiredRank, x, y, size, state);
    if (
      typeof drawSkillIcon !== 'function' ||
      typeof drawRoundRect !== 'function' ||
      typeof drawCanvasText !== 'function'
    ) {
      return false;
    }
    drawSkillIcon({ skill, x, y, size });
    drawRoundRect(cue.frame);
    drawRoundRect(cue.badge);
    drawCanvasText(cue.text);
    return true;
  }

  function normalizeFallbackSkillIconKind(value) {
    return String(value || 'slash').replace(/[^a-z0-9-]/gi, '') || 'slash';
  }

  function drawCanvasFallbackSkillIcon(ctx, kind, x, y, size, options) {
    if (!ctx) return false;
    const settings = options || {};
    const iconClass = typeof settings.iconClass === 'function' ? settings.iconClass : normalizeFallbackSkillIconKind;
    const cx = x + size / 2;
    const cy = y + size / 2;
    const r = size * 0.32;
    ctx.save();
    ctx.strokeStyle = '#eef6ff';
    ctx.fillStyle = '#7bdff2';
    ctx.lineWidth = Math.max(2, size * 0.08);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const iconKind = iconClass(kind);
    if (iconKind === 'guard') {
      ctx.beginPath();
      ctx.moveTo(cx, y + size * 0.22);
      ctx.lineTo(x + size * 0.72, y + size * 0.34);
      ctx.lineTo(x + size * 0.64, y + size * 0.72);
      ctx.lineTo(cx, y + size * 0.84);
      ctx.lineTo(x + size * 0.36, y + size * 0.72);
      ctx.lineTo(x + size * 0.28, y + size * 0.34);
      ctx.closePath();
      ctx.stroke();
    } else if (iconKind === 'arrow') {
      ctx.beginPath();
      ctx.moveTo(x + size * 0.24, cy + size * 0.12);
      ctx.lineTo(x + size * 0.75, cy - size * 0.12);
      ctx.lineTo(x + size * 0.58, cy - size * 0.22);
      ctx.moveTo(x + size * 0.75, cy - size * 0.12);
      ctx.lineTo(x + size * 0.62, cy + size * 0.08);
      ctx.stroke();
    } else if (iconKind === 'mark') {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.moveTo(cx - r * 1.3, cy);
      ctx.lineTo(cx + r * 1.3, cy);
      ctx.moveTo(cx, cy - r * 1.3);
      ctx.lineTo(cx, cy + r * 1.3);
      ctx.stroke();
    } else if (iconKind === 'field' || iconKind === 'area' || iconKind === 'buff') {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.arc(cx, cy, r * 0.55, 0, Math.PI * 2);
      ctx.stroke();
    } else if (iconKind === 'blink' || iconKind === 'mobility') {
      ctx.beginPath();
      ctx.moveTo(x + size * 0.2, cy + size * 0.16);
      ctx.lineTo(x + size * 0.7, cy - size * 0.16);
      ctx.moveTo(x + size * 0.28, cy);
      ctx.lineTo(x + size * 0.78, cy - size * 0.32);
      ctx.moveTo(x + size * 0.2, cy + size * 0.3);
      ctx.lineTo(x + size * 0.5, cy + size * 0.12);
      ctx.stroke();
    } else if (iconKind === 'magic' || iconKind === 'burst') {
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.72, 0, Math.PI * 2);
      ctx.moveTo(cx - r * 1.15, cy);
      ctx.lineTo(cx + r * 1.15, cy);
      ctx.moveTo(cx, cy - r * 1.15);
      ctx.lineTo(cx, cy + r * 1.15);
      ctx.stroke();
    } else if (iconKind === 'break') {
      ctx.beginPath();
      ctx.rect(x + size * 0.27, y + size * 0.28, size * 0.46, size * 0.44);
      ctx.moveTo(cx - size * 0.05, y + size * 0.28);
      ctx.lineTo(cx + size * 0.06, cy);
      ctx.lineTo(cx - size * 0.03, y + size * 0.72);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.moveTo(x + size * 0.24, y + size * 0.72);
      ctx.lineTo(x + size * 0.74, y + size * 0.28);
      ctx.moveTo(x + size * 0.34, y + size * 0.78);
      ctx.lineTo(x + size * 0.82, y + size * 0.52);
      ctx.stroke();
    }
    ctx.restore();
    return true;
  }

  function drawCanvasSkillIcon(ctx, skill, image, x, y, size, options) {
    const settings = options || {};
    const drawCanvasUiSlot = settings.drawCanvasUiSlot;
    const drawFallbackSkillIcon = settings.drawFallbackSkillIcon;
    if (typeof drawCanvasUiSlot !== 'function') return false;
    drawCanvasUiSlot({
      x,
      y,
      w: size,
      h: size,
      options: {
        variant: 'dark',
        radius: Math.min(8, size / 4),
        stroke: 'rgba(89,216,255,0.42)'
      }
    });
    if (image && ctx && typeof ctx.drawImage === 'function') {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, y + 2, Math.max(1, size - 4), Math.max(1, size - 4));
      ctx.clip();
      ctx.drawImage(image, x + 2, y + 2, Math.max(1, size - 4), Math.max(1, size - 4));
      ctx.restore();
      return true;
    }
    if (typeof drawFallbackSkillIcon !== 'function') return false;
    drawFallbackSkillIcon({
      kind: skill ? skill.iconKind : 'slash',
      x,
      y,
      size
    });
    return true;
  }

  function createSkillDisplayUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      clamp: settings.clamp || clampValue,
      compactStats: typeof settings.compactStats === 'function' ? settings.compactStats : () => '',
      formatAbbreviatedInteger: typeof settings.formatAbbreviatedInteger === 'function' ? settings.formatAbbreviatedInteger : formatIntegerFallback,
      formatCooldownLabel: typeof settings.formatCooldownLabel === 'function' ? settings.formatCooldownLabel : formatCooldownLabel,
      formatSkillPercent: typeof settings.formatSkillPercent === 'function' ? settings.formatSkillPercent : UiSkillMetadata.formatSkillPercent || ((value) => `${Math.max(1, Math.round(Number(value || 0) * 100))}%`),
      getSkillLineCount: typeof settings.getSkillLineCount === 'function' ? settings.getSkillLineCount : UiSkillMetadata.getSkillLineCount || (() => 1),
      getSkillMpCost: typeof settings.getSkillMpCost === 'function' ? settings.getSkillMpCost : UiSkillMetadata.getSkillMpCost || (() => 0),
      getVisibleSkillCooldown: typeof settings.getVisibleSkillCooldown === 'function' ? settings.getVisibleSkillCooldown : UiSkillMetadata.getVisibleSkillCooldown || (() => 0),
      iconClass: typeof settings.iconClass === 'function' ? settings.iconClass : normalizeFallbackSkillIconKind,
      isBuffSkill: typeof settings.isBuffSkill === 'function' ? settings.isBuffSkill : UiSkillMetadata.isBuffSkill || (() => false),
      isDefensiveSkill: typeof settings.isDefensiveSkill === 'function' ? settings.isDefensiveSkill : UiSkillMetadata.isDefensiveSkill || (() => false),
      isMobilitySkill: typeof settings.isMobilitySkill === 'function' ? settings.isMobilitySkill : UiSkillMetadata.isMobilitySkill || (() => false),
      isPassiveSkill: typeof settings.isPassiveSkill === 'function' ? settings.isPassiveSkill : UiSkillMetadata.isPassiveSkill || (() => false)
    });
    function getOptions(extra) {
      return Object.assign({}, helperOptions, extra || {});
    }
    return Object.freeze({
      getDamageFloorRatio: (snapshot, extraOptions) => getDamageFloorRatio(snapshot, getOptions(extraOptions)),
      formatDamageRange: (range, extraOptions) => formatDamageRange(range, getOptions(extraOptions)),
      formatSkillDamageRange: (maxValue, floorRatio, extraOptions) => formatSkillDamageRange(maxValue, floorRatio, getOptions(extraOptions)),
      getSkillDamageEstimate: (snapshot, skill, nextRank, extraOptions) => getSkillDamageEstimate(snapshot, skill, nextRank, getOptions(extraOptions)),
      limitSkillSummaryText,
      getSkillPassiveBonusSummary: (skill, rank, extraOptions) => getSkillPassiveBonusSummary(skill, rank, getOptions(extraOptions)),
      getSkillPrimaryEffectSummary: (snapshot, skill, rank, extraOptions) => getSkillPrimaryEffectSummary(snapshot, skill, rank, getOptions(extraOptions)),
      formatSkillMissingRequirementSummary,
      getSkillBreakpointLabels: (skill, extraOptions) => getSkillBreakpointLabels(skill, getOptions(extraOptions)),
      getSkillNextLevelSummary: (snapshot, skill, extraOptions) => getSkillNextLevelSummary(snapshot, skill, getOptions(extraOptions)),
      getCanvasSkillDependencyCueMetadata,
      drawCanvasSkillDependencyCue,
      drawCanvasFallbackSkillIcon: (ctx, kind, x, y, size, extraOptions) => drawCanvasFallbackSkillIcon(ctx, kind, x, y, size, Object.assign({}, extraOptions || {}, {
        iconClass: helperOptions.iconClass
      })),
      drawCanvasSkillIcon
    });
  }

  const api = {
    getDamageFloorRatio,
    formatDamageRange,
    formatSkillDamageRange,
    getSkillDamageEstimate,
    limitSkillSummaryText,
    getSkillPassiveBonusSummary,
    getSkillPrimaryEffectSummary,
    formatSkillMissingRequirementSummary,
    getSkillBreakpointLabels,
    getSkillNextLevelSummary,
    getCanvasSkillDependencyCueMetadata,
    drawCanvasSkillDependencyCue,
    normalizeFallbackSkillIconKind,
    drawCanvasFallbackSkillIcon,
    drawCanvasSkillIcon,
    createSkillDisplayUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.skillDisplay = Object.assign({}, modules.skillDisplay || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
