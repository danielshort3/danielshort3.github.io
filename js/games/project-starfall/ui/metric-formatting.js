(function initProjectStarfallUiMetricFormatting(global) {
  'use strict';

  function clampValue(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatSigned(value) {
    const number = Number(value) || 0;
    return `${number >= 0 ? '+' : ''}${number}`;
  }

  function formatMetricRate(value, maxDecimals, options) {
    const settings = options || {};
    const formatAbbreviatedNumber = settings.formatAbbreviatedNumber || function formatAbbreviatedNumberFallback(number) {
      return String(Math.round(Number(number) || 0));
    };
    const number = Math.max(0, Number(value) || 0);
    if (number >= 1000) return formatAbbreviatedNumber(number, { decimals: 1 });
    const decimals = Number.isFinite(Number(maxDecimals)) ? Math.max(0, Math.floor(Number(maxDecimals))) : 1;
    return number.toFixed(decimals);
  }

  function formatSignedMetricRate(value, maxDecimals, options) {
    const number = Number(value) || 0;
    if (number >= 0) return formatMetricRate(number, maxDecimals, options);
    return `-${formatMetricRate(Math.abs(number), maxDecimals, options)}`;
  }

  function formatMetricEta(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    if (!value) return '--';
    if (value < 1) return '1s';
    if (value < 60) return `${Math.ceil(value)}s`;
    const totalSeconds = Math.ceil(value);
    const minutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = totalSeconds % 60;
    if (minutes < 60) return remainingSeconds ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
    const hours = Math.floor(totalSeconds / 3600);
    const remainingMinutes = minutes % 60;
    if (hours < 24) return remainingMinutes ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    return remainingHours ? `${days}d ${remainingHours}h` : `${days}d`;
  }

  function formatMeterPercent(value, max, options) {
    const clamp = options && options.clamp || clampValue;
    const total = Math.max(1, Number(max) || 1);
    const amount = clamp(Number(value) || 0, 0, total);
    const percent = amount / total * 100;
    if (percent > 0 && percent < 1) return '<1%';
    return `${Math.floor(percent)}%`;
  }

  function formatXpMeterPercent(value, max, options) {
    const clamp = options && options.clamp || clampValue;
    const total = Math.max(1, Number(max) || 1);
    const amount = clamp(Number(value) || 0, 0, total);
    return `${(amount / total * 100).toFixed(2)}%`;
  }

  function formatMonsterRange(range, options) {
    const formatAbbreviatedInteger = options && options.formatAbbreviatedInteger || function formatAbbreviatedIntegerFallback(number) {
      return String(Math.round(Number(number) || 0));
    };
    if (!range) return '???';
    const min = Math.round(Number(range.min || 0));
    const max = Math.round(Number(range.max || min));
    return min === max ? formatAbbreviatedInteger(min) : `${formatAbbreviatedInteger(min)}-${formatAbbreviatedInteger(max)}`;
  }

  function formatMonsterPercent(value) {
    const percent = Math.max(0, Number(value || 0) * 100);
    return percent >= 10 ? `${Math.round(percent)}%` : `${percent.toFixed(1).replace(/\.0$/, '')}%`;
  }

  function formatCooldownLabelFallback(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    if (value < 10) return `${value.toFixed(1).replace(/\.0$/, '')}s`;
    return `${Math.ceil(value)}s`;
  }

  function formatDungeonRespawnLabel(dungeon, options) {
    if (!dungeon || !dungeon.bossRespawning) return '';
    const formatCooldownLabel = options && options.formatCooldownLabel || formatCooldownLabelFallback;
    return `Boss respawns in ${formatCooldownLabel(dungeon.bossRespawnRemaining)}`;
  }

  function getStarfallLevelXp(level) {
    const normalizedLevel = Math.max(1, Number(level) || 1);
    return Math.round(360 + 130 * normalizedLevel + 18 * normalizedLevel * normalizedLevel + 0.16 * normalizedLevel * normalizedLevel * normalizedLevel);
  }

  function getSnapshotNextLevelXp(snapshot) {
    const player = (((snapshot || {}).state || {}).player) || {};
    return Math.max(1, Number((snapshot || {}).nextLevelXp || getStarfallLevelXp(player.level)) || 1);
  }

  function getCombatMetricRates(engineRates, snapshot, options) {
    const settings = options || {};
    const formatEta = settings.formatMetricEta || formatMetricEta;
    if (engineRates) {
      const rates = engineRates;
      const levelEtaSeconds = Number(rates.levelEtaSeconds || 0);
      return {
        damagePerSecond: Number(rates.damagePerSecond || 0),
        currencyPerHour: Number(rates.currencyPerHour || 0),
        currencySpentPerHour: Number(rates.currencySpentPerHour || 0),
        netCurrencyPerHour: Number(rates.netCurrencyPerHour || 0),
        xpPerHour: Number(rates.xpPerHour || 0),
        killsPerHour: Number(rates.killsPerHour || 0),
        bossKillsPerHour: Number(rates.bossKillsPerHour || 0),
        eliteKillsPerHour: Number(rates.eliteKillsPerHour || 0),
        dropsPerHour: Number(rates.dropsPerHour || 0),
        lootPerHour: Number(rates.lootPerHour || 0),
        consumablesPerHour: Number(rates.consumablesPerHour || 0),
        potionsPerHour: Number(rates.potionsPerHour || 0),
        potionCostPerHour: Number(rates.potionCostPerHour || 0),
        deathsPerHour: Number(rates.deathsPerHour || 0),
        damageTakenPerHour: Number(rates.damageTakenPerHour || 0),
        xpRemaining: Math.max(0, Number(rates.xpRemaining || 0)),
        levelEtaSeconds,
        levelEtaLabel: formatEta(levelEtaSeconds),
        showPanel: !!rates.showPanel,
        windowSeconds: Number(rates.windowSeconds || 0),
        elapsedSeconds: Number(rates.elapsedSeconds || 0)
      };
    }
    const metrics = (((snapshot || {}).state || {}).session || {}).combatMetrics || {};
    const player = (((snapshot || {}).state || {}).player) || {};
    const getNextLevelXp = settings.getSnapshotNextLevelXp || getSnapshotNextLevelXp;
    const nowMs = typeof settings.nowMs === 'function' ? settings.nowMs : () => Date.now();
    const startedAt = Number(metrics.startedAt || 0);
    const damageDealt = Number(metrics.damageDealt || 0);
    const currencyGained = Number(metrics.currencyGained || 0);
    const currencySpent = Number(metrics.currencySpent || 0);
    const xpGained = Number(metrics.xpGained || 0);
    const kills = Number(metrics.kills || 0);
    const bossKills = Number(metrics.bossKills || 0);
    const eliteKills = Number(metrics.eliteKills || 0);
    const dropsGenerated = Number(metrics.dropsGenerated || 0);
    const dropsLooted = Number(metrics.dropsLooted || 0);
    const consumablesUsed = Number(metrics.consumablesUsed || 0);
    const potionsUsed = Number(metrics.potionsUsed || 0);
    const potionCost = Number(metrics.potionCost || 0);
    const deaths = Number(metrics.deaths || 0);
    const damageTaken = Number(metrics.damageTaken || 0);
    const elapsedSeconds = Math.max(0.0001, nowMs() / 1000 - startedAt);
    const elapsedHours = elapsedSeconds / 3600;
    const xpPerHour = xpGained / elapsedHours;
    const xpNeeded = getNextLevelXp(snapshot);
    const xpRemaining = Math.max(0, xpNeeded - Number(player.xp || 0));
    const levelEtaSeconds = xpPerHour > 0 && xpRemaining > 0 ? xpRemaining / xpPerHour * 3600 : 0;
    return {
      damagePerSecond: damageDealt / elapsedSeconds,
      currencyPerHour: currencyGained / elapsedHours,
      currencySpentPerHour: currencySpent / elapsedHours,
      netCurrencyPerHour: (currencyGained - currencySpent) / elapsedHours,
      xpPerHour,
      killsPerHour: kills / elapsedHours,
      bossKillsPerHour: bossKills / elapsedHours,
      eliteKillsPerHour: eliteKills / elapsedHours,
      dropsPerHour: dropsGenerated / elapsedHours,
      lootPerHour: dropsLooted / elapsedHours,
      consumablesPerHour: consumablesUsed / elapsedHours,
      potionsPerHour: potionsUsed / elapsedHours,
      potionCostPerHour: potionCost / elapsedHours,
      deathsPerHour: deaths / elapsedHours,
      damageTakenPerHour: damageTaken / elapsedHours,
      xpRemaining,
      levelEtaSeconds,
      levelEtaLabel: formatEta(levelEtaSeconds),
      showPanel: !!metrics.showPanel
    };
  }

  function getCombatMetricsPanelMetadata(combatRates, box, options) {
    if (!combatRates || !combatRates.showPanel || !box) return null;
    const settings = options || {};
    const formatRate = typeof settings.formatMetricRate === 'function'
      ? settings.formatMetricRate
      : (value, decimals) => formatMetricRate(value, decimals, settings);
    const formatSignedRate = typeof settings.formatSignedMetricRate === 'function'
      ? settings.formatSignedMetricRate
      : (value, decimals) => formatSignedMetricRate(value, decimals, settings);
    const x = box.x;
    const y = box.y;
    const w = box.w;
    const h = box.h;
    const rowValues = [
      ['DPS', `${formatRate(combatRates.damagePerSecond, 1)}/s`],
      ['Kills', `${formatRate(combatRates.killsPerHour, 0)}/h`],
      ['Coins', `${formatRate(combatRates.currencyPerHour, 0)}/h`],
      ['Spent', `${formatRate(combatRates.currencySpentPerHour, 0)}/h`],
      ['Net', `${formatSignedRate(combatRates.netCurrencyPerHour, 0)}/h`],
      ['XP', `${formatRate(combatRates.xpPerHour, 0)}/h`],
      ['Drops', `${formatRate(combatRates.dropsPerHour, 0)}/h`],
      ['Potions', `${formatRate(combatRates.potionsPerHour, 1)}/h`],
      ['Deaths', `${formatRate(combatRates.deathsPerHour, 1)}/h`],
      ['Level ETA', combatRates.levelEtaLabel]
    ];
    return {
      box: { x, y, w, h },
      shadow: {
        color: 'rgba(9,31,59,0.34)',
        blur: 14,
        offsetY: 5
      },
      frame: {
        x,
        y,
        w,
        h,
        radius: 9,
        fill: 'rgba(9,31,59,0.88)',
        stroke: 'rgba(255,255,255,0.28)'
      },
      region: { type: 'combat-metrics-drag', x, y, w, h },
      titleText: {
        value: 'Combat Metrics',
        x: x + 12,
        y: y + 9,
        color: '#ffffff',
        font: '950 11px system-ui',
        maxWidth: 132,
        lineHeight: 12,
        maxLines: 1
      },
      shortcutText: {
        value: settings.keyLabel || '',
        x: x + w - 12,
        y: y + 9,
        color: '#9be7ff',
        font: '900 10px system-ui',
        align: 'right',
        maxWidth: 48,
        lineHeight: 11,
        maxLines: 1
      },
      rows: rowValues.map((row, index) => {
        const rowY = y + 28 + index * 12;
        return {
          label: row[0],
          value: row[1],
          labelText: {
            value: row[0],
            x: x + 12,
            y: rowY,
            color: '#d8e5ec',
            font: '850 9px system-ui',
            maxWidth: 58,
            lineHeight: 10,
            maxLines: 1
          },
          valueText: {
            value: row[1],
            x: x + w - 12,
            y: rowY,
            color: index === 0 ? '#ffd166' : '#9be7ff',
            font: '900 10px system-ui',
            align: 'right',
            maxWidth: 112,
            lineHeight: 11,
            maxLines: 1
          }
        };
      })
    };
  }

  function getCombatMetricsControlsMetadata(showPanel, x, y, w) {
    const h = 68;
    return {
      h,
      frame: {
        x,
        y,
        w,
        h,
        radius: 7,
        fill: '#fbfaf6',
        stroke: 'rgba(16,32,51,0.14)'
      },
      titleText: {
        value: 'Combat Metrics',
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      statusText: {
        value: `F4 toggles: ${showPanel ? 'On' : 'Off'}`,
        x: x + w - 12,
        y: y + 10,
        color: showPanel ? '#2f7dd6' : '#5f6f7a',
        font: '900 11px system-ui',
        align: 'right'
      },
      toggleButton: {
        label: showPanel ? 'Hide Panel' : 'Show Panel',
        x: x + 12,
        y: y + 34,
        w: Math.min(132, w - 24),
        h: 28,
        region: { type: 'toggle-combat-metrics' },
        disabled: false
      },
      nextY: y + h
    };
  }

  function createMetricFormattingUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      formatAbbreviatedNumber: settings.formatAbbreviatedNumber,
      formatAbbreviatedInteger: settings.formatAbbreviatedInteger,
      formatCooldownLabel: settings.formatCooldownLabel,
      formatMetricEta: settings.formatMetricEta,
      formatMetricRate: settings.formatMetricRate,
      formatSignedMetricRate: settings.formatSignedMetricRate,
      getSnapshotNextLevelXp: settings.getSnapshotNextLevelXp,
      nowMs: settings.nowMs,
      clamp: settings.clamp
    });
    const mergeOptions = (override) => Object.assign({}, helperOptions, override || {});
    return Object.freeze({
      formatSigned,
      formatMetricRate: (value, maxDecimals, override) => formatMetricRate(value, maxDecimals, mergeOptions(override)),
      formatSignedMetricRate: (value, maxDecimals, override) => formatSignedMetricRate(value, maxDecimals, mergeOptions(override)),
      formatMetricEta,
      formatMeterPercent: (value, max, override) => formatMeterPercent(value, max, mergeOptions(override)),
      formatXpMeterPercent: (value, max, override) => formatXpMeterPercent(value, max, mergeOptions(override)),
      formatMonsterRange: (range, override) => formatMonsterRange(range, mergeOptions(override)),
      formatMonsterPercent,
      formatDungeonRespawnLabel: (dungeon, override) => formatDungeonRespawnLabel(dungeon, mergeOptions(override)),
      getStarfallLevelXp,
      getSnapshotNextLevelXp,
      getCombatMetricRates: (engineRates, snapshot, override) => getCombatMetricRates(engineRates, snapshot, mergeOptions(override)),
      getCombatMetricsPanelMetadata: (combatRates, box, override) => getCombatMetricsPanelMetadata(combatRates, box, mergeOptions(override)),
      getCombatMetricsControlsMetadata
    });
  }

  const api = {
    formatSigned,
    formatMetricRate,
    formatSignedMetricRate,
    formatMetricEta,
    formatMeterPercent,
    formatXpMeterPercent,
    formatMonsterRange,
    formatMonsterPercent,
    formatDungeonRespawnLabel,
    getStarfallLevelXp,
    getSnapshotNextLevelXp,
    getCombatMetricRates,
    getCombatMetricsPanelMetadata,
    getCombatMetricsControlsMetadata,
    createMetricFormattingUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.metricFormatting = Object.assign({}, modules.metricFormatting || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
