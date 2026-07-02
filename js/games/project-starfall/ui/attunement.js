(function initProjectStarfallUiAttunement(global) {
  'use strict';

  const UiModules = global.ProjectStarfallUiModules || {};
  const UiCanvasWindows = (typeof require === 'function' ? require('./canvas-windows.js') : null) || UiModules.canvasWindows || {};
  const POTENTIAL_MAX_LINE_COUNT = 5;
  const POTENTIAL_LINE_UPGRADE_CHANCES = Object.freeze({ 2: 0.8, 3: 0.55, 4: 0.35, 5: 0.2 });
  const POTENTIAL_LINE_DOWNGRADE_CHANCE_ON_FAIL = 1;
  const DOM_POTENTIAL_PROMPT_DRAG_HANDLE_ATTRIBUTES = Object.freeze([
    'data-starfall-potential-drag'
  ]);
  const DOM_POTENTIAL_PROMPT_DRAG_HANDLE_SELECTOR = DOM_POTENTIAL_PROMPT_DRAG_HANDLE_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function isPotentialLinePoolValidForSlot(pool, slot) {
    const allowedSlots = Array.isArray(pool && pool.slots) ? pool.slots : null;
    return !allowedSlots || !slot || allowedSlots.includes(String(slot || ''));
  }

  function isPotentialLinePoolValidForTier(pool, tierId) {
    return !!(pool && pool.values && Object.prototype.hasOwnProperty.call(pool.values, tierId));
  }

  function isPotentialLinePoolEligible(pool, tierId, slot) {
    return isPotentialLinePoolValidForTier(pool, tierId) && isPotentialLinePoolValidForSlot(pool, slot);
  }

  function normalizePotentialLine(line, tierId, item, options) {
    const settings = options || {};
    const getPotentialLinePoolForStat = typeof settings.getPotentialLinePoolForStat === 'function'
      ? settings.getPotentialLinePoolForStat
      : () => null;
    const isPoolEligible = typeof settings.isPotentialLinePoolEligible === 'function'
      ? settings.isPotentialLinePoolEligible
      : isPotentialLinePoolEligible;
    if (!line || typeof line !== 'object') return null;
    const stat = String(line.stat || '').trim();
    const value = Math.round(Number(line.value || 0) || 0);
    const pool = getPotentialLinePoolForStat(stat);
    if (!stat || !value || !pool || !isPoolEligible(pool, tierId, item && item.slot)) return null;
    return { stat, value };
  }

  function normalizePotentialLineCount(value, options) {
    const maxLineCount = Math.max(1, Math.floor(Number(options && options.maxLineCount || POTENTIAL_MAX_LINE_COUNT) || POTENTIAL_MAX_LINE_COUNT));
    return clamp(Math.floor(Number(value || 1) || 1), 1, maxLineCount);
  }

  function normalizePotentialLineArchive(archive, tierId, item, lineCount, options) {
    const settings = options || {};
    const maxLineCount = Math.max(1, Math.floor(Number(settings.maxLineCount || POTENTIAL_MAX_LINE_COUNT) || POTENTIAL_MAX_LINE_COUNT));
    const source = archive && typeof archive === 'object' ? archive : {};
    return Object.entries(source).reduce((normalized, [rawIndex, rawLine]) => {
      const index = Math.floor(Number(rawIndex));
      if (!Number.isFinite(index) || index < lineCount || index >= maxLineCount) return normalized;
      const line = normalizePotentialLine(rawLine, tierId, item, settings);
      if (line) normalized[index] = line;
      return normalized;
    }, {});
  }

  function normalizeItemPotential(potential, item, options) {
    const settings = options || {};
    const getPotentialTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    if (!potential || typeof potential !== 'object') return null;
    const tier = getPotentialTierDefinition(potential.tier);
    if (!tier) return null;
    const requestedLineCount = Object.prototype.hasOwnProperty.call(potential, 'lineCount')
      ? normalizePotentialLineCount(potential.lineCount, settings)
      : normalizePotentialLineCount(Array.isArray(potential.lines) && potential.lines.length ? potential.lines.length : 1, settings);
    const lines = (Array.isArray(potential.lines) ? potential.lines : [])
      .map((line) => normalizePotentialLine(line, tier.id, item, settings))
      .filter(Boolean)
      .slice(0, requestedLineCount);
    if (!lines.length) return null;
    const lineCount = normalizePotentialLineCount(Math.min(requestedLineCount, lines.length), settings);
    const lineArchive = normalizePotentialLineArchive(potential.lineArchive, tier.id, item, lineCount, settings);
    const normalized = { tier: tier.id, lineCount, lines: lines.slice(0, lineCount) };
    if (Object.keys(lineArchive).length) normalized.lineArchive = lineArchive;
    return normalized;
  }

  function getItemPotentialStats(item, options) {
    const stats = {};
    const normalize = options && typeof options.normalizeItemPotential === 'function'
      ? options.normalizeItemPotential
      : normalizeItemPotential;
    const potential = normalize(item && item.potential, item, options);
    (potential && potential.lines || []).forEach((line) => {
      stats[line.stat] = (stats[line.stat] || 0) + Number(line.value || 0);
    });
    return stats;
  }

  function getPotentialLineSummary(item, fallback, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (key) => String(key || '');
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (key, value) => String(value);
    const potential = normalize(item && item.potential, item, settings);
    if (!potential) return fallback || 'No attunement';
    return potential.lines
      .map((line) => `${formatStatName(line.stat)} ${formatStatValue(line.stat, line.value)}`)
      .join(', ');
  }

  function getPotentialLineRollRows(item, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    const getPotentialLinePoolForStat = typeof settings.getPotentialLinePoolForStat === 'function'
      ? settings.getPotentialLinePoolForStat
      : () => null;
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (key) => String(key || '');
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (key, value) => String(value);
    const potential = normalize(item && item.potential, item, settings);
    if (!potential) return [];
    return potential.lines.map((line, index) => {
      const pool = getPotentialLinePoolForStat(line.stat);
      const range = pool && pool.values && pool.values[potential.tier] || [];
      const min = Number(range[0]);
      const max = Number(range[1]);
      const value = Number(line.value || 0);
      const hasRange = Number.isFinite(min) && Number.isFinite(max);
      const percentile = !hasRange
        ? null
        : min === max
          ? 100
          : clamp(Math.round(((value - min) / (max - min)) * 100), 0, 100);
      return {
        stat: line.stat,
        lineIndex: index,
        value,
        label: `${formatStatName(line.stat)} ${formatStatValue(line.stat, value)}`,
        rangeLabel: hasRange ? `${formatStatValue(line.stat, min)} to ${formatStatValue(line.stat, max)}` : 'Range unknown',
        percentileLabel: percentile == null ? '' : `${percentile}%`
      };
    });
  }

  function getPotentialSummaryFromPotential(potential, fallback, item, options) {
    const target = Object.assign({}, item || {}, { potential });
    return getPotentialLineSummary(target, fallback, options);
  }

  function cloneNormalizedPotential(potential) {
    return potential ? {
      tier: potential.tier,
      lineCount: potential.lineCount,
      lines: (potential.lines || []).map((line) => Object.assign({}, line)),
      lineArchive: Object.assign({}, potential.lineArchive || {})
    } : null;
  }

  function getItemWithPotential(item, potential, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    if (!item) return null;
    const normalized = normalize(potential, item, settings);
    return Object.assign({}, item, {
      potential: cloneNormalizedPotential(normalized)
    });
  }

  function cloneItemPotentialForPrompt(potential, item, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    return cloneNormalizedPotential(normalize(potential, item, settings));
  }

  function getPotentialComparisonLineIndexFallback(row) {
    if (!row) return 999;
    const index = Number(row.lineIndex);
    return Number.isFinite(index) ? index : 999;
  }

  function getPotentialLineRowMap(potential, item, options) {
    const settings = options || {};
    const getItem = typeof settings.getItemWithPotential === 'function'
      ? settings.getItemWithPotential
      : (targetItem, targetPotential) => getItemWithPotential(targetItem, targetPotential, settings);
    const getRows = typeof settings.getPotentialLineRollRows === 'function'
      ? settings.getPotentialLineRollRows
      : (targetItem) => getPotentialLineRollRows(targetItem, settings);
    const getRank = typeof settings.getPotentialRollRank === 'function'
      ? settings.getPotentialRollRank
      : () => null;
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const target = getItem(item, potential);
    return getRows(target).reduce((rows, row, index) => {
      const lineIndex = Number.isFinite(Number(row.lineIndex)) ? Number(row.lineIndex) : index;
      const key = `line_${lineIndex}`;
      rows[key] = Object.assign({}, row, {
        key,
        lineIndex,
        lineCount: 1,
        rank: getRank(row.percentileLabel),
        label: formatStatName(row.stat)
      });
      return rows;
    }, {});
  }

  function getPotentialComparisonSide(potential, item, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    const getPotentialTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    const getItem = typeof settings.getItemWithPotential === 'function'
      ? settings.getItemWithPotential
      : (targetItem, targetPotential) => getItemWithPotential(targetItem, targetPotential, settings);
    const getRows = typeof settings.getPotentialLineRowMap === 'function'
      ? settings.getPotentialLineRowMap
      : (targetPotential, targetItem) => getPotentialLineRowMap(targetPotential, targetItem, settings);
    const normalized = normalize(potential, item, settings);
    return {
      potential: normalized,
      tier: normalized ? getPotentialTierDefinition(normalized.tier) : null,
      lineCount: normalized && Array.isArray(normalized.lines) ? normalized.lines.length : 0,
      item: getItem(item, normalized),
      rows: getRows(normalized, item)
    };
  }

  function getAttunementPotentialComparison(item, choice, options) {
    const settings = options || {};
    if (!item || !choice) return null;
    const getComparisonSide = typeof settings.getPotentialComparisonSide === 'function'
      ? settings.getPotentialComparisonSide
      : (potential, targetItem) => getPotentialComparisonSide(potential, targetItem, settings);
    const getLineIndex = typeof settings.getPotentialComparisonLineIndex === 'function'
      ? settings.getPotentialComparisonLineIndex
      : getPotentialComparisonLineIndexFallback;
    const getDeltaClass = typeof settings.getPotentialComparisonDeltaClass === 'function'
      ? settings.getPotentialComparisonDeltaClass
      : (delta) => Number(delta || 0) > 0 ? 'is-gain' : Number(delta || 0) < 0 ? 'is-loss' : 'is-same';
    const getStatAbbreviation = typeof settings.getPotentialStatAbbreviation === 'function'
      ? settings.getPotentialStatAbbreviation
      : (stat) => String(stat || '').slice(0, 2).toUpperCase();
    const getStrength = typeof settings.getItemStrength === 'function'
      ? settings.getItemStrength
      : () => 0;
    const getItem = typeof settings.getItemWithPotential === 'function'
      ? settings.getItemWithPotential
      : (targetItem, targetPotential) => getItemWithPotential(targetItem, targetPotential, settings);
    const getTierRank = typeof settings.getPotentialTierRank === 'function'
      ? settings.getPotentialTierRank
      : () => -1;
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const previous = getComparisonSide(choice.originalPotential, item);
    const next = getComparisonSide(choice.newPotential, item);
    const keys = Array.from(new Set(Object.keys(previous.rows).concat(Object.keys(next.rows))))
      .sort((a, b) => {
        const rowA = previous.rows[a] || next.rows[a] || {};
        const rowB = previous.rows[b] || next.rows[b] || {};
        const indexA = getLineIndex(rowA);
        const indexB = getLineIndex(rowB);
        return indexA - indexB || a.localeCompare(b);
      });
    const rows = keys.map((key) => {
      const previousRow = previous.rows[key] || null;
      const nextRow = next.rows[key] || null;
      const previousStat = previousRow && previousRow.stat || '';
      const nextStat = nextRow && nextRow.stat || previousStat;
      const displayStat = nextStat || previousStat || key;
      const statChanged = !!(previousRow && nextRow && previousStat !== nextStat);
      const previousValue = previousRow ? Number(previousRow.value || 0) : 0;
      const nextValue = nextRow ? Number(nextRow.value || 0) : 0;
      const delta = statChanged ? 0 : nextValue - previousValue;
      const deltaClass = statChanged || (!previousRow && nextRow) ? 'is-gain' : getDeltaClass(delta);
      return {
        key,
        lineIndex: Math.min(
          getLineIndex(previousRow),
          getLineIndex(nextRow)
        ),
        stat: displayStat,
        previousStat,
        nextStat,
        statChanged,
        label: statChanged && previousStat && nextStat ? `${formatStatName(previousStat)} -> ${formatStatName(nextStat)}` : formatStatName(displayStat),
        icon: getStatAbbreviation(displayStat),
        previous: previousRow,
        next: nextRow,
        previousValue,
        nextValue,
        delta,
        deltaClass,
        verdict: statChanged ? 'CHANGED' : delta > 0 ? 'GAIN' : delta < 0 ? 'LOSS' : 'SAME',
        verdictSymbol: statChanged ? '~' : delta > 0 ? '+' : delta < 0 ? '-' : '='
      };
    });
    const previousPower = previous.item ? getStrength(previous.item) : getStrength(getItem(item, null));
    const nextPower = next.item ? getStrength(next.item) : getStrength(getItem(item, null));
    const tierDelta = getTierRank(next.tier && next.tier.id) - getTierRank(previous.tier && previous.tier.id);
    const changedCount = rows.filter((row) => row.statChanged || row.delta !== 0 || !!row.previous !== !!row.next).length;
    return {
      previous,
      next,
      rows,
      previousPower,
      nextPower,
      powerDelta: nextPower - previousPower,
      tierDelta,
      changedCount,
      totalRows: Math.max(rows.length, previous.lineCount, next.lineCount)
    };
  }

  function getEchoPotentialComparison(item, pendingChoice, options) {
    const getComparison = options && typeof options.getAttunementPotentialComparison === 'function'
      ? options.getAttunementPotentialComparison
      : (targetItem, choice) => getAttunementPotentialComparison(targetItem, choice, options);
    return getComparison(item, pendingChoice);
  }

  function renderAttunementStatIconMarkup(stat, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const getIconMeta = typeof settings.getAttunementStatIconMeta === 'function'
      ? settings.getAttunementStatIconMeta
      : () => ({ color: '#8a5d27', bg: '#fff0cf', svg: '' });
    const getTooltip = typeof settings.getAttunementStatTooltip === 'function'
      ? settings.getAttunementStatTooltip
      : () => '';
    if (!stat) {
      return '<span class="project-starfall-echo-stat-icon project-starfall-attunement-stat-icon is-empty" aria-hidden="true">-</span>';
    }
    const meta = getIconMeta(stat);
    const tooltip = getTooltip(stat, settings.tooltipOptions || settings);
    const statClass = String(stat || 'stat').replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase();
    const style = `--attunement-color:${meta.color};--attunement-bg:${meta.bg};`;
    return `
      <span class="project-starfall-echo-stat-icon project-starfall-attunement-stat-icon is-${escapeHtml(statClass)}" tabindex="0" title="${escapeHtml(tooltip)}" aria-label="${escapeHtml(tooltip)}" data-starfall-attunement-tooltip="${escapeHtml(tooltip)}" style="${escapeHtml(style)}">
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">${meta.svg}</svg>
      </span>
    `;
  }

  function renderAttunementPrismIcon(mode, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const getConsumableItem = typeof settings.getConsumableItem === 'function'
      ? settings.getConsumableItem
      : () => null;
    const getItemAsset = typeof settings.getItemAsset === 'function'
      ? settings.getItemAsset
      : () => '';
    const renderAssetImage = typeof settings.renderAssetImage === 'function'
      ? settings.renderAssetImage
      : () => '';
    const isEcho = mode === 'echo';
    const isLineCatalyst = mode === 'line_catalyst';
    const prism = getConsumableItem(isLineCatalyst ? 'line_catalyst' : isEcho ? 'preservation_cube' : 'potential_cube');
    const asset = prism ? getItemAsset(prism) : '';
    const label = prism && prism.icon || (isLineCatalyst ? 'LC' : isEcho ? 'ECHO' : 'PRS');
    return `
      <span class="project-starfall-echo-crystal ${isLineCatalyst ? 'is-line-catalyst' : ''} ${asset ? 'has-image' : ''}" aria-hidden="true">
        ${asset ? renderAssetImage(asset, '', 'project-starfall-item-art') : escapeHtml(label)}
      </span>
    `;
  }

  function renderEchoRankMarkup(rowData, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const getPotentialRollRank = typeof settings.getPotentialRollRank === 'function'
      ? settings.getPotentialRollRank
      : () => null;
    if (!rowData) {
      return '<small class="project-starfall-potential-rank project-starfall-potential-rank-empty"><span>-</span></small>';
    }
    const rank = rowData.rank || getPotentialRollRank(rowData.percentileLabel);
    return `
      <small class="project-starfall-potential-rank ${escapeHtml(rank.className)}">
        <span>${escapeHtml(rank.label)}</span>
      </small>
    `;
  }

  function renderEchoVerdictRows(comparison, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    return (comparison && comparison.rows || []).map((row) => `
      <div class="project-starfall-echo-verdict-row ${escapeHtml(row.deltaClass)}" aria-label="${escapeHtml(`${row.label} ${row.verdict}`)}">
        <strong>${escapeHtml(row.verdictSymbol)}</strong>
        <span>${escapeHtml(row.verdict)}</span>
      </div>
    `).join('');
  }

  function renderEchoComparisonRows(comparison, side, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const formatStatTotalValue = typeof settings.formatStatTotalValue === 'function'
      ? settings.formatStatTotalValue
      : (stat, value) => String(value);
    const getPotentialRollRank = typeof settings.getPotentialRollRank === 'function'
      ? settings.getPotentialRollRank
      : () => null;
    const renderStatIconMarkup = typeof settings.renderAttunementStatIconMarkup === 'function'
      ? settings.renderAttunementStatIconMarkup
      : (stat, iconOptions) => renderAttunementStatIconMarkup(stat, Object.assign({}, settings, iconOptions || {}));
    const renderRankMarkup = typeof settings.renderEchoRankMarkup === 'function'
      ? settings.renderEchoRankMarkup
      : (rowData) => renderEchoRankMarkup(rowData, settings);
    const isNext = side === 'next';
    return comparison.rows.map((row) => {
      const rowData = isNext ? row.next : row.previous;
      const label = rowData ? formatStatName(rowData.stat) : 'No line';
      const rank = rowData ? rowData.rank || getPotentialRollRank(rowData.percentileLabel) : null;
      const value = rowData ? formatStatTotalValue(rowData.stat, rowData.value) : '-';
      const previousText = row.previous
        ? `${formatStatName(row.previous.stat)} ${formatStatTotalValue(row.previous.stat, row.previous.value)}`
        : 'No line';
      const nextText = row.next
        ? `${formatStatName(row.next.stat)} ${formatStatTotalValue(row.next.stat, row.next.value)}`
        : 'No line';
      const title = [
        `${isNext ? 'New' : 'Previous'}: ${isNext ? nextText : previousText}`,
        `Previous: ${previousText}`,
        `New: ${nextText}`
      ].join('. ');
      return `
        <div class="project-starfall-echo-stat-row ${escapeHtml(row.deltaClass)}" data-starfall-potential-line-index="${escapeHtml(row.lineIndex)}" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}">
          ${renderStatIconMarkup(rowData && rowData.stat, rowData ? { value: rowData.value, rangeLabel: rowData.rangeLabel, rankLabel: rank && rank.label } : null)}
          <b>${escapeHtml(label)}</b>
          <strong>${escapeHtml(value)}</strong>
          ${renderRankMarkup(rowData)}
        </div>
      `;
    }).join('');
  }

  function renderAttunementCurrentRows(item, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const getRows = typeof settings.getPotentialLineRollRows === 'function'
      ? settings.getPotentialLineRollRows
      : (targetItem) => getPotentialLineRollRows(targetItem, settings);
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (stat, value) => String(value);
    const getPotentialRollRank = typeof settings.getPotentialRollRank === 'function'
      ? settings.getPotentialRollRank
      : () => null;
    const renderStatIconMarkup = typeof settings.renderAttunementStatIconMarkup === 'function'
      ? settings.renderAttunementStatIconMarkup
      : (stat, iconOptions) => renderAttunementStatIconMarkup(stat, Object.assign({}, settings, iconOptions || {}));
    const renderRankMarkup = typeof settings.renderEchoRankMarkup === 'function'
      ? settings.renderEchoRankMarkup
      : (rowData) => renderEchoRankMarkup(rowData, settings);
    const rows = getRows(item);
    if (!rows.length) {
      return `
        <div class="project-starfall-echo-stat-row is-same project-starfall-attunement-empty-row">
          ${renderStatIconMarkup('')}
          <b>No attunement stats</b>
          <strong>-</strong>
          ${renderRankMarkup(null)}
        </div>
      `;
    }
    return rows.map((row) => {
      const rank = getPotentialRollRank(row.percentileLabel);
      const title = `${formatStatName(row.stat)} ${formatStatValue(row.stat, row.value)}. Range: ${row.rangeLabel}. Roll rank: ${rank.label}.`;
      return `
        <div class="project-starfall-echo-stat-row project-starfall-potential-line-row is-same" data-starfall-potential-line-index="${escapeHtml(row.lineIndex)}" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}">
          ${renderStatIconMarkup(row.stat, { value: row.value, rangeLabel: row.rangeLabel, rankLabel: rank.label })}
          <b>${escapeHtml(formatStatName(row.stat))}</b>
          <strong>${escapeHtml(formatStatValue(row.stat, row.value))}</strong>
          <em>${escapeHtml(rank.percentLabel || '--')}</em>
          ${renderRankMarkup(row)}
        </div>
      `;
    }).join('');
  }

  function renderLineCatalystDetails(item, lineInfo, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const getRows = typeof settings.getPotentialLineRollRows === 'function'
      ? settings.getPotentialLineRollRows
      : (targetItem) => getPotentialLineRollRows(targetItem, settings);
    const getTierLabel = typeof settings.getPotentialTierLabel === 'function'
      ? settings.getPotentialTierLabel
      : (targetItem) => getPotentialTierLabel(targetItem, settings);
    const getSlotMeta = typeof settings.getSlotMeta === 'function'
      ? settings.getSlotMeta
      : () => ({ icon: '', label: '' });
    const getClassRequirementLabel = typeof settings.getItemClassRequirementLabel === 'function'
      ? settings.getItemClassRequirementLabel
      : () => '';
    const renderGearIconMarkup = typeof settings.renderGearIconMarkup === 'function'
      ? settings.renderGearIconMarkup
      : () => '';
    const renderCurrentRows = typeof settings.renderAttunementCurrentRows === 'function'
      ? settings.renderAttunementCurrentRows
      : (targetItem) => renderAttunementCurrentRows(targetItem, settings);
    const renderStatIconMarkup = typeof settings.renderAttunementStatIconMarkup === 'function'
      ? settings.renderAttunementStatIconMarkup
      : (stat, iconOptions) => renderAttunementStatIconMarkup(stat, Object.assign({}, settings, iconOptions || {}));
    const renderRankMarkup = typeof settings.renderEchoRankMarkup === 'function'
      ? settings.renderEchoRankMarkup
      : (rowData) => renderEchoRankMarkup(rowData, settings);
    const formatAbbreviatedInteger = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (value) => String(value);
    const formatChance = typeof settings.formatChance === 'function'
      ? settings.formatChance
      : (value) => `${value}%`;
    const rows = getRows(item);
    const hasPotential = !!(lineInfo && lineInfo.hasPotential);
    const isMaxed = hasPotential && lineInfo.lineCount >= lineInfo.maxLineCount;
    const successChance = hasPotential && !isMaxed ? Math.max(0, Number(lineInfo.chance || 0)) : 0;
    const failChance = hasPotential && !isMaxed ? Math.max(0, 1 - successChance) : 0;
    const lineCountLabel = hasPotential ? `${lineInfo.lineCount} / ${lineInfo.maxLineCount}` : '0 / 5';
    const nextLineLabel = hasPotential
      ? isMaxed ? 'Maxed' : `${lineInfo.targetLineCount} / ${lineInfo.maxLineCount}`
      : 'Attune first';
    const outcomeNote = hasPotential
      ? isMaxed
        ? 'This gear already has the maximum number of active attunement lines.'
        : lineInfo.lineCount > 1
          ? 'If the attempt fails, no new line is added and one active line is hidden.'
          : 'If the attempt fails, no new line is added.'
      : 'Use an Attunement Prism before applying Line Catalysts.';
    return `
      <div class="project-starfall-line-catalyst-layout" aria-label="Line Catalyst details">
        <section class="project-starfall-line-catalyst-card project-starfall-line-catalyst-gear" data-starfall-potential-drop-zone>
          <header>
            <strong>Ready For Line Upgrade</strong>
            <small>${escapeHtml(item ? getTierLabel(item) : 'Select gear')}</small>
          </header>
          ${item ? `
            <div class="project-starfall-line-catalyst-gear-row">
              ${renderGearIconMarkup(item, getSlotMeta(item.slot).icon)}
              <span>
                <strong>${escapeHtml(item.rarity || 'Common')} ${escapeHtml(item.name || 'Gear')} +${item.upgrade || 0}</strong>
                <small>${escapeHtml(getSlotMeta(item.slot).label)} - ${escapeHtml(getClassRequirementLabel(item))}</small>
              </span>
            </div>
          ` : `
            <div class="project-starfall-line-catalyst-gear-row">
              <span class="project-starfall-gear-icon" aria-hidden="true">GEAR</span>
              <span>
                <strong>Drop gear here</strong>
                <small>Drag inventory or equipped gear here, or click to choose.</small>
              </span>
            </div>
          `}
        </section>
        <section class="project-starfall-line-catalyst-card project-starfall-line-catalyst-counts">
          <header>
            <strong>Line Count</strong>
            <small>${escapeHtml(nextLineLabel)}</small>
          </header>
          <div class="project-starfall-line-catalyst-metrics">
            <span>
              <small>Current Lines</small>
              <strong>${escapeHtml(lineCountLabel)}</strong>
            </span>
            <span>
              <small>Next Target</small>
              <strong>${escapeHtml(nextLineLabel)}</strong>
            </span>
          </div>
        </section>
        <section class="project-starfall-line-catalyst-card project-starfall-line-catalyst-current">
          <header>
            <strong>Current Lines</strong>
            <small>${escapeHtml(lineCountLabel)}</small>
          </header>
          <div class="project-starfall-line-catalyst-lines">
            ${rows.length ? renderCurrentRows(item) : `
              <div class="project-starfall-echo-stat-row is-same project-starfall-attunement-empty-row">
                ${renderStatIconMarkup('')}
                <b>No attunement stats</b>
                <strong>-</strong>
                ${renderRankMarkup(null)}
              </div>
            `}
          </div>
        </section>
        <section class="project-starfall-line-catalyst-card project-starfall-line-catalyst-odds-card">
          <header>
            <strong>Attempt Odds</strong>
            <small>${escapeHtml(formatAbbreviatedInteger(lineInfo && lineInfo.catalystCount || 0))} owned</small>
          </header>
          <div class="project-starfall-line-catalyst-odds">
            <span class="is-gain">
              <small>New Line Added</small>
              <strong>${escapeHtml(formatChance(successChance * 100))}</strong>
            </span>
            <span class="is-loss">
              <small>Not Added</small>
              <strong>${escapeHtml(formatChance(failChance * 100))}</strong>
            </span>
          </div>
          <p>${escapeHtml(outcomeNote)}</p>
        </section>
      </div>
    `;
  }

  function renderAttunementTargetPanel(item, mode, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const getSlotMeta = typeof settings.getSlotMeta === 'function'
      ? settings.getSlotMeta
      : () => ({ icon: '', label: '' });
    const getClassRequirementLabel = typeof settings.getItemClassRequirementLabel === 'function'
      ? settings.getItemClassRequirementLabel
      : () => '';
    const getTierLabel = typeof settings.getPotentialTierLabel === 'function'
      ? settings.getPotentialTierLabel
      : (targetItem) => getPotentialTierLabel(targetItem, settings);
    const renderGearIconMarkup = typeof settings.renderGearIconMarkup === 'function'
      ? settings.renderGearIconMarkup
      : () => '';
    const isEcho = mode === 'echo';
    const isLineCatalyst = mode === 'line_catalyst';
    const modeLabel = isLineCatalyst ? 'Line Catalyst' : isEcho ? 'Echo Prism' : 'Attunement Prism';
    const readyTitle = item
      ? isLineCatalyst ? 'Ready For Line Upgrade' : 'Ready For Next Roll'
      : 'Select Gear';
    const emptyTitle = isLineCatalyst ? 'No line attempt yet' : 'No new result yet';
    const emptyText = item
      ? isLineCatalyst ? 'Use a Line Catalyst to unlock another active line.' : `Use a ${modeLabel} to roll attunement stats.`
      : 'Choose a gear item to begin.';
    return `
      <section class="project-starfall-echo-stat-table project-starfall-echo-stat-table-next project-starfall-attunement-target-table" aria-label="${escapeHtml(modeLabel)} target">
        <header>
          <strong>${escapeHtml(readyTitle)}</strong>
          <small>${escapeHtml(modeLabel)}</small>
        </header>
        <div class="project-starfall-attunement-target-zone ${item ? 'has-item' : ''}" data-starfall-potential-drop-zone>
          ${item ? `
            ${renderGearIconMarkup(item, getSlotMeta(item.slot).icon)}
            <span>
              <strong>${escapeHtml(item.rarity || 'Common')} ${escapeHtml(item.name || 'Gear')} +${item.upgrade || 0}</strong>
              <small>${escapeHtml(getSlotMeta(item.slot).label)} - ${escapeHtml(getClassRequirementLabel(item))} - ${escapeHtml(getTierLabel(item))}</small>
            </span>
          ` : `
            <span class="project-starfall-gear-icon" aria-hidden="true">GEAR</span>
            <span>
              <strong>Drop gear here</strong>
              <small>Drag inventory or equipped gear here, or click to choose.</small>
            </span>
          `}
        </div>
        <div class="project-starfall-attunement-empty-result">
          <strong>${escapeHtml(emptyTitle)}</strong>
          <small>${escapeHtml(emptyText)}</small>
        </div>
      </section>
    `;
  }

  function renderPotentialAutoControls(item, mode, prismCount, hasPendingChoice, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const normalizeTarget = typeof settings.normalizePotentialAutoTarget === 'function'
      ? settings.normalizePotentialAutoTarget
      : () => ({ statGoals: [], tier: '', maxRolls: 0 });
    const getStatOptions = typeof settings.getPotentialAutoStatOptions === 'function'
      ? settings.getPotentialAutoStatOptions
      : () => [];
    const getTierOptions = typeof settings.getPotentialAutoTierOptions === 'function'
      ? settings.getPotentialAutoTierOptions
      : () => [];
    const hasGoals = typeof settings.potentialAutoTargetHasGoals === 'function'
      ? settings.potentialAutoTargetHasGoals
      : () => false;
    const getEstimate = typeof settings.getPotentialAutoEstimate === 'function'
      ? settings.getPotentialAutoEstimate
      : () => ({ chanceLabel: '-', estimateLabel: '-' });
    const getTargetLabel = typeof settings.getPotentialAutoTargetLabel === 'function'
      ? settings.getPotentialAutoTargetLabel
      : () => '';
    const getStatGoalRange = typeof settings.getPotentialAutoStatGoalRange === 'function'
      ? settings.getPotentialAutoStatGoalRange
      : () => ({ min: 0, max: 0 });
    const maxPrisms = Math.max(0, Math.floor(Number(prismCount || 0) || 0));
    const target = normalizeTarget(item, maxPrisms) || { statGoals: [], tier: '', maxRolls: 0 };
    const statOptions = getStatOptions(item) || [];
    const tierOptions = getTierOptions(item) || [];
    const selectedStats = new Set((target.statGoals || []).map((goal) => goal.stat));
    const canAddRequest = statOptions.some((option) => !selectedStats.has(option.id));
    const hasTarget = hasGoals(target);
    const disabled = !item || hasPendingChoice || maxPrisms <= 0 || !hasTarget || target.maxRolls <= 0;
    const repeatDisabled = !item || hasPendingChoice || maxPrisms <= 0 || !settings.lastPotentialAutoTarget;
    const isEcho = mode === 'echo';
    const launcherLabel = isEcho ? 'Auto Echo' : 'Auto Attunement';
    const runLabel = isEcho ? 'Start Auto Echo' : 'Start Auto Attunement';
    const estimate = getEstimate(item, target) || { chanceLabel: '-', estimateLabel: '-' };
    const summary = target.lastSummary || (hasTarget
      ? `Stops when all goals match: ${getTargetLabel(target, item)}.`
      : 'Select one or more stat thresholds, a higher tier, or both.');
    const panelOpen = !!settings.potentialAutoPanelOpen;
    return `
      <div class="project-starfall-attunement-auto-shell ${panelOpen ? 'is-open' : ''}" aria-label="Auto attunement">
        <div class="project-starfall-attunement-auto-launcher">
          <button type="button" class="project-starfall-attunement-auto-toggle" data-starfall-potential-auto-toggle aria-expanded="${panelOpen ? 'true' : 'false'}">
            ${escapeHtml(launcherLabel)}
          </button>
          <button type="button" class="project-starfall-attunement-auto-repeat" data-starfall-potential-auto-repeat ${repeatDisabled ? 'disabled' : ''}>Repeat Last</button>
        </div>
        ${panelOpen ? `
          <div class="project-starfall-attunement-auto" role="group" aria-label="Auto attunement goals">
            <header>
              <strong>Auto Goals</strong>
              <button type="button" data-starfall-potential-auto-close aria-label="Close auto attunement goals">X</button>
            </header>
            <div class="project-starfall-attunement-auto-requests" aria-label="Stat threshold requests">
              <div class="project-starfall-attunement-auto-request-list">
                ${(target.statGoals || []).length ? target.statGoals.map((goal, index) => {
                  const range = getStatGoalRange(goal.stat, item, target);
                  return `
                    <div class="project-starfall-attunement-auto-request">
                      <label>
                        <span>Requested Stat</span>
                        <select data-starfall-potential-auto-stat-row="${index}">
                          ${statOptions.map((option) => `<option value="${escapeHtml(option.id)}" ${option.id === goal.stat ? 'selected' : ''} ${selectedStats.has(option.id) && option.id !== goal.stat ? 'disabled' : ''}>${escapeHtml(option.label)}</option>`).join('')}
                        </select>
                      </label>
                      <label>
                        <span>Threshold</span>
                        <input type="number" min="${range.min}" max="${range.max}" value="${goal.min}" inputmode="numeric" data-starfall-potential-auto-stat-min-index="${index}">
                      </label>
                      <button type="button" data-starfall-potential-auto-stat-remove="${index}" aria-label="Remove stat request">Remove</button>
                    </div>
                  `;
                }).join('') : '<div class="project-starfall-attunement-auto-request-empty">No stat requests yet.</div>'}
              </div>
              <button type="button" data-starfall-potential-auto-stat-add ${canAddRequest ? '' : 'disabled'}>Add Request</button>
            </div>
            <label class="project-starfall-attunement-auto-tier">
              <span>Minimum Tier</span>
              <select data-starfall-potential-auto-tier>
                <option value="">Choose tier...</option>
                ${tierOptions.map((tier) => `<option value="${escapeHtml(tier.id)}" ${target.tier === tier.id ? 'selected' : ''}>${escapeHtml(tier.name || tier.id)}+</option>`).join('')}
                ${tierOptions.length ? '' : '<option value="" disabled>No higher tier</option>'}
              </select>
            </label>
            <label class="project-starfall-attunement-auto-max">
              <span>Prisms To Use</span>
              <span class="project-starfall-attunement-auto-slider">
                <input type="range" min="0" max="${maxPrisms}" step="1" value="${target.maxRolls}" data-starfall-potential-auto-max aria-label="Prisms to use" ${maxPrisms <= 0 ? 'disabled' : ''}>
                <strong>${target.maxRolls} / ${maxPrisms}</strong>
              </span>
              <small>Locks to 25-prism intervals; drag to the end to use all.</small>
            </label>
            <div class="project-starfall-attunement-auto-estimate" aria-label="Auto attunement odds">
              <span><small>Chance</small><strong>${escapeHtml(estimate.chanceLabel)}</strong></span>
              <span><small>Estimate</small><strong>${escapeHtml(estimate.estimateLabel)}</strong></span>
            </div>
            <button type="button" data-starfall-potential-auto-run ${disabled ? 'disabled' : ''}>${escapeHtml(runLabel)}</button>
            <small>${escapeHtml(summary)}</small>
          </div>
        ` : ''}
      </div>
    `;
  }

  function renderAttunementEmptyPrompt(mode, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const getPromptPlacementClass = typeof settings.getPotentialPromptPlacementClass === 'function'
      ? settings.getPotentialPromptPlacementClass
      : () => '';
    const getPromptDomStyle = typeof settings.getPotentialPromptDomStyle === 'function'
      ? settings.getPotentialPromptDomStyle
      : () => '';
    const renderPrismIcon = typeof settings.renderAttunementPrismIcon === 'function'
      ? settings.renderAttunementPrismIcon
      : (targetMode) => renderAttunementPrismIcon(targetMode, settings);
    const isRegular = mode === 'regular';
    const isLineCatalyst = mode === 'line_catalyst';
    const title = isLineCatalyst ? 'Line Catalyst' : isRegular ? 'Attunement Prism' : 'Echo Prism Attunement';
    const sectionLabel = isLineCatalyst ? 'Line Catalyst item selection' : isRegular ? 'Attunement Prism item selection' : 'Echo Prism item selection';
    const actionLabel = isLineCatalyst ? 'Use Line Catalyst' : isRegular ? 'Use Attunement Prism' : 'Use Echo Prism';
    const actionAttr = isLineCatalyst ? 'data-starfall-potential-line-upgrade=""' : isRegular ? 'data-starfall-potential-confirm=""' : 'data-starfall-potential-preserve=""';
    const promptHint = isLineCatalyst ? 'Select attuned gear from inventory or equipment' : 'Select gear from inventory or equipment';
    const waitingText = isLineCatalyst ? 'Select attuned gear before using this item.' : 'Select gear before using this item.';
    const cardClass = `project-starfall-upgrade-confirm-card project-starfall-echo-choice-card project-starfall-attunement-choice-card project-starfall-attunement-empty-card${getPromptPlacementClass()}`;
    return `
      <section class="project-starfall-upgrade-confirm project-starfall-potential-confirm project-starfall-attunement-choice project-starfall-echo-choice project-starfall-attunement-empty-prompt ${isLineCatalyst ? 'is-line-catalyst' : isRegular ? 'is-regular' : 'is-echo'}" role="dialog" aria-modal="true" aria-label="${escapeHtml(sectionLabel)}">
        <div class="${cardClass}"${getPromptDomStyle()}>
          <button type="button" class="project-starfall-upgrade-confirm-close project-starfall-echo-help" data-starfall-potential-help aria-label="Show attunement odds">?</button>
          <button type="button" class="project-starfall-upgrade-confirm-close project-starfall-echo-close" data-starfall-potential-close aria-label="Close attunement selection">X</button>
          <header class="project-starfall-echo-choice-title" data-starfall-potential-drag>
            ${renderPrismIcon(isLineCatalyst ? 'line_catalyst' : isRegular ? 'regular' : 'echo')}
            <span>
              <strong>${escapeHtml(title)}</strong>
              <small>${escapeHtml(promptHint)}</small>
            </span>
          </header>
          <div class="project-starfall-attunement-drop-only" data-starfall-potential-drop-zone>
            <span class="project-starfall-gear-icon" aria-hidden="true">GEAR</span>
            <span>
              <strong>Drop gear here</strong>
              <small>Drag inventory or equipped gear here, or click to choose.</small>
            </span>
          </div>
          <div class="project-starfall-upgrade-actions project-starfall-echo-actions project-starfall-attunement-result-actions">
            <span>${escapeHtml(waitingText)}</span>
            <button type="button" class="project-starfall-echo-accept" ${actionAttr} disabled>${escapeHtml(actionLabel)} (0)</button>
          </div>
        </div>
      </section>
    `;
  }

  function renderAttunementComparisonPrompt(item, choice, options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const formatAbbreviatedInteger = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (value) => String(value);
    const formatChance = typeof settings.formatChance === 'function'
      ? settings.formatChance
      : (value) => `${value}%`;
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (stat, value) => String(value);
    const getComparison = typeof settings.getAttunementPotentialComparison === 'function'
      ? settings.getAttunementPotentialComparison
      : (targetItem, targetChoice) => getAttunementPotentialComparison(targetItem, targetChoice, settings);
    const getDeltaClass = typeof settings.getPotentialComparisonDeltaClass === 'function'
      ? settings.getPotentialComparisonDeltaClass
      : (delta) => Number(delta || 0) > 0 ? 'is-gain' : Number(delta || 0) < 0 ? 'is-loss' : 'is-same';
    const getAutoPrismCount = typeof settings.getPotentialAutoPrismCount === 'function'
      ? settings.getPotentialAutoPrismCount
      : () => 0;
    const getSlotMeta = typeof settings.getSlotMeta === 'function'
      ? settings.getSlotMeta
      : () => ({ label: '' });
    const getClassRequirementLabel = typeof settings.getItemClassRequirementLabel === 'function'
      ? settings.getItemClassRequirementLabel
      : () => '';
    const getTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    const getStrength = typeof settings.getItemStrength === 'function'
      ? settings.getItemStrength
      : () => 0;
    const getLineUpgradeInfo = typeof settings.getPotentialLineUpgradeInfo === 'function'
      ? settings.getPotentialLineUpgradeInfo
      : (targetItem, snapshot) => getPotentialLineUpgradeInfo(targetItem, snapshot, settings);
    const getTierAdvanceLabel = typeof settings.getPotentialTierAdvanceLabel === 'function'
      ? settings.getPotentialTierAdvanceLabel
      : (targetItem) => getPotentialTierAdvanceLabel(targetItem, settings);
    const getPromptPlacementClass = typeof settings.getPotentialPromptPlacementClass === 'function'
      ? settings.getPotentialPromptPlacementClass
      : () => '';
    const getPromptDomStyle = typeof settings.getPotentialPromptDomStyle === 'function'
      ? settings.getPotentialPromptDomStyle
      : () => '';
    const renderPrismIcon = typeof settings.renderAttunementPrismIcon === 'function'
      ? settings.renderAttunementPrismIcon
      : (targetMode) => renderAttunementPrismIcon(targetMode, settings);
    const renderEmptyPrompt = typeof settings.renderAttunementEmptyPrompt === 'function'
      ? settings.renderAttunementEmptyPrompt
      : (targetMode) => renderAttunementEmptyPrompt(targetMode, settings);
    const renderLineDetails = typeof settings.renderLineCatalystDetails === 'function'
      ? settings.renderLineCatalystDetails
      : (targetItem, lineInfo) => renderLineCatalystDetails(targetItem, lineInfo, settings);
    const renderComparisonRows = typeof settings.renderEchoComparisonRows === 'function'
      ? settings.renderEchoComparisonRows
      : (comparisonData, side) => renderEchoComparisonRows(comparisonData, side, settings);
    const renderCurrentRows = typeof settings.renderAttunementCurrentRows === 'function'
      ? settings.renderAttunementCurrentRows
      : (targetItem) => renderAttunementCurrentRows(targetItem, settings);
    const renderVerdictRows = typeof settings.renderEchoVerdictRows === 'function'
      ? settings.renderEchoVerdictRows
      : (comparisonData) => renderEchoVerdictRows(comparisonData, settings);
    const renderTargetPanel = typeof settings.renderAttunementTargetPanel === 'function'
      ? settings.renderAttunementTargetPanel
      : (targetItem, targetMode) => renderAttunementTargetPanel(targetItem, targetMode, settings);
    const renderAutoControls = typeof settings.renderPotentialAutoControls === 'function'
      ? settings.renderPotentialAutoControls
      : (targetItem, targetMode, prismCount, hasPendingChoice) => renderPotentialAutoControls(targetItem, targetMode, prismCount, hasPendingChoice, settings);
    const snapshot = settings.snapshot || {};
    const state = snapshot.state || {};
    const mode = settings.mode === 'line_catalyst' ? 'line_catalyst' : settings.mode === 'regular' ? 'regular' : 'echo';
    const comparison = item && choice ? settings.comparison || getComparison(item, choice) : null;
    if (!item && !comparison) return renderEmptyPrompt(mode);
    const shortTier = (tier) => tier ? String(tier.name || '').replace(' Attunement', '') : 'None';
    const powerClass = comparison ? getDeltaClass(comparison.powerDelta) : 'is-same';
    const tierClass = comparison ? getDeltaClass(comparison.tierDelta) : 'is-same';
    const changedClass = comparison && comparison.changedCount > 0 ? 'is-gain' : 'is-same';
    const prismCount = getAutoPrismCount('regular');
    const echoPrismCount = getAutoPrismCount('echo');
    const isRegular = mode === 'regular';
    const isLineCatalyst = mode === 'line_catalyst';
    const title = isRegular
      ? comparison ? 'Attunement Prism Result' : 'Attunement Prism'
      : isLineCatalyst ? 'Line Catalyst' : 'Echo Prism Attunement';
    const nextTitle = isLineCatalyst ? 'Line Catalyst Target' : isRegular ? 'Applied New Stats' : 'Optional New Stats (Echo)';
    const sectionLabel = isRegular ? 'Attunement Prism result screen' : isLineCatalyst ? 'Line Catalyst attunement screen' : 'Echo Prism attunement screen';
    const itemMeta = item
      ? `${getSlotMeta(item.slot).label} - ${getClassRequirementLabel(item)} - ${item.rarity || 'Common'} ${item.name} +${item.upgrade || 0}`
      : 'Select gear from inventory or equipment';
    const currentTier = item ? getTierDefinition(item.potential && item.potential.tier) : null;
    const tierSummaryTier = comparison ? isRegular ? comparison.next.tier : comparison.previous.tier : currentTier;
    const tierSummaryClass = comparison && isRegular ? tierClass : 'is-same';
    const tierSummaryDelta = comparison && isRegular
      ? comparison.tierDelta > 0 ? `+${comparison.tierDelta}` : comparison.tierDelta < 0 ? String(comparison.tierDelta) : '-'
      : '-';
    const currentPower = item ? getStrength(item) : 0;
    const lineInfo = Object.assign({
      catalystCount: 0,
      canUpgrade: false,
      chance: 0,
      hasPotential: false,
      lineCount: 0,
      maxLineCount: POTENTIAL_MAX_LINE_COUNT,
      targetLineCount: 1
    }, getLineUpgradeInfo(item, snapshot) || {});
    const actionCount = isLineCatalyst ? lineInfo.catalystCount : isRegular ? prismCount : echoPrismCount;
    const shardCount = Math.max(0, Math.floor(Number(state.materials && state.materials.cubeFragment || 0) || 0));
    const shardHint = isLineCatalyst
      ? `${formatAbbreviatedInteger(shardCount)} Prism Shards; 20 -> Line Catalyst`
      : isRegular
        ? `${formatAbbreviatedInteger(shardCount)} Prism Shards; 10 -> Attunement Prism`
        : `${formatAbbreviatedInteger(shardCount)} Prism Shards; 25 -> Echo Prism`;
    const hasPendingChoice = !isRegular && !isLineCatalyst && !!comparison;
    const hasAnyPendingChoice = !!(state && state.pendingPotentialChoice);
    const lineChanceLabel = lineInfo.chance > 0 ? formatChance(lineInfo.chance * 100) : '-';
    const lineDisabled = !item || hasAnyPendingChoice || !lineInfo.canUpgrade;
    const lineSummaryValue = !item
      ? '-'
      : !lineInfo.hasPotential
        ? 'Attune first'
        : lineInfo.lineCount >= lineInfo.maxLineCount
          ? `${lineInfo.lineCount}/${lineInfo.maxLineCount}`
          : `${lineInfo.lineCount}/${lineInfo.maxLineCount} -> ${lineInfo.targetLineCount}/${lineInfo.maxLineCount}`;
    const lineSummaryStatus = !item
      ? 'Waiting'
      : !lineInfo.hasPotential
        ? 'Needs attunement'
        : lineInfo.lineCount >= lineInfo.maxLineCount
          ? 'Maxed'
          : lineInfo.canUpgrade ? `${lineChanceLabel}` : 'No Catalyst';
    const lineStatus = !item
      ? 'Select gear to upgrade lines.'
      : !lineInfo.hasPotential
        ? 'Attune gear before upgrading lines.'
        : lineInfo.lineCount >= lineInfo.maxLineCount
          ? `Lines ${lineInfo.lineCount}/${lineInfo.maxLineCount} maxed.`
          : `Lines ${lineInfo.lineCount}/${lineInfo.maxLineCount} -> ${lineInfo.targetLineCount}/${lineInfo.maxLineCount} at ${lineChanceLabel}; failed attempts ${lineInfo.lineCount > 1 ? 'hide one line' : 'do not add a line'}.`;
    const actionAttr = isLineCatalyst
      ? `data-starfall-potential-line-upgrade="${escapeHtml(item && item.uid || '')}"`
      : isRegular
        ? `data-starfall-potential-confirm="${escapeHtml(item && item.uid || '')}"`
        : `data-starfall-potential-preserve="${escapeHtml(item && item.uid || '')}"`;
    const actionDisabled = !item || hasPendingChoice || actionCount <= 0;
    const cardClass = `project-starfall-upgrade-confirm-card project-starfall-echo-choice-card project-starfall-attunement-choice-card${getPromptPlacementClass()}`;
    return `
      <section class="project-starfall-upgrade-confirm project-starfall-potential-confirm project-starfall-attunement-choice project-starfall-echo-choice ${isLineCatalyst ? 'is-line-catalyst' : isRegular ? 'is-regular' : 'is-echo'} ${comparison ? 'has-comparison' : 'has-ready-item'}" role="dialog" aria-modal="true" aria-label="${escapeHtml(sectionLabel)}">
        <div class="${cardClass}"${getPromptDomStyle()}>
          <button type="button" class="project-starfall-upgrade-confirm-close project-starfall-echo-help" data-starfall-potential-help aria-label="Show attunement odds">?</button>
          <button type="button" class="project-starfall-upgrade-confirm-close project-starfall-echo-close" data-starfall-potential-close aria-label="Close attunement comparison">X</button>
          <header class="project-starfall-echo-choice-title" data-starfall-potential-drag>
            ${renderPrismIcon(isLineCatalyst ? 'line_catalyst' : isRegular ? 'regular' : 'echo')}
            <span>
              <strong>${escapeHtml(title)}</strong>
              <small>${escapeHtml(itemMeta)}</small>
            </span>
          </header>
          <div class="project-starfall-echo-summary" aria-label="Attunement summary">
            <span>
              <small>Total Power</small>
              <strong>${comparison ? `${formatAbbreviatedInteger(comparison.previousPower)} <i aria-hidden="true">&raquo;</i> ${formatAbbreviatedInteger(comparison.nextPower)}` : formatAbbreviatedInteger(currentPower)}</strong>
              <em class="${escapeHtml(powerClass)}">${comparison ? escapeHtml(formatStatValue('', comparison.powerDelta)) : '-'}</em>
            </span>
            <span>
              <small>Tier</small>
              <strong>${escapeHtml(shortTier(tierSummaryTier))}</strong>
              <em class="${escapeHtml(tierSummaryClass)}">${escapeHtml(tierSummaryDelta)}</em>
            </span>
            <span>
              <small>${comparison ? 'Changed Lines' : isLineCatalyst ? 'Active Lines' : isRegular ? 'Use Attunement Prism' : 'Prisms'}</small>
              <strong>${comparison ? `${comparison.changedCount} / ${comparison.totalRows || 0}` : isLineCatalyst ? escapeHtml(lineSummaryValue) : `${actionCount} ready`}</strong>
              <em class="${escapeHtml(isLineCatalyst && !comparison && lineInfo.canUpgrade ? 'is-gain' : changedClass)}">${comparison ? comparison.changedCount ? 'Updated' : 'Same' : isLineCatalyst ? escapeHtml(lineSummaryStatus) : item ? 'Ready' : 'Waiting'}</em>
            </span>
          </div>
          ${isLineCatalyst && !comparison ? renderLineDetails(item, lineInfo) : `
          <div class="project-starfall-echo-comparison-grid ${comparison ? '' : 'is-empty'}">
            <section class="project-starfall-echo-stat-table project-starfall-echo-stat-table-previous" aria-label="Previous complete stats">
              <header>
                <strong>${comparison ? 'Previous Complete Stats' : 'Current attunement'}</strong>
                <small>${comparison ? escapeHtml(shortTier(comparison.previous.tier)) : escapeHtml(shortTier(currentTier))}</small>
              </header>
              <div class="project-starfall-echo-stat-list">
                ${comparison ? renderComparisonRows(comparison, 'previous') : renderCurrentRows(item)}
                <div class="project-starfall-echo-total-row">
                  <b>Total Power</b>
                  <strong>${formatAbbreviatedInteger(comparison ? comparison.previousPower : currentPower)}</strong>
                </div>
              </div>
            </section>
            <section class="project-starfall-echo-verdict-table" aria-label="Stat verdicts">
              <header><strong>${comparison ? 'Verdict' : 'Status'}</strong></header>
              ${comparison ? `
                <div class="project-starfall-echo-verdict-list">
                  ${renderVerdictRows(comparison)}
                  <div class="project-starfall-echo-verdict-row ${escapeHtml(powerClass)}">
                    <strong>${comparison.powerDelta > 0 ? '+' : comparison.powerDelta < 0 ? '-' : '='}</strong>
                    <span>POWER</span>
                  </div>
                </div>
              ` : `
                <div class="project-starfall-attunement-ready-state">
                  <strong>${item ? 'READY' : 'PICK'}</strong>
                  <span>${item ? isLineCatalyst ? 'Use a Line Catalyst to add an active line.' : 'Use a prism or configure auto target.' : 'Drop or choose gear.'}</span>
                </div>
              `}
            </section>
            ${comparison ? `
              <section class="project-starfall-echo-stat-table project-starfall-echo-stat-table-next" aria-label="${escapeHtml(nextTitle)}">
                <header>
                  <strong>${escapeHtml(nextTitle)}</strong>
                  <small>${escapeHtml(shortTier(comparison.next.tier))}</small>
                </header>
                <div class="project-starfall-echo-stat-list">
                  ${renderComparisonRows(comparison, 'next')}
                  <div class="project-starfall-echo-total-row ${escapeHtml(powerClass)}">
                    <b>Total Power</b>
                    <strong>${formatAbbreviatedInteger(comparison.nextPower)}</strong>
                  </div>
                </div>
              </section>
            ` : renderTargetPanel(item, isLineCatalyst ? 'line_catalyst' : isRegular ? 'regular' : 'echo')}
          </div>
          `}
          <footer class="project-starfall-attunement-footer">
            ${isLineCatalyst ? `<div class="project-starfall-upgrade-actions project-starfall-echo-actions project-starfall-attunement-line-actions">
              <span>${escapeHtml(lineStatus)} Line Catalysts: ${formatAbbreviatedInteger(lineInfo.catalystCount)}. ${escapeHtml(shardHint)}.</span>
              <button type="button" data-starfall-potential-line-upgrade="${escapeHtml(item && item.uid || '')}" ${lineDisabled ? 'disabled' : ''}>Use Line Catalyst</button>
            </div>` : ''}
            ${isLineCatalyst ? '' : renderAutoControls(item, isRegular ? 'regular' : 'echo', actionCount, hasPendingChoice)}
            ${isLineCatalyst ? '' : isRegular ? `
              <div class="project-starfall-upgrade-actions project-starfall-echo-actions project-starfall-attunement-result-actions">
                <span>${comparison ? 'New attunement is already applied.' : `Ready to use ${actionCount} Attunement Prism${actionCount === 1 ? '' : 's'}. ${getTierAdvanceLabel(item)} ${shardHint}.`}</span>
                ${comparison ? `
                  <button type="button" class="project-starfall-echo-accept" data-starfall-potential-repeat="${escapeHtml(item && item.uid || '')}" ${prismCount > 0 ? '' : 'disabled'}>Use Again (${prismCount})</button>
                ` : `
                  <button type="button" class="project-starfall-echo-accept" ${actionAttr} ${actionDisabled ? 'disabled' : ''}>Use Attunement Prism (${actionCount})</button>
                `}
              </div>
            ` : comparison ? `
              <div class="project-starfall-upgrade-actions project-starfall-echo-actions project-starfall-echo-actions-four">
                <button type="button" data-starfall-potential-choice="keep">Keep Current Attunement</button>
                <button type="button" data-starfall-potential-choice="keep-repeat" ${echoPrismCount > 0 ? '' : 'disabled'}>Keep Current + Use Again (${echoPrismCount})</button>
                <button type="button" class="project-starfall-echo-accept" data-starfall-potential-choice="apply">Apply New Attunement</button>
                <button type="button" class="project-starfall-echo-accept" data-starfall-potential-choice="apply-repeat" ${echoPrismCount > 0 ? '' : 'disabled'}>Apply New + Use Again (${echoPrismCount})</button>
              </div>
            ` : `
              <div class="project-starfall-upgrade-actions project-starfall-echo-actions project-starfall-attunement-result-actions">
                <span>Ready to preview without replacing current stats. ${escapeHtml(shardHint)}.</span>
                <button type="button" class="project-starfall-echo-accept" ${actionAttr} ${actionDisabled ? 'disabled' : ''}>Use Echo Prism (${actionCount})</button>
              </div>
            `}
          </footer>
        </div>
      </section>
    `;
  }

  function renderEchoPotentialChoicePrompt(item, pendingChoice, options) {
    const settings = options || {};
    const renderComparisonPrompt = typeof settings.renderAttunementComparisonPrompt === 'function'
      ? settings.renderAttunementComparisonPrompt
      : (targetItem, targetChoice, targetOptions) => renderAttunementComparisonPrompt(targetItem, targetChoice, Object.assign({}, settings, targetOptions || {}));
    return renderComparisonPrompt(item, pendingChoice, { mode: 'echo', comparison: settings.comparison || null });
  }

  function getPotentialPromptConsumableId(consumableId, options) {
    const settings = options || {};
    const id = String(consumableId || settings.potentialPromptConsumableId || 'potential_cube');
    const getConsumable = typeof settings.getConsumableItem === 'function'
      ? settings.getConsumableItem
      : () => null;
    const consumable = getConsumable(id);
    return consumable && consumable.preservationCube
      ? 'preservation_cube'
      : consumable && consumable.lineCatalyst
        ? 'line_catalyst'
        : 'potential_cube';
  }

  function getPotentialPromptRenderState(item, options) {
    const settings = options || {};
    const consumableId = String(settings.potentialPromptConsumableId || 'potential_cube');
    const mode = consumableId === 'line_catalyst'
      ? 'line_catalyst'
      : consumableId === 'preservation_cube'
        ? 'preservation_cube'
        : 'potential_cube';
    const hasCandidate = typeof settings.hasPotentialPromptComparisonCandidate === 'function'
      ? settings.hasPotentialPromptComparisonCandidate(item)
      : !!getPotentialPromptComparisonSelection(item, settings);
    const getComparisonState = typeof settings.getPotentialPromptComparisonState === 'function'
      ? settings.getPotentialPromptComparisonState
      : (targetItem) => getPotentialPromptComparisonState(targetItem, settings);
    const comparisonState = hasCandidate ? getComparisonState(item) : null;
    const hasChoice = !!(item && comparisonState && comparisonState.choice);
    return {
      item,
      consumableId,
      mode,
      promptMode: mode === 'line_catalyst' ? 'line_catalyst' : mode === 'preservation_cube' ? 'echo' : 'regular',
      comparisonState,
      choice: hasChoice ? comparisonState.choice : null,
      comparison: hasChoice ? comparisonState.comparison : null,
      renderMode: hasChoice
        ? comparisonState.mode === 'regular' ? 'regular-comparison' : 'echo-choice'
        : 'prompt'
    };
  }

  function renderPotentialPrompt(item, options) {
    const settings = options || {};
    const renderState = getPotentialPromptRenderState(item, settings);
    const renderComparisonPrompt = typeof settings.renderAttunementComparisonPrompt === 'function'
      ? settings.renderAttunementComparisonPrompt
      : (targetItem, targetChoice, targetOptions) => renderAttunementComparisonPrompt(targetItem, targetChoice, Object.assign({}, settings, targetOptions || {}));
    const renderEchoChoicePrompt = typeof settings.renderEchoPotentialChoicePrompt === 'function'
      ? settings.renderEchoPotentialChoicePrompt
      : (targetItem, targetChoice, targetOptions) => renderEchoPotentialChoicePrompt(targetItem, targetChoice, Object.assign({}, settings, targetOptions || {}));
    if (renderState.renderMode === 'regular-comparison') {
      return renderComparisonPrompt(item, renderState.choice, { mode: 'regular', comparison: renderState.comparison });
    }
    if (renderState.renderMode === 'echo-choice') {
      return renderEchoChoicePrompt(item, renderState.choice, { comparison: renderState.comparison });
    }
    return renderComparisonPrompt(item, null, { mode: renderState.promptMode });
  }

  function getShardCraftRecipeConfig(recipe, options) {
    const settings = options || {};
    const id = String(recipe || settings.shardCraftRecipe || 'potential');
    if (id === 'preservation') return { id: 'preservation', cost: 25, label: 'Echo Prism', actionLabel: 'Make Echo Prism' };
    if (id === 'line_catalyst' || id === 'lineCatalyst') return { id: 'line_catalyst', cost: 20, label: 'Line Catalyst', actionLabel: 'Make Line Catalyst' };
    return { id: 'potential', cost: 10, label: 'Attunement Prism', actionLabel: 'Make Attunement Prism' };
  }

  function getShardCraftMaxQuantity(recipe, options) {
    const settings = options || {};
    const snapshot = settings.snapshot || {};
    const state = snapshot.state || {};
    const materials = state.materials || {};
    const fragmentCount = Object.prototype.hasOwnProperty.call(settings, 'fragmentCount')
      ? Math.max(0, Math.floor(Number(settings.fragmentCount || 0) || 0))
      : Math.max(0, Math.floor(Number(materials.cubeFragment || 0) || 0));
    const getRecipeConfig = typeof settings.getShardCraftRecipeConfig === 'function'
      ? settings.getShardCraftRecipeConfig
      : (targetRecipe) => getShardCraftRecipeConfig(targetRecipe, settings);
    const config = getRecipeConfig(recipe);
    return Math.max(0, Math.floor(fragmentCount / Math.max(1, Number(config.cost || 1))));
  }

  function getShardCraftRecipeSelectionState(recipe, options) {
    const settings = options || {};
    const getRecipeConfig = typeof settings.getShardCraftRecipeConfig === 'function'
      ? settings.getShardCraftRecipeConfig
      : (targetRecipe) => getShardCraftRecipeConfig(targetRecipe, settings);
    const config = getRecipeConfig(recipe);
    return {
      recipe: config.id,
      quantity: 1
    };
  }

  function getShardCraftPromptOpenState(options) {
    const settings = options || {};
    return {
      shardCraftPromptOpen: true,
      shardCraftRecipe: settings.shardCraftRecipe || 'potential',
      shardCraftQuantity: settings.shardCraftQuantity || 1,
      isCommandOpen: false
    };
  }

  function getShardCraftPromptCloseState(options) {
    const settings = options || {};
    return {
      shouldClose: !!settings.shardCraftPromptOpen,
      shardCraftPromptOpen: false,
      shardCraftPromptDrag: null
    };
  }

  function normalizeShardCraftPromptQuantity(quantity, maxQuantity) {
    const max = Math.max(0, Math.floor(Number(maxQuantity || 0) || 0));
    return max > 0 ? clamp(Math.floor(Number(quantity) || 1), 1, max) : 1;
  }

  function getShardCraftQuantityAdjustment(quantity, delta) {
    return Number(quantity || 1) + Number(delta || 0);
  }

  function normalizeShardCraftCombineQuantity(quantity, fallback) {
    return Math.max(1, Math.floor(Number(quantity || fallback || 1) || 1));
  }

  function getShardCraftConfirmAction(recipe, options) {
    const settings = options || {};
    const getRecipeConfig = typeof settings.getShardCraftRecipeConfig === 'function'
      ? settings.getShardCraftRecipeConfig
      : (targetRecipe) => getShardCraftRecipeConfig(targetRecipe, settings);
    const config = recipe && typeof recipe === 'object'
      ? recipe
      : getRecipeConfig(recipe || settings.shardCraftRecipe);
    const type = config.id === 'preservation'
      ? 'combinePreservationCubeFragments'
      : config.id === 'line_catalyst'
        ? 'combineLineCatalysts'
        : 'combineCubeFragments';
    return {
      handled: true,
      type,
      recipeId: config.id,
      quantity: settings.quantity
    };
  }

  function getShardCraftPromptRenderState(options) {
    const settings = options || {};
    const snapshot = settings.snapshot || {};
    const state = snapshot.state || {};
    const materials = state.materials || {};
    const getRecipeConfig = typeof settings.getShardCraftRecipeConfig === 'function'
      ? settings.getShardCraftRecipeConfig
      : (recipe) => getShardCraftRecipeConfig(recipe, settings);
    const fragmentCount = Math.max(0, Math.floor(Number(materials.cubeFragment || 0) || 0));
    const recipe = getRecipeConfig(settings.shardCraftRecipe);
    const getMaxQuantity = typeof settings.getShardCraftMaxQuantity === 'function'
      ? settings.getShardCraftMaxQuantity
      : (recipeId) => getShardCraftMaxQuantity(recipeId, Object.assign({}, settings, {
        fragmentCount,
        getShardCraftRecipeConfig: getRecipeConfig
      }));
    const maxQuantity = Math.max(0, Math.floor(Number(getMaxQuantity(recipe.id, recipe, fragmentCount)) || 0));
    const quantity = normalizeShardCraftPromptQuantity(settings.shardCraftQuantity || 1, maxQuantity);
    if (typeof settings.setShardCraftPromptState === 'function') {
      settings.setShardCraftPromptState({ recipe: recipe.id, quantity, maxQuantity });
    }
    return {
      fragmentCount,
      recipe,
      maxQuantity,
      quantity,
      totalCost: recipe.cost * quantity
    };
  }

  function renderShardCraftPrompt(options) {
    const settings = options || {};
    const escapeHtml = typeof settings.escapeHtml === 'function'
      ? settings.escapeHtml
      : (value) => String(value == null ? '' : value);
    const formatAbbreviatedInteger = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (value) => String(value);
    const prompt = getShardCraftPromptRenderState(settings);
    const fragmentCount = prompt.fragmentCount;
    const recipe = prompt.recipe;
    const maxQuantity = prompt.maxQuantity;
    const quantity = prompt.quantity;
    const totalCost = prompt.totalCost;
    return `
      <section class="project-starfall-upgrade-confirm project-starfall-shard-craft" role="dialog" aria-modal="true" aria-label="Prism Shard crafting">
        <div class="project-starfall-upgrade-confirm-card">
          <button type="button" class="project-starfall-upgrade-confirm-close" data-starfall-shard-craft-close aria-label="Close shard crafting">X</button>
          <div class="project-starfall-potential-current">
            <strong>Prism Shards</strong>
            <span>${formatAbbreviatedInteger(fragmentCount)}</span>
            <small>Choose what to make.</small>
          </div>
          <div class="project-starfall-upgrade-actions">
            <button type="button" class="${recipe.id === 'potential' ? 'is-selected' : ''}" data-starfall-shard-craft-recipe="potential" ${fragmentCount >= 10 ? '' : 'disabled'}>Make Attunement Prism</button>
            <button type="button" class="${recipe.id === 'line_catalyst' ? 'is-selected' : ''}" data-starfall-shard-craft-recipe="line_catalyst" ${fragmentCount >= 20 ? '' : 'disabled'}>Make Line Catalyst</button>
            <button type="button" class="${recipe.id === 'preservation' ? 'is-selected' : ''}" data-starfall-shard-craft-recipe="preservation" ${fragmentCount >= 25 ? '' : 'disabled'}>Make Echo Prism</button>
          </div>
          <div class="project-starfall-shard-craft-quantity">
            <span>${escapeHtml(recipe.label)} <strong>${formatAbbreviatedInteger(quantity)}/${formatAbbreviatedInteger(Math.max(1, maxQuantity))}</strong></span>
            <div>
              <button type="button" data-starfall-shard-craft-delta="-1" ${quantity <= 1 ? 'disabled' : ''}>-</button>
              <button type="button" data-starfall-shard-craft-delta="1" ${quantity >= maxQuantity ? 'disabled' : ''}>+</button>
              <button type="button" data-starfall-shard-craft-all ${quantity >= maxQuantity ? 'disabled' : ''}>All</button>
            </div>
            <small>Costs ${formatAbbreviatedInteger(totalCost)} Prism Shards.</small>
          </div>
          <button type="button" data-starfall-shard-craft-confirm ${maxQuantity > 0 ? '' : 'disabled'}>${escapeHtml(recipe.actionLabel)} x${quantity}</button>
        </div>
      </section>
    `;
  }

  function getCanvasShardCraftPromptLayout(x, y, w, h) {
    const headerH = 36;
    const recipeGap = 6;
    const recipeW = Math.floor((w - 32 - recipeGap * 2) / 3);
    const recipeY = y + 116;
    const qtyY = y + 212;
    return {
      headerH,
      titleX: x + 14,
      titleY: y + 10,
      titleMaxW: w - 58,
      headerRegionW: w - 42,
      closeX: x + w - 34,
      closeY: y + 7,
      closeW: 22,
      closeH: 22,
      summaryX: x + 16,
      summaryY: y + 50,
      summaryW: w - 32,
      summaryH: 54,
      summaryTextX: x + 28,
      fragmentY: y + 62,
      summaryHintY: y + 88,
      contentMaxW: w - 56,
      recipeY,
      recipeGap,
      recipeW,
      recipeH: 26,
      potentialRecipeX: x + 16,
      catalystRecipeX: x + 16 + recipeW + recipeGap,
      echoRecipeX: x + 16 + (recipeW + recipeGap) * 2,
      detailX: x + 16,
      detailY: y + 152,
      detailW: w - 32,
      detailH: 54,
      detailTextX: x + 28,
      detailLabelY: y + 162,
      detailCostY: y + 184,
      qtyY,
      decrementX: x + 16,
      incrementX: x + 56,
      allX: x + 96,
      decrementW: 34,
      incrementW: 34,
      allW: 48,
      quantityButtonH: 24,
      confirmX: x + 154,
      confirmW: w - 170
    };
  }

  function drawCanvasShardCraftPrompt(ctx, width, height, options) {
    const settings = options || {};
    if (settings.shardCraftPromptOpen === false) return false;
    const formatAbbreviatedInteger = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (value) => String(value);
    const getPromptBox = typeof settings.getShardCraftPromptBox === 'function'
      ? settings.getShardCraftPromptBox
      : (boxWidth, boxHeight) => getShardCraftPromptBox(boxWidth, boxHeight, settings.shardCraftPromptWindow || {}, settings);
    const addCanvasRegion = typeof settings.addCanvasRegion === 'function'
      ? settings.addCanvasRegion
      : () => {};
    const drawCanvasUiWindowFrame = typeof settings.drawCanvasUiWindowFrame === 'function'
      ? settings.drawCanvasUiWindowFrame
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const drawCanvasButton = typeof settings.drawCanvasButton === 'function'
      ? settings.drawCanvasButton
      : () => {};
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const prompt = getShardCraftPromptRenderState(settings);
    const fragmentCount = prompt.fragmentCount;
    const recipe = prompt.recipe;
    const maxQuantity = prompt.maxQuantity;
    const quantity = prompt.quantity;
    const totalCost = prompt.totalCost;
    const box = getPromptBox(width, height);
    const x = box.x;
    const y = box.y;
    const w = box.w;
    const h = box.h;
    const layout = getCanvasShardCraftPromptLayout(x, y, w, h);
    addCanvasRegion({ type: 'shard-craft-shell', x, y, w, h });
    drawCanvasUiWindowFrame(ctx, x, y, w, h, layout.headerH, { radius: 10, gems: true });
    drawCanvasText(ctx, 'Prism Shards', layout.titleX, layout.titleY, { color: '#ffffff', font: '900 13px system-ui', maxWidth: layout.titleMaxW, lineHeight: 15 });
    addCanvasRegion({ type: 'shard-craft-header', x, y, w: layout.headerRegionW, h: layout.headerH });
    drawCanvasButton(ctx, 'X', layout.closeX, layout.closeY, layout.closeW, layout.closeH, { type: 'shard-craft-close' }, false);
    drawRoundRect(ctx, layout.summaryX, layout.summaryY, layout.summaryW, layout.summaryH, 7, '#fbfaf6', 'rgba(16,32,51,0.12)');
    drawCanvasText(ctx, formatAbbreviatedInteger(fragmentCount), layout.summaryTextX, layout.fragmentY, { color: '#102033', font: '950 18px system-ui', maxWidth: layout.contentMaxW, maxLines: 1, lineHeight: 20 });
    drawCanvasText(ctx, 'Choose what to make.', layout.summaryTextX, layout.summaryHintY, { color: '#5f6f7a', font: '10px system-ui', maxWidth: layout.contentMaxW, maxLines: 1, lineHeight: 11 });
    drawCanvasButton(ctx, 'Attunement', layout.potentialRecipeX, layout.recipeY, layout.recipeW, layout.recipeH, { type: 'shard-craft-recipe', recipe: 'potential' }, fragmentCount < 10);
    drawCanvasButton(ctx, 'Line Catalyst', layout.catalystRecipeX, layout.recipeY, layout.recipeW, layout.recipeH, { type: 'shard-craft-recipe', recipe: 'line_catalyst' }, fragmentCount < 20);
    drawCanvasButton(ctx, 'Echo Prism', layout.echoRecipeX, layout.recipeY, layout.recipeW, layout.recipeH, { type: 'shard-craft-recipe', recipe: 'preservation' }, fragmentCount < 25);
    drawRoundRect(ctx, layout.detailX, layout.detailY, layout.detailW, layout.detailH, 7, '#fbfaf6', 'rgba(16,32,51,0.12)');
    drawCanvasText(ctx, `${recipe.label} ${formatAbbreviatedInteger(quantity)}/${formatAbbreviatedInteger(Math.max(1, maxQuantity))}`, layout.detailTextX, layout.detailLabelY, { color: '#102033', font: '900 11px system-ui', maxWidth: layout.contentMaxW, maxLines: 1, lineHeight: 12 });
    drawCanvasText(ctx, `Costs ${formatAbbreviatedInteger(totalCost)} Prism Shards.`, layout.detailTextX, layout.detailCostY, { color: '#5f6f7a', font: '9px system-ui', maxWidth: layout.contentMaxW, maxLines: 1, lineHeight: 10 });
    drawCanvasButton(ctx, '-', layout.decrementX, layout.qtyY, layout.decrementW, layout.quantityButtonH, { type: 'shard-craft-decrease' }, quantity <= 1);
    drawCanvasButton(ctx, '+', layout.incrementX, layout.qtyY, layout.incrementW, layout.quantityButtonH, { type: 'shard-craft-increase' }, quantity >= maxQuantity);
    drawCanvasButton(ctx, 'All', layout.allX, layout.qtyY, layout.allW, layout.quantityButtonH, { type: 'shard-craft-all' }, quantity >= maxQuantity);
    drawCanvasButton(ctx, `${recipe.actionLabel} x${quantity}`, layout.confirmX, layout.qtyY, layout.confirmW, layout.quantityButtonH, { type: 'shard-craft-confirm' }, maxQuantity < 1);
    return true;
  }

  function getCanvasPotentialPromptRenderState(width, height, options) {
    const settings = options || {};
    if (settings.potentialPromptOpen === false) return null;
    const consumableId = String(settings.consumableId || settings.potentialPromptConsumableId || 'potential_cube');
    const mode = consumableId === 'line_catalyst'
      ? 'line_catalyst'
      : consumableId === 'preservation_cube'
        ? 'preservation_cube'
        : 'potential_cube';
    const getPromptItem = typeof settings.getPotentialPromptItem === 'function'
      ? settings.getPotentialPromptItem
      : () => settings.item || null;
    const item = Object.prototype.hasOwnProperty.call(settings, 'item')
      ? settings.item
      : getPromptItem(settings.potentialPromptUid);
    const promptOptions = Object.assign({}, settings, { consumableId });
    const hasComparisonCandidate = typeof settings.hasPotentialPromptComparisonCandidate === 'function'
      ? settings.hasPotentialPromptComparisonCandidate(item)
      : !!getPotentialPromptComparisonSelection(item, promptOptions);
    const getComparisonState = typeof settings.getPotentialPromptComparisonState === 'function'
      ? settings.getPotentialPromptComparisonState
      : (targetItem) => {
        const result = getPotentialPromptComparisonState(targetItem, promptOptions);
        return result ? result.state : null;
      };
    const comparisonState = hasComparisonCandidate ? getComparisonState(item) : null;
    const getPromptBox = typeof settings.getPotentialPromptBox === 'function'
      ? settings.getPotentialPromptBox
      : (targetWidth, targetHeight, targetItem, targetComparisonState) => getPotentialPromptBox(targetWidth, targetHeight, settings.potentialPromptWindow || {}, Object.assign({}, promptOptions, {
        item: targetItem,
        comparisonState: targetComparisonState
      }));
    const box = getPromptBox(width, height, item, comparisonState) || {};
    const renderMode = comparisonState && comparisonState.choice
      ? 'comparison'
      : item
        ? 'ready'
        : 'empty';
    return {
      item,
      mode,
      consumableId,
      comparisonState,
      box,
      x: box.x,
      y: box.y,
      boxW: box.w,
      boxH: box.h,
      renderMode
    };
  }

  function getCanvasPotentialHelpLayout(x, y, boxW, boxH) {
    const headerH = 36;
    const bodyX = x + 14;
    const bodyY = y + headerH + 12;
    const bodyW = boxW - 28;
    const bodyH = boxH - headerH - 24;
    const tierW = 74;
    const linesW = 42;
    const oddsW = 70;
    const rangeW = 110;
    const statX = bodyX + tierW + linesW + 8;
    const rangeX = bodyX + bodyW - oddsW - rangeW - 8;
    const oddsX = bodyX + bodyW - oddsW;
    return {
      headerH,
      bodyX,
      bodyY,
      bodyW,
      bodyH,
      tierW,
      linesW,
      oddsW,
      rangeW,
      statX,
      rangeX,
      oddsX
    };
  }

  function drawCanvasPotentialHelp(ctx, width, height, options) {
    const settings = options || {};
    if (settings.potentialHelpOpen === false) return false;
    const getPromptItem = typeof settings.getPotentialPromptItem === 'function'
      ? settings.getPotentialPromptItem
      : () => settings.item || null;
    const item = Object.prototype.hasOwnProperty.call(settings, 'item')
      ? settings.item
      : getPromptItem(settings.potentialPromptUid);
    const getRows = typeof settings.getPotentialHelpRows === 'function'
      ? settings.getPotentialHelpRows
      : (targetItem) => getPotentialHelpRows(targetItem, settings);
    const getPromptBox = typeof settings.getPotentialHelpBox === 'function'
      ? settings.getPotentialHelpBox
      : (boxWidth, boxHeight) => getPotentialHelpBox(boxWidth, boxHeight, settings.potentialHelpWindow || {}, settings);
    const getHelpWindow = typeof settings.getPotentialHelpWindow === 'function'
      ? settings.getPotentialHelpWindow
      : () => settings.potentialHelpWindow || {};
    const getSlotMeta = typeof settings.getSlotMeta === 'function'
      ? settings.getSlotMeta
      : (slot) => ({ label: String(slot || '') });
    const getRollSummary = typeof settings.getPotentialTierRollSummary === 'function'
      ? settings.getPotentialTierRollSummary
      : (targetItem) => getPotentialTierRollSummary(targetItem, settings);
    const addCanvasRegion = typeof settings.addCanvasRegion === 'function'
      ? settings.addCanvasRegion
      : () => {};
    const drawCanvasUiWindowFrame = typeof settings.drawCanvasUiWindowFrame === 'function'
      ? settings.drawCanvasUiWindowFrame
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const drawCanvasButton = typeof settings.drawCanvasButton === 'function'
      ? settings.drawCanvasButton
      : () => {};
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const rows = getRows(item);
    const box = getPromptBox(width, height) || {};
    const x = box.x;
    const y = box.y;
    const boxW = box.w;
    const boxH = box.h;
    const layout = getCanvasPotentialHelpLayout(x, y, boxW, boxH);
    const headerH = layout.headerH;
    const bodyX = layout.bodyX;
    const bodyY = layout.bodyY;
    const bodyW = layout.bodyW;
    const bodyH = layout.bodyH;
    const tierW = layout.tierW;
    const linesW = layout.linesW;
    const oddsW = layout.oddsW;
    const rangeW = layout.rangeW;
    const statX = layout.statX;
    const rangeX = layout.rangeX;
    const oddsX = layout.oddsX;
    const helpWindow = getHelpWindow() || {};
    addCanvasRegion({ type: 'potential-help-shell', x, y, w: boxW, h: boxH });
    drawCanvasUiWindowFrame(ctx, x, y, boxW, boxH, headerH, { radius: 10, gems: true });
    drawCanvasText(ctx, 'Attunement Odds', x + 14, y + 10, { color: '#ffffff', font: '900 13px system-ui', maxWidth: boxW - 62, lineHeight: 15 });
    drawCanvasButton(ctx, 'X', x + boxW - 34, y + 7, 22, 22, { type: 'potential-help-close' }, false);
    addCanvasRegion({ type: 'potential-help-body', x: bodyX, y: bodyY, w: bodyW, h: bodyH });
    ctx.save();
    ctx.beginPath();
    ctx.rect(bodyX, bodyY, bodyW, bodyH);
    ctx.clip();
    const scroll = Number(helpWindow.scroll || 0);
    let cy = bodyY - scroll;
    drawCanvasText(ctx, item ? `${item.name} - ${getSlotMeta(item.slot).label}` : 'All eligible stat pools', bodyX, cy, {
      color: '#102033',
      font: '950 12px system-ui',
      maxWidth: bodyW,
      lineHeight: 14,
      maxLines: 1
    });
    cy += 17;
    drawCanvasText(ctx, getRollSummary(item), bodyX, cy, {
      color: '#8856c5',
      font: '850 10px system-ui',
      maxWidth: bodyW,
      lineHeight: 12,
      maxLines: 2
    });
    cy += 28;
    drawRoundRect(ctx, bodyX, cy, bodyW, 20, 6, '#f4f0fb', 'rgba(136,86,197,0.18)');
    drawCanvasText(ctx, 'Tier', bodyX + 8, cy + 6, { color: '#5f6f7a', font: '900 8px system-ui', maxWidth: tierW - 12, lineHeight: 9 });
    drawCanvasText(ctx, 'Quality', bodyX + tierW, cy + 6, { color: '#5f6f7a', font: '900 8px system-ui', maxWidth: linesW, lineHeight: 9 });
    drawCanvasText(ctx, 'Stat', statX, cy + 6, { color: '#5f6f7a', font: '900 8px system-ui', maxWidth: rangeX - statX - 6, lineHeight: 9 });
    drawCanvasText(ctx, 'Range', rangeX, cy + 6, { color: '#5f6f7a', font: '900 8px system-ui', maxWidth: rangeW, lineHeight: 9 });
    drawCanvasText(ctx, 'Odds/line', oddsX, cy + 6, { color: '#5f6f7a', font: '900 8px system-ui', maxWidth: oddsW - 8, lineHeight: 9 });
    cy += 24;
    (rows || []).forEach((row, index) => {
      const rowH = item ? 21 : 29;
      if (cy + rowH >= bodyY && cy <= bodyY + bodyH) {
        drawRoundRect(ctx, bodyX, cy, bodyW, rowH - 3, 5, index % 2 ? '#ffffff' : '#fbfaf6', 'rgba(16,32,51,0.06)');
        drawCanvasText(ctx, row.tier.name.replace(' Attunement', ''), bodyX + 8, cy + 5, { color: '#102033', font: '850 8px system-ui', maxWidth: tierW - 12, lineHeight: 9, maxLines: 1 });
        drawCanvasText(ctx, 'Range', bodyX + tierW + 8, cy + 5, { color: '#5f6f7a', font: '850 8px system-ui', maxWidth: linesW - 8, lineHeight: 9 });
        drawCanvasText(ctx, row.stat, statX, cy + 5, { color: '#102033', font: '900 9px system-ui', maxWidth: rangeX - statX - 8, lineHeight: 10, maxLines: 1 });
        if (!item) drawCanvasText(ctx, row.slot, statX, cy + 17, { color: '#5f6f7a', font: '750 7px system-ui', maxWidth: rangeX - statX - 8, lineHeight: 8, maxLines: 1 });
        drawCanvasText(ctx, row.range, rangeX, cy + 5, { color: '#5f6f7a', font: '850 8px system-ui', maxWidth: rangeW, lineHeight: 9, maxLines: 1 });
        drawCanvasText(ctx, row.chance, oddsX, cy + 5, { color: '#8856c5', font: '900 8px system-ui', maxWidth: oddsW - 8, lineHeight: 9, maxLines: 1 });
      }
      cy += rowH;
    });
    const contentHeight = cy - (bodyY - scroll);
    helpWindow.maxScroll = Math.max(0, contentHeight - bodyH);
    helpWindow.scroll = clamp(Number(helpWindow.scroll || 0), 0, helpWindow.maxScroll);
    ctx.restore();
    if (helpWindow.maxScroll > 0) {
      const barH = Math.max(34, bodyH * bodyH / Math.max(bodyH, contentHeight));
      const barY = bodyY + (bodyH - barH) * (helpWindow.scroll / helpWindow.maxScroll);
      drawRoundRect(ctx, x + boxW - 10, barY, 4, barH, 4, 'rgba(16,32,51,0.32)', '');
    }
    return true;
  }

  function getCanvasAttunementPromptModeMeta(mode) {
    const isEcho = mode === 'preservation_cube';
    const isLineCatalyst = mode === 'line_catalyst';
    return {
      isEcho,
      isLineCatalyst,
      consumableId: isLineCatalyst ? 'line_catalyst' : isEcho ? 'preservation_cube' : 'potential_cube',
      title: isLineCatalyst ? 'LINE CATALYST' : isEcho ? 'ECHO PRISM ATTUNEMENT' : 'ATTUNEMENT PRISM',
      accentColor: isLineCatalyst ? '#8fd66f' : isEcho ? '#f4d7ff' : '#fff3cf',
      borderColor: isLineCatalyst ? 'rgba(143,214,111,0.58)' : isEcho ? 'rgba(188,143,255,0.52)' : 'rgba(245,207,114,0.52)',
      fallbackIcon: isLineCatalyst ? 'LC' : isEcho ? 'ECHO' : 'PRS',
      emptySubtitle: isLineCatalyst ? 'Select attuned gear from inventory or equipment' : 'Select gear from inventory or equipment'
    };
  }

  function drawCanvasAttunementPromptChrome(ctx, x, y, boxW, boxH, mode, subtitle, options) {
    const settings = options || {};
    const getConsumableItem = typeof settings.getConsumableItem === 'function'
      ? settings.getConsumableItem
      : () => null;
    const addCanvasRegion = typeof settings.addCanvasRegion === 'function'
      ? settings.addCanvasRegion
      : () => {};
    const drawCanvasUiWindowFrame = typeof settings.drawCanvasUiWindowFrame === 'function'
      ? settings.drawCanvasUiWindowFrame
      : () => {};
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawItemIcon = typeof settings.drawItemIcon === 'function'
      ? settings.drawItemIcon
      : () => false;
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const drawCanvasButton = typeof settings.drawCanvasButton === 'function'
      ? settings.drawCanvasButton
      : () => {};
    const meta = getCanvasAttunementPromptModeMeta(mode);
    const prismItem = getConsumableItem(meta.consumableId);
    const subtitleText = subtitle == null ? meta.emptySubtitle : String(subtitle || '');
    addCanvasRegion({ type: 'potential-prompt-shell', x, y, w: boxW, h: boxH });
    drawCanvasUiWindowFrame(ctx, x, y, boxW, boxH, 54, { radius: 10, gems: true, shadowBlur: 22, shadowOffsetY: 8 });
    drawRoundRect(ctx, x + 18, y + 12, 34, 34, 6, '#071323', meta.borderColor);
    if (!drawItemIcon(ctx, prismItem, x + 21, y + 15, 28, { showAura: false, frame: false })) {
      drawCanvasText(ctx, prismItem && prismItem.icon || meta.fallbackIcon, x + 35, y + 23, { color: meta.accentColor, font: '950 8px system-ui', align: 'center', maxWidth: 28, lineHeight: 9 });
    }
    drawCanvasText(ctx, meta.title, x + 62, y + 13, { color: meta.accentColor, font: '950 18px system-ui', maxWidth: boxW - 148, maxLines: 1, lineHeight: 21 });
    drawCanvasText(ctx, subtitleText, x + 63, y + 36, { color: '#9ccfe4', font: '850 10px system-ui', maxWidth: boxW - 150, maxLines: 1, lineHeight: 11 });
    addCanvasRegion({ type: 'potential-prompt-header', x, y, w: boxW - 72, h: 52 });
    drawCanvasButton(ctx, '?', x + boxW - 62, y + 13, 22, 22, { type: 'potential-help-open' }, false);
    drawCanvasButton(ctx, 'X', x + boxW - 34, y + 13, 22, 22, { type: 'potential-prompt-close' }, false);
    return meta;
  }

  function drawCanvasAttunementEmptyPrompt(ctx, x, y, boxW, boxH, mode, options) {
    const settings = options || {};
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const addCanvasRegion = typeof settings.addCanvasRegion === 'function'
      ? settings.addCanvasRegion
      : () => {};
    const drawIconBadge = typeof settings.drawIconBadge === 'function'
      ? settings.drawIconBadge
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const meta = drawCanvasAttunementPromptChrome(ctx, x, y, boxW, boxH, mode, null, settings);
    const dropX = x + 18;
    const dropY = y + 72;
    const dropW = boxW - 36;
    const dropH = Math.max(106, boxH - 94);
    drawRoundRect(ctx, dropX, dropY, dropW, dropH, 8, '#fffaf0', meta.isLineCatalyst ? 'rgba(143,214,111,0.5)' : 'rgba(136,86,197,0.42)');
    addCanvasRegion({ type: 'potential-prompt-drop-zone', x: dropX, y: dropY, w: dropW, h: dropH });
    addCanvasRegion({ type: 'gear-picker-open', context: 'attunement', x: dropX, y: dropY, w: dropW, h: dropH });
    const badgeSize = 46;
    drawIconBadge(ctx, 'GEAR', dropX + 20, dropY + Math.floor((dropH - badgeSize) / 2), badgeSize, '#ffffff', 'rgba(136,86,197,0.3)');
    drawCanvasText(ctx, 'Drop gear here', dropX + 84, dropY + Math.floor(dropH / 2) - 14, { color: '#102033', font: '950 14px system-ui', maxWidth: dropW - 104, maxLines: 1, lineHeight: 16 });
    drawCanvasText(ctx, 'Drag gear here, or click to choose.', dropX + 84, dropY + Math.floor(dropH / 2) + 10, { color: '#5f6f7a', font: '850 10px system-ui', maxWidth: dropW - 104, maxLines: 1, lineHeight: 11 });
    return true;
  }

  function getCanvasAttunementSummaryGeometry(x, y, boxW) {
    const summaryY = y + 58;
    const summaryX = x + 16;
    const summaryW = boxW - 32;
    const summaryStacked = boxW < 560;
    const summaryH = summaryStacked ? 102 : 58;
    return { summaryX, summaryY, summaryW, summaryH, summaryStacked };
  }

  function drawCanvasAttunementReadySummary(ctx, x, y, boxW, mode, options) {
    const settings = options || {};
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const formatAbbreviatedInteger = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (value) => String(value);
    const formatChance = typeof settings.formatChance === 'function'
      ? settings.formatChance
      : (value) => `${value}%`;
    const meta = getCanvasAttunementPromptModeMeta(mode);
    const currentTier = settings.currentTier || null;
    const currentPower = Number(settings.currentPower || 0) || 0;
    const item = settings.item || null;
    const lineInfo = settings.lineInfo || {};
    const chance = Math.max(0, Number(lineInfo.chance || 0) || 0);
    const lineChanceLabel = settings.lineChanceLabel || (chance > 0 ? formatChance(chance * 100) : '-');
    const geometry = getCanvasAttunementSummaryGeometry(x, y, boxW);
    const summaryX = geometry.summaryX;
    const summaryY = geometry.summaryY;
    const summaryW = geometry.summaryW;
    const summaryH = geometry.summaryH;
    const summaryStacked = geometry.summaryStacked;
    drawRoundRect(ctx, summaryX, summaryY, summaryW, summaryH, 4, '#fff0cf', 'rgba(137,93,39,0.34)');
    const metrics = [
      ['TOTAL POWER', formatAbbreviatedInteger(currentPower), '-'],
      ['TIER', currentTier ? String(currentTier.name || '').replace(' Attunement', '') : 'None', '-'],
      [meta.isLineCatalyst ? 'LINE CATALYST' : meta.isEcho ? 'ECHO PRISM' : 'ATTUNEMENT', item ? `${lineInfo.lineCount}/${lineInfo.maxLineCount}` : '-', chance > 0 ? `${lineChanceLabel} next` : lineInfo.hasPotential ? 'Max' : 'Attune']
    ];
    metrics.forEach((metric, index) => {
      const metricW = summaryStacked ? summaryW : summaryW / 3;
      const metricH = summaryStacked ? summaryH / 3 : summaryH;
      const mx = summaryStacked ? summaryX : summaryX + metricW * index;
      const my = summaryStacked ? summaryY + metricH * index : summaryY;
      if (index > 0 && ctx && ctx.beginPath && ctx.moveTo && ctx.lineTo && ctx.stroke) {
        ctx.strokeStyle = 'rgba(137,93,39,0.22)';
        ctx.beginPath();
        if (summaryStacked) {
          ctx.moveTo(summaryX, my);
          ctx.lineTo(summaryX + summaryW, my);
        } else {
          ctx.moveTo(mx, summaryY);
          ctx.lineTo(mx, summaryY + summaryH);
        }
        ctx.stroke();
      }
      drawCanvasText(ctx, metric[0], mx + 12, my + 8, { color: '#8a5d27', font: '950 9px system-ui', maxWidth: metricW - 24, maxLines: 1, lineHeight: 10 });
      drawCanvasText(ctx, metric[1], summaryStacked ? mx + metricW - 82 : mx + metricW / 2, my + 27, { color: '#1c2631', font: '950 15px system-ui', align: summaryStacked ? 'right' : 'center', maxWidth: metricW - 86, maxLines: 1, lineHeight: 17 });
      drawCanvasText(ctx, metric[2], mx + metricW - 14, my + 28, { color: '#65727c', font: '950 10px system-ui', align: 'right', maxWidth: 72, maxLines: 1, lineHeight: 11 });
    });
    return geometry;
  }

  function getCanvasAttunementReadyCurrentPanelLayout(x, y, w, h) {
    const rowH = 30;
    const rowStartY = y + 40;
    const rowClipY = y + h - 38;
    const totalY = y + h - 38;
    return {
      rowH,
      rowStartY,
      rowClipY,
      totalY,
      rowFillX: x + 1,
      rowFillW: w - 2,
      footerH: 37
    };
  }

  function drawCanvasAttunementReadyCurrentPanel(ctx, x, y, w, h, options) {
    const settings = options || {};
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const getPotentialRollRank = typeof settings.getPotentialRollRank === 'function'
      ? settings.getPotentialRollRank
      : () => ({ label: '', percentLabel: '--', fill: '#ffffff', color: '#1c2631' });
    const getPotentialStatAbbreviation = typeof settings.getPotentialStatAbbreviation === 'function'
      ? settings.getPotentialStatAbbreviation
      : (stat) => String(stat || '').slice(0, 2).toUpperCase();
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (stat, value) => String(value);
    const formatAbbreviatedInteger = typeof settings.formatAbbreviatedInteger === 'function'
      ? settings.formatAbbreviatedInteger
      : (value) => String(value);
    const rows = Array.isArray(settings.rows) ? settings.rows : [];
    const currentTier = settings.currentTier || null;
    const currentPower = Number(settings.currentPower || 0) || 0;
    const tierText = currentTier ? String(currentTier.name || '').replace(' Attunement', '') : 'None';
    drawRoundRect(ctx, x, y, w, h, 4, '#fff0cf', 'rgba(137,93,39,0.42)');
    drawCanvasText(ctx, 'Current attunement', x + 12, y + 12, { color: '#8a5d27', font: '950 10px system-ui', maxWidth: w - 94, maxLines: 1, lineHeight: 11 });
    drawCanvasText(ctx, tierText, x + w - 12, y + 12, { color: '#8a5d27', font: '950 9px system-ui', align: 'right', maxWidth: 78, maxLines: 1, lineHeight: 10 });
    const layout = getCanvasAttunementReadyCurrentPanelLayout(x, y, w, h);
    if (rows.length) {
      rows.forEach((row, index) => {
        const rowY = layout.rowStartY + index * layout.rowH;
        const rank = getPotentialRollRank(row.percentileLabel);
        if (rowY + layout.rowH > layout.rowClipY) return;
        if (ctx && ctx.fillRect) {
          ctx.fillStyle = index % 2 ? 'rgba(137,93,39,0.035)' : 'rgba(255,255,255,0.2)';
          ctx.fillRect(layout.rowFillX, rowY, layout.rowFillW, layout.rowH);
        }
        drawRoundRect(ctx, x + 8, rowY + 5, 20, 20, 10, 'rgba(137,93,39,0.08)', 'rgba(137,93,39,0.2)');
        drawCanvasText(ctx, getPotentialStatAbbreviation(row.stat), x + 18, rowY + 10, { color: '#8a5d27', font: '950 7px system-ui', align: 'center', maxWidth: 16, lineHeight: 8 });
        drawCanvasText(ctx, formatStatName(row.stat), x + 36, rowY + 9, { color: '#1c2631', font: '850 10px system-ui', maxWidth: Math.max(52, w - 142), maxLines: 1, lineHeight: 11 });
        drawCanvasText(ctx, formatStatValue(row.stat, row.value), x + w - 68, rowY + 8, { color: '#1c2631', font: '950 10px system-ui', align: 'right', maxWidth: 58, maxLines: 1, lineHeight: 11 });
        drawRoundRect(ctx, x + w - 58, rowY + 7, 30, 16, 3, rank.fill, 'rgba(137,93,39,0.26)');
        drawCanvasText(ctx, rank.label, x + w - 43, rowY + 10, { color: rank.color, font: '950 9px system-ui', align: 'center', maxWidth: 22, maxLines: 1, lineHeight: 10 });
        drawCanvasText(ctx, rank.percentLabel || '--', x + w - 10, rowY + 10, { color: '#65727c', font: '850 8px system-ui', align: 'right', maxWidth: 34, maxLines: 1, lineHeight: 9 });
      });
    } else {
      drawCanvasText(ctx, 'No attunement stats', x + 12, y + 54, { color: '#65727c', font: '10px system-ui', maxWidth: w - 24, maxLines: 1, lineHeight: 11 });
    }
    const totalY = layout.totalY;
    if (ctx && ctx.fillRect) {
      ctx.fillStyle = 'rgba(137,93,39,0.08)';
      ctx.fillRect(layout.rowFillX, totalY, layout.rowFillW, layout.footerH);
    }
    drawCanvasText(ctx, 'TOTAL POWER', x + 12, totalY + 12, { color: '#8a5d27', font: '950 10px system-ui', maxWidth: w - 118, maxLines: 1, lineHeight: 11 });
    drawCanvasText(ctx, formatAbbreviatedInteger(currentPower), x + w - 12, totalY + 11, { color: '#1c2631', font: '950 12px system-ui', align: 'right', maxWidth: 78, maxLines: 1, lineHeight: 13 });
    return true;
  }

  function getCanvasAttunementReadyTargetPanelLayout(x, y, w, h) {
    const dropX = x + 12;
    const dropY = y + 50;
    const dropW = w - 24;
    const dropH = Math.min(86, Math.max(68, h - 126));
    return {
      dropX,
      dropY,
      dropW,
      dropH
    };
  }

  function drawCanvasAttunementReadyTargetPanel(ctx, x, y, w, h, mode, options) {
    const settings = options || {};
    const item = settings.item || null;
    const shardHint = settings.shardHint || '';
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const drawItemIcon = typeof settings.drawItemIcon === 'function'
      ? settings.drawItemIcon
      : () => false;
    const drawIconBadge = typeof settings.drawIconBadge === 'function'
      ? settings.drawIconBadge
      : () => {};
    const addCanvasRegion = typeof settings.addCanvasRegion === 'function'
      ? settings.addCanvasRegion
      : () => {};
    const getSlotMeta = typeof settings.getSlotMeta === 'function'
      ? settings.getSlotMeta
      : () => ({ label: '' });
    const getTierLabel = typeof settings.getPotentialTierLabel === 'function'
      ? settings.getPotentialTierLabel
      : (targetItem) => getPotentialTierLabel(targetItem, settings);
    const meta = getCanvasAttunementPromptModeMeta(mode);
    const titleText = item
      ? meta.isLineCatalyst ? 'READY FOR LINE UPGRADE' : 'READY FOR NEXT ROLL'
      : 'SELECT GEAR';
    const tierText = meta.isLineCatalyst ? 'Line' : meta.isEcho ? 'Echo' : 'Prism';
    drawRoundRect(ctx, x, y, w, h, 4, '#fff0cf', 'rgba(137,93,39,0.42)');
    drawCanvasText(ctx, titleText, x + 12, y + 12, { color: '#8a5d27', font: '950 10px system-ui', maxWidth: w - 94, maxLines: 1, lineHeight: 11 });
    drawCanvasText(ctx, tierText, x + w - 12, y + 12, { color: '#8a5d27', font: '950 9px system-ui', align: 'right', maxWidth: 78, maxLines: 1, lineHeight: 10 });
    const layout = getCanvasAttunementReadyTargetPanelLayout(x, y, w, h);
    const dropX = layout.dropX;
    const dropY = layout.dropY;
    const dropW = layout.dropW;
    const dropH = layout.dropH;
    drawRoundRect(ctx, dropX, dropY, dropW, dropH, 6, item ? '#fffaf0' : '#fbfaf6', item ? meta.isLineCatalyst ? 'rgba(143,214,111,0.38)' : 'rgba(47,125,214,0.28)' : 'rgba(136,86,197,0.42)');
    addCanvasRegion({ type: 'potential-prompt-drop-zone', x: dropX, y: dropY, w: dropW, h: dropH });
    addCanvasRegion({ type: 'gear-picker-open', context: 'attunement', x: dropX, y: dropY, w: dropW, h: dropH });
    if (item) {
      const slotMeta = getSlotMeta(item.slot) || {};
      drawItemIcon(ctx, item, x + 22, dropY + 12, 46);
      drawCanvasText(ctx, `${item.rarity || 'Common'} ${item.name} +${item.upgrade || 0}`, x + 78, dropY + 15, { color: '#102033', font: '950 12px system-ui', maxWidth: w - 96, maxLines: 1, lineHeight: 14 });
      drawCanvasText(ctx, `${slotMeta.label || ''} - ${getTierLabel(item)}`, x + 78, dropY + 39, { color: '#5f6f7a', font: '10px system-ui', maxWidth: w - 96, maxLines: 1, lineHeight: 11 });
    } else {
      drawIconBadge(ctx, 'GEAR', x + 24, dropY + 17, 40, '#ffffff', 'rgba(136,86,197,0.3)');
      drawCanvasText(ctx, 'Drop gear here', x + 78, dropY + 18, { color: '#102033', font: '950 12px system-ui', maxWidth: w - 96, maxLines: 1, lineHeight: 14 });
      drawCanvasText(ctx, 'Click this slot to choose.', x + 78, dropY + 42, { color: '#5f6f7a', font: '10px system-ui', maxWidth: w - 96, maxLines: 1, lineHeight: 11 });
    }
    drawCanvasText(ctx, meta.isLineCatalyst ? 'Attempts the next active line' : shardHint, x + w / 2, dropY + dropH + 12, { color: '#102033', font: '950 12px system-ui', align: 'center', maxWidth: w - 24, maxLines: 1, lineHeight: 14 });
    return true;
  }

  function drawCanvasAttunementReadyStatusPanel(ctx, x, y, w, h, mode, options) {
    const settings = options || {};
    const item = settings.item || null;
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const meta = getCanvasAttunementPromptModeMeta(mode);
    drawRoundRect(ctx, x, y, w, h, 4, '#fff0cf', 'rgba(137,93,39,0.42)');
    drawCanvasText(ctx, 'STATUS', x + w / 2, y + 12, { color: '#8a5d27', font: '950 10px system-ui', align: 'center', maxWidth: w - 16, maxLines: 1, lineHeight: 11 });
    drawCanvasText(ctx, item ? 'READY' : 'PICK', x + w / 2, y + Math.floor(h / 2) - 10, { color: item ? '#8fd66f' : '#65727c', font: '950 17px system-ui', align: 'center', maxWidth: w - 12, lineHeight: 18 });
    drawCanvasText(ctx, item ? meta.isLineCatalyst ? 'Add line' : 'Use prism' : 'Choose gear', x + w / 2, y + Math.floor(h / 2) + 12, { color: '#65727c', font: '950 8px system-ui', align: 'center', maxWidth: w - 12, lineHeight: 9 });
    return true;
  }

  function getCanvasAttunementReadyActionLayout(boxW, stackedLayout) {
    const autoW = stackedLayout ? Math.floor((boxW - 44) / 2) : Math.min(142, Math.floor((boxW - 70) / 4));
    const repeatW = stackedLayout ? Math.floor((boxW - 44) / 2) : Math.min(112, Math.floor((boxW - 70) / 4));
    const actionW = stackedLayout ? boxW - 36 : Math.min(220, Math.floor((boxW - 54) / 2));
    const lineButtonW = Math.min(180, Math.max(132, Math.floor((boxW - 54) / 3)));
    return {
      autoW,
      repeatW,
      actionW,
      lineButtonW
    };
  }

  function drawCanvasAttunementReadyActionBar(ctx, x, y, boxW, actionY, mode, options) {
    const settings = options || {};
    const item = settings.item || null;
    const actionCount = Math.max(0, Math.floor(Number(settings.actionCount || 0) || 0));
    const stackedLayout = typeof settings.stackedLayout === 'boolean' ? settings.stackedLayout : boxW < 680;
    const hasLastPotentialAutoTarget = Boolean(settings.hasLastPotentialAutoTarget);
    const hasAutoTarget = Boolean(settings.hasAutoTarget);
    const shardFooterHint = settings.shardFooterHint || '';
    const lineInfo = settings.lineInfo || {};
    const drawCanvasButton = typeof settings.drawCanvasButton === 'function'
      ? settings.drawCanvasButton
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const getAutoSummaryText = typeof settings.getAutoSummaryText === 'function'
      ? settings.getAutoSummaryText
      : () => shardFooterHint;
    const meta = getCanvasAttunementPromptModeMeta(mode);
    const actionLayout = getCanvasAttunementReadyActionLayout(boxW, stackedLayout);
    const autoW = actionLayout.autoW;
    const repeatW = actionLayout.repeatW;
    const actionW = actionLayout.actionW;
    if (!meta.isLineCatalyst) {
      const autoLabel = meta.isEcho ? 'Auto Echo' : 'Auto Attunement';
      drawCanvasButton(ctx, autoLabel, x + 18, actionY, autoW, 30, { type: 'potential-auto-toggle' }, false);
      drawCanvasButton(ctx, 'Repeat Last', x + 26 + autoW, actionY, repeatW, 30, { type: 'potential-auto-repeat' }, !hasLastPotentialAutoTarget || !item || actionCount <= 0);
      if (!stackedLayout) {
        drawCanvasText(ctx, getAutoSummaryText(), x + 34 + autoW + repeatW, actionY + 8, { color: '#65727c', font: '850 10px system-ui', maxWidth: Math.max(40, boxW - autoW - repeatW - actionW - 68), maxLines: 1, lineHeight: 11 });
      }
    } else {
      const buttonW = actionLayout.lineButtonW;
      drawCanvasButton(ctx, 'Use Line Catalyst', x + 18, actionY, buttonW, 30, { type: 'potential-line-upgrade', uid: item && item.uid || '' }, !item || !lineInfo.canUpgrade);
      drawCanvasText(ctx, `Line Catalysts ${lineInfo.catalystCount}; ${shardFooterHint}`, x + 28 + buttonW, actionY + 8, { color: '#65727c', font: '850 10px system-ui', maxWidth: Math.max(40, boxW - buttonW - 64), maxLines: 1, lineHeight: 11 });
    }
    return { autoW, repeatW, actionW };
  }

  function getCanvasAttunementReadyAutoGoalsLayout(x, boxW, actionY, actionW) {
    const panelW = Math.min(390, Math.max(280, boxW - actionW - 70));
    const panelH = 132;
    const panelX = x + 18;
    const panelY = actionY - panelH - 8;
    const fieldW = Math.floor((panelW - 34) / 2);
    const fieldY = panelY + 44;
    const maxY = fieldY + 38;
    const sliderX = panelX + 12;
    const sliderW = Math.max(96, panelW - 150);
    return {
      panelW,
      panelH,
      panelX,
      panelY,
      fieldW,
      fieldY,
      maxY,
      sliderX,
      sliderW
    };
  }

  function drawCanvasAttunementReadyAutoGoalsPanel(ctx, x, y, boxW, actionY, actionW, mode, options) {
    const settings = options || {};
    const item = settings.item || null;
    const target = settings.target || {};
    const actionCount = Math.max(0, Math.floor(Number(settings.actionCount || 0) || 0));
    const hasAutoTarget = Boolean(settings.hasAutoTarget);
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const drawCanvasButton = typeof settings.drawCanvasButton === 'function'
      ? settings.drawCanvasButton
      : () => {};
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (stat, value) => String(value);
    const getPotentialTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    const meta = getCanvasAttunementPromptModeMeta(mode);
    const layout = getCanvasAttunementReadyAutoGoalsLayout(x, boxW, actionY, actionW);
    const panelW = layout.panelW;
    const panelH = layout.panelH;
    const panelX = layout.panelX;
    const panelY = layout.panelY;
    drawRoundRect(ctx, panelX, panelY, panelW, panelH, 6, '#fff0cf', 'rgba(137,93,39,0.42)');
    drawCanvasText(ctx, 'AUTO GOALS', panelX + 12, panelY + 12, { color: meta.isEcho ? '#8856c5' : '#8a5d27', font: '950 10px system-ui', maxWidth: panelW - 54, maxLines: 1, lineHeight: 11 });
    drawCanvasButton(ctx, 'X', panelX + panelW - 30, panelY + 8, 20, 20, { type: 'potential-auto-close' }, false);
    drawCanvasText(ctx, 'Click fields to cycle goals', panelX + 12, panelY + 28, { color: '#65727c', font: '850 9px system-ui', maxWidth: panelW - 54, maxLines: 1, lineHeight: 10 });
    const fieldW = layout.fieldW;
    const fieldY = layout.fieldY;
    const firstGoal = target.statGoals && target.statGoals[0];
    const statName = firstGoal ? formatStatName(firstGoal.stat)
      .replace('Attack Damage', 'Atk Dmg')
      .replace('Boss Damage', 'Boss Dmg')
      .replace('Cooldown Recovery', 'CD Recovery')
      .replace('Resource Cost Reduction', 'Cost Down') : '';
    const statLabel = firstGoal ? `${statName} >= ${formatStatValue(firstGoal.stat, firstGoal.min)}` : 'Choose stat';
    const tierName = target.tier ? String((getPotentialTierDefinition(target.tier) || {}).name || target.tier).replace(' Attunement', '') : '';
    const tierLabel = target.tier ? `${tierName}+` : 'Choose tier';
    drawCanvasButton(ctx, `Stat: ${statLabel}`, panelX + 12, fieldY, fieldW, 28, { type: 'potential-auto-stat-cycle', delta: 1 }, false);
    drawCanvasButton(ctx, `Tier: ${tierLabel}`, panelX + 22 + fieldW, fieldY, fieldW, 28, { type: 'potential-auto-tier-cycle', delta: 1 }, false);
    const maxY = layout.maxY;
    const sliderX = layout.sliderX;
    const sliderW = layout.sliderW;
    const sliderRatio = actionCount > 0 ? clamp(target.maxRolls / actionCount, 0, 1) : 0;
    drawCanvasText(ctx, `Use ${target.maxRolls}/${actionCount} prisms`, sliderX, maxY + 2, { color: '#102033', font: '900 11px system-ui', maxWidth: sliderW, maxLines: 1, lineHeight: 12 });
    drawRoundRect(ctx, sliderX, maxY + 20, sliderW, 6, 3, 'rgba(137,93,39,0.22)', 'rgba(137,93,39,0.12)');
    drawRoundRect(ctx, sliderX, maxY + 20, Math.max(6, sliderW * sliderRatio), 6, 3, '#8856c5', 'rgba(136,86,197,0.3)');
    drawRoundRect(ctx, sliderX + Math.max(0, sliderW * sliderRatio - 4), maxY + 16, 8, 14, 4, '#fffaf0', 'rgba(136,86,197,0.58)');
    drawCanvasText(ctx, '25-step slider in the prompt', sliderX, maxY + 32, { color: '#65727c', font: '800 8px system-ui', maxWidth: sliderW, maxLines: 1, lineHeight: 9 });
    drawCanvasButton(ctx, meta.isEcho ? 'Start Echo' : 'Start Auto', panelX + panelW - 122, maxY, 110, 28, { type: 'potential-auto-run' }, !item || actionCount <= 0 || !hasAutoTarget || target.maxRolls <= 0);
    return { panelX, panelY, panelW, panelH };
  }

  function getCanvasAttunementReadyFinalActionLayout(x, boxW, actionY, actionW, stackedLayout) {
    const buttonX = stackedLayout ? x + 18 : x + boxW - actionW - 18;
    const buttonY = stackedLayout ? actionY + 38 : actionY;
    return {
      buttonX,
      buttonY,
      buttonW: actionW,
      buttonH: 30
    };
  }

  function drawCanvasAttunementReadyFinalActionButton(ctx, x, y, boxW, actionY, actionW, mode, options) {
    const settings = options || {};
    const item = settings.item || null;
    const actionCount = Math.max(0, Math.floor(Number(settings.actionCount || 0) || 0));
    const stackedLayout = typeof settings.stackedLayout === 'boolean' ? settings.stackedLayout : boxW < 680;
    const drawCanvasButton = typeof settings.drawCanvasButton === 'function'
      ? settings.drawCanvasButton
      : () => {};
    const meta = getCanvasAttunementPromptModeMeta(mode);
    if (meta.isLineCatalyst) return null;
    const layout = getCanvasAttunementReadyFinalActionLayout(x, boxW, actionY, actionW, stackedLayout);
    const label = `Use ${meta.isEcho ? 'Echo Prism' : 'Attunement Prism'} (${actionCount})`;
    const region = meta.isEcho
      ? { type: 'potential-prompt-preserve', uid: item && item.uid || '' }
      : { type: 'potential-prompt-confirm', uid: item && item.uid || '' };
    drawCanvasButton(ctx, label, layout.buttonX, layout.buttonY, layout.buttonW, layout.buttonH, region, !item || actionCount <= 0);
    return { x: layout.buttonX, y: layout.buttonY, w: layout.buttonW, h: layout.buttonH, label, region };
  }

  function getCanvasAttunementReadyBaseLayout(y, boxW, boxH, summaryY, summaryH, isLineCatalyst) {
    const stackedLayout = boxW < 680;
    const actionY = y + boxH - (stackedLayout && !isLineCatalyst ? 82 : 44);
    const tableY = summaryY + summaryH + 14;
    const gap = 10;
    const verdictW = 104;
    return {
      stackedLayout,
      actionY,
      tableY,
      gap,
      verdictW
    };
  }

  function getCanvasAttunementReadyPanelLayout(x, boxW, actionY, tableY, stackedLayout, gap, verdictW) {
    const fullW = boxW - 32;
    const availableH = Math.max(240, actionY - tableY - 10);
    const contentH = Math.max(210, availableH - gap);
    const currentH = Math.max(120, Math.floor(contentH * 0.52));
    const targetH = Math.max(100, contentH - currentH);
    const tableH = Math.max(210, actionY - tableY - 10);
    const sideW = Math.floor((boxW - 32 - verdictW - gap * 2) / 2);
    const statusX = x + 16 + sideW + gap;
    return {
      fullW,
      availableH,
      contentH,
      currentH,
      targetH,
      tableH,
      sideW,
      statusX,
      currentPanelX: x + 16,
      currentPanelY: tableY,
      currentPanelW: stackedLayout ? fullW : sideW,
      currentPanelH: stackedLayout ? currentH : tableH,
      targetPanelX: stackedLayout ? x + 16 : statusX + verdictW + gap,
      targetPanelY: stackedLayout ? tableY + currentH + gap : tableY,
      targetPanelW: stackedLayout ? fullW : sideW,
      targetPanelH: stackedLayout ? targetH : tableH,
      statusPanelX: statusX,
      statusPanelY: tableY,
      statusPanelW: verdictW,
      statusPanelH: tableH
    };
  }

  function getCanvasAttunementReadyLineLayout(x, y, boxW, boxH, summaryY, summaryH) {
    const bodyY = summaryY + summaryH + 12;
    const actionY = y + boxH - 44;
    const bodyH = Math.max(210, actionY - bodyY - 10);
    const stacked = boxW < 560;
    const gap = 10;
    const linePanelW = stacked ? boxW - 32 : Math.floor((boxW - 42) * 0.58);
    const sidePanelW = stacked ? boxW - 32 : boxW - 32 - linePanelW - gap;
    const linePanelX = x + 16;
    const sidePanelX = stacked ? x + 16 : linePanelX + linePanelW + gap;
    const sidePanelY = stacked ? bodyY + Math.max(170, Math.floor(bodyH * 0.56)) + gap : bodyY;
    const sidePanelH = stacked ? Math.max(120, actionY - sidePanelY - 10) : bodyH;
    const linePanelH = stacked ? Math.max(160, sidePanelY - bodyY - gap) : bodyH;
    return {
      bodyY,
      actionY,
      bodyH,
      stacked,
      gap,
      linePanelW,
      sidePanelW,
      linePanelX,
      sidePanelX,
      sidePanelY,
      sidePanelH,
      linePanelH
    };
  }

  function getCanvasAttunementReadyLineUpgradePanelLayout(x, y, w, h) {
    const oddsY = y + 42;
    const oddsW = Math.max(72, Math.floor((w - 34) / 2));
    const detailY = oddsY + 66;
    const footerY = Math.min(y + h - 26, detailY + 34);
    return {
      oddsY,
      oddsW,
      detailY,
      footerY,
      successOddsX: x + 12,
      failOddsX: x + 22 + oddsW
    };
  }

  function drawCanvasAttunementReadyLineUpgradePanel(ctx, x, y, w, h, options) {
    const settings = options || {};
    const lineInfo = settings.lineInfo || {};
    const successChance = Math.max(0, Number(settings.successChance || 0) || 0);
    const failChance = Math.max(0, Number(settings.failChance || 0) || 0);
    const shardFooterHint = settings.shardFooterHint || '';
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const formatChance = typeof settings.formatChance === 'function'
      ? settings.formatChance
      : (value) => `${value}%`;
    drawRoundRect(ctx, x, y, w, h, 4, '#fff0cf', 'rgba(137,93,39,0.42)');
    drawCanvasText(ctx, 'Ready For Line Upgrade', x + 12, y + 12, { color: '#8a5d27', font: '950 10px system-ui', maxWidth: w - 94, maxLines: 1, lineHeight: 11 });
    drawCanvasText(ctx, lineInfo.lineCount >= lineInfo.maxLineCount ? 'Maxed' : `${lineInfo.targetLineCount}/${lineInfo.maxLineCount}`, x + w - 12, y + 12, { color: '#8a5d27', font: '950 9px system-ui', align: 'right', maxWidth: 78, maxLines: 1, lineHeight: 10 });
    const layout = getCanvasAttunementReadyLineUpgradePanelLayout(x, y, w, h);
    const oddsY = layout.oddsY;
    const oddsW = layout.oddsW;
    const detailY = layout.detailY;
    drawRoundRect(ctx, layout.successOddsX, oddsY, oddsW, 50, 5, '#e1f9da', 'rgba(73,150,75,0.38)');
    drawCanvasText(ctx, 'New Line Added', x + 20, oddsY + 9, { color: '#2f7a5d', font: '950 9px system-ui', maxWidth: oddsW - 16, maxLines: 1, lineHeight: 10 });
    drawCanvasText(ctx, formatChance(successChance * 100), x + 20, oddsY + 28, { color: '#1c2631', font: '950 15px system-ui', maxWidth: oddsW - 16, maxLines: 1, lineHeight: 16 });
    drawRoundRect(ctx, layout.failOddsX, oddsY, oddsW, 50, 5, '#ffe8dd', 'rgba(202,86,67,0.3)');
    drawCanvasText(ctx, 'Not Added', x + 30 + oddsW, oddsY + 9, { color: '#9a4d3a', font: '950 9px system-ui', maxWidth: oddsW - 16, maxLines: 1, lineHeight: 10 });
    drawCanvasText(ctx, formatChance(failChance * 100), x + 30 + oddsW, oddsY + 28, { color: '#1c2631', font: '950 15px system-ui', maxWidth: oddsW - 16, maxLines: 1, lineHeight: 16 });
    drawCanvasText(ctx, lineInfo.lineCount > 1 ? 'Failure does not add a line and hides one active line.' : 'Failure does not add a line.', x + 12, detailY, { color: '#65727c', font: '850 10px system-ui', maxWidth: w - 24, maxLines: 2, lineHeight: 12 });
    drawCanvasText(ctx, `Line Catalysts ${lineInfo.catalystCount}; ${shardFooterHint}`, x + 12, layout.footerY, { color: '#65727c', font: '850 10px system-ui', maxWidth: w - 24, maxLines: 1, lineHeight: 11 });
    return { oddsY, oddsW, detailY };
  }

  function getCanvasAttunementReadyLineActionLayout(x, boxW, actionY) {
    const buttonW = Math.min(190, Math.max(150, Math.floor((boxW - 54) / 3)));
    return {
      buttonX: x + 18,
      buttonY: actionY,
      buttonW,
      buttonH: 30,
      statusX: x + 28 + buttonW,
      statusY: actionY + 8,
      statusMaxW: Math.max(40, boxW - buttonW - 64)
    };
  }

  function drawCanvasAttunementReadyLineActionBar(ctx, x, y, boxW, actionY, options) {
    const settings = options || {};
    const item = settings.item || null;
    const lineInfo = settings.lineInfo || {};
    const successChance = Math.max(0, Number(settings.successChance || 0) || 0);
    const drawCanvasButton = typeof settings.drawCanvasButton === 'function'
      ? settings.drawCanvasButton
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const formatChance = typeof settings.formatChance === 'function'
      ? settings.formatChance
      : (value) => `${value}%`;
    const layout = getCanvasAttunementReadyLineActionLayout(x, boxW, actionY);
    const buttonW = layout.buttonW;
    drawCanvasButton(ctx, 'Use Line Catalyst', layout.buttonX, layout.buttonY, buttonW, layout.buttonH, { type: 'potential-line-upgrade', uid: item && item.uid || '' }, !item || !lineInfo.canUpgrade);
    const statusText = `${lineInfo.lineCount}/${lineInfo.maxLineCount} current; ${lineInfo.lineCount < lineInfo.maxLineCount ? `${formatChance(successChance * 100)} add` : 'maxed'}`;
    drawCanvasText(ctx, statusText, layout.statusX, layout.statusY, { color: '#65727c', font: '850 10px system-ui', maxWidth: layout.statusMaxW, maxLines: 1, lineHeight: 11 });
    return { buttonW, statusText };
  }

  function getCanvasAttunementReadyLineCurrentPanelLayout(x, y, w, h, rowCount) {
    const rowH = Math.max(23, Math.min(30, Math.floor((h - 78) / Math.max(1, rowCount || 1))));
    const rowStartY = y + 38;
    const rowClipY = y + h - 38;
    const lineTotalY = y + h - 38;
    return {
      rowH,
      rowStartY,
      rowClipY,
      lineTotalY,
      rowFillX: x + 1,
      rowFillW: w - 2,
      footerH: 37
    };
  }

  function drawCanvasAttunementReadyLineCurrentPanel(ctx, x, y, w, h, options) {
    const settings = options || {};
    const rows = Array.isArray(settings.rows) ? settings.rows : [];
    const lineInfo = settings.lineInfo || {};
    const drawRoundRect = typeof settings.drawRoundRect === 'function'
      ? settings.drawRoundRect
      : () => {};
    const drawCanvasText = typeof settings.drawCanvasText === 'function'
      ? settings.drawCanvasText
      : () => 0;
    const getPotentialRollRank = typeof settings.getPotentialRollRank === 'function'
      ? settings.getPotentialRollRank
      : () => ({ label: '', percentLabel: '--', fill: '#ffffff', color: '#1c2631' });
    const getPotentialStatAbbreviation = typeof settings.getPotentialStatAbbreviation === 'function'
      ? settings.getPotentialStatAbbreviation
      : (stat) => String(stat || '').slice(0, 2).toUpperCase();
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const formatStatValue = typeof settings.formatStatValue === 'function'
      ? settings.formatStatValue
      : (stat, value) => String(value);
    drawRoundRect(ctx, x, y, w, h, 4, '#fff0cf', 'rgba(137,93,39,0.42)');
    drawCanvasText(ctx, 'Current Lines', x + 12, y + 12, { color: '#8a5d27', font: '950 10px system-ui', maxWidth: w - 94, maxLines: 1, lineHeight: 11 });
    drawCanvasText(ctx, `${lineInfo.lineCount}/${lineInfo.maxLineCount}`, x + w - 12, y + 12, { color: '#8a5d27', font: '950 9px system-ui', align: 'right', maxWidth: 78, maxLines: 1, lineHeight: 10 });
    const layout = getCanvasAttunementReadyLineCurrentPanelLayout(x, y, w, h, rows.length);
    const rowH = layout.rowH;
    if (rows.length) {
      rows.forEach((row, index) => {
        const rowY = layout.rowStartY + index * rowH;
        const rank = getPotentialRollRank(row.percentileLabel);
        if (rowY + rowH > layout.rowClipY) return;
        if (ctx && ctx.fillRect) {
          ctx.fillStyle = index % 2 ? 'rgba(47,122,93,0.035)' : 'rgba(255,255,255,0.22)';
          ctx.fillRect(layout.rowFillX, rowY, layout.rowFillW, rowH);
        }
        drawRoundRect(ctx, x + 8, rowY + Math.max(3, Math.floor((rowH - 20) / 2)), 20, 20, 10, 'rgba(47,122,93,0.1)', 'rgba(47,122,93,0.22)');
        drawCanvasText(ctx, getPotentialStatAbbreviation(row.stat), x + 18, rowY + Math.max(7, Math.floor((rowH - 8) / 2)), { color: '#2f7a5d', font: '950 7px system-ui', align: 'center', maxWidth: 16, lineHeight: 8 });
        drawCanvasText(ctx, formatStatName(row.stat), x + 36, rowY + Math.max(7, Math.floor((rowH - 10) / 2)), { color: '#1c2631', font: '850 10px system-ui', maxWidth: Math.max(52, w - 142), maxLines: 1, lineHeight: 11 });
        drawCanvasText(ctx, formatStatValue(row.stat, row.value), x + w - 68, rowY + Math.max(6, Math.floor((rowH - 10) / 2)), { color: '#1c2631', font: '950 10px system-ui', align: 'right', maxWidth: 58, maxLines: 1, lineHeight: 11 });
        drawRoundRect(ctx, x + w - 58, rowY + Math.max(5, Math.floor((rowH - 16) / 2)), 30, 16, 3, rank.fill, 'rgba(47,122,93,0.26)');
        drawCanvasText(ctx, rank.label, x + w - 43, rowY + Math.max(8, Math.floor((rowH - 9) / 2)), { color: rank.color, font: '950 9px system-ui', align: 'center', maxWidth: 22, maxLines: 1, lineHeight: 10 });
        drawCanvasText(ctx, rank.percentLabel || '--', x + w - 10, rowY + Math.max(8, Math.floor((rowH - 9) / 2)), { color: '#65727c', font: '850 8px system-ui', align: 'right', maxWidth: 34, maxLines: 1, lineHeight: 9 });
      });
    } else {
      drawCanvasText(ctx, 'No attunement stats', x + 12, y + 54, { color: '#65727c', font: '10px system-ui', maxWidth: w - 24, maxLines: 1, lineHeight: 11 });
    }
    const lineTotalY = layout.lineTotalY;
    if (ctx && ctx.fillRect) {
      ctx.fillStyle = 'rgba(47,122,93,0.08)';
      ctx.fillRect(layout.rowFillX, lineTotalY, layout.rowFillW, layout.footerH);
    }
    drawCanvasText(ctx, 'NUMBER OF LINES', x + 12, lineTotalY + 12, { color: '#2f7a5d', font: '950 10px system-ui', maxWidth: w - 118, maxLines: 1, lineHeight: 11 });
    drawCanvasText(ctx, `${lineInfo.lineCount}/${lineInfo.maxLineCount}`, x + w - 12, lineTotalY + 11, { color: '#1c2631', font: '950 12px system-ui', align: 'right', maxWidth: 78, maxLines: 1, lineHeight: 13 });
    return { rowH, lineTotalY };
  }

  function getPotentialTierLabel(item, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    const getPotentialTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    const potential = normalize(item && item.potential, item, settings);
    const tier = potential ? getPotentialTierDefinition(potential.tier) : null;
    return tier ? tier.name : 'No Attunement';
  }

  function formatPotentialTierChance(value, options) {
    const formatChance = options && typeof options.formatChance === 'function'
      ? options.formatChance
      : (amount) => `${amount}%`;
    return formatChance(Math.max(0, Number(value || 0)) * 100);
  }

  function getPotentialTierAdvanceLabel(item, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    const getPotentialTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    const formatTierChance = typeof settings.formatPotentialTierChance === 'function'
      ? settings.formatPotentialTierChance
      : (value) => formatPotentialTierChance(value, settings);
    const potential = normalize(item && item.potential, item, settings);
    if (!potential) {
      const firstTier = getPotentialTierDefinition('rare');
      return `First roll: ${firstTier ? firstTier.name : 'Rare Attunement'} (100%)`;
    }
    const tier = getPotentialTierDefinition(potential.tier);
    const nextTier = tier && tier.nextTier ? getPotentialTierDefinition(tier.nextTier) : null;
    if (!tier || !nextTier || !Number(tier.tierUpChance || 0)) return 'Max tier reached';
    return `Next tier: ${nextTier.name} (${formatTierChance(tier.tierUpChance)})`;
  }

  function getPotentialTierRollSummary(item, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    const getPotentialTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    const formatTierChance = typeof settings.formatPotentialTierChance === 'function'
      ? settings.formatPotentialTierChance
      : (value) => formatPotentialTierChance(value, settings);
    const potential = normalize(item && item.potential, item, settings);
    if (!potential) return 'First roll: Rare Attunement 100% - 1 active line';
    const tier = getPotentialTierDefinition(potential.tier);
    const nextTier = tier && tier.nextTier ? getPotentialTierDefinition(tier.nextTier) : null;
    if (!tier || !nextTier || !Number(tier.tierUpChance || 0)) return `${tier ? tier.name : 'Current tier'}: max tier reached`;
    const tierUpChance = Math.max(0, Number(tier.tierUpChance || 0));
    return `${tier.name}: ${(100 - tierUpChance * 100).toFixed(tierUpChance < 0.01 ? 2 : 1).replace(/\.0+$/, '')}% stay, ${formatTierChance(tierUpChance)} -> ${nextTier.name}`;
  }

  function getPotentialLineUpgradeChanceForTarget(targetLineCount, options) {
    const settings = options || {};
    const chances = settings.lineUpgradeChances || POTENTIAL_LINE_UPGRADE_CHANCES;
    const target = normalizePotentialLineCount(targetLineCount, settings);
    return clamp(Number(chances[target] || 0), 0, 1);
  }

  function getPotentialLineUpgradeInfo(item, snapshot, options) {
    const settings = options || {};
    const normalize = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : normalizeItemPotential;
    const getLineUpgradeChance = typeof settings.getPotentialLineUpgradeChanceForTarget === 'function'
      ? settings.getPotentialLineUpgradeChanceForTarget
      : (targetLineCount) => getPotentialLineUpgradeChanceForTarget(targetLineCount, settings);
    const maxLineCount = Math.max(1, Math.floor(Number(settings.maxLineCount || POTENTIAL_MAX_LINE_COUNT) || POTENTIAL_MAX_LINE_COUNT));
    const downgradeChance = settings.downgradeChance == null ? POTENTIAL_LINE_DOWNGRADE_CHANCE_ON_FAIL : Number(settings.downgradeChance);
    const potential = normalize(item && item.potential, item, settings);
    const consumables = snapshot && snapshot.state && snapshot.state.consumables || {};
    const catalystCount = Math.max(0, Math.floor(Number(consumables.line_catalyst || 0) || 0));
    const lineCount = potential ? potential.lineCount : 0;
    const targetLineCount = potential && lineCount < maxLineCount ? lineCount + 1 : lineCount;
    const chance = potential && lineCount < maxLineCount ? getLineUpgradeChance(targetLineCount) : 0;
    return {
      hasPotential: !!potential,
      lineCount,
      targetLineCount,
      maxLineCount,
      catalystCount,
      chance,
      downgradeChance,
      canUpgrade: !!(item && item.slot && potential && lineCount < maxLineCount && catalystCount > 0)
    };
  }

  function getPotentialEligibleLinePools(tierId, item, options) {
    const settings = options || {};
    const getPotentialTierDefinition = typeof settings.getPotentialTierDefinition === 'function'
      ? settings.getPotentialTierDefinition
      : () => null;
    const isValidForTier = typeof settings.isPotentialLinePoolValidForTier === 'function'
      ? settings.isPotentialLinePoolValidForTier
      : isPotentialLinePoolValidForTier;
    const isValidForSlot = typeof settings.isPotentialLinePoolValidForSlot === 'function'
      ? settings.isPotentialLinePoolValidForSlot
      : isPotentialLinePoolValidForSlot;
    const tier = getPotentialTierDefinition(tierId);
    if (!tier) return [];
    return (Array.isArray(settings.potentialLinePools) ? settings.potentialLinePools : [])
      .filter((pool) => isValidForTier(pool, tier.id) && (!item || isValidForSlot(pool, item.slot)));
  }

  function getPotentialRangeLabel(pool, tierId, options) {
    const formatStatValue = options && typeof options.formatStatValue === 'function'
      ? options.formatStatValue
      : (key, value) => String(value);
    const values = pool && pool.values && pool.values[tierId] || [];
    if (!values.length) return '-';
    return `${formatStatValue(pool.stat, values[0])} to ${formatStatValue(pool.stat, values[1])}`;
  }

  function getPotentialSlotRestrictionLabel(pool) {
    const slots = Array.isArray(pool && pool.slots) ? pool.slots : [];
    return slots.length ? slots.map((slot) => slot.replace(/\b\w/g, (char) => char.toUpperCase())).join(', ') : 'Any gear';
  }

  function getPotentialHelpRows(item, options) {
    const settings = options || {};
    const formatChance = typeof settings.formatChance === 'function'
      ? settings.formatChance
      : (amount) => `${amount}%`;
    const formatStatName = typeof settings.formatStatName === 'function'
      ? settings.formatStatName
      : (stat) => String(stat || '');
    const getEligibleLinePools = typeof settings.getPotentialEligibleLinePools === 'function'
      ? settings.getPotentialEligibleLinePools
      : (tierId, targetItem) => getPotentialEligibleLinePools(tierId, targetItem, settings);
    const getRangeLabel = typeof settings.getPotentialRangeLabel === 'function'
      ? settings.getPotentialRangeLabel
      : (pool, tierId) => getPotentialRangeLabel(pool, tierId, settings);
    const getSlotRestrictionLabel = typeof settings.getPotentialSlotRestrictionLabel === 'function'
      ? settings.getPotentialSlotRestrictionLabel
      : getPotentialSlotRestrictionLabel;
    const tiers = Array.isArray(settings.potentialTiers) ? settings.potentialTiers : [];
    const rows = [];
    tiers.forEach((tier) => {
      const pools = getEligibleLinePools(tier.id, item);
      const chance = pools.length ? formatChance(100 / pools.length) : '0%';
      pools.forEach((pool) => {
        rows.push({
          tier,
          stat: formatStatName(pool.stat),
          range: getRangeLabel(pool, tier.id),
          chance,
          slot: getSlotRestrictionLabel(pool)
        });
      });
    });
    return rows;
  }

  function createAttunementPotentialUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      maxLineCount: settings.maxLineCount || POTENTIAL_MAX_LINE_COUNT,
      lineUpgradeChances: settings.lineUpgradeChances || POTENTIAL_LINE_UPGRADE_CHANCES,
      downgradeChance: settings.downgradeChance == null ? POTENTIAL_LINE_DOWNGRADE_CHANCE_ON_FAIL : settings.downgradeChance,
      potentialTiers: settings.potentialTiers || [],
      potentialLinePools: settings.potentialLinePools || [],
      getPotentialTierDefinition: typeof settings.getPotentialTierDefinition === 'function' ? settings.getPotentialTierDefinition : () => null,
      getPotentialTierRank: typeof settings.getPotentialTierRank === 'function' ? settings.getPotentialTierRank : () => -1,
      getPotentialLinePoolForStat: typeof settings.getPotentialLinePoolForStat === 'function' ? settings.getPotentialLinePoolForStat : () => null,
      getPotentialRollRank: typeof settings.getPotentialRollRank === 'function' ? settings.getPotentialRollRank : () => null,
      getPotentialComparisonDeltaClass: typeof settings.getPotentialComparisonDeltaClass === 'function' ? settings.getPotentialComparisonDeltaClass : (delta) => Number(delta || 0) > 0 ? 'is-gain' : Number(delta || 0) < 0 ? 'is-loss' : 'is-same',
      getPotentialComparisonLineIndex: typeof settings.getPotentialComparisonLineIndex === 'function' ? settings.getPotentialComparisonLineIndex : getPotentialComparisonLineIndexFallback,
      getPotentialStatAbbreviation: typeof settings.getPotentialStatAbbreviation === 'function' ? settings.getPotentialStatAbbreviation : (stat) => String(stat || '').slice(0, 2).toUpperCase(),
      getConsumableItem: typeof settings.getConsumableItem === 'function' ? settings.getConsumableItem : () => null,
      getItemStrength: typeof settings.getItemStrength === 'function' ? settings.getItemStrength : () => 0,
      getAttunementStatIconMeta: typeof settings.getAttunementStatIconMeta === 'function' ? settings.getAttunementStatIconMeta : () => ({ color: '#8a5d27', bg: '#fff0cf', svg: '' }),
      getAttunementStatTooltip: typeof settings.getAttunementStatTooltip === 'function' ? settings.getAttunementStatTooltip : () => '',
      formatStatName: typeof settings.formatStatName === 'function' ? settings.formatStatName : (stat) => String(stat || ''),
      formatStatValue: typeof settings.formatStatValue === 'function' ? settings.formatStatValue : (key, value) => String(value),
      formatChance: typeof settings.formatChance === 'function' ? settings.formatChance : (amount) => `${amount}%`,
      escapeHtml: typeof settings.escapeHtml === 'function' ? settings.escapeHtml : (value) => String(value == null ? '' : value)
    });
    let helpers = null;
    function getOptions(extra) {
      return Object.assign({}, helperOptions, extra || {}, {
        isPotentialLinePoolValidForSlot: helpers.isPotentialLinePoolValidForSlot,
        isPotentialLinePoolValidForTier: helpers.isPotentialLinePoolValidForTier,
        isPotentialLinePoolEligible: helpers.isPotentialLinePoolEligible,
        normalizeItemPotential: helpers.normalizeItemPotential,
        getItemWithPotential: helpers.getItemWithPotential,
        getPotentialLineRollRows: helpers.getPotentialLineRollRows,
        getPotentialLineRowMap: helpers.getPotentialLineRowMap,
        getPotentialComparisonSide: helpers.getPotentialComparisonSide,
        getAttunementPotentialComparison: helpers.getAttunementPotentialComparison,
        getPotentialLineUpgradeChanceForTarget: helpers.getPotentialLineUpgradeChanceForTarget,
        getPotentialEligibleLinePools: helpers.getPotentialEligibleLinePools,
        getPotentialRangeLabel: helpers.getPotentialRangeLabel,
        getPotentialSlotRestrictionLabel: helpers.getPotentialSlotRestrictionLabel
      });
    }
    helpers = Object.freeze({
      isPotentialLinePoolValidForSlot,
      isPotentialLinePoolValidForTier,
      isPotentialLinePoolEligible,
      normalizePotentialLine: (line, tierId, item, extraOptions) => normalizePotentialLine(line, tierId, item, getOptions(extraOptions)),
      normalizePotentialLineCount: (value, extraOptions) => normalizePotentialLineCount(value, getOptions(extraOptions)),
      normalizePotentialLineArchive: (archive, tierId, item, lineCount, extraOptions) => normalizePotentialLineArchive(archive, tierId, item, lineCount, getOptions(extraOptions)),
      normalizeItemPotential: (potential, item, extraOptions) => normalizeItemPotential(potential, item, getOptions(extraOptions)),
      getItemPotentialStats: (item, extraOptions) => getItemPotentialStats(item, getOptions(extraOptions)),
      getPotentialLineSummary: (item, fallback, extraOptions) => getPotentialLineSummary(item, fallback, getOptions(extraOptions)),
      getPotentialLineRollRows: (item, extraOptions) => getPotentialLineRollRows(item, getOptions(extraOptions)),
      getPotentialSummaryFromPotential: (potential, fallback, item, extraOptions) => getPotentialSummaryFromPotential(potential, fallback, item, getOptions(extraOptions)),
      getItemWithPotential: (item, potential, extraOptions) => getItemWithPotential(item, potential, getOptions(extraOptions)),
      cloneItemPotentialForPrompt: (potential, item, extraOptions) => cloneItemPotentialForPrompt(potential, item, getOptions(extraOptions)),
      getPotentialLineRowMap: (potential, item, extraOptions) => getPotentialLineRowMap(potential, item, getOptions(extraOptions)),
      getPotentialComparisonSide: (potential, item, extraOptions) => getPotentialComparisonSide(potential, item, getOptions(extraOptions)),
      getAttunementPotentialComparison: (item, choice, extraOptions) => getAttunementPotentialComparison(item, choice, getOptions(extraOptions)),
      getEchoPotentialComparison: (item, pendingChoice, extraOptions) => getEchoPotentialComparison(item, pendingChoice, getOptions(extraOptions)),
      renderAttunementStatIconMarkup: (stat, extraOptions) => renderAttunementStatIconMarkup(stat, getOptions(extraOptions)),
      renderAttunementPrismIcon: (mode, extraOptions) => renderAttunementPrismIcon(mode, getOptions(extraOptions)),
      renderEchoRankMarkup: (rowData, extraOptions) => renderEchoRankMarkup(rowData, getOptions(extraOptions)),
      renderEchoVerdictRows: (comparison, extraOptions) => renderEchoVerdictRows(comparison, getOptions(extraOptions)),
      renderEchoComparisonRows: (comparison, side, extraOptions) => renderEchoComparisonRows(comparison, side, getOptions(extraOptions)),
      renderAttunementCurrentRows: (item, extraOptions) => renderAttunementCurrentRows(item, getOptions(extraOptions)),
      renderLineCatalystDetails: (item, lineInfo, extraOptions) => renderLineCatalystDetails(item, lineInfo, getOptions(extraOptions)),
      renderAttunementTargetPanel: (item, mode, extraOptions) => renderAttunementTargetPanel(item, mode, getOptions(extraOptions)),
      renderPotentialAutoControls: (item, mode, prismCount, hasPendingChoice, extraOptions) => renderPotentialAutoControls(item, mode, prismCount, hasPendingChoice, getOptions(extraOptions)),
      renderAttunementEmptyPrompt: (mode, extraOptions) => renderAttunementEmptyPrompt(mode, getOptions(extraOptions)),
      renderAttunementComparisonPrompt: (item, choice, extraOptions) => renderAttunementComparisonPrompt(item, choice, getOptions(extraOptions)),
      renderEchoPotentialChoicePrompt: (item, pendingChoice, extraOptions) => renderEchoPotentialChoicePrompt(item, pendingChoice, getOptions(extraOptions)),
      getPotentialPromptConsumableId: (consumableId, extraOptions) => getPotentialPromptConsumableId(consumableId, getOptions(extraOptions)),
      getPotentialPromptChoiceCacheKey,
      getPotentialPromptComparisonCacheKey: (item, mode, choice, extraOptions) => getPotentialPromptComparisonCacheKey(item, mode, choice, getOptions(extraOptions)),
      getPotentialPromptRenderState: (item, extraOptions) => getPotentialPromptRenderState(item, getOptions(extraOptions)),
      renderPotentialPrompt: (item, extraOptions) => renderPotentialPrompt(item, getOptions(extraOptions)),
      getShardCraftRecipeConfig: (recipe, extraOptions) => getShardCraftRecipeConfig(recipe, getOptions(extraOptions)),
      getShardCraftMaxQuantity: (recipe, extraOptions) => getShardCraftMaxQuantity(recipe, getOptions(extraOptions)),
      getShardCraftRecipeSelectionState: (recipe, extraOptions) => getShardCraftRecipeSelectionState(recipe, getOptions(extraOptions)),
      getShardCraftPromptOpenState: (extraOptions) => getShardCraftPromptOpenState(getOptions(extraOptions)),
      getShardCraftPromptCloseState: (extraOptions) => getShardCraftPromptCloseState(getOptions(extraOptions)),
      normalizeShardCraftPromptQuantity,
      getShardCraftQuantityAdjustment,
      normalizeShardCraftCombineQuantity,
      getShardCraftConfirmAction: (recipe, extraOptions) => getShardCraftConfirmAction(recipe, getOptions(extraOptions)),
      getShardCraftPromptRenderState: (extraOptions) => getShardCraftPromptRenderState(getOptions(extraOptions)),
      renderShardCraftPrompt: (extraOptions) => renderShardCraftPrompt(getOptions(extraOptions)),
      getCanvasShardCraftPromptLayout: (x, y, w, h) => getCanvasShardCraftPromptLayout(x, y, w, h),
      drawCanvasShardCraftPrompt: (ctx, width, height, extraOptions) => drawCanvasShardCraftPrompt(ctx, width, height, getOptions(extraOptions)),
      getCanvasPotentialPromptRenderState: (width, height, extraOptions) => getCanvasPotentialPromptRenderState(width, height, getOptions(extraOptions)),
      getCanvasPotentialHelpLayout: (x, y, boxW, boxH) => getCanvasPotentialHelpLayout(x, y, boxW, boxH),
      drawCanvasPotentialHelp: (ctx, width, height, extraOptions) => drawCanvasPotentialHelp(ctx, width, height, getOptions(extraOptions)),
      drawCanvasAttunementPromptChrome: (ctx, x, y, boxW, boxH, mode, subtitle, extraOptions) => drawCanvasAttunementPromptChrome(ctx, x, y, boxW, boxH, mode, subtitle, getOptions(extraOptions)),
      drawCanvasAttunementEmptyPrompt: (ctx, x, y, boxW, boxH, mode, extraOptions) => drawCanvasAttunementEmptyPrompt(ctx, x, y, boxW, boxH, mode, getOptions(extraOptions)),
      drawCanvasAttunementReadySummary: (ctx, x, y, boxW, mode, extraOptions) => drawCanvasAttunementReadySummary(ctx, x, y, boxW, mode, getOptions(extraOptions)),
      getCanvasAttunementReadyCurrentPanelLayout: (x, y, w, h) => getCanvasAttunementReadyCurrentPanelLayout(x, y, w, h),
      drawCanvasAttunementReadyCurrentPanel: (ctx, x, y, w, h, extraOptions) => drawCanvasAttunementReadyCurrentPanel(ctx, x, y, w, h, getOptions(extraOptions)),
      getCanvasAttunementReadyTargetPanelLayout: (x, y, w, h) => getCanvasAttunementReadyTargetPanelLayout(x, y, w, h),
      drawCanvasAttunementReadyTargetPanel: (ctx, x, y, w, h, mode, extraOptions) => drawCanvasAttunementReadyTargetPanel(ctx, x, y, w, h, mode, getOptions(extraOptions)),
      drawCanvasAttunementReadyStatusPanel: (ctx, x, y, w, h, mode, extraOptions) => drawCanvasAttunementReadyStatusPanel(ctx, x, y, w, h, mode, getOptions(extraOptions)),
      getCanvasAttunementReadyActionLayout: (boxW, stackedLayout) => getCanvasAttunementReadyActionLayout(boxW, stackedLayout),
      drawCanvasAttunementReadyActionBar: (ctx, x, y, boxW, actionY, mode, extraOptions) => drawCanvasAttunementReadyActionBar(ctx, x, y, boxW, actionY, mode, getOptions(extraOptions)),
      getCanvasAttunementReadyAutoGoalsLayout: (x, boxW, actionY, actionW) => getCanvasAttunementReadyAutoGoalsLayout(x, boxW, actionY, actionW),
      drawCanvasAttunementReadyAutoGoalsPanel: (ctx, x, y, boxW, actionY, actionW, mode, extraOptions) => drawCanvasAttunementReadyAutoGoalsPanel(ctx, x, y, boxW, actionY, actionW, mode, getOptions(extraOptions)),
      getCanvasAttunementReadyFinalActionLayout: (x, boxW, actionY, actionW, stackedLayout) => getCanvasAttunementReadyFinalActionLayout(x, boxW, actionY, actionW, stackedLayout),
      drawCanvasAttunementReadyFinalActionButton: (ctx, x, y, boxW, actionY, actionW, mode, extraOptions) => drawCanvasAttunementReadyFinalActionButton(ctx, x, y, boxW, actionY, actionW, mode, getOptions(extraOptions)),
      getCanvasAttunementReadyBaseLayout: (y, boxW, boxH, summaryY, summaryH, isLineCatalyst) => getCanvasAttunementReadyBaseLayout(y, boxW, boxH, summaryY, summaryH, isLineCatalyst),
      getCanvasAttunementReadyPanelLayout: (x, boxW, actionY, tableY, stackedLayout, gap, verdictW) => getCanvasAttunementReadyPanelLayout(x, boxW, actionY, tableY, stackedLayout, gap, verdictW),
      getCanvasAttunementReadyLineLayout: (x, y, boxW, boxH, summaryY, summaryH) => getCanvasAttunementReadyLineLayout(x, y, boxW, boxH, summaryY, summaryH),
      getCanvasAttunementReadyLineUpgradePanelLayout: (x, y, w, h) => getCanvasAttunementReadyLineUpgradePanelLayout(x, y, w, h),
      drawCanvasAttunementReadyLineUpgradePanel: (ctx, x, y, w, h, extraOptions) => drawCanvasAttunementReadyLineUpgradePanel(ctx, x, y, w, h, getOptions(extraOptions)),
      getCanvasAttunementReadyLineActionLayout: (x, boxW, actionY) => getCanvasAttunementReadyLineActionLayout(x, boxW, actionY),
      drawCanvasAttunementReadyLineActionBar: (ctx, x, y, boxW, actionY, extraOptions) => drawCanvasAttunementReadyLineActionBar(ctx, x, y, boxW, actionY, getOptions(extraOptions)),
      getCanvasAttunementReadyLineCurrentPanelLayout: (x, y, w, h, rowCount) => getCanvasAttunementReadyLineCurrentPanelLayout(x, y, w, h, rowCount),
      drawCanvasAttunementReadyLineCurrentPanel: (ctx, x, y, w, h, extraOptions) => drawCanvasAttunementReadyLineCurrentPanel(ctx, x, y, w, h, getOptions(extraOptions)),
      getPotentialTierLabel: (item, extraOptions) => getPotentialTierLabel(item, getOptions(extraOptions)),
      formatPotentialTierChance: (value, extraOptions) => formatPotentialTierChance(value, getOptions(extraOptions)),
      getPotentialTierAdvanceLabel: (item, extraOptions) => getPotentialTierAdvanceLabel(item, getOptions(extraOptions)),
      getPotentialTierRollSummary: (item, extraOptions) => getPotentialTierRollSummary(item, getOptions(extraOptions)),
      getPotentialLineUpgradeChanceForTarget: (targetLineCount, extraOptions) => getPotentialLineUpgradeChanceForTarget(targetLineCount, getOptions(extraOptions)),
      getPotentialLineUpgradeInfo: (item, snapshot, extraOptions) => getPotentialLineUpgradeInfo(item, snapshot, getOptions(extraOptions)),
      getPotentialEligibleLinePools: (tierId, item, extraOptions) => getPotentialEligibleLinePools(tierId, item, getOptions(extraOptions)),
      getPotentialRangeLabel: (pool, tierId, extraOptions) => getPotentialRangeLabel(pool, tierId, getOptions(extraOptions)),
      getPotentialSlotRestrictionLabel,
      getPotentialHelpRows: (item, extraOptions) => getPotentialHelpRows(item, getOptions(extraOptions)),
      getPotentialPromptItem: (uid, extraOptions) => getPotentialPromptItem(uid, getOptions(extraOptions)),
      getPotentialPromptTargetSelectionState: (uid, extraOptions) => getPotentialPromptTargetSelectionState(uid, getOptions(extraOptions)),
      getPotentialPromptOpenState: (extraOptions) => getPotentialPromptOpenState(getOptions(extraOptions)),
      getPotentialPromptCloseState: (extraOptions) => getPotentialPromptCloseState(getOptions(extraOptions)),
      getPotentialHelpToggleState: (open, extraOptions) => getPotentialHelpToggleState(open, getOptions(extraOptions)),
      getPotentialPromptResultClearState: (uid, extraOptions) => getPotentialPromptResultClearState(uid, getOptions(extraOptions)),
      getPotentialPromptResult: (item, extraOptions) => getPotentialPromptResult(item, getOptions(extraOptions)),
      getPotentialPromptRerollResultState: (originalItem, updatedItem, extraOptions) => getPotentialPromptRerollResultState(originalItem, updatedItem, getOptions(extraOptions)),
      getPotentialPromptComparisonSelection: (item, extraOptions) => getPotentialPromptComparisonSelection(item, getOptions(extraOptions)),
      getPotentialPromptComparisonState: (item, extraOptions) => getPotentialPromptComparisonState(item, getOptions(extraOptions)),
      getPotentialPromptBox: (width, height, state, extraOptions) => getPotentialPromptBox(width, height, state, getOptions(extraOptions)),
      getPotentialHelpBox: (width, height, state, extraOptions) => getPotentialHelpBox(width, height, state, getOptions(extraOptions)),
      getShardCraftPromptBox: (width, height, state, extraOptions) => getShardCraftPromptBox(width, height, state, getOptions(extraOptions)),
      getPotentialDropZoneDomAction,
      getPotentialPromptShellDomAction,
      getShardCraftPromptDomAction,
      getPotentialPromptRollDomAction,
      getPotentialAutoDomAction,
      getPotentialAutoInputDomAction,
      getPotentialChoiceDomAction,
      getPotentialPromptDomDragHandle,
      getPotentialPromptDomDragStartAction: (event, extraOptions) => getPotentialPromptDomDragStartAction(event, getOptions(extraOptions)),
      getPotentialPromptDomDragMoveAction: (event, drag, extraOptions) => getPotentialPromptDomDragMoveAction(event, drag, getOptions(extraOptions)),
      getPotentialPromptDomDragStopAction,
      getAttunementPromptPointerAction,
      getAttunementPromptRegionAction
    });
    return helpers;
  }

  function getPotentialPromptItem(uid, options) {
    const settings = options || {};
    const id = String(uid || settings.potentialPromptUid || '');
    const snapshot = settings.snapshot || null;
    const getGearItem = typeof settings.getGearItemByUid === 'function'
      ? settings.getGearItemByUid
      : () => null;
    if (!id || !snapshot) return null;
    return getGearItem(id);
  }

  function getPotentialPromptTargetSelectionState(uid, options) {
    const settings = options || {};
    const id = String(uid || '');
    const snapshot = settings.snapshot || null;
    const getGearItem = typeof settings.getGearItemByUid === 'function'
      ? settings.getGearItemByUid
      : () => null;
    const item = id && snapshot ? getGearItem(id) : null;
    return {
      id,
      item,
      isValid: !!(item && item.slot)
    };
  }

  function getPotentialPromptOpenState(options) {
    const settings = options || {};
    return {
      potentialPromptConsumableId: getPotentialPromptConsumableId(settings.consumableId, settings),
      potentialPromptOpen: true,
      potentialAutoPanelOpen: false,
      isCommandOpen: false,
      shouldClearResult: true,
      potentialPromptUid: '',
      pendingInventoryDrop: null,
      dropQuantityPrompt: null,
      canvasInventoryClick: null,
      canvasHoverTarget: { type: '', key: '' }
    };
  }

  function getPotentialPromptCloseState(options) {
    const settings = options || {};
    const shouldClose = !!settings.potentialPromptOpen;
    return {
      shouldClose,
      potentialPromptOpen: false,
      potentialAutoPanelOpen: false,
      potentialPromptUid: '',
      shouldClearResult: shouldClose,
      potentialHelpOpen: false,
      potentialPromptDrag: null,
      potentialPromptDomDrag: null,
      canvasGearDrag: null,
      canvasInventoryDrag: null,
      canvasHoverTarget: { type: '', key: '' },
      shouldCloseGearPicker: shouldClose
    };
  }

  function getPotentialHelpToggleState(open, options) {
    const settings = options || {};
    return {
      potentialHelpOpen: typeof open === 'boolean' ? open : !settings.potentialHelpOpen,
      canvasHoverTarget: { type: '', key: '' }
    };
  }

  function getPotentialPromptResultClearState(uid, options) {
    const settings = options || {};
    const result = Object.prototype.hasOwnProperty.call(settings, 'promptResult')
      ? settings.promptResult
      : null;
    const shouldClear = !uid || !!(result && result.itemUid === uid);
    return {
      result: shouldClear ? null : result,
      shouldClear
    };
  }

  function getPotentialPromptResult(item, options) {
    const settings = options || {};
    const result = settings.promptResult || null;
    if (!item || !result || result.itemUid !== item.uid || result.mode !== 'regular') return null;
    const getComparison = typeof settings.getAttunementPotentialComparison === 'function'
      ? settings.getAttunementPotentialComparison
      : getAttunementPotentialComparison;
    return getComparison(item, result) ? result : null;
  }

  function getPotentialPromptRerollResultState(originalItem, updatedItem, options) {
    const settings = options || {};
    if (!updatedItem) {
      return {
        potentialPromptUid: '',
        potentialPromptResult: null,
        shouldClearResult: true
      };
    }
    const getNow = typeof settings.getNow === 'function'
      ? settings.getNow
      : () => Date.now();
    const clonePotential = typeof settings.cloneItemPotentialForPrompt === 'function'
      ? settings.cloneItemPotentialForPrompt
      : (potential, item) => cloneItemPotentialForPrompt(potential, item, settings);
    const originalPotential = Object.prototype.hasOwnProperty.call(settings, 'originalPotential')
      ? settings.originalPotential
      : clonePotential(originalItem && originalItem.potential, originalItem);
    const idTimestamp = getNow();
    return {
      potentialPromptUid: updatedItem.uid,
      potentialPromptResult: {
        mode: 'regular',
        id: `regular_${idTimestamp.toString(36)}_${updatedItem.uid || 'item'}`,
        itemUid: updatedItem.uid,
        itemId: updatedItem.id,
        itemName: updatedItem.name,
        originalPotential,
        newPotential: clonePotential(updatedItem.potential, updatedItem),
        createdAt: getNow()
      },
      shouldClearResult: false
    };
  }

  function getPotentialPromptComparisonSelection(item, options) {
    const settings = options || {};
    const state = Object.prototype.hasOwnProperty.call(settings, 'state')
      ? settings.state
      : settings.snapshot && settings.snapshot.state || null;
    if (!item || !state) return null;
    const pendingChoice = state.pendingPotentialChoice && state.pendingPotentialChoice.itemUid === item.uid
      ? state.pendingPotentialChoice
      : null;
    if (pendingChoice) return { mode: 'echo', choice: pendingChoice };
    const result = settings.promptResult || null;
    if (settings.consumableId === 'potential_cube' &&
      result &&
      result.itemUid === item.uid &&
      result.mode === 'regular') {
      return { mode: 'regular', choice: result };
    }
    return null;
  }

  function getPotentialPromptChoiceCacheKey(choice) {
    if (!choice) return '';
    const stableId = String(choice.id || '');
    if (stableId) return stableId;
    return JSON.stringify({
      itemUid: choice.itemUid || '',
      createdAt: choice.createdAt || 0,
      originalPotential: choice.originalPotential || null,
      newPotential: choice.newPotential || null
    });
  }

  function getPotentialPromptComparisonCacheKey(item, mode, choice, options) {
    const settings = options || {};
    const snapshot = settings.snapshot || {};
    const revisions = snapshot.domainRevisions || {};
    const consumableId = Object.prototype.hasOwnProperty.call(settings, 'potentialPromptConsumableId')
      ? settings.potentialPromptConsumableId
      : settings.consumableId;
    const getChoiceCacheKey = typeof settings.getPotentialPromptChoiceCacheKey === 'function'
      ? settings.getPotentialPromptChoiceCacheKey
      : getPotentialPromptChoiceCacheKey;
    return [
      mode || '',
      item && item.uid || '',
      consumableId || '',
      Number(snapshot.cacheRevision || 0),
      Number(revisions.inventory || 0),
      Number(revisions.equipment || 0),
      Number(revisions.hud || 0),
      getChoiceCacheKey(choice)
    ].join('|');
  }

  function getPotentialPromptComparisonState(item, options) {
    const settings = options || {};
    const state = settings.state || settings.snapshot && settings.snapshot.state || null;
    if (!item || !state) {
      return { state: null, cache: settings.cache || null, shouldUpdateCache: false };
    }
    const selection = getPotentialPromptComparisonSelection(item, settings);
    if (!selection || !selection.choice) {
      return { state: null, cache: settings.cache || null, shouldUpdateCache: false };
    }
    const getCacheKey = typeof settings.getCacheKey === 'function'
      ? settings.getCacheKey
      : () => '';
    const key = getCacheKey(item, selection.mode, selection.choice);
    const cached = settings.cache || null;
    if (cached && cached.key === key) {
      return {
        state: cached.state,
        cache: cached,
        shouldUpdateCache: false,
        key,
        mode: selection.mode,
        choice: selection.choice
      };
    }
    const getComparison = typeof settings.getComparison === 'function'
      ? settings.getComparison
      : getAttunementPotentialComparison;
    const comparison = getComparison(item, selection.choice);
    const nextState = comparison ? {
      mode: selection.mode,
      choice: selection.choice,
      comparison
    } : null;
    return {
      state: nextState,
      cache: { key, state: nextState },
      shouldUpdateCache: true,
      key,
      mode: selection.mode,
      choice: selection.choice
    };
  }

  function getPotentialHelpBox(width, height, state, options) {
    const settings = options || {};
    const bottomLimit = Number(settings.bottomLimit || height);
    const boxW = Math.min(540, Math.max(330, width - 32));
    const boxH = Math.min(430, Math.max(280, bottomLimit - 28));
    const target = state || { x: 0, y: 0, w: boxW, h: boxH, scroll: 0, userPlaced: false };
    const getCenteredPromptBox = typeof settings.getCenteredPromptBox === 'function'
      ? settings.getCenteredPromptBox
      : UiCanvasWindows.getCenteredPromptBox;
    if (getCenteredPromptBox) {
      const box = getCenteredPromptBox(width, height, target, {
        minWidth: boxW,
        maxWidth: boxW,
        height: boxH,
        bottomLimit,
        defaultYMin: 14,
        constrainDefaultY: false
      });
      target.scroll = clamp(Number(target.scroll || 0), 0, Number(target.maxScroll || 0));
      return box;
    }
    target.w = boxW;
    target.h = boxH;
    if (!target.userPlaced) {
      target.x = Math.round((width - boxW) / 2);
      target.y = Math.round(Math.max(14, (bottomLimit - boxH) / 2));
    }
    target.x = clamp(Number(target.x || 0), 8, Math.max(8, width - boxW - 8));
    target.y = clamp(Number(target.y || 0), 8, Math.max(8, bottomLimit - boxH));
    target.scroll = clamp(Number(target.scroll || 0), 0, Number(target.maxScroll || 0));
    return { x: target.x, y: target.y, w: boxW, h: boxH };
  }

  function getPotentialPromptBox(width, height, state, options) {
    const settings = options || {};
    const comparisonState = settings.comparisonState || null;
    const comparison = settings.comparison || comparisonState && comparisonState.comparison || null;
    const item = settings.item || null;
    const isEmpty = !item && !comparison;
    const isLineCatalyst = String(settings.consumableId || '') === 'line_catalyst';
    const targetW = isEmpty ? 430 : comparison ? 860 : isLineCatalyst ? 620 : 760;
    const minW = isEmpty ? 320 : comparison ? 520 : isLineCatalyst ? 430 : 560;
    const maxAvailableW = Math.max(300, width - 32);
    const boxW = Math.max(Math.min(minW, maxAvailableW), Math.min(targetW, maxAvailableW));
    const stacked = boxW < 680;
    const targetH = isEmpty
      ? 238
      : comparison
        ? (stacked ? 660 : 380)
        : isLineCatalyst
          ? (stacked ? 500 : 410)
          : stacked ? 610 : 430;
    const minH = isEmpty
      ? 210
      : comparison
        ? (stacked ? 520 : 340)
        : isLineCatalyst
          ? (stacked ? 440 : 360)
          : stacked ? 520 : 390;
    const bottomLimit = Number(settings.bottomLimit || height);
    const maxAvailableH = Math.max(220, bottomLimit - 16);
    const boxH = Math.max(Math.min(minH, maxAvailableH), Math.min(targetH, maxAvailableH));
    const target = state || { x: 0, y: 0, w: boxW, h: boxH, userPlaced: false };
    const getCenteredPromptBox = typeof settings.getCenteredPromptBox === 'function'
      ? settings.getCenteredPromptBox
      : UiCanvasWindows.getCenteredPromptBox;
    if (getCenteredPromptBox) {
      return getCenteredPromptBox(width, height, target, {
        minWidth: boxW,
        maxWidth: boxW,
        height: boxH,
        bottomLimit,
        defaultYMin: 18,
        defaultYOffset: 18,
        constrainDefaultY: false
      });
    }
    target.w = boxW;
    target.h = boxH;
    if (!target.userPlaced) {
      target.x = Math.round((width - boxW) / 2);
      target.y = Math.round(Math.max(18, (bottomLimit - boxH) / 2 + 18));
    }
    target.x = clamp(Number(target.x || 0), 8, Math.max(8, width - boxW - 8));
    target.y = clamp(Number(target.y || 0), 8, Math.max(8, bottomLimit - boxH));
    return { x: target.x, y: target.y, w: boxW, h: boxH };
  }

  function getShardCraftPromptBox(width, height, state, options) {
    const settings = options || {};
    const bottomLimit = Number(settings.bottomLimit || height);
    const boxW = Math.min(380, Math.max(300, width - 40));
    const boxH = 252;
    const target = state || { x: 0, y: 0, w: boxW, h: boxH, userPlaced: false };
    const getCenteredPromptBox = typeof settings.getCenteredPromptBox === 'function'
      ? settings.getCenteredPromptBox
      : UiCanvasWindows.getCenteredPromptBox;
    if (getCenteredPromptBox) {
      return getCenteredPromptBox(width, height, target, {
        minWidth: 300,
        maxWidth: 380,
        horizontalInset: 40,
        height: boxH,
        bottomLimit,
        defaultYMin: 18,
        defaultYOffset: 18,
        constrainDefaultY: false
      });
    }
    target.w = boxW;
    target.h = boxH;
    if (!target.userPlaced) {
      target.x = Math.round((width - boxW) / 2);
      target.y = Math.round(Math.max(18, (bottomLimit - boxH) / 2 + 18));
    }
    target.x = clamp(Number(target.x || 0), 8, Math.max(8, width - boxW - 8));
    target.y = clamp(Number(target.y || 0), 8, Math.max(8, bottomLimit - boxH));
    return { x: target.x, y: target.y, w: boxW, h: boxH };
  }

  function getPotentialDropZoneDomAction(target) {
    const source = target || null;
    if (source && typeof source.hasAttribute === 'function' && source.hasAttribute('data-starfall-potential-drop-zone')) {
      return { handled: true, type: 'openGearPicker', mode: 'attunement' };
    }
    return { handled: false, type: '' };
  }

  function getPotentialPromptShellDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    if (hasAttribute('data-starfall-potential-close')) return { handled: true, type: 'closePotentialPrompt' };
    if (hasAttribute('data-starfall-potential-help')) return { handled: true, type: 'openPotentialHelp' };
    return { handled: false, type: '' };
  }

  function getShardCraftPromptDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (hasAttribute('data-starfall-shard-craft-close')) return { handled: true, type: 'closeShardCraftPrompt' };
    const recipeId = getAttribute('data-starfall-shard-craft-recipe');
    if (recipeId) return { handled: true, type: 'setShardCraftRecipe', recipeId };
    const quantityDelta = getAttribute('data-starfall-shard-craft-delta');
    if (quantityDelta) return { handled: true, type: 'adjustShardCraftQuantity', delta: Number(quantityDelta) };
    if (hasAttribute('data-starfall-shard-craft-all')) return { handled: true, type: 'setShardCraftQuantityToMax' };
    if (hasAttribute('data-starfall-shard-craft-confirm')) return { handled: true, type: 'confirmShardCraftPrompt' };
    if (hasAttribute('data-starfall-combine-cube-fragments')) return { handled: true, type: 'combineCubeFragments' };
    if (hasAttribute('data-starfall-combine-preservation-cube-fragments')) return { handled: true, type: 'combinePreservationCubeFragments' };
    return { handled: false, type: '' };
  }

  function getPotentialPromptRollDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const confirmUid = getAttribute('data-starfall-potential-confirm');
    if (confirmUid) return { handled: true, type: 'confirmPotentialPrompt', uid: confirmUid };
    const preserveUid = getAttribute('data-starfall-potential-preserve');
    if (preserveUid) return { handled: true, type: 'startPreservationPotentialPrompt', uid: preserveUid };
    const repeatUid = getAttribute('data-starfall-potential-repeat');
    if (repeatUid) return { handled: true, type: 'repeatPotentialPrompt', uid: repeatUid };
    const lineUpgradeUid = getAttribute('data-starfall-potential-line-upgrade');
    if (lineUpgradeUid) return { handled: true, type: 'upgradePotentialLinePrompt', uid: lineUpgradeUid };
    return { handled: false, type: '' };
  }

  function getPotentialAutoDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    if (hasAttribute('data-starfall-potential-auto-toggle')) return { handled: true, type: 'toggleAutoPanel' };
    if (hasAttribute('data-starfall-potential-auto-close')) return { handled: true, type: 'closeAutoPanel' };
    const statId = getAttribute('data-starfall-potential-auto-stat-toggle');
    if (statId) return { handled: true, type: 'toggleStatGoal', statId };
    if (hasAttribute('data-starfall-potential-auto-stat-add')) return { handled: true, type: 'addStatGoal' };
    const removeIndex = getAttribute('data-starfall-potential-auto-stat-remove');
    if (removeIndex !== null) return { handled: true, type: 'removeStatGoal', index: removeIndex };
    if (hasAttribute('data-starfall-potential-auto-repeat')) return { handled: true, type: 'repeatLastAutoAttunement' };
    if (hasAttribute('data-starfall-potential-auto-run')) return { handled: true, type: 'runAutoAttunement' };
    return { handled: false, type: '' };
  }

  function getPotentialAutoInputDomAction(target) {
    const source = target || null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const value = source && Object.prototype.hasOwnProperty.call(source, 'value') ? source.value : undefined;
    if (hasAttribute('data-starfall-potential-auto-stat')) {
      return { handled: true, type: 'setPotentialAutoTarget', field: 'stat', value };
    }
    const statGoalIndex = getAttribute('data-starfall-potential-auto-stat-row');
    if (statGoalIndex !== null) return { handled: true, type: 'setPotentialAutoStatGoalAt', index: statGoalIndex, value };
    const statMinId = getAttribute('data-starfall-potential-auto-stat-min');
    if (statMinId) return { handled: true, type: 'setPotentialAutoStatGoalMin', statId: statMinId, value };
    const statMinIndex = getAttribute('data-starfall-potential-auto-stat-min-index');
    if (statMinIndex !== null) return { handled: true, type: 'setPotentialAutoStatGoalMinAt', index: statMinIndex, value };
    if (hasAttribute('data-starfall-potential-auto-tier')) {
      return { handled: true, type: 'setPotentialAutoTarget', field: 'tier', value };
    }
    if (hasAttribute('data-starfall-potential-auto-max')) {
      return { handled: true, type: 'setPotentialAutoTarget', field: 'max', value };
    }
    return { handled: false, type: '' };
  }

  function getPotentialChoiceDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const choiceId = getAttribute('data-starfall-potential-choice');
    if (!choiceId) return { handled: false, type: '' };
    return {
      handled: true,
      type: 'resolvePreservationPotentialChoice',
      choiceId,
      acceptNew: choiceId === 'apply' || choiceId === 'apply-repeat',
      repeat: choiceId === 'keep-repeat' || choiceId === 'apply-repeat'
    };
  }

  function getPotentialPromptDomDragHandle(target, selector) {
    const source = target || null;
    const handleSelector = selector || DOM_POTENTIAL_PROMPT_DRAG_HANDLE_SELECTOR;
    return source && typeof source.closest === 'function'
      ? source.closest(handleSelector)
      : null;
  }

  function getPotentialPromptDomDragStartAction(event, options) {
    const settings = options || {};
    const target = event && event.target || null;
    const root = settings.root || null;
    const handle = getPotentialPromptDomDragHandle(target);
    const inRoot = !!(handle && root && typeof root.contains === 'function' && root.contains(handle));
    const blockedTarget = target && typeof target.closest === 'function'
      ? target.closest('button, input, select, textarea, [data-starfall-potential-drop-zone]')
      : null;
    const card = handle && typeof handle.closest === 'function'
      ? handle.closest('.project-starfall-attunement-choice-card')
      : null;
    const handled = !!(handle && inRoot && settings.potentialPromptOpen && !blockedTarget && card);
    return {
      handled,
      type: handled ? 'startPotentialPromptDomDrag' : '',
      handle: handled ? handle : null,
      card: handled ? card : null,
      pointerId: event && event.pointerId,
      shouldSetPointerCapture: handled,
      shouldPreventDefault: handled
    };
  }

  function getPotentialPromptDomDragMoveAction(event, drag, options) {
    const sourceDrag = drag || null;
    if (!sourceDrag) {
      return {
        handled: false,
        type: '',
        x: 0,
        y: 0,
        w: 0,
        h: 0,
        userPlaced: false,
        shouldQueueUiRefresh: false,
        refreshOptions: null,
        shouldPreventDefault: false
      };
    }
    const settings = options || {};
    const viewportW = Number(settings.viewportWidth || 1280) || 1280;
    const viewportH = Number(settings.viewportHeight || 806) || 806;
    const w = Number(sourceDrag.w || 0);
    const h = Number(sourceDrag.h || 0);
    return {
      handled: true,
      type: 'movePotentialPromptDomDrag',
      x: clamp(Number(event && event.clientX || 0) - Number(sourceDrag.dx || 0), 8, Math.max(8, viewportW - w - 8)),
      y: clamp(Number(event && event.clientY || 0) - Number(sourceDrag.dy || 0), 8, Math.max(8, viewportH - h - 8)),
      w,
      h,
      userPlaced: true,
      shouldQueueUiRefresh: true,
      refreshOptions: { domains: ['inventory', 'equipment'], panel: true, draw: false },
      shouldPreventDefault: true
    };
  }

  function getPotentialPromptDomDragStopAction(drag) {
    const handled = !!drag;
    return {
      handled,
      type: handled ? 'stopPotentialPromptDomDrag' : '',
      shouldClearDrag: handled
    };
  }

  function getAttunementPromptPointerAction(region) {
    const source = region || {};
    if (source.type === 'potential-prompt-header') {
      return {
        handled: true,
        type: 'startPotentialPromptDrag',
        dragKey: 'potentialPromptDrag',
        boxType: 'potentialPrompt',
        activePanel: 'potentialPrompt',
        shouldPreventDefault: true
      };
    }
    if (source.type === 'shard-craft-header') {
      return {
        handled: true,
        type: 'startShardCraftPromptDrag',
        dragKey: 'shardCraftPromptDrag',
        boxType: 'shardCraftPrompt',
        activePanel: 'shardCraftPrompt',
        shouldPreventDefault: true
      };
    }
    return {
      handled: false,
      type: '',
      dragKey: '',
      boxType: '',
      activePanel: '',
      shouldPreventDefault: false
    };
  }

  function getAttunementPromptRegionAction(region) {
    const source = region || {};
    if (source.type === 'potential-prompt-close') return { handled: true, type: 'closePotentialPrompt' };
    if (source.type === 'potential-help-open') return { handled: true, type: 'openPotentialHelp' };
    if (source.type === 'potential-help-close') return { handled: true, type: 'closePotentialHelp' };
    if (source.type === 'shard-craft-close') return { handled: true, type: 'closeShardCraftPrompt' };
    if (source.type === 'shard-craft-recipe') return { handled: true, type: 'setShardCraftRecipe', recipeId: source.recipe };
    if (source.type === 'shard-craft-decrease') return { handled: true, type: 'adjustShardCraftQuantity', delta: -1 };
    if (source.type === 'shard-craft-increase') return { handled: true, type: 'adjustShardCraftQuantity', delta: 1 };
    if (source.type === 'shard-craft-all') return { handled: true, type: 'setShardCraftQuantityToMax' };
    if (source.type === 'shard-craft-confirm') return { handled: true, type: 'confirmShardCraftPrompt' };
    if (source.type === 'potential-prompt-confirm') return { handled: true, type: 'confirmPotentialPrompt', uid: source.uid };
    if (source.type === 'potential-prompt-preserve') return { handled: true, type: 'startPreservationPotentialPrompt', uid: source.uid };
    if (source.type === 'potential-repeat') return { handled: true, type: 'repeatPotentialPrompt', uid: source.uid };
    if (source.type === 'potential-line-upgrade') return { handled: true, type: 'upgradePotentialLinePrompt', uid: source.uid };
    if (source.type === 'potential-auto-toggle') return { handled: true, type: 'toggleAutoPanel' };
    if (source.type === 'potential-auto-close') return { handled: true, type: 'closeAutoPanel' };
    if (source.type === 'potential-auto-stat-cycle') return { handled: true, type: 'cycleAutoTarget', field: 'stat', delta: source.delta };
    if (source.type === 'potential-auto-tier-cycle') return { handled: true, type: 'cycleAutoTarget', field: 'tier', delta: source.delta };
    if (source.type === 'potential-auto-repeat') return { handled: true, type: 'repeatLastAutoAttunement' };
    if (source.type === 'potential-auto-run') return { handled: true, type: 'runAutoAttunement' };
    if (source.type === 'combine-cube-fragments') return { handled: true, type: 'combineCubeFragments' };
    if (source.type === 'combine-preservation-cube-fragments') return { handled: true, type: 'combinePreservationCubeFragments' };
    if (source.type === 'potential-choice') {
      return { handled: true, type: 'resolvePreservationPotentialChoice', acceptNew: source.acceptNew, repeat: !!source.repeat };
    }
    return { handled: false, type: '' };
  }

  function createAttunementDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_POTENTIAL_PROMPT_DRAG_HANDLE_ATTRIBUTES,
      DOM_POTENTIAL_PROMPT_DRAG_HANDLE_SELECTOR
    });
  }

  const api = {
    DOM_POTENTIAL_PROMPT_DRAG_HANDLE_ATTRIBUTES,
    DOM_POTENTIAL_PROMPT_DRAG_HANDLE_SELECTOR,
    POTENTIAL_MAX_LINE_COUNT,
    POTENTIAL_LINE_UPGRADE_CHANCES,
    POTENTIAL_LINE_DOWNGRADE_CHANCE_ON_FAIL,
    isPotentialLinePoolValidForSlot,
    isPotentialLinePoolValidForTier,
    isPotentialLinePoolEligible,
    normalizePotentialLine,
    normalizePotentialLineCount,
    normalizePotentialLineArchive,
    normalizeItemPotential,
    getItemPotentialStats,
    getPotentialLineSummary,
    getPotentialLineRollRows,
    getPotentialSummaryFromPotential,
    getItemWithPotential,
    cloneItemPotentialForPrompt,
    getPotentialLineRowMap,
    getPotentialComparisonSide,
    getAttunementPotentialComparison,
    getEchoPotentialComparison,
    renderAttunementStatIconMarkup,
    renderAttunementPrismIcon,
    renderEchoRankMarkup,
    renderEchoVerdictRows,
    renderEchoComparisonRows,
    renderAttunementCurrentRows,
    renderLineCatalystDetails,
    renderAttunementTargetPanel,
    renderPotentialAutoControls,
    renderAttunementEmptyPrompt,
    renderAttunementComparisonPrompt,
    renderEchoPotentialChoicePrompt,
    getPotentialPromptConsumableId,
    getPotentialPromptRenderState,
    renderPotentialPrompt,
    getShardCraftRecipeConfig,
    getShardCraftMaxQuantity,
    getShardCraftRecipeSelectionState,
    getShardCraftPromptOpenState,
    getShardCraftPromptCloseState,
    normalizeShardCraftPromptQuantity,
    getShardCraftQuantityAdjustment,
    normalizeShardCraftCombineQuantity,
    getShardCraftConfirmAction,
    getShardCraftPromptRenderState,
    renderShardCraftPrompt,
    getCanvasShardCraftPromptLayout,
    drawCanvasShardCraftPrompt,
    getCanvasPotentialPromptRenderState,
    getCanvasPotentialHelpLayout,
    drawCanvasPotentialHelp,
    drawCanvasAttunementPromptChrome,
    drawCanvasAttunementEmptyPrompt,
    drawCanvasAttunementReadySummary,
    getCanvasAttunementReadyCurrentPanelLayout,
    drawCanvasAttunementReadyCurrentPanel,
    getCanvasAttunementReadyTargetPanelLayout,
    drawCanvasAttunementReadyTargetPanel,
    drawCanvasAttunementReadyStatusPanel,
    getCanvasAttunementReadyActionLayout,
    drawCanvasAttunementReadyActionBar,
    getCanvasAttunementReadyAutoGoalsLayout,
    drawCanvasAttunementReadyAutoGoalsPanel,
    getCanvasAttunementReadyFinalActionLayout,
    drawCanvasAttunementReadyFinalActionButton,
    getCanvasAttunementReadyBaseLayout,
    getCanvasAttunementReadyPanelLayout,
    getCanvasAttunementReadyLineLayout,
    getCanvasAttunementReadyLineUpgradePanelLayout,
    drawCanvasAttunementReadyLineUpgradePanel,
    getCanvasAttunementReadyLineActionLayout,
    drawCanvasAttunementReadyLineActionBar,
    getCanvasAttunementReadyLineCurrentPanelLayout,
    drawCanvasAttunementReadyLineCurrentPanel,
    getPotentialTierLabel,
    formatPotentialTierChance,
    getPotentialTierAdvanceLabel,
    getPotentialTierRollSummary,
    getPotentialLineUpgradeChanceForTarget,
    getPotentialLineUpgradeInfo,
    getPotentialEligibleLinePools,
    getPotentialRangeLabel,
    getPotentialSlotRestrictionLabel,
    getPotentialHelpRows,
    createAttunementPotentialUiHelpers,
    getPotentialPromptItem,
    getPotentialPromptTargetSelectionState,
    getPotentialPromptOpenState,
    getPotentialPromptCloseState,
    getPotentialHelpToggleState,
    getPotentialPromptResultClearState,
    getPotentialPromptResult,
    getPotentialPromptRerollResultState,
    getPotentialPromptChoiceCacheKey,
    getPotentialPromptComparisonCacheKey,
    getPotentialPromptComparisonSelection,
    getPotentialPromptComparisonState,
    getPotentialPromptBox,
    getPotentialHelpBox,
    getShardCraftPromptBox,
    getPotentialDropZoneDomAction,
    getPotentialPromptShellDomAction,
    getShardCraftPromptDomAction,
    getPotentialPromptRollDomAction,
    getPotentialAutoDomAction,
    getPotentialAutoInputDomAction,
    getPotentialChoiceDomAction,
    getPotentialPromptDomDragHandle,
    getPotentialPromptDomDragStartAction,
    getPotentialPromptDomDragMoveAction,
    getPotentialPromptDomDragStopAction,
    getAttunementPromptPointerAction,
    getAttunementPromptRegionAction,
    createAttunementDomSelectorUiHelpers
  };

  const modules = UiModules;
  modules.attunement = Object.assign({}, modules.attunement || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
