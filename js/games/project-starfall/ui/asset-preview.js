(function initProjectStarfallUiAssetPreview(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  function escapeHtmlFallback(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  const ASSET_PREVIEW_CATEGORY_OPTIONS = Object.freeze([
    { id: 'all', label: 'All' },
    { id: 'Maps', label: 'Maps' },
    { id: 'Backgrounds', label: 'Backgrounds' },
    { id: 'Environment', label: 'Environment' },
    { id: 'Players', label: 'Players' },
    { id: 'Enemies', label: 'Enemies' },
    { id: 'Items', label: 'Items' },
    { id: 'Equipment', label: 'Equipment' },
    { id: 'Skills', label: 'Skills' },
    { id: 'Combat FX', label: 'Combat FX' },
    { id: 'UI', label: 'UI' },
    { id: 'Portals', label: 'Portals' },
    { id: 'Other', label: 'Other' }
  ]);
  const ASSET_PREVIEW_SOURCE_OPTIONS = Object.freeze([
    { id: 'all', label: 'All' },
    { id: 'ai', label: 'AI' },
    { id: 'procedural', label: 'Procedural' }
  ]);
  const DOM_ASSET_PREVIEW_QUERY_INPUT_ATTRIBUTES = Object.freeze([
    'data-starfall-asset-preview-query'
  ]);
  const DOM_ASSET_PREVIEW_QUERY_INPUT_SELECTOR = DOM_ASSET_PREVIEW_QUERY_INPUT_ATTRIBUTES
    .map((attribute) => `[${attribute}]`)
    .join(', ');

  function normalizeAssetPreviewCategory(value) {
    const id = normalizeId(value) || 'all';
    return ASSET_PREVIEW_CATEGORY_OPTIONS.some((option) => option.id === id) ? id : 'all';
  }

  function normalizeAssetPreviewSource(value) {
    const id = normalizeId(value).toLowerCase() || 'all';
    return ASSET_PREVIEW_SOURCE_OPTIONS.some((option) => option.id === id) ? id : 'all';
  }

  function getAssetPreviewSourceLabel(entry) {
    const sourceType = normalizeAssetPreviewSource(entry && entry.sourceType);
    if (sourceType === 'ai') return 'AI-generated';
    if (sourceType === 'procedural') return 'Procedural';
    return entry && entry.sourceLabel || 'All sources';
  }

  function getAssetPreviewStatusLabel(entry) {
    if (!entry) return 'Missing';
    if (entry.loaded) return entry.width && entry.height ? `${entry.width}x${entry.height}` : 'Loaded';
    if (entry.failed || entry.status === 'error') return 'Load failed';
    return 'Loading';
  }

  function getAssetPreviewEntryCategories(entry) {
    return entry && entry.categories && entry.categories.length ? entry.categories : [entry && entry.category || 'Other'];
  }

  function getAssetPreviewCounts(catalog) {
    const counts = { all: (catalog || []).length };
    (catalog || []).forEach((entry) => {
      getAssetPreviewEntryCategories(entry).forEach((category) => {
        counts[category] = (counts[category] || 0) + 1;
      });
    });
    return counts;
  }

  function getAssetPreviewEntriesForCategory(catalog, categoryId) {
    const category = normalizeAssetPreviewCategory(categoryId);
    if (category === 'all') return (catalog || []).slice();
    return (catalog || []).filter((entry) => getAssetPreviewEntryCategories(entry).includes(category));
  }

  function getAssetPreviewEntriesForSource(catalog, sourceType) {
    const normalizedSource = normalizeAssetPreviewSource(sourceType);
    if (normalizedSource === 'all') return (catalog || []).slice();
    return (catalog || []).filter((entry) => normalizeAssetPreviewSource(entry && entry.sourceType) === normalizedSource);
  }

  function getFilteredAssetPreviewCatalog(catalog, state) {
    const previewState = state || {};
    const sourceType = normalizeAssetPreviewSource(previewState.sourceType);
    const category = normalizeAssetPreviewCategory(previewState.category);
    const query = normalizeId(previewState.query).toLowerCase();
    return getAssetPreviewEntriesForCategory(getAssetPreviewEntriesForSource(catalog || [], sourceType), category).filter((entry) => {
      const categories = getAssetPreviewEntryCategories(entry);
      if (!query) return true;
      return [
        entry.label,
        entry.path,
        entry.kind,
        entry.category,
        entry.sourceType,
        entry.sourceLabel,
        categories.join(' '),
        (entry.labels || []).join(' '),
        (entry.tags || []).join(' '),
        (entry.sourceIds || []).join(' ')
      ].join(' ').toLowerCase().includes(query);
    });
  }

  function getAssetPreviewSelectionMetadata(catalog, filtered, state) {
    const sourceCatalog = catalog || [];
    const previewState = state || {};
    const selectedPathByCategory = previewState.selectedPathByCategory || {};
    const visible = filtered || getFilteredAssetPreviewCatalog(sourceCatalog, previewState);
    const sourceEntries = getAssetPreviewEntriesForSource(sourceCatalog, previewState.sourceType);
    const category = normalizeAssetPreviewCategory(previewState.category);
    const categoryEntries = getAssetPreviewEntriesForCategory(sourceEntries, category);
    const sourceKey = normalizeAssetPreviewSource(previewState.sourceType);
    const selectionKey = `${sourceKey}:${category}`;
    const legacySelectionKey = category;
    const pathForCategory = previewState.selectedPath || selectedPathByCategory[selectionKey] || selectedPathByCategory[legacySelectionKey] || '';
    const selected = categoryEntries.find((entry) => entry.path === pathForCategory);
    return {
      visible,
      sourceEntries,
      category,
      categoryEntries,
      sourceKey,
      selectionKey,
      legacySelectionKey,
      pathForCategory,
      selected,
      next: selected || visible[0] || categoryEntries[0] || sourceEntries[0] || sourceCatalog[0] || null
    };
  }

  function getAssetPreviewCategoryChangeMetadata(state, value) {
    const previewState = state || {};
    const selectedPathByCategory = previewState.selectedPathByCategory || {};
    const previousCategory = normalizeAssetPreviewCategory(previewState.category);
    const sourceKey = normalizeAssetPreviewSource(previewState.sourceType);
    const nextCategory = normalizeAssetPreviewCategory(value);
    const previousSelectionKey = `${sourceKey}:${previousCategory}`;
    const nextSelectionKey = `${sourceKey}:${nextCategory}`;
    return {
      previousCategory,
      sourceKey,
      nextCategory,
      previousSelectionKey,
      nextSelectionKey,
      selectedPath: previewState.selectedPath || '',
      nextSelectedPath: selectedPathByCategory[nextSelectionKey] || selectedPathByCategory[nextCategory] || ''
    };
  }

  function getAssetPreviewAssetSelectionMetadata(state, path) {
    const previewState = state || {};
    const selectedPath = normalizeId(path);
    const category = normalizeAssetPreviewCategory(previewState.category);
    const sourceKey = normalizeAssetPreviewSource(previewState.sourceType);
    return {
      selectedPath,
      category,
      sourceKey,
      selectionKey: `${sourceKey}:${category}`
    };
  }

  function getAssetPreviewSourceTypeChangeMetadata(value) {
    return {
      sourceType: normalizeAssetPreviewSource(value),
      selectedPath: '',
      animationState: '',
      categoryScroll: 0
    };
  }

  function getAssetPreviewQueryChangeMetadata(value) {
    return {
      query: normalizeId(value)
    };
  }

  function getAssetPreviewQueryInput(root, selector) {
    const source = root || null;
    const querySelector = selector || DOM_ASSET_PREVIEW_QUERY_INPUT_SELECTOR;
    return source && typeof source.querySelector === 'function'
      ? source.querySelector(querySelector)
      : null;
  }

  function getAssetPreviewAnimationStateChangeMetadata(stateId) {
    return {
      animationState: normalizeId(stateId)
    };
  }

  function getAssetPreviewPausedToggleMetadata(state) {
    const previewState = state || {};
    return {
      paused: !previewState.paused
    };
  }

  function getAssetPreviewCategoryButtonMetadata(summary, selected) {
    const entry = summary || {};
    const isSelected = !!selected;
    const count = Number(entry.count || 0);
    const disabled = !count;
    return {
      id: entry.id,
      label: entry.label,
      count,
      countLabel: `${count} assets`,
      representative: entry.representative || null,
      disabled,
      selected: isSelected,
      className: `project-starfall-asset-kind${isSelected ? ' is-selected' : ''}`,
      ariaPressed: isSelected ? 'true' : 'false'
    };
  }

  function getAssetPreviewSourceToggleMetadata(sourceType) {
    const selectedSource = normalizeAssetPreviewSource(sourceType);
    return {
      selectedSource,
      options: ASSET_PREVIEW_SOURCE_OPTIONS.map((option) => ({
        id: option.id,
        label: option.label,
        selected: option.id === selectedSource,
        ariaPressed: option.id === selectedSource ? 'true' : 'false'
      }))
    };
  }

  function getAssetPreviewThumbMetadata(entry, selected) {
    const item = entry || {};
    const isSelected = !!selected;
    const status = getAssetPreviewStatusLabel(item);
    const sourceLabel = getAssetPreviewSourceLabel(item);
    return {
      entry: item,
      selected: isSelected,
      path: item.path,
      label: item.label || item.path,
      kindText: `${item.category || 'Other'} - ${item.kind || 'image'}`,
      status,
      sourceLabel,
      statusText: `${status} - ${sourceLabel}`,
      className: `project-starfall-asset-entry${isSelected ? ' is-selected' : ''}${item.loaded ? ' is-loaded' : item.failed ? ' is-failed' : ' is-loading'}`
    };
  }

  function getAssetPreviewDetailMetadata(entry, animationState, groups, paused) {
    if (!entry) {
      return {
        emptyText: 'No matching assets.'
      };
    }
    const animation = entry.animation;
    const sourceLabel = getAssetPreviewSourceLabel(entry);
    const statusLabel = getAssetPreviewStatusLabel(entry);
    const detailGroups = Array.isArray(groups) ? groups : getAssetPreviewAnimationGroups(entry);
    const hasAnimation = !!(animation && animationState);
    const frameWidth = Number(animation && animation.frameWidth || 160);
    const frameHeight = Number(animation && animation.frameHeight || 160);
    const frameCount = Math.max(1, Number(animationState && animationState.frames || 1) || 1);
    const fps = Math.max(1, Number(animationState && animationState.fps || 1) || 1);
    return {
      entry,
      hasAnimation,
      title: entry.label || entry.path,
      summaryText: `${(entry.categories || [entry.category || 'Other']).join(' / ')} - ${entry.kind || 'image'} - ${sourceLabel} - ${statusLabel}`,
      path: entry.path,
      groups: detailGroups.map((group) => ({
        id: group.id,
        label: group.label,
        states: (group.states || []).map((state) => ({
          id: state.id,
          label: state.label || state.id,
          selected: !!(animationState && state.id === animationState.id),
          ariaPressed: animationState && state.id === animationState.id ? 'true' : 'false'
        }))
      })),
      metaItems: hasAnimation
        ? [
          `Frame ${frameWidth}x${frameHeight}`,
          `${frameCount} frames`,
          `${fps} FPS`,
          animationState.loop ? 'Looping' : 'One-shot',
          sourceLabel
        ]
        : [
          entry.width && entry.height ? `${entry.width}x${entry.height}` : 'Dimensions pending',
          entry.status || 'loading',
          sourceLabel
        ],
      toggleLabel: paused ? 'Play Animation' : 'Pause Animation'
    };
  }

  function getAssetPreviewPanelMetadata(state, sourceCatalog, categories, filtered) {
    const previewState = state || {};
    const sourceType = normalizeAssetPreviewSource(previewState.sourceType);
    const category = normalizeAssetPreviewCategory(previewState.category);
    const sourceEntries = Array.isArray(sourceCatalog) ? sourceCatalog : [];
    const categorySummaries = Array.isArray(categories) ? categories : [];
    const filteredEntries = Array.isArray(filtered) ? filtered : [];
    const activeSource = ASSET_PREVIEW_SOURCE_OPTIONS.find((option) => option.id === sourceType) || ASSET_PREVIEW_SOURCE_OPTIONS[0];
    const activeCategory = categorySummaries.find((entry) => entry && entry.id === category) || null;
    const categoryLabel = activeCategory && activeCategory.label || 'All';
    return {
      sourceType,
      category,
      query: previewState.query || '',
      activeSource,
      activeCategory,
      categoryLabel,
      toolbarSummary: `${categoryLabel} / ${activeSource.label} - ${filteredEntries.length}/${sourceEntries.length} visible`,
      paneTitle: activeCategory && activeCategory.label || 'Assets',
      entryCountText: `${filteredEntries.length} entries`
    };
  }

  function getAssetPreviewArtMetadata(entry, className, options, animationState, paused) {
    const settings = options || {};
    const item = entry || {};
    const state = animationState || null;
    const isPaused = !!(settings.selected && paused);
    const hasLoopDelay = !!(state && Number(state.loopDelay || 0) > 0);
    return {
      entry: item,
      className: className || '',
      animationState: state,
      hasAnimation: !!(item.animation && state),
      wrapperClassName: `${className || ''} project-starfall-asset-animation-art`,
      frameClassName: `project-starfall-asset-animation-frame${settings.thumb ? ' is-thumb' : ''}${isPaused ? ' is-paused' : ''}${hasLoopDelay ? ' has-loop-delay' : ''}`,
      imagePath: item.path || '',
      imageAlt: item.label,
      imageClassName: 'project-starfall-asset-card-image',
      fallbackText: 'AST'
    };
  }

  function getAssetPreviewCategorySummaries(catalog, counts) {
    return ASSET_PREVIEW_CATEGORY_OPTIONS.map((option) => {
      const entries = getAssetPreviewEntriesForCategory(catalog || [], option.id);
      const representative = entries.find((entry) => entry.loaded) || entries.find((entry) => entry.kind === 'animation') || entries[0] || null;
      return Object.assign({}, option, {
        count: Number(counts && counts[option.id] || 0),
        representative
      });
    });
  }

  function getAssetPreviewAnimationGroups(entry) {
    const states = entry && entry.animation && entry.animation.states || [];
    if (!states.length) return [];
    const category = entry.category || '';
    const groups = category === 'Enemies'
      ? [
        { id: 'movement', label: 'Movement', states: ['idle', 'move'] },
        { id: 'combat', label: 'Combat', states: ['telegraph', 'attack', 'projectile', 'buff', 'melee'] },
        { id: 'reaction', label: 'Reaction', states: ['hit', 'defeat'] }
      ]
      : category === 'Players' || category === 'Equipment'
        ? [
          { id: 'locomotion', label: 'Locomotion', states: ['idle', 'run', 'jump', 'fall', 'climb'] },
          { id: 'combat', label: 'Combat', states: ['basic', 'skill'] },
          { id: 'support', label: 'Support', states: ['party', 'buff'] },
          { id: 'reaction', label: 'Reaction', states: ['hit', 'defeat'] }
        ]
        : [
          { id: 'cast', label: 'Cast & Launch', states: ['cast', 'telegraph', 'projectile', 'trail'] },
          { id: 'impact', label: 'Impact & Area', states: ['impact', 'area', 'melee', 'buff'] },
          { id: 'idle', label: 'Idle', states: ['idle'] }
        ];
    const stateById = states.reduce((map, state) => {
      map[state.id] = state;
      return map;
    }, {});
    const used = new Set();
    const result = groups.map((group) => {
      const groupStates = group.states.map((id) => stateById[id]).filter(Boolean);
      groupStates.forEach((state) => used.add(state.id));
      return Object.assign({}, group, { states: groupStates });
    }).filter((group) => group.states.length);
    const other = states.filter((state) => !used.has(state.id));
    if (other.length) result.push({ id: 'other', label: 'Other', states: other });
    return result;
  }

  function getAssetPreviewAnimationState(entry, stateId) {
    if (!entry || !entry.animation) return null;
    const states = entry.animation.states || [];
    return states.find((state) => state.id === stateId) || states[0] || null;
  }

  function getAssetPreviewDefaultAnimationState(entry) {
    if (!entry || !entry.animation) return null;
    const states = entry.animation.states || [];
    return states.find((state) => ['idle', 'move', 'run'].includes(state.id)) || states[0] || null;
  }

  function getAssetPreviewAnimationStyle(entry, animationState, options) {
    const settings = options || {};
    const escapeHtml = settings.escapeHtml || escapeHtmlFallback;
    const animation = entry && entry.animation;
    const state = animationState || getAssetPreviewDefaultAnimationState(entry);
    if (!animation || !state) return '';
    const frameWidth = Number(animation.frameWidth || 160);
    const frameHeight = Number(animation.frameHeight || 160);
    const frameCount = Math.max(1, Number(state.frames || 1) || 1);
    const fps = Math.max(1, Number(state.fps || 1) || 1);
    const rowOffset = -Math.max(0, Number(state.row || 0)) * frameHeight;
    const endOffset = -frameWidth * frameCount;
    const duration = Math.max(0.1, frameCount / fps + Math.max(0, Number(state.loopDelay || 0) || 0));
    return `--asset-sheet:url(&quot;${escapeHtml(entry.path)}&quot;); --asset-frame-width:${frameWidth}px; --asset-frame-height:${frameHeight}px; --asset-row-offset:${rowOffset}px; --asset-end-offset:${endOffset}px; --asset-frames:${frameCount}; --asset-duration:${duration}s;`;
  }

  function getAssetPreviewAnimationFrameIndex(state, previewState, nowSeconds) {
    const animationState = state || {};
    const frameCount = Math.max(1, Number(animationState.frames || 1) || 1);
    const sequence = Array.isArray(animationState.sequence) && animationState.sequence.length
      ? animationState.sequence.map((frame) => clamp(Math.floor(Number(frame || 0) || 0), 0, frameCount - 1))
      : null;
    const stepCount = sequence ? sequence.length : frameCount;
    const frameForStep = (step) => sequence
      ? sequence[clamp(Math.floor(Number(step || 0) || 0), 0, stepCount - 1)]
      : clamp(Math.floor(Number(step || 0) || 0), 0, frameCount - 1);
    const fps = Math.max(1, Number(animationState.fps || 1) || 1);
    const assetPreview = previewState || {};
    if (assetPreview.paused) return 0;
    const elapsed = Math.max(0, Number(nowSeconds || 0) - Number(assetPreview.startedAt || 0));
    const loopDelay = animationState.loop ? Math.max(0, Number(animationState.loopDelay || 0) || 0) : 0;
    if (animationState.loop) {
      const playDuration = Math.max(0.01, stepCount / fps);
      const cycleDuration = Math.max(playDuration, playDuration + loopDelay);
      const cycleElapsed = cycleDuration > 0 ? elapsed % cycleDuration : elapsed;
      if (loopDelay > 0 && cycleElapsed >= playDuration) return frameForStep(stepCount - 1);
      return frameForStep(Math.floor(cycleElapsed * fps) % stepCount);
    }
    return frameForStep(Math.min(stepCount - 1, Math.floor(elapsed * fps)));
  }

  function getAssetPreviewAdminControlsMetadata(catalog, progress, x, y, w) {
    const entries = Array.isArray(catalog) ? catalog : [];
    const loadProgress = progress || {};
    const loaded = Number(loadProgress.loaded || 0);
    const total = Number(loadProgress.total || 0);
    const failed = Number(loadProgress.failed || 0);
    const h = 74;
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
        value: 'Asset Preview',
        x: x + 12,
        y: y + 10,
        color: '#102033',
        font: '900 12px system-ui'
      },
      countText: {
        value: `${entries.length} entries`,
        x: x + w - 12,
        y: y + 10,
        color: '#2f7dd6',
        font: '900 11px system-ui',
        align: 'right'
      },
      progressText: {
        value: `${loaded}/${total} loaded${failed ? ` - ${failed} failed` : ''}`,
        x: x + 12,
        y: y + 31,
        color: '#5f6f7a',
        font: '800 10px system-ui',
        maxWidth: w - 24,
        lineHeight: 11,
        maxLines: 1
      },
      openButton: {
        label: 'Open Asset Preview',
        x: x + 12,
        y: y + 44,
        w: 154,
        h: 26,
        region: { type: 'asset-preview-open' },
        disabled: !entries.length
      },
      nextY: y + h
    };
  }

  function getAssetPreviewScrollIndicatorMetadata(x, y, w, h, contentH, scroll) {
    const maxScroll = Math.max(0, Number(contentH || 0) - Number(h || 0));
    if (maxScroll <= 0) return null;
    const barH = Math.max(28, h * h / Math.max(h, contentH));
    const barY = y + (h - barH) * (clamp(scroll || 0, 0, maxScroll) / maxScroll);
    return {
      x: x + w - 6,
      y: barY,
      w: 3,
      h: barH,
      radius: 3,
      fill: 'rgba(16,32,51,0.28)',
      stroke: ''
    };
  }

  function getAssetPreviewCanvasLayout(x, y, w, bodyHeight) {
    const bodyH = Math.max(360, Number(bodyHeight || 520));
    const headerH = 54;
    const gap = 8;
    let kindW = Math.min(164, Math.max(126, Math.floor(w * 0.22)));
    let detailW = Math.min(310, Math.max(220, Math.floor(w * 0.38)));
    let entryW = w - kindW - detailW - gap * 2;
    if (entryW < 180) {
      detailW = Math.max(190, w - kindW - 180 - gap * 2);
      entryW = w - kindW - detailW - gap * 2;
    }
    if (entryW < 160) {
      kindW = Math.max(112, kindW - (160 - entryW));
      entryW = w - kindW - detailW - gap * 2;
    }
    const paneY = y + headerH;
    const paneH = Math.max(260, bodyH - headerH);
    const kindX = x;
    const entryX = kindX + kindW + gap;
    const detailX = entryX + entryW + gap;
    return {
      bodyH,
      headerH,
      gap,
      paneY,
      paneH,
      kindX,
      kindW,
      entryX,
      entryW,
      detailX,
      detailW,
      categoryPane: {
        x: kindX,
        y: paneY,
        w: kindW,
        h: paneH,
        radius: 8,
        fill: '#fbfaf6',
        stroke: 'rgba(16,32,51,0.14)'
      },
      entryPane: {
        x: entryX,
        y: paneY,
        w: entryW,
        h: paneH,
        radius: 8,
        fill: '#ffffff',
        stroke: 'rgba(16,32,51,0.14)'
      },
      detailPane: {
        x: detailX,
        y: paneY,
        w: detailW,
        h: paneH,
        radius: 8,
        fill: '#ffffff',
        stroke: 'rgba(16,32,51,0.14)'
      }
    };
  }

  function getAssetPreviewHeaderMetadata(activeCategory, activeSource, filteredCount, sourceCount, query, x, y, w) {
    const category = activeCategory || {};
    const source = activeSource || ASSET_PREVIEW_SOURCE_OPTIONS[0];
    const sourceType = normalizeAssetPreviewSource(source.id);
    const sourceButtonY = y + 22;
    let sourceButtonX = x + w;
    const sourceButtons = ASSET_PREVIEW_SOURCE_OPTIONS.slice().reverse().map((option) => {
      const buttonW = option.id === 'procedural' ? 82 : 48;
      sourceButtonX -= buttonW;
      const button = {
        label: option.label,
        x: sourceButtonX,
        y: sourceButtonY,
        w: buttonW,
        h: 23,
        region: { type: 'asset-preview-source', sourceType: option.id },
        disabled: false,
        selectedHighlight: option.id === sourceType ? {
          x: sourceButtonX + 2,
          y: sourceButtonY + 2,
          w: buttonW - 4,
          h: 19,
          radius: 5,
          fill: 'rgba(47,125,214,0.16)',
          stroke: 'rgba(47,125,214,0.72)'
        } : null
      };
      sourceButtonX -= 5;
      return button;
    });
    return {
      titleText: {
        value: 'Asset Catalog',
        x,
        y,
        color: '#102033',
        font: '900 13px system-ui'
      },
      summaryText: {
        value: `${category.label || 'All'} / ${source.label} - ${Number(filteredCount || 0)}/${Number(sourceCount || 0)} visible`,
        x: x + w,
        y: y + 1,
        color: '#2f7dd6',
        font: '900 10px system-ui',
        align: 'right',
        maxWidth: Math.max(180, w - 150),
        lineHeight: 12,
        maxLines: 1
      },
      helperText: {
        value: query ? `Search: ${query}` : 'Categories, entries, and selected preview stay visible while browsing.',
        x,
        y: y + 17,
        color: '#5f6f7a',
        font: '800 9px system-ui',
        maxWidth: Math.max(160, w - 300),
        lineHeight: 10,
        maxLines: 1
      },
      sourceButtons
    };
  }

  function getAssetPreviewCategoryListMetadata(categories, category, x, y, w, h, scroll) {
    const rows = Array.isArray(categories) ? categories : [];
    const categoryRowH = 56;
    const categoryGap = 6;
    const categoryPad = 8;
    const contentH = categoryPad * 2 + rows.length * categoryRowH + Math.max(0, rows.length - 1) * categoryGap;
    const maxScroll = Math.max(0, contentH - h);
    const clampedScroll = clamp(Number(scroll || 0), 0, maxScroll);
    return {
      rowH: categoryRowH,
      gap: categoryGap,
      pad: categoryPad,
      contentH,
      maxScroll,
      scroll: clampedScroll,
      scrollRegion: { type: 'asset-preview-category-scroll', x, y, w, h, maxScroll },
      clipRect: { x, y, w, h },
      rows: rows.map((summary, index) => {
        const entry = summary || {};
        const by = y + categoryPad + index * (categoryRowH + categoryGap) - clampedScroll;
        if (by + categoryRowH < y || by > y + h) return null;
        const selectedCategory = entry.id === category;
        const disabled = !entry.count;
        return {
          summary: entry,
          frame: {
            x: x + 6,
            y: by,
            w: w - 12,
            h: categoryRowH,
            radius: 7,
            fill: selectedCategory ? '#eef6ff' : disabled ? '#f2f1eb' : '#ffffff',
            stroke: selectedCategory ? 'rgba(47,125,214,0.68)' : 'rgba(16,32,51,0.14)'
          },
          preview: {
            x: x + 12,
            y: by + 8,
            w: 38,
            h: 38,
            representative: entry.representative || null,
            iconLabel: 'AST',
            iconFill: '#eef6ff'
          },
          labelText: {
            value: entry.label,
            x: x + 56,
            y: by + 9,
            color: disabled ? '#75818c' : '#102033',
            font: '900 10px system-ui',
            maxWidth: w - 66,
            lineHeight: 11,
            maxLines: 1
          },
          countText: {
            value: `${Number(entry.count || 0)} assets`,
            x: x + 56,
            y: by + 27,
            color: selectedCategory ? '#2f7dd6' : '#5f6f7a',
            font: '800 9px system-ui',
            maxWidth: w - 66,
            lineHeight: 10,
            maxLines: 1
          },
          region: disabled ? null : { type: 'asset-preview-category', category: entry.id, x: x + 6, y: by, w: w - 12, h: categoryRowH }
        };
      }).filter(Boolean),
      scrollIndicator: getAssetPreviewScrollIndicatorMetadata(x, y + 6, w, h - 12, contentH, clampedScroll)
    };
  }

  function getAssetPreviewEntryListMetadata(entries, selected, category, x, y, w, h, scroll) {
    const rows = Array.isArray(entries) ? entries : [];
    const entryRowH = 70;
    const entryGap = 6;
    const contentH = rows.length ? rows.length * entryRowH + Math.max(0, rows.length - 1) * entryGap + 8 : h;
    const maxScroll = Math.max(0, contentH - h);
    const clampedScroll = clamp(Number(scroll || 0), 0, maxScroll);
    return {
      rowH: entryRowH,
      gap: entryGap,
      contentH,
      maxScroll,
      scroll: clampedScroll,
      scrollRegion: { type: 'asset-preview-entry-scroll', category, x, y, w, h, maxScroll },
      clipRect: { x, y, w, h },
      emptyText: rows.length ? null : {
        value: 'No matching assets in this category.',
        x: x + 10,
        y: y + 12,
        color: '#5f6f7a',
        font: '12px system-ui',
        maxWidth: w - 20,
        lineHeight: 14,
        maxLines: 3
      },
      rows: rows.map((entry, index) => {
        const item = entry || {};
        const by = y + 4 + index * (entryRowH + entryGap) - clampedScroll;
        if (by + entryRowH < y || by > y + h) return null;
        const active = !!(selected && item.path === selected.path);
        return {
          entry: item,
          active,
          frame: {
            x: x + 8,
            y: by,
            w: w - 16,
            h: entryRowH,
            radius: 7,
            fill: active ? '#eef6ff' : '#fbfaf6',
            stroke: active ? 'rgba(47,125,214,0.68)' : 'rgba(16,32,51,0.12)'
          },
          preview: {
            x: x + 16,
            y: by + 9,
            w: 52,
            h: 52
          },
          labelText: {
            value: item.label || item.path,
            x: x + 76,
            y: by + 11,
            color: '#102033',
            font: '900 10px system-ui',
            maxWidth: w - 92,
            lineHeight: 11,
            maxLines: 2
          },
          kindText: {
            value: `${item.category || 'Other'} - ${item.kind || 'image'}`,
            x: x + 76,
            y: by + 38,
            color: '#2f7dd6',
            font: '800 9px system-ui',
            maxWidth: w - 92,
            lineHeight: 10,
            maxLines: 1
          },
          statusText: {
            value: `${getAssetPreviewStatusLabel(item)} - ${getAssetPreviewSourceLabel(item)}`,
            x: x + 76,
            y: by + 53,
            color: '#5f6f7a',
            font: '800 8px system-ui',
            maxWidth: w - 92,
            lineHeight: 9,
            maxLines: 1
          },
          region: { type: 'asset-preview-select', path: item.path, x: x + 8, y: by, w: w - 16, h: entryRowH }
        };
      }).filter(Boolean),
      scrollIndicator: getAssetPreviewScrollIndicatorMetadata(x, y + 6, w, h - 12, contentH, clampedScroll)
    };
  }

  function getAssetPreviewEntryHeaderMetadata(activeCategory, entryCount, x, y, w, paneH) {
    const category = activeCategory || {};
    const headerH = 42;
    const listY = y + headerH;
    const listH = Math.max(80, Number(paneH || 0) - headerH - 6);
    return {
      headerH,
      titleText: {
        value: category.label || 'Assets',
        x: x + 10,
        y: y + 9,
        color: '#102033',
        font: '900 12px system-ui',
        maxWidth: w - 20,
        lineHeight: 13,
        maxLines: 1
      },
      countText: {
        value: `${Number(entryCount || 0)} entries`,
        x: x + 10,
        y: y + 26,
        color: '#5f6f7a',
        font: '800 9px system-ui',
        maxWidth: w - 20,
        lineHeight: 10,
        maxLines: 1
      },
      listY,
      listH
    };
  }

  function getAssetPreviewDetailSummaryMetadata(selected, selectedState, x, y, w, h) {
    if (!selected) {
      return {
        emptyText: {
          value: 'No matching assets.',
          x: x + 12,
          y: y + 16,
          color: '#5f6f7a',
          font: '12px system-ui',
          maxWidth: w - 24,
          lineHeight: 14
        }
      };
    }
    const detailPad = 10;
    const stageH = Math.min(174, Math.max(126, Math.floor(h * 0.32)));
    let detailY = y + detailPad + stageH + 10;
    const titleText = {
      value: selected.label || selected.path,
      x: x + detailPad,
      y: detailY,
      color: '#102033',
      font: '900 13px system-ui',
      maxWidth: w - detailPad * 2,
      lineHeight: 15,
      maxLines: 2
    };
    detailY += 36;
    const metaText = {
      value: `${(selected.categories || [selected.category || 'Other']).join(' / ')} - ${selected.kind || 'image'} - ${getAssetPreviewSourceLabel(selected)} - ${getAssetPreviewStatusLabel(selected)}`,
      x: x + detailPad,
      y: detailY,
      color: '#2f7dd6',
      font: '850 10px system-ui',
      maxWidth: w - detailPad * 2,
      lineHeight: 12,
      maxLines: 2
    };
    detailY += 28;
    const pathText = {
      value: selected.path,
      x: x + detailPad,
      y: detailY,
      color: '#5f6f7a',
      font: '800 9px monospace',
      maxWidth: w - detailPad * 2,
      lineHeight: 11,
      maxLines: 2
    };
    detailY += 30;
    const animationMetaText = selected.animation && selectedState ? {
      value: `Frame ${selected.animation.frameWidth}x${selected.animation.frameHeight} - ${selectedState.frames} frames - ${selectedState.fps} FPS - ${selectedState.loop ? 'loop' : 'one-shot'}`,
      x: x + detailPad,
      y: detailY,
      color: '#5f6f7a',
      font: '850 9px system-ui',
      maxWidth: w - detailPad * 2,
      lineHeight: 11,
      maxLines: 1
    } : null;
    const dimensionsText = selected.animation && selectedState ? null : {
      value: selected.width && selected.height ? `${selected.width}x${selected.height}` : 'Dimensions pending',
      x: x + detailPad,
      y: detailY,
      color: '#5f6f7a',
      font: '850 10px system-ui',
      maxWidth: w - detailPad * 2,
      lineHeight: 12,
      maxLines: 1
    };
    return {
      detailPad,
      stageH,
      preview: {
        x: x + detailPad,
        y: y + detailPad,
        w: w - detailPad * 2,
        h: stageH
      },
      titleText,
      metaText,
      pathText,
      animationMetaText,
      dimensionsText,
      nextY: animationMetaText ? detailY + 18 : detailY
    };
  }

  function getAssetPreviewAnimationControlsMetadata(groups, selectedState, paused, x, startY, w, paneY, paneH, detailPad) {
    const sourceGroups = Array.isArray(groups) ? groups : [];
    const pad = Number(detailPad || 0);
    const stateGap = 5;
    const stateW = Math.max(52, Math.floor((w - pad * 2 - stateGap * 2) / 3));
    let detailY = Number(startY || 0);
    const limitY = paneY + paneH;
    const groupMetadata = [];
    sourceGroups.forEach((group) => {
      const entry = group || {};
      const states = Array.isArray(entry.states) ? entry.states : [];
      if (detailY > limitY - 46) return;
      const labelText = {
        value: entry.label,
        x: x + pad,
        y: detailY,
        color: '#102033',
        font: '900 9px system-ui',
        maxWidth: w - pad * 2,
        lineHeight: 10,
        maxLines: 1
      };
      const buttonStartY = detailY + 14;
      const stateButtons = states.map((animationState, index) => {
        const state = animationState || {};
        const bx = x + pad + (index % 3) * (stateW + stateGap);
        const by = buttonStartY + Math.floor(index / 3) * 25;
        if (by > limitY - 42) return null;
        const selected = !!(selectedState && state.id === selectedState.id);
        return {
          label: state.label || state.id,
          x: bx,
          y: by,
          w: stateW,
          h: 21,
          region: { type: 'asset-preview-state', stateId: state.id },
          disabled: false,
          selectedHighlight: selected ? {
            x: bx + 2,
            y: by + 2,
            w: stateW - 4,
            h: 17,
            radius: 5,
            fill: 'rgba(47,125,214,0.14)',
            stroke: 'rgba(47,125,214,0.68)'
          } : null
        };
      }).filter(Boolean);
      groupMetadata.push({
        labelText,
        stateButtons
      });
      detailY = buttonStartY + Math.ceil(states.length / 3) * 25 + 6;
    });
    return {
      stateGap,
      stateW,
      groups: groupMetadata,
      toggleButton: {
        label: paused ? 'Play' : 'Pause',
        x: x + pad,
        y: paneY + paneH - 34,
        w: 76,
        h: 24,
        region: { type: 'asset-preview-toggle' },
        disabled: false
      },
      nextY: detailY
    };
  }

  function getAssetPreviewRegionAction(region) {
    const source = region || {};
    if (source.type === 'asset-preview-open') return { handled: true, type: 'open' };
    if (source.type === 'asset-preview-category') return { handled: true, type: 'category', category: source.category };
    if (source.type === 'asset-preview-source') return { handled: true, type: 'source', sourceType: source.sourceType };
    if (source.type === 'asset-preview-select') return { handled: true, type: 'select', path: source.path };
    if (source.type === 'asset-preview-state') return { handled: true, type: 'state', stateId: source.stateId };
    if (source.type === 'asset-preview-toggle') return { handled: true, type: 'toggle' };
    return { handled: false, type: '' };
  }

  function getAssetPreviewDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    if (hasAttribute('data-starfall-asset-preview-open')) return { handled: true, type: 'open' };
    const category = getAttribute('data-starfall-asset-preview-category');
    if (category) return { handled: true, type: 'category', category };
    const sourceType = getAttribute('data-starfall-asset-preview-source');
    if (sourceType) return { handled: true, type: 'source', sourceType };
    const path = getAttribute('data-starfall-asset-preview-select');
    if (path) return { handled: true, type: 'select', path };
    const stateId = getAttribute('data-starfall-asset-preview-state');
    if (stateId) return { handled: true, type: 'state', stateId };
    if (hasAttribute('data-starfall-asset-preview-toggle')) return { handled: true, type: 'toggle' };
    return { handled: false, type: '' };
  }

  function getAssetPreviewInputDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const hasAttribute = source && typeof source.hasAttribute === 'function'
      ? (name) => source.hasAttribute(name)
      : () => false;
    const value = source && Object.prototype.hasOwnProperty.call(source, 'value') ? source.value : undefined;
    if (hasAttribute('data-starfall-asset-preview-category')) return { handled: true, type: 'setCategory', value };
    if (hasAttribute('data-starfall-asset-preview-source')) {
      return { handled: true, type: 'setSourceType', value: value || getAttribute('data-starfall-asset-preview-source') };
    }
    if (hasAttribute('data-starfall-asset-preview-query')) return { handled: true, type: 'setQuery', value };
    return { handled: false, type: '' };
  }

  function createAssetPreviewUiHelpers() {
    return Object.freeze({
      normalizeAssetPreviewCategory,
      normalizeAssetPreviewSource,
      getAssetPreviewSourceLabel,
      getAssetPreviewStatusLabel,
      getAssetPreviewEntryCategories,
      getAssetPreviewCounts,
      getAssetPreviewEntriesForCategory,
      getAssetPreviewEntriesForSource,
      getFilteredAssetPreviewCatalog,
      getAssetPreviewSelectionMetadata,
      getAssetPreviewCategoryChangeMetadata,
      getAssetPreviewAssetSelectionMetadata,
      getAssetPreviewSourceTypeChangeMetadata,
      getAssetPreviewQueryChangeMetadata,
      getAssetPreviewQueryInput,
      getAssetPreviewAnimationStateChangeMetadata,
      getAssetPreviewPausedToggleMetadata,
      getAssetPreviewCategoryButtonMetadata,
      getAssetPreviewSourceToggleMetadata,
      getAssetPreviewThumbMetadata,
      getAssetPreviewDetailMetadata,
      getAssetPreviewPanelMetadata,
      getAssetPreviewArtMetadata,
      getAssetPreviewCategorySummaries,
      getAssetPreviewAnimationGroups,
      getAssetPreviewAnimationState,
      getAssetPreviewDefaultAnimationState,
      getAssetPreviewAnimationStyle,
      getAssetPreviewAnimationFrameIndex,
      getAssetPreviewAdminControlsMetadata,
      getAssetPreviewScrollIndicatorMetadata,
      getAssetPreviewCanvasLayout,
      getAssetPreviewHeaderMetadata,
      getAssetPreviewCategoryListMetadata,
      getAssetPreviewEntryListMetadata,
      getAssetPreviewEntryHeaderMetadata,
      getAssetPreviewDetailSummaryMetadata,
      getAssetPreviewAnimationControlsMetadata,
      getAssetPreviewRegionAction,
      getAssetPreviewDomAction,
      getAssetPreviewInputDomAction
    });
  }

  function createAssetPreviewDomSelectorUiHelpers() {
    return Object.freeze({
      DOM_ASSET_PREVIEW_QUERY_INPUT_ATTRIBUTES,
      DOM_ASSET_PREVIEW_QUERY_INPUT_SELECTOR
    });
  }

  const api = {
    ASSET_PREVIEW_CATEGORY_OPTIONS,
    ASSET_PREVIEW_SOURCE_OPTIONS,
    DOM_ASSET_PREVIEW_QUERY_INPUT_ATTRIBUTES,
    DOM_ASSET_PREVIEW_QUERY_INPUT_SELECTOR,
    createAssetPreviewUiHelpers,
    createAssetPreviewDomSelectorUiHelpers,
    normalizeAssetPreviewCategory,
    normalizeAssetPreviewSource,
    getAssetPreviewSourceLabel,
    getAssetPreviewStatusLabel,
    getAssetPreviewEntryCategories,
    getAssetPreviewCounts,
    getAssetPreviewEntriesForCategory,
    getAssetPreviewEntriesForSource,
    getFilteredAssetPreviewCatalog,
    getAssetPreviewSelectionMetadata,
    getAssetPreviewCategoryChangeMetadata,
    getAssetPreviewAssetSelectionMetadata,
    getAssetPreviewSourceTypeChangeMetadata,
    getAssetPreviewQueryChangeMetadata,
    getAssetPreviewQueryInput,
    getAssetPreviewAnimationStateChangeMetadata,
    getAssetPreviewPausedToggleMetadata,
    getAssetPreviewCategoryButtonMetadata,
    getAssetPreviewSourceToggleMetadata,
    getAssetPreviewThumbMetadata,
    getAssetPreviewDetailMetadata,
    getAssetPreviewPanelMetadata,
    getAssetPreviewArtMetadata,
    getAssetPreviewCategorySummaries,
    getAssetPreviewAnimationGroups,
    getAssetPreviewAnimationState,
    getAssetPreviewDefaultAnimationState,
    getAssetPreviewAnimationStyle,
    getAssetPreviewAnimationFrameIndex,
    getAssetPreviewAdminControlsMetadata,
    getAssetPreviewScrollIndicatorMetadata,
    getAssetPreviewCanvasLayout,
    getAssetPreviewHeaderMetadata,
    getAssetPreviewCategoryListMetadata,
    getAssetPreviewEntryListMetadata,
    getAssetPreviewEntryHeaderMetadata,
    getAssetPreviewDetailSummaryMetadata,
    getAssetPreviewAnimationControlsMetadata,
    getAssetPreviewRegionAction,
    getAssetPreviewDomAction,
    getAssetPreviewInputDomAction
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.assetPreview = Object.assign({}, modules.assetPreview || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
