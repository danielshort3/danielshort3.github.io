(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  const form = $('#povcheck-form');
  const textInput = $('#povcheck-text');
  const includeItToggle = $('#povcheck-include-it');
  const thirdReferencesInput = $('#povcheck-third-references');
  const ignoreTermsInput = $('#povcheck-ignore-terms');
  const ignoreQuotedToggle = $('#povcheck-ignore-quoted');
  const ignoreMetadataToggle = $('#povcheck-ignore-metadata');
  const modeBasicInput = $('#povcheck-mode-basic');
  const modeAdvancedInput = $('#povcheck-mode-advanced');
  const advancedPanel = $('#povcheck-advanced-panel');

  const summaryEl = $('#povcheck-summary');
  const outputEl = $('#povcheck-output');
  const clearBtn = $('#povcheck-clear');

  const pasteBtn = $('#povcheck-paste');
  const importBtn = $('#povcheck-import');
  const fileInput = $('#povcheck-file');
  const inputStatusEl = $('#povcheck-input-status');

  const copyResultsBtn = $('#povcheck-copy-results');
  const exportCsvBtn = $('#povcheck-export-csv');
  const exportJsonBtn = $('#povcheck-export-json');
  const copyHtmlBtn = $('#povcheck-copy-html');
  const resultsStatusEl = $('#povcheck-results-status');

  const firstColorInput = $('#povcheck-first-color');
  const secondColorInput = $('#povcheck-second-color');
  const thirdColorInput = $('#povcheck-third-color');
  const resetColorsBtn = $('#povcheck-reset-colors');

  const firstBadge = $('#povcheck-first-badge');
  const firstCount = $('#povcheck-first-count');
  const firstList = $('#povcheck-first-list');
  const secondBadge = $('#povcheck-second-badge');
  const secondCount = $('#povcheck-second-count');
  const secondList = $('#povcheck-second-list');
  const thirdBadge = $('#povcheck-third-badge');
  const thirdCount = $('#povcheck-third-count');
  const thirdList = $('#povcheck-third-list');

  const driftSummaryEl = $('#povcheck-drift-summary');
  const driftListEl = $('#povcheck-drift-list');

  const presetNameInput = $('#povcheck-preset-name');
  const presetSelect = $('#povcheck-preset-select');
  const presetSaveBtn = $('#povcheck-preset-save');
  const presetLoadBtn = $('#povcheck-preset-load');
  const presetDeleteBtn = $('#povcheck-preset-delete');
  const presetExportBtn = $('#povcheck-preset-export');
  const presetImportBtn = $('#povcheck-preset-import');
  const presetImportFile = $('#povcheck-presets-file');
  const presetStatusEl = $('#povcheck-preset-status');
  const shareConfigBtn = $('#povcheck-share-config');

  if (!form || !textInput || !summaryEl || !outputEl) return;
  if (!firstBadge || !firstCount || !firstList) return;
  if (!secondBadge || !secondCount || !secondList) return;
  if (!thirdBadge || !thirdCount || !thirdList) return;
  if (!driftSummaryEl || !driftListEl) return;

  const TOOL_ID = 'point-of-view-checker';
  const MAX_IMPORT_BYTES = 24 * 1024 * 1024;
  const MAX_SAVED_OUTPUT_HTML_CHARS = 120_000;
  const MAX_DRIFT_ROWS = 220;
  const PDF_WORKER_PATH = '/js/vendor/pdfjs/pdf.worker.min.js';

  const COLOR_STORAGE_KEY = 'povcheck-highlight-colors';
  const PRESET_STORAGE_KEY = 'povcheck-presets-v1';

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const WORD_REGEX = (() => {
    try {
      new RegExp('\\p{L}', 'u');
      return {
        source: "[\\p{L}\\p{N}]+(?:['’][\\p{L}\\p{N}]+)*",
        flags: 'gu',
        unicode: true,
        letter: /\p{L}/u
      };
    } catch {
      return {
        source: "[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*",
        flags: 'g',
        unicode: false,
        letter: /[A-Za-z]/
      };
    }
  })();

  const WORD_BOUNDARY_CLASS = WORD_REGEX.unicode ? '\\p{L}\\p{N}_' : 'A-Za-z0-9_';

  const createWordRegex = () => new RegExp(WORD_REGEX.source, WORD_REGEX.flags);

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const escapeAttr = (value) => escapeHtml(value).replace(/'/g, '&#39;');

  const escapeRegExp = (value) => String(value || '')
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const collapseWhitespace = (value) => String(value || '')
    .trim()
    .replace(/\s+/g, ' ');

  const formatNumber = (value) => Number(value || 0).toLocaleString('en-US');

  const normalizeText = (text) => String(text || '')
    .replace(/\u0000/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/[\u2018\u2019\u201B\uFF07`]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/\u00A0/g, ' ');

  const normalizeImportedText = (text) => normalizeText(text)
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const normalizeToken = (token) => normalizeText(token).toLowerCase();

  const clampText = (value, maxChars) => {
    const text = String(value || '');
    if (text.length <= maxChars) return { text, truncated: false };
    return { text: text.slice(0, maxChars), truncated: true };
  };

  const setInputStatus = (message, tone) => {
    if (!inputStatusEl) return;
    inputStatusEl.textContent = String(message || '');
    inputStatusEl.dataset.tone = tone || '';
  };

  const setResultsStatus = (message, tone) => {
    if (!resultsStatusEl) return;
    resultsStatusEl.textContent = String(message || '');
    resultsStatusEl.dataset.tone = tone || '';
  };

  const setPresetStatus = (message, tone) => {
    if (!presetStatusEl) return;
    presetStatusEl.textContent = String(message || '');
    presetStatusEl.dataset.tone = tone || '';
  };

  const setInputBusy = (busy) => {
    const disabled = Boolean(busy);
    if (pasteBtn) pasteBtn.disabled = disabled;
    if (importBtn) importBtn.disabled = disabled;
    if (fileInput) fileInput.disabled = disabled;
  };

  const normalizeHexColor = (value, fallback) => {
    const s = String(value || '').trim();
    if (/^#[0-9a-f]{6}$/i.test(s)) return s.toUpperCase();
    return fallback;
  };

  const getDefaultHighlightColors = () => ({
    first: normalizeHexColor(firstColorInput?.defaultValue, '#2396AD'),
    second: normalizeHexColor(secondColorInput?.defaultValue, '#E0A328'),
    third: normalizeHexColor(thirdColorInput?.defaultValue, '#38B37E')
  });

  const getHighlightColors = () => {
    const defaults = getDefaultHighlightColors();
    return {
      first: normalizeHexColor(firstColorInput?.value, defaults.first),
      second: normalizeHexColor(secondColorInput?.value, defaults.second),
      third: normalizeHexColor(thirdColorInput?.value, defaults.third)
    };
  };

  const applyHighlightColors = (colors) => {
    if (!colors || !document.body?.style) return;
    document.body.style.setProperty('--povcheck-first-color', colors.first);
    document.body.style.setProperty('--povcheck-second-color', colors.second);
    document.body.style.setProperty('--povcheck-third-color', colors.third);
  };

  const saveHighlightColors = (colors) => {
    try {
      window.localStorage.setItem(COLOR_STORAGE_KEY, JSON.stringify(colors));
    } catch {}
  };

  const clearStoredHighlightColors = () => {
    try {
      window.localStorage.removeItem(COLOR_STORAGE_KEY);
    } catch {}
  };

  const loadHighlightColors = () => {
    try {
      const raw = window.localStorage.getItem(COLOR_STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return null;
      const defaults = getDefaultHighlightColors();
      return {
        first: normalizeHexColor(parsed.first, defaults.first),
        second: normalizeHexColor(parsed.second, defaults.second),
        third: normalizeHexColor(parsed.third, defaults.third)
      };
    } catch {
      return null;
    }
  };

  const syncHighlightControls = () => {
    const stored = loadHighlightColors();
    const colors = stored || getHighlightColors();
    if (stored) {
      if (firstColorInput) firstColorInput.value = colors.first;
      if (secondColorInput) secondColorInput.value = colors.second;
      if (thirdColorInput) thirdColorInput.value = colors.third;
    }
    applyHighlightColors(colors);
  };

  const FIRST_PERSON = new Set([
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
    "i'm", "i'd", "i'll", "i've", "we're", "we'd", "we'll", "we've"
  ]);

  const SECOND_PERSON = new Set([
    'you', 'your', 'yours', 'yourself', 'yourselves',
    "you're", "you'd", "you'll", "you've", "y'all"
  ]);

  const THIRD_PERSON_BASE = new Set([
    'he', 'him', 'his', 'himself', "he's", "he'd", "he'll",
    'she', 'her', 'hers', 'herself', "she's", "she'd", "she'll",
    'they', 'them', 'their', 'theirs', 'themself', 'themselves', "they're", "they'd", "they'll", "they've"
  ]);

  const THIRD_PERSON_NEUTRAL = new Set(['it', 'its', 'itself', "it's"]);

  const buildThirdSet = (includeNeutral) => {
    if (!includeNeutral) return THIRD_PERSON_BASE;
    return new Set([...THIRD_PERSON_BASE, ...THIRD_PERSON_NEUTRAL]);
  };

  const parseThirdReferencesText = (rawText) => {
    const rawLines = String(rawText || '').split(/\r?\n/);
    const seen = new Set();
    const phrases = [];

    rawLines.forEach((line) => {
      const normalized = collapseWhitespace(normalizeText(line)).toLowerCase();
      if (!normalized) return;
      if (seen.has(normalized)) return;
      seen.add(normalized);
      phrases.push(normalized);
    });

    return phrases;
  };

  const parseIgnoreTermsText = (rawText) => {
    const tokens = String(rawText || '').split(/[\n,;]+/);
    const seen = new Set();
    const phrases = [];
    tokens.forEach((token) => {
      const normalized = collapseWhitespace(normalizeText(token)).toLowerCase();
      if (!normalized) return;
      if (seen.has(normalized)) return;
      seen.add(normalized);
      phrases.push(normalized);
    });
    return phrases;
  };

  const buildBoundaryRegex = (body) => new RegExp(
    `(^|[^${WORD_BOUNDARY_CLASS}])(${body})(?=$|[^${WORD_BOUNDARY_CLASS}])`,
    WORD_REGEX.unicode ? 'gu' : 'g'
  );

  const buildPhraseRegex = (phrase, options) => {
    const normalized = collapseWhitespace(normalizeText(phrase)).toLowerCase();
    if (!normalized) return null;

    const allowPossessive = Boolean(options?.allowPossessive);
    const alreadyPossessive = /(?:'s|')$/.test(normalized);
    const body = normalized
      .split(' ')
      .map(escapeRegExp)
      .join('\\s+');
    const suffix = allowPossessive && !alreadyPossessive ? "(?:'s)?" : '';

    return buildBoundaryRegex(`${body}${suffix}`);
  };

  const collectPhraseMatches = (text, phrases, options) => {
    const source = String(text || '');
    const output = [];

    phrases.forEach((phrase) => {
      const rx = buildPhraseRegex(phrase, options);
      if (!rx) return;
      rx.lastIndex = 0;
      let match;
      while ((match = rx.exec(source)) !== null) {
        const prefix = String(match[1] || '');
        const target = String(match[2] || '');
        const start = match.index + prefix.length;
        const end = start + target.length;
        if (end > start) {
          output.push({
            phrase,
            start,
            end,
            matchText: target
          });
        }
        if (rx.lastIndex === match.index) rx.lastIndex += 1;
      }
    });

    return output;
  };

  const mergeRanges = (ranges) => {
    const sorted = (ranges || [])
      .filter((range) => Number.isFinite(range?.start) && Number.isFinite(range?.end) && range.end > range.start)
      .sort((a, b) => a.start - b.start || a.end - b.end);

    const merged = [];
    sorted.forEach((range) => {
      const last = merged[merged.length - 1];
      if (!last || range.start > last.end) {
        merged.push({ start: range.start, end: range.end });
        return;
      }
      if (range.end > last.end) last.end = range.end;
    });
    return merged;
  };

  const rangeOverlaps = (start, end, ranges) => {
    for (let i = 0; i < ranges.length; i += 1) {
      const range = ranges[i];
      if (end <= range.start) return false;
      if (start < range.end && end > range.start) return true;
    }
    return false;
  };

  const stripQuotedText = (text) => {
    const source = String(text || '');
    if (!source) return '';

    let output = '';
    let inQuote = false;
    let escaped = false;

    for (let i = 0; i < source.length; i += 1) {
      const ch = source[i];
      if (inQuote) {
        output += ' ';
        if (ch === '"' && !escaped) inQuote = false;
        escaped = ch === '\\' && !escaped;
        continue;
      }
      if (ch === '"') {
        inQuote = true;
        escaped = false;
        output += ' ';
        continue;
      }
      escaped = false;
      output += ch;
    }

    return output;
  };

  const stripMetadataLines = (text) => {
    const lines = String(text || '').split('\n');
    let inFrontMatter = false;

    return lines.map((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) return line;

      if (index === 0 && /^---+$/.test(trimmed)) {
        inFrontMatter = true;
        return ' '.repeat(line.length);
      }

      if (inFrontMatter) {
        const mask = ' '.repeat(line.length);
        if (/^---+$/.test(trimmed)) inFrontMatter = false;
        return mask;
      }

      const isMarkdownHeading = /^#{1,6}\s+\S+/.test(trimmed);
      const isUpperHeading = /^[A-Z][A-Z0-9 &'/-]{4,}$/.test(trimmed) && trimmed.length <= 90;
      const isHeadingLabel = /^[A-Za-z][A-Za-z0-9 _-]{1,32}:\s*$/.test(trimmed);
      const isMetadataLine = /^[A-Za-z][A-Za-z0-9 _-]{1,32}:\s+\S+/.test(trimmed) && trimmed.length <= 140;

      if (isMarkdownHeading || isUpperHeading || isHeadingLabel || isMetadataLine) {
        return ' '.repeat(line.length);
      }
      return line;
    }).join('\n');
  };

  const finalizeMatches = (matches) => {
    const sorted = [...(matches || [])].sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return (b.end - b.start) - (a.end - a.start);
    });

    const finalRanges = [];
    let lastEnd = -1;

    sorted.forEach((range) => {
      if (!range || range.end <= range.start) return;
      if (range.start < lastEnd) return;
      finalRanges.push(range);
      lastEnd = range.end;
    });

    return finalRanges;
  };

  const createGroupState = () => ({
    total: 0,
    counts: new Map(),
    labels: new Map()
  });

  const addGroupHit = (group, key, label) => {
    group.total += 1;
    group.counts.set(key, (group.counts.get(key) || 0) + 1);
    if (!group.labels.has(key) && label) group.labels.set(key, label);
  };

  const getCurrentMode = () => (modeAdvancedInput?.checked ? 'advanced' : 'basic');

  const syncModeUi = () => {
    const advanced = getCurrentMode() === 'advanced';
    if (advancedPanel) advancedPanel.hidden = !advanced;
    form.dataset.mode = advanced ? 'advanced' : 'basic';
  };

  const normalizeMultiline = (value, maxLen) => String(value || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .slice(0, maxLen);

  const getRawConfigFromControls = () => ({
    mode: getCurrentMode(),
    includeNeutral: Boolean(includeItToggle?.checked),
    ignoreQuoted: Boolean(ignoreQuotedToggle?.checked),
    ignoreMetadata: Boolean(ignoreMetadataToggle?.checked),
    ignoreTerms: normalizeMultiline(ignoreTermsInput?.value || '', 12_000),
    thirdReferences: normalizeMultiline(thirdReferencesInput?.value || '', 12_000),
    colors: getHighlightColors()
  });

  const getEffectiveConfig = () => {
    const raw = getRawConfigFromControls();
    if (raw.mode === 'advanced') return raw;
    return {
      ...raw,
      includeNeutral: true,
      ignoreQuoted: false,
      ignoreMetadata: false,
      ignoreTerms: '',
      thirdReferences: ''
    };
  };

  const sanitizeConfig = (raw) => {
    const source = raw && typeof raw === 'object' ? raw : {};
    const defaults = getDefaultHighlightColors();

    const mode = source.mode === 'advanced' ? 'advanced' : 'basic';
    const colors = source.colors && typeof source.colors === 'object' ? source.colors : {};

    return {
      mode,
      includeNeutral: source.includeNeutral !== false,
      ignoreQuoted: Boolean(source.ignoreQuoted),
      ignoreMetadata: Boolean(source.ignoreMetadata),
      ignoreTerms: normalizeMultiline(source.ignoreTerms || '', 12_000),
      thirdReferences: normalizeMultiline(source.thirdReferences || '', 12_000),
      colors: {
        first: normalizeHexColor(colors.first, defaults.first),
        second: normalizeHexColor(colors.second, defaults.second),
        third: normalizeHexColor(colors.third, defaults.third)
      }
    };
  };

  const applyConfigToControls = (config, options) => {
    const safe = sanitizeConfig(config);

    if (modeBasicInput && modeAdvancedInput) {
      modeBasicInput.checked = safe.mode !== 'advanced';
      modeAdvancedInput.checked = safe.mode === 'advanced';
    }

    if (includeItToggle) includeItToggle.checked = safe.includeNeutral;
    if (ignoreQuotedToggle) ignoreQuotedToggle.checked = safe.ignoreQuoted;
    if (ignoreMetadataToggle) ignoreMetadataToggle.checked = safe.ignoreMetadata;
    if (ignoreTermsInput) ignoreTermsInput.value = safe.ignoreTerms;
    if (thirdReferencesInput) thirdReferencesInput.value = safe.thirdReferences;

    if (firstColorInput) firstColorInput.value = safe.colors.first;
    if (secondColorInput) secondColorInput.value = safe.colors.second;
    if (thirdColorInput) thirdColorInput.value = safe.colors.third;
    applyHighlightColors(safe.colors);

    syncModeUi();

    if (options?.persistColors) saveHighlightColors(safe.colors);
    if (options?.markDirty) markSessionDirty();
  };

  const getEffectiveInput = () => {
    const raw = textInput.value || '';
    const hasUser = Boolean(raw.trim());
    return {
      text: hasUser ? raw : (textInput.placeholder || ''),
      hasUser
    };
  };

  const buildProcessedText = (sourceText, config) => {
    let processed = normalizeText(sourceText);
    if (config.ignoreQuoted) processed = stripQuotedText(processed);
    if (config.ignoreMetadata) processed = stripMetadataLines(processed);
    return processed;
  };

  const collectPronounMatches = (lowerProcessed, sourceText, options) => {
    const thirdSet = buildThirdSet(options.includeNeutral);
    const matches = [];
    const rx = createWordRegex();

    let match;
    while ((match = rx.exec(lowerProcessed)) !== null) {
      const tokenText = String(match[0] || '');
      if (!tokenText) continue;
      const start = match.index;
      const end = start + tokenText.length;

      if (rangeOverlaps(start, end, options.ignoreRanges)) continue;
      if (rangeOverlaps(start, end, options.thirdRanges)) continue;

      const key = normalizeToken(tokenText);

      let pov = '';
      if (FIRST_PERSON.has(key)) pov = 'first';
      else if (SECOND_PERSON.has(key)) pov = 'second';
      else if (thirdSet.has(key)) pov = 'third';

      if (!pov) continue;

      const label = collapseWhitespace(sourceText.slice(start, end)) || key;
      matches.push({
        start,
        end,
        pov,
        key,
        label,
        type: 'pronoun'
      });
    }

    return matches;
  };

  const collectThirdReferenceMatches = (lowerProcessed, sourceText, phrases, ignoreRanges) => {
    const rawMatches = collectPhraseMatches(lowerProcessed, phrases, { allowPossessive: true });

    return rawMatches
      .filter((match) => !rangeOverlaps(match.start, match.end, ignoreRanges))
      .map((match) => {
        const key = collapseWhitespace(match.matchText).toLowerCase();
        const label = collapseWhitespace(sourceText.slice(match.start, match.end)) || key;
        return {
          start: match.start,
          end: match.end,
          pov: 'third',
          key,
          label,
          type: 'reference'
        };
      });
  };

  const splitSentenceSpans = (processedText, sourceText) => {
    const spans = [];
    const rx = /[^.!?\n]+(?:[.!?]+["')\]]*)?(?:\s+|$)|\n+/g;
    let match;

    while ((match = rx.exec(processedText)) !== null) {
      const chunk = String(match[0] || '');
      if (!chunk) continue;

      const firstNonSpace = chunk.search(/\S/);
      if (firstNonSpace < 0) continue;

      let trimEnd = chunk.length;
      while (trimEnd > firstNonSpace && /\s/.test(chunk[trimEnd - 1])) {
        trimEnd -= 1;
      }

      if (trimEnd <= firstNonSpace) continue;

      const start = match.index + firstNonSpace;
      const end = match.index + trimEnd;
      const sentenceText = sourceText.slice(start, end);
      if (!WORD_REGEX.letter.test(sentenceText)) continue;

      spans.push({ start, end, text: sentenceText });
    }

    if (!spans.length && collapseWhitespace(sourceText)) {
      spans.push({ start: 0, end: sourceText.length, text: sourceText });
    }

    return spans;
  };

  const computeDrift = (matches, processedText, sourceText) => {
    const sentences = splitSentenceSpans(processedText, sourceText);
    const rows = [];

    let switches = 0;
    let lastDominant = '';

    sentences.forEach((sentence, index) => {
      const counts = { first: 0, second: 0, third: 0 };
      let total = 0;

      matches.forEach((range) => {
        if (range.start >= sentence.start && range.end <= sentence.end) {
          counts[range.pov] += 1;
          total += 1;
        }
      });

      let dominant = 'none';
      if (total > 0) {
        const ordered = [
          { key: 'first', value: counts.first },
          { key: 'second', value: counts.second },
          { key: 'third', value: counts.third }
        ].sort((a, b) => b.value - a.value);

        if (ordered[0].value === ordered[1].value) dominant = 'mixed';
        else dominant = ordered[0].key;

        if (lastDominant && dominant !== lastDominant) switches += 1;
        lastDominant = dominant;
      }

      rows.push({
        index: index + 1,
        start: sentence.start,
        end: sentence.end,
        counts,
        total,
        dominant,
        snippet: collapseWhitespace(sentence.text).slice(0, 200)
      });
    });

    const highlightedRows = rows.filter((row) => row.total > 0);
    const dominantCounts = { first: 0, second: 0, third: 0, mixed: 0, none: 0 };
    highlightedRows.forEach((row) => {
      dominantCounts[row.dominant] = (dominantCounts[row.dominant] || 0) + 1;
    });

    return {
      rows: highlightedRows.slice(0, MAX_DRIFT_ROWS),
      totalSentences: sentences.length,
      highlightedSentences: highlightedRows.length,
      switches,
      dominantCounts
    };
  };

  const buildAnalysis = ({ text, hasUser }, config) => {
    const sourceText = normalizeText(text);
    const hasText = Boolean(collapseWhitespace(sourceText));

    const groups = {
      first: createGroupState(),
      second: createGroupState(),
      third: createGroupState()
    };

    if (!hasText) {
      return {
        sourceText,
        hasText,
        hasUser,
        config,
        groups,
        total: 0,
        matches: [],
        thirdReferences: [],
        ignoreTerms: [],
        drift: {
          rows: [],
          totalSentences: 0,
          highlightedSentences: 0,
          switches: 0,
          dominantCounts: { first: 0, second: 0, third: 0, mixed: 0, none: 0 }
        }
      };
    }

    const processedText = buildProcessedText(sourceText, config);
    const lowerProcessed = processedText.toLowerCase();

    const ignoreTerms = parseIgnoreTermsText(config.ignoreTerms);
    const ignoreMatches = collectPhraseMatches(lowerProcessed, ignoreTerms, { allowPossessive: false });
    const ignoreRanges = mergeRanges(ignoreMatches);

    const thirdReferences = parseThirdReferencesText(config.thirdReferences);
    const thirdMatches = collectThirdReferenceMatches(lowerProcessed, sourceText, thirdReferences, ignoreRanges);
    const thirdRanges = mergeRanges(thirdMatches);

    const pronounMatches = collectPronounMatches(lowerProcessed, sourceText, {
      includeNeutral: config.includeNeutral,
      ignoreRanges,
      thirdRanges
    });

    const matches = finalizeMatches([...thirdMatches, ...pronounMatches]);

    matches.forEach((match) => {
      addGroupHit(groups[match.pov], match.key, match.label);
    });

    const total = groups.first.total + groups.second.total + groups.third.total;

    return {
      sourceText,
      hasText,
      hasUser,
      config,
      groups,
      total,
      matches,
      thirdReferences,
      ignoreTerms,
      drift: computeDrift(matches, processedText, sourceText)
    };
  };

  const renderBadge = (badgeEl, total) => {
    const detected = total > 0;
    badgeEl.textContent = detected ? 'Detected' : 'Not found';
    badgeEl.classList.toggle('povcheck-badge-detected', detected);
    badgeEl.classList.toggle('povcheck-badge-muted', !detected);
  };

  const sortTokenEntries = (group) => {
    const entries = [...group.counts.entries()].map(([key, count]) => ({
      key,
      count,
      label: group.labels.get(key) || key
    }));

    entries.sort((a, b) => {
      if (b.count === a.count) return a.label.localeCompare(b.label);
      return b.count - a.count;
    });

    return entries;
  };

  const activeTokenMatches = (token, pov) => {
    if (!activeTokenFilter) return false;
    return activeTokenFilter.key === token && activeTokenFilter.pov === pov;
  };

  const renderTokenList = (listEl, group, pov) => {
    const entries = sortTokenEntries(group);
    listEl.innerHTML = '';

    if (!entries.length) {
      const empty = document.createElement('li');
      empty.className = 'povcheck-token-empty';
      empty.textContent = 'None found.';
      listEl.appendChild(empty);
      return;
    }

    entries.forEach((entry) => {
      const li = document.createElement('li');
      li.className = 'povcheck-token-pill';

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'povcheck-token-btn';
      btn.dataset.pov = pov;
      btn.dataset.tokenKey = entry.key;
      btn.setAttribute('aria-pressed', activeTokenMatches(entry.key, pov) ? 'true' : 'false');
      btn.innerHTML = `
        <span class="povcheck-token">${escapeHtml(entry.label)}</span>
        <span class="povcheck-token-count">${formatNumber(entry.count)}x</span>
      `;

      li.appendChild(btn);
      listEl.appendChild(li);
    });
  };

  const dominantGroupFromTotals = (analysis) => {
    const ordered = [
      { key: 'first', value: analysis.groups.first.total },
      { key: 'second', value: analysis.groups.second.total },
      { key: 'third', value: analysis.groups.third.total }
    ].sort((a, b) => b.value - a.value);
    if (!ordered[0].value) return '';
    return ordered[0].key;
  };

  const dominantLabel = (key) => {
    if (key === 'first') return 'first person';
    if (key === 'second') return 'second person';
    if (key === 'third') return 'third person';
    if (key === 'mixed') return 'mixed';
    return 'none';
  };

  const renderSummary = (analysis) => {
    if (!analysis.hasText) {
      summaryEl.textContent = 'Paste text above and click Check.';
      return;
    }

    if (!analysis.total) {
      summaryEl.textContent = 'No pronouns or third-person references were detected with the current settings.';
      return;
    }

    const detected = [];
    if (analysis.groups.first.total) detected.push('first person');
    if (analysis.groups.second.total) detected.push('second person');
    if (analysis.groups.third.total) detected.push('third person');

    const modeLabel = analysis.config.mode === 'advanced' ? 'Advanced mode' : 'Basic mode';
    const dominant = dominantLabel(dominantGroupFromTotals(analysis));

    const parts = [
      `${modeLabel}.`,
      `Total matches: <strong>${formatNumber(analysis.total)}</strong>.`,
      `Detected: <strong>${detected.join(', ')}</strong>.`,
      dominant !== 'none' ? `Dominant POV: <strong>${dominant}</strong>.` : '',
      analysis.drift.switches ? `Sentence-level switches: <strong>${formatNumber(analysis.drift.switches)}</strong>.` : ''
    ].filter(Boolean);

    const lead = detected.length > 1 ? 'Mixed point of view. ' : '';
    summaryEl.innerHTML = `${lead}${parts.join(' ')}`;
  };

  const renderHighlightedOutput = (analysis) => {
    const text = analysis.sourceText;
    if (!analysis.hasText) {
      outputEl.innerHTML = '<p class="povcheck-empty">Waiting for input.</p>';
      return;
    }

    if (!analysis.matches.length) {
      outputEl.innerHTML = escapeHtml(text);
      return;
    }

    let html = '';
    let cursor = 0;
    const hasFilter = Boolean(activeTokenFilter);

    analysis.matches.forEach((range) => {
      html += escapeHtml(text.slice(cursor, range.start));

      const classes = ['povcheck-mark', `povcheck-mark-${range.pov}`];
      const isFocus = hasFilter && activeTokenMatches(range.key, range.pov);
      if (hasFilter) {
        if (isFocus) classes.push('povcheck-mark-focus');
        else classes.push('povcheck-mark-dim');
      }

      html += `<mark class="${classes.join(' ')}" data-pov="${escapeAttr(range.pov)}" data-key="${escapeAttr(range.key)}" data-start="${range.start}" data-end="${range.end}">${escapeHtml(text.slice(range.start, range.end))}</mark>`;
      cursor = range.end;
    });

    html += escapeHtml(text.slice(cursor));
    outputEl.innerHTML = html || '<p class="povcheck-empty">No output.</p>';
  };

  const renderDrift = (analysis) => {
    if (!analysis.hasText) {
      driftSummaryEl.textContent = 'Run a check to inspect sentence-level point-of-view shifts.';
      driftListEl.innerHTML = '<li class="povcheck-token-empty">Run a check to inspect sentence-level point-of-view shifts.</li>';
      return;
    }

    if (!analysis.drift.rows.length) {
      driftSummaryEl.textContent = 'No sentence-level POV terms were detected with the current settings.';
      driftListEl.innerHTML = '<li class="povcheck-token-empty">No sentence-level POV terms were detected.</li>';
      return;
    }

    driftSummaryEl.textContent = `Detected POV terms in ${formatNumber(analysis.drift.highlightedSentences)} of ${formatNumber(analysis.drift.totalSentences)} sentences. Switches between dominant POV: ${formatNumber(analysis.drift.switches)}.`;

    driftListEl.innerHTML = '';
    analysis.drift.rows.forEach((row) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'povcheck-drift-item';
      btn.dataset.sentenceStart = String(row.start);
      btn.dataset.sentenceEnd = String(row.end);
      btn.innerHTML = `
        <span class="povcheck-drift-meta">Sentence ${formatNumber(row.index)} · ${formatNumber(row.total)} matches</span>
        <span class="povcheck-drift-pill povcheck-drift-pill-${escapeAttr(row.dominant)}">${escapeHtml(dominantLabel(row.dominant))}</span>
        <span class="povcheck-drift-snippet">${escapeHtml(row.snippet)}</span>
      `;
      li.appendChild(btn);
      driftListEl.appendChild(li);
    });
  };

  const renderCounts = (analysis) => {
    firstCount.textContent = formatNumber(analysis.groups.first.total);
    secondCount.textContent = formatNumber(analysis.groups.second.total);
    thirdCount.textContent = formatNumber(analysis.groups.third.total);

    renderBadge(firstBadge, analysis.groups.first.total);
    renderBadge(secondBadge, analysis.groups.second.total);
    renderBadge(thirdBadge, analysis.groups.third.total);

    renderTokenList(firstList, analysis.groups.first, 'first');
    renderTokenList(secondList, analysis.groups.second, 'second');
    renderTokenList(thirdList, analysis.groups.third, 'third');
  };

  const ensureActiveTokenExists = (analysis) => {
    if (!activeTokenFilter) return;
    const group = analysis.groups[activeTokenFilter.pov];
    if (!group || !group.counts.has(activeTokenFilter.key)) {
      activeTokenFilter = null;
    }
  };

  const renderAnalysis = (analysis) => {
    ensureActiveTokenExists(analysis);
    renderCounts(analysis);
    renderSummary(analysis);
    renderHighlightedOutput(analysis);
    renderDrift(analysis);
  };

  const resetUI = () => {
    hasRun = false;
    lastAnalysis = null;
    activeTokenFilter = null;

    [firstCount, secondCount, thirdCount].forEach((el) => {
      el.textContent = '0';
    });

    [firstBadge, secondBadge, thirdBadge].forEach((el) => {
      el.textContent = 'Not found';
      el.classList.add('povcheck-badge-muted');
      el.classList.remove('povcheck-badge-detected');
    });

    [firstList, secondList, thirdList].forEach((list) => {
      list.innerHTML = '<li class="povcheck-token-empty">Waiting for input.</li>';
    });

    summaryEl.textContent = 'Paste text above and click Check.';
    outputEl.innerHTML = '<p class="povcheck-empty">Waiting for input.</p>';
    driftSummaryEl.textContent = 'Run a check to inspect sentence-level point-of-view shifts.';
    driftListEl.innerHTML = '<li class="povcheck-token-empty">Run a check to inspect sentence-level point-of-view shifts.</li>';
    setResultsStatus('');
  };

  const runAnalysis = () => {
    const input = getEffectiveInput();
    const config = getEffectiveConfig();

    const analysis = buildAnalysis(input, config);
    lastAnalysis = analysis;
    renderAnalysis(analysis);
    hasRun = true;
  };

  const csvEscape = (value) => {
    const text = String(value || '');
    if (/[,"\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
    return text;
  };

  const flattenGroupRows = (groupName, group) => {
    const entries = sortTokenEntries(group);
    return entries.map((entry) => ({
      group: groupName,
      token: entry.label,
      normalizedToken: entry.key,
      count: entry.count,
      sharePct: group.total ? ((entry.count / group.total) * 100) : 0
    }));
  };

  const buildResultsText = (analysis) => {
    const rows = [];
    rows.push('Point of View Checker Results');
    rows.push(`Mode: ${analysis.config.mode}`);
    rows.push(`Summary: ${collapseWhitespace(summaryEl.textContent || '')}`);
    rows.push('');
    rows.push(`First person: ${formatNumber(analysis.groups.first.total)}`);
    rows.push(`Second person: ${formatNumber(analysis.groups.second.total)}`);
    rows.push(`Third person: ${formatNumber(analysis.groups.third.total)}`);
    rows.push(`Total matches: ${formatNumber(analysis.total)}`);
    rows.push('');

    ['first', 'second', 'third'].forEach((groupName) => {
      const group = analysis.groups[groupName];
      rows.push(`${groupName.toUpperCase()} TOKENS`);
      if (!group.total) {
        rows.push('  - none');
      } else {
        sortTokenEntries(group).forEach((entry) => {
          rows.push(`  - ${entry.label}: ${formatNumber(entry.count)}`);
        });
      }
      rows.push('');
    });

    rows.push(`Sentence-level switches: ${formatNumber(analysis.drift.switches)}`);
    rows.push(`Sentences with POV terms: ${formatNumber(analysis.drift.highlightedSentences)} / ${formatNumber(analysis.drift.totalSentences)}`);

    return rows.join('\n').trim();
  };

  const buildResultsCsv = (analysis) => {
    const rows = [
      ['group', 'token', 'normalized_token', 'count', 'group_share_pct']
    ];

    const allRows = [
      ...flattenGroupRows('first', analysis.groups.first),
      ...flattenGroupRows('second', analysis.groups.second),
      ...flattenGroupRows('third', analysis.groups.third)
    ];

    allRows.forEach((row) => {
      rows.push([
        row.group,
        row.token,
        row.normalizedToken,
        String(row.count),
        row.sharePct.toFixed(2)
      ]);
    });

    return rows.map((row) => row.map(csvEscape).join(',')).join('\n');
  };

  const buildResultsJson = (analysis) => ({
    tool: TOOL_ID,
    generatedAt: new Date().toISOString(),
    summary: collapseWhitespace(summaryEl.textContent || ''),
    mode: analysis.config.mode,
    settings: {
      includeNeutral: analysis.config.includeNeutral,
      ignoreQuoted: analysis.config.ignoreQuoted,
      ignoreMetadata: analysis.config.ignoreMetadata,
      ignoreTerms: analysis.ignoreTerms,
      thirdReferences: analysis.thirdReferences
    },
    totals: {
      first: analysis.groups.first.total,
      second: analysis.groups.second.total,
      third: analysis.groups.third.total,
      overall: analysis.total
    },
    groups: {
      first: flattenGroupRows('first', analysis.groups.first),
      second: flattenGroupRows('second', analysis.groups.second),
      third: flattenGroupRows('third', analysis.groups.third)
    },
    drift: {
      totalSentences: analysis.drift.totalSentences,
      highlightedSentences: analysis.drift.highlightedSentences,
      switches: analysis.drift.switches,
      dominantCounts: analysis.drift.dominantCounts,
      rows: analysis.drift.rows
    }
  });

  const downloadFile = (filename, content, mimeType) => {
    const blob = new Blob([content], { type: mimeType || 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  };

  const legacyCopyText = (text) => {
    try {
      const area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', 'readonly');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.focus();
      area.select();
      const copied = document.execCommand('copy');
      area.remove();
      return copied;
    } catch {
      return false;
    }
  };

  const writeClipboardText = async (text) => {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(text);
      return true;
    }
    return legacyCopyText(text);
  };

  const copyResults = async () => {
    if (!lastAnalysis) {
      setResultsStatus('Run Check before copying results.', 'error');
      return;
    }

    const text = buildResultsText(lastAnalysis);
    try {
      const copied = await writeClipboardText(text);
      if (!copied) throw new Error('Clipboard unavailable.');
      setResultsStatus('Results copied to clipboard.', 'success');
    } catch {
      setResultsStatus('Unable to copy results in this browser.', 'error');
    }
  };

  const exportResultsCsv = () => {
    if (!lastAnalysis) {
      setResultsStatus('Run Check before exporting CSV.', 'error');
      return;
    }

    const csv = buildResultsCsv(lastAnalysis);
    downloadFile('point-of-view-results.csv', csv, 'text/csv;charset=utf-8');
    setResultsStatus('CSV exported.', 'success');
  };

  const exportResultsJson = () => {
    if (!lastAnalysis) {
      setResultsStatus('Run Check before exporting JSON.', 'error');
      return;
    }

    const json = JSON.stringify(buildResultsJson(lastAnalysis), null, 2);
    downloadFile('point-of-view-results.json', json, 'application/json;charset=utf-8');
    setResultsStatus('JSON exported.', 'success');
  };

  const copyHighlightedHtml = async () => {
    if (!lastAnalysis) {
      setResultsStatus('Run Check before copying highlighted HTML.', 'error');
      return;
    }

    const html = String(outputEl.innerHTML || '').trim();
    if (!html) {
      setResultsStatus('No highlighted output to copy.', 'error');
      return;
    }

    try {
      if (navigator.clipboard && typeof navigator.clipboard.write === 'function' && window.ClipboardItem) {
        const htmlBlob = new Blob([html], { type: 'text/html' });
        const textBlob = new Blob([outputEl.textContent || ''], { type: 'text/plain' });
        await navigator.clipboard.write([
          new ClipboardItem({
            'text/html': htmlBlob,
            'text/plain': textBlob
          })
        ]);
      } else {
        const copied = await writeClipboardText(html);
        if (!copied) throw new Error('Clipboard unavailable.');
      }
      setResultsStatus('Highlighted HTML copied.', 'success');
    } catch {
      setResultsStatus('Unable to copy highlighted HTML in this browser.', 'error');
    }
  };

  const setTokenFilter = (token, pov) => {
    if (activeTokenMatches(token, pov)) {
      activeTokenFilter = null;
      return;
    }
    activeTokenFilter = { key: token, pov };
  };

  const onTokenListClick = (event) => {
    const button = event.target?.closest?.('button[data-token-key][data-pov]');
    if (!button || !lastAnalysis) return;

    const token = String(button.dataset.tokenKey || '').trim();
    const pov = String(button.dataset.pov || '').trim();
    if (!token || !pov) return;

    setTokenFilter(token, pov);
    renderAnalysis(lastAnalysis);
    markSessionDirty();
  };

  const scrollOutputToSentence = (start, end) => {
    const marks = [...outputEl.querySelectorAll('mark[data-start][data-end]')];
    const target = marks.find((mark) => {
      const s = Number.parseInt(mark.getAttribute('data-start') || '-1', 10);
      const e = Number.parseInt(mark.getAttribute('data-end') || '-1', 10);
      return Number.isFinite(s) && Number.isFinite(e) && s >= start && e <= end;
    });

    if (!target) return;

    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.classList.add('povcheck-mark-focus');
    setTimeout(() => {
      target.classList.remove('povcheck-mark-focus');
      if (lastAnalysis) renderHighlightedOutput(lastAnalysis);
    }, 1000);
  };

  const onDriftClick = (event) => {
    const button = event.target?.closest?.('button[data-sentence-start][data-sentence-end]');
    if (!button) return;

    const start = Number.parseInt(button.dataset.sentenceStart || '-1', 10);
    const end = Number.parseInt(button.dataset.sentenceEnd || '-1', 10);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return;

    scrollOutputToSentence(start, end);
  };

  const decodeXmlEntities = (text) => String(text || '')
    .replace(/&#x([0-9a-f]+);/gi, (_match, hex) => {
      const code = Number.parseInt(hex, 16);
      if (Number.isNaN(code)) return '';
      try {
        return String.fromCodePoint(code);
      } catch {
        return '';
      }
    })
    .replace(/&#([0-9]+);/g, (_match, dec) => {
      const code = Number.parseInt(dec, 10);
      if (Number.isNaN(code)) return '';
      try {
        return String.fromCodePoint(code);
      } catch {
        return '';
      }
    })
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');

  const extractHtmlText = (html) => {
    const raw = String(html || '');
    const withBreakHints = raw
      .replace(/<\s*br\s*\/?>/gi, '\n')
      .replace(/<\/(p|div|li|tr|h[1-6])>/gi, '\n')
      .replace(/<li\b[^>]*>/gi, '- ');

    if (typeof DOMParser !== 'function') {
      return withBreakHints.replace(/<[^>]+>/g, ' ');
    }

    const doc = new DOMParser().parseFromString(withBreakHints, 'text/html');
    doc.querySelectorAll('script, style, noscript').forEach((node) => node.remove());
    return doc.body?.textContent || '';
  };

  const extractRtfText = (rtf) => String(rtf || '')
    .replace(/\\par[d]?/gi, '\n')
    .replace(/\\line\b/gi, '\n')
    .replace(/\\tab\b/gi, '\t')
    .replace(/\\'([0-9a-f]{2})/gi, (_match, hex) => String.fromCharCode(Number.parseInt(hex, 16) || 32))
    .replace(/\\u(-?\d+)\??/g, (_match, num) => {
      let code = Number.parseInt(num, 10);
      if (Number.isNaN(code)) return '';
      if (code < 0) code += 65536;
      try {
        return String.fromCharCode(code);
      } catch {
        return '';
      }
    })
    .replace(/\\[a-z]+-?\d* ?/gi, '')
    .replace(/[{}]/g, '');

  const extractDocxXmlText = (xmlText) => {
    const text = String(xmlText || '')
      .replace(/<w:tab[^>]*\/>/gi, '\t')
      .replace(/<w:(br|cr)[^>]*\/>/gi, '\n')
      .replace(/<\/w:tc>/gi, '\t')
      .replace(/<\/w:(p|tr)>/gi, '\n')
      .replace(/<[^>]+>/g, '');
    return decodeXmlEntities(text);
  };

  const extractLegacyDocText = (bytes) => {
    const clean = (text) => String(text || '')
      .replace(/[^ -~\n\r\t]/g, ' ')
      .replace(/[ \t]{2,}/g, ' ')
      .replace(/\n{4,}/g, '\n\n\n');

    const latin = clean(new TextDecoder('latin1').decode(bytes));
    const utf16 = clean(new TextDecoder('utf-16le').decode(bytes));
    return utf16.length > latin.length ? utf16 : latin;
  };

  const parseDocxFile = async (file) => {
    const api = window.fflate;
    if (!api || typeof api.unzipSync !== 'function') {
      throw new Error('DOCX import is unavailable: zip parser failed to load.');
    }

    const bytes = new Uint8Array(await file.arrayBuffer());
    const files = api.unzipSync(bytes);
    const decoder = new TextDecoder('utf-8');
    const xmlPaths = Object.keys(files)
      .filter((name) => /^word\/(document|header\d+|footer\d+|footnotes|endnotes)\.xml$/i.test(name))
      .sort((a, b) => a.localeCompare(b));

    if (!xmlPaths.length) {
      throw new Error('Unable to find readable text in this DOCX file.');
    }

    const chunks = [];
    xmlPaths.forEach((path) => {
      try {
        const xml = decoder.decode(files[path]);
        const text = normalizeImportedText(extractDocxXmlText(xml));
        if (text) chunks.push(text);
      } catch {
        // Skip malformed sections and keep best-effort extraction.
      }
    });

    return chunks.join('\n\n');
  };

  const parsePdfFile = async (file) => {
    const pdfjs = window.pdfjsLib;
    if (!pdfjs || typeof pdfjs.getDocument !== 'function') {
      throw new Error('PDF import is unavailable: parser failed to load.');
    }
    if (pdfjs.GlobalWorkerOptions && !pdfjs.GlobalWorkerOptions.workerSrc) {
      pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER_PATH;
    }

    const loadingTask = pdfjs.getDocument({ data: await file.arrayBuffer() });
    let pdf = null;

    try {
      pdf = await loadingTask.promise;
      const pages = [];

      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const textContent = await page.getTextContent();
        let pageText = '';

        textContent.items.forEach((item) => {
          const value = String(item?.str || '');
          if (!value) return;
          pageText += value;
          pageText += item?.hasEOL ? '\n' : ' ';
        });

        const normalized = normalizeImportedText(pageText);
        if (normalized) pages.push(normalized);
      }

      return pages.join('\n\n');
    } finally {
      try {
        await loadingTask.destroy();
      } catch {}
      try {
        pdf?.cleanup?.();
      } catch {}
    }
  };

  const detectImportType = (file) => {
    const name = String(file?.name || '').toLowerCase();
    const type = String(file?.type || '').toLowerCase();

    if (name.endsWith('.pdf') || type.includes('pdf')) return 'pdf';
    if (name.endsWith('.docx') || type.includes('officedocument.wordprocessingml.document')) return 'docx';
    if (name.endsWith('.doc') || type === 'application/msword') return 'doc';
    if (name.endsWith('.rtf') || type.includes('rtf')) return 'rtf';
    if (name.endsWith('.html') || name.endsWith('.htm') || type.includes('html') || type.includes('xml')) return 'html';
    return 'text';
  };

  const parseImportedFile = async (file) => {
    const importType = detectImportType(file);
    if (importType === 'pdf') {
      return { text: await parsePdfFile(file), warning: '' };
    }
    if (importType === 'docx') {
      return { text: await parseDocxFile(file), warning: '' };
    }
    if (importType === 'doc') {
      const bytes = new Uint8Array(await file.arrayBuffer());
      return {
        text: extractLegacyDocText(bytes),
        warning: 'Legacy .doc extraction is best-effort. Convert to .docx for highest fidelity.'
      };
    }
    if (importType === 'rtf') {
      return { text: extractRtfText(await file.text()), warning: '' };
    }
    if (importType === 'html') {
      return { text: extractHtmlText(await file.text()), warning: '' };
    }
    return { text: await file.text(), warning: '' };
  };

  const pasteIntoInput = async () => {
    setInputBusy(true);
    setInputStatus('Reading clipboard...', 'info');

    try {
      if (!navigator.clipboard || typeof navigator.clipboard.readText !== 'function') {
        throw new Error('Clipboard read is unavailable.');
      }

      const pasted = normalizeImportedText(await navigator.clipboard.readText());
      if (!pasted.trim()) {
        setInputStatus('Clipboard is empty.', 'error');
        return;
      }

      textInput.value = pasted;
      textInput.focus();
      markSessionDirty();
      setInputStatus(`Pasted ${formatNumber(pasted.length)} characters.`, 'success');

      if (hasRun) runAnalysis();
    } catch {
      setInputStatus('Clipboard access blocked. Use Ctrl/Cmd+V in the text box.', 'error');
    } finally {
      setInputBusy(false);
    }
  };

  const importIntoInput = async () => {
    const file = fileInput?.files?.[0];
    if (!file) return;

    setInputBusy(true);
    setInputStatus(`Importing ${file.name}...`, 'info');

    try {
      if (file.size > MAX_IMPORT_BYTES) {
        const maxMb = Math.round(MAX_IMPORT_BYTES / (1024 * 1024));
        throw new Error(`File is too large. Max supported size is ${maxMb} MB.`);
      }

      const parsed = await parseImportedFile(file);
      const importedText = normalizeImportedText(parsed.text);
      if (!importedText.trim()) {
        throw new Error('No readable text was found in this file.');
      }

      textInput.value = importedText;
      textInput.focus();
      markSessionDirty();

      const baseMsg = `${file.name} imported (${formatNumber(importedText.length)} characters).`;
      const statusMsg = parsed.warning ? `${baseMsg} ${parsed.warning}` : baseMsg;
      setInputStatus(statusMsg, parsed.warning ? 'info' : 'success');

      if (hasRun) runAnalysis();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to import this file.';
      setInputStatus(message, 'error');
    } finally {
      if (fileInput) fileInput.value = '';
      setInputBusy(false);
    }
  };

  const serializePresetList = (presets) => presets.map((entry) => ({
    name: entry.name,
    config: sanitizeConfig(entry.config)
  }));

  const loadPresetList = () => {
    try {
      const raw = window.localStorage.getItem(PRESET_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];

      return parsed
        .map((item) => {
          if (!item || typeof item !== 'object') return null;
          const name = collapseWhitespace(item.name).slice(0, 80);
          if (!name) return null;
          return {
            name,
            config: sanitizeConfig(item.config)
          };
        })
        .filter(Boolean)
        .sort((a, b) => a.name.localeCompare(b.name));
    } catch {
      return [];
    }
  };

  const savePresetList = (presets) => {
    try {
      window.localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(serializePresetList(presets)));
    } catch {}
  };

  const renderPresetOptions = (selectedName) => {
    if (!presetSelect) return;

    presetSelect.innerHTML = '';

    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = presets.length ? 'Select a preset' : 'No presets saved';
    presetSelect.appendChild(placeholder);

    presets.forEach((preset) => {
      const option = document.createElement('option');
      option.value = preset.name;
      option.textContent = preset.name;
      if (selectedName && preset.name === selectedName) {
        option.selected = true;
      }
      presetSelect.appendChild(option);
    });
  };

  const findPresetByName = (name) => presets.find((preset) => preset.name === name);

  const saveCurrentPreset = () => {
    const name = collapseWhitespace(presetNameInput?.value || presetSelect?.value || '').slice(0, 80);
    if (!name) {
      setPresetStatus('Enter a preset name before saving.', 'error');
      return;
    }

    const config = sanitizeConfig(getRawConfigFromControls());
    const existing = presets.findIndex((preset) => preset.name.toLowerCase() === name.toLowerCase());

    if (existing >= 0) {
      presets[existing] = { name: presets[existing].name, config };
      setPresetStatus(`Updated preset "${presets[existing].name}".`, 'success');
      renderPresetOptions(presets[existing].name);
      if (presetNameInput) presetNameInput.value = presets[existing].name;
    } else {
      presets.push({ name, config });
      presets.sort((a, b) => a.name.localeCompare(b.name));
      setPresetStatus(`Saved preset "${name}".`, 'success');
      renderPresetOptions(name);
      if (presetNameInput) presetNameInput.value = name;
    }

    savePresetList(presets);
    markSessionDirty();
  };

  const loadSelectedPreset = () => {
    const name = String(presetSelect?.value || '').trim();
    if (!name) {
      setPresetStatus('Choose a preset to load.', 'error');
      return;
    }

    const preset = findPresetByName(name);
    if (!preset) {
      setPresetStatus('Preset not found.', 'error');
      return;
    }

    applyConfigToControls(preset.config, { persistColors: true, markDirty: true });
    if (presetNameInput) presetNameInput.value = preset.name;
    setPresetStatus(`Loaded preset "${preset.name}".`, 'success');

    if (hasRun) runAnalysis();
  };

  const deleteSelectedPreset = () => {
    const name = String(presetSelect?.value || '').trim();
    if (!name) {
      setPresetStatus('Choose a preset to delete.', 'error');
      return;
    }

    const previousLength = presets.length;
    presets = presets.filter((preset) => preset.name !== name);
    if (presets.length === previousLength) {
      setPresetStatus('Preset not found.', 'error');
      return;
    }

    savePresetList(presets);
    renderPresetOptions('');
    setPresetStatus(`Deleted preset "${name}".`, 'success');
    if (presetNameInput && presetNameInput.value.trim() === name) {
      presetNameInput.value = '';
    }
    markSessionDirty();
  };

  const exportPresets = () => {
    const payload = {
      tool: TOOL_ID,
      exportedAt: new Date().toISOString(),
      presets: serializePresetList(presets)
    };
    downloadFile('point-of-view-presets.json', JSON.stringify(payload, null, 2), 'application/json;charset=utf-8');
    setPresetStatus('Presets exported.', 'success');
  };

  const importPresets = async () => {
    const file = presetImportFile?.files?.[0];
    if (!file) return;

    try {
      const raw = await file.text();
      const parsed = JSON.parse(raw);
      const incoming = Array.isArray(parsed)
        ? parsed
        : (Array.isArray(parsed?.presets) ? parsed.presets : []);

      if (!incoming.length) {
        throw new Error('No presets were found in this file.');
      }

      incoming.forEach((entry) => {
        const name = collapseWhitespace(entry?.name).slice(0, 80);
        if (!name) return;
        const config = sanitizeConfig(entry?.config);
        const existing = presets.findIndex((preset) => preset.name.toLowerCase() === name.toLowerCase());
        if (existing >= 0) {
          presets[existing] = { name: presets[existing].name, config };
        } else {
          presets.push({ name, config });
        }
      });

      presets.sort((a, b) => a.name.localeCompare(b.name));
      savePresetList(presets);
      renderPresetOptions('');
      setPresetStatus(`Imported ${formatNumber(incoming.length)} preset entries.`, 'success');
      markSessionDirty();
    } catch (error) {
      setPresetStatus(error instanceof Error ? error.message : 'Unable to import presets.', 'error');
    } finally {
      if (presetImportFile) presetImportFile.value = '';
    }
  };

  const encodeConfigForUrl = (config) => {
    const json = JSON.stringify(sanitizeConfig(config));
    try {
      if (typeof TextEncoder === 'function') {
        const bytes = new TextEncoder().encode(json);
        let binary = '';
        bytes.forEach((byte) => {
          binary += String.fromCharCode(byte);
        });
        return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
      }
      return btoa(unescape(encodeURIComponent(json))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
    } catch {
      return '';
    }
  };

  const decodeConfigFromUrl = (encoded) => {
    const safe = String(encoded || '').trim();
    if (!safe) return null;

    const normalized = safe.replace(/-/g, '+').replace(/_/g, '/');
    const padded = `${normalized}${'='.repeat((4 - (normalized.length % 4)) % 4)}`;

    try {
      const binary = atob(padded);
      let json = '';
      if (typeof TextDecoder === 'function') {
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
          bytes[i] = binary.charCodeAt(i);
        }
        json = new TextDecoder().decode(bytes);
      } else {
        json = decodeURIComponent(escape(binary));
      }
      return sanitizeConfig(JSON.parse(json));
    } catch {
      return null;
    }
  };

  const copyShareLink = async () => {
    const encoded = encodeConfigForUrl(getRawConfigFromControls());
    if (!encoded) {
      setPresetStatus('Unable to build share link for current settings.', 'error');
      return;
    }

    try {
      const shareUrl = new URL(window.location.href);
      shareUrl.searchParams.set('povcfg', encoded);
      const copied = await writeClipboardText(shareUrl.toString());
      if (!copied) throw new Error('Clipboard unavailable.');
      setPresetStatus('Share link copied to clipboard.', 'success');
    } catch {
      setPresetStatus('Unable to copy share link in this browser.', 'error');
    }
  };

  const applyConfigFromUrlIfPresent = () => {
    try {
      const params = new URLSearchParams(window.location.search || '');
      const raw = params.get('povcfg');
      if (!raw) return;
      const config = decodeConfigFromUrl(raw);
      if (!config) {
        setPresetStatus('Share link settings could not be parsed.', 'error');
        return;
      }
      applyConfigToControls(config, { persistColors: true, markDirty: false });
      setPresetStatus('Loaded settings from share link.', 'info');
    } catch {
      // Ignore malformed URLs and continue with defaults.
    }
  };

  let hasRun = false;
  let lastAnalysis = null;
  let activeTokenFilter = null;
  let presets = loadPresetList();

  const captureSummary = () => String(summaryEl?.textContent || '').replace(/\s+/g, ' ').trim();

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
    markSessionDirty();
  });

  form.addEventListener('input', () => {
    markSessionDirty();
  });

  if (modeBasicInput) {
    modeBasicInput.addEventListener('change', () => {
      syncModeUi();
      if (hasRun) runAnalysis();
      markSessionDirty();
    });
  }

  if (modeAdvancedInput) {
    modeAdvancedInput.addEventListener('change', () => {
      syncModeUi();
      if (hasRun) runAnalysis();
      markSessionDirty();
    });
  }

  [includeItToggle, thirdReferencesInput, ignoreTermsInput, ignoreQuotedToggle, ignoreMetadataToggle].forEach((el) => {
    el?.addEventListener('input', () => {
      if (!hasRun) return;
      runAnalysis();
    });
    el?.addEventListener('change', () => {
      if (!hasRun) return;
      runAnalysis();
    });
  });

  clearBtn?.addEventListener('click', () => {
    textInput.value = '';
    setInputStatus('');
    setResultsStatus('');
    resetUI();
    markSessionDirty();
    textInput.focus();
  });

  const handleHighlightColorInput = () => {
    const colors = getHighlightColors();
    applyHighlightColors(colors);
    saveHighlightColors(colors);
    if (hasRun && lastAnalysis) renderHighlightedOutput(lastAnalysis);
    markSessionDirty();
  };

  [firstColorInput, secondColorInput, thirdColorInput].forEach((el) => {
    el?.addEventListener('input', handleHighlightColorInput);
  });

  resetColorsBtn?.addEventListener('click', () => {
    const defaults = getDefaultHighlightColors();
    if (firstColorInput) firstColorInput.value = defaults.first;
    if (secondColorInput) secondColorInput.value = defaults.second;
    if (thirdColorInput) thirdColorInput.value = defaults.third;
    applyHighlightColors(defaults);
    clearStoredHighlightColors();
    if (hasRun && lastAnalysis) renderHighlightedOutput(lastAnalysis);
    markSessionDirty();
  });

  pasteBtn?.addEventListener('click', () => {
    void pasteIntoInput();
  });

  importBtn?.addEventListener('click', () => {
    fileInput?.click();
  });

  fileInput?.addEventListener('change', () => {
    void importIntoInput();
  });

  copyResultsBtn?.addEventListener('click', () => {
    void copyResults();
  });

  exportCsvBtn?.addEventListener('click', exportResultsCsv);
  exportJsonBtn?.addEventListener('click', exportResultsJson);

  copyHtmlBtn?.addEventListener('click', () => {
    void copyHighlightedHtml();
  });

  [firstList, secondList, thirdList].forEach((list) => {
    list.addEventListener('click', onTokenListClick);
  });

  driftListEl.addEventListener('click', onDriftClick);

  presetSaveBtn?.addEventListener('click', saveCurrentPreset);
  presetLoadBtn?.addEventListener('click', loadSelectedPreset);
  presetDeleteBtn?.addEventListener('click', deleteSelectedPreset);
  presetExportBtn?.addEventListener('click', exportPresets);
  presetImportBtn?.addEventListener('click', () => {
    presetImportFile?.click();
  });
  presetImportFile?.addEventListener('change', () => {
    void importPresets();
  });

  presetSelect?.addEventListener('change', () => {
    const selected = String(presetSelect.value || '');
    if (presetNameInput) presetNameInput.value = selected;
  });

  shareConfigBtn?.addEventListener('click', () => {
    void copyShareLink();
  });

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    payload.outputSummary = captureSummary();
    payload.inputs = {
      Text: textInput.value || '',
      Mode: getCurrentMode(),
      IncludeNeutral: includeItToggle?.checked ? 'yes' : 'no',
      IgnoreQuoted: ignoreQuotedToggle?.checked ? 'yes' : 'no',
      IgnoreMetadata: ignoreMetadataToggle?.checked ? 'yes' : 'no',
      IgnoreTerms: ignoreTermsInput?.value || '',
      ThirdReferences: thirdReferencesInput?.value || ''
    };

    const html = String(outputEl?.innerHTML || '').trim();
    if (html && html.length <= MAX_SAVED_OUTPUT_HTML_CHARS) {
      payload.output = { kind: 'html', html, summary: payload.outputSummary };
      return;
    }

    const rawText = String(outputEl?.textContent || '').trim();
    const { text, truncated } = clampText(rawText, 120_000);
    if (text) payload.output = { kind: 'text', text, summary: payload.outputSummary, truncated };
  });

  document.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    requestAnimationFrame(() => {
      try {
        runAnalysis();
      } catch {}
    });
  });

  syncHighlightControls();
  renderPresetOptions('');
  applyConfigFromUrlIfPresent();
  syncModeUi();
  resetUI();
})();
