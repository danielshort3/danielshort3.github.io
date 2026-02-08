(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  const form = $('#wordfreq-form');
  const textInput = $('#wordfreq-text');
  const topInput = $('#wordfreq-top');
  const clearBtn = $('#wordfreq-clear');
  const resultsList = $('#wordfreq-results');
  const summaryEl = $('#wordfreq-summary');
  const emptyEl = $('#wordfreq-empty');
  const copyBtn = $('#wordfreq-copy');
  const exportCsvBtn = $('#wordfreq-export-csv');
  const exportJsonBtn = $('#wordfreq-export-json');
  const copyStatusEl = $('#wordfreq-copy-status');
  const occurrencePanel = $('#wordfreq-occurrence-panel');
  const occurrenceSummaryEl = $('#wordfreq-occurrence-summary');
  const occurrenceListEl = $('#wordfreq-occurrence-list');
  const fullTextPanel = $('#wordfreq-fulltext-panel');
  const fullTextSummaryEl = $('#wordfreq-fulltext-summary');
  const fullTextEl = $('#wordfreq-fulltext');

  const pasteBtn = $('#wordfreq-paste');
  const importBtn = $('#wordfreq-import');
  const fileInput = $('#wordfreq-file');
  const inputStatusEl = $('#wordfreq-input-status');

  const stopwordsSelect = $('#wordfreq-stopwords');
  const ngramSelect = $('#wordfreq-ngram');
  const scoreSelect = $('#wordfreq-score');
  const sortSelect = $('#wordfreq-sort');
  const minLengthInput = $('#wordfreq-minlen');
  const stemInput = $('#wordfreq-stem');
  const foldInput = $('#wordfreq-fold');
  const numbersInput = $('#wordfreq-numbers');
  const includeInput = $('#wordfreq-include');
  const excludeInput = $('#wordfreq-exclude');

  if (
    !form || !textInput || !topInput || !resultsList || !summaryEl || !emptyEl ||
    !stopwordsSelect || !ngramSelect || !scoreSelect || !sortSelect || !minLengthInput ||
    !stemInput || !foldInput || !numbersInput || !includeInput || !excludeInput
  ) {
    return;
  }

  const TOOL_ID = 'word-frequency';
  const MAX_IMPORT_BYTES = 24 * 1024 * 1024;
  const MAX_OCCURRENCE_SNIPPETS = 12;
  const MAX_STORED_HITS_PER_TERM = 600;
  const MAX_FULLTEXT_PREVIEW_CHARS = 180_000;
  const PDF_WORKER_PATH = '/js/vendor/pdfjs/pdf.worker.min.js';
  const PDFJS_SRC = '/js/vendor/pdfjs/pdf.min.js';
  const FFLATE_SRC = '/js/vendor/fflate/fflate.min.js';
  const vendorScriptPromises = new Map();

  const STOPWORDS_BASIC = new Set([
    'a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by',
    'can','cannot','can\'t','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t',
    'have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into',
    'is','isn\'t','it','it\'s','its','itself','just','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours',
    'ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves',
    'then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll',
    'we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you',
    'you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves'
  ]);

  const STOPWORDS_AGGRESSIVE = new Set([
    ...STOPWORDS_BASIC,
    'also','among','amongst','another','anyone','anything','anyway','anywhere','become','becomes','becoming','beside','besides','beyond','cause','causes','couldve','d','e','eg',
    'else','elsewhere','etc','ever','every','everybody','everyone','everything','everywhere','f','g','get','gets','getting','go','goes','going','gone','got','gotten','h','however',
    'ie','im','ive','k','l','least','less','m','mainly','many','may','maybe','might','mightn\'t','much','n','namely','nearly','need','needed','needing','needs','next','o','often',
    'ok','okay','onto','p','per','perhaps','q','quite','r','really','s','say','says','seem','seemed','seeming','seems','seen','several','since','still','t','take','takes','taken',
    'taking','thing','things','think','thinks','thus','u','v','via','w','want','wants','way','ways','x','y','yes','yet','z'
  ]);

  const EMPTY_STOPWORDS = new Set();

  const regexes = (() => {
    try {
      return {
        token: /[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*/gu,
        letter: /\p{L}/u,
        digit: /\p{N}/u,
        marks: /\p{M}+/gu
      };
    } catch {
      return {
        token: /[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*/g,
        letter: /[A-Za-z]/,
        digit: /[0-9]/,
        marks: /[\u0300-\u036f]+/g
      };
    }
  })();

  const defaults = {
    top: String(topInput.value || '15'),
    stopwords: String(stopwordsSelect.value || 'english-basic'),
    ngram: String(ngramSelect.value || '1'),
    score: String(scoreSelect.value || 'count'),
    sort: String(sortSelect.value || 'score-desc'),
    minLength: String(minLengthInput.value || '2'),
    stem: Boolean(stemInput.checked),
    fold: Boolean(foldInput.checked),
    includeNumbers: Boolean(numbersInput.checked),
    includeTerms: String(includeInput.value || ''),
    excludeTerms: String(excludeInput.value || '')
  };

  const DEFAULT_SUMMARY = 'Click Analyze to run the built-in example, or paste your own text.';
  const DEFAULT_EMPTY = 'Click Analyze to run the built-in example.';

  const getAnalysisSourceText = () => {
    const userText = normalizeWhitespace(textInput.value || '');
    if (userText.trim()) return userText;
    return normalizeWhitespace(textInput.placeholder || '');
  };

  const formatNumber = (n) => Number(n || 0).toLocaleString('en-US');

  const clampInt = (value, min, max, fallback) => {
    const parsed = Number.parseInt(String(value || '').trim(), 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const escapeAttr = (value) => escapeHtml(value).replace(/'/g, '&#39;');

  const normalizeWhitespace = (text) => String(text || '')
    .replace(/\u0000/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');

  const setCopyStatus = (message, tone) => {
    if (!copyStatusEl) return;
    copyStatusEl.textContent = String(message || '');
    copyStatusEl.dataset.tone = tone || '';
  };

  const setInputStatus = (message, tone) => {
    if (!inputStatusEl) return;
    inputStatusEl.textContent = String(message || '');
    inputStatusEl.dataset.tone = tone || '';
  };

  const setInputBusy = (busy) => {
    const state = Boolean(busy);
    if (pasteBtn) pasteBtn.disabled = state;
    if (importBtn) importBtn.disabled = state;
    if (fileInput) fileInput.disabled = state;
  };

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const splitCustomList = (raw) => String(raw || '')
    .split(/[\n,;]+/)
    .map((entry) => entry.trim())
    .filter(Boolean);

  const stripDiacritics = (text) => {
    const value = String(text || '');
    if (!value || typeof value.normalize !== 'function') return value;
    return value.normalize('NFD').replace(regexes.marks, '');
  };

  const normalizeApostrophes = (text) => String(text || '')
    .replace(/[’`]/g, "'")
    .replace(/[‐‑‒–—]/g, '-')
    .replace(/-{2,}/g, '-');

  const simpleStem = (word) => {
    let value = String(word || '');
    if (value.length <= 3) return value;

    if (value.endsWith('ies') && value.length > 4) {
      value = `${value.slice(0, -3)}y`;
    } else if (/(sses|ches|shes|xes|zes)$/.test(value) && value.length > 5) {
      value = value.slice(0, -2);
    } else if (value.endsWith('s') && !/(ss|us|is)$/.test(value) && value.length > 3) {
      value = value.slice(0, -1);
    }

    if (value.endsWith('ing') && value.length > 5) {
      value = value.slice(0, -3);
      if (/([b-df-hj-np-tv-z])\1$/.test(value)) value = value.slice(0, -1);
    } else if (value.endsWith('ed') && value.length > 4) {
      value = value.slice(0, -2);
      if (/([b-df-hj-np-tv-z])\1$/.test(value)) value = value.slice(0, -1);
    }

    if (value.endsWith('ly') && value.length > 5) value = value.slice(0, -2);
    if (value.endsWith('ment') && value.length > 6) value = value.slice(0, -4);
    if (value.endsWith('ness') && value.length > 6) value = value.slice(0, -4);

    return value;
  };

  const createTokenRegex = () => new RegExp(regexes.token.source, regexes.token.flags);

  const tokenizeText = (text) => {
    const source = String(text || '');
    if (!source) return [];
    const regex = createTokenRegex();
    const tokens = [];
    let match;
    while ((match = regex.exec(source)) !== null) {
      const raw = String(match[0] || '');
      if (!raw) continue;
      tokens.push({
        raw,
        start: match.index,
        end: match.index + raw.length,
        tokenIndex: tokens.length
      });
    }
    return tokens;
  };

  const normalizeToken = (raw, settings, options) => {
    let value = normalizeApostrophes(raw);
    if (!value) return '';

    if (settings.foldDiacritics) value = stripDiacritics(value);
    value = value.toLowerCase().replace(/^['-]+|['-]+$/g, '');
    if (!value) return '';

    const hasLetter = regexes.letter.test(value);
    const hasDigit = regexes.digit.test(value);

    if (!settings.includeNumbers && !hasLetter) return '';

    const minLength = options?.ignoreMinLength ? 1 : settings.minLength;
    if (value.length < minLength && !hasDigit) return '';

    if (settings.useStemming && !options?.disableStemming && hasLetter) {
      value = simpleStem(value);
    }

    if (!value) return '';
    if (value.length < minLength && !hasDigit) return '';
    return value;
  };

  const normalizeCustomEntry = (entry, settings, ngramSize) => {
    const tokens = tokenizeText(entry);
    if (!tokens.length) return '';
    const parts = tokens
      .map((token) => normalizeToken(token.raw, settings, { ignoreMinLength: true }))
      .filter(Boolean);
    if (!parts.length) return '';
    if (ngramSize <= 1) return parts[0];
    if (parts.length < ngramSize) return '';
    return parts.slice(0, ngramSize).join(' ');
  };

  const bestDisplayForm = (forms, fallback) => {
    if (!(forms instanceof Map) || !forms.size) return fallback;
    let winner = fallback;
    let winnerCount = -1;
    forms.forEach((count, form) => {
      if (count > winnerCount) {
        winner = form;
        winnerCount = count;
        return;
      }
      if (count === winnerCount && form.length < winner.length) {
        winner = form;
      }
    });
    return winner;
  };

  const getStopwordSet = (mode) => {
    if (mode === 'none') return EMPTY_STOPWORDS;
    if (mode === 'english-aggressive') return STOPWORDS_AGGRESSIVE;
    return STOPWORDS_BASIC;
  };

  const getSettings = () => ({
    top: clampInt(topInput.value, 1, 200, clampInt(defaults.top, 1, 200, 15)),
    minLength: clampInt(minLengthInput.value, 1, 12, clampInt(defaults.minLength, 1, 12, 2)),
    stopwordMode: String(stopwordsSelect.value || defaults.stopwords),
    ngramSize: clampInt(ngramSelect.value, 1, 3, clampInt(defaults.ngram, 1, 3, 1)),
    scoreMode: String(scoreSelect.value || defaults.score),
    sortMode: String(sortSelect.value || defaults.sort),
    useStemming: Boolean(stemInput.checked),
    foldDiacritics: Boolean(foldInput.checked),
    includeNumbers: Boolean(numbersInput.checked),
    includeTermsRaw: String(includeInput.value || ''),
    excludeTermsRaw: String(excludeInput.value || '')
  });

  const buildCustomSets = (settings) => {
    const includeEntries = splitCustomList(settings.includeTermsRaw);
    const excludeEntries = splitCustomList(settings.excludeTermsRaw);

    const tokenInclude = new Set(includeEntries
      .map((entry) => normalizeCustomEntry(entry, settings, 1))
      .filter(Boolean));

    const tokenExclude = new Set(excludeEntries
      .map((entry) => normalizeCustomEntry(entry, settings, 1))
      .filter(Boolean));

    const phraseExclude = new Set(excludeEntries
      .map((entry) => normalizeCustomEntry(entry, settings, settings.ngramSize))
      .filter(Boolean));

    return { tokenInclude, tokenExclude, phraseExclude };
  };

  const initEntryRecord = (key, parts) => ({
    key,
    parts,
    count: 0,
    forms: new Map(),
    occurrences: []
  });

  const registerEntryHit = (entryMap, key, display, start, end, parts) => {
    let record = entryMap.get(key);
    if (!record) {
      record = initEntryRecord(key, parts);
      entryMap.set(key, record);
    }
    record.count += 1;
    const normalizedDisplay = String(display || '').replace(/\s+/g, ' ').trim();
    if (normalizedDisplay) {
      record.forms.set(normalizedDisplay, (record.forms.get(normalizedDisplay) || 0) + 1);
    }
    if (record.occurrences.length < MAX_STORED_HITS_PER_TERM) {
      record.occurrences.push({ start, end });
    }
  };

  const analyzeInput = (rawText, settings) => {
    const sourceText = normalizeWhitespace(rawText);
    const sourceTokens = tokenizeText(sourceText);
    const stopwords = getStopwordSet(settings.stopwordMode);
    const custom = buildCustomSets(settings);

    const normalizedTokens = [];
    const sourceTokenViews = [];
    const fullTokenRecords = new Map();
    let removedStopwords = 0;
    let removedExcluded = 0;

    sourceTokens.forEach((token) => {
      const normalized = normalizeToken(token.raw, settings);
      sourceTokenViews.push({
        ...token,
        normalized
      });

      if (!normalized) return;

      let fullRecord = fullTokenRecords.get(normalized);
      if (!fullRecord) {
        fullRecord = {
          count: 0,
          forms: new Map(),
          hits: []
        };
        fullTokenRecords.set(normalized, fullRecord);
      }
      fullRecord.count += 1;
      fullRecord.forms.set(token.raw, (fullRecord.forms.get(token.raw) || 0) + 1);
      fullRecord.hits.push({ start: token.start, end: token.end });

      const excluded = custom.tokenExclude.has(normalized);
      if (excluded) {
        removedExcluded += 1;
        return;
      }

      const forced = custom.tokenInclude.has(normalized);
      const isStopword = stopwords.has(normalized);
      if (!forced && isStopword) {
        removedStopwords += 1;
        return;
      }

      normalizedTokens.push({
        ...token,
        normalized
      });
    });

    const fullTokenMap = new Map();
    fullTokenRecords.forEach((record, key) => {
      fullTokenMap.set(key, {
        key,
        display: bestDisplayForm(record.forms, key),
        count: record.count,
        hits: record.hits
      });
    });

    const unigramCounts = new Map();
    normalizedTokens.forEach((token) => {
      unigramCounts.set(token.normalized, (unigramCounts.get(token.normalized) || 0) + 1);
    });

    const entryMap = new Map();
    let totalUnits = 0;

    if (settings.ngramSize === 1) {
      normalizedTokens.forEach((token) => {
        totalUnits += 1;
        registerEntryHit(entryMap, token.normalized, token.raw, token.start, token.end, [token.normalized]);
      });
    } else {
      const n = settings.ngramSize;
      for (let i = 0; i <= normalizedTokens.length - n; i += 1) {
        const window = normalizedTokens.slice(i, i + n);
        let contiguous = true;
        for (let j = 1; j < window.length; j += 1) {
          if (window[j].tokenIndex !== window[0].tokenIndex + j) {
            contiguous = false;
            break;
          }
        }
        if (!contiguous) continue;

        const parts = window.map((token) => token.normalized);
        const key = parts.join(' ');
        if (custom.phraseExclude.has(key)) {
          removedExcluded += 1;
          continue;
        }

        const start = window[0].start;
        const end = window[window.length - 1].end;
        const display = sourceText.slice(start, end).replace(/\s+/g, ' ').trim();
        totalUnits += 1;
        registerEntryHit(entryMap, key, display || key, start, end, parts);
      }
    }

    const entries = Array.from(entryMap.values()).map((record) => {
      const display = bestDisplayForm(record.forms, record.key);
      const count = record.count;
      const per1000 = totalUnits ? (count / totalUnits) * 1000 : 0;
      const sharePct = totalUnits ? (count / totalUnits) * 100 : 0;

      let pmi = 0;
      let association = 0;
      if (settings.ngramSize > 1 && totalUnits > 0 && normalizedTokens.length > 0) {
        const pxy = count / totalUnits;
        let product = 1;
        record.parts.forEach((part) => {
          const partCount = unigramCounts.get(part) || 1;
          product *= (partCount / normalizedTokens.length);
        });
        if (product > 0 && pxy > 0) {
          pmi = Math.log2(pxy / product);
          association = pmi * Math.log2(count + 1);
        }
      }

      let score = count;
      let scoreLabel = `${formatNumber(count)}×`;
      let scoreValueText = `${count}`;

      if (settings.scoreMode === 'per1000') {
        score = per1000;
        scoreLabel = `${per1000.toFixed(2)} /1k`;
        scoreValueText = per1000.toFixed(6);
      } else if (settings.scoreMode === 'share') {
        score = sharePct;
        scoreLabel = `${sharePct.toFixed(2)}%`;
        scoreValueText = sharePct.toFixed(6);
      } else if (settings.scoreMode === 'pmi' && settings.ngramSize > 1) {
        score = association;
        scoreLabel = `${association.toFixed(3)} assoc`;
        scoreValueText = association.toFixed(6);
      }

      return {
        key: record.key,
        display,
        count,
        per1000,
        sharePct,
        pmi,
        association,
        score,
        scoreLabel,
        scoreValueText,
        occurrences: record.occurrences,
        parts: record.parts
      };
    });

    const sortMode = settings.sortMode;
    entries.sort((left, right) => {
      if (sortMode === 'alpha-asc') {
        return left.display.localeCompare(right.display, undefined, { sensitivity: 'base' });
      }
      if (right.score !== left.score) return right.score - left.score;
      if (right.count !== left.count) return right.count - left.count;
      return left.display.localeCompare(right.display, undefined, { sensitivity: 'base' });
    });

    const topEntries = entries.slice(0, settings.top);

    const usesPmiFallback = settings.scoreMode === 'pmi' && settings.ngramSize === 1;

    return {
      text: sourceText,
      textPreview: sourceText.length > MAX_FULLTEXT_PREVIEW_CHARS
        ? `${sourceText.slice(0, MAX_FULLTEXT_PREVIEW_CHARS)}\n\n[Preview truncated for performance.]`
        : sourceText,
      settings,
      sourceTokenCount: sourceTokens.length,
      analyzedTokenCount: normalizedTokens.length,
      uniqueCount: entries.length,
      totalUnits,
      removedStopwords,
      removedExcluded,
      usesPmiFallback,
      entries,
      topEntries,
      sourceTokenViews,
      fullTokenMap,
      entryMap: new Map(entries.map((entry) => [entry.key, entry])),
      occurrenceMap: new Map(entries.map((entry) => [entry.key, entry.occurrences]))
    };
  };

  const scoreToWidth = (score, minScore, maxScore) => {
    if (!Number.isFinite(score)) return 8;
    if (!Number.isFinite(minScore) || !Number.isFinite(maxScore) || minScore === maxScore) return 100;
    const normalized = (score - minScore) / (maxScore - minScore);
    return 8 + (Math.max(0, Math.min(1, normalized)) * 92);
  };

  const renderEmpty = (summary, reason) => {
    summaryEl.textContent = summary;
    resultsList.innerHTML = '';
    emptyEl.hidden = false;
    emptyEl.textContent = reason;
    hideOccurrences();
    hideFullText();
  };

  const buildSummaryText = (analysis) => {
    const label = analysis.settings.ngramSize === 1
      ? 'words'
      : `${analysis.settings.ngramSize}-word phrases`;

    const pieces = [
      `Showing ${analysis.topEntries.length} of ${analysis.uniqueCount} unique ${label}`,
      `${formatNumber(analysis.sourceTokenCount)} source tokens`,
      `${formatNumber(analysis.removedStopwords)} stopwords removed`
    ];

    if (analysis.removedExcluded) {
      pieces.push(`${formatNumber(analysis.removedExcluded)} manually excluded`);
    }

    if (analysis.usesPmiFallback) {
      pieces.push('PMI requires phrase mode (using raw frequency)');
    }

    return pieces.join(' · ');
  };

  const createResultRow = (entry, index, scoreRange, selectedKey) => {
    const item = document.createElement('li');
    item.className = 'wordfreq-row';
    if (selectedKey && selectedKey === entry.key) item.classList.add('is-selected');

    const meta = document.createElement('div');
    meta.className = 'wordfreq-meta';

    const rank = document.createElement('span');
    rank.className = 'wordfreq-rank';
    rank.textContent = String(index + 1);

    const wrap = document.createElement('div');
    wrap.className = 'wordfreq-wordwrap';

    const termBtn = document.createElement('button');
    termBtn.type = 'button';
    termBtn.className = 'wordfreq-term-btn';
    termBtn.dataset.termKey = entry.key;
    termBtn.dataset.termLabel = entry.display;
    termBtn.textContent = entry.display;

    const count = document.createElement('span');
    count.className = 'wordfreq-count';
    count.textContent = `${formatNumber(entry.count)}×`;

    const score = document.createElement('span');
    score.className = 'wordfreq-score';
    score.textContent = entry.scoreLabel;

    wrap.append(termBtn, count, score);
    meta.append(rank, wrap);

    const bar = document.createElement('div');
    bar.className = 'wordfreq-bar';
    bar.setAttribute('role', 'presentation');
    bar.setAttribute('aria-hidden', 'true');

    const fill = document.createElement('span');
    fill.style.width = `${scoreToWidth(entry.score, scoreRange.min, scoreRange.max).toFixed(1)}%`;
    bar.appendChild(fill);

    item.append(meta, bar);
    return item;
  };

  const getScoreRange = (entries) => {
    if (!entries.length) return { min: 0, max: 0 };
    let min = entries[0].score;
    let max = entries[0].score;
    entries.forEach((entry) => {
      if (entry.score < min) min = entry.score;
      if (entry.score > max) max = entry.score;
    });
    return { min, max };
  };

  const normalizeSnippetSegment = (segment) => escapeHtml(String(segment || '').replace(/\s+/g, ' '));

  const buildOccurrenceSnippet = (text, start, end, radius) => {
    const left = Math.max(0, start - radius);
    const right = Math.min(text.length, end + radius);
    const prefix = left > 0 ? '…' : '';
    const suffix = right < text.length ? '…' : '';

    const before = normalizeSnippetSegment(text.slice(left, start));
    const hit = normalizeSnippetSegment(text.slice(start, end));
    const after = normalizeSnippetSegment(text.slice(end, right));

    return `${prefix}${before}<mark>${hit}</mark>${after}${suffix}`;
  };

  const hideOccurrences = () => {
    if (occurrencePanel) occurrencePanel.hidden = true;
    if (occurrenceListEl) occurrenceListEl.innerHTML = '';
    if (occurrenceSummaryEl) {
      occurrenceSummaryEl.textContent = 'Select a term to inspect where it appears in your source text.';
    }
  };

  const hideFullText = () => {
    if (fullTextPanel) fullTextPanel.hidden = true;
    if (fullTextEl) fullTextEl.innerHTML = '';
    if (fullTextSummaryEl) {
      fullTextSummaryEl.textContent = 'Select a term from results (or click a word below) to highlight it throughout the source text.';
    }
  };

  const resolveSelectedTerm = (analysis, termKey, preferredLabel) => {
    if (!analysis) return null;
    const key = String(termKey || '').trim();
    if (!key) return null;

    const entry = analysis.entryMap?.get(key) || null;
    const fullToken = analysis.fullTokenMap?.get(key) || null;

    if (!entry && !fullToken) return null;

    const label = preferredLabel || entry?.display || fullToken?.display || key;
    const useFullTokenHits = !key.includes(' ') && fullToken;
    const hits = useFullTokenHits ? fullToken.hits : (entry?.occurrences || fullToken?.hits || []);
    const total = useFullTokenHits ? fullToken.count : (entry?.count || fullToken?.count || hits.length);

    return {
      key,
      label,
      hits,
      total
    };
  };

  const renderOccurrences = (analysis, selectedTerm) => {
    if (!occurrencePanel || !occurrenceListEl || !occurrenceSummaryEl || !analysis) return;

    if (!selectedTerm) {
      hideOccurrences();
      return;
    }

    const hits = selectedTerm.hits || [];
    const total = Number(selectedTerm.total || hits.length);
    if (!hits.length) {
      occurrencePanel.hidden = false;
      occurrenceSummaryEl.textContent = `No occurrences found for “${selectedTerm.label}”.`;
      occurrenceListEl.innerHTML = '';
      return;
    }

    occurrencePanel.hidden = false;
    const visibleHits = hits.slice(0, MAX_OCCURRENCE_SNIPPETS);
    occurrenceSummaryEl.textContent = `Showing ${visibleHits.length} of ${formatNumber(total)} matches for “${selectedTerm.label}”.`;

    occurrenceListEl.innerHTML = '';
    visibleHits.forEach((hit) => {
      const item = document.createElement('li');
      item.innerHTML = buildOccurrenceSnippet(analysis.text, hit.start, hit.end, 72);
      occurrenceListEl.appendChild(item);
    });
  };

  const buildFullTextHtml = (analysis, selectedTerm) => {
    const sourceText = String(analysis?.textPreview || '');
    const sourceTokens = Array.isArray(analysis?.sourceTokenViews) ? analysis.sourceTokenViews : [];
    if (!sourceText) return '';
    if (!sourceTokens.length) return escapeHtml(sourceText);

    const hasSelection = Boolean(selectedTerm?.key);
    const selectedKey = String(selectedTerm?.key || '');
    const ranges = (selectedTerm?.hits || []).slice().sort((a, b) => a.start - b.start || a.end - b.end);

    let html = '';
    let cursor = 0;
    let rangeIndex = 0;

    for (let i = 0; i < sourceTokens.length; i += 1) {
      const token = sourceTokens[i];
      if (token.start >= sourceText.length) break;
      const safeStart = Math.max(0, Math.min(token.start, sourceText.length));
      const safeEnd = Math.max(safeStart, Math.min(token.end, sourceText.length));

      html += escapeHtml(sourceText.slice(cursor, safeStart));

      while (rangeIndex < ranges.length && ranges[rangeIndex].end <= safeStart) {
        rangeIndex += 1;
      }
      const inSelectedRange = hasSelection &&
        rangeIndex < ranges.length &&
        ranges[rangeIndex].start < safeEnd &&
        ranges[rangeIndex].end > safeStart;

      const tokenText = sourceText.slice(safeStart, safeEnd);
      if (!token.normalized) {
        html += escapeHtml(tokenText);
      } else {
        const classes = ['wordfreq-fulltext-token'];
        if (inSelectedRange) classes.push('is-hit');
        if (selectedKey && token.normalized === selectedKey) classes.push('is-selected');

        html += `<button type="button" class="${classes.join(' ')}" data-term-key="${escapeAttr(token.normalized)}" data-term-label="${escapeAttr(tokenText)}">${escapeHtml(tokenText)}</button>`;
      }

      cursor = safeEnd;
    }

    html += escapeHtml(sourceText.slice(cursor));
    return html;
  };

  const renderFullText = (analysis, selectedTerm) => {
    if (!fullTextPanel || !fullTextEl || !fullTextSummaryEl || !analysis) return;

    fullTextPanel.hidden = false;
    fullTextEl.innerHTML = buildFullTextHtml(analysis, selectedTerm);

    if (analysis.textPreview !== analysis.text) {
      fullTextSummaryEl.textContent = 'Showing an interactive preview of the source text. Click any word below to highlight it throughout the preview.';
      return;
    }

    if (!selectedTerm) {
      fullTextSummaryEl.textContent = 'Click a term in results or click any word below to highlight it throughout the source text.';
      return;
    }

    fullTextSummaryEl.textContent = `Highlighted ${formatNumber(selectedTerm.total)} matches for “${selectedTerm.label}”. Click any other word below to switch focus.`;
  };

  const renderResults = (analysis, selectedKey) => {
    resultsList.innerHTML = '';
    if (!analysis.topEntries.length) return;

    const scoreRange = getScoreRange(analysis.topEntries);
    analysis.topEntries.forEach((entry, index) => {
      resultsList.appendChild(createResultRow(entry, index, scoreRange, selectedKey));
    });
  };

  const csvEscape = (value) => {
    const raw = String(value ?? '');
    if (!/[",\n]/.test(raw)) return raw;
    return `"${raw.replace(/"/g, '""')}"`;
  };

  const downloadTextFile = (text, filename, type) => {
    const blob = new Blob([text], { type: type || 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const buildExportRows = (analysis) => analysis.topEntries.map((entry, index) => ({
    rank: index + 1,
    term: entry.display,
    key: entry.key,
    count: entry.count,
    per1000: Number(entry.per1000.toFixed(6)),
    sharePct: Number(entry.sharePct.toFixed(6)),
    score: Number(entry.scoreValueText),
    scoreLabel: entry.scoreLabel
  }));

  const buildResultsText = (analysis) => {
    const rows = buildExportRows(analysis);
    if (!rows.length) return '';
    const lines = [`# ${captureSummary()}`];
    rows.forEach((row) => {
      lines.push(`${row.rank}. ${row.term} | count=${row.count} | score=${row.scoreLabel}`);
    });
    return lines.join('\n');
  };

  const copyText = async (text) => {
    const value = String(text || '');
    if (!value) return false;

    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        await navigator.clipboard.writeText(value);
        return true;
      }
    } catch {}

    try {
      const el = document.createElement('textarea');
      el.value = value;
      el.setAttribute('readonly', '');
      el.style.position = 'fixed';
      el.style.top = '-9999px';
      el.style.left = '-9999px';
      document.body.appendChild(el);
      el.select();
      const ok = document.execCommand('copy');
      el.remove();
      return ok;
    } catch {
      return false;
    }
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

  const loadVendorScript = (src) => {
    const safeSrc = String(src || '').trim();
    if (!safeSrc) return Promise.reject(new Error('Vendor script source is missing.'));
    if (vendorScriptPromises.has(safeSrc)) return vendorScriptPromises.get(safeSrc);

    const existing = document.querySelector(`script[src="${safeSrc}"]`);
    if (existing && existing.dataset.vendorLoaded === 'true') {
      return Promise.resolve();
    }

    const pending = new Promise((resolve, reject) => {
      const script = existing || document.createElement('script');

      const finish = (error) => {
        script.removeEventListener('load', onLoad);
        script.removeEventListener('error', onError);
        if (error) {
          vendorScriptPromises.delete(safeSrc);
          reject(error);
          return;
        }
        script.dataset.vendorLoaded = 'true';
        resolve();
      };
      const onLoad = () => finish();
      const onError = () => finish(new Error(`Unable to load ${safeSrc}.`));

      script.addEventListener('load', onLoad);
      script.addEventListener('error', onError);

      if (!existing) {
        script.src = safeSrc;
        script.async = true;
        document.head.appendChild(script);
      }
    });

    vendorScriptPromises.set(safeSrc, pending);
    return pending;
  };

  const ensureFflate = async () => {
    const api = window.fflate;
    if (api && typeof api.unzipSync === 'function') return api;
    await loadVendorScript(FFLATE_SRC);
    const loaded = window.fflate;
    if (!loaded || typeof loaded.unzipSync !== 'function') {
      throw new Error('DOCX import is unavailable: zip parser failed to load.');
    }
    return loaded;
  };

  const ensurePdfjs = async () => {
    const api = window.pdfjsLib;
    if (api && typeof api.getDocument === 'function') return api;
    await loadVendorScript(PDFJS_SRC);
    const loaded = window.pdfjsLib;
    if (!loaded || typeof loaded.getDocument !== 'function') {
      throw new Error('PDF import is unavailable: parser failed to load.');
    }
    return loaded;
  };

  const parseDocxFile = async (file) => {
    const api = await ensureFflate();

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
        const text = normalizeWhitespace(extractDocxXmlText(xml)).trim();
        if (text) chunks.push(text);
      } catch {}
    });

    return chunks.join('\n\n');
  };

  const parsePdfFile = async (file) => {
    const pdfjs = await ensurePdfjs();

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
        const normalized = normalizeWhitespace(pageText).trim();
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
    if (importType === 'pdf') return { text: await parsePdfFile(file), warning: '' };
    if (importType === 'docx') return { text: await parseDocxFile(file), warning: '' };
    if (importType === 'doc') {
      const bytes = new Uint8Array(await file.arrayBuffer());
      return {
        text: extractLegacyDocText(bytes),
        warning: 'Legacy .doc extraction is best-effort. Convert to .docx for highest fidelity.'
      };
    }
    if (importType === 'rtf') return { text: extractRtfText(await file.text()), warning: '' };
    if (importType === 'html') return { text: extractHtmlText(await file.text()), warning: '' };
    return { text: await file.text(), warning: '' };
  };

  const clearFormToDefaults = () => {
    textInput.value = '';
    topInput.value = defaults.top;
    stopwordsSelect.value = defaults.stopwords;
    ngramSelect.value = defaults.ngram;
    scoreSelect.value = defaults.score;
    sortSelect.value = defaults.sort;
    minLengthInput.value = defaults.minLength;
    stemInput.checked = defaults.stem;
    foldInput.checked = defaults.fold;
    numbersInput.checked = defaults.includeNumbers;
    includeInput.value = defaults.includeTerms;
    excludeInput.value = defaults.excludeTerms;
  };

  const captureSummary = () => String(summaryEl?.textContent || '').replace(/\s+/g, ' ').trim();

  let lastAnalysis = null;
  let selectedTermKey = '';
  let selectedTermLabel = '';

  const runAnalysis = () => {
    setCopyStatus('', '');

    const settings = getSettings();
    topInput.value = String(settings.top);
    minLengthInput.value = String(settings.minLength);

    const analysisSource = getAnalysisSourceText();
    const analysis = analyzeInput(analysisSource, settings);

    if (!analysis.sourceTokenCount) {
      lastAnalysis = null;
      selectedTermKey = '';
      selectedTermLabel = '';
      renderEmpty(DEFAULT_SUMMARY, DEFAULT_EMPTY);
      markSessionDirty();
      return;
    }

    if (!analysis.topEntries.length) {
      lastAnalysis = null;
      selectedTermKey = '';
      selectedTermLabel = '';
      const reason = analysis.analyzedTokenCount
        ? 'No terms available after filtering and settings.'
        : 'No analyzable words remained after filtering.';
      renderEmpty('No results found with current settings.', reason);
      markSessionDirty();
      return;
    }

    lastAnalysis = analysis;

    const summaryText = buildSummaryText(analysis);
    summaryEl.innerHTML = escapeHtml(summaryText)
      .replace(/(Showing \d+ of \d+ unique [^.·]+)/, '<strong>$1</strong>')
      .replace(/(\d[\d,]* source tokens)/, '<strong>$1</strong>');

    emptyEl.hidden = true;
    const selectedTerm = resolveSelectedTerm(analysis, selectedTermKey, selectedTermLabel);
    if (!selectedTerm) {
      selectedTermKey = '';
      selectedTermLabel = '';
    } else {
      selectedTermKey = selectedTerm.key;
      selectedTermLabel = selectedTerm.label;
    }

    renderResults(analysis, selectedTermKey);
    renderOccurrences(analysis, selectedTerm);
    renderFullText(analysis, selectedTerm);

    markSessionDirty();
  };

  const handleCopyResults = async () => {
    if (!lastAnalysis || !lastAnalysis.topEntries.length) {
      setCopyStatus('Nothing to copy yet.', 'error');
      return;
    }

    setCopyStatus('Copying…', 'info');
    const ok = await copyText(buildResultsText(lastAnalysis));
    setCopyStatus(ok ? 'Copied results.' : 'Copy failed.', ok ? 'success' : 'error');
  };

  const handleExportCsv = () => {
    if (!lastAnalysis || !lastAnalysis.topEntries.length) {
      setCopyStatus('No results to export.', 'error');
      return;
    }

    const headers = ['rank', 'term', 'key', 'count', 'per_1000', 'share_pct', 'score', 'score_label'];
    const rows = buildExportRows(lastAnalysis);
    const csv = [
      headers.join(','),
      ...rows.map((row) => [
        row.rank,
        row.term,
        row.key,
        row.count,
        row.per1000,
        row.sharePct,
        row.score,
        row.scoreLabel
      ].map(csvEscape).join(','))
    ].join('\n');

    downloadTextFile(csv, 'word-frequency-results.csv', 'text/csv;charset=utf-8');
    setCopyStatus('CSV exported.', 'success');
  };

  const handleExportJson = () => {
    if (!lastAnalysis || !lastAnalysis.topEntries.length) {
      setCopyStatus('No results to export.', 'error');
      return;
    }

    const payload = {
      summary: captureSummary(),
      settings: {
        top: lastAnalysis.settings.top,
        minLength: lastAnalysis.settings.minLength,
        stopwords: lastAnalysis.settings.stopwordMode,
        ngramSize: lastAnalysis.settings.ngramSize,
        scoreMode: lastAnalysis.settings.scoreMode,
        sortMode: lastAnalysis.settings.sortMode,
        useStemming: lastAnalysis.settings.useStemming,
        foldDiacritics: lastAnalysis.settings.foldDiacritics,
        includeNumbers: lastAnalysis.settings.includeNumbers,
        includeTerms: splitCustomList(lastAnalysis.settings.includeTermsRaw),
        excludeTerms: splitCustomList(lastAnalysis.settings.excludeTermsRaw)
      },
      metrics: {
        sourceTokenCount: lastAnalysis.sourceTokenCount,
        analyzedTokenCount: lastAnalysis.analyzedTokenCount,
        uniqueCount: lastAnalysis.uniqueCount,
        totalUnits: lastAnalysis.totalUnits,
        removedStopwords: lastAnalysis.removedStopwords,
        removedExcluded: lastAnalysis.removedExcluded
      },
      results: buildExportRows(lastAnalysis)
    };

    downloadTextFile(JSON.stringify(payload, null, 2), 'word-frequency-results.json', 'application/json;charset=utf-8');
    setCopyStatus('JSON exported.', 'success');
  };

  const handlePasteInput = async () => {
    setInputBusy(true);
    setInputStatus('Reading clipboard…', 'info');

    try {
      if (!navigator.clipboard || typeof navigator.clipboard.readText !== 'function') {
        throw new Error('Clipboard read is unavailable.');
      }

      const clipboardText = normalizeWhitespace(await navigator.clipboard.readText());
      if (!clipboardText.trim()) {
        setInputStatus('Clipboard is empty.', 'error');
        return;
      }

      textInput.value = clipboardText;
      textInput.dispatchEvent(new Event('input', { bubbles: true }));
      setInputStatus(`Pasted ${clipboardText.length.toLocaleString('en-US')} characters.`, 'success');
      runAnalysis();
      textInput.focus();
    } catch {
      setInputStatus('Clipboard access blocked. Use Ctrl/Cmd+V in the text box.', 'error');
    } finally {
      setInputBusy(false);
    }
  };

  const handleImportInput = async () => {
    const file = fileInput?.files?.[0];
    if (!file) return;

    setInputBusy(true);
    setInputStatus(`Importing ${file.name}…`, 'info');

    try {
      if (file.size > MAX_IMPORT_BYTES) {
        const maxMb = Math.round(MAX_IMPORT_BYTES / (1024 * 1024));
        throw new Error(`File is too large. Max supported size is ${maxMb} MB.`);
      }

      const parsed = await parseImportedFile(file);
      const text = normalizeWhitespace(parsed.text).trim();
      if (!text) {
        throw new Error('No readable text was found in this file.');
      }

      textInput.value = text;
      textInput.dispatchEvent(new Event('input', { bubbles: true }));
      const status = parsed.warning
        ? `${file.name} imported. ${parsed.warning}`
        : `${file.name} imported (${text.length.toLocaleString('en-US')} characters).`;
      setInputStatus(status, parsed.warning ? 'info' : 'success');
      runAnalysis();
      textInput.focus();
    } catch (error) {
      setInputStatus(error instanceof Error ? error.message : 'Unable to import this file.', 'error');
    } finally {
      if (fileInput) fileInput.value = '';
      setInputBusy(false);
    }
  };

  const captureOutputText = () => {
    if (!lastAnalysis || !lastAnalysis.topEntries.length) return '';
    return buildResultsText(lastAnalysis);
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  clearBtn?.addEventListener('click', () => {
    clearFormToDefaults();
    setCopyStatus('', '');
    setInputStatus('', '');
    selectedTermKey = '';
    selectedTermLabel = '';
    lastAnalysis = null;
    hideOccurrences();
    hideFullText();
    renderEmpty(DEFAULT_SUMMARY, DEFAULT_EMPTY);
    markSessionDirty();
    textInput.focus();
  });

  copyBtn?.addEventListener('click', () => {
    void handleCopyResults();
  });

  exportCsvBtn?.addEventListener('click', handleExportCsv);
  exportJsonBtn?.addEventListener('click', handleExportJson);

  pasteBtn?.addEventListener('click', () => {
    void handlePasteInput();
  });

  importBtn?.addEventListener('click', () => {
    fileInput?.click();
  });

  fileInput?.addEventListener('change', () => {
    void handleImportInput();
  });

  resultsList.addEventListener('click', (event) => {
    const btn = event.target instanceof Element ? event.target.closest('.wordfreq-term-btn') : null;
    if (!btn || !lastAnalysis) return;

    const termKey = String(btn.getAttribute('data-term-key') || '').trim();
    const termLabel = String(btn.getAttribute('data-term-label') || termKey).trim();
    if (!termKey) return;

    selectedTermKey = termKey;
    selectedTermLabel = termLabel || termKey;
    const selectedTerm = resolveSelectedTerm(lastAnalysis, selectedTermKey, selectedTermLabel);
    renderResults(lastAnalysis, selectedTermKey);
    renderOccurrences(lastAnalysis, selectedTerm);
    renderFullText(lastAnalysis, selectedTerm);
  });

  fullTextEl?.addEventListener('click', (event) => {
    const btn = event.target instanceof Element ? event.target.closest('.wordfreq-fulltext-token[data-term-key]') : null;
    if (!btn || !lastAnalysis) return;

    const termKey = String(btn.getAttribute('data-term-key') || '').trim();
    if (!termKey) return;

    const termLabel = String(btn.getAttribute('data-term-label') || termKey).trim();
    selectedTermKey = termKey;
    selectedTermLabel = termLabel || termKey;

    const selectedTerm = resolveSelectedTerm(lastAnalysis, selectedTermKey, selectedTermLabel);
    if (!selectedTerm) return;

    renderResults(lastAnalysis, selectedTerm.key);
    renderOccurrences(lastAnalysis, selectedTerm);
    renderFullText(lastAnalysis, selectedTerm);
  });

  const markDirtyFromControl = () => {
    markSessionDirty();
  };

  [textInput, topInput, minLengthInput, includeInput, excludeInput].forEach((el) => {
    el?.addEventListener('input', markDirtyFromControl);
  });

  [stopwordsSelect, ngramSelect, scoreSelect, sortSelect, stemInput, foldInput, numbersInput].forEach((el) => {
    el?.addEventListener('change', () => {
      markSessionDirty();
      if (!getAnalysisSourceText().trim()) return;
      runAnalysis();
    });
  });

  topInput.addEventListener('change', () => {
    if (!getAnalysisSourceText().trim()) return;
    runAnalysis();
  });

  minLengthInput.addEventListener('change', () => {
    if (!getAnalysisSourceText().trim()) return;
    runAnalysis();
  });

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const settings = getSettings();
    const summary = captureSummary();

    payload.outputSummary = summary;
    payload.inputs = {
      Text: textInput.value || '',
      'Top results': String(settings.top),
      'Stopwords preset': settings.stopwordMode,
      'N-gram size': String(settings.ngramSize),
      'Score mode': settings.scoreMode,
      'Sort mode': settings.sortMode,
      'Min token length': String(settings.minLength),
      'Use stemming': settings.useStemming ? 'Yes' : 'No',
      'Fold diacritics': settings.foldDiacritics ? 'Yes' : 'No',
      'Include numbers': settings.includeNumbers ? 'Yes' : 'No',
      'Always keep terms': settings.includeTermsRaw,
      'Always ignore terms': settings.excludeTermsRaw
    };

    const outputText = captureOutputText();
    if (outputText) {
      payload.output = {
        kind: 'text',
        text: outputText,
        summary
      };
    }
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

  renderEmpty(DEFAULT_SUMMARY, DEFAULT_EMPTY);
})();
