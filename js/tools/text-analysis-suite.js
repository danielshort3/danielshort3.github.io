(() => {
  'use strict';

  const TOOL_ID = 'text-analysis-suite';
  const $ = (sel) => document.querySelector(sel);

  const root = document.getElementById('main');
  const form = $('#textsuite-form');
  const textInput = $('#textsuite-text');
  const pasteBtn = $('#textsuite-paste');
  const clearBtn = $('#textsuite-clear');
  const topInput = $('#textsuite-top');
  const statusEl = $('#textsuite-status');
  const summaryEl = $('#textsuite-summary');

  const enableFrequencyInput = $('#textsuite-enable-frequency');
  const enablePovInput = $('#textsuite-enable-pov');
  const enableOxfordInput = $('#textsuite-enable-oxford');
  const enableNbspInput = $('#textsuite-enable-nbsp');

  const frequencyPanel = $('#textsuite-frequency-panel');
  const frequencySummaryEl = $('#textsuite-frequency-summary');
  const frequencyListEl = $('#textsuite-frequency-list');

  const povPanel = $('#textsuite-pov-panel');
  const povSummaryEl = $('#textsuite-pov-summary');
  const povListEl = $('#textsuite-pov-list');

  const oxfordPanel = $('#textsuite-oxford-panel');
  const oxfordSummaryEl = $('#textsuite-oxford-summary');
  const oxfordListEl = $('#textsuite-oxford-list');

  const nbspPanel = $('#textsuite-nbsp-panel');
  const nbspSummaryEl = $('#textsuite-nbsp-summary');
  const nbspListEl = $('#textsuite-nbsp-list');

  if (
    !root || !form || !textInput || !topInput || !summaryEl ||
    !enableFrequencyInput || !enablePovInput || !enableOxfordInput || !enableNbspInput ||
    !frequencyPanel || !frequencySummaryEl || !frequencyListEl ||
    !povPanel || !povSummaryEl || !povListEl ||
    !oxfordPanel || !oxfordSummaryEl || !oxfordListEl ||
    !nbspPanel || !nbspSummaryEl || !nbspListEl
  ) {
    return;
  }

  const DEFAULT_SUMMARY = 'Choose at least one tool and click Analyze selected tools.';

  const defaults = {
    top: String(topInput.value || '15'),
    frequency: Boolean(enableFrequencyInput.checked),
    pov: Boolean(enablePovInput.checked),
    oxford: Boolean(enableOxfordInput.checked),
    nbsp: Boolean(enableNbspInput.checked)
  };

  const regexes = (() => {
    try {
      return {
        token: /[\p{L}\p{N}]+(?:['’\-][\p{L}\p{N}]+)*/gu,
        letter: /\p{L}/u
      };
    } catch {
      return {
        token: /[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*/g,
        letter: /[A-Za-z]/
      };
    }
  })();

  const STOPWORDS = new Set([
    'a','an','and','are','as','at','be','been','being','but','by','for','from','had','has','have','he','her','hers','him','his','i',
    'if','in','into','is','it','its','itself','me','my','myself','of','on','or','our','ours','ourselves','she','so','that','the','their',
    'theirs','them','themselves','there','these','they','this','those','to','too','us','was','we','were','what','when','where','which','who',
    'with','you','your','yours','yourself','yourselves'
  ]);

  const FIRST_PERSON = new Set([
    'i','me','my','mine','myself','we','us','our','ours','ourselves',
    "i'm","i'd","i'll","i've","we're","we'd","we'll","we've"
  ]);

  const SECOND_PERSON = new Set([
    'you','your','yours','yourself','yourselves',
    "you're","you'd","you'll","you've","y'all"
  ]);

  const THIRD_PERSON = new Set([
    'he','him','his','himself','she','her','hers','herself','they','them','their','theirs','themself','themselves',
    "he's","he'd","he'll","she's","she'd","she'll","they're","they'd","they'll","they've",
    'it','its','itself',"it's"
  ]);

  const HARD_SPACES = [
    { char: '\u00A0', label: 'Non-breaking space', code: 'U+00A0' },
    { char: '\u202F', label: 'Narrow no-break space', code: 'U+202F' },
    { char: '\u2007', label: 'Figure space', code: 'U+2007' },
    { char: '\u2009', label: 'Thin space', code: 'U+2009' }
  ];

  const OXFORD_CONJUNCTIONS = ['and', 'or', 'nor'];

  const formatNumber = (value) => Number(value || 0).toLocaleString('en-US');

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const truncate = (value, max = 190) => {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (text.length <= max) return text;
    return `${text.slice(0, Math.max(0, max - 3)).trimEnd()}...`;
  };

  const normalizeWhitespace = (text) => String(text || '')
    .replace(/\u0000/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');

  const normalizeToken = (token) => String(token || '')
    .replace(/[’`]/g, "'")
    .toLowerCase()
    .replace(/^['\-]+|['\-]+$/g, '');

  const clampInt = (value, min, max, fallback) => {
    const parsed = Number.parseInt(String(value || '').trim(), 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const escapeRegExp = (value) => String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const setStatus = (message, tone) => {
    if (!statusEl) return;
    statusEl.textContent = String(message || '');
    statusEl.dataset.tone = tone || '';
  };

  const getSelectedModules = () => ({
    frequency: Boolean(enableFrequencyInput.checked),
    pov: Boolean(enablePovInput.checked),
    oxford: Boolean(enableOxfordInput.checked),
    nbsp: Boolean(enableNbspInput.checked)
  });

  const getTopLimit = () => {
    const top = clampInt(topInput.value, 5, 60, clampInt(defaults.top, 5, 60, 15));
    topInput.value = String(top);
    return top;
  };

  const getTextForAnalysis = () => {
    const userText = normalizeWhitespace(textInput.value || '');
    if (userText.trim()) return userText;
    return normalizeWhitespace(textInput.placeholder || '');
  };

  const tokenize = (text) => {
    const source = String(text || '');
    if (!source) return [];

    const regex = new RegExp(regexes.token.source, regexes.token.flags);
    const tokens = [];
    let match;

    while ((match = regex.exec(source)) !== null) {
      const raw = String(match[0] || '');
      const normalized = normalizeToken(raw);
      if (!normalized) continue;

      tokens.push({
        raw,
        normalized,
        start: match.index,
        end: match.index + raw.length
      });
    }

    return tokens;
  };

  const analyzeFrequency = (tokens, topN) => {
    const counts = new Map();
    let analyzedTokenCount = 0;

    tokens.forEach((token) => {
      if (!regexes.letter.test(token.normalized)) return;
      if (token.normalized.length < 2) return;
      if (STOPWORDS.has(token.normalized)) return;

      analyzedTokenCount += 1;
      counts.set(token.normalized, (counts.get(token.normalized) || 0) + 1);
    });

    const rows = Array.from(counts.entries())
      .map(([term, count]) => ({ term, count }))
      .sort((left, right) => {
        if (right.count !== left.count) return right.count - left.count;
        return left.term.localeCompare(right.term);
      });

    return {
      analyzedTokenCount,
      uniqueCount: rows.length,
      rows: rows.slice(0, topN)
    };
  };

  const analyzePov = (tokens) => {
    const counts = {
      first: 0,
      second: 0,
      third: 0
    };

    const examples = {
      first: new Set(),
      second: new Set(),
      third: new Set()
    };

    tokens.forEach((token) => {
      const value = token.normalized;
      if (FIRST_PERSON.has(value)) {
        counts.first += 1;
        if (examples.first.size < 5) examples.first.add(value);
        return;
      }
      if (SECOND_PERSON.has(value)) {
        counts.second += 1;
        if (examples.second.size < 5) examples.second.add(value);
        return;
      }
      if (THIRD_PERSON.has(value)) {
        counts.third += 1;
        if (examples.third.size < 5) examples.third.add(value);
      }
    });

    const activeGroups = ['first', 'second', 'third'].filter((key) => counts[key] > 0);
    const mixed = activeGroups.length > 1;

    let dominant = '';
    let dominantCount = -1;
    activeGroups.forEach((key) => {
      if (counts[key] > dominantCount) {
        dominant = key;
        dominantCount = counts[key];
      }
    });

    return {
      counts,
      examples: {
        first: Array.from(examples.first),
        second: Array.from(examples.second),
        third: Array.from(examples.third)
      },
      activeGroups,
      mixed,
      dominant
    };
  };

  const buildOxfordRegex = () => {
    const conjunctionPattern = OXFORD_CONJUNCTIONS.map((value) => escapeRegExp(value)).join('|');
    return new RegExp(
      `([^.!?\\n;:]+(?:,[^.!?\\n;:]+)+\\s+\\b(${conjunctionPattern})\\b\\s+[^.!?\\n;:]+)(?=[.!?\\n;:]|$)`,
      'gi'
    );
  };

  const findLastConjunctionIndex = (text, conjunction) => {
    const regex = new RegExp(`\\b${escapeRegExp(conjunction)}\\b`, 'gi');
    let index = -1;
    let match;
    while ((match = regex.exec(text)) !== null) {
      index = match.index;
    }
    return index;
  };

  const analyzeOxford = (text) => {
    const regex = buildOxfordRegex();
    let total = 0;
    let present = 0;
    let missing = 0;
    const snippets = [];

    let match;
    while ((match = regex.exec(text)) !== null) {
      const raw = String(match[0] || '');
      const conjunction = String(match[2] || '').toLowerCase();
      if (!conjunction) continue;

      const conjunctionIndex = findLastConjunctionIndex(raw, conjunction);
      if (conjunctionIndex < 0) continue;

      const before = raw.slice(0, conjunctionIndex).trimEnd();
      const hasOxford = /,\s*$/.test(before);

      total += 1;
      if (hasOxford) {
        present += 1;
      } else {
        missing += 1;
        if (snippets.length < 8) snippets.push(raw);
      }
    }

    return {
      total,
      present,
      missing,
      snippets
    };
  };

  const analyzeHardSpaces = (text) => {
    const counts = new Map(HARD_SPACES.map((entry) => [entry.char, 0]));

    for (const char of String(text || '')) {
      if (!counts.has(char)) continue;
      counts.set(char, (counts.get(char) || 0) + 1);
    }

    const byType = HARD_SPACES
      .map((entry) => ({
        ...entry,
        count: counts.get(entry.char) || 0
      }))
      .filter((entry) => entry.count > 0);

    const total = byType.reduce((sum, entry) => sum + entry.count, 0);
    return { total, byType };
  };

  const hidePanels = () => {
    frequencyPanel.hidden = true;
    povPanel.hidden = true;
    oxfordPanel.hidden = true;
    nbspPanel.hidden = true;

    frequencyListEl.innerHTML = '';
    povListEl.innerHTML = '';
    oxfordListEl.innerHTML = '';
    nbspListEl.innerHTML = '';

    frequencySummaryEl.textContent = '';
    povSummaryEl.textContent = '';
    oxfordSummaryEl.textContent = '';
    nbspSummaryEl.textContent = '';
  };

  const renderFrequency = (enabled, frequency) => {
    if (!enabled) {
      frequencyPanel.hidden = true;
      frequencyListEl.innerHTML = '';
      frequencySummaryEl.textContent = '';
      return;
    }

    frequencyPanel.hidden = false;

    if (!frequency || !frequency.rows.length) {
      frequencySummaryEl.textContent = 'No non-stopword frequency terms were found with the current text/settings.';
      frequencyListEl.innerHTML = '';
      return;
    }

    frequencySummaryEl.textContent = `Showing ${frequency.rows.length} of ${frequency.uniqueCount} unique non-stopwords from ${formatNumber(frequency.analyzedTokenCount)} analyzed tokens.`;
    frequencyListEl.innerHTML = frequency.rows.map((row) => (
      `<li><span class="textsuite-term">${escapeHtml(row.term)}</span><span class="textsuite-count">${formatNumber(row.count)}x</span></li>`
    )).join('');
  };

  const renderPov = (enabled, pov) => {
    if (!enabled) {
      povPanel.hidden = true;
      povListEl.innerHTML = '';
      povSummaryEl.textContent = '';
      return;
    }

    povPanel.hidden = false;

    if (!pov || !pov.activeGroups.length) {
      povSummaryEl.textContent = 'No first-, second-, or third-person pronouns were detected.';
      povListEl.innerHTML = '';
      return;
    }

    if (pov.mixed) {
      povSummaryEl.textContent = 'Mixed point of view detected.';
    } else {
      const dominantLabel = pov.dominant === 'first'
        ? 'First person dominates.'
        : pov.dominant === 'second'
          ? 'Second person dominates.'
          : 'Third person dominates.';
      povSummaryEl.textContent = dominantLabel;
    }

    const rows = [
      { key: 'first', label: 'First person', value: pov.counts.first, examples: pov.examples.first },
      { key: 'second', label: 'Second person', value: pov.counts.second, examples: pov.examples.second },
      { key: 'third', label: 'Third person', value: pov.counts.third, examples: pov.examples.third }
    ];

    povListEl.innerHTML = rows
      .filter((row) => row.value > 0)
      .map((row) => {
        const sampleText = row.examples.length ? ` (${row.examples.join(', ')})` : '';
        return `<li><strong>${row.label}:</strong> ${formatNumber(row.value)}${escapeHtml(sampleText)}</li>`;
      })
      .join('');
  };

  const renderOxford = (enabled, oxford) => {
    if (!enabled) {
      oxfordPanel.hidden = true;
      oxfordListEl.innerHTML = '';
      oxfordSummaryEl.textContent = '';
      return;
    }

    oxfordPanel.hidden = false;

    if (!oxford || !oxford.total) {
      oxfordSummaryEl.textContent = 'No serial-list candidates were detected.';
      oxfordListEl.innerHTML = '';
      return;
    }

    oxfordSummaryEl.textContent = `${formatNumber(oxford.total)} list candidates found: ${formatNumber(oxford.present)} with Oxford comma, ${formatNumber(oxford.missing)} without.`;

    if (!oxford.snippets.length) {
      oxfordListEl.innerHTML = '';
      return;
    }

    oxfordListEl.innerHTML = oxford.snippets
      .map((snippet) => `<li>${escapeHtml(truncate(snippet))}</li>`)
      .join('');
  };

  const renderNbsp = (enabled, nbsp) => {
    if (!enabled) {
      nbspPanel.hidden = true;
      nbspListEl.innerHTML = '';
      nbspSummaryEl.textContent = '';
      return;
    }

    nbspPanel.hidden = false;

    if (!nbsp || !nbsp.total) {
      nbspSummaryEl.textContent = 'No hard-space characters were detected.';
      nbspListEl.innerHTML = '';
      return;
    }

    nbspSummaryEl.textContent = `${formatNumber(nbsp.total)} hard-space characters detected.`;
    nbspListEl.innerHTML = nbsp.byType
      .map((entry) => `<li><strong>${escapeHtml(entry.label)} (${entry.code}):</strong> ${formatNumber(entry.count)}</li>`)
      .join('');
  };

  const buildSummary = (enabledCount, charCount, tokenCount) => {
    const label = enabledCount === 1 ? 'tool' : 'tools';
    return `Ran ${enabledCount} selected ${label} on ${formatNumber(charCount)} characters and ${formatNumber(tokenCount)} tokens.`;
  };

  let lastReport = null;

  const runAnalysis = () => {
    setStatus('', '');

    const modules = getSelectedModules();
    const enabledCount = Object.values(modules).filter(Boolean).length;

    if (!enabledCount) {
      lastReport = null;
      summaryEl.textContent = 'Select at least one tool to analyze your text.';
      hidePanels();
      markSessionDirty();
      return;
    }

    const text = getTextForAnalysis();
    if (!text.trim()) {
      lastReport = null;
      summaryEl.textContent = DEFAULT_SUMMARY;
      hidePanels();
      markSessionDirty();
      return;
    }

    const tokens = tokenize(text);
    const topLimit = getTopLimit();

    const report = {
      text,
      modules,
      enabledCount,
      charCount: text.length,
      tokenCount: tokens.length,
      frequency: modules.frequency ? analyzeFrequency(tokens, topLimit) : null,
      pov: modules.pov ? analyzePov(tokens) : null,
      oxford: modules.oxford ? analyzeOxford(text) : null,
      nbsp: modules.nbsp ? analyzeHardSpaces(text) : null
    };

    summaryEl.textContent = buildSummary(report.enabledCount, report.charCount, report.tokenCount);

    renderFrequency(modules.frequency, report.frequency);
    renderPov(modules.pov, report.pov);
    renderOxford(modules.oxford, report.oxford);
    renderNbsp(modules.nbsp, report.nbsp);

    lastReport = report;
    markSessionDirty();
  };

  const clearToDefaults = () => {
    textInput.value = '';
    topInput.value = defaults.top;
    enableFrequencyInput.checked = defaults.frequency;
    enablePovInput.checked = defaults.pov;
    enableOxfordInput.checked = defaults.oxford;
    enableNbspInput.checked = defaults.nbsp;

    setStatus('', '');
    summaryEl.textContent = DEFAULT_SUMMARY;
    hidePanels();
    lastReport = null;
    markSessionDirty();
    textInput.focus();
  };

  const handlePaste = async () => {
    setStatus('Reading clipboard...', 'info');
    if (pasteBtn) pasteBtn.disabled = true;

    try {
      if (!navigator.clipboard || typeof navigator.clipboard.readText !== 'function') {
        throw new Error('Clipboard read is unavailable in this browser.');
      }

      const clipboardText = normalizeWhitespace(await navigator.clipboard.readText());
      if (!clipboardText.trim()) {
        setStatus('Clipboard is empty.', 'error');
        return;
      }

      textInput.value = clipboardText;
      textInput.dispatchEvent(new Event('input', { bubbles: true }));
      setStatus(`Pasted ${formatNumber(clipboardText.length)} characters.`, 'success');
      runAnalysis();
      textInput.focus();
    } catch {
      setStatus('Clipboard access blocked. Use Ctrl/Cmd+V in the text field.', 'error');
    } finally {
      if (pasteBtn) pasteBtn.disabled = false;
    }
  };

  const buildSnapshotText = () => {
    if (!lastReport) return '';

    const lines = [summaryEl.textContent.trim()];

    if (lastReport.modules.frequency && lastReport.frequency) {
      lines.push('', '[Word frequency]');
      if (!lastReport.frequency.rows.length) {
        lines.push('No non-stopword terms found.');
      } else {
        lastReport.frequency.rows.forEach((row, index) => {
          lines.push(`${index + 1}. ${row.term} (${row.count})`);
        });
      }
    }

    if (lastReport.modules.pov && lastReport.pov) {
      lines.push('', '[Point of view]');
      lines.push(`First person: ${lastReport.pov.counts.first}`);
      lines.push(`Second person: ${lastReport.pov.counts.second}`);
      lines.push(`Third person: ${lastReport.pov.counts.third}`);
    }

    if (lastReport.modules.oxford && lastReport.oxford) {
      lines.push('', '[Oxford comma]');
      lines.push(`List candidates: ${lastReport.oxford.total}`);
      lines.push(`Present: ${lastReport.oxford.present}`);
      lines.push(`Missing: ${lastReport.oxford.missing}`);
    }

    if (lastReport.modules.nbsp && lastReport.nbsp) {
      lines.push('', '[Hard-space scan]');
      lines.push(`Total hard spaces: ${lastReport.nbsp.total}`);
      lastReport.nbsp.byType.forEach((entry) => {
        lines.push(`${entry.code} ${entry.label}: ${entry.count}`);
      });
    }

    return lines.join('\n').trim();
  };

  const getToolInputs = () => {
    const modules = getSelectedModules();
    return {
      Text: textInput.value || '',
      'Word frequency': modules.frequency ? 'On' : 'Off',
      'Point of view': modules.pov ? 'On' : 'Off',
      'Oxford comma': modules.oxford ? 'On' : 'Off',
      'Hard-space scan': modules.nbsp ? 'On' : 'Off',
      'Top frequency terms': String(getTopLimit())
    };
  };

  const getToolSnapshotOutput = () => ({
    kind: 'text',
    text: buildSnapshotText(),
    summary: String(summaryEl.textContent || '').trim() || DEFAULT_SUMMARY
  });

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  clearBtn?.addEventListener('click', clearToDefaults);

  pasteBtn?.addEventListener('click', () => {
    void handlePaste();
  });

  [textInput, topInput].forEach((input) => {
    input?.addEventListener('input', markSessionDirty);
  });

  [enableFrequencyInput, enablePovInput, enableOxfordInput, enableNbspInput].forEach((input) => {
    input?.addEventListener('change', markSessionDirty);
  });

  root.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail || {};
    if (detail.toolId !== TOOL_ID) return;

    const payload = detail.payload || {};
    const output = getToolSnapshotOutput();
    const inputs = getToolInputs();

    payload.outputSummary = String(output.summary || payload.outputSummary || '').trim();
    payload.inputs = inputs;

    if (output.text) {
      payload.output = {
        kind: 'text',
        text: output.text,
        summary: output.summary
      };
    }

    if (detail.snapshot && typeof detail.snapshot === 'object') {
      detail.snapshot.output = payload.output || output;
      detail.snapshot.inputs = inputs;
    }
  });

  root.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail || {};
    if (detail.toolId !== TOOL_ID) return;
    requestAnimationFrame(() => {
      try {
        runAnalysis();
      } catch {}
    });
  });

  summaryEl.textContent = DEFAULT_SUMMARY;
  hidePanels();
})();
