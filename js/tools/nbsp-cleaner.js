(() => {
  'use strict';
  const $ = (sel) => document.querySelector(sel);
  const form = $('#nbsp-form');
  const input = $('#nbsp-input');
  const output = $('#nbsp-output');
  const summary = $('#nbsp-summary');
  const countsList = $('#nbsp-counts');
  const copyBtn = $('#nbsp-copy');
  const copyStatus = $('#nbsp-copy-status');
  const clearBtn = $('#nbsp-clear');
  const preview = $('#nbsp-preview');
  const fixHardToggle = $('#nbsp-fix-hard');
  const stripNonAsciiToggle = $('#nbsp-strip-nonascii');

  if (!form || !input || !output || !summary || !countsList || !preview) return;

  const TOOL_ID = 'nbsp-cleaner';

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const HARD_SPACES = [
    { char: '\u00A0', label: 'Non-breaking space', code: 'U+00A0' },
    { char: '\u202F', label: 'Narrow no-break space', code: 'U+202F' },
    { char: '\u2007', label: 'Figure space', code: 'U+2007' },
    { char: '\u2009', label: 'Thin space', code: 'U+2009' },
  ];
  const HARD_SET = new Set(HARD_SPACES.map((h) => h.char));

  const escapeForSet = (str) => str.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const hardChars = HARD_SPACES.map((h) => h.char).join('');
  const hardRegex = new RegExp(`[${escapeForSet(hardChars)}]`, 'g');
  const NON_ASCII_REGEX = /[^\x00-\x7F]/g;

  const formatNumber = (n) => n.toLocaleString('en-US');
  const escapeHtml = (s) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const codePointLabel = (ch) => {
    const cp = ch.codePointAt(0).toString(16).toUpperCase().padStart(4, '0');
    return `U+${cp}`;
  };

  const analyze = (text) => {
    const perType = HARD_SPACES.map((h) => ({ ...h, count: 0 }));
    let total = 0;
    for (const ch of text) {
      const idx = HARD_SPACES.findIndex((h) => h.char === ch);
      if (idx !== -1) {
        perType[idx].count += 1;
        total += 1;
      }
    }
    return { perType, total };
  };

  const countNonAscii = (text) => {
    const counts = new Map();
    for (const ch of text) {
      if (HARD_SET.has(ch)) continue;
      if (ch.charCodeAt(0) > 127) {
        counts.set(ch, (counts.get(ch) || 0) + 1);
      }
    }
    return counts;
  };

  const renderCounts = (perType, nonAsciiCounts) => {
    countsList.innerHTML = '';
    const active = perType.filter((p) => p.count > 0);
    active.forEach((entry) => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${formatNumber(entry.count)}×</strong>
        <div>
          <div>${entry.label}</div>
          <small>${entry.code}</small>
        </div>`;
      countsList.appendChild(li);
    });
    const nonAsciiTotal = [...nonAsciiCounts.values()].reduce((sum, v) => sum + v, 0);
    if (nonAsciiTotal > 0) {
      const li = document.createElement('li');
      const samples = [...nonAsciiCounts.entries()].slice(0, 3)
        .map(([ch, c]) => `${codePointLabel(ch)} (${formatNumber(c)}×)`).join(', ');
      li.innerHTML = `<strong>${formatNumber(nonAsciiTotal)}×</strong>
        <div>
          <div>Other non-ASCII characters</div>
          <small>${samples}</small>
        </div>`;
      countsList.appendChild(li);
    }
  };

  const renderPreview = (text) => {
    const parts = [];
    for (const ch of text) {
      if (HARD_SET.has(ch)) {
        parts.push(`<span class="nbsp-chip nbsp-hard" title="${codePointLabel(ch)} hard space">␠</span>`);
      } else if (ch.charCodeAt(0) > 127) {
        parts.push(`<span class="nbsp-chip nbsp-nonascii" title="Non-ASCII ${codePointLabel(ch)}">${escapeHtml(ch)}</span>`);
      } else {
        parts.push(escapeHtml(ch));
      }
    }
    preview.innerHTML = parts.join('') || '<span class="nbsp-status">Preview will appear after you paste text.</span>';
  };

  const buildCleaned = (text) => {
    let cleaned = text;
    const replaceHard = !fixHardToggle || fixHardToggle.checked;
    const stripNonAscii = Boolean(stripNonAsciiToggle?.checked);
    if (replaceHard) {
      cleaned = cleaned.replace(hardRegex, ' ');
    }
    if (stripNonAscii) {
      cleaned = cleaned.replace(NON_ASCII_REGEX, '');
    }
    return { cleaned, replacedHard: replaceHard, strippedNonAscii: stripNonAscii };
  };

  const setCopyStatus = (msg, tone) => {
    if (!copyStatus) return;
    copyStatus.textContent = msg;
    copyStatus.dataset.tone = tone || '';
  };

  const runCleaner = () => {
    const text = input.value || '';
    if (!text.trim()) {
      summary.textContent = 'Paste text above, then run the cleaner.';
      countsList.innerHTML = '';
      output.value = '';
      setCopyStatus('');
      preview.innerHTML = '<span class="nbsp-status">Preview will appear after you paste text.</span>';
      return;
    }
    const { perType, total } = analyze(text);
    const nonAsciiCounts = countNonAscii(text);
    renderCounts(perType, nonAsciiCounts);
    const { cleaned, replacedHard, strippedNonAscii } = buildCleaned(text);
    output.value = cleaned;
    const nonAsciiTotal = [...nonAsciiCounts.values()].reduce((sum, v) => sum + v, 0);
    const findings = [];
    if (total > 0) findings.push(`${formatNumber(total)} hard spaces${replacedHard ? ' replaced' : ' detected (kept)'}`);
    if (nonAsciiTotal > 0) findings.push(`${formatNumber(nonAsciiTotal)} other non-ASCII ${strippedNonAscii ? 'removed' : 'detected (kept)'}`);
    if (findings.length) {
      summary.innerHTML = `Found ${findings.join(' and ')}.${strippedNonAscii ? ' Output is ASCII-only.' : ''}`;
    } else {
      summary.textContent = 'No hard spaces or non-ASCII characters detected. Output matches your input.';
    }
    renderPreview(text);
    setCopyStatus('');
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runCleaner();
  });

  clearBtn?.addEventListener('click', () => {
    input.value = '';
    output.value = '';
    countsList.innerHTML = '';
    summary.textContent = 'Paste text and run the cleaner to see findings.';
    setCopyStatus('');
    preview.innerHTML = '<span class="nbsp-status">Preview will appear after you paste text.</span>';
    markSessionDirty();
    input.focus();
  });

  copyBtn?.addEventListener('click', async () => {
    if (!output.value) {
      setCopyStatus('Nothing to copy yet.', 'error');
      return;
    }
    try {
      await navigator.clipboard.writeText(output.value);
      setCopyStatus('Cleaned text copied.', 'success');
    } catch {
      setCopyStatus('Copy failed. Please copy manually.', 'error');
    }
  });

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    payload.outputSummary = String(summary?.textContent || '').replace(/\s+/g, ' ').trim();
    payload.inputs = { Input: input.value || '' };
  });
})();
