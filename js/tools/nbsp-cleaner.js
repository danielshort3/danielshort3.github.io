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

  if (!form || !input || !output || !summary || !countsList) return;

  const HARD_SPACES = [
    { char: '\u00A0', label: 'Non-breaking space', code: 'U+00A0' },
    { char: '\u202F', label: 'Narrow no-break space', code: 'U+202F' },
    { char: '\u2007', label: 'Figure space', code: 'U+2007' },
    { char: '\u2009', label: 'Thin space', code: 'U+2009' },
  ];

  const escapeForSet = (str) => str.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const hardChars = HARD_SPACES.map((h) => h.char).join('');
  const hardRegex = new RegExp(`[${escapeForSet(hardChars)}]`, 'g');

  const formatNumber = (n) => n.toLocaleString('en-US');

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
    const cleaned = text.replace(hardRegex, ' ');
    return { perType, total, cleaned };
  };

  const renderCounts = (perType) => {
    countsList.innerHTML = '';
    const active = perType.filter((p) => p.count > 0);
    if (!active.length) return;
    active.forEach((entry) => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${formatNumber(entry.count)}×</strong>
        <div>
          <div>${entry.label}</div>
          <small>${entry.code}</small>
        </div>`;
      countsList.appendChild(li);
    });
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
      return;
    }
    const { perType, total, cleaned } = analyze(text);
    renderCounts(perType);
    output.value = cleaned;
    const totalTypes = perType.filter((p) => p.count > 0).length;
    if (total > 0) {
      summary.innerHTML = `Found <strong>${formatNumber(total)}</strong> hard spaces across <strong>${totalTypes}</strong> type(s). Replaced them with regular spaces below.`;
    } else {
      summary.textContent = 'No hard spaces detected. Output matches your input.';
    }
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
})();
