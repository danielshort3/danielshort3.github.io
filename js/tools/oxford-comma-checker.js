(() => {
  'use strict';
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const form = $('[data-oxford-form]');
  const input = $('[data-oxford-input]');
  const summary = $('[data-oxford-summary]');
  const counts = $('[data-oxford-counts]');
  const resultsList = $('[data-oxford-results]');
  const empty = $('[data-oxford-empty]');
  const output = $('[data-oxford-output]');
  const clearBtn = $('[data-oxford-clear]');
  const conjInputs = $$('[data-oxford-conjunction]');
  const presentColorInput = $('[data-oxford-color="present"]');
  const absentColorInput = $('[data-oxford-color="absent"]');
  const resetColorsBtn = $('[data-oxford-reset-colors]');
  let hasRun = false;

  if (!form || !input || !summary || !counts || !output) return;

  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const escapeHtml = (value) => value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const formatNumber = (value) => value.toLocaleString('en-US');
  const truncate = (value, max = 180) => (value.length <= max ? value : `${value.slice(0, max - 3)}...`);

  const defaultColors = {
    present: presentColorInput?.value || '',
    absent: absentColorInput?.value || '',
  };

  const applyColorVars = () => {
    if (presentColorInput?.value) {
      document.body.style.setProperty('--oxford-present-color', presentColorInput.value);
    }
    if (absentColorInput?.value) {
      document.body.style.setProperty('--oxford-absent-color', absentColorInput.value);
    }
  };

  applyColorVars();

  const getTextForAnalysis = () => {
    const raw = input.value || '';
    if (raw.trim()) {
      return { text: raw, isPlaceholder: false };
    }
    const placeholder = input.placeholder || '';
    if (placeholder.trim()) {
      return { text: placeholder, isPlaceholder: true };
    }
    return { text: '', isPlaceholder: false };
  };

  const getConjunctions = () => conjInputs
    .filter((item) => item.checked)
    .map((item) => (item.dataset.oxfordConjunction || '').toLowerCase())
    .filter(Boolean);

  const buildRegex = (conjunctions) => {
    if (!conjunctions.length) return null;
    const conjPattern = conjunctions.map((c) => escapeRegExp(c)).join('|');
    return new RegExp(
      `([^.!?\\n;:]+(?:,[^.!?\\n;:]+)+\\s+\\b(${conjPattern})\\b\\s+[^.!?\\n;:]+)(?=[.!?\\n;:]|$)`,
      'gi'
    );
  };

  const findLastConjunctionIndex = (text, conj) => {
    const regex = new RegExp(`\\b${escapeRegExp(conj)}\\b`, 'gi');
    let idx = -1;
    let match;
    while ((match = regex.exec(text)) !== null) {
      idx = match.index;
    }
    return idx;
  };

  const findMatches = (text, conjunctions) => {
    const regex = buildRegex(conjunctions);
    if (!regex) return [];
    const matches = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
      const raw = match[0];
      const conj = (match[2] || '').toLowerCase();
      if (!conj) continue;
      const conjIndex = findLastConjunctionIndex(raw, conj);
      if (conjIndex < 0) continue;
      const before = raw.slice(0, conjIndex).trimEnd();
      const hasOxford = /,\s*$/.test(before);
      matches.push({
        start: match.index,
        end: match.index + raw.length,
        text: raw,
        conj,
        hasOxford,
      });
    }
    return matches;
  };

  const renderCounts = (total, present, missing) => {
    counts.innerHTML = `
      <li>
        <strong>${formatNumber(total)}</strong>
        <span>List candidates</span>
      </li>
      <li>
        <strong>${formatNumber(present)}</strong>
        <span>Oxford comma present</span>
      </li>
      <li>
        <strong>${formatNumber(missing)}</strong>
        <span>Oxford comma absent</span>
      </li>
    `;
  };

  const renderResults = (matches) => {
    if (!resultsList) return;
    resultsList.innerHTML = '';
    if (!matches.length) {
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'No list candidates detected yet.';
      }
      return;
    }
    if (empty) empty.hidden = true;
    matches.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'oxford-result';
      li.dataset.status = item.hasOxford ? 'present' : 'missing';
      const snippet = truncate(item.text.trim());
      li.innerHTML = `
        <div class="oxford-result-header">
          <span class="oxford-result-status">${item.hasOxford ? 'Oxford comma present' : 'Oxford comma absent'}</span>
          <span class="oxford-result-tag">${item.conj}</span>
        </div>
        <p class="oxford-result-text">${escapeHtml(snippet)}</p>
      `;
      resultsList.appendChild(li);
    });
  };

  const renderOutput = (text, highlights) => {
    if (!text.trim()) {
      output.innerHTML = '<p class="oxford-empty">Paste text and click Check.</p>';
      return;
    }
    if (!highlights.length) {
      output.textContent = text;
      return;
    }
    const sorted = highlights.slice().sort((a, b) => a.start - b.start);
    let cursor = 0;
    let html = '';
    sorted.forEach((range) => {
      if (range.start < cursor) return;
      const tone = range.status === 'present' ? 'present' : 'missing';
      html += escapeHtml(text.slice(cursor, range.start));
      html += `<mark class="oxford-mark oxford-mark-${tone}">${escapeHtml(text.slice(range.start, range.end))}</mark>`;
      cursor = range.end;
    });
    html += escapeHtml(text.slice(cursor));
    output.innerHTML = html;
  };

  const runAnalysis = () => {
    const { text } = getTextForAnalysis();
    if (!text.trim()) {
      hasRun = false;
      summary.textContent = 'Paste text and click Check.';
      counts.innerHTML = '';
      if (resultsList) resultsList.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'Waiting for input.';
      }
      renderOutput('', []);
      return;
    }
    const conjunctions = getConjunctions();
    if (!conjunctions.length) {
      hasRun = true;
      summary.textContent = 'Select at least one conjunction to scan.';
      counts.innerHTML = '';
      if (resultsList) resultsList.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'Choose "and", "or", or "nor" to scan lists.';
      }
      renderOutput(text, []);
      return;
    }
    const matches = findMatches(text, conjunctions);
    const missing = matches.filter((item) => !item.hasOxford);
    const missingCount = missing.length;
    const present = matches.length - missingCount;
    renderCounts(matches.length, present, missingCount);
    hasRun = true;
    if (matches.length) {
      const majorityLabel = present === missingCount
        ? 'Tie'
        : present > missingCount
          ? 'Oxford comma present'
          : 'Oxford comma absent';
      summary.textContent = `${formatNumber(matches.length)} list candidate${matches.length === 1 ? '' : 's'} found. Oxford comma present in ${formatNumber(present)}, absent in ${formatNumber(missingCount)}. Majority: ${majorityLabel}.`;
    } else {
      summary.textContent = 'No list candidates detected. Try a longer sample.';
    }
    const highlights = matches.map((item) => ({
      start: item.start,
      end: item.end,
      status: item.hasOxford ? 'present' : 'missing',
    }));
    renderResults(matches);
    renderOutput(text, highlights);
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  const maybeRerun = () => {
    if (!hasRun) return;
    runAnalysis();
  };

  conjInputs.forEach((item) => {
    item.addEventListener('change', maybeRerun);
  });
  presentColorInput?.addEventListener('input', applyColorVars);
  absentColorInput?.addEventListener('input', applyColorVars);
  resetColorsBtn?.addEventListener('click', () => {
    if (presentColorInput && defaultColors.present) {
      presentColorInput.value = defaultColors.present;
    }
    if (absentColorInput && defaultColors.absent) {
      absentColorInput.value = defaultColors.absent;
    }
    applyColorVars();
  });
  clearBtn?.addEventListener('click', () => {
    input.value = '';
    hasRun = false;
    counts.innerHTML = '';
    if (resultsList) resultsList.innerHTML = '';
    summary.textContent = 'Paste text and click Check.';
    if (empty) {
      empty.hidden = false;
      empty.textContent = 'Waiting for input.';
    }
    output.innerHTML = '<p class="oxford-empty">Paste text and click Check.</p>';
    input.focus();
  });
})();
