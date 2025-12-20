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
  const onlyMissingToggle = $('[data-oxford-only-missing]');
  const conjInputs = $$('[data-oxford-conjunction]');

  if (!form || !input || !summary || !counts || !resultsList || !output) return;

  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const escapeHtml = (value) => value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  const formatNumber = (value) => value.toLocaleString('en-US');
  const truncate = (value, max = 180) => (value.length <= max ? value : `${value.slice(0, max - 3)}...`);

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

  const buildSuggestion = (text, conjIndex) => {
    if (conjIndex < 0) return '';
    const before = text.slice(0, conjIndex).replace(/\s*$/, '');
    const after = text.slice(conjIndex).replace(/^\s+/, ' ');
    return `${before},${after}`;
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
        suggestion: hasOxford ? '' : buildSuggestion(raw, conjIndex),
      });
    }
    return matches;
  };

  const renderCounts = (total, missing, present) => {
    counts.innerHTML = `
      <li>
        <strong>${formatNumber(total)}</strong>
        <span>List candidates</span>
      </li>
      <li>
        <strong>${formatNumber(missing)}</strong>
        <span>Missing Oxford commas</span>
      </li>
      <li>
        <strong>${formatNumber(present)}</strong>
        <span>Oxford comma present</span>
      </li>
    `;
  };

  const renderResults = (matches, onlyMissing) => {
    resultsList.innerHTML = '';
    const visible = onlyMissing ? matches.filter((item) => !item.hasOxford) : matches;
    if (!visible.length) {
      if (empty) {
        empty.hidden = false;
        empty.textContent = matches.length
          ? 'No missing Oxford commas found with the current filters.'
          : 'No list candidates detected yet.';
      }
      return;
    }
    if (empty) empty.hidden = true;
    visible.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'oxford-result';
      li.dataset.status = item.hasOxford ? 'present' : 'missing';
      const snippet = truncate(item.text.trim());
      const suggestion = item.suggestion ? truncate(item.suggestion.trim()) : '';
      li.innerHTML = `
        <div class="oxford-result-header">
          <span class="oxford-result-status">${item.hasOxford ? 'Oxford comma present' : 'Missing Oxford comma'}</span>
          <span class="oxford-result-tag">${item.conj}</span>
        </div>
        <p class="oxford-result-text">${escapeHtml(snippet)}</p>
        ${item.hasOxford ? '' : `<p class="oxford-result-fix"><span>Suggestion:</span> ${escapeHtml(suggestion)}</p>`}
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
    const text = input.value || '';
    if (!text.trim()) {
      summary.textContent = 'Paste text and click Check.';
      counts.innerHTML = '';
      resultsList.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'Waiting for input.';
      }
      renderOutput('', []);
      return;
    }
    const conjunctions = getConjunctions();
    if (!conjunctions.length) {
      summary.textContent = 'Select at least one conjunction to scan.';
      counts.innerHTML = '';
      resultsList.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = 'Choose "and", "or", or "nor" to scan lists.';
      }
      renderOutput(text, []);
      return;
    }
    const matches = findMatches(text, conjunctions);
    const missing = matches.filter((item) => !item.hasOxford);
    const present = matches.length - missing.length;
    renderCounts(matches.length, missing.length, present);
    if (matches.length) {
      summary.textContent = `${formatNumber(matches.length)} list candidate${matches.length === 1 ? '' : 's'} found. ${formatNumber(missing.length)} missing Oxford comma${missing.length === 1 ? '' : 's'}.`;
    } else {
      summary.textContent = 'No list candidates detected. Try a longer sample.';
    }
    const onlyMissing = Boolean(onlyMissingToggle?.checked);
    const highlights = matches.map((item) => ({
      start: item.start,
      end: item.end,
      status: item.hasOxford ? 'present' : 'missing',
    }));
    renderResults(matches, onlyMissing);
    renderOutput(text, highlights);
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  const maybeRerun = () => {
    if (!input.value.trim()) return;
    runAnalysis();
  };

  conjInputs.forEach((item) => {
    item.addEventListener('change', maybeRerun);
  });
  onlyMissingToggle?.addEventListener('change', maybeRerun);

  clearBtn?.addEventListener('click', () => {
    input.value = '';
    counts.innerHTML = '';
    resultsList.innerHTML = '';
    summary.textContent = 'Paste text and click Check.';
    if (empty) {
      empty.hidden = false;
      empty.textContent = 'Waiting for input.';
    }
    output.innerHTML = '<p class="oxford-empty">Paste text and click Check.</p>';
    input.focus();
  });
})();
