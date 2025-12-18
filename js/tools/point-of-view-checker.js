(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  const form = $('#povcheck-form');
  const textInput = $('#povcheck-text');
  const includeItToggle = $('#povcheck-include-it');
  const thirdReferencesInput = $('#povcheck-third-references');
  const summaryEl = $('#povcheck-summary');
  const outputEl = $('#povcheck-output');
  const clearBtn = $('#povcheck-clear');

  const firstBadge = $('#povcheck-first-badge');
  const firstCount = $('#povcheck-first-count');
  const firstList = $('#povcheck-first-list');
  const secondBadge = $('#povcheck-second-badge');
  const secondCount = $('#povcheck-second-count');
  const secondList = $('#povcheck-second-list');
  const thirdBadge = $('#povcheck-third-badge');
  const thirdCount = $('#povcheck-third-count');
  const thirdList = $('#povcheck-third-list');

  if (!form || !textInput || !summaryEl) return;
  if (!outputEl) return;
  if (!firstBadge || !firstCount || !firstList) return;
  if (!secondBadge || !secondCount || !secondList) return;
  if (!thirdBadge || !thirdCount || !thirdList) return;

  const normalizeText = (text) => String(text || '')
    .replace(/[\u2018\u2019\u201B\uFF07]/g, "'")
    .replace(/\u00A0/g, ' ');

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const collapseWhitespace = (value) => String(value || '')
    .trim()
    .replace(/\s+/g, ' ');

  const escapeRegExp = (value) => String(value || '')
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const extractTokens = (text) => {
    const normalized = normalizeText(text).toLowerCase();
    return normalized.match(/[a-z]+(?:'[a-z]+)*/g) || [];
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

  const getEffectiveThirdReferences = (textHasUser) => {
    if (!thirdReferencesInput) return [];
    const raw = thirdReferencesInput.value || '';
    if (raw.trim()) return parseThirdReferencesText(raw);
    if (!textHasUser) return parseThirdReferencesText(thirdReferencesInput.placeholder || '');
    return [];
  };

  const buildThirdReferenceRegex = (phrase) => {
    const base = collapseWhitespace(phrase);
    if (!base) return null;

    const endsWithPossessive = /'s$/.test(base) || /'$/.test(base);
    const body = base
      .split(' ')
      .map(escapeRegExp)
      .join('\\s+');
    const possessive = endsWithPossessive ? '' : "(?:'s)?";
    return new RegExp(`\\b${body}${possessive}\\b`, 'g');
  };

  const countThirdReferences = (text, phrases) => {
    const normalizedText = normalizeText(text).toLowerCase();
    const counts = new Map();
    let total = 0;

    phrases.forEach((phrase) => {
      const rx = buildThirdReferenceRegex(phrase);
      if (!rx) return;
      rx.lastIndex = 0;
      let match;
      while ((match = rx.exec(normalizedText)) !== null) {
        const token = collapseWhitespace(match[0]);
        counts.set(token, (counts.get(token) || 0) + 1);
        total += 1;
        if (match.index === rx.lastIndex) rx.lastIndex += 1;
      }
    });

    return { counts, total };
  };

  const sortTokenCounts = (counts) => [...counts.entries()].sort((a, b) => {
    if (b[1] === a[1]) return a[0].localeCompare(b[0]);
    return b[1] - a[1];
  });

  const renderBadge = (badgeEl, total) => {
    const detected = total > 0;
    badgeEl.textContent = detected ? 'Detected' : 'Not found';
    badgeEl.classList.toggle('povcheck-badge-detected', detected);
    badgeEl.classList.toggle('povcheck-badge-muted', !detected);
  };

  const renderTokenList = (listEl, counts) => {
    const entries = sortTokenCounts(counts);
    listEl.innerHTML = '';
    if (!entries.length) {
      const empty = document.createElement('li');
      empty.className = 'povcheck-token-empty';
      empty.textContent = 'None found.';
      listEl.appendChild(empty);
      return;
    }
    entries.forEach(([token, count]) => {
      const li = document.createElement('li');
      li.className = 'povcheck-token-pill';
      li.innerHTML = `
        <span class="povcheck-token">${token}</span>
        <span class="povcheck-token-count">${count.toLocaleString('en-US')}×</span>
      `;
      listEl.appendChild(li);
    });
  };

  const buildCounts = (tokens, allowed) => {
    const counts = new Map();
    let total = 0;
    tokens.forEach((token) => {
      if (!allowed.has(token)) return;
      counts.set(token, (counts.get(token) || 0) + 1);
      total++;
    });
    return { total, counts };
  };

  const FIRST_PERSON = new Set([
    'i','me','my','mine','myself','we','us','our','ours','ourselves',
    "i'm","i'd","i'll","i've","we're","we'd","we'll","we've"
  ]);

  const SECOND_PERSON = new Set([
    'you','your','yours','yourself','yourselves',
    "you're","you'd","you'll","you've","y'all"
  ]);

  const THIRD_PERSON_BASE = new Set([
    'he','him','his','himself',"he's","he'd","he'll",
    'she','her','hers','herself',"she's","she'd","she'll",
    'they','them','their','theirs','themself','themselves',"they're","they'd","they'll","they've"
  ]);

  const THIRD_PERSON_NEUTRAL = new Set(['it','its','itself',"it's"]);

  const buildThirdSet = (includeNeutral) => {
    if (!includeNeutral) return THIRD_PERSON_BASE;
    return new Set([...THIRD_PERSON_BASE, ...THIRD_PERSON_NEUTRAL]);
  };

  const buildHighlightRanges = ({ displayText, includeNeutral, thirdRefs }) => {
    const text = String(displayText || '');
    const lowerText = text.toLowerCase();
    const thirdSet = buildThirdSet(includeNeutral);
    const ranges = [];

    thirdRefs.forEach((phrase) => {
      const rx = buildThirdReferenceRegex(phrase);
      if (!rx) return;
      rx.lastIndex = 0;
      let match;
      while ((match = rx.exec(lowerText)) !== null) {
        ranges.push({ start: match.index, end: match.index + match[0].length, pov: 'third' });
        if (match.index === rx.lastIndex) rx.lastIndex += 1;
      }
    });

    const tokenRx = /[a-z]+(?:'[a-z]+)*/g;
    let tokenMatch;
    while ((tokenMatch = tokenRx.exec(lowerText)) !== null) {
      const token = tokenMatch[0];
      let pov = null;
      if (FIRST_PERSON.has(token)) pov = 'first';
      else if (SECOND_PERSON.has(token)) pov = 'second';
      else if (thirdSet.has(token)) pov = 'third';
      if (!pov) continue;
      ranges.push({ start: tokenMatch.index, end: tokenMatch.index + token.length, pov });
    }

    ranges.sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return (b.end - b.start) - (a.end - a.start);
    });

    const finalRanges = [];
    let lastEnd = -1;
    ranges.forEach((range) => {
      if (range.start < lastEnd) return;
      finalRanges.push(range);
      lastEnd = range.end;
    });

    return finalRanges;
  };

  const renderHighlightedOutput = ({ text, includeNeutral, thirdRefs }) => {
    const displayText = normalizeText(text);
    if (!displayText.trim()) return '<p class="povcheck-empty">Waiting for input.</p>';

    const ranges = buildHighlightRanges({ displayText, includeNeutral, thirdRefs });
    if (!ranges.length) return escapeHtml(displayText);

    let html = '';
    let cursor = 0;
    ranges.forEach((range) => {
      html += escapeHtml(displayText.slice(cursor, range.start));
      html += `<mark class="povcheck-mark povcheck-mark-${range.pov}" data-pov="${range.pov}">${escapeHtml(displayText.slice(range.start, range.end))}</mark>`;
      cursor = range.end;
    });
    html += escapeHtml(displayText.slice(cursor));
    return html || '<p class="povcheck-empty">No output.</p>';
  };

  let hasRun = false;

  const getEffectiveInput = () => {
    const raw = textInput.value || '';
    const hasUser = Boolean(raw.trim());
    return {
      text: hasUser ? raw : (textInput.placeholder || ''),
      hasUser
    };
  };

  const renderSummary = ({ firstTotal, secondTotal, thirdTotal, hasText }) => {
    const total = firstTotal + secondTotal + thirdTotal;

    if (!hasText) {
      summaryEl.textContent = 'Paste text above and click Check.';
      return;
    }

    if (!total) {
      summaryEl.textContent = 'No pronouns or third-person references detected in this text.';
      return;
    }

    const detected = [];
    if (firstTotal) detected.push('first person');
    if (secondTotal) detected.push('second person');
    if (thirdTotal) detected.push('third person');

    const sorted = [
      { label: 'first person', count: firstTotal },
      { label: 'second person', count: secondTotal },
      { label: 'third person', count: thirdTotal }
    ].sort((a, b) => b.count - a.count);
    const dominant = sorted[0]?.count ? sorted[0].label : '';

    const pieces = [
      `Total matches: <strong>${total.toLocaleString('en-US')}</strong>.`,
      `Detected: <strong>${detected.join(', ')}</strong>.`,
      dominant ? `Dominant POV: <strong>${dominant}</strong>.` : ''
    ].filter(Boolean);

    summaryEl.innerHTML = pieces.join(' ');
    if (detected.length > 1) {
      summaryEl.innerHTML = `Mixed point of view. ${summaryEl.innerHTML}`;
    }
  };

  const resetUI = () => {
    hasRun = false;
    [firstCount, secondCount, thirdCount].forEach((el) => { el.textContent = '0'; });
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
  };

  const runAnalysis = () => {
    const { text, hasUser } = getEffectiveInput();
    const hasText = Boolean(String(text || '').trim());
    const tokens = extractTokens(text);
    const includeNeutral = includeItToggle ? includeItToggle.checked : true;
    const thirdSet = buildThirdSet(includeNeutral);

    const first = buildCounts(tokens, FIRST_PERSON);
    const second = buildCounts(tokens, SECOND_PERSON);
    const third = buildCounts(tokens, thirdSet);

    const thirdRefs = getEffectiveThirdReferences(hasUser);
    if (thirdRefs.length) {
      const refs = countThirdReferences(text, thirdRefs);
      refs.counts.forEach((count, key) => {
        third.counts.set(key, (third.counts.get(key) || 0) + count);
      });
      third.total += refs.total;
    }

    firstCount.textContent = first.total.toLocaleString('en-US');
    secondCount.textContent = second.total.toLocaleString('en-US');
    thirdCount.textContent = third.total.toLocaleString('en-US');

    renderBadge(firstBadge, first.total);
    renderBadge(secondBadge, second.total);
    renderBadge(thirdBadge, third.total);

    renderTokenList(firstList, first.counts);
    renderTokenList(secondList, second.counts);
    renderTokenList(thirdList, third.counts);

    renderSummary({
      firstTotal: first.total,
      secondTotal: second.total,
      thirdTotal: third.total,
      hasText
    });

    outputEl.innerHTML = renderHighlightedOutput({
      text,
      includeNeutral,
      thirdRefs
    });

    hasRun = true;
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  includeItToggle?.addEventListener('change', () => {
    if (!hasRun) return;
    runAnalysis();
  });

  thirdReferencesInput?.addEventListener('input', () => {
    if (!hasRun) return;
    runAnalysis();
  });

  clearBtn?.addEventListener('click', () => {
    textInput.value = '';
    resetUI();
    textInput.focus();
  });

  resetUI();
})();
