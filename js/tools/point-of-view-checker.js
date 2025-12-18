(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);

  const form = $('#povcheck-form');
  const textInput = $('#povcheck-text');
  const includeItToggle = $('#povcheck-include-it');
  const summaryEl = $('#povcheck-summary');
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
  if (!firstBadge || !firstCount || !firstList) return;
  if (!secondBadge || !secondCount || !secondList) return;
  if (!thirdBadge || !thirdCount || !thirdList) return;

  const normalizeText = (text) => String(text || '')
    .replace(/[\u2018\u2019\u201B\uFF07]/g, "'")
    .replace(/\u00A0/g, ' ');

  const extractTokens = (text) => {
    const normalized = normalizeText(text).toLowerCase();
    return normalized.match(/[a-z]+(?:'[a-z]+)*/g) || [];
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

  const renderSummary = ({ firstTotal, secondTotal, thirdTotal, tokenCount }) => {
    const total = firstTotal + secondTotal + thirdTotal;

    if (!tokenCount) {
      summaryEl.textContent = 'Paste text above and click Check.';
      return;
    }

    if (!total) {
      summaryEl.textContent = 'No personal pronouns detected in this text.';
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
      `Total pronouns: <strong>${total.toLocaleString('en-US')}</strong>.`,
      `Detected: <strong>${detected.join(', ')}</strong>.`,
      dominant ? `Dominant POV: <strong>${dominant}</strong>.` : ''
    ].filter(Boolean);

    summaryEl.innerHTML = pieces.join(' ');
    if (detected.length > 1) {
      summaryEl.innerHTML = `Mixed point of view. ${summaryEl.innerHTML}`;
    }
  };

  const resetUI = () => {
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
  };

  const runAnalysis = () => {
    const tokens = extractTokens(textInput.value);
    const includeNeutral = includeItToggle ? includeItToggle.checked : true;
    const thirdSet = buildThirdSet(includeNeutral);

    const first = buildCounts(tokens, FIRST_PERSON);
    const second = buildCounts(tokens, SECOND_PERSON);
    const third = buildCounts(tokens, thirdSet);

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
      tokenCount: tokens.length
    });
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  includeItToggle?.addEventListener('change', () => {
    if (!extractTokens(textInput.value).length) return;
    runAnalysis();
  });

  clearBtn?.addEventListener('click', () => {
    textInput.value = '';
    resetUI();
    textInput.focus();
  });

  resetUI();
})();
