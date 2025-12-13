(() => {
  'use strict';
  const $ = (sel) => document.querySelector(sel);
  const form = $('#wordfreq-form');
  const textInput = $('#wordfreq-text');
  const topInput = $('#wordfreq-top');
  const resultsList = $('#wordfreq-results');
  const summaryEl = $('#wordfreq-summary');
  const emptyEl = $('#wordfreq-empty');
  const clearBtn = $('#wordfreq-clear');

  if (!form || !textInput || !topInput || !resultsList || !summaryEl) return;

  const STOPWORDS = new Set([
    'a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by',
    'can','cannot','can\'t','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t',
    'have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into',
    'is','isn\'t','it','it\'s','its','itself','just','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours',
    'ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves',
    'then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll',
    'we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you',
    'you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves'
  ]);

  const formatNumber = (n) => n.toLocaleString('en-US');
  const clampTop = (value) => {
    const parsed = parseInt(value, 10);
    if (Number.isNaN(parsed)) return 10;
    return Math.max(1, Math.min(200, parsed));
  };

  const extractWords = (text) => {
    if (!text) return [];
    const cleaned = text.replace(/[_-]/g, ' ').toLowerCase();
    const matches = cleaned.match(/[a-z']+/g) || [];
    return matches
      .map((word) => word.replace(/^'+|'+$/g, ''))
      .filter(Boolean);
  };

  const filterStopwords = (words) => words.filter((word) => !STOPWORDS.has(word));

  const buildCounts = (words) => {
    const counts = new Map();
    words.forEach((word) => {
      counts.set(word, (counts.get(word) || 0) + 1);
    });
    return counts;
  };

  const renderEmpty = (reason) => {
    resultsList.innerHTML = '';
    if (emptyEl) {
      emptyEl.hidden = false;
      emptyEl.textContent = reason;
    }
  };

  const renderResults = (entries, maxCount) => {
    resultsList.innerHTML = '';
    entries.forEach(([word, count], index) => {
      const item = document.createElement('li');
      item.className = 'wordfreq-row';
      const percent = maxCount ? Math.max(8, (count / maxCount) * 100) : 0;
      item.innerHTML = `
        <div class="wordfreq-meta">
          <span class="wordfreq-rank">${index + 1}</span>
          <div class="wordfreq-wordwrap">
            <span class="wordfreq-word">${word}</span>
            <span class="wordfreq-count">${formatNumber(count)}×</span>
          </div>
        </div>
        <div class="wordfreq-bar" role="presentation" aria-hidden="true">
          <span style="width:${percent.toFixed(1)}%"></span>
        </div>
      `;
      resultsList.appendChild(item);
    });
  };

  const runAnalysis = () => {
    const top = clampTop(topInput.value);
    topInput.value = top;
    const rawWords = extractWords(textInput.value);
    const filteredWords = filterStopwords(rawWords);
    if (!filteredWords.length) {
      summaryEl.textContent = rawWords.length
        ? 'All tokens were stopwords. Try a different passage.'
        : 'Paste text and submit to see the most common words.';
      renderEmpty(rawWords.length ? 'No words left after removing stopwords.' : 'Waiting for input.');
      return;
    }

    const counts = buildCounts(filteredWords);
    const sorted = [...counts.entries()].sort((a, b) => {
      if (b[1] === a[1]) return a[0].localeCompare(b[0]);
      return b[1] - a[1];
    });
    const topEntries = sorted.slice(0, top);
    const maxCount = topEntries[0]?.[1] || 0;
    const removed = rawWords.length - filteredWords.length;

    if (emptyEl) emptyEl.hidden = true;
    renderResults(topEntries, maxCount);
    summaryEl.innerHTML = `
      Showing <strong>${topEntries.length}</strong> of <strong>${counts.size}</strong> unique words
      from <strong>${formatNumber(rawWords.length)}</strong> tokens
      (${formatNumber(removed)} stopwords removed).
    `;
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runAnalysis();
  });

  clearBtn?.addEventListener('click', () => {
    textInput.value = '';
    resultsList.innerHTML = '';
    if (emptyEl) {
      emptyEl.hidden = false;
      emptyEl.textContent = 'Paste text to get started.';
    }
    summaryEl.textContent = 'Waiting for input.';
    textInput.focus();
  });
})();
