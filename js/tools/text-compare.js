(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const form = $('#textcompare-form');
  const originalEl = $('#textcompare-original');
  const revisedEl = $('#textcompare-revised');
  const outputEl = $('#textcompare-output');
  const summaryEl = $('#textcompare-summary');
  const clearBtn = $('#textcompare-clear');
  const swapBtn = $('#textcompare-swap');

  if (!form || !originalEl || !revisedEl || !outputEl || !summaryEl) return;

  const MAX_CHARS = 200_000;
  const MAX_TOKENS = 20_000;

  const escapeHtml = (s) => String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const countWords = (s) => (String(s || '').match(/\S+/g) || []).length;

  const tokenize = (text) => {
    const s = String(text || '');
    const tokens = [];
    const len = s.length;
    let i = 0;

    if (i < len && /\s/.test(s[i])) {
      let j = i;
      while (j < len && /\s/.test(s[j])) j += 1;
      tokens.push(s.slice(i, j));
      i = j;
    }

    while (i < len) {
      let wordEnd = i;
      while (wordEnd < len && !/\s/.test(s[wordEnd])) wordEnd += 1;
      let spaceEnd = wordEnd;
      while (spaceEnd < len && /\s/.test(s[spaceEnd])) spaceEnd += 1;
      tokens.push(s.slice(i, spaceEnd));
      i = spaceEnd;
    }
    return tokens;
  };

  const myersEdits = (a, b) => {
    const n = a.length;
    const m = b.length;
    const max = n + m;
    const offset = max;
    let v = new Array(2 * max + 1).fill(0);
    const trace = [];

    const backtrack = () => {
      let x = n;
      let y = m;
      const edits = [];
      const D = trace.length - 1;

      for (let d = D; d > 0; d -= 1) {
        const vPrev = trace[d - 1];
        const k = x - y;
        const kIdx = k + offset;
        const down = k === -d || (k !== d && vPrev[kIdx - 1] < vPrev[kIdx + 1]);
        const prevK = down ? k + 1 : k - 1;
        const prevX = vPrev[prevK + offset];
        const prevY = prevX - prevK;

        while (x > prevX && y > prevY) {
          edits.push({ type: 'equal', value: a[x - 1] });
          x -= 1;
          y -= 1;
        }

        if (down) {
          edits.push({ type: 'insert', value: b[prevY] });
        } else {
          edits.push({ type: 'delete', value: a[prevX] });
        }

        x = prevX;
        y = prevY;
      }

      while (x > 0 && y > 0) {
        edits.push({ type: 'equal', value: a[x - 1] });
        x -= 1;
        y -= 1;
      }
      while (x > 0) {
        edits.push({ type: 'delete', value: a[x - 1] });
        x -= 1;
      }
      while (y > 0) {
        edits.push({ type: 'insert', value: b[y - 1] });
        y -= 1;
      }

      edits.reverse();
      return edits;
    };

    for (let d = 0; d <= max; d += 1) {
      for (let k = -d; k <= d; k += 2) {
        const kIdx = k + offset;
        let x;
        const down = k === -d || (k !== d && v[kIdx - 1] < v[kIdx + 1]);
        if (down) {
          x = v[kIdx + 1];
        } else {
          x = v[kIdx - 1] + 1;
        }
        let y = x - k;
        while (x < n && y < m && a[x] === b[y]) {
          x += 1;
          y += 1;
        }
        v[kIdx] = x;
        if (x >= n && y >= m) {
          trace.push(v.slice());
          return backtrack();
        }
      }
      trace.push(v.slice());
    }
    return [];
  };

  const groupRuns = (edits) => {
    const runs = [];
    let current = null;
    edits.forEach((edit) => {
      if (!current || current.type !== edit.type) {
        current = { type: edit.type, tokens: [edit.value] };
        runs.push(current);
      } else {
        current.tokens.push(edit.value);
      }
    });
    return runs;
  };

  const normalizeRuns = (runs) => {
    const out = [];
    for (let i = 0; i < runs.length; i += 1) {
      const run = runs[i];
      const next = runs[i + 1];
      if (run.type === 'delete' && next && next.type === 'insert') {
        out.push({ type: 'replace', delTokens: run.tokens, insTokens: next.tokens });
        i += 1;
        continue;
      }
      if (run.type === 'insert' && next && next.type === 'delete') {
        out.push({ type: 'replace', delTokens: next.tokens, insTokens: run.tokens });
        i += 1;
        continue;
      }
      out.push(run);
    }
    return out;
  };

  const mergeEdits = (edits) => {
    const merged = [];
    let current = null;
    edits.forEach((edit) => {
      if (!current || current.type !== edit.type) {
        current = { type: edit.type, text: edit.value };
        merged.push(current);
      } else {
        current.text += edit.value;
      }
    });
    return merged;
  };

  const sharedEdgeScore = (a, b) => {
    const left = String(a || '');
    const right = String(b || '');
    const maxCheck = Math.min(left.length, right.length);
    let prefix = 0;
    while (prefix < maxCheck && left[prefix] === right[prefix]) prefix += 1;
    let suffix = 0;
    while (
      suffix < maxCheck - prefix &&
      left[left.length - 1 - suffix] === right[right.length - 1 - suffix]
    ) {
      suffix += 1;
    }
    return prefix + suffix;
  };

  const renderCharDiff = (delCore, insCore, kind) => {
    const edits = myersEdits(Array.from(delCore), Array.from(insCore));
    const segments = mergeEdits(edits);
    return segments.map((seg) => {
      if (seg.type === 'equal') return escapeHtml(seg.text);
      if (kind === 'del' && seg.type === 'delete') return `<span class="diff-char-del">${escapeHtml(seg.text)}</span>`;
      if (kind === 'ins' && seg.type === 'insert') return `<span class="diff-char-ins">${escapeHtml(seg.text)}</span>`;
      return '';
    }).join('');
  };

  const renderReplace = (delText, insText) => {
    const delTrim = String(delText || '').trim();
    const insTrim = String(insText || '').trim();
    const singleToken = delTrim && insTrim && !/\s/.test(delTrim) && !/\s/.test(insTrim);
    const smallEnough = delTrim.length <= 42 && insTrim.length <= 42;
    const similarEnough = sharedEdgeScore(delTrim, insTrim) >= 2;

    if (!singleToken || !smallEnough || !similarEnough) {
      return `<del class="diff-del">${escapeHtml(delText)}</del><ins class="diff-ins">${escapeHtml(insText)}</ins>`;
    }

    const delLeading = (String(delText || '').match(/^\s+/) || [''])[0];
    const delTrailing = (String(delText || '').match(/\s+$/) || [''])[0];
    const insLeading = (String(insText || '').match(/^\s+/) || [''])[0];
    const insTrailing = (String(insText || '').match(/\s+$/) || [''])[0];
    const delCore = String(delText || '').slice(delLeading.length, String(delText || '').length - delTrailing.length);
    const insCore = String(insText || '').slice(insLeading.length, String(insText || '').length - insTrailing.length);

    const delInner = `${escapeHtml(delLeading)}${renderCharDiff(delCore, insCore, 'del')}${escapeHtml(delTrailing)}`;
    const insInner = `${escapeHtml(insLeading)}${renderCharDiff(delCore, insCore, 'ins')}${escapeHtml(insTrailing)}`;
    return `<del class="diff-del">${delInner}</del><ins class="diff-ins">${insInner}</ins>`;
  };

  const renderOutput = (runs) => runs.map((run) => {
    if (run.type === 'equal') return escapeHtml(run.tokens.join(''));
    if (run.type === 'insert') return `<ins class="diff-ins">${escapeHtml(run.tokens.join(''))}</ins>`;
    if (run.type === 'delete') return `<del class="diff-del">${escapeHtml(run.tokens.join(''))}</del>`;
    if (run.type === 'replace') {
      const delText = run.delTokens.join('');
      const insText = run.insTokens.join('');
      return renderReplace(delText, insText);
    }
    return '';
  }).join('');

  const setEmpty = (msg) => {
    outputEl.innerHTML = `<p class="textcompare-empty">${escapeHtml(msg)}</p>`;
  };

  const runCompare = () => {
    const original = originalEl.value || '';
    const revised = revisedEl.value || '';
    if (!original.trim() && !revised.trim()) {
      summaryEl.textContent = 'Paste text above and click Compare.';
      setEmpty('Waiting for input.');
      return;
    }

    if (original.length + revised.length > MAX_CHARS) {
      summaryEl.textContent = 'Text is too large to compare in-browser. Please compare smaller sections.';
      setEmpty('Input too large.');
      return;
    }

    summaryEl.textContent = 'Comparing…';
    setEmpty('Comparing…');

    requestAnimationFrame(() => {
      const aTokens = tokenize(original);
      const bTokens = tokenize(revised);
      if (aTokens.length + bTokens.length > MAX_TOKENS) {
        summaryEl.textContent = 'Text is too large to compare quickly. Please compare smaller sections.';
        setEmpty('Input too large.');
        return;
      }

      const edits = myersEdits(aTokens, bTokens);
      const runs = normalizeRuns(groupRuns(edits));
      const html = renderOutput(runs);
      outputEl.innerHTML = html || '<p class="textcompare-empty">No output.</p>';

      let insertedWords = 0;
      let deletedWords = 0;
      let replacements = 0;
      runs.forEach((run) => {
        if (run.type === 'insert') insertedWords += countWords(run.tokens.join(''));
        if (run.type === 'delete') deletedWords += countWords(run.tokens.join(''));
        if (run.type === 'replace') {
          replacements += 1;
          insertedWords += countWords(run.insTokens.join(''));
          deletedWords += countWords(run.delTokens.join(''));
        }
      });

      if (!insertedWords && !deletedWords) {
        summaryEl.textContent = 'No differences found.';
        return;
      }
      const parts = [];
      if (insertedWords) parts.push(`${insertedWords.toLocaleString('en-US')} inserted words`);
      if (deletedWords) parts.push(`${deletedWords.toLocaleString('en-US')} deleted words`);
      if (replacements) parts.push(`${replacements.toLocaleString('en-US')} replacements`);
      summaryEl.textContent = `Changes: ${parts.join(' · ')}.`;
    });
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runCompare();
  });

  clearBtn?.addEventListener('click', () => {
    originalEl.value = '';
    revisedEl.value = '';
    summaryEl.textContent = 'Paste text above and click Compare.';
    setEmpty('Waiting for input.');
    originalEl.focus();
  });

  swapBtn?.addEventListener('click', () => {
    const a = originalEl.value;
    originalEl.value = revisedEl.value;
    revisedEl.value = a;
    runCompare();
  });
})();
