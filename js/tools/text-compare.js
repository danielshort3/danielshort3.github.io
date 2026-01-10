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
  const copyBtn = $('#textcompare-copy');
  const copyStatus = $('#textcompare-copy-status');
  const mergeWordsEl = $('#textcompare-merge-words');
  const mergeCharsEl = $('#textcompare-merge-chars');
  const mergeRunsEl = $('#textcompare-merge-runs');
  const softWordsEl = $('#textcompare-soft-words');
  const softWordLenEl = $('#textcompare-soft-word-length');
  const commonMinEl = $('#textcompare-common-min');
  const insBgEl = $('#textcompare-ins-bg');
  const insTextEl = $('#textcompare-ins-text');
  const delBgEl = $('#textcompare-del-bg');
  const delTextEl = $('#textcompare-del-text');
  const delStrikeEl = $('#textcompare-del-strike');

  if (!form || !originalEl || !revisedEl || !outputEl || !summaryEl) return;

  const TOOL_ID = 'text-compare';
  const MAX_CHARS = 600_000;
  const MAX_TOKENS = 200_000;
  const MAX_TRACE_CELLS = 16_000_000;
  const DEFAULT_COMPARE_SETTINGS = {
    mergeMinWords: 8,
    mergeMinChars: 40,
    mergeMinRuns: 4,
    softEqualMaxWords: 2,
    softEqualMaxWordLength: 2,
    commonWordMinCount: 2
  };
  let lastRuns = null;
  let lastRevisedText = '';

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const escapeHtml = (s) => String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const setCopyStatus = (msg, tone) => {
    if (!copyStatus) return;
    copyStatus.textContent = msg;
    copyStatus.dataset.tone = tone || '';
  };

  const clampInt = (value, min, max, fallback) => {
    const parsed = Number.parseInt(value, 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const getCompareSettings = () => ({
    mergeMinWords: clampInt(mergeWordsEl?.value, 0, 200, DEFAULT_COMPARE_SETTINGS.mergeMinWords),
    mergeMinChars: clampInt(mergeCharsEl?.value, 0, 500, DEFAULT_COMPARE_SETTINGS.mergeMinChars),
    mergeMinRuns: clampInt(mergeRunsEl?.value, 1, 200, DEFAULT_COMPARE_SETTINGS.mergeMinRuns),
    softEqualMaxWords: clampInt(softWordsEl?.value, 0, 12, DEFAULT_COMPARE_SETTINGS.softEqualMaxWords),
    softEqualMaxWordLength: clampInt(softWordLenEl?.value, 0, 12, DEFAULT_COMPARE_SETTINGS.softEqualMaxWordLength),
    commonWordMinCount: clampInt(commonMinEl?.value, 1, 20, DEFAULT_COMPARE_SETTINGS.commonWordMinCount)
  });

  const buildTokenRegexes = () => {
    const unicodeWord = "[\\p{L}\\p{N}_]+(?:['-][\\p{L}\\p{N}_]+)*";
    try {
      return {
        word: new RegExp(unicodeWord, 'gu'),
        wordToken: new RegExp(`^${unicodeWord}$`, 'u'),
        wordChar: new RegExp('[\\p{L}\\p{N}_]', 'u'),
        token: new RegExp(`${unicodeWord}|\\s+|[^\\p{L}\\p{N}_\\s]+`, 'gu')
      };
    } catch {
      const asciiWord = "[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*";
      return {
        word: new RegExp(asciiWord, 'g'),
        wordToken: new RegExp(`^${asciiWord}$`),
        wordChar: new RegExp('[A-Za-z0-9_]'),
        token: new RegExp(`${asciiWord}|\\s+|[^A-Za-z0-9_\\s]+`, 'g')
      };
    }
  };

  const {
    word: WORD_RE,
    token: TOKEN_RE,
    wordToken: WORD_TOKEN_RE,
    wordChar: WORD_CHAR_RE
  } = buildTokenRegexes();

  const matchAll = (re, text) => {
    re.lastIndex = 0;
    return text.match(re) || [];
  };

  const countWords = (s) => matchAll(WORD_RE, String(s || '')).length;

  const tokenize = (text) => matchAll(TOKEN_RE, String(text || ''));

  const isWordToken = (token) => WORD_TOKEN_RE.test(token);

  const hasWordChar = (text) => WORD_CHAR_RE.test(text);

  const countWordTokens = (tokens) => tokens.reduce(
    (sum, token) => sum + (isWordToken(token) ? 1 : 0),
    0
  );

  const buildWordCounts = (tokens) => {
    const counts = new Map();
    tokens.forEach((token) => {
      if (!isWordToken(token)) return;
      const key = token.toLowerCase();
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    return counts;
  };

  const isCommonWord = (token, countsA, countsB, settings) => {
    const key = token.toLowerCase();
    const threshold = settings.commonWordMinCount;
    return (countsA.get(key) || 0) >= threshold || (countsB.get(key) || 0) >= threshold;
  };

  const isSoftEqual = (run, countsA, countsB, settings) => {
    if (run.type !== 'equal') return false;
    const text = run.tokens.join('');
    if (!text) return true;
    if (!hasWordChar(text)) return true;
    const words = run.tokens.filter(isWordToken);
    if (!words.length) return true;
    if (settings.softEqualMaxWords <= 0) return false;
    if (words.length <= settings.softEqualMaxWords && words.every((word) => isCommonWord(word, countsA, countsB, settings))) return true;
    if (
      settings.softEqualMaxWordLength > 0 &&
      words.length <= settings.softEqualMaxWords &&
      words.every((word) => word.length <= settings.softEqualMaxWordLength)
    ) {
      return true;
    }
    return false;
  };

  const coalesceRuns = (runs, countsA, countsB, settings) => {
    const merged = [];
    const MIN_MERGE_WORDS = settings.mergeMinWords;
    const MIN_MERGE_CHARS = settings.mergeMinChars;
    const MIN_MERGE_RUNS = settings.mergeMinRuns;

    for (let i = 0; i < runs.length; i += 1) {
      const run = runs[i];
      if (run.type === 'equal') {
        merged.push(run);
        continue;
      }

      const segment = [];
      let hasInsert = false;
      let hasDelete = false;
      let softEqualCount = 0;
      let editRunCount = 0;
      let editWordCount = 0;
      let editCharCount = 0;
      let j = i;

      while (j < runs.length) {
        const current = runs[j];
        if (current.type === 'equal' && !isSoftEqual(current, countsA, countsB, settings)) break;
        segment.push(current);
        if (current.type === 'equal') {
          softEqualCount += 1;
        } else {
          editRunCount += 1;
          if (current.type === 'insert') {
            hasInsert = true;
            editWordCount += countWordTokens(current.tokens);
            editCharCount += current.tokens.join('').length;
          } else if (current.type === 'delete') {
            hasDelete = true;
            editWordCount += countWordTokens(current.tokens);
            editCharCount += current.tokens.join('').length;
          } else if (current.type === 'replace') {
            hasInsert = true;
            hasDelete = true;
            editWordCount += countWordTokens(current.delTokens) + countWordTokens(current.insTokens);
            editCharCount += current.delTokens.join('').length + current.insTokens.join('').length;
          }
        }
        j += 1;
      }

      const shouldMerge = hasInsert && hasDelete && softEqualCount > 0 && (
        editWordCount >= MIN_MERGE_WORDS ||
        editCharCount >= MIN_MERGE_CHARS ||
        editRunCount >= MIN_MERGE_RUNS
      );

      if (shouldMerge) {
        const delTokens = [];
        const insTokens = [];
        segment.forEach((seg) => {
          if (seg.type === 'equal') {
            delTokens.push(...seg.tokens);
            insTokens.push(...seg.tokens);
          } else if (seg.type === 'insert') {
            insTokens.push(...seg.tokens);
          } else if (seg.type === 'delete') {
            delTokens.push(...seg.tokens);
          } else if (seg.type === 'replace') {
            delTokens.push(...seg.delTokens);
            insTokens.push(...seg.insTokens);
          }
        });
        merged.push({ type: 'replace', delTokens, insTokens });
      } else {
        merged.push(...segment);
      }

      i = j - 1;
    }
    return merged;
  };

  const myersEdits = (a, b, options) => {
    const n = a.length;
    const m = b.length;
    if (!n && !m) return [];
    if (!n) return b.map((value) => ({ type: 'insert', value }));
    if (!m) return a.map((value) => ({ type: 'delete', value }));

    const max = n + m;
    const maxDFromTrace = Math.max(0, Math.floor(Math.sqrt(MAX_TRACE_CELLS)) - 1);
    const maxD = Math.min(
      max,
      Math.max(0, options?.maxD ?? maxDFromTrace)
    );

    const trace = [];
    let vPrev = new Int32Array(1);
    let prefix = 0;
    while (prefix < n && prefix < m && a[prefix] === b[prefix]) prefix += 1;
    vPrev[0] = prefix;
    trace.push(vPrev);
    if (prefix >= n && prefix >= m) {
      return a.map((value) => ({ type: 'equal', value }));
    }

    const backtrack = (D) => {
      let x = n;
      let y = m;
      const edits = [];

      for (let d = D; d > 0; d -= 1) {
        const v = trace[d - 1];
        const offset = d - 1;
        const k = x - y;
        const down = k === -d || (k !== d && v[k - 1 + offset] < v[k + 1 + offset]);
        const prevK = down ? k + 1 : k - 1;
        const prevX = v[prevK + offset];
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

    for (let d = 1; d <= maxD; d += 1) {
      const offsetPrev = d - 1;
      const vNext = new Int32Array(2 * d + 1);
      const offset = d;

      for (let k = -d; k <= d; k += 2) {
        const kIdx = k + offset;
        const down = k === -d || (k !== d && vPrev[k - 1 + offsetPrev] < vPrev[k + 1 + offsetPrev]);
        let x;
        if (down) {
          x = vPrev[k + 1 + offsetPrev];
        } else {
          x = vPrev[k - 1 + offsetPrev] + 1;
        }
        let y = x - k;
        while (x < n && y < m && a[x] === b[y]) {
          x += 1;
          y += 1;
        }
        vNext[kIdx] = x;

        if (x >= n && y >= m) {
          trace.push(vNext);
          return backtrack(d);
        }
      }

      trace.push(vNext);
      vPrev = vNext;
    }

    return null;
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

  const splitWhitespace = (text) => {
    const s = String(text || '');
    if (!s) return { leading: '', core: '', trailing: '' };
    if (/^\s+$/.test(s)) return { leading: s, core: '', trailing: '' };
    const leading = (s.match(/^\s+/) || [''])[0];
    const trailing = (s.match(/\s+$/) || [''])[0];
    const core = s.slice(leading.length, s.length - trailing.length);
    return { leading, core, trailing };
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
    const delParts = splitWhitespace(delText);
    const insParts = splitWhitespace(insText);
    const leading = insParts.leading;
    const trailing = insParts.trailing;
    const delCoreText = delParts.core;
    const insCoreText = insParts.core;
    const delTrim = delCoreText.trim();
    const insTrim = insCoreText.trim();
    const singleToken = delTrim && insTrim && !/\s/.test(delTrim) && !/\s/.test(insTrim);
    const smallEnough = delTrim.length <= 42 && insTrim.length <= 42;
    const similarEnough = sharedEdgeScore(delTrim, insTrim) >= 2;

    if (!singleToken || !smallEnough || !similarEnough) {
      return `${escapeHtml(leading)}<del class="diff-del">${escapeHtml(delCoreText)}</del><ins class="diff-ins">${escapeHtml(insCoreText)}</ins>${escapeHtml(trailing)}`;
    }

    const delInner = renderCharDiff(delCoreText, insCoreText, 'del');
    const insInner = renderCharDiff(delCoreText, insCoreText, 'ins');
    return `${escapeHtml(leading)}<del class="diff-del">${delInner}</del><ins class="diff-ins">${insInner}</ins>${escapeHtml(trailing)}`;
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

  const escapeHtmlWithBreaks = (text) => escapeHtml(text).replace(/\r\n|\r|\n/g, '<br>');

  const getCopyStyle = () => ({
    insBg: insBgEl?.value || '#00FF00',
    insColor: insTextEl?.value || '#000000',
    delBg: delBgEl?.value || '#FF0000',
    delColor: delTextEl?.value || '#000000',
    delStrike: delStrikeEl?.value || '#000000'
  });

  const normalizeHexColor = (value, fallback) => {
    const s = String(value || '').trim();
    if (/^#[0-9a-f]{6}$/i.test(s)) return s.toUpperCase();
    return fallback;
  };

  const applyPreviewStyle = () => {
    const style = getCopyStyle();
    document.body.style.setProperty('--textcompare-ins-bg', normalizeHexColor(style.insBg, '#00FF00'));
    document.body.style.setProperty('--textcompare-ins-text', normalizeHexColor(style.insColor, '#000000'));
    document.body.style.setProperty('--textcompare-del-bg', normalizeHexColor(style.delBg, '#FF0000'));
    document.body.style.setProperty('--textcompare-del-text', normalizeHexColor(style.delColor, '#000000'));
    document.body.style.setProperty('--textcompare-del-strike', normalizeHexColor(style.delStrike, '#000000'));
  };

  const hexToRgb = (hex) => {
    const h = String(hex || '').replace('#', '');
    const r = parseInt(h.slice(0, 2), 16) || 0;
    const g = parseInt(h.slice(2, 4), 16) || 0;
    const b = parseInt(h.slice(4, 6), 16) || 0;
    return { r, g, b };
  };

  const escapeRtf = (text) => {
    const s = String(text || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    let out = '';
    for (let i = 0; i < s.length; i += 1) {
      const code = s.codePointAt(i);
      const ch = String.fromCodePoint(code);
      if (code > 0xFFFF) i += 1;
      if (ch === '\\') out += '\\\\';
      else if (ch === '{') out += '\\{';
      else if (ch === '}') out += '\\}';
      else if (ch === '\n') out += '\\line\n';
      else if (code <= 0x7F) out += ch;
      else if (code <= 0xFFFF) {
        const signed = code > 0x7FFF ? code - 0x10000 : code;
        out += `\\u${signed}?`;
      } else {
        const cp = code - 0x10000;
        const hi = 0xD800 + (cp >> 10);
        const lo = 0xDC00 + (cp & 0x3FF);
        const hiSigned = hi > 0x7FFF ? hi - 0x10000 : hi;
        const loSigned = lo > 0x7FFF ? lo - 0x10000 : lo;
        out += `\\u${hiSigned}?\\u${loSigned}?`;
      }
    }
    return out;
  };

  const buildClipboardFragment = (runs, style) => {
    const insBg = normalizeHexColor(style.insBg, '#00FF00');
    const insColor = normalizeHexColor(style.insColor, '#000000');
    const delBg = normalizeHexColor(style.delBg, '#FF0000');
    const delColor = normalizeHexColor(style.delColor, '#000000');
    const delStrike = normalizeHexColor(style.delStrike, '#000000');

    const insStyle = `background:${insBg};background-color:${insBg};color:${insColor};mso-highlight:${insBg};`;
    const delWrapStyle = `background:${delBg};background-color:${delBg};mso-highlight:${delBg};`;
    const delInnerStyle = `color:${delColor};text-decoration:line-through;text-decoration-color:${delStrike};mso-text-decoration:line-through;`;
    return runs.map((run) => {
      if (run.type === 'equal') return escapeHtmlWithBreaks(run.tokens.join(''));
      if (run.type === 'insert') return `<span style="${insStyle}">${escapeHtmlWithBreaks(run.tokens.join(''))}</span>`;
      if (run.type === 'delete') return `<span style="${delWrapStyle}"><s style="${delInnerStyle}">${escapeHtmlWithBreaks(run.tokens.join(''))}</s></span>`;
      if (run.type === 'replace') {
        const delText = run.delTokens.join('');
        const insText = run.insTokens.join('');
        const delParts = splitWhitespace(delText);
        const insParts = splitWhitespace(insText);
        const leading = insParts.leading;
        const trailing = insParts.trailing;
        return `${escapeHtmlWithBreaks(leading)}<span style="${delWrapStyle}"><s style="${delInnerStyle}">${escapeHtmlWithBreaks(delParts.core)}</s></span><span style="${insStyle}">${escapeHtmlWithBreaks(insParts.core)}</span>${escapeHtmlWithBreaks(trailing)}`;
      }
      return '';
    }).join('');
  };

  const buildClipboardHtml = (runs, style) => {
    const fragment = buildClipboardFragment(runs, style);
    const bodyStyle = [
      'font-family:Calibri, Arial, sans-serif',
      'font-size:11pt',
      'line-height:1.5',
      'color:#000',
      'background:#fff'
    ].join(';');
    return `<div style="${bodyStyle}"><!--StartFragment-->${fragment}<!--EndFragment--></div>`;
  };

  const buildClipboardRtf = (runs, style) => {
    const insBg = normalizeHexColor(style.insBg, '#00FF00');
    const insColor = normalizeHexColor(style.insColor, '#000000');
    const delBg = normalizeHexColor(style.delBg, '#FF0000');
    const delColor = normalizeHexColor(style.delColor, '#000000');

    const insBgRgb = hexToRgb(insBg);
    const insColorRgb = hexToRgb(insColor);
    const delBgRgb = hexToRgb(delBg);
    const delColorRgb = hexToRgb(delColor);

    const colors = [
      { r: 0, g: 0, b: 0 }, // index 1: black (fallback)
      insBgRgb,            // index 2: inserted highlight
      insColorRgb,         // index 3: inserted text color
      delBgRgb,            // index 4: deleted highlight
      delColorRgb          // index 5: deleted text color
    ];
    const colorTable = `{\n\\colortbl ;${colors.map(c => `\\red${c.r}\\green${c.g}\\blue${c.b};`).join('')}\n}\n`;

    const normalPrefix = '\\highlight0\\cf1\\strike0 ';
    const insertPrefix = '\\highlight2\\cf3\\strike0 ';
    const deletePrefix = '\\highlight4\\cf5\\strike ';

    const body = runs.map((run) => {
      if (run.type === 'equal') return escapeRtf(run.tokens.join(''));
      if (run.type === 'insert') return `${insertPrefix}${escapeRtf(run.tokens.join(''))}${normalPrefix}`;
      if (run.type === 'delete') return `${deletePrefix}${escapeRtf(run.tokens.join(''))}${normalPrefix}`;
      if (run.type === 'replace') {
        const delText = run.delTokens.join('');
        const insText = run.insTokens.join('');
        const delParts = splitWhitespace(delText);
        const insParts = splitWhitespace(insText);
        const leading = insParts.leading;
        const trailing = insParts.trailing;
        return `${escapeRtf(leading)}${deletePrefix}${escapeRtf(delParts.core)}${normalPrefix}${insertPrefix}${escapeRtf(insParts.core)}${normalPrefix}${escapeRtf(trailing)}`;
      }
      return '';
    }).join('');

    return `{\\rtf1\\ansi\\deff0\n{\\fonttbl{\\f0 Calibri;}}\n${colorTable}\\viewkind4\\uc1\\pard\\f0\\fs22 ${normalPrefix}${body}\\par\n}`;
  };

  const copyFormatted = async () => {
    if (!lastRuns || !lastRuns.length) {
      setCopyStatus('Nothing to copy yet.', 'error');
      return;
    }

    setCopyStatus('Copying…');
    const style = getCopyStyle();
    const html = buildClipboardHtml(lastRuns, style);
    const rtf = buildClipboardRtf(lastRuns, style);
    const plainText = lastRevisedText || '';

    try {
      if (navigator.clipboard && window.ClipboardItem) {
        const item = new ClipboardItem({
          'text/html': new Blob([html], { type: 'text/html' }),
          'text/plain': new Blob([plainText], { type: 'text/plain' }),
          'text/rtf': new Blob([rtf], { type: 'text/rtf' })
        });
        await navigator.clipboard.write([item]);
        setCopyStatus('Copied with formatting (Outlook-friendly).', 'success');
        return;
      }
    } catch {
      // fall through to selection-based copy
    }

    try {
      const temp = document.createElement('div');
      temp.style.position = 'fixed';
      temp.style.left = '-9999px';
      temp.style.top = '0';
      temp.style.whiteSpace = 'normal';
      temp.contentEditable = 'true';
      temp.innerHTML = html;
      document.body.appendChild(temp);

      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(temp);
      selection?.removeAllRanges();
      selection?.addRange(range);
      const handleCopy = (event) => {
        if (!event.clipboardData) return;
        event.clipboardData.setData('text/plain', plainText);
        event.clipboardData.setData('text/html', html);
        try {
          event.clipboardData.setData('text/rtf', rtf);
        } catch {
          // ignore if the browser blocks RTF
        }
        event.preventDefault();
      };
      document.addEventListener('copy', handleCopy);
      let ok = false;
      try {
        ok = document.execCommand('copy');
      } finally {
        document.removeEventListener('copy', handleCopy);
        selection?.removeAllRanges();
        temp.remove();
      }

      setCopyStatus(ok ? 'Copied with formatting.' : 'Copy failed.', ok ? 'success' : 'error');
    } catch {
      setCopyStatus('Copy failed. Try selecting the output and copying manually.', 'error');
    }
  };

  const runCompare = () => {
    setCopyStatus('');
    markSessionDirty();
    const originalInput = originalEl.value || '';
    const revisedInput = revisedEl.value || '';
    const originalHasUser = Boolean(originalInput.trim());
    const revisedHasUser = Boolean(revisedInput.trim());
    const original = (!originalHasUser && !revisedHasUser) ? (originalEl.placeholder || '') : originalInput;
    const revised = (!originalHasUser && !revisedHasUser) ? (revisedEl.placeholder || '') : revisedInput;

    if ((!originalHasUser && revisedHasUser) || (originalHasUser && !revisedHasUser)) {
      summaryEl.textContent = 'Paste both versions to compare.';
      setEmpty('Paste text in both boxes, then click Compare.');
      lastRuns = null;
      lastRevisedText = '';
      markSessionDirty();
      return;
    }

    if (!original.trim() && !revised.trim()) {
      summaryEl.textContent = 'Paste text above and click Compare.';
      setEmpty('Waiting for input.');
      lastRuns = null;
      lastRevisedText = '';
      markSessionDirty();
      return;
    }

    if (original.length + revised.length > MAX_CHARS) {
      summaryEl.textContent = 'Text is too large to compare in-browser. Please compare smaller sections.';
      setEmpty('Input too large.');
      lastRuns = null;
      lastRevisedText = '';
      markSessionDirty();
      return;
    }

    summaryEl.textContent = 'Comparing…';
    setEmpty('Comparing…');

    requestAnimationFrame(() => {
      const aTokens = tokenize(original);
      const bTokens = tokenize(revised);
      if (aTokens.length + bTokens.length > MAX_TOKENS) {
        summaryEl.textContent = 'Text is too large to compare in-browser. Please compare smaller sections.';
        setEmpty('Input too large.');
        lastRuns = null;
        lastRevisedText = '';
        markSessionDirty();
        return;
      }

      const edits = myersEdits(aTokens, bTokens);
      let runs;
      if (!edits) {
        runs = [{ type: 'replace', delTokens: aTokens, insTokens: bTokens }];
      } else {
        const rawRuns = normalizeRuns(groupRuns(edits));
        const settings = getCompareSettings();
        const countsA = buildWordCounts(aTokens);
        const countsB = buildWordCounts(bTokens);
        runs = coalesceRuns(rawRuns, countsA, countsB, settings);
      }
      lastRuns = runs;
      lastRevisedText = revised;
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
        markSessionDirty();
        return;
      }
      const parts = [];
      if (insertedWords) parts.push(`${insertedWords.toLocaleString('en-US')} inserted words`);
      if (deletedWords) parts.push(`${deletedWords.toLocaleString('en-US')} deleted words`);
      if (replacements) parts.push(`${replacements.toLocaleString('en-US')} replacements`);
      summaryEl.textContent = `Changes: ${parts.join(' · ')}.`;
      markSessionDirty();
    });
  };

  [insBgEl, insTextEl, delBgEl, delTextEl, delStrikeEl].forEach((el) => {
    el?.addEventListener('input', applyPreviewStyle);
  });
  applyPreviewStyle();

  [mergeWordsEl, mergeCharsEl, mergeRunsEl, softWordsEl, softWordLenEl, commonMinEl].forEach((el) => {
    el?.addEventListener('input', () => {
      if (!lastRuns) return;
      runCompare();
    });
  });

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runCompare();
  });

  clearBtn?.addEventListener('click', () => {
    originalEl.value = '';
    revisedEl.value = '';
    summaryEl.textContent = 'Paste text above and click Compare.';
    setEmpty('Waiting for input.');
    lastRuns = null;
    lastRevisedText = '';
    setCopyStatus('');
    markSessionDirty();
    originalEl.focus();
  });

  swapBtn?.addEventListener('click', () => {
    const a = originalEl.value;
    originalEl.value = revisedEl.value;
    revisedEl.value = a;
    runCompare();
  });

  copyBtn?.addEventListener('click', copyFormatted);
  const MAX_SAVED_OUTPUT_HTML_CHARS = 120_000;
  const MAX_SAVED_OUTPUT_TEXT_CHARS = 120_000;

  const clampText = (value, maxChars) => {
    const text = String(value || '');
    if (text.length <= maxChars) return { text, truncated: false };
    return { text: text.slice(0, maxChars), truncated: true };
  };

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const summary = String(summaryEl?.textContent || '').trim();
    payload.outputSummary = summary;

    const html = String(outputEl?.innerHTML || '').trim();
    if (html && html.length <= MAX_SAVED_OUTPUT_HTML_CHARS) {
      payload.output = { kind: 'html', html, summary };
      return;
    }

    const content = String(outputEl?.textContent || '').trim();
    const { text, truncated } = clampText(content, MAX_SAVED_OUTPUT_TEXT_CHARS);
    payload.output = { kind: 'text', text, summary, truncated };
  });

  document.addEventListener('tools:session-applied', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const snapshot = detail?.snapshot;
    const output = snapshot?.output;
    if (!output || typeof output !== 'object') return;

    const summary = String(output.summary || '').trim();
    if (summary) summaryEl.textContent = summary;

    const kind = String(output.kind || '').trim();
    if (kind === 'html') {
      outputEl.innerHTML = String(output.html || '').trim() || '<p class="textcompare-empty">No output.</p>';
      return;
    }

    if (kind === 'text') {
      const raw = String(output.text || '').trim();
      outputEl.innerHTML = raw
        ? `<pre>${escapeHtml(raw)}</pre>`
        : '<p class="textcompare-empty">No output.</p>';
    }
  });
})();
