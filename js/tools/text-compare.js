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
  const originalPasteBtn = $('#textcompare-original-paste');
  const revisedPasteBtn = $('#textcompare-revised-paste');
  const originalImportBtn = $('#textcompare-original-import');
  const revisedImportBtn = $('#textcompare-revised-import');
  const originalFileInput = $('#textcompare-original-file');
  const revisedFileInput = $('#textcompare-revised-file');
  const originalStatusEl = $('#textcompare-original-status');
  const revisedStatusEl = $('#textcompare-revised-status');
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
  const MAX_IMPORT_BYTES = 24 * 1024 * 1024;
  const AUTO_COMPARE_BASE_SETTINGS = {
    mergeMinWords: 8,
    mergeMinChars: 40,
    mergeMinRuns: 4,
    softEqualMaxWords: 2,
    softEqualMaxWordLength: 2,
    commonWordMinCount: 2
  };
  const PDF_WORKER_PATH = '/js/vendor/pdfjs/pdf.worker.min.js';
  let lastRuns = null;
  let lastRevisedText = '';
  const fields = {
    original: {
      textarea: originalEl,
      pasteBtn: originalPasteBtn,
      importBtn: originalImportBtn,
      fileInput: originalFileInput,
      statusEl: originalStatusEl
    },
    revised: {
      textarea: revisedEl,
      pasteBtn: revisedPasteBtn,
      importBtn: revisedImportBtn,
      fileInput: revisedFileInput,
      statusEl: revisedStatusEl
    }
  };

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

  const clampNumber = (value, min, max) => Math.min(max, Math.max(min, value));

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

  const collectLowerWords = (text) => matchAll(WORD_RE, String(text || '')).map((word) => word.toLowerCase());

  const wordOverlapScore = (aWords, bWords) => {
    if (!aWords.length || !bWords.length) return 0;
    const counts = new Map();
    aWords.forEach((word) => counts.set(word, (counts.get(word) || 0) + 1));
    let overlap = 0;
    bWords.forEach((word) => {
      const available = counts.get(word) || 0;
      if (available <= 0) return;
      overlap += 1;
      counts.set(word, available - 1);
    });
    return (2 * overlap) / (aWords.length + bWords.length);
  };

  const textSimilarityScore = (leftText, rightText) => {
    const left = String(leftText || '').trim();
    const right = String(rightText || '').trim();
    if (!left && !right) return 1;
    if (!left || !right) return 0;
    if (left === right) return 1;

    const aWords = collectLowerWords(left);
    const bWords = collectLowerWords(right);
    const lexical = wordOverlapScore(aWords, bWords);
    const maxLength = Math.max(left.length, right.length) || 1;
    const lengthRatio = Math.min(left.length, right.length) / maxLength;
    const edgeRatio = sharedEdgeScore(left, right) / maxLength;
    return (lexical * 0.7) + (lengthRatio * 0.2) + (edgeRatio * 0.1);
  };

  const shouldPairAsReplace = (delTokens, insTokens) => {
    const delText = (delTokens || []).join('');
    const insText = (insTokens || []).join('');
    if (!delText && !insText) return false;

    const delTrim = delText.trim();
    const insTrim = insText.trim();
    if (!delTrim || !insTrim) return false;
    if (delTrim === insTrim) return true;

    const delWords = countWords(delTrim);
    const insWords = countWords(insTrim);
    const longEnough = delWords >= 3 || insWords >= 3 || delTrim.length >= 28 || insTrim.length >= 28;
    const score = textSimilarityScore(delTrim, insTrim);
    if (!longEnough) return score >= 0.18;
    return score >= 0.3;
  };

  const getAutoCompareSettings = (aTokens, bTokens) => {
    const totalTokenCount = aTokens.length + bTokens.length;
    const totalWordCount = countWordTokens(aTokens) + countWordTokens(bTokens);
    const meanWords = totalWordCount / 2;
    const wordDensity = totalTokenCount ? totalWordCount / totalTokenCount : 0;
    const sizeFactor = clampNumber(meanWords / 1_800, 0, 1);
    const punctuationHeavy = wordDensity < 0.42;
    const denseLargeDoc = meanWords > 2_800;

    return {
      mergeMinWords: clampNumber(
        Math.round(AUTO_COMPARE_BASE_SETTINGS.mergeMinWords + (sizeFactor * 8) + (punctuationHeavy ? 2 : 0)),
        4,
        24
      ),
      mergeMinChars: clampNumber(
        Math.round(AUTO_COMPARE_BASE_SETTINGS.mergeMinChars + (sizeFactor * 72) + (punctuationHeavy ? 10 : 0)),
        20,
        160
      ),
      mergeMinRuns: clampNumber(
        Math.round(AUTO_COMPARE_BASE_SETTINGS.mergeMinRuns + (sizeFactor * 4)),
        2,
        10
      ),
      softEqualMaxWords: denseLargeDoc ? 1 : AUTO_COMPARE_BASE_SETTINGS.softEqualMaxWords,
      softEqualMaxWordLength: meanWords > 2_000 ? 1 : AUTO_COMPARE_BASE_SETTINGS.softEqualMaxWordLength,
      commonWordMinCount: meanWords > 1_200 ? 3 : AUTO_COMPARE_BASE_SETTINGS.commonWordMinCount
    };
  };

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
    const pushRun = (run) => {
      if (!run) return;
      const last = out[out.length - 1];
      if (!last || last.type !== run.type) {
        out.push(run);
        return;
      }
      if (run.type === 'replace') {
        last.delTokens.push(...run.delTokens);
        last.insTokens.push(...run.insTokens);
        return;
      }
      last.tokens.push(...run.tokens);
    };

    for (let i = 0; i < runs.length; i += 1) {
      const run = runs[i];
      const next = runs[i + 1];
      if (next && ((run.type === 'delete' && next.type === 'insert') || (run.type === 'insert' && next.type === 'delete'))) {
        const delTokens = run.type === 'delete' ? run.tokens : next.tokens;
        const insTokens = run.type === 'insert' ? run.tokens : next.tokens;
        if (shouldPairAsReplace(delTokens, insTokens)) {
          pushRun({ type: 'replace', delTokens: [...delTokens], insTokens: [...insTokens] });
          i += 1;
          continue;
        }
      }
      if (run.type === 'replace') {
        pushRun({ type: 'replace', delTokens: [...run.delTokens], insTokens: [...run.insTokens] });
        continue;
      }
      pushRun({ type: run.type, tokens: [...run.tokens] });
    }
    return out;
  };

  const cloneRun = (run) => {
    if (!run || !run.type) return null;
    if (run.type === 'replace') {
      return {
        type: 'replace',
        delTokens: [...(run.delTokens || [])],
        insTokens: [...(run.insTokens || [])],
        moveId: run.moveId,
        moveRole: run.moveRole
      };
    }
    return {
      type: run.type,
      tokens: [...(run.tokens || [])],
      moveId: run.moveId,
      moveRole: run.moveRole
    };
  };

  const pushMergedRun = (target, run) => {
    const next = cloneRun(run);
    if (!next) return;

    if (next.type === 'replace') {
      if (!next.delTokens.length && !next.insTokens.length) return;
    } else if (!next.tokens.length) {
      return;
    }

    const last = target[target.length - 1];
    const canMerge = Boolean(last) &&
      last.type === next.type &&
      !last.moveId &&
      !next.moveId;

    if (!canMerge) {
      target.push(next);
      return;
    }

    if (next.type === 'replace') {
      last.delTokens.push(...next.delTokens);
      last.insTokens.push(...next.insTokens);
      return;
    }
    last.tokens.push(...next.tokens);
  };

  const appendRuns = (target, runs) => {
    (runs || []).forEach((run) => pushMergedRun(target, run));
  };

  const makeSingleRun = (type, text) => {
    const tokens = tokenize(text);
    if (!tokens.length) return [];
    return [{ type, tokens }];
  };

  const tokenDiffRuns = (leftText, rightText) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    if (!left && !right) return [];
    if (!left) return makeSingleRun('insert', right);
    if (!right) return makeSingleRun('delete', left);

    const aTokens = tokenize(left);
    const bTokens = tokenize(right);
    const edits = myersEdits(aTokens, bTokens);
    if (!edits) return [{ type: 'replace', delTokens: aTokens, insTokens: bTokens }];
    return normalizeRuns(groupRuns(edits));
  };

  const splitParagraphBlocks = (text) => {
    const source = String(text || '');
    if (!source) return [];
    const parts = source.split(/(\n{2,})/);
    const blocks = [];
    for (let i = 0; i < parts.length; i += 2) {
      const body = parts[i] || '';
      const separator = parts[i + 1] || '';
      const block = `${body}${separator}`;
      if (block) blocks.push(block);
    }
    return blocks.length ? blocks : [source];
  };

  const splitSentenceBlocks = (text) => {
    const source = String(text || '');
    if (!source) return [];
    const chunks = source.match(/[^.!?\n]+(?:[.!?]+)?(?:\s+|$)|\n+/g);
    if (!chunks || !chunks.length) return [source];
    return chunks.filter(Boolean);
  };

  const createAnchorSignature = (text) => String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const buildTextUnits = (text, splitter) => (splitter(text) || []).map((chunk) => {
    const value = String(chunk || '');
    return {
      text: value,
      signature: createAnchorSignature(value),
      words: countWords(value),
      chars: value.trim().length
    };
  });

  const joinUnitRange = (units, from, to) => units.slice(from, to).map((unit) => unit.text).join('');

  const selectIncreasingPairs = (pairs, maxPairs) => {
    if (!pairs.length) return [];
    if (pairs.length > maxPairs) {
      const greedy = [];
      let lastB = -1;
      pairs.forEach((pair) => {
        if (pair.bIndex <= lastB) return;
        greedy.push(pair);
        lastB = pair.bIndex;
      });
      return greedy;
    }

    const count = pairs.length;
    const best = new Array(count).fill(0);
    const prev = new Array(count).fill(-1);

    for (let i = 0; i < count; i += 1) {
      best[i] = pairs[i].score;
      for (let j = 0; j < i; j += 1) {
        if (pairs[j].bIndex >= pairs[i].bIndex) continue;
        const candidate = best[j] + pairs[i].score;
        if (candidate > best[i]) {
          best[i] = candidate;
          prev[i] = j;
        }
      }
    }

    let bestIndex = 0;
    for (let i = 1; i < count; i += 1) {
      if (best[i] > best[bestIndex]) bestIndex = i;
    }

    const selected = [];
    let cursor = bestIndex;
    while (cursor !== -1) {
      selected.push(pairs[cursor]);
      cursor = prev[cursor];
    }
    selected.reverse();
    return selected;
  };

  const findUnitAnchors = (aUnits, bUnits, options) => {
    if (!aUnits.length || !bUnits.length) return [];
    const minWords = Math.max(1, options?.minWords || 1);
    const minChars = Math.max(1, options?.minChars || 1);
    const maxPairs = Math.max(80, options?.maxPairs || 420);

    const indexUnique = (units) => {
      const counts = new Map();
      units.forEach((unit) => {
        if (!unit.signature) return;
        if (unit.words < minWords && unit.chars < minChars) return;
        counts.set(unit.signature, (counts.get(unit.signature) || 0) + 1);
      });

      const unique = new Map();
      units.forEach((unit, index) => {
        if (!unit.signature) return;
        if (unit.words < minWords && unit.chars < minChars) return;
        if ((counts.get(unit.signature) || 0) !== 1) return;
        unique.set(unit.signature, index);
      });
      return unique;
    };

    const aUnique = indexUnique(aUnits);
    const bUnique = indexUnique(bUnits);
    if (!aUnique.size || !bUnique.size) return [];

    const candidates = [];
    aUnique.forEach((aIndex, signature) => {
      const bIndex = bUnique.get(signature);
      if (typeof bIndex !== 'number') return;
      const unit = aUnits[aIndex];
      const score = Math.max(1, (unit.words * 2) + Math.floor(unit.chars / 30));
      candidates.push({ aIndex, bIndex, score });
    });

    candidates.sort((left, right) => {
      if (left.aIndex !== right.aIndex) return left.aIndex - right.aIndex;
      return left.bIndex - right.bIndex;
    });

    return selectIncreasingPairs(candidates, maxPairs);
  };

  const diffWithAnchors = (aUnits, bUnits, options) => {
    const fallback = options?.fallback || tokenDiffRuns;
    const anchors = findUnitAnchors(aUnits, bUnits, options);
    if (!anchors.length) {
      return fallback(joinUnitRange(aUnits, 0, aUnits.length), joinUnitRange(bUnits, 0, bUnits.length));
    }

    const runs = [];
    let aCursor = 0;
    let bCursor = 0;

    anchors.forEach((anchor) => {
      const leftGap = joinUnitRange(aUnits, aCursor, anchor.aIndex);
      const rightGap = joinUnitRange(bUnits, bCursor, anchor.bIndex);
      appendRuns(runs, fallback(leftGap, rightGap));

      const aAnchorText = aUnits[anchor.aIndex]?.text || '';
      const bAnchorText = bUnits[anchor.bIndex]?.text || '';
      if (aAnchorText === bAnchorText) {
        appendRuns(runs, makeSingleRun('equal', aAnchorText));
      } else {
        appendRuns(runs, tokenDiffRuns(aAnchorText, bAnchorText));
      }

      aCursor = anchor.aIndex + 1;
      bCursor = anchor.bIndex + 1;
    });

    const leftTail = joinUnitRange(aUnits, aCursor, aUnits.length);
    const rightTail = joinUnitRange(bUnits, bCursor, bUnits.length);
    appendRuns(runs, fallback(leftTail, rightTail));
    return runs;
  };

  const diffBySentenceAnchors = (leftText, rightText) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    if (!left && !right) return [];

    const aUnits = buildTextUnits(left, splitSentenceBlocks);
    const bUnits = buildTextUnits(right, splitSentenceBlocks);
    if (aUnits.length < 3 && bUnits.length < 3) {
      return tokenDiffRuns(left, right);
    }

    return diffWithAnchors(aUnits, bUnits, {
      minWords: 2,
      minChars: 22,
      maxPairs: 520,
      fallback: tokenDiffRuns
    });
  };

  const diffByParagraphAnchors = (leftText, rightText) => {
    const left = String(leftText || '');
    const right = String(rightText || '');
    if (!left && !right) return [];

    const aUnits = buildTextUnits(left, splitParagraphBlocks);
    const bUnits = buildTextUnits(right, splitParagraphBlocks);
    if (aUnits.length < 2 && bUnits.length < 2) {
      return diffBySentenceAnchors(left, right);
    }

    return diffWithAnchors(aUnits, bUnits, {
      minWords: 4,
      minChars: 36,
      maxPairs: 260,
      fallback: diffBySentenceAnchors
    });
  };

  const buildBestEffortRuns = (leftText, rightText) => diffByParagraphAnchors(leftText, rightText);

  const normalizeMoveSignature = (text) => String(text || '')
    .replace(/\s+/g, ' ')
    .trim();

  const markMovedRuns = (runs) => {
    const insertBySignature = new Map();
    const usedInsertIndexes = new Set();

    runs.forEach((run, index) => {
      if (!run || run.type !== 'insert') return;
      const text = normalizeMoveSignature(run.tokens.join(''));
      if (!text) return;
      const words = countWords(text);
      if (words < 8 && text.length < 60) return;
      const existing = insertBySignature.get(text) || [];
      existing.push(index);
      insertBySignature.set(text, existing);
    });

    let moveCount = 0;
    let moveId = 1;

    runs.forEach((run, index) => {
      if (!run || run.type !== 'delete') return;
      const text = normalizeMoveSignature(run.tokens.join(''));
      if (!text) return;
      const words = countWords(text);
      if (words < 8 && text.length < 60) return;
      const candidates = insertBySignature.get(text);
      if (!candidates || !candidates.length) return;

      let insertIndex = -1;
      for (let i = 0; i < candidates.length; i += 1) {
        const candidateIndex = candidates[i];
        if (usedInsertIndexes.has(candidateIndex)) continue;
        if (Math.abs(candidateIndex - index) <= 1) continue;
        insertIndex = candidateIndex;
        break;
      }
      if (insertIndex < 0) return;

      usedInsertIndexes.add(insertIndex);
      run.moveId = moveId;
      run.moveRole = 'from';
      runs[insertIndex].moveId = moveId;
      runs[insertIndex].moveRole = 'to';
      moveCount += 1;
      moveId += 1;
    });

    return moveCount;
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

  const buildMoveAttrs = (run) => {
    if (!run?.moveId) return { className: '', attrs: '' };
    const role = run.moveRole === 'from' ? 'from' : 'to';
    return {
      className: ' diff-move',
      attrs: ` data-move-role="${role}" data-move-id="${run.moveId}"`
    };
  };

  const renderOutput = (runs) => runs.map((run) => {
    if (run.type === 'equal') return escapeHtml(run.tokens.join(''));
    if (run.type === 'insert') {
      const move = buildMoveAttrs(run);
      return `<ins class="diff-ins${move.className}"${move.attrs}>${escapeHtml(run.tokens.join(''))}</ins>`;
    }
    if (run.type === 'delete') {
      const move = buildMoveAttrs(run);
      return `<del class="diff-del${move.className}"${move.attrs}>${escapeHtml(run.tokens.join(''))}</del>`;
    }
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

  const setFieldStatus = (fieldKey, message, tone) => {
    const field = fields[fieldKey];
    if (!field?.statusEl) return;
    field.statusEl.textContent = String(message || '');
    field.statusEl.dataset.tone = tone || '';
  };

  const setFieldBusy = (fieldKey, busy) => {
    const field = fields[fieldKey];
    if (!field) return;
    if (field.pasteBtn) field.pasteBtn.disabled = Boolean(busy);
    if (field.importBtn) field.importBtn.disabled = Boolean(busy);
    if (field.fileInput) field.fileInput.disabled = Boolean(busy);
  };

  const clearFieldStatuses = () => {
    setFieldStatus('original', '', '');
    setFieldStatus('revised', '', '');
  };

  const normalizeInputText = (text) => String(text || '')
    .replace(/\u0000/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/\u00A0/g, ' ');

  const normalizeImportedText = (text) => normalizeInputText(text)
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{4,}/g, '\n\n\n')
    .trim();

  const applyTextToField = (fieldKey, text) => {
    const field = fields[fieldKey];
    if (!field?.textarea) return;
    field.textarea.value = String(text || '');
    field.textarea.dispatchEvent(new Event('input', { bubbles: true }));
    field.textarea.dispatchEvent(new Event('change', { bubbles: true }));
    markSessionDirty();
    field.textarea.focus();
  };

  const decodeXmlEntities = (text) => String(text || '')
    .replace(/&#x([0-9a-f]+);/gi, (_match, hex) => {
      const code = Number.parseInt(hex, 16);
      if (Number.isNaN(code)) return '';
      try {
        return String.fromCodePoint(code);
      } catch {
        return '';
      }
    })
    .replace(/&#([0-9]+);/g, (_match, dec) => {
      const code = Number.parseInt(dec, 10);
      if (Number.isNaN(code)) return '';
      try {
        return String.fromCodePoint(code);
      } catch {
        return '';
      }
    })
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');

  const extractHtmlText = (html) => {
    const raw = String(html || '');
    const withBreakHints = raw
      .replace(/<\s*br\s*\/?>/gi, '\n')
      .replace(/<\/(p|div|li|tr|h[1-6])>/gi, '\n')
      .replace(/<li\b[^>]*>/gi, '- ');
    const doc = new DOMParser().parseFromString(withBreakHints, 'text/html');
    doc.querySelectorAll('script, style, noscript').forEach((node) => node.remove());
    return doc.body?.textContent || '';
  };

  const extractRtfText = (rtf) => String(rtf || '')
    .replace(/\\par[d]?/gi, '\n')
    .replace(/\\line\b/gi, '\n')
    .replace(/\\tab\b/gi, '\t')
    .replace(/\\'([0-9a-f]{2})/gi, (_match, hex) => String.fromCharCode(Number.parseInt(hex, 16) || 32))
    .replace(/\\u(-?\d+)\??/g, (_match, num) => {
      let code = Number.parseInt(num, 10);
      if (Number.isNaN(code)) return '';
      if (code < 0) code += 65536;
      try {
        return String.fromCharCode(code);
      } catch {
        return '';
      }
    })
    .replace(/\\[a-z]+-?\d* ?/gi, '')
    .replace(/[{}]/g, '');

  const extractDocxXmlText = (xmlText) => {
    const text = String(xmlText || '')
      .replace(/<w:tab[^>]*\/>/gi, '\t')
      .replace(/<w:(br|cr)[^>]*\/>/gi, '\n')
      .replace(/<\/w:tc>/gi, '\t')
      .replace(/<\/w:(p|tr)>/gi, '\n')
      .replace(/<[^>]+>/g, '');
    return decodeXmlEntities(text);
  };

  const extractLegacyDocText = (bytes) => {
    const clean = (text) => String(text || '')
      .replace(/[^ -~\n\r\t]/g, ' ')
      .replace(/[ \t]{2,}/g, ' ')
      .replace(/\n{4,}/g, '\n\n\n');

    const latin = clean(new TextDecoder('latin1').decode(bytes));
    const utf16 = clean(new TextDecoder('utf-16le').decode(bytes));
    return utf16.length > latin.length ? utf16 : latin;
  };

  const parseDocxFile = async (file) => {
    const api = window.fflate;
    if (!api || typeof api.unzipSync !== 'function') {
      throw new Error('DOCX import is unavailable: zip parser failed to load.');
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    const files = api.unzipSync(bytes);
    const decoder = new TextDecoder('utf-8');
    const xmlPaths = Object.keys(files)
      .filter((name) => /^word\/(document|header\d+|footer\d+|footnotes|endnotes)\.xml$/i.test(name))
      .sort((a, b) => a.localeCompare(b));

    if (!xmlPaths.length) {
      throw new Error('Unable to find readable text in this DOCX file.');
    }

    const chunks = [];
    xmlPaths.forEach((path) => {
      try {
        const xml = decoder.decode(files[path]);
        const text = normalizeImportedText(extractDocxXmlText(xml));
        if (text) chunks.push(text);
      } catch {
        // Skip malformed sections and keep best-effort extraction.
      }
    });
    return chunks.join('\n\n');
  };

  const parsePdfFile = async (file) => {
    const pdfjs = window.pdfjsLib;
    if (!pdfjs || typeof pdfjs.getDocument !== 'function') {
      throw new Error('PDF import is unavailable: parser failed to load.');
    }
    if (pdfjs.GlobalWorkerOptions && !pdfjs.GlobalWorkerOptions.workerSrc) {
      pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER_PATH;
    }

    const loadingTask = pdfjs.getDocument({ data: await file.arrayBuffer() });
    let pdf = null;

    try {
      pdf = await loadingTask.promise;
      const pages = [];
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const textContent = await page.getTextContent();
        let pageText = '';
        textContent.items.forEach((item) => {
          const value = String(item?.str || '');
          if (!value) return;
          pageText += value;
          pageText += item?.hasEOL ? '\n' : ' ';
        });
        const normalized = normalizeImportedText(pageText);
        if (normalized) pages.push(normalized);
      }
      return pages.join('\n\n');
    } finally {
      try {
        await loadingTask.destroy();
      } catch {}
      try {
        pdf?.cleanup?.();
      } catch {}
    }
  };

  const detectImportType = (file) => {
    const name = String(file?.name || '').toLowerCase();
    const type = String(file?.type || '').toLowerCase();

    if (name.endsWith('.pdf') || type.includes('pdf')) return 'pdf';
    if (name.endsWith('.docx') || type.includes('officedocument.wordprocessingml.document')) return 'docx';
    if (name.endsWith('.doc') || type === 'application/msword') return 'doc';
    if (name.endsWith('.rtf') || type.includes('rtf')) return 'rtf';
    if (name.endsWith('.html') || name.endsWith('.htm') || type.includes('html') || type.includes('xml')) return 'html';
    return 'text';
  };

  const parseImportedFile = async (file) => {
    const importType = detectImportType(file);
    if (importType === 'pdf') {
      return { text: await parsePdfFile(file), warning: '' };
    }
    if (importType === 'docx') {
      return { text: await parseDocxFile(file), warning: '' };
    }
    if (importType === 'doc') {
      const bytes = new Uint8Array(await file.arrayBuffer());
      return {
        text: extractLegacyDocText(bytes),
        warning: 'Legacy .doc extraction is best-effort. Convert to .docx for highest fidelity.'
      };
    }
    if (importType === 'rtf') {
      return { text: extractRtfText(await file.text()), warning: '' };
    }
    if (importType === 'html') {
      return { text: extractHtmlText(await file.text()), warning: '' };
    }
    return { text: await file.text(), warning: '' };
  };

  const pasteIntoField = async (fieldKey) => {
    const field = fields[fieldKey];
    if (!field?.textarea) return;

    setFieldBusy(fieldKey, true);
    setFieldStatus(fieldKey, 'Reading clipboard…', 'info');

    try {
      if (!navigator.clipboard || typeof navigator.clipboard.readText !== 'function') {
        throw new Error('Clipboard read is unavailable.');
      }
      const clipboardText = normalizeInputText(await navigator.clipboard.readText());
      if (!clipboardText.trim()) {
        setFieldStatus(fieldKey, 'Clipboard is empty.', 'error');
        return;
      }
      applyTextToField(fieldKey, clipboardText);
      setFieldStatus(fieldKey, `Pasted ${clipboardText.length.toLocaleString('en-US')} characters.`, 'success');
    } catch {
      setFieldStatus(fieldKey, 'Clipboard access blocked. Use Ctrl/Cmd+V in the text box.', 'error');
    } finally {
      setFieldBusy(fieldKey, false);
    }
  };

  const importIntoField = async (fieldKey) => {
    const field = fields[fieldKey];
    const file = field?.fileInput?.files?.[0];
    if (!field?.textarea || !file) return;

    setFieldBusy(fieldKey, true);
    setFieldStatus(fieldKey, `Importing ${file.name}…`, 'info');

    try {
      if (file.size > MAX_IMPORT_BYTES) {
        const maxMb = Math.round(MAX_IMPORT_BYTES / (1024 * 1024));
        throw new Error(`File is too large. Max supported size is ${maxMb} MB.`);
      }

      const parsed = await parseImportedFile(file);
      const importedText = normalizeImportedText(parsed.text);
      if (!importedText.trim()) {
        throw new Error('No readable text was found in this file.');
      }

      applyTextToField(fieldKey, importedText);
      const summary = `${file.name} imported (${importedText.length.toLocaleString('en-US')} characters).`;
      const statusText = parsed.warning ? `${summary} ${parsed.warning}` : summary;
      setFieldStatus(fieldKey, statusText, parsed.warning ? 'info' : 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to import this file.';
      setFieldStatus(fieldKey, message, 'error');
    } finally {
      if (field.fileInput) field.fileInput.value = '';
      setFieldBusy(fieldKey, false);
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
      summaryEl.textContent = 'Click Compare to run the built-in example, or paste your own drafts.';
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

      const settings = getAutoCompareSettings(aTokens, bTokens);
      const countsA = buildWordCounts(aTokens);
      const countsB = buildWordCounts(bTokens);
      const initialRuns = buildBestEffortRuns(original, revised);
      let runs = normalizeRuns(initialRuns);
      runs = coalesceRuns(runs, countsA, countsB, settings);
      const movedBlocks = markMovedRuns(runs);
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
      if (movedBlocks) parts.push(`${movedBlocks.toLocaleString('en-US')} moved blocks`);
      summaryEl.textContent = `Changes: ${parts.join(' · ')}.`;
      markSessionDirty();
    });
  };

  [insBgEl, insTextEl, delBgEl, delTextEl, delStrikeEl].forEach((el) => {
    el?.addEventListener('input', applyPreviewStyle);
  });
  applyPreviewStyle();

  originalPasteBtn?.addEventListener('click', () => {
    void pasteIntoField('original');
  });
  revisedPasteBtn?.addEventListener('click', () => {
    void pasteIntoField('revised');
  });
  originalImportBtn?.addEventListener('click', () => {
    originalFileInput?.click();
  });
  revisedImportBtn?.addEventListener('click', () => {
    revisedFileInput?.click();
  });
  originalFileInput?.addEventListener('change', () => {
    void importIntoField('original');
  });
  revisedFileInput?.addEventListener('change', () => {
    void importIntoField('revised');
  });

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    runCompare();
  });

  clearBtn?.addEventListener('click', () => {
    originalEl.value = '';
    revisedEl.value = '';
    summaryEl.textContent = 'Click Compare to run the built-in example, or paste your own drafts.';
    setEmpty('Waiting for input.');
    lastRuns = null;
    lastRevisedText = '';
    setCopyStatus('');
    clearFieldStatuses();
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
