const fs = require('fs');

const core = require('../js/tools/text-compare-core.js');

module.exports = function runTextCompareCoreTests({ assert }) {
  assert(core && typeof core.compareText === 'function', 'text compare core should export compareText');
  assert(core && typeof core.detectComparisonMode === 'function', 'text compare core should export detectComparisonMode');
  assert(core && typeof core.diffChars === 'function', 'text compare core should export diffChars');
  assert(core && typeof core.tokenize === 'function', 'text compare core should export tokenize');

  {
    const result = core.compareText({
      leftText: 'Product analytics should be easy to use.',
      rightText: 'Product analytics should be easy to use.',
      modeOverride: 'auto',
      sourceHints: { leftKind: 'text', rightKind: 'text' }
    });
    assert(result.inferredMode === core.MODES.DOCUMENT, 'plain text should default to document mode');
    assert(result.counts.hasChanges === false, 'identical plain text should have no changes');
    assert(result.runs.every((run) => run.type === 'equal'), 'identical plain text should only emit equal runs');
  }

  {
    const result = core.compareText({
      leftText: 'Colour choices matter.',
      rightText: 'Color choices matter.',
      modeOverride: 'prose',
      sourceHints: { leftKind: 'text', rightKind: 'text' }
    });
    assert(result.inferredMode === core.MODES.DOCUMENT, 'legacy prose mode should map to document mode');
    assert(result.counts.replacements >= 1, 'small word edits should be recognized as replacements');
  }

  {
    const left = [
      'Editorial summary stays in the same place.',
      '',
      'The release note stayed the same while Section A reviewed onboarding retention, instrumentation coverage, ownership routing, and support handoff.',
      '',
      'The release note stayed the same while Section B reviewed activation cohorts, release readiness, experiment cleanup, dashboard debt, and support coverage.'
    ].join('\n');
    const right = [
      'Editorial summary stays in the same place.',
      '',
      'The release note stayed the same while Section B reviewed activation cohorts, release readiness, experiment cleanup, dashboard debt, and support coverage.',
      '',
      'The release note stayed the same while Section A reviewed onboarding retention, instrumentation coverage, ownership routing, and support handoff.'
    ].join('\n');
    const result = core.compareText({
      leftText: left,
      rightText: right,
      modeOverride: 'prose',
      sourceHints: { leftKind: 'text', rightKind: 'text' }
    });
    assert(result.counts.movedBlocks >= 1, 'reordered document blocks should be flagged as moved');
  }

  {
    const left = [
      'Editorial summary stays in the same place.',
      '',
      'The support handoff playbook now covers triage notes, escalation owners, response windows, and weekly close-out reviews.',
      '',
      'The activation section tracks first-session completion, follow-through within seven days, and checklist drop-off by cohort.'
    ].join('\n');
    const right = [
      'Editorial summary stays in the same place.',
      '',
      'The activation section tracks first-session completion, follow-through within seven days, and checklist drop-off by cohort.',
      '',
      'The support handoff playbook now covers triage notes, escalation owners, response windows, and weekly close-out reviews with partner handoff notes.'
    ].join('\n');
    const result = core.compareText({
      leftText: left,
      rightText: right,
      modeOverride: 'document',
      sourceHints: { leftKind: 'text', rightKind: 'text' }
    });
    assert(result.counts.movedBlocks >= 1, 'moved paragraphs with light edits should still be marked as moves');
  }

  {
    const result = core.compareText({
      leftText: 'Launch review stays on Monday. Ownership routing updates happen next. Support handoff follows on Friday. Nothing else changed.',
      rightText: 'Launch review stays on Monday. Support handoff follows on Friday. Ownership routing updates happen next. Nothing else changed.',
      modeOverride: 'document',
      sourceHints: { leftKind: 'text', rightKind: 'text' }
    });
    assert(result.counts.movedBlocks >= 1, 'reordered sentence groups should be marked as moves');
  }

  {
    const result = core.compareText({
      leftText: '- Activation review covers day-one completion.\n- Ownership routing is sent after every incident.\n- Support handoff closes with a Friday summary.\n',
      rightText: '- Ownership routing is sent after every incident.\n- Support handoff closes with a Friday summary and escalation note.\n- Activation review covers day-one completion.\n',
      modeOverride: 'document',
      sourceHints: { leftKind: 'text', rightKind: 'text' }
    });
    assert(result.counts.movedBlocks >= 1, 'reordered list items should be marked as moves');
  }

  {
    const sharedPrefix = Array.from({ length: 1_800 }, (_, index) => `keep${index}`).join(' ');
    const leftMiddle = Array.from({ length: 2_200 }, (_, index) => `left${index}`).join(' ');
    const rightMiddle = Array.from({ length: 2_200 }, (_, index) => `right${index}`).join(' ');
    const sharedSuffix = Array.from({ length: 1_800 }, (_, index) => `tail${index}`).join(' ');
    const result = core.compareText({
      leftText: `${sharedPrefix} ${leftMiddle} ${sharedSuffix}`,
      rightText: `${sharedPrefix} ${rightMiddle} ${sharedSuffix}`,
      modeOverride: 'document',
      sourceHints: { leftKind: 'text', rightKind: 'text' }
    });
    assert(result.warnings.length > 0, 'large document comparisons should surface a coarse fallback warning');
    assert(result.runs.some((run) => run.type === 'equal'), 'large document fallback should preserve equal regions');
    assert(result.runs.some((run) => run.type !== 'equal'), 'large document fallback should still emit edits');
  }

  {
    const result = core.compareText({
      leftText: 'name,score,status\nalpha,10,ready\nbeta,12,hold\n',
      rightText: 'name,score,status\nalpha,11,ready\nbeta,12,done\n',
      modeOverride: 'auto',
      sourceHints: { leftKind: 'csv', rightKind: 'csv' }
    });
    assert(result.inferredMode === core.MODES.STRUCTURED, 'CSV inputs should use structured mode');
    assert(result.counts.hasChanges === true, 'CSV inputs should still report edits');
    assert(result.counts.movedBlocks === 0, 'structured mode should not report moved prose blocks');
  }

  {
    const result = core.compareText({
      leftText: '{\n  "name": "alpha",\n  "score": 10\n}\n',
      rightText: '{\n  "name": "alpha",\n  "score": 11,\n  "status": "ready"\n}\n',
      modeOverride: 'auto',
      sourceHints: { leftKind: 'json', rightKind: 'json' }
    });
    assert(result.inferredMode === core.MODES.STRUCTURED, 'JSON inputs should use structured mode');
  }

  const pageScript = fs.readFileSync('js/tools/text-compare.js', 'utf8');
  assert(pageScript.includes('requestId !== latestCompareRequestId'), 'text compare page should ignore stale compare responses');
  assert(pageScript.includes('Compared on the main thread because the background worker was unavailable.'),
    'text compare page should surface worker fallback warnings');
  assert(pageScript.includes('Auto mode used document comparison.'),
    'text compare page should surface the document auto-mode notice');
  assert(pageScript.includes('new Worker(COMPARE_WORKER_PATH)'), 'text compare page should spin up a worker');

  const workerScript = fs.readFileSync('js/tools/text-compare-worker.js', 'utf8');
  assert(workerScript.includes("importScripts(CORE_PATH)"), 'text compare worker should load the shared core');
  assert(workerScript.includes('requestId'), 'text compare worker should round-trip request ids');
};
