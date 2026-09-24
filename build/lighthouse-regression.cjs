'use strict';

function median(values) {
  if (!values.length || values.some(value => !Number.isFinite(value))) {
    throw Error('Lighthouse comparison needs finite measurements');
  }
  return [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
}

function compareLighthouseRoute(before, result) {
  if (!before) throw Error(`Missing reviewed baseline: ${result.route}`);
  for (const key of ['lcp', 'tbt']) {
    if (!Number.isFinite(before[key]) || !Number.isFinite(result.median[key])) {
      throw Error(`${result.route}: missing ${key} measurement`);
    }
  }
  if (!Number.isFinite(result.median.cls)) throw Error(`${result.route}: missing CLS measurement`);

  const referenceRuns = result.referenceRuns || null;
  if (referenceRuns && (!Array.isArray(referenceRuns) || referenceRuns.length !== result.runs.length || referenceRuns.length !== 3)) {
    throw Error(`${result.route}: expected three matched reference and candidate runs`);
  }
  const pairedTbtDeltas = referenceRuns
    ? result.runs.map((run, index) => run.tbt - referenceRuns[index].tbt)
    : null;
  const referenceTbt = referenceRuns ? median(referenceRuns.map(run => run.tbt)) : null;
  const tbtDelta = pairedTbtDeltas ? median(pairedTbtDeltas) : null;
  const adjustedTbt = pairedTbtDeltas ? before.tbt + tbtDelta : result.median.tbt;
  const limits = { cls: 0.1, lcp: before.lcp * 1.2 + 250, tbt: before.tbt * 1.2 + 50 };
  const failures = [];
  if (result.median.cls > limits.cls) failures.push(`${result.route}: CLS exceeds 0.1`);
  if (result.median.lcp > limits.lcp) failures.push(`${result.route}: lcp regressed beyond baseline tolerance`);
  if (adjustedTbt > limits.tbt) failures.push(`${result.route}: tbt regressed beyond baseline tolerance`);

  return {
    limits,
    observed: { cls: result.median.cls, lcp: result.median.lcp, tbt: result.median.tbt },
    referenceTbt,
    pairedTbtDeltas,
    tbtDelta,
    adjustedTbt,
    failures
  };
}

module.exports = { median, compareLighthouseRoute };
