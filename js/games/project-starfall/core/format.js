(function initProjectStarfallCoreFormat(global) {
  'use strict';

  const ABBREVIATED_NUMBER_SUFFIXES = Object.freeze(['', 'k', 'm', 'b', 't', 'q', 'qi', 'sx', 'sp', 'oc', 'no', 'dc']);

  function formatPerformanceMs(value) {
    return `${(Number(value) || 0).toFixed(2)}ms`;
  }

  function formatPerformanceFps(value) {
    return `${Math.round(Number(value) || 0)} fps`;
  }

  function formatCooldownLabel(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    if (value < 10) return `${value.toFixed(1).replace(/\.0$/, '')}s`;
    return `${Math.ceil(value)}s`;
  }

  function formatIntegerWithCommas(value) {
    const number = Math.round(Number(value) || 0);
    const sign = number < 0 ? '-' : '';
    let amount = Math.abs(number);
    if (amount < 1000) return `${sign}${String(amount)}`;
    let suffixIndex = 0;
    while (amount >= 1000 && suffixIndex < ABBREVIATED_NUMBER_SUFFIXES.length - 1) {
      amount /= 1000;
      suffixIndex += 1;
    }
    if (Math.round(amount * 10) / 10 >= 1000 && suffixIndex < ABBREVIATED_NUMBER_SUFFIXES.length - 1) {
      amount /= 1000;
      suffixIndex += 1;
    }
    return `${sign}${amount.toFixed(1)}${ABBREVIATED_NUMBER_SUFFIXES[suffixIndex]}`;
  }

  const api = {
    ABBREVIATED_NUMBER_SUFFIXES,
    formatPerformanceMs,
    formatPerformanceFps,
    formatCooldownLabel,
    formatIntegerWithCommas
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
