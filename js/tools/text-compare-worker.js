(() => {
  'use strict';

  const CORE_PATH = '/js/tools/text-compare-core.js';
  let coreApi = null;

  const getCoreApi = () => {
    if (coreApi) return coreApi;
    if (typeof importScripts !== 'function') {
      throw new Error('Worker importScripts is unavailable.');
    }
    importScripts(CORE_PATH);
    coreApi = self.TextCompareCore;
    if (!coreApi || typeof coreApi.compareText !== 'function') {
      throw new Error('Text compare core failed to load in the worker.');
    }
    return coreApi;
  };

  self.addEventListener('message', (event) => {
    const data = event?.data || {};
    const requestId = data.requestId;
    try {
      const api = getCoreApi();
      const result = api.compareText({
        leftText: data.leftText,
        rightText: data.rightText,
        modeOverride: data.modeOverride,
        sourceHints: data.sourceHints
      });
      self.postMessage({
        requestId,
        ok: true,
        runs: result.runs,
        counts: result.counts,
        inferredMode: result.inferredMode,
        warnings: result.warnings
      });
    } catch (error) {
      self.postMessage({
        requestId,
        ok: false,
        error: error instanceof Error ? error.message : 'Worker compare failed.'
      });
    }
  });
})();
