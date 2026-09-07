(() => {
  'use strict';

  // These published, precomputed datasets are served as ordinary site assets.
  // Filtering and chart updates happen in the browser; no demo API is needed.
  const DATA_ROOT = '/demos/data/';

  function createLoader(fetchData = (...args) => fetch(...args)) {
    const readJson = async (path, validate) => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);
      try {
        const response = await fetchData(`${DATA_ROOT}${path}`, {
          cache: 'default',
          credentials: 'omit',
          signal: controller.signal
        });
        if (!response.ok) throw new Error(`Dataset could not be loaded (${response.status}).`);
        const data = await response.json();
        if (!validate(data)) throw new Error('Dataset is incomplete or invalid.');
        return data;
      } finally {
        clearTimeout(timeout);
      }
    };

    const loadDashboardData = (dashboard) => {
      if (!['target-empty-package', 'retail-loss-sales'].includes(dashboard)) {
        throw new Error('Unknown dashboard.');
      }
      return readJson(`${dashboard}/data.json`, (data) => dashboard === 'target-empty-package'
        ? Boolean(data?.meta && Array.isArray(data.rows) && data.rows.length === data.meta.recordCount)
        : Boolean(data?.meta && Array.isArray(data.sales?.weekly) && Array.isArray(data.incidents?.stores)));
    };

    const loadCovidMeta = () => readJson('covid-outbreak/meta.json', (data) =>
      Array.isArray(data?.dates) && data.dates.length > 0 && Array.isArray(data.states));

    const loadCovidDate = (date) => {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(String(date))) throw new Error('Invalid date.');
      return readJson(`covid-outbreak/by-date/${date}.json`, (data) =>
        data?.date === date && Array.isArray(data.states) && Array.isArray(data.hotspots));
    };

    const loadCovidState = (state) => {
      if (!/^[A-Z]{2}$/.test(String(state))) throw new Error('Invalid state.');
      return readJson(`covid-outbreak/state/${state}.json`, (data) =>
        data?.state?.id === state && Array.isArray(data.history));
    };

    return { loadDashboardData, loadCovidMeta, loadCovidDate, loadCovidState };
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = { createLoader };
  } else {
    window.DemoDashboardData = createLoader();
  }
})();
