(function bootProjectStarfall(global) {
  'use strict';

  function init() {
    const root = document.querySelector('[data-starfall-root]');
    const canvas = document.getElementById('project-starfall-canvas');
    if (!root || !canvas || !global.createProjectStarfallEngine || !global.createProjectStarfallUi) return;

    const engine = global.createProjectStarfallEngine(canvas, global.ProjectStarfallData);
    const ui = global.createProjectStarfallUi(root, engine);

    if (engine.setAssetLoadProgressHandler && ui.updateLoadProgress) {
      engine.setAssetLoadProgressHandler((progress) => ui.updateLoadProgress(progress));
    }
    ui.init();

    const beginGame = () => {
      if (ui.completeInitialLoad) ui.completeInitialLoad();
    };
    const ready = engine.whenAssetsLoaded ? engine.whenAssetsLoaded() : null;
    const renderReady = engine.whenRenderAssetsLoaded ? engine.whenRenderAssetsLoaded() : null;
    const pending = [ready, renderReady].filter((item) => item && typeof item.then === 'function');
    if (pending.length) {
      Promise.allSettled(pending).then(beginGame, beginGame);
    } else {
      beginGame();
    }

    root.ProjectStarfall = {
      engine,
      ui
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})(typeof window !== 'undefined' ? window : globalThis);
