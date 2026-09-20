(function bootProjectStarfall(global) {
  'use strict';

  function init() {
    const root = document.querySelector('[data-starfall-root]');
    const canvas = document.getElementById('project-starfall-canvas');
    if (!root || !canvas || !global.createProjectStarfallEngine || !global.createProjectStarfallUi) return;

    const engine = global.createProjectStarfallEngine(canvas, global.ProjectStarfallData);
    const ui = global.createProjectStarfallUi(root, engine);

    const configuredSource = root.getAttribute('data-starfall-hurtboxes-src');
    const combatSource = new URL(configuredSource || '/js/games/project-starfall/data/enemy-hurtboxes.js', document.baseURI).href;
    const combatReady = () => Boolean(global.ProjectStarfallEnemyHurtboxesData &&
      (!configuredSource || global.ProjectStarfallEnemyHurtboxesSource === combatSource));
    let pendingStart = null;
    const openCharacterSelect = ui.openCharacterSelect.bind(ui);
    const setStartLoading = (loading) => {
      const start = root.querySelector('[data-starfall-start-screen]');
      start?.setAttribute('aria-busy', String(loading));
      start?.querySelectorAll('button').forEach(button => { button.disabled = loading; });
      if (loading) {
        ui.elements.stage?.classList.add('is-loading');
        ui.elements.loader?.setAttribute('aria-hidden', 'false');
        ui.setLoaderProgress({ percent: 0 }, 'Preparing game');
      } else {
        ui.updateLoadProgress({ percent: 100, complete: true });
        ui.elements.loader?.setAttribute('aria-hidden', 'true');
      }
    };
    ui.openCharacterSelect = () => {
      if (pendingStart) return pendingStart;
      if (combatReady()) return openCharacterSelect();
      setStartLoading(true);
      global.ProjectStarfallEnemyHurtboxesData = null;
      pendingStart = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = combatSource;
        script.async = true;
        let timer;
        const finish = (error) => {
          global.clearTimeout(timer);
          script.onload = null;
          script.onerror = null;
          script.remove();
          if (error) reject(error);
          else resolve();
        };
        script.onload = () => {
          if (!global.ProjectStarfallEnemyHurtboxesData) return finish(new Error('Combat data was not initialized.'));
          global.ProjectStarfallEnemyHurtboxesSource = combatSource;
          finish();
        };
        script.onerror = () => finish(new Error('Combat data could not be downloaded.'));
        timer = global.setTimeout(() => finish(new Error('Combat data download timed out.')), 30000);
        document.head.append(script);
      }).then(() => {
        if (!root.isConnected) return false;
        setStartLoading(false);
        return openCharacterSelect();
      }, () => {
        if (root.isConnected) {
          setStartLoading(false);
          ui.showToast('Could not start the game. Select Start to retry.');
        }
        return false;
      }).finally(() => { pendingStart = null; });
      return pendingStart;
    };

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
