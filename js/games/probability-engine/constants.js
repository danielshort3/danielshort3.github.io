(() => {
  "use strict";

  const PE = window.ProbabilityEngine = window.ProbabilityEngine || {};

  PE.SAVE_KEY = "probability-engine-save-v1";
  PE.AUTOSAVE_MS = 10000;
  PE.TICK_MS = 100;
  PE.LIVE_UI_REFRESH_MS = 250;
  PE.DECK_EV_BASELINE_SAMPLES = 40;
  PE.DECK_EV_DELTA_SAMPLES = 24;
  PE.MIN_DECK_SIZE = 10;
  PE.MAX_DECK_SIZE = 32;
  PE.PITY_LIMIT = 10;
  PE.BASE_STARTING_CASH = 120;
})();
