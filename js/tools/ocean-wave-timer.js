(() => {
  'use strict';

  const create = ({ onTick = () => {}, onComplete = () => {}, now = () => Date.now(),
    schedule = (fn, delay) => window.setTimeout(fn, delay), cancelSchedule = id => window.clearTimeout(id) } = {}) => {
    let deadline = 0;
    let timeout = 0;
    let disposed = false;
    const clear = () => { if (timeout) cancelSchedule(timeout); timeout = 0; };
    const refresh = () => {
      clear();
      if (disposed || !deadline) return;
      const remainingMs = Math.max(0, deadline - now());
      // The final minute fades steadily; waking a background tab uses the
      // original deadline rather than extending the session by missed ticks.
      const progress = Math.max(0, Math.min(1, 1 - remainingMs / 60000));
      const fade = progress * progress * (3 - 2 * progress);
      onTick({ active: remainingMs > 0, remainingMs, fade });
      if (!remainingMs) {
        deadline = 0;
        onComplete();
      } else timeout = schedule(refresh, Math.min(1000, remainingMs));
    };
    const cancel = () => {
      clear();
      deadline = 0;
      if (!disposed) onTick({ active: false, remainingMs: 0, fade: 0 });
    };
    return {
      start(minutes) {
        if (disposed) return;
        if (![15, 30, 60].includes(Number(minutes))) { cancel(); return; }
        deadline = now() + Number(minutes) * 60000;
        refresh();
      },
      cancel, refresh,
      dispose() { disposed = true; clear(); deadline = 0; },
    };
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = { create };
  else window.OceanWaveTimer = { create };
})();
