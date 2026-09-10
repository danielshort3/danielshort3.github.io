(() => {
  'use strict';

  const stage = document.querySelector('#ocean-wave-stage');
  if (!stage) return;
  const find = (id) => stage.querySelector(`#ocean-wave-${id}`);
  const fullscreenButton = find('fullscreen');
  const soundButton = find('sound');
  const settingsButton = find('settings-toggle');
  const settings = find('settings');
  const volume = find('volume');
  const status = find('status');
  const hud = stage.querySelector('.ocean-wave-hud');
  if (!fullscreenButton || !soundButton || !settingsButton || !settings || !volume) return;

  let disposed = false;
  let fullscreenPending = false;
  let fallbackPlaceholder = null;
  let previousOverflow = '';
  let idleTimer = 0;
  let soundEnabled = false;
  let resumeWithSound = false;
  const timerInput = find('timer');
  const timerStatus = find('timer-status');
  const restScreen = find('rest-screen');
  const listeners = [];
  const on = (target, name, callback, options) => {
    target.addEventListener(name, callback, options);
    listeners.push(() => target.removeEventListener(name, callback, options));
  };
  const announce = (text) => { if (status) status.textContent = text; };
  const isFullscreen = () => document.fullscreenElement === stage || Boolean(fallbackPlaceholder);

  const revealControls = () => {
    window.clearTimeout(idleTimer);
    stage.classList.remove('is-idle');
    if (!isFullscreen() || !settings.hidden || disposed) return;
    idleTimer = window.setTimeout(() => {
      if (!disposed && settings.hidden && !hud.contains(document.activeElement)) stage.classList.add('is-idle');
    }, 4200);
  };

  const setSettingsOpen = (open, restoreFocus = false) => {
    settings.hidden = !open;
    stage.classList.toggle('is-adjusting', open);
    settingsButton.setAttribute('aria-expanded', String(open));
    settingsButton.setAttribute('aria-label', open ? 'Close ocean settings' : 'Open ocean settings');
    if (open) find('wind').focus({ preventScroll: true });
    else if (restoreFocus) settingsButton.focus({ preventScroll: true });
    revealControls();
  };
  on(settingsButton, 'click', () => setSettingsOpen(settings.hidden, !settings.hidden));
  on(find('settings-close'), 'click', () => setSettingsOpen(false, true));
  on(stage, 'pointerdown', (event) => {
    if (!settings.hidden && !settings.contains(event.target) && !settingsButton.contains(event.target)) setSettingsOpen(false);
    revealControls();
  });
  on(stage, 'pointermove', revealControls, { passive: true });
  on(stage, 'focusin', revealControls);
  on(stage, 'focusout', revealControls);
  on(find('controls'), 'submit', (event) => event.preventDefault());

  const syncFullscreen = () => {
    const active = isFullscreen();
    fullscreenButton.setAttribute('aria-pressed', String(active));
    fullscreenButton.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
    fullscreenButton.title = active ? 'Exit fullscreen (Esc)' : 'Fullscreen (F)';
    const label = fullscreenButton.querySelector('[data-ocean-fullscreen-label]');
    if (label) label.textContent = active ? 'Exit' : 'Full screen';
    fullscreenButton.querySelector('path').setAttribute('d', active
      ? 'M3 8h5V3m8 0v5h5M8 21v-5H3m18 0h-5v5'
      : 'M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5');
    revealControls();
    window.dispatchEvent(new Event('resize'));
  };
  const exitFallback = () => {
    if (!fallbackPlaceholder) return;
    fallbackPlaceholder.replaceWith(stage);
    fallbackPlaceholder = null;
    stage.classList.remove('is-fullscreen');
    document.body.style.overflow = previousOverflow;
  };
  const enterFallback = () => {
    // Portalling avoids the shared site's transformed and clipped viewport.
    fallbackPlaceholder = document.createElement('div');
    fallbackPlaceholder.style.height = `${stage.getBoundingClientRect().height}px`;
    stage.before(fallbackPlaceholder);
    previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    document.body.appendChild(stage);
    stage.classList.add('is-fullscreen');
  };
  const toggleFullscreen = async () => {
    if (fullscreenPending || disposed) return;
    fullscreenPending = true;
    try {
      if (fallbackPlaceholder) exitFallback();
      else if (document.fullscreenElement === stage) await document.exitFullscreen();
      else {
        try {
          if (!stage.requestFullscreen) throw new Error('Native fullscreen unavailable');
          await stage.requestFullscreen();
        } catch {
          if (!disposed) enterFallback();
        }
      }
      if (disposed) {
        if (document.fullscreenElement === stage) await document.exitFullscreen();
        return;
      }
      syncFullscreen();
      stage.focus({ preventScroll: true });
    } catch {
      announce('Fullscreen could not be changed. Try again.');
    } finally {
      fullscreenPending = false;
    }
  };
  on(fullscreenButton, 'click', toggleFullscreen);
  on(document, 'fullscreenchange', () => {
    if (!disposed) syncFullscreen();
  });
  on(stage, 'keydown', (event) => {
    if (event.key === 'Tab' && isFullscreen()) {
      revealControls();
      const controls = Array.from(stage.querySelectorAll('button, input, select, textarea, a[href], [tabindex]'))
        .filter((element) => !element.disabled && element.tabIndex >= 0
          && !element.closest('[hidden], [inert]') && element.getClientRects().length > 0
          && getComputedStyle(element).visibility !== 'hidden');
      const first = controls[0];
      const last = controls[controls.length - 1];
      const active = document.activeElement;
      if (!first) { event.preventDefault(); stage.focus({ preventScroll: true }); }
      else if (event.shiftKey && (active === first || active === stage || !stage.contains(active))) {
        event.preventDefault();
        last.focus({ preventScroll: true });
      } else if (!event.shiftKey && (active === last || active === stage || !stage.contains(active))) {
        event.preventDefault();
        first.focus({ preventScroll: true });
      }
      return;
    }
    if (event.key === 'Escape') {
      if (!settings.hidden) { setSettingsOpen(false, true); event.preventDefault(); }
      else if (fallbackPlaceholder) { exitFallback(); syncFullscreen(); stage.focus({ preventScroll: true }); event.preventDefault(); }
      else if (document.fullscreenElement === stage) {
        event.preventDefault();
        document.exitFullscreen().then(() => {
          if (!disposed) stage.focus({ preventScroll: true });
        }).catch(() => {
          if (!disposed) announce('Fullscreen could not be changed. Try again.');
        });
      }
      return;
    }
    if (event.target.closest('input, select, textarea') || event.ctrlKey || event.altKey || event.metaKey) return;
    if (event.key.toLowerCase() === 'f') { event.preventDefault(); toggleFullscreen(); }
    else if (event.code === 'Space' && event.target === stage) { event.preventDefault(); find('toggle').click(); }
    revealControls();
  });

  const syncSoundButton = () => {
    soundButton.setAttribute('aria-pressed', String(soundEnabled));
    soundButton.setAttribute('aria-label', soundEnabled ? 'Mute ocean sound' : 'Enable ocean sound');
    soundButton.title = soundEnabled ? 'Mute ocean sound' : 'Enable ocean sound';
    soundButton.querySelector('[data-ocean-sound-waves]').setAttribute('d', soundEnabled
      ? 'M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14'
      : 'm16 9 5 6m0-6-5 6');
  };

  const audio = window.OceanWaveAudio.create({
    onStatus: ({ state, enabled }) => {
      if (disposed) return;
      soundEnabled = enabled;
      syncSoundButton();
      soundButton.setAttribute('aria-busy', String(state === 'loading'));
      const label = soundButton.querySelector('[data-ocean-sound-label]');
      if (label) label.textContent = state === 'loading' ? 'Loading' : 'Sound';
    },
    onError: () => { if (!disposed) announce('Sound could not load. Tap Sound to try again.'); },
  });
  const syncAudioVisibility = () => audio.setVisible(!document.hidden && stage.dataset.oceanVisible !== 'false');
  const setSound = async (enabled) => {
    if (disposed) return;
    soundEnabled = enabled;
    syncSoundButton();
    await audio.setEnabled(enabled);
  };
  const syncVolume = () => {
    const value = Math.max(0, Math.min(100, Number(volume.value) || 0));
    find('volume-value').textContent = `${value}%`;
    volume.setAttribute('aria-valuetext', `${value} percent`);
    audio.setVolume(value / 100);
  };
  try {
    const saved = window.localStorage.getItem('ds-ocean-volume-v1');
    if (saved !== null && Number.isFinite(Number(saved))) volume.value = String(Math.max(0, Math.min(100, Number(saved))));
  } catch {}
  syncVolume();
  audio.setScene(stage.dataset.oceanScene === 'cove' ? 'cove' : 'ocean');
  audio.setConditions({
    wind: find('wind').value,
    waveHeight: find('height').value,
    shore: stage.dataset.oceanShore,
  });
  syncAudioVisibility();
  on(soundButton, 'click', () => { setSound(!soundEnabled).catch(() => {}); });
  on(volume, 'input', () => {
    syncVolume();
    try { window.localStorage.setItem('ds-ocean-volume-v1', volume.value); } catch {}
  });
  on(stage, 'ocean:scene-change', event => audio.setScene(event.detail.scene));
  on(stage, 'ocean:conditions', event => audio.setConditions(event.detail));
  on(document, 'visibilitychange', syncAudioVisibility);
  on(stage, 'ocean:visibility', syncAudioVisibility);

  const sessionTimer = window.OceanWaveTimer.create({
    onTick: ({ active, remainingMs, fade }) => {
      if (disposed) return;
      stage.style.setProperty('--ocean-rest-fade', String(fade));
      audio.setFade(1 - fade);
      timerStatus.hidden = !active;
      if (active) {
        const minutes = Math.ceil(remainingMs / 60000);
        timerStatus.textContent = `${minutes} min · Cancel`;
        timerStatus.setAttribute('aria-label', `Cancel timer, ${minutes} minutes remaining`);
      }
    },
    onComplete: () => {
      if (disposed) return;
      resumeWithSound = soundEnabled;
      setSound(false).catch(() => {});
      setSettingsOpen(false);
      stage.classList.add('is-resting');
      stage.classList.remove('is-idle');
      restScreen.hidden = false;
      stage.dispatchEvent(new CustomEvent('ocean:rest'));
      find('resume').focus({ preventScroll: true });
      announce('Your ocean session has gently faded out.');
    },
  });
  const cancelTimer = () => { timerInput.value = '0'; sessionTimer.cancel(); };
  on(timerInput, 'change', () => sessionTimer.start(Number(timerInput.value)));
  on(timerStatus, 'click', () => { cancelTimer(); announce('Fade-out timer cancelled.'); });
  on(find('resume'), 'click', () => {
    cancelTimer();
    restScreen.hidden = true;
    stage.classList.remove('is-resting');
    stage.dispatchEvent(new CustomEvent('ocean:resume'));
    if (resumeWithSound) setSound(true).catch(() => {});
    resumeWithSound = false;
    stage.focus({ preventScroll: true });
    revealControls();
  });
  on(document, 'visibilitychange', () => sessionTimer.refresh());
  on(stage, 'ocean:renderer', (event) => {
    stage.classList.toggle('has-render-error', event.detail.available === false);
  });

  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    soundEnabled = false;
    window.clearTimeout(idleTimer);

    listeners.forEach((remove) => remove());
    exitFallback();
    stage.classList.remove('is-idle');
    if (document.fullscreenElement === stage) document.exitFullscreen().catch(() => {});
    sessionTimer.dispose();
    audio.dispose();
  };
  on(window, 'pagehide', (event) => {
    if (!event.persisted) { cleanup(); return; }
    // Browser Back can restore this same document without rerunning scripts.
    // Keep the controls intact while suspending work in the page cache.
    window.clearTimeout(idleTimer);

    stage.classList.remove('is-idle');
    audio.setVisible(false);
  });
  on(window, 'pageshow', (event) => {
    if (!event.persisted || disposed) return;
    revealControls();
    sessionTimer.refresh();
    syncAudioVisibility();
  });
  window.SiteRoutes?.addCleanup(cleanup);
})();
