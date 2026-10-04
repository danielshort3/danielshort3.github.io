(function (root) {
  'use strict';

  const WIDTH = 96;
  const HEIGHT = 64;
  const FRAME_MS = 180;
  // First 12 SHA-256 characters of each authored atlas; verify when replacing art.
  const FILES = {
    actors: 'actors.png?v=51f333248522',
    props: 'props.png?v=c2d43879b391',
    realms: 'realms.png?v=5f5dccb03a2d'
  };
  const ACTORS = ['walk1', 'walk2', 'walk3', 'idle', 'miner1', 'miner2', 'smith1', 'smith2', 'scholar1', 'cook1', 'forager', 'cartographer', 'leader', 'alchemist', 'fox', 'owl', 'tortoise', 'inspect', 'scholar2', 'cook2'];
  const PROPS = ['ore', 'furnace', 'anvil', 'lantern', 'crate', 'cauldron', 'desk', 'mapTable', 'basket', 'banner', 'alchemy', 'sign'];
  const REALMS = ['greenway', 'copperhills', 'mistwood', 'frostpass', 'sunkenreach', 'starfall', 'frontier', 'room'];
  const ROOMS = {
    trail: { actor: ['walk1', 'walk2', 'walk3', 'walk2'], x: 20 },
    mine: { actor: ['miner1', 'miner2'], x: 21, props: [['ore', 48, 55, 32], ['crate', 76, 55, 18]] },
    forge: { actor: ['smith1', 'smith2'], x: 34, props: [['furnace', 8, 55, 32], ['anvil', 57, 55, 25]] },
    forage: { actor: ['forager'], x: 23, props: [['basket', 51, 55, 23], ['crate', 73, 55, 18]] },
    kitchen: { actor: ['cook1', 'cook2'], x: 25, props: [['cauldron', 52, 55, 27], ['crate', 77, 55, 16]] },
    study: { actor: ['scholar1', 'scholar2'], x: 23, props: [['desk', 51, 55, 32]] },
    cartography: { actor: ['cartographer'], x: 23, props: [['mapTable', 51, 55, 32]] },
    hall: { actor: ['leader'], x: 27, props: [['banner', 65, 43, 28]] },
    alchemy: { actor: ['alchemist'], x: 23, props: [['alchemy', 51, 55, 32]] }
  };
  const ALIASES = { adventuring: 'trail', mining: 'mine', smithing: 'forge', foraging: 'forage', cooking: 'kitchen', scholarship: 'study', leadership: 'hall', research: 'study' };
  // Workers and their stations share the right side of a floor. The left wall
  // stays clear for the host's accessible room name and production labels.
  const STATIONS = {
    mine: { x: 43, props: [['ore', 66, 26], ['crate', 81, 13]] },
    forge: { x: 49, props: [['furnace', 30, 23], ['anvil', 69, 23]] },
    forage: { x: 49, props: [['basket', 70, 23], ['crate', 83, 12]] },
    kitchen: { x: 49, props: [['cauldron', 69, 25], ['crate', 84, 12]] },
    study: { x: 48, props: [['desk', 67, 28]] },
    cartography: { x: 48, props: [['mapTable', 66, 29]] },
    hall: { x: 51, props: [['banner', 76, 24]] },
    alchemy: { x: 48, props: [['alchemy', 67, 28]] }
  };

  function normalizeView(value, previous) {
    const incoming = value && typeof value === 'object' ? value : {};
    const next = Object.assign({}, previous, incoming);
    next.room = ALIASES[next.room] || next.room;
    if (!ROOMS[next.room]) next.room = 'trail';
    next.realm = typeof next.realm === 'object' && next.realm ? next.realm.id : next.realm;
    if (!REALMS.includes(next.realm) || next.realm === 'room') next.realm = 'greenway';
    next.progress = Math.max(0, Math.min(1, Number(next.progress) || 0));
    next.workers = Array.isArray(next.workers) ? next.workers.length : Math.max(0, Math.min(20, Number(next.workers) || 0));
    next.companion = typeof next.companion === 'object' && next.companion ? next.companion.id : next.companion;
    if (!['fox', 'owl', 'tortoise'].includes(next.companion)) next.companion = null;
    return next;
  }

  function create(canvas, options) {
    if (!canvas || typeof canvas.getContext !== 'function') throw new TypeError('WayfarersScene requires a canvas.');
    const settings = options || {};
    const files = Object.assign({}, FILES);
    const art = settings.sceneArt && typeof settings.sceneArt === 'object' ? settings.sceneArt : {};
    ['trail', 'room', ...Object.keys(ROOMS), ...REALMS].forEach(name => {
      if (typeof art[name] === 'string' && art[name].trim()) files['art-' + name] = art[name];
    });
    const assetCount = Object.keys(files).length;
    const document = canvas.ownerDocument || root.document;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Canvas rendering is unavailable.');
    const stage = document.createElement('canvas');
    stage.width = WIDTH;
    stage.height = HEIGHT;
    const paint = stage.getContext('2d', { alpha: false });
    if (!paint) throw new Error('Canvas rendering is unavailable.');
    const media = typeof root.matchMedia === 'function' ? root.matchMedia('(prefers-reduced-motion: reduce)') : null;
    let view = normalizeView({ room: 'trail', realm: 'greenway', progress: 0, workers: 1, companion: null, reducedMotion: settings.reducedMotion }, {});
    let disposed = false;
    let inView = true;
    let frameTimer = null;
    let dirty = true;
    let tick = 0;
    let draws = 0;
    let generation = 0;
    const images = {};
    const failed = new Set();
    const pending = new Set();
    const loadTimers = new Map();
    let status = 'loading';
    canvas.style.imageRendering = 'pixelated';

    function visible() { return !disposed && inView && !document.hidden; }
    function reduced() { return view.reducedMotion === true || !!(media && media.matches); }
    function animated() { return !reduced() && view.workers > 0; }
    function getStatus() {
      return { state: status, loaded: Object.keys(images).length, total: assetCount, failed: Array.from(failed), draws, room: view.room, layout: settings.layout || 'panorama' };
    }
    function notify() {
      canvas.dataset.sceneStatus = status;
      if (typeof settings.onStatus === 'function') settings.onStatus(getStatus());
    }

    function sprite(kind, name, x, baseline, size) {
      const list = kind === 'actors' ? ACTORS : PROPS;
      const index = list.indexOf(name);
      if (index < 0 || !images[kind]) return;
      const side = size || 32;
      paint.drawImage(images[kind], (index % 4) * 32, Math.floor(index / 4) * 32, 32, 32,
        Math.round(x), Math.round(baseline - 30 * side / 32), side, side);
    }

    function livingScene(width, height) {
      const portrait = settings.layout === 'portrait' && view.room === 'trail';
      // Width owns pixel size in the tall trail and the stacked rooms. Using
      // canvas height here would enlarge a 32px actor to fill the portrait.
      const scale = width / WIDTH;
      const logicalHeight = Math.max(1, Math.ceil(height / scale));
      if (stage.width !== WIDTH) stage.width = WIDTH;
      if (stage.height !== logicalHeight) stage.height = logicalHeight;
      paint.setTransform(1, 0, 0, 1, 0, 0);
      paint.imageSmoothingEnabled = false;
      context.imageSmoothingEnabled = false;
      canvas.dataset.sceneScale = String(scale);
      canvas.dataset.sceneLogicalWidth = String(WIDTH);
      canvas.dataset.sceneLogicalHeight = String(logicalHeight);
      canvas.dataset.sceneLayout = portrait ? 'portrait' : 'room';
      const isTrail = view.room === 'trail';
      const backdrop = isTrail ? images['art-' + view.realm] || (view.realm === 'greenway' ? images['art-trail'] : null) : images['art-' + view.room] || images['art-room'];
      const realmIndex = REALMS.indexOf(isTrail ? view.realm : 'room');
      const sky = ['#8cc6e9', '#f5e9cc', '#849f9c', '#c6dfed', '#506e81', '#14283a', '#102139', '#f5e9cc'];
      paint.fillStyle = sky[realmIndex];
      paint.fillRect(0, 0, WIDTH, logicalHeight);
      let portraitArtHeight = logicalHeight;
      let portraitArtTop = 0;
      if (backdrop) {
        if (portrait) {
          // Preserve the whole trail, including its future Mine entrance. Extra
          // portrait height extends the sky above art anchored to the bottom.
          portraitArtHeight = WIDTH * backdrop.naturalHeight / backdrop.naturalWidth;
          portraitArtTop = logicalHeight - portraitArtHeight;
          if (portraitArtTop > 0) paint.drawImage(backdrop, 0, 0, backdrop.naturalWidth, 1, 0, 0, WIDTH, portraitArtTop);
          paint.drawImage(backdrop, 0, 0, backdrop.naturalWidth, backdrop.naturalHeight, 0, portraitArtTop, WIDTH, portraitArtHeight);
        } else {
          // Room backgrounds cover their floor; actors retain square cells.
          const ratio = Math.max(WIDTH / backdrop.naturalWidth, logicalHeight / backdrop.naturalHeight);
          const sourceWidth = WIDTH / ratio, sourceHeight = logicalHeight / ratio;
          paint.drawImage(backdrop, (backdrop.naturalWidth - sourceWidth) / 2, (backdrop.naturalHeight - sourceHeight) / 2, sourceWidth, sourceHeight, 0, 0, WIDTH, logicalHeight);
        }
      } else if (images.realms) {
        const sourceX = (realmIndex % 2) * WIDTH, sourceY = Math.floor(realmIndex / 2) * HEIGHT;
        if (portrait) {
          // Legacy realm art remains recognizable on later expeditions. Extend
          // its sky upward instead of stretching the horizon and foreground.
          const backgroundHeight = Math.min(HEIGHT, logicalHeight);
          paint.drawImage(images.realms, sourceX, sourceY, WIDTH, HEIGHT, 0, logicalHeight - backgroundHeight, WIDTH, backgroundHeight);
        } else {
          paint.drawImage(images.realms, sourceX, sourceY, WIDTH, isTrail ? HEIGHT : 54, 0, 0, WIDTH, logicalHeight);
        }
      }
      const phase = animated() ? tick : 0;
      const floor = portrait ? backdrop ? Math.round(portraitArtTop + portraitArtHeight * 0.86) : logicalHeight - 8 : Math.round(logicalHeight * (backdrop ? 0.8 : 0.9));
      canvas.dataset.sceneFloor = String(floor);
      canvas.dataset.sceneArtTop = String(portraitArtTop);
      canvas.dataset.sceneArtHeight = String(portraitArtHeight);
      const size = Math.min(portrait ? 28 : 30, Math.max(8, floor - 2));
      const room = ROOMS[view.room], station = STATIONS[view.room];
      if (!isTrail) {
        if (!backdrop) sprite('props', 'lantern', 82, 14, 11);
        station.props.forEach(prop => sprite('props', prop[0], prop[1], floor, Math.min(prop[2], size)));
      }
      if (view.workers > 0) sprite('actors', room.actor[phase % room.actor.length], isTrail ? portrait ? 8 : 25 : station.x, floor, size);
      if (view.companion) sprite('actors', view.companion, isTrail ? portrait ? 31 : 49 : 80, floor, Math.min(18, size));
      if (view.room === 'forge' && animated() && phase % 4 === 1) {
        paint.fillStyle = '#ffbd43';
        paint.fillRect(73, floor - 15, 1, 1);
        paint.fillRect(76, floor - 17, 1, 1);
      }
      context.fillStyle = '#14283a';
      context.fillRect(0, 0, width, height);
      // Uniform nearest-neighbor scaling also applies at fractional DPR. A
      // partial final pixel may clip below the canvas, never distort an actor.
      context.drawImage(stage, 0, 0, WIDTH * scale, logicalHeight * scale);
    }

    function draw() {
      if (!visible()) { dirty = true; return; }
      const bounds = canvas.getBoundingClientRect();
      const cssWidth = bounds.width || Number(canvas.getAttribute('width')) || 384;
      const cssHeight = bounds.height || cssWidth * HEIGHT / WIDTH;
      const dpr = Math.min(3, Math.max(1, root.devicePixelRatio || 1));
      const width = Math.max(WIDTH, Math.round(cssWidth * dpr));
      const height = Math.max(HEIGHT, Math.round(cssHeight * dpr));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      if (settings.layout === 'portrait' || settings.layout === 'room') {
        livingScene(width, height);
        dirty = false;
        draws += 1;
        return;
      }
      // A compact panorama may be much wider than its authored96x64 room.
      // Expand the logical background; never stretch square actor/prop cells.
      const scale = height / HEIGHT;
      const logicalWidth = Math.max(WIDTH, Math.ceil(width / scale));
      if (stage.width !== logicalWidth) stage.width = logicalWidth;
      const offset = Math.floor((logicalWidth - WIDTH) / 2);
      canvas.dataset.sceneScale = String(scale);
      canvas.dataset.sceneLogicalWidth = String(logicalWidth);
      canvas.dataset.sceneLogicalHeight = String(HEIGHT);
      canvas.dataset.sceneLayout = 'panorama';
      context.imageSmoothingEnabled = false;
      paint.setTransform(1, 0, 0, 1, 0, 0);
      paint.imageSmoothingEnabled = false;
      paint.fillStyle = '#14283a';
      paint.fillRect(0, 0, logicalWidth, HEIGHT);
      paint.translate(offset, 0);
      const room = ROOMS[view.room];
      const realmIndex = REALMS.indexOf(view.room === 'trail' ? view.realm : 'room');
      if (images.realms) {
        // The imported room has a lower floor than the trail; crop its excess floor
        // before nearest-neighbor scaling so all actors share the same baseline.
        const sourceX = (realmIndex % 2) * WIDTH;
        const sourceY = Math.floor(realmIndex / 2) * HEIGHT;
        const sourceHeight = view.room === 'trail' ? HEIGHT : 54;
        if (offset > 0) {
          paint.drawImage(images.realms, sourceX, sourceY, 1, sourceHeight, -offset, 0, offset, HEIGHT);
          paint.drawImage(images.realms, sourceX + WIDTH - 1, sourceY, 1, sourceHeight, WIDTH, 0, logicalWidth - WIDTH - offset, HEIGHT);
        }
        paint.drawImage(images.realms, (realmIndex % 2) * WIDTH, Math.floor(realmIndex / 2) * HEIGHT,
          WIDTH, view.room === 'trail' ? HEIGHT : 54, 0, 0, WIDTH, HEIGHT);
      }
      paint.fillStyle = '#14283a';
      paint.fillRect(-offset, 57, logicalWidth, 7);
      const phase = animated() ? tick : 0;
      if (view.room !== 'trail') {
        sprite('props', 'lantern', 33, 24, 15);
        room.props.forEach(prop => sprite('props', ...prop));
      }
      if (view.workers > 0) {
        const frame = room.actor[phase % room.actor.length];
        // Actors work in place: travel is automatic and no art suggests direct movement controls.
        sprite('actors', frame, room.x, 56, 32);
      }
      if (view.companion) sprite('actors', view.companion, view.room === 'trail' ? 5 : 74, 56, 25);
      if (view.room === 'forge' && animated() && phase % 4 === 1) {
        paint.fillStyle = '#ffbd43';
        paint.fillRect(66, 36, 1, 1);
        paint.fillRect(69, 34, 1, 1);
      }
      if (view.room === 'trail' && view.progress >= 0.98) {
        paint.fillStyle = '#ffbd43';
        paint.fillRect(85, 24, 1, 3);
      }
      context.fillStyle = '#14283a';
      context.fillRect(0, 0, width, height);
      const drawnWidth = logicalWidth * scale;
      context.drawImage(stage, (width - drawnWidth) / 2, 0, drawnWidth, height);
      if (!images.realms && status !== 'ready') {
        context.fillStyle = '#f5e9cc';
        context.font = `${Math.max(12, Math.round(12 * dpr))}px sans-serif`;
        context.textAlign = 'center';
        context.fillText(status === 'error' ? 'Artwork unavailable. Reconnect to retry.' : 'Loading guild artwork…', width / 2, height / 2);
      }
      dirty = false;
      draws += 1;
    }

    function stop() {
      if (frameTimer !== null) { root.clearTimeout(frameTimer); frameTimer = null; }
    }
    function schedule() {
      if (!visible()) { stop(); return; }
      if (dirty) draw();
      if (animated() && status === 'ready' && frameTimer === null) {
        frameTimer = root.setTimeout(function frame() {
          frameTimer = null;
          if (!visible()) return;
          tick += 1;
          dirty = true;
          draw();
          schedule();
        }, FRAME_MS);
      } else if (!animated() || status !== 'ready') stop();
    }

    function loadAssets() {
      if (disposed) return;
      pending.forEach(image => { image.onload = null; image.onerror = null; });
      pending.clear();
      loadTimers.forEach(timer => root.clearTimeout(timer));
      loadTimers.clear();
      const current = ++generation;
      status = 'loading';
      failed.clear();
      const names = Object.keys(files).filter(name => !images[name]);
      if (!names.length) { status = 'ready'; notify(); schedule(); return; }
      names.forEach(name => {
        const image = new root.Image();
        let settled = false;
        pending.add(image);
        function finish(ok) {
          if (settled) return;
          settled = true;
          root.clearTimeout(loadTimers.get(image));
          loadTimers.delete(image);
          pending.delete(image);
          image.onload = null;
          image.onerror = null;
          if (disposed || current !== generation) return;
          if (ok && image.naturalWidth) images[name] = image;
          else failed.add(name);
          status = Object.keys(images).length === assetCount ? 'ready' : pending.size ? 'loading' : 'error';
          dirty = true;
          notify();
          schedule();
        }
        image.onload = () => finish(true);
        image.onerror = () => finish(false);
        loadTimers.set(image, root.setTimeout(() => finish(false), 12000));
        const base = String(settings.assetBase || '/img/wayfarers-guild/').replace(/\/?$/, '/');
        image.src = /^(?:\/|https?:)/.test(files[name]) ? files[name] : base + files[name];
      });
      notify();
    }

    function update(next) {
      if (disposed) return;
      const previous = JSON.stringify(view);
      view = normalizeView(next, view);
      if (JSON.stringify(view) !== previous) dirty = true;
      schedule();
    }
    function onVisibility() { if (visible()) { dirty = true; schedule(); } else stop(); }
    function onResize() { dirty = true; schedule(); }
    function onMotion() { dirty = true; schedule(); }
    function onOnline() { if (status === 'error') loadAssets(); }
    document.addEventListener('visibilitychange', onVisibility);
    root.addEventListener('online', onOnline);
    if (media && media.addEventListener) media.addEventListener('change', onMotion);
    const resizeObserver = root.ResizeObserver ? new root.ResizeObserver(onResize) : null;
    if (resizeObserver) resizeObserver.observe(canvas);
    else root.addEventListener('resize', onResize);
    const intersectionObserver = root.IntersectionObserver ? new root.IntersectionObserver(entries => {
      inView = entries.some(entry => entry.isIntersecting);
      onVisibility();
    }) : null;
    if (intersectionObserver) intersectionObserver.observe(canvas);
    loadAssets();
    schedule();

    return {
      update,
      setRoom(id) { update({ room: id }); },
      getStatus,
      retryAssets: loadAssets,
      destroy() {
        if (disposed) return;
        disposed = true;
        generation += 1;
        stop();
        pending.forEach(image => { image.onload = null; image.onerror = null; });
        pending.clear();
        loadTimers.forEach(timer => root.clearTimeout(timer));
        loadTimers.clear();
        if (resizeObserver) resizeObserver.disconnect();
        if (intersectionObserver) intersectionObserver.disconnect();
        if (media && media.removeEventListener) media.removeEventListener('change', onMotion);
        document.removeEventListener('visibilitychange', onVisibility);
        root.removeEventListener('online', onOnline);
        root.removeEventListener('resize', onResize);
        canvas.dataset.sceneStatus = 'destroyed';
      }
    };
  }

  const api = { create, roomIds: Object.keys(ROOMS), realmIds: REALMS.filter(id => id !== 'room') };
  root.WayfarersScene = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
