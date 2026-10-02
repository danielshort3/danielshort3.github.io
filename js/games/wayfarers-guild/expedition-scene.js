(function (root) {
  'use strict';

  // A working model of the current expedition. All progress, queues and visible
  // milestones come from the engine; the renderer never awards resources.
  const PALETTES = {
    greenway: { sky: '#91bfd4', grass: '#809e58', light: '#95ad62', dark: '#557c49', tree: '#2c6547', leaf: '#438453', water: '#5b9fbc', rock: '#66707b' },
    copperhills: { sky: '#c5b898', grass: '#a49061', light: '#beac77', dark: '#776e4b', tree: '#426548', leaf: '#748650', water: '#70999e', rock: '#7c7775' },
    mistwood: { sky: '#a8bebb', grass: '#718b70', light: '#88a583', dark: '#4b7063', tree: '#36594f', leaf: '#587c69', water: '#789eac', rock: '#63737c' },
    frostpass: { sky: '#b8d1e1', grass: '#c1d2ca', light: '#dce5d6', dark: '#95b3aa', tree: '#4e7770', leaf: '#8faba0', water: '#709cae', rock: '#748391' },
    sunkenreach: { sky: '#a9bdc1', grass: '#66917c', light: '#82a791', dark: '#477466', tree: '#305f52', leaf: '#588675', water: '#538397', rock: '#667b80' },
    starfall: { sky: '#617796', grass: '#627584', light: '#849299', dark: '#43596d', tree: '#314c64', leaf: '#506b83', water: '#537ea5', rock: '#687891' },
    frontier: { sky: '#a9bac0', grass: '#8a9870', light: '#a4ae7e', dark: '#687854', tree: '#49684c', leaf: '#71895d', water: '#6a99a8', rock: '#748083' }
  };
  const INK = '#172c3c';
  const GOLD = '#f0bf58';
  const CREAM = '#f3e1b4';
  const LETTERS = {
    A: [14, 17, 17, 31, 17, 17, 17], B: [30, 17, 17, 30, 17, 17, 30], C: [14, 17, 16, 16, 16, 17, 14], D: [30, 17, 17, 17, 17, 17, 30], E: [31, 16, 16, 30, 16, 16, 31], F: [31, 16, 16, 30, 16, 16, 16],
    G: [14, 17, 16, 23, 17, 17, 15], H: [17, 17, 17, 31, 17, 17, 17], I: [14, 4, 4, 4, 4, 4, 14], J: [7, 2, 2, 2, 18, 18, 12], K: [17, 18, 20, 24, 20, 18, 17], L: [16, 16, 16, 16, 16, 16, 31],
    M: [17, 27, 21, 21, 17, 17, 17], N: [17, 25, 21, 19, 17, 17, 17], O: [14, 17, 17, 17, 17, 17, 14], P: [30, 17, 17, 30, 16, 16, 16], Q: [14, 17, 17, 17, 21, 18, 13], R: [30, 17, 17, 30, 20, 18, 17],
    S: [15, 16, 16, 14, 1, 1, 30], T: [31, 4, 4, 4, 4, 4, 4], U: [17, 17, 17, 17, 17, 17, 14], V: [17, 17, 17, 17, 17, 10, 4], W: [17, 17, 17, 21, 21, 21, 10], X: [17, 17, 10, 4, 10, 17, 17], Y: [17, 17, 10, 4, 4, 4, 4], Z: [31, 1, 2, 4, 8, 16, 31],
    0: [14, 17, 19, 21, 25, 17, 14], 1: [4, 12, 4, 4, 4, 4, 14], 2: [14, 17, 1, 2, 4, 8, 31], 3: [30, 1, 1, 14, 1, 1, 30], 4: [2, 6, 10, 18, 31, 2, 2], 5: [31, 16, 16, 30, 1, 1, 30], 6: [14, 16, 16, 30, 17, 17, 14], 7: [31, 1, 2, 4, 8, 8, 8], 8: [14, 17, 17, 14, 17, 17, 14], 9: [14, 17, 17, 15, 1, 1, 14]
  };
  // Bounds within the approved 1448 x 1086 generated prop sheet. Transparent
  // gutters are removed at draw time so actors retain the same visual scale.
  const ART = { tree: [78, 56, 219, 294], explorer: [439, 80, 205, 269], tent: [1095, 783, 312, 237], fox: [747, 808, 274, 216] };
  const clamp = (value, low, high) => Math.max(low, Math.min(high, Number.isFinite(Number(value)) ? Number(value) : low));
  const finite = value => Math.max(0, Number.isFinite(Number(value)) ? Number(value) : 0);

  function normalizeView(input) {
    const source = input && (input.scene || input.expedition && input.expedition.scene || input) || {};
    const kind = ['greenway', 'quarry', 'watchtower'].includes(source.kind) ? source.kind : 'greenway';
    return {
      kind,
      index: Math.floor(finite(source.index)),
      region: PALETTES[source.region] ? source.region : 'greenway',
      progress: clamp(source.progress, 0, 1),
      completed: source.completed === true,
      ranks: source.ranks || {},
      rates: source.rates || {},
      flows: source.flows || null,
      unlocked: Array.isArray(source.unlocked) ? source.unlocked : null,
      route: source.route === 'supply' ? 'supply' : 'short',
      oreBuffer: finite(source.oreBuffer),
      smeltBuffer: finite(source.smeltBuffer),
      capacity: Math.max(1, finite(source.capacity)),
      workers: { repair: finite(source.workers && source.workers.repair), protection: finite(source.workers && source.workers.protection), total: finite(source.workers && source.workers.total) },
      beacon: clamp(source.beacon, 0, 1),
      checkpoint: Math.floor(clamp(source.checkpoint, 0, 4)),
      bottleneck: ['picks', 'carts', 'furnace'].includes(source.bottleneck) ? source.bottleneck : null,
      allocation: ['repair', 'protect'].includes(source.allocation) ? source.allocation : 'balanced',
      quality: source.quality === 'quality' ? 'quality' : 'throughput',
      pressure: finite(source.pressure),
      hazard: source.hazard === true,
      banner: ['banner-amber', 'banner-moon'].includes(source.banner) ? source.banner : null,
      companion: ['fox', 'owl', 'tortoise'].includes(source.companion) ? source.companion : null
    };
  }

  function create(canvas, options) {
    if (!canvas || typeof canvas.getContext !== 'function') throw new TypeError('An expedition canvas is required.');
    const settings = options || {};
    const document = canvas.ownerDocument || root.document;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Canvas rendering is unavailable.');
    const buffer = document.createElement('canvas');
    const paint = buffer.getContext('2d', { alpha: false });
    if (!paint) throw new Error('Canvas rendering is unavailable.');
    const motion = root.matchMedia ? root.matchMedia('(prefers-reduced-motion: reduce)') : null;
    let view = normalizeView({});
    let quiet = settings.quiet === true;
    let disposed = false;
    let inView = true;
    let frame = null;
    let dirty = true;
    let lastPaint = -Infinity;
    let animation = 0;
    let cartPhase = 0;
    let previousTime = null;
    let width = 192;
    let height = 210;
    let viewportHeight = 245;
    let hotspots = [];
    let actorImage = null;
    let optionalArt = null;
    let pendingImages = [];
    let pointerStart = null;
    let deliveredFlash = 0;
    let previousDelivered = 0;
    canvas.style.imageRendering = 'pixelated';
    canvas.dataset.sceneStatus = 'ready';

    function rectangle(x, y, w, h, color) {
      paint.fillStyle = color;
      paint.fillRect(Math.round(x), Math.round(y), Math.max(1, Math.round(w)), Math.max(1, Math.round(h)));
    }
    function polygon(points, color) {
      paint.fillStyle = color;
      paint.beginPath();
      points.forEach((point, i) => i ? paint.lineTo(Math.round(point[0]), Math.round(point[1])) : paint.moveTo(Math.round(point[0]), Math.round(point[1])));
      paint.closePath();
      paint.fill();
    }
    function line(points, color, thickness) {
      paint.strokeStyle = color;
      paint.lineWidth = thickness || 1;
      paint.lineCap = 'square';
      paint.lineJoin = 'miter';
      paint.beginPath();
      points.forEach((point, i) => i ? paint.lineTo(Math.round(point[0]) + 0.5, Math.round(point[1]) + 0.5) : paint.moveTo(Math.round(point[0]) + 0.5, Math.round(point[1]) + 0.5));
      paint.stroke();
    }
    function label(text, x, y, accent) {
      const word = String(text).toUpperCase();
      const length = word.length * 6 - 1;
      const left = Math.round(clamp(x - length / 2 - 4, 3, width - length - 11));
      rectangle(left, y, length + 8, 13, INK);
      if (accent) rectangle(left, y + 12, length + 8, 1, accent);
      Array.from(word).forEach((character, column) => {
        const glyph = LETTERS[character];
        if (!glyph) return;
        glyph.forEach((bits, row) => {
          for (let pixel = 0; pixel < 5; pixel += 1) if (bits & (1 << (4 - pixel))) rectangle(left + 4 + column * 6 + pixel, y + 3 + row, 1, 1, CREAM);
        });
      });
    }
    function bar(x, y, w, fraction, color) {
      rectangle(x, y, w, 4, INK);
      if (fraction > 0) rectangle(x + 1, y + 1, (w - 2) * clamp(fraction, 0, 1), 2, color || GOLD);
    }
    function shadow(x, y, w) {
      rectangle(x - w / 2 + 2, y, w - 4, 3, '#304d4566');
      rectangle(x - w / 2, y + 1, w, 1, '#304d4566');
    }
    function artwork(id, x, y, w, h) {
      const crop = ART[id];
      if (!optionalArt || !crop) return false;
      paint.drawImage(optionalArt, crop[0], crop[1], crop[2], crop[3], Math.round(x), Math.round(y), Math.round(w), Math.round(h));
      return true;
    }
    function tree(x, y, size, palette) {
      const s = size || 1;
      shadow(x, y, 15 * s);
      if (view.region === 'greenway' && artwork('tree', x - 11 * s, y - 30 * s, 22 * s, 30 * s)) return;
      rectangle(x - 2 * s, y - 8 * s, 4 * s, 8 * s, '#755b3d');
      polygon([[x, y - 30 * s], [x - 8 * s, y - 16 * s], [x - 5 * s, y - 16 * s], [x - 11 * s, y - 7 * s], [x + 11 * s, y - 7 * s], [x + 5 * s, y - 16 * s], [x + 8 * s, y - 16 * s]], palette.tree);
      polygon([[x - 1 * s, y - 28 * s], [x - 6 * s, y - 17 * s], [x - 3 * s, y - 17 * s], [x - 7 * s, y - 11 * s], [x + 2 * s, y - 11 * s], [x + 4 * s, y - 16 * s]], palette.leaf);
    }
    function crate(x, y, size) {
      const s = size || 10;
      rectangle(x, y - s, s, s, '#533e2e');
      rectangle(x + 1, y - s + 1, s - 2, s - 2, '#a87b49');
      rectangle(x + 3, y - s + 1, 1, s - 2, '#d1aa66');
      line([[x + 1, y - 2], [x + s - 2, y - s + 2]], '#735031', 1);
    }
    function actor(x, y, role, moving, facing) {
      const step = !reduced() && moving ? Math.floor(animation * 4) % 4 : 3;
      shadow(x, y - 1, 13);
      if (role === 'explorer' && optionalArt) {
        const bob = moving && !reduced() && step % 2 ? 1 : 0;
        paint.save();
        paint.translate(Math.round(x), Math.round(y - bob));
        if (facing < 0) paint.scale(-1, 1);
        artwork('explorer', -9, -24, 18, 24);
        paint.restore();
        return;
      }
      if (actorImage) {
        const indices = role === 'miner' ? [4, 5, 4, 5] : role === 'smith' ? [6, 7, 6, 7] : [0, 1, 2, 1];
        const cell = moving ? indices[step] : role === 'miner' ? 4 : role === 'smith' ? 6 : 3;
        paint.save();
        paint.translate(Math.round(x), Math.round(y));
        if (facing < 0) paint.scale(-1, 1);
        paint.drawImage(actorImage, cell % 4 * 32, Math.floor(cell / 4) * 32, 32, 32, -16, -31, 32, 32);
        paint.restore();
        return;
      }
      const bob = moving && !reduced() && step % 2 ? 1 : 0;
      rectangle(x - 4, y - 11 - bob, 8, 8, role === 'miner' ? '#e2a14e' : '#49788d');
      rectangle(x - 4, y - 18 - bob, 8, 7, '#ebba7e');
      rectangle(x - 6, y - 19 - bob, 12, 3, role === 'miner' ? GOLD : '#876443');
      rectangle(x - 3, y - 22 - bob, 7, 4, role === 'miner' ? GOLD : '#876443');
      rectangle(x - 4, y - 3, 3, 3 + (moving && step % 2 ? 1 : 0), INK);
      rectangle(x + 1, y - 3, 3, 3 + (moving && !(step % 2) ? 1 : 0), INK);
      rectangle(x + (facing < 0 ? -3 : 2), y - 15 - bob, 1, 1, INK);
    }
    function companion(x, y) {
      if (!view.companion) return;
      shadow(x, y, 10);
      if (view.companion === 'fox' && artwork('fox', x - 9, y - 13, 18, 14)) return;
      if (actorImage) {
        const cell = { fox:14, owl:15, tortoise:16 }[view.companion];
        paint.drawImage(actorImage, cell % 4 * 32, Math.floor(cell / 4) * 32, 32, 32, Math.round(x - 10), Math.round(y - 18), 20, 20);
      } else {
        rectangle(x - 5, y - 6, 10, 5, view.companion === 'fox' ? '#db893f' : view.companion === 'owl' ? '#bcada0' : '#54754e');
        rectangle(x + 2, y - 9, 5, 5, '#d6c28e');
      }
    }
    function guildBanner(x, y) {
      if (!view.banner) return;
      rectangle(x - 7, y - 21, 16, 2, '#c7a363');
      rectangle(x - 6, y - 19, 13, 18, INK);
      const moon = view.banner === 'banner-moon';
      rectangle(x - 5, y - 18, 11, 15, moon ? '#476690' : '#d79d41');
      if (moon) {
        rectangle(x - 2, y - 15, 5, 8, CREAM);
        rectangle(x + 1, y - 15, 3, 6, '#476690');
      } else {
        polygon([[x, y - 15], [x + 4, y - 11], [x, y - 7], [x - 3, y - 11]], '#7c4e35');
        rectangle(x, y - 12, 1, 3, CREAM);
      }
      polygon([[x - 5, y - 3], [x, y], [x + 6, y - 3]], moon ? '#476690' : '#d79d41');
    }
    function hotspot(id, kind, labelText, x, y, w, h) {
      const unlocked = !view.unlocked || kind !== 'upgrade' || view.unlocked.includes(id);
      if (!unlocked) return;
      hotspots.push({ id, kind, label: labelText, x: x / width, y: y / viewportHeight, width: w / width, height: h / viewportHeight });
    }
    function rank(id) { return finite(view.ranks[id]); }
    function unlocked(id) { return !view.unlocked || view.unlocked.includes(id); }
    function haulRate() { return view.flows ? finite(view.flows.carts) : finite(view.rates.carts) || 0.4 + rank('carts') * 0.2; }
    function haulPeriod() { return clamp(10 / Math.sqrt(Math.max(0.1, haulRate())), 2, 12); }
    function pathPosition(points, fraction) {
      const lengths = points.slice(1).map((point, index) => Math.hypot(point[0] - points[index][0], point[1] - points[index][1]));
      const total = lengths.reduce((sum, value) => sum + value, 0);
      let distance = total * clamp(fraction, 0, 1);
      for (let i = 0; i < lengths.length; i += 1) {
        if (distance <= lengths[i] || i === lengths.length - 1) {
          const part = lengths[i] ? distance / lengths[i] : 0;
          return [points[i][0] + (points[i + 1][0] - points[i][0]) * part, points[i][1] + (points[i + 1][1] - points[i][1]) * part];
        }
        distance -= lengths[i];
      }
      return points[0];
    }
    function dottedPath(points, color) {
      const count = Math.max(6, Math.floor(points.slice(1).reduce((sum, point, i) => sum + Math.hypot(point[0] - points[i][0], point[1] - points[i][1]), 0) / 8));
      for (let i = 0; i <= count; i += 1) {
        const at = pathPosition(points, i / count);
        rectangle(at[0] - 1, at[1] - 1, 3, 3, color);
      }
    }
    function flag(x, y, complete) {
      rectangle(x, y - 12, 2, 12, '#735b40');
      rectangle(x + 2, y - 12, 7, 5, complete ? GOLD : '#b6c3a5');
      if (complete) rectangle(x + 3, y - 11, 2, 2, CREAM);
    }

    function drawGreenway(palette) {
      rectangle(0, 0, width, height, palette.grass);
      const floor = height * 0.82;
      const junction = [width * 0.38, height * 0.47];
      const bridgeY = height * 0.33;
      polygon([[0, height * 0.12], [width * 0.25, height * 0.2], [width * 0.49, height * 0.05], [width, height * 0.16], [width, 0], [0, 0]], palette.light);
      const river = [[width * 0.72, -10], [width * 0.67, height * 0.13], [width * 0.8, bridgeY], [width * 0.87, height * 0.57], [width * 1.02, height * 0.75]];
      line(river, '#accad0', 23);
      line(river, palette.water, 17);
      for (let i = 0; i < 8; i += 1) {
        const at = pathPosition(river, (i / 9 + (reduced() ? 0 : animation * 0.012)) % 1);
        rectangle(at[0] - 3, at[1], 5, 1, '#a6c8d0');
      }
      const start = [width * 0.12, floor];
      const short = [start, [width * 0.23, height * 0.62], junction, [width * 0.58, height * 0.37], [width * 0.79, bridgeY]];
      const supply = [start, [width * 0.43, height * 0.77], [width * 0.63, height * 0.66], [width * 0.58, height * 0.48], [width * 0.79, bridgeY]];
      line(short, '#b9ae79', 8);
      line(supply, '#b9ae79', 7);
      line(view.route === 'supply' ? supply : short, '#dcca98', 5);
      if (rank('scouts') > 0 || view.progress > 0.2) dottedPath(view.route === 'supply' ? short : supply, '#d6d9b6');
      dottedPath(view.route === 'supply' ? supply : short, '#f1d78b');
      const bridgeX = width * 0.78;
      const repair = view.completed ? 1 : view.beacon;
      line([[bridgeX - 16, bridgeY - 8], [bridgeX + 17, bridgeY - 8]], '#72523b', 3);
      line([[bridgeX - 16, bridgeY + 7], [bridgeX + 17, bridgeY + 7]], '#72523b', 3);
      for (let i = 0; i < 9; i += 1) {
        const existing = i < 2 || i > 6;
        if (existing || i - 2 < repair * 5) rectangle(bridgeX - 17 + i * 4, bridgeY - 7, 3, 14, existing ? '#b78b52' : '#dfb568');
      }
      [-18, 16].forEach(dx => { rectangle(bridgeX + dx, bridgeY - 12, 3, 24, '#694a37'); rectangle(bridgeX + dx, bridgeY - 12, 3, 2, '#bf965b'); });
      tree(width * 0.13, height * 0.24, 0.9, palette);
      tree(width * 0.47, height * 0.18, 1, palette);
      tree(width * 0.95, height * 0.19, 0.82, palette);
      tree(width * 0.13, height * 0.54, 0.82, palette);
      tree(width * 0.79, height * 0.87, 0.86, palette);
      tree(width * 0.53, height * 0.94, 0.72, palette);
      const campX = width * 0.48, campY = height * 0.78;
      shadow(campX, campY, 27);
      if (!artwork('tent', campX - 16, campY - 24, 32, 24)) {
        polygon([[campX - 15, campY], [campX, campY - 20], [campX + 16, campY]], '#e4d4aa');
        polygon([[campX, campY - 20], [campX + 16, campY], [campX + 5, campY]], '#b5aa85');
        polygon([[campX - 5, campY], [campX, campY - 12], [campX + 5, campY]], INK);
      }
      crate(campX + 17, campY + 1, 8);
      guildBanner(campX + 25, campY - 10);
      if (rank('porters') >= 3) crate(campX + 24, campY + 5, 7);
      const route = view.route === 'supply' ? supply : short;
      const explorer = view.completed ? [bridgeX + 23, bridgeY] : pathPosition(route, Math.min(0.84, view.progress / 0.78 * 0.84));
      actor(explorer[0], explorer[1] - 2, 'explorer', !view.completed, 1);
      companion(explorer[0] - 17, explorer[1] + 2);
      if (rank('porters') >= 3) crate(explorer[0] - 12, explorer[1] - 3, 6);
      [0.22, 0.49, 0.72].forEach((fraction, i) => {
        const at = pathPosition(route, fraction);
        flag(at[0] + 9, at[1] - 6, view.checkpoint > i || view.progress >= fraction);
      });
      if (view.completed) { flag(bridgeX + 23, bridgeY - 3, true); label('Outpost established', width / 2, height - 20, GOLD); }
      else if (repair > 0) { bar(bridgeX - 17, bridgeY + 18, 35, repair); label('Bridge', bridgeX, bridgeY + 24); }
      hotspot('boots', 'upgrade', 'Inspect boots', explorer[0] - 14, explorer[1] - 28, 28, 32);
      hotspot('porters', 'upgrade', 'Inspect satchel', campX - 19, campY - 25, 46, 32);
      hotspot('scouts', 'upgrade', 'Inspect scout', junction[0] - 15, junction[1] - 22, 30, 30);
    }

    function pile(x, y, amount, color, capacity) {
      const count = Math.ceil(clamp(amount / Math.max(1, capacity), 0, 1) * 12);
      for (let i = 0; i < count; i += 1) {
        const row = i < 5 ? 0 : i < 9 ? 1 : 2;
        const column = row === 0 ? i : row === 1 ? i - 5 : i - 9;
        const px = x - 12 + column * 5 + row * 2;
        const py = y - row * 4;
        rectangle(px, py - 4, 5, 4, '#51443d');
        rectangle(px + 1, py - 5, 4, 3, color);
      }
    }
    function arrow(x, y, active) {
      const color = active ? GOLD : '#5d6b72';
      rectangle(x - 5, y - 1, 8, 3, color);
      polygon([[x + 2, y - 4], [x + 7, y], [x + 2, y + 5]], color);
    }
    function cart(x, y, load, second) {
      shadow(x, y + 3, 23);
      rectangle(x - 11, y - 14, 22, 3, '#bcc0b1');
      polygon([[x - 10, y - 11], [x + 10, y - 11], [x + 7, y - 2], [x - 7, y - 2]], second ? '#927350' : '#9e7856');
      rectangle(x - 8, y - 10, 2, 7, '#4d4a46');
      rectangle(x + 6, y - 10, 2, 7, '#4d4a46');
      if (load > 0) pile(x, y - 14, load, '#d99c68', 1);
      [-7, 7].forEach(dx => { rectangle(x + dx - 3, y - 2, 6, 6, '#293942'); rectangle(x + dx - 1, y, 2, 2, '#a4aca8'); });
    }
    function drawQuarry(palette) {
      rectangle(0, 0, width, height, '#243343');
      polygon([[0, 0], [width, 0], [width, height * 0.16], [width * 0.78, height * 0.06], [width * 0.59, height * 0.17], [width * 0.34, height * 0.1], [0, height * 0.22]], palette.rock);
      const floor = Math.round(height * 0.59);
      rectangle(0, floor, width, height - floor, '#38434a');
      rectangle(0, floor, width, 5, '#8c755c');
      rectangle(0, floor + 5, width, 3, '#554b42');
      [width * 0.06, width * 0.91].forEach(x => {
        rectangle(x, height * 0.16, 5, floor - height * 0.16, '#70533e');
        rectangle(x + 1, height * 0.16, 1, floor - height * 0.16, '#a07e51');
      });
      rectangle(width * 0.045, height * 0.16, width * 0.9, 6, '#8f6946');
      rectangle(width * 0.045, height * 0.16, width * 0.9, 2, '#ba9158');
      const lampY = height * 0.16;
      [width * 0.31, width * 0.7].forEach(x => { rectangle(x, lampY + 5, 1, 11, '#b19462'); rectangle(x - 3, lampY + 16, 7, 9, '#121f2c'); rectangle(x - 1, lampY + 18, 3, 5, GOLD); });
      const miningX = width * 0.16;
      const furnaceX = width * 0.82;
      polygon([[2, floor - 3], [2, floor - 25], [width * 0.06, floor - 36], [width * 0.14, floor - 29], [width * 0.2, floor - 10], [width * 0.18, floor - 3]], '#6e777a');
      [[5, -21], [14, -29], [22, -17], [11, -11]].forEach(([x, y]) => { rectangle(x, floor + y, 5, 4, '#d39a68'); rectangle(x + 1, floor + y, 2, 1, '#ecc098'); });
      if (rank('picks') >= 3) { rectangle(6, floor - 37, 6, 4, '#e2b673'); rectangle(13, floor - 41, 4, 3, '#d59f55'); }
      const extracting = view.flows ? finite(view.flows.picks) > 0 : view.oreBuffer < view.capacity;
      actor(miningX + 9, floor - 1, 'miner', extracting && !view.completed, -1);
      companion(miningX - 9, floor + 1);
      guildBanner(width * 0.85, height * 0.29);
      const minedFraction = clamp(view.oreBuffer / view.capacity, 0, 1);
      pile(width * 0.35, floor - 1, view.oreBuffer, '#d69861', view.capacity);
      const railStart = width * 0.41, railEnd = width * 0.71;
      for (let x = railStart - 4; x < railEnd + 7; x += 7) rectangle(x, floor - 1, 3, 5, '#9a7e59');
      rectangle(railStart - 5, floor - 1, railEnd - railStart + 13, 1, '#b2b4a7');
      rectangle(railStart - 5, floor + 2, railEnd - railStart + 13, 1, '#788a90');
      if (rank('carts') >= 3) {
        line([[railStart, floor], [railStart - 7, floor + 7], [railStart - 7, floor + 19], [railStart, floor + 25], [railEnd, floor + 25], [railEnd + 7, floor + 18], [railEnd + 7, floor + 6], [railEnd, floor]], '#9fa99c', 1);
        for (let x = railStart; x < railEnd; x += 7) rectangle(x, floor + 24, 3, 3, '#8e7759');
      }
      // A trip's period follows actual haul throughput (or a bounded rank
      // fallback for older saves). Animation is a rate cue, not an extra tick.
      const rate = haulRate();
      const phase = reduced() || view.completed || rate === 0 ? 0.3 : cartPhase % 1;
      const distance = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
      const hasLoad = rate > 0 || view.oreBuffer > 0.02 || view.smeltBuffer > 0.02;
      cart(railStart + (railEnd - railStart) * distance, floor - 4, hasLoad && phase < 0.55 ? Math.max(0.25, minedFraction) : 0, false);
      if (rank('carts') >= 3) {
        const secondPhase = (phase + 0.5) % 1;
        const secondDistance = secondPhase < 0.5 ? secondPhase * 2 : (1 - secondPhase) * 2;
        cart(railStart + (railEnd - railStart) * secondDistance, floor + 21, hasLoad && secondPhase < 0.55 ? 0.45 : 0, true);
        rectangle(railStart - 5, floor + 25, railEnd - railStart + 13, 1, '#9fa99c');
      }
      rectangle(furnaceX - 14, floor - 29, 28, 29, '#596570');
      rectangle(furnaceX - 12, floor - 31, 24, 3, '#9aa19c');
      rectangle(furnaceX - 4, floor - 48, 9, 18, '#737e80');
      rectangle(furnaceX - 6, floor - 49, 13, 3, '#a7a89b');
      rectangle(furnaceX - 10, floor - 23, 20, 21, '#303740');
      rectangle(furnaceX - 7, floor - 19, 14, 17, '#171f29');
      const smelting = (view.flows ? finite(view.flows.furnace) > 0 : view.smeltBuffer > 0.02) && !view.completed;
      if (smelting || view.completed) {
        rectangle(furnaceX - 6, floor - 15, 12, 12, '#cb642f');
        const flicker = reduced() ? 0 : Math.floor(animation * 6) % 3;
        polygon([[furnaceX - 4, floor - 4], [furnaceX - 3, floor - 12 - flicker], [furnaceX, floor - 8], [furnaceX + 3, floor - 15 + flicker], [furnaceX + 5, floor - 4]], GOLD);
        rectangle(furnaceX - 1, floor - 9, 3, 6, '#ffedbb');
      }
      if (rank('furnace') >= 3) rectangle(furnaceX - 13, floor - 27, 26, 3, view.quality === 'quality' ? GOLD : '#a4b8bd');
      pile(furnaceX + 16, floor - 1, view.smeltBuffer, '#c88c5b', view.capacity);
      const bars = Math.min(6, Math.floor(view.progress / 0.78 * 7));
      for (let i = 0; i < bars; i += 1) { rectangle(furnaceX - 10 + (i % 3) * 8, floor + 16 - Math.floor(i / 3) * 4, 7, 4, '#d79354'); rectangle(furnaceX - 9 + (i % 3) * 8, floor + 16 - Math.floor(i / 3) * 4, 5, 1, '#f2c58c'); }
      arrow(width * 0.33, floor - 29, view.oreBuffer > 0);
      arrow(width * 0.68, floor - 29, smelting);
      label('Mine', width * 0.16, floor + 8);
      label('Haul', width * 0.52, floor + (rank('carts') >= 3 ? 31 : 8));
      label('Smelt', width * 0.84, floor + 8);
      bar(width * 0.35 - 10, floor + 6, 21, minedFraction, '#dca970');
      bar(width * 0.92 - 8, floor + 3, 17, view.smeltBuffer / view.capacity, '#dca970');
      if (height - floor > 68) {
        const shaftX = width * 0.52;
        const shaftBottom = height - 13;
        const liftY = shaftBottom - (shaftBottom - floor - 44) * view.beacon;
        rectangle(shaftX - 22, floor + 48, 44, shaftBottom - floor - 42, '#1b2b38');
        rectangle(shaftX - 21, floor + 47, 2, shaftBottom - floor - 43, '#72553d');
        rectangle(shaftX + 19, floor + 47, 2, shaftBottom - floor - 43, '#72553d');
        rectangle(shaftX, floor + 46, 1, liftY - floor - 44, '#b09a6a');
        rectangle(shaftX - 16, liftY, 33, 3, '#a88454');
        for (let i = 0; i < Math.min(4, bars); i += 1) rectangle(shaftX - 12 + i % 2 * 12, liftY - 4 - Math.floor(i / 2) * 4, 10, 4, '#dca970');
        if (view.beacon > 0 && !view.completed) bar(shaftX - 18, shaftBottom + 6, 36, view.beacon);
      }
      if (!view.completed && view.progress < 0.78) {
        const full = view.oreBuffer >= view.capacity * 0.95;
        const waiting = !smelting && view.progress < 0.78;
        if (full) label('Full', width * 0.36, floor - 24, GOLD);
        if (waiting) label('Waiting', furnaceX, floor - 64, CREAM);
        if (!full && !waiting && view.bottleneck) {
          const x = view.bottleneck === 'picks' ? miningX : view.bottleneck === 'carts' ? width * 0.53 : furnaceX;
          const y = floor - (view.bottleneck === 'furnace' ? 64 : 54);
          label('Slowest', x, y, GOLD);
          polygon([[x - 3, y + 15], [x + 3, y + 15], [x, y + 19]], GOLD);
        }
      } else if (view.completed) label('Mine established', width / 2, height - 19, GOLD);
      if (deliveredFlash > 0 && !reduced()) { const rise = (1 - deliveredFlash) * 8; rectangle(furnaceX + 19, floor - 32 - rise, 3, 3, GOLD); rectangle(furnaceX + 21, floor - 37 - rise, 1, 1, CREAM); }
      hotspot('picks', 'upgrade', 'Inspect picks', 0, floor - 48, width * 0.36, 52);
      hotspot('carts', 'upgrade', 'Inspect carts', width * 0.37, floor - 25, width * 0.31, 57);
      hotspot('furnace', 'upgrade', 'Inspect furnace', width * 0.7, floor - 58, width * 0.3, 63);
    }

    function drawWatchtower(palette) {
      rectangle(0, 0, width, height, palette.sky);
      polygon([[0, height * 0.53], [width * 0.18, height * 0.23], [width * 0.38, height * 0.43], [width * 0.55, height * 0.25], [width * 0.84, height * 0.48], [width, height * 0.28], [width, height], [0, height]], '#7293a7');
      polygon([[0, height * 0.66], [width * 0.25, height * 0.45], [width * 0.53, height * 0.67], [width * 0.82, height * 0.43], [width, height * 0.61], [width, height], [0, height]], '#537b88');
      const floor = Math.round(height * 0.82);
      rectangle(0, floor - 9, width, height - floor + 9, palette.grass);
      rectangle(0, floor + 7, width, height - floor - 7, palette.light);
      tree(width * 0.11, floor - 8, 0.74, palette);
      tree(width * 0.89, floor - 17, 0.86, palette);
      const towerX = width * 0.49;
      const towerWidth = Math.min(47, width * 0.25);
      const towerHeight = Math.min(height * 0.55, 118);
      const top = floor - towerHeight;
      const courses = 9;
      const built = Math.min(courses, 2 + Math.floor(view.progress / 0.78 * 7));
      rectangle(towerX - towerWidth / 2 - 5, floor - 2, towerWidth + 10, 5, '#5b655e');
      // Empty scaffold preserves the landmark silhouette; only completed
      // masonry courses appear, so progress has a readable physical result.
      [towerX - towerWidth / 2 - 8, towerX + towerWidth / 2 + 5].forEach(x => {
        rectangle(x, top - 7, 3, towerHeight + 7, '#765536');
        rectangle(x + 1, top - 7, 1, towerHeight + 7, '#bb975c');
      });
      for (let i = 0; i < courses; i += 1) {
        const courseY = floor - (i + 1) * towerHeight / courses;
        if (i < built) {
          rectangle(towerX - towerWidth / 2, courseY, towerWidth, towerHeight / courses + 1, i < 2 ? '#aba88e' : '#c7bda1');
          rectangle(towerX + towerWidth / 2 - 8, courseY, 8, towerHeight / courses + 1, '#a19b82');
          rectangle(towerX - towerWidth / 2, courseY, towerWidth, 1, '#8b8b77');
          for (let x = towerX - towerWidth / 2 + (i % 2 ? 7 : 0); x < towerX + towerWidth / 2 - 8; x += 15) rectangle(x, courseY, 1, towerHeight / courses, '#9e9d84');
        }
        if (i % 3 === 0) { rectangle(towerX - towerWidth / 2 - 10, courseY + 1, towerWidth + 23, 3, '#866345'); }
      }
      rectangle(towerX - 6, floor - 21, 12, 21, '#4a554b');
      rectangle(towerX - 3, floor - 18, 6, 18, '#283b3b');
      if (built >= 5) rectangle(towerX - 3, floor - towerHeight * 0.53, 6, 13, '#4e5c58');
      if (built >= 8) rectangle(towerX - 3, top + 13, 6, 11, '#4e5c58');
      const workY = floor - built * towerHeight / courses + 2;
      const workerCount = Math.min(4, Math.max(1, Math.ceil(view.workers.repair / 2)));
      for (let i = 0; i < workerCount; i += 1) {
        const walk = reduced() || view.completed ? 0 : Math.sin(animation * (0.6 + rank('crew') * 0.03) + i * 2) * 6;
        actor(towerX - towerWidth / 2 - 5 + i * 11 + walk, i % 2 ? floor - 2 : workY, 'miner', !view.completed, i % 2 ? -1 : 1);
      }
      const guardCount = Math.min(3, Math.ceil(view.workers.protection / 2));
      for (let i = 0; i < guardCount; i += 1) {
        const guardX = width * (0.18 + i * 0.31);
        actor(guardX, floor + 15, 'explorer', false, i % 2 ? -1 : 1);
        rectangle(guardX + 7, floor + 1, 5, 7, '#718f9e');
        rectangle(guardX + 8, floor + 3, 3, 1, CREAM);
      }
      const craneX = towerX + towerWidth / 2 + 11;
      rectangle(craneX, top + 19, 4, towerHeight - 19, '#72523b');
      line([[craneX - 5, top + 27], [craneX + 28, top + 13]], '#8c6540', 4);
      const liftRate = finite(view.rates.repair) || 1 + rank('lift') * 0.25;
      const liftPhase = reduced() || view.completed ? 0.65 : (Math.sin(animation * 0.4 * Math.sqrt(liftRate)) + 1) / 2;
      const liftY = top + 30 + liftPhase * (towerHeight - 39);
      rectangle(craneX + 24, top + 15, 1, liftY - top - 15, '#e1c283');
      crate(craneX + 18, liftY + 10, rank('lift') >= 3 ? 14 : 10);
      for (let i = 0; i < 4; i += 1) rectangle(width * 0.78 + (i % 2) * 7, floor + 7 - Math.floor(i / 2) * 4, 6, 4, '#8f9995');
      const beaconY = top - 14;
      rectangle(towerX - 9, beaconY + 4, 18, 4, '#6f523a');
      rectangle(towerX - 7, beaconY, 2, 8, '#ac8953');
      rectangle(towerX + 5, beaconY, 2, 8, '#ac8953');
      if (view.completed || view.beacon >= 1) {
        const flicker = reduced() ? 0 : Math.floor(animation * 5) % 3;
        polygon([[towerX - 5, beaconY + 4], [towerX - 7, beaconY - 5], [towerX - 2, beaconY - 2], [towerX + 1, beaconY - 14 - flicker], [towerX + 4, beaconY - 5], [towerX + 7, beaconY + 4]], '#df7d35');
        polygon([[towerX - 3, beaconY + 4], [towerX - 1, beaconY - 7], [towerX + 3, beaconY - 2], [towerX + 4, beaconY + 4]], GOLD);
        rectangle(towerX - 7, top + 19, 14, 26, '#335c81');
        rectangle(towerX - 1, top + 22, 3, 16, GOLD);
        rectangle(towerX - 4, top + 27, 9, 3, GOLD);
        label('Beacon restored', width / 2, height - 18, GOLD);
      } else if (unlocked('beacon')) bar(towerX - 10, beaconY - 7, 20, view.beacon, GOLD);
      if (!view.completed && view.workers.total > 0) {
        label(String(Math.round(view.workers.repair)) + ' build', width * 0.25, height - 18);
        label(String(Math.round(view.workers.protection)) + ' guard', width * 0.75, height - 18);
      }
      guildBanner(towerX - towerWidth / 2 - 2, floor - 3);
      companion(width * 0.09, floor + 16);
      if (view.hazard && !view.completed) {
        const cloudX = width * 0.17;
        const cloudY = height * 0.1;
        rectangle(cloudX - 16, cloudY + 4, 38, 7, '#71828e');
        rectangle(cloudX - 8, cloudY, 18, 5, '#71828e');
        for (let i = 0; i < 5; i += 1) {
          const dropY = cloudY + 16 + (i * 9 + (reduced() ? 8 : animation * 24)) % 35;
          line([[cloudX - 12 + i * 7, dropY], [cloudX - 14 + i * 7, dropY + 4]], '#c3d6dd', 1);
        }
      }
      hotspot('crew', 'upgrade', 'Inspect crew', towerX - towerWidth / 2 - 15, workY - 23, towerWidth + 12, 31);
      hotspot('lift', 'upgrade', 'Inspect winch', craneX + 10, top + 9, 27, towerHeight - 5);
      hotspot('beacon', 'upgrade', 'Inspect beacon', towerX - 17, beaconY - 21, 34, 35);
    }

    function reduced() { return quiet || !!(motion && motion.matches); }
    function visible() { return !disposed && inView && !document.hidden && canvas.getClientRects().length > 0; }
    function draw(now) {
      const bounds = canvas.getBoundingClientRect();
      if (bounds.width < 1 || bounds.height < 1) return;
      const dpr = clamp(root.devicePixelRatio || 1, 1, 3);
      const physicalWidth = Math.round(bounds.width * dpr);
      const physicalHeight = Math.round(bounds.height * dpr);
      // Each logical art pixel maps to an integer number of device pixels.
      // A narrow, centered margin absorbs any non-divisible remainder.
      const scale = Math.max(1, Math.round(Math.max(1, Math.round(bounds.width / 192)) * dpr));
      width = Math.max(96, Math.floor(physicalWidth / scale));
      viewportHeight = Math.max(1, Math.floor(physicalHeight / scale));
      // The host's one-thumb choice button occupies the bottom of the world.
      // Reserve its space from the first frame so later unlocks cannot cover
      // workers/queues or force the scene composition to jump.
      height = Math.max(48, viewportHeight - Math.ceil(70 * dpr / scale));
      if (canvas.width !== physicalWidth || canvas.height !== physicalHeight) { canvas.width = physicalWidth; canvas.height = physicalHeight; }
      if (buffer.width !== width || buffer.height !== viewportHeight) { buffer.width = width; buffer.height = viewportHeight; }
      paint.imageSmoothingEnabled = false;
      context.imageSmoothingEnabled = false;
      hotspots = [];
      const palette = PALETTES[view.region];
      rectangle(0, 0, width, viewportHeight, view.kind === 'quarry' ? '#38434a' : palette.light);
      if (view.kind === 'quarry') drawQuarry(palette);
      else if (view.kind === 'watchtower') drawWatchtower(palette);
      else drawGreenway(palette);
      context.fillStyle = view.kind === 'quarry' ? '#243343' : palette.grass;
      context.fillRect(0, 0, physicalWidth, physicalHeight);
      context.drawImage(buffer, Math.floor((physicalWidth - width * scale) / 2), Math.floor((physicalHeight - viewportHeight * scale) / 2), width * scale, viewportHeight * scale);
      canvas.dataset.sceneKind = view.kind;
      canvas.dataset.sceneCheckpoint = String(view.checkpoint);
      canvas.dataset.sceneProgress = String(Math.round(view.progress * 100));
      canvas.dataset.scenePixelScale = String(scale);
      canvas.dataset.sceneReducedMotion = String(reduced());
      canvas.dataset.sceneDraws = String((Number(canvas.dataset.sceneDraws) || 0) + 1);
      lastPaint = now;
      dirty = false;
    }
    function stop() {
      if (frame !== null) root.cancelAnimationFrame(frame);
      frame = null;
      previousTime = null;
    }
    function tick(now) {
      frame = null;
      if (!visible()) { previousTime = null; return; }
      const delta = previousTime === null ? 0 : Math.min(0.2, Math.max(0, (now - previousTime) / 1000));
      previousTime = now;
      if (!reduced()) {
        animation += delta;
        if (view.kind === 'quarry' && haulRate() > 0) cartPhase += delta / haulPeriod();
        deliveredFlash = Math.max(0, deliveredFlash - delta * 1.5);
      }
      if (dirty || !reduced() && now - lastPaint >= 100) draw(now);
      if (!reduced() && !view.completed) frame = root.requestAnimationFrame(tick);
    }
    function schedule() {
      if (visible() && frame === null) frame = root.requestAnimationFrame(tick);
    }
    function onVisibility() { if (visible()) { dirty = true; schedule(); } else stop(); }
    function onResize() { dirty = true; schedule(); }
    function onMotion() { stop(); dirty = true; schedule(); }
    function select(event) {
      if (disposed || !pointerStart || Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 12) { pointerStart = null; return; }
      pointerStart = null;
      const bounds = canvas.getBoundingClientRect();
      const x = (event.clientX - bounds.left) / bounds.width;
      const y = (event.clientY - bounds.top) / bounds.height;
      const hit = hotspots.slice().reverse().find(item => x >= item.x && x <= item.x + item.width && y >= item.y && y <= item.y + item.height);
      if (hit && typeof settings.onSelect === 'function') settings.onSelect({ kind: hit.kind, id: hit.id });
    }
    function beginPointer(event) { pointerStart = { x: event.clientX, y: event.clientY }; }
    function cancelPointer() { pointerStart = null; }
    function loadImage(path, receive) {
      if (!root.Image) return;
      const image = new root.Image();
      pendingImages.push(image);
      image.onload = () => { if (!disposed && image.naturalWidth) { receive(image); dirty = true; schedule(); } };
      image.onerror = () => {};
      image.src = path;
    }
    const base = String(settings.assetBase || 'img/wayfarers-guild/').replace(/\/?$/, '/');
    loadImage(base + 'actors.png?v=51f333248522', image => { actorImage = image; });
    loadImage(settings.art || base + 'expedition-props.png?v=a86c28a4803c', image => { optionalArt = image; });
    document.addEventListener('visibilitychange', onVisibility);
    if (motion && motion.addEventListener) motion.addEventListener('change', onMotion);
    canvas.addEventListener('pointerdown', beginPointer);
    canvas.addEventListener('pointerup', select);
    canvas.addEventListener('pointercancel', cancelPointer);
    const resize = root.ResizeObserver ? new root.ResizeObserver(onResize) : null;
    if (resize) resize.observe(canvas);
    else root.addEventListener('resize', onResize);
    const intersection = root.IntersectionObserver ? new root.IntersectionObserver(entries => { inView = entries.some(entry => entry.isIntersecting); onVisibility(); }) : null;
    if (intersection) intersection.observe(canvas);
    schedule();

    return {
      update(next) {
        if (disposed) return;
        const incoming = normalizeView(next);
        if (JSON.stringify(incoming) === JSON.stringify(view)) return;
        if (incoming.kind !== view.kind || incoming.index !== view.index) { animation = 0; cartPhase = 0; previousDelivered = incoming.progress; }
        if (incoming.kind === 'quarry' && incoming.progress > previousDelivered + 0.007) { deliveredFlash = 1; previousDelivered = incoming.progress; }
        view = incoming;
        dirty = true;
        schedule();
      },
      setQuiet(value) { const next = value === true; if (quiet !== next) { quiet = next; onMotion(); } },
      getHotspots() { return hotspots.map(item => Object.assign({}, item)); },
      getStatus() { return { state: disposed ? 'destroyed' : 'ready', kind: view.kind, progress: view.progress, reducedMotion: reduced(), draws: Number(canvas.dataset.sceneDraws) || 0, animated: frame !== null && !reduced(), hasArt: !!optionalArt }; },
      dispose() {
        if (disposed) return;
        disposed = true;
        stop();
        if (resize) resize.disconnect();
        if (intersection) intersection.disconnect();
        pendingImages.forEach(image => { image.onload = null; image.onerror = null; });
        pendingImages = [];
        document.removeEventListener('visibilitychange', onVisibility);
        if (motion && motion.removeEventListener) motion.removeEventListener('change', onMotion);
        root.removeEventListener('resize', onResize);
        canvas.removeEventListener('pointerdown', beginPointer);
        canvas.removeEventListener('pointerup', select);
        canvas.removeEventListener('pointercancel', cancelPointer);
        canvas.dataset.sceneStatus = 'destroyed';
      }
    };
  }

  const api = { create, normalizeView };
  root.WayfarersExpeditionScene = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
