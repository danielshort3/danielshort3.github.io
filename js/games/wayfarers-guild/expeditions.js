(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers);
  if (common) module.exports = api;
  if (root) root.WayfarersExpeditions = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N) {
  'use strict';
  const EPS = 1e-8;
  const KINDS = ['greenway', 'quarry', 'watchtower'];
  const REGIONS = ['Greenway', 'Copper Hills', 'Mistwood', 'Frostpass', 'Sunken Reach', 'Starfall Heights'];
  const TRACKS = {
    greenway: [
      { id: 'boots', label: 'Boots', icon: 'boots', base: 6, growth: 1.75, effect: 'Faster travel', milestone: 'Shortcut: travel +25%' },
      { id: 'porters', label: 'Porters', icon: 'coins', base: 10, growth: 1.85, effect: 'More coins, faster repairs', milestone: 'Double bundles: income +35%' },
      { id: 'scouts', label: 'Scouts', icon: 'maps', base: 18, growth: 1.8, effect: 'Faster route and bridge work', milestone: 'Alternate crossing: bridge work +50%' }
    ],
    quarry: [
      { id: 'picks', label: 'Picks', icon: 'mine', base: 20, growth: 1.65, effect: 'More ore extracted', milestone: 'Rich vein: extraction +50%' },
      { id: 'carts', label: 'Carts', icon: 'ore', base: 20, growth: 1.65, effect: 'More ore delivered', milestone: 'Second cart: hauling +50%' },
      { id: 'furnace', label: 'Furnace', icon: 'forge', base: 20, growth: 1.65, effect: 'More ingots smelted', milestone: 'Batch recipe: smelting +50%' }
    ],
    watchtower: [
      { id: 'crew', label: 'Crew', icon: 'hall', base: 36, growth: 1.6, effect: 'More workers and coins', milestone: 'Foreman: one extra worker' },
      { id: 'lift', label: 'Lift', icon: 'equipment', base: 40, growth: 1.65, effect: 'Faster construction', milestone: 'Counterweight: repairs +50%' },
      { id: 'beacon', label: 'Beacon', icon: 'knowledge', base: 40, growth: 1.65, effect: 'Protection and faster finale', milestone: 'Storm lens: pressure −1' }
    ]
  };
  const ALL_TRACKS = Object.values(TRACKS).flat().map(item => item.id);
  const BLUEPRINTS = [
    { id: 'caravan', label: 'Trade roads', icon: 'coins', effectText: '+20% expedition coin income', description: 'Porters and established outposts support faster local investment.' },
    { id: 'engineering', label: 'Guild engineering', icon: 'forge', effectText: '+15% construction and throughput', description: 'Quarries process material faster; bridges and towers are built faster.' },
    { id: 'pathfinding', label: 'Wayfinder charts', icon: 'maps', effectText: '+15% travel and +1 protection', description: 'Find quicker paths and protect tower crews from regional hazards.' }
  ];
  const entitlements = new WeakMap();
  const setEntitlements = (parent, ids) => entitlements.set(parent, new Set(ids));
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const exact = (value, keys) => object(value) && Object.keys(value).length === keys.length && keys.every(key => Object.prototype.hasOwnProperty.call(value, key));
  const finite = (value, max = 1e15, integer = false) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= max && (!integer || Number.isSafeInteger(value));
  const clone = value => JSON.parse(JSON.stringify(value));
  const kind = x => KINDS[x.index % 3];
  const rank = (x, id) => x.ranks[id] || 0;
  const sumRanks = x => ALL_TRACKS.reduce((sum, id) => sum + rank(x, id), 0);
  const stageScale = x => x.index < 3 ? 1 : 3.5 + Math.min(20, Math.floor(x.index / 3) * 0.65);
  const moneyScale = x => N.pow(4, Math.floor(x.index / 3));
  const def = (x, id) => TRACKS[kind(x)].find(item => item.id === id);
  const blueprint = (x, id) => x.blueprints.includes(id);
  const counts = cleared => Object.fromEntries(KINDS.map((id, offset) => [id, Math.max(0, Math.floor((cleared - offset) / 3) + 1)]));
  const progress = x => {
    const target = targets(x);
    return x.completed ? 1 : Math.min(1, 0.78 * x.work / target.work + 0.22 * x.finaleWork / target.finale);
  };
  function targets(x) {
    const scale = stageScale(x);
    return kind(x) === 'greenway' ? { work: 115 * scale, finale: 42 * scale }
      : kind(x) === 'quarry' ? { work: 110 * scale, finale: 30 * scale }
        : { work: 450 * scale, finale: 120 * scale };
  }
  function create(parent) {
    const index = parent && parent.route ? parent.route.index : 0;
    const cleared = parent && parent.lifetime ? parent.lifetime.highestRoute : -1;
    const x = {
      version: 1, index, cleared, completed: false, elapsed: 0, work: 0, finaleWork: 0,
      buffers: { ore: 0, smelt: 0 }, ranks: Object.fromEntries(ALL_TRACKS.map(id => [id, 0])),
      choices: { route: 'short', smelting: 'throughput', equipment: 'tools', allocation: 'balanced' },
      blueprints: [], mastery: { greenway: 0, quarry: 0, watchtower: 0 },
      automation: { enabled: false, priority: 'balanced', dispatch: false },
      sequence: 0, seen: 0, recent: [], purchases: 0, stagePurchases: 0
    };
    return x;
  }
  function hydrate(parent, fraction) {
    const x = create(parent), target = targets(x), p = Math.max(0, Math.min(0.999999, fraction || 0));
    x.work = Math.min(target.work, p / 0.78 * target.work);
    x.finaleWork = Math.max(0, (p - 0.78) / 0.22 * target.finale);
    return x;
  }
  function validate(x) {
    if (!exact(x, ['version', 'index', 'cleared', 'completed', 'elapsed', 'work', 'finaleWork', 'buffers', 'ranks', 'choices', 'blueprints', 'mastery', 'automation', 'sequence', 'seen', 'recent', 'purchases', 'stagePurchases'])) return false;
    if (x.version !== 1 || !finite(x.index, 1e9, true) || !Number.isSafeInteger(x.cleared) || x.cleared < -1 || x.cleared > 1e9 || x.index > x.cleared + 1 || typeof x.completed !== 'boolean' || !finite(x.elapsed, 8.64e12) || !finite(x.work) || !finite(x.finaleWork)) return false;
    if (!exact(x.ranks, ALL_TRACKS) || ALL_TRACKS.some(id => !finite(x.ranks[id], 12, true) || !def(x, id) && x.ranks[id] !== 0)) return false;
    if (!exact(x.buffers, ['ore', 'smelt']) || !finite(x.buffers.ore, 10 + x.ranks.carts * 3 + EPS) || !finite(x.buffers.smelt, 10 + x.ranks.carts * 3 + EPS) || kind(x) !== 'quarry' && (x.buffers.ore || x.buffers.smelt)) return false;
    if (!exact(x.choices, ['route', 'smelting', 'equipment', 'allocation']) || !['short', 'supply'].includes(x.choices.route) || !['throughput', 'quality'].includes(x.choices.smelting) || !['tools', 'boots'].includes(x.choices.equipment) || !['balanced', 'repair', 'protect'].includes(x.choices.allocation)) return false;
    if (!Array.isArray(x.blueprints) || new Set(x.blueprints).size !== x.blueprints.length || x.blueprints.some(id => !BLUEPRINTS.some(item => item.id === id)) || x.blueprints.length > Math.floor((x.cleared + 1) / 3)) return false;
    if (!exact(x.mastery, KINDS) || KINDS.some(id => !finite(x.mastery[id], 1e6, true))) return false;
    if (!exact(x.automation, ['enabled', 'priority', 'dispatch']) || typeof x.automation.enabled !== 'boolean' || typeof x.automation.dispatch !== 'boolean' || !['balanced', 'progress', 'income'].includes(x.automation.priority) || x.cleared < 2 && (x.automation.enabled || x.automation.dispatch)) return false;
    if (!finite(x.sequence, 1e12, true) || !finite(x.seen, x.sequence, true) || !finite(x.purchases, 1e12, true) || !finite(x.stagePurchases, 36, true) || x.stagePurchases !== sumRanks(x) || x.purchases < x.stagePurchases) return false;
    if (!Array.isArray(x.recent) || x.recent.length > 12 || x.recent.some((event, i) => !exact(event, ['sequence', 'kind', 'title', 'text', 'stage']) || !finite(event.sequence, x.sequence, true) || !event.sequence || i && event.sequence <= x.recent[i - 1].sequence || !['milestone', 'stage', 'blueprint'].includes(event.kind) || !finite(event.stage, 1e9, true) || typeof event.title !== 'string' || event.title.length > 100 || typeof event.text !== 'string' || event.text.length > 300)) return false;
    const t = targets(x);
    return x.work <= t.work + EPS && x.finaleWork <= t.finale + EPS && (!x.completed || x.index <= x.cleared && x.work >= t.work - EPS && x.finaleWork >= t.finale - EPS) && (x.work >= t.work - EPS || x.finaleWork === 0);
  }
  function event(x, type, title, text) {
    x.sequence += 1;
    x.recent.push({ sequence: x.sequence, kind: type, title, text, stage: x.index });
    x.recent = x.recent.slice(-12);
  }
  function visible(x, id) {
    if (!def(x, id)) return false;
    if (kind(x) !== 'greenway' || x.index > 0) return true;
    return id === 'boots' || id === 'porters' && (rank(x, 'boots') >= 2 || x.work >= 28) || id === 'scouts' && (sumRanks(x) >= 5 || x.work >= 70);
  }
  function cost(x, id) {
    const d = def(x, id);
    if (!d) return N.zero();
    const amount = N.mul(moneyScale(x), N.mul(d.base * (x.index >= 3 ? 3.1 : 1), N.pow(d.growth, rank(x, id))));
    return N.cmp(amount, 1e6) < 0 ? N.from(Math.ceil(N.toNumber(amount) - 1e-9)) : amount;
  }
  function localRates(parent) {
    const x = parent.expedition, r = x.ranks, t = targets(x), type = kind(x);
    const engineering = blueprint(x, 'engineering') ? 1.15 : 1;
    const pace = blueprint(x, 'pathfinding') ? 1.15 : 1;
    const mastery = 1 + Math.min(0.5, x.mastery[type] * 0.02);
    const crew = parent.challenges.active === 'quiet-company' ? [] : parent.crew.specialists;
    const companion = parent.challenges.active === 'quiet-company' ? null : parent.crew.companion;
    const equipped = id => parent.challenges.active === 'old-tools' ? 0 : parent.upgrades[id] || 0;
    const globalTools = (1 + Math.sqrt(equipped('gear-tools')) * 0.12) * (crew.includes('prospector') ? 1.35 : 1) * (companion === 'tortoise' ? 1.2 : 1);
    const paid = entitlements.get(parent);
    const owns = id => parent.premium.owned.includes(id) || !!(paid && paid.has(id));
    const floor = parent.guild.plan.reserves.provisions;
    const supplied = parent.guild.supply !== 'save' && parent.challenges.active !== 'light-pack' && N.cmp(parent.resources.provisions, floor) > 0;
    const prepared = id => parent.guild.prepared.kinds.includes(id);
    const mealStrength = parent.guild.supply === 'push' ? 1.5 : 1;
    const collection = Math.pow(1.08, parent.collections.length);
    const travelKit = parent.luck.active === 'living-crucible' && parent.luck.kit.active === 'travel' && parent.luck.kit.remainingSeconds > 0;
    const miningKit = parent.luck.active === 'living-crucible' && parent.luck.kit.active === 'mining' && parent.luck.kit.remainingSeconds > 0;
    const globalBoots = (1 + Math.sqrt(equipped('gear-boots')) * 0.12 + Math.sqrt(parent.refitUpgrades.pace) * 0.08 + Math.sqrt(parent.legacy.waystones) * 0.1 + Math.sqrt(parent.upgrades.preparation) * 0.06)
      * (crew.includes('scout') ? 1.3 : 1) * (companion === 'fox' ? 1.15 : 1) * (owns('compass') ? 1.1 : 1)
      * (supplied && parent.meal === 'meal-travel' ? 1 + .25 * mealStrength : 1)
      * (prepared('scout') ? 1.35 : 1) * (prepared('scout') && parent.luck.active === 'frost-compass' ? 1.4 : 1)
      * (parent.luck.active === 'wayfarer-standard' && parent.guild.prepared.kinds.length ? 1.25 : 1)
      * (travelKit ? 2 : 1) * (parent.doctrine === 'industry' ? .85 : parent.doctrine === 'expedition' ? 1.35 : 1)
      * (parent.route.mode === 'supply' ? .6 : parent.route.mode === 'discovery' ? .7 : 1)
      * (parent.luck.hunt !== 'balanced' ? .85 : 1) * collection * (1 + Math.min(1, counts(x.cleared).watchtower * .03));
    const regional = Math.floor(x.index / 3) % 3;
    const result = { work: 0, finale: 0, income: N.zero(), ore: 0, smelt: 0, capacity: 10, picks: 0, carts: 0, furnace: 0, actualPicks: 0, actualCarts: 0, actualFurnace: 0, travel: 0, repair: 0, beacon: 0, workers: { repair: 0, protection: 0, total: 0 }, pressure: 0, wave: false, bottleneck: null };
    if (x.completed) return result;
    let income = 0;
    if (type === 'greenway') {
      result.travel = (1 + r.boots * 0.24) * (1 + r.scouts * 0.12) * (r.boots >= 3 ? 1.25 : 1) * (x.choices.route === 'supply' ? 0.88 : 1.08) * pace * globalBoots * mastery;
      result.repair = (0.8 + r.porters * 0.26 + r.boots * 0.08) * (r.scouts >= 3 ? 1.5 : 1) * engineering * mastery;
      if (x.index >= 3 && regional === 1 && x.choices.route === 'short') result.travel *= r.scouts >= 3 ? 1.25 : 0.72;
      if (x.index >= 3 && regional === 2) result.repair *= r.scouts >= 3 ? 1.15 : 0.7;
      income = 0.7 * Math.pow(1.2, r.boots) * Math.pow(1.35, r.porters) * (r.porters >= 3 ? 1.35 : 1) * (x.choices.route === 'supply' ? 1.45 : 1);
      result.work = result.travel; result.finale = result.repair;
    } else if (type === 'quarry') {
      const m = 0.6 * Math.pow(1.35, r.picks) * (r.picks >= 3 ? 1.5 : 1) * engineering * globalTools * mastery * (owns('artisan') ? 1.1 : 1) * (supplied && parent.meal === 'meal-mining' ? 1 + .25 * mealStrength : 1) * (parent.luck.active === 'golden-pickaxe' ? 1.2 : 1) * (miningKit ? 2 : 1) * (prepared('supply') ? 1.5 : 1) * collection;
      const h = 0.45 * Math.pow(1.4, r.carts) * (r.carts >= 3 ? 1.5 : 1) * engineering * mastery * (x.index >= 3 && regional === 1 ? 0.7 : 1);
      const f = 0.38 * Math.pow(1.4, r.furnace) * (r.furnace >= 3 ? 1.5 : 1) * (x.choices.smelting === 'quality' ? 0.76 : 1) * engineering * mastery * (parent.research.includes('efficient-smelting') ? 1.25 : 1) * (crew.includes('quartermaster') ? 1.15 : 1) * (x.index >= 3 && regional === 2 ? 0.65 : 1);
      result.capacity = 10 + r.carts * 3;
      result.picks = m; result.carts = h; result.furnace = f;
      result.bottleneck = m <= h && m <= f ? 'picks' : h <= f ? 'carts' : 'furnace';
      let haul = x.buffers.ore > EPS ? h : Math.min(h, m);
      const smelt = x.buffers.smelt > EPS ? f : Math.min(f, haul);
      if (x.buffers.smelt >= result.capacity - EPS) haul = Math.min(haul, smelt);
      const mine = x.buffers.ore >= result.capacity - EPS ? Math.min(m, haul) : m;
      result.actualPicks = mine; result.actualCarts = haul; result.actualFurnace = smelt;
      result.ore = mine - haul; result.smelt = haul - smelt;
      result.work = smelt; result.finale = smelt * 1.15;
      income = 1.1 + smelt * (x.choices.smelting === 'quality' ? 2.6 : 1.5);
    } else {
      const total = 3 + r.crew + (r.crew >= 3 ? 1 : 0);
      const protection = x.choices.allocation === 'repair' ? Math.max(1, Math.floor(total * 0.2)) : x.choices.allocation === 'protect' ? Math.ceil(total * 0.6) : Math.ceil(total * 0.4);
      const builders = total - protection;
      const wave = Math.floor(x.elapsed / 35) % 3 === 2;
      const pressure = wave ? 4 + Math.floor(x.index / 3) * 0.3 : x.index >= 3 && regional === 2 ? 1.8 : 0.8;
      const shielding = r.beacon * 0.25 + (r.beacon >= 3 ? 1 : 0) + (blueprint(x, 'pathfinding') ? 1 : 0);
      const efficiency = Math.min(1, (protection + shielding) / pressure);
      result.workers = { repair: builders, protection, total };
      result.pressure = pressure; result.wave = wave;
      result.repair = builders * 0.42 * Math.pow(1.2, r.lift) * (r.lift >= 3 ? 1.5 : 1) * Math.max(0.2, efficiency) * engineering * mastery * (crew.includes('quartermaster') ? 1.15 : 1);
      result.beacon = (0.7 + r.beacon * 0.35 + builders * 0.15) * Math.max(0.35, efficiency) * engineering * mastery;
      result.work = result.repair; result.finale = result.beacon;
      income = 2.2 + total * 0.3 + r.lift * 0.1;
    }
    result.income = N.mul(moneyScale(x), income * (blueprint(x, 'caravan') ? 1.2 : 1));
    if (x.work >= t.work - EPS) result.work = 0;
    else result.finale = 0;
    return result;
  }
  function contribution(parent) {
    const x = parent.expedition;
    if (!x) return { coins: N.zero(), ore: N.zero(), production: 1, travel: 1 };
    const c = counts(x.cleared), rates = localRates(parent);
    return { coins: N.add(rates.income, c.greenway * 0.25), ore: N.from(c.quarry * 0.012), production: 1 + Math.min(1, c.quarry * 0.025), travel: 1 + Math.min(1, c.watchtower * 0.03) };
  }
  function nextEvent(parent, gains) {
    const x = parent.expedition;
    if (!x || x.completed) return Infinity;
    const r = localRates(parent), t = targets(x);
    let seconds = x.work < t.work - EPS ? (t.work - x.work) / r.work : (t.finale - x.finaleWork) / r.finale;
    if (kind(x) === 'quarry') {
      for (const id of ['ore', 'smelt']) {
        if (r[id] > EPS) seconds = Math.min(seconds, Math.max(EPS, (r.capacity - x.buffers[id]) / r[id]));
        else if (r[id] < -EPS) seconds = Math.min(seconds, Math.max(EPS, x.buffers[id] / -r[id]));
      }
    }
    if (kind(x) === 'watchtower') seconds = Math.min(seconds, 35 - x.elapsed % 35);
    if (x.automation.enabled && x.cleared >= 2) {
      for (const item of TRACKS[kind(x)].filter(item => visible(x, item.id) && rank(x, item.id) < 12)) {
        const due = N.sub(cost(x, item.id), parent.resources.coins);
        const wait = N.cmp(cost(x, item.id), parent.resources.coins) <= 0 ? EPS : N.toNumber(N.div(due, gains.coins));
        seconds = Math.min(seconds, Math.max(EPS, wait));
      }
    }
    return Number.isFinite(seconds) ? Math.max(EPS, seconds) : Infinity;
  }
  function tick(parent, seconds) {
    const x = parent.expedition;
    if (!x || x.completed) return;
    const r = localRates(parent), t = targets(x);
    x.elapsed += seconds;
    x.work = Math.min(t.work, x.work + r.work * seconds);
    x.finaleWork = Math.min(t.finale, x.finaleWork + r.finale * seconds);
    if (kind(x) === 'quarry') {
      x.buffers.ore = Math.max(0, Math.min(r.capacity, x.buffers.ore + r.ore * seconds));
      x.buffers.smelt = Math.max(0, Math.min(r.capacity, x.buffers.smelt + r.smelt * seconds));
    }
  }
  function finish(parent) {
    const x = parent.expedition, t = targets(x);
    if (x.completed || x.work < t.work - EPS || x.finaleWork < t.finale - EPS) return false;
    x.work = t.work; x.finaleWork = t.finale; x.completed = true;
    const first = x.index > x.cleared;
    x.cleared = Math.max(x.cleared, x.index);
    x.mastery[kind(x)] = Math.min(1e6, x.mastery[kind(x)] + 1);
    if (first && kind(x) === 'quarry') parent.upgrades[x.choices.equipment === 'tools' ? 'gear-tools' : 'gear-boots'] += 1;
    event(x, 'stage', stageName(x) + ' complete', first ? 'A productive outpost now supports your guild.' : 'Outpost mastery improved. The guild keeps producing while you prepare the next expedition.');
    return true;
  }
  function restart(parent, index) {
    const old = parent.expedition || create(parent), next = create(parent);
    next.index = index;
    for (const key of ['cleared', 'blueprints', 'mastery', 'automation', 'sequence', 'seen', 'recent', 'purchases']) next[key] = clone(old[key]);
    parent.expedition = next;
  }
  function autoBuy(parent) {
    const x = parent.expedition;
    if (!x || x.completed || !x.automation.enabled || x.cleared < 2) return;
    const r = localRates(parent);
    const best = kind(x) === 'quarry' ? r.bottleneck : kind(x) === 'greenway' ? x.automation.priority === 'income' ? 'porters' : 'boots' : x.automation.priority === 'income' ? 'crew' : x.work >= targets(x).work ? 'beacon' : 'lift';
    const items = TRACKS[kind(x)].slice().sort((a, b) => x.automation.priority === 'balanced' ? rank(x, a.id) - rank(x, b.id) : (a.id === best ? -1 : 0) - (b.id === best ? -1 : 0));
    for (const item of items) if (visible(x, item.id) && rank(x, item.id) < 12 && N.cmp(parent.resources.coins, cost(x, item.id)) >= 0) act(parent, { type: 'expedition-buy', id: item.id });
  }
  function act(parent, action) {
    const x = parent.expedition;
    if (!x) return { ok: false, message: 'Expedition progress is unavailable.' };
    if (action.type === 'expedition-buy') {
      const item = def(x, action.id);
      if (!item || !visible(x, action.id) || x.completed || rank(x, action.id) >= 12) return { ok: false, message: 'That upgrade is not available in this expedition.' };
      const price = cost(x, action.id);
      if (N.cmp(parent.resources.coins, price) < 0) return { ok: false, message: 'Need ' + N.format(price) + ' coins.' };
      parent.resources.coins = N.sub(parent.resources.coins, price);
      x.ranks[action.id] += 1; x.purchases += 1; x.stagePurchases += 1;
      if (rank(x, action.id) === 3) event(x, 'milestone', item.label + ' milestone', item.milestone);
      return { ok: true, message: item.label + ' improved to rank ' + rank(x, action.id) + '.' };
    }
    if (action.type === 'expedition-choice') {
      if (x.completed) return { ok: false, message: 'Choose an approach on your next expedition.' };
      const type = kind(x);
      if (type === 'greenway' && (x.index > 0 || sumRanks(x) >= 2 || x.work >= 28) && ['short', 'supply'].includes(action.id)) x.choices.route = action.id;
      else if (type === 'quarry' && ['throughput', 'quality'].includes(action.id)) x.choices.smelting = action.id;
      else if (type === 'quarry' && ['tools', 'equipment-boots'].includes(action.id)) x.choices.equipment = action.id === 'tools' ? 'tools' : 'boots';
      else if (type === 'watchtower' && ['balanced', 'repair', 'protect'].includes(action.id)) x.choices.allocation = action.id;
      else return { ok: false, message: 'That approach has not opened here.' };
      return { ok: true, message: 'Expedition approach updated. Your crew continues automatically.' };
    }
    if (action.type === 'expedition-blueprint') {
      const item = BLUEPRINTS.find(item => item.id === action.id);
      if (!item || x.blueprints.includes(action.id) || x.blueprints.length >= Math.floor((x.cleared + 1) / 3)) return { ok: false, message: 'Complete another region to choose this blueprint.' };
      x.blueprints.push(action.id); event(x, 'blueprint', item.label, item.effectText);
      return { ok: true, message: item.effectText + '. Retained through Refits and Charters.' };
    }
    if (action.type === 'expedition-automation') {
      if (x.cleared < 2 || typeof action.enabled !== 'boolean' || !['balanced', 'progress', 'income'].includes(action.priority) || action.dispatch !== undefined && typeof action.dispatch !== 'boolean') return { ok: false, message: 'Complete the Watchtower to earn expedition automation.' };
      x.automation.enabled = action.enabled; x.automation.priority = action.priority;
      if (action.dispatch !== undefined) x.automation.dispatch = action.dispatch;
      return { ok: true, message: 'Expedition plan saved. It works while you are away.' };
    }
    if (action.type === 'expedition-seen' && finite(action.sequence, x.sequence, true)) { x.seen = Math.max(x.seen, action.sequence); return { ok: true, message: '' }; }
    return { ok: false, message: 'Unknown expedition action.' };
  }
  function stageName(x) {
    const names = ['Old Footpath', 'Copper Quarry', 'Watchtower Road'];
    if (x.index < 3) return names[x.index];
    return REGIONS[Math.min(5, Math.floor(x.index / 3))] + ' ' + ['Crossing', 'Works', 'Beacon'][x.index % 3] + (x.index >= 18 ? ' ' + (Math.floor(x.index / 3) - 4) : '');
  }
  function view(parent) {
    const x = parent.expedition;
    if (!x) return { local: false };
    const type = kind(x), r = localRates(parent), t = targets(x), p = progress(x), finale = x.work >= t.work - EPS;
    const region = REGIONS[Math.min(5, Math.floor(x.index / 3))];
    const labels = type === 'greenway' ? ['Leave camp', 'Find the crossing', 'Repair the bridge', 'Bridge crossed'] : type === 'quarry' ? ['Open the vein', 'Balance deliveries', 'Restart the lift', 'Lift restored'] : ['Set up the crew', 'Raise the tower', 'Relight the beacon', 'Beacon lit'];
    const options = (entries, selected) => entries.map(([id, label, effectText]) => ({ id, label, effectText, selected: selected === id, action: { type: 'expedition-choice', id } }));
    const roughPath = x.index >= 3 && Math.floor(x.index / 3) % 3 === 1 && rank(x, 'scouts') < 3;
    const choice = type === 'greenway' ? { id: 'route', title: 'Choose your path', visible: x.index > 0 || sumRanks(x) >= 2 || x.work >= 28, options: options([['short', 'Short path', roughPath ? 'Rough until Scout rank 3' : 'Faster crossing'], ['supply', 'Supply detour', roughPath ? '+45% expedition coins; bypasses the rough path' : '+45% expedition coins; slower travel']], x.choices.route) }
      : type === 'quarry' ? { id: 'smelting', title: 'Furnace plan', visible: true, options: options([['throughput', 'Fast batches', 'More ingots toward the lift'], ['quality', 'Fine ingots', 'Slower smelting; more coins per ingot']], x.choices.smelting) }
        : { id: 'allocation', title: r.wave ? 'Protect the crew' : 'Assign your crew', visible: true, options: options([['balanced', 'Balanced', 'Safe automatic progress'], ['repair', 'Build', 'More builders; exposed during hazards'], ['protect', 'Guard', 'Fewer builders; stronger hazard protection']], x.choices.allocation) };
    const equipment = type === 'quarry' ? { id: 'equipment', title: 'Forge your completion reward', visible: p >= 0.35, options: options([['tools', 'Mining tools', '+1 guild mining-tools rank'], ['equipment-boots', 'Travel boots', '+1 guild expedition-boots rank']], x.choices.equipment === 'tools' ? 'tools' : 'equipment-boots') } : null;
    const cards = TRACKS[type].map(item => {
      const price = cost(x, item.id), old = r, copy = Object.assign({}, parent, { expedition: Object.assign({}, x, { ranks: Object.assign({}, x.ranks, { [item.id]: rank(x, item.id) + 1 }) }) });
      if (entitlements.has(parent)) entitlements.set(copy, entitlements.get(parent));
      const improved = localRates(copy), field = item.id === 'boots' || item.id === 'scouts' ? finale ? 'repair' : 'travel' : item.id === 'porters' ? 'income' : type === 'quarry' ? item.id : item.id === 'beacon' ? 'beacon' : 'repair';
      const currentRate = field === 'income' ? N.toNumber(old.income) : old[field];
      const nextRate = field === 'income' ? N.toNumber(improved.income) : improved[field];
      const chainOutput = type === 'quarry' ? { current: Math.min(old.picks, old.carts, old.furnace), next: Math.min(improved.picks, improved.carts, improved.furnace) } : null;
      return { id: item.id, label: item.label, icon: item.icon, rank: rank(x, item.id), maxRank: 12, visible: visible(x, item.id), disabled: x.completed || rank(x, item.id) >= 12 || N.cmp(parent.resources.coins, price) < 0, cost: [{ resource: 'coins', amount: price, text: N.format(price) + ' coins' }], effectText: type === 'greenway' && finale && field === 'repair' ? 'Bridge repair capacity' : item.effect, comparison: currentRate.toFixed(2) + ' → ' + nextRate.toFixed(2) + '/s', chainOutput, nextMilestone: rank(x, item.id) < 3 ? { rank: 3, label: item.milestone, remaining: 3 - rank(x, item.id) } : null, action: { type: 'expedition-buy', id: item.id } };
    });
    const stations = type === 'quarry' ? [['picks', 'Mine', r.actualPicks, r.picks, x.buffers.ore], ['carts', 'Cart', r.actualCarts, r.carts, x.buffers.smelt], ['furnace', 'Furnace', r.actualFurnace, r.furnace, x.work]].map(([id, label, rate, maxRate, buffer]) => ({ id, label, rate, maxRate, rateText: rate.toFixed(2) + '/s', buffer, capacity: id === 'furnace' ? t.work : r.capacity, bottleneck: id === r.bottleneck, status: rate < maxRate - EPS ? id === 'picks' ? 'Buffer full' : 'Waiting for supply' : id === r.bottleneck ? 'Bottleneck' : 'Working' })) : [];
    const outposts = Object.entries(counts(x.cleared)).filter(([, count]) => count).map(([id, count]) => ({ id, count, label: { greenway: 'Trail outpost', quarry: 'Working mine', watchtower: 'Guard post' }[id], effectText: id === 'greenway' ? '+' + (count * .25).toFixed(2) + ' guild coins/s' : id === 'quarry' ? '+' + (count * .012).toFixed(3) + ' ore/s; stronger production' : '+' + Math.min(100, count * 3) + '% Greenway travel', mastery: x.mastery[id] }));
    return {
      local: true, stage: { id: 'expedition-' + x.index, index: x.index, kind: type, name: stageName(x), region, objective: x.completed ? labels[3] : finale ? labels[2] : labels[p < .35 ? 0 : 1], progress: p, completed: x.completed, elapsed: x.elapsed },
      wallet: { resource: 'coins', amount: N.from(parent.resources.coins), formatted: N.format(parent.resources.coins), rateText: '+' + N.format(r.income) + '/s expedition' }, cards, checkpoints: labels.map((label, i) => ({ id: 'checkpoint-' + i, label, progress: Math.max(0, Math.min(1, (p - [0, .3, .78, 1][i]) / (i === 2 ? .22 : .3))), complete: p >= [0.3, .6, .78, 1][i] })),
      stations, choice, choices: equipment ? [choice, equipment] : [choice],
      finale: { ready: finale && !x.completed, completed: x.completed, label: labels[2], progress: x.finaleWork / t.finale, automatic: true, action: { type: 'expedition-complete' } },
      next: { label: 'Next expedition', description: 'Develop a fresh local crew. Your outposts, guild equipment, mastery and blueprints remain.', disabled: !x.completed, action: { type: 'expedition-next' } }, outposts,
      blueprints: BLUEPRINTS.map(item => Object.assign({}, item, { visible: x.cleared >= 2, owned: x.blueprints.includes(item.id), disabled: x.blueprints.includes(item.id) || x.blueprints.length >= Math.floor((x.cleared + 1) / 3), action: { type: 'expedition-blueprint', id: item.id } })),
      automation: Object.assign({ unlocked: x.cleared >= 2, choices: ['balanced', 'progress', 'income'].map(id => ({ id, label: id === 'progress' ? 'Finish stages' : id === 'income' ? 'Build income' : 'Balanced', action: { type: 'expedition-automation', enabled: true, priority: id } })) }, x.automation),
      events: clone(x.recent.filter(event => event.sequence > x.seen)), sequence: x.sequence, purchases: x.purchases,
      condition: x.index < 3 ? 'Learn the expedition' : type === 'greenway' ? Math.floor(x.index / 3) % 3 === 1 ? 'Rough direct path: supply detours avoid the slowdown' : Math.floor(x.index / 3) % 3 === 2 ? 'Hidden crossing: Scout rank 3 speeds the bridge' : 'Long crossing: develop porters and scouting' : type === 'quarry' ? Math.floor(x.index / 3) % 3 === 1 ? 'Steep haul: cart capacity is reduced' : Math.floor(x.index / 3) % 3 === 2 ? 'Hard ore: furnace throughput is reduced' : 'Deep workings: balance the full chain' : 'Regional hazards need stronger protection',
      guildLinks: ['Guild equipment supports travel and mining', 'Scouts, prospectors and quartermasters support expedition work', 'Supplied travel and mining meals improve local work', 'Efficient smelting improves furnace capacity'],
      scene: { kind: type, index: x.index, region: region.toLowerCase().replace(/ /g, '-'), progress: p, completed: x.completed, ranks: Object.fromEntries(TRACKS[type].map(item => [item.id, rank(x, item.id)])), route: x.choices.route, oreBuffer: x.buffers.ore, smeltBuffer: x.buffers.smelt, capacity: r.capacity, workers: r.workers, beacon: x.finaleWork / t.finale, checkpoint: p < .3 ? 0 : p < .6 ? 1 : p < .78 ? 2 : 3, bottleneck: r.bottleneck, allocation: x.choices.allocation, quality: x.choices.smelting, rates: { travel: r.travel, picks: r.picks, carts: r.carts, furnace: r.furnace, repair: r.repair, beacon: r.beacon }, flows: { picks: r.actualPicks, carts: r.actualCarts, furnace: r.actualFurnace }, unlocked: cards.filter(item => item.visible).map(item => item.id), pressure: r.pressure, hazard: r.wave, hazardIn: type === 'watchtower' ? 35 - x.elapsed % 35 : 0 }
    };
  }
  return { create, hydrate, validate, progress, targets, localRates, contribution, nextEvent, tick, finish, restart, autoBuy, act, view, stageName, setEntitlements };
});
