(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./collections.js') : root.WayfarersCollections);
  if (common) module.exports = api;
  if (root) root.WayfarersExpeditions = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, Collection) {
  'use strict';
  const EPS = 1e-8;
  let tierProvider = () => true;
  const setTierProvider = fn => { tierProvider = fn; };
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
    const scale = x.goalMode === 'regional' && x.index >= 3 ? Math.min(1e12, 4.2 * Math.pow(2.4, Math.min(40, Math.floor(x.index / 3) - 1))) : stageScale(x);
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
    const collectionGear = Collection.modifiers(parent);
    const globalBoots = (1 + Math.sqrt(equipped('gear-boots')) * 0.12 + Math.sqrt(parent.refitUpgrades.pace) * 0.08 + Math.sqrt(parent.legacy.waystones) * 0.1 + Math.sqrt(parent.upgrades.preparation) * 0.06) * (1 + (collectionGear.travel || 0) + (collectionGear.voyage || 0))
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
  // The v1 formulas and validator above are retained for honest save migration.
  // A v2 world owns three permanent establishments. Selection is presentation
  // state; the frontier project and every producing area advance independently.
  const AREA_NAMES = { greenway: 'Greenway', quarry: 'Copper Quarry', watchtower: 'Watchtower' };
  const AREA_ICONS = { greenway: 'trail', quarry: 'mine', watchtower: 'observatory' };
  const DEVELOPMENTS = [
    { id: 'trail-caravans', name: 'Caravan routes', group: 'transport', at: 1, costs: { coins: 65, ore: 2 }, from: ['quarry'], to: ['greenway', 'quarry'], kind: 'allocation', description: 'Open freight dispatch on the Greenway. Divert coin couriers to relieve the quarry cart bottleneck.' },
    { id: 'paved-roads', name: 'Paved roads', group: 'transport', at: 1, requires: ['trail-caravans'], costs: { coins: 180, ore: 5 }, from: ['quarry'], to: ['greenway'], kind: 'capacity', description: 'Quarry stone improves Greenway travel by 25% and makes each freight assignment carry more.' },
    { id: 'trail-depot', name: 'Cargo depot', group: 'transport', at: 1, requires: ['trail-caravans'], costs: { coins: 280, ore: 8 }, from: ['greenway', 'quarry'], to: ['quarry'], kind: 'storage', description: 'Add 24 units to both quarry queues. Surplus extraction can cover later hauling or smelting shortages.' },
    { id: 'efficient-crucibles', name: 'Efficient crucibles', group: 'metallurgy', at: 1, costs: { coins: 300, ore: 10 }, from: ['quarry'], to: ['quarry'], kind: 'conversion', description: 'Recover 15% more usable ore from every smelted unit, without increasing raw-ore demand.' },
    { id: 'tower-survey', name: 'Survey routes', group: 'discovery', at: 2, costs: { coins: 240, ore: 8 }, from: ['watchtower'], to: ['greenway'], kind: 'recipe', description: 'Open a Greenway survey route. Scouts produce maps instead of most coin deliveries.' },
    { id: 'quarry-precision', name: 'Precision smelting', group: 'metallurgy', at: 2, costs: { coins: 800, knowledge: 4 }, from: ['watchtower'], to: ['quarry'], kind: 'recipe', description: 'Open precision batches: 40% more ore per raw unit, with 40% lower furnace processing capacity. Best when raw ore is scarce.' },
    { id: 'signal-network', name: 'Courier signals', group: 'coordination', at: 3, costs: { coins: 550, maps: 2 }, from: ['watchtower'], to: ['greenway'], kind: 'allocation', description: 'Tower signal crews coordinate coin deliveries. A Signal assignment favours trade over quarry escorts.' },
    { id: 'protective-escorts', name: 'Quarry escorts', group: 'coordination', at: 3, costs: { coins: 550, knowledge: 5 }, from: ['watchtower'], to: ['quarry'], kind: 'allocation', description: 'Tower guards improve quarry hauling. A Guard assignment favours raw-material delivery over trade signals.' },
    { id: 'relay-network', name: 'Two-lane relay', group: 'coordination', at: 3, requires: ['trail-caravans', 'tower-survey'], costs: { coins: 1800, maps: 6, knowledge: 12 }, from: ['watchtower'], to: ['greenway', 'quarry'], kind: 'parallel', description: 'Open Relay dispatch: coin trade and freight run together at 80% strength. Pure trade or freight remains stronger for a single goal.' },
    { id: 'recovery-chutes', name: 'Overflow recovery', group: 'metallurgy', at: 3, costs: { coins: 900, ore: 25, knowledge: 8 }, from: ['quarry'], to: ['quarry', 'greenway'], kind: 'conversion', description: 'Sell excess newly mined rock when the raw queue is full. It earns a small coin return instead of stopping extraction.' },
    { id: 'dispatch-ledgers', name: 'Dispatch ledgers', group: 'automation', at: 3, costs: { coins: 1200, knowledge: 12 }, from: ['watchtower'], to: KINDS, kind: 'automation', description: 'Open Materials automation: fund the quarry bottleneck and freight porters before other area ranks. Developments remain your choice.' },
    { id: 'shared-workshops', name: 'Split-batch workshop', group: 'metallurgy', at: 4, requires: ['quarry-precision'], costs: { coins: 3500, ore: 50, knowledge: 25 }, from: ['watchtower', 'quarry'], to: ['quarry'], kind: 'parallel', description: 'Open mixed batches: divide the furnace equally between volume and precision processing, with intermediate yield and capacity.' },
    { id: 'deep-veins', name: 'Deep-vein surveys', group: 'expansion', at: 5, requires: ['tower-survey'], costs: { coins: 2400, ore: 40, maps: 5 }, from: ['greenway', 'watchtower'], to: ['quarry'], kind: 'unlock', description: 'Raise the quarry rank ceiling by 8 and improve extraction by 20%. Earlier mining infrastructure can keep growing.' },
    { id: 'survey-charters', name: 'Regional survey office', group: 'expansion', at: 5, requires: ['tower-survey'], costs: { coins: 5000, maps: 15, knowledge: 30 }, from: ['greenway'], to: ['greenway', 'watchtower'], kind: 'conversion', description: 'Survey dispatch also produces knowledge. Maps, trade coins and quarry materials continue funding regional expansion.' },
    { id: 'optical-foundry', name: 'Optical foundry', group: 'metallurgy', at: 8, requires: ['quarry-precision'], costs: { coins: 100000, ore: 800, knowledge: 250, maps: 30 }, from: ['quarry'], to: ['quarry', 'watchtower'], kind: 'recipe', description: 'Open optical batches: divert 70% of usable ore into Tower research. Beacon capacity limits how many optical parts can be studied.' },
    { id: 'trail-prospectors', name: 'Prospecting network', group: 'discovery', at: 11, charters: 1, requires: ['tower-survey'], costs: { coins: 1000000, maps: 300, knowledge: 1000 }, from: ['greenway'], to: ['quarry'], kind: 'allocation', description: 'While Greenway runs a survey route, Scout ranks also improve quarry extraction. Pure coin trade and freight keep their own advantages.' },
    { id: 'tower-control-room', name: 'Queue control room', group: 'automation', at: 14, charters: 2, requires: ['quarry-precision', 'signal-network'], costs: { coins: 10000000, ore: 100000, knowledge: 10000, maps: 1000 }, from: ['watchtower'], to: ['quarry'], kind: 'automation', description: 'Open Adaptive smelting. It runs volume batches while an input backlog exists, then precision when incoming raw supply is scarce.' },
    { id: 'survey-exchange', name: 'Survey exchange', group: 'coordination', at: 17, charters: 3, requires: ['relay-network', 'survey-charters'], costs: { coins: 100000000, maps: 20000, knowledge: 100000 }, from: ['watchtower'], to: ['greenway'], kind: 'parallel', description: 'Open Trade + survey dispatch: 75% of trade coins and 60% of survey maps together. It carries no quarry freight.' }
  ];
  let reserveProvider = parent => parent.guild.plan.reserves.coins;
  const setReserveProvider = provider => { reserveProvider = provider; };
  let rateProvider = null;
  const setRateProvider = provider => { rateProvider = provider; };
  const ownsDevelopment = (x, id) => x.developments.includes(id);
  const areaId = index => KINDS[index % 3];
  const expansionOf = (x, id) => Math.max(0, Math.floor((Math.max(x.cleared, x.index) - KINDS.indexOf(id)) / 3));
  const areaMoney = (x, id) => N.pow(4, x.areas[id].legacyRegion);
  const areaKeys = ['index', 'goalMode', 'legacyRegion', 'seenSequence', 'established', 'elapsed', 'work', 'finaleWork', 'buffers', 'ranks', 'choices', 'purchases'];
  function makeArea(index, established) {
    const seed = create({ route: { index }, lifetime: { highestRoute: -1 } });
    const t = targets(seed);
    return { index, goalMode: 'legacy', legacyRegion: 0, seenSequence: 0, established: !!established, elapsed: 0, work: established ? t.work : 0, finaleWork: established ? t.finale : 0, buffers: seed.buffers, ranks: seed.ranks,
      choices: Object.assign(seed.choices, { dispatch: 'trade' }), purchases: 0 };
  }
  function createWorld(parent) {
    const seed = create(parent);
    const x = { version: 2, index: seed.index, cleared: seed.cleared, completed: false, selectedArea: areaId(seed.index), areas: {}, developments: [],
      blueprints: seed.blueprints, mastery: seed.mastery, automation: seed.automation, sequence: 0, seen: 0, recent: [], purchases: 0 };
    for (let offset = 0; offset < KINDS.length; offset += 1) {
      if (offset > Math.max(x.index, x.cleared)) continue;
      const latest = offset + Math.max(0, Math.floor((x.cleared - offset) / 3)) * 3;
      x.areas[KINDS[offset]] = makeArea(offset === x.index % 3 ? x.index : latest, latest <= x.cleared);
    }
    return x;
  }
  function projectArea(x) { return x.areas[areaId(x.index)]; }
  function context(parent, id) {
    const x = parent.expedition, a = x.areas[id];
    if (!a) return null;
    const local = Object.assign({}, x, a, { version: 1, completed: false, stagePurchases: sumRanks(a) });
    const result = Object.assign({}, parent, { expedition: local });
    if (entitlements.has(parent)) entitlements.set(result, entitlements.get(parent));
    return result;
  }
  function hydrateWorld(parent, fraction) {
    const x = createWorld(parent), a = projectArea(x), t = targets(a), p = Math.max(0, Math.min(.999999, fraction || 0));
    a.work = Math.min(t.work, p / .78 * t.work);
    a.finaleWork = Math.max(0, (p - .78) / .22 * t.finale);
    return x;
  }
  function upgradeWorld(parent) {
    const old = parent.expedition;
    if (!old || old.version === 2) return old;
    if (!validate(old)) return null;
    const x = createWorld({ route: { index: old.index }, lifetime: { highestRoute: old.cleared } }), a = makeArea(old.index, old.cleared >= old.index % 3);
    for (const key of ['index', 'cleared', 'completed', 'blueprints', 'mastery', 'automation', 'sequence', 'seen', 'recent', 'purchases']) x[key] = clone(old[key]);
    for (const key of ['elapsed', 'work', 'finaleWork', 'buffers', 'ranks']) a[key] = clone(old[key]);
    a.choices = Object.assign({}, old.choices, { dispatch: 'trade' }); a.purchases = old.stagePurchases;
    a.legacyRegion = Math.floor(old.index / 3);
    a.seenSequence = old.seen;
    x.areas[areaId(old.index)] = a; x.selectedArea = areaId(old.index);
    return x;
  }
  function worldProgress(x) { return progress(Object.assign({}, projectArea(x), { completed: x.completed })); }
  function worldTargets(x) { return targets(x.version === 2 ? projectArea(x) : x); }
  function maxRank(x, id) { return Math.min(200, 12 + (x.areas[id].established ? 8 : 0) + Math.max(0, Math.floor((x.cleared + 1) / 3)) * 4 + (id === 'quarry' && ownsDevelopment(x, 'deep-veins') ? 8 : 0)); }
  function validateWorld(x, parent) {
    if (x && x.version === 1) return validate(x);
    if (!exact(x, ['version', 'index', 'cleared', 'completed', 'selectedArea', 'areas', 'developments', 'blueprints', 'mastery', 'automation', 'sequence', 'seen', 'recent', 'purchases']) || x.version !== 2) return false;
    if (!finite(x.index, 1e9, true) || !Number.isSafeInteger(x.cleared) || x.cleared < -1 || x.cleared > 1e9 || x.index > x.cleared + 1 || typeof x.completed !== 'boolean' || !KINDS.includes(x.selectedArea) || !object(x.areas) || !x.areas[x.selectedArea] || !projectArea(x)) return false;
    if (Object.keys(x.areas).some(id => !KINDS.includes(id)) || KINDS.some((id, offset) => (offset <= Math.max(x.index, x.cleared)) !== !!x.areas[id])) return false;
    if (!Array.isArray(x.developments) || new Set(x.developments).size !== x.developments.length || x.developments.some(id => !DEVELOPMENTS.some(d => d.id === id))) return false;
    if (x.developments.some(id => developmentDependencies(x, DEVELOPMENTS.find(d => d.id === id), parent).some(item => !item.met))) return false;
    for (const id of Object.keys(x.areas)) {
      const a = x.areas[id];
      if (!exact(a, areaKeys) || !['legacy', 'regional'].includes(a.goalMode) || !finite(a.legacyRegion, Math.floor(Math.max(x.index, x.cleared) / 3), true) || !finite(a.seenSequence, x.sequence, true) || !finite(a.index, 1e9, true) || areaId(a.index) !== id || a.index > Math.max(x.index, x.cleared) || typeof a.established !== 'boolean' || a.established !== (x.cleared >= KINDS.indexOf(id)) || !finite(a.elapsed, 8.64e12) || !finite(a.work) || !finite(a.finaleWork)) return false;
      if (!exact(a.ranks, ALL_TRACKS) || ALL_TRACKS.some(track => !finite(a.ranks[track], maxRank(x, id), true) || !def(a, track) && a.ranks[track] !== 0) || !finite(a.purchases, 1e12, true) || a.purchases < sumRanks(a)) return false;
      if (!exact(a.buffers, ['ore', 'smelt']) || !finite(a.buffers.ore, 10 + a.ranks.carts * 3 + (ownsDevelopment(x, 'trail-depot') ? 24 : 0) + EPS) || !finite(a.buffers.smelt, 10 + a.ranks.carts * 3 + (ownsDevelopment(x, 'trail-depot') ? 24 : 0) + EPS) || id !== 'quarry' && (a.buffers.ore || a.buffers.smelt)) return false;
      if (!exact(a.choices, ['route', 'smelting', 'equipment', 'allocation', 'dispatch']) || !['short', 'supply'].includes(a.choices.route) || !['throughput', 'quality', 'precision', 'mixed', 'optics', 'adaptive'].includes(a.choices.smelting) || !['tools', 'boots'].includes(a.choices.equipment) || !['balanced', 'repair', 'protect'].includes(a.choices.allocation) || !['trade', 'freight', 'survey', 'relay', 'trade-survey'].includes(a.choices.dispatch)) return false;
      if (a.choices.dispatch === 'freight' && !ownsDevelopment(x, 'trail-caravans') || a.choices.dispatch === 'survey' && !ownsDevelopment(x, 'tower-survey') || a.choices.dispatch === 'relay' && !ownsDevelopment(x, 'relay-network') || a.choices.dispatch === 'trade-survey' && !ownsDevelopment(x, 'survey-exchange') || a.choices.smelting === 'precision' && !ownsDevelopment(x, 'quarry-precision') || a.choices.smelting === 'mixed' && !ownsDevelopment(x, 'shared-workshops') || a.choices.smelting === 'optics' && !ownsDevelopment(x, 'optical-foundry') || a.choices.smelting === 'adaptive' && !ownsDevelopment(x, 'tower-control-room')) return false;
      const t = targets(a);
      if (a.work > t.work + EPS || a.finaleWork > t.finale + EPS || a.work < t.work - EPS && a.finaleWork !== 0) return false;
    }
    if (!Array.isArray(x.blueprints) || new Set(x.blueprints).size !== x.blueprints.length || x.blueprints.some(id => !BLUEPRINTS.some(d => d.id === id)) || x.blueprints.length > Math.floor((x.cleared + 1) / 3) || !exact(x.mastery, KINDS) || KINDS.some(id => !finite(x.mastery[id], 1e6, true))) return false;
    if (!exact(x.automation, ['enabled', 'priority', 'dispatch']) || typeof x.automation.enabled !== 'boolean' || typeof x.automation.dispatch !== 'boolean' || !['balanced', 'progress', 'income', 'materials'].includes(x.automation.priority) || x.cleared < 2 && (x.automation.enabled || x.automation.dispatch) || x.automation.priority === 'materials' && !ownsDevelopment(x, 'dispatch-ledgers')) return false;
    if (!finite(x.sequence, 1e12, true) || !finite(x.seen, x.sequence, true) || !finite(x.purchases, 1e12, true) || x.purchases < Object.values(x.areas).reduce((total, a) => total + a.purchases, 0)) return false;
    if (!Array.isArray(x.recent) || x.recent.length > 12 || x.recent.some((e, i) => !object(e) || !finite(e.sequence, x.sequence, true) || !e.sequence || i && e.sequence <= x.recent[i - 1].sequence || !['milestone', 'stage', 'blueprint', 'development'].includes(e.kind) || !finite(e.stage, 1e9, true) || typeof e.title !== 'string' || e.title.length > 100 || typeof e.text !== 'string' || e.text.length > 300 || Object.keys(e).some(key => !['sequence', 'kind', 'title', 'text', 'stage', 'areaId', 'sourceAreas', 'targetAreas'].includes(key)) || e.areaId !== undefined && !KINDS.includes(e.areaId) || ['sourceAreas', 'targetAreas'].some(key => e[key] !== undefined && (!Array.isArray(e[key]) || e[key].some(id => !KINDS.includes(id)))))) return false;
    const a = projectArea(x), t = targets(a);
    return a.index === x.index && (!x.completed || x.index <= x.cleared && a.work >= t.work - EPS && a.finaleWork >= t.finale - EPS);
  }
  function worldRates(parent) {
    const x = parent.expedition, all = {};
    if (!x || x.version !== 2) return all;
    for (const id of KINDS) {
      const ctx = context(parent, id);
      if (!ctx) continue;
      // New recipes reuse the established capacity formulas before applying
      // their explicit raw-material conversion. No output feeds itself.
      ctx.expedition.choices = Object.assign({}, ctx.expedition.choices);
      if (['precision', 'mixed', 'optics', 'adaptive'].includes(ctx.expedition.choices.smelting)) ctx.expedition.choices.smelting = 'throughput';
      all[id] = localRates(ctx);
      all[id].income = N.mul(all[id].income, N.pow(4, x.areas[id].legacyRegion - Math.floor(x.areas[id].index / 3)));
      Object.assign(all[id], { materials: N.zero(), knowledge: N.zero(), maps: N.zero(), overflow: 0, freight: 0, yield: 1 });
    }
    const trail = all.greenway, mine = all.quarry, tower = all.watchtower;
    const g = x.areas.greenway, q = x.areas.quarry, w = x.areas.watchtower;
    if (trail) {
      if (ownsDevelopment(x, 'paved-roads')) { trail.travel *= 1.25; trail.work *= 1.25; }
      const dispatch = g.choices.dispatch;
      const freightStrength = .30 + g.ranks.porters * .08 + (ownsDevelopment(x, 'paved-roads') ? .25 : 0);
      if (dispatch === 'freight') { trail.income = N.mul(trail.income, .55); trail.freight = freightStrength; }
      if (dispatch === 'survey' || dispatch === 'trade-survey') {
        trail.income = N.mul(trail.income, dispatch === 'survey' ? .4 : .75);
        trail.maps = N.mul(areaMoney(x, 'greenway'), .035 * (1 + .25 * g.ranks.scouts) * (dispatch === 'survey' ? 1 : .6));
        if (ownsDevelopment(x, 'survey-charters')) trail.knowledge = N.mul(trail.maps, .75);
      }
      if (dispatch === 'relay') { trail.income = N.mul(trail.income, .8); trail.freight = freightStrength * .8; }
      if (tower && ownsDevelopment(x, 'signal-network')) trail.income = N.mul(trail.income, 1 + tower.workers.repair * .09);
    }
    if (mine) {
      if (ownsDevelopment(x, 'deep-veins')) mine.picks *= 1.2;
      if (ownsDevelopment(x, 'trail-prospectors') && g && ['survey', 'trade-survey'].includes(g.choices.dispatch)) mine.picks *= 1 + Math.sqrt(g.ranks.scouts) * .08;
      mine.carts *= 1 + (trail ? trail.freight : 0);
      if (tower && ownsDevelopment(x, 'protective-escorts')) mine.carts *= 1 + tower.workers.protection * .12;
      if (ownsDevelopment(x, 'trail-depot')) mine.capacity += 24;
      let plan = q.choices.smelting;
      if (plan === 'adaptive') plan = q.buffers.smelt > EPS || q.buffers.ore > EPS || Math.min(mine.picks, mine.carts) >= mine.furnace * .6 ? 'throughput' : 'precision';
      mine.processing = plan;
      if (plan === 'precision') { mine.furnace *= .6; mine.yield = 1.4; }
      if (plan === 'mixed') { mine.furnace *= .8; mine.yield = 1.15; }
      if (plan === 'quality') mine.yield = .9;
      if (plan === 'optics') { mine.yield = .3; mine.furnace = Math.min(mine.furnace, (.08 + w.ranks.beacon * .02) / .35); }
      if (ownsDevelopment(x, 'efficient-crucibles')) mine.yield *= 1.15;
      let haul = q.buffers.ore > EPS ? mine.carts : Math.min(mine.carts, mine.picks);
      const smelt = q.buffers.smelt > EPS ? mine.furnace : Math.min(mine.furnace, haul);
      if (q.buffers.smelt >= mine.capacity - EPS) haul = Math.min(haul, smelt);
      let extracted = q.buffers.ore >= mine.capacity - EPS ? Math.min(mine.picks, haul) : mine.picks;
      if (ownsDevelopment(x, 'recovery-chutes') && q.buffers.ore >= mine.capacity - EPS) { mine.overflow = Math.max(0, mine.picks - haul); extracted = mine.picks; }
      mine.actualPicks = extracted; mine.actualCarts = haul; mine.actualFurnace = smelt;
      mine.ore = extracted - haul - mine.overflow; mine.smelt = haul - smelt;
      mine.bottleneck = mine.picks <= mine.carts && mine.picks <= mine.furnace ? 'picks' : mine.carts <= mine.furnace ? 'carts' : 'furnace';
      const t = targets(q);
      mine.work = q.work >= t.work - EPS ? 0 : smelt;
      mine.finale = q.work >= t.work - EPS ? smelt * 1.15 : 0;
      mine.materials = N.mul(areaMoney(x, 'quarry'), smelt * mine.yield * .08);
      mine.income = N.mul(areaMoney(x, 'quarry'), (1.1 + smelt * (plan === 'quality' ? 2.6 : 1.5) + mine.overflow * .15) * (blueprint(x, 'caravan') ? 1.2 : 1));
    }
    if (tower) {
      // Tower knowledge and maps bootstrap its cross-area projects before the
      // later Study/Map Room. Both are visible as soon as the Tower opens.
      tower.knowledge = N.mul(areaMoney(x, 'watchtower'), (.012 + tower.workers.repair * .004) * (1 + w.ranks.beacon * .08));
      tower.maps = N.mul(areaMoney(x, 'watchtower'), .008 * (1 + w.ranks.crew * .15));
      if (mine && q.choices.smelting === 'optics') tower.knowledge = N.add(tower.knowledge, N.mul(areaMoney(x, 'quarry'), mine.actualFurnace * .35));
    }
    return all;
  }
  function localWorldRates(parent, id) {
    if (!parent.expedition || parent.expedition.version !== 2) return localRates(parent);
    return worldRates(parent)[id || parent.expedition.selectedArea];
  }
  function worldContribution(parent) {
    const x = parent.expedition;
    if (!x || x.version !== 2) return contribution(parent);
    const c = counts(x.cleared), r = worldRates(parent);
    const total = key => Object.values(r).reduce((sum, item) => N.add(sum, item[key]), N.zero());
    // Established operations also coordinate their corresponding guild base.
    // Sublinear throughput scaling keeps further ranks relevant without a
    // self-feeding resource loop. Dispatch and actual constrained smelting,
    // rather than nominal station capacity, determine the support delivered.
    const guild = { coins: 1, ore: 1, knowledge: 1, maps: 1 };
    if (r.greenway) {
      const local = N.toNumber(N.div(r.greenway.income, areaMoney(x, 'greenway')));
      const plan = x.areas.greenway.choices.dispatch;
      const share = { trade: 1, freight: .55, survey: .4, relay: .8, 'trade-survey': .75 }[plan];
      guild.coins += .2 * Math.pow(local / share / .7, .25) * share;
      guild.maps += .3 * Math.sqrt(N.toNumber(N.div(r.greenway.maps, areaMoney(x, 'greenway'))) / .035);
      guild.knowledge += .3 * Math.sqrt(N.toNumber(N.div(r.greenway.knowledge, areaMoney(x, 'greenway'))) / .02);
    }
    if (r.quarry) guild.ore += .25 * Math.pow(r.quarry.actualFurnace * r.quarry.yield / .5, .35);
    if (r.watchtower) {
      const w = x.areas.watchtower;
      const knowledge = (.012 + r.watchtower.workers.repair * .004) * (1 + w.ranks.beacon * .08) + (r.quarry && x.areas.quarry.choices.smelting === 'optics' ? r.quarry.actualFurnace * .35 : 0);
      guild.knowledge += .3 * Math.sqrt(knowledge / .02);
      guild.maps += .3 * Math.sqrt(N.toNumber(N.div(r.watchtower.maps, areaMoney(x, 'watchtower'))) / .008);
    }
    return { coins: N.add(total('income'), c.greenway * .25), ore: N.add(total('materials'), c.quarry * .012), knowledge: total('knowledge'), maps: total('maps'), guild, production: 1 + Math.min(1, c.quarry * .025), travel: 1 + Math.min(1, c.watchtower * .03) };
  }
  function areaCost(x, id, track) {
    const a = x.areas[id], d = def(a, track), invested = rank(a, track);
    if (!d) return N.zero();
    const amount = N.mul(N.mul(d.base, N.pow(d.growth, invested)), N.pow(1.12, Math.pow(Math.max(0, invested - 8), 1.3)));
    return N.cmp(amount, 1e6) < 0 ? N.from(Math.ceil(N.toNumber(amount) - 1e-9)) : amount;
  }
  // An analytically scheduled purchase can land a few mantissa ULPs below its
  // price. Treat only that representation error as affordable, rather than
  // inventing another minimum-duration earning interval after every purchase.
  function canPay(wallet, amount) {
    return N.cmp(wallet, amount) >= 0 || N.cmp(wallet, N.mul(amount, 1 - Number.EPSILON * 16)) >= 0;
  }
  function areaOffers(parent) {
    const x = parent.expedition;
    return KINDS.flatMap(id => x.areas[id] ? TRACKS[id].filter(d => visible(x.areas[id], d.id) && tierProvider(parent, { type: 'expedition-buy', areaId: id, id: d.id }) && rank(x.areas[id], d.id) < maxRank(x, id)).map(d => ({ areaId: id, id: d.id, cost: areaCost(x, id, d.id) })) : []);
  }
  function worldNextEvent(parent, gains) {
    const x = parent.expedition;
    if (!x || x.version !== 2) return nextEvent(parent, gains);
    const all = worldRates(parent), a = projectArea(x), t = targets(a), r = all[areaId(x.index)];
    let seconds = x.completed ? Infinity : a.work < t.work - EPS ? (t.work - a.work) / r.work : (t.finale - a.finaleWork) / r.finale;
    for (const id of Object.keys(all)) {
      const area = x.areas[id], rates = all[id];
      if (id === 'quarry') for (const key of ['ore', 'smelt']) {
        if (rates[key] > EPS) seconds = Math.min(seconds, Math.max(EPS, (rates.capacity - area.buffers[key]) / rates[key]));
        else if (rates[key] < -EPS) seconds = Math.min(seconds, Math.max(EPS, area.buffers[key] / -rates[key]));
      }
      if (id === 'watchtower') seconds = Math.min(seconds, 35 - area.elapsed % 35);
    }
    if (x.automation.enabled && x.cleared >= 2) {
      const reserve = reserveProvider(parent);
      for (const offer of areaOffers(parent)) {
        const required = N.add(offer.cost, reserve);
        const wait = canPay(parent.resources.coins, required) ? EPS : N.toNumber(N.div(N.sub(required, parent.resources.coins), gains.coins));
        seconds = Math.min(seconds, Math.max(EPS, wait));
      }
    }
    return Number.isFinite(seconds) ? Math.max(EPS, seconds) : Infinity;
  }
  function worldTick(parent, seconds) {
    const x = parent.expedition;
    if (!x || x.version !== 2) return tick(parent, seconds);
    const all = worldRates(parent);
    for (const id of Object.keys(all)) {
      const a = x.areas[id], r = all[id], t = targets(a);
      a.elapsed += seconds;
      if (id === 'watchtower') {
        const boundary = Math.round(a.elapsed / 35) * 35;
        if (Math.abs(a.elapsed - boundary) < EPS) a.elapsed = boundary;
      }
      // Only the frontier project grants route work. All establishments continue
      // production and their internal queues, including when not displayed.
      if (a.index === x.index && !x.completed) {
        a.work = Math.min(t.work, a.work + r.work * seconds);
        a.finaleWork = Math.min(t.finale, a.finaleWork + r.finale * seconds);
      }
      if (id === 'quarry') {
        a.buffers.ore = Math.max(0, Math.min(r.capacity, a.buffers.ore + r.ore * seconds));
        a.buffers.smelt = Math.max(0, Math.min(r.capacity, a.buffers.smelt + r.smelt * seconds));
        for (const key of ['ore', 'smelt']) {
          if (a.buffers[key] < EPS) a.buffers[key] = 0;
          if (r.capacity - a.buffers[key] < EPS) a.buffers[key] = r.capacity;
        }
      }
    }
  }
  function worldFinish(parent) {
    const x = parent.expedition, a = projectArea(x), t = targets(a);
    if (x.completed || a.work < t.work - EPS || a.finaleWork < t.finale - EPS) return false;
    a.work = t.work; a.finaleWork = t.finale; a.established = true; x.completed = true;
    const first = x.index > x.cleared;
    x.cleared = Math.max(x.cleared, x.index);
    x.mastery[areaId(x.index)] = Math.min(1e6, x.mastery[areaId(x.index)] + 1);
    if (first && areaId(x.index) === 'quarry') parent.upgrades[a.choices.equipment === 'tools' ? 'gear-tools' : 'gear-boots'] += 1;
    event(x, 'stage', stageName(a) + ' established', 'This area keeps producing. Its upgrades, workers and choices remain available.');
    x.recent[x.recent.length - 1].areaId = areaId(x.index);
    return true;
  }
  function beginProject(parent, index, select) {
    const x = parent.expedition;
    if (!x || x.version !== 2) { parent.expedition = upgradeWorld(parent) || createWorld(parent); return beginProject(parent, index, select); }
    const id = areaId(index), a = x.areas[id] || makeArea(index, index <= x.cleared);
    // Regional projects renew only project work. The permanent establishment,
    // production clock, upgrade ranks, choices and material buffers all remain.
    a.index = index; a.goalMode = 'regional'; a.work = 0; a.finaleWork = 0;
    x.areas[id] = a; x.index = index; x.completed = false;
    if (select !== false) x.selectedArea = id;
  }
  function developmentDependencies(x, d, parent) {
    return [{ label: d.at < 3 ? 'Open ' + AREA_NAMES[KINDS[d.at]] : 'Establish ' + d.at + ' regional landmarks', met: d.at < 3 ? !!x.areas[KINDS[d.at]] : x.cleared + 1 >= d.at }]
      .concat(d.charters ? [{ label: 'Earn ' + d.charters + ' Guild Charter' + (d.charters === 1 ? '' : 's'), met: !parent || parent.lifetime.charters >= d.charters }] : [])
      .concat((d.requires || []).map(id => ({ label: DEVELOPMENTS.find(item => item.id === id).name, met: ownsDevelopment(x, id) })));
  }
  function developmentTask(parent, id) {
    const d = DEVELOPMENTS.find(item => item.id === id);
    if (!d) return null;
    const x = parent && parent.expedition;
    return { name: d.name, costs: Object.fromEntries(Object.entries(d.costs).map(([key, value]) => [key, N.from(value)])), open: !!(x && x.version === 2 && developmentDependencies(x, d, parent).every(item => item.met)), done: !!(x && x.version === 2 && ownsDevelopment(x, id)) };
  }
  function worldAct(parent, action) {
    if (!tierProvider(parent, action)) return { ok: false, message: 'Unlock this ready upgrade tier first.' };
    const x = parent.expedition;
    if (!x || x.version !== 2) return act(parent, action);
    const id = action.areaId || x.selectedArea, a = x.areas[id];
    if (action.type === 'expedition-select') {
      if (!a || !KINDS.includes(action.areaId)) return { ok: false, message: 'Discover that area first.' };
      x.selectedArea = id;
      a.seenSequence = x.sequence;
      return { ok: true, message: AREA_NAMES[id] + ' selected. Every area keeps working.' };
    }
    if (action.type === 'expedition-buy') {
      const d = a && def(a, action.id);
      if (!d || !visible(a, d.id) || rank(a, d.id) >= maxRank(x, id)) return { ok: false, message: 'Discover another regional expansion to raise this rank ceiling.' };
      const price = areaCost(x, id, d.id);
      if (!canPay(parent.resources.coins, price)) return { ok: false, message: 'Need ' + N.format(price) + ' coins.' };
      parent.resources.coins = N.sub(parent.resources.coins, price);
      a.ranks[d.id] += 1; a.purchases += 1; x.purchases += 1;
      if ([3, 6, 10, 20, 30, 50, 75, 100, 150, 200].includes(a.ranks[d.id])) {
        event(x, 'milestone', d.label + ' rank ' + a.ranks[d.id], a.ranks[d.id] === 3 ? d.milestone : 'Permanent ' + d.label.toLowerCase() + ' investment reached rank ' + a.ranks[d.id] + '.');
        x.recent[x.recent.length - 1].areaId = id;
      }
      return { ok: true, message: d.label + ' improved to rank ' + a.ranks[d.id] + '.' };
    }
    if (action.type === 'expedition-choice') {
      if (!a) return { ok: false, message: 'Discover that area first.' };
      if (id === 'greenway' && ['short', 'supply'].includes(action.id) && (a.established || sumRanks(a) >= 2 || a.work >= 28)) a.choices.route = action.id;
      else if (id === 'greenway' && (action.id === 'trade' || action.id === 'freight' && ownsDevelopment(x, 'trail-caravans') || action.id === 'survey' && ownsDevelopment(x, 'tower-survey') || action.id === 'relay' && ownsDevelopment(x, 'relay-network') || action.id === 'trade-survey' && ownsDevelopment(x, 'survey-exchange'))) a.choices.dispatch = action.id;
      else if (id === 'quarry' && (['throughput', 'quality'].includes(action.id) || action.id === 'precision' && ownsDevelopment(x, 'quarry-precision') || action.id === 'mixed' && ownsDevelopment(x, 'shared-workshops') || action.id === 'optics' && ownsDevelopment(x, 'optical-foundry') || action.id === 'adaptive' && ownsDevelopment(x, 'tower-control-room'))) a.choices.smelting = action.id;
      else if (id === 'quarry' && ['tools', 'equipment-boots'].includes(action.id)) a.choices.equipment = action.id === 'tools' ? 'tools' : 'boots';
      else if (id === 'watchtower' && ['balanced', 'repair', 'protect'].includes(action.id)) a.choices.allocation = action.id;
      else return { ok: false, message: 'That working plan has not opened here.' };
      return { ok: true, message: AREA_NAMES[id] + ' plan saved. It keeps working while you visit other areas.' };
    }
    if (action.type === 'expedition-development') {
      const d = DEVELOPMENTS.find(item => item.id === action.id);
      if (!d || ownsDevelopment(x, d.id) || developmentDependencies(x, d, parent).some(item => !item.met)) return { ok: false, message: 'Complete the listed discoveries first.' };
      if (Object.entries(d.costs).some(([key, value]) => N.cmp(parent.resources[key], value) < 0)) return { ok: false, message: 'Gather the listed guild resources first.' };
      Object.entries(d.costs).forEach(([key, value]) => { parent.resources[key] = N.sub(parent.resources[key], value); });
      x.developments.push(d.id);
      event(x, 'development', d.name, d.description);
      Object.assign(x.recent[x.recent.length - 1], { areaId: d.to[0], sourceAreas: d.from.slice(), targetAreas: d.to.slice() });
      return { ok: true, message: d.name + ' built. Return to ' + d.to.map(key => AREA_NAMES[key]).join(' and ') + ' to use it.' };
    }
    if (action.type === 'expedition-automation') {
      if (x.cleared < 2 || typeof action.enabled !== 'boolean' || !['balanced', 'progress', 'income', 'materials'].includes(action.priority) || action.priority === 'materials' && !ownsDevelopment(x, 'dispatch-ledgers') || action.dispatch !== undefined && typeof action.dispatch !== 'boolean') return { ok: false, message: 'Establish the Watchtower and any required working plan first.' };
      x.automation.enabled = action.enabled; x.automation.priority = action.priority;
      if (action.dispatch !== undefined) x.automation.dispatch = action.dispatch;
      return { ok: true, message: 'All-area automation saved. Shared coin reserves and your saved objective remain protected.' };
    }
    // Blueprint and acknowledgement state belongs to the world, not an area.
    return act(parent, action);
  }
  function worldAutoBuy(parent) {
    const x = parent.expedition;
    if (!x || x.version !== 2) return autoBuy(parent);
    if (!x.automation.enabled || x.cleared < 2) return;
    let all = worldRates(parent);
    const priority = x.automation.priority, frontier = areaId(x.index);
    function score(offer) {
      if (priority === 'income') return offer.id === 'porters' ? 0 : offer.id === 'crew' ? 1 : 2;
      if (priority === 'materials') return offer.areaId === 'quarry' && offer.id === all.quarry.bottleneck ? 0 : offer.id === 'porters' && x.areas.greenway.choices.dispatch !== 'trade' ? 1 : 2;
      if (priority === 'progress') return offer.areaId === frontier && offer.id === (frontier === 'quarry' ? all.quarry.bottleneck : frontier === 'greenway' ? 'boots' : projectArea(x).work >= targets(projectArea(x)).work ? 'beacon' : 'lift') ? 0 : 2;
      return rank(x.areas[offer.areaId], offer.id);
    }
    // Exhaust the finite set of immediately affordable ranks at this event.
    // A funded wallet must not earn extra micro-intervals merely to buy its
    // second rank, or depend on how often the display requests a frame.
    for (let pass = 0; pass < 200; pass += 1) {
      all = worldRates(parent);
      const offers = areaOffers(parent).sort((a, b) => score(a) - score(b) || N.cmp(a.cost, b.cost));
      let bought = false;
      for (const offer of offers) {
        if (canPay(parent.resources.coins, N.add(offer.cost, reserveProvider(parent)))) bought = worldAct(parent, { type: 'expedition-buy', areaId: offer.areaId, id: offer.id }).ok || bought;
      }
      if (!bought) break;
    }
  }
  function detached(parent) {
    const result = Object.assign({}, parent, { expedition: clone(parent.expedition) });
    if (entitlements.has(parent)) entitlements.set(result, entitlements.get(parent));
    return result;
  }
  function rateMetrics(parent) {
    const all = worldRates(parent), result = {};
    if (rateProvider) {
      const gain = rateProvider(parent, entitlements.get(parent));
      for (const key of ['coins', 'ore', 'knowledge', 'maps']) result['guild:' + key] = { areaId: { coins: 'greenway', ore: 'quarry', knowledge: 'watchtower', maps: 'greenway' }[key], label: 'Guild · Total ' + key, value: N.from(gain[key]), unit: '/s' };
    }
    for (const id of Object.keys(all)) {
      const r = all[id];
      for (const [key, label, value] of [['coins', 'Coins', r.income], ['ore', 'Usable ore', r.materials], ['maps', 'Maps', r.maps], ['knowledge', 'Knowledge', r.knowledge]]) result[id + ':' + key] = { areaId: id, label: AREA_NAMES[id] + ' · ' + label, value: N.from(value), unit: '/s' };
      if (id === 'greenway') result['greenway:exploration'] = { areaId: id, label: 'Greenway · Exploration', value: N.from(r.travel), unit: '/s' };
      if (id === 'quarry') {
        for (const [key, label] of [['picks', 'Extraction'], ['carts', 'Hauling'], ['furnace', 'Smelting']]) result['quarry:' + key] = { areaId: id, label: 'Quarry · ' + label + ' capacity', value: N.from(r[key]), unit: '/s' };
        result['quarry:queue'] = { areaId: id, label: 'Quarry · Queue capacity', value: N.from(r.capacity), unit: ' units' };
        result['quarry:capacity'] = { areaId: id, label: 'Quarry · Sustainable raw processing', value: N.from(Math.min(r.picks, r.carts, r.furnace)), unit: '/s' };
      }
      if (id === 'watchtower') result['watchtower:construction'] = { areaId: id, label: 'Watchtower · Expansion construction capacity', value: N.from(r.repair), unit: '/s' };
    }
    return result;
  }
  function impactFor(parent, changed) {
    const before = rateMetrics(parent), after = rateMetrics(changed);
    return Object.keys(after).filter(key => before[key] && N.cmp(before[key].value, after[key].value) !== 0).map(key => ({ metric: key, areaId: after[key].areaId, label: after[key].label, current: N.format(before[key].value), next: N.format(after[key].value), currentValue: before[key].value, nextValue: after[key].value, unit: after[key].unit }));
  }
  function worldCatalog(parent) {
    const x = parent.expedition;
    if (!x || x.version !== 2) return [];
    const cards = KINDS.flatMap(id => !x.areas[id] ? [] : TRACKS[id].map(d => {
      const a = x.areas[id], price = areaCost(x, id, d.id), copy = detached(parent);
      copy.expedition.areas[id].ranks[d.id] += 1;
      const impact = impactFor(parent, copy), cap = maxRank(x, id);
      const milestone = [3, 6, 10, 20, 30, 50, 75, 100, 150, 200].find(value => value > rank(a, d.id) && value <= cap);
      return { id: 'area:' + id + ':' + d.id, catalogId: 'area:' + id + ':' + d.id, trackId: d.id, areaId: id, name: d.label, label: d.label, icon: d.icon, group: 'area', effectKind: id === 'quarry' ? d.id === 'carts' ? 'transport' : 'throughput' : d.id === 'porters' ? 'allocation' : d.id === 'scouts' ? 'discovery' : 'throughput', sourceAreas: [id], targetAreas: [...new Set([id].concat(impact.map(item => item.areaId)))], dependencies: [], impact,
        rank: rank(a, d.id), level: rank(a, d.id), maxRank: cap, maxLevel: cap, maxed: rank(a, d.id) >= cap, visible: visible(a, d.id) && tierProvider(parent, { type: 'expedition-buy', areaId: id, id: d.id }), disabled: rank(a, d.id) >= cap || N.cmp(parent.resources.coins, price) < 0,
        description: (d.id === 'lift' ? 'Construction capacity for expansions; a completed Tower keeps its capacity for future building projects' : d.effect) + '. Permanent area infrastructure; visiting another area never removes it.', effectText: d.id === 'lift' ? 'Expansion construction capacity' : d.effect, comparison: impact.length ? impact.slice(0, 2).map(item => item.label + ' ' + item.current + ' → ' + item.next + item.unit).join(' · ') : 'Capacity improves; another station currently limits output.',
        cost: [{ resource: 'coins', amount: price, text: N.format(price) + ' coins' }], nextMilestone: milestone ? { rank: milestone, remaining: milestone - rank(a, d.id), label: milestone === 3 ? d.milestone : 'Stronger permanent ' + d.label.toLowerCase() } : null,
        action: { type: 'expedition-buy', areaId: id, id: d.id } };
    }));
    return cards.concat(DEVELOPMENTS.map(d => {
      const dependencies = developmentDependencies(x, d, parent), owned = ownsDevelopment(x, d.id), costs = Object.entries(d.costs).map(([resource, amount]) => ({ resource, amount: N.from(amount), text: N.format(amount) + ' ' + resource }));
      const copy = detached(parent); if (!owned) copy.expedition.developments.push(d.id);
      const impact = owned ? [] : impactFor(parent, copy);
      if (!owned && ['allocation', 'recipe', 'parallel', 'automation', 'unlock'].includes(d.kind)) impact.unshift({ metric: 'behavior:' + d.id, label: d.name, current: 'Locked', next: 'Available', unit: '' });
      return { id: 'development:' + d.id, catalogId: 'development:' + d.id, name: d.name, label: d.name, icon: AREA_ICONS[d.to[0]], group: d.group, effectKind: d.kind, areaId: d.to[0], sourceAreas: d.from.slice(), targetAreas: d.to.slice(), dependencies, impact, description: d.description, effectText: d.description,
        cost: costs, owned, maxed: owned, level: owned ? 1 : 0, maxLevel: 1, visible: owned || dependencies.every(item => item.met) && tierProvider(parent, { type: 'expedition-development', id: d.id }), disabled: owned || dependencies.some(item => !item.met) || costs.some(item => N.cmp(parent.resources[item.resource], item.amount) < 0),
        action: { type: 'expedition-development', id: d.id } };
    }));
  }
  function worldView(parent) {
    const x = parent.expedition;
    if (!x || x.version !== 2) return view(parent);
    const id = x.selectedArea, a = x.areas[id], ctx = context(parent, id), all = worldRates(parent), r = all[id], t = targets(a), catalog = worldCatalog(parent);
    const result = view(ctx), frontier = a.index === x.index, done = a.work >= t.work - EPS && a.finaleWork >= t.finale - EPS;
    result.catalog = catalog;
    const options = (entries, selected) => entries.map(([key, label, effectText]) => {
      const copy = detached(parent); worldAct(copy, { type: 'expedition-choice', areaId: id, id: key });
      return { id: key, label, effectText, selected: selected === key, impact: impactFor(parent, copy), action: { type: 'expedition-choice', areaId: id, id: key } };
    });
    result.stage = Object.assign(result.stage, { id: 'area-' + id + '-expansion-' + expansionOf(x, id), areaId: id, name: AREA_NAMES[id], expansion: expansionOf(x, id), established: a.established, completed: done, frontier, objective: done ? 'Producing for your guild' : result.stage.objective });
    result.cards = catalog.filter(item => item.group === 'area' && item.areaId === id).map(item => Object.assign({}, item, { id: item.trackId }));
    result.wallet.rateText = '+' + N.format(r.income) + '/s from ' + AREA_NAMES[id];
    result.choices.forEach(choice => choice.options.forEach(option => { option.action.areaId = id; }));
    if (id === 'greenway' && (ownsDevelopment(x, 'trail-caravans') || ownsDevelopment(x, 'tower-survey'))) {
      const entries = [['trade', 'Trade', 'Full coin deliveries']];
      if (ownsDevelopment(x, 'trail-caravans')) entries.push(['freight', 'Freight', 'Fewer coins; faster quarry hauling']);
      if (ownsDevelopment(x, 'tower-survey')) entries.push(['survey', 'Survey', 'Maps instead of most coin deliveries']);
      if (ownsDevelopment(x, 'relay-network')) entries.push(['relay', 'Relay', 'Trade and freight together at 80% strength']);
      if (ownsDevelopment(x, 'survey-exchange')) entries.push(['trade-survey', 'Trade + survey', '75% trade income and 60% survey maps; no freight']);
      result.choices.push({ id: 'dispatch', title: 'Greenway dispatch', visible: true, options: options(entries, a.choices.dispatch) });
    }
    if (id === 'quarry') {
      const entries = [['throughput', 'Volume', 'Full raw throughput and standard ore yield'], ['quality', 'Trade ingots', 'Slower processing; more coins, 10% less usable ore']];
      if (ownsDevelopment(x, 'quarry-precision')) entries.push(['precision', 'Precision', '40% more ore per raw unit; 40% lower furnace capacity']);
      if (ownsDevelopment(x, 'shared-workshops')) entries.push(['mixed', 'Split batches', 'Volume and precision together: intermediate yield and capacity']);
      if (ownsDevelopment(x, 'optical-foundry')) entries.push(['optics', 'Optical parts', 'Trade 70% of ore yield for Tower knowledge; limited by the beacon']);
      if (ownsDevelopment(x, 'tower-control-room')) entries.push(['adaptive', 'Adaptive', 'Clear input queues with volume; use precision under sustained raw scarcity']);
      result.choices[0] = { id: 'smelting', title: 'Processing plan', visible: true, options: options(entries, a.choices.smelting) };
      result.stations = [['picks', 'Mine', r.actualPicks, r.picks, a.buffers.ore], ['carts', 'Cart', r.actualCarts, r.carts, a.buffers.smelt], ['furnace', 'Furnace', r.actualFurnace, r.furnace, 0]].map(([key, label, rate, maxRate, buffer]) => ({ id: key, label, rate, maxRate, rateText: rate.toFixed(2) + '/s', buffer, capacity: r.capacity, bottleneck: key === r.bottleneck, status: rate < maxRate - EPS ? key === 'picks' ? 'Queue full' : 'Waiting for supply' : key === r.bottleneck ? 'Bottleneck' : 'Working' }));
    }
    if (id === 'watchtower') result.choices[0] = { id: 'allocation', title: 'Assign the permanent crew', visible: true, options: options([['balanced', 'Balanced', 'Split construction, trade signals and escorts'], ['repair', 'Signal', 'More construction and trade signals; fewer quarry escorts'], ['protect', 'Guard', 'More quarry escorts and protection; fewer trade signals']], a.choices.allocation) };
    result.choice = result.choices[0];
    result.finale.completed = done; result.finale.ready = frontier && !done && a.work >= t.work - EPS;
    result.next = { label: x.index < 2 ? 'Discover ' + AREA_NAMES[areaId(x.index + 1)] : 'Expand ' + AREA_NAMES[areaId(x.index + 1)], description: 'Every opened area keeps producing. All area ranks, queues and working plans remain.', disabled: !x.completed, action: { type: 'expedition-next' } };
    result.areas = KINDS.filter(key => x.areas[key]).map(key => {
      const area = x.areas[key], rates = all[key], target = targets(area), complete = area.work >= target.work - EPS && area.finaleWork >= target.finale - EPS;
      return { id: key, label: AREA_NAMES[key], icon: AREA_ICONS[key], unlocked: true, selected: key === id, established: area.established, expansion: expansionOf(x, key), objective: complete ? 'Producing' : 'Establishing', progress: progress(area), rateText: key === 'quarry' ? '+' + N.format(rates.materials) + ' ore/s' : key === 'watchtower' ? '+' + N.format(rates.knowledge) + ' knowledge/s' : '+' + N.format(rates.income) + ' coins/s', status: key === 'quarry' ? rates.bottleneck + ' limits throughput' : 'Working automatically', attention: x.recent.some(e => e.sequence > area.seenSequence && (e.areaId === key || (e.targetAreas || []).includes(key))), action: { type: 'expedition-select', areaId: key } };
    });
    result.outposts = result.areas.filter(area => area.established).map(area => Object.assign({}, area, { name: area.label, effectText: area.rateText, count: Math.floor(x.areas[area.id].index / 3) + 1, mastery: x.mastery[area.id] }));
    result.automation = Object.assign({ unlocked: x.cleared >= 2, choices: ['balanced', 'progress', 'income'].concat(ownsDevelopment(x, 'dispatch-ledgers') ? ['materials'] : []).map(key => ({ id: key, label: key[0].toUpperCase() + key.slice(1), action: { type: 'expedition-automation', enabled: true, priority: key } })) }, x.automation);
    result.purchases = x.purchases;
    result.guildLinks = ['All opened areas work at the same time', 'Quarry materials build Greenway transport', 'Tower discoveries open earlier-area recipes and routes'];
    result.scene = Object.assign(result.scene, { established: a.established, expansion: expansionOf(x, id), developments: x.developments.slice(), completed: done, capacity: r.capacity, bottleneck: r.bottleneck, workers: r.workers,
      dispatch: a.choices.dispatch, rates: { travel: r.travel, picks: r.picks, carts: r.carts, furnace: r.furnace, repair: r.repair, beacon: r.beacon, income: N.toNumber(r.income), maps: N.toNumber(r.maps), knowledge: N.toNumber(r.knowledge), materials: N.toNumber(r.materials) }, flows: { picks: r.actualPicks, carts: r.actualCarts, furnace: r.actualFurnace } });
    return result;
  }
  return { create: createWorld, hydrate: hydrateWorld, validate: validateWorld, normalize: upgradeWorld, progress: worldProgress, targets: worldTargets, localRates: localWorldRates, rates: worldRates, contribution: worldContribution, nextEvent: worldNextEvent, tick: worldTick, finish: worldFinish, restart: beginProject, autoBuy: worldAutoBuy, act: worldAct, view: worldView, catalog: worldCatalog, impact: impactFor, developmentTask, stageName, setEntitlements, setReserveProvider, setRateProvider, setTierProvider, tierDefinitions: () => DEVELOPMENTS };
});
