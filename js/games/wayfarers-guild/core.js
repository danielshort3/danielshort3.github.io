(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./content.js') : root.WayfarersContent);
  if (common) module.exports = api;
  if (root) root.WayfarersCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, C) {
  'use strict';
  const VERSION = 4;
  const MAX_TIME = 8.64e15;
  const DROP_MEAN_SECONDS = 2000 * 60;
  const MAX_DROP_SECONDS = Math.ceil(DROP_MEAN_SECONDS * Math.log(4294967296) * 1000) / 1000;
  // Store-verified ownership is deliberately absent from serialized game state.
  const premiumEntitlements = new WeakMap();
  const ORDINARY = ['coins', 'ore', 'herbs', 'provisions', 'knowledge', 'maps'];
  const DAY = 86400000;
  const LUCK_KEYS = ['rng', 'commonRemainingMs', 'relicRemainingMs', 'pitySeconds', 'owned', 'active', 'duplicateProgress', 'hunt', 'scheduledHunt', 'research', 'kit', 'ledger', 'collectionBanner', 'commonFinds', 'relicFinds'];
  const CARAVAN_KEYS = ['rng', 'remainingMs', 'offer', 'sequence', 'completed', 'receipts', 'pendingQuote', 'surge'];
  const EPS = 1e-7;
  const own = (x, k) => Object.prototype.hasOwnProperty.call(x, k);
  const object = x => x !== null && typeof x === 'object' && !Array.isArray(x);
  const find = (list, id) => list.find(x => x.id === id);
  const clone = x => JSON.parse(JSON.stringify(x));
  const dictionary = (items, value) => Object.fromEntries(items.map(item => [item.id, typeof value === 'function' ? value() : value]));
  const level = (state, id) => state.upgrades[id] || 0;
  const researched = (state, id) => state.research.includes(id);
  const unlocked = (state, id) => state.rooms.includes(id);
  const completed = (state, id) => state.challenges.completed.includes(id);
  const timeValue = now => typeof now === 'number' && Number.isFinite(now) && now >= 0 && now <= MAX_TIME ? now : Date.now();
  const product = (...values) => values.reduce((out, value) => N.mul(out, value), N.from(1));
  const power = (base, exp) => N.pow(base, exp);
  const ratio = (a, b) => Math.min(1e100, N.toNumber(N.div(a, N.max(b, '1e-100'))));
  const affordable = (state, costs) => Object.keys(costs).every(id => N.cmp(state.resources[id], costs[id]) >= 0);
  const spend = (state, costs) => Object.keys(costs).forEach(id => { state.resources[id] = N.sub(state.resources[id], costs[id]); });
  function pushEvent(state, message) {
    state.events.push(message);
    if (state.events.length > 12) state.events.shift();
  }

  function nextDrop(premium) {
    let seed = premium.rng;
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    premium.rng = seed >>> 0;
    return Math.max(1, Math.ceil(-Math.log(premium.rng / 4294967296) * DROP_MEAN_SECONDS * 1000)) / 1000;
  }
  function createPremium(time) {
    // Timestamp mixing creates a stable initial seed; loading never samples again.
    let seed = (Math.floor(time) ^ Math.floor(time / 4294967296) ^ 0x9e3779b9) >>> 0;
    seed = Math.imul(seed ^ (seed >>> 16), 0x21f0aaad) >>> 0;
    seed = Math.imul(seed ^ (seed >>> 15), 0x735a2d97) >>> 0;
    const premium = { rng: (seed ^ (seed >>> 15)) >>> 0 || 1, eligibleSeconds: 0, untilDrop: 0, drops: 0, claimedMilestones: [], owned: [], equipped: null };
    premium.untilDrop = nextDrop(premium);
    return premium;
  }
  function random(stream) {
    let seed = stream.rng;
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    stream.rng = seed >>> 0;
    return stream.rng / 4294967296;
  }
  function createLuck(time) {
    const luck = { rng: (createPremium(time).rng ^ 0x63d83595) >>> 0 || 1, commonRemainingMs: 0, relicRemainingMs: 0, pitySeconds: 0, owned: [], active: null, duplicateProgress: dictionary(C.RELICS, 0), hunt: 'balanced', scheduledHunt: 'balanced', research: [], kit: { prepared: null, active: null, remainingSeconds: 0 }, ledger: { seq: 0, seen: 0, recent: [] }, collectionBanner: false, commonFinds: 0, relicFinds: 0 };
    luck.commonRemainingMs = Math.floor((480 + random(luck) * 240) * 1000);
    luck.relicRemainingMs = Math.min(2700000, Math.max(1000, Math.ceil(-Math.log(random(luck)) * 21600000)));
    return luck;
  }
  function createCaravan(time) {
    const caravan = { rng: (createPremium(time).rng ^ 0xb5297a4d) >>> 0 || 1, remainingMs: 0, offer: null, sequence: 0, completed: [], receipts: [], pendingQuote: null, surge: { resource: null, remainingSeconds: 0 } };
    caravan.remainingMs = Math.floor((3600 + random(caravan) * 1800) * 1000);
    return caravan;
  }
  function setPremiumEntitlements(state, ids) {
    if (!object(state) || !Array.isArray(ids) || ids.some(id => typeof id !== 'string' || !find(C.PREMIUM_ITEMS, id))) return { ok: false, message: 'Invalid premium entitlements.' };
    premiumEntitlements.set(state, new Set(ids));
    return { ok: true, message: 'Account ownership refreshed.' };
  }
  function premiumOwned(state, id) {
    const account = premiumEntitlements.get(state);
    return state.premium.owned.includes(id) || !!(account && account.has(id));
  }
  function premiumGift(state, id) {
    const def = find(C.PREMIUM_MILESTONES, id);
    if (id === 'first-refit' && state.lifetime.refits > 0 || id === 'first-charter' && state.lifetime.charters > 0 || id.startsWith('challenge-') && completed(state, id.slice(10))) return 0;
    return def && !state.premium.claimedMilestones.includes(id) ? def.amount : 0;
  }
  function grantPremiumGift(state, id) {
    const amount = premiumGift(state, id);
    if (!amount) return;
    state.premium.claimedMilestones.push(id);
    state.resources.starshards = N.floor(N.add(state.resources.starshards, amount));
    pushEvent(state, 'Milestone gift: ' + amount + ' earned Starshards. This gift is awarded once.');
  }
  function advancePremium(state, seconds) {
    if (!unlocked(state, 'forge') || seconds <= 0) return;
    const premium = state.premium;
    // Count absolute clock boundaries, rather than rounding each foreground tick.
    // Integer milliseconds keep rare-drop boundaries identical across partitions.
    const milliseconds = Math.max(0, Math.round(state.lastUpdate + seconds * 1000) - Math.round(state.lastUpdate));
    let remaining = milliseconds;
    let untilDrop = Math.round(premium.untilDrop * 1000);
    let drops = 0;
    // Only real expedition time counts. Routes, speed, and resets cannot reroll or
    // accelerate this schedule. Work is proportional to rare drops, never seconds.
    while (remaining >= untilDrop) {
      remaining -= untilDrop;
      drops += 1;
      untilDrop = Math.round(nextDrop(premium) * 1000);
    }
    premium.untilDrop = (untilDrop - remaining) / 1000;
    premium.eligibleSeconds = (Math.round(premium.eligibleSeconds * 1000) + milliseconds) / 1000;
    premium.drops += drops;
    if (drops) {
      state.resources.starshards = N.floor(N.add(state.resources.starshards, drops));
      pushEvent(state, 'Expedition find: ' + drops + ' earned Starshard' + (drops === 1 ? '' : 's') + '.');
    }
  }

  function createState(now) {
    const time = timeValue(now);
    return {
      schemaVersion: VERSION, createdAt: time, lastUpdate: time,
      resources: dictionary(C.RESOURCES, N.zero), upgrades: dictionary(C.UPGRADES, 0),
      research: [], refitUpgrades: dictionary(C.REFIT_UPGRADES, 0), legacy: dictionary(C.LEGACY_UPGRADES, 0),
      rooms: ['trail'], mastery: dictionary(C.ROOMS, N.zero), ranks: dictionary(C.ROOMS, 0),
      route: { index: 0, progress: N.zero(), mode: 'frontier' },
      run: { id: 1, completed: -1, work: N.zero(), elapsed: 0 },
      chapter: { work: N.zero(), refits: 0 },
      lifetime: { work: N.zero(), coins: N.zero(), refits: 0, charters: 0, highestRoute: -1, frontier: 0 },
      crew: { owned: [], specialists: [null, null], companions: [], companion: null },
      meal: 'meal-none', doctrine: 'balanced', automations: { operations: false, equipment: false, routes: false },
      challenges: { active: null, completed: [] }, collections: [], planner: 0, events: [], premium: createPremium(time), luck: createLuck(time), caravan: createCaravan(time), guild: createGuild(false), introductions: { seen: [] }
    };
  }

  function validateSchema(state, version) {
    const errors = [];
    if (!object(state)) return { valid: false, errors: ['State must be an object.'] };
    if (state.schemaVersion !== version) errors.push('Unsupported game state version.');
    const numeric = (x, name, integer, min, max) => {
      if (typeof x !== 'number' || !Number.isFinite(x) || x < (min === undefined ? 0 : min) || x > (max === undefined ? integer ? 1e9 : Number.MAX_SAFE_INTEGER / 16 : max) || (integer && !Number.isSafeInteger(x))) errors.push('Invalid ' + name + '.');
    };
    const big = (x, name) => { if (!N.valid(x)) errors.push('Invalid ' + name + '.'); };
    const map = (x, list, checker, name) => {
      if (!object(x)) { errors.push('Missing ' + name + '.'); return; }
      list.forEach(item => checker(x[item.id], name + '.' + item.id));
      if (Object.keys(x).some(key => !list.some(item => item.id === key))) errors.push('Unknown ' + name + ' entry.');
    };
    const ids = (x, list, name) => {
      if (!Array.isArray(x) || x.length > list.length || new Set(x).size !== x.length || x.some(id => typeof id !== 'string' || !find(list, id))) errors.push('Invalid ' + name + '.');
    };
    numeric(state.createdAt, 'creation time', false, 0, MAX_TIME);
    numeric(state.lastUpdate, 'simulation time', false, 0, MAX_TIME);
    map(state.resources, version === 1 ? C.RESOURCES.filter(x => x.id !== 'starshards') : C.RESOURCES, big, 'resources');
    [ ['upgrades', C.UPGRADES], ['refitUpgrades', C.REFIT_UPGRADES], ['legacy', C.LEGACY_UPGRADES] ].forEach(([key, list]) => map(state[key], list, (x, name) => numeric(x, name, true, 0, 1e6), key));
    map(state.ranks, C.ROOMS, (x, name) => numeric(x, name, true), 'ranks');
    map(state.mastery, C.ROOMS, big, 'mastery');
    ids(state.rooms, C.ROOMS, 'rooms');
    if (Array.isArray(state.rooms) && !state.rooms.includes('trail')) errors.push('Missing starting trail.');
    ids(state.research, C.RESEARCH, 'research');
    if (!object(state.route)) errors.push('Missing route.');
    else { numeric(state.route.index, 'route index', true); big(state.route.progress, 'route progress'); if (!find(C.MODES, state.route.mode)) errors.push('Invalid route mode.'); }
    if (!object(state.run)) errors.push('Missing expedition.');
    else { numeric(state.run.id, 'run id', true, 1); numeric(state.run.completed, 'completed route', true, -1); numeric(state.run.elapsed, 'run elapsed'); big(state.run.work, 'run work'); }
    if (!object(state.chapter)) errors.push('Missing charter.');
    else { big(state.chapter.work, 'chapter work'); numeric(state.chapter.refits, 'chapter refits', true); }
    if (!object(state.lifetime)) errors.push('Missing lifetime progress.');
    else {
      ['work', 'coins'].forEach(key => big(state.lifetime[key], 'lifetime ' + key));
      ['refits', 'charters', 'frontier'].forEach(key => numeric(state.lifetime[key], key, true));
      numeric(state.lifetime.highestRoute, 'highest route', true, -1);
    }
    if (!object(state.crew)) errors.push('Missing crew.');
    else {
      ids(state.crew.owned, C.SPECIALISTS, 'owned specialists'); ids(state.crew.companions, C.COMPANIONS, 'owned companions');
      if (!Array.isArray(state.crew.specialists) || state.crew.specialists.length !== 2 || state.crew.specialists.some(id => id !== null && (!find(C.SPECIALISTS, id) || !Array.isArray(state.crew.owned) || !state.crew.owned.includes(id))) || (state.crew.specialists[0] !== null && state.crew.specialists[0] === state.crew.specialists[1])) errors.push('Invalid specialist assignments.');
      if (state.crew.companion !== null && (!find(C.COMPANIONS, state.crew.companion) || !Array.isArray(state.crew.companions) || !state.crew.companions.includes(state.crew.companion))) errors.push('Invalid companion.');
    }
    if (!find(C.RECIPES.filter(x => x.id.startsWith('meal-')), state.meal)) errors.push('Invalid provision recipe.');
    if (!find(C.DOCTRINES, state.doctrine)) errors.push('Invalid doctrine.');
    if (!object(state.automations) || C.AUTOMATIONS.some(x => typeof state.automations[x.id] !== 'boolean')) errors.push('Invalid automation settings.');
    if (!object(state.challenges)) errors.push('Missing challenges.');
    else { ids(state.challenges.completed, C.CHALLENGES, 'completed challenges'); if (state.challenges.active !== null && !find(C.CHALLENGES, state.challenges.active)) errors.push('Invalid active challenge.'); }
    ids(state.collections, C.REALMS, 'collections');
    numeric(state.planner, 'planning clock', false, 0, 60);
    if (!Array.isArray(state.events) || state.events.length > 12 || state.events.some(x => typeof x !== 'string' || x.length > 300)) errors.push('Invalid event history.');
    if (version >= 2) {
      if (!object(state.premium)) errors.push('Missing premium progression.');
      else {
        numeric(state.premium.rng, 'premium random state', true, 1, 4294967295);
        numeric(state.premium.eligibleSeconds, 'premium eligible time', false, 0, MAX_TIME / 1000);
        numeric(state.premium.untilDrop, 'premium drop countdown', false, 0.001, MAX_DROP_SECONDS);
        numeric(state.premium.drops, 'premium drop count', true);
        ids(state.premium.owned, C.PREMIUM_ITEMS, 'earned premium ownership');
        ids(state.premium.claimedMilestones, C.PREMIUM_MILESTONES, 'premium milestones');
        // A paid banner preference may remain in a save, but conveys no ownership.
        if (state.premium.equipped !== null && !find(C.PREMIUM_ITEMS.filter(x => x.kind === 'banner'), state.premium.equipped)) errors.push('Invalid guild banner.');
        if (Object.keys(state.premium).some(key => !['rng', 'eligibleSeconds', 'untilDrop', 'drops', 'claimedMilestones', 'owned', 'equipped'].includes(key))) errors.push('Unknown premium progression field.');
        ['eligibleSeconds', 'untilDrop'].forEach(key => { const value = state.premium[key] * 1000; if (Math.abs(value - Math.round(value)) > Math.max(0.00001, Math.abs(value) * Number.EPSILON * 2)) errors.push('Premium time must use whole milliseconds.'); });
      }
    }
    // Reject semantic mismatches that can otherwise strand a valid-looking imported save.
    if (!errors.length) {
      if (state.route.index > state.lifetime.highestRoute + 1) errors.push('Route is not discovered.');
      if (state.run.completed > state.lifetime.highestRoute) errors.push('Run exceeds lifetime progress.');
      if (state.challenges.active && state.route.index > state.run.completed + 1) errors.push('Challenge route skips fresh expedition progress.');
      if (C.ROOMS.some(room => state.rooms.includes(room.id) && room.at > state.lifetime.highestRoute + 1)) errors.push('Room is not discovered.');
      if (N.cmp(state.route.progress, getRoute(state.route.index, version >= 4 ? state : null).distance) >= 0) errors.push('Route progress exceeds its target.');
      if (C.ROOMS.some(room => state.ranks[room.id] !== masteryRank(state.mastery[room.id]))) errors.push('Mastery rank does not match experience.');
      if (version >= 2) {
        const claims = state.premium.claimedMilestones;
        if (claims.includes('first-refit') !== (state.lifetime.refits > 0) || claims.includes('first-charter') !== (state.lifetime.charters > 0) || C.CHALLENGES.some(x => claims.includes('challenge-' + x.id) !== completed(state, x.id))) errors.push('Premium milestone record does not match completed progress.');
        const shards = state.resources.starshards;
        if (shards.e < 14 && Math.abs(N.toNumber(shards) - Math.round(N.toNumber(shards))) > 1e-7) errors.push('Starshards must be whole units.');
      }
    }
    if (version >= 3) validateLuck(state, errors, version);
    if (version >= 4) validateGuild(state, errors);
    if (version >= 4 && own(state, 'introductions')) {
      if (!exactKeys(state.introductions, ['seen'])) errors.push('Invalid presentation introductions.');
      else ids(state.introductions.seen, C.PRESENTATION_SYSTEMS, 'seen introductions');
    }
    return { valid: errors.length === 0, errors };
  }

  function validateState(state) { return validateSchema(state, VERSION); }

  function migrateState(input) {
    // Migration does not repair missing values, import premium ownership, or mint
    // rewards for historical play. Only the exact supported v1 schema is admitted.
    try {
      if (!object(input) || ![1, 2, 3].includes(input.schemaVersion) || !validateSchema(input, input.schemaVersion).valid) return null;
      const keys = Object.keys(createState(input.createdAt)).filter(key => !['guild', 'introductions'].includes(key) && (input.schemaVersion >= 3 || !['luck', 'caravan'].includes(key)) && (input.schemaVersion !== 1 || key !== 'premium'));
      if (Object.keys(input).length !== keys.length || Object.keys(input).some(key => !keys.includes(key))) return null;
      const expected = { route: ['index', 'progress', 'mode'], run: ['id', 'completed', 'work', 'elapsed'], chapter: ['work', 'refits'], lifetime: ['work', 'coins', 'refits', 'charters', 'highestRoute', 'frontier'], crew: ['owned', 'specialists', 'companions', 'companion'], challenges: ['active', 'completed'], automations: C.AUTOMATIONS.map(x => x.id) };
      if (Object.keys(expected).some(key => Object.keys(input[key]).length !== expected[key].length || Object.keys(input[key]).some(field => !expected[key].includes(field)))) return null;
      const out = clone(input);
      out.schemaVersion = VERSION;
      if (input.schemaVersion === 1) {
        out.resources.starshards = N.zero();
        out.premium = createPremium(out.createdAt);
        if (out.lifetime.refits) out.premium.claimedMilestones.push('first-refit');
        if (out.lifetime.charters) out.premium.claimedMilestones.push('first-charter');
        out.challenges.completed.forEach(id => out.premium.claimedMilestones.push('challenge-' + id));
      }
      if (input.schemaVersion < 3) { out.luck = createLuck(out.createdAt); out.caravan = createCaravan(out.createdAt); }
      else C.RELICS.filter(def => !own(out.luck.duplicateProgress, def.id)).forEach(def => { out.luck.duplicateProgress[def.id] = 0; });
      out.guild = createGuild(true);
      out.guild.chapterProject.number = out.lifetime.charters;
      out.introductions = { seen: presentationSystems(out).map(system => system.id) };
      return validateState(out).valid ? out : null;
    } catch (_) { return null; }
  }

  function normalizeState(input, now) {
    if (!validateState(input).valid) return createState(now);
    const out = createState(input.createdAt);
    // Only fields from the schema are admitted. Never merge untrusted keys into live objects.
    Object.keys(out).forEach(key => { if (key !== 'introductions' || own(input, key)) out[key] = clone(input[key]); });
    if (!own(input, 'introductions')) out.introductions = { seen: presentationSystems(out).map(system => system.id) };
    return out;
  }

  const exactKeys = (value, keys) => object(value) && Object.keys(value).length === keys.length && keys.every(key => own(value, key));
  const bounded = (value, max, integer) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= max && (!integer || Number.isSafeInteger(value));
  const validId = value => typeof value === 'string' && /^[A-Za-z0-9_-]{1,160}$/.test(value);
  const canonical = value => JSON.stringify(value, (key, item) => object(item) ? Object.fromEntries(Object.keys(item).sort().map(name => [name, item[name]])) : item);
  function createPlan() {
    return { reserves: Object.fromEntries(ORDINARY.map(id => [id, N.zero()])), goal: null, priorities: { operations: 'balanced', equipment: 'balanced' }, queue: [], kit: 'off', preparation: 'off' };
  }
  function createGuild(grandfathered) {
    return { grandfathered, projects: [], capabilities: [], supply: 'steady', prepared: { index: -1, kinds: [] }, chapterProject: { number: 0, choice: null }, plan: createPlan(), loadouts: [], audit: [] };
  }
  function validPlanAction(action) {
    return exactKeys(action, ['type', 'id']) && (action.type === 'buy' && !!find(C.UPGRADES, action.id) || action.type === 'research' && !!find(C.RESEARCH, action.id) || action.type === 'project' && (!!find(C.PROJECTS, action.id) || ['chapter-supply', 'chapter-survey', 'chapter-industry'].includes(action.id)));
  }
  function validPlan(plan) {
    return exactKeys(plan, ['reserves', 'goal', 'priorities', 'queue', 'kit', 'preparation']) && exactKeys(plan.reserves, ORDINARY) && ORDINARY.every(id => N.valid(plan.reserves[id])) && (plan.goal === null || validPlanAction(plan.goal)) && exactKeys(plan.priorities, ['operations', 'equipment']) && ['balanced', 'travel', 'production'].includes(plan.priorities.operations) && ['balanced', 'tools', 'boots', 'research'].includes(plan.priorities.equipment) && Array.isArray(plan.queue) && plan.queue.length <= 6 && plan.queue.every(validPlanAction) && ['off', 'mining', 'travel'].includes(plan.kit) && ['off', 'scout', 'supply', 'survey'].includes(plan.preparation);
  }
  function validateGuild(state, errors) {
    const g = state.guild;
    const ids = (value, list) => Array.isArray(value) && new Set(value).size === value.length && value.every(id => !!find(list, id));
    if (!exactKeys(g, ['grandfathered', 'projects', 'capabilities', 'supply', 'prepared', 'chapterProject', 'plan', 'loadouts', 'audit'])) { errors.push('Invalid guild development.'); return; }
    if (typeof g.grandfathered !== 'boolean' || !ids(g.projects, C.PROJECTS) || !ids(g.capabilities, C.CAPABILITIES) || !['save', 'steady', 'push'].includes(g.supply)) errors.push('Invalid guild progression.');
    if (!exactKeys(g.prepared, ['index', 'kinds']) || !Number.isSafeInteger(g.prepared.index) || g.prepared.index < -1 || g.prepared.index > 1e9 || !ids(g.prepared.kinds, C.PREPARATIONS) || g.prepared.index !== state.route.index && g.prepared.kinds.length) errors.push('Invalid route preparation.');
    if (!exactKeys(g.chapterProject, ['number', 'choice']) || !bounded(g.chapterProject.number, state.lifetime.charters, true) || ![null, 'supply', 'survey', 'industry'].includes(g.chapterProject.choice) || g.chapterProject.number !== state.lifetime.charters) errors.push('Invalid charter project.');
    if (!validPlan(g.plan)) errors.push('Invalid automation plan.');
    if (!Array.isArray(g.loadouts) || g.loadouts.length > 3 || new Set(g.loadouts.map(x => x && x.id)).size !== g.loadouts.length || g.loadouts.some(x => !exactKeys(x, ['id', 'name', 'plan', 'specialists', 'companion', 'doctrine', 'meal', 'relic', 'supply', 'mode']) || ![0, 1, 2].includes(x.id) || typeof x.name !== 'string' || !x.name.trim() || x.name.length > 40 || !validPlan(x.plan) || !Array.isArray(x.specialists) || x.specialists.length !== 2 || x.specialists.some(id => id !== null && !state.crew.owned.includes(id)) || x.specialists[0] !== null && x.specialists[0] === x.specialists[1] || x.companion !== null && !state.crew.companions.includes(x.companion) || !find(C.DOCTRINES, x.doctrine) || !find(C.RECIPES, x.meal) || x.relic !== null && !state.luck.owned.includes(x.relic) || !['save', 'steady', 'push'].includes(x.supply) || !find(C.MODES, x.mode))) errors.push('Invalid saved loadouts.');
    if (!Array.isArray(g.audit) || g.audit.length > 20 || g.audit.some(x => !exactKeys(x, ['at', 'message']) || !bounded(x.at, MAX_TIME) || typeof x.message !== 'string' || x.message.length > 250)) errors.push('Invalid planning audit.');
  }
  function validQuote(quote) {
    if (!exactKeys(quote, ['version', 'offerId', 'kind', 'golden', 'issuedAt', 'runId', 'charters', 'reward']) || quote.version !== 1 || !validId(quote.offerId) || !['shipment', 'surge', 'relic'].includes(quote.kind) || typeof quote.golden !== 'boolean' || !bounded(quote.issuedAt, MAX_TIME) || !bounded(quote.runId, 1e9, true) || quote.runId < 1 || !bounded(quote.charters, 1e9, true)) return false;
    const reward = quote.reward;
    const keys = quote.kind === 'shipment' ? ['resources'] : ['resources', quote.kind];
    if (!exactKeys(reward, keys) || !object(reward.resources) || Object.keys(reward.resources).some(id => !ORDINARY.includes(id) || !N.valid(reward.resources[id]) || !exactKeys(reward.resources[id], ['m', 'e']))) return false;
    if (quote.kind === 'shipment') return Object.keys(reward.resources).length >= 1 && Object.keys(reward.resources).length <= 2 && own(reward.resources, 'coins');
    if (quote.kind === 'surge') return !Object.keys(reward.resources).length && exactKeys(reward.surge, ['resource', 'multiplier', 'seconds']) && ORDINARY.includes(reward.surge.resource) && reward.surge.multiplier === 3 && reward.surge.seconds === (quote.golden ? 5400 : 2700);
    return Object.keys(reward.resources).length <= 2 && exactKeys(reward.relic, ['id', 'progress']) && !!find(C.RELICS, reward.relic.id) && reward.relic.progress === (quote.golden ? 40 : 20);
  }
  function validateLuck(state, errors, version) {
    const luck = state.luck, caravan = state.caravan;
    const relics = version < 4 ? C.RELICS.filter(def => def.chapter === 0) : C.RELICS;
    const ids = (value, content) => Array.isArray(value) && new Set(value).size === value.length && value.every(id => !!find(content, id));
    const target = id => id === 'balanced' || !!find(relics, id);
    if (!exactKeys(luck, LUCK_KEYS)) { errors.push('Invalid discovery state.'); return; }
    if (!bounded(luck.rng, 4294967295, true) || !luck.rng || !bounded(luck.commonRemainingMs, 720000) || !bounded(luck.relicRemainingMs, 28800000) || !bounded(luck.pitySeconds, 28800) || !ids(luck.owned, C.RELICS) || luck.active !== null && !luck.owned.includes(luck.active) || !target(luck.hunt) || !target(luck.scheduledHunt) || !ids(luck.research, C.LUCK_RESEARCH) || typeof luck.collectionBanner !== 'boolean' || !bounded(luck.commonFinds, Number.MAX_SAFE_INTEGER, true) || !bounded(luck.relicFinds, Number.MAX_SAFE_INTEGER, true)) errors.push('Invalid relic progression.');
    if (!exactKeys(luck.duplicateProgress, relics.map(x => x.id)) || relics.some(x => !bounded(luck.duplicateProgress[x.id], 100))) errors.push('Invalid relic study progress.');
    if (!exactKeys(luck.kit, ['prepared', 'active', 'remainingSeconds']) || ![null, 'mining', 'travel'].includes(luck.kit.prepared) || ![null, 'mining', 'travel'].includes(luck.kit.active) || !bounded(luck.kit.remainingSeconds, 1800) || (luck.kit.active === null) !== (luck.kit.remainingSeconds === 0)) errors.push('Invalid expedition kit.');
    const ledger = luck.ledger;
    if (!exactKeys(ledger, ['seq', 'seen', 'recent']) || !bounded(ledger.seq, Number.MAX_SAFE_INTEGER, true) || !bounded(ledger.seen, ledger.seq, true) || !Array.isArray(ledger.recent) || ledger.recent.length > 32 || ledger.recent.some((entry, i) => !object(entry) || !bounded(entry.seq, ledger.seq, true) || entry.seq < 1 || i && entry.seq <= ledger.recent[i - 1].seq || !['common', 'rare', 'epic', 'legendary'].includes(entry.rarity) || !['type', 'title', 'reward', 'effect'].every(key => typeof entry[key] === 'string' && entry[key].length <= 350) || !bounded(entry.at, MAX_TIME) || entry.relicId !== undefined && !find(C.RELICS, entry.relicId))) errors.push('Invalid discovery ledger.');
    if (Array.isArray(luck.owned) && (luck.owned.some(id => !find(relics, id)) || luck.collectionBanner !== C.RELICS.filter(def => def.chapter === 0).every(def => luck.owned.includes(def.id)))) errors.push('Relic collection reward does not match ownership.');
    if (!exactKeys(caravan, CARAVAN_KEYS)) { errors.push('Invalid caravan state.'); return; }
    if (!bounded(caravan.rng, 4294967295, true) || !caravan.rng || !bounded(caravan.remainingMs, 5400000) || !bounded(caravan.sequence, Number.MAX_SAFE_INTEGER, true)) errors.push('Invalid caravan clock.');
    if (!Array.isArray(caravan.completed) || caravan.completed.length > 3 || caravan.completed.some((at, i) => !bounded(at, MAX_TIME) || i && at < caravan.completed[i - 1])) errors.push('Invalid caravan completion history.');
    if (!Array.isArray(caravan.receipts) || caravan.receipts.length > 256 || caravan.receipts.some(id => !validId(id)) || new Set(caravan.receipts).size !== caravan.receipts.length) errors.push('Invalid caravan receipts.');
    if (!exactKeys(caravan.surge, ['resource', 'remainingSeconds']) || ![null, ...ORDINARY].includes(caravan.surge.resource) || !bounded(caravan.surge.remainingSeconds, 5400) || (caravan.surge.resource === null) !== (caravan.surge.remainingSeconds === 0)) errors.push('Invalid caravan surge.');
    if (caravan.offer !== null) {
      const offer = caravan.offer;
      if (!exactKeys(offer, ['id', 'golden', 'minutes', 'arrivedAt', 'quote', 'locked']) || !validId(offer.id) || typeof offer.golden !== 'boolean' || typeof offer.locked !== 'boolean' || offer.locked && !offer.quote || !bounded(offer.minutes, 120, true) || offer.minutes < 90 || !bounded(offer.arrivedAt, MAX_TIME) || offer.quote !== null && (!validQuote(offer.quote) || offer.quote.offerId !== offer.id || offer.quote.golden !== offer.golden)) errors.push('Invalid caravan offer.');
    }
    if (caravan.pendingQuote !== null && (!validQuote(caravan.pendingQuote) || !caravan.offer || !caravan.offer.locked || caravan.pendingQuote.offerId !== caravan.offer.id || canonical(caravan.pendingQuote) !== canonical(caravan.offer.quote))) errors.push('Invalid pending caravan quote.');
    if (version < 4 && (ledger.recent.some(entry => entry.relicId && !find(relics, entry.relicId)) || [caravan.pendingQuote, caravan.offer && caravan.offer.quote].some(quote => quote && quote.reward.relic && !find(relics, quote.reward.relic.id)))) errors.push('Relic was not part of this legacy schema.');
  }
  function recordDiscovery(state, type, rarity, title, reward, effect, at, relicId) {
    const ledger = state.luck.ledger;
    ledger.seq += 1;
    const entry = { seq: ledger.seq, type, rarity, title, reward, effect, at: at === undefined ? state.lastUpdate : at };
    if (relicId) entry.relicId = relicId;
    ledger.recent.push(entry);
    if (ledger.recent.length > 32) ledger.recent.shift();
  }
  function unlockedMaterials(state) { return ORDINARY.filter(id => unlocked(state, find(C.RESOURCES, id).room)); }
  function eligibleRelics(state) { return C.RELICS.filter(def => def.chapter <= state.lifetime.charters); }
  function missingRelic(state, preferred) { const eligible = eligibleRelics(state); return eligible.find(x => x.id === preferred && !state.luck.owned.includes(x.id)) || eligible.find(x => !state.luck.owned.includes(x.id)); }
  function awardRelic(state, id, at, reason) {
    const relic = find(C.RELICS, id);
    if (state.luck.owned.includes(id)) return false;
    state.luck.owned.push(id); state.luck.duplicateProgress[id] = 100;
    recordDiscovery(state, 'relic', relic.rarity, relic.name + ' discovered', reason || 'Permanent relic added to your collection.', relic.description, at, id);
    if (!state.luck.collectionBanner && C.RELICS.filter(def => def.chapter === 0).every(def => state.luck.owned.includes(def.id))) {
      state.luck.collectionBanner = true;
      recordDiscovery(state, 'collection', 'legendary', 'The Founders’ Collection', 'Constellation pennant unlocked permanently.', 'A cosmetic guild pennant celebrates all three relic discoveries.', at);
    }
    return true;
  }
  function addRelicProgress(state, preferred, amount, at, reason) {
    const target = missingRelic(state, preferred);
    if (!target) return false;
    state.luck.duplicateProgress[target.id] = Math.min(100, state.luck.duplicateProgress[target.id] + amount);
    if (state.luck.duplicateProgress[target.id] >= 100) awardRelic(state, target.id, at, reason);
    else recordDiscovery(state, 'relic-progress', target.rarity, target.name + ' study', '+' + amount + '% guaranteed progress · ' + state.luck.duplicateProgress[target.id] + '/100%', reason || 'Complete the study to discover this relic without another roll.', at, target.id);
    return true;
  }
  function pityLimit(state) { return state.luck.relicFinds === 0 ? 2700 : state.luck.research.includes('relic-lore') ? 21600 : 28800; }
  function commonFind(state, at) {
    const luck = state.luck;
    const materials = unlockedMaterials(state);
    const id = materials[Math.min(materials.length - 1, Math.floor(random(luck) * materials.length))];
    const scale = luck.research.includes('careful-salvage') ? 1.25 : 1;
    const rates = getRates(state, true);
    const amount = N.mul(rates.gain[id], (30 + random(luck) * 30) * scale);
    state.resources[id] = N.add(state.resources[id], amount);
    luck.commonFinds += 1;
    recordDiscovery(state, 'supplies', 'common', 'A useful trail find', '+' + N.format(amount) + ' ' + find(C.RESOURCES, id).name.toLowerCase(), 'Supplies have already joined your stockpile.', at);
    luck.commonRemainingMs = Math.floor((480 + random(luck) * 240) * 1000);
  }
  function relicFind(state, at) {
    const luck = state.luck;
    const guaranteed = luck.pitySeconds >= pityLimit(state) - EPS;
    const missing = missingRelic(state, luck.scheduledHunt);
    let relic;
    if (guaranteed && missing) relic = missing;
    else {
      const eligible = eligibleRelics(state);
      const weights = eligible.map(x => x.weight * (x.id === luck.scheduledHunt ? 4 : 1));
      let roll = random(luck) * weights.reduce((a, b) => a + b, 0);
      relic = eligible.find((x, i) => { roll -= weights[i]; return roll < 0; }) || eligible[0];
    }
    if (!awardRelic(state, relic.id, at, guaranteed ? 'A guaranteed discovery: your patient search paid off.' : 'A rare expedition discovery.')) {
      const amount = luck.research.includes('duplicate-study') ? 25 : 20;
      if (!addRelicProgress(state, luck.scheduledHunt, amount, at, 'A familiar ' + relic.name + ' became guaranteed study progress.')) {
        const coins = N.mul(getRates(state, true).gain.coins, 300);
        state.resources.coins = N.add(state.resources.coins, coins);
        recordDiscovery(state, 'duplicate', relic.rarity, 'A familiar ' + relic.name, '+' + N.format(coins) + ' coins', 'Your collection is complete; duplicate finds become a modest supply reward.', at, relic.id);
      }
    }
    luck.relicFinds += 1; luck.pitySeconds = 0; luck.scheduledHunt = luck.hunt;
    const mean = luck.research.includes('relic-lore') ? 14400 : 21600;
    luck.relicRemainingMs = Math.min(pityLimit(state) * 1000, Math.max(1000, Math.ceil(-Math.log(random(luck)) * mean * 1000)));
  }
  function caravanArrives(state, at) {
    const caravan = state.caravan;
    caravan.sequence += 1;
    const golden = random(caravan) < 0.1;
    const minutes = 90 + Math.floor(random(caravan) * 31);
    caravan.offer = { id: 'wg-caravan-' + Math.floor(state.createdAt).toString(36) + '-' + caravan.sequence.toString(36) + '-' + caravan.rng.toString(36), golden, minutes, arrivedAt: at, quote: null, locked: false };
    caravan.remainingMs = 0;
    recordDiscovery(state, 'caravan', golden ? 'legendary' : 'rare', golden ? 'A golden caravan has arrived' : 'A caravan has arrived', golden ? 'Every quoted reward is doubled.' : 'Choose a guaranteed reward before an optional ad.', 'This offer stays until used or dismissed. Free progress continues.', at);
  }
  function tickLuck(state, dt, at, eligible) {
    const luck = state.luck, caravan = state.caravan;
    if (eligible.mine) luck.commonRemainingMs = Math.max(0, luck.commonRemainingMs - dt * 1000);
    if (eligible.study) { luck.relicRemainingMs = Math.max(0, luck.relicRemainingMs - dt * 1000); luck.pitySeconds += dt; }
    if (eligible.forge && !caravan.offer) caravan.remainingMs = Math.max(0, caravan.remainingMs - dt * 1000);
    if (luck.kit.active) { luck.kit.remainingSeconds = Math.max(0, luck.kit.remainingSeconds - dt); if (luck.kit.remainingSeconds <= EPS) { luck.kit.active = null; luck.kit.remainingSeconds = 0; } }
    if (caravan.surge.resource) { caravan.surge.remainingSeconds = Math.max(0, caravan.surge.remainingSeconds - dt); if (caravan.surge.remainingSeconds <= EPS) { caravan.surge.resource = null; caravan.surge.remainingSeconds = 0; } }
    if (eligible.mine && luck.commonRemainingMs <= EPS * 1000) commonFind(state, at);
    if (eligible.study && (luck.relicRemainingMs <= EPS * 1000 || luck.pitySeconds >= pityLimit(state) - EPS)) relicFind(state, at);
    if (eligible.forge && !caravan.offer && caravan.remainingMs <= EPS * 1000) caravanArrives(state, at);
  }
  // Between production-changing events, supply finds are additive. Replay their
  // retained RNG in order, aggregate amounts, and materialize only the recent
  // ledger. This preserves every roll without millions of full economy passes.
  function batchLuck(state, seconds, start, rates, eligible) {
    const luck = state.luck, caravan = state.caravan, materials = unlockedMaterials(state);
    const sums = Object.fromEntries(ORDINARY.map(id => [id, 0]));
    const scale = luck.research.includes('careful-salvage') ? 1.25 : 1;
    const end = seconds * 1000;
    let elapsed = 0;
    function recent(type, rarity, title, id, units, effect, at, relicId) {
      const ledger = luck.ledger;
      ledger.seq += 1;
      // At least 32 later common finds replace these older display-only entries.
      if (end - (at - start) > 24000000) return;
      const entry = { seq: ledger.seq, type, rarity, title, reward: '+' + N.format(N.mul(rates.gain[id], units)) + ' ' + find(C.RESOURCES, id).name.toLowerCase(), effect, at };
      if (relicId) entry.relicId = relicId;
      ledger.recent.push(entry);
      if (ledger.recent.length > 32) ledger.recent.shift();
    }
    while (elapsed < end - EPS * 1000) {
      const common = eligible.mine ? luck.commonRemainingMs : Infinity;
      const relic = eligible.study ? Math.min(luck.relicRemainingMs, Math.max(0, pityLimit(state) - luck.pitySeconds) * 1000) : Infinity;
      const arrival = eligible.forge && !caravan.offer ? caravan.remainingMs : Infinity;
      const dt = Math.min(end - elapsed, common, relic, arrival);
      if (eligible.mine) luck.commonRemainingMs = Math.max(0, luck.commonRemainingMs - dt);
      if (eligible.study) { luck.relicRemainingMs = Math.max(0, luck.relicRemainingMs - dt); luck.pitySeconds += dt / 1000; }
      if (eligible.forge && !caravan.offer) caravan.remainingMs = Math.max(0, caravan.remainingMs - dt);
      elapsed += dt;
      const at = start + elapsed;
      if (eligible.mine && luck.commonRemainingMs <= EPS * 1000) {
        const id = materials[Math.min(materials.length - 1, Math.floor(random(luck) * materials.length))];
        const units = (30 + random(luck) * 30) * scale;
        sums[id] += units; luck.commonFinds += 1;
        recent('supplies', 'common', 'A useful trail find', id, units, 'Supplies have already joined your stockpile.', at);
        luck.commonRemainingMs = Math.floor((480 + random(luck) * 240) * 1000);
      }
      if (eligible.study && (luck.relicRemainingMs <= EPS * 1000 || luck.pitySeconds >= pityLimit(state) - EPS)) {
        if (missingRelic(state)) relicFind(state, at);
        else {
          const eligible = eligibleRelics(state);
          const weights = eligible.map(x => x.weight * (x.id === luck.scheduledHunt ? 4 : 1));
          let roll = random(luck) * weights.reduce((a, b) => a + b, 0);
          const found = eligible.find((x, i) => { roll -= weights[i]; return roll < 0; }) || eligible[0];
          sums.coins += 300;
          recent('duplicate', found.rarity, 'A familiar ' + found.name, 'coins', 300, 'Your collection is complete; duplicate finds become a modest supply reward.', at, found.id);
          luck.relicFinds += 1; luck.pitySeconds = 0; luck.scheduledHunt = luck.hunt;
          const mean = luck.research.includes('relic-lore') ? 14400 : 21600;
          luck.relicRemainingMs = Math.min(pityLimit(state) * 1000, Math.max(1000, Math.ceil(-Math.log(random(luck)) * mean * 1000)));
        }
      }
      if (eligible.forge && !caravan.offer && caravan.remainingMs <= EPS * 1000) caravanArrives(state, at);
    }
    ORDINARY.forEach(id => { if (sums[id]) state.resources[id] = N.add(state.resources[id], N.mul(rates.gain[id], sums[id])); });
    luck.ledger.recent = luck.ledger.recent.filter(entry => entry.seq > luck.ledger.seq - 32);
    if (luck.kit.active) { luck.kit.remainingSeconds = Math.max(0, luck.kit.remainingSeconds - seconds); if (luck.kit.remainingSeconds <= EPS) { luck.kit.active = null; luck.kit.remainingSeconds = 0; } }
    if (caravan.surge.resource) { caravan.surge.remainingSeconds = Math.max(0, caravan.surge.remainingSeconds - seconds); if (caravan.surge.remainingSeconds <= EPS) { caravan.surge.resource = null; caravan.surge.remainingSeconds = 0; } }
  }
  function luckEventTime(state, batch) {
    const values = [];
    if (!batch && unlocked(state, 'mine')) values.push(state.luck.commonRemainingMs / 1000);
    if (!batch && unlocked(state, 'study')) values.push(state.luck.relicRemainingMs / 1000, pityLimit(state) - state.luck.pitySeconds);
    if (!batch && unlocked(state, 'forge') && !state.caravan.offer) values.push(state.caravan.remainingMs / 1000);
    if (state.luck.kit.active) values.push(state.luck.kit.remainingSeconds);
    if (state.caravan.surge.resource) values.push(state.caravan.surge.remainingSeconds);
    return values.length ? Math.max(EPS, Math.min(...values)) : Infinity;
  }
  function kitCosts(state) { const material = power(3.5, tier(state) - 1); return { ore: product(material, 20, productionInvestment(state, 'ore')), provisions: product(material, 12, productionInvestment(state, 'provisions')), knowledge: product(material, 8, productionInvestment(state, 'knowledge')) }; }
  function selectCaravan(state, action) {
    const caravan = state.caravan, offer = caravan.offer;
    if (!offer || caravan.pendingQuote) return { ok: false, message: caravan.pendingQuote ? 'A quoted reward is awaiting verified ad completion.' : 'Wait for a caravan to arrive.' };
    if (offer.locked) return { ok: false, message: 'This arrival’s reward was fixed when its first ad was requested. Retry it or dismiss this arrival.' };
    if (!['shipment', 'surge', 'relic'].includes(action.kind)) return { ok: false, message: 'Choose a caravan reward.' };
    const reward = { resources: {} }, rates = getRates(state, true), materials = unlockedMaterials(state), factor = offer.golden ? 2 : 1;
    if (action.kind === 'shipment') {
      const material = action.material || materials.filter(id => id !== 'coins').slice(-1)[0];
      if (!material || material === 'coins' || !materials.includes(material)) return { ok: false, message: 'Choose an unlocked material.' };
      reward.resources.coins = N.mul(rates.gain.coins, offer.minutes * 60 * factor);
      reward.resources[material] = N.mul(rates.gain[material], offer.minutes * 60 * factor);
    } else if (action.kind === 'surge') {
      const resource = action.resource || 'ore';
      if (!materials.includes(resource)) return { ok: false, message: 'Choose an unlocked profession.' };
      reward.surge = { resource, multiplier: 3, seconds: 2700 * factor };
    } else {
      if (action.relicId && !find(eligibleRelics(state), action.relicId)) return { ok: false, message: 'Earn this relic’s chapter before choosing it as a reward.' };
      const target = missingRelic(state, action.relicId || state.luck.hunt);
      if (!unlocked(state, 'study') || !target) return { ok: false, message: 'Relic study needs a missing relic and the Study.' };
      reward.relic = { id: target.id, progress: 20 * factor };
      reward.resources.coins = N.mul(rates.gain.coins, 300 * factor);
    }
    offer.quote = { version: 1, offerId: offer.id, kind: action.kind, golden: offer.golden, issuedAt: Math.floor(state.lastUpdate), runId: state.run.id, charters: state.lifetime.charters, reward };
    return { ok: true, message: 'Guaranteed reward selected. Its exact amounts are shown before any ad.' };
  }
  function getCaravanQuote(state) { return clone(state.caravan.pendingQuote || state.caravan.offer && state.caravan.offer.quote || null); }
  function beginCaravanReward(state, quote) {
    const selected = getCaravanQuote(state);
    const recent = state.caravan.completed.filter(at => at > state.lastUpdate - DAY);
    if (recent.length >= 3) return { ok: false, message: 'Three caravan rewards have been completed in the last 24 hours. Your offer will wait.' };
    if (!validQuote(quote) || !selected || canonical(quote) !== canonical(selected)) return { ok: false, message: 'Review the current caravan quote before starting an ad.' };
    state.caravan.pendingQuote = clone(selected);
    state.caravan.offer.locked = true;
    return { ok: true, message: 'The displayed reward is reserved until verified completion.' };
  }
  function cancelCaravanReward(state, offerId) {
    if (!state.caravan.pendingQuote || state.caravan.pendingQuote.offerId !== offerId) return { ok: false, message: 'No matching ad reward is pending.' };
    state.caravan.pendingQuote = null;
    return { ok: true, message: 'No reward was claimed. Your caravan remains available.' };
  }
  function grantCaravanReward(state, receipt) {
    if (!exactKeys(receipt, ['receiptId', 'offerId', 'quote', 'completedAt']) || !validId(receipt.receiptId) || !validId(receipt.offerId) || !validQuote(receipt.quote) || receipt.offerId !== receipt.quote.offerId || !bounded(receipt.completedAt, MAX_TIME) || receipt.completedAt < receipt.quote.issuedAt) return { ok: false, message: 'A verified matching caravan receipt is required.' };
    if (state.caravan.receipts.includes(receipt.receiptId)) return { ok: true, duplicate: true, message: 'This verified reward was already applied.' };
    const quote = state.caravan.pendingQuote;
    if (!quote || canonical(quote) !== canonical(receipt.quote)) return { ok: false, message: 'This receipt does not match a reserved caravan reward.' };
    const completions = state.caravan.completed.filter(at => at > receipt.completedAt - DAY);
    if (completions.length >= 3 || receipt.completedAt > state.lastUpdate + 300000) return { ok: false, message: 'The receipt is outside the current verified reward allowance.' };
    const reward = quote.reward;
    Object.keys(reward.resources).forEach(id => { state.resources[id] = N.add(state.resources[id], reward.resources[id]); });
    if (reward.surge) {
      const surge = state.caravan.surge;
      surge.remainingSeconds = Math.min(5400, (surge.resource === reward.surge.resource ? surge.remainingSeconds : 0) + reward.surge.seconds);
      surge.resource = reward.surge.resource;
    }
    if (reward.relic) {
      if (!addRelicProgress(state, reward.relic.id, reward.relic.progress, state.lastUpdate, 'Guaranteed progress from a verified caravan reward.')) {
        state.resources.coins = N.add(state.resources.coins, reward.resources.coins || N.zero());
      }
    }
    state.caravan.completed = completions.concat(receipt.completedAt).sort((a, b) => a - b);
    state.caravan.receipts.push(receipt.receiptId); if (state.caravan.receipts.length > 256) state.caravan.receipts.shift();
    state.caravan.offer = null; state.caravan.pendingQuote = null;
    state.caravan.remainingMs = Math.floor((3600 + random(state.caravan) * 1800) * 1000);
    recordDiscovery(state, 'caravan-reward', quote.golden ? 'legendary' : 'rare', 'Caravan reward delivered', caravanRewardText(quote), 'Granted from a verified receipt. Watching or timing an ad locally never grants a reward.', state.lastUpdate);
    return { ok: true, message: 'Your guaranteed caravan reward has been delivered.' };
  }
  function caravanRewardText(quote) {
    if (!quote) return 'Select a reward to see its exact guaranteed contents.';
    const reward = quote.reward;
    const resources = Object.keys(reward.resources).map(id => N.format(reward.resources[id]) + ' ' + find(C.RESOURCES, id).name.toLowerCase());
    if (reward.surge) resources.push('3× ' + find(C.RESOURCES, reward.surge.resource).name.toLowerCase() + ' for ' + reward.surge.seconds / 60 + ' minutes');
    if (reward.relic) resources.push(reward.relic.progress + '% ' + find(C.RELICS, reward.relic.id).name + ' study (if already found: another missing relic; if collection complete: another ' + N.format(reward.resources.coins || N.zero()) + ' coins)');
    return resources.join(' + ');
  }

  function getRoute(index, state) {
    const revised = state && state.guild && !state.guild.grandfathered;
    const opening = [1100, 2200, 30000, 350000, 1200000, 200000000000, 100000000000000, 500000000000000, 2000000000000000];
    if (index < C.ROUTES.length) return Object.assign({}, C.ROUTES[index], { distance: revised ? index < opening.length ? N.from(opening[index]) : product(opening[8], power(4, index - 8)) : N.from(C.ROUTES[index].distance) });
    const frontier = index - C.ROUTES.length + 1;
    return { id: 'route-' + index, index, name: 'Frontier ' + frontier, realm: 'frontier', realmName: 'Endless Frontier', tier: 6 + Math.ceil(frontier / 3), material: 'Astral alloy', distance: revised ? product(opening[8], power(4, index - 8)) : product(C.ROUTES[17].distance, power(8, index - 17)), description: 'An endless expedition with stronger gathering tiers and new prestige targets.' };
  }
  function masteryRank(xp) { return Math.max(0, Math.floor(N.log10(N.add(N.div(xp, 300), 1)) / Math.log10(2) + 1e-10)); }
  function nextMastery(rank) { return N.mul(N.sub(power(2, rank + 1), 1), 300); }
  function tier(state) { return Math.max(1, 1 + Math.floor((state.lifetime.highestRoute + 1) / 3)); }
  function hasSpecialist(state, id) { return state.challenges.active !== 'quiet-company' && state.crew.specialists.includes(id); }
  function hasCompanion(state, id) { return state.challenges.active !== 'quiet-company' && state.crew.companion === id; }
  function hasCapability(state, id) { return state.guild.capabilities.includes(id); }
  function prepared(state, id) { return state.guild.prepared.index === state.route.index && state.guild.prepared.kinds.includes(id); }
  function caravanUnlocked(state) { return unlocked(state, 'forge') && (state.guild.grandfathered || state.lifetime.refits > 0) || !!state.caravan.offer; }
  function preparationCosts(state, id) {
    const def = find(C.PREPARATIONS, id);
    if (!def) return null;
    const scale = power(3.5, Math.max(0, getRoute(state.route.index, state).tier - 1));
    return Object.fromEntries(['maps', 'provisions', 'knowledge'].filter(resource => def[resource]).map(resource => [resource, product(scale, def[resource], productionInvestment(state, resource))]));
  }
  function productionInvestment(state, resource) {
    if (state.guild.grandfathered) return N.from(1);
    if (resource === 'maps') return product(power(1.35, level(state, 'surveyors')), power(1.4, level(state, 'gear-instruments')));
    if (resource === 'knowledge') return product(power(1.35, level(state, 'scholars')), power(1.4, level(state, 'gear-instruments')));
    if (resource === 'provisions') return power(1.3, level(state, 'cooks'));
    if (resource === 'herbs') return power(1.35, level(state, 'foragers'));
    if (resource === 'ore') return product(power(1.32, level(state, 'miners')), power(1.45, level(state, 'gear-tools')));
    return N.from(1);
  }
  function continuousReserve(state, resource) {
    const plan = state.guild.plan;
    let floor = N.from(plan.reserves[resource] || 0);
    const tasks = [plan.goal, plan.queue[0]].filter((task, i, list) => task && list.findIndex(other => other && canonical(other) === canonical(task)) === i);
    tasks.forEach(action => { const detail = taskDetails(state, action); if (detail && !detail.done && detail.costs[resource]) floor = N.add(floor, detail.costs[resource]); });
    return floor;
  }
  function regionCondition(state) {
    const realm = getRoute(state.route.index, state).realm;
    if (state.guild.grandfathered) return { text: 'Established expedition rules are preserved.', factor: 1 };
    const condition = { mistwood: { id: 'supply', text: 'Mistwood: a field camp bypasses a 25% travel detour.' }, frostpass: { id: 'scout', text: 'Frostpass: scouting bypasses a 30% travel detour.' }, sunkenreach: { id: 'survey', text: 'Sunken Reach: a survey bypasses a 30% travel detour.' }, starfall: { id: 'scout', text: 'Starfall Heights: scouting bypasses a 35% travel detour.' } }[realm];
    return condition ? { text: condition.text + (prepared(state, condition.id) ? ' Preparation active.' : ' Ordinary travel continues.'), factor: prepared(state, condition.id) ? 1 : realm === 'mistwood' ? 0.75 : realm === 'starfall' ? 0.65 : 0.7 } : { text: 'Open paths: no regional detour.', factor: 1 };
  }
  function getRates(state, unboosted) {
    const rates = dictionary(C.RESOURCES, N.zero);
    const drain = dictionary(C.RESOURCES, N.zero);
    const xp = dictionary(C.ROOMS, N.zero);
    const equip = id => state.challenges.active === 'old-tools' ? 0 : level(state, id);
    const master = room => power(1.08, state.ranks[room]);
    const material = power(3.5, tier(state) - 1);
    const collection = power(1.08, state.collections.length);
    const foundations = power(1.3, state.legacy.foundations);
    const specialists = researched(state, 'specialist-training');
    let travel = product(power(1.28, level(state, 'boots')), power(1.35, equip('gear-boots')), power(1.15, level(state, 'preparation')), power(1.2, state.refitUpgrades.pace), power(1.35, state.legacy.waystones), master('trail'), collection);
    rates.coins = product(0.3, power(1.18, level(state, 'boots')), power(1.2, equip('gear-boots')), material, foundations, master('trail'), collection);
    if (unlocked(state, 'mine')) rates.ore = product(state.guild.grandfathered ? 0.025 : 0.018, power(1.32, level(state, 'miners')), power(1.45, equip('gear-tools')), material, power(1.25, state.refitUpgrades.supply), foundations, master('mine'), collection);
    if (unlocked(state, 'forage')) rates.herbs = product(0.025, power(1.35, level(state, 'foragers')), material, power(1.25, state.refitUpgrades.supply), foundations, master('forage'), collection);
    if (unlocked(state, 'study')) rates.knowledge = product(0.015, power(1.35, level(state, 'scholars')), power(1.4, equip('gear-instruments')), material, power(1.25, state.refitUpgrades.insight), power(1.35, state.legacy.curriculum), master('study'), collection);
    if (unlocked(state, 'cartography')) rates.maps = product(0.006, power(1.35, level(state, 'surveyors')), power(1.4, equip('gear-instruments')), material, power(1.25, state.refitUpgrades.insight), master('cartography'), collection);
    if (hasSpecialist(state, 'scout')) travel = N.mul(travel, 1.4 * (specialists ? 1 + 0.04 * state.ranks.mine : 1));
    if (hasSpecialist(state, 'prospector')) { rates.ore = N.mul(rates.ore, 1.65); if (specialists) rates.herbs = N.mul(rates.herbs, 1 + 0.04 * state.ranks.forge); }
    if (hasSpecialist(state, 'naturalist')) rates.herbs = N.mul(rates.herbs, 1.7);
    if (hasSpecialist(state, 'scholar')) { rates.knowledge = N.mul(rates.knowledge, 1.65); if (specialists) rates.maps = N.mul(rates.maps, 1 + 0.05 * state.ranks.kitchen); }
    if (hasCompanion(state, 'fox')) { travel = N.mul(travel, 1.2); rates.coins = N.mul(rates.coins, 1.3); }
    if (hasCompanion(state, 'owl')) { rates.knowledge = N.mul(rates.knowledge, 1.4); rates.maps = N.mul(rates.maps, 1.2); }
    if (hasCompanion(state, 'tortoise')) { rates.ore = N.mul(rates.ore, 1.35); rates.herbs = N.mul(rates.herbs, 1.35); }
    if (state.doctrine === 'industry') { rates.ore = N.mul(rates.ore, 1.45); rates.herbs = N.mul(rates.herbs, 1.45); travel = N.mul(travel, 0.85); }
    if (state.doctrine === 'expedition') { travel = N.mul(travel, 1.35); rates.knowledge = N.mul(rates.knowledge, 0.8); }
    if (state.doctrine === 'scholarship') { rates.knowledge = N.mul(rates.knowledge, 1.5); rates.maps = N.mul(rates.maps, 1.5); rates.coins = N.mul(rates.coins, 0.85); }
    if (state.route.mode === 'supply') { travel = N.mul(travel, 0.6); rates.ore = N.mul(rates.ore, 1.85); rates.herbs = N.mul(rates.herbs, 1.85); rates.coins = N.mul(rates.coins, 1.25); }
    if (state.route.mode === 'discovery') { travel = N.mul(travel, 0.7); rates.knowledge = N.mul(rates.knowledge, 2.2); rates.maps = N.mul(rates.maps, 2.2); }
    if (researched(state, 'map-survey')) rates.ore = N.mul(rates.ore, 1 + state.ranks.cartography * 0.12);
    if (state.legacy.waystones) travel = N.mul(travel, 1 + state.ranks.cartography * 0.05);
    travel = N.mul(travel, regionCondition(state).factor);
    if (prepared(state, 'scout')) travel = N.mul(travel, 1.35);
    if (prepared(state, 'supply')) {
      rates.ore = N.mul(rates.ore, 1.5);
      if (hasCapability(state, 'regional-logistics')) rates.herbs = N.mul(rates.herbs, 1.3);
    }
    if (prepared(state, 'survey')) {
      rates.knowledge = N.mul(rates.knowledge, 1.6);
      if (hasCapability(state, 'archive-network')) rates.maps = N.mul(rates.maps, 1.4);
    }
    if (researched(state, 'frontier-compass')) { const boost = power(1.15, state.lifetime.frontier); rates.ore = N.mul(rates.ore, boost); rates.herbs = N.mul(rates.herbs, boost); }
    if (premiumOwned(state, 'compass')) travel = N.mul(travel, 1.1);
    if (premiumOwned(state, 'artisan')) { rates.ore = N.mul(rates.ore, 1.1); rates.herbs = N.mul(rates.herbs, 1.1); }
    if (premiumOwned(state, 'scholar')) { rates.knowledge = N.mul(rates.knowledge, 1.1); rates.maps = N.mul(rates.maps, 1.1); }
    if (state.luck.active === 'golden-pickaxe') rates.ore = N.mul(rates.ore, 1.5);
    if (state.luck.active === 'surveyors-lens') {
      rates.knowledge = N.mul(rates.knowledge, 1 + Math.min(0.75, 0.25 + state.ranks.cartography * 0.02));
      rates.maps = N.mul(rates.maps, 1 + Math.min(0.6, 0.2 + state.ranks.study * 0.02));
    }
    if (state.luck.active === 'marsh-lantern') rates.herbs = N.mul(rates.herbs, 1.3);
    if (state.luck.active === 'frost-compass' && prepared(state, 'scout')) travel = N.mul(travel, 1.4);
    if (state.luck.active === 'archive-quill' && prepared(state, 'survey')) { rates.knowledge = N.mul(rates.knowledge, 1.5); rates.maps = N.mul(rates.maps, 1.35); }
    if (state.luck.active === 'wayfarer-standard' && state.guild.prepared.kinds.length) { rates.coins = N.mul(rates.coins, 1.6); travel = N.mul(travel, 1.25); }
    if (state.luck.hunt !== 'balanced') { travel = N.mul(travel, 0.85); rates.coins = N.mul(rates.coins, 0.9); }
    if (!unboosted && state.luck.active === 'living-crucible' && state.luck.kit.active) {
      if (state.luck.kit.active === 'mining') rates.ore = N.mul(rates.ore, 2);
      else travel = N.mul(travel, 2);
    }
    if (!unboosted && state.caravan.surge.resource && state.caravan.surge.resource !== 'provisions') rates[state.caravan.surge.resource] = N.mul(rates[state.caravan.surge.resource], 3);
    let kitchenRatio = 1;
    if (unlocked(state, 'kitchen')) {
      const capacity = product(0.018, power(1.3, level(state, 'cooks')), material, master('kitchen'));
      const floor = continuousReserve(state, 'herbs'), relation = N.cmp(state.resources.herbs, floor);
      kitchenRatio = relation > 0 ? 1 : relation < 0 ? 0 : Math.min(1, ratio(rates.herbs, capacity));
      drain.herbs = N.mul(capacity, kitchenRatio);
      rates.provisions = N.mul(drain.herbs, hasSpecialist(state, 'naturalist') ? 1.875 : 1.5);
      if (premiumOwned(state, 'artisan')) rates.provisions = N.mul(rates.provisions, 1.1);
      if (!unboosted && state.caravan.surge.resource === 'provisions') rates.provisions = N.mul(rates.provisions, 3);
    }
    let mealRatio = 0;
    if (unlocked(state, 'kitchen') && state.meal !== 'meal-none' && state.challenges.active !== 'light-pack' && state.guild.supply !== 'save') {
      const reduction = (researched(state, 'balanced-meals') ? 0.5 : 1) * (hasSpecialist(state, 'quartermaster') ? 0.5 : 1) * (hasCompanion(state, 'tortoise') ? 0.5 : 1) * (completed(state, 'light-pack') ? 0.85 : 1) * (hasCapability(state, 'regional-logistics') ? 0.8 : 1) * (state.luck.active === 'marsh-lantern' ? 0.6 : 1);
      const scope = state.guild.grandfathered ? N.from(1) : product(material, power(1.2, equip('gear-tools') + equip('gear-boots')), power(1.12, level(state, 'boots')));
      const push = state.guild.supply === 'push';
      const need = product(state.meal === 'meal-travel' ? 0.015 : 0.02, scope, reduction, push ? 2 : 1);
      const floor = continuousReserve(state, 'provisions'), relation = N.cmp(state.resources.provisions, floor);
      mealRatio = relation > 0 ? 1 : relation < 0 ? 0 : Math.min(1, ratio(rates.provisions, need));
      drain.provisions = N.mul(need, mealRatio);
      const strength = mealRatio * (push ? 1.5 : 1);
      if (!unboosted && state.meal === 'meal-travel') travel = N.mul(travel, 1 + 0.5 * strength);
      if (!unboosted && state.meal === 'meal-study') rates.knowledge = N.mul(rates.knowledge, 1 + 0.9 * strength);
      if (!unboosted && state.meal === 'meal-mining') rates.ore = N.mul(rates.ore, 1 + 0.75 * strength);
    }
    const experience = product(power(1.08, level(state, 'mentors')), power(1.35, state.legacy.curriculum), completed(state, 'quiet-company') ? 1.2 : 1);
    C.ROOMS.forEach(room => {
      if (unlocked(state, room.id)) xp[room.id] = N.mul(experience, room.id === 'forge' ? 0.3 * (1 + level(state, 'forge')) : room.id === 'kitchen' ? 0.3 * kitchenRatio : 0.3);
    });
    return { gain: rates, drain, xp, travel, kitchenRatio, mealRatio };
  }

  function upgradeCost(state, def) {
    const invested = level(state, def.id);
    // Advanced ranks need disproportionately better supply chains, not just more clicks.
    // The opening eight purchases keep a simple geometric curve.
    let price = product(def.base, power(def.scale, invested), power(state.guild.grandfathered ? 1.045 : 1.025, Math.pow(Math.max(0, invested - 8), 2) / (state.guild.grandfathered ? 1 : 1 + Math.max(0, invested - 8) / 40)));
    if (def.equipment) price = product(price, power(0.95, level(state, 'forge')), researched(state, 'efficient-smelting') ? 0.65 : 1, hasSpecialist(state, 'quartermaster') ? 0.8 : 1, completed(state, 'old-tools') ? 0.85 : 1);
    if (def.equipment && state.luck.active === 'starsteel-anvil') price = N.mul(price, 0.8);
    const costs = { [def.resource]: price };
    if (def.id === 'gear-instruments') costs.herbs = N.mul(price, 0.4);
    return costs;
  }
  function upgradeOpen(state, def) { return unlocked(state, def.room) && (!def.at || state.lifetime.highestRoute + 1 >= def.at); }
  function recipeTrade(state, id) {
    const scale = power(3.5, tier(state) - 1);
    if (id === 'alloy') return { costs: { herbs: N.mul(scale, 50) }, gains: { ore: N.mul(scale, state.luck.active === 'starsteel-anvil' ? 60 : 20) } };
    if (id === 'survey') return { costs: { maps: N.mul(scale, 10) }, gains: { knowledge: N.mul(scale, 25), coins: N.mul(scale, 500) } };
    return { costs: {}, gains: {} };
  }
  function purchase(state, id) {
    const def = find(C.UPGRADES, id);
    if (!def || !upgradeOpen(state, def)) return { ok: false, message: 'That upgrade is not available yet.' };
    const costs = upgradeCost(state, def);
    if (!affordable(state, costs)) return { ok: false, message: 'More resources are needed for ' + def.name.toLowerCase() + '.' };
    spend(state, costs); state.upgrades[id] += 1;
    if (def.equipment) { state.mastery.forge = N.add(state.mastery.forge, 30); state.ranks.forge = masteryRank(state.mastery.forge); }
    return { ok: true, message: def.name + ' improved.' };
  }
  function automationAvailable(state, id) { return id === 'operations' && state.lifetime.refits > 0 || researched(state, find(C.AUTOMATIONS, id).research); }

  function planningOpen(state) { return state.lifetime.refits > 0 || researched(state, 'auto-work'); }
  function taskDetails(state, action) {
    if (!validPlanAction(action)) return null;
    if (action.type === 'buy') { const def = find(C.UPGRADES, action.id); return { name: def.name, costs: upgradeCost(state, def), open: upgradeOpen(state, def), done: false }; }
    if (action.type === 'research') { const def = find(C.RESEARCH, action.id); return { name: def.name, costs: { knowledge: N.from(def.cost) }, open: unlocked(state, 'study') && state.lifetime.highestRoute + 1 >= def.at, done: researched(state, def.id) }; }
    const def = find(C.PROJECTS, action.id);
    return { name: projectName(action.id), costs: projectCosts(state, action.id), open: projectOpen(state, action.id), done: def ? unlocked(state, def.room) : !!state.guild.chapterProject.choice };
  }
  function cleanPlan(state) {
    const plan = state.guild.plan;
    if (plan.goal && taskDetails(state, plan.goal).done) plan.goal = null;
    while (plan.queue.length && taskDetails(state, plan.queue[0]).done) plan.queue.shift();
  }
  function reservedCosts(state, action, costs, routine) {
    const plan = state.guild.plan;
    const result = clone(costs);
    if (routine && result.coins) result.coins = N.mul(result.coins, 1.25);
    // The explicit objective has precedence over the queue. Lower-priority
    // purchases protect it; it must never wait for the queue's full price.
    const tasks = plan.goal && canonical(plan.goal) === canonical(action) ? [] : [plan.goal, plan.queue[0]];
    const protectedTasks = tasks.filter((task, i, list) => task && canonical(task) !== canonical(action) && list.findIndex(other => other && canonical(other) === canonical(task)) === i);
    for (const task of protectedTasks) {
      const detail = taskDetails(state, task);
      if (detail && !detail.done) Object.keys(detail.costs).forEach(id => { if (result[id]) result[id] = N.add(result[id], detail.costs[id]); });
    }
    Object.keys(result).forEach(id => { result[id] = N.add(result[id], plan.reserves[id] || N.zero()); });
    return result;
  }
  function planningCandidates(state) {
    if (!planningOpen(state)) return [];
    const plan = state.guild.plan, result = [];
    const add = (action, routine) => {
      const detail = taskDetails(state, action);
      if (detail && detail.open && !detail.done) result.push({ action, name: detail.name, costs: detail.costs, routine, required: reservedCosts(state, action, detail.costs, routine) });
    };
    if (plan.goal) add(plan.goal, false);
    if (hasCapability(state, 'purchase-queue') && plan.queue.length) add(plan.queue[0], false);
    if (state.automations.operations && automationAvailable(state, 'operations')) {
      const order = C.UPGRADES.filter(x => !x.equipment && x.id !== 'preparation' && upgradeOpen(state, x));
      order.sort((a, b) => {
        const rank = def => plan.priorities.operations === 'travel' ? def.id === 'boots' ? -1000 : 0 : plan.priorities.operations === 'production' ? def.id === 'boots' ? 1000 : 0 : 0;
        return rank(a) - rank(b) || level(state, a.id) - level(state, b.id);
      });
      order.forEach(def => add({ type: 'buy', id: def.id }, true));
    }
    if (state.automations.equipment && automationAvailable(state, 'equipment')) {
      const priority = plan.priorities.equipment === 'balanced' ? state.doctrine === 'industry' ? 'tools' : 'boots' : plan.priorities.equipment;
      const first = { tools: 'gear-tools', boots: 'gear-boots', research: 'gear-instruments' }[priority];
      const order = [first, ...['gear-boots', 'gear-tools', 'gear-instruments'].filter(id => id !== first)];
      order.forEach(id => {
        const oldLength = result.length;
        add({ type: 'buy', id }, false);
        if (result.length > oldLength && id !== 'gear-boots' && researched(state, 'smart-reserve')) {
          result[result.length - 1].preserveBoots = true;
          result[result.length - 1].required.ore = N.add(result[result.length - 1].required.ore, upgradeCost(state, find(C.UPGRADES, 'gear-boots')).ore);
        }
      });
    }
    if (plan.kit !== 'off' && hasCapability(state, 'kit-plan') && state.luck.active === 'living-crucible' && !state.luck.kit.active && (!state.luck.kit.prepared || state.luck.kit.prepared === plan.kit) && unlocked(state, 'study') && unlocked(state, 'kitchen')) {
      const costs = state.luck.kit.prepared ? {} : kitCosts(state), action = { type: 'kit-plan-run', id: plan.kit };
      result.push({ action, name: 'Renew ' + plan.kit + ' kit', costs, required: reservedCosts(state, action, costs, false) });
    }
    if (plan.preparation !== 'off' && hasCapability(state, 'dispatch-preparation') && unlocked(state, 'cartography') && !state.guild.prepared.kinds.length) {
      const action = { type: 'prepare-route', id: plan.preparation }, costs = preparationCosts(state, action.id);
      result.push({ action, name: find(C.PREPARATIONS, action.id).name, costs, required: reservedCosts(state, action, costs, false) });
    }
    return result;
  }
  function auditPurchase(state, name, costs, at) {
    const price = Object.keys(costs).map(id => N.format(costs[id]) + ' ' + id).join(' + ');
    state.guild.audit.push({ at, message: (name + (price ? ' · spent ' + price : '')).slice(0, 250) });
    if (state.guild.audit.length > 20) state.guild.audit.shift();
  }
  function planPurchases(state, at) {
    cleanPlan(state);
    const candidates = planningCandidates(state);
    const acted = new Set();
    for (const candidate of candidates) {
      const key = canonical(candidate.action);
      if (acted.has(key)) continue;
      // Recompute protection after each purchase: earlier spending changes costs,
      // and fulfilled goals must stop reserving resources in this same minute.
      const detail = taskDetails(state, candidate.action);
      let current = null;
      if (detail && detail.open && !detail.done) current = { name: detail.name, costs: detail.costs, required: reservedCosts(state, candidate.action, detail.costs, candidate.routine) };
      else if (candidate.action.type === 'prepare-route' && !state.guild.prepared.kinds.length) { const costs = preparationCosts(state, candidate.action.id); current = { name: candidate.name, costs, required: reservedCosts(state, candidate.action, costs, false) }; }
      else if (candidate.action.type === 'kit-plan-run' && !state.luck.kit.active) { const costs = state.luck.kit.prepared ? {} : kitCosts(state); current = { name: candidate.name, costs, required: reservedCosts(state, candidate.action, costs, false) }; }
      if (current && candidate.preserveBoots) current.required.ore = N.add(current.required.ore, upgradeCost(state, find(C.UPGRADES, 'gear-boots')).ore);
      if (!current || !affordable(state, current.required)) continue;
      let result;
      if (candidate.action.type === 'kit-plan-run') {
        if (!state.luck.kit.prepared) result = act(state, { type: 'kit-prepare', id: candidate.action.id });
        if (state.luck.kit.prepared) result = act(state, { type: 'kit-use' });
      } else result = act(state, candidate.action);
      if (result && result.ok) { acted.add(key); auditPurchase(state, current.name, current.costs, at); }
    }
  }
  function nextPlanningTime(state, rates, allowance) {
    const candidates = planningCandidates(state);
    let earliest = Infinity;
    candidates.forEach(candidate => {
      let wait = 0;
      Object.keys(candidate.required).forEach(id => {
        const stock = allowance ? N.add(state.resources[id], allowance[id] || N.zero()) : state.resources[id];
        if (N.cmp(stock, candidate.required[id]) >= 0) return;
        const net = N.sub(rates.gain[id], rates.drain[id]);
        wait = Math.max(wait, net.m ? ratio(N.sub(candidate.required[id], stock), net) : Infinity);
      });
      earliest = Math.min(earliest, wait);
    });
    if (!Number.isFinite(earliest)) return Infinity;
    const first = 60 - state.planner;
    return first + Math.max(0, Math.ceil((earliest - first - EPS) / 60)) * 60;
  }

  function settleRoute(state) {
    const index = state.route.index;
    const route = getRoute(index, state);
    state.route.progress = N.zero();
    state.guild.prepared = { index: -1, kinds: [] };
    state.run.completed = Math.max(state.run.completed, index);
    const newRoute = index > state.lifetime.highestRoute;
    if (newRoute) {
      state.lifetime.highestRoute = index;
      state.lifetime.frontier = Math.max(0, index - 17);
      pushEvent(state, route.name + ' completed.');
      C.ROOMS.filter(room => room.id !== 'forge' && (state.guild.grandfathered || !C.PROJECTS.some(project => project.room === room.id)) && room.at <= index + 1 && !unlocked(state, room.id)).forEach(room => { state.rooms.push(room.id); pushEvent(state, room.name + ' discovered: ' + room.description); });
      if (index < 18 && index % 3 === 2 && !state.collections.includes(route.realm)) { state.collections.push(route.realm); pushEvent(state, route.realmName + ' landmark added to the guild collection.'); }
      if (unlocked(state, 'study')) state.resources.knowledge = N.add(state.resources.knowledge, product(5, power(3, route.tier - 1), researched(state, 'field-notes') ? 2 : 1));
    }
    const challenge = find(C.CHALLENGES, state.challenges.active);
    if (challenge && index >= challenge.target) {
      if (!completed(state, challenge.id)) { grantPremiumGift(state, 'challenge-' + challenge.id); state.challenges.completed.push(challenge.id); }
      state.challenges.active = null;
      pushEvent(state, challenge.name + ' challenge completed. Its guild reward is permanent.');
    }
    // The opening has one continuous trail. Route selection becomes a choice in the Map Room.
    if (!unlocked(state, 'cartography') || state.automations.routes && automationAvailable(state, 'routes')) state.route.index = index + 1;
  }

  function advance(state, seconds) {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0) return { seconds: 0, gained: {}, events: [] };
    const actual = Math.min(seconds, (MAX_TIME - state.lastUpdate) / 1000);
    if (actual <= 0) return { seconds: 0, gained: {}, events: [] };
    const before = clone(state.resources);
    const oldEvents = state.events.slice();
    const prior = { rooms: state.rooms.slice(), research: state.research.slice(), highestRoute: state.lifetime.highestRoute, seq: state.luck.ledger.seq, at: state.lastUpdate };
    let remaining = actual;
    let elapsed = 0;
    let iterations = 0;
    while (remaining > EPS) {
      cleanPlan(state);
      const rates = getRates(state);
      const luckEligible = { mine: unlocked(state, 'mine'), study: unlocked(state, 'study'), forge: caravanUnlocked(state) };
      const batch = rates.kitchenRatio === 1 && (state.meal === 'meal-none' || rates.mealRatio === 1) && C.RESOURCES.every(resource => N.cmp(rates.gain[resource.id], rates.drain[resource.id]) >= 0);
      const unboosted = batch ? getRates(state, true) : null;
      const target = getRoute(state.route.index, state).distance;
      const reserveEvents = {};
      let dt = remaining;
      dt = Math.min(dt, luckEventTime(state, batch));
      const dispatch = !unlocked(state, 'cartography') || state.automations.routes && automationAvailable(state, 'routes');
      const repeat = state.route.index <= state.run.completed && !dispatch && !state.guild.prepared.kinds.length;
      const routeTime = repeat ? Infinity : ratio(N.sub(target, state.route.progress), rates.travel);
      if (!repeat) dt = Math.min(dt, Math.max(EPS, routeTime));
      // A conservative bound includes the largest possible find before the
      // first check, then maximum subsequent find frequency. No automated
      // purchase can become affordable inside a batched interval.
      const planningRates = batch ? { gain: {}, drain: rates.drain } : rates;
      const allowance = batch ? {} : null;
      if (batch) C.RESOURCES.forEach(resource => {
        const id = resource.id, common = luckEligible.mine && ORDINARY.includes(id) ? 75 : 0, duplicate = luckEligible.study && id === 'coins' ? 300 : 0;
        allowance[id] = N.mul(unboosted.gain[id], common + duplicate);
        planningRates.gain[id] = N.add(rates.gain[id], N.mul(unboosted.gain[id], common / 480 + duplicate));
      });
      const planningTime = nextPlanningTime(state, planningRates, allowance);
      if (Number.isFinite(planningTime)) dt = Math.min(dt, Math.max(EPS, planningTime));
      C.ROOMS.forEach(room => {
        if (rates.xp[room.id].m) dt = Math.min(dt, Math.max(EPS, ratio(N.sub(nextMastery(state.ranks[room.id]), state.mastery[room.id]), rates.xp[room.id])));
      });
      C.RESOURCES.forEach(resource => {
        const id = resource.id;
        const floor = ['herbs', 'provisions'].includes(id) ? continuousReserve(state, id) : N.zero();
        const direction = N.cmp(rates.gain[id], rates.drain[id]), relation = N.cmp(state.resources[id], floor);
        if (relation > 0 && direction < 0 || relation < 0 && direction > 0) {
          const distance = relation > 0 ? N.sub(state.resources[id], floor) : N.sub(floor, state.resources[id]);
          const rate = direction > 0 ? N.sub(rates.gain[id], rates.drain[id]) : N.sub(rates.drain[id], rates.gain[id]);
          const time = Math.max(EPS, ratio(distance, rate));
          reserveEvents[id] = { time, floor }; dt = Math.min(dt, time);
        }
      });
      const distance = N.mul(rates.travel, dt);
      state.route.progress = N.add(state.route.progress, distance);
      state.run.work = N.add(state.run.work, distance); state.chapter.work = N.add(state.chapter.work, distance); state.lifetime.work = N.add(state.lifetime.work, distance);
      state.run.elapsed += dt;
      state.lifetime.coins = N.add(state.lifetime.coins, N.mul(rates.gain.coins, dt));
      C.RESOURCES.forEach(resource => {
        const id = resource.id;
        const net = N.cmp(rates.gain[id], rates.drain[id]);
        state.resources[id] = net >= 0 ? N.add(state.resources[id], N.mul(N.sub(rates.gain[id], rates.drain[id]), dt)) : N.sub(state.resources[id], N.mul(N.sub(rates.drain[id], rates.gain[id]), dt));
        if (reserveEvents[id] && reserveEvents[id].time <= dt + EPS) state.resources[id] = N.from(reserveEvents[id].floor);
        if (state.resources[id].m && state.resources[id].e < -9 && net < 0) state.resources[id] = N.zero();
      });
      if (batch) batchLuck(state, dt, state.lastUpdate + elapsed * 1000, unboosted, luckEligible);
      C.ROOMS.forEach(room => { state.mastery[room.id] = N.add(state.mastery[room.id], N.mul(rates.xp[room.id], dt)); state.ranks[room.id] = masteryRank(state.mastery[room.id]); });
      state.planner += dt;
      if (repeat && N.cmp(state.route.progress, target) >= 0) {
        const laps = N.div(state.route.progress, target);
        // Repeated known routes have no completion grant. Batch their animation phase;
        // above mantissa precision the fractional lap is intentionally represented as zero.
        state.route.progress = laps.e > 13 ? N.zero() : N.mul(target, N.toNumber(laps) % 1);
      } else if (N.cmp(state.route.progress, target) >= 0 || routeTime <= dt + EPS) settleRoute(state);
      if (!batch) tickLuck(state, dt, state.lastUpdate + (elapsed + dt) * 1000, luckEligible);
      if (state.planner >= 60 - EPS) { state.planner %= 60; if (state.planner > 60 - EPS) state.planner = 0; }
      if (planningTime <= dt + EPS) planPurchases(state, state.lastUpdate + (elapsed + dt) * 1000);
      // Summing processed intervals avoids cancellation when a huge absence contains
      // very short events (for example an extreme imported travel multiplier).
      elapsed += dt;
      remaining = Math.max(0, actual - elapsed);
      iterations += 1;
      // A pathological imported state cannot monopolize the UI. Leave the timestamp
      // at the exact processed point so the caller can continue without losing time.
      if (iterations >= 2048) break;
    }
    const processed = remaining === 0 ? actual : elapsed;
    advancePremium(state, processed);
    state.lastUpdate += processed * 1000;
    const events = state.events.filter(event => !oldEvents.includes(event));
    const goal = getGoal(state);
    const summary = {
      completed: events.filter(event => /completed|researched|built|discovered/.test(event)).slice(-6),
      changes: state.rooms.filter(id => !prior.rooms.includes(id)).map(id => find(C.ROOMS, id).name + ' opened').concat(state.research.filter(id => !prior.research.includes(id)).map(id => find(C.RESEARCH, id).name + ' researched')).slice(-6),
      routesDiscovered: state.lifetime.highestRoute - prior.highestRoute,
      discoveries: state.luck.ledger.seq - prior.seq,
      blockedReason: planBlockedReason(state),
      nextChoices: [goal.title].concat(unlocked(state, 'cartography') && !state.guild.chapterProject.choice ? ['Choose a regional supply, atlas, or waystation project'] : []).slice(0, 2),
      spending: state.guild.audit.filter(entry => entry.at > prior.at).slice(-6)
    };
    return { seconds: processed, pendingSeconds: Math.max(0, remaining), gained: Object.fromEntries(C.RESOURCES.map(resource => [resource.id, N.sub(state.resources[resource.id], before[resource.id])])), events, summary };
  }
  function advanceTo(state, now) { return typeof now === 'number' && Number.isFinite(now) && now >= 0 && now <= MAX_TIME ? advance(state, Math.max(0, (now - state.lastUpdate) / 1000)) : { seconds: 0, gained: {}, events: [] }; }

  function getRefitPreview(state) {
    const starshards = premiumGift(state, 'first-refit');
    const requiredWork = product(12000, power(2, state.chapter.refits));
    let reward = N.floor(product(2, power(N.div(state.run.work, 12000), 0.32), researched(state, 'field-notes') ? 1.25 : 1));
    if (state.run.completed < 2 || state.challenges.active) reward = N.zero();
    const requirements = [{ label: 'Build the Forge and reinforce both mining tools and expedition boots', met: unlocked(state, 'forge') && level(state, 'gear-tools') >= 1 && level(state, 'gear-boots') >= 1 }, { label: 'Complete the Abandoned Quarry in this expedition', met: state.run.completed >= 2 }, { label: 'Earn ' + N.format(requiredWork) + ' expedition distance this run', met: N.cmp(state.run.work, requiredWork) >= 0 }, { label: 'Finish or leave the current challenge', met: state.challenges.active === null }];
    return { available: requirements.every(x => x.met), reward, rewardText: N.format(reward) + ' field notes', premiumGift: starshards, requirements, ...resetAdvice(state, 'notes', reward),
      keeps: ['Discovered rooms and routes', 'Profession operations and mastery', 'Equipment reinforcement and recipes', 'Ore, herbs, provisions, knowledge, and maps', 'Crew ownership and assignments', 'Research, notes, collections, and legacy', 'Starshards, lasting charms, banners, and rare-drop timing', 'Relics, research, study progress, hunts, kits, and discovery timers', 'Caravan offers, reserved ad quotes, verified receipts, and remaining surge time'],
      resets: ['Coins', 'Current route and route completion this run', 'Field preparation', 'Active meal returns to Save provisions', 'Unstarted caravan selections are repriced; reserved ad rewards keep their exact quote'],
      gains: [N.format(reward) + ' field notes for permanent Refit improvements', state.lifetime.refits ? 'A new expedition using your retained guild' : 'Automatic operation purchases become available'].concat(starshards ? [starshards + ' earned Starshards · first Refit gift, once only'] : []) };
  }
  function getCharterPreview(state) {
    const starshards = premiumGift(state, 'first-charter');
    const targetRoute = charterTarget(state);
    const requiredWork = getRoute(targetRoute, state).distance;
    let reward = N.floor(product(2, power(N.div(state.chapter.work, requiredWork), 0.25)));
    const requirements = [{ label: 'Complete ' + getRoute(targetRoute, state).name + ' (route ' + (targetRoute + 1) + ') in this expedition', met: state.run.completed >= targetRoute }, state.guild.grandfathered ? { label: 'Complete two Refits in this charter (preserved legacy rules)', met: state.chapter.refits >= 2 } : { label: 'Complete one regional supply, atlas, or waystation project', met: state.guild.chapterProject.choice !== null }, { label: 'Earn ' + N.format(requiredWork) + ' distance in this charter', met: N.cmp(state.chapter.work, requiredWork) >= 0 }, { label: 'Finish or leave the current challenge', met: state.challenges.active === null }];
    if (!requirements.every(x => x.met)) reward = N.zero();
    return { available: requirements.every(x => x.met), reward, rewardText: N.format(reward) + ' guild crests', premiumGift: starshards, requirements, ...resetAdvice(state, 'crests', reward),
      keeps: ['All room discoveries and equipment patterns', 'Lifetime profession mastery', 'Research and automation licenses', 'Crew ownership, recipes, and unlocked route choices', 'Collections and completed challenges', 'Guild crests and legacy upgrades', 'Starshards, lasting charms, banners, and rare-drop timing', 'Relics, research, study progress, hunts, kits, and discovery timers', 'Caravan offers, reserved ad quotes, verified receipts, and remaining surge time'],
      resets: ['All ordinary resource stockpiles and field notes', 'Operations, field preparation, and equipment reinforcement', 'Refit improvement levels', 'Route progress and this charter’s expedition distance', 'Unstarted caravan selections are repriced; reserved ad rewards keep their exact quote'],
      gains: [N.format(reward) + ' guild crests for permanent legacy capabilities', 'An established guild opening with 100 coins', 'Discovered professions are immediately available'].concat(starshards ? [starshards + ' earned Starshards · first Charter gift, once only'] : []) };
  }
  function charterTarget(state) { return 8 + state.lifetime.charters * (state.guild.grandfathered ? 1 : 3); }
  function resetAdvice(state, resource, reward) {
    const budget = N.add(state.resources[resource], reward);
    const purchases = C.CAPABILITIES.filter(def => def.resource === resource && !hasCapability(state, def.id) && (resource === 'notes' ? state.lifetime.refits + 1 : state.lifetime.charters + 1) >= def.at && N.cmp(budget, def.cost) >= 0).map(def => ({ name: def.name, costText: def.cost + ' ' + resource, effectText: def.description }));
    const retainedDistance = N.from(state.run.work);
    const rate = getRates(state).travel;
    return { purchases, recovery: resource === 'notes' ? { seconds: ratio(retainedDistance, rate), text: 'Estimate at the current travel rate, before new purchases and route preparations. Returning to this reach may be faster as the guild improves.' } : { seconds: null, text: 'Operations and equipment restart. Retained mastery, research, playbooks, and your crest purchase determine recovery; no fixed-time promise.' }, planText: 'Reserves, priorities, saved objective, queue, automation settings, capabilities, and saved loadouts remain. Meals and paid route preparations restart; queued materials are earned again.' };
  }
  function resetRun(state) {
    state.resources.coins = N.zero(); state.upgrades.preparation = 0;
    state.route = { index: 0, progress: N.zero(), mode: 'frontier' };
    state.run = { id: state.run.id + 1, completed: -1, work: N.zero(), elapsed: 0 };
    state.meal = 'meal-none'; state.planner = 0;
    state.guild.prepared = { index: -1, kinds: [] };
    if (state.caravan.offer && !state.caravan.offer.locked) state.caravan.offer.quote = null;
  }
  function projectCosts(state, id) {
    const def = find(C.PROJECTS, id);
    if (def) return Object.fromEntries(Object.entries(def.costs).map(([resource, amount]) => [resource, N.from(amount)]));
    const scale = power(3.5, Math.max(2, getRoute(charterTarget(state), state).tier - 1));
    const amount = (resource, value) => product(scale, value, productionInvestment(state, resource));
    if (id === 'chapter-supply') return { provisions: amount('provisions', 3000), herbs: amount('herbs', 1500), maps: amount('maps', 400) };
    if (id === 'chapter-survey') return { knowledge: amount('knowledge', 2500), maps: amount('maps', 900) };
    if (id === 'chapter-industry') return { ore: amount('ore', 5000), provisions: amount('provisions', 1000), maps: amount('maps', 200) };
    return null;
  }
  function projectOpen(state, id) {
    const def = find(C.PROJECTS, id);
    return def ? unlocked(state, 'forge') && !unlocked(state, def.room) && state.lifetime.highestRoute + 1 >= def.at && unlocked(state, def.room === 'study' ? 'kitchen' : def.room === 'hall' ? 'study' : 'hall') : /^chapter-(supply|survey|industry)$/.test(id) && unlocked(state, 'cartography') && state.guild.chapterProject.choice === null;
  }
  function projectName(id) { const def = find(C.PROJECTS, id); return def ? def.name : { 'chapter-supply': 'Establish regional supply camps', 'chapter-survey': 'Complete a regional atlas', 'chapter-industry': 'Build a regional waystation' }[id] || id; }
  function performAction(state, action) {
    if (!object(action) || typeof action.type !== 'string') return { ok: false, message: 'Choose an available guild action.' };
    if (action.type === 'introduction-seen') {
      const earned = presentationSystems(state).map(system => system.id);
      if (!Array.isArray(action.ids) || !action.ids.length || action.ids.length > C.PRESENTATION_SYSTEMS.length || new Set(action.ids).size !== action.ids.length || action.ids.some(id => !earned.includes(id))) return { ok: false, message: 'Only introductions for earned systems can be acknowledged.' };
      if (!state.introductions) state.introductions = { seen: earned.slice() };
      state.introductions.seen = Array.from(new Set(state.introductions.seen.concat(action.ids)));
      return { ok: true, message: 'Introduction saved.' };
    }
    if (action.type.startsWith('plan-')) {
      if (!planningOpen(state)) return { ok: false, message: 'Earn your first Refit or Workshop ledgers before authoring a plan.' };
      const plan = state.guild.plan;
      if (action.type === 'plan-reserve') {
        if (!ORDINARY.includes(action.id) || typeof action.amount !== 'string' || action.amount.length > 80) return { ok: false, message: 'Choose an ordinary resource and a nonnegative reserve.' };
        try { plan.reserves[action.id] = N.from(action.amount.trim()); } catch (_) { return { ok: false, message: 'Enter a valid nonnegative amount; scientific notation is accepted.' }; }
      } else if (action.type === 'plan-goal') {
        if (action.action !== null && !validPlanAction(action.action)) return { ok: false, message: 'Choose an upgrade, research item, or project to save for.' };
        plan.goal = clone(action.action);
      } else if (action.type === 'plan-priority') {
        const allowed = { operations: ['balanced', 'travel', 'production'], equipment: ['balanced', 'tools', 'boots', 'research'] };
        if (!allowed[action.group] || !allowed[action.group].includes(action.id)) return { ok: false, message: 'Choose an available purchase priority.' };
        plan.priorities[action.group] = action.id;
      } else if (action.type === 'plan-queue') {
        if (!hasCapability(state, 'purchase-queue') || plan.queue.length >= 6 || !validPlanAction(action.action)) return { ok: false, message: 'Standing purchase plan allows up to six upgrades, research items, or projects.' };
        plan.queue.push(clone(action.action));
      } else if (action.type === 'plan-remove') {
        if (!Number.isSafeInteger(action.index) || action.index < 0 || action.index >= plan.queue.length) return { ok: false, message: 'Choose an existing queue entry.' };
        plan.queue.splice(action.index, 1);
      } else if (action.type === 'plan-kit') {
        if (!hasCapability(state, 'kit-plan') || !['off', 'travel', 'mining'].includes(action.id)) return { ok: false, message: 'Learn Expedition outfitter before automating a kit.' };
        plan.kit = action.id;
      } else if (action.type === 'plan-preparation') {
        if (!hasCapability(state, 'dispatch-preparation') || !['off', 'scout', 'supply', 'survey'].includes(action.id)) return { ok: false, message: 'Learn Survey dispatch to plan automatic route preparation.' };
        plan.preparation = action.id;
      } else return { ok: false, message: 'Unknown planning action.' };
      cleanPlan(state);
      return { ok: true, message: 'Plan saved. Automatic purchases, herb cooking, and meals honor reserves and the saved objective. Manual purchases may use reserved resources.' };
    }
    if (action.type.startsWith('loadout-')) {
      if (!hasCapability(state, 'loadouts') || ![0, 1, 2].includes(action.id)) return { ok: false, message: 'Learn Guild playbooks to use three saved plans.' };
      if (action.type === 'loadout-save') {
        if (typeof action.name !== 'string' || !action.name.trim() || action.name.length > 40) return { ok: false, message: 'Name the playbook with 1–40 characters.' };
        const entry = { id: action.id, name: action.name.trim(), plan: clone(state.guild.plan), specialists: state.crew.specialists.slice(), companion: state.crew.companion, doctrine: state.doctrine, meal: state.meal, relic: state.luck.active, supply: state.guild.supply, mode: state.route.mode };
        state.guild.loadouts = state.guild.loadouts.filter(x => x.id !== action.id).concat(entry).sort((a, b) => a.id - b.id);
        return { ok: true, message: 'Playbook saved. No route progress or reward is stored in a playbook.' };
      }
      const entry = state.guild.loadouts.find(x => x.id === action.id);
      if (!entry) return { ok: false, message: 'Save that playbook first.' };
      if (action.type === 'loadout-delete') state.guild.loadouts = state.guild.loadouts.filter(x => x.id !== action.id);
      else if (action.type === 'loadout-use') {
        if (state.challenges.active) return { ok: false, message: 'Finish or leave the current challenge before applying a full playbook.' };
        state.guild.plan = clone(entry.plan); state.guild.supply = entry.supply;
        state.crew.specialists = entry.specialists.slice(); state.crew.companion = entry.companion;
        state.doctrine = entry.doctrine; state.meal = entry.meal; state.luck.active = entry.relic; state.route.mode = entry.mode;
        cleanPlan(state);
      } else return { ok: false, message: 'Unknown playbook action.' };
      return { ok: true, message: action.type === 'loadout-delete' ? 'Playbook deleted.' : entry.name + ' applied. Current route, stockpiles, and preparation remain.' };
    }
    if (action.type === 'capability') {
      const def = find(C.CAPABILITIES, action.id);
      if (!def || hasCapability(state, def.id) || (def.resource === 'notes' ? state.lifetime.refits < 1 : state.lifetime.charters < def.at)) return { ok: false, message: 'Earn the required Refit or Charter before learning this capability.' };
      const costs = { [def.resource]: N.from(def.cost) };
      if (!affordable(state, costs)) return { ok: false, message: 'More ' + def.resource + ' are needed.' };
      spend(state, costs); state.guild.capabilities.push(def.id);
      return { ok: true, message: def.name + ' learned permanently. ' + def.description };
    }
    if (action.type === 'supply-plan') {
      if (!unlocked(state, 'kitchen') || !['save', 'steady', 'push'].includes(action.id)) return { ok: false, message: 'Build the Kitchen to choose a supply commitment.' };
      state.guild.supply = action.id;
      return { ok: true, message: action.id === 'save' ? 'Provisions saved. Basic expeditions continue.' : action.id === 'push' ? 'Meals consume twice the provisions for 50% stronger meal bonuses.' : 'Standard meals support the expedition while supplied.' };
    }
    if (action.type === 'prepare-route') {
      const costs = preparationCosts(state, action.id);
      if (!costs || !unlocked(state, 'cartography') || state.guild.prepared.kinds.length) return { ok: false, message: 'Choose one preparation for the current route; it lasts until completion, route change, or reset.' };
      if (!affordable(state, costs)) return { ok: false, message: 'More supplies are needed for this preparation.' };
      spend(state, costs); state.guild.prepared = { index: state.route.index, kinds: [action.id] };
      return { ok: true, message: find(C.PREPARATIONS, action.id).description + ' Consumed on route completion, change, or reset; no refund.' };
    }
    if (action.type === 'project') {
      const costs = projectCosts(state, action.id);
      if (!costs || !projectOpen(state, action.id)) return { ok: false, message: 'Meet this project’s route and profession requirements first.' };
      if (!affordable(state, costs)) return { ok: false, message: 'Gather the materials for ' + projectName(action.id).toLowerCase() + '.' };
      spend(state, costs);
      const def = find(C.PROJECTS, action.id);
      if (def) { state.guild.projects.push(def.id); state.rooms.push(def.room); }
      else state.guild.chapterProject.choice = action.id.slice(8);
      pushEvent(state, projectName(action.id) + ' completed.');
      return { ok: true, message: projectName(action.id) + ' completed.' };
    }
    if (action.type === 'discovery-seen') {
      if (!bounded(action.seq, state.luck.ledger.seq, true)) return { ok: false, message: 'Choose an existing discovery.' };
      state.luck.ledger.seen = Math.max(state.luck.ledger.seen, action.seq);
      return { ok: true, message: 'Discoveries recorded.' };
    }
    if (action.type === 'relic-equip') {
      if (action.id !== null && !state.luck.owned.includes(action.id)) return { ok: false, message: 'Discover that relic first.' };
      state.luck.active = action.id;
      return { ok: true, message: action.id ? find(C.RELICS, action.id).name + ' equipped in your one relic slot.' : 'Relic unequipped.' };
    }
    if (action.type === 'relic-hunt') {
      if (!unlocked(state, 'cartography') || action.id !== 'balanced' && !find(eligibleRelics(state), action.id)) return { ok: false, message: 'Discover the Map Room and the relic’s chapter before planning this hunt.' };
      state.luck.hunt = action.id;
      return { ok: true, message: 'Future searches target this relic. The current search keeps its saved destination; no timer is rerolled.' };
    }
    if (action.type === 'luck-research') {
      const def = find(C.LUCK_RESEARCH, action.id);
      if (!def || !unlocked(state, def.room || 'study') || state.luck.research.includes(def.id)) return { ok: false, message: 'That discovery research is unavailable.' };
      const costs = { knowledge: N.from(def.cost) };
      if (!affordable(state, costs)) return { ok: false, message: 'More knowledge is needed.' };
      spend(state, costs); state.luck.research.push(def.id);
      return { ok: true, message: def.description };
    }
    if (action.type === 'kit-prepare') {
      if (!find(C.KITS, action.id) || state.luck.active !== 'living-crucible' || !unlocked(state, 'kitchen') || !unlocked(state, 'study') || state.luck.kit.prepared) return { ok: false, message: 'Equip the Living Crucible and leave the prepared-kit slot empty.' };
      const costs = kitCosts(state);
      if (!affordable(state, costs)) return { ok: false, message: 'This kit needs ore, provisions, and knowledge.' };
      spend(state, costs); state.luck.kit.prepared = action.id;
      return { ok: true, message: find(C.KITS, action.id).name + ' prepared. Choose when to begin its burst.' };
    }
    if (action.type === 'kit-use') {
      if (!state.luck.kit.prepared || state.luck.kit.active || state.luck.active !== 'living-crucible') return { ok: false, message: 'Equip the Crucible and prepare a kit; only one burst can run at once.' };
      state.luck.kit.active = state.luck.kit.prepared; state.luck.kit.prepared = null; state.luck.kit.remainingSeconds = 1800;
      return { ok: true, message: 'Kit active for 30 expedition minutes. Unequipping the Crucible suppresses its effect but does not pause time.' };
    }
    if (action.type === 'caravan-select') return selectCaravan(state, action);
    if (action.type === 'caravan-dismiss') {
      if (!state.caravan.offer || state.caravan.pendingQuote) return { ok: false, message: 'A reserved verified reward cannot be dismissed.' };
      state.caravan.offer = null; state.caravan.remainingMs = Math.floor((3600 + random(state.caravan) * 1800) * 1000);
      return { ok: true, message: 'Caravan dismissed. Another may arrive after 60–90 expedition minutes.' };
    }
    if (action.type === 'premium-buy') {
      const def = find(C.PREMIUM_ITEMS, action.id);
      if (!def || !unlocked(state, 'forge')) return { ok: false, message: 'Build the Forge to visit the Starshard shop.' };
      if (premiumOwned(state, def.id)) return { ok: false, message: 'You already own that lasting item.' };
      const costs = { starshards: N.from(def.cost) };
      if (!affordable(state, costs)) return { ok: false, message: 'This item needs ' + def.cost + ' earned Starshards.' };
      spend(state, costs); state.resources.starshards = N.floor(state.resources.starshards);
      state.premium.owned.push(def.id);
      return { ok: true, message: def.name + ' is yours permanently.' };
    }
    if (action.type === 'premium-equip') {
      const def = find(C.PREMIUM_ITEMS, action.id);
      if (!unlocked(state, 'forge') || action.id !== null && (!def || def.kind !== 'banner' || !premiumOwned(state, def.id))) return { ok: false, message: 'Choose a banner you own.' };
      state.premium.equipped = action.id;
      return { ok: true, message: def ? def.name + ' selected.' : 'Default guild banner selected.' };
    }
    if (action.type === 'buy') return purchase(state, action.id);
    if (action.type === 'build-room' && action.id === 'forge') {
      if (!unlocked(state, 'mine') || unlocked(state, 'forge') || state.lifetime.highestRoute < 1) return { ok: false, message: 'Explore Watchtower Road before building the Forge.' };
      const costs = { ore: N.from(20) };
      if (!affordable(state, costs)) return { ok: false, message: 'The Forge needs 20 ore from your Mine.' };
      spend(state, costs); state.rooms.push('forge'); pushEvent(state, 'Forge built. Ore can now reinforce tools and expedition boots.');
      return { ok: true, message: 'The Forge is ready. Equipment and mining tools share your ore supply.' };
    }
    if (action.type === 'research') {
      const def = find(C.RESEARCH, action.id);
      if (!def || !unlocked(state, 'study') || state.lifetime.highestRoute + 1 < def.at || researched(state, def.id)) return { ok: false, message: 'That research is not available.' };
      const costs = { knowledge: N.from(def.cost) };
      if (!affordable(state, costs)) return { ok: false, message: 'More knowledge is needed.' };
      spend(state, costs); state.research.push(def.id); pushEvent(state, def.name + ' researched.');
      return { ok: true, message: def.description };
    }
    if (action.type === 'refit-upgrade' || action.type === 'legacy-upgrade') {
      const isLegacy = action.type === 'legacy-upgrade';
      const list = isLegacy ? C.LEGACY_UPGRADES : C.REFIT_UPGRADES;
      const map = isLegacy ? state.legacy : state.refitUpgrades;
      const def = find(list, action.id); const resource = isLegacy ? 'crests' : 'notes';
      if (!def || !(isLegacy ? state.lifetime.charters : state.lifetime.refits)) return { ok: false, message: 'Complete the relevant prestige first.' };
      const costs = { [resource]: product(def.base, power(def.scale, map[def.id])) };
      if (!affordable(state, costs)) return { ok: false, message: 'More ' + resource + ' are needed.' };
      spend(state, costs); map[def.id] += 1; return { ok: true, message: def.name + ' improved.' };
    }
    if (action.type === 'route') {
      const index = typeof action.id === 'string' && /^route-\d+$/.test(action.id) ? Number(action.id.slice(6)) : -1;
      if (!Number.isSafeInteger(index) || index < 0 || index > state.lifetime.highestRoute + 1) return { ok: false, message: 'Discover that route first.' };
      if (state.challenges.active && index > state.run.completed + 1) return { ok: false, message: 'Complete the previous route in this challenge expedition first.' };
      const mode = action.mode || state.route.mode;
      if (!find(C.MODES, mode) || mode !== 'frontier' && !unlocked(state, 'cartography')) return { ok: false, message: 'Discover the Map Room to choose a route purpose.' };
      if (state.route.index !== index) { state.route.progress = N.zero(); state.guild.prepared = { index: -1, kinds: [] }; }
      state.route.index = index; state.route.mode = mode;
      return { ok: true, message: getRoute(index, state).name + ' · ' + find(C.MODES, mode).name };
    }
    if (action.type === 'mode') return act(state, { type: 'route', id: 'route-' + state.route.index, mode: action.id });
    if (action.type === 'recipe') {
      const def = find(C.RECIPES, action.id);
      if (!def || !unlocked(state, def.room) || def.research && !researched(state, def.research)) return { ok: false, message: 'That recipe is not available yet.' };
      if (def.id.startsWith('meal-')) {
        if (state.challenges.active === 'light-pack' && def.id !== 'meal-none') return { ok: false, message: 'Light Pack requires an expedition without meals.' };
        state.meal = def.id; return { ok: true, message: def.name + ' selected.' };
      }
      const trade = recipeTrade(state, def.id);
      if (!affordable(state, trade.costs)) return { ok: false, message: def.id === 'alloy' ? 'Gather more herbs for this alloy.' : 'Chart more maps first.' };
      spend(state, trade.costs);
      Object.keys(trade.gains).forEach(id => { state.resources[id] = N.add(state.resources[id], trade.gains[id]); });
      return { ok: true, message: def.name + ' completed.' };
    }
    if (action.type === 'recruit') {
      const companion = action.kind === 'companion';
      const def = find(companion ? C.COMPANIONS : C.SPECIALISTS, action.id);
      const owned = companion ? state.crew.companions : state.crew.owned;
      if (!def || !unlocked(state, 'hall') || owned.includes(def.id)) return { ok: false, message: 'That guild member is not available to recruit.' };
      const costs = companion ? { coins: N.from(2000), knowledge: N.from(20) } : { coins: N.from(1000), knowledge: N.from(15) };
      if (!affordable(state, costs)) return { ok: false, message: 'Recruitment needs more coins and knowledge.' };
      spend(state, costs); owned.push(def.id);
      if (companion && state.crew.companion === null) state.crew.companion = def.id;
      if (!companion) { const free = state.crew.specialists.indexOf(null); if (free !== -1) state.crew.specialists[free] = def.id; }
      return { ok: true, message: def.name + ' joined the guild.' };
    }
    if (action.type === 'specialist') {
      if (![0, 1].includes(action.slot) || action.id !== null && !state.crew.owned.includes(action.id)) return { ok: false, message: 'Choose a recruited specialist and one of the two slots.' };
      const old = state.crew.specialists.indexOf(action.id);
      if (action.id !== null && old !== -1) state.crew.specialists[old] = null;
      state.crew.specialists[action.slot] = action.id; return { ok: true, message: 'Specialist assignment updated.' };
    }
    if (action.type === 'companion') {
      if (action.id !== null && !state.crew.companions.includes(action.id)) return { ok: false, message: 'Recruit that companion first.' };
      state.crew.companion = action.id; return { ok: true, message: 'Companion updated.' };
    }
    if (action.type === 'doctrine') {
      if (!unlocked(state, 'hall') || !find(C.DOCTRINES, action.id)) return { ok: false, message: 'Discover the Guild Hall first.' };
      state.doctrine = action.id; return { ok: true, message: find(C.DOCTRINES, action.id).name + ' adopted.' };
    }
    if (action.type === 'automation') {
      if (!find(C.AUTOMATIONS, action.id) || typeof action.enabled !== 'boolean' || !automationAvailable(state, action.id)) return { ok: false, message: 'Earn that automation license first.' };
      state.automations[action.id] = action.enabled; return { ok: true, message: 'Automation ' + (action.enabled ? 'enabled.' : 'paused.') };
    }
    if (action.type === 'refit') {
      const preview = getRefitPreview(state);
      if (!preview.available) return { ok: false, message: 'Complete the Refit requirements first.' };
      grantPremiumGift(state, 'first-refit');
      state.resources.notes = N.add(state.resources.notes, preview.reward); state.lifetime.refits += 1; state.chapter.refits += 1;
      resetRun(state); pushEvent(state, 'Expedition refitted: earned ' + preview.rewardText + '.');
      return { ok: true, message: 'Refit complete. Your developed professions and equipment remain.' };
    }
    if (action.type === 'charter') {
      const preview = getCharterPreview(state);
      if (!preview.available) return { ok: false, message: 'Complete the Charter requirements first.' };
      const crests = N.add(state.resources.crests, preview.reward);
      const starshards = state.resources.starshards;
      state.resources = dictionary(C.RESOURCES, N.zero); state.resources.crests = crests; state.resources.coins = N.from(100);
      state.resources.starshards = starshards;
      grantPremiumGift(state, 'first-charter');
      state.upgrades = dictionary(C.UPGRADES, 0); state.refitUpgrades = dictionary(C.REFIT_UPGRADES, 0);
      C.UPGRADES.filter(x => !x.equipment && x.id !== 'preparation' && unlocked(state, x.room)).forEach(x => { state.upgrades[x.id] = state.legacy.foundations; });
      state.chapter = { work: N.zero(), refits: 0 }; state.lifetime.charters += 1;
      state.guild.chapterProject = { number: state.lifetime.charters, choice: null };
      resetRun(state); state.resources.coins = N.from(100); pushEvent(state, 'New Guild Charter: earned ' + preview.rewardText + '.');
      return { ok: true, message: 'New Charter founded. Your discoveries, mastery, crew, research, and legacy remain.' };
    }
    if (action.type === 'challenge') {
      if (action.id === null) { state.challenges.active = null; return { ok: true, message: 'Challenge left. Current route progress remains.' }; }
      const def = find(C.CHALLENGES, action.id);
      if (!def || state.lifetime.highestRoute + 1 < def.at || completed(state, def.id) || state.challenges.active) return { ok: false, message: 'That challenge is not available.' };
      resetRun(state); state.challenges.active = def.id;
      pushEvent(state, def.name + ' challenge started. Route, coins, preparation, and meal were reset without a prestige reward.');
      return { ok: true, message: def.description };
    }
    return { ok: false, message: 'Unknown guild action.' };
  }


  function act(state, action) {
    const result = performAction(state, action);
    if (result.ok && validPlanAction(action)) {
      const plan = state.guild.plan;
      if (plan.goal && canonical(plan.goal) === canonical(action)) plan.goal = null;
      if (plan.queue.length && canonical(plan.queue[0]) === canonical(action)) plan.queue.shift();
      cleanPlan(state);
    }
    return result;
  }
  function baseGoal(state) {
    const current = getRoute(state.route.index, state);
    const progress = Math.min(1, ratio(state.route.progress, current.distance));
    const base = { progress, progressText: N.format(state.route.progress) + ' / ' + N.format(current.distance) + ' distance', action: null };
    if (state.challenges.active) {
      const challenge = find(C.CHALLENGES, state.challenges.active);
      const next = state.run.completed + 1;
      return Object.assign(base, { title: 'Complete ' + challenge.name, description: challenge.description + ' Follow the fresh expedition from Old Footpath; current route: ' + current.name + '.', action: state.route.index < next ? { type: 'route', id: 'route-' + next, mode: 'frontier' } : null });
    }
    if (!unlocked(state, 'mine')) return Object.assign(base, { title: 'Complete Old Footpath', description: 'Travel happens automatically. Finish Old Footpath to discover the Mine; better boots shorten the journey.', action: level(state, 'boots') < 3 ? { type: 'buy', id: 'boots' } : null });
    if (!unlocked(state, 'forge')) return Object.assign(base, { title: 'Build the Forge', description: state.lifetime.highestRoute < 1 ? 'Complete Watchtower Road and gather 20 ore. The Mine supplies your first workshop.' : 'Spend 20 ore from the Mine to build the Forge and choose between tools and expedition boots.', progress: Math.min(1, ratio(state.resources.ore, N.from(20))), progressText: N.format(state.resources.ore) + ' / 20 ore', action: { type: 'build-room', id: 'forge' } });
    if (!level(state, 'gear-tools') || !level(state, 'gear-boots')) {
      const equipped = Number(level(state, 'gear-tools') > 0) + Number(level(state, 'gear-boots') > 0);
      return Object.assign(base, { title: 'Connect your professions', description: 'Forge tools strengthen the Mine; reinforced boots advance the expedition. Both spend ore.', progress: equipped / 2, progressText: equipped + ' / 2 equipment improvements', action: { type: 'ui', tab: 'guild', room: 'forge' } });
    }
    const refit = getRefitPreview(state);
    if (refit.available && state.lifetime.refits === 0) return Object.assign(base, { title: 'Prepare your first Expedition Refit', description: 'Bank field notes and unlock automatic operation purchases. Preview exactly what stays before refitting.', progress: 1, progressText: 'Ready · ' + refit.rewardText, action: { type: 'ui', tab: 'journey' } });
    if (state.lifetime.refits && !state.automations.operations) return Object.assign(base, { title: 'Let the guild handle familiar work', description: 'Your first Refit earned operation automation. Enable it to improve rooms automatically.', progress: 0, progressText: 'Ready to enable', action: { type: 'automation', id: 'operations', enabled: true } });
    if (unlocked(state, 'study') && !researched(state, 'auto-forge')) {
      const cost = N.from(find(C.RESEARCH, 'auto-forge').cost);
      return Object.assign(base, { title: 'Research standing forge orders', description: 'Automate equipment, then choose a doctrine to set its purchase priorities.', progress: Math.min(1, ratio(state.resources.knowledge, cost)), progressText: N.format(state.resources.knowledge) + ' / ' + N.format(cost) + ' knowledge', action: { type: 'ui', tab: 'research' } });
    }
    if (unlocked(state, 'cartography') && state.route.index <= state.lifetime.highestRoute && !state.automations.routes) return Object.assign(base, { title: 'Choose the next expedition', description: 'Continue into undiscovered territory, or stay here for supplies and discoveries.', action: { type: 'route', id: 'route-' + (state.lifetime.highestRoute + 1), mode: 'frontier' } });
    const charter = getCharterPreview(state);
    if (charter.available) return Object.assign(base, { title: 'Found a new Guild Charter', description: 'Turn this guild’s work into lasting crests. Review the deeper reset before committing.', progress: 1, progressText: 'Ready · ' + charter.rewardText, action: { type: 'ui', tab: 'journey' } });
    if (state.lifetime.highestRoute >= 8) {
      const target = getRoute(charterTarget(state), state);
      let charterProgress, charterText;
      if (state.guild.grandfathered && state.chapter.refits < 2) {
        charterProgress = state.chapter.refits / 2;
        charterText = state.chapter.refits + ' / 2 Refits in this charter';
      } else if (state.run.completed < target.index) {
        charterProgress = Math.min(1, (state.run.completed + 1) / (target.index + 1));
        charterText = (state.run.completed + 1) + ' / ' + (target.index + 1) + ' routes this expedition';
      } else {
        charterProgress = Math.min(1, ratio(state.chapter.work, target.distance));
        charterText = N.format(state.chapter.work) + ' / ' + N.format(target.distance) + ' charter distance';
      }
      const missing = state.guild.grandfathered && state.chapter.refits < 2 ? charter.requirements[1] : charter.requirements.find(requirement => !requirement.met);
      return Object.assign(base, { title: 'Build a charter-worthy guild', description: missing ? missing.label + '.' : 'Review your completed charter requirements.', progress: charterProgress, progressText: charterText, action: { type: 'ui', tab: 'journey' } });
    }
    const nextRoom = C.ROOMS.find(room => !unlocked(state, room.id));
    const nextProject = nextRoom && !state.guild.grandfathered && C.PROJECTS.find(def => def.room === nextRoom.id);
    return Object.assign(base, { title: nextRoom ? nextProject ? nextProject.name : 'Discover ' + nextRoom.name : state.route.index >= 18 ? 'Chart the endless frontier' : 'Reach ' + current.name, description: nextRoom ? nextProject ? roomRequirement(state, nextRoom) + ' ' + nextRoom.description : nextRoom.description + ' Complete ' + nextRoom.at + ' routes to discover it.' : 'Improve a limiting profession, choose a run purpose, and push toward the next landmark.', progress: nextRoom ? Math.min(1, (state.lifetime.highestRoute + 1) / nextRoom.at) : base.progress, progressText: nextRoom ? (state.lifetime.highestRoute + 1) + ' / ' + nextRoom.at + ' routes discovered' : base.progressText, action: null });
  }

  function costInfo(state, costs, rates) {
    const missing = Object.keys(costs).filter(id => N.cmp(state.resources[id], costs[id]) < 0);
    let eta = 0;
    for (const id of missing) {
      const net = N.sub(rates.gain[id], rates.drain[id]);
      if (!net.m) { eta = null; break; }
      eta = Math.max(eta, ratio(N.sub(costs[id], state.resources[id]), net));
    }
    return { shortageText: missing.map(id => N.format(N.sub(costs[id], state.resources[id])) + ' more ' + find(C.RESOURCES, id).name.toLowerCase()).join(' + '), etaSeconds: eta };
  }
  function nearbyGoal(state, longGoal) {
    const rates = getRates(state), plan = state.guild.plan;
    const candidates = ['boots', 'gear-boots'].map(id => {
      const action = { type: 'buy', id }, detail = taskDetails(state, action);
      return { action, detail, info: costInfo(state, detail.costs, rates) };
    }).filter(item => item.detail.open && item.info.etaSeconds !== null).sort((a, b) => a.info.etaSeconds - b.info.etaSeconds);
    const saved = plan.goal && taskDetails(state, plan.goal);
    const next = saved && saved.open && !saved.done ? { action: clone(plan.goal), detail: saved, info: costInfo(state, saved.costs, rates) } : candidates[0];
    if (!next) return longGoal;
    const canPay = affordable(state, next.detail.costs);
    const options = [];
    const option = (id, label, description, action, effectText) => ({ id, icon: action.type === 'relic-equip' ? action.id : action.type === 'supply-plan' ? 'provisions' : action.action.id, label, description, effectText: effectText || description, action, cost: [], visible: true, unlocked: true, affordable: true, disabled: false, reason: '', shortageText: '', etaSeconds: 0 });
    const relics = C.RELICS.filter(def => state.luck.owned.includes(def.id) && def.id !== state.luck.active).map(def => {
      const preview = Object.assign({}, state, { luck: Object.assign({}, state.luck, { active: def.id }) });
      const entitlements = premiumEntitlements.get(state); if (entitlements) premiumEntitlements.set(preview, entitlements);
      return { def, travel: getRates(preview).travel };
    }).filter(item => N.cmp(item.travel, N.mul(rates.travel, 1.05)) > 0).sort((a, b) => N.cmp(b.travel, a.travel));
    if (relics.length) {
      const def = relics[0].def, comparison = comparisonFor(state, { luck: Object.assign({}, state.luck, { active: def.id }) });
      options.push(option('nearby-relic', 'Equip ' + def.name, comparison.text + '. Uses your one relic slot; its conditional bonus follows the current preparation.', { type: 'relic-equip', id: def.id }, comparison.text + ' · replaces the active relic.'));
    }
    if (plan.kit !== 'off' && hasCapability(state, 'kit-plan') && state.luck.active === 'living-crucible' && !state.luck.kit.active && state.guild.supply !== 'save') {
      const costs = kitCosts(state), provisionTime = ratio(N.sub(costs.provisions, state.resources.provisions), rates.gain.provisions);
      options.push(option('nearby-kit-supply', 'Save provisions for ' + plan.kit + ' kits', 'Suspend the meal bonus to accumulate provisions. Each renewal costs ' + N.format(costs.provisions) + ' provisions plus ore and knowledge; ' + (Number.isFinite(provisionTime) ? 'provision funding alone takes about ' + Math.ceil(provisionTime / 60) + ' minutes at current output' : 'develop the Kitchen to fund provisions') + '. The planned kit doubles ' + (plan.kit === 'travel' ? 'travel' : 'ore') + ' for 30 minutes while the Crucible remains equipped.', { type: 'supply-plan', id: 'save' }, 'Pause meals · ' + (Number.isFinite(provisionTime) ? 'fund kit provisions in ~' + Math.ceil(provisionTime / 60) + 'm' : 'develop provision supply') + ' · ' + plan.kit + ' kit ×2 for 30m.'));
    }
    if (options.length < 2) {
      const alternate = find(C.UPGRADES, next.action.id === 'gear-tools' ? 'gear-boots' : 'gear-tools');
      const costs = upgradeCost(state, alternate), info = costInfo(state, costs, rates);
      options.push(option('nearby-production', 'Save for ' + alternate.name.toLowerCase(), alternate.description + ' Costs ' + Object.keys(costs).map(id => N.format(costs[id]) + ' ' + id).join(' + ') + '. ' + (info.shortageText || 'Already affordable') + '; this becomes your primary saved objective.', { type: 'plan-goal', action: { type: 'buy', id: alternate.id } }, alternate.description + ' · cost ' + Object.keys(costs).map(id => N.format(costs[id]) + ' ' + id).join(' + ')));
    }
    const fractions = Object.keys(next.detail.costs).map(id => Math.min(1, ratio(state.resources[id], next.detail.costs[id])));
    const effect = next.action.type === 'buy' ? find(C.UPGRADES, next.action.id).description : 'Fund the objective selected in your guild plan.';
    return { title: 'Next improvement: ' + next.detail.name, description: effect + ' Save for this nearer improvement while the expedition advances toward its Charter.', longGoal: longGoal.description, progress: Math.min(...fractions), progressText: canPay ? 'Ready to improve' : 'Funding the next improvement', action: canPay ? next.action : { type: 'plan-goal', action: next.action }, targetCosts: next.detail.costs, options: options.slice(0, 2) };
  }
  function getGoal(state) {
    let goal = baseGoal(state), costs = {};
    const project = C.PROJECTS.find(def => projectOpen(state, def.id));
    if (project && !state.challenges.active && !(state.lifetime.refits === 0 && getRefitPreview(state).available)) {
      costs = projectCosts(state, project.id);
      const fractions = Object.keys(costs).map(id => Math.min(1, ratio(state.resources[id], costs[id])));
      goal = { title: project.name, description: roomRequirement(state, find(C.ROOMS, project.room)) + ' ' + project.description, progress: Math.min(...fractions), progressText: affordable(state, costs) ? 'Ready to build' : 'Gather the displayed project materials', action: { type: 'project', id: project.id } };
    } else if (goal.title === 'Connect your professions') {
      goal.action = { type: 'buy', id: level(state, 'gear-tools') ? 'gear-boots' : 'gear-tools' };
    } else if (goal.title === 'Research standing forge orders') goal.action = { type: 'research', id: 'auto-forge' };
    if (!state.guild.grandfathered && unlocked(state, 'cartography') && !state.guild.chapterProject.choice && !project && !state.challenges.active) {
      goal = { title: 'Prepare a regional Charter project', description: 'Choose supply camps, a regional atlas, or a waystation. Each uses a different profession chain and its completion stays through Refits.', progress: 0, progressText: 'Choose one regional project', action: { type: 'ui', tab: 'guild', section: 'projects' } };
    }
    if (!state.guild.grandfathered && goal.title === 'Build a charter-worthy guild' && state.guild.chapterProject.choice && planningOpen(state)) goal = nearbyGoal(state, goal);
    if (goal.targetCosts) { costs = goal.targetCosts; delete goal.targetCosts; }
    if (goal.action && goal.action.type === 'buy') costs = upgradeCost(state, find(C.UPGRADES, goal.action.id));
    if (goal.action && goal.action.type === 'research') costs = { knowledge: N.from(find(C.RESEARCH, goal.action.id).cost) };
    if (goal.action && goal.action.type === 'build-room') costs = { ore: N.from(20) };
    const info = costInfo(state, costs, getRates(state));
    const gatedForge = goal.action && goal.action.type === 'build-room' && state.lifetime.highestRoute < 1;
    const actionLabel = goal.action && goal.action.section === 'projects' ? 'Choose regional project' : goal.action ? goal.action.type === 'plan-goal' ? 'Save for ' + taskDetails(state, goal.action.action).name.toLowerCase() : goal.action.type === 'build-room' ? 'Build the Forge' : goal.action.type === 'project' ? projectName(goal.action.id) : goal.action.type === 'buy' ? find(C.UPGRADES, goal.action.id).name : goal.action.type === 'research' ? 'Research ' + find(C.RESEARCH, goal.action.id).name : goal.action.type === 'automation' ? 'Enable operation upgrades' : goal.action.type === 'route' ? 'Begin next expedition' : 'Review ' + (/charter/i.test(goal.title) ? 'Charter' : 'Refit') : '';
    return Object.assign(goal, { actionLabel, ready: !!goal.action && goal.action.section !== 'projects' && !gatedForge && (goal.action.type === 'plan-goal' || !info.shortageText), remainingText: gatedForge ? 'Complete Watchtower Road first' : info.shortageText || (goal.progress >= 1 ? 'Ready' : goal.progressText), etaSeconds: gatedForge ? null : Object.keys(costs).length ? info.etaSeconds : goal.progress >= 1 ? 0 : null });
  }

  function planBlockedReason(state) {
    if (!planningOpen(state)) return 'Earn a Refit or Workshop ledgers to author a plan.';
    const plan = state.guild.plan;
    const task = plan.goal || plan.queue[0];
    if (task) {
      const detail = taskDetails(state, task);
      if (!detail.open && !detail.done) return detail.name + ': meet its discovery requirements first.';
      const info = costInfo(state, reservedCosts(state, task, detail.costs, false), getRates(state));
      return info.shortageText ? detail.name + ': saving ' + info.shortageText + ', including reserves.' : detail.name + ': ready for the next planning minute.';
    }
    if (plan.kit !== 'off' && state.luck.active !== 'living-crucible') return 'Kit plan paused: equip the Living Crucible.';
    if (plan.kit !== 'off' && state.luck.kit.prepared && state.luck.kit.prepared !== plan.kit) return 'Kit plan paused: use the prepared ' + state.luck.kit.prepared + ' kit first, or select it as your kit plan. Your ' + plan.kit + ' renewal will wait.';
    const candidates = planningCandidates(state);
    if (!candidates.length) return state.luck.kit.active ? 'Kit active; its renewal waits until the current burst ends.' : 'No automatic purchase is currently scheduled.';
    const candidate = candidates.find(item => affordable(state, item.required)) || candidates[0];
    const info = costInfo(state, candidate.required, getRates(state));
    return info.shortageText ? candidate.name + ': need ' + info.shortageText + ', including reserves.' : candidate.name + ': ready for the next planning minute.';
  }
  function comparisonFor(state, change) {
    const preview = Object.assign({}, state, change);
    if (premiumEntitlements.has(state)) premiumEntitlements.set(preview, premiumEntitlements.get(state));
    const before = getRates(state), after = getRates(preview), changes = [];
    const b = {}, a = {};
    for (const id of ['travel', ...ORDINARY]) {
      const from = id === 'travel' ? before.travel : N.sub(before.gain[id], before.drain[id]);
      const to = id === 'travel' ? after.travel : N.sub(after.gain[id], after.drain[id]);
      b[id] = N.format(from) + '/s'; a[id] = N.format(to) + '/s';
      if (N.cmp(from, to)) changes.push((id === 'travel' ? 'Travel' : find(C.RESOURCES, id).name) + ' ' + b[id] + ' → ' + a[id]);
    }
    if (N.cmp(before.drain.provisions, after.drain.provisions)) changes.push('Provision use ' + N.format(before.drain.provisions) + ' → ' + N.format(after.drain.provisions) + '/s');
    return { before: b, after: a, text: changes.join('; ') || 'No production change under the current plan; inspect its conditional effect.' };
  }

  function roomRequirement(state, room) {
    if (room.id === 'forge') return 'Complete Watchtower Road (route 2), then spend 20 ore from the Mine to build the Forge.';
    const project = C.PROJECTS.find(def => def.room === room.id);
    const route = C.ROUTES[room.at - 1];
    if (!project || state.guild.grandfathered) return 'Complete ' + route.name + ' (route ' + room.at + ').';
    const prior = room.id === 'study' ? 'Kitchen' : room.id === 'hall' ? 'Study' : 'Guild Hall';
    const price = Object.keys(project.costs).map(id => N.format(project.costs[id]) + ' ' + id).join(' + ');
    return 'Complete ' + route.name + ' (route ' + room.at + '), have the Forge and ' + prior + ', then spend ' + price + ' to build it.';
  }
  function presentationSystems(state) {
    const hasRecord = state.luck.commonFinds > 0 || state.luck.relicFinds > 0 || state.luck.ledger.seq > 0;
    const account = premiumEntitlements.get(state);
    const seen = state.introductions && state.introductions.seen || [];
    const earned = {
      finds: hasRecord,
      collections: state.luck.owned.length > 0 || state.luck.collectionBanner,
      refit: state.lifetime.refits > 0 || state.lifetime.charters > 0 || getRefitPreview(state).available || seen.includes('refit') && unlocked(state, 'forge') && state.lifetime.highestRoute >= 2,
      planning: planningOpen(state),
      keepsakes: N.cmp(state.resources.starshards, 0) > 0 || state.premium.owned.length > 0 || !!(account && account.size),
      relics: unlocked(state, 'study') || state.luck.owned.length > 0,
      caravan: !!state.caravan.offer || !!state.caravan.pendingQuote || state.caravan.receipts.length > 0,
      contracts: C.CHALLENGES.some(def => state.lifetime.highestRoute + 1 >= def.at) || state.challenges.completed.length > 0 || !!state.challenges.active,
      charter: unlocked(state, 'cartography') || state.lifetime.charters > 0
    };
    const requirements = {
      finds: 'Find the first supply cache while travelling after discovering the Mine.',
      collections: 'Discover the first relic. Realm landmarks remain listed in the Guild record.',
      refit: 'Build the Forge, reinforce mining tools and expedition boots, complete Abandoned Quarry in this expedition, and earn the required expedition distance.',
      planning: 'Complete the first Expedition Refit or research Workshop ledgers.',
      keepsakes: 'Earn your first Starshards or already own a lasting charm or banner.',
      relics: 'Build or discover the Study. The first relic arrives within 45 eligible expedition minutes.',
      caravan: state.guild.grandfathered ? 'After building the Forge, wait for an actual caravan arrival. Established guild eligibility is preserved.' : 'After the first Refit and built Forge, wait for an actual caravan arrival.',
      contracts: 'Complete Mosslight Way (route 7) to make the first optional contract available.',
      charter: 'Open the Map Room to plan a regional project and review the deeper renewal.'
    };
    return C.PRESENTATION_SYSTEMS.filter(def => own(earned, def.id) ? earned[def.id] : unlocked(state, def.id)).map(def => {
      const room = find(C.ROOMS, def.id);
      return Object.assign({}, clone(def), { requirement: room ? roomRequirement(state, room) : requirements[def.id] });
    });
  }
  function getPresentation(state) {
    const systems = presentationSystems(state), ids = systems.map(system => system.id), has = id => ids.includes(id);
    const seen = state.introductions ? state.introductions.seen : ids;
    const rooms = C.ROOMS.filter(room => unlocked(state, room.id));
    const guildSections = rooms.length > 1 ? ['rooms'] : [];
    if (has('study')) guildSections.push('research');
    if (has('hall')) guildSections.push('crew');
    if (has('planning')) guildSections.push('planning');
    const journalSections = [];
    if (has('finds') || has('relics')) journalSections.push('discoveries');
    if (has('collections')) journalSections.push('collections');
    if (has('refit') || has('charter')) journalSections.push('renewals');
    if (has('contracts')) journalSections.push('challenges');
    if (journalSections.length) journalSections.push('record');
    const primary = ['trail'];
    if (guildSections.length) primary.push('guild');
    if (journalSections.length) primary.push('journey');
    const nextRoom = C.ROOMS.find(room => room.id !== 'trail' && !unlocked(state, room.id));
    let nextUnlock = null;
    if (nextRoom) {
      const def = find(C.PRESENTATION_SYSTEMS, nextRoom.id), project = C.PROJECTS.find(item => item.room === nextRoom.id);
      const completedRoutes = state.lifetime.highestRoute + 1;
      const routePart = state.route.index === completedRoutes ? Math.min(1, ratio(state.route.progress, getRoute(state.route.index, state).distance)) : 0;
      let progress = Math.min(1, (completedRoutes + routePart) / nextRoom.at), progressText = Math.min(completedRoutes, nextRoom.at) + ' / ' + nextRoom.at + ' routes completed';
      let action = { type: 'ui', tab: 'trail' };
      if (nextRoom.id === 'forge' && completedRoutes >= nextRoom.at) {
        progress = Math.min(1, ratio(state.resources.ore, 20)); progressText = N.format(N.min(state.resources.ore, 20)) + ' / 20 ore';
        action = affordable(state, { ore: N.from(20) }) ? { type: 'build-room', id: 'forge' } : { type: 'ui', tab: 'guild', room: 'mine' };
      } else if (project && !state.guild.grandfathered && completedRoutes >= nextRoom.at) {
        const costs = projectCosts(state, project.id), open = projectOpen(state, project.id);
        progress = open ? Math.min(...Object.keys(costs).map(id => Math.min(1, ratio(state.resources[id], costs[id])))) : 0;
        progressText = !open ? 'Build the required earlier rooms first' : affordable(state, costs) ? 'Materials ready — build to unlock' : costInfo(state, costs, getRates(state)).shortageText;
        action = open && affordable(state, costs) ? { type: 'project', id: project.id } : { type: 'ui', tab: 'guild', section: 'projects' };
      }
      nextUnlock = { id: nextRoom.id, label: def.label, requirement: roomRequirement(state, nextRoom), effect: def.effect, progress, progressText, action };
    }
    const resourceIds = C.RESOURCES.filter(def => def.id === 'coins' || def.id === 'starshards' ? def.id === 'coins' || has('keepsakes') : def.id === 'notes' ? state.lifetime.refits > 0 || N.cmp(state.resources.notes, 0) > 0 : def.id === 'crests' ? state.lifetime.charters > 0 || N.cmp(state.resources.crests, 0) > 0 : unlocked(state, def.room)).map(def => def.id);
    return { version: 1, opening: rooms.length === 1, primary, guildSections, journalSections, roomIds: rooms.map(room => room.id), resourceIds, systems, unlockedIds: ids, nextUnlock, introductions: systems.filter(system => !seen.includes(system.id)), show: { mastery: rooms.length > 1, travelStats: rooms.length > 1, supplies: has('kitchen'), routes: has('cartography'), research: has('study'), crew: has('hall'), planning: has('planning'), finds: has('finds') || has('relics'), collections: has('collections'), contracts: has('contracts'), projects: C.PROJECTS.some(def => projectOpen(state, def.id)) || has('cartography'), refit: has('refit'), charter: has('charter'), shop: has('keepsakes'), caravan: !!state.caravan.offer || !!state.caravan.pendingQuote } };
  }

  function getView(state) {
    const rates = getRates(state);
    const costRows = costs => Object.keys(costs).map(id => ({ resource: id, amount: N.from(costs[id]), text: N.format(costs[id]) + ' ' + find(C.RESOURCES, id).name.toLowerCase() }));
    function descriptor(id, label, description, action, costs, open, extra) {
      costs = costs || {};
      const canPay = affordable(state, costs);
      return Object.assign({ id, label, description, effectText: description, action, visible: true, unlocked: open !== false, affordable: canPay, disabled: open === false || !canPay, reason: open === false ? 'Continue exploring to unlock this.' : canPay ? '' : 'Needs ' + costRows(costs).filter(row => N.cmp(state.resources[row.resource], row.amount) < 0).map(row => row.text).join(' and '), cost: costRows(costs), level: 0 }, costInfo(state, costs, rates), extra || {});
    }
    const upgrades = C.UPGRADES.map(def => descriptor(def.id, def.name, def.description, { type: 'buy', id: def.id }, upgradeCost(state, def), upgradeOpen(state, def), { room: def.room, level: level(state, def.id), visible: upgradeOpen(state, def), effectText: upgradeOpen(state, def) ? comparisonFor(state, { upgrades: Object.assign({}, state.upgrades, { [def.id]: level(state, def.id) + 1 }) }).text : def.description }));
    upgrades.push(descriptor('build-forge', 'Build the Forge', 'Use 20 ore to open equipment crafting. Tools develop ore supply; boots help expeditions.', { type: 'build-room', id: 'forge' }, { ore: N.from(20) }, unlocked(state, 'mine') && state.lifetime.highestRoute >= 1 && !unlocked(state, 'forge'), { room: 'mine', visible: unlocked(state, 'mine') && !unlocked(state, 'forge'), reason: state.lifetime.highestRoute < 1 ? 'Complete Watchtower Road first.' : N.cmp(state.resources.ore, 20) < 0 ? 'Needs 20 ore.' : '' }));
    const research = C.RESEARCH.map(def => {
      const owned = researched(state, def.id);
      return descriptor(def.id, def.name, def.description, { type: 'research', id: def.id }, { knowledge: N.from(def.cost) }, unlocked(state, 'study') && state.lifetime.highestRoute + 1 >= def.at && !owned, { room: 'study', owned, maxed: owned, visible: owned || unlocked(state, 'study') && def.at <= state.lifetime.highestRoute + 1, reason: owned ? 'Researched permanently.' : undefined });
    });
    const recipes = C.RECIPES.map(def => {
      const trade = recipeTrade(state, def.id);
      const costs = trade.costs;
      const description = Object.keys(trade.gains).length ? 'Exchange ' + costRows(trade.costs).map(x => x.text).join(' + ') + ' for ' + costRows(trade.gains).map(x => x.text).join(' + ') + '. Material tier ' + tier(state) + '; all mastery is retained.' : def.description;
      const open = unlocked(state, def.room) && (!def.research || researched(state, def.research));
      return descriptor(def.id, def.name, description, { type: 'recipe', id: def.id }, costs, open && !(state.challenges.active === 'light-pack' && def.id !== 'meal-none' && def.id.startsWith('meal-')), { room: def.room, visible: open, selected: def.id === state.meal, outputs: costRows(trade.gains) });
    });
    const specialists = C.SPECIALISTS.flatMap(def => {
      const owned = state.crew.owned.includes(def.id);
      if (!owned) return [descriptor(def.id, 'Recruit ' + def.name, def.description, { type: 'recruit', kind: 'specialist', id: def.id }, { coins: N.from(1000), knowledge: N.from(15) }, unlocked(state, 'hall'), { visible: unlocked(state, 'hall'), owned: false, room: 'hall' })];
      return [0, 1].map(slot => descriptor(def.id + '-' + slot, def.name + ' · slot ' + (slot + 1), def.description, { type: 'specialist', slot, id: def.id }, {}, true, { room: 'hall', slot, specialistId: def.id, owned: true, visible: unlocked(state, 'hall'), selected: state.crew.specialists[slot] === def.id }));
    });
    const companions = C.COMPANIONS.map(def => {
      const owned = state.crew.companions.includes(def.id);
      return descriptor(def.id, (owned ? 'Travel with ' : 'Befriend ') + def.name, def.description, owned ? { type: 'companion', id: def.id } : { type: 'recruit', kind: 'companion', id: def.id }, owned ? {} : { coins: N.from(2000), knowledge: N.from(20) }, unlocked(state, 'hall'), { owned, visible: unlocked(state, 'hall'), selected: state.crew.companion === def.id, room: 'hall' });
    });
    const doctrines = C.DOCTRINES.map(def => descriptor(def.id, def.name, def.description, { type: 'doctrine', id: def.id }, {}, unlocked(state, 'hall'), { visible: unlocked(state, 'hall'), selected: state.doctrine === def.id, room: 'hall' }));
    const challenges = C.CHALLENGES.map(def => {
      const starshards = premiumGift(state, 'challenge-' + def.id);
      return descriptor(def.id, completed(state, def.id) ? def.name + ' · completed' : 'Begin ' + def.name, def.description + (starshards ? ' First completion also grants ' + starshards + ' earned Starshards.' : '') + ' Starting resets route, coins, preparation, and meal without a Refit reward.', { type: 'challenge', id: def.id }, {}, state.lifetime.highestRoute + 1 >= def.at && !completed(state, def.id) && !state.challenges.active, { visible: state.lifetime.highestRoute + 1 >= def.at || completed(state, def.id), owned: completed(state, def.id), selected: state.challenges.active === def.id, premiumGift: starshards });
    });
    if (state.challenges.active) challenges.push(descriptor('leave-challenge', 'Leave challenge', 'Keep current progress and restore unrestricted equipment and crew.', { type: 'challenge', id: null }, {}, true));
    const automations = C.AUTOMATIONS.map(def => descriptor(def.id, def.name, def.description, { type: 'automation', id: def.id, enabled: !state.automations[def.id] }, {}, automationAvailable(state, def.id), { visible: state.lifetime.refits > 0 || unlocked(state, 'study'), selected: state.automations[def.id] }));
    const refitUpgrades = C.REFIT_UPGRADES.map(def => descriptor('refit-' + def.id, def.name, def.description, { type: 'refit-upgrade', id: def.id }, { notes: product(def.base, power(def.scale, state.refitUpgrades[def.id])) }, state.lifetime.refits > 0, { visible: state.lifetime.refits > 0, level: state.refitUpgrades[def.id], group: 'refit' }));
    const legacyUpgrades = C.LEGACY_UPGRADES.map(def => descriptor('legacy-' + def.id, def.name, def.description, { type: 'legacy-upgrade', id: def.id }, { crests: product(def.base, power(def.scale, state.legacy[def.id])) }, state.lifetime.charters > 0, { visible: state.lifetime.charters > 0, level: state.legacy[def.id], group: 'legacy' }));
    const routes = [];
    for (let index = 0; index <= Math.min(17, state.lifetime.highestRoute + 1); index += 1) {
      const route = getRoute(index, state);
      const challengeBlocked = state.challenges.active && index > state.run.completed + 1;
      routes.push(descriptor(route.id, route.name, route.realmName + ' · ' + route.material + ' tier · ' + N.format(route.distance) + ' distance', { type: 'route', id: route.id }, {}, index <= state.lifetime.highestRoute + 1 && !challengeBlocked, { visible: unlocked(state, 'cartography'), selected: state.route.index === index, completed: index <= state.lifetime.highestRoute, realm: route.realm, tier: route.tier, reason: challengeBlocked ? 'Complete the previous challenge route first.' : '' }));
    }
    if (state.lifetime.highestRoute >= 17) {
      [...new Set([Math.max(18, state.route.index), Math.max(18, state.lifetime.highestRoute + 1)])].forEach(index => { const route = getRoute(index, state); const blocked = state.challenges.active && index > state.run.completed + 1; routes.push(descriptor(route.id, route.name, route.description, { type: 'route', id: route.id }, {}, !blocked, { selected: state.route.index === index, reason: blocked ? 'Complete the previous challenge route first.' : '' })); });
    }
    const modes = C.MODES.map(def => descriptor(def.id, def.name, def.description, { type: 'mode', id: def.id }, {}, unlocked(state, 'cartography'), { visible: unlocked(state, 'cartography'), selected: state.route.mode === def.id }));
    const rooms = C.ROOMS.map(room => ({ id: room.id, name: room.name, description: room.description, profession: room.profession, unlocked: unlocked(state, room.id), level: room.id === 'trail' ? level(state, 'boots') : Math.max(0, ...C.UPGRADES.filter(x => x.room === room.id && !x.equipment).map(x => level(state, x.id))), mastery: state.ranks[room.id], status: !unlocked(state, room.id) ? 'Undiscovered' : room.id === 'kitchen' && rates.kitchenRatio < 0.999 ? 'Limited by herb supply' : 'Working automatically', actions: upgrades.filter(x => x.room === room.id && x.visible).concat(recipes.filter(x => x.room === room.id && x.visible)) }));
    const current = getRoute(state.route.index, state);
    const resources = C.RESOURCES.map(def => {
      const negative = N.cmp(rates.gain[def.id], rates.drain[def.id]) < 0;
      const rate = negative ? N.sub(rates.drain[def.id], rates.gain[def.id]) : N.sub(rates.gain[def.id], rates.drain[def.id]);
      return { id: def.id, name: def.name, value: N.from(state.resources[def.id]), formatted: N.format(state.resources[def.id]), rate, rateFormatted: (negative ? '−' : '+') + N.format(rate) + '/s', visible: def.id === 'notes' ? state.lifetime.refits > 0 : def.id === 'crests' ? state.lifetime.charters > 0 : unlocked(state, def.room) };
    });
    const accountOwned = C.PREMIUM_ITEMS.filter(def => {
      const entitlements = premiumEntitlements.get(state);
      return entitlements && entitlements.has(def.id);
    }).map(def => def.id);
    const owned = C.PREMIUM_ITEMS.filter(def => premiumOwned(state, def.id)).map(def => def.id);
    const equipped = owned.includes(state.premium.equipped) ? state.premium.equipped : null;
    const premium = {
      unlocked: unlocked(state, 'forge'), balance: N.from(state.resources.starshards), balanceText: N.format(state.resources.starshards),
      dropDescription: 'After building the Forge, expeditions very rarely find 1 Starshard: about one per 2,000 expedition minutes (33 hours 20 minutes) on average. Offline time counts equally. Finds are random, not guaranteed on a timer.',
      owned, earnedOwned: state.premium.owned.slice(), accountOwned, equipped,
      drops: state.premium.drops, eligibleSeconds: state.premium.eligibleSeconds,
      items: C.PREMIUM_ITEMS.map(def => {
        const has = owned.includes(def.id);
        const banner = def.kind === 'banner';
        const selected = banner && equipped === def.id;
        const usable = !has || banner && !selected;
        return descriptor(def.id, def.name, def.description, { type: has && banner ? 'premium-equip' : 'premium-buy', id: def.id }, has ? {} : { starshards: N.from(def.cost) }, unlocked(state, 'forge') && usable, { visible: unlocked(state, 'forge'), owned: has, earnedOwned: state.premium.owned.includes(def.id), accountOwned: accountOwned.includes(def.id), selected, kind: def.kind, price: def.cost, maxed: has && !banner, reason: has ? selected ? 'Equipped.' : banner ? '' : 'Permanent charm active.' : undefined });
      })
    };
    const relicEffect = id => id === 'golden-pickaxe' ? 'Ore ×1.50 while equipped.' : id === 'surveyors-lens' ? 'Knowledge ×' + (1 + Math.min(0.75, 0.25 + state.ranks.cartography * 0.02)).toFixed(2) + '; maps ×' + (1 + Math.min(0.6, 0.2 + state.ranks.study * 0.02)).toFixed(2) + ' from cross-profession mastery.' : id === 'living-crucible' ? 'Prepare one kit with ' + costRows(kitCosts(state)).map(x => x.text).join(' + ') + '; mining or travel ×2 for 30 minutes.' : find(C.RELICS, id).description;
    const luck = {
      unlocked: unlocked(state, 'mine'), relicsUnlocked: unlocked(state, 'study'), active: state.luck.active, owned: state.luck.owned.slice(), duplicateProgress: clone(state.luck.duplicateProgress), collectionBanner: state.luck.collectionBanner,
      relics: C.RELICS.map(def => descriptor(def.id, def.name, def.description + ' ' + relicEffect(def.id), { type: 'relic-equip', id: def.id }, {}, state.luck.owned.includes(def.id), { visible: unlocked(state, 'study') && def.chapter <= state.lifetime.charters + 1, owned: state.luck.owned.includes(def.id), selected: state.luck.active === def.id, rarity: def.rarity, family: def.family, chapter: def.chapter, comparison: comparisonFor(state, { luck: Object.assign({}, state.luck, { active: def.id }) }), effect: relicEffect(def.id), progress: state.luck.duplicateProgress[def.id], reason: def.chapter > state.lifetime.charters ? 'Earn Charter ' + def.chapter + ' to search this family.' : state.luck.owned.includes(def.id) ? '' : 'Discover this relic or complete its guaranteed study.' })),
      research: C.LUCK_RESEARCH.map(def => descriptor(def.id, def.name, def.description, { type: 'luck-research', id: def.id }, { knowledge: N.from(def.cost) }, unlocked(state, def.room || 'study') && !state.luck.research.includes(def.id), { visible: unlocked(state, def.room || 'study'), room: def.room || 'study', owned: state.luck.research.includes(def.id), reason: state.luck.research.includes(def.id) ? 'Researched permanently.' : undefined })),
      hunts: [{ id: 'balanced', name: 'Open exploration' }, ...eligibleRelics(state)].map(def => descriptor(def.id, def.name, def.id === 'balanced' ? 'Normal travel and coins. No target weighting.' : 'Future searches give this relic 4× relative weight. Travel ×0.85; coins ×0.90. The current saved search destination does not change.', { type: 'relic-hunt', id: def.id }, {}, unlocked(state, 'cartography'), { visible: unlocked(state, 'cartography'), selected: state.luck.hunt === def.id })),
      kits: C.KITS.map(def => descriptor('prepare-' + def.id, 'Prepare ' + def.name, def.description + ' One prepared kit slot; burst effects require the Crucible to stay equipped.', { type: 'kit-prepare', id: def.id }, kitCosts(state), state.luck.active === 'living-crucible' && !state.luck.kit.prepared && unlocked(state, 'kitchen') && unlocked(state, 'study'), { visible: state.luck.owned.includes('living-crucible'), room: 'forge' })).concat(state.luck.kit.prepared ? [descriptor('use-kit', 'Use ' + find(C.KITS, state.luck.kit.prepared).name, 'Begin a 30-minute burst. Time continues offline; boosts never stack with another kit.', { type: 'kit-use' }, {}, state.luck.active === 'living-crucible' && !state.luck.kit.active, { room: 'forge' })] : []),
      kit: clone(state.luck.kit), scheduledHunt: state.luck.scheduledHunt,
      pity: { elapsedSeconds: state.luck.pitySeconds, guaranteeSeconds: pityLimit(state), progress: Math.min(1, state.luck.pitySeconds / pityLimit(state)), text: 'Next guaranteed discovery within ' + Math.ceil(Math.max(0, pityLimit(state) - state.luck.pitySeconds) / 60) + ' expedition minutes. A guarantee chooses a missing relic when possible.' },
      ledger: clone(state.luck.ledger),
      collections: [...new Set(C.RELICS.map(def => def.family))].map(family => { const members = C.RELICS.filter(def => def.family === family); return { id: family.toLowerCase().replace(/[^a-z]+/g, '-'), name: family, total: members.length, owned: members.filter(def => state.luck.owned.includes(def.id)).length, unlocked: members[0].chapter <= state.lifetime.charters, completed: members.every(def => state.luck.owned.includes(def.id)), reward: family + ' pennant · cosmetic recognition, no additional multiplier' }; })
    };
    const offer = state.caravan.offer, recentCompletions = state.caravan.completed.filter(at => at > state.lastUpdate - DAY), quoted = getCaravanQuote(state);
    const caravan = {
      unlocked: caravanUnlocked(state), offer: clone(offer), quote: quoted, pending: !!state.caravan.pendingQuote, pendingQuote: clone(state.caravan.pendingQuote), quoteText: caravanRewardText(quoted), surge: clone(state.caravan.surge),
      quota: { used: recentCompletions.length, limit: 3, nextAt: recentCompletions.length >= 3 ? recentCompletions[0] + DAY : 0 },
      description: 'One non-expiring offer; 3 verified rewards per rolling 24 hours. Ordinary play never requires ads. Golden offers double the displayed reward. Resource quotes exclude temporary kits, meals, and caravan boosts.',
      materials: unlockedMaterials(state).map(id => ({ id, label: find(C.RESOURCES, id).name })),
      relicTargets: eligibleRelics(state).filter(def => !state.luck.owned.includes(def.id)).map(def => ({ id: def.id, label: def.name })),
      choices: [
        descriptor('shipment', 'Supply shipment', (offer ? offer.minutes * (offer.golden ? 2 : 1) : '90–120') + ' minutes of unboosted coins and a chosen unlocked material. Select to view exact amounts.', { type: 'caravan-select', kind: 'shipment' }, {}, !!offer && !offer.locked && !state.caravan.pendingQuote, { visible: !!offer, selected: !!quoted && quoted.kind === 'shipment' }),
        descriptor('surge', 'Profession surge', 'Production ×3 for ' + (offer && offer.golden ? 90 : 45) + ' minutes in one profession. Maximum 90 remaining minutes; extends duration, never stacks multipliers.', { type: 'caravan-select', kind: 'surge' }, {}, !!offer && !offer.locked && !state.caravan.pendingQuote, { visible: !!offer, selected: !!quoted && quoted.kind === 'surge' }),
        descriptor('relic', 'Relic expedition', (offer && offer.golden ? 40 : 20) + '% guaranteed study toward a chosen missing relic, plus a modest coin cache. No second luck roll.', { type: 'caravan-select', kind: 'relic' }, {}, !!offer && !offer.locked && !state.caravan.pendingQuote && unlocked(state, 'study') && !!missingRelic(state), { visible: !!offer && unlocked(state, 'study'), selected: !!quoted && quoted.kind === 'relic' })
      ]
    };
    const projects = C.PROJECTS.map(def => descriptor(def.id, def.name, def.description, { type: 'project', id: def.id }, projectCosts(state, def.id), projectOpen(state, def.id), { visible: projectOpen(state, def.id), room: def.room, owned: unlocked(state, def.room) }));
    const stageRoom = C.ROOMS.filter(room => unlocked(state, room.id)).slice(-1)[0];
    const development = { grandfathered: state.guild.grandfathered, stage: stageRoom.id, title: stageRoom.name, description: state.guild.grandfathered ? 'Established guild: existing economy, discoveries, and reset requirements are preserved. New planning tools are optional.' : stageRoom.description, projects, chapter: { name: C.REALMS[Math.min(5, state.lifetime.charters + 2)].name, condition: 'Complete a regional project and its landmark to earn a Charter.', completed: !!state.guild.chapterProject.choice, projects: ['supply', 'survey', 'industry'].map(id => descriptor('chapter-' + id, projectName('chapter-' + id), 'A different profession chain can prepare this charter. Choose one project; progress remains through Refits.', { type: 'project', id: 'chapter-' + id }, projectCosts(state, 'chapter-' + id), projectOpen(state, 'chapter-' + id), { visible: unlocked(state, 'cartography'), owned: state.guild.chapterProject.choice === id })) } };
    const expedition = {
      supply: state.guild.supply,
      supplyChoices: [{ id: 'save', name: 'Save provisions', description: 'Suspend meal use and its bonus; ordinary travel continues.' }, { id: 'steady', name: 'Steady expedition', description: 'Standard meal strength and provision demand.' }, { id: 'push', name: 'Supplied push', description: 'Twice the provision demand for a 50% stronger meal bonus.' }].map(def => descriptor(def.id, def.name, def.description, { type: 'supply-plan', id: def.id }, {}, unlocked(state, 'kitchen'), { visible: unlocked(state, 'kitchen'), selected: state.guild.supply === def.id })),
      preparation: C.PREPARATIONS.map(def => descriptor(def.id, def.name, def.description + ' One preparation per route; consumed on completion, route change, or reset.', { type: 'prepare-route', id: def.id }, preparationCosts(state, def.id), unlocked(state, 'cartography') && !state.guild.prepared.kinds.length, { visible: unlocked(state, 'cartography'), selected: prepared(state, def.id), reason: state.guild.prepared.kinds.length ? 'A preparation is already active on this route.' : undefined })),
      prepared: state.guild.prepared.kinds.slice(), conditionText: regionCondition(state).text, demandText: 'Provisions: +' + N.format(rates.gain.provisions) + '/s produced; ' + N.format(rates.drain.provisions) + '/s consumed. ' + (rates.mealRatio < 1 && state.meal !== 'meal-none' && state.guild.supply !== 'save' ? 'Meal strength is limited by supply.' : 'Basic travel always continues.')
    };
    const planning = {
      unlocked: planningOpen(state), reserves: ORDINARY.map(id => ({ id, name: find(C.RESOURCES, id).name, value: N.from(state.guild.plan.reserves[id]), formatted: N.format(state.guild.plan.reserves[id]) })), goal: clone(state.guild.plan.goal), priorities: clone(state.guild.plan.priorities), queue: state.guild.plan.queue.map(action => Object.assign({}, action, { name: taskDetails(state, action).name })), blockedReason: planBlockedReason(state), audit: clone(state.guild.audit), kit: state.guild.plan.kit, preparation: state.guild.plan.preparation, loadouts: state.guild.loadouts.map(entry => ({ id: entry.id, name: entry.name })),
      capabilities: C.CAPABILITIES.map(def => descriptor(def.id, def.name, def.description, { type: 'capability', id: def.id }, { [def.resource]: N.from(def.cost) }, !hasCapability(state, def.id) && (def.resource === 'notes' ? state.lifetime.refits > 0 : state.lifetime.charters >= def.at), { visible: def.resource === 'notes' ? state.lifetime.refits > 0 : state.lifetime.charters >= Math.max(1, def.at - 1), owned: hasCapability(state, def.id), maxed: hasCapability(state, def.id), reason: hasCapability(state, def.id) ? 'Learned permanently.' : undefined }))
    };
    return {
      development, expedition, planning, presentation: getPresentation(state),
      title: "Wayfarers’ Guild", resources, rooms, actions: upgrades.concat(refitUpgrades, legacyUpgrades), routes, modes, recipes, research, specialists, companions, doctrines, challenges, automations, refitUpgrades, legacyUpgrades, premium, luck, caravan,
      refit: getRefitPreview(state), charter: getCharterPreview(state), goal: getGoal(state), unlocks: state.rooms.slice(), events: state.events.slice(),
      progression: { routeId: current.id, routeIndex: current.index, routeName: current.name, realm: current.realm, realmName: current.realmName, routeProgress: N.format(state.route.progress), routeTarget: N.format(current.distance), routePercent: Math.min(1, ratio(state.route.progress, current.distance)), mode: state.route.mode, highestRoute: state.lifetime.highestRoute, frontier: state.lifetime.frontier, refits: state.lifetime.refits, charters: state.lifetime.charters, challenge: state.challenges.active, notes: N.format(state.resources.notes), mastery: clone(state.ranks), collections: state.collections.map(id => ({ id, name: find(C.REALMS, id).name, description: 'Permanent +8% production and travel.' })), specialists: state.crew.specialists.slice(), companion: state.crew.companion, doctrine: state.doctrine, materialTier: tier(state), meal: state.meal },
      stats: { travelRate: N.format(rates.travel) + '/s', travel: rates.travel, runDistance: N.format(state.run.work), chapterDistance: N.format(state.chapter.work), lifetimeDistance: N.format(state.lifetime.work), elapsed: state.run.elapsed, kitchenSupply: rates.kitchenRatio, provisionSupply: rates.mealRatio, recipes: C.RECIPES.filter(def => unlocked(state, def.room) && (!def.research || researched(state, def.research))).length }
    };
  }
  return { VERSION, MAX_TIME, Numbers: N, Content: C, format: N.format, createState, normalizeState, validateState, migrateState, setPremiumEntitlements, getRoute, getRates, upgradeCost, advance, advanceTo, act, getView, getPresentation, getGoal, getRefitPreview, getCharterPreview, getCaravanQuote, beginCaravanReward, cancelCaravanReward, grantCaravanReward };
});
