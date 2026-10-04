(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./progression-content.js') : root.WayfarersProgressionContent, common ? require('./upgrade-tiers.js') : root.WayfarersUpgradeTiers, common ? require('./practice-lessons.js') : root.WayfarersPractice);
  if (common) module.exports = api;
  if (root) root.WayfarersOnboarding = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, D, Tiers, Practice) {
  'use strict';
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const clone = value => JSON.parse(JSON.stringify(value));
  const exact = (value, keys) => object(value) && Object.keys(value).length === keys.length && keys.every(key => own(value, key));
  const REWARDS = { greenway: 12, quarry: 60, watchtower: 120, workshop: 200, ruins: 360, harbor: 600 };
  const GUIDES = D.AREAS.map(area => area.id).concat('cards', 'equipment');
  const tierCache = new Map();
  const tierRows = state => {
    const version = [3,4].includes(state.expedition?.version) ? 3 : 2;
    if (!tierCache.has(version)) tierCache.set(version, Tiers.describe(state));
    return tierCache.get(version);
  };
  const firstArea = 'area:greenway';
  let contextProvider = null, attentionProvider = null, attentionIds = new Set();
  const setContextProvider = fn => { contextProvider = fn; Practice.setContextProvider(fn); };
  const configureAttention = options => { attentionProvider = options.provider; attentionIds = new Set(options.ids); };
  const attentionInitial = () => ({ version: 1, seen: [], currencies: ['coins'], finds: {}, findSequence: 0 });
  const WALLET = ['coins', 'ore', 'herbs', 'provisions', 'knowledge', 'maps', 'notes', 'crests', 'starshards', 'ink'];
  function earnedCurrencies(state) {
    const area = id => !!state.expedition?.areas?.[id], room = id => state.rooms.includes(id);
    const open = { coins: true, ore: area('quarry') || room('mine'), herbs: area('ruins') || room('forage'), provisions: area('workshop') || room('kitchen'), knowledge: area('watchtower') || room('study'), maps: area('watchtower') || room('cartography'), notes: state.lifetime.refits > 0, crests: state.lifetime.charters > 0, starshards: area('quarry') || room('forge'), ink: !!state.collection?.cardsUnlocked };
    return WALLET.filter(id => open[id] || state.onboarding?.attention?.currencies.includes(id) || state.onboarding?.practice?.currencyRead?.includes(id));
  }
  function currencyRows(state) {
    return earnedCurrencies(state).map(id => {
      const value = N.from(id === 'ink' ? state.collection.ink : state.resources[id]);
      return { ...Practice.currencyInfo(id), value, formatted: N.format(value), ...(id === 'starshards' ? { balanceSource: 'earned-plus-verified-account', earnedValue: value, paidValue: null } : {}) };
    });
  }
  function attentionRows(state, context) {
    const c = context || contextProvider?.(state) || {}, options = attentionProvider?.(state) || [];
    const result = [];
    const add = (id, kind, label, extra = {}) => { if (!result.some(row => row.id === id)) result.push({ id, kind, label, ...extra }); };
    for (const area of D.AREAS) if (hasArea(state, area.id)) add('area:' + area.id, 'area', area.name, { areaId: area.id, destination: ui('expedition', { areaId: area.id }) });
    for (const row of c.globalUpgrades || []) if (row.visible !== false && row.status !== 'locked' && row.state !== 'locked') add('upgrade:' + row.id, 'upgrade', row.label || row.name, { areaId: row.areaId || null, itemId: row.id, destination: row.action?.type === 'expedition-buy' ? ui('expedition', { areaId: row.areaId, upgradeId: row.action.id }) : ui('upgrades', { catalogId: row.id }) });
    for (const row of options) add('option:' + row.areaId + ':' + row.group + ':' + row.id, 'option', row.label, { areaId: row.areaId, itemId: row.id, group: row.group, destination: ui('expedition', { areaId: row.areaId, control: 'plans', group: row.group, optionId: row.id }) });
    for (const id of Object.keys(state.collection?.cards || {})) add('card:' + id, 'card', id, { itemId: id, destination: ui('cards', { cardId: id }) });
    for (const id of Object.keys(state.collection?.gear || {})) add('gear:' + id, 'gear', id, { itemId: id, destination: ui('equipment', { itemId: id }) });
    for (const row of currencyRows(state)) add('currency:' + row.id, 'currency', row.name, { itemId: row.id, destination: ui('currency', { currencyId: row.id }) });
    for (const row of definitions(state)) if (row.earned && !row.retired && !['area', 'project'].includes(row.kind) && (row.kind !== 'tier-ready' || row.pending)) add('discovery:' + row.id, row.kind, row.label, { areaId: row.targetAreaId || null, itemId: row.id, destination: row.goToAction });
    return result;
  }
  const attentionToken = (state, row) => [state.createdAt, state.run.id, 'inspect', row.id, state.onboarding?.attention?.finds?.[row.id]?.latest || 0, JSON.stringify(row.destination)].join('|');
  function syncFinds(state) {
    const x = state.onboarding.attention, migrated = !own(x, 'finds');
    if (migrated) { x.finds = {}; x.findSequence = 0; }
    for (const event of state.collection?.recent || []) {
      if (event.id <= x.findSequence || event.outcome !== 'found' || !(event.delta?.owned > 0 || event.delta?.copies > 0)) continue;
      const id = event.cardId ? 'card:' + event.cardId : event.itemId ? 'gear:' + event.itemId : null;
      if (!id) continue;
      const previous = x.finds[id];
      x.finds[id] = { latest: event.id, seen: migrated && x.seen.includes(id) ? event.id : previous?.seen || 0 };
    }
    x.findSequence = state.collection?.sequence || 0;
  }
  function attentionView(state, context) {
    const seen = state.onboarding?.attention?.seen || [];
    const items = attentionRows(state, context).map(row => ({ ...row, unseen: (!seen.includes(row.id) || (state.onboarding.attention?.finds?.[row.id]?.latest || 0) > (state.onboarding.attention?.finds?.[row.id]?.seen || 0)) && !(row.kind === 'currency' && state.onboarding.practice?.currencyRead?.includes(row.itemId)), inspectAction: { type: 'onboarding-item-inspect', id: row.id, token: attentionToken(state, row) } }));
    const unseen = items.filter(row => row.unseen), areaIds = Object.fromEntries(D.AREAS.map(area => [area.id, unseen.some(row => row.areaId === area.id)]));
    return { items, count: unseen.length, areas: Object.values(areaIds).some(Boolean), areaIds, upgrades: unseen.some(row => ['upgrade', 'tier', 'tier-ready', 'batch'].includes(row.kind)), guild: unseen.some(row => row.kind === 'upgrade' && !row.areaId || row.id === 'discovery:feature:guild'), cards: unseen.some(row => row.kind === 'card' || row.id === 'discovery:feature:cards'), equipment: unseen.some(row => row.kind === 'gear' || row.id === 'discovery:feature:equipment'), currency: unseen.some(row => row.kind === 'currency') };
  }
  function initializeAttention(state) {
    const x = state.onboarding, rows = attentionRows(state), unread = definitions(state).filter(row => row.earned && !x.read.includes(row.id));
    const newItems = new Set((state.collection?.recent || []).filter(row => row.id > state.collection.seen).flatMap(row => [row.cardId ? 'card:' + row.cardId : null, row.itemId ? 'gear:' + row.itemId : null].filter(Boolean)));
    x.attention = { version: 1, currencies: earnedCurrencies(state), seen: rows.filter(row => !newItems.has(row.id) && !unread.some(notice => row.id === 'discovery:' + notice.id || row.id === notice.id || notice.kind === 'tier' && row.kind === 'upgrade' && row.areaId === notice.targetAreaId)).map(row => row.id) };
  }
  function initial() {
    return { version: 1, progress: { greenway: 0 }, active: null, entries: [firstArea], announced: [firstArea], read: [firstArea], rewardClaims: [], practice: Practice.initial(), attention: attentionInitial() };
  }
  const hasArea = (state, id) => !!state.expedition?.areas?.[id];
  const available = (state, id) => GUIDES.includes(id) && (id === 'cards' ? hasArea(state, 'quarry') : id === 'equipment' ? hasArea(state, 'watchtower') : hasArea(state, id));
  const ui = (screen, extra = {}) => Object.assign({ type: 'ui', screen }, extra);
  function tierDestination(row) {
    const purchase = row.actions[0];
    return row.local ? ui('expedition', { areaId: row.areaId, upgradeId: purchase.id }) : ui('upgrades', { tierId: row.id, purchase: clone(purchase) });
  }
  function retiredDestination(state, row) {
    if (row.local && row.actions[0].id === 'lift' && [3,4].includes(state.expedition?.version)) return ui('expedition', { areaId: row.areaId, upgradeId: 'signals' });
    return ui('expedition', { areaId: hasArea(state, row.areaId) ? row.areaId : state.expedition?.selectedArea || 'greenway' });
  }
  function definitions(state) {
    const areaEffects = { greenway: 'Automatic deliveries earn coins and expand the Trail.', quarry: 'Extract, haul and refine ore for the guild.', watchtower: state.expedition?.version === 2 ? 'Restore the Tower and coordinate earlier areas.' : 'Survey for knowledge, maps and project research.', workshop: 'Turn available ore into useful manufacturing output.', ruins: 'Delve, interpret and recover useful discoveries.', harbor: 'Fund voyages with provisions and receive cargo on arrival.' };
    const rows = D.AREAS.map(area => ({ id: 'area:' + area.id, kind: 'area', label: area.name + ' unlocked', effect: areaEffects[area.id], icon: area.icon, targetAreaId: area.id, goToAction: ui('expedition', { areaId: area.id }), earned: hasArea(state, area.id) }));
    const features = [
      ['cards', 'Cards', 'Build saved decks from cards you discover.', 'relic', hasArea(state, 'quarry'), ui('cards')],
      ['equipment', 'Equipment', 'Equip items and improve them with earned scrolls.', 'equipment', hasArea(state, 'watchtower'), ui('equipment')],
      ['upgrades', 'Upgrades', 'Browse the upgrade tiers you have explicitly unlocked.', 'research', Object.keys(state.expedition?.areas || {}).length > 1 || state.rooms.includes('hall'), ui('upgrades')],
      ['guild', 'Guild', 'Manage your earned guild systems and lasting progress.', 'guild', hasArea(state, 'watchtower') || state.rooms.includes('hall'), ui('guild')],
      ['focus', 'Focus', 'Spend a shared charge to briefly accelerate one area.', 'focus', state.lifetime.refits > 0, ui('expedition', { areaId: state.expedition?.selectedArea || 'greenway', control: 'focus' })]
    ];
    features.forEach(([id, name, effect, icon, earned, goToAction]) => rows.push({ id: 'feature:' + id, kind: 'feature', label: name + ' unlocked', effect, icon, earned, goToAction }));
    D.BATCHES.filter(batch => batch.count > 1).forEach(batch => rows.push({ id: 'batch:' + batch.count, kind: 'batch', label: 'Buy ×' + batch.count + ' unlocked', effect: 'Choose an exact batch of ' + batch.count + ' ranks when you can afford the complete price.', icon: 'coins', earned: state.lifetime.refits >= batch.refits && state.lifetime.charters >= batch.charters, goToAction: ui('expedition', { areaId: state.expedition?.selectedArea || 'greenway', control: 'batch' }) }));
    const tiers = tierRows(state);
    tiers.filter(row => !row.local || row.index > 0).forEach(row => rows.push({ id: 'ready:' + row.id, kind: 'tier-ready', label: row.label + ' available', effect: row.shortEffect, icon: row.icon, targetAreaId: row.areaId, earned: !!state.upgradeTiers && (state.upgradeTiers.pending.includes(row.id) || state.upgradeTiers.claimed.includes(row.id)), pending: !!state.upgradeTiers?.pending.includes(row.id), tierId: row.id, goToAction: tierDestination(row) }));
    tiers.filter(row => !row.local || row.index > 0).forEach(row => rows.push({ id: 'tier:' + row.id, kind: 'tier', label: row.label + ' unlocked', effect: row.shortEffect, icon: row.icon, targetAreaId: row.areaId, earned: !!state.upgradeTiers?.claimed.includes(row.id), goToAction: tierDestination(row) }));
    // Prior-version entries remain meaningful after explicit adoption.
    const other = tierRows({ ...state, expedition: { ...state.expedition, version: [3,4].includes(state.expedition?.version) ? 2 : 3 } });
    other.filter(row => (!row.local || row.index > 0) && !rows.some(item => item.id === 'ready:' + row.id)).forEach(row => rows.push({ id: 'ready:' + row.id, kind: 'tier-ready', label: row.label + ' · earlier expedition', effect: 'Retained history. Current options are shown in the area.', icon: row.icon, targetAreaId: row.areaId, earned: !!state.upgradeTiers && (state.upgradeTiers.pending.includes(row.id) || state.upgradeTiers.claimed.includes(row.id)), pending: false, retired: true, tierId: row.id, goToAction: retiredDestination(state, row) }));
    other.filter(row => (!row.local || row.index > 0) && !rows.some(item => item.id === 'tier:' + row.id)).forEach(row => rows.push({ id: 'tier:' + row.id, kind: 'tier', label: row.label + ' · earlier expedition', effect: 'Retained history. Current options are shown in the area.', icon: row.icon, targetAreaId: row.areaId, retired: true, earned: !!state.upgradeTiers?.claimed.includes(row.id), goToAction: retiredDestination(state, row) }));
    D.PROJECTS.forEach(project => rows.push({ id: 'project:' + project.id, kind: 'project', label: project.name + ' completed', effect: project.effect, icon: 'research', sourceAreaId: project.source, targetAreaId: project.target, earned: !!state.expedition?.projects?.includes(project.id), goToAction: ui('expedition', { areaId: project.target, control: project.unlock.track ? 'upgrade-tier' : 'plans', upgradeId: project.unlock.track || null }) }));
    return rows;
  }
  function migrate(state) {
    const ids = GUIDES.filter(id => available(state, id));
    const entries = definitions(state).filter(row => row.earned).map(row => row.id);
    return { version: 1, progress: Object.fromEntries(ids.map(id => [id, 0])), active: null, entries, announced: entries.slice(), read: entries.slice(), rewardClaims: ids.slice() };
  }
  function sync(state) {
    if (!state.onboarding) return;
    const x = state.onboarding;
    GUIDES.filter(id => available(state, id)).forEach(id => { if (!own(x.progress, id)) x.progress[id] = 0; });
    for (const row of definitions(state)) {
      if (row.earned && !x.entries.includes(row.id)) x.entries.push(row.id);
      if (row.earned && (row.retired || row.kind === 'tier-ready' && !row.pending)) {
        if (!x.announced.includes(row.id)) x.announced.push(row.id);
        if (!x.read.includes(row.id)) x.read.push(row.id);
      }
    }
    if (x.active && !['cards', 'equipment'].includes(x.active) && state.expedition?.selectedArea !== x.active) x.active = null;
    Practice.sync(state);
    if (!x.attention) initializeAttention(state);
    x.attention.currencies = [...new Set(x.attention.currencies.concat(earnedCurrencies(state)))];
    syncFinds(state);
  }
  function adopt(state) {
    const x = state.onboarding;
    if (!x) return;
    // An explicitly adopted run can rename retained tracks and projects. Treat
    // those mapped entitlements as known, while genuinely new features remain new.
    for (const row of definitions(state)) if (row.earned && (row.kind === 'project' || (row.id.startsWith('tier:area:') || row.id.startsWith('ready:area:')) && own(x.progress, row.targetAreaId))) {
      if (!x.entries.includes(row.id)) x.entries.push(row.id);
      if (!x.announced.includes(row.id)) x.announced.push(row.id);
      if (!x.read.includes(row.id)) x.read.push(row.id);
    }
  }
  function steps(state, id) {
    const legacy = state.expedition?.version === 2;
    const step = (key, target, heading, body, fallbackTarget = 'scene') => ({ id: key, target, fallbackTarget, heading, body, ackLabel: 'Next' });
    const purpose = {
      greenway: ['Keep the Trail moving', 'The Trail works automatically. Deliveries earn coins while the landmark meter fills.'],
      quarry: ['Follow the ore', 'Extraction feeds hauling, then refining. The slowest station limits the finished ore you receive.'],
      watchtower: legacy ? ['Restore and protect', 'The Tower divides workers between repairs and protection. Both help complete its landmark.'] : ['Turn surveys into knowledge', 'The Tower makes knowledge and maps. Its research capacity also advances the projects you fund.'],
      workshop: ['Manufacture with real inputs', 'The Workshop consumes available ore to manufacture supplies. Empty inputs or protected reserves can stop its work.'],
      ruins: ['Recover discoveries', 'Delving feeds interpretation, then recovery. Finds provide resources and later support artifact assignments.'],
      harbor: ['Prepare, sail, deliver', 'Voyages spend provisions before departure. Resources arrive when a funded voyage finishes, not every second.']
    };
    const operation = {
      greenway: ['Inspect before upgrading', 'The upgrade name opens its next improvement and full price. Nothing is purchased during this guide.'],
      quarry: ['Upgrade the limiting station', 'Inspect an earned upgrade. More capacity helps most when that station is limiting the chain.'],
      watchtower: ['Choose your next improvement', legacy ? 'Inspect worker, lift or beacon upgrades as their tiers become available.' : 'Inspect Surveying first. Later claimed tiers add coordination and specialist capacity.'],
      workshop: ['Inspect manufacturing', 'Processing shows the Workshop’s inputs and output. Earned plans let you choose a manufacturing template.'],
      ruins: ['Inspect discovery recovery', 'Processing shows discovery, interpretation and recovery. Earned plans select which type of find to recover.'],
      harbor: ['Inspect the next departure', 'Processing shows departure supplies, ships at sea and arriving cargo. Already funded voyages keep their original costs and rewards.']
    };
    if (id === 'cards') return [step('purpose', 'cards-introduction', 'Cards need a deck', 'Only cards in the active deck apply their effects. Owning a card alone gives no production bonus.', 'cards'), step('operation', 'cards-decks', 'Choose a saved deck', 'A card can occupy one slot in each deck. Only the active deck supplies its effects.', 'cards'), step('next-step', 'cards-library', 'Keep useful duplicates', 'Fusion spends duplicates for a guaranteed rank improvement. Card rarity stays unchanged.', 'cards')];
    if (id === 'equipment') return [step('purpose', 'equipment-introduction', 'Equipment is a lasting choice', 'Equipped items apply their base effects. Owning an item alone does not change production.', 'equipment'), step('operation', 'equipment-slots', 'Use the matching slot', 'Tool, head, coat and boots each hold one item. Inspect an item before replacing what is equipped.', 'equipment'), step('next-step', 'equipment-inventory', 'Improve an owned item', 'Open an owned item to choose a scroll. A failed attempt uses a slot but keeps the item and its bonuses.', 'equipment')];
    const plan = ['workshop', 'ruins', 'harbor'].includes(id);
    const last = id === 'greenway' ? ['Follow the next goal', 'The goal names the next landmark or unlock. Three foundation skills show what to earn next. A ready skill becomes available when you choose Unlock.']
      : id === 'quarry' ? ['Keep earlier areas working', 'The Trail continues producing while you develop the Quarry. New projects can improve both areas.']
        : id === 'watchtower' ? ['Connect your areas', 'Fund unlocked research in Upgrades. Its result can add a new option or improve an earlier area.']
          : ['Invest in the current operation', 'Inspect an earned upgrade here. Its comparison shows which part of this operation will improve.'];
    return [step('purpose', 'scene', ...purpose[id]), step('operation', plan ? 'area-plans' : 'area-upgrades', ...operation[id], plan ? 'area-goal' : 'scene'), step('next-step', plan ? 'area-upgrades' : 'area-goal', ...last)];
  }
  function guide(state, id) {
    const x = state.onboarding, area = D.AREAS.find(item => item.id === id), progress = x?.progress[id] || 0;
    const authored = steps(state, id);
    authored[authored.length - 1].ackLabel = 'Start working';
    const amount = x?.rewardClaims.includes(id) ? 0 : REWARDS[id] || 0;
    return { id, areaId: area ? id : null, title: area?.name || (id === 'cards' ? 'Cards' : 'Equipment'), mandatory: !!area || !!state.collection?.[id === 'cards' ? 'cardsUnlocked' : 'equipmentUnlocked'], complete: progress === authored.length, progress, steps: authored, visitAction: { type: 'onboarding-visit', id }, rewardPreview: { resource: 'coins', amount: N.from(amount), text: amount ? amount + ' coins after completing this guide, once for this guild.' : 'No reward is granted for replay or an already discovered area.', available: amount > 0 } };
  }
  function act(state, action) {
    const x = state.onboarding;
    if (!x) return { ok: false, message: 'Reload this guild before starting a guide.' };
    if (action.type === 'onboarding-item-inspect') {
      if (!exact(action, ['type', 'id', 'token']) || !x.attention) return { ok: false, message: 'Open an earned item’s details first.' };
      const row = attentionRows(state).find(item => item.id === action.id);
      if (!row || attentionToken(state, row) !== action.token) return { ok: false, message: 'This inspection belongs to another item or expedition.' };
      const find = x.attention.finds?.[row.id];
      if (x.attention.seen.includes(row.id) && (!find || find.seen === find.latest)) return { ok: false, message: 'This inspection was already saved.' };
      if (!x.attention.seen.includes(row.id)) x.attention.seen.push(row.id);
      if (find) find.seen = find.latest;
      if (row.kind === 'currency' && x.practice && !x.practice.currencyRead.includes(row.itemId)) x.practice.currencyRead.push(row.itemId);
      return { ok: true, message: 'Inspection saved.' };
    }
    if (x.practice && ['onboarding-visit', 'onboarding-next', 'onboarding-leave', 'onboarding-inspect', 'onboarding-help-open', 'onboarding-currency-ack'].includes(action.type)) return Practice.act(state, action);
    if (action.type === 'onboarding-visit') {
      if (!exact(action, ['type', 'id']) || !available(state, action.id) || !own(x.progress, action.id) || !['cards', 'equipment'].includes(action.id) && state.expedition.selectedArea !== action.id || x.progress[action.id] === 3) return { ok: false, message: 'Visit an available, unfinished guide.' };
      if (x.active && x.active !== action.id) return { ok: false, message: 'Close the current guide before starting another.' };
      x.active = action.id; return { ok: true, message: 'Guide opened.' };
    }
    if (action.type === 'onboarding-leave') {
      if (!exact(action, ['type', 'id']) || x.active !== action.id) return { ok: false, message: 'That guide is not open.' };
      x.active = null; return { ok: true, message: 'Guide progress saved.' };
    }
    if (action.type === 'onboarding-next') {
      if (!exact(action, ['type', 'id', 'stepId']) || x.active !== action.id || !available(state, action.id) || !['cards', 'equipment'].includes(action.id) && state.expedition.selectedArea !== action.id) return { ok: false, message: 'Open this area’s guide first.' };
      const current = guide(state, action.id), expected = current.steps[current.progress];
      if (!expected || expected.id !== action.stepId) return { ok: false, message: 'This guide step has changed. Continue from the current step.' };
      x.progress[action.id] += 1;
      if (x.progress[action.id] !== current.steps.length) return { ok: true, message: 'Guide step saved.' };
      const amount = x.rewardClaims.includes(action.id) ? 0 : REWARDS[action.id] || 0;
      if (!x.rewardClaims.includes(action.id)) x.rewardClaims.push(action.id);
      if (amount) state.resources.coins = N.add(state.resources.coins, amount);
      x.active = null;
      const discovery = (['cards', 'equipment'].includes(action.id) ? 'feature:' : 'area:') + action.id;
      if (x.entries.includes(discovery)) {
        if (!x.announced.includes(discovery)) x.announced.push(discovery);
        if (!x.read.includes(discovery)) x.read.push(discovery);
      }
      return { ok: true, completedGuide: action.id, reward: { coins: N.from(amount) }, message: current.title + ' guide complete.' + (amount ? ' +' + amount + ' coins.' : '') };
    }
    if (action.type === 'onboarding-open') {
      const row = definitions(state).find(item => item.id === action.id && item.earned);
      if (!exact(action, ['type', 'id']) || !row || !x.entries.includes(action.id)) return { ok: false, message: 'That discovery is not available.' };
      if (row.kind === 'tier-ready' && row.pending) {
        const result = Tiers.act(state, { type: 'upgrade-tier-unlock', id: row.tierId });
        if (!result.ok) return result;
        Tiers.sync(state); sync(state);
      }
      const ids = row.kind === 'tier-ready' ? [row.id, 'tier:' + row.tierId] : [row.id];
      ids.filter(id => x.entries.includes(id)).forEach(id => { if (!x.announced.includes(id)) x.announced.push(id); if (!x.read.includes(id)) x.read.push(id); });
      return { ok: true, goToAction: clone(row.goToAction), destination: clone(row.goToAction), message: row.kind === 'tier-ready' ? 'Upgrade tier unlocked.' : 'Discovery opened.' };
    }
    if (['onboarding-announce', 'onboarding-read'].includes(action.type)) {
      if (!exact(action, ['type', 'ids']) || !Array.isArray(action.ids) || !action.ids.length || new Set(action.ids).size !== action.ids.length || action.ids.some(id => !x.entries.includes(id))) return { ok: false, message: 'Choose existing discoveries to acknowledge.' };
      action.ids.forEach(id => {
        if (!x.announced.includes(id)) x.announced.push(id);
        if (action.type === 'onboarding-read' && !x.read.includes(id)) x.read.push(id);
        if (id.startsWith('ready:')) {
          const tierId = id.slice(6);
          if (state.upgradeTiers?.pending.includes(tierId) && !state.upgradeTiers.prompted.includes(tierId)) state.upgradeTiers.prompted.push(tierId);
        }
      });
      return { ok: true, message: 'Discovery notice saved.' };
    }
    return { ok: false, message: 'Choose a valid guide action.' };
  }
  function view(state, context) {
    const x = state.onboarding;
    if (!x) return { identity: String(state.createdAt) + ':1', guides: [], active: null, inbox: { entries: [], unreadCount: 0 }, notice: null };
    const guides = GUIDES.filter(id => available(state, id)).map(id => guide(state, id));
    const current = guides.find(item => item.id === x.active);
    const active = current && !current.complete ? { ...clone(current), ...clone(current.steps[current.progress]), id: current.id, guideId: current.id, stepId: current.steps[current.progress].id, index: current.progress, total: current.steps.length, action: { type: 'onboarding-next', id: current.id, stepId: current.steps[current.progress].id }, leaveAction: { type: 'onboarding-leave', id: current.id } } : null;
    const rows = definitions(state);
    const entries = x.entries.map(id => rows.find(row => row.id === id)).filter(Boolean).map(row => { const item = clone(row); delete item.earned; return { ...item, read: x.read.includes(row.id), announced: x.announced.includes(row.id), readAction: { type: 'onboarding-read', ids: [row.id] }, openAction: { type: 'onboarding-open', id: row.id }, openLabel: row.kind === 'tier-ready' && row.pending ? 'Unlock & go' : row.retired ? 'View area' : 'Go to' }; });
    const unseen = entries.filter(row => !row.announced);
    const practice = x.practice ? Practice.view(state, context) : null;
    return { identity: String(state.createdAt) + (practice ? ':practice-1' : ':1'), guides: practice?.guides || guides, active: practice?.active || (practice ? null : active), currencies: currencyRows(state), attention: attentionView(state, context), triggers: practice?.triggers || [], helpQueue: practice?.helpQueue || [], inbox: { entries, unreadCount: entries.filter(row => !row.read).length }, notice: unseen.length ? { id: unseen.map(row => row.id).join('|'), title: unseen.map(row => row.id).length === 1 ? 'New discovery' : unseen.length + ' new discoveries', items: unseen, deferAction: { type: 'onboarding-announce', ids: unseen.map(row => row.id) } } : null };
  }
  function validate(x, state) {
    try {
      if (!exact(x, ['version', 'progress', 'active', 'entries', 'announced', 'read', 'rewardClaims'].concat(own(x, 'practice') ? ['practice'] : [], own(x, 'attention') ? ['attention'] : [])) || x.version !== 1 || !object(x.progress)) return false;
      if (own(x, 'practice') && !Practice.validate(x.practice, state)) return false;
      if (own(x, 'attention') && (!exact(x.attention, ['version', 'seen', 'currencies'].concat(own(x.attention, 'finds') ? ['finds', 'findSequence'] : [])) || x.attention.version !== 1 || !Array.isArray(x.attention.seen) || x.attention.seen.length > 1000 || new Set(x.attention.seen).size !== x.attention.seen.length || x.attention.seen.some(id => !attentionIds.has(id) && !definitions(state).some(row => 'discovery:' + row.id === id)) || !Array.isArray(x.attention.currencies) || new Set(x.attention.currencies).size !== x.attention.currencies.length || x.attention.currencies.some(id => !WALLET.includes(id)))) return false;
      if (x.attention && own(x.attention, 'finds') && (!object(x.attention.finds) || Object.keys(x.attention.finds).length > 26 || !Number.isSafeInteger(x.attention.findSequence) || x.attention.findSequence < 0 || x.attention.findSequence > state.collection.sequence || Object.entries(x.attention.finds).some(([id, value]) => !attentionIds.has(id) || !/^(card|gear):/.test(id) || !exact(value, ['latest', 'seen']) || !Number.isSafeInteger(value.latest) || value.latest < 1 || value.latest > x.attention.findSequence || !Number.isSafeInteger(value.seen) || value.seen < 0 || value.seen > value.latest))) return false;
      if (Object.entries(x.progress).some(([id, value]) => !available(state, id) || !Number.isSafeInteger(value) || value < 0 || value > 3)) return false;
      if (x.active !== null && (typeof x.active !== 'string' || !own(x.progress, x.active) || x.progress[x.active] >= 3)) return false;
      if (x.active !== null && !['cards', 'equipment'].includes(x.active) && state.expedition?.selectedArea !== x.active) return false;
      const rows = definitions(state), earned = new Set(rows.filter(row => row.earned).map(row => row.id));
      for (const key of ['entries', 'announced', 'read']) if (!Array.isArray(x[key]) || x[key].length > rows.length || new Set(x[key]).size !== x[key].length || x[key].some(id => typeof id !== 'string' || !earned.has(id))) return false;
      if (x.announced.some(id => !x.entries.includes(id)) || x.read.some(id => !x.announced.includes(id))) return false;
      if (!Array.isArray(x.rewardClaims) || x.rewardClaims.length > GUIDES.length || new Set(x.rewardClaims).size !== x.rewardClaims.length || x.rewardClaims.some(id => typeof id !== 'string' || !available(state, id) || !own(x.progress, id))) return false;
      if (Object.entries(x.progress).some(([id, value]) => value === 3 && !x.rewardClaims.includes(id))) return false;
      return true;
    } catch (_) { return false; }
  }
  function mergePracticeReceipts(incoming, current) {
    Practice.mergeReceipts(incoming, current);
    if (incoming.createdAt === current.createdAt && incoming.onboarding?.attention && current.onboarding?.attention) {
      incoming.onboarding.attention.seen = [...new Set(incoming.onboarding.attention.seen.concat(current.onboarding.attention.seen))];
      incoming.onboarding.attention.currencies = [...new Set(incoming.onboarding.attention.currencies.concat(current.onboarding.attention.currencies))];
      // Keep the imported branch's event cursor. Importing a future receipt
      // number would hide genuinely new finds after an older save is restored.
      for (const [id, find] of Object.entries(incoming.onboarding.attention.finds || {})) if (current.onboarding.attention.seen.includes(id) && (!current.onboarding.attention.finds?.[id] || current.onboarding.attention.finds[id].seen >= find.latest)) find.seen = find.latest;
    }
    return incoming;
  }
  return { configureAttention, initial, migrate, sync, adopt, act, view, validate, setContextProvider, setCostProvider: Practice.setCostProvider, guardAction: Practice.guardAction, preparePractice: Practice.prepare, capture: Practice.capture, observeAction: Practice.observe, mergePracticeReceipts };
});
