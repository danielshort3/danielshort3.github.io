(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./progression-content.js') : root.WayfarersProgressionContent, common ? require('./collections.js') : root.WayfarersCollections);
  if (common) module.exports = api;
  if (root) root.WayfarersPractice = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, D, Collection) {
  'use strict';
  const own = (x, k) => Object.prototype.hasOwnProperty.call(x, k);
  const object = x => !!x && typeof x === 'object' && !Array.isArray(x);
  const clone = x => JSON.parse(JSON.stringify(x));
  const exact = (x, keys) => object(x) && Object.keys(x).length === keys.length && keys.every(k => own(x, k));
  const AREAS = D.AREAS.map(a => a.id);
  const PRIMARY = AREAS.concat('cards', 'equipment');
  const OPTIONAL = ['tiers', 'expansion', 'plans', 'bulk', 'guild-upgrades', 'projects', 'techniques', 'technique-config', 'focus', 'automation', 'reserves', 'crew', 'companions', 'relics', 'meals', 'kits', 'card-archive', 'card-craft', 'gear-craft', 'gear-repair', 'gear-reforge', 'specialization', 'configuration', 'supply', 'planner', 'playbooks', 'refit', 'charter', 'shop', 'caravan'];
  const IDS = PRIMARY.concat(OPTIONAL);
  const TITLES = { greenway: 'Trail', quarry: 'Quarry', watchtower: 'Tower', workshop: 'Workshop', ruins: 'Ruins', harbor: 'Harbor', cards: 'Cards and decks', equipment: 'Equipment and scrolls', tiers: 'Unlock an upgrade tier', expansion: 'Expand the guild', plans: 'Choose a working plan', bulk: 'Buy an exact batch', 'guild-upgrades': 'Guild improvements', projects: 'Fund a connected project', techniques: 'Improve an area technique', 'technique-config': 'Choose a technique mode', focus: 'Use a Focus charge', automation: 'Set a standing plan', reserves: 'Protect a reserve', crew: 'Recruit and assign crew', companions: 'Choose a companion', relics: 'Equip a relic', meals: 'Choose a meal', kits: 'Prepare an expedition kit', 'card-archive': 'Archive a duplicate', 'card-craft': 'Craft a discovered card', 'gear-craft': 'Craft equipment', 'gear-repair': 'Restore a failed slot', 'gear-reforge': 'Review reforging', specialization: 'Assign a specialist track', configuration: 'Configure an operation', supply: 'Set provision demand', planner: 'Save an upgrade goal', playbooks: 'Save a guild playbook', refit: 'Review a Refit', charter: 'Review a Charter', shop: 'Review the Starshard shop', caravan: 'Review a caravan reward' };
  const KEYS = Object.fromEntries(IDS.map(id => [id, AREAS.includes(id) ? ['inspect', 'upgrade', 'operate'] : id === 'cards' ? ['equip', 'fuse', 'deck', 'second-deck', 'return-deck'] : id === 'equipment' ? ['equip', 'scroll', 'result'] : id === 'bulk' ? ['quantity', 'purchase'] : ['inspect', 'practice']]));
  let contextProvider = null;
  let costProvider = null;
  const setContextProvider = fn => { contextProvider = fn; };
  const setCostProvider = fn => { costProvider = fn; };
  const CURRENCIES = {
    coins: ['Coins', 'coins', 'Spend on upgrades and guild helpers.', 'Earn coins as your Trail crew works and completes deliveries.'],
    ore: ['Ore', 'ore', 'Build equipment, improve industry and supply manufacturing.', 'The Quarry extracts, hauls and refines ore.'],
    herbs: ['Herbs', 'herbs', 'Prepare recipes and fund discoveries.', 'Foragers and botanical discoveries produce herbs.'],
    provisions: ['Provisions', 'provisions', 'Supply meals and fund a voyage before it departs.', 'Kitchens, recipes and manufacturing produce provisions.'],
    knowledge: ['Knowledge', 'knowledge', 'Research improvements and fund connected projects.', 'Surveys, study and recovered discoveries produce knowledge.'],
    maps: ['Maps', 'maps', 'Fund exploration and later connected projects.', 'Surveys, cartography and voyage cargo provide maps.'],
    notes: ['Field notes', 'notes', 'Buy lasting Refit improvements.', 'Complete an eligible Refit to earn field notes.'],
    crests: ['Guild crests', 'crests', 'Buy lasting Charter improvements.', 'Complete an eligible Guild Charter to earn crests.'],
    starshards: ['Starshards', 'starshards', 'Buy permanent shop benefits. Earned and purchased balances are tracked separately.', 'Find rare Starshards or make a verified optional purchase.'],
    ink: ['Archive Ink', 'relic', 'Craft extra copies of a card you have already discovered.', 'Archive loose duplicates to earn Ink.'],
    copies: ['Duplicate cards', 'relic', 'Fusion spends copies for a guaranteed rank increase; rarity stays the same.', 'Discover another copy or craft a previously discovered card.'],
    steady: ['Steady Scrolls', 'equipment', 'Use one attempt slot for a guaranteed enhancement point.', 'Find scrolls in earned supply caches.'],
    bold: ['Bold Scrolls', 'equipment', 'Spend one attempt slot: 60% chance of two enhancement points.', 'Find scrolls in earned supply caches.'],
    brilliant: ['Brilliant Scrolls', 'equipment', 'Spend one attempt slot: 15% chance of five enhancement points.', 'Find scrolls in earned supply caches.'],
    restoration: ['Restoration Scrolls', 'equipment', 'Recover one failed attempt slot. Successful enhancements stay unchanged.', 'Find Restoration Scrolls in rare earned supply caches.'],
    focus: ['Focus charges', 'focus', 'Temporarily accelerate one area while respecting its available inputs.', 'Unlocked after a Refit; shared charges recharge over time.']
  };
  const currencyInfo = id => own(CURRENCIES, id) ? { id, name: CURRENCIES[id][0], label: CURRENCIES[id][0], icon: CURRENCIES[id][1], purpose: CURRENCIES[id][2], earnedFrom: CURRENCIES[id][3] } : null;
  const context = (state, supplied) => supplied || (contextProvider ? contextProvider(state) : {});
  const TRIGGERS = { tiers: ['upgrade-tier-unlock', 'onboarding-open'], expansion: ['expedition-next'], plans: ['expedition-choice'], bulk: ['expedition-batch'], 'guild-upgrades': ['buy', 'refit-upgrade', 'legacy-upgrade', 'capability'], projects: ['expedition-development', 'project', 'research', 'luck-research'], 'technique-config': ['area-skill-config'], focus: ['expedition-focus'], automation: ['automation', 'expedition-automation'], reserves: ['plan-reserve'], crew: ['recruit', 'specialist'], companions: ['companion', 'recruit'], relics: ['relic-equip', 'relic-hunt'], meals: ['recipe'], kits: ['kit-prepare', 'kit-use'], 'card-archive': ['card-recycle'], 'card-craft': ['card-craft'], 'gear-craft': ['gear-forge'], 'gear-repair': ['gear-scroll'], specialization: ['expedition-specialize'], configuration: ['expedition-config'], supply: ['supply-plan'], planner: ['plan-goal', 'plan-priority', 'plan-queue', 'plan-kit', 'plan-preparation'], playbooks: ['loadout-save', 'loadout-use'] };
  const initial = () => ({ version: 1, progress: { greenway: 0 }, active: null, bindings: {}, intentions: {}, proofs: [], supplies: [], rewards: [], helpRewards: [], currencyRead: [] });
  const ledger = state => state.onboarding?.practice;
  const hasArea = (s, id) => !!s.expedition?.areas?.[id];
  const bindingUsable = (s, id, b) => b && (!b.cardId || !!s.collection.cards[b.cardId]) && (!b.itemId || !!s.collection.gear[b.itemId]) && (!b.trackId || own(s.expedition.areas[id]?.ranks || {}, b.trackId));
  function earned(s, id) {
    if (AREAS.includes(id)) return hasArea(s, id);
    if (id === 'cards' || id.startsWith('card-')) return !!s.collection?.cardsUnlocked;
    if (['equipment', 'gear-craft'].includes(id)) return !!s.collection?.equipmentUnlocked;
    if (id === 'gear-repair') return !!s.collection?.equipmentUnlocked && Object.values(s.collection.gear).some(item => item.failed > 0) || own(ledger(s)?.progress || {}, id);
    if (id === 'gear-reforge') return !!s.collection?.equipmentUnlocked && hasArea(s, 'workshop');
    if (id === 'tiers') return (s.upgradeTiers?.pending.length || 0) > 0 || (s.upgradeTiers?.claimed.length || 0) > 1;
    if (id === 'expansion') return !!s.expedition?.completed || Object.keys(s.expedition?.areas || {}).length > 1;
    if (['plans', 'guild-upgrades'].includes(id)) return hasArea(s, 'quarry');
    if (id === 'techniques') return !!s.areaSkills?.unlocked?.length;
    if (id === 'technique-config') return !!s.areaSkills && Object.values(s.areaSkills.ranks || {}).some(rank => rank > 0);
    if (['projects', 'configuration'].includes(id)) return hasArea(s, 'watchtower');
    if (['bulk', 'focus'].includes(id)) return s.lifetime.refits > 0;
    if (id === 'automation') return s.lifetime.refits > 0 || s.rooms.includes('study');
    if (id === 'playbooks') return s.guild.capabilities.includes('loadouts');
    if (['reserves', 'planner'].includes(id)) return s.lifetime.refits > 0 || s.guild.capabilities.includes('workshop-ledgers');
    if (['crew', 'companions'].includes(id)) return s.rooms.includes('hall');
    if (id === 'relics') return s.luck.owned.length > 0;
    if (['meals', 'supply'].includes(id)) return s.rooms.includes('kitchen');
    if (id === 'kits') return s.rooms.includes('study') && s.rooms.includes('kitchen') && s.luck.owned.includes('living-crucible');
    if (id === 'specialization') return s.expedition?.version === 3 && Object.values(s.expedition.areas).some(a => Object.values(a.highRanks).some(r => r >= 25));
    if (id === 'refit') return s.rooms.includes('forge');
    if (id === 'charter') return s.rooms.includes('cartography') || s.lifetime.charters > 0;
    if (id === 'shop') return hasArea(s, 'quarry');
    if (id === 'caravan') return !!s.caravan.offer || own(ledger(s)?.progress || {}, id);
    return false;
  }
  function sync(state) {
    if (!state.onboarding) return;
    if (!state.onboarding.practice) state.onboarding.practice = initial();
    const x = ledger(state);
    if (!own(x, 'currencyRead')) x.currencyRead = [];
    for (const id of IDS) if (earned(state, id) && !own(x.progress, id)) x.progress[id] = 0;
    for (const id of Object.keys(x.bindings)) if (earned(state, id) && !bindingUsable(state, id, x.bindings[id])) x.bindings[id] = bind(state, id, {});
    for (const id of PRIMARY) if (earned(state, id) && x.progress[id] === KEYS[id].length) {
      state.onboarding.progress[id] = 3;
      if (!state.onboarding.rewardClaims.includes(id)) state.onboarding.rewardClaims.push(id);
    }
  }
  const cleanAction = a => Object.fromEntries(Object.entries(a || {}).filter(([k]) => !['quote', 'lesson', 'epoch'].includes(k)));
  const actionKey = a => JSON.stringify(Object.entries(cleanAction(a)).sort(([a], [b]) => a.localeCompare(b)));
  function normalizedAction(state, a) {
    const out = cleanAction(a);
    if (out.type === 'expedition-buy') { out.areaId ||= state.expedition.selectedArea; out.count ||= state.expedition.version === 3 ? state.expedition.batch : 1; }
    if (['buy', 'refit-upgrade', 'legacy-upgrade'].includes(out.type) && state.expedition.version === 3) out.count ||= 1;
    if (out.type === 'expedition-automation' && out.dispatch === undefined && state.expedition.version === 3) out.dispatch = state.expedition.automation.dispatch;
    return out;
  }
  const rows = value => Array.isArray(value) ? value : [];
  const availableRows = value => rows(value).filter(row => row.visible !== false);
  const usable = value => availableRows(value).filter(row => !row.disabled && !row.selected && !row.owned && !row.maxed);
  const destination = (screen, extra = {}) => ({ type: 'ui', screen, ...extra });
  function bind(state, id, c) {
    const a = state.expedition.areas[id], local = rows(c.globalUpgrades).filter(row => row.areaId === id && row.action?.type === 'expedition-buy');
    const b = {};
    if (a) { b.areaId = id; b.trackId = (local.find(row => row.rank === 0) || local[0])?.trackId || Object.keys(a.ranks)[0]; }
    if (id === 'cards') { b.cardId = Object.keys(state.collection.cards).find(key => state.collection.cards[key].rank === 1) || Object.keys(state.collection.cards)[0]; b.deckId = state.collection.activeDeck; b.returnDeckId = state.collection.activeDeck; b.otherDeckId = state.collection.decks.find(d => d.id !== b.deckId).id; }
    if (id === 'equipment') b.itemId = Object.keys(state.collection.gear).find(key => Collection.used(state.collection.gear[key]) < 6) || Object.keys(state.collection.gear)[0];
    return b;
  }
  function makeStep(id, mode, target, heading, body, action, dest, targetData = {}, extra = {}) {
    return { id, mode, target, fallbackTarget: 'none', heading, body, requiredAction: action || null, destination: dest, targetData, ...extra };
  }
  function steps(state, id, supplied) {
    const c = context(state, supplied), x = ledger(state), saved = x?.bindings[id], b = bindingUsable(state, id, saved) ? saved : bind(state, id, c);
    const selected = state.expedition.selectedArea, areaName = TITLES[id] || id;
    const inspect = (key, target, title, body, dest, data = {}) => makeStep(key, 'inspect', target, title, body, null, dest, data);
    const action = (key, target, title, body, a, dest, data = {}, extra = {}) => makeStep(key, a ? 'action' : 'wait', target, title, body, a, dest, data, extra);
    const collection = c.collection || Collection.view(state), cardRows = collection.cards?.items || collection.cards || [];
    if (AREAS.includes(id)) {
      const local = rows(c.globalUpgrades).find(row => row.action?.type === 'expedition-buy' && row.areaId === id && (row.trackId || row.action.id) === b.trackId);
      const dest = destination('expedition', { areaId: id });
      const bought = (state.expedition.areas[id].ranks[b.trackId] || 0) > 0;
      const buy = { type: 'expedition-buy', areaId: id, id: b.trackId, count: 1 };
      // First visits teach an always-available inspector. Choosing a different
      // plan has its own real-action lesson when an alternative is earned.
      // Keep the saved operate proof and target identity for stuck older saves.
      const operate = id === 'greenway'
        ? inspect('operate', 'area-goal', 'Find your next destination', 'Open the current objective. Completed landmarks keep producing while you choose the next expansion.', dest, { areaId: id, section: 'objective' })
        : inspect('operate', 'area-plans', 'Open Processing', 'Open Processing to see this area’s working stages and what currently limits its output. New working plans appear here when you earn them.', dest, { areaId: id, section: 'plans' });
      return [inspect('inspect', 'area-upgrades', 'Open ' + (local?.name || 'an earned upgrade'), 'Open its comparison and price. The next step uses this real upgrade control.', dest, { areaId: id, trackId: b.trackId }), bought ? inspect('upgrade', 'area-upgrades', 'Review your existing investment', 'You already invested in this track. Inspect its earned rank and next effect; no extra purchase or reset is required.', dest, { areaId: id, trackId: b.trackId, mastered: true }) : action('upgrade', 'area-buy', 'Buy the first improvement', 'Use the upgrade button. The guild supplies this one practice rank; the improvement remains in your guild.', local ? buy : null, dest, { areaId: id, trackId: b.trackId }, { supply: 'rank', suppliesText: 'Guild supplies one first-rank purchase. Your saved resources are not spent.' }), operate];
    }
    if (id === 'cards') {
      const owned = state.collection.cards[b.cardId], deck = state.collection.decks.find(d => d.id === b.deckId), other = state.collection.decks.find(d => d.id === b.otherDeckId);
      const card = Collection.Content.CARDS.find(d => d.id === b.cardId), slot = deck?.slots.slice(0, Collection.slots(state)).findIndex(v => v === null) ?? -1, otherSlot = other?.slots.slice(0, Collection.slots(state)).findIndex(v => v === null) ?? -1;
      const already = deck?.slots.includes(b.cardId), another = other?.slots.includes(b.cardId), dest = destination('cards', { cardId: b.cardId });
      return [already || slot < 0 ? inspect('equip', 'cards-decks', 'Inspect the equipped cards', 'Your deck already contains an investment. Inspect its saved arrangement; practice never replaces a full deck.', dest, { cardId: b.cardId, deckId: b.deckId, slot: Math.max(0, deck.slots.indexOf(b.cardId)), mastered: true }) : action('equip', 'card-equip', 'Place ' + card.name, 'Choose the highlighted saved deck slot. Only the active deck contributes effects.', { type: 'card-equip', id: b.cardId, deckId: b.deckId, slot }, dest, { cardId: b.cardId, deckId: b.deckId, slot }),
        owned.rank > 1 ? inspect('fuse', 'card-detail', 'Inspect the fused rank', 'You already fused this card. Its rarity stays fixed and all decks share the improved rank.', dest, { cardId: b.cardId, mastered: true }) : action('fuse', 'card-fuse', 'Fuse a real duplicate', 'Confirm a guaranteed fusion. The guild supplies the practice copies; your saved duplicates remain available.', { type: 'card-fuse', id: b.cardId }, dest, { cardId: b.cardId }, { supply: 'copies', suppliesText: 'Guild supplies exactly two practice copies, consumed by this fusion.' }),
        state.collection.activeDeck === b.otherDeckId ? inspect('deck', 'cards-decks', 'Inspect the active alternate deck', 'This deck is already active. Review its saved arrangement.', destination('cards'), { deckId: b.otherDeckId, mastered: true }) : action('deck', 'deck-select', 'Activate another saved deck', 'Switch to the highlighted deck. Different decks keep independent card arrangements.', { type: 'deck-select', id: b.otherDeckId }, destination('cards'), { deckId: b.otherDeckId }),
        another || otherSlot < 0 ? inspect('second-deck', 'cards-decks', 'Inspect this deck’s cards', 'The same owned card can appear once in each saved deck. Your existing arrangement is preserved.', dest, { cardId: b.cardId, deckId: b.otherDeckId, slot: Math.max(0, other.slots.indexOf(b.cardId)), mastered: true }) : action('second-deck', 'card-equip', 'Build the second deck', 'Put the same owned card in this deck. Cards are not consumed by equipping.', { type: 'card-equip', id: b.cardId, deckId: b.otherDeckId, slot: otherSlot }, dest, { cardId: b.cardId, deckId: b.otherDeckId, slot: otherSlot }),
        state.collection.activeDeck === b.returnDeckId ? inspect('return-deck', 'cards-decks', 'Inspect your original active deck', 'Your original deck is already active. Both saved arrangements remain available.', destination('cards'), { deckId: b.returnDeckId, mastered: true }) : action('return-deck', 'deck-select', 'Return to your original deck', 'Restore your original active deck. Both saved arrangements remain available.', { type: 'deck-select', id: b.returnDeckId }, destination('cards'), { deckId: b.returnDeckId })];
    }
    if (id === 'equipment') {
      const def = Collection.Content.GEAR.find(d => d.id === b.itemId), item = state.collection.gear[b.itemId], dest = destination('equipment', { itemId: b.itemId });
      return [state.collection.equipped[def.slot] === def.id ? inspect('equip', 'equipment-slots', 'Inspect your equipped item', 'This item already occupies its correct slot. Open it to inspect its real bonuses.', dest, { itemId: def.id, slot: def.slot, mastered: true }) : action('equip', 'gear-equip', 'Equip ' + def.name, 'Use Equip on this owned item. Its bonuses become active in the matching character slot.', { type: 'gear-equip', id: def.id, slot: def.slot }, dest, { itemId: def.id, slot: def.slot }),
        Collection.used(item) >= 6 ? inspect('scroll', 'gear-detail', 'Inspect the enhanced item', 'All six attempt slots are already used. Review the successful points; practice never forces a destructive reforge.', dest, { itemId: def.id, mastered: true }) : action('scroll', 'gear-scroll', 'Use a guaranteed Steady Scroll', 'Confirm the supplied practice scroll on this real item. It adds one point and uses one attempt slot.', { type: 'gear-scroll', id: def.id, scrollId: 'steady' }, dest, { itemId: def.id, scrollId: 'steady' }, { supply: 'scroll', suppliesText: 'Guild supplies one Steady Scroll for this attempt. Your saved scrolls are not spent.' }),
        inspect('result', 'gear-detail', 'Check the actual improvement', 'Open the enhanced item. Compare its points and remaining slots. Riskier scrolls can fail without destroying the item.', dest, { itemId: def.id })];
    }
    let dest = destination('guild'), data = { section: id }, a = null, target = 'guild-action', explanation = 'Use the real control when you are ready. Its normal effect remains in your guild.', supply = null;
    const select = list => availableRows(list).find(row => row.action && !row.disabled && !row.selected && !row.maxed) || availableRows(list).find(row => row.action && !row.selected && !row.owned && !row.maxed);
    if (id === 'tiers') { const r = rows(c.upgradeTiers?.ready)[0]; a = r?.unlockAction; dest = r?.areaId ? destination('expedition', { areaId: r.areaId }) : destination('upgrades'); target = 'tier-unlock'; data = { tierId: r?.id }; }
    if (id === 'expansion') { a = !c.expedition?.next?.disabled ? c.expedition.next.action : null; dest = destination('expedition', { areaId: selected }); target = 'area-expand'; explanation = 'Choose the real next expansion. Existing areas and their investments keep working.'; }
    if (['plans', 'configuration', 'specialization'].includes(id)) { const source = id === 'plans' ? rows(c.expedition?.choices).flatMap(g => g.options || [g]) : id === 'specialization' ? rows(c.expedition?.specializations) : rows(c.expedition?.configurations).flatMap(g => rows(g.slotOptions).flatMap(slot => rows(slot.options).map(option => ({ ...option, selected: option.id === slot.selected })))); const r = select(source); a = r?.action; dest = destination('expedition', { areaId: selected }); target = id === 'plans' ? 'area-plan-option' : id === 'specialization' ? 'area-specialization' : 'area-configuration'; data = { areaId: selected, choiceId: r?.id }; }
    if (id === 'bulk') {
      const intendedCount = x?.intentions.bulk?.count;
      const mode = rows(c.expedition?.batch?.options).find(row => row.unlocked && row.count > 1 && (!intendedCount || row.count === intendedCount)), count = mode?.count, row = rows(c.expedition?.cards).find(row => !row.maxed && row.rank + (count || 0) <= row.maxRank);
      dest = destination('expedition', { areaId: selected });
      return [state.expedition.batch === count ? inspect('quantity', 'batch-select', 'Inspect the selected quantity', 'This earned batch is already selected. Inspect its exact quantity before purchasing.', dest, { count, mastered: true }) : action('quantity', 'batch-select', 'Choose an exact quantity', 'Select the earned batch size. The price covers every rank; purchases never silently buy a partial batch.', mode?.action, dest, { count }), action('purchase', 'area-buy', 'Buy the quoted batch', 'Buy this exact batch. The guild supplies this first lesson purchase; future batches use your resources.', row && count ? { type: 'expedition-buy', areaId: selected, id: row.trackId || row.action.id, count } : null, dest, { areaId: selected, trackId: row?.trackId, count }, { supply: 'cost', suppliesText: 'Guild supplies this exact practice batch once.' })];
    }
    if (['guild-upgrades', 'projects'].includes(id)) { const r = select(rows(c.globalUpgrades).filter(row => id === 'projects' ? ['expedition-development', 'project', 'research'].includes(row.action?.type) : row.action?.type === 'buy')); a = r?.action; dest = destination('upgrades'); target = 'catalog-buy'; data = { catalogId: r?.id, actionId: a?.id }; }
    if (id === 'techniques' || id === 'technique-config') {
      const skills = rows(c.areaSkills?.items || c.areaSkills?.rows || c.globalUpgrades).filter(row => row.skillId);
      const r = id === 'techniques'
        ? skills.find(row => row.state === 'learned' && row.rank < row.maxRank)
        : skills.find(row => rows(row.configuration?.options || row.options).some(option => option.action && !option.disabled && !option.selected));
      const option = id === 'technique-config' && rows(r?.configuration?.options || r?.options).find(item => item.action && !item.disabled && !item.selected);
      a = id === 'techniques' && r ? { type: 'area-skill-buy', id: r.skillId, count: 1 } : option?.action;
      dest = destination('upgrades', { areaId: r?.areaId }); target = id === 'techniques' ? 'catalog-buy' : 'technique-config';
      data = { catalogId: r?.catalogId || r?.id, skillId: r?.skillId, areaId: r?.areaId, value: option?.value ?? option?.id };
      explanation = id === 'techniques' ? 'Inspect this earned technique, then use its real upgrade button. This lesson supplies one practice rank; future ranks use the displayed resources.' : 'Choose the highlighted mode. Its actual effect changes this area’s operation, and you can change modes again later.';
    }
    if (id === 'focus') { a = select(c.expedition?.focus?.actions)?.action; dest = destination('expedition', { areaId: selected }); target = 'focus-action'; explanation = 'Use one guild-supplied practice charge on this operation. Your regular charges stay available. Production still needs its normal inputs.'; }
    if (id === 'automation') { a = select(c.automations)?.action; if (state.lifetime.refits && state.expedition.version === 3) a = { type: 'expedition-automation', enabled: !state.expedition.automation.enabled, priority: state.expedition.automation.priority, dispatch: state.expedition.automation.dispatch }; data.section = 'automation'; }
    if (id === 'reserves') { a = { type: 'plan-reserve', id: 'ore', amount: N.cmp(state.guild.plan.reserves.ore, 10) ? '10' : '20' }; data.resourceId = 'ore'; data.amount = a.amount; data.section = 'planning'; explanation = 'Enter ' + a.amount + ' for the ore reserve and Save. Automatic conversions and purchases protect it; you can edit it again.'; }
    if (id === 'planner') { const r = rows(c.globalUpgrades).find(row => ['buy', 'research', 'project', 'expedition-development'].includes(row.action?.type) && !row.owned && !row.maxed); a = r ? { type: 'plan-goal', action: { type: r.action.type, id: r.action.id } } : null; data.section = 'planning'; }
    if (id === 'playbooks') { const slot = [0, 1, 2].find(id => !state.guild.loadouts.some(row => row.id === id)); a = slot === undefined ? null : { type: 'loadout-save', id: slot, name: 'Plan ' + (slot + 1) }; data.section = 'planning'; if (slot === undefined) return [inspect('inspect', 'guild-action', 'Inspect a saved playbook', 'All three playbooks already exist. Open their saved settings without replacing one.', destination('guild', { section: 'planning' }), data), inspect('practice', 'playbook-details', 'Review your saved configuration', 'The existing playbook proves you have used this feature. No overwrite is required.', destination('guild', { section: 'planning' }), data)]; }
    if (id === 'crew') { a = select(c.specialists)?.action; data.section = 'crew'; }
    if (id === 'companions') { a = select(c.companions)?.action; data.section = 'crew'; }
    if (id === 'relics') { const r = select(c.luck?.relics); a = r?.action || (state.luck.owned.find(key => key !== state.luck.active) ? { type: 'relic-equip', id: state.luck.owned.find(key => key !== state.luck.active) } : null); data.section = 'discoveries'; }
    if (id === 'meals') { a = select(rows(c.recipes).filter(row => row.action?.id?.startsWith('meal-') && row.action.id !== 'meal-none'))?.action; data.section = 'kitchen'; }
    if (id === 'supply') { a = select(c.expedition?.supplyChoices)?.action; data.section = 'expedition'; }
    if (id === 'kits') { a = state.luck.active === 'living-crucible' ? select(c.luck?.kits)?.action || (state.luck.kit.prepared && !state.luck.kit.active ? { type: 'kit-use' } : null) : null; data.section = 'discoveries'; }
    if (['card-archive', 'card-craft'].includes(id)) { const key = Object.keys(state.collection.cards)[0]; a = key ? { type: id === 'card-archive' ? 'card-recycle' : 'card-craft', id: key } : null; dest = destination('cards', { cardId: key }); target = id === 'card-archive' ? 'card-recycle' : 'card-craft'; data = { cardId: key }; }
    if (['gear-craft', 'gear-repair'].includes(id)) { const item = id === 'gear-craft' ? rows(collection.equipment?.items).find(r => !r.owned && Collection.areaOpen(state, r.area)) : rows(collection.equipment?.items).find(r => r.failedSlots > 0 || state.collection.gear[r.id]?.failed > 0); a = item ? { type: id === 'gear-craft' ? 'gear-forge' : 'gear-scroll', id: item.id, ...(id === 'gear-repair' ? { scrollId: 'restoration' } : {}) } : null; dest = destination('equipment', { itemId: item?.id }); target = id === 'gear-craft' ? 'gear-forge' : 'gear-scroll'; data = { itemId: item?.id, ...(id === 'gear-repair' ? { scrollId: 'restoration' } : {}) }; }
    if (['refit', 'charter', 'shop', 'caravan', 'gear-reforge'].includes(id)) {
      dest = destination(id === 'gear-reforge' ? 'equipment' : 'guild', { section: id }); target = id + '-review'; data = { section: id };
      if (id === 'gear-reforge') { data.itemId = Object.keys(state.collection.gear).find(key => Collection.used(state.collection.gear[key]) > 0); dest.itemId = data.itemId; }
      return [inspect('inspect', target, TITLES[id], 'Review the current effects, costs and what remains. Opening this preview commits nothing.', dest, data), inspect('practice', target + '-details', 'Inspect the confirmation details', 'Read the actual current quote and its consequences. Only your separate explicit confirmation can reset, reforge, spend currency or request an ad.', dest, data)];
    }
    const intended = x?.intentions[id];
    if (intended) { a = clone(intended); data.actionId = a.id; if(id === 'tiers') data.tierId = String(a.id).replace(/^ready:/, ''); if (['guild-upgrades', 'projects'].includes(id)) { const row = rows(c.globalUpgrades).find(row => row.action?.type === a.type && row.action?.id === a.id); if (row) data.catalogId = row.id; else delete data.catalogId; } if (id === 'technique-config') { const row = rows(c.areaSkills?.items || c.areaSkills?.rows || c.globalUpgrades).find(item => item.skillId === a.id); data.skillId = a.id; data.value = a.value; data.catalogId = row?.catalogId || row?.id || 'skill:' + a.id; data.areaId = row?.areaId; dest = destination('upgrades', { areaId: row?.areaId }); } if (['plans', 'specialization', 'configuration'].includes(id)) { data.choiceId = a.id; if (a.kind) data.kind = a.kind; if (a.slot !== undefined) data.slot = a.slot; } if (a.areaId) { data.areaId = a.areaId; dest = destination('expedition', { areaId: a.areaId }); } if (a.type.startsWith('card-')) { data.cardId = a.id; dest = destination('cards', { cardId: a.id }); } if (a.type.startsWith('gear-')) { data.itemId = a.id; dest = destination('equipment', { itemId: a.id }); } if (id === 'reserves') { data.resourceId = a.id; data.amount = a.amount; explanation = 'Enter ' + a.amount + ' as the ' + a.id + ' reserve and Save. You can edit it again.'; } }
    supply = 'cost';
    return [inspect('inspect', target, TITLES[id], explanation, { ...dest, ...data }, data), action('practice', target, TITLES[id], explanation + ' The guild supplies the first practice cost; the real result remains.', a, { ...dest, ...data }, data, { supply, suppliesText: 'Guild supplies this first practice action’s exact inputs once.' })];
  }
  function token(state, id, step) {
    const x = ledger(state);
    return [state.createdAt, state.run.id, state.expedition.revision || 0, state.collection.revision, id, x.progress[id], step.id, actionKey(step.requiredAction), JSON.stringify(step.targetData)].join('|');
  }
  const helpToken = (state, id) => [state.createdAt, state.run.id, 'optional-help', id].join('|');
  const mandatory = (state, id) => PRIMARY.includes(id) || !!ledger(state)?.intentions[id];
  function pendingCurrency(state, id, supplied) {
    const x = ledger(state), all = steps(state, id, supplied), current = all[x.progress[id]];
    // Explain the next real price before opening its comparison.
    const step = current?.mode === 'action' ? current : all.slice(x.progress[id]).find(row => row.mode === 'action');
    if (!step?.requiredAction || !costProvider) return null;
    const costs = costProvider(state, step.requiredAction) || {};
    const missing = Object.keys(costs).find(key => own(CURRENCIES, key) && N.cmp(N.from(costs[key]), 0) > 0 && !x.currencyRead?.includes(key));
    return missing ? { ...currencyInfo(missing), coverageText: step.supply ? 'The guild supplies this first practice purchase. The shown price is its usual cost; your saved materials remain unchanged.' : 'This action uses your saved resources at the displayed price.', cost: N.from(costs[missing]), supplied: !!step.supply } : null;
  }
  function currencyToken(state, id, info, step) {
    return token(state, id, step || steps(state, id)[ledger(state).progress[id]]) + '|currency|' + info.id;
  }
  function guardAction(state, action) {
    const x = ledger(state); if (!x?.active || String(action.type).startsWith('onboarding-') && action.type !== 'onboarding-open') return null;
    const step = steps(state, x.active).slice(x.progress[x.active]).find(row => row.requiredAction);
    if (step?.requiredAction && actionKey(normalizedAction(state, step.requiredAction)) === actionKey(normalizedAction(state, action)) && pendingCurrency(state, x.active)) return { ok: false, message: 'Read the currency explanation before this lesson action.' };
    return null;
  }
  function descriptor(state, id, supplied) {
    const x = ledger(state), all = steps(state, id, supplied), progress = x?.progress[id] || 0;
    const list = all.map(step => ({ ...step, targetData: Object.fromEntries(Object.entries(step.targetData || {}).filter(([, v]) => v !== undefined)), completed: x?.proofs.includes(id + ':' + step.id) || false }));
    const optional = OPTIONAL.includes(id), reward = optional ? x?.rewards.includes(id) ? 0 : 8 : state.onboarding.rewardClaims.includes(id) ? 0 : ({ greenway: 12, quarry: 60, watchtower: 120, workshop: 200, ruins: 360, harbor: 600 })[id] || 0;
    return { id, areaId: AREAS.includes(id) ? id : null, title: TITLES[id], mandatory: mandatory(state, id), optional, available: !list.some(step => step.mode === 'wait'), reason: list.some(step => step.mode === 'wait') ? 'Return when this operation has an available action.' : '', complete: progress >= list.length, progress, steps: list, visitAction: { type: 'onboarding-visit', id }, ...(optional && !x?.helpRewards.includes(id) ? { helpOpenAction: { type: 'onboarding-help-open', id, token: helpToken(state, id) } } : {}), helpReward: { amount: optional && !x?.helpRewards.includes(id) ? 4 : 0, claimed: !optional || x?.helpRewards.includes(id), text: optional && !x?.helpRewards.includes(id) ? '+4 coins the first time you open this help.' : 'First-open help reward already saved.' }, rewardPreview: { resource: 'coins', amount: N.from(reward), available: reward > 0, text: reward ? reward + ' coins once after completing this hands-on lesson.' : 'This lesson cannot grant a repeat reward.' } };
  }
  function view(state, supplied) {
    const x = ledger(state); if (!x) return null;
    const c = context(state, supplied), guides = IDS.filter(id => own(x.progress, id) && earned(state, id)).map(id => descriptor(state, id, c));
    const current = guides.find(g => g.id === x.active && !g.complete);
    let active = null;
    if (current) {
      const step = current.steps[current.progress], t = token(state, current.id, step);
      active = { ...clone(current), ...clone(step), id: current.id, guideId: current.id, stepId: step.id, index: current.progress, total: current.steps.length, token: t, leaveAction: { type: 'onboarding-leave', id: current.id } };
      if (step.mode === 'inspect') active.inspectAction = { type: 'onboarding-inspect', id: current.id, stepId: step.id, target: step.target, token: t };
      if (step.mode === 'action' && step.requiredAction) active.practiceAction = { type: 'onboarding-perform', id: current.id, stepId: step.id, token: t, action: clone(step.requiredAction) };
      delete active.helpOpenAction;
      if (current.optional && !x.intentions[current.id] && !x.helpRewards.includes(current.id)) active.helpOpenAction = clone(current.helpOpenAction);
      active.canLeave = !mandatory(state, current.id);
      if (!active.canLeave) delete active.leaveAction;
      active.costCoverage = step.supply ? 'guild' : 'wallet';
      active.suppliedCost = step.supply && step.requiredAction && costProvider ? costProvider(state, step.requiredAction) : null;
      active.walletSpend = step.supply ? {} : step.requiredAction && costProvider ? costProvider(state, step.requiredAction) : {};
      active.quotedCost = step.requiredAction && costProvider ? costProvider(state, step.requiredAction) : {};
      const info = pendingCurrency(state, current.id, c);
      if (info) {
        active.mode = 'currency'; active.stepId = 'currency:' + info.id; active.target = 'currency-info'; active.currencyInfo = info;
        active.heading = info.name; active.body = info.purpose + ' ' + info.earnedFrom + ' ' + info.coverageText;
        active.token = currencyToken(state, current.id, info, step);
        active.action = active.ackAction = { type: 'onboarding-currency-ack', id: current.id, currencyId: info.id, token: active.token };
        delete active.practiceAction; delete active.inspectAction;
      }
    }
    return { guides, active, triggers: guides.filter(g => !g.complete && g.available && TRIGGERS[g.id]).flatMap(g => ['crew', 'companions'].includes(g.id) ? [{ id: g.id, actionTypes: TRIGGERS[g.id].filter(t => t !== 'recruit'), visitAction: g.visitAction }, { id: g.id, actionTypes: ['recruit'], match: { kind: g.id === 'crew' ? 'specialist' : 'companion' }, visitAction: g.visitAction }] : [{ id: g.id, actionTypes: TRIGGERS[g.id], ...(g.id === 'gear-repair' ? { match: { scrollId: 'restoration' } } : {}), visitAction: g.visitAction }]), helpQueue: guides.filter(g => g.optional && !g.complete && g.available).slice(0, 3) };
  }
  function completeStep(state, id, step, supply = false) {
    const x = ledger(state), receipt = id + ':' + step.id;
    if (x.proofs.includes(receipt)) return { ok: false, message: 'This lesson action was already saved.' };
    x.proofs.push(receipt); if (supply) x.supplies.push(receipt); x.progress[id] += 1;
    if (x.progress[id] < KEYS[id].length) return { ok: true, lessonStepComplete: { id, stepId: step.id }, message: 'Action saved. The next lesson step is ready.' };
    let coins = 0;
    if (OPTIONAL.includes(id)) { if (!x.rewards.includes(id)) { coins = 8; x.rewards.push(id); } }
    else { if (!state.onboarding.rewardClaims.includes(id)) coins = ({ greenway: 12, quarry: 60, watchtower: 120, workshop: 200, ruins: 360, harbor: 600 })[id] || 0; if (!state.onboarding.rewardClaims.includes(id)) state.onboarding.rewardClaims.push(id); state.onboarding.progress[id] = 3; }
    if (coins) state.resources.coins = N.add(state.resources.coins, coins);
    x.active = null;
    return { ok: true, completedGuide: id, reward: { coins: N.from(coins) }, message: TITLES[id] + ' lesson complete.' + (coins ? ' +' + coins + ' coins.' : '') };
  }
  function act(state, action) {
    const x = ledger(state); if (!x) return { ok: false, message: 'Reload this guild before practicing.' };
    if (action.type === 'onboarding-currency-ack') {
      if (!exact(action, ['type', 'id', 'currencyId', 'token']) || x.active !== action.id) return { ok: false, message: 'Open the current currency explanation.' };
      const info = pendingCurrency(state, action.id);
      if (!info || info.id !== action.currencyId || currencyToken(state, action.id, info) !== action.token) return { ok: false, message: 'This currency explanation changed or was already saved.' };
      x.currencyRead.push(info.id);
      return { ok: true, message: info.name + ' explained.' };
    }
    if (action.type === 'onboarding-help-open') {
      if (!exact(action, ['type', 'id', 'token']) || !OPTIONAL.includes(action.id) || !earned(state, action.id) || x.active === action.id && x.intentions[action.id] || helpToken(state, action.id) !== action.token || x.helpRewards.includes(action.id)) return { ok: false, message: 'This first-open help reward is unavailable or already saved.' };
      x.helpRewards.push(action.id); state.resources.coins = N.add(state.resources.coins, 4);
      return { ok: true, reward: { coins: N.from(4) }, message: 'Helpful discovery · +4 coins, once.' };
    }
    if (action.type === 'onboarding-visit') {
      if (!(exact(action, ['type', 'id']) || exact(action, ['type', 'id', 'intendedAction']) && validIntention(action.id, action.intendedAction)) || !IDS.includes(action.id) || !earned(state, action.id) || !own(x.progress, action.id) || x.progress[action.id] >= KEYS[action.id].length || x.active && x.active !== action.id) return { ok: false, message: 'Open an earned unfinished lesson.' };
      if (action.id === 'plans' && !steps(state, 'plans').some(step => step.requiredAction && step.mode === 'action')) return { ok: false, message: 'Working plans are taught when another plan is available.' };
      if (action.id === 'technique-config' && action.intendedAction) {
        const skills = context(state).areaSkills?.items || [];
        const row = skills.find(item => item.skillId === action.intendedAction.id);
        const option = rows(row?.configuration?.options || row?.options).find(item => item.action && !item.disabled && !item.selected && actionKey(item.action) === actionKey(cleanAction(action.intendedAction)));
        if (!option) return { ok: false, message: 'Choose a different earned technique mode for this lesson.' };
      }
      if (action.intendedAction) x.intentions[action.id] = cleanAction(action.intendedAction);
      else if (x.active !== action.id) delete x.intentions[action.id];
      if (!x.bindings[action.id]) x.bindings[action.id] = bind(state, action.id, context(state));
      x.active = action.id; state.onboarding.active = null;
      return { ok: true, message: 'Hands-on lesson opened.' };
    }
    if (action.type === 'onboarding-leave') {
      if (!exact(action, ['type', 'id']) || x.active !== action.id) return { ok: false, message: 'That lesson is not open.' };
      if (mandatory(state, action.id)) return { ok: false, message: 'Complete this first-use lesson to continue. Settings and save recovery remain available.' };
      x.active = null; return { ok: true, message: 'Lesson progress saved.' };
    }
    if (action.type === 'onboarding-next') return { ok: false, message: 'Use the highlighted game control to complete this action.' };
    if (action.type === 'onboarding-inspect') {
      if (!exact(action, ['type', 'id', 'stepId', 'target', 'token']) || x.active !== action.id) return { ok: false, message: 'Open the current lesson’s actual control.' };
      if (pendingCurrency(state, action.id)) return { ok: false, message: 'Read the currency explanation first.' };
      const step = steps(state, action.id)[x.progress[action.id]];
      if (!step || step.mode !== 'inspect' || step.id !== action.stepId || step.target !== action.target || token(state, action.id, step) !== action.token) return { ok: false, message: 'This lesson target changed. Open its current control.' };
      return completeStep(state, action.id, step);
    }
    return { ok: false, message: 'Choose a valid hands-on lesson action.' };
  }
  function prepare(state, action) {
    const x = ledger(state);
    if (!x || !exact(action, ['type', 'id', 'stepId', 'token', 'action']) || x.active !== action.id || !object(action.action)) return { ok: false, message: 'Open the current practice action first.' };
    if (pendingCurrency(state, action.id)) return { ok: false, message: 'Read the currency explanation first.' };
    const step = steps(state, action.id)[x.progress[action.id]];
    if (!step || step.mode !== 'action' || step.id !== action.stepId || actionKey(action.action) !== actionKey(step.requiredAction) || token(state, action.id, step) !== action.token || x.proofs.includes(action.id + ':' + step.id)) return { ok: false, message: 'The practice action changed. Review the current lesson.' };
    return { ok: true, step, action: clone(step.requiredAction), supply: step.supply || null };
  }
  function observe(state, action, before) {
    const x = ledger(state); if (!x?.active || !before || before.id !== x.active || before.progress !== x.progress[x.active]) return null;
    const step = before.step;
    if (step.mode !== 'action' || actionKey(normalizedAction(state, action)) !== actionKey(normalizedAction(state, step.requiredAction))) return null;
    if (before.fingerprint === fingerprint(state)) return null;
    return completeStep(state, x.active, step);
  }
  function fingerprint(state) { return JSON.stringify(state, (key, value) => ['onboarding', 'trailDeliveries', 'revision', 'sequence', 'recent', 'events', 'seen', 'seenSequence', 'clock'].includes(key) ? undefined : value); }
  function validIntention(id, action) { return own(TRIGGERS, id) && object(action) && TRIGGERS[id].includes(action.type) && !(id === 'gear-repair' && action.scrollId !== 'restoration') && !(action.type === 'onboarding-open' && !/^ready:/.test(action.id)) && JSON.stringify(action).length <= 500 && Object.keys(action).every(key => !['__proto__', 'constructor', 'prototype'].includes(key)) && Object.values(action).every(value => value === null || ['string', 'boolean'].includes(typeof value) || typeof value === 'number' && Number.isFinite(value) || object(value) && Object.keys(value).every(key => ['type', 'id'].includes(key)) && Object.values(value).every(v => typeof v === 'string')); }
  function capture(state, action) {
    const x = ledger(state); if (!x?.active || String(action.type).startsWith('onboarding-') && action.type !== 'onboarding-open') return null;
    const step = steps(state, x.active)[x.progress[x.active]];
    return step?.mode === 'action' ? { id: x.active, progress: x.progress[x.active], step, fingerprint: fingerprint(state) } : null;
  }
  function validate(x, state) {
    try {
      if (!exact(x, ['version', 'progress', 'active', 'bindings', 'intentions', 'proofs', 'supplies', 'rewards', 'helpRewards'].concat(own(x, 'currencyRead') ? ['currencyRead'] : [])) || x.version !== 1 || !object(x.progress) || !object(x.bindings) || !object(x.intentions) || Object.entries(x.intentions).some(([id, action]) => !own(x.progress, id) || !validIntention(id, action))) return false;
      if (Object.entries(x.progress).some(([id, p]) => !IDS.includes(id) || !Number.isSafeInteger(p) || p < 0 || p > KEYS[id].length)) return false;
      if (x.active !== null && (typeof x.active !== 'string' || !own(x.progress, x.active) || x.progress[x.active] === KEYS[x.active].length)) return false;
      if (Object.entries(x.bindings).some(([id, b]) => !own(x.progress, id) || !object(b) || Object.entries(b).some(([k, v]) => !['areaId', 'trackId', 'cardId', 'deckId', 'returnDeckId', 'otherDeckId', 'itemId'].includes(k) || typeof v !== 'string' || v.length > 80 || !/^[a-z0-9-]+$/.test(v)))) return false;
      for (const key of ['proofs', 'supplies', 'rewards', 'helpRewards']) if (!Array.isArray(x[key]) || x[key].length > 150 || new Set(x[key]).size !== x[key].length || x[key].some(v => typeof v !== 'string')) return false;
      const expected = Object.entries(x.progress).flatMap(([id, p]) => KEYS[id].slice(0, p).map(step => id + ':' + step));
      if (expected.length !== x.proofs.length || expected.some(key => !x.proofs.includes(key))) return false;
      if (x.supplies.some(key => !x.proofs.includes(key))) return false;
      if (x.helpRewards.some(id => !OPTIONAL.includes(id))) return false;
      if (own(x, 'currencyRead') && (!Array.isArray(x.currencyRead) || new Set(x.currencyRead).size !== x.currencyRead.length || x.currencyRead.some(id => typeof id !== 'string' || !own(CURRENCIES, id)))) return false;
      if (x.rewards.some(id => !OPTIONAL.includes(id) || x.progress[id] !== KEYS[id].length) || OPTIONAL.some(id => x.progress[id] === KEYS[id].length && !x.rewards.includes(id))) return false;
      for (const [id, b] of Object.entries(x.bindings)) {
        if (b.areaId && !AREAS.includes(b.areaId) || b.trackId && !D.AREAS.find(a => a.id === b.areaId)?.tracks.some(t => t.id === b.trackId) && !['lift', 'beacon'].includes(b.trackId) || b.cardId && !Collection.Content.CARDS.some(c => c.id === b.cardId) || b.itemId && !Collection.Content.GEAR.some(g => g.id === b.itemId)) return false;
        if (['deckId', 'returnDeckId', 'otherDeckId'].some(k => b[k] && !state.collection.decks.some(d => d.id === b[k]))) return false;
      }
      return true;
    } catch (_) { return false; }
  }
  function mergeReceipts(incoming, current) {
    if (incoming.createdAt !== current.createdAt || !current.onboarding?.practice) return incoming;
    sync(incoming);
    const a = ledger(incoming), b = ledger(current);
    for (const id of IDS) if ((b.progress[id] || 0) > (a.progress[id] || 0)) {
      a.progress[id] = b.progress[id];
      if (b.bindings[id]) a.bindings[id] = clone(b.bindings[id]);
      if (PRIMARY.includes(id) && earned(incoming, id) && a.progress[id] === KEYS[id].length) { incoming.onboarding.progress[id] = 3; if (!incoming.onboarding.rewardClaims.includes(id)) incoming.onboarding.rewardClaims.push(id); }
    }
    a.proofs = Object.entries(a.progress).flatMap(([id, p]) => KEYS[id].slice(0, p).map(step => id + ':' + step));
    a.supplies = [...new Set(a.supplies.concat(b.supplies))].filter(key => a.proofs.includes(key));
    a.rewards = OPTIONAL.filter(id => a.progress[id] === KEYS[id].length);
    a.helpRewards = [...new Set(a.helpRewards.concat(b.helpRewards))];
    a.currencyRead = [...new Set((a.currencyRead || []).concat(b.currencyRead || []))];
    a.active = null;
    sync(incoming);
    return incoming;
  }
  return { initial, sync, view, act, prepare, capture, observe, validate, mergeReceipts, setContextProvider, setCostProvider, guardAction, currencyInfo, completeStep, IDS, KEYS };
});
