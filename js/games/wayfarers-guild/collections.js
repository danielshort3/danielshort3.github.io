(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./numbers.js') : root.WayfarersNumbers, common ? require('./collection-content.js') : root.WayfarersCollectionContent);
  if (common) module.exports = api;
  if (root) root.WayfarersCollections = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N, D) {
  'use strict';
  const EPS = 1e-6;
  const clone = value => JSON.parse(JSON.stringify(value));
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const exact = (value, keys) => object(value) && Object.keys(value).length === keys.length && keys.every(key => Object.prototype.hasOwnProperty.call(value, key));
  const integer = (value, max = 1e9) => Number.isSafeInteger(value) && value >= 0 && value <= max;
  const card = id => D.CARDS.find(value => value.id === id);
  const gear = id => D.GEAR.find(value => value.id === id);
  const scroll = id => D.SCROLLS.find(value => value.id === id);
  const ownership = new WeakMap();
  let rateProvider = null;
  const setRateProvider = fn => { rateProvider = fn; };
  const setEntitlements = (state, values) => ownership.set(state, new Set(values));
  function random(x, key) { let v = x[key]; v ^= v << 13; v ^= v >>> 17; v ^= v << 5; x[key] = v >>> 0; return x[key] / 4294967296; }
  function create(time) {
    const seed = (Math.floor(time) ^ Math.floor(time / 4294967296) ^ 0x51b43c29) >>> 0 || 1;
    return { version: 1, cardsUnlocked: false, equipmentUnlocked: false, cards: {}, gear: {},
      decks: ['Travel', 'Industry', 'Discovery'].map((name, i) => ({ id: 'deck-' + (i + 1), name, slots: [null, null, null, null] })), activeDeck: 'deck-1',
      equipped: Object.fromEntries(D.SLOTS.map(id => [id, null])), scrolls: Object.fromEntries(D.SCROLLS.map(def => [def.id, 0])), ink: 0,
      lootRng: seed, scrollRng: (seed ^ 0xf3987a61) >>> 0 || 1, cardRemainingMs: 900000, scrollRemainingMs: 3600000,
      cardEligibleMs: 0, scrollEligibleMs: 0, cardFinds: 0, scrollFinds: 0, sinceEpic: 0, sinceLegendary: 0,
      revision: 0, sequence: 0, seen: 0, slotsAnnounced: 2, recent: [] };
  }
  function areaOpen(state, id) {
    if (state.expedition?.areas) return !!state.expedition.areas[id];
    return Math.max(state.route.index, state.lifetime.highestRoute) >= ({ greenway: 0, quarry: 1, watchtower: 2 }[id] ?? Infinity);
  }
  function slots(state) { return areaOpen(state, 'workshop') ? 4 : areaOpen(state, 'watchtower') ? 3 : 2; }
  function points(item) { return item.successes.steady + item.successes.bold * 2 + item.successes.brilliant * 5; }
  function used(item) { return item.failed + Object.values(item.successes).reduce((a, b) => a + b, 0); }
  function event(x, outcome, title, detail, values = {}) {
    x.sequence += 1;
    x.recent.push({ id: x.sequence, outcome, title, detail, rarity: 'common', consumed: {}, delta: {}, ...values });
    x.recent = x.recent.slice(-24);
  }
  function grantCard(x, id, reason) {
    const definition = card(id), duplicate = !!x.cards[id];
    if (duplicate) x.cards[id].copies += 1;
    else x.cards[id] = { rank: 1, copies: 0 };
    event(x, 'found', definition.name, duplicate ? 'Duplicate retained for guaranteed fusion or Archive Ink.' : reason || 'Add this card to a saved deck to use its effect.', { cardId: id, rarity: definition.rarity, delta: { copies: duplicate ? 1 : 0, owned: duplicate ? 0 : 1 } });
  }
  function syncUnlocks(state) {
    const x = state.collection;
    if (!x?.cardsUnlocked || slots(state) <= x.slotsAnnounced) return;
    x.slotsAnnounced = slots(state);
    event(x, 'found', 'Deck slot ' + x.slotsAnnounced + ' unlocked', 'Every saved deck can now equip ' + x.slotsAnnounced + ' different cards. Choose the new slot in Cards.', { delta: { deckSlots: x.slotsAnnounced } });
    x.revision += 1;
  }
  function nextEvent(state) {
    const x = state.collection;
    if (!x) return Infinity;
    return Math.max(1e-9, Math.min(x.cardsUnlocked ? x.cardRemainingMs / 1000 : Infinity, x.equipmentUnlocked ? x.scrollRemainingMs / 1000 : Infinity));
  }
  function tick(state, seconds) {
    const x = state.collection;
    if (!x) return;
    if (x.cardsUnlocked) { x.cardEligibleMs += seconds * 1000; x.cardRemainingMs -= seconds * 1000; }
    if (x.equipmentUnlocked) { x.scrollEligibleMs += seconds * 1000; x.scrollRemainingMs -= seconds * 1000; }
    while (x.cardsUnlocked && x.cardRemainingMs <= EPS) {
      const pool = D.CARDS.filter(def => areaOpen(state, def.area)), rarities = D.RARITIES.filter(rarity => pool.some(def => def.rarity === rarity));
      let rarity;
      if (x.sinceLegendary >= 79 && rarities.includes('legendary')) rarity = 'legendary';
      else if (x.sinceEpic >= 11 && rarities.includes('epic')) rarity = 'epic';
      else {
        let roll = random(x, 'lootRng') * rarities.reduce((sum, value) => sum + D.WEIGHTS[value], 0);
        rarity = rarities.find(value => { roll -= D.WEIGHTS[value]; return roll < 0; }) || rarities[0];
      }
      const choices = pool.filter(def => def.rarity === rarity);
      grantCard(x, choices[Math.floor(random(x, 'lootRng') * choices.length)].id);
      x.cardFinds += 1;
      x.sinceEpic = ['epic', 'legendary'].includes(rarity) ? 0 : Math.min(12, x.sinceEpic + 1);
      x.sinceLegendary = rarity === 'legendary' ? 0 : Math.min(80, x.sinceLegendary + 1);
      x.cardRemainingMs += Math.floor((2700 + random(x, 'lootRng') * 1800) * 1000);
      x.revision += 1;
    }
    while (x.equipmentUnlocked && x.scrollRemainingMs <= EPS) {
      let roll = random(x, 'lootRng') * 100;
      const found = D.SCROLLS.find(def => { roll -= def.weight; return roll < 0; }) || D.SCROLLS[0];
      x.scrolls[found.id] += 1; x.scrollFinds += 1;
      event(x, 'found', found.name, 'Earned supply cache. Equipment and scrolls survive every reset.', { scrollId: found.id, rarity: found.id === 'brilliant' || found.id === 'restoration' ? 'epic' : found.id === 'bold' ? 'rare' : 'common', delta: { scrolls: 1 } });
      x.scrollRemainingMs += Math.floor((5400 + random(x, 'lootRng') * 3600) * 1000);
      x.revision += 1;
    }
  }
  function modifiers(state) {
    const values = {}, tags = { journey: 0, industry: 0, discovery: 0 }, x = state.collection;
    const add = (effects, strength = 1) => Object.entries(effects).forEach(([id, amount]) => { values[id] = (values[id] || 0) + amount * strength; });
    if (!x) return values;
    if (x.cardsUnlocked) {
      const deck = x.decks.find(def => def.id === x.activeDeck);
      for (const id of deck.slots.slice(0, slots(state))) if (id && x.cards[id]) { const def = card(id); add(def.effects, D.RANK_SCALE[x.cards[id].rank]); tags[def.tag] += 1; }
      const synergy = count => count >= 4 ? .09 : count === 3 ? .07 : count === 2 ? .04 : 0;
      add({ cargo: synergy(tags.journey), oreSaving: synergy(tags.industry), research: synergy(tags.discovery) });
    }
    if (x.equipmentUnlocked) for (const id of Object.values(x.equipped)) if (id && x.gear[id]) { const def = gear(id); add(def.effects); add(def.perPoint, points(x.gear[id])); }
    values.oreSaving = Math.min(.4, values.oreSaving || 0);
    return values;
  }
  function quote(state, action) {
    const x = state.collection, result = { disabled: false, reason: '', cost: {}, token: '' };
    if (!x || !object(action)) return { ...result, disabled: true, reason: 'Choose a collection purchase in a supported save.' };
    const definition = card(action.id), item = definition && x.cards[action.id], equipment = gear(action.id), owned = equipment && x.gear[action.id];
    if (action.type === 'card-fuse') {
      result.costCopies = item && D.FUSION[item.rank] || 0; result.rankAfter = item ? Math.min(5, item.rank + 1) : 1;
      if (!x.cardsUnlocked || !item || item.rank >= 5 || item.copies < result.costCopies) result.reason = !item ? 'Discover this card first.' : item.rank >= 5 ? 'Maximum rank. Exchange duplicates for Archive Ink.' : 'Keep ' + result.costCopies + ' duplicate copies for fusion.';
    } else if (action.type === 'card-recycle') {
      result.costCopies = 1; result.inkGain = definition ? D.INK[definition.rarity] : 0;
      if (!x.cardsUnlocked || !item || !item.copies) result.reason = 'Only loose duplicates can become Archive Ink.';
    } else if (action.type === 'card-craft') {
      result.costInk = definition ? D.INK[definition.rarity] * 10 : 0;
      if (!x.cardsUnlocked || !item || x.ink < result.costInk) result.reason = !item ? 'Discover this card before crafting copies.' : 'Save ' + result.costInk + ' Archive Ink.';
    } else if (action.type === 'gear-forge') {
      result.cost = equipment ? Object.fromEntries(Object.entries(equipment.costs).map(([id, amount]) => [id, N.from(amount)])) : {};
      if (!x.equipmentUnlocked || !equipment || !areaOpen(state, equipment.area) || owned) result.reason = owned ? 'This equipment is already owned.' : 'Discover its area and open Equipment first.';
      else if (Object.entries(result.cost).some(([id, amount]) => N.cmp(state.resources[id], amount) < 0)) result.reason = 'Gather the complete crafting cost.';
    } else if (action.type === 'gear-reforge') {
      result.cost = equipment ? Object.fromEntries(Object.entries(equipment.costs).map(([id, amount]) => [id, N.from(amount * 10)])) : {};
      result.scrollCost = 3; result.scrollId = 'restoration';
      result.pointsLost = owned ? points(owned) : 0; result.slotsAfter = 6;
      if (!x.equipmentUnlocked || !owned || !areaOpen(state, 'workshop')) result.reason = 'Discover the Workshop and own this equipment first.';
      else if (!used(owned)) result.reason = 'This equipment already has six unused slots.';
      else if (x.scrolls.restoration < 3 || Object.entries(result.cost).some(([id, amount]) => N.cmp(state.resources[id], amount) < 0)) result.reason = 'Save three earned Restoration Scrolls and the entire reforging cost.';
    } else if (action.type === 'gear-scroll') {
      const def = scroll(action.scrollId);
      if (!x.equipmentUnlocked || !owned || !def || !x.scrolls[action.scrollId]) result.reason = 'Choose owned equipment and an earned scroll.';
      else if (action.scrollId === 'restoration' ? !owned.failed : used(owned) >= 6) result.reason = action.scrollId === 'restoration' ? 'There is no failed slot to restore.' : 'All six slots are used. Only failed slots can be restored.';
    } else result.reason = 'Choose a collection purchase.';
    result.disabled = !!result.reason;
    result.token = [state.run.id, x.revision, action.type, action.id, action.scrollId || '', JSON.stringify(result.cost)].join(':');
    return result;
  }
  function act(state, action) {
    const x = state.collection;
    if (!x || !object(action)) return { ok: false, message: 'Choose a collection action.' };
    const previousSequence = x.sequence;
    if (action.type === 'collection-unlock') {
      const key = action.kind === 'cards' ? 'cardsUnlocked' : action.kind === 'equipment' ? 'equipmentUnlocked' : null;
      if (!key || x[key] || !areaOpen(state, action.kind === 'cards' ? 'quarry' : 'watchtower')) return { ok: false, message: 'This introduction is not available.' };
      x[key] = true;
      if (action.kind === 'cards') { grantCard(x, 'trail-courier', 'Choose a deck slot to activate this card. Ownership alone gives no bonus.'); grantCard(x, 'quarry-mole'); }
      else { x.gear['trail-boots'] = { successes: { steady: 0, bold: 0, brilliant: 0 }, failed: 0 }; x.scrolls.steady += 2; x.scrolls.bold += 1; x.scrolls.restoration += 1; event(x, 'found', 'Your equipment satchel', 'Trail Boots and four earned scrolls. Equip the boots explicitly; failed scrolls never destroy equipment.', { itemId: 'trail-boots', delta: { owned: 1, scrolls: 4 } }); }
    } else if (action.type === 'collection-ack') {
      if (!integer(action.sequence, x.sequence) || action.sequence < x.seen) return { ok: false, message: 'This discovery acknowledgement is stale.' };
      x.seen = action.sequence; return { ok: true, message: 'Collection discoveries acknowledged.' };
    } else if (action.type === 'deck-select' || action.type === 'deck-name') {
      const deck = x.decks.find(def => def.id === action.id);
      if (!x.cardsUnlocked || !deck || action.type === 'deck-name' && (typeof action.name !== 'string' || !action.name.trim() || action.name.trim().length > 24 || /[\u0000-\u001f]/.test(action.name))) return { ok: false, message: 'Choose a deck and a name of 1–24 characters.' };
      if (action.type === 'deck-select') x.activeDeck = deck.id; else deck.name = action.name.trim();
    } else if (action.type === 'card-equip') {
      const deck = x.decks.find(def => def.id === action.deckId);
      if (!x.cardsUnlocked || !deck || !integer(action.slot, slots(state) - 1) || action.id !== null && (!card(action.id) || !x.cards[action.id] || deck.slots.some((id, slot) => id === action.id && slot !== action.slot))) return { ok: false, message: 'Use an owned card once per deck, in an unlocked slot.' };
      deck.slots[action.slot] = action.id;
    } else if (action.type === 'gear-equip') {
      if (!x.equipmentUnlocked || !D.SLOTS.includes(action.slot) || action.id !== null && (!x.gear[action.id] || gear(action.id)?.slot !== action.slot)) return { ok: false, message: 'Choose owned equipment for this slot.' };
      x.equipped[action.slot] = action.id;
    } else {
      const q = quote(state, action);
      if (q.disabled || action.quote !== q.token) return { ok: false, message: q.reason || 'This collection quote changed. Review the current cost.' };
      if (action.type === 'gear-reforge' && action.confirm !== true) return { ok: false, message: 'Confirm that reforging removes every enhancement point and used slot.' };
      if (action.type === 'card-fuse') {
        const item = x.cards[action.id]; item.copies -= q.costCopies; item.rank += 1;
        event(x, 'fused', card(action.id).name + ' · rank ' + item.rank, 'Guaranteed fusion. Rarity stays the same; every saved deck uses this improved card.', { cardId: action.id, rarity: card(action.id).rarity, consumed: { copies: q.costCopies }, delta: { rank: 1 } });
      } else if (action.type === 'card-recycle') {
        x.cards[action.id].copies -= 1; x.ink += q.inkGain;
        event(x, 'found', 'Archive Ink +' + q.inkGain, 'One loose duplicate became Archive Ink. Your owned card and every saved deck remain intact.', { cardId: action.id, rarity: card(action.id).rarity, consumed: { copies: 1 }, delta: { copies: -1, ink: q.inkGain } });
      }
      else if (action.type === 'card-craft') { x.ink -= q.costInk; grantCard(x, action.id, 'Crafted with Archive Ink.'); }
      else if (action.type === 'gear-forge') { Object.entries(q.cost).forEach(([id, amount]) => { state.resources[id] = N.sub(state.resources[id], amount); }); x.gear[action.id] = { successes: { steady: 0, bold: 0, brilliant: 0 }, failed: 0 }; event(x, 'found', gear(action.id).name, 'Crafted permanently. Choose Equip to activate its bonuses.', { itemId: action.id, rarity: gear(action.id).rarity, delta: { owned: 1 } }); }
      else if (action.type === 'gear-reforge') {
        const item = x.gear[action.id], oldPoints = points(item), oldFailed = item.failed, oldUsed = used(item);
        Object.entries(q.cost).forEach(([id, amount]) => { state.resources[id] = N.sub(state.resources[id], amount); });
        x.scrolls.restoration -= 3;
        item.successes = { steady: 0, bold: 0, brilliant: 0 }; item.failed = 0;
        event(x, 'reforged', gear(action.id).name + ' reforged', 'Removed ' + oldPoints + ' enhancement points. The owned item, equipped slot and base bonuses remain; all six attempt slots are unused again.', { itemId: action.id, consumed: { scrollId: 'restoration', count: 3, ...Object.fromEntries(Object.entries(gear(action.id).costs).map(([id, amount]) => [id, amount * 10])) }, delta: { points: -oldPoints, failedSlots: -oldFailed, freeSlots: oldUsed } });
      }
      else if (action.type === 'gear-scroll') {
        const item = x.gear[action.id], def = scroll(action.scrollId); x.scrolls[def.id] -= 1;
        if (def.id === 'restoration') { item.failed -= 1; event(x, 'restored', 'Failed slot restored', 'One failed attempt slot is usable again. Successful points remain unchanged.', { itemId: action.id, consumed: { scrollId: def.id, count: 1 }, delta: { points: 0, failedSlots: -1, freeSlots: 1 } }); }
        else {
          const success = random(x, 'scrollRng') < def.success;
          if (success) item.successes[def.id] += 1; else item.failed += 1;
          event(x, success ? 'success' : 'failed', success ? gear(action.id).name + ' +' + def.points : 'Scroll did not take', success ? '+' + def.points + ' permanent enhancement points.' : 'Equipment and existing points are safe. An earned Restoration Scroll can recover this failed slot.', { itemId: action.id, rarity: success ? def.id === 'brilliant' ? 'legendary' : 'rare' : 'common', consumed: { scrollId: def.id, count: 1, attemptSlots: 1 }, delta: { points: success ? def.points : 0, failedSlots: success ? 0 : 1, freeSlots: -1 } });
        }
      }
    }
    x.revision += 1;
    return { ok: true, message: 'Collection updated.', collectionEvent: x.sequence > previousSequence ? x.recent.at(-1).id : null };
  }
  function comparison(state, change, baseline) {
    if (!rateProvider) return [];
    const copy = clone(state); change(copy);
    const context = ownership.get(state), before = baseline || rateProvider(state, context), after = rateProvider(copy, context);
    const labels = { travel: 'Expansion travel', trailTravel: 'Trail travel capacity', picks: 'Extraction capacity', haul: 'Hauling capacity', smelt: 'Refining capacity', assembly: 'Assembly capacity', oreInput: 'Workshop ore demand', research: 'Commission research', delving: 'Delving capacity', recovery: 'Recovery capacity', voyage: 'Funded voyage travel', cargo: 'New manifest coin cargo', mapCargo: 'New manifest map cargo', oreCargo: 'New manifest ore cargo', provisionCargo: 'New manifest provision cargo', manifestSupply: 'New manifest provisions cost', manifestReady: 'New manifest affordable now' };
    return Object.keys(before).filter(id => after[id] !== undefined && N.cmp(before[id], after[id]) && Math.abs(N.toNumber(N.div(after[id], N.max(before[id], 1e-12))) - 1) > 1e-9).map(id => ({ metric: 'guild:' + id, label: labels[id] || 'Guild ' + id, currentValue: before[id], nextValue: after[id], current: id === 'manifestReady' ? N.cmp(before[id], 0) ? 'Ready' : 'Needs supplies' : N.format(before[id]), next: id === 'manifestReady' ? N.cmp(after[id], 0) ? 'Ready' : 'Needs supplies' : N.format(after[id]), direction: ['oreInput', 'manifestSupply'].includes(id) ? 'lower' : 'higher', unit: ['cargo', 'mapCargo', 'oreCargo', 'provisionCargo', 'manifestSupply', 'manifestReady'].includes(id) ? '' : '/s' }));
  }
  function effectText(state, effects, strength = 1) {
    const labels = { coins: 'Trail delivery coins', maps: 'maps', travel: 'Trail travel', cargo: 'new voyage cargo', picks: 'extraction capacity', haul: 'hauling capacity', smelt: 'refining capacity', oreYield: 'ore yield', knowledge: 'knowledge', research: 'commission research', assembly: 'assembly capacity', workshopYield: 'manufacturing yield', oreSaving: 'Workshop ore per item', delving: 'delving capacity', recovery: 'recovery capacity', artifacts: 'artifact support', interpretation: 'interpretation capacity', voyage: 'funded voyage travel' };
    if (state.expedition?.version !== 3) {
      const groups = { coins: ['coins', 'cargo'], ore: ['picks', 'haul', 'smelt', 'oreYield'], herbs: ['delving', 'recovery', 'artifacts'], provisions: ['assembly', 'workshopYield', 'oreSaving'], knowledge: ['knowledge', 'research', 'interpretation'], maps: ['maps'], travel: ['travel', 'voyage'] };
      return 'Retained run: ' + Object.entries(groups).map(([id, keys]) => [id, keys.reduce((sum, key) => sum + (effects[key] || 0), 0)]).filter(([, amount]) => amount).map(([id, amount]) => '+' + Number((amount * strength * 100).toFixed(2)) + '% ' + id).join('; ');
    }
    return Object.entries(effects).map(([id, amount]) => (id === 'oreSaving' ? '−' : '+') + Number((amount * strength * 100).toFixed(2)) + '% ' + labels[id]).join('; ');
  }
  function view(state) {
    const x = state.collection, slotCount = slots(state);
    const baseline = rateProvider && rateProvider(state, ownership.get(state));
    const compare = change => comparison(state, change, baseline);
    const activeCards = x.decks.find(deck => deck.id === x.activeDeck).slots.slice(0, slotCount).filter(Boolean).map(card);
    const synergies = ['journey', 'industry', 'discovery'].map(tag => {
      const count = activeCards.filter(def => def.tag === tag).length;
      const amount = count >= 4 ? .09 : count === 3 ? .07 : count === 2 ? .04 : 0;
      const effects = { [tag === 'journey' ? 'cargo' : tag === 'industry' ? 'oreSaving' : 'research']: amount };
      return { id: tag, count, active: count >= 2, nextCount: count < 2 ? 2 : count < 4 ? count + 1 : null, effectText: effectText(state, effects), thresholds: '2 / 3 / 4 matching cards: 4% / 7% / 9%' };
    });
    const availableRarities = D.RARITIES.filter(rarity => D.CARDS.some(def => def.rarity === rarity && areaOpen(state, def.area)));
    const guaranteed = x.sinceLegendary >= 79 && availableRarities.includes('legendary') ? 'legendary' : x.sinceEpic >= 11 && availableRarities.includes('epic') ? 'epic' : null;
    const weightSum = availableRarities.reduce((sum, rarity) => sum + D.WEIGHTS[rarity], 0);
    const odds = Object.fromEntries(D.RARITIES.map(rarity => [rarity, guaranteed ? rarity === guaranteed ? 100 : 0 : availableRarities.includes(rarity) ? D.WEIGHTS[rarity] / weightSum * 100 : 0]));
    const oddsText = guaranteed ? 'Next find: guaranteed ' + guaranteed + ' from the discovered-area pool.' : 'Next find: ' + availableRarities.map(rarity => Number(odds[rarity].toFixed(2)) + '% ' + rarity).join(', ') + '. Only discovered areas contribute cards.';
    const purchase = action => { const q = quote(state, action); return { ...q, action: { ...action, quote: q.token } }; };
    const unlocked = kind => {
      const available = areaOpen(state, kind === 'cards' ? 'quarry' : 'watchtower'), owned = x[kind === 'cards' ? 'cardsUnlocked' : 'equipmentUnlocked'];
      const description = kind === 'cards' ? 'Two starter cards; choose which cards to equip. New finds arrive during online and offline play.' : 'A pair of Trail Boots and earned scrolls. Equip gear yourself; scroll failure never destroys it.';
      return { available, unlocked: owned, disabled: !available || owned, reason: owned ? 'Already introduced.' : available ? '' : 'Discover the ' + (kind === 'cards' ? 'Quarry' : 'Tower') + ' first.', action: { type: 'collection-unlock', kind }, description, starterText: description };
    };
    return { cardsAvailable: areaOpen(state, 'quarry'), cardsUnlocked: x.cardsUnlocked, equipmentAvailable: areaOpen(state, 'watchtower'), equipmentUnlocked: x.equipmentUnlocked, unlocks: { cards: unlocked('cards'), equipment: unlocked('equipment') }, attention: x.sequence > x.seen, ink: x.ink, slotsUnlocked: slotCount, synergies,
      acquisition: { cards: 'First find after 15 eligible minutes, then every 45–75 minutes. Offline time counts.', scrolls: 'First cache after one eligible hour, then every 90–150 minutes. Restoration is scarce (3% of caches).', odds, oddsText, baseWeights: D.WEIGHTS, guaranteed, pity: 'An Epic-or-better by the 12th eligible find; a Legendary by the 80th when that rarity is available.', nextCardSeconds: x.cardRemainingMs / 1000, nextScrollSeconds: x.scrollRemainingMs / 1000 },
      decks: x.decks.map(deck => ({ id: deck.id, name: deck.name, selected: deck.id === x.activeDeck, slots: deck.slots.map((cardId, index) => ({ index, cardId, locked: index >= slotCount, clearAction: { type: 'card-equip', deckId: deck.id, slot: index, id: null } })), selectAction: { type: 'deck-select', id: deck.id }, renameAction: { type: 'deck-name', id: deck.id } })),
      cards: D.CARDS.filter(def => x.cards[def.id] || areaOpen(state, def.area)).map(def => {
        const item = x.cards[def.id], rank = item?.rank || 0, fusion = purchase({ type: 'card-fuse', id: def.id });
        if (item && rank < 5) fusion.impact = compare(next => { next.collection.cards[def.id].rank += 1; });
        fusion.currentEffectText = effectText(state, def.effects, D.RANK_SCALE[rank || 1]);
        fusion.nextEffectText = effectText(state, def.effects, D.RANK_SCALE[Math.min(5, (rank || 1) + 1)]);
        fusion.rarityText = 'Guaranteed rank improvement; ' + def.rarity + ' rarity stays unchanged.';
        return { ...def, owned: !!item, rank, maxRank: 5, copies: item?.copies || 0, effectText: effectText(state, def.effects, D.RANK_SCALE[rank || 1]), passive: false, fusion, recycle: purchase({ type: 'card-recycle', id: def.id }), craft: purchase({ type: 'card-craft', id: def.id }),
          equipActions: x.decks.flatMap(deck => deck.slots.map((id, slot) => {
            const disabled = !item || slot >= slotCount || deck.slots.some((id, index) => id === def.id && index !== slot);
            return { deckId: deck.id, slot, selected: id === def.id, disabled, reason: disabled ? 'Use an owned card once per deck, in an unlocked slot.' : '', context: deck.id === x.activeDeck ? 'Replace active deck slot ' + (slot + 1) : 'Save to ' + deck.name + '; activate this deck to apply its bonuses.', impact: !disabled && deck.id === x.activeDeck ? compare(next => { next.collection.decks.find(d => d.id === deck.id).slots[slot] = def.id; }) : [], action: { type: 'card-equip', deckId: deck.id, slot, id: def.id } };
          })),
          impact: item ? compare(next => { const deck = next.collection.decks.find(d => d.id === next.collection.activeDeck); deck.slots = deck.slots.map(id => id === def.id ? null : id); deck.slots[0] = def.id; }) : [] };
      }),
      equipment: { slots: D.SLOTS.map(id => ({ id, name: { tool: 'Tool', head: 'Head', coat: 'Coat', boots: 'Boots' }[id], itemId: x.equipped[id], clearAction: { type: 'gear-equip', slot: id, id: null } })), scrolls: D.SCROLLS.map(def => ({ ...def, count: x.scrolls[def.id], successPercent: def.success * 100 })), items: D.GEAR.filter(def => x.gear[def.id] || areaOpen(state, def.area)).map(def => {
        const item = x.gear[def.id];
        const effects = Object.fromEntries(Object.keys(def.effects).map(key => [key, def.effects[key] + def.perPoint[key] * (item ? points(item) : 0)]));
        const reforge = { ...purchase({ type: 'gear-reforge', id: def.id, confirm: true }), visible: areaOpen(state, 'workshop') && !!item, currentEffectText: effectText(state, effects), nextEffectText: effectText(state, def.effects), impact: item && used(item) ? compare(next => { next.collection.gear[def.id] = { successes: { steady: 0, bold: 0, brilliant: 0 }, failed: 0 }; }) : [] };
        return { ...def, owned: !!item, equipped: x.equipped[def.slot] === def.id, points: item ? points(item) : 0, slotsUsed: item ? used(item) : 0, failedSlots: item?.failed || 0, slotsMax: 6, effectText: effectText(state, effects), stats: effects, forge: purchase({ type: 'gear-forge', id: def.id }), equipAction: { type: 'gear-equip', slot: def.slot, id: def.id },
          reforge,
          impact: item ? compare(next => { next.collection.equipped[def.slot] = def.id; }) : [],
          scrolls: D.SCROLLS.map(s => ({ ...s, ...purchase({ type: 'gear-scroll', id: def.id, scrollId: s.id }), count: x.scrolls[s.id], successPercent: s.success * 100, pointsAfterSuccess: (item ? points(item) : 0) + s.points, currentEffectText: effectText(state, effects), nextEffectText: effectText(state, Object.fromEntries(Object.keys(effects).map(key => [key, effects[key] + def.perPoint[key] * s.points]))), impact: item && s.id !== 'restoration' && used(item) < 6 ? compare(next => { next.collection.gear[def.id].successes[s.id] += 1; }) : [], successText: s.id === 'restoration' ? 'Restore one failed slot; keep every successful point.' : '+' + s.points + ' enhancement points; consume one attempt slot.', failureText: s.id === 'restoration' ? 'Guaranteed restoration.' : s.success === 1 ? 'Guaranteed success.' : 'Consume the scroll and one attempt slot. Equipment and existing points stay safe.' })) };
      }) }, events: x.recent.filter(entry => entry.id > x.seen).map(entry => ({ ...clone(entry), sequence: entry.id })), ackAction: { type: 'collection-ack', sequence: x.sequence } };
  }
  function validate(x, state) {
    const template = create(state.createdAt);
    if (!exact(x, Object.keys(template)) || x.version !== 1 || typeof x.cardsUnlocked !== 'boolean' || typeof x.equipmentUnlocked !== 'boolean' || x.cardsUnlocked && !areaOpen(state, 'quarry') || x.equipmentUnlocked && !areaOpen(state, 'watchtower')) return false;
    if (!object(x.cards) || Object.entries(x.cards).some(([id, item]) => !card(id) || !areaOpen(state, card(id).area) || !exact(item, ['rank', 'copies']) || !integer(item.rank, 5) || item.rank < 1 || !integer(item.copies)) || !x.cardsUnlocked && Object.keys(x.cards).length || x.cardsUnlocked && (!x.cards['trail-courier'] || !x.cards['quarry-mole'])) return false;
    if (!Array.isArray(x.decks) || x.decks.length !== 3 || x.decks.some((deck, index) => !exact(deck, ['id', 'name', 'slots']) || deck.id !== 'deck-' + (index + 1) || typeof deck.name !== 'string' || !deck.name.trim() || deck.name.length > 24 || /[\u0000-\u001f]/.test(deck.name) || !Array.isArray(deck.slots) || deck.slots.length !== 4 || deck.slots.some((id, slot) => id !== null && (!card(id) || !own(x.cards, id) || slot >= slots(state))) || new Set(deck.slots.filter(Boolean)).size !== deck.slots.filter(Boolean).length) || !x.decks.some(deck => deck.id === x.activeDeck)) return false;
    if (!object(x.gear) || Object.entries(x.gear).some(([id, item]) => !gear(id) || !areaOpen(state, gear(id).area) || !exact(item, ['successes', 'failed']) || !exact(item.successes, ['steady', 'bold', 'brilliant']) || !Object.values(item.successes).every(value => integer(value, 6)) || !integer(item.failed, 6) || used(item) > 6) || !x.equipmentUnlocked && Object.keys(x.gear).length || x.equipmentUnlocked && !x.gear['trail-boots']) return false;
    if (!exact(x.equipped, D.SLOTS) || D.SLOTS.some(slot => x.equipped[slot] !== null && (!gear(x.equipped[slot]) || !own(x.gear, x.equipped[slot]) || gear(x.equipped[slot]).slot !== slot)) || !exact(x.scrolls, D.SCROLLS.map(s => s.id)) || !Object.values(x.scrolls).every(value => integer(value)) || !integer(x.ink)) return false;
    if (!integer(x.lootRng, 4294967295) || !x.lootRng || !integer(x.scrollRng, 4294967295) || !x.scrollRng || ![x.cardRemainingMs, x.scrollRemainingMs].every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 9000000) || ![x.cardEligibleMs, x.scrollEligibleMs].every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 8.64e15)) return false;
    if (![x.cardFinds, x.scrollFinds, x.revision, x.sequence, x.seen].every(value => integer(value)) || x.seen > x.sequence || !integer(x.sinceEpic, 12) || !integer(x.sinceLegendary, 80) || !Array.isArray(x.recent) || x.recent.length > 24) return false;
    if (![2, 3, 4].includes(x.slotsAnnounced) || x.slotsAnnounced > slots(state)) return false;
    if (!x.cardsUnlocked && (x.cardEligibleMs || x.cardFinds || x.sinceEpic || x.sinceLegendary || x.ink || x.cardRemainingMs !== template.cardRemainingMs) || !x.equipmentUnlocked && (x.scrollEligibleMs || x.scrollFinds || Object.values(x.scrolls).some(Boolean) || x.scrollRemainingMs !== template.scrollRemainingMs)) return false;
    if (x.recent.some((entry, index) => !object(entry) || Object.keys(entry).some(key => !['id', 'outcome', 'title', 'detail', 'rarity', 'consumed', 'delta', 'cardId', 'itemId', 'scrollId'].includes(key)) || !integer(entry.id, x.sequence) || entry.id < 1 || index > 0 && entry.id <= x.recent[index - 1].id || !['found', 'fused', 'success', 'failed', 'restored', 'reforged'].includes(entry.outcome) || !D.RARITIES.includes(entry.rarity) || !['title', 'detail'].every(key => typeof entry[key] === 'string' && entry[key].length <= 350) || !object(entry.consumed) || !object(entry.delta) || Object.entries(entry.consumed).some(([key, value]) => key === 'scrollId' ? !scroll(value) : ['coins', 'ore', 'herbs', 'provisions', 'knowledge', 'maps'].includes(key) ? !integer(value) : !['copies', 'count', 'attemptSlots'].includes(key) || !integer(value, 100)) || Object.entries(entry.delta).some(([key, value]) => !['copies', 'owned', 'scrolls', 'rank', 'points', 'failedSlots', 'freeSlots', 'deckSlots', 'ink'].includes(key) || !Number.isSafeInteger(value) || value < (key === 'points' ? -30 : -6) || value > 100) || entry.cardId !== undefined && !card(entry.cardId) || entry.itemId !== undefined && !gear(entry.itemId) || entry.scrollId !== undefined && !scroll(entry.scrollId))) return false;
    return true;
  }
  return { Content: D, create, validate, areaOpen, slots, points, used, modifiers, nextEvent, tick, quote, act, view, syncUnlocks, setRateProvider, setEntitlements };
});
