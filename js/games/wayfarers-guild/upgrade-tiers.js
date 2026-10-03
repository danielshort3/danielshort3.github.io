(function (root, factory) {
  'use strict';
  const common = typeof module === 'object' && module.exports;
  const api = factory(common ? require('./content.js') : root.WayfarersContent, common ? require('./progression-content.js') : root.WayfarersProgressionContent);
  if (common) module.exports = api;
  if (root) root.WayfarersUpgradeTiers = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (C, D) {
  'use strict';
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const legacyTracks = { greenway: ['boots', 'porters', 'scouts'], quarry: ['picks', 'carts', 'furnace'], watchtower: ['crew', 'lift', 'beacon'] };
  const areaName = id => D.AREAS.find(area => area.id === id)?.name || 'Guild';
  let baseOpen = () => false, legacyDevelopments = [], foundationRequirement = null;
  const cache = new Map();
  const configure = providers => { baseOpen = providers.baseOpen; legacyDevelopments = providers.legacyDevelopments || []; foundationRequirement = providers.foundationRequirement || null; cache.clear(); };
  const initial = () => ({ version: 1, claimed: ['area:greenway:boots'], pending: [], prompted: [] });
  const identity = action => action.type + ':' + action.id;
  function definitions(state) {
    const x = state.expedition, rows = [], progression = x?.version === 3;
    if (cache.has(progression)) return cache.get(progression);
    for (const area of D.AREAS) {
      const tracks = progression ? area.tracks : (legacyTracks[area.id] || []).map((id, index) => {
        const def = area.tracks.find(track => track.id === id);
        return { id, name: id === 'lift' ? 'Tower lift' : def?.name || 'Tower crew', effect: id === 'lift' ? 'Construction capacity for expansions' : def?.effect || 'Repair and protection allocation', icon: def?.icon || 'equipment', index };
      });
      tracks.forEach((track, index) => rows.push({ id: 'area:' + area.id + ':' + track.id, areaId: area.id, tier: index + 1, label: area.name + ' · ' + track.name, shortEffect: track.effect, icon: track.icon,
        tracks: [{ id: track.id, label: track.name, shortEffect: track.effect, icon: track.icon }], actions: [{ type: 'expedition-buy', areaId: area.id, id: track.id }], local: true, previous: index ? 'area:' + area.id + ':' + tracks[index - 1].id : null, index }));
    }
    const groups = new Map();
    function add(id, label, action, areaId, description, icon, tier = 1) {
      if (!groups.has(id)) groups.set(id, { id, areaId, tier, label, shortEffect: description, icon, tracks: [], actions: [] });
      const row = groups.get(id); row.actions.push(action); row.tracks.push({ id: action.id, label: label === description ? action.id : description, shortEffect: description, icon });
    }
    C.UPGRADES.forEach(def => add('guild:' + def.room + ':' + (def.at || 0), (C.ROOMS.find(room => room.id === def.room)?.name || 'Guild') + ' upgrades', { type: 'buy', id: def.id }, def.room === 'trail' ? 'greenway' : ['mine','forge'].includes(def.room) ? 'quarry' : 'watchtower', def.name, def.equipment ? 'equipment' : 'guild', def.at || 1));
    const researchNames = { 5: 'Working methods', 6: 'Field knowledge', 7: 'Resource exchanges', 8: 'Route dispatch', 9: 'Resource planning', 10: 'Shared training', 14: 'Frontier research' };
    C.RESEARCH.forEach(def => add('research:' + def.at, researchNames[def.at] || 'New research', { type: 'research', id: def.id }, 'watchtower', def.name, 'research', def.at));
    C.LUCK_RESEARCH.forEach(def => add('finds:' + (def.room || 'study'), def.room === 'cartography' ? 'Familiar relic studies' : 'Discovery studies', { type: 'luck-research', id: def.id }, 'watchtower', def.name, 'relic', def.at || 1));
    C.REFIT_UPGRADES.forEach(def => add('notes', 'Field note upgrades', { type: 'refit-upgrade', id: def.id }, 'greenway', def.name, 'notes'));
    C.LEGACY_UPGRADES.forEach(def => add('crests', 'Guild crest upgrades', { type: 'legacy-upgrade', id: def.id }, 'watchtower', def.name, 'crests'));
    C.CAPABILITIES.forEach(def => add('planning:' + def.resource + ':' + def.at, 'Guild planning tools', { type: 'capability', id: def.id }, 'watchtower', def.name, 'guild', def.at || 1));
    C.PROJECTS.forEach(def => add('project:' + def.id, def.name, { type: 'project', id: def.id }, 'watchtower', def.description, 'guild'));
    for (const id of ['supply', 'survey', 'industry']) add('chapter-projects', 'Regional charter projects', { type: 'project', id: 'chapter-' + id }, 'watchtower', 'Choose a regional ' + id + ' project', 'guild');
    for (const def of progression ? D.PROJECTS : legacyDevelopments) add('development:' + (progression ? 'p:' : 'e:') + def.id, def.name, { type: 'expedition-development', id: def.id }, def.target || def.to[0], def.effect || def.description, 'research', def.chapter || 1);
    const result = rows.concat([...groups.values()]); cache.set(progression, result); return result;
  }
  function isOwned(state, action) {
    if (action.type === 'buy') return state.upgrades[action.id] > 0;
    if (action.type === 'research') return state.research.includes(action.id);
    if (action.type === 'luck-research') return state.luck.research.includes(action.id);
    if (action.type === 'refit-upgrade') return state.refitUpgrades[action.id] > 0;
    if (action.type === 'legacy-upgrade') return state.legacy[action.id] > 0;
    if (action.type === 'capability') return state.guild.capabilities.includes(action.id);
    if (action.type === 'project') return state.guild.projects.includes(action.id) || action.id === 'chapter-' + state.guild.chapterProject.choice;
    if (action.type === 'expedition-development') return (state.expedition?.projects || state.expedition?.developments || []).includes(action.id) || state.expedition?.commission?.id === action.id;
    return false;
  }
  function localReady(state, row, migration) {
    const x = state.expedition, area = x?.areas?.[row.areaId], track = row.actions[0].id;
    if (!area) return false;
    if (x.version === 3) return area.learned.includes(track) || area.highRanks[track] > 0 || area.ranks[track] > 0;
    if (area.ranks[track] > 0 || row.index === 0) return true;
    if (migration) return row.areaId !== 'greenway' || area.index > 0 || track === 'porters' && (area.ranks.boots >= 2 || area.work >= 28) || track === 'scouts' && (Object.values(area.ranks).reduce((a,b) => a+b,0) >= 5 || area.work >= 70);
    const previous = legacyTracks[row.areaId][row.index - 1];
    return area.ranks[previous] >= 2 || area.elapsed >= row.index * 180;
  }
  function eligible(state, row, migration = false) {
    if (row.local) return localReady(state, row, migration) && (migration || row.index === 0 || state.upgradeTiers.claimed.includes(row.previous));
    // The existing guild-wide boot system is introduced after the first area;
    // the opening remains one understandable local purchase surface.
    if (!migration && row.id.startsWith('guild:trail:') && state.expedition && state.lifetime.highestRoute < 0) return false;
    return row.actions.some(action => isOwned(state, action) || baseOpen(state, action));
  }
  function migrate(state) {
    const rows = definitions(state), claimed = rows.filter(row => eligible(state, row, true)).map(row => row.id);
    return { version: 1, claimed, pending: [], prompted: [] };
  }
  function sync(state) {
    if (!state.upgradeTiers) state.upgradeTiers = migrate(state);
    const x = state.upgradeTiers;
    for (const row of definitions(state)) {
      if (x.claimed.includes(row.id) || x.pending.includes(row.id) || !eligible(state, row)) continue;
      if (row.local && row.index === 0) x.claimed.push(row.id);
      else x.pending.push(row.id);
    }
  }
  function adopt(state) {
    if (!state.upgradeTiers) return;
    const x = state.upgradeTiers;
    for (const row of definitions(state)) {
      if (!(row.local && localReady(state, row, true) || !row.local && row.actions.some(action => isOwned(state, action)))) continue;
      if (!x.claimed.includes(row.id)) x.claimed.push(row.id);
      x.pending = x.pending.filter(id => id !== row.id); x.prompted = x.prompted.filter(id => id !== row.id);
    }
  }
  function rowFor(state, action) {
    if (action.type === 'expedition-buy') return definitions(state).find(row => row.local && row.areaId === (action.areaId || state.expedition?.selectedArea) && row.actions[0].id === action.id);
    return definitions(state).find(row => !row.local && row.actions.some(value => identity(value) === identity(action)));
  }
  function allows(state, action) {
    if (!state.upgradeTiers) return true; // Supported historical state before normalization.
    const row = rowFor(state, action);
    return !row || state.upgradeTiers.claimed.includes(row.id);
  }
  function visible(state, action) {
    const row = rowFor(state, action);
    return allows(state, action) && (!row || row.local || isOwned(state, action) || baseOpen(state, action));
  }
  function readyFor(state, action) {
    const row = rowFor(state, action);
    return row && state.upgradeTiers?.pending.includes(row.id) ? view(state).ready.find(item => item.id === row.id) : null;
  }
  function act(state, action) {
    const x = state.upgradeTiers;
    if (!x) return { ok: false, message: 'Reload this guild before choosing an upgrade tier.' };
    if (action.type === 'upgrade-tier-unlock') {
      const row = definitions(state).find(value => value.id === action.id);
      if (!row || !x.pending.includes(action.id)) return { ok: false, message: 'That upgrade tier is not ready, or is already unlocked.' };
      x.pending = x.pending.filter(id => id !== action.id); x.prompted = x.prompted.filter(id => id !== action.id); x.claimed.push(action.id);
      return { ok: true, message: row.label + ' unlocked. Choose its upgrades when you are ready.' };
    }
    if (action.type === 'upgrade-tier-defer') {
      if (!Array.isArray(action.ids) || !action.ids.length || new Set(action.ids).size !== action.ids.length || action.ids.some(id => !x.pending.includes(id))) return { ok: false, message: 'Only ready upgrade tiers can be acknowledged.' };
      x.prompted = [...new Set(x.prompted.concat(action.ids))]; return { ok: true, message: 'Ready upgrades saved for later.' };
    }
    return { ok: false, message: 'Choose a valid upgrade tier action.' };
  }
  function foundationView(state) {
    const x = state.upgradeTiers;
    if (!x) return [];
    return definitions(state).filter(row => row.local && row.index < 3 && state.expedition?.areas[row.areaId]).map(row => {
      const area = state.expedition.areas[row.areaId], track = row.actions[0].id;
      const claimed = x.claimed.includes(row.id), ready = x.pending.includes(row.id);
      let requirements = [];
      if (!claimed && !ready && row.index) {
        const canonical = state.expedition.version === 3 && foundationRequirement?.(state, row.areaId, track);
        if (canonical) requirements = canonical.requirements.map(requirement => ({ ...requirement }));
        else {
          const previous = definitions(state).find(item => item.id === row.previous), rank = area.ranks[previous.actions[0].id] || 0;
          // Released E2 keeps its rank OR elapsed-time path. State both routes.
          const elapsed = area.elapsed || 0, duration = row.index * 180;
          requirements = [{ label: previous.tracks[0].label + ' rank 2 or ' + Math.ceil(duration / 60) + ' minutes here', icon: previous.icon, current: rank, required: 2, met: rank >= 2 || elapsed >= duration, alternate: { current: elapsed, required: duration, unit: 'seconds' } }];
        }
        if (row.previous && !x.claimed.includes(row.previous)) {
          const previous = definitions(state).find(item => item.id === row.previous);
          requirements.push({ label: 'Unlock ' + previous.tracks[0].label, icon: previous.icon, current: 0, required: 1, met: false });
        }
      }
      return { id: row.id, tierId: row.id, areaId: row.areaId, trackId: track, name: row.tracks[0].label, label: row.tracks[0].label, icon: row.icon, effectText: row.shortEffect, rank: area.ranks[track] || 0, maxRank: area.cap || null,
        status: claimed ? 'learned' : ready ? 'ready' : 'locked', requirements,
        ...(ready ? { unlockAction: { type: 'upgrade-tier-unlock', id: row.id } } : {}) };
    });
  }
  function view(state) {
    const x = state.upgradeTiers;
    if (!x) return { ready: [], notice: null, claimedCount: 0, foundations: [] };
    const ready = definitions(state).filter(row => x.pending.includes(row.id)).map(row => ({ id: row.id, areaId: row.areaId, tier: row.tier, label: row.label, shortEffect: row.shortEffect, icon: row.icon, tracks: row.tracks, status: 'ready', unlockAction: { type: 'upgrade-tier-unlock', id: row.id }, deferAction: { type: 'upgrade-tier-defer', ids: [row.id] } }));
    const unseen = ready.filter(row => !x.prompted.includes(row.id));
    return { ready, foundations: foundationView(state), claimedCount: x.claimed.length, notice: unseen.length ? { id: unseen.map(row => row.id).join('|'), title: unseen.length === 1 ? 'An upgrade tier is ready' : unseen.length + ' upgrade tiers are ready', items: unseen, deferAction: { type: 'upgrade-tier-defer', ids: unseen.map(row => row.id) } } : null };
  }
  function validate(value, state) {
    try {
    if (!object(value) || Object.keys(value).length !== 4 || !['version','claimed','pending','prompted'].every(key => own(value,key)) || value.version !== 1) return false;
    const all = new Set([...definitions({ ...state, expedition: { ...state.expedition, version: 2 } }), ...definitions({ ...state, expedition: { ...state.expedition, version: 3 } })].map(row => row.id));
    for (const key of ['claimed','pending','prompted']) if (!Array.isArray(value[key]) || value[key].length > all.size || new Set(value[key]).size !== value[key].length || value[key].some(id => typeof id !== 'string' || !all.has(id))) return false;
    if (value.pending.some(id => value.claimed.includes(id)) || !value.prompted.every(id => value.pending.includes(id))) return false;
    const rows = definitions(state);
    for (const id of value.claimed.concat(value.pending)) {
      const row = rows.find(row => row.id === id);
      // Prior-version tier IDs are historical knowledge after a deliberate
      // adoption reset. They cannot name a current purchasable future track.
      if (row && !eligible(state, row, true)) return false;
    }
    return value.claimed.includes('area:greenway:boots');
    } catch (_) { return false; }
  }
  const describe = state => definitions(state).map(row => JSON.parse(JSON.stringify(row)));
  return { initial, configure, migrate, sync, adopt, allows, visible, readyFor, act, view, validate, describe };
});
