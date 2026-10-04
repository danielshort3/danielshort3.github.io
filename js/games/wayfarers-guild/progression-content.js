(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersProgressionContent = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  // Track identities and dependencies are permanent knowledge. Their purchased
  // ranks are run investments; no UI code owns a price, unlock or reward.
  const AREAS = [
    { id: 'greenway', name: 'Trail', icon: 'trail', resource: 'coins', tracks: [
      ['boots', 'Pathfinding', 'boots', 6, 'Travel capacity and shortcuts'], ['porters', 'Porters', 'coins', 12, 'Cargo per delivery'], ['scouts', 'Scouting', 'maps', 24, 'Discovery and route surveys'],
      ['caravans', 'Caravans', 'hall', 150, 'Freight support alongside the selected dispatch'], ['waystations', 'Waystations', 'provisions', 300, 'Supply coverage and staging'], ['railways', 'Railways', 'equipment', 600, 'A dedicated bulk freight lane'] ] },
    { id: 'quarry', name: 'Quarry', icon: 'mine', resource: 'ore', tracks: [
      ['picks', 'Excavation', 'mine', 30, 'Raw extraction capacity'], ['carts', 'Haulage', 'ore', 36, 'Transport and queue capacity'], ['furnace', 'Refining', 'forge', 42, 'Processing capacity'],
      ['geology', 'Geology', 'maps', 180, 'Selectable deposits and material yields'], ['recovery', 'Recovery', 'equipment', 350, 'Recover waste without new extraction'], ['deepworks', 'Deepworks', 'mine', 750, 'Deep extraction alongside the selected deposit'] ] },
    { id: 'watchtower', name: 'Tower', icon: 'observatory', resource: 'knowledge', tracks: [
      ['beacon', 'Surveying', 'knowledge', 60, 'Mapped territory and research'], ['signals', 'Signals', 'maps', 75, 'Coordination capacity'], ['crew', 'Command', 'hall', 90, 'Specialist assignment capacity'],
      ['optics', 'Optics', 'research', 250, 'Distant survey targets'], ['forecasting', 'Forecasting', 'maps', 450, 'Ocean surveys and voyage speed support'], ['relay-grid', 'Relay Grid', 'observatory', 900, 'Coordinate multiple regions'] ] },
    { id: 'workshop', name: 'Workshop', icon: 'forge', resource: 'provisions', tracks: [
      ['assembly', 'Assembly', 'forge', 100, 'Manufacturing capacity'], ['toolmaking', 'Toolmaking', 'equipment', 120, 'Equipment support'], ['metallurgy', 'Metallurgy', 'ore', 140, 'Alloy conversion yield'],
      ['mechanisms', 'Mechanisms', 'equipment', 300, 'Machinery allocation'], ['precision', 'Precision', 'research', 550, 'Efficient processing recipes'], ['replication', 'Replication', 'forge', 1100, 'Parallel manufacturing templates'] ] },
    { id: 'ruins', name: 'Ruins', icon: 'relic', resource: 'herbs', tracks: [
      ['delving', 'Delving', 'trail', 180, 'Discovery reach'], ['archaeology', 'Archaeology', 'knowledge', 210, 'Interpretation capacity'], ['recovery-teams', 'Recovery Teams', 'hall', 240, 'Extract interpreted finds'],
      ['restoration', 'Restoration', 'equipment', 450, 'Restore usable artifacts'], ['attunement', 'Attunement', 'relic', 750, 'Assign discoveries to specialist roles'], ['resonance', 'Resonance', 'observatory', 1500, 'Combine distinct artifact effects'] ] },
    { id: 'harbor', name: 'Harbor', icon: 'caravan', resource: 'maps', tracks: [
      ['shipbuilding', 'Shipbuilding', 'caravan', 300, 'Fleet carrying capacity'], ['seamanship', 'Seamanship', 'trail', 350, 'Voyage completion speed'], ['stowage', 'Stowage', 'provisions', 400, 'Mixed cargo loads'],
      ['contracts', 'Contracts', 'coins', 700, 'Standing exchanges'], ['navigation', 'Navigation', 'maps', 1100, 'Distant ports'], ['fleet-command', 'Fleet Command', 'hall', 2000, 'Concurrent expeditions'] ] }
  ];
  AREAS.forEach(area => { area.tracks = area.tracks.map(([id, name, icon, base, effect], index) => ({ id, name, icon, base, effect, areaId: area.id, index, max: 1000 })); });
  const PROJECTS = [];
  const add = (id, name, chapter, source, target, requires, costs, work, effect, unlock) => PROJECTS.push({ id, name, chapter, source, target, requires, costs, work, effect, unlock });
  add('wheelworks', 'Quarry wheelworks', 1, 'quarry', 'greenway', [], { coins: 150, ore: 15 }, 120, 'Add caravan freight support while keeping your selected Trail dispatch.', { track: 'caravans' });
  add('tower-surveys', 'Tower survey office', 1, 'watchtower', 'greenway', [], { coins: 250, knowledge: 10 }, 180, 'Open Waystations and supply staging.', { track: 'waystations' });
  add('deposit-maps', 'Mapped deposits', 1, 'watchtower', 'quarry', [], { maps: 8, knowledge: 15 }, 240, 'Choose rich, balanced or alloy deposits.', { track: 'geology' });
  add('workshop-foundation', 'Found the Workshop', 2, 'quarry', 'workshop', ['wheelworks', 'tower-surveys'], { coins: 2000, ore: 150, knowledge: 80 }, 50000, 'Manufacture equipment and allocate alloys across the guild.', { area: 'workshop' });
  add('alloy-machinery', 'Alloy machinery', 2, 'quarry', 'workshop', ['workshop-foundation', 'deposit-maps'], { ore: 500, knowledge: 200 }, 70000, 'Assign machinery to extraction, manufacture or survey.', { track: 'mechanisms' });
  add('sorting-lines', 'Sorting lines', 2, 'workshop', 'quarry', ['workshop-foundation'], { coins: 5000, provisions: 250 }, 85000, 'Recover refining waste into usable production.', { track: 'recovery' });
  add('workshop-lenses', 'Workshop lenses', 3, 'workshop', 'watchtower', ['alloy-machinery'], { ore: 700, provisions: 600, knowledge: 700 }, 120000, 'Survey distant targets; Tower rank ceiling becomes 250.', { track: 'optics', cap: 250 });
  add('rail-network', 'Regional rail network', 3, 'workshop', 'greenway', ['sorting-lines'], { coins: 15000, ore: 1500, provisions: 800 }, 150000, 'Open a dedicated freight lane alongside trade; Trail ceiling 250.', { track: 'railways', cap: 250 });
  add('industrial-supports', 'Industrial supports', 3, 'workshop', 'quarry', ['sorting-lines', 'alloy-machinery'], { ore: 1800, provisions: 1000 }, 180000, 'Run rich deposits without losing all alloy supply; Quarry ceiling 250.', { cap: 250, behavior: 'dual-deposit' });
  add('ruins-expedition', 'Open the Ancient Ruins', 3, 'watchtower', 'ruins', ['workshop-lenses', 'rail-network'], { maps: 1800, knowledge: 3000, provisions: 1500 }, 240000, 'Delve, interpret and recover discoveries in a new production chain.', { area: 'ruins' });
  add('restoration-laboratory', 'Restoration laboratory', 4, 'workshop', 'ruins', ['ruins-expedition'], { provisions: 2400, knowledge: 4000 }, 300000, 'Restore finds into productive artifacts; Ruins ceiling 250.', { track: 'restoration', cap: 250 });
  add('precision-patterns', 'Ancient precision patterns', 4, 'ruins', 'workshop', ['ruins-expedition'], { herbs: 1600, knowledge: 5000 }, 340000, 'Switch between high-output and material-efficient manufacturing; Workshop ceiling 250.', { track: 'precision', cap: 250 });
  add('tower-attunement', 'Tower attunement studies', 4, 'watchtower', 'ruins', ['restoration-laboratory'], { herbs: 2200, maps: 3000, knowledge: 6000 }, 390000, 'Assign recovered discoveries to industry, survey or trade.', { track: 'attunement' });
  add('harbor-foundation', 'Found the Harbor', 4, 'greenway', 'harbor', ['precision-patterns', 'tower-attunement'], { coins: 100000, ore: 15000, provisions: 12000, maps: 5000 }, 480000, 'Build vessels, balance cargo and send automatic voyages.', { area: 'harbor' });
  add('standing-contracts', 'Standing trade contracts', 5, 'greenway', 'harbor', ['harbor-foundation'], { coins: 180000, maps: 9000 }, 550000, 'Trade and material contracts use the original Trail supply network.', { track: 'contracts' });
  add('ocean-charts', 'Ocean charts', 5, 'watchtower', 'harbor', ['harbor-foundation'], { knowledge: 20000, maps: 12000 }, 610000, 'Discover distant ports and mixed voyages; Harbor ceiling 250.', { track: 'navigation', cap: 250 });
  add('harbor-observations', 'Harbor weather observations', 5, 'harbor', 'watchtower', ['ocean-charts'], { maps: 18000, provisions: 16000 }, 700000, 'Open ocean survey targets and improve voyage speed through forecast support.', { track: 'forecasting' });
  add('deepwater-equipment', 'Deepwater expedition equipment', 5, 'harbor', 'ruins', ['standing-contracts', 'restoration-laboratory'], { ore: 45000, provisions: 30000, maps: 25000 }, 820000, 'Simultaneous shallow recovery and deep exploration; Ruins ceiling 1000.', { cap: 1000, behavior: 'parallel-delving' });
  add('deepworks-commission', 'Deepworks commission', 6, 'ruins', 'quarry', ['deepwater-equipment', 'industrial-supports'], { herbs: 45000, knowledge: 55000, ore: 60000 }, 940000, 'Add deep extraction to the selected deposit; Quarry ceiling 1000.', { track: 'deepworks', cap: 1000 });
  add('ruins-resonators', 'Ancient resonator network', 6, 'ruins', 'watchtower', ['tower-attunement', 'harbor-observations'], { herbs: 55000, maps: 50000, knowledge: 80000 }, 1100000, 'Coordinate multiple areas without abandoning the first assignment; Tower ceiling 1000.', { track: 'relay-grid', cap: 1000 });
  add('standardized-production', 'Standardized production', 6, 'harbor', 'workshop', ['precision-patterns', 'standing-contracts'], { ore: 90000, provisions: 65000, maps: 55000 }, 1250000, 'Run two manufacturing templates in parallel; Workshop ceiling 1000.', { track: 'replication', cap: 1000 });
  add('continental-exchange', 'Continental exchange', 6, 'harbor', 'greenway', ['rail-network', 'standing-contracts'], { coins: 1000000, maps: 70000, provisions: 80000 }, 1400000, 'Overseas trade uses Trail freight while domestic routes remain active; Trail ceiling 1000.', { cap: 1000, behavior: 'continental-trade' });
  add('navigation-artifact', 'Navigation artifact', 7, 'ruins', 'harbor', ['ruins-resonators', 'ocean-charts'], { herbs: 100000, knowledge: 150000, maps: 90000 }, 1600000, 'Command concurrent trade and discovery expeditions; Harbor ceiling 1000.', { track: 'fleet-command', cap: 1000 });
  add('deepwater-resonance', 'Deepwater resonance', 7, 'harbor', 'ruins', ['deepwater-equipment', 'ruins-resonators'], { herbs: 120000, provisions: 140000, maps: 120000 }, 1800000, 'Combine two different attuned artifact roles.', { track: 'resonance' });
  add('guild-industry', 'Guild commission: industry', 7, 'workshop', 'quarry', ['deepworks-commission', 'standardized-production'], { ore: 250000, provisions: 250000 }, 2000000, 'Enable an integrated extraction-to-manufacture allocation.', { behavior: 'industry-commission' });
  add('guild-discovery', 'Guild commission: discovery', 7, 'watchtower', 'ruins', ['deepwater-resonance', 'navigation-artifact'], { knowledge: 300000, maps: 240000 }, 2200000, 'Interpretation and survey may share a coordinated assignment.', { behavior: 'discovery-commission' });
  add('guild-commerce', 'Guild commission: commerce', 7, 'greenway', 'harbor', ['continental-exchange', 'navigation-artifact'], { coins: 3000000, provisions: 300000 }, 2400000, 'Run continental freight and discovery cargo in the same fleet.', { behavior: 'commerce-commission' });
  // Later commissions are sustained production goals, not login/calendar gates.
  // Their work continues through Refits while players rebuild their operations.
  const researchScale = [0, 1, 25, 100, 200, 6000, 10000, 20000];
  // Spread the midgame research budget across usable discoveries. The Harbor
  // should follow several new decisions, never one week-long capped plateau.
  const midgameWork = { 'restoration-laboratory': 8, 'precision-patterns': 15, 'tower-attunement': 20, 'harbor-foundation': 25 };
  PROJECTS.forEach(project => { project.work *= researchScale[project.chapter] * (midgameWork[project.id] || 1); });
  const BATCHES = [{ count: 1, refits: 0, charters: 0 }, { count: 5, refits: 1, charters: 0 }, { count: 10, refits: 3, charters: 0 }, { count: 25, refits: 0, charters: 1 }, { count: 100, refits: 10, charters: 1 }];
  const FOCUS = { capacity: 3, recharge: 14400, duration: 90, multiplier: 25 };
  return { AREAS, PROJECTS, BATCHES, FOCUS };
});
