(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersAreaSkillsContent = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const AREA = {
    greenway: { name: 'Trail', resource: 'coins', base: 6, roles: ['Travel', 'Carry', 'Discover'], unit: 'deliveries', foundation: [3, 18], operations: [30, 45, 60] },
    quarry: { name: 'Quarry', resource: 'ore', base: 30, roles: ['Mine', 'Move', 'Refine'], unit: 'refined units', foundation: [6, 18], operations: [40, 70, 100] },
    watchtower: { name: 'Tower', resource: 'knowledge', base: 60, roles: ['Survey', 'Signal', 'Command'], unit: 'survey work', foundation: [60, 180], operations: [360, 600, 900] },
    workshop: { name: 'Workshop', resource: 'provisions', base: 100, roles: ['Assemble', 'Equip', 'Convert'], unit: 'assembled units', foundation: [6, 18], operations: [40, 70, 100] },
    ruins: { name: 'Ruins', resource: 'herbs', base: 180, roles: ['Delve', 'Interpret', 'Recover'], unit: 'recovered finds', foundation: [3, 9], operations: [20, 35, 50] },
    harbor: { name: 'Harbor', resource: 'maps', base: 300, roles: ['Build', 'Sail', 'Stow'], unit: 'voyages', foundation: [1, 3], operations: [5, 8, 12] }
  };
  const SKILLS = [];
  const add = (areaId, id, name, role, module, project, from, to, unit, effect, options) => {
    const index = SKILLS.filter(d => d.areaId === areaId).length;
    const area = AREA[areaId];
    SKILLS.push({ id, areaId, name, label: name, icon: 'skill-' + id, functionalRole: area.roles[role], role, module, moduleLabel: ['Foundations', 'Operations', 'Connections', 'Industry', 'Mastery'][module], order: index,
      project: project || null, from, to, unit, effect, maxRank: 10, base: area.base * [1, 8, 32, 128, 512][module], resource: area.resource,
      outputRequired: index < 3 ? area.operations[index] : area.operations[2] * [4, 8, 12, 20, 30, 40][index - 3],
      options: (options || []).map(value => ({ value, label: value === 'off' ? 'Off' : value[0].toUpperCase() + value.slice(1).replace(/-/g, ' ') })) });
  };
  add('greenway', 'express-routes', 'Express Routes', 0, 1, null, .2, .35, 'less trip work', 'Express deliveries finish sooner and carry 15% less arrival cargo.', ['off', 'express']);
  add('greenway', 'cargo-lashing', 'Cargo Lashing', 1, 1, null, .25, .5, 'more arrival cargo', 'Secure cargo increases arrival rewards without changing continuous income.');
  add('greenway', 'field-journals', 'Field Journals', 2, 1, null, 10, 25, 'seconds of maps per arrival', 'Each completed delivery also records maps from unboosted Scouting.');
  add('greenway', 'caravan-escorts', 'Caravan Escorts', 1, 2, 'wheelworks', .15, .3, 'lost trade retained', 'Freight retains some of the coin income normally surrendered by its working plan.');
  add('greenway', 'supply-depots', 'Supply Depots', 1, 2, 'tower-surveys', .2, .6, 'extra buffer capacity', 'Waystations enlarge Quarry and Ruins buffers.');
  add('greenway', 'relay-runners', 'Relay Runners', 2, 3, 'rail-network', 2, 8, 'seconds of research per arrival', 'Deliveries advance an already funded commission using unboosted Trail research.');
  add('greenway', 'return-cargo', 'Return Cargo', 1, 4, 'standing-contracts', .05, .15, 'paid provisions returned', 'Returning voyages refund a bounded share of their frozen provision bill.');
  add('greenway', 'bonded-routes', 'Bonded Routes', 1, 4, 'continental-exchange', .1, .25, 'extra voyage cargo', 'Divert 20% of Trail coin production to enlarge newly funded Harbor cargo.', ['off', 'bonded']);
  add('greenway', 'continental-logistics', 'Continental Logistics', 2, 4, 'guild-commerce', .2, .4, 'secondary dispatch strength', 'Run an earned domestic dispatch alongside continental freight.', ['off', 'trade', 'freight', 'survey']);
  add('quarry', 'stockpiles', 'Stockpiles', 1, 1, null, .5, 1.5, 'extra buffer capacity', 'Keep ore moving through larger extraction and furnace queues.');
  add('quarry', 'ore-sorting', 'Ore Sorting', 1, 1, null, .25, .55, 'extra graded yield', 'Grade ore for greater yield at 20% lower handling capacity.', ['off', 'graded']);
  add('quarry', 'batch-kilns', 'Batch Kilns', 2, 1, null, .2, .5, 'extra batch yield', 'Settle stronger refining batches every 5 to 3 seconds instead of continuously.', ['off', 'batch']);
  add('quarry', 'reinforced-shafts', 'Reinforced Shafts', 0, 2, 'industrial-supports', .05, .2, 'additional alloy support', 'Rich deposits retain alloy support alongside the existing industrial permission.');
  add('quarry', 'rail-transfer', 'Rail Transfer', 1, 3, 'rail-network', .2, .5, 'rail support to refining', 'Trail rail hauling also feeds the furnaces.');
  add('quarry', 'slag-processing', 'Slag Processing', 2, 3, 'sorting-lines', .05, .15, 'provisions per refined unit', 'Turn actual refining byproducts into provisions.');
  add('quarry', 'shift-planning', 'Shift Planning', 1, 3, 'guild-industry', 30, 5, 'seconds between checks', 'Switch earned rich and balanced plans when queues cross 25% or 75%.', ['off', 'automatic']);
  add('quarry', 'resonant-drills', 'Resonant Drills', 0, 4, 'deepworks-commission', 20, 50, 'boosted units per 100 extracted', 'Actual extraction charges a double-speed drill burst; finite queues still apply.');
  add('quarry', 'parallel-furnaces', 'Parallel Furnaces', 2, 4, 'standardized-production', .25, .5, 'second furnace capacity', 'Run a different earned recipe alongside the original furnace.', ['off', 'balanced', 'rich', 'alloy']);
  add('watchtower', 'field-notebooks', 'Field Notebooks', 0, 1, null, 60, 180, 'seconds of stored survey', 'Store unused survey work for a bounded contribution to the next funded commission.');
  add('watchtower', 'triangulation', 'Triangulation', 0, 1, null, .25, .5, 'extra research capacity', 'Trade 25% of map output for stronger research.', ['off', 'triangulate']);
  add('watchtower', 'dispatch-codes', 'Dispatch Codes', 2, 1, null, .1, .25, 'echoed assignment strength', 'Echo a share of coordination to a second region.', ['off', 'trade', 'industry', 'survey']);
  add('watchtower', 'mineral-cartography', 'Mineral Cartography', 0, 2, 'deposit-maps', .78, .9, 'rich extraction retained', 'Mapped rich deposits retain more of their raw extraction capacity.');
  add('watchtower', 'logistics-charts', 'Logistics Charts', 1, 3, 'rail-network', .15, .35, 'assigned logistics bonus', 'Trade assignments strengthen Trail arrivals; industry assignments enlarge Quarry buffers.');
  add('watchtower', 'weather-stations', 'Weather Stations', 0, 3, 'harbor-observations', .3, .7, 'weather slowdown removed', 'Reduce bad-weather sailing penalties without exceeding clear-weather pace.');
  add('watchtower', 'research-exchanges', 'Research Exchanges', 0, 4, 'ruins-resonators', .1, .25, 'lost maps retained', 'Deep-record surveying retains part of its sacrificed map output.');
  add('watchtower', 'long-signals', 'Long Signals', 1, 4, 'ruins-resonators', .2, .45, 'extra weakest-region coordination', 'Support the least developed assigned region.');
  add('watchtower', 'celestial-calendar', 'Celestial Calendar', 0, 4, 'guild-discovery', .2, .4, 'secondary survey strength', 'Survey a second earned target alongside the primary target.', ['off', 'near', 'deep', 'ocean']);
  add('workshop', 'material-hoppers', 'Material Hoppers', 0, 1, null, 20, 60, 'seconds of prepaid ore', 'Prepay ore above the reserve into a visible assembly buffer.', ['off', 'buffer']);
  add('workshop', 'template-queue', 'Template Queue', 0, 1, null, 20, 5, 'assembled units per rotation', 'Rotate earned recipes automatically, one at a time.', ['off', 'alternate']);
  add('workshop', 'offcut-recovery', 'Offcut Recovery', 2, 1, null, .05, .15, 'consumed ore returned', 'Return a bounded share of ore actually used by manufacturing.');
  add('workshop', 'standard-tools', 'Standard Tools', 1, 2, 'alloy-machinery', .2, .45, 'Quarry capacity support', 'Divert 20% of manufacturing output into Quarry extraction tools.', ['off', 'extraction']);
  add('workshop', 'spare-parts', 'Spare Parts', 1, 3, 'sorting-lines', .2, .45, 'Quarry capacity support', 'Use the same manufacturing allocation for hauling or refining instead.', ['off', 'hauling', 'refining']);
  add('workshop', 'instrument-cases', 'Instrument Cases', 1, 3, 'workshop-lenses', .1, .25, 'ordinary provisions retained', 'Instrument batches also supply a portion of normal provision production.');
  add('workshop', 'precision-fixtures', 'Precision Fixtures', 2, 4, 'precision-patterns', .2, .5, 'lost assembly speed retained', 'Recover some of the assembly speed surrendered by Precision.');
  add('workshop', 'modular-frames', 'Modular Frames', 0, 4, 'standardized-production', 3, 10, 'allocation divisions', 'Split parallel templates unevenly, from thirds to tenths.', ['off', 'primary-heavy', 'secondary-heavy']);
  add('workshop', 'export-crates', 'Export Crates', 1, 4, 'guild-industry', .08, .2, 'launch provisions saved', 'Divert 10% of provision output to packaging for cheaper new Harbor launches.', ['off', 'package']);
  add('ruins', 'field-camps', 'Field Camps', 0, 1, null, .5, 1.5, 'extra find capacity', 'Larger camps hold more discovered and interpreted finds.');
  add('ruins', 'careful-recovery', 'Careful Recovery', 2, 1, null, .25, .55, 'extra recovery yield', 'Careful handling increases yield at 20% lower recovery capacity.', ['off', 'careful']);
  add('ruins', 'site-catalogues', 'Site Catalogues', 1, 1, null, 10, 3, 'finds per rotation', 'Rotate automatically toward the least represented earned discovery type.', ['off', 'rotate']);
  add('ruins', 'survey-tablets', 'Survey Tablets', 1, 2, 'restoration-laboratory', .1, .3, 'maps per inscribed unit', 'Interpreted inscriptions also produce maps.');
  add('ruins', 'botanical-remedies', 'Botanical Remedies', 2, 3, 'restoration-laboratory', .1, .3, 'provisions per botanical unit', 'Recovered botanical finds also supply provisions.');
  add('ruins', 'reclaimed-alloys', 'Reclaimed Alloys', 2, 3, 'precision-patterns', .1, .3, 'Workshop alloy support', 'Metallic recovery with an industry artifact strengthens Workshop conversion.');
  add('ruins', 'expedition-rigs', 'Expedition Rigs', 0, 4, 'deepwater-equipment', .2, .45, 'spare interpretation recovered', 'Direct spare interpretation capacity into existing shallow recovery.');
  add('ruins', 'resonant-pairings', 'Resonant Pairings', 1, 4, 'deepwater-resonance', .15, .35, 'distinct-role pairing bonus', 'Two different attuned artifact roles strengthen each other.');
  add('ruins', 'archive-network', 'Archive Network', 1, 4, 'guild-discovery', .3, .65, 'diverted research equivalent', 'Divert 20% of recovery yield into an already funded commission.', ['off', 'archive']);
  add('harbor', 'provision-packing', 'Provision Packing', 2, 1, null, .08, .2, 'launch provisions saved', 'Pay fewer provisions for future voyages; funded manifests remain unchanged.');
  add('harbor', 'coastal-tenders', 'Coastal Tenders', 1, 1, null, .2, .35, 'less coastal voyage work', 'Short coastal trips carry 15% less cargo.', ['off', 'short']);
  add('harbor', 'mixed-holds', 'Mixed Holds', 2, 1, null, .1, .25, 'material cargo retained', 'Trade voyages also carry a portion of material-route ore.');
  add('harbor', 'scheduled-convoys', 'Scheduled Convoys', 0, 2, 'standing-contracts', .1, .25, 'prepared convoy cargo', 'A returning voyage rewards an immediately provisioned next departure.');
  add('harbor', 'salvage-nets', 'Salvage Nets', 2, 3, 'ocean-charts', .1, .25, 'material cargo salvaged', 'Discovery voyages recover a portion of material-route ore.');
  add('harbor', 'weather-routing', 'Weather Routing', 1, 3, 'harbor-observations', .1, .04, 'cargo surrendered', 'Automatically choose the least weather-affected earned port.', ['off', 'route']);
  add('harbor', 'deepwater-holds', 'Deepwater Holds', 2, 4, 'deepwater-equipment', .2, .5, 'lost maps retained', 'Material voyages retain more of their sacrificed map cargo.');
  add('harbor', 'twin-manifests', 'Twin Manifests', 0, 4, 'navigation-artifact', .7, 1, 'secondary manifest strength', 'Concurrent fleet slots may carry different earned manifest types.', ['off', 'trade', 'materials', 'discovery', 'commerce']);
  add('harbor', 'exchange-houses', 'Continental Exchange Houses', 2, 4, 'guild-commerce', .15, .35, 'domestic logistics support', 'Divert 15% of voyage coin cargo into Trail arrivals and Quarry hauling.', ['off', 'domestic']);
  const TRACK_LAYOUT = {
    greenway: { boots: [0, 0], porters: [0, 1], scouts: [0, 2], caravans: [2, 1], waystations: [2, 0], railways: [3, 0] },
    quarry: { picks: [0, 0], carts: [0, 1], furnace: [0, 2], geology: [2, 0], recovery: [2, 2], deepworks: [4, 0] },
    watchtower: { beacon: [0, 0], signals: [0, 1], crew: [0, 2], optics: [2, 0], forecasting: [3, 0], 'relay-grid': [4, 1] },
    workshop: { assembly: [0, 0], toolmaking: [0, 1], metallurgy: [0, 2], mechanisms: [2, 1], precision: [3, 2], replication: [4, 0] },
    ruins: { delving: [0, 0], archaeology: [0, 1], 'recovery-teams': [0, 2], restoration: [2, 2], attunement: [3, 1], resonance: [4, 1] },
    harbor: { shipbuilding: [0, 0], seamanship: [0, 1], stowage: [0, 2], contracts: [2, 2], navigation: [3, 1], 'fleet-command': [4, 0] }
  };
  // Each reveal module contains three entries when old production tracks and
  // new techniques are shown together. Quarry keeps the approved concept order.
  TRACK_LAYOUT.greenway.waystations = [3, 0];
  TRACK_LAYOUT.greenway.railways = [4, 0];
  SKILLS.forEach(d => {
    if (d.areaId !== 'quarry') {
      d.module = d.order < 3 ? 1 : d.order < 5 ? 2 : d.order < 7 ? 3 : 4;
      d.moduleLabel = ['Foundations', 'Operations', 'Connections', 'Industry', 'Mastery'][d.module];
      d.base = AREA[d.areaId].base * [1, 8, 32, 128, 512][d.module];
    }
  });
  return { AREA, SKILLS, TRACK_LAYOUT };
});
