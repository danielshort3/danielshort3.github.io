(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersStationContent = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  // IDs and semantic subjects are stable. Atlas order never identifies an upgrade.
  const AREAS = [
    ['greenway', 'Trail', 'coins', 24, 1800, 1, 'trail'],
    ['quarry', 'Quarry', 'ore', 30, 3000, 2, 'mine'],
    ['watchtower', 'Tower', 'knowledge', 60, 180, .1, 'observatory'],
    ['workshop', 'Workshop', 'provisions', 100, 360, .2, 'forge'],
    ['ruins', 'Ruins', 'herbs', 180, 180, .1, 'relic'],
    ['harbor', 'Harbor', 'maps', 300, 4, .06, 'caravan']
  ].map(([id, name, resource, base, target, rate, icon]) => ({ id, name, resource, base, target, rate, icon }));
  const MATRIX = {
    greenway: [
      ['path', 'Path', ['Pathfinding','Waymarks','Trailcraft'], ['boots',null,null], ['Express Routes','Hidden Tracks','Landmark Cairns'], ['express-routes',null,null], 'distance', 'discovery'],
      ['porter-camp', 'Porter Camp', ['Porters','Loading Harnesses','Trade Ledgers'], ['porters',null,null], ['Cargo Lashing','Bonded Routes','Parcel Sorting'], ['cargo-lashing','bonded-routes',null], 'cargo', 'value'],
      ['scout-post', 'Scout Post', ['Scouting','Survey Lines','Keen Eyes'], ['scouts',null,null], ['Field Journals','Survey Satchels','Targeted Surveys'], ['field-journals',null,null], 'surveys', 'discovery'],
      ['supply-depot', 'Supply Depot', ['Waystations','Packing Stations','Provision Standards'], ['waystations',null,null], ['Supply Depots','Return Cargo','Pack Recovery'], ['supply-depots','return-cargo',null], 'supplies', 'efficiency'],
      ['caravan-terminus', 'Caravan Terminus', ['Caravans','Railways','Contract Seals'], ['caravans','railways',null], ['Caravan Escorts','Relay Runners','Continental Logistics'], ['caravan-escorts','relay-runners','continental-logistics'], 'freight', 'support']
    ],
    quarry: [
      ['mine', 'Mine', ['Deep Cut','Quick Swing','Rich Veins'], ['picks',null,null], ['Reinforced Shafts','Power Drill','Vein Mapping'], ['reinforced-shafts','resonant-drills','deepworks'], 'ore', 'discovery'],
      ['hauling', 'Hauling', ['Cart Beds','Axle Bearings','Dispatch Flags'], ['carts',null,null], ['Stockpiles','Rail Transfer','Rail Junctions'], ['stockpiles','rail-transfer',null], 'shipments', 'value'],
      ['sorting', 'Sorting', ['Fine Sieves','Gem Trays','Reclamation'], ['geology',null,'recovery'], ['Ore Sorting','Slag Processing','Gem Inspection'], ['ore-sorting','slag-processing',null], 'graded ore', 'efficiency'],
      ['tool-forge', 'Tool Forge', ['Bellows','Molds','Tempering'], ['furnace',null,null], ['Batch Kilns','Parallel Furnaces','Tempered Picks'], ['batch-kilns','parallel-furnaces',null], 'tools', 'support'],
      ['crystal-lab', 'Crystal Lab', ['Crystal Cutters','Resonance','Sample Banks'], [null,null,null], ['Adaptive Scheduling','Discovery Protection','Resonant Blueprints'], ['shift-planning',null,null], 'samples', 'discovery']
    ],
    watchtower: [
      ['survey-deck', 'Survey Deck', ['Surveying','Sweep Patterns','Optics'], ['beacon',null,'optics'], ['Field Notebooks','Triangulation','Mineral Cartography'], ['field-notebooks','triangulation','mineral-cartography'], 'observations', 'discovery'],
      ['signal-gallery', 'Signal Gallery', ['Signals','Pulse Relays','Channel Tuning'], ['signals',null,null], ['Dispatch Codes','Logistics Charts','Echo Loops'], ['dispatch-codes','logistics-charts',null], 'signals', 'support'],
      ['command-room', 'Command Room', ['Command','Crew Drills','Shift Rosters'], ['crew',null,null], ['Long Signals','Crew Rotation','Focus Orders'], ['long-signals',null,null], 'instructions', 'efficiency'],
      ['weather-station', 'Weather Station', ['Forecasting','Wind Readings','Observation Windows'], ['forecasting',null,null], ['Weather Stations','Research Exchanges','Sail Observations'], ['weather-stations','research-exchanges',null], 'forecasts', 'support'],
      ['relay-observatory', 'Relay Observatory', ['Relay Grid','Signal Archives','Harmonic Control'], ['relay-grid',null,null], ['Celestial Calendar','Shared Archives','Relay Blueprints'], ['celestial-calendar',null,null], 'relay work', 'support']
    ],
    workshop: [
      ['workbench', 'Workbench', ['Assembly','Bench Fixtures','Cutting Jigs'], ['assembly',null,null], ['Material Hoppers','Template Queue','Assembly Jigs'], ['material-hoppers','template-queue',null], 'assemblies', 'efficiency'],
      ['tool-bench', 'Tool Bench', ['Toolmaking','Cutting Dies','Sharpening Wheels'], ['toolmaking',null,null], ['Standard Tools','Spare Parts','Tempered Edges'], ['standard-tools','spare-parts',null], 'tools', 'support'],
      ['alloy-forge', 'Alloy Forge', ['Metallurgy','Crucible Beds','Flux Mixtures'], ['metallurgy',null,null], ['Offcut Recovery','Slag Reuse','Alloy Recipes'], ['offcut-recovery',null,null], 'alloys', 'efficiency'],
      ['tinker-bay', 'Tinker Bay', ['Mechanisms','Clockwork Drives','Precision Fittings'], ['mechanisms',null,null], ['Instrument Cases','Export Crates','Survey Modules'], ['instrument-cases','export-crates',null], 'mechanisms', 'support'],
      ['pattern-hall', 'Pattern Hall', ['Precision','Replication','Pattern Libraries'], ['precision','replication',null], ['Precision Fixtures','Modular Frames','Pattern Replicas'], ['precision-fixtures','modular-frames',null], 'patterns', 'efficiency']
    ],
    ruins: [
      ['dig-site', 'Dig Site', ['Delving','Field Tools','Strata Reading'], ['delving',null,null], ['Field Camps','Expedition Rigs','Echo Sounding'], ['field-camps','expedition-rigs',null], 'finds', 'discovery'],
      ['archives', 'Archives', ['Archaeology','Reading Tables','Translation Keys'], ['archaeology',null,null], ['Site Catalogues','Survey Tablets','Archive Network'], ['site-catalogues','survey-tablets','archive-network'], 'interpretations', 'discovery'],
      ['recovery-camp', 'Recovery Camp', ['Recovery Teams','Lift Rigs','Conservation Kits'], ['recovery-teams',null,null], ['Careful Recovery','Botanical Remedies','Safe Rigging'], ['careful-recovery','botanical-remedies',null], 'recovered finds', 'value'],
      ['restoration-hall', 'Restoration Hall', ['Restoration','Cleaning Baths','Salvage Grades'], ['restoration',null,null], ['Reclaimed Alloys','Artifact Lenses','Restored Patterns'], ['reclaimed-alloys',null,null], 'restorations', 'support'],
      ['attunement-chamber', 'Attunement Chamber', ['Attunement','Resonance','Focal Crystals'], ['attunement','resonance',null], ['Resonant Pairings','Second Sigil','Memory Stones'], ['resonant-pairings',null,null], 'artifact work', 'support']
    ],
    harbor: [
      ['shipyard', 'Shipyard', ['Shipbuilding','Hull Timbers','Dry Dock'], ['shipbuilding',null,null], ['Provision Packing','Scheduled Convoys','Slipway Cradles'], ['provision-packing','scheduled-convoys',null], 'ship parts', 'efficiency'],
      ['sailing-pier', 'Sailing Pier', ['Seamanship','Sail Rigging','Coastal Knowledge'], ['seamanship',null,null], ['Coastal Tenders','Weather Routing','Trade Winds'], ['coastal-tenders','weather-routing',null], 'passages', 'value'],
      ['cargo-warehouse', 'Cargo Warehouse', ['Stowage','Balanced Crates','Cargo Seals'], ['stowage',null,null], ['Mixed Holds','Salvage Nets','Deepwater Holds'], ['mixed-holds','salvage-nets','deepwater-holds'], 'cargo', 'discovery'],
      ['trade-house', 'Trade House', ['Contracts','Market Brokers','Trade Stamps'], ['contracts',null,null], ['Exchange Houses','Standing Invoices','Return Markets'], ['exchange-houses',null,null], 'contracts', 'value'],
      ['chart-room', 'Chart Room', ['Navigation','Fleet Command','Chart Libraries'], ['navigation','fleet-command',null], ['Twin Manifests','Distant Charts','Navigation Relays'], ['twin-manifests',null,null], 'charts', 'discovery']
    ]
  };
  const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  const PROJECT_GATES = { greenway: ['tower-surveys','rail-network'], quarry: ['sorting-lines','deepworks-commission'], watchtower: ['harbor-observations','ruins-resonators'], workshop: ['alloy-machinery','standardized-production'], ruins: ['restoration-laboratory','deepwater-resonance'], harbor: ['standing-contracts','navigation-artifact'] };
  const SECONDARY = { greenway: 'maps', quarry: 'knowledge', watchtower: 'maps', workshop: 'ore', ruins: 'knowledge', harbor: 'coins' };
  // Every earned technique names a concrete recipient or operation. Values
  // interpolate logarithmically between rank 1 and rank 10; none compound
  // another station's bonuses back into themselves.
  const TECHNIQUES = {
    greenway: [
      [['cadence-tech',null,.2,.35],['byproduct','maps',.05,.15],['link','greenway:scout-post',.2,.5]],
      [['batch',null,.25,.5,12],['link','harbor:trade-house',.1,.25],['byproduct','provisions',.05,.15]],
      [['research',null,.1,.25],['link','watchtower:survey-deck',.15,.35],['byproduct','knowledge',.05,.15]],
      [['saving',null,.08,.2],['byproduct','provisions',.08,.2],['backlink',null,.1,.3]],
      [['link','quarry:hauling',.15,.3],['research',null,.2,.4],['backlink',null,.2,.4]]
    ],
    quarry: [
      [['local',null,.15,.35],['batch',null,.2,.5,8],['byproduct','maps',.05,.15]],
      [['batch',null,.15,.35,12],['link','quarry:tool-forge',.2,.5],['backlink',null,.1,.3]],
      [['local',null,.25,.55],['byproduct','provisions',.05,.15],['byproduct','knowledge',.08,.2]],
      [['batch',null,.2,.5,5],['parallel',null,.25,.5],['backlink',null,.15,.35]],
      [['cadence-tech',null,.15,.3],['byproduct','maps',.1,.25],['link','workshop:pattern-hall',.15,.35]]
    ],
    watchtower: [
      [['research',null,.25,.6],['local',null,.25,.5],['link','quarry:mine',.15,.35]],
      [['link','greenway:porter-camp',.1,.25],['link','quarry:hauling',.15,.35],['batch',null,.1,.3,16]],
      [['backlink',null,.2,.45],['saving',null,.08,.2],['link','watchtower:survey-deck',.15,.4]],
      [['link','harbor:sailing-pier',.15,.35],['byproduct','maps',.1,.25],['link','harbor:shipyard',.1,.3]],
      [['byproduct','maps',.2,.4],['research',null,.3,.65],['backlink',null,.2,.45]]
    ],
    workshop: [
      [['batch',null,.2,.5,10],['cadence-tech',null,.15,.35],['saving',null,.08,.2]],
      [['link','quarry:mine',.2,.45],['link','quarry:hauling',.2,.45],['backlink',null,.15,.35]],
      [['byproduct','ore',.05,.15],['saving',null,.08,.2],['link','workshop:tool-bench',.2,.45]],
      [['byproduct','knowledge',.1,.25],['link','harbor:shipyard',.15,.35],['link','watchtower:survey-deck',.1,.3]],
      [['cadence-tech',null,.2,.5],['backlink',null,.15,.4],['batch',null,.2,.45,24]]
    ],
    ruins: [
      [['batch',null,.2,.45,12],['link','ruins:recovery-camp',.2,.45],['byproduct','ore',.05,.15]],
      [['cadence-tech',null,.15,.35],['byproduct','maps',.1,.3],['research',null,.3,.65]],
      [['local',null,.25,.55],['byproduct','provisions',.1,.3],['backlink',null,.15,.35]],
      [['link','workshop:alloy-forge',.1,.3],['link','watchtower:survey-deck',.15,.35],['link','workshop:pattern-hall',.15,.4]],
      [['backlink',null,.15,.35],['byproduct','maps',.15,.35],['research',null,.2,.5]]
    ],
    harbor: [
      [['saving',null,.08,.2],['batch',null,.1,.25,20],['cadence-tech',null,.15,.35]],
      [['cadence-tech',null,.2,.35],['link','watchtower:weather-station',.1,.3],['backlink',null,.15,.35]],
      [['byproduct','ore',.1,.25],['byproduct','herbs',.1,.25],['local',null,.2,.5]],
      [['link','greenway:porter-camp',.15,.35],['batch',null,.15,.35,24],['backlink',null,.15,.4]],
      [['batch',null,.2,.5,30],['research',null,.2,.5],['link','watchtower:relay-observatory',.15,.35]]
    ]
  };
  const STATIONS = [], SKILLS = [];
  for (const area of AREAS) MATRIX[area.id].forEach((row, index) => {
    const [localId, name, starters, aliases, techniques, techAliases, unit, specialty] = row;
    const id = area.id + ':' + localId;
    const station = { id, localId, areaId: area.id, name, index, icon: area.icon, unit, specialty, secondary: SECONDARY[area.id], baseCycle: [8,12,16,20,24][index], factor: [1,.6,.45,.35,.25][index], project: index >= 3 ? PROJECT_GATES[area.id][index - 3] : null, skillIds: [] };
    for (let n = 0; n < 6; n += 1) {
      const title = n < 3 ? starters[n] : techniques[n - 3], alias = n < 3 ? aliases[n] : techAliases[n - 3];
      const skillId = 'station:' + area.id + ':' + localId + ':' + slug(title);
      const mechanic = n >= 3 ? TECHNIQUES[area.id][index][n - 3] : null;
      const kind = n === 0 ? 'yield' : n === 1 ? 'cadence' : n === 2 ? specialty : mechanic[0];
      const skill = { id: skillId, stationId: id, areaId: area.id, name: title, icon: 'station-' + area.id + '-' + localId + '-' + slug(title), alias, order: n, kind, core: n < 3, maxRank: n < 3 ? 100 : 10, unit, resource: area.resource, secondary: station.secondary, project: n === 5 ? station.project : null,
        target: mechanic?.[1] || null, from: mechanic?.[2] || 0, to: mechanic?.[3] || 0, interval: mechanic?.[4] || 0 };
      station.skillIds.push(skillId); SKILLS.push(skill);
    }
    STATIONS.push(station);
  });
  const AREA_UPGRADES = AREAS.flatMap(area => [
    { id: 'area:' + area.id + ':training', areaId: area.id, name: 'Crew Training', kind: 'training', icon: 'guild', maxRank: 10, effect: '+3% base output at every built station per rank' },
    { id: 'area:' + area.id + ':tools', areaId: area.id, name: 'Shared Tools', kind: 'discount', icon: 'equipment', maxRank: 10, effect: '2% less ordinary station upgrade cost per rank; capped at 20%' },
    { id: 'area:' + area.id + ':shifts', areaId: area.id, name: 'Shift Planning', kind: 'duration', icon: 'focus', maxRank: 15, effect: '+2 seconds of optional area boost per rank; unlimited offline production stays available' }
  ]);
  return { AREAS, STATIONS, SKILLS, AREA_UPGRADES, PROJECT_GATES };
});
