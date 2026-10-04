(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) {
    const host = root.WayfarersContent;
    // The APK supplies trusted document metadata independently of this catalog.
    // Preserve it regardless of whether native bootstrap or this file runs first.
    if (host && Number.isSafeInteger(host.version) && host.version > 0 && typeof host.documentToken === 'string') {
      ['version', 'label', 'apkVersion', 'documentToken', 'recoveryToken', 'restoreFailed'].forEach(key => {
        if (Object.prototype.hasOwnProperty.call(host, key)) api[key] = host[key];
      });
    }
    root.WayfarersContent = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const RESOURCES = [
    { id: 'coins', name: 'Coins', room: 'trail' }, { id: 'ore', name: 'Ore', room: 'mine' },
    { id: 'herbs', name: 'Herbs', room: 'forage' }, { id: 'provisions', name: 'Provisions', room: 'kitchen' },
    { id: 'knowledge', name: 'Knowledge', room: 'study' }, { id: 'maps', name: 'Maps', room: 'cartography' },
    { id: 'notes', name: 'Field notes', room: 'forge' }, { id: 'crests', name: 'Guild crests', room: 'hall' },
    { id: 'starshards', name: 'Starshards', room: 'forge' }
  ];
  const ROOMS = [
    { id: 'trail', name: "Wayfarer's Trail", profession: 'Adventuring', at: 0, description: 'Automatic expeditions earn coins and reveal new territory.' },
    { id: 'mine', name: 'Mine', profession: 'Mining', at: 1, description: 'Mine ore. Stronger picks and deeper routes improve its quality.' },
    { id: 'forge', name: 'Forge', profession: 'Smithing', at: 2, description: 'Spend ore on mining tools or expedition equipment. Both share the same supply.' },
    { id: 'forage', name: "Forager's Camp", profession: 'Foraging', at: 3, description: 'Gather herbs for provisions and research instruments.' },
    { id: 'kitchen', name: 'Kitchen', profession: 'Cooking', at: 4, description: 'Turn herbs into provisions. Choose a meal for the current expedition.' },
    { id: 'study', name: 'Study', profession: 'Scholarship', at: 5, description: 'Discover knowledge and research new rules for familiar professions.' },
    { id: 'hall', name: 'Guild Hall', profession: 'Leadership', at: 6, description: 'Assign two specialists, a companion, and a doctrine to shape the expedition.' },
    { id: 'cartography', name: 'Map Room', profession: 'Cartography', at: 7, description: 'Produce maps, choose route rewards, and chart the endless frontier.' }
  ];
  const REALMS = [
    { id: 'greenway', name: 'Greenway', material: 'Copper', routes: ['Old Footpath', 'Watchtower Road', 'Abandoned Quarry'] },
    { id: 'copperhills', name: 'Copper Hills', material: 'Iron', routes: ['Copper Ridge', 'Forager Crossing', 'Bellstone Ruins'] },
    { id: 'mistwood', name: 'Mistwood', material: 'Silver', routes: ['Mosslight Way', 'Lantern Marsh', 'The Hollow Gate'] },
    { id: 'frostpass', name: 'Frostpass', material: 'Cobalt', routes: ['Frozen Steps', 'Whitewind Bridge', 'Crown of Ice'] },
    { id: 'sunkenreach', name: 'Sunken Reach', material: 'Aetherglass', routes: ['Tidal Causeway', 'Drowned Archive', 'Deepwater Beacon'] },
    { id: 'starfall', name: 'Starfall Heights', material: 'Starsteel', routes: ['Falling-Star Road', 'Observatory Stair', 'The Last Lantern'] }
  ];
  const ROUTES = REALMS.flatMap((realm, r) => realm.routes.map((name, i) => ({
    id: 'route-' + (r * 3 + i), index: r * 3 + i, name, realm: realm.id, realmName: realm.name,
    tier: r + 1, material: realm.material, distance: r === 0 ? [1100, 2200, 14000][i] : 14000 * Math.pow(8, r * 3 + i - 2),
    description: i === 2 ? 'A realm landmark. First completion adds a permanent collection.' : 'An automatic expedition route.'
  })));
  const MODES = [
    { id: 'frontier', name: 'Frontier', description: 'Full travel speed. Push toward the next landmark.' },
    { id: 'supply', name: 'Supply', description: '+85% ore and herbs, +25% coins; travel is 40% slower.' },
    { id: 'discovery', name: 'Discovery', description: '+120% knowledge and maps; travel is 30% slower.' }
  ];
  // A quick, one-control opening. Starter prices meet the established curve;
  // the opening's stronger gains taper smoothly to the normal gain per rank.
  const OPENING = {
    routeDistances: [120, 450],
    forgeOre: 8,
    discounts: { boots: { firstCost: 2.4, untilRank: 16 }, miners: { firstCost: 12, untilRank: 8 }, 'gear-tools': { firstCost: 1, untilRank: 8 }, 'gear-boots': { firstCost: 2, untilRank: 8 } },
    boots: { untilRank: 8, bonus: 0.17, decay: 0.5 }
  };
  const UPGRADES = [
    { id: 'boots', room: 'trail', name: 'Improve travel boots', description: '+28% travel and +18% coins per level.', resource: 'coins', base: 20, scale: 1.65 },
    { id: 'preparation', room: 'trail', name: 'Field preparation', description: '+15% travel per level. Rebuilt after Refit.', resource: 'coins', base: 150, scale: 1.9, at: 2 },
    { id: 'miners', room: 'mine', name: 'Expand the mining team', description: '+32% ore per level.', resource: 'coins', base: 90, scale: 1.8 },
    { id: 'forge', room: 'forge', name: 'Improve the workshop', description: 'Makes equipment 5% cheaper per level and speeds smithing mastery.', resource: 'coins', base: 250, scale: 2.1 },
    { id: 'foragers', room: 'forage', name: 'Equip the foragers', description: '+35% herbs per level.', resource: 'coins', base: 600, scale: 2 },
    { id: 'cooks', room: 'kitchen', name: 'Expand the kitchen', description: '+30% provision output and herb demand per level.', resource: 'coins', base: 1500, scale: 2.1 },
    { id: 'scholars', room: 'study', name: 'Train a research circle', description: '+35% knowledge per level.', resource: 'coins', base: 5000, scale: 2.15 },
    { id: 'surveyors', room: 'cartography', name: 'Equip surveyors', description: '+35% map production per level.', resource: 'coins', base: 14000, scale: 2.1 },
    { id: 'mentors', room: 'hall', name: 'Train guild mentors', description: '+8% mastery experience in every profession per level.', resource: 'knowledge', base: 25, scale: 2 },
    { id: 'gear-tools', room: 'forge', name: 'Reinforce mining tools', description: '+45% ore per reinforcement. Competes with boots for ore.', resource: 'ore', base: 5, scale: 1.75, equipment: true },
    { id: 'gear-boots', room: 'forge', name: 'Reinforce expedition boots', description: '+35% travel and +20% coins per reinforcement.', resource: 'ore', base: 10, scale: 1.8, equipment: true },
    { id: 'gear-instruments', room: 'forge', name: 'Craft survey instruments', description: '+40% knowledge and maps per reinforcement; also costs herbs.', resource: 'ore', base: 80, scale: 2, equipment: true, at: 5 }
  ];
  const RESEARCH = [
    { id: 'auto-work', name: 'Workshop ledgers', cost: 20, description: 'Unlock automatic room upgrades and travel boots. Purchases keep a coin reserve.', at: 5 },
    { id: 'auto-forge', name: 'Standing forge orders', cost: 40, description: 'Unlock automatic equipment purchases with a tools/boots priority.', at: 5 },
    { id: 'efficient-smelting', name: 'Efficient smelting', cost: 60, description: 'Equipment costs 35% less ore.', at: 5 },
    { id: 'field-notes', name: 'Field journals', cost: 100, description: 'New route discoveries grant extra knowledge; Refit notes +25%.', at: 6 },
    { id: 'balanced-meals', name: 'Lasting provisions', cost: 150, description: 'Meals use half as many provisions.', at: 6 },
    { id: 'ore-conversion', name: 'Alternative alloys', cost: 250, description: 'Unlock a recipe exchanging herbs for ore; useful on supply runs.', at: 7 },
    { id: 'map-survey', name: 'Field surveying', cost: 400, description: 'Cartography mastery improves ore output; unlock map-funded survey commissions.', at: 7 },
    { id: 'auto-route', name: 'Route dispatch', cost: 600, description: 'Automatically enter the next unfinished route after completion.', at: 8 },
    { id: 'smart-reserve', name: 'Resource reservations', cost: 1000, description: 'Automated forging preserves enough ore for the next expedition-boots upgrade.', at: 9 },
    { id: 'specialist-training', name: 'Shared apprenticeships', cost: 1800, description: 'Specialists gain a second benefit from mastery in a related profession.', at: 10 },
    { id: 'frontier-compass', name: 'Frontier compass', cost: 4000, description: 'Infinite frontier milestones improve every gathering tier.', at: 14 }
  ];
  const RECIPES = [
    { id: 'meal-none', room: 'kitchen', name: 'Save provisions', description: 'No meal consumed. Preserve supplies for a later push.' },
    { id: 'meal-travel', room: 'kitchen', name: 'Trail stew', description: '+50% travel while supplied. Demand grows with expedition scale; the supply plan shows current consumption.' },
    { id: 'meal-study', room: 'kitchen', name: 'Scholar tea', description: '+90% knowledge while supplied. Demand grows with expedition scale; the supply plan shows current consumption.' },
    { id: 'meal-mining', room: 'kitchen', name: "Miner's lunch", description: '+75% ore while supplied. Demand grows with expedition scale; the supply plan shows current consumption.' },
    { id: 'alloy', room: 'forge', name: 'Make alternative alloy', description: 'Trade 50 herbs for 20 ore, scaled by material tier.', research: 'ore-conversion' },
    { id: 'survey', room: 'cartography', name: 'Survey commission', description: 'Trade maps for knowledge and coins; retains all mastery.', research: 'map-survey' }
  ];
  const SPECIALISTS = [
    { id: 'scout', name: 'Scout', description: '+40% travel; trained scouts use mining mastery for another travel bonus.' },
    { id: 'prospector', name: 'Prospector', description: '+65% ore; trained prospectors also improve herb gathering.' },
    { id: 'naturalist', name: 'Naturalist', description: '+70% herbs and +25% provisions.' },
    { id: 'scholar', name: 'Scholar', description: '+65% knowledge; trained scholars gain maps from cooking mastery.' },
    { id: 'quartermaster', name: 'Quartermaster', description: 'Half provision consumption and 20% cheaper equipment.' }
  ];
  const COMPANIONS = [
    { id: 'fox', name: 'Trail fox', description: '+20% travel and +30% coins.' },
    { id: 'owl', name: 'Archive owl', description: '+40% knowledge and +20% maps.' },
    { id: 'tortoise', name: 'Pack tortoise', description: '+35% ore and herbs, and half provision consumption.' }
  ];
  const DOCTRINES = [
    { id: 'balanced', name: 'Open Roads', description: 'Balanced output and evenly ordered automatic purchases.' },
    { id: 'industry', name: 'Patient Industry', description: '+45% ore and herbs, -15% travel. Forge automation favors tools.' },
    { id: 'expedition', name: 'Far Horizons', description: '+35% travel, -20% knowledge. Forge automation favors boots.' },
    { id: 'scholarship', name: 'Shared Discovery', description: '+50% knowledge and maps, -15% coins.' }
  ];
  const CHALLENGES = [
    { id: 'light-pack', name: 'Light Pack', description: 'Reach route 7 without consuming meals. Reward: permanently 15% less provision demand.', target: 6, at: 7 },
    { id: 'old-tools', name: 'Old Tools', description: 'Reach route 10 with equipment reinforcements inactive. Reward: permanently 15% cheaper ore equipment.', target: 9, at: 10 },
    { id: 'quiet-company', name: 'Quiet Company', description: 'Reach route 13 with specialists and companion inactive. Reward: permanently +20% mastery experience.', target: 12, at: 13 }
  ];
  const REFIT_UPGRADES = [
    { id: 'pace', name: 'Established trails', description: '+20% travel per level.', base: 2, scale: 2.5 },
    { id: 'supply', name: 'Reliable suppliers', description: '+25% ore and herbs per level.', base: 2, scale: 2.5 },
    { id: 'insight', name: 'Shared fieldwork', description: '+25% knowledge and maps per level.', base: 3, scale: 2.5 }
  ];
  const LEGACY_UPGRADES = [
    { id: 'foundations', name: 'Guild foundations', description: '+30% coins and material output per level; start each Charter with stronger operations.', base: 1, scale: 3 },
    { id: 'curriculum', name: 'Living curriculum', description: '+35% knowledge and mastery experience per level.', base: 1, scale: 3 },
    { id: 'waystones', name: 'Waystone network', description: '+35% travel per level; cartography mastery adds another travel bonus.', base: 1, scale: 3 }
  ];
  const AUTOMATIONS = [
    { id: 'operations', name: 'Upgrade operations', research: 'auto-work', description: 'Every planning minute, buy useful room upgrades while retaining 20% of coins.' },
    { id: 'equipment', name: 'Forge equipment', research: 'auto-forge', description: 'Every planning minute, follow doctrine priorities; reservations protect expedition boots.' },
    { id: 'routes', name: 'Dispatch routes', research: 'auto-route', description: 'Continue to the next route on completion. Never performs a prestige.' }
  ];
  const PREMIUM_ITEMS = [
    { id: 'compass', name: 'Wayfarer’s Compass', cost: 10, kind: 'charm', description: 'Permanently +10% travel. One lasting charm; survives every Refit and Charter.' },
    { id: 'artisan', name: 'Artisan’s Charm', cost: 10, kind: 'charm', description: 'Permanently +10% ore, herbs, and provision output. Does not increase herb consumption.' },
    { id: 'scholar', name: 'Scholar’s Charm', cost: 10, kind: 'charm', description: 'Permanently +10% knowledge and maps. One lasting charm; survives every Refit and Charter.' },
    { id: 'banner-amber', name: 'Amber Pennant', cost: 5, kind: 'banner', description: 'A warm amber guild banner. Cosmetic only; select one owned banner at a time.' },
    { id: 'banner-moon', name: 'Moonlit Pennant', cost: 5, kind: 'banner', description: 'A moonlit blue guild banner. Cosmetic only; select one owned banner at a time.' }
  ];
  const PREMIUM_MILESTONES = [
    { id: 'first-refit', amount: 3 }, { id: 'first-charter', amount: 5 },
    ...CHALLENGES.map(challenge => ({ id: 'challenge-' + challenge.id, amount: 2 }))
  ];
  const RELICS = [
    { id: 'golden-pickaxe', name: 'Golden Pickaxe', rarity: 'rare', weight: 65, chapter: 0, family: 'Founders', description: 'While equipped, ore production is 50% stronger.' },
    { id: 'surveyors-lens', name: 'Surveyor’s Lens', rarity: 'epic', weight: 30, chapter: 0, family: 'Founders', description: 'Map Room mastery strengthens knowledge; Study mastery strengthens maps. Each bonus has a fixed ceiling.' },
    { id: 'living-crucible', name: 'Living Crucible', rarity: 'legendary', weight: 5, chapter: 0, family: 'Founders', description: 'Turn ore, provisions, and knowledge into a prepared mining or travel kit. Its burst works while this relic is equipped.' },
    { id: 'marsh-lantern', name: 'Marsh Lantern', rarity: 'rare', weight: 30, chapter: 1, family: 'Regional Guides', description: 'Herb gathering +30% and meal provision demand −40%. A conservation choice for hungry expeditions.' },
    { id: 'frost-compass', name: 'Frost Compass', rarity: 'epic', weight: 15, chapter: 1, family: 'Regional Guides', description: 'A scouted route receives another +40% travel. No bonus without spending maps on scouting.' },
    { id: 'archive-quill', name: 'Archive Quill', rarity: 'epic', weight: 15, chapter: 2, family: 'Archive', description: 'A surveyed route receives another +50% knowledge and +35% maps. Supports ongoing survey funding.' },
    { id: 'starsteel-anvil', name: 'Starsteel Anvil', rarity: 'legendary', weight: 8, chapter: 3, family: 'Highland Craft', description: 'Alternative alloy recipes produce three times the ore; equipment costs 20% less ore. A conversion build for herb-rich guilds.' },
    { id: 'wayfarer-standard', name: 'Wayfarer’s Standard', rarity: 'legendary', weight: 5, chapter: 4, family: 'Beyond the Lantern', description: 'Any prepared route earns +60% coins and +25% travel. Trades specialization for a balanced prepared expedition.' }
  ];
  const LUCK_RESEARCH = [
    { id: 'careful-salvage', name: 'Careful salvage', cost: 120, description: 'Common supply finds contain 25% more resources.' },
    { id: 'relic-lore', name: 'Relic lore', cost: 500, description: 'Future relic searches average four hours instead of six; the guarantee falls from eight hours to six.' },
    { id: 'duplicate-study', name: 'Study familiar relics', cost: 1500, description: 'Duplicate relics give 25% progress toward a missing relic instead of 20%.', room: 'cartography' }
  ];
  const KITS = [
    { id: 'mining', name: 'Mining kit', description: 'Double ore production for 30 expedition minutes while the Living Crucible is equipped.' },
    { id: 'travel', name: 'Trail kit', description: 'Double travel for 30 expedition minutes while the Living Crucible is equipped.' }
  ];
  const PROJECTS = [
    { id: 'study-foundation', name: 'Build the Study', room: 'study', at: 5, costs: { ore: 700, herbs: 1200, provisions: 500 }, description: 'Build a place to turn supplies into research, new working rules, and relic discoveries.' },
    { id: 'hall-foundation', name: 'Establish the Guild Hall', room: 'hall', at: 6, costs: { ore: 20000, provisions: 18000, knowledge: 7000 }, description: 'Recruit a two-person specialist team, choose a companion, and set a guild doctrine.' },
    { id: 'maps-foundation', name: 'Open the Map Room', room: 'cartography', at: 7, costs: { ore: 150000, provisions: 100000, knowledge: 45000 }, description: 'Plan expeditions with map allocations, regional preparations, and targeted relic hunts.' }
  ];
  const CAPABILITIES = [
    { id: 'purchase-queue', name: 'Standing purchase plan', resource: 'notes', cost: 2, at: 0, description: 'Queue up to six chosen upgrades, research items, or projects. These are purchased before routine work.' },
    { id: 'kit-plan', name: 'Expedition outfitter', resource: 'notes', cost: 4, at: 0, description: 'Prepare and renew one chosen Crucible kit automatically, respecting your reserves and saved objective.' },
    { id: 'loadouts', name: 'Guild playbooks', resource: 'crests', cost: 1, at: 1, description: 'Save three complete plans: crew, companion, doctrine, meal, relic, supplies, route mode, reserves, and purchase priorities.' },
    { id: 'dispatch-preparation', name: 'Survey dispatch', resource: 'crests', cost: 1, at: 1, description: 'Buy your chosen route preparation automatically when resources permit, preserving your reserves.' },
    { id: 'regional-logistics', name: 'Regional logistics', resource: 'crests', cost: 2, at: 2, description: 'Supply preparations also strengthen herbs by 30%; regional provision demand is reduced by 20%.' },
    { id: 'archive-network', name: 'Archive network', resource: 'crests', cost: 2, at: 3, description: 'Survey preparations add 40% map production, creating a reusable discovery expedition plan.' }
  ];
  const PREPARATIONS = [
    { id: 'scout', name: 'Scout a safe route', description: 'Spend maps for +35% travel on this route. Also bypasses the Frostpass detour.', maps: 20 },
    { id: 'supply', name: 'Stock a field camp', description: 'Spend provisions and maps for +50% ore on this route. Also bypasses the Mistwood detour.', maps: 10, provisions: 100 },
    { id: 'survey', name: 'Survey the region', description: 'Spend maps and knowledge for +60% knowledge on this route. Also bypasses the Sunken Reach detour.', maps: 30, knowledge: 40 }
  ];
  // Presentation copy names actual game prerequisites. Eligibility remains in
  // the core so a label cannot accidentally unlock an economic action.
  const PRESENTATION_SYSTEMS = [
    { id: 'mine', label: 'Mine and Guild', effect: 'The Guild opens with a Mine. Ore arrives automatically; improve its mining team with coins.', action: { type: 'ui', tab: 'guild', room: 'mine' } },
    { id: 'finds', label: 'Trail finds and Journal', effect: 'The Journal records supplies found during travel. Rewards are already granted; opening it does not claim them.', action: { type: 'ui', tab: 'finds', journal: 'discoveries' } },
    { id: 'forge', label: 'Forge', effect: 'Forge equipment spends ore. Choose stronger mining tools or faster expedition boots.', action: { type: 'ui', tab: 'guild', room: 'forge' } },
    { id: 'forage', label: 'Forager’s Camp', effect: 'Gather herbs automatically for cooking and research instruments.', action: { type: 'ui', tab: 'guild', room: 'forage' } },
    { id: 'collections', label: 'Relic collections', effect: 'Review owned relics and their families. Relics and collection rewards persist through renewals.', action: { type: 'ui', tab: 'finds', journal: 'collections' } },
    { id: 'refit', label: 'Expedition Refit', effect: 'Preview what stays and renews. Refit earns Field notes and unlocks automatic operation purchases.', action: { type: 'ui', tab: 'journey', journal: 'renewals' } },
    { id: 'planning', label: 'Guild plans', effect: 'Choose an objective, reserve resources, and set automatic purchase priorities. The plan continues offline.', action: { type: 'ui', tab: 'planning' } },
    { id: 'keepsakes', label: 'Starshard keepsakes', effect: 'Spend earned Starshards on lasting charms or banners. Every item can be earned through play.', action: { type: 'ui', tab: 'shop' } },
    { id: 'kitchen', label: 'Kitchen and meals', effect: 'Turn herbs into provisions. Choose a meal bonus or save supplies for later.', action: { type: 'ui', tab: 'guild', room: 'kitchen' } },
    { id: 'study', label: 'Study and research', effect: 'Knowledge funds permanent research that changes how your professions work.', action: { type: 'ui', tab: 'research' } },
    { id: 'relics', label: 'Relic discoveries', effect: 'The Study begins a guaranteed first relic search. Equip one relic to change the guild’s strengths.', action: { type: 'ui', tab: 'finds', journal: 'discoveries' } },
    { id: 'caravan', label: 'Travelling caravan', effect: 'An arrival is waiting. Review its guaranteed optional reward before choosing whether to watch an ad.', action: { type: 'ui', tab: 'caravan' } },
    { id: 'hall', label: 'Guild Hall and crew', effect: 'Recruit two specialists, choose a companion, and select a doctrine for the expedition.', action: { type: 'ui', tab: 'crew' } },
    { id: 'contracts', label: 'Guild contracts', effect: 'Optional challenge expeditions impose a clear constraint and grant a lasting completion reward.', action: { type: 'ui', tab: 'journey', journal: 'challenges' } },
    { id: 'cartography', label: 'Map Room and route plans', effect: 'Choose route rewards, spend maps on one-route preparations, and target missing relics.', action: { type: 'ui', tab: 'guild', room: 'cartography' } },
    { id: 'charter', label: 'Guild Charter', effect: 'Review the deeper renewal. Crests buy lasting capabilities; the preview lists every retained and reset system.', action: { type: 'ui', tab: 'journey', journal: 'renewals' } }
  ];
  return { RESOURCES, ROOMS, REALMS, ROUTES, MODES, OPENING, UPGRADES, RESEARCH, RECIPES, SPECIALISTS, COMPANIONS, DOCTRINES, CHALLENGES, REFIT_UPGRADES, LEGACY_UPGRADES, AUTOMATIONS, PREMIUM_ITEMS, PREMIUM_MILESTONES, RELICS, LUCK_RESEARCH, KITS, PROJECTS, CAPABILITIES, PREPARATIONS, PRESENTATION_SYSTEMS };
});
