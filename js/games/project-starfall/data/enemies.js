(function initProjectStarfallDataEnemies(global) {
  'use strict';

  function createEmptyDropPool() {
    return Object.freeze({
      materials: Object.freeze([]),
      equipment: Object.freeze([]),
      consumables: Object.freeze([]),
      cards: Object.freeze([]),
      currencyWeight: 0,
      globalRareEligible: false,
      basicConsumables: true
    });
  }

  function createEnemyData(options) {
    const settings = options || {};
    const ENEMY_ASSETS = settings.ENEMY_ASSETS || {};
    const ENEMY_ANIMATION_ASSETS = settings.ENEMY_ANIMATION_ASSETS || {};
    const MONSTER_DROP_POOLS = settings.MONSTER_DROP_POOLS || {};
    const MONSTER_GUIDE_FUTURE_ENEMY_IDS = settings.MONSTER_GUIDE_FUTURE_ENEMY_IDS || Object.freeze([]);
    const MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS = settings.MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS || Object.freeze([]);
    const getEnemyAnimationBehavior = typeof settings.getEnemyAnimationBehavior === 'function'
      ? settings.getEnemyAnimationBehavior
      : () => 'melee';
    const createFallbackDropPool = typeof settings.createFallbackDropPool === 'function'
      ? settings.createFallbackDropPool
      : createEmptyDropPool;

    function attachEnemyDropPool(enemy) {
      return Object.freeze(Object.assign({}, enemy, {
        dropPool: MONSTER_DROP_POOLS[enemy.id] || createFallbackDropPool()
      }));
    }

    function attachEnemyAssets(enemy) {
      const animationBehavior = getEnemyAnimationBehavior(enemy);
      return Object.freeze(Object.assign({}, enemy, {
        asset: ENEMY_ASSETS[enemy.id] || '',
        animation: ENEMY_ANIMATION_ASSETS[enemy.id] || null,
        animationBehavior
      }));
    }

    function uniqueGuideValues(values) {
      return Object.freeze(Array.from(new Set((values || [])
        .map((value) => String(value || '').trim())
        .filter(Boolean))));
    }

    function enemyGuideText(enemy) {
      return [
        enemy && enemy.id,
        enemy && enemy.name,
        enemy && enemy.family,
        enemy && enemy.role,
        enemy && enemy.behavior,
        enemy && enemy.mechanic,
        enemy && enemy.counter,
        (enemy && enemy.drops || []).join(' ')
      ].join(' ').toLowerCase();
    }

    function inferMonsterGuideCategory(enemy) {
      if (enemy && enemy.behavior === 'boss') return 'boss';
      if (enemy && (enemy.behavior === 'elite' || enemy.id === 'crackedMimic' || enemy.id === 'riftAberration')) return 'elite';
      return 'normal';
    }

    function inferMonsterGuideThreatTier(enemy) {
      const category = inferMonsterGuideCategory(enemy);
      if (category === 'boss') return 'Boss';
      if (category === 'elite') return 'Elite';
      const range = Array.isArray(enemy && enemy.levelRange) ? enemy.levelRange : [1, 1];
      const maxLevel = Math.max(1, Number(range[1] || range[0]) || 1);
      if (maxLevel >= 90) return 'Astral';
      if (maxLevel >= 55) return 'Dangerous';
      if (maxLevel >= 25) return 'Veteran';
      return 'Field';
    }

    function inferMonsterGuideRegionTags(enemy) {
      const text = enemyGuideText(enemy);
      const tags = [];
      if (/slime|moss|thorn|vine|briar|bramble|boar/.test(text)) tags.push('Greenroot');
      if (/clock|rust|coil|scrap|ore|quarry|titan/.test(text)) tags.push('Rustcoil');
      if (/ember|ash|lava|cinder|volcanic/.test(text)) tags.push('Cinder');
      if (/frost|rime|shard|glacier|snow|ice/.test(text)) tags.push('Frostfen');
      if (/gale|storm|thunder|cloud|roc/.test(text)) tags.push('Stormbreak');
      if (/astral|index|lumen|void|archivist/.test(text)) tags.push('Astral');
      if (/eclipse|rift|sovereign|duelist/.test(text)) tags.push('Eclipse');
      if (/bandit|dust/.test(text)) tags.push('Bandit Ridge');
      return uniqueGuideValues(tags.length ? tags : ['Frontier']);
    }

    function inferMonsterGuideWeaknesses(enemy) {
      const text = enemyGuideText(enemy);
      const values = [];
      if (/fire/.test(text)) values.push('Fire');
      if (/cold|ice|frost/.test(text)) values.push('Cold');
      if (/lightning|storm/.test(text)) values.push('Lightning');
      if (/armor break|armor crack|weak-point|weak point/.test(text)) values.push('Armor Break');
      if (/stun|interrupt|seal/.test(text)) values.push('Control');
      if (/range|ranged|marks/.test(text)) values.push('Ranged Uptime');
      if (/aoe|area/.test(text)) values.push('Area Damage');
      if (/back attacks|back attack/.test(text)) values.push('Back Attacks');
      return uniqueGuideValues(values.length ? values : ['Clean positioning']);
    }

    function inferMonsterGuideResistances(enemy) {
      const text = enemyGuideText(enemy);
      const values = [];
      if (/volcanic|ember|ash|lava|cinder/.test(text)) values.push('Fire');
      if (/frost|rime|glacier|ice|snow/.test(text)) values.push('Cold');
      if (/construct|armor|armored|plate|sentinel/.test(text)) values.push('Physical');
      if (/astral|void|eclipse/.test(text)) values.push('Arcane');
      return uniqueGuideValues(values);
    }

    function inferMonsterGuideStatusVulnerabilities(enemy) {
      const text = enemyGuideText(enemy);
      const values = [];
      if (/charger|skirmisher|flyer|thrower/.test(text)) values.push('Slow');
      if (/healer|thrower|turret|boss/.test(text)) values.push('Interrupt');
      if (/blocker|armored|construct/.test(text)) values.push('Armor Crack');
      if (/ambush|flee|retreat/.test(text)) values.push('Stun');
      return uniqueGuideValues(values.length ? values : ['Knockback']);
    }

    function inferMonsterGuideAttackPatterns(enemy) {
      const behavior = String(enemy && enemy.behavior || '');
      const mechanic = String(enemy && enemy.mechanic || '');
      const patterns = {
        hopper: ['Contact hop', 'Swarm pressure'],
        bruiser: ['Slow melee', 'Brace window'],
        turret: ['Stationary projectile', 'Line break'],
        skirmisher: ['Burst movement', 'Retreat window'],
        charger: ['Telegraphed charge', 'Recovery window'],
        armored: ['Armor crack', 'Slow pressure'],
        flyer: ['Floating projectile', 'Ranged uptime check'],
        thrower: ['Arcing projectile', 'Kiting'],
        blocker: ['Frontal guard', 'Counter swing'],
        healer: ['Ally heal', 'Priority target'],
        elite: ['Ambush', 'Reward burst'],
        boss: ['Phase attacks', 'Arena mechanics']
      };
      return uniqueGuideValues((patterns[behavior] || ['Melee pressure']).concat(mechanic ? [mechanic] : []));
    }

    function inferMonsterGuideBehaviorTags(enemy) {
      const text = enemyGuideText(enemy);
      const tags = [enemy && enemy.behavior, enemy && enemy.family, inferMonsterGuideCategory(enemy)];
      if (/ranged|throw|turret|projectile|bolt|arrow/.test(text)) tags.push('Ranged');
      if (/fly|wisp|harrier|roc/.test(text)) tags.push('Flying');
      if (/heal|support|oracle|acolyte/.test(text)) tags.push('Support');
      if (/armor|block|plate|shell|construct/.test(text)) tags.push('Armored');
      if (/charge|dash|lunge/.test(text)) tags.push('Mobility');
      if (/boss/.test(text)) tags.push('Boss');
      return uniqueGuideValues(tags);
    }

    function inferMonsterGuideLore(enemy) {
      const regions = inferMonsterGuideRegionTags(enemy);
      const family = String(enemy && enemy.family || 'Monster');
      const name = String(enemy && enemy.name || 'Unknown monster');
      return `${name} is a ${family.toLowerCase()} threat documented by frontier scouts around ${regions[0] || 'the frontier'}.`;
    }

    function inferMonsterGuideRespawnClass(enemy) {
      const category = inferMonsterGuideCategory(enemy);
      if (category === 'boss') return 'Dungeon boss';
      if (category === 'elite') return 'Rare elite';
      if (enemy && enemy.behavior === 'turret') return 'Fixed field post';
      return 'Field pack';
    }

    function attachEnemyGuideMetadata(enemy) {
      const id = String(enemy && enemy.id || '');
      const future = MONSTER_GUIDE_FUTURE_ENEMY_IDS.includes(id);
      const excluded = MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS.includes(id);
      return Object.freeze(Object.assign({}, enemy, {
        guide: Object.freeze({
          category: inferMonsterGuideCategory(enemy),
          excludedFromCollection: excluded,
          visibility: future ? 'future' : excluded ? 'debug' : 'live',
          threatTier: inferMonsterGuideThreatTier(enemy),
          regionTags: inferMonsterGuideRegionTags(enemy),
          biomeTags: inferMonsterGuideRegionTags(enemy),
          behaviorTags: inferMonsterGuideBehaviorTags(enemy),
          weaknesses: inferMonsterGuideWeaknesses(enemy),
          resistances: inferMonsterGuideResistances(enemy),
          statusVulnerabilities: inferMonsterGuideStatusVulnerabilities(enemy),
          attackPatterns: inferMonsterGuideAttackPatterns(enemy),
          lore: inferMonsterGuideLore(enemy),
          spawnConditions: Object.freeze(future ? ['Future content'] : ['Standard field spawn']),
          respawnClass: inferMonsterGuideRespawnClass(enemy),
          questTags: Object.freeze(inferMonsterGuideCategory(enemy) === 'boss' ? ['Dungeon clear'] : [])
        })
      }));
    }

    const ENEMIES = Object.freeze([
      { id: 'glassback', name: 'Glassback', levelRange: [1, 8], role: 'Starter charge lesson', family: 'Meteor Scavenger', hpMult: 0.9, damageMult: 0.62, defenseMult: 0.82, expMult: 0.82, speed: 64, behavior: 'charger', mechanic: 'Its star-glass mantle flares before a short committed charge.', counter: 'Step or jump past the tell, then punish the recovery.', drops: ['Star-glass Chip', 'Training gear'] },
      { id: 'riftLantern', name: 'Rift Lantern', levelRange: [2, 9], role: 'Starter vertical pressure', family: 'Rift Fauna', hpMult: 0.72, damageMult: 0.64, defenseMult: 0.58, expMult: 0.88, speed: 54, behavior: 'flyer', mechanic: 'Its core focuses before releasing a slow, readable spark.', counter: 'Reposition vertically or close the distance during its aim tell.', drops: ['Lantern Core', 'Focus accessory'] },
      { id: 'faultSkitter', name: 'Fault Skitter', levelRange: [1, 7], role: 'Starter ground cleanup', family: 'Faultborn Arthropod', hpMult: 0.58, damageMult: 0.48, defenseMult: 0.55, expMult: 0.62, speed: 84, behavior: 'melee', mechanic: 'Jittering ground movement keeps the lower lane active while Rift Lanterns pressure above.', counter: 'Sweep its low profile with a basic attack or briefly control the pack.', drops: ['Star-glass Chip', 'Training gear'] },
      { id: 'slimelet', name: 'Slimelet', levelRange: [1, 6], role: 'Basic/swarm-light', family: 'Ooze', hpMult: 0.75, damageMult: 0.6, defenseMult: 0.6, expMult: 0.7, speed: 52, behavior: 'hopper', mechanic: 'Slow hop contact.', counter: 'Any basic attack, AoE.', drops: ['Gel Drop', 'Training gear'] },
      { id: 'dewSlime', name: 'Dew Slime', levelRange: [1, 8], role: 'Starter swarm', family: 'Ooze', hpMult: 0.68, damageMult: 0.55, defenseMult: 0.55, expMult: 0.72, speed: 58, behavior: 'hopper', mechanic: 'Quick wet hops and low-contact pressure.', counter: 'Basic attacks, short AoE, and knockback.', drops: ['Gel Drop', 'Dew Bead', 'Training gear'] },
      { id: 'mossback', name: 'Mossback', levelRange: [4, 10], role: 'Durable melee', family: 'Beast', hpMult: 1.25, damageMult: 0.85, defenseMult: 1.2, expMult: 1.1, speed: 42, behavior: 'bruiser', mechanic: 'Braces against knockback.', counter: 'Fire, armor break.', drops: ['Moss Hide', 'Guard Ring'] },
      { id: 'thornSprout', name: 'Thorn Sprout', levelRange: [5, 12], role: 'Stationary ranged', family: 'Plant', hpMult: 0.7, damageMult: 0.85, defenseMult: 0.7, expMult: 1, speed: 0, behavior: 'turret', mechanic: 'Fires dodgeable thorns.', counter: 'Fire, range, line attacks.', drops: ['Thorn Fiber', 'Focus Ring'] },
      { id: 'vineSnapper', name: 'Vine Snapper', levelRange: [8, 22], role: 'Ambush skirmisher', family: 'Plant', hpMult: 0.95, damageMult: 1, defenseMult: 0.85, expMult: 1.08, speed: 92, behavior: 'skirmisher', mechanic: 'Lunges from brush, then retreats to cover.', counter: 'Fire, traps, and quick ranged tags.', drops: ['Vine Fiber', 'Upgrade Dust', 'Focus accessory'] },
      { id: 'bristleBoar', name: 'Bristle Boar', levelRange: [8, 16], role: 'Charger', family: 'Beast', hpMult: 1.15, damageMult: 1.25, defenseMult: 1, expMult: 1.2, speed: 78, behavior: 'charger', mechanic: 'Telegraph charge.', counter: 'Jumping, traps, blocks.', drops: ['Bristle Hide', 'Traveler Boots'] },
      { id: 'briarStag', name: 'Briar Stag', levelRange: [12, 28], role: 'Heavy charger', family: 'Beast / Plant', hpMult: 1.3, damageMult: 1.22, defenseMult: 1.05, expMult: 1.24, speed: 84, behavior: 'charger', mechanic: 'Antler charge leaves a brief thorn hazard trail.', counter: 'Jump timing, slows, and burst after the charge.', drops: ['Briar Antler', 'Traveler Boots', 'Upgrade Catalyst'] },
      { id: 'dustImp', name: 'Dust Imp', levelRange: [10, 18], role: 'Fast melee', family: 'Imp', hpMult: 0.85, damageMult: 1, defenseMult: 0.75, expMult: 1.1, speed: 118, behavior: 'skirmisher', mechanic: 'Leap slash and retreat.', counter: 'AoE, traps, fast hits.', drops: ['Dust Claw', 'Upgrade Dust'] },
      { id: 'clockbug', name: 'Clockbug', levelRange: [12, 22], role: 'Armored tank', family: 'Construct', hpMult: 1.7, damageMult: 0.8, defenseMult: 1.7, expMult: 1.4, speed: 38, behavior: 'armored', mechanic: 'Armor crack state.', counter: 'Armor break, bossing skills.', drops: ['Clockwork Scrap', 'Focus Amulet'] },
      { id: 'rustRatchet', name: 'Rust Ratchet', levelRange: [12, 26], role: 'Construct skirmisher', family: 'Construct', hpMult: 0.9, damageMult: 1.04, defenseMult: 1, expMult: 1.12, speed: 104, behavior: 'skirmisher', mechanic: 'Skates along gear rails and snaps at close range.', counter: 'Area control, slows, and armor break.', drops: ['Clockwork Scrap', 'Upgrade Dust', 'Rustcoil gear'] },
      { id: 'coilSentry', name: 'Coil Sentry', levelRange: [14, 32], role: 'Stationary construct ranged', family: 'Construct', hpMult: 0.95, damageMult: 1.05, defenseMult: 1.35, expMult: 1.2, speed: 0, behavior: 'turret', mechanic: 'Charges visible electric bolts from fixed posts.', counter: 'Line-of-sight breaks, burst, and ranged attacks.', drops: ['Charged Coil', 'Focus Amulet', 'Upgrade Catalyst'] },
      { id: 'scrapWarden', name: 'Scrap Warden', levelRange: [24, 46], role: 'Armored blocker', family: 'Construct / Humanoid', hpMult: 1.55, damageMult: 1.08, defenseMult: 1.45, expMult: 1.36, speed: 56, behavior: 'blocker', mechanic: 'Raises a shield plate before short counter-swings.', counter: 'Back attacks, armor break, and stun windows.', drops: ['Scrap Plate', 'Guard Ring', 'Rustcoil gear'] },
      { id: 'emberWisp', name: 'Ember Wisp', levelRange: [16, 26], role: 'Flying ranged', family: 'Spirit', hpMult: 0.85, damageMult: 0.95, defenseMult: 0.7, expMult: 1.15, speed: 70, behavior: 'flyer', mechanic: 'Floating firebolt.', counter: 'Marks, lightning, ranged attacks.', drops: ['Ember Dust', 'Ember Ring'] },
      { id: 'ashCrawler', name: 'Ash Crawler', levelRange: [16, 40], role: 'Volcanic bruiser', family: 'Volcanic Beast', hpMult: 1.35, damageMult: 1.02, defenseMult: 1.18, expMult: 1.22, speed: 46, behavior: 'bruiser', mechanic: 'Shrugs off light hits while ember plates cool.', counter: 'Cold pressure, armor break, and sustained hits.', drops: ['Ash Carapace', 'Upgrade Dust', 'Cinder gear'] },
      { id: 'lavaTick', name: 'Lava Tick', levelRange: [22, 50], role: 'Fast burn skirmisher', family: 'Volcanic Beast', hpMult: 0.72, damageMult: 1.16, defenseMult: 0.78, expMult: 1.18, speed: 124, behavior: 'skirmisher', mechanic: 'Rapid bites with short burst movement between vents.', counter: 'AoE, traps, and burst before it retreats.', drops: ['Molten Fang', 'Ember Dust', 'Cinder gear'] },
      { id: 'cinderSpitter', name: 'Cinder Spitter', levelRange: [28, 55], role: 'Volcanic thrower', family: 'Volcanic Beast', hpMult: 0.88, damageMult: 1.12, defenseMult: 0.9, expMult: 1.26, speed: 62, behavior: 'thrower', mechanic: 'Lobs arcing cinder globs from mid-range.', counter: 'Gap closers, vertical movement, and stuns.', drops: ['Cinder Gland', 'Focus accessory', 'Upgrade Catalyst'] },
      { id: 'banditCutter', name: 'Bandit Cutter', levelRange: [18, 28], role: 'Melee blocker', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Frontal block.', counter: 'Back attacks, burst, stun.', drops: ['Bandit Cloth', 'Steel weapon'] },
      { id: 'banditCutterDirect', name: 'Direct Sheet Bandit', levelRange: [18, 28], role: 'Asset test - direct sheet', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Direct single-prompt sheet comparison.', counter: 'Back attacks, burst, stun.', drops: [] },
      { id: 'banditCutterReference', name: 'Reference Sheet Bandit', levelRange: [18, 28], role: 'Asset test - reference sheet', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Reference-guided sheet comparison.', counter: 'Back attacks, burst, stun.', drops: [] },
      { id: 'banditCutterHybrid', name: 'Hybrid Keyframe Bandit', levelRange: [18, 28], role: 'Asset test - hybrid keyframes', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Generated key poses with deterministic assembly.', counter: 'Back attacks, burst, stun.', drops: [] },
      { id: 'banditCutterPuppet', name: 'Puppet Composite Bandit', levelRange: [18, 28], role: 'Asset test - puppet composite', family: 'Humanoid', hpMult: 1.05, damageMult: 1.05, defenseMult: 1, expMult: 1.1, speed: 88, behavior: 'blocker', mechanic: 'Generated puppet poses with deterministic transforms.', counter: 'Back attacks, burst, stun.', drops: [] },
      { id: 'banditThrower', name: 'Bandit Thrower', levelRange: [20, 30], role: 'Ranged priority', family: 'Humanoid', hpMult: 0.8, damageMult: 0.9, defenseMult: 0.75, expMult: 1.05, speed: 74, behavior: 'thrower', mechanic: 'Retreats and throws arcs.', counter: 'Gap closers, long range.', drops: ['Throwing Knife Scrap', 'Sharp Ring'] },
      { id: 'orebackBeetle', name: 'Oreback Beetle', levelRange: [24, 35], role: 'Tank/material', family: 'Beast / Mineral', hpMult: 1.9, damageMult: 0.9, defenseMult: 1.8, expMult: 1.5, speed: 35, behavior: 'armored', mechanic: 'Shell crack.', counter: 'Armor break, weak points.', drops: ['Ore Chunks', 'Upgrade Catalyst'] },
      { id: 'glowcapHealer', name: 'Glowcap Healer', levelRange: [22, 34], role: 'Support', family: 'Plant', hpMult: 0.75, damageMult: 0.4, defenseMult: 0.7, expMult: 1.2, speed: 34, behavior: 'healer', mechanic: 'Heals nearby enemies.', counter: 'Fire, seal, target priority.', drops: ['Glow Spores', 'Support accessory'] },
      { id: 'crackedMimic', name: 'Cracked Mimic', levelRange: [15, 35], role: 'Rare elite', family: 'Construct / Treasure', hpMult: 4, damageMult: 1.5, defenseMult: 1.2, expMult: 4, speed: 96, behavior: 'elite', mechanic: 'Ambush and flee at low HP.', counter: 'Burst windows, stuns.', drops: ['Currency burst', 'Rare gear', 'Upgrade Catalyst'] },
      { id: 'brambleking', name: 'Brambleking', levelRange: [24, 34], role: 'Boss', family: 'Plant', hpMult: 6.8, damageMult: 1.45, defenseMult: 1.25, expMult: 5.8, speed: 24, behavior: 'boss', mechanic: 'Root waves, thorn volleys, and vine cages.', counter: 'Fire, mobility, and controlled burst windows.', drops: ['Bramble Crown', 'Boss gear', 'Upgrade Catalyst'] },
      { id: 'clockworkTitan', name: 'Clockwork Titan', levelRange: [30, 42], role: 'Boss', family: 'Construct', hpMult: 7.4, damageMult: 1.55, defenseMult: 1.9, expMult: 6.4, speed: 34, behavior: 'boss', mechanic: 'Gear slams, armor plates, and timed vulnerability phases.', counter: 'Armor break, bossing skills, and weak-point timing.', drops: ['Titan Core', 'Boss gear', 'Upgrade Catalyst'] },
      { id: 'quarryColossus', name: 'Quarry Colossus', levelRange: [34, 48], role: 'Boss', family: 'Mineral Construct', hpMult: 7.8, damageMult: 1.65, defenseMult: 2, expMult: 6.8, speed: 30, behavior: 'boss', mechanic: 'Ore armor, falling rocks, and healer add pressure.', counter: 'Sustained armor crack uptime and priority add control.', drops: ['Colossus Ore', 'Boss gear', 'Rare catalyst'] },
      { id: 'emberjawGolem', name: 'Emberjaw Golem', levelRange: [30, 40], role: 'Boss', family: 'Volcanic Construct', hpMult: 8, damageMult: 1.8, defenseMult: 1.6, expMult: 8, speed: 52, behavior: 'boss', mechanic: 'Ground slams, fire cracks, arena charge, ore minions, Overheat vulnerability.', counter: 'Dodging, control during Overheat, weak-point marks.', drops: ['Emberjaw Badge', 'Boss gear', 'Rare catalyst'] },
      { id: 'frostlingScout', name: 'Frostling Scout', levelRange: [45, 58], role: 'Fast melee', family: 'Frostkin', hpMult: 0.95, damageMult: 1.12, defenseMult: 0.9, expMult: 1.24, speed: 112, behavior: 'skirmisher', mechanic: 'Dashes across slick ground after short feints.', counter: 'Area control, slows, and quick burst windows.', drops: ['Rime Shard', 'Frost gear'] },
      { id: 'shardling', name: 'Shardling', levelRange: [45, 60], role: 'Frost swarm', family: 'Frost Construct', hpMult: 0.78, damageMult: 0.9, defenseMult: 0.95, expMult: 1.08, speed: 66, behavior: 'hopper', mechanic: 'Skips across ice in short crystalline hops.', counter: 'AoE and knockback before it surrounds you.', drops: ['Rime Shard', 'Upgrade Dust', 'Frost gear'] },
      { id: 'rimebackBrute', name: 'Rimeback Brute', levelRange: [48, 64], role: 'Armored tank', family: 'Frost Beast', hpMult: 2.05, damageMult: 1.05, defenseMult: 1.85, expMult: 1.55, speed: 42, behavior: 'armored', mechanic: 'Ice shell cracks under sustained armor break.', counter: 'Armor break and persistent damage.', drops: ['Frozen Hide', 'Upgrade Catalyst'] },
      { id: 'glacierSentinel', name: 'Glacier Sentinel', levelRange: [54, 70], role: 'Frozen turret', family: 'Frost Construct', hpMult: 1.35, damageMult: 1.12, defenseMult: 1.55, expMult: 1.42, speed: 0, behavior: 'turret', mechanic: 'Anchors itself and fires slow piercing ice lances.', counter: 'Line-of-sight breaks, armor break, and burst.', drops: ['Glacier Core', 'Frost gear', 'Upgrade Catalyst'] },
      { id: 'snowglareWisp', name: 'Snowglare Wisp', levelRange: [50, 66], role: 'Flying ranged', family: 'Frost Spirit', hpMult: 0.9, damageMult: 1.1, defenseMult: 0.75, expMult: 1.3, speed: 78, behavior: 'flyer', mechanic: 'Floats above ice lanes and fires chill bolts.', counter: 'Ranged attacks, marks, and mobility skills.', drops: ['Snowglare Dust', 'Focus accessory'] },
      { id: 'icebloomOracle', name: 'Icebloom Oracle', levelRange: [48, 66], role: 'Frost support', family: 'Plant / Frost Spirit', hpMult: 0.82, damageMult: 0.55, defenseMult: 0.82, expMult: 1.26, speed: 38, behavior: 'healer', mechanic: 'Channels frost blooms that heal nearby enemies.', counter: 'Target priority, fire, and interrupts.', drops: ['Icebloom Petal', 'Support accessory', 'Prism Shards'] },
      { id: 'galeHarrier', name: 'Gale Harrier', levelRange: [55, 72], role: 'Wind flyer', family: 'Storm Spirit', hpMult: 0.86, damageMult: 1.1, defenseMult: 0.78, expMult: 1.28, speed: 92, behavior: 'flyer', mechanic: 'Strafes through cliff lanes with gust bolts.', counter: 'Marks, lightning resistance, and ranged attacks.', drops: ['Gale Feather', 'Storm gear', 'Focus accessory'] },
      { id: 'stormboundArcher', name: 'Stormbound Archer', levelRange: [55, 72], role: 'Ranged pressure', family: 'Humanoid / Storm', hpMult: 0.92, damageMult: 1.14, defenseMult: 0.9, expMult: 1.3, speed: 76, behavior: 'thrower', mechanic: 'Kites between platforms while firing charged arrows.', counter: 'Gap closers, stuns, and vertical pressure.', drops: ['Storm Fletching', 'Sharp Ring', 'Storm gear'] },
      { id: 'thunderRam', name: 'Thunder Ram', levelRange: [58, 74], role: 'Storm charger', family: 'Beast / Storm', hpMult: 1.42, damageMult: 1.28, defenseMult: 1.15, expMult: 1.4, speed: 96, behavior: 'charger', mechanic: 'Crackling charge after a visible hoof spark.', counter: 'Jump timing, slows, and post-charge burst.', drops: ['Thunder Horn', 'Traveler Boots', 'Upgrade Catalyst'] },
      { id: 'cloudcallAcolyte', name: 'Cloudcall Acolyte', levelRange: [60, 74], role: 'Storm support', family: 'Humanoid / Storm', hpMult: 0.88, damageMult: 0.65, defenseMult: 0.9, expMult: 1.32, speed: 48, behavior: 'healer', mechanic: 'Calls cloud pulses that restore nearby allies.', counter: 'Target priority and interrupts.', drops: ['Cloud Silk', 'Support accessory', 'Prism Shards'] },
      { id: 'indexScribe', name: 'Index Scribe', levelRange: [70, 90], role: 'Astral ranged', family: 'Astral Humanoid', hpMult: 0.94, damageMult: 1.18, defenseMult: 0.95, expMult: 1.35, speed: 70, behavior: 'thrower', mechanic: 'Throws rune pages that arc over low cover.', counter: 'Gap closers, line movement, and burst.', drops: ['Runic Page', 'Focus accessory', 'Astral gear'] },
      { id: 'lumenSentinel', name: 'Lumen Sentinel', levelRange: [72, 100], role: 'Astral armored tank', family: 'Astral Construct', hpMult: 1.8, damageMult: 1.08, defenseMult: 1.85, expMult: 1.58, speed: 44, behavior: 'armored', mechanic: 'Luminous shell dims as armor breaks.', counter: 'Armor break, sustained damage, and weak-point marks.', drops: ['Lumen Plate', 'Guard Ring', 'Astral gear'] },
      { id: 'voidMote', name: 'Void Mote', levelRange: [75, 100], role: 'Astral flyer', family: 'Void Spirit', hpMult: 0.82, damageMult: 1.22, defenseMult: 0.78, expMult: 1.38, speed: 96, behavior: 'flyer', mechanic: 'Blinks in short arcs before firing void sparks.', counter: 'Ranged tracking, marks, and burst timing.', drops: ['Void Dust', 'Focus accessory', 'Prism Shards'] },
      { id: 'eclipseDuelist', name: 'Eclipse Duelist', levelRange: [85, 105], role: 'Late-game blocker', family: 'Astral Humanoid', hpMult: 1.16, damageMult: 1.28, defenseMult: 1.12, expMult: 1.46, speed: 92, behavior: 'blocker', mechanic: 'Parries from the front and lunges after blocking.', counter: 'Back attacks, stuns, and vertical repositioning.', drops: ['Eclipse Silk', 'Sharp Ring', 'Eclipse gear'] },
      { id: 'riftAberration', name: 'Rift Aberration', levelRange: [100, 100], role: 'Rift elite', family: 'Void Aberration', hpMult: 3.2, damageMult: 1.55, defenseMult: 1.25, expMult: 3.6, speed: 86, behavior: 'elite', mechanic: 'Warps through lanes and enrages at low HP.', counter: 'Burst windows, stuns, and target focus.', drops: ['Rift Splinter', 'Rare gear', 'Rare catalyst'] },
      { id: 'rimewarden', name: 'Rimewarden', levelRange: [58, 70], role: 'Boss', family: 'Frost Construct', hpMult: 8.2, damageMult: 1.72, defenseMult: 1.75, expMult: 7.2, speed: 38, behavior: 'boss', mechanic: 'Freezing shockwaves, armor phases, and slippery arena control.', counter: 'Positioning, armor crack uptime, and controlled burst.', drops: ['Rimewarden Sigil', 'Boss gear', 'Rare catalyst'] },
      { id: 'stormbreakRoc', name: 'Aurelion, Stormbreak Roc', levelRange: [72, 84], role: 'Boss', family: 'Storm Beast', hpMult: 8.8, damageMult: 1.84, defenseMult: 1.5, expMult: 7.8, speed: 74, behavior: 'boss', mechanic: 'Flying divebombs, lightning rods, and wind lane pressure.', counter: 'Ranged uptime, rod baiting, and vertical repositioning.', drops: ['Stormbreak Plume', 'Boss gear', 'Rare catalyst'] },
      { id: 'astralArchivist', name: 'The Astral Archivist', levelRange: [88, 98], role: 'Boss', family: 'Astral Humanoid', hpMult: 9.2, damageMult: 1.88, defenseMult: 1.62, expMult: 8.4, speed: 46, behavior: 'boss', mechanic: 'Rune pages, action-memory resistance, and mirrored delayed attacks.', counter: 'Skill variety, add control, and target focus.', drops: ['Archivist Index', 'Boss gear', 'Rare catalyst'] },
      { id: 'eclipseSovereign', name: 'Eclipse Sovereign', levelRange: [100, 112], role: 'Boss', family: 'Astral Royalty', hpMult: 10.2, damageMult: 2.02, defenseMult: 1.76, expMult: 9.2, speed: 54, behavior: 'boss', mechanic: 'Solar and lunar stance swaps, eclipse zones, and totality sigils.', counter: 'Zone swaps, coordinated burst, and phase awareness.', drops: ['Sovereign Corona', 'Boss gear', 'Rare catalyst'] }
    ].map(attachEnemyDropPool).map(attachEnemyAssets).map(attachEnemyGuideMetadata));

    return Object.freeze({
      ENEMIES
    });
  }

  const api = Object.freeze({
    createEnemyData
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.enemies = api;
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
