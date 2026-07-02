(function initProjectStarfallDataClassSkillDesign(global) {
  'use strict';

  const CLASS_SKILL_GUIDE_CONTRACT = Object.freeze({
    guidePath: 'CLASS_AND_SKILL_DESIGN_GUIDE.md',
    runtimeDataFiles: Object.freeze([
      'js/games/project-starfall/data/classes.js',
      'js/games/project-starfall/data/skills.js',
      'js/games/project-starfall/data/specializations.js',
      'js/games/project-starfall/data/combat-fx.js',
      'js/games/project-starfall/data/animations.js',
      'js/games/project-starfall/data/enemies.js',
      'js/games/project-starfall/data/combat-modifiers.js',
      'js/games/project-starfall/data/equipment-catalog.js',
      'js/games/project-starfall/data/upgrades.js'
    ]),
    runtimeHookFiles: Object.freeze([
      'js/games/project-starfall/project-starfall-engine.js',
      'js/games/project-starfall/engine/skills.js',
      'js/games/project-starfall/engine/movement.js',
      'js/games/project-starfall/engine/skill-modifiers.js',
      'js/games/project-starfall/engine/combat-formulas.js',
      'js/games/project-starfall/ui/skill-metadata.js',
      'js/games/project-starfall/ui/skill-display.js',
      'js/games/project-starfall/ui/skill-prerequisites.js',
      'js/games/project-starfall/ui/skill-state.js',
      'js/games/project-starfall/ui/resource-widgets.js',
      'js/games/project-starfall/ui/keybindings.js',
      'js/games/project-starfall/ui/input.js'
    ]),
    expectedBaseClasses: Object.freeze(['fighter', 'mage', 'archer']),
    expectedAdvancedClasses: Object.freeze([
      'guardian',
      'berserker',
      'duelist',
      'fireMage',
      'runeMage',
      'stormMage',
      'sniper',
      'trapper',
      'beastArcher'
    ]),
    requiredGuideSections: Object.freeze([
      'Project-Specific Combat And Class Analysis',
      'Core Design Goals For The Class System',
      'Class Identity Framework',
      'Skill Design Framework',
      'Skill Expression And Player Mastery',
      'Recommended Class Archetypes',
      'Progression And Unlock Structure',
      'Combat Readability And Game Feel',
      'Balance Framework',
      'Implementation Guidance',
      'Required Animation And Asset Guidance For Skills',
      'Enemy And Encounter Interaction',
      'UI And Player Communication',
      'Class And Skill Documentation Templates',
      'Playtesting Checklist'
    ])
  });

  const CLASS_RUNTIME_REQUIRED_FIELDS = Object.freeze([
    'id',
    'name',
    'asset',
    'animation',
    'resourceName',
    'resourceColor',
    'roleProfile',
    'description'
  ]);

  const ADVANCED_CLASS_REQUIRED_FIELDS = Object.freeze([
    'baseClass',
    'levelRequirement',
    'partySkillId'
  ]);

  const SKILL_RUNTIME_REQUIRED_FIELDS = Object.freeze([
    'id',
    'name',
    'owner',
    'batch',
    'type',
    'category',
    'roleTags',
    'prerequisites',
    'maxRank',
    'resourceCost',
    'cooldown',
    'lineCount',
    'lineDamageScale',
    'iconAsset',
    'iconKind',
    'purpose',
    'description'
  ]);

  const SKILL_DESIGN_OPTIONAL_FIELDS = Object.freeze([
    'startupSeconds',
    'activeSeconds',
    'recoverySeconds',
    'cancelRules',
    'hitbox',
    'airUse',
    'facingLock',
    'movementLock',
    'riskLevel',
    'skillTier',
    'statusEffects',
    'assetRequirements'
  ]);

  const CLASS_RESOURCE_DEFINITIONS = Object.freeze({
    fighter: Object.freeze({
      id: 'momentum',
      label: 'Momentum',
      classId: 'fighter',
      gain: Object.freeze(['close-range hits', 'guard pressure', 'same-lane combat uptime']),
      spend: Object.freeze(['Momentum Burst', 'advanced Fighter branch payoffs']),
      playerDecision: 'Stay close enough to keep pressure without eating unsafe hits.',
      uiWidget: 'warm red-orange meter with a burst-ready flash',
      feedback: Object.freeze(['impact pulse on gain', 'meter flare on spender readiness'])
    }),
    mage: Object.freeze({
      id: 'energy',
      label: 'Energy',
      classId: 'mage',
      gain: Object.freeze(['spell hits', 'marks', 'safe casting windows']),
      spend: Object.freeze(['Energy Release', 'advanced Mage branch payoffs']),
      playerDecision: 'Create enough distance to cast while saving burst for marked or clustered enemies.',
      uiWidget: 'blue arcane meter with mark-linked pulses',
      feedback: Object.freeze(['cast sparkle on gain', 'marked-target pulse on spender readiness'])
    }),
    archer: Object.freeze({
      id: 'focus',
      label: 'Focus',
      classId: 'archer',
      gain: Object.freeze(['ranged hits', 'marks', 'clean spacing']),
      spend: Object.freeze(['Focused Volley', 'advanced Archer branch payoffs']),
      playerDecision: 'Keep an effective firing lane while choosing the best target to mark.',
      uiWidget: 'green focus meter with reticle readiness',
      feedback: Object.freeze(['arrow glint on gain', 'reticle flash on marked payoff'])
    }),
    guardian: Object.freeze({
      id: 'storedImpact',
      label: 'Stored Impact',
      classId: 'guardian',
      gain: Object.freeze(['blocked hits', 'shield contact', 'front-line stagger']),
      spend: Object.freeze(['Retaliation Wave', 'Guardian\'s Verdict', 'Oath Barrier scaling']),
      playerDecision: 'Time guard windows and decide whether to convert stored pressure into control or shielding.',
      uiWidget: 'segmented blue-steel impact meter',
      feedback: Object.freeze(['shield clang on gain', 'segment flash on counter-ready'])
    }),
    berserker: Object.freeze({
      id: 'rage',
      label: 'Rage',
      classId: 'berserker',
      gain: Object.freeze(['dealing damage', 'taking controlled risk', 'low-HP pressure']),
      spend: Object.freeze(['Last Stand windows', 'rage-scaled sustain and burst']),
      playerDecision: 'Stay in the danger band long enough to gain value without turning risk into a death spiral.',
      uiWidget: 'red rage gauge with low-HP threshold marker',
      feedback: Object.freeze(['pulse on threshold entry', 'strong flash on burst-ready'])
    }),
    duelist: Object.freeze({
      id: 'tempo',
      label: 'Tempo',
      classId: 'duelist',
      gain: Object.freeze(['repeated hits on the same target', 'clean repositioning', 'future ripostes']),
      spend: Object.freeze(['finishers', 'tempo-scaled burst windows']),
      playerDecision: 'Stay on the correct target and decide when to preserve or cash out same-target pressure.',
      uiWidget: 'gold tempo pips with target-lock indicator',
      feedback: Object.freeze(['thin slash tick on gain', 'finisher glint at cap'])
    }),
    fireMage: Object.freeze({
      id: 'heat',
      label: 'Heat',
      classId: 'fireMage',
      gain: Object.freeze(['fire hits', 'burning enemies', 'burn spread']),
      spend: Object.freeze(['Heat Vent', 'Inferno Burst', 'wildfire payoffs']),
      playerDecision: 'Build enough Heat for strong detonations while preventing waste or unsafe overheat.',
      uiWidget: 'orange heat thermometer with burn-count accent',
      feedback: Object.freeze(['ember tick on gain', 'heat shimmer at high Heat'])
    }),
    runeMage: Object.freeze({
      id: 'runicEnergy',
      label: 'Runic Energy',
      classId: 'runeMage',
      gain: Object.freeze(['rune marks', 'active glyphs', 'linked targets']),
      spend: Object.freeze(['Rune Detonation', 'Grand Inscription', 'seal control']),
      playerDecision: 'Place runes where enemies will remain and detonate when setup density is worth the cast.',
      uiWidget: 'teal rune counter plus energy meter',
      feedback: Object.freeze(['glyph pulse on gain', 'linked-line brightening before detonation'])
    }),
    stormMage: Object.freeze({
      id: 'charge',
      label: 'Charge',
      classId: 'stormMage',
      gain: Object.freeze(['chain hits', 'clustered enemies', 'static repositioning']),
      spend: Object.freeze(['future storm spenders', 'Stormfront scaling']),
      playerDecision: 'Route lightning through clusters without wasting casts on isolated targets.',
      uiWidget: 'cyan charge meter with chain-count sparks',
      feedback: Object.freeze(['spark jumps per chain', 'static flash on high Charge'])
    }),
    sniper: Object.freeze({
      id: 'aim',
      label: 'Aim',
      classId: 'sniper',
      gain: Object.freeze(['steady shots', 'weak point marks', 'long-range uptime']),
      spend: Object.freeze(['One Perfect Shot', 'Execution Shot optimization']),
      playerDecision: 'Hold safe aim space and choose whether to spend on a weak point or save for execution.',
      uiWidget: 'gold aim meter with steady reticle',
      feedback: Object.freeze(['reticle tighten on gain', 'weak-point flash when ready'])
    }),
    trapper: Object.freeze({
      id: 'preparation',
      label: 'Preparation',
      classId: 'trapper',
      gain: Object.freeze(['armed traps', 'controlled enemies', 'manual detonations']),
      spend: Object.freeze(['Kill Zone', 'Detonate scaling', 'Tactical Field value']),
      playerDecision: 'Prepare the lane enemies will cross instead of chasing their current position.',
      uiWidget: 'brown-gold preparation slots with trap readiness rings',
      feedback: Object.freeze(['trap arm pulse', 'wire brightening on trigger-ready'])
    }),
    beastArcher: Object.freeze({
      id: 'bond',
      label: 'Bond',
      classId: 'beastArcher',
      gain: Object.freeze(['companion strikes', 'pack marks', 'sustained coordinated attacks']),
      spend: Object.freeze(['Pack Call', 'future companion commands and Bond spenders']),
      playerDecision: 'Coordinate shots and companion pressure while deciding when to spend Bond for sustain or burst.',
      uiWidget: 'green-gold bond meter with companion status icon',
      feedback: Object.freeze(['companion spark on gain', 'pack aura when spender-ready'])
    })
  });

  const STATUS_EFFECT_DEFINITIONS = Object.freeze({
    burn: Object.freeze({
      id: 'burn',
      category: 'damageOverTime',
      sourceClasses: Object.freeze(['fireMage']),
      playerRead: 'orange flame icon and small non-shaking tick numbers',
      bossRule: 'Bosses can take burn damage, but spread and tick scaling must respect caps.'
    }),
    mark: Object.freeze({
      id: 'mark',
      category: 'setup',
      sourceClasses: Object.freeze(['mage', 'archer', 'fireMage', 'runeMage', 'sniper', 'trapper', 'beastArcher']),
      playerRead: 'small target sigil above the enemy',
      bossRule: 'Boss marks should enable payoff without permanent full vulnerability.'
    }),
    crack: Object.freeze({
      id: 'crack',
      category: 'armorBreak',
      sourceClasses: Object.freeze(['fighter', 'guardian', 'sniper']),
      playerRead: 'fractured armor icon and brief impact ring',
      bossRule: 'Feeds boss break gauge and improves damage inside controlled caps.'
    }),
    slow: Object.freeze({
      id: 'slow',
      category: 'control',
      sourceClasses: Object.freeze(['runeMage', 'stormMage', 'trapper']),
      playerRead: 'blue drag trail or foot snare pulse',
      bossRule: 'Boss movement effects should be reduced or converted into break-gauge value.'
    }),
    weakPoint: Object.freeze({
      id: 'weakPoint',
      category: 'precisionSetup',
      sourceClasses: Object.freeze(['sniper']),
      playerRead: 'gold reticle flash on the target',
      bossRule: 'Bosses should telegraph weak-point windows instead of allowing permanent weak-point uptime.'
    }),
    runeLink: Object.freeze({
      id: 'runeLink',
      category: 'linkedDamage',
      sourceClasses: Object.freeze(['runeMage']),
      playerRead: 'teal line between linked enemies',
      bossRule: 'Boss links can splash to adds or contribute reduced echo damage.'
    }),
    packMark: Object.freeze({
      id: 'packMark',
      category: 'companionSetup',
      sourceClasses: Object.freeze(['beastArcher']),
      playerRead: 'green-gold claw mark',
      bossRule: 'Boss pack marks should interact with companion commands and sustain, not free passive damage.'
    }),
    stagger: Object.freeze({
      id: 'stagger',
      category: 'hitReaction',
      sourceClasses: Object.freeze(['fighter', 'guardian']),
      playerRead: 'short enemy recoil with a clear no-input window',
      bossRule: 'Boss stagger should convert to break progress unless a phase specifically permits stagger.'
    }),
    shield: Object.freeze({
      id: 'shield',
      category: 'defense',
      sourceClasses: Object.freeze(['mage', 'guardian']),
      playerRead: 'visible shell and HUD absorb value',
      bossRule: 'Large boss hits should visibly consume shield without hiding danger telegraphs.'
    }),
    lure: Object.freeze({
      id: 'lure',
      category: 'enemyRouting',
      sourceClasses: Object.freeze(['trapper']),
      playerRead: 'green pull line and destination pulse',
      bossRule: 'Bosses resist pull, but adds and nearby regular enemies should route toward the lure.'
    }),
    haste: Object.freeze({
      id: 'haste',
      category: 'buff',
      sourceClasses: Object.freeze(['duelist', 'runeMage', 'stormMage']),
      playerRead: 'light foot or cast-speed aura',
      bossRule: 'Party haste should obey buff stacking rules and not remove boss recovery windows.'
    })
  });

  const CLASS_SKILL_TOOLTIP_FORMAT = Object.freeze({
    lines: Object.freeze([
      '[Skill Name] - Rank [current]/[max]',
      '[Type] | [Purpose Tags]',
      '[One-sentence fantasy and use case.]',
      'Cost: [MP/class resource/charges]',
      'Cooldown: [seconds]',
      'Range: [short/medium/long or exact if useful]',
      'Current: [specific behavior at current rank]',
      'Next Rank: [specific change]',
      'Breakpoint: [Rank 5/10 behavior if relevant]',
      'Use when: [decision prompt]',
      'Weak when: [counter-situation]',
      'Prerequisite: [missing requirement or Ready]'
    ]),
    requiredConcepts: Object.freeze(['cost', 'cooldown', 'range', 'current', 'nextRank', 'useWhen', 'weakWhen', 'prerequisite'])
  });

  const CLASS_SKILL_BALANCE_TUNING_FIELDS = Object.freeze([
    'skill',
    'owner',
    'tier',
    'damage',
    'cooldown',
    'mpCost',
    'classResource',
    'range',
    'startup',
    'recovery',
    'areaSize',
    'crowdControlValue',
    'mobilityValue',
    'risk',
    'bestUse',
    'weakness'
  ]);

  const CLASS_SKILL_LOADOUT_SLOTS = Object.freeze([
    Object.freeze({ id: 'basicAttack', label: 'Basic Attack', required: true, source: 'base class' }),
    Object.freeze({ id: 'primarySkill', label: 'Primary Skill', required: true, source: 'class primary training candidate' }),
    Object.freeze({ id: 'mobilityDefense', label: 'Mobility / Defense', required: true, source: 'learned active skill' }),
    Object.freeze({ id: 'active1', label: 'Active Skill 1', required: false, source: 'learned active skill' }),
    Object.freeze({ id: 'active2', label: 'Active Skill 2', required: false, source: 'learned active skill' }),
    Object.freeze({ id: 'active3', label: 'Active Skill 3', required: false, source: 'learned active skill' }),
    Object.freeze({ id: 'partySkill', label: 'Party Skill', required: true, source: 'advanced class partySkillId' })
  ]);

  const CLASS_SKILL_ENCOUNTER_TEST_CASES = Object.freeze([
    Object.freeze({
      id: 'clusteredMobbing',
      enemyRoles: Object.freeze(['hopper', 'skirmisher', 'support']),
      terrain: 'shared lane or compact platform pack',
      expectedStrongClasses: Object.freeze(['fireMage', 'stormMage', 'trapper']),
      expectedPressureClasses: Object.freeze(['sniper', 'duelist']),
      validates: Object.freeze(['area target caps', 'burn spread', 'chain falloff', 'trap arming'])
    }),
    Object.freeze({
      id: 'armoredElite',
      enemyRoles: Object.freeze(['armored', 'blocker', 'bruiser']),
      terrain: 'medium lane with safe punish windows',
      expectedStrongClasses: Object.freeze(['guardian', 'fighter', 'sniper', 'runeMage']),
      expectedPressureClasses: Object.freeze(['fireMage', 'stormMage']),
      validates: Object.freeze(['crack value', 'break gauge contribution', 'single-target rotations'])
    }),
    Object.freeze({
      id: 'flyingSpread',
      enemyRoles: Object.freeze(['flying', 'ranged', 'turret']),
      terrain: 'vertical platforms and separated lanes',
      expectedStrongClasses: Object.freeze(['archer', 'sniper', 'stormMage', 'mage']),
      expectedPressureClasses: Object.freeze(['guardian', 'berserker']),
      validates: Object.freeze(['line of sight', 'projectile readability', 'melee access routes'])
    }),
    Object.freeze({
      id: 'sustainBoss',
      enemyRoles: Object.freeze(['boss', 'adds', 'hazard']),
      terrain: 'boss arena with one meaningful platform feature',
      expectedStrongClasses: Object.freeze(['guardian', 'berserker', 'duelist', 'sniper']),
      expectedPressureClasses: Object.freeze(['trapper', 'stormMage']),
      validates: Object.freeze(['boss TTK', 'defensive timing', 'break windows', 'class resource spending'])
    }),
    Object.freeze({
      id: 'routeControl',
      enemyRoles: Object.freeze(['charger', 'support', 'swarm']),
      terrain: 'switchback lane with ladders or chokepoints',
      expectedStrongClasses: Object.freeze(['trapper', 'runeMage', 'guardian', 'fireMage']),
      expectedPressureClasses: Object.freeze(['sniper', 'beastArcher']),
      validates: Object.freeze(['trap visibility', 'field placement', 'lure behavior', 'platform surface snapping'])
    })
  ]);

  const CLASS_SKILL_DEBUG_SCENARIOS = Object.freeze([
    Object.freeze({ id: 'rank1Starter', level: 1, skillRanks: 'default', gearProfile: 'starter' }),
    Object.freeze({ id: 'firstBuildChoice', level: 12, skillRanks: 'mixed base ranks', gearProfile: 'shop expected' }),
    Object.freeze({ id: 'advancedEntry', level: 25, skillRanks: 'rank 1 advanced trainer and inherited base loop', gearProfile: 'level 25 class offhand' }),
    Object.freeze({ id: 'rank5Breakpoint', level: 35, skillRanks: 'rank 5 branch core', gearProfile: 'regional expected' }),
    Object.freeze({ id: 'specializationEntry', level: 60, skillRanks: 'specialization unlocked', gearProfile: 'boss set partial' }),
    Object.freeze({ id: 'attunedEndgame', level: 100, skillRanks: 'late mastery', gearProfile: 'attuned boss set' })
  ]);

  const api = {
    CLASS_SKILL_GUIDE_CONTRACT,
    CLASS_RUNTIME_REQUIRED_FIELDS,
    ADVANCED_CLASS_REQUIRED_FIELDS,
    SKILL_RUNTIME_REQUIRED_FIELDS,
    SKILL_DESIGN_OPTIONAL_FIELDS,
    CLASS_RESOURCE_DEFINITIONS,
    STATUS_EFFECT_DEFINITIONS,
    CLASS_SKILL_TOOLTIP_FORMAT,
    CLASS_SKILL_BALANCE_TUNING_FIELDS,
    CLASS_SKILL_LOADOUT_SLOTS,
    CLASS_SKILL_ENCOUNTER_TEST_CASES,
    CLASS_SKILL_DEBUG_SCENARIOS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.classSkillDesign = Object.assign({}, modules.classSkillDesign || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
