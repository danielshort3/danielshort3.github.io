(function initProjectStarfallDataSkills(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const BASE_SKILL_ICON_ROOT = DataAssets.BASE_SKILL_ICON_ROOT;
  const ADVANCED_SKILL_ICON_ROOT = DataAssets.ADVANCED_SKILL_ICON_ROOT;

  const BASE_SKILL_ICONS = Object.freeze({
    fighter_heavy_strike: `${BASE_SKILL_ICON_ROOT}/fighter-heavy-strike.png`,
    fighter_dash_slash: `${BASE_SKILL_ICON_ROOT}/fighter-dash-slash.png`,
    fighter_guard: `${BASE_SKILL_ICON_ROOT}/fighter-guard.png`,
    fighter_ground_slam: `${BASE_SKILL_ICON_ROOT}/fighter-ground-slam.png`,
    fighter_power_break: `${BASE_SKILL_ICON_ROOT}/fighter-power-break.png`,
    fighter_momentum_burst: `${BASE_SKILL_ICON_ROOT}/fighter-momentum-burst.png`,
    fighter_damage_mastery: `${BASE_SKILL_ICON_ROOT}/fighter-damage-mastery.png`,
    mage_magic_bolt: `${BASE_SKILL_ICON_ROOT}/mage-magic-bolt.png`,
    mage_blink: `${BASE_SKILL_ICON_ROOT}/mage-blink.png`,
    mage_arcane_burst: `${BASE_SKILL_ICON_ROOT}/mage-arcane-burst.png`,
    mage_mana_shield: `${BASE_SKILL_ICON_ROOT}/mage-mana-shield.png`,
    mage_spell_mark: `${BASE_SKILL_ICON_ROOT}/mage-spell-mark.png`,
    mage_energy_release: `${BASE_SKILL_ICON_ROOT}/mage-energy-release.png`,
    mage_damage_mastery: `${BASE_SKILL_ICON_ROOT}/mage-damage-mastery.png`,
    archer_quick_shot: `${BASE_SKILL_ICON_ROOT}/archer-quick-shot.png`,
    archer_roll_shot: `${BASE_SKILL_ICON_ROOT}/archer-roll-shot.png`,
    archer_marked_shot: `${BASE_SKILL_ICON_ROOT}/archer-marked-shot.png`,
    archer_piercing_arrow: `${BASE_SKILL_ICON_ROOT}/archer-piercing-arrow.png`,
    archer_eagle_stance: `${BASE_SKILL_ICON_ROOT}/archer-eagle-stance.png`,
    archer_focused_volley: `${BASE_SKILL_ICON_ROOT}/archer-focused-volley.png`,
    archer_damage_mastery: `${BASE_SKILL_ICON_ROOT}/archer-damage-mastery.png`
  });

  const ADVANCED_SKILL_ICONS = Object.freeze({
    guardian_shield_bash: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-shield-bash.png`,
    guardian_shield_dash: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-shield-dash.png`,
    guardian_impact_guard: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-impact-guard.png`,
    guardian_oath_barrier: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-oath-barrier.png`,
    guardian_retaliation_wave: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-retaliation-wave.png`,
    guardian_hold_the_line: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-hold-the-line.png`,
    guardian_verdict: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-verdict.png`,
    guardian_shield_wall: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-shield-wall.png`,
    guardian_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/guardian/guardian-damage-mastery.png`,
    berserker_blood_cleave: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-blood-cleave.png`,
    berserker_rage_surge: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-rage-surge.png`,
    berserker_reckless_leap: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-reckless-leap.png`,
    berserker_crimson_recovery: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-crimson-recovery.png`,
    berserker_pain_to_power: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-pain-to-power.png`,
    berserker_last_stand: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-last-stand.png`,
    berserker_war_cry: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-war-cry.png`,
    berserker_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/berserker/berserker-damage-mastery.png`,
    duelist_quick_cut: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-quick-cut.png`,
    duelist_flash_step: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-flash-step.png`,
    duelist_rallying_flourish: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-rallying-flourish.png`,
    duelist_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/duelist/duelist-damage-mastery.png`,
    fire_mage_fireball: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-fireball.png`,
    fire_mage_flame_trail: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-flame-trail.png`,
    fire_mage_burning_mark: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-burning-mark.png`,
    fire_mage_heat_vent: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-heat-vent.png`,
    fire_mage_wildfire: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-wildfire.png`,
    fire_mage_inferno_burst: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-inferno-burst.png`,
    fire_mage_ignition_aura: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-ignition-aura.png`,
    fire_mage_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/fire-mage/fire-mage-damage-mastery.png`,
    rune_mage_rune_mark: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-mark.png`,
    rune_mage_rune_blink: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-blink.png`,
    rune_mage_ground_glyph: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-ground-glyph.png`,
    rune_mage_arcane_link: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-arcane-link.png`,
    rune_mage_rune_detonation: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-detonation.png`,
    rune_mage_mana_seal: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-mana-seal.png`,
    rune_mage_grand_inscription: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-grand-inscription.png`,
    rune_mage_rune_circle: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-rune-circle.png`,
    rune_mage_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/rune-mage/rune-mage-damage-mastery.png`,
    storm_mage_chain_bolt: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-chain-bolt.png`,
    storm_mage_static_shift: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-static-shift.png`,
    storm_mage_stormfront: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-stormfront.png`,
    storm_mage_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/storm-mage/storm-mage-damage-mastery.png`,
    sniper_aimed_shot: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-aimed-shot.png`,
    sniper_combat_roll: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-combat-roll.png`,
    sniper_weak_point_mark: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-weak-point-mark.png`,
    sniper_steady_breath: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-steady-breath.png`,
    sniper_pierce_armor: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-pierce-armor.png`,
    sniper_execution_shot: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-execution-shot.png`,
    sniper_one_perfect_shot: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-one-perfect-shot.png`,
    sniper_eagle_eye: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-eagle-eye.png`,
    sniper_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/sniper/sniper-damage-mastery.png`,
    trapper_snare_trap: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-snare-trap.png`,
    trapper_grapple_dash: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-grapple-dash.png`,
    trapper_spike_trap: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-spike-trap.png`,
    trapper_lure_shot: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-lure-shot.png`,
    trapper_tripwire: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-tripwire.png`,
    trapper_detonate: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-detonate.png`,
    trapper_kill_zone: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-kill-zone.png`,
    trapper_tactical_field: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-tactical-field.png`,
    trapper_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/trapper/trapper-damage-mastery.png`,
    beast_archer_companion_strike: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-companion-strike.png`,
    beast_archer_pounce_roll: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-pounce-roll.png`,
    beast_archer_pack_call: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-pack-call.png`,
    beast_archer_damage_mastery: `${ADVANCED_SKILL_ICON_ROOT}/beast-archer/beast-archer-damage-mastery.png`
  });

  const CLASS_ICON_ASSETS = Object.freeze({
    fighter: BASE_SKILL_ICONS.fighter_heavy_strike,
    mage: BASE_SKILL_ICONS.mage_magic_bolt,
    archer: BASE_SKILL_ICONS.archer_quick_shot,
    guardian: ADVANCED_SKILL_ICONS.guardian_shield_bash,
    berserker: ADVANCED_SKILL_ICONS.berserker_blood_cleave,
    duelist: ADVANCED_SKILL_ICONS.duelist_quick_cut,
    fireMage: ADVANCED_SKILL_ICONS.fire_mage_fireball,
    runeMage: ADVANCED_SKILL_ICONS.rune_mage_rune_mark,
    stormMage: ADVANCED_SKILL_ICONS.storm_mage_chain_bolt,
    sniper: ADVANCED_SKILL_ICONS.sniper_aimed_shot,
    trapper: ADVANCED_SKILL_ICONS.trapper_snare_trap,
    beastArcher: ADVANCED_SKILL_ICONS.beast_archer_companion_strike
  });

  const SKILL_PURPOSES = Object.freeze({
    trainer: Object.freeze({ label: 'Trainer', description: 'Reliable low-cooldown skill for normal leveling.' }),
    mobility: Object.freeze({ label: 'Mobility', description: 'Repositioning, gap crossing, or escape.' }),
    setup: Object.freeze({ label: 'Setup', description: 'Marks, cracks, links, or prepares a stronger follow-up.' }),
    control: Object.freeze({ label: 'Control', description: 'Slows, staggers, pulls, or otherwise shapes enemy movement.' }),
    mobbing: Object.freeze({ label: 'Mobbing', description: 'Clears clustered regular enemies.' }),
    bossing: Object.freeze({ label: 'Bossing', description: 'Focused damage or debuffs for durable targets.' }),
    defense: Object.freeze({ label: 'Defense', description: 'Prevents damage or turns pressure into resources.' }),
    sustain: Object.freeze({ label: 'Sustain', description: 'Restores HP, MP, or keeps resources stable.' }),
    resource: Object.freeze({ label: 'Resource', description: 'Builds, vents, or spends a class resource deliberately.' }),
    buff: Object.freeze({ label: 'Buff', description: 'Temporary self-enhancement for a combat window.' }),
    finisher: Object.freeze({ label: 'Finisher', description: 'Payoff skill with higher cost, setup, or cooldown.' }),
    passive: Object.freeze({ label: 'Passive', description: 'Permanent stat or mechanic upgrade.' }),
    party: Object.freeze({ label: 'Party', description: 'Solo self-buff now, party support hook later.' })
  });

  function normalizeSkillPurpose(value) {
    const purpose = String(value || '').trim();
    return Object.prototype.hasOwnProperty.call(SKILL_PURPOSES, purpose) ? purpose : '';
  }

  function inferSkillPurpose(id, type, roleTags, options) {
    const skillId = String(id || '').toLowerCase();
    const skillType = String(type || '').toLowerCase();
    const tags = roleTags || [];
    const config = options || {};
    const configured = normalizeSkillPurpose(config.purpose);
    if (configured) return configured;
    if (config.primaryTraining || skillType.includes('primary training') || skillType.includes('basic attack')) return 'trainer';
    if (skillType.includes('party') || config.partyEffect) return 'party';
    if (skillType.includes('passive') || config.passiveStats) return 'passive';
    if (config.movementEffect || skillId.includes('blink') || skillId.includes('dash') || skillId.includes('roll') || skillId.includes('leap')) return 'mobility';
    if (skillType.includes('defense') || skillId.includes('_guard') || skillId.startsWith('guard_') || skillId.includes('shield') || skillId.includes('barrier')) return 'defense';
    if (skillType.includes('sustain') || skillId.includes('recovery')) return 'sustain';
    if (skillType.includes('resource') || skillId.includes('heat_vent')) return 'resource';
    if (skillType.includes('finisher') || skillType.includes('ultimate') || skillType.includes('burst') || skillId.includes('detonate')) return 'finisher';
    if ((!skillType.includes('debuff') && skillType.includes('buff')) || skillType.includes('stance') || skillId.includes('surge') || skillId.includes('breath')) return 'buff';
    if (skillType.includes('setup') || skillType.includes('combo') || skillType.includes('debuff') || skillId.includes('mark') || skillId.includes('break') || skillId.includes('armor')) return 'setup';
    if (skillType.includes('control') || skillType.includes('utility') || skillId.includes('seal') || skillId.includes('lure')) return 'control';
    if (tags.includes('Mobbing') || skillType.includes('area') || skillId.includes('trap') || skillId.includes('glyph') || skillId.includes('zone')) return 'mobbing';
    if (tags.includes('Bossing')) return 'bossing';
    return 'trainer';
  }

  function inferSkillIconKind(id, type, roleTags) {
    const skillId = String(id || '');
    const skillType = String(type || '').toLowerCase();
    const tags = roleTags || [];
    if (skillId.includes('blink')) return 'blink';
    if (skillId.includes('dash') || skillId.includes('roll') || skillId.includes('leap')) return 'mobility';
    if (skillId.includes('guard') || skillId.includes('shield') || skillId.includes('barrier') || skillType.includes('defense')) return 'guard';
    if (skillId.includes('trap') || skillId.includes('field') || skillId.includes('glyph') || skillId.includes('circle')) return 'field';
    if (skillId.includes('mark') || skillId.includes('eye') || skillType.includes('setup')) return 'mark';
    if (skillId.includes('arrow') || skillId.includes('shot') || skillId.includes('volley')) return 'arrow';
    if (skillId.includes('bolt') || skillId.includes('fireball') || skillId.includes('rune')) return 'magic';
    if (skillId.includes('break') || skillId.includes('armor')) return 'break';
    if (skillType.includes('area') || tags.includes('Mobbing')) return 'area';
    if (skillType.includes('buff') || tags.includes('Party') || tags.includes('Support')) return 'buff';
    if (skillType.includes('finisher') || skillId.includes('burst')) return 'burst';
    return 'slash';
  }

  function inferSkillCategory(id, type, options) {
    const skillId = String(id || '').toLowerCase();
    const skillType = String(type || '').toLowerCase();
    const config = options || {};
    if (config.category) return config.category;
    if (skillType.includes('passive') || config.passiveStats) return 'passive';
    if (config.movementEffect || skillId.includes('blink') || skillId.includes('dash') || skillId.includes('roll') || skillId.includes('leap')) return 'mobility';
    if (skillType.includes('debuff')) return 'attack';
    if (skillType.includes('buff') || skillType.includes('defense') || skillType.includes('party') || skillType.includes('stance')) return 'buff';
    return 'attack';
  }

  function inferSkillMaxRank(id, type, options) {
    const skillType = String(type || '').toLowerCase();
    const category = inferSkillCategory(id, type, options);
    if (category === 'mobility' || category === 'buff') return 5;
    if (category === 'passive') return 10;
    if (skillType.includes('setup') || skillType.includes('debuff') || skillType.includes('finisher') || skillType.includes('ultimate') || skillType.includes('resource') || skillType.includes('utility')) return 10;
    return 20;
  }

  function projectileTargeting(config) {
    return { mode: 'projectile', ...(config || {}) };
  }

  function skillVisual(kind, color, accent, options) {
    return Object.freeze(Object.assign({
      kind,
      color,
      accent,
      glow: 18,
      trail: 'spark',
      impact: 'spark'
    }, options || {}));
  }

  const SKILL_VISUALS = Object.freeze({
    arcaneBolt: skillVisual('orb', '#69c8ff', '#e7fbff', { trail: 'star', impact: 'arcaneRing' }),
    arcaneBurst: skillVisual('orbBurst', '#8f8cff', '#ffffff', { glow: 24, trail: 'comet', impact: 'burstRing' }),
    spellMark: skillVisual('markOrb', '#b794f4', '#fff2ff', { trail: 'runeDust', impact: 'markSigil' }),
    energyRelease: skillVisual('comet', '#7bdff2', '#fff7b8', { glow: 28, trail: 'comet', impact: 'burstRing' }),
    fireball: skillVisual('fireball', '#ff7b3a', '#ffd166', { glow: 28, trail: 'ember', impact: 'flameBloom' }),
    burningMark: skillVisual('fireMark', '#ff5a3d', '#ffd166', { trail: 'ember', impact: 'brand' }),
    wildfire: skillVisual('wildfire', '#ff8c3a', '#fff0a6', { glow: 30, trail: 'ember', impact: 'flameBloom' }),
    infernoBurst: skillVisual('inferno', '#ff3d2e', '#ffe16a', { glow: 34, trail: 'ember', impact: 'infernoBloom' }),
    heatVent: skillVisual('flameCone', '#ff7b3a', '#ffe16a', { trail: 'heat', impact: 'flameBloom' }),
    runeMark: skillVisual('runeBolt', '#28c7b7', '#b8fff2', { trail: 'runeDust', impact: 'runeSeal' }),
    runeLink: skillVisual('runeLink', '#37d6c7', '#f0fffb', { trail: 'runeDust', impact: 'runeLink' }),
    manaSeal: skillVisual('seal', '#4cc7ff', '#e7fbff', { trail: 'runeDust', impact: 'runeSeal' }),
    groundGlyph: skillVisual('glyph', '#28c7b7', '#b8fff2', { impact: 'groundGlyph' }),
    runeDetonation: skillVisual('runeBurst', '#28c7b7', '#ffffff', { glow: 30, impact: 'runeBurst' }),
    grandInscription: skillVisual('grandRune', '#38e6d0', '#ffffff', { glow: 34, impact: 'grandRune' }),
    chainBolt: skillVisual('lightning', '#d8f6ff', '#7aa7ff', { glow: 26, trail: 'lightning', impact: 'staticFork' }),
    quickShot: skillVisual('arrow', '#f2d273', '#ffffff', { trail: 'line', impact: 'arrowChip' }),
    markedShot: skillVisual('markedArrow', '#ffe16a', '#ffef99', { trail: 'line', impact: 'markReticle' }),
    piercingArrow: skillVisual('piercingArrow', '#d8c25f', '#ffffff', { trail: 'pierceLine', impact: 'arrowPierce' }),
    focusedVolley: skillVisual('volleyArrow', '#ffd166', '#ffffff', { trail: 'line', impact: 'volleySpark' }),
    aimedShot: skillVisual('sniperTracer', '#e9f7ff', '#ffd166', { glow: 16, trail: 'tracer', impact: 'reticleHit' }),
    weakPointMark: skillVisual('weakPointArrow', '#ffd166', '#ffffff', { trail: 'tracer', impact: 'weakPoint' }),
    pierceArmor: skillVisual('armorPierce', '#b7c3ca', '#ffe16a', { trail: 'pierceLine', impact: 'armorCrack' }),
    executionShot: skillVisual('executionTracer', '#ffef99', '#ffffff', { glow: 24, trail: 'tracer', impact: 'executionFlash' }),
    perfectShot: skillVisual('perfectShot', '#ffffff', '#ffd166', { glow: 30, trail: 'tracer', impact: 'perfectStar' }),
    lureShot: skillVisual('lureArrow', '#66d79a', '#dbffe6', { trail: 'line', impact: 'lurePulse' }),
    trapSnare: skillVisual('trapSnare', '#66d79a', '#dbffe6', { impact: 'trapCircle' }),
    trapSpike: skillVisual('trapSpike', '#b07a47', '#ffd166', { impact: 'spikeTrap' }),
    trapTripwire: skillVisual('tripwire', '#7bdff2', '#ffffff', { impact: 'tripwire' }),
    trapDetonate: skillVisual('trapDetonate', '#ffbe55', '#fff0a6', { impact: 'trapBurst' }),
    killZone: skillVisual('killZone', '#66d79a', '#ffffff', { impact: 'killZone' }),
    companionStrike: skillVisual('companionArrow', '#8ed174', '#fff0a6', { trail: 'line', impact: 'clawSpark' })
  });

  const SKILL_VISUAL_IDS = Object.freeze({
    mage_magic_bolt: 'arcaneBolt',
    mage_arcane_burst: 'arcaneBurst',
    mage_spell_mark: 'spellMark',
    mage_energy_release: 'energyRelease',
    fire_mage_fireball: 'fireball',
    fire_mage_burning_mark: 'burningMark',
    fire_mage_heat_vent: 'heatVent',
    fire_mage_wildfire: 'wildfire',
    fire_mage_inferno_burst: 'infernoBurst',
    rune_mage_rune_mark: 'runeMark',
    rune_mage_ground_glyph: 'groundGlyph',
    rune_mage_arcane_link: 'runeLink',
    rune_mage_rune_detonation: 'runeDetonation',
    rune_mage_mana_seal: 'manaSeal',
    rune_mage_grand_inscription: 'grandInscription',
    storm_mage_chain_bolt: 'chainBolt',
    archer_quick_shot: 'quickShot',
    archer_marked_shot: 'markedShot',
    archer_piercing_arrow: 'piercingArrow',
    archer_focused_volley: 'focusedVolley',
    sniper_aimed_shot: 'aimedShot',
    sniper_weak_point_mark: 'weakPointMark',
    sniper_pierce_armor: 'pierceArmor',
    sniper_execution_shot: 'executionShot',
    sniper_one_perfect_shot: 'perfectShot',
    trapper_snare_trap: 'trapSnare',
    trapper_spike_trap: 'trapSpike',
    trapper_lure_shot: 'lureShot',
    trapper_tripwire: 'trapTripwire',
    trapper_detonate: 'trapDetonate',
    trapper_kill_zone: 'killZone',
    beast_archer_companion_strike: 'companionStrike'
  });

  function createSkillData(options) {
    const settings = options || {};
    const baseSkillIcons = settings.baseSkillIcons || BASE_SKILL_ICONS;
    const advancedSkillIcons = settings.advancedSkillIcons || ADVANCED_SKILL_ICONS;
    function skill(id, name, owner, batch, type, roleTags, prerequisites, description, options) {
      const config = options || {};
      const category = inferSkillCategory(id, type, config);
      return Object.freeze({
        id,
        name,
        owner,
        batch,
        type,
        category,
        roleTags,
        prerequisites: prerequisites || [],
        maxRank: config.maxRank || inferSkillMaxRank(id, type, config),
        defaultRank: Object.prototype.hasOwnProperty.call(config, 'defaultRank') ? Math.max(0, Math.floor(Number(config.defaultRank || 0) || 0)) : null,
        resourceCost: config.resourceCost || 0,
        cooldown: config.cooldown || 0,
        lineCount: Math.max(0, Math.floor(Number(config.lineCount || 0) || 0)),
        lineDamageScale: Number(config.lineDamageScale || 1) || 1,
        iconAsset: config.iconAsset || baseSkillIcons[id] || advancedSkillIcons[id] || '',
        iconKind: config.iconKind || inferSkillIconKind(id, type, roleTags),
        visualId: config.visualId || SKILL_VISUAL_IDS[id] || '',
        purpose: inferSkillPurpose(id, type, roleTags, config),
        primaryTraining: !!config.primaryTraining,
        passiveStats: config.passiveStats ? Object.freeze({ ...config.passiveStats }) : null,
        movementEffect: config.movementEffect ? Object.freeze({ ...config.movementEffect }) : null,
        targeting: config.targeting ? Object.freeze({ ...config.targeting }) : null,
        targetCaps: config.targetCaps ? Object.freeze({ ...config.targetCaps }) : null,
        partyEffect: config.partyEffect || '',
        futurePartyEffect: config.futurePartyEffect || '',
        description
      });
    }


    const SKILLS = Object.freeze([
      skill('fighter_heavy_strike', 'Heavy Strike', 'fighter', 'Base Skill Batch', 'Basic attack', ['Hybrid'], [], 'Trainer strike for close-range Momentum. Repeated hits on one target add extra stagger.', { resourceCost: 6, cooldown: 0.34, lineCount: 1 }),
      skill('fighter_dash_slash', 'Dash Slash', 'fighter', 'Base Skill Batch', 'Mobility', ['Hybrid'], [{ skillId: 'fighter_heavy_strike', rank: 3 }], 'Mobility slash for crossing gaps, dodging through lanes, and clipping enemies while moving.', { resourceCost: 16, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 250, distancePerRank: 5, duration: 0.2, damageRadius: 96, hitOffset: 124, invulnerable: 0.16 } }),
      skill('fighter_guard', 'Guard', 'fighter', 'Base Skill Batch', 'Defense', ['Support', 'Bossing'], [{ skillId: 'fighter_heavy_strike', rank: 3 }], 'Defensive window that reduces incoming damage and turns pressure into Momentum.', { resourceCost: 14, cooldown: 6 }),
      skill('fighter_ground_slam', 'Ground Slam', 'fighter', 'Base Skill Batch', 'Area damage', ['Mobbing'], [{ skillId: 'fighter_heavy_strike', rank: 5 }], 'Mobbing slam that staggers clustered enemies and rewards hitting three or more targets.', { resourceCost: 28, cooldown: 5, lineCount: 2 }),
      skill('fighter_power_break', 'Power Break', 'fighter', 'Base Skill Batch', 'Debuff', ['Bossing'], [{ skillId: 'fighter_ground_slam', rank: 3, any: true }, { skillId: 'fighter_heavy_strike', rank: 5, any: true }], 'Setup strike that cracks a durable target so later attacks hit harder.', { resourceCost: 22, cooldown: 6, lineCount: 1 }),
      skill('fighter_momentum_burst', 'Momentum Burst', 'fighter', 'Base Skill Batch', 'Finisher', ['Hybrid'], [{ skillId: 'fighter_heavy_strike', rank: 5 }, { anyOf: ['fighter_dash_slash', 'fighter_ground_slam', 'fighter_power_break'], rank: 3 }], 'Momentum finisher that spends stored resource for a stronger shockwave payoff.', { resourceCost: 55, cooldown: 8, lineCount: 2 }),
      skill('fighter_damage_mastery', 'Fighter Damage Mastery', 'fighter', 'Base Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Fighter damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),

      skill('mage_magic_bolt', 'Magic Bolt', 'mage', 'Base Skill Batch', 'Basic attack', ['Hybrid'], [], 'Trainer projectile that safely builds Energy at range.', { resourceCost: 7, cooldown: 0.44, lineCount: 1, targeting: projectileTargeting({ projectileType: 'magic', range: 480, rangePerRank: 4, speed: 620 }) }),
      skill('mage_blink', 'Blink', 'mage', 'Base Skill Batch', 'Mobility', ['Support'], [{ skillId: 'mage_magic_bolt', rank: 3 }], 'Mobility teleport for changing platforms or escaping pressure without adding burst damage.', { resourceCost: 12, cooldown: 0.5, movementEffect: { mode: 'blink', distance: 190, distancePerRank: 4, duration: 0, damageRadius: 68, hitOffset: 88, invulnerable: 0.24 } }),
      skill('mage_arcane_burst', 'Arcane Burst', 'mage', 'Base Skill Batch', 'Area damage', ['Mobbing'], [{ skillId: 'mage_magic_bolt', rank: 5 }], 'Mobbing blast that clears nearby enemies more efficiently than Magic Bolt.', { resourceCost: 24, cooldown: 5, lineCount: 2, targeting: projectileTargeting({ projectileType: 'magic', range: 455, rangePerRank: 5, speed: 560, explodeRadius: 108, explodeRadiusPerRank: 3 }) }),
      skill('mage_mana_shield', 'Mana Shield', 'mage', 'Base Skill Batch', 'Defense', ['Support', 'Bossing'], [{ skillId: 'mage_magic_bolt', rank: 3 }], 'Defensive conversion that spends MP to absorb incoming damage.', { resourceCost: 22, cooldown: 8 }),
      skill('mage_spell_mark', 'Spell Mark', 'mage', 'Base Skill Batch', 'Setup', ['Bossing'], [{ skillId: 'mage_magic_bolt', rank: 5 }], 'Setup mark that makes the next major spell payoff stronger.', { resourceCost: 18, cooldown: 5, lineCount: 1, targeting: projectileTargeting({ projectileType: 'magic', range: 500, rangePerRank: 4, speed: 640, applyMark: true }) }),
      skill('mage_energy_release', 'Energy Release', 'mage', 'Base Skill Batch', 'Finisher', ['Hybrid'], [{ skillId: 'mage_arcane_burst', rank: 5, any: true }, { skillId: 'mage_spell_mark', rank: 5, any: true }], 'Energy finisher that detonates around marked targets and spends stored resource for scale.', { resourceCost: 60, cooldown: 8, lineCount: 2, targeting: projectileTargeting({ projectileType: 'magic', range: 540, rangePerRank: 5, speed: 660, explodeRadius: 124, explodeRadiusPerRank: 3 }) }),
      skill('mage_damage_mastery', 'Mage Damage Mastery', 'mage', 'Base Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),

      skill('archer_quick_shot', 'Quick Shot', 'archer', 'Base Skill Batch', 'Basic attack', ['Hybrid'], [], 'Trainer shot that builds Focus quickly without needing setup.', { resourceCost: 6, cooldown: 0.38, lineCount: 2, targeting: projectileTargeting({ projectileType: 'arrow', range: 560, rangePerRank: 5, speed: 780, count: 2, spreadY: 6 }) }),
      skill('archer_roll_shot', 'Roll Shot', 'archer', 'Base Skill Batch', 'Mobility', ['Hybrid'], [{ skillId: 'archer_quick_shot', rank: 3 }], 'Mobility roll that creates a firing lane instead of replacing damage skills.', { resourceCost: 14, cooldown: 0.5, movementEffect: { mode: 'roll', direction: 'backward', distance: 170, distancePerRank: 3, duration: 0.24, damageRadius: 72, hitOffset: 92, invulnerable: 0.18 } }),
      skill('archer_marked_shot', 'Marked Shot', 'archer', 'Base Skill Batch', 'Setup', ['Bossing'], [{ skillId: 'archer_quick_shot', rank: 3 }], 'Setup shot that marks a priority enemy for Focus spender payoff.', { resourceCost: 14, cooldown: 4, lineCount: 1, targeting: projectileTargeting({ projectileType: 'arrow', range: 585, rangePerRank: 6, speed: 760, applyMark: true }) }),
      skill('archer_piercing_arrow', 'Piercing Arrow', 'archer', 'Base Skill Batch', 'Damage', ['Mobbing'], [{ skillId: 'archer_quick_shot', rank: 5 }], 'Mobbing arrow that travels through enemies lined up in a lane.', { resourceCost: 24, cooldown: 5, lineCount: 3, targeting: projectileTargeting({ projectileType: 'arrow', range: 640, rangePerRank: 7, speed: 790, pierce: 4 }) }),
      skill('archer_eagle_stance', 'Eagle Stance', 'archer', 'Base Skill Batch', 'Buff', ['Bossing'], [{ skillId: 'archer_marked_shot', rank: 3, any: true }, { skillId: 'archer_piercing_arrow', rank: 3, any: true }], 'Bossing buff window for longer range, better crits, and cleaner marked-target pressure.', { resourceCost: 18, cooldown: 10 }),
      skill('archer_focused_volley', 'Focused Volley', 'archer', 'Base Skill Batch', 'Finisher', ['Hybrid'], [{ skillId: 'archer_marked_shot', rank: 5, any: true }, { skillId: 'archer_piercing_arrow', rank: 5, any: true }], 'Focus finisher that concentrates multiple damage lines into a marked target.', { resourceCost: 55, cooldown: 8, lineCount: 5, targeting: projectileTargeting({ projectileType: 'arrow', range: 620, rangePerRank: 6, speed: 800, count: 5, spreadY: 10 }) }),
      skill('archer_damage_mastery', 'Archer Damage Mastery', 'archer', 'Base Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Archer damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),

      skill('guardian_shield_bash', 'Shield Bash', 'guardian', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Control'], [], 'Low-cooldown guarded strike for Guardian training. Staggers enemies and builds Stored Impact for larger counters.', { resourceCost: 4, cooldown: 0.48, lineCount: 1, lineDamageScale: 1.04, primaryTraining: true }),
      skill('guardian_damage_mastery', 'Guardian Damage Mastery', 'guardian', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Guardian damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('guardian_shield_dash', 'Shield Dash', 'guardian', 'Advanced Skill Batch', 'Mobility / control', ['Hybrid', 'Control'], [{ skillId: 'fighter_dash_slash', rank: 5 }, { skillId: 'guardian_shield_bash', rank: 3 }], 'Dash forward behind your shield to claim space or cross short gaps.', { resourceCost: 22, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 300, distancePerRank: 6, duration: 0.24, damageRadius: 112, hitOffset: 118, invulnerable: 0.24 } }),
      skill('guardian_impact_guard', 'Impact Guard', 'guardian', 'Advanced Skill Batch', 'Defense', ['Support', 'Bossing'], [{ skillId: 'fighter_guard', rank: 5 }], 'Block incoming damage and convert part of it into Stored Impact.', { resourceCost: 18, cooldown: 7 }),
      skill('guardian_oath_barrier', 'Oath Barrier', 'guardian', 'Advanced Skill Batch', 'Defense', ['Support'], [{ skillId: 'guardian_impact_guard', rank: 5 }], 'Create a temporary shield for yourself, stronger with Stored Impact.', { resourceCost: 25, cooldown: 12 }),
      skill('guardian_retaliation_wave', 'Retaliation Wave', 'guardian', 'Advanced Skill Batch', 'Counterattack', ['Mobbing', 'Hybrid'], [{ skillId: 'guardian_impact_guard', rank: 5 }, { skillId: 'fighter_ground_slam', rank: 5 }], 'Release Stored Impact in a forward shockwave.', { resourceCost: 40, cooldown: 8, lineCount: 2 }),
      skill('guardian_hold_the_line', 'Hold the Line', 'guardian', 'Advanced Skill Batch', 'Passive', ['Support', 'Bossing'], [{ skillId: 'guardian_impact_guard', rank: 5 }], 'Permanently improve block, defense, and Momentum control while holding ground.', { cooldown: 0, passiveStats: { defense: 0.8, block: 0.8, resourceGain: 0.4 } }),
      skill('guardian_verdict', 'Guardian\'s Verdict', 'guardian', 'Advanced Skill Batch', 'Finisher', ['Bossing', 'Party'], [{ skillId: 'guardian_retaliation_wave', rank: 5 }, { skillId: 'fighter_guard', rank: 5 }], 'Spend all Stored Impact to create a defensive explosion and personal shield.', { resourceCost: 70, cooldown: 16, lineCount: 3 }),
      skill('guardian_shield_wall', 'Shield Wall', 'guardian', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'guardian_impact_guard', rank: 3 }], 'Self-buff prototype: damage reduction, shield, knockback resistance, and Momentum when hit.', { resourceCost: 45, cooldown: 60, partyEffect: 'Self: 20% damage reduction and temporary shield.', futurePartyEffect: 'Future party: nearby allies gain 10% damage reduction, shield, and reduced knockback.' }),

      skill('berserker_blood_cleave', 'Blood Cleave', 'berserker', 'Advanced Skill Batch', 'Primary training attack', ['Bossing', 'Hybrid'], [], 'Low-cooldown risk cleave for Berserker training. Damage rises as HP drops and is best against durable targets.', { resourceCost: 5, cooldown: 0.6, lineCount: 2, lineDamageScale: 0.72, primaryTraining: true }),
      skill('berserker_damage_mastery', 'Berserker Damage Mastery', 'berserker', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Berserker damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('berserker_rage_surge', 'Rage Surge', 'berserker', 'Advanced Skill Batch', 'Buff', ['Bossing'], [{ skillId: 'berserker_blood_cleave', rank: 3 }], 'Increase attack speed and damage while reducing defense.', { resourceCost: 20, cooldown: 12 }),
      skill('berserker_reckless_leap', 'Reckless Leap', 'berserker', 'Advanced Skill Batch', 'Mobility', ['Hybrid'], [{ skillId: 'fighter_dash_slash', rank: 5 }, { skillId: 'berserker_blood_cleave', rank: 3 }], 'Leap aggressively across gaps or into position.', { resourceCost: 24, cooldown: 0.5, movementEffect: { mode: 'leap', distance: 330, distancePerRank: 7, duration: 0.34, verticalVelocity: -420, damageRadius: 128, hitOffset: 140, invulnerable: 0.26 } }),
      skill('berserker_crimson_recovery', 'Crimson Recovery', 'berserker', 'Advanced Skill Batch', 'Sustain', ['Bossing', 'Support'], [{ skillId: 'berserker_rage_surge', rank: 5 }], 'Damage enemies and heal for a portion of damage dealt.', { resourceCost: 34, cooldown: 10, lineCount: 2 }),
      skill('berserker_pain_to_power', 'Pain to Power', 'berserker', 'Advanced Skill Batch', 'Passive', ['Bossing'], [{ skillId: 'berserker_blood_cleave', rank: 5 }], 'Permanently improve power and Rage generation as wounds become fuel.', { cooldown: 0, passiveStats: { power: 0.8, resourceGain: 0.9 } }),
      skill('berserker_last_stand', 'Last Stand', 'berserker', 'Advanced Skill Batch', 'Finisher', ['Bossing'], [{ skillId: 'berserker_pain_to_power', rank: 5 }, { skillId: 'fighter_momentum_burst', rank: 10 }], 'Lethal damage leaves you at 1 HP briefly and greatly increases damage.', { resourceCost: 70, cooldown: 18, lineCount: 3 }),
      skill('berserker_war_cry', 'War Cry', 'berserker', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'berserker_rage_surge', rank: 3 }], 'Self-buff prototype: attack power, Rage generation, and small lifesteal.', { resourceCost: 48, cooldown: 75, partyEffect: 'Self: 18% attack power, Rage generation, and capped lifesteal.', futurePartyEffect: 'Future party: nearby allies gain 8% attack power, minor healing on hit, and resource generation.' }),

      skill('duelist_quick_cut', 'Quick Cut', 'duelist', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Bossing'], [], 'Low-cooldown precise cut for Duelist training. Repeated hits on one target build Tempo.', { resourceCost: 4, cooldown: 0.46, lineCount: 3, lineDamageScale: 0.58, primaryTraining: true }),
      skill('duelist_damage_mastery', 'Duelist Damage Mastery', 'duelist', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Duelist damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('duelist_flash_step', 'Flash Step', 'duelist', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'fighter_dash_slash', rank: 5 }, { skillId: 'duelist_quick_cut', rank: 3 }], 'Dash through a short opening while preserving Tempo for follow-up attacks.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 315, distancePerRank: 8, duration: 0.16, damageRadius: 74, hitOffset: 112, invulnerable: 0.24 } }),
      skill('duelist_rallying_flourish', 'Rallying Flourish', 'duelist', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'duelist_quick_cut', rank: 3 }], 'Self-buff prototype: haste, precision chance, and Tempo generation after chained attacks.', { resourceCost: 44, cooldown: 65, partyEffect: 'Self: movement speed, precision chance, and faster Tempo generation.', futurePartyEffect: 'Future party: nearby allies gain minor haste and precision chance during burst windows.' }),

      skill('fire_mage_fireball', 'Fireball', 'fireMage', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Mobbing'], [], 'Low-cooldown fire projectile for Fire Mage training. Explodes on impact and starts burn spread setups.', { resourceCost: 5, cooldown: 0.62, lineCount: 2, lineDamageScale: 0.76, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'fire', range: 500, rangePerRank: 5, speed: 560, explodeRadius: 92, explodeRadiusPerRank: 3, applyBurn: true }) }),
      skill('fire_mage_damage_mastery', 'Fire Mage Damage Mastery', 'fireMage', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Fire Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('fire_mage_flame_trail', 'Flame Trail', 'fireMage', 'Advanced Skill Batch', 'Movement effect', ['Mobbing', 'Support'], [{ skillId: 'mage_blink', rank: 5 }, { skillId: 'fire_mage_fireball', rank: 3 }], 'Dash forward and leave a burning trail behind.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 285, distancePerRank: 7, duration: 0.22, damageRadius: 102, hitOffset: 104, invulnerable: 0.22, trail: 'flame' } }),
      skill('fire_mage_burning_mark', 'Burning Mark', 'fireMage', 'Advanced Skill Batch', 'Damage-over-time', ['Bossing', 'Hybrid'], [{ skillId: 'mage_spell_mark', rank: 5 }, { skillId: 'fire_mage_fireball', rank: 3 }], 'Mark an enemy with burn. Fire skills spread the burn.', { resourceCost: 18, cooldown: 6, lineCount: 1, targeting: projectileTargeting({ projectileType: 'fire', range: 510, rangePerRank: 4, speed: 610, applyMark: true, applyBurn: true }) }),
      skill('fire_mage_heat_vent', 'Heat Vent', 'fireMage', 'Advanced Skill Batch', 'Resource control', ['Hybrid'], [{ skillId: 'fire_mage_fireball', rank: 5 }], 'Release Heat in a cone of flame, preventing Overheat.', { resourceCost: 30, cooldown: 8, lineCount: 2 }),
      skill('fire_mage_wildfire', 'Wildfire', 'fireMage', 'Advanced Skill Batch', 'Area damage', ['Mobbing'], [{ skillId: 'fire_mage_burning_mark', rank: 5 }, { skillId: 'mage_arcane_burst', rank: 5 }], 'Burn spreads between nearby enemies.', { resourceCost: 35, cooldown: 10, lineCount: 3, targeting: projectileTargeting({ projectileType: 'fire', range: 500, rangePerRank: 4, speed: 560, explodeRadius: 148, explodeRadiusPerRank: 4, applyBurn: true }) }),
      skill('fire_mage_inferno_burst', 'Inferno Burst', 'fireMage', 'Advanced Skill Batch', 'Finisher', ['Hybrid', 'Bossing'], [{ skillId: 'fire_mage_heat_vent', rank: 5 }, { skillId: 'mage_energy_release', rank: 10 }], 'Consume all Heat for a large explosion. At max Heat, harms the caster slightly.', { resourceCost: 75, cooldown: 16, lineCount: 3, targeting: projectileTargeting({ projectileType: 'fire', range: 470, rangePerRank: 4, speed: 530, explodeRadius: 168, explodeRadiusPerRank: 5, applyBurn: true }) }),
      skill('fire_mage_ignition_aura', 'Ignition Aura', 'fireMage', 'Advanced Skill Batch', 'Party skill', ['Mobbing', 'Party'], [{ skillId: 'fire_mage_fireball', rank: 3 }], 'Self-buff prototype: fire damage, burn power, Heat generation, and longer burns.', { resourceCost: 52, cooldown: 75, partyEffect: 'Self: 20% increased fire and burn damage, faster Heat, and stronger burn uptime.', futurePartyEffect: 'Future party: nearby allies deal 8% bonus damage against burning enemies and can apply minor burn.' }),

      skill('rune_mage_rune_mark', 'Rune Mark', 'runeMage', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Control'], [], 'Low-cooldown rune bolt for Rune Mage training. Marks and links targets for later detonations.', { resourceCost: 4, cooldown: 0.58, lineCount: 1, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'rune', range: 510, rangePerRank: 5, speed: 620, applyMark: true, applySlow: true }) }),
      skill('rune_mage_damage_mastery', 'Rune Mage Damage Mastery', 'runeMage', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Rune Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('rune_mage_rune_blink', 'Rune Blink', 'runeMage', 'Advanced Skill Batch', 'Mobility', ['Support', 'Hybrid'], [{ skillId: 'mage_blink', rank: 5 }, { skillId: 'rune_mage_rune_mark', rank: 3 }], 'Blink through a short rune fold to reposition without breaking setup flow.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'blink', distance: 285, distancePerRank: 7, duration: 0, damageRadius: 72, hitOffset: 88, invulnerable: 0.3 } }),
      skill('rune_mage_ground_glyph', 'Ground Glyph', 'runeMage', 'Advanced Skill Batch', 'Area setup', ['Mobbing', 'Control'], [{ skillId: 'rune_mage_rune_mark', rank: 3, any: true }, { skillId: 'mage_arcane_burst', rank: 5, any: true }], 'Place a wide rune field that damages enemies, slows and weakens enemies inside it, and grants modest recovery and casting haste while you stand within it.', { resourceCost: 32, cooldown: 9, lineCount: 2, targetCaps: { area: 6, field: 6 } }),
      skill('rune_mage_arcane_link', 'Arcane Link', 'runeMage', 'Advanced Skill Batch', 'Combo', ['Bossing', 'Hybrid'], [{ skillId: 'rune_mage_rune_mark', rank: 5 }], 'Link two runes so damage to one partially affects another.', { resourceCost: 24, cooldown: 7, lineCount: 2, targeting: projectileTargeting({ projectileType: 'rune', range: 520, rangePerRank: 5, speed: 620, count: 2, spreadY: 12, applyMark: true }) }),
      skill('rune_mage_rune_detonation', 'Rune Detonation', 'runeMage', 'Advanced Skill Batch', 'Burst', ['Hybrid'], [{ skillId: 'rune_mage_ground_glyph', rank: 5, any: true }, { skillId: 'rune_mage_arcane_link', rank: 5, any: true }], 'Detonate active runes for burst damage.', { resourceCost: 35, cooldown: 9, lineCount: 3, targetCaps: { runeDetonation: 6 } }),
      skill('rune_mage_mana_seal', 'Mana Seal', 'runeMage', 'Advanced Skill Batch', 'Control', ['Bossing', 'Support'], [{ skillId: 'rune_mage_rune_mark', rank: 5 }, { skillId: 'mage_mana_shield', rank: 5 }], 'Seal an enemy briefly, reducing movement or casting.', { resourceCost: 28, cooldown: 12, lineCount: 1, targeting: projectileTargeting({ projectileType: 'rune', range: 500, rangePerRank: 4, speed: 600, applySlow: true, applyMark: true }) }),
      skill('rune_mage_grand_inscription', 'Grand Inscription', 'runeMage', 'Advanced Skill Batch', 'Finisher', ['Mobbing', 'Support'], [{ skillId: 'rune_mage_rune_detonation', rank: 5 }, { skillId: 'mage_energy_release', rank: 10 }], 'Place a massive inscription field with stronger recovery, haste, enemy slow, and Rune Mark explosion bonuses while you stand within it.', { resourceCost: 88, cooldown: 24, lineCount: 3, targetCaps: { finisherArea: 8, field: 8 } }),
      skill('rune_mage_rune_circle', 'Rune Circle', 'runeMage', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'rune_mage_ground_glyph', rank: 3 }], 'Self-buff prototype with selectable Power, Guard, Focus, Cleanse, or Haste rune modes.', { resourceCost: 45, cooldown: 60, partyEffect: 'Self: selected rune grants skill damage, reduction, resource generation, cleanse, or haste.', futurePartyEffect: 'Future party: allies inside the circle receive reduced selected rune effects.' }),

      skill('storm_mage_chain_bolt', 'Chain Bolt', 'stormMage', 'Advanced Skill Batch', 'Primary training attack', ['Mobbing', 'Hybrid'], [], 'Low-cooldown lightning chain for Storm Mage training. Efficiently clears clustered enemies but has lower single-target boss value.', { resourceCost: 6, cooldown: 0.78, lineCount: 3, lineDamageScale: 0.58, primaryTraining: true, targeting: projectileTargeting({ mode: 'chain', projectileType: 'lightning', range: 380, rangePerRank: 4, speed: 720, chainRange: 220, chainRangePerRank: 6, chainTargets: 3, chainTargetsPerRanks: 2, maxChainTargets: 8, chainDamageFalloff: 0.92, applySlow: true }) }),
      skill('storm_mage_damage_mastery', 'Storm Mage Damage Mastery', 'stormMage', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Storm Mage damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('storm_mage_static_shift', 'Static Shift', 'stormMage', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'mage_blink', rank: 5 }, { skillId: 'storm_mage_chain_bolt', rank: 3 }], 'Blink through static current to reposition without breaking spell flow.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'blink', distance: 300, distancePerRank: 8, duration: 0, damageRadius: 70, hitOffset: 88, invulnerable: 0.3 } }),
      skill('storm_mage_stormfront', 'Stormfront', 'stormMage', 'Advanced Skill Batch', 'Party skill', ['Mobbing', 'Party'], [{ skillId: 'storm_mage_chain_bolt', rank: 3 }], 'Self-buff prototype: faster Charge generation, bonus area damage, and lightning chain reach.', { resourceCost: 50, cooldown: 70, partyEffect: 'Self: Charge generation, area damage, and lightning reach.', futurePartyEffect: 'Future party: nearby allies gain minor skill haste and bonus damage to shocked enemies.' }),

      skill('sniper_aimed_shot', 'Aimed Shot', 'sniper', 'Advanced Skill Batch', 'Primary training attack', ['Bossing'], [], 'Low-cooldown precision shot for Sniper training. Strongest when held at range or aimed into weak points.', { resourceCost: 5, cooldown: 0.65, lineCount: 1, lineDamageScale: 0.92, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'arrow', range: 720, rangePerRank: 8, speed: 900 }) }),
      skill('sniper_damage_mastery', 'Sniper Damage Mastery', 'sniper', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Sniper damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('sniper_combat_roll', 'Combat Roll', 'sniper', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'archer_roll_shot', rank: 5 }, { skillId: 'sniper_aimed_shot', rank: 3 }], 'Roll backward into a clean firing lane while briefly avoiding damage.', { resourceCost: 16, cooldown: 0.5, movementEffect: { mode: 'roll', direction: 'backward', distance: 245, distancePerRank: 5, duration: 0.24, damageRadius: 74, hitOffset: 90, invulnerable: 0.25 } }),
      skill('sniper_weak_point_mark', 'Weak Point Mark', 'sniper', 'Advanced Skill Batch', 'Setup', ['Bossing'], [{ skillId: 'archer_marked_shot', rank: 5 }], 'Mark a target weak point. The next heavy shot deals bonus damage.', { resourceCost: 18, cooldown: 6, lineCount: 1, targeting: projectileTargeting({ projectileType: 'arrow', range: 725, rangePerRank: 8, speed: 880, applyMark: true }) }),
      skill('sniper_steady_breath', 'Steady Breath', 'sniper', 'Advanced Skill Batch', 'Passive', ['Bossing'], [{ skillId: 'sniper_aimed_shot', rank: 5 }], 'Permanently improve range, precision damage, and steady Aim control.', { cooldown: 0, passiveStats: { range: 4, critDamage: 2, resourceGain: 0.4 } }),
      skill('sniper_pierce_armor', 'Pierce Armor', 'sniper', 'Advanced Skill Batch', 'Debuff', ['Bossing', 'Support'], [{ skillId: 'sniper_weak_point_mark', rank: 5 }, { skillId: 'archer_piercing_arrow', rank: 5 }], 'Shot that lowers enemy defense, stronger against marked enemies.', { resourceCost: 26, cooldown: 7, lineCount: 3, targeting: projectileTargeting({ projectileType: 'arrow', range: 730, rangePerRank: 8, speed: 860, pierce: 3, applyCrack: true }) }),
      skill('sniper_execution_shot', 'Execution Shot', 'sniper', 'Advanced Skill Batch', 'Finisher', ['Bossing'], [{ skillId: 'sniper_weak_point_mark', rank: 5 }, { skillId: 'sniper_aimed_shot', rank: 5 }], 'Massive damage against low-health or weak-point-marked enemies.', { resourceCost: 45, cooldown: 12, lineCount: 4, targeting: projectileTargeting({ projectileType: 'arrow', range: 720, rangePerRank: 8, speed: 890, pierce: 1 }) }),
      skill('sniper_one_perfect_shot', 'One Perfect Shot', 'sniper', 'Advanced Skill Batch', 'Ultimate-style', ['Bossing'], [{ skillId: 'sniper_execution_shot', rank: 5 }, { skillId: 'archer_focused_volley', rank: 10 }], 'Consume all Aim to fire a huge single-target attack.', { resourceCost: 80, cooldown: 18, lineCount: 1, targeting: projectileTargeting({ projectileType: 'arrow', range: 790, rangePerRank: 10, speed: 940, width: 36, height: 12 }) }),
      skill('sniper_eagle_eye', 'Eagle Eye', 'sniper', 'Advanced Skill Batch', 'Party skill', ['Bossing', 'Party'], [{ skillId: 'sniper_weak_point_mark', rank: 3 }], 'Self-buff prototype: precision chance, precision damage, range, and Aim generation.', { resourceCost: 48, cooldown: 75, partyEffect: 'Self: 15% precision chance, 20% precision damage, increased range, and Aim against marked enemies.', futurePartyEffect: 'Future party: nearby allies gain 8% precision chance, 10% precision damage, and weak-point payoff.' }),

      skill('trapper_snare_trap', 'Snare Trap', 'trapper', 'Advanced Skill Batch', 'Primary training attack', ['Mobbing', 'Control'], [], 'Low-cooldown trap for Trapper training. Arms quickly, slows enemies, and sets up manual detonations.', { resourceCost: 4, cooldown: 0.52, lineCount: 1, lineDamageScale: 0.82, primaryTraining: true }),
      skill('trapper_damage_mastery', 'Trapper Damage Mastery', 'trapper', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Trapper damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('trapper_grapple_dash', 'Grapple Dash', 'trapper', 'Advanced Skill Batch', 'Mobility / utility', ['Hybrid', 'Control'], [{ skillId: 'archer_roll_shot', rank: 5 }, { skillId: 'trapper_snare_trap', rank: 3 }], 'Fire a short grapple line and dash into a better trap angle.', { resourceCost: 18, cooldown: 0.5, movementEffect: { mode: 'dash', distance: 285, distancePerRank: 7, duration: 0.25, damageRadius: 82, hitOffset: 110, invulnerable: 0.22 } }),
      skill('trapper_spike_trap', 'Spike Trap', 'trapper', 'Advanced Skill Batch', 'Damage', ['Mobbing'], [{ skillId: 'trapper_snare_trap', rank: 3 }], 'Place a trap that deals damage when stepped on.', { resourceCost: 20, cooldown: 6, lineCount: 3 }),
      skill('trapper_lure_shot', 'Lure Shot', 'trapper', 'Advanced Skill Batch', 'Utility', ['Control'], [{ skillId: 'archer_marked_shot', rank: 5 }, { skillId: 'trapper_snare_trap', rank: 3 }], 'Fire a shot that draws enemies toward a target area.', { resourceCost: 18, cooldown: 7, lineCount: 2, targeting: projectileTargeting({ projectileType: 'arrow', range: 590, rangePerRank: 6, speed: 760, applySlow: true, applyMark: true }) }),
      skill('trapper_tripwire', 'Tripwire', 'trapper', 'Advanced Skill Batch', 'Combo trap', ['Mobbing', 'Control'], [{ skillId: 'trapper_snare_trap', rank: 5 }, { skillId: 'trapper_spike_trap', rank: 3 }], 'Place a line trap that triggers when enemies cross it.', { resourceCost: 25, cooldown: 8, lineCount: 3 }),
      skill('trapper_detonate', 'Detonate', 'trapper', 'Advanced Skill Batch', 'Trigger', ['Hybrid'], [{ skillId: 'trapper_spike_trap', rank: 5, any: true }, { skillId: 'trapper_tripwire', rank: 5, any: true }], 'Manually trigger active traps.', { resourceCost: 24, cooldown: 7, lineCount: 3, targetCaps: { trapDetonate: 8 } }),
      skill('trapper_kill_zone', 'Kill Zone', 'trapper', 'Advanced Skill Batch', 'Finisher', ['Mobbing'], [{ skillId: 'trapper_detonate', rank: 5 }, { skillId: 'archer_focused_volley', rank: 10 }], 'Place a large trap field that chains trap activations.', { resourceCost: 70, cooldown: 18, lineCount: 5 }),
      skill('trapper_tactical_field', 'Tactical Field', 'trapper', 'Advanced Skill Batch', 'Party skill', ['Control', 'Party'], [{ skillId: 'trapper_snare_trap', rank: 3 }], 'Self-buff prototype: trap damage, arming speed, damage reduction inside field, and Focus on trap trigger.', { resourceCost: 46, cooldown: 70, partyEffect: 'Self: 15% trap damage, faster arming speed, field mitigation, and Focus from traps.', futurePartyEffect: 'Future party: allies gain damage reduction from enemies inside the field and bonus damage to controlled enemies.' }),

      skill('beast_archer_companion_strike', 'Companion Strike', 'beastArcher', 'Advanced Skill Batch', 'Primary training attack', ['Hybrid', 'Support'], [], 'Low-cooldown coordinated shot for Beast Archer training. Marks targets for companion pressure and sustain.', { resourceCost: 5, cooldown: 0.58, lineCount: 3, lineDamageScale: 0.62, primaryTraining: true, targeting: projectileTargeting({ projectileType: 'arrow', range: 575, rangePerRank: 6, speed: 780, count: 2, spreadY: 14, applyMark: true }) }),
      skill('beast_archer_damage_mastery', 'Beast Archer Damage Mastery', 'beastArcher', 'Advanced Skill Batch', 'Passive', ['Hybrid'], [], 'Permanently tighten Beast Archer damage rolls by raising the minimum damage floor.', { cooldown: 0, maxRank: 20, defaultRank: 0, passiveStats: { damageFloor: 2 } }),
      skill('beast_archer_pounce_roll', 'Pounce Roll', 'beastArcher', 'Advanced Skill Batch', 'Mobility', ['Hybrid', 'Support'], [{ skillId: 'archer_roll_shot', rank: 5 }, { skillId: 'beast_archer_companion_strike', rank: 3 }], 'Roll into a pounce lane, repositioning while your companion covers the retreat.', { resourceCost: 17, cooldown: 0.5, movementEffect: { mode: 'roll', direction: 'backward', distance: 255, distancePerRank: 5, duration: 0.24, damageRadius: 72, hitOffset: 92, invulnerable: 0.25 } }),
      skill('beast_archer_pack_call', 'Pack Call', 'beastArcher', 'Advanced Skill Batch', 'Party skill', ['Support', 'Party'], [{ skillId: 'beast_archer_companion_strike', rank: 3 }], 'Self-buff prototype: Bond generation, max HP, and steady MP recovery during coordinated attacks.', { resourceCost: 45, cooldown: 70, partyEffect: 'Self: Bond generation, max HP, and MP recovery while attacking.', futurePartyEffect: 'Future party: nearby allies gain minor sustain and resource recovery.' })
    ]);

    return {
      SKILL_PURPOSES,
      SKILL_VISUALS,
      SKILL_VISUAL_IDS,
      SKILLS
    };
  }

  const api = {
    BASE_SKILL_ICONS,
    ADVANCED_SKILL_ICONS,
    CLASS_ICON_ASSETS,
    SKILL_PURPOSES,
    SKILL_VISUALS,
    SKILL_VISUAL_IDS,
    createSkillData
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.skills = Object.assign({}, modules.skills || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
