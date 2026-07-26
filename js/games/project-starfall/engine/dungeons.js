(function initProjectStarfallEngineDungeons(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  const DEFAULT_BOSS_RESPAWN_SECONDS = 30;
  const DUNGEON_ENCOUNTER_FLOW_VERSION = 1;
  const DEFAULT_DUNGEON_BOSS_INTRO_DELAY_MS = 2600;

  function getDungeonData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getDungeonObjectiveDefinitions(dungeonId, options) {
    const data = getDungeonData(options);
    const dungeon = getById(data.DUNGEONS || [], dungeonId);
    if (!dungeon) return [];
    return data.DUNGEON_OBJECTIVES || [];
  }

  function getActiveDungeonDefinition(dungeons, options) {
    const data = getDungeonData(options);
    const state = dungeons && typeof dungeons === 'object' ? dungeons : {};
    const id = normalizeId(state.activeDungeonId);
    if (!id) return null;
    return (data.DUNGEONS || []).find((dungeon) => normalizeId(dungeon && dungeon.id) === id) || null;
  }

  function getDungeonDefinitionById(dungeonId, options) {
    const data = getDungeonData(options);
    const id = normalizeId(dungeonId);
    if (!id) return null;
    return (data.DUNGEONS || []).find((dungeon) => normalizeId(dungeon && dungeon.id) === id) || null;
  }

  function getDungeonBossIds(map, options) {
    if (!map || !map.isDungeon) return [];
    const dungeon = map.dungeonId ? getDungeonDefinitionById(map.dungeonId, options) : null;
    const configured = map.bossId
      ? [map.bossId]
      : dungeon && Array.isArray(dungeon.bossIds) && dungeon.bossIds.length
      ? dungeon.bossIds
      : dungeon && dungeon.bossId
        ? [dungeon.bossId]
        : [];
    return Array.from(new Set(configured.filter(Boolean)));
  }

  function getDungeonDefinitionByMapId(mapId, options) {
    const data = getDungeonData(options);
    const id = normalizeId(mapId);
    if (!id) return null;
    return (data.DUNGEONS || []).find((dungeon) => normalizeId(dungeon && dungeon.mapId) === id) || null;
  }

  function normalizeDungeonEncounterIdList(value) {
    return Array.from(new Set((Array.isArray(value) ? value : [])
      .map(normalizeId)
      .filter(Boolean)));
  }

  function normalizeDungeonEncounterBeat(value, index) {
    const source = value && typeof value === 'object' ? value : {};
    const kind = normalizeId(source.kind || source.type) === 'boss' ? 'boss' : 'combat';
    const enemyIds = (Array.isArray(source.enemyIds) ? source.enemyIds : [])
      .map(normalizeId)
      .filter(Boolean);
    const bossIds = normalizeDungeonEncounterIdList(source.bossIds);
    return {
      id: normalizeId(source.id) || `beat_${Math.max(0, Number(index || 0)) + 1}`,
      kind,
      name: String(source.name || source.label || `Beat ${Math.max(0, Number(index || 0)) + 1}`),
      summary: String(source.summary || ''),
      sectionIds: normalizeDungeonEncounterIdList(source.sectionIds),
      spawnGroupIds: normalizeDungeonEncounterIdList(source.spawnGroupIds),
      enemyIds: kind === 'boss' ? [] : enemyIds,
      bossIds: kind === 'boss' ? bossIds : [],
      gateX: Math.max(0, Number(source.gateX || 0))
    };
  }

  function getDungeonEncounterFlowDefinition(dungeonId, options) {
    const dungeon = getDungeonDefinitionById(dungeonId, options);
    const source = dungeon && dungeon.encounterFlow && typeof dungeon.encounterFlow === 'object'
      ? dungeon.encounterFlow
      : null;
    if (!source || !Array.isArray(source.beats) || !source.beats.length) return null;
    const beats = source.beats.map(normalizeDungeonEncounterBeat).filter((beat) =>
      beat.id && (beat.kind === 'boss' ? beat.bossIds.length : beat.enemyIds.length));
    if (!beats.length) return null;
    return {
      version: DUNGEON_ENCOUNTER_FLOW_VERSION,
      id: normalizeId(source.id) || `${normalizeId(dungeon.id)}_route`,
      dungeonId: normalizeId(dungeon.id),
      mapId: normalizeId(dungeon.mapId),
      bossIntroDelayMs: Math.max(1200, Number(source.bossIntroDelaySeconds || 0) * 1000 || DEFAULT_DUNGEON_BOSS_INTRO_DELAY_MS),
      bossHpScale: Math.max(1, Number(source.bossHpScale || 1) || 1),
      beats,
      bossBeatIndex: beats.findIndex((beat) => beat.kind === 'boss'),
      preBossBeatCount: beats.filter((beat) => beat.kind !== 'boss').length
    };
  }

  function getDungeonEncounterBeatEnemyIds(beat) {
    if (!beat) return [];
    return (beat.kind === 'boss' ? beat.bossIds : beat.enemyIds).slice();
  }

  function createDungeonEncounterBeatSlots(beat) {
    return getDungeonEncounterBeatEnemyIds(beat).map((enemyId, index) => ({
      id: `${beat.id}__${index + 1}`,
      beatId: beat.id,
      enemyId,
      index,
      boss: beat.kind === 'boss'
    }));
  }

  function createDungeonEncounterFlowRunState(dungeonId, value, options) {
    const settings = options || {};
    const definition = getDungeonEncounterFlowDefinition(dungeonId, settings);
    if (!definition) return null;
    const source = value && typeof value === 'object' ? value : {};
    const beatIds = definition.beats.map((beat) => beat.id);
    const validBeatIds = new Set(beatIds);
    const slots = definition.beats.flatMap(createDungeonEncounterBeatSlots);
    const validSlotIds = new Set(slots.map((slot) => slot.id));
    const requestedBeatId = normalizeId(source.activeBeatId);
    const requestedBeatIndex = requestedBeatId && validBeatIds.has(requestedBeatId)
      ? beatIds.indexOf(requestedBeatId)
      : clamp(Math.floor(Number(source.activeBeatIndex || 0) || 0), 0, definition.beats.length - 1);
    const completedBeatIds = normalizeDungeonEncounterIdList(source.completedBeatIds)
      .filter((id) => validBeatIds.has(id));
    for (let index = 0; index < requestedBeatIndex; index += 1) {
      if (!completedBeatIds.includes(beatIds[index])) completedBeatIds.push(beatIds[index]);
    }
    const legacyComplete = (!value || typeof value !== 'object') && (
      Number(settings.completedAt || 0) > 0 ||
      settings.bossDefeated === true
    );
    const requestedComplete = normalizeId(source.status) === 'complete' || legacyComplete;
    if (requestedComplete) {
      beatIds.forEach((id) => {
        if (!completedBeatIds.includes(id)) completedBeatIds.push(id);
      });
    }
    const defeatedSlotIds = normalizeDungeonEncounterIdList(source.defeatedSlotIds)
      .filter((id) => validSlotIds.has(id));
    definition.beats.forEach((beat) => {
      if (!completedBeatIds.includes(beat.id)) return;
      createDungeonEncounterBeatSlots(beat).forEach((slot) => {
        if (!defeatedSlotIds.includes(slot.id)) defeatedSlotIds.push(slot.id);
      });
    });
    const complete = beatIds.every((id) => completedBeatIds.includes(id));
    const firstIncompleteIndex = definition.beats.findIndex((beat) => !completedBeatIds.includes(beat.id));
    const activeBeatIndex = complete
      ? definition.beats.length - 1
      : firstIncompleteIndex >= 0
        ? firstIncompleteIndex
        : requestedBeatIndex;
    const activeBeat = definition.beats[activeBeatIndex];
    const nowMs = Number.isFinite(Number(settings.nowMs)) ? Math.max(0, Number(settings.nowMs)) : Date.now();
    let bossRevealedAt = Math.max(0, Number(source.bossRevealedAt || 0));
    let bossArmedAt = Math.max(0, Number(source.bossArmedAt || 0));
    if (!complete && activeBeat && activeBeat.kind === 'boss') {
      if (!bossRevealedAt) bossRevealedAt = nowMs;
      if (!bossArmedAt) bossArmedAt = bossRevealedAt + definition.bossIntroDelayMs;
    }
    return {
      version: DUNGEON_ENCOUNTER_FLOW_VERSION,
      id: definition.id,
      dungeonId: definition.dungeonId,
      mapId: definition.mapId,
      status: complete ? 'complete' : activeBeat && activeBeat.kind === 'boss' ? 'boss' : 'active',
      activeBeatIndex,
      activeBeatId: activeBeat ? activeBeat.id : '',
      completedBeatIds,
      spawnedBeatIds: normalizeDungeonEncounterIdList(source.spawnedBeatIds).filter((id) => validBeatIds.has(id)),
      defeatedSlotIds,
      startedAt: Math.max(0, Number(source.startedAt || settings.startedAt || 0)),
      beatStartedAt: Math.max(0, Number(source.beatStartedAt || settings.startedAt || 0)),
      bossRevealedAt,
      bossArmedAt,
      completedAt: complete ? Math.max(0, Number(source.completedAt || settings.completedAt || nowMs)) : 0
    };
  }

  function ensureDungeonEncounterFlowRunState(run, options) {
    if (!run || !run.dungeonId) return null;
    const settings = Object.assign({}, options || {}, {
      startedAt: Number(run.startedAt || 0),
      completedAt: Number(run.completedAt || 0),
      bossDefeated: !!run.bossDefeated
    });
    const flow = createDungeonEncounterFlowRunState(run.dungeonId, run.encounterFlow, settings);
    if (flow) run.encounterFlow = flow;
    else delete run.encounterFlow;
    return flow;
  }

  function markDungeonEncounterBeatSpawned(run, beatId, options) {
    const flow = ensureDungeonEncounterFlowRunState(run, options);
    const definition = flow ? getDungeonEncounterFlowDefinition(run.dungeonId, options) : null;
    const id = normalizeId(beatId);
    if (!flow || !definition || !definition.beats.some((beat) => beat.id === id)) return false;
    if (!flow.spawnedBeatIds.includes(id)) flow.spawnedBeatIds.push(id);
    return true;
  }

  function recordDungeonEncounterEnemyDefeat(run, enemy, options) {
    const settings = options || {};
    const flow = ensureDungeonEncounterFlowRunState(run, settings);
    const definition = flow ? getDungeonEncounterFlowDefinition(run.dungeonId, settings) : null;
    const rejected = (reason) => ({ accepted: false, advanced: false, complete: false, bossRevealed: false, reason });
    if (!flow || !definition || flow.status === 'complete' || Number(run.completedAt || 0) > 0) return rejected('inactive');
    const beat = definition.beats[flow.activeBeatIndex];
    const beatId = normalizeId(enemy && enemy.dungeonBeatId);
    const slotId = normalizeId(enemy && enemy.dungeonBeatSlotId);
    if (!enemy || beatId !== beat.id || !slotId) return rejected('wrong-beat');
    if (normalizeId(enemy.dungeonBeatDungeonId) !== definition.dungeonId) return rejected('wrong-dungeon');
    if (normalizeId(enemy.dungeonBeatMapId) !== definition.mapId) return rejected('wrong-map');
    if (Number(enemy.dungeonBeatRunStartedAt || 0) !== Number(run.startedAt || 0)) return rejected('wrong-run');
    if (settings.mapId && normalizeId(settings.mapId) !== definition.mapId) return rejected('wrong-runtime-map');
    const slots = createDungeonEncounterBeatSlots(beat);
    const matchedSlot = slots.find((slot) => slot.id === slotId);
    if (!matchedSlot) return rejected('wrong-slot');
    if (normalizeId(enemy.id) !== matchedSlot.enemyId) return rejected('wrong-enemy');
    if (flow.defeatedSlotIds.includes(slotId)) return rejected('duplicate');
    flow.defeatedSlotIds.push(slotId);
    const beatComplete = slots.every((slot) => flow.defeatedSlotIds.includes(slot.id));
    if (!beatComplete) {
      return {
        accepted: true,
        advanced: false,
        complete: false,
        bossRevealed: false,
        beatId: beat.id,
        nextBeatId: beat.id,
        defeated: slots.filter((slot) => flow.defeatedSlotIds.includes(slot.id)).length,
        goal: slots.length
      };
    }
    if (!flow.completedBeatIds.includes(beat.id)) flow.completedBeatIds.push(beat.id);
    const nowMs = Number.isFinite(Number(settings.nowMs)) ? Math.max(0, Number(settings.nowMs)) : Date.now();
    const nextBeatIndex = flow.activeBeatIndex + 1;
    if (nextBeatIndex >= definition.beats.length) {
      flow.status = 'complete';
      flow.completedAt = nowMs;
      return {
        accepted: true,
        advanced: true,
        complete: true,
        bossRevealed: false,
        beatId: beat.id,
        nextBeatId: '',
        defeated: slots.length,
        goal: slots.length
      };
    }
    const nextBeat = definition.beats[nextBeatIndex];
    flow.activeBeatIndex = nextBeatIndex;
    flow.activeBeatId = nextBeat.id;
    flow.status = nextBeat.kind === 'boss' ? 'boss' : 'active';
    flow.beatStartedAt = nowMs;
    if (nextBeat.kind === 'boss') {
      flow.bossRevealedAt = nowMs;
      flow.bossArmedAt = nowMs + definition.bossIntroDelayMs;
    }
    return {
      accepted: true,
      advanced: true,
      complete: false,
      bossRevealed: nextBeat.kind === 'boss',
      beatId: beat.id,
      nextBeatId: nextBeat.id,
      defeated: slots.length,
      goal: slots.length
    };
  }

  function isDungeonEncounterFlowComplete(dungeonId, run, options) {
    const definition = getDungeonEncounterFlowDefinition(dungeonId, options);
    if (!definition) return true;
    if (!run || normalizeId(run.dungeonId) !== definition.dungeonId) return false;
    const flow = createDungeonEncounterFlowRunState(dungeonId, run.encounterFlow, Object.assign({}, options || {}, {
      startedAt: Number(run.startedAt || 0),
      completedAt: Number(run.completedAt || 0),
      bossDefeated: !!run.bossDefeated
    }));
    if (!flow || flow.status !== 'complete') return false;
    const finalBeat = definition.beats[definition.beats.length - 1];
    return finalBeat.kind === 'boss' && createDungeonEncounterBeatSlots(finalBeat)
      .every((slot) => flow.defeatedSlotIds.includes(slot.id));
  }

  function getDungeonEncounterCompletionBlockReason(dungeonId, run, options) {
    const definition = getDungeonEncounterFlowDefinition(dungeonId, options);
    if (!definition) return '';
    if (isDungeonEncounterFlowComplete(dungeonId, run, options)) return '';
    if (!run || normalizeId(run.dungeonId) !== definition.dungeonId) return 'Start the expedition route before claiming a clear.';
    const flow = createDungeonEncounterFlowRunState(dungeonId, run.encounterFlow, Object.assign({}, options || {}, {
      startedAt: Number(run.startedAt || 0)
    }));
    const beat = flow && definition.beats[flow.activeBeatIndex];
    return beat ? `Complete ${beat.name} before claiming the expedition clear.` : 'Complete the expedition route before claiming a clear.';
  }

  function createDungeonEncounterFlowSnapshot(dungeonId, run, options) {
    const settings = options || {};
    const definition = getDungeonEncounterFlowDefinition(dungeonId, settings);
    if (!definition || !run || normalizeId(run.dungeonId) !== definition.dungeonId) return null;
    const flow = createDungeonEncounterFlowRunState(dungeonId, run.encounterFlow, Object.assign({}, settings, {
      startedAt: Number(run.startedAt || 0),
      completedAt: Number(run.completedAt || 0),
      bossDefeated: !!run.bossDefeated
    }));
    if (!flow) return null;
    const nowMs = Number.isFinite(Number(settings.nowMs)) ? Math.max(0, Number(settings.nowMs)) : Date.now();
    const defeated = new Set(flow.defeatedSlotIds);
    const completed = new Set(flow.completedBeatIds);
    const beats = definition.beats.map((beat, index) => {
      const slots = createDungeonEncounterBeatSlots(beat);
      const value = slots.filter((slot) => defeated.has(slot.id)).length;
      return {
        id: beat.id,
        kind: beat.kind,
        name: beat.name,
        summary: beat.summary,
        index,
        number: index + 1,
        sectionIds: beat.sectionIds.slice(),
        spawnGroupIds: beat.spawnGroupIds.slice(),
        enemyIds: beat.enemyIds.slice(),
        bossIds: beat.bossIds.slice(),
        gateX: beat.gateX,
        value,
        goal: slots.length,
        complete: completed.has(beat.id),
        active: index === flow.activeBeatIndex && flow.status !== 'complete'
      };
    });
    const activeBeat = beats[flow.activeBeatIndex] || null;
    const bossIntroRemainingMs = activeBeat && activeBeat.kind === 'boss'
      ? Math.max(0, Number(flow.bossArmedAt || 0) - nowMs)
      : 0;
    const bossIntroActive = bossIntroRemainingMs > 0 && flow.status !== 'complete';
    const status = flow.status === 'complete'
      ? 'complete'
      : activeBeat && activeBeat.kind === 'boss'
        ? bossIntroActive ? 'boss_intro' : 'boss_active'
        : 'active';
    const completedBeatCount = beats.filter((beat) => beat.complete).length;
    const activeGateX = flow.status === 'complete' || !activeBeat || activeBeat.kind === 'boss'
      ? 0
      : Math.max(0, Number(activeBeat.gateX || 0));
    const remaining = activeBeat ? Math.max(0, activeBeat.goal - activeBeat.value) : 0;
    const hudStatus = flow.status === 'complete'
      ? 'Expedition route secured'
      : bossIntroActive
        ? `Boss arming in ${Math.max(1, Math.ceil(bossIntroRemainingMs / 1000))}s`
        : activeBeat && activeBeat.kind === 'boss'
          ? `${remaining} boss ${remaining === 1 ? 'target' : 'targets'} remaining`
          : `${remaining} enemies remaining`;
    return {
      version: flow.version,
      id: flow.id,
      dungeonId: flow.dungeonId,
      mapId: flow.mapId,
      status,
      complete: flow.status === 'complete',
      activeBeatIndex: flow.activeBeatIndex,
      activeBeatId: flow.activeBeatId,
      beatCount: beats.length,
      preBossBeatCount: definition.preBossBeatCount,
      completedBeatCount,
      completedBeatIds: flow.completedBeatIds.slice(),
      defeatedSlotIds: flow.defeatedSlotIds.slice(),
      activeGateX,
      activeBeat,
      beats,
      bossReveal: {
        active: !!(activeBeat && activeBeat.kind === 'boss' && flow.status !== 'complete'),
        introActive: bossIntroActive,
        revealedAt: flow.bossRevealedAt,
        armedAt: flow.bossArmedAt,
        remainingMs: bossIntroRemainingMs
      },
      hud: {
        title: flow.status === 'complete' ? 'Expedition Route Complete' : `Route ${flow.activeBeatIndex + 1}/${beats.length}`,
        label: flow.status === 'complete' ? 'Expedition route secured' : activeBeat && activeBeat.name || 'Advance the route',
        summary: flow.status === 'complete' ? 'All encounter beats are clear.' : activeBeat && activeBeat.summary || '',
        status: hudStatus,
        value: flow.status === 'complete' ? beats.length : activeBeat && activeBeat.value || 0,
        goal: flow.status === 'complete' ? beats.length : Math.max(1, activeBeat && activeBeat.goal || 1),
        complete: flow.status === 'complete',
        kind: flow.status === 'complete' ? 'complete' : activeBeat && activeBeat.kind || 'combat',
        activeGateX
      }
    };
  }

  function createDungeonObjectiveRunState(dungeonId, value, options) {
    const data = getDungeonData(options);
    const source = value && typeof value === 'object' ? value : {};
    const configured = data.DUNGEON_OBJECTIVES || [];
    return configured.reduce((objectives, objective) => {
      const id = normalizeId(objective && objective.id);
      if (!id) return objectives;
      const entry = source[id] && typeof source[id] === 'object' ? source[id] : {};
      const goal = Math.max(1, Number(objective.goal || 1) || 1);
      objectives[id] = {
        progress: clamp(Number(entry.progress || 0), 0, goal),
        complete: !!entry.complete || Number(entry.progress || 0) >= goal,
        failed: !!entry.failed,
        claimed: !!entry.claimed
      };
      return objectives;
    }, {});
  }

  function ensureDungeonRunObjectives(run, options) {
    if (!run || !run.dungeonId) return {};
    run.objectives = createDungeonObjectiveRunState(run.dungeonId, run.objectives, options);
    return run.objectives;
  }

  function recordDungeonObjectiveRunProgress(run, type, amount, options) {
    if (!run || run.completedAt || !run.dungeonId) return false;
    const data = getDungeonData(options);
    const objectives = ensureDungeonRunObjectives(run, options);
    let changed = false;
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      if (!objective || objective.type !== type) return;
      const entry = objectives[objective.id];
      if (!entry || entry.failed || entry.complete) return;
      const goal = Math.max(1, Number(objective.goal || 1));
      entry.progress = clamp(Number(entry.progress || 0) + Math.max(0, Number(amount || 1)), 0, goal);
      entry.complete = entry.progress >= goal;
      changed = true;
    });
    return changed;
  }

  function failDungeonObjectiveRun(run, type, options) {
    if (!run || run.completedAt || !run.dungeonId) return false;
    const data = getDungeonData(options);
    const objectives = ensureDungeonRunObjectives(run, options);
    let changed = false;
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      if (!objective || objective.type !== type) return;
      const entry = objectives[objective.id];
      if (!entry || entry.complete) return;
      entry.failed = true;
      changed = true;
    });
    return changed;
  }

  function finalizeDungeonObjectiveRunState(run, options) {
    if (!run || !run.dungeonId) return;
    const settings = options || {};
    const data = getDungeonData(settings);
    const nowMs = Object.prototype.hasOwnProperty.call(settings, 'nowMs')
      ? Number(settings.nowMs || 0)
      : Date.now();
    const fallbackStartedAt = Object.prototype.hasOwnProperty.call(settings, 'fallbackStartedAt')
      ? Number(settings.fallbackStartedAt || 0)
      : Date.now();
    const elapsed = Math.max(0, (nowMs - Number(run.startedAt || fallbackStartedAt)) / 1000);
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      if (!objective || objective.type !== 'timedClear') return;
      const entry = ensureDungeonRunObjectives(run, settings)[objective.id];
      if (!entry) return;
      const goal = Math.max(1, Number(objective.goal || 1));
      entry.progress = Math.min(goal, elapsed <= goal ? goal : 0);
      entry.complete = elapsed <= goal;
      entry.failed = elapsed > goal;
    });
    const survival = (data.DUNGEON_OBJECTIVES || []).find((objective) => objective.type === 'partySurvival');
    if (survival) {
      const entry = ensureDungeonRunObjectives(run, settings)[survival.id];
      if (entry) {
        entry.progress = Number(run.partyDefeats || 0) > 0 ? 0 : 1;
        entry.complete = Number(run.partyDefeats || 0) <= 0;
        entry.failed = Number(run.partyDefeats || 0) > 0;
      }
    }
  }

  function awardDungeonObjectiveRunRewards(run, awardReward, options) {
    if (!run || !run.objectives || typeof awardReward !== 'function') return;
    const data = getDungeonData(options);
    (data.DUNGEON_OBJECTIVES || []).forEach((objective) => {
      const entry = run.objectives[objective.id];
      if (!entry || !entry.complete || entry.claimed || !objective.reward) return;
      awardReward(objective.reward);
      entry.claimed = true;
    });
  }

  function createDungeonState(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const completedDungeonIds = Array.isArray(source.completedDungeonIds)
      ? source.completedDungeonIds.map(normalizeId).filter(Boolean)
      : [];
    const completionCounts = {};
    Object.entries(source.completionCounts || {}).forEach(([id, count]) => {
      const key = normalizeId(id);
      if (key) completionCounts[key] = Math.max(0, Math.floor(Number(count) || 0));
    });
    const bossRespawnAt = {};
    Object.entries(source.bossRespawnAt || {}).forEach(([id, value]) => {
      const key = normalizeId(id);
      const timestamp = Number(value || 0);
      if (key && timestamp > 0) bossRespawnAt[key] = timestamp;
    });
    const currentRun = source.currentRun && typeof source.currentRun === 'object'
      ? {
          dungeonId: normalizeId(source.currentRun.dungeonId),
          startedAt: Number(source.currentRun.startedAt || 0),
          completedAt: Number(source.currentRun.completedAt || 0),
          bossDefeated: !!source.currentRun.bossDefeated,
          bossEncounterId: normalizeId(source.currentRun.bossEncounterId),
          adminEncounter: !!source.currentRun.adminEncounter,
          objectives: createDungeonObjectiveRunState(source.currentRun.dungeonId, source.currentRun.objectives, options),
          partyDefeats: Math.max(0, Math.floor(Number(source.currentRun.partyDefeats || 0) || 0))
        }
      : null;
    if (currentRun && currentRun.dungeonId) {
      const encounterFlow = createDungeonEncounterFlowRunState(
        currentRun.dungeonId,
        source.currentRun.encounterFlow,
        Object.assign({}, options || {}, {
          startedAt: currentRun.startedAt,
          completedAt: currentRun.completedAt,
          bossDefeated: currentRun.bossDefeated
        })
      );
      if (encounterFlow) currentRun.encounterFlow = encounterFlow;
    }
    return {
      activeDungeonId: normalizeId(source.activeDungeonId),
      currentRun: currentRun && currentRun.dungeonId ? currentRun : null,
      completedDungeonIds: Array.from(new Set(completedDungeonIds)),
      completionCounts,
      bossRespawnAt,
      lastCompletedAt: Number(source.lastCompletedAt || 0)
    };
  }

  function createDungeonStartBlockReason(dungeon, player) {
    const activePlayer = player || {};
    if (!dungeon) return 'Dungeon is unavailable.';
    if (!activePlayer.classId) return 'Choose a class first.';
    if (activePlayer.level < Number(dungeon.levelRequirement || 1)) return `Level ${dungeon.levelRequirement} required.`;
    if (dungeon.requiresAdvancedClass && !activePlayer.advancedClassId) return 'Choose an advanced class first.';
    return '';
  }

  function createDungeonBossRespawnInfo(dungeonId, dungeons, options) {
    const settings = options || {};
    const id = normalizeId(dungeonId);
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const respawnAt = id ? Number(state.bossRespawnAt && state.bossRespawnAt[id] || 0) : 0;
    const remainingMs = Math.max(0, respawnAt - Number(settings.nowMs || 0));
    const bossRespawnSeconds = Math.max(0, Number(settings.bossRespawnSeconds || DEFAULT_BOSS_RESPAWN_SECONDS));
    return {
      respawnAt,
      remainingMs,
      remaining: Math.min(bossRespawnSeconds, remainingMs / 1000),
      respawning: remainingMs > 0
    };
  }

  function isDungeonBossRespawning(dungeonId, dungeons, options) {
    return createDungeonBossRespawnInfo(dungeonId, dungeons, options).respawning;
  }

  function refreshDungeonBossRespawnState(dungeonId, dungeons, options) {
    const settings = options || {};
    const id = normalizeId(dungeonId);
    if (!id) return false;
    if (getDungeonEncounterFlowDefinition(id, settings)) return false;
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const respawnAt = Number(state.bossRespawnAt && state.bossRespawnAt[id] || 0);
    if (!respawnAt || Number(settings.nowMs || 0) < respawnAt) return false;
    delete state.bossRespawnAt[id];
    state.activeDungeonId = id;
    state.currentRun = createDungeonStartRunState(id, false, settings);
    return true;
  }

  function createDungeonStartRunState(dungeonId, bossRespawning, options) {
    const settings = options || {};
    const id = normalizeId(dungeonId);
    const encounterDefinition = getDungeonEncounterFlowDefinition(id, settings);
    const respawning = !!bossRespawning && !encounterDefinition;
    const run = {
      dungeonId: id,
      startedAt: Number(settings.startedAt || 0),
      completedAt: respawning ? Number(settings.completedAt || 0) : 0,
      bossDefeated: respawning,
      objectives: createDungeonObjectiveRunState(id, null, settings),
      partyDefeats: 0
    };
    const encounterFlow = createDungeonEncounterFlowRunState(id, null, Object.assign({}, settings, {
      completedAt: respawning ? Number(settings.completedAt || 0) : 0,
      bossDefeated: respawning
    }));
    if (encounterFlow) run.encounterFlow = encounterFlow;
    return run;
  }

  function createMapChangeDungeonRunState(dungeonId, bossRespawning, options) {
    return createDungeonStartRunState(dungeonId, bossRespawning, options);
  }

  function startDungeonState(dungeonId, dungeons, bossRespawning, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createDungeonStartRunState(dungeonId, bossRespawning, settings);
    return state.currentRun;
  }

  function transitionDungeonMapState(dungeonId, dungeons, run, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createDungeonMapTransitionRunState(dungeonId, run, settings);
    return state.currentRun;
  }

  function changeMapDungeonState(dungeonId, dungeons, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    if (dungeonId) {
      state.activeDungeonId = dungeonId;
      if (!state.currentRun || state.currentRun.dungeonId !== state.activeDungeonId) {
        state.currentRun = createMapChangeDungeonRunState(state.activeDungeonId, !!settings.bossRespawning, settings);
      }
    } else {
      state.activeDungeonId = '';
      state.currentRun = null;
    }
    return state.currentRun;
  }

  function createDungeonCompletionRunState(dungeonId, currentRun, options) {
    const settings = options || {};
    const source = currentRun && typeof currentRun === 'object' ? currentRun : {};
    const existingRun = settings.existingRun && typeof settings.existingRun === 'object' ? settings.existingRun : null;
    const id = normalizeId(dungeonId);
    const completedAt = Number(settings.completedAt || 0);
    return Object.assign({}, source, {
      dungeonId: id,
      startedAt: Number(source.startedAt || completedAt),
      completedAt,
      bossDefeated: true,
      objectives: existingRun ? existingRun.objectives : createDungeonObjectiveRunState(id, null, settings),
      partyDefeats: existingRun ? Number(existingRun.partyDefeats || 0) : 0
    });
  }

  function completeDungeonState(dungeonId, dungeons, options) {
    const settings = options || {};
    const id = dungeonId;
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    if (!Array.isArray(state.completedDungeonIds)) state.completedDungeonIds = [];
    if (!state.completionCounts || typeof state.completionCounts !== 'object') state.completionCounts = {};
    if (!state.completedDungeonIds.includes(id)) state.completedDungeonIds.push(id);
    state.completionCounts[id] = Math.max(0, Number(state.completionCounts[id] || 0)) + 1;
    const completedAt = Number(settings.completedAt || 0);
    state.lastCompletedAt = completedAt;
    state.activeDungeonId = id;
    state.bossRespawnAt = state.bossRespawnAt && typeof state.bossRespawnAt === 'object' ? state.bossRespawnAt : {};
    const bossRespawnSeconds = Math.max(0, Number(settings.bossRespawnSeconds || DEFAULT_BOSS_RESPAWN_SECONDS));
    state.bossRespawnAt[id] = completedAt + bossRespawnSeconds * 1000;
    state.currentRun = createDungeonCompletionRunState(id, state.currentRun, settings);
    return state;
  }

  function createBossEncounterCompletionRunState(dungeonId, currentRun, options) {
    const settings = options || {};
    const source = currentRun && typeof currentRun === 'object' ? currentRun : {};
    const completedAt = Number(settings.completedAt || 0);
    return Object.assign({}, source, {
      dungeonId,
      startedAt: Number(source.startedAt || completedAt),
      completedAt,
      bossDefeated: true,
      bossEncounterId: settings.bossEncounterId,
      adminEncounter: true
    });
  }

  function completeBossEncounterState(dungeonId, dungeons, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createBossEncounterCompletionRunState(state.activeDungeonId, state.currentRun, settings);
    return state;
  }

  function createBossEncounterStartRunState(dungeonId, bossEncounterId, adminEncounter, options) {
    const settings = options || {};
    return {
      dungeonId,
      startedAt: Number(settings.startedAt || 0),
      completedAt: 0,
      bossDefeated: false,
      bossEncounterId,
      adminEncounter: !!adminEncounter
    };
  }

  function startBossEncounterState(dungeonId, dungeons, bossEncounterId, adminEncounter, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    state.activeDungeonId = dungeonId;
    state.currentRun = createBossEncounterStartRunState(state.activeDungeonId, bossEncounterId, adminEncounter, settings);
    return state.currentRun;
  }

  function createDungeonMapTransitionRunState(dungeonId, run, options) {
    const settings = options || {};
    const source = run && typeof run === 'object' ? run : {};
    const id = normalizeId(dungeonId);
    const transitioned = Object.assign({}, source, {
      objectives: createDungeonObjectiveRunState(id, source.objectives, settings),
      partyDefeats: Number(source.partyDefeats || 0)
    });
    const encounterFlow = createDungeonEncounterFlowRunState(id, source.encounterFlow, Object.assign({}, settings, {
      startedAt: Number(source.startedAt || 0),
      completedAt: Number(source.completedAt || 0),
      bossDefeated: !!source.bossDefeated
    }));
    if (encounterFlow) transitioned.encounterFlow = encounterFlow;
    else delete transitioned.encounterFlow;
    return transitioned;
  }

  function createDungeonObjectiveSnapshots(objectives, options) {
    const data = getDungeonData(options);
    return Object.entries(objectives || {}).map(([id, entry]) => {
      const objective = getById(data.DUNGEON_OBJECTIVES || [], id) || {};
      const goal = Math.max(1, Number(objective.goal || 1));
      return Object.assign({}, objective, entry, {
        id,
        goal,
        progress: clamp(Number(entry && entry.progress || 0), 0, goal)
      });
    });
  }

  function createDungeonSummary(dungeon, dungeons, objectives, options) {
    if (!dungeon) return null;
    const data = getDungeonData(options);
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const map = getById(data.MAPS || [], dungeon.mapId);
    const boss = getById(data.ENEMIES || [], dungeon.bossId);
    const count = Math.max(0, Number(state.completionCounts && state.completionCounts[dungeon.id] || 0));
    const respawn = createDungeonBossRespawnInfo(dungeon.id, state, settings);
    const run = state.currentRun && state.currentRun.dungeonId === dungeon.id ? state.currentRun : null;
    const encounterFlow = run ? createDungeonEncounterFlowSnapshot(dungeon.id, run, settings) : null;
    return {
      id: dungeon.id,
      name: dungeon.name,
      summary: dungeon.summary,
      mapId: dungeon.mapId,
      mapName: map ? map.name : dungeon.mapId,
      bossId: dungeon.bossId,
      bossName: boss ? boss.name : dungeon.bossId,
      levelRequirement: dungeon.levelRequirement,
      recommendedPartySize: dungeon.recommendedPartySize,
      requiresAdvancedClass: !!dungeon.requiresAdvancedClass,
      active: state.activeDungeonId === dungeon.id,
      complete: (state.completedDungeonIds || []).includes(dungeon.id),
      completionCount: count,
      bossRespawnAt: respawn.respawnAt,
      bossRespawnRemaining: respawn.remaining,
      bossRespawning: respawn.respawning,
      lockedReason: createDungeonStartBlockReason(dungeon, settings.player),
      encounterFlow,
      objectives: run ? createDungeonObjectiveSnapshots(objectives || run.objectives, settings) : []
    };
  }

  function createDungeonSummaryFromState(dungeon, dungeons, options) {
    if (!dungeon) return null;
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const run = state.currentRun && state.currentRun.dungeonId === dungeon.id ? state.currentRun : null;
    const objectives = run ? ensureDungeonRunObjectives(run, settings) : null;
    return createDungeonSummary(dungeon, state, objectives, settings);
  }

  function createDungeonRunSnapshot(run) {
    if (!run || typeof run !== 'object') return null;
    const snapshot = Object.assign({}, run, {
      objectives: Object.entries(run.objectives || {}).reduce((entries, pair) => {
        const id = pair[0];
        const entry = pair[1] && typeof pair[1] === 'object' ? pair[1] : {};
        entries[id] = Object.assign({}, entry);
        return entries;
      }, {})
    });
    const flow = run.encounterFlow && typeof run.encounterFlow === 'object' ? run.encounterFlow : null;
    if (flow) {
      snapshot.encounterFlow = Object.assign({}, flow, {
        completedBeatIds: Array.isArray(flow.completedBeatIds) ? flow.completedBeatIds.slice() : [],
        spawnedBeatIds: Array.isArray(flow.spawnedBeatIds) ? flow.spawnedBeatIds.slice() : [],
        defeatedSlotIds: Array.isArray(flow.defeatedSlotIds) ? flow.defeatedSlotIds.slice() : []
      });
    }
    return snapshot;
  }

  function createDungeonSnapshot(dungeons, activeDungeonSummary, dungeonSummaries) {
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState();
    return {
      activeDungeon: activeDungeonSummary || null,
      encounterFlow: activeDungeonSummary && activeDungeonSummary.encounterFlow || null,
      dungeons: Array.isArray(dungeonSummaries) ? dungeonSummaries : [],
      completedDungeonIds: (state.completedDungeonIds || []).slice(),
      completionCounts: Object.assign({}, state.completionCounts),
      bossRespawnAt: Object.assign({}, state.bossRespawnAt || {}),
      currentRun: createDungeonRunSnapshot(state.currentRun)
    };
  }

  function getDungeonSnapshotSummaryOptions(options) {
    const settings = options || {};
    const nowMs = typeof settings.nowMsProvider === 'function'
      ? settings.nowMsProvider()
      : Number(settings.nowMs || 0);
    return Object.assign({}, settings, { nowMs });
  }

  function createDungeonSnapshotFromState(dungeons, options) {
    const settings = options || {};
    const data = getDungeonData(settings);
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const activeDungeon = getActiveDungeonDefinition(state, settings);
    const activeSummary = activeDungeon
      ? createDungeonSummaryFromState(activeDungeon, state, getDungeonSnapshotSummaryOptions(settings))
      : null;
    const summaries = (data.DUNGEONS || []).map((dungeon) =>
      createDungeonSummaryFromState(dungeon, state, getDungeonSnapshotSummaryOptions(settings))
    );
    return createDungeonSnapshot(state, activeSummary, summaries);
  }

  function createDungeonTrackerSnapshot(dungeons, activeDungeonSummary) {
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState();
    return {
      activeDungeon: activeDungeonSummary || null,
      encounterFlow: activeDungeonSummary && activeDungeonSummary.encounterFlow || null,
      dungeons: [],
      completedDungeonIds: [],
      completionCounts: {},
      bossRespawnAt: {},
      currentRun: createDungeonRunSnapshot(state.currentRun)
    };
  }

  function createDungeonTrackerSnapshotFromState(dungeons, options) {
    const settings = options || {};
    const state = dungeons && typeof dungeons === 'object' ? dungeons : createDungeonState(null, settings);
    const activeDungeon = getDungeonDefinitionById(state.activeDungeonId, settings);
    const activeSummary = activeDungeon
      ? createDungeonSummaryFromState(activeDungeon, state, getDungeonSnapshotSummaryOptions(settings))
      : null;
    return createDungeonTrackerSnapshot(state, activeSummary);
  }

  const api = {
    DEFAULT_BOSS_RESPAWN_SECONDS,
    DUNGEON_ENCOUNTER_FLOW_VERSION,
    DEFAULT_DUNGEON_BOSS_INTRO_DELAY_MS,
    getDungeonObjectiveDefinitions,
    getActiveDungeonDefinition,
    getDungeonDefinitionById,
    getDungeonBossIds,
    getDungeonDefinitionByMapId,
    normalizeDungeonEncounterIdList,
    normalizeDungeonEncounterBeat,
    getDungeonEncounterFlowDefinition,
    getDungeonEncounterBeatEnemyIds,
    createDungeonEncounterBeatSlots,
    createDungeonEncounterFlowRunState,
    ensureDungeonEncounterFlowRunState,
    markDungeonEncounterBeatSpawned,
    recordDungeonEncounterEnemyDefeat,
    isDungeonEncounterFlowComplete,
    getDungeonEncounterCompletionBlockReason,
    createDungeonEncounterFlowSnapshot,
    createDungeonObjectiveRunState,
    ensureDungeonRunObjectives,
    recordDungeonObjectiveRunProgress,
    failDungeonObjectiveRun,
    finalizeDungeonObjectiveRunState,
    awardDungeonObjectiveRunRewards,
    createDungeonState,
    createDungeonStartBlockReason,
    createDungeonBossRespawnInfo,
    isDungeonBossRespawning,
    refreshDungeonBossRespawnState,
    createDungeonStartRunState,
    createMapChangeDungeonRunState,
    startDungeonState,
    transitionDungeonMapState,
    changeMapDungeonState,
    createDungeonCompletionRunState,
    completeDungeonState,
    createBossEncounterCompletionRunState,
    completeBossEncounterState,
    createBossEncounterStartRunState,
    startBossEncounterState,
    createDungeonMapTransitionRunState,
    createDungeonObjectiveSnapshots,
    createDungeonSummary,
    createDungeonSummaryFromState,
    createDungeonRunSnapshot,
    createDungeonSnapshot,
    createDungeonSnapshotFromState,
    createDungeonTrackerSnapshot,
    createDungeonTrackerSnapshotFromState
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.dungeons = Object.assign({}, modules.dungeons || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
