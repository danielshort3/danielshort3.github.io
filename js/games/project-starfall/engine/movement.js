(function initProjectStarfallEngineMovement(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreGeometry = (typeof require === 'function' ? require('../core/geometry.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};

  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const rectsOverlap = CoreMath.rectsOverlap || function rectsOverlapFallback(a, b) {
    return !!(a && b &&
      a.x < b.x + b.w &&
      a.x + a.w > b.x &&
      a.y < b.y + b.h &&
      a.y + a.h > b.y);
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const platformContainsX = CoreGeometry.platformContainsX || function platformContainsXFallback(platform, x, padding) {
    const pad = Number(padding) || 0;
    return !!platform && x >= platform.x - pad && x <= platform.x + platform.w + pad;
  };
  const getPlatformSurfaceY = CoreGeometry.getPlatformSurfaceY || function getPlatformSurfaceYFallback(platform, x) {
    if (!platform) return 0;
    const y = Number(platform.y || 0);
    if (!(platform.shape === 'slope' && Number.isFinite(Number(platform.y2)))) return y;
    const width = Math.max(1, Number(platform.w || 0));
    const ratio = clamp((Number(x || 0) - Number(platform.x || 0)) / width, 0, 1);
    return y + (Number(platform.y2 || y) - y) * ratio;
  };

  function isLevelTeleportTarget(player, platform, x) {
    if (!player || !platform) return false;
    const minX = platform.x + 10;
    const maxX = platform.x + platform.w - player.w - 10;
    return minX <= maxX && x >= minX && x <= maxX;
  }

  function findVerticalTeleportTarget(player, platforms, verticalDirection, distance, worldWidth) {
    if (!player || !Array.isArray(platforms)) return null;
    const centerX = player.x + player.w / 2;
    const footY = player.y + player.h;
    const xPadding = Math.max(70, player.w * 1.5);
    const maxDistance = Math.max(0, Number(distance) || 0) + (verticalDirection < 0 ? 48 : 64);
    const candidates = platforms
      .filter((platform) => platform && platformContainsX(platform, centerX, xPadding))
      .map((platform) => ({ platform, surfaceY: getPlatformSurfaceY(platform, centerX) }))
      .filter((entry) => verticalDirection < 0 ? entry.surfaceY < footY - 24 : entry.surfaceY > footY + 24)
      .map((entry) => ({ platform: entry.platform, gap: verticalDirection < 0 ? footY - entry.surfaceY : entry.surfaceY - footY }))
      .filter((entry) => entry.gap <= maxDistance)
      .sort((a, b) =>
        a.gap - b.gap ||
        Math.abs((a.platform.x + a.platform.w / 2) - centerX) - Math.abs((b.platform.x + b.platform.w / 2) - centerX) ||
        Number(a.platform.index || 0) - Number(b.platform.index || 0));
    const match = candidates[0];
    if (!match) return null;
    const platform = match.platform;
    const minX = platform.x + 10;
    const maxX = platform.x + platform.w - player.w - 10;
    const targetX = minX <= maxX ? clamp(player.x, minX, maxX) : platform.x + platform.w / 2 - player.w / 2;
    return {
      platform,
      x: clamp(targetX, 10, worldWidth - player.w - 10),
      y: Math.max(0, getPlatformSurfaceY(platform, targetX + player.w / 2) - player.h)
    };
  }

  function findHorizontalTeleportTarget(player, platform, direction, distance, worldWidth) {
    if (!player || !platform) return null;
    const stepDirection = Math.sign(direction || player.facing || 1) || 1;
    const minX = platform.x + 10;
    const maxX = platform.x + platform.w - player.w - 10;
    if (minX > maxX) return null;
    const startX = clamp(player.x, minX, maxX);
    const desiredX = clamp(startX + stepDirection * Math.max(0, Number(distance) || 0), minX, maxX);
    const step = Math.max(6, Math.min(16, player.w / 2));
    for (let x = desiredX; stepDirection > 0 ? x >= startX : x <= startX; x -= stepDirection * step) {
      if (isLevelTeleportTarget(player, platform, x)) {
        return {
          platform,
          x: clamp(x, 10, worldWidth - player.w - 10),
          y: Math.max(0, getPlatformSurfaceY(platform, x + player.w / 2) - player.h)
        };
      }
    }
    return {
      platform,
      x: startX,
      y: Math.max(0, getPlatformSurfaceY(platform, startX + player.w / 2) - player.h)
    };
  }

  function getMovementSkillPlan(skill, rank, player, input, options) {
    const settings = options || {};
    const controls = input || {};
    const effect = skill && skill.movementEffect || {};
    const isTeleport = effect.mode === 'blink' || String(skill && skill.id || '').includes('blink');
    if (!isTeleport) return null;
    const verticalDirection = controls.up ? -1 : controls.down ? 1 : 0;
    const distance = Math.max(0, Number(effect.distance || 190) + Number(rank || 0) * Number(effect.distancePerRank || 0));
    if (!verticalDirection) {
      const horizontalDirection = controls.left && !controls.right ? -1 : controls.right && !controls.left ? 1 : player.facing || 1;
      const target = findHorizontalTeleportTarget(player, settings.bodyPlatform, horizontalDirection, distance, settings.worldWidth);
      if (!target) return { type: 'blink-horizontal', blockReason: 'No level ground in teleport range.', silent: true };
      return { type: 'blink-horizontal', target };
    }
    const target = findVerticalTeleportTarget(player, settings.platforms, verticalDirection, distance, settings.worldWidth);
    if (!target) return { type: 'blink-vertical', blockReason: 'No platform in teleport range.', silent: true };
    return { type: 'blink-vertical', target };
  }

  function getRuntimePlatformPlacementStatePlan(player, platform, x) {
    const placedX = clamp(Number(x || platform.x + 160), platform.x + 16, platform.x + platform.w - player.w - 16);
    const placedY = getPlatformSurfaceY(platform, placedX + player.w / 2) - player.h;
    return {
      x: placedX,
      y: placedY,
      previousY: placedY,
      grounded: true,
      groundedPlatformId: platform.id || '',
      groundedPlatformIndex: platform.index || 0,
      vx: 0,
      vy: 0,
      climbing: false,
      climbMoving: false,
      climbableId: ''
    };
  }

  function getTrialInstanceStartPlacementStatePlan(player, platform) {
    const placedX = 150;
    const placedY = platform ? getPlatformSurfaceY(platform, placedX + player.w / 2) - player.h : 360;
    return {
      x: placedX,
      y: placedY,
      previousY: placedY,
      grounded: !!platform,
      groundedPlatformId: platform ? platform.id : '',
      groundedPlatformIndex: platform ? platform.index : 0,
      facing: 1
    };
  }

  function getHometownReturnPlacementStatePlan(player, platform, instance, worldWidth) {
    const placedX = clamp(Number(instance && instance.returnX || 1845) - player.w / 2, 16, worldWidth - player.w - 16);
    const placedY = platform ? getPlatformSurfaceY(platform, placedX + player.w / 2) - player.h : Number(instance && instance.returnY || 360);
    return {
      x: placedX,
      y: placedY,
      previousY: placedY,
      grounded: !!platform,
      groundedPlatformId: platform ? platform.id : '',
      groundedPlatformIndex: platform ? platform.index : 0,
      facing: -1
    };
  }

  function getMapEntryPlacementStatePlan(player, entryPortal, platform, worldWidth) {
    const placedX = clamp(entryPortal.x + entryPortal.w / 2 - player.w / 2, 16, worldWidth - player.w - 16);
    const placedY = platform ? getPlatformSurfaceY(platform, placedX + player.w / 2) - player.h : entryPortal.y + entryPortal.h - player.h;
    return {
      x: placedX,
      y: placedY,
      previousY: placedY,
      grounded: !!platform,
      groundedPlatformId: platform ? platform.id : '',
      groundedPlatformIndex: platform ? platform.index : 0
    };
  }

  function getChannelPositionLandingSearchPlan(player, previousPlayer, worldWidth) {
    const previous = previousPlayer || player || {};
    return {
      centerX: clamp(Number(previous.x || player.x || 140) + Number(player.w || 0) / 2, 32, Math.max(32, worldWidth - 32)),
      floorY: Number(previous.y || player.y || 250) + Number(player.h || 0) + 18
    };
  }

  function getChannelPositionPlacementStatePlan(player, previousPlayer, landingSurface, platform, worldWidth, worldHeight, landingSearch) {
    const previous = previousPlayer || player || {};
    const search = landingSearch || getChannelPositionLandingSearchPlan(player, previousPlayer, worldWidth);
    const placedX = clamp((landingSurface ? landingSurface.x : search.centerX) - player.w / 2, 16, worldWidth - player.w - 16);
    const placedY = platform ? getPlatformSurfaceY(platform, placedX + player.w / 2) - player.h : clamp(Number(previous.y || 250), 80, worldHeight - player.h - 20);
    return {
      x: placedX,
      y: placedY,
      previousY: placedY,
      grounded: !!platform,
      groundedPlatformId: platform ? platform.id : '',
      groundedPlatformIndex: platform ? platform.index : 0,
      facing: Number(previous.facing || player.facing || 1) >= 0 ? 1 : -1
    };
  }

  function getInstanceTravelResetStatePlan() {
    return {
      player: {
        vx: 0,
        vy: 0,
        activeStation: '',
        activePortalId: '',
        activeQuestNpcId: '',
        activeShopVendorId: '',
        climbing: false,
        climbMoving: false,
        climbableId: '',
        climbLockUntil: 0,
        mobility: null
      },
      input: {
        attack: false,
        loot: false
      },
      nextHeldLootAt: 0
    };
  }

  function getMapChangeTravelStatePlan(map) {
    const safeZone = !!(map && map.safeZone);
    return {
      player: {
        x: 140,
        y: safeZone ? 360 : 250,
        vx: 0,
        vy: 0,
        activeStation: '',
        activePortalId: '',
        activeQuestNpcId: '',
        activeShopVendorId: '',
        climbing: false,
        climbMoving: false,
        climbableId: '',
        climbLockUntil: 0
      },
      fallbackPlatformX: safeZone ? 220 : 180,
      input: {
        attack: false,
        loot: false
      },
      nextHeldLootAt: 0
    };
  }

  function getSkillMovementApplicationPlan(skill, rank, player) {
    const effect = skill.movementEffect || {};
    const facing = player.facing || 1;
    const direction = effect.direction === 'backward' ? -facing : facing;
    const startedAirborne = !player.grounded;
    const preservesAirMomentum = startedAirborne && (effect.mode === 'dash' || effect.mode === 'roll' || effect.mode === 'leap' || String(skill.id || '').match(/dash|roll|leap/));
    const distance = Math.max(0, Number(effect.distance || 190) + Number(rank || 0) * Number(effect.distancePerRank || 0));
    const duration = Math.max(0, Number(effect.duration || 0.2));
    const startX = player.x + player.w / 2;
    const startY = player.y + player.h / 2;
    const hitOffset = Number(effect.hitOffset || Math.max(78, distance * 0.48));
    return {
      effect,
      direction,
      preservesAirMomentum,
      distance,
      duration,
      startX,
      startY,
      hitOffset
    };
  }

  function getSkillMovementInvulnerabilityPlan(player, movementApplication, currentTime, mobilityWindowScale) {
    const effect = movementApplication.effect;
    if (!effect.invulnerable) return null;
    return {
      invulnerableUntil: Math.max(player.invulnerableUntil || 0, currentTime + Number(effect.invulnerable) * mobilityWindowScale)
    };
  }

  function getSkillMovementBlinkStartEffectPlan(movementApplication) {
    return {
      type: 'burst',
      x: movementApplication.startX,
      y: movementApplication.startY,
      r: 44,
      ttl: 0.28
    };
  }

  function getSkillMovementBlinkEndEffectPlan(player) {
    return {
      type: 'burst',
      x: player.x + player.w / 2,
      y: player.y + player.h / 2,
      r: 58,
      ttl: 0.32
    };
  }

  function isSkillMovementInstant(movementApplication) {
    const effect = movementApplication.effect;
    return effect.mode === 'blink' || movementApplication.duration <= 0.01;
  }

  function getSkillMovementInstantVelocityStatePlan() {
    return {
      vx: 0
    };
  }

  function getSkillMovementBlinkFallbackStatePlan(player, movementApplication, worldWidth) {
    const worldClamp = getWorldXClampPlan({
      x: player.x + movementApplication.direction * movementApplication.distance,
      w: player.w
    }, worldWidth);
    return {
      x: worldClamp.x
    };
  }

  function isSkillMovementBlinkTargetPlan(movementPlan) {
    return !!(movementPlan && (movementPlan.type === 'blink-vertical' || movementPlan.type === 'blink-horizontal') && movementPlan.target);
  }

  function getSkillMovementMobilityStatePlan(skill, player, movementApplication, currentTime) {
    const effect = movementApplication.effect;
    const speed = movementApplication.distance / movementApplication.duration;
    const state = {
      mobility: {
        direction: movementApplication.direction,
        remaining: movementApplication.distance,
        speed,
        endsAt: currentTime + movementApplication.duration,
        preserveMomentumUntilGround: !!movementApplication.preservesAirMomentum
      },
      vx: movementApplication.direction * speed,
      preserveAirMomentum: !!movementApplication.preservesAirMomentum
    };
    if (movementApplication.preservesAirMomentum) {
      state.airMobilitySkillId = skill.id;
      state.airDashMomentumUntilGround = false;
    }
    if (Number.isFinite(Number(effect.verticalVelocity))) {
      state.vy = Math.min(player.vy, Number(effect.verticalVelocity));
      state.grounded = false;
      state.hasVerticalVelocity = true;
    }
    return state;
  }

  function getSkillMovementBlinkTargetStatePlan(movementPlan) {
    const target = movementPlan.target;
    const y = target.y;
    return {
      x: target.x,
      y,
      previousY: y,
      vy: 0,
      grounded: true,
      groundedPlatformId: target.platform.id,
      groundedPlatformIndex: target.platform.index,
      climbing: false,
      climbMoving: false,
      climbableId: '',
      mobility: null,
      airMobilitySkillId: '',
      airDashMomentumUntilGround: false
    };
  }

  function getSkillMovementActionEffectPlan(skill, movementApplication) {
    const effect = movementApplication.effect;
    const skillId = String(skill.id || '');
    return {
      type: skillId.includes('roll') ? 'arrowRelease' : skillId.includes('blink') ? 'cast' : 'slash',
      options: {
        forward: Math.max(44, Math.min(150, movementApplication.hitOffset)),
        yOffset: skillId.includes('blink') ? 28 : 34,
        r: Number(effect.damageRadius || 74),
        ttl: 0.28,
        duration: 0.28
      }
    };
  }

  function getSkillMovementFlameTrailEffectPlan(skill, rank, stats, player, movementApplication) {
    const effect = movementApplication.effect;
    if (effect.trail !== 'flame') return null;
    return {
      type: 'field',
      x: movementApplication.startX + movementApplication.direction * Math.max(48, movementApplication.distance * 0.45),
      y: player.y + player.h - 8,
      r: Math.max(90, Number(effect.damageRadius || 100)),
      ttl: 3.2,
      duration: 3.2,
      color: '#ff7b3a',
      damage: stats.power * (0.55 + rank * 0.035),
      slow: false,
      skillId: skill.id
    };
  }

  function getMapMovementProfile(map) {
    if (!map || map.safeZone) return null;
    if (map.movementProfile === 'ice') {
      return {
        id: 'ice',
        groundAccelerationScale: 0.68,
        activeFriction: 0.982,
        idleFriction: 0.94,
        maxSpeedScale: 1.08
      };
    }
    return null;
  }

  function getMovementInputIntent(input) {
    const rawDirection = (input.right ? 1 : 0) - (input.left ? 1 : 0);
    return {
      dropJumpReleased: !input.jump,
      rawDirection,
      verticalIntent: (input.down ? 1 : 0) - (input.up ? 1 : 0)
    };
  }

  function getActiveMobilityState(mobility) {
    if (!mobility || mobility.remaining <= 0) return null;
    return mobility;
  }

  function getMobilityStepPlan(mobility, delta, grounded) {
    const step = Math.min(mobility.remaining, Math.max(0, mobility.speed * delta));
    const remaining = Math.max(0, mobility.remaining - step);
    const complete = remaining <= 0;
    return {
      step,
      remaining,
      complete,
      preserveMomentum: complete && mobility.preserveMomentumUntilGround && !grounded
    };
  }

  function getMobilityMovementPlan(mobility, delta, grounded) {
    const mobilityStep = getMobilityStepPlan(mobility, delta, grounded);
    const activeVx = mobility.direction * mobility.speed;
    return {
      xDelta: mobility.direction * mobilityStep.step,
      vx: mobilityStep.complete && !mobilityStep.preserveMomentum ? 0 : activeVx,
      remaining: mobilityStep.remaining,
      complete: mobilityStep.complete,
      preserveMomentum: mobilityStep.preserveMomentum
    };
  }

  function getAirDashMomentumMovementPlan(player, delta) {
    return {
      xDelta: player.vx * delta
    };
  }

  function getHorizontalMovementStepPlan(player, stats, direction, delta, movementLocked, movementProfile, options) {
    const settings = options || {};
    let vx = player.vx;
    let facing;
    if (movementLocked && player.grounded) vx = 0;
    if (direction) {
      const groundAcceleration = settings.groundAcceleration * (movementProfile && player.grounded ? movementProfile.groundAccelerationScale : 1);
      const acceleration = player.grounded ? groundAcceleration : settings.airAcceleration;
      vx += direction * stats.speed * acceleration * delta;
      facing = direction > 0 ? 1 : -1;
    }
    const friction = player.grounded
      ? (movementProfile
          ? (direction ? movementProfile.activeFriction : movementProfile.idleFriction)
          : (direction ? settings.groundFrictionActive : settings.groundFrictionIdle))
      : settings.airFriction;
    vx *= friction;
    if (!direction && Math.abs(vx) < 2) vx = 0;
    const maxSpeed = stats.speed * (movementProfile ? movementProfile.maxSpeedScale : 1);
    vx = clamp(vx, -maxSpeed, maxSpeed);
    return {
      vx,
      facing,
      xDelta: vx * delta
    };
  }

  function getJumpMovementAction(player, input, stats, movementLocked, mobility, options) {
    const controls = input || {};
    const settings = options || {};
    if (!movementLocked && !mobility && controls.jump && controls.down && player.grounded && !player.dropJumpConsumed) {
      return {
        type: 'drop-through',
        vy: Math.max(player.vy, settings.dropThroughVy)
      };
    }
    if (!movementLocked && !mobility && controls.jump && !controls.down && player.grounded) {
      return {
        type: 'jump',
        vy: -stats.jump
      };
    }
    return null;
  }

  function getVerticalMovementStepPlan(player, delta, options) {
    const settings = options || {};
    const vy = player.vy + settings.gravity * delta;
    const previousY = player.y;
    return {
      vy,
      previousY,
      y: player.y + vy * delta,
      grounded: false
    };
  }

  function getClimbActivationPlan(player, movementLocked, mobility, climbable, verticalIntent, canMount, currentTime) {
    const canUseClimbMovement = !movementLocked && !mobility;
    const wantsClimb = !!(canUseClimbMovement &&
      climbable &&
      verticalIntent !== 0 &&
      canMount &&
      currentTime >= Number(player.climbLockUntil || 0));
    return {
      wantsClimb,
      shouldUpdateClimbing: !!(canUseClimbMovement && (player.climbing || wantsClimb))
    };
  }

  function getClimbClearStatePlan() {
    return {
      climbing: false,
      climbMoving: false,
      climbableId: ''
    };
  }

  function getClimbJumpExitPlan(input, stats, options) {
    if (!input.jump) return null;
    const settings = options || {};
    return {
      climbing: false,
      climbMoving: false,
      climbableId: '',
      climbLockDuration: settings.climbLockDuration,
      vy: -stats.jump,
      grounded: false
    };
  }

  function getClimbMountStatePlan(player, climbable, wasClimbing) {
    return {
      climbing: true,
      climbableId: climbable.id,
      debugEvent: wasClimbing
        ? null
        : {
            type: 'ladder',
            action: 'mount',
            payload: {
              climbableId: climbable.id,
              x: Math.round(player.x),
              y: Math.round(player.y)
            }
          },
      grounded: false,
      vy: 0,
      attack: false,
      vx: 0
    };
  }

  function getCurrentClimbable(climbable, climbables, climbableId) {
    return climbable || climbables.find((item) => item.id === climbableId) || null;
  }

  function getClimbMovementStepPlan(player, climbable, input, delta, wasClimbing, options) {
    const controls = input || {};
    const settings = options || {};
    const vertical = (controls.down ? 1 : 0) - (controls.up ? 1 : 0);
    const centerTarget = climbable.x + climbable.w / 2 - player.w / 2;
    const topExitY = climbable.y - player.h;
    const bottomExitY = climbable.y + climbable.h - player.h;
    let y = player.y;
    if (!wasClimbing) {
      const groundedPlatformId = normalizeId(player.groundedPlatformId);
      if (vertical > 0 && groundedPlatformId === climbable.topPlatformId) y = topExitY;
      else if (vertical < 0 && groundedPlatformId === climbable.bottomPlatformId) y = bottomExitY;
    }
    y += vertical * settings.climbSpeed * delta;
    y = clamp(y, topExitY, bottomExitY);
    return {
      vertical,
      climbMoving: vertical !== 0,
      x: centerTarget,
      y,
      nearTop: y <= topExitY + settings.mountTolerance,
      nearBottom: y >= bottomExitY - settings.mountTolerance
    };
  }

  function getClimbStepDismountEndpoint(climbStep) {
    if (climbStep.vertical < 0 && climbStep.nearTop) return 'top';
    if (climbStep.vertical > 0 && climbStep.nearBottom) return 'bottom';
    return null;
  }

  function getWorldXClampPlan(body, worldWidth) {
    return {
      x: clamp(body.x, 10, worldWidth - body.w - 10)
    };
  }

  function getClimbStepFinalizationPlan(player, worldWidth) {
    const worldClamp = getWorldXClampPlan(player, worldWidth);
    return {
      previousY: player.y,
      x: worldClamp.x
    };
  }

  function getClimbDismountPlan(player, climbable, endpoint, platforms) {
    if (!player || !climbable || !Array.isArray(platforms)) return null;
    const platformIndex = endpoint === 'top' ? climbable.topPlatformIndex : climbable.bottomPlatformIndex;
    const platform = platforms[platformIndex];
    if (!platform) return null;
    const ladderCenterX = climbable.x + climbable.w / 2 - player.w / 2;
    const x = clamp(ladderCenterX, platform.x + 4, platform.x + platform.w - player.w - 4);
    return {
      platform,
      platformIndex,
      x,
      y: getPlatformSurfaceY(platform, x + player.w / 2) - player.h
    };
  }

  function getClimbDismountStatePlan(climbable, endpoint, dismountPlan, options) {
    const settings = options || {};
    const platform = dismountPlan.platform;
    return {
      climbing: false,
      climbMoving: false,
      climbableId: '',
      climbLockDuration: settings.climbLockDuration,
      grounded: true,
      groundedPlatformId: platform.id,
      groundedPlatformIndex: platform.index,
      clearDropThrough: true,
      vy: 0,
      x: dismountPlan.x,
      y: dismountPlan.y,
      previousY: dismountPlan.y,
      debugEvent: {
        type: 'ladder',
        action: 'dismount',
        payload: {
          climbableId: climbable.id,
          endpoint,
          platformId: platform.id,
          x: Math.round(dismountPlan.x),
          y: Math.round(dismountPlan.y)
        }
      }
    };
  }

  function getClimbableMountMetrics(body, climbable, options) {
    const settings = options || {};
    const check = {
      x: body.x + body.w * 0.25,
      y: body.y + 8,
      w: body.w * 0.5,
      h: Math.max(12, body.h - 12)
    };
    return {
      check,
      centerX: body.x + body.w / 2,
      bottom: body.y + body.h,
      ladderCenter: climbable.x + climbable.w / 2,
      mountWidth: Math.max(settings.mountWidth, climbable.w + 58)
    };
  }

  function bodyMatchesPlatformMountHeight(body, platform, options) {
    if (!body || !platform) return false;
    const settings = options || {};
    const centerX = Number(body.x || 0) + Number(body.w || 0) / 2;
    const bottom = Number(body.y || 0) + Number(body.h || 0);
    const surfaceY = getPlatformSurfaceY(platform, centerX);
    return Math.abs(bottom - surfaceY) <= settings.mountTolerance ||
      Math.abs(bottom - Number(platform.y || 0)) <= settings.mountTolerance;
  }

  function getClimbablePlatformEndpointMatches(bodyPlatform, climbable, bottom, options) {
    const settings = options || {};
    const platformMatchesHeight = bodyPlatform && Math.abs(bottom - bodyPlatform.y) <= settings.mountTolerance;
    return {
      onTopPlatform: !!(platformMatchesHeight && bodyPlatform.id === climbable.topPlatformId),
      onBottomPlatform: !!(platformMatchesHeight && bodyPlatform.id === climbable.bottomPlatformId)
    };
  }

  function bodyMatchesClimbableMountReach(check, climbable, bottom, options) {
    if (!check || !climbable) return false;
    const settings = options || {};
    return rectsOverlap(check, climbable) ||
      Math.abs(bottom - climbable.y) <= settings.mountTolerance ||
      Math.abs(bottom - (climbable.y + climbable.h)) <= settings.mountTolerance;
  }

  function getOverlappingClimbable(body, climbables, bodyPlatform, verticalIntent, currentTime, options) {
    if (!body || !climbables) return null;
    const settings = options || {};
    const intent = Math.sign(Number(verticalIntent || 0));
    const now = currentTime;
    if (intent && Number(body.dropThroughUntil || 0) > now) return null;
    const check = {
      x: body.x + body.w * 0.25,
      y: body.y + 8,
      w: body.w * 0.5,
      h: Math.max(12, body.h - 12)
    };
    const centerX = body.x + body.w / 2;
    const bottom = body.y + body.h;
    const getMountMetrics = (climbable) => getClimbableMountMetrics(body, climbable, {
      mountWidth: settings.mountWidth
    });
    if (intent && !body.climbing) {
      if (!body.grounded) {
        if (intent >= 0) return null;
        return climbables.find((climbable) => {
          const metrics = getMountMetrics(climbable);
          const ladderCenter = metrics.ladderCenter;
          const mountWidth = metrics.mountWidth;
          if (Math.abs(centerX - ladderCenter) > mountWidth / 2) return false;
          return bodyMatchesClimbableMountReach(check, climbable, bottom, {
            mountTolerance: settings.mountTolerance
          });
        }) || null;
      }
      if (!bodyPlatform) return null;
      return climbables.find((climbable) => {
        const metrics = getMountMetrics(climbable);
        const ladderCenter = metrics.ladderCenter;
        const mountWidth = metrics.mountWidth;
        if (Math.abs(centerX - ladderCenter) > mountWidth / 2) return false;
        const endpointMatches = getClimbablePlatformEndpointMatches(bodyPlatform, climbable, bottom, {
          mountTolerance: settings.mountTolerance
        });
        const onTopPlatform = endpointMatches.onTopPlatform;
        const onBottomPlatform = endpointMatches.onBottomPlatform;
        return intent < 0 ? onBottomPlatform : onTopPlatform;
      }) || null;
    }
    const overlap = climbables.find((climbable) => rectsOverlap(check, climbable));
    if (overlap) return overlap;
    return climbables.find((climbable) => {
      const metrics = getMountMetrics(climbable);
      const ladderCenter = metrics.ladderCenter;
      const mountWidth = metrics.mountWidth;
      if (Math.abs(centerX - ladderCenter) > mountWidth / 2) return false;
      const endpointMatches = getClimbablePlatformEndpointMatches(bodyPlatform, climbable, bottom, {
        mountTolerance: settings.mountTolerance
      });
      const onTopPlatform = endpointMatches.onTopPlatform;
      const onBottomPlatform = endpointMatches.onBottomPlatform;
      return onTopPlatform ||
        onBottomPlatform ||
        bodyMatchesClimbableMountReach(check, climbable, bottom, {
          mountTolerance: settings.mountTolerance
        });
    }) || null;
  }

  function canMountClimbableFromGround(body, climbable, verticalIntent, bodyPlatform, currentTime, options) {
    if (!body || !climbable || !verticalIntent) return false;
    if (Number(body.dropThroughUntil || 0) > currentTime) return false;
    const settings = options || {};
    const metrics = getClimbableMountMetrics(body, climbable, {
      mountWidth: settings.mountWidth
    });
    const centerX = metrics.centerX;
    const bottom = metrics.bottom;
    const ladderCenter = metrics.ladderCenter;
    const mountWidth = metrics.mountWidth;
    if (Math.abs(centerX - ladderCenter) > mountWidth / 2) return false;
    if (!body.grounded) {
      if (verticalIntent >= 0) return false;
      const check = metrics.check;
      return bodyMatchesClimbableMountReach(check, climbable, bottom, {
        mountTolerance: settings.mountTolerance
      });
    }
    const isAtPlatformMountHeight = (platform) => {
      return bodyMatchesPlatformMountHeight(body, platform, {
        mountTolerance: settings.mountTolerance
      });
    };
    const onTopPlatform = bodyPlatform &&
      bodyPlatform.id === climbable.topPlatformId &&
      isAtPlatformMountHeight(bodyPlatform);
    const onBottomPlatform = bodyPlatform &&
      bodyPlatform.id === climbable.bottomPlatformId &&
      isAtPlatformMountHeight(bodyPlatform);
    if (verticalIntent < 0) return !!onBottomPlatform;
    if (verticalIntent > 0) return !!onTopPlatform;
    return false;
  }

  function createDropThroughState(body, duration, platform, currentTime) {
    return {
      dropThroughUntil: currentTime + Math.max(0, Number(duration || 0.28) || 0.28),
      dropThroughPlatformId: platform && platform.id || normalizeId(body && body.groundedPlatformId),
      dropThroughPlatformIndex: platform && Number.isFinite(Number(platform.index))
        ? Number(platform.index)
        : Number.isFinite(Number(body && body.groundedPlatformIndex))
          ? Number(body.groundedPlatformIndex)
          : -1
    };
  }

  function getPlatformLandingResolution(metrics, platform) {
    if (!metrics || !platform) return null;
    const bodyX = Number(metrics.bodyX || 0);
    const bodyY = Number(metrics.bodyY || 0);
    const bodyW = Number(metrics.bodyW || 0);
    const centerX = Number(metrics.centerX || 0);
    const bottom = Number(metrics.bottom || 0);
    const previousBottom = Number(metrics.previousBottom || 0);
    const horizontal = bodyX + bodyW > platform.x && bodyX < platform.x + platform.w && platformContainsX(platform, centerX, bodyW / 2);
    if (!horizontal) return null;
    const surfaceY = getPlatformSurfaceY(platform, centerX);
    return {
      surfaceY,
      canLand: previousBottom <= surfaceY + 10 && bottom >= surfaceY && bodyY < surfaceY + 4
    };
  }

  function shouldSkipDropThroughPlatform(body, platform, nowSeconds) {
    if (!body || !platform || !platform.dropThrough) return false;
    if (Number(body.dropThroughUntil || 0) <= Number(nowSeconds || 0)) return false;
    const platformId = normalizeId(body.dropThroughPlatformId);
    if (platformId) return platform.id === platformId;
    const platformIndex = Number(body.dropThroughPlatformIndex);
    return Number.isFinite(platformIndex) && platformIndex >= 0 && platform.index === platformIndex;
  }

  const api = {
    isLevelTeleportTarget,
    findVerticalTeleportTarget,
    findHorizontalTeleportTarget,
    getMovementSkillPlan,
    getRuntimePlatformPlacementStatePlan,
    getTrialInstanceStartPlacementStatePlan,
    getHometownReturnPlacementStatePlan,
    getMapEntryPlacementStatePlan,
    getChannelPositionLandingSearchPlan,
    getChannelPositionPlacementStatePlan,
    getInstanceTravelResetStatePlan,
    getMapChangeTravelStatePlan,
    getSkillMovementApplicationPlan,
    getSkillMovementInvulnerabilityPlan,
    getSkillMovementBlinkStartEffectPlan,
    getSkillMovementBlinkEndEffectPlan,
    isSkillMovementInstant,
    getSkillMovementInstantVelocityStatePlan,
    getSkillMovementBlinkFallbackStatePlan,
    isSkillMovementBlinkTargetPlan,
    getSkillMovementMobilityStatePlan,
    getSkillMovementBlinkTargetStatePlan,
    getSkillMovementActionEffectPlan,
    getSkillMovementFlameTrailEffectPlan,
    getMapMovementProfile,
    getMovementInputIntent,
    getActiveMobilityState,
    getMobilityStepPlan,
    getMobilityMovementPlan,
    getAirDashMomentumMovementPlan,
    getHorizontalMovementStepPlan,
    getJumpMovementAction,
    getVerticalMovementStepPlan,
    getClimbActivationPlan,
    getClimbClearStatePlan,
    getClimbJumpExitPlan,
    getClimbMountStatePlan,
    getCurrentClimbable,
    getClimbMovementStepPlan,
    getClimbStepDismountEndpoint,
    getWorldXClampPlan,
    getClimbStepFinalizationPlan,
    getClimbDismountPlan,
    getClimbDismountStatePlan,
    getClimbableMountMetrics,
    bodyMatchesPlatformMountHeight,
    getClimbablePlatformEndpointMatches,
    bodyMatchesClimbableMountReach,
    getOverlappingClimbable,
    canMountClimbableFromGround,
    createDropThroughState,
    getPlatformLandingResolution,
    shouldSkipDropThroughPlatform
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.movement = Object.assign({}, modules.movement || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
