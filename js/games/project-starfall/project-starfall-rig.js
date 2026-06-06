(function initProjectStarfallRig(global) {
  'use strict';

  const TAU = Math.PI * 2;

  function canDraw(ctx) {
    return !!(ctx && typeof ctx.save === 'function' && typeof ctx.fillRect === 'function');
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, Number(value) || 0));
  }

  function lerp(a, b, t) {
    return a + (b - a) * clamp(t, 0, 1);
  }

  function easeOut(t) {
    const next = 1 - clamp(t, 0, 1);
    return 1 - next * next;
  }

  function weaponKindFromContext(context) {
    return context && context.weaponKind ? String(context.weaponKind) : 'sword';
  }

  function isCasterWeapon(kind) {
    return kind === 'wand' || kind === 'staff';
  }

  function isBowWeapon(kind) {
    return kind === 'bow';
  }

	  function normalizeItemId(item) {
	    return String(item && (item.id || item.visualId) || '').trim();
	  }

	  function equippedItem(equipment, slot) {
	    return equipment && equipment[slot] ? equipment[slot] : null;
	  }

	  function equipmentVisual(rig, item, fallback) {
	    const visualIds = [
	      normalizeItemId(item),
	      String(item && item.visualId || '').trim()
	    ].filter(Boolean);
	    if (rig && rig.equipmentVisuals) {
	      for (const id of visualIds) {
	        if (rig.equipmentVisuals[id]) return rig.equipmentVisuals[id];
	      }
	    }
	    return fallback || null;
	  }

  function animationFrame(rig, state, elapsed) {
    const states = rig && rig.animationStates ? rig.animationStates : {};
    const config = states[state] || states.idle || { frames: 1, fps: 1, loop: true };
    const frames = Math.max(1, Number(config.frames) || 1);
    const fps = Math.max(1, Number(config.fps) || 1);
    const rawFrame = Math.max(0, Number(elapsed) || 0) * fps;
    const frameIndex = config.loop
      ? Math.floor(rawFrame) % frames
      : Math.min(frames - 1, Math.floor(rawFrame));
    const duration = frames / fps;
    return {
      config,
      frameIndex,
      phase: config.loop
        ? (rawFrame % frames) / frames
        : clamp((Number(elapsed) || 0) / Math.max(0.001, duration), 0, 1),
      framePhase: rawFrame - Math.floor(rawFrame)
    };
  }

  function basePose(rig) {
    const anchors = rig && rig.anchors ? rig.anchors : {};
    return {
      rootX: 0,
      rootY: 0,
      bodyY: 0,
      torsoTilt: 0,
      headX: 0,
      headY: 0,
      frontShoulder: Object.assign({ x: 10, y: -27 }, anchors.frontShoulder),
      backShoulder: Object.assign({ x: -8, y: -28 }, anchors.backShoulder),
      frontHip: Object.assign({ x: 7, y: 3 }, anchors.frontHip),
      backHip: Object.assign({ x: -5, y: 3 }, anchors.backHip),
      frontLeg: { upper: -0.05, lower: 0.1, foot: 0.02 },
      backLeg: { upper: 0.08, lower: 0.02, foot: -0.04 },
      frontArm: { upper: -0.18, lower: -0.2 },
      backArm: { upper: 0.12, lower: 0.1 },
      weaponAngle: -0.06,
      weaponLift: 0,
      weaponReach: 0,
      slashArc: 0,
      bowDraw: 0,
      bowRelease: 0,
      shieldX: -7,
      shieldY: 0,
      aura: 0,
      recoil: 0,
      defeated: false
    };
  }

  function resolvePose(rig, state, elapsed, context) {
    const frame = animationFrame(rig, state, elapsed);
    const pose = basePose(rig);
    const weaponKind = weaponKindFromContext(context);
    const phase = frame.phase;
    const cycle = Math.sin(phase * TAU);
    const bounce = Math.abs(cycle);

    if (state === 'idle') {
      pose.bodyY = bounce > 0.92 ? -1 : 0;
      pose.headY = cycle > 0.2 ? -1 : 0;
      pose.headX = 2;
      pose.frontArm.upper = -0.18 + cycle * 0.03;
      pose.backArm.upper = 0.12 - cycle * 0.02;
      return pose;
    }

    if (state === 'run') {
      const frameIndex = frame.frameIndex % 4;
      const walk = [
        {
          bodyY: 0,
          headY: 0,
          torsoTilt: 0.13,
          frontLeg: { upper: -0.34, lower: 0.12, foot: 0.02, knee: { x: 4, y: 21 }, ankle: { x: 0, y: 39 } },
          backLeg: { upper: 0.32, lower: 0.04, foot: -0.08, knee: { x: -12, y: 22 }, ankle: { x: -18, y: 39 } },
          frontArm: { upper: 0.23, lower: 0.03 },
          backArm: { upper: -0.25, lower: -0.08 },
          weaponAngle: -0.08,
          weaponLift: 0
        },
        {
          bodyY: -1,
          headY: -1,
          torsoTilt: 0.11,
          frontLeg: { upper: -0.08, lower: 0.24, foot: -0.01, knee: { x: 5, y: 22 }, ankle: { x: -2, y: 39 } },
          backLeg: { upper: 0.12, lower: 0.2, foot: -0.04, knee: { x: -8, y: 20 }, ankle: { x: -10, y: 35 } },
          frontArm: { upper: -0.03, lower: -0.14 },
          backArm: { upper: 0.02, lower: 0.12 },
          weaponAngle: -0.14,
          weaponLift: -1
        },
        {
          bodyY: 0,
          headY: 0,
          torsoTilt: 0.13,
          frontLeg: { upper: 0.24, lower: 0.04, foot: -0.08, knee: { x: -2, y: 21 }, ankle: { x: -10, y: 39 } },
          backLeg: { upper: -0.28, lower: 0.12, foot: 0.02, knee: { x: 2, y: 22 }, ankle: { x: 0, y: 39 } },
          frontArm: { upper: -0.3, lower: -0.12 },
          backArm: { upper: 0.24, lower: 0.05 },
          weaponAngle: -0.18,
          weaponLift: -1
        },
        {
          bodyY: -1,
          headY: -1,
          torsoTilt: 0.11,
          frontLeg: { upper: 0.1, lower: 0.2, foot: -0.04, knee: { x: 5, y: 20 }, ankle: { x: 2, y: 35 } },
          backLeg: { upper: -0.08, lower: 0.24, foot: -0.01, knee: { x: -8, y: 22 }, ankle: { x: -14, y: 39 } },
          frontArm: { upper: -0.03, lower: -0.14 },
          backArm: { upper: 0.02, lower: 0.12 },
          weaponAngle: -0.14,
          weaponLift: -1
        }
      ][frameIndex];
      Object.assign(pose, walk);
      pose.frontLeg = walk.frontLeg;
      pose.backLeg = walk.backLeg;
      pose.frontArm = walk.frontArm;
      pose.backArm = walk.backArm;
      pose.frontShoulder.x += 2;
      pose.backShoulder.x += 2;
      pose.headX = 5;
      return pose;
    }

    if (state === 'jump') {
      pose.rootY = -2;
      pose.bodyY = 0;
      pose.headX = 2;
      pose.headY = -1;
      pose.frontLeg = { upper: -0.3, lower: 0.5, foot: 0.1 };
      pose.backLeg = { upper: 0.24, lower: 0.44, foot: -0.04 };
      pose.frontArm = { upper: -0.62, lower: -0.34 };
      pose.backArm = { upper: 0.34, lower: 0.12 };
      pose.weaponAngle = -0.32;
      return pose;
    }

    if (state === 'fall') {
      pose.rootY = 1;
      pose.headX = 2;
      pose.headY = 1;
      pose.frontLeg = { upper: -0.16, lower: 0.16, foot: 0.02 };
      pose.backLeg = { upper: 0.2, lower: 0.12, foot: -0.06 };
      pose.frontArm = { upper: -0.5, lower: -0.18 };
      pose.backArm = { upper: 0.44, lower: 0.14 };
      pose.weaponAngle = -0.36;
      return pose;
    }

    if (state === 'climb') {
      pose.rootY = cycle > 0 ? -1 : 1;
      pose.bodyY = 0;
      pose.headY = -Math.round(bounce);
      pose.headX = 1;
      pose.frontArm = { upper: -0.56 * cycle - 0.22, lower: -0.18 };
      pose.backArm = { upper: 0.56 * cycle + 0.08, lower: 0.16 };
      pose.frontLeg = { upper: 0.22 * cycle, lower: 0.16 + 0.18 * Math.max(0, cycle), foot: 0 };
      pose.backLeg = { upper: -0.22 * cycle, lower: 0.16 + 0.18 * Math.max(0, -cycle), foot: 0 };
      pose.weaponAngle = -0.24;
      return pose;
    }

    if (state === 'basic') {
      const windup = clamp(phase / 0.34, 0, 1);
      const strike = clamp((phase - 0.28) / 0.34, 0, 1);
      const recover = clamp((phase - 0.62) / 0.38, 0, 1);
      const reach = easeOut(strike) * (1 - recover * 0.75);
      if (isCasterWeapon(weaponKind)) {
        pose.rootX = Math.round(1 * reach);
        pose.rootY = -Math.round(1 * strike);
        pose.headX = 2 + Math.round(1 * reach);
        pose.headY = -Math.round(1 * strike);
        pose.frontLeg = { upper: -0.1, lower: 0.18, foot: 0.03 };
        pose.backLeg = { upper: 0.16, lower: 0.08, foot: -0.05 };
        pose.frontArm = {
          upper: lerp(-0.22, -0.98, strike),
          lower: lerp(-0.18, -0.72, strike)
        };
        pose.backArm = { upper: lerp(0.08, 0.24, windup), lower: 0.1 };
        pose.weaponAngle = lerp(-0.34, -0.72, strike);
        pose.weaponLift = lerp(-3, -11, strike) + recover * 4;
        pose.weaponReach = Math.round(3 * reach);
        pose.aura = 0.2 + reach * 0.65;
        return pose;
      }
      if (isBowWeapon(weaponKind)) {
        const draw = clamp(windup * (1 - recover * 0.7), 0, 1);
        const release = clamp((phase - 0.48) / 0.2, 0, 1) * (1 - recover * 0.9);
        pose.rootX = Math.round(1 + 1.5 * reach);
        pose.headX = 3;
        pose.frontLeg = { upper: -0.12, lower: 0.2, foot: 0.04 };
        pose.backLeg = { upper: 0.22, lower: 0.06, foot: -0.07 };
        pose.frontArm = {
          upper: lerp(-0.36, -0.94, draw),
          lower: lerp(-0.24, -1.08, draw)
        };
        pose.backArm = {
          upper: lerp(0.1, -0.88, draw),
          lower: lerp(0.06, 1.02, draw)
        };
        pose.weaponAngle = lerp(-0.08, -0.16, draw);
        pose.weaponLift = lerp(-2, -5, draw) + recover * 2;
        pose.weaponReach = Math.round(3 * reach);
        pose.bowDraw = draw;
        pose.bowRelease = release;
        pose.aura = Math.max(reach * 0.35, draw * 0.22);
        return pose;
      }
      const swing = easeOut(strike);
      pose.rootX = Math.round(3 * reach);
      pose.rootY = -Math.round(1 * windup);
      pose.headX = 2 + Math.round(1 * reach);
      pose.frontLeg = { upper: -0.18, lower: 0.18, foot: 0.05 };
      pose.backLeg = { upper: 0.24, lower: 0.04, foot: -0.08 };
      pose.frontArm = {
        upper: lerp(-0.22, -1.12, swing),
        lower: lerp(-0.92, -0.2, swing)
      };
      pose.backArm = { upper: lerp(-0.18, 0.24, windup), lower: lerp(0.02, 0.18, windup) };
      pose.weaponAngle = lerp(-1.24, 0.5, swing) - recover * 0.3;
      pose.weaponLift = lerp(-12, 4, swing) + recover * 2;
      pose.weaponReach = Math.round(4 * reach);
      pose.slashArc = reach;
      pose.aura = reach * 0.4;
      return pose;
    }

    if (state === 'skill') {
      const windup = clamp(phase / 0.3, 0, 1);
      const cleave = clamp((phase - 0.26) / 0.36, 0, 1);
      const recover = clamp((phase - 0.64) / 0.36, 0, 1);
      const power = easeOut(cleave) * (1 - recover * 0.6);
      if (isCasterWeapon(weaponKind)) {
        pose.rootX = Math.round(1 * power);
        pose.rootY = -Math.round(2 * windup);
        pose.headX = 2;
        pose.headY = -2;
        pose.frontLeg = { upper: -0.08, lower: 0.22, foot: 0.02 };
        pose.backLeg = { upper: 0.18, lower: 0.12, foot: -0.06 };
        pose.frontArm = { upper: lerp(-0.46, -1.18, cleave), lower: lerp(-0.3, -0.9, cleave) };
        pose.backArm = { upper: lerp(0.1, -0.36, windup), lower: lerp(0.1, -0.28, cleave) };
        pose.weaponAngle = lerp(-0.48, -0.86, cleave);
        pose.weaponLift = lerp(-8, -17, cleave) + recover * 6;
        pose.weaponReach = Math.round(4 * power);
        pose.aura = 0.35 + power * 0.8;
        return pose;
      }
      if (isBowWeapon(weaponKind)) {
        const draw = clamp(windup * (1 - recover * 0.65), 0, 1);
        const release = clamp((phase - 0.5) / 0.18, 0, 1) * (1 - recover * 0.85);
        pose.rootX = Math.round(1 + 2 * power);
        pose.rootY = -Math.round(1 * windup);
        pose.headX = 3;
        pose.headY = -1;
        pose.frontLeg = { upper: -0.16, lower: 0.22, foot: 0.04 };
        pose.backLeg = { upper: 0.28, lower: 0.08, foot: -0.08 };
        pose.frontArm = { upper: lerp(-0.44, -1.02, draw), lower: lerp(-0.28, -1.12, draw) };
        pose.backArm = { upper: lerp(0.12, -0.94, draw), lower: lerp(0.02, 1.08, draw) };
        pose.weaponAngle = lerp(-0.08, -0.18, draw);
        pose.weaponLift = lerp(-4, -7, draw) + recover * 2;
        pose.weaponReach = Math.round(4 * power);
        pose.bowDraw = draw;
        pose.bowRelease = release;
        pose.aura = 0.2 + Math.max(power * 0.45, draw * 0.35);
        return pose;
      }
      pose.rootX = Math.round(2 + 8 * power);
      pose.rootY = -Math.round(1 * windup);
      pose.headX = 2;
      pose.headY = -1;
      pose.frontLeg = { upper: -0.26, lower: 0.24, foot: 0.04 };
      pose.backLeg = { upper: 0.32, lower: 0.08, foot: -0.08 };
      pose.frontArm = { upper: lerp(-0.72, -1.28, cleave), lower: lerp(-0.2, -0.86, cleave) };
      pose.backArm = { upper: 0.24, lower: 0.1 };
      pose.weaponAngle = lerp(-1.08, 0.16, cleave);
      pose.weaponLift = lerp(-12, 3, cleave);
      pose.weaponReach = Math.round(12 * power);
      pose.aura = 0.25 + power * 0.65;
      return pose;
    }

    if (state === 'party') {
      const rise = clamp(phase / 0.35, 0, 1);
      const flare = clamp((phase - 0.28) / 0.46, 0, 1);
      pose.bodyY = -Math.round(Math.sin(phase * Math.PI) * 2);
      pose.headX = 2;
      pose.headY = -2;
      pose.frontArm = { upper: lerp(-0.18, -0.82, rise), lower: lerp(-0.2, -0.44, rise) };
      pose.backArm = { upper: lerp(0.12, 0.78, rise), lower: lerp(0.1, 0.4, rise) };
      pose.weaponAngle = -0.58;
      pose.weaponLift = -8;
      pose.aura = 0.35 + Math.sin(flare * Math.PI) * 0.65;
      return pose;
    }

    if (state === 'hit') {
      pose.rootX = -5;
      pose.headX = -3;
      pose.headY = 1;
      pose.frontArm = { upper: 0.16, lower: -0.02 };
      pose.backArm = { upper: -0.22, lower: 0.04 };
      pose.frontLeg = { upper: 0.18, lower: 0.08, foot: -0.06 };
      pose.backLeg = { upper: -0.16, lower: 0.12, foot: 0.02 };
      pose.weaponAngle = -0.42;
      pose.recoil = 1;
      return pose;
    }

    if (state === 'defeat') {
      pose.rootY = 10;
      pose.defeated = true;
      return pose;
    }

    return pose;
  }

  function segmentEnd(start, angle, length) {
    return {
      x: start.x - Math.sin(angle) * length,
      y: start.y + Math.cos(angle) * length
    };
  }

  function limbPoints(start, upperAngle, lowerAngle, upperLength, lowerLength) {
    const elbow = segmentEnd(start, upperAngle, upperLength);
    const hand = segmentEnd(elbow, lowerAngle, lowerLength);
    return { start, elbow, hand };
  }

  function limbPointsFromTargets(start, limb, upperLength, lowerLength) {
    if (!limb || !limb.knee || !limb.ankle) return limbPoints(start, limb.upper, limb.lower, upperLength, lowerLength);
    return {
      start,
      elbow: { x: Number(limb.knee.x) || 0, y: Number(limb.knee.y) || 0 },
      hand: { x: Number(limb.ankle.x) || 0, y: Number(limb.ankle.y) || 0 }
    };
  }

  function legFootPoint(pose, which) {
    const isFront = which === 'front';
    const hip = isFront ? pose.frontHip : pose.backHip;
    const limb = isFront ? pose.frontLeg : pose.backLeg;
    return limbPointsFromTargets(hip, limb, 18, 18).hand;
  }

  function drawBlock(ctx, x, y, w, h, color, outline) {
    const px = Math.round(x);
    const py = Math.round(y);
    const pw = Math.round(w);
    const ph = Math.round(h);
    if (outline) {
      ctx.fillStyle = outline;
      ctx.fillRect(px - 2, py - 2, pw + 4, ph + 4);
    }
    ctx.fillStyle = color;
    ctx.fillRect(px, py, pw, ph);
  }

  function drawSegment(ctx, start, angle, length, width, color, outline) {
    ctx.save();
    ctx.translate(Math.round(start.x), Math.round(start.y));
    ctx.rotate(angle);
    drawBlock(ctx, -width / 2, 0, width, length, color, outline);
    ctx.restore();
  }

  function angleBetween(start, end) {
    return Math.atan2(start.x - end.x, end.y - start.y);
  }

  function distanceBetween(start, end) {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function drawSegmentBetween(ctx, start, end, width, color, outline) {
    drawSegment(ctx, start, angleBetween(start, end), distanceBetween(start, end), width, color, outline);
  }

  function drawFoot(ctx, ankle, angle, palette, bootVisual, front) {
    const leather = bootVisual && bootVisual.leather ? bootVisual.leather : palette.boot;
    const sole = bootVisual && bootVisual.sole ? bootVisual.sole : palette.outline;
    const buckle = bootVisual && bootVisual.buckle ? bootVisual.buckle : palette.belt;
    ctx.save();
    ctx.translate(Math.round(ankle.x), Math.round(ankle.y));
    ctx.rotate(angle);
    drawBlock(ctx, -3, -1, 15, 6, leather, palette.outline);
    drawBlock(ctx, -2, 4, 16, 2, sole, null);
    if (front && bootVisual) drawBlock(ctx, 7, 1, 3, 2, buckle, null);
    ctx.restore();
  }

  function drawLeg(ctx, pose, which, palette, bootVisual) {
    const isFront = which === 'front';
    const hip = isFront ? pose.frontHip : pose.backHip;
    const limb = isFront ? pose.frontLeg : pose.backLeg;
    const upperColor = isFront ? palette.pants : palette.pantsDark;
    const lowerColor = isFront ? palette.pantsDark : '#1f2b38';
    const points = limbPointsFromTargets(hip, limb, 18, 18);
    drawSegmentBetween(ctx, points.start, points.elbow, 8, upperColor, palette.outline);
    drawSegmentBetween(ctx, points.elbow, points.hand, 7, lowerColor, palette.outline);
    drawFoot(ctx, points.hand, limb.foot || limb.lower, palette, bootVisual, isFront);
    return points.hand;
  }

  function drawArm(ctx, pose, which, palette, gloveVisual) {
    const isFront = which === 'front';
    const shoulder = isFront ? pose.frontShoulder : pose.backShoulder;
    const limb = isFront ? pose.frontArm : pose.backArm;
    const sleeve = isFront ? palette.shirtLight : palette.shirt;
    const hand = gloveVisual && gloveVisual.dark ? gloveVisual.dark : palette.hand;
    const points = limbPoints(shoulder, limb.upper, limb.lower, 15, 14);
    drawSegment(ctx, points.start, limb.upper, 15, 7, sleeve, palette.outline);
    drawSegment(ctx, points.elbow, limb.lower, 14, 6, sleeve, palette.outline);
    drawBlock(ctx, points.hand.x - 3, points.hand.y - 3, 7, 6, hand, palette.outline);
    return points.hand;
  }

  function drawContactShadows(ctx, pose, palette) {
    const front = legFootPoint(pose, 'front');
    const back = legFootPoint(pose, 'back');
    ctx.save();
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = palette.shadow;
    ctx.fillRect(Math.round(back.x - 7), 41, 13, 2);
    ctx.fillRect(Math.round(front.x - 5), 42, 15, 2);
    ctx.restore();
  }

  function drawTorso(ctx, pose, palette, chestVisual) {
    ctx.save();
    ctx.translate(0, pose.bodyY);
    ctx.rotate(pose.torsoTilt - 0.04);
    drawBlock(ctx, -1, -45, 9, 10, palette.skin, palette.outline);
    drawBlock(ctx, -12, -39, 25, 42, palette.outline, null);
    drawBlock(ctx, -10, -36, 21, 34, palette.shirt, null);
    drawBlock(ctx, 4, -35, 7, 32, palette.shirtLight, null);
    drawBlock(ctx, -10, -4, 22, 6, palette.belt, null);
    if (chestVisual) {
      drawBlock(ctx, -10, -34, 22, 29, chestVisual.cloth || '#8c5a3a', null);
      drawBlock(ctx, -8, -32, 19, 5, chestVisual.trim || '#d39a5c', null);
      drawBlock(ctx, 3, -27, 3, 18, chestVisual.stitch || '#f3d5a0', null);
      drawBlock(ctx, -8, -8, 19, 4, chestVisual.trim || '#d39a5c', null);
    }
    ctx.restore();
  }

  function drawHead(ctx, pose, palette, headVisual) {
    const x = 2 + pose.headX;
    const y = pose.headY;
    drawBlock(ctx, -12 + x, -64 + y, 25, 25, palette.outline, null);
    drawBlock(ctx, -9 + x, -60 + y, 20, 18, palette.skin, null);
    drawBlock(ctx, -10 + x, -65 + y, 23, 8, palette.hair, null);
    drawBlock(ctx, -10 + x, -58 + y, 5, 10, palette.hair, null);
    if (headVisual) {
      drawBlock(ctx, -11 + x, -68 + y, 24, 6, headVisual.trim || palette.belt, palette.outline);
    }
  }

  function drawSword(ctx, hand, pose, visual) {
    const blade = visual && visual.blade ? visual.blade : '#dce6ee';
    const shine = visual && visual.shine ? visual.shine : '#ffffff';
    const grip = visual && visual.grip ? visual.grip : '#8f5f39';
    ctx.save();
    ctx.translate(Math.round(hand.x + pose.weaponReach * 0.45), Math.round(hand.y + pose.weaponLift));
    ctx.rotate(pose.weaponAngle);
    drawBlock(ctx, -6, -3, 13, 6, grip, '#13222f');
    drawBlock(ctx, 0, -8, 6, 16, grip, '#13222f');
    drawBlock(ctx, 6, -4, 35, 7, blade, '#13222f');
    drawBlock(ctx, 35, -5, 8, 5, shine, null);
    drawBlock(ctx, 16, -3, 17, 2, shine, null);
    ctx.restore();
  }

  function drawAxe(ctx, hand, pose, visual) {
    const blade = visual && visual.blade ? visual.blade : '#d6e1e8';
    const shine = visual && visual.shine ? visual.shine : '#ffffff';
    const grip = visual && visual.grip ? visual.grip : '#815334';
    ctx.save();
    ctx.translate(Math.round(hand.x + pose.weaponReach), Math.round(hand.y + pose.weaponLift));
    ctx.rotate(pose.weaponAngle);
    drawBlock(ctx, -7, -3, 37, 6, grip, '#13222f');
    drawBlock(ctx, 25, -13, 16, 20, blade, '#13222f');
    drawBlock(ctx, 28, -9, 12, 4, shine, null);
    ctx.restore();
  }

  function drawWand(ctx, hand, pose, visual, staff) {
    const rod = visual && visual.rod ? visual.rod : '#724c2f';
    const glow = visual && visual.glow ? visual.glow : '#8bd7ff';
    const gem = visual && visual.gem ? visual.gem : '#cbe8ff';
    const length = staff ? 43 : 31;
    ctx.save();
    ctx.translate(Math.round(hand.x + pose.weaponReach), Math.round(hand.y + pose.weaponLift));
    ctx.rotate(pose.weaponAngle - 0.12);
    drawBlock(ctx, -5, -2, length, 4, rod, '#13222f');
    drawBlock(ctx, length - 4, -5, 8, 8, gem, '#13222f');
    ctx.save();
    ctx.globalAlpha = 0.18 + pose.aura * 0.18;
    ctx.fillStyle = glow;
    ctx.fillRect(length + 4, -8, 13, 13);
    ctx.restore();
    ctx.restore();
  }

  function drawBow(ctx, hand, backHand, pose, visual) {
    const wood = visual && visual.wood ? visual.wood : '#8a5c2f';
    const string = visual && visual.string ? visual.string : '#f5efd6';
    const arrow = visual && visual.arrow ? visual.arrow : '#ffe16a';
    const outline = '#13222f';
    const height = visual && visual.long ? 68 : 56;
    const halfHeight = height / 2;
    const bellyX = visual && visual.long ? 17 : 14;
    const angle = Number(pose.weaponAngle || -0.08);
    const forward = { x: Math.cos(angle), y: Math.sin(angle) };
    const normal = { x: -Math.sin(angle), y: Math.cos(angle) };
    const grip = {
      x: hand.x + 3 + Number(pose.weaponReach || 0) * 0.35,
      y: hand.y + Number(pose.weaponLift || 0)
    };
    const topBelly = {
      x: grip.x + forward.x * bellyX - normal.x * halfHeight * 0.52,
      y: grip.y + forward.y * bellyX - normal.y * halfHeight * 0.52
    };
    const bottomBelly = {
      x: grip.x + forward.x * bellyX + normal.x * halfHeight * 0.52,
      y: grip.y + forward.y * bellyX + normal.y * halfHeight * 0.52
    };
    const topTip = {
      x: grip.x + forward.x * 2 - normal.x * halfHeight,
      y: grip.y + forward.y * 2 - normal.y * halfHeight
    };
    const bottomTip = {
      x: grip.x + forward.x * 2 + normal.x * halfHeight,
      y: grip.y + forward.y * 2 + normal.y * halfHeight
    };
    const restNock = {
      x: grip.x - forward.x * 11,
      y: grip.y - forward.y * 11
    };
    const drawAmount = clamp(Number(pose.bowDraw || 0), 0, 1);
    const pulledNock = backHand
      ? { x: backHand.x - forward.x * 4, y: backHand.y - forward.y * 4 }
      : restNock;
    const nock = {
      x: lerp(restNock.x, pulledNock.x, drawAmount),
      y: lerp(restNock.y, pulledNock.y, drawAmount)
    };
    const drawArrow = drawAmount > 0.12 || Number(pose.bowRelease || 0) > 0.05 || pose.aura > 0.18;
    ctx.save();
    drawSegmentBetween(ctx, grip, topBelly, 5, wood, outline);
    drawSegmentBetween(ctx, topBelly, topTip, 5, wood, outline);
    drawSegmentBetween(ctx, grip, bottomBelly, 5, wood, outline);
    drawSegmentBetween(ctx, bottomBelly, bottomTip, 5, wood, outline);
    drawSegmentBetween(ctx, topTip, nock, 2, string, null);
    drawSegmentBetween(ctx, bottomTip, nock, 2, string, null);
    drawBlock(ctx, grip.x - 4, grip.y - 5, 8, 10, visual && visual.grip ? visual.grip : wood, outline);
    if (drawArrow) {
      const arrowAngle = Math.atan2(grip.y - nock.y, grip.x - nock.x);
      const arrowLength = Math.max(38, distanceBetween(nock, grip) + 30);
      ctx.save();
      ctx.translate(Math.round(nock.x), Math.round(nock.y));
      ctx.rotate(arrowAngle);
      drawBlock(ctx, -4, -1, arrowLength, 3, arrow, outline);
      drawBlock(ctx, arrowLength - 2, -4, 8, 7, arrow, outline);
      drawBlock(ctx, -9, -4, 5, 3, string, null);
      drawBlock(ctx, -9, 1, 5, 3, string, null);
      ctx.restore();
    }
    ctx.restore();
  }

  function drawWeapon(ctx, hand, backHand, pose, visual) {
    if (visual && visual.kind === 'axe') {
      drawAxe(ctx, hand, pose, visual);
      return;
    }
    if (visual && (visual.kind === 'wand' || visual.kind === 'staff')) {
      drawWand(ctx, hand, pose, visual, visual.kind === 'staff');
      return;
    }
    if (visual && visual.kind === 'bow') {
      drawBow(ctx, hand, backHand, pose, visual);
      return;
    }
    drawSword(ctx, hand, pose, visual);
  }

  function drawOffhand(ctx, hand, pose, visual, palette) {
    if (!visual) return;
    if (visual.kind === 'shield') {
      drawBlock(ctx, hand.x - 10 + pose.shieldX, hand.y - 15 + pose.shieldY, 24, 30, visual.trim, palette.outline);
      drawBlock(ctx, hand.x - 7 + pose.shieldX, hand.y - 11 + pose.shieldY, 18, 22, visual.face, null);
      drawBlock(ctx, hand.x - 3 + pose.shieldX, hand.y - 8 + pose.shieldY, 9, 16, visual.metal, null);
      return;
    }
    if (visual.kind === 'core' || visual.kind === 'focus') {
      const pulse = 1 + pose.aura * 5;
      ctx.save();
      ctx.globalAlpha = 0.24 + pose.aura * 0.24;
      ctx.fillStyle = visual.glow;
      ctx.fillRect(hand.x - 10 - pulse, hand.y - 10 - pulse, 20 + pulse * 2, 20 + pulse * 2);
      ctx.restore();
      drawBlock(ctx, hand.x - 4, hand.y - 4, 9, 9, visual.core, palette.outline);
      return;
    }
    if (visual.kind === 'grip') {
      drawBlock(ctx, hand.x - 6, hand.y - 6, 13, 13, visual.dark, palette.outline);
      drawBlock(ctx, hand.x - 4, hand.y - 4, 11, 9, visual.metal, null);
      drawBlock(ctx, hand.x + 3, hand.y - 5, 8, 4, visual.edge, null);
      return;
    }
    if (visual.kind === 'scope') {
      drawBlock(ctx, hand.x - 5, hand.y - 7, 21, 8, visual.metal, palette.outline);
      drawBlock(ctx, hand.x + 10, hand.y - 8, 8, 10, visual.trim, null);
      drawBlock(ctx, hand.x + 13, hand.y - 5, 4, 5, visual.lens, null);
      return;
    }
    if (visual.kind === 'kit') {
      drawBlock(ctx, hand.x - 9, hand.y - 4, 17, 14, visual.leather, palette.outline);
      drawBlock(ctx, hand.x - 7, hand.y - 1, 12, 3, visual.metal, null);
    }
  }

  function drawAura(ctx, pose, visual) {
    if (!visual && pose.aura <= 0) return;
    const color = visual && visual.glow ? visual.glow : '#ffcf63';
    ctx.save();
    ctx.globalAlpha = visual ? 0.16 + pose.aura * 0.24 : pose.aura * 0.22;
    ctx.strokeStyle = color;
    ctx.lineWidth = visual ? 3 : 4;
    ctx.beginPath();
    ctx.ellipse(0, -17, 25 + pose.aura * 9, 39 + pose.aura * 7, 0, 0, TAU);
    ctx.stroke();
    ctx.restore();
  }

  function drawSlash(ctx, pose, visual) {
    if (pose.aura <= 0) return;
    const kind = visual && visual.kind ? visual.kind : 'sword';
    ctx.save();
    ctx.globalAlpha = 0.16 + pose.aura * 0.24;
    if (kind === 'bow') {
      ctx.fillStyle = visual.arrow || '#ffe16a';
      ctx.fillRect(39 + pose.weaponReach, -16 + pose.weaponLift, 38, 3);
      ctx.fillRect(70 + pose.weaponReach, -19 + pose.weaponLift, 7, 7);
    } else if (kind === 'wand' || kind === 'staff') {
      ctx.fillStyle = visual.glow || '#8bd7ff';
      ctx.fillRect(42 + pose.weaponReach, -28 + pose.weaponLift, 14, 14);
      ctx.fillRect(48 + pose.weaponReach, -34 + pose.weaponLift, 5, 26);
    } else {
      ctx.fillStyle = '#ffcf63';
      ctx.translate(26 + pose.weaponReach, -18 + pose.weaponLift);
      ctx.rotate(pose.weaponAngle - 0.38);
      const arc = Math.max(0.25, Number(pose.slashArc || pose.aura || 0));
      ctx.fillRect(-10, -18, 10 + arc * 4, 4);
      ctx.fillRect(-2, -13, 17 + arc * 6, 4);
      ctx.fillRect(10, -7, 22 + arc * 8, 4);
      ctx.fillRect(25, 0, 16 + arc * 5, 4);
    }
    ctx.restore();
  }

  function drawDefeated(ctx, palette, equipment) {
    const chest = equipment.chest;
    const boots = equipment.boots;
    const weapon = equipment.weapon;
    ctx.save();
    ctx.globalAlpha = 0.9;
    drawBlock(ctx, -32, 34, 22, 2, palette.shadow, null);
    drawBlock(ctx, 8, 34, 24, 2, palette.shadow, null);
    drawBlock(ctx, -23, 8, 36, 20, palette.shirt, palette.outline);
    if (chest) {
      drawBlock(ctx, -20, 10, 30, 14, chest.cloth || '#8c5a3a', null);
      drawBlock(ctx, -18, 11, 25, 4, chest.trim || '#d39a5c', null);
    }
    drawBlock(ctx, 9, 4, 22, 21, palette.skin, palette.outline);
    drawBlock(ctx, 9, 2, 22, 7, palette.hair, null);
    drawBlock(ctx, -31, 18, 24, 8, palette.pantsDark, palette.outline);
    drawBlock(ctx, -38, 23, 17, 7, boots && boots.leather ? boots.leather : palette.boot, palette.outline);
    drawBlock(ctx, -17, 26, 26, 8, palette.pants, palette.outline);
    drawBlock(ctx, -7, 31, 18, 7, boots && boots.leather ? boots.leather : palette.boot, palette.outline);
    if (weapon) {
      drawBlock(ctx, 29, 27, 43, 5, weapon.kind === 'axe' ? weapon.grip : weapon.blade, palette.outline);
    }
    ctx.restore();
  }

  function drawCharacter(ctx, actor, rig, options) {
    if (!canDraw(ctx) || !actor || !rig) return false;
    const settings = options || {};
    const state = settings.state || actor.animationState || 'idle';
    const elapsed = Math.max(0, Number(settings.elapsed) || 0);
    const equipment = settings.equipment || {};
    const visuals = {
      weapon: equipmentVisual(rig, equippedItem(equipment, 'weapon'), rig.equipmentVisuals && rig.equipmentVisuals.training_sword),
      offhand: equipmentVisual(rig, equippedItem(equipment, 'offhand')),
      chest: equipmentVisual(rig, equippedItem(equipment, 'chest')),
      boots: equipmentVisual(rig, equippedItem(equipment, 'boots')),
      ring: equipmentVisual(rig, equippedItem(equipment, 'ring')),
      head: equipmentVisual(rig, equippedItem(equipment, 'head')),
      gloves: equipmentVisual(rig, equippedItem(equipment, 'gloves'))
    };
    const pose = resolvePose(rig, state, elapsed, { weaponKind: visuals.weapon && visuals.weapon.kind });
    const palette = Object.assign({}, rig.palette || {}, settings.palette || {});
    const scale = Math.max(0.1, Number(settings.scale || rig.scale) || 1);
    const facing = (Number(actor.facing) || 1) < 0 ? -1 : 1;

    ctx.save();
    const previousSmoothing = ctx.imageSmoothingEnabled;
    ctx.imageSmoothingEnabled = false;
    ctx.translate(
      Math.round(actor.x + actor.w / 2),
      Math.round(actor.y + actor.h - (Number(rig.groundY || 39) * scale))
    );
    ctx.scale(facing * scale, scale);
    ctx.translate(pose.rootX, pose.rootY);

    if (pose.defeated) {
      drawDefeated(ctx, palette, visuals);
      ctx.imageSmoothingEnabled = previousSmoothing;
      ctx.restore();
      return true;
    }

    drawContactShadows(ctx, pose, palette);
    drawAura(ctx, pose, visuals.ring);
    const backHand = drawArm(ctx, pose, 'back', palette, visuals.gloves);
    drawLeg(ctx, pose, 'back', palette, visuals.boots);
    drawLeg(ctx, pose, 'front', palette, visuals.boots);
    drawTorso(ctx, pose, palette, visuals.chest);
    drawHead(ctx, pose, palette, visuals.head);
    drawOffhand(ctx, backHand, pose, visuals.offhand, palette);
    const frontHandPreview = limbPoints(pose.frontShoulder, pose.frontArm.upper, pose.frontArm.lower, 15, 14).hand;
    drawSlash(ctx, pose, visuals.weapon);
    drawWeapon(ctx, frontHandPreview, backHand, pose, visuals.weapon);
    drawArm(ctx, pose, 'front', palette, visuals.gloves);
    if (pose.recoil) {
      ctx.save();
      ctx.globalAlpha = 0.28;
      drawBlock(ctx, -25, -54, 7, 7, '#ffffff', null);
      drawBlock(ctx, 21, -40, 7, 7, '#ffffff', null);
      ctx.restore();
    }
    drawAura(ctx, pose, null);

    ctx.imageSmoothingEnabled = previousSmoothing;
    ctx.restore();
    return true;
  }

  const ProjectStarfallRig = Object.freeze({
    drawCharacter,
    resolvePose
  });

  global.ProjectStarfallRig = ProjectStarfallRig;

  if (typeof module === 'object' && module.exports) {
    module.exports = ProjectStarfallRig;
  }
})(typeof window !== 'undefined' ? window : globalThis);
