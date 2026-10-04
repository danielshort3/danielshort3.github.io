const assert = require("node:assert/strict");
const { WHEEL_ORDER, betDefinitions, calculateSettlement, drawUniformIndex,
  createSpinPlan, sampleSpinMotion, getWheelGeometry } =
  require("../../js/games/roulette/app.js");

assert.equal(WHEEL_ORDER.length, 38);
assert.equal(new Set(WHEEL_ORDER).size, 38);
assert(WHEEL_ORDER.includes("0") && WHEEL_ORDER.includes("00"));
assert.equal(betDefinitions.size, 161);

const insideCounts = { split: 62, street: 12, trio: 3, corner: 22, "six-line": 11 };
const insideRules = { split: [2, 17], street: [3, 11], trio: [3, 11], corner: [4, 8], "six-line": [6, 5] };
Object.entries(insideCounts).forEach(([group, count]) => {
  const bets = Array.from(betDefinitions.values()).filter((bet) => bet.group === group);
  assert.equal(bets.length, count, `${group} count`);
  bets.forEach((bet) => {
    assert.equal(bet.numbers.size, insideRules[group][0], `${bet.id} coverage`);
    assert.equal(bet.payout, insideRules[group][1], `${bet.id} payout`);
    bet.numbers.forEach((pocket) => assert(WHEEL_ORDER.includes(pocket), `${bet.id}: ${pocket}`));
  });
});

[
  ["split-0-00", ["0", "00"]],
  ["trio-0-1-2", ["0", "1", "2"]],
  ["trio-0-2-00", ["0", "2", "00"]],
  ["trio-00-2-3", ["00", "2", "3"]],
  ["corner-1-2-4-5", ["1", "2", "4", "5"]],
  ["six-line-1-2-3-4-5-6", ["1", "2", "3", "4", "5", "6"]]
].forEach(([id, numbers]) => {
  assert.deepEqual(Array.from(betDefinitions.get(id).numbers), numbers, id);
});

const zeroBets = new Map([
  ["straight-0", 1],
  ["split-0-00", 2],
  ["trio-0-2-00", 3],
  ["basket-first-five", 4],
  ["outside-even", 5],
  ["straight-00", 6]
]);
const zeroResult = calculateSettlement(zeroBets, "0");
assert.equal(zeroResult.returned, (1 * 36) + (2 * 18) + (3 * 12) + (4 * 7));
assert.deepEqual(zeroResult.winningBets.map((bet) => bet.betId),
  ["straight-0", "split-0-00", "trio-0-2-00", "basket-first-five"]);
assert.equal(calculateSettlement(zeroBets, "00").returned,
  (2 * 18) + (3 * 12) + (4 * 7) + (6 * 36));
assert.equal(calculateSettlement(new Map([["outside-even", 5]]), "00").returned, 0);

const insideBets = new Map([
  ["split-1-2", 5],
  ["street-1-2-3", 5],
  ["corner-1-2-4-5", 5],
  ["six-line-1-2-3-4-5-6", 5]
]);
assert.equal(calculateSettlement(insideBets, "1").returned, 5 * (18 + 12 + 9 + 6));
assert.equal(calculateSettlement(insideBets, "7").returned, 0);

for (let sample = 0; sample < 38; sample += 1) {
  assert.equal(drawUniformIndex(38, () => sample), sample);
}
let draws = 0;
const limit = Math.floor(0x100000000 / 38) * 38;
assert.equal(drawUniformIndex(38, () => (++draws === 1 ? limit : 37)), 37);
assert.equal(draws, 2, "out-of-range 32-bit samples are rejected to avoid modulo bias");
assert.throws(() => drawUniformIndex(38, () => -1), RangeError);
assert.throws(() => drawUniformIndex(0, () => 0), RangeError);

const slice = 360 / WHEEL_ORDER.length;
const normalized = (degrees) => ((degrees % 360) + 360) % 360;
const angularError = (first, second) => Math.abs(normalized(first - second + 180) - 180);
const geometry = getWheelGeometry(276);
assert(geometry.outerRadius > geometry.pocketRadius);
assert(geometry.outerRadius < 138 && geometry.pocketRadius > 80);
assert.equal(geometry.labelRadius - geometry.pocketRadius, 8);
assert.equal(getWheelGeometry(380).labelRadius - getWheelGeometry(380).pocketRadius, 8);
assert.equal(geometry.deflectorRadius,
  geometry.pocketRadius + ((geometry.outerRadius - geometry.pocketRadius) * 0.45));

WHEEL_ORDER.forEach((pocket, index) => {
  const plan = createSpinPlan(pocket, 137, -92, {
    durationMs: 8700,
    wheelTurns: 5,
    ballTurns: 8,
    landingOffsetDeg: 0.5
  });
  const beginning = sampleSpinMotion(plan, 0);
  const orbit = sampleSpinMotion(plan, plan.lockMs * 0.4);
  const drop = sampleSpinMotion(plan, plan.lockMs * 0.78);
  const lock = sampleSpinMotion(plan, plan.lockMs);
  const finish = sampleSpinMotion(plan, plan.durationMs);

  assert.equal(beginning.phase, "launch");
  assert.equal(beginning.trackFraction, 0);
  assert.equal(orbit.phase, "orbit");
  assert.equal(orbit.trackFraction, 1);
  assert.equal(drop.phase, "drop");
  assert(drop.trackFraction > 0 && drop.trackFraction < 1);
  assert(Math.abs(drop.bouncePx) > 0.1);
  assert.equal(lock.phase, "locked");
  assert.equal(finish.phase, "locked");
  assert.equal(finish.trackFraction, 0);
  assert.equal(finish.bouncePx, 0);
  assert(orbit.wheelDeg > beginning.wheelDeg, "rotor travels clockwise");
  assert(orbit.ballDeg < beginning.ballDeg, "ball travels counterclockwise");
  assert(finish.wheelDeg > lock.wheelDeg, "rotor keeps moving after ball enters the pocket");
  assert(angularError(lock.ballDeg - lock.wheelDeg, (index + 0.5) * slice) < 0.000001,
    `${pocket} aligns at pocket lock`);
  assert(angularError(finish.ballDeg - finish.wheelDeg, (index + 0.5) * slice) < 0.000001,
    `${pocket} stays in its pocket as the rotor coasts`);
  assert.equal(Math.floor(normalized(finish.ballDeg - finish.wheelDeg) / slice), index,
    `${pocket} is the visible winning slice`);
});
assert.throws(() => createSpinPlan("37", 0, 0), RangeError);
const variantA = createSpinPlan("17", 0, 0,
  { wheelTurns: 4, ballTurns: 7, landingOffsetDeg: -130 });
const variantB = createSpinPlan("17", 0, 0,
  { wheelTurns: 6, ballTurns: 9, landingOffsetDeg: 80 });
assert.notEqual(normalized(variantA.endWheelDeg), normalized(variantB.endWheelDeg));

console.log("Roulette rules, fair selection, and wheel/ball motion geometry: passed");
