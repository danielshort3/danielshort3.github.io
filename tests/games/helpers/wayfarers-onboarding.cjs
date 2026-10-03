'use strict';

const assert = require('node:assert/strict');
const Core = require('../../../js/games/wayfarers-guild/core');

// These fixtures isolate established controls from first-visit teaching. The
// onboarding browser suite separately exercises the actual initial walkthrough.
function completeAreaGuides(state) {
  const selected = state.expedition.selectedArea;
  for (const guide of Core.getView(state).onboarding.guides.filter(item => item.mandatory && !item.complete)) {
    if (guide.areaId) assert(Core.act(state, { type: 'expedition-select', areaId: guide.areaId }).ok);
    assert(Core.act(state, guide.visitAction).ok);
    for (const step of guide.steps.slice(guide.progress)) {
      assert(Core.act(state, { type: 'onboarding-next', id: guide.id, stepId: step.id }).ok);
    }
  }
  assert(Core.act(state, { type: 'expedition-select', areaId: selected }).ok);
  return state;
}

function announceDiscoveries(state) {
  const notice = Core.getView(state).onboarding.notice;
  if (notice) assert(Core.act(state, notice.deferAction).ok);
  return state;
}

async function finishCurrentGuide(page) {
  await page.clock.runFor(1000);
  for (let count = 0; count < 3 && await page.locator('.wx-guide[open]').count(); count += 1) {
    await page.locator('[data-guide-next]').click();
    await page.clock.runFor(20);
  }
  assert.equal(await page.locator('.wx-guide[open]').count(), 0);
}

module.exports = { completeAreaGuides, announceDiscoveries, finishCurrentGuide };
