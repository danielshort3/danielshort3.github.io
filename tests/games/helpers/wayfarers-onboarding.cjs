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
    for (let count=0;count<40;count+=1) {
      const active=Core.getView(state).onboarding.active;
      if(!active)break;
      const action=active.practiceAction || active.inspectAction || active.action;
      assert(action,'Fixture guide has a canonical action: '+active.guideId);
      const result=Core.act(state,action);assert(result.ok,result.message);
    }
    assert(Core.getView(state).onboarding.guides.find(item=>item.id===guide.id).complete,'Fixture guide completed through canonical actions: '+guide.id);
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
  for (let count = 0; count < 40 && await page.locator('.wx-guide[open]').count(); count += 1) {
    const realTarget=page.locator('[data-guide-target]');
    if(await page.locator('.wx-guide[data-interactive="true"][open]').count())await realTarget.click();
    else await page.locator('[data-guide-next]').click();
    await page.clock.runFor(1000);
    const result=page.locator('.wx-sheet[data-kind="collection-result"][open] .wx-confirm');
    if(await result.count()) {await result.click();await page.clock.runFor(1000);}
  }
  assert.equal(await page.locator('.wx-guide[open]').count(), 0);
}

module.exports = { completeAreaGuides, announceDiscoveries, finishCurrentGuide };
