'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const hud = require(path.join(ROOT, 'js/games/project-starfall/ui/hud.js'));
const panels = require(path.join(ROOT, 'js/games/project-starfall/ui/panels.js'));
const input = require(path.join(ROOT, 'js/games/project-starfall/ui/input.js'));
const canvasWindows = require(path.join(ROOT, 'js/games/project-starfall/ui/canvas-windows.js'));

const hudBox = hud.getCanvasStatusHudBox(1280, 806, {
  settings: { video: { hudScale: 1 } },
  runtime: { hudTop: 722 },
  statusHudHeight: 84
});
const quickActions = hud.getCanvasHudQuickActions(['inventory', 'equipment', 'skills', 'worldmap'], {
  getBindableAction: (panel) => ({ panel, label: `${panel} Popup` })
});
const hudLayout = hud.getCanvasStatusHudLayout(hudBox, quickActions);
const hudFrame = hud.getCanvasStatusHudFrameMetadata(hudBox, hudLayout.frame, '#74ddff', {});
const commandGroups = hud.getCanvasMenuGroups({
  dailyLogin: { claimable: true },
  channel: { currentId: 'ch1', channels: [{ id: 'ch1', label: 'Ch. 1', current: true }] }
});
const commandFooter = hud.getCanvasMenuFooterAction();

assert.strictEqual(quickActions.length, 4, 'the compact HUD should preserve its four direct panel shortcuts');
assert.strictEqual(hudFrame.shadowRect, null, 'the status HUD should not render a full-width background slab');
assert.strictEqual(hudFrame.panels.length, 3, 'character, vitals, and shortcut groups should render as floating modules');
assert(hudFrame.panels.every((panel) => !String(panel.stroke).includes('245,207,114')),
  'floating HUD modules should use restrained cyan lines instead of ornate gold borders');
assert.deepStrictEqual(commandGroups.map((group) => group.title), ['OPERATIVE', 'FIELD SYSTEMS', 'SIGNAL LINKS', 'COMMAND'],
  'the command deck should use Starfall-native information architecture instead of generic MMO menu headings');
assert(commandGroups[1].items.some((item) => item.panel === 'worldmap' && item.label === 'Starfall Atlas'));
assert(commandGroups[1].items.some((item) => item.panel === 'daily' && item.label === 'Beacon Ready!'));
assert.strictEqual(commandGroups[2].items[0].label, 'Signal 1');
assert.strictEqual(commandFooter.label, 'Return to Observatory');
assert.strictEqual(canvasWindows.getWindowTitle('worldmap'), 'Starfall Atlas',
  'canvas windows should use the same Starfall-native vocabulary as the command deck');
assert.strictEqual(panels.getDomPanelPresentation('skills').title, 'Techniques',
  'semantic panel titles should mirror the command-deck label without changing panel IDs');
Object.keys(canvasWindows.WINDOW_TITLES).forEach((panelId) => {
  assert.strictEqual(panels.getDomPanelPresentation(panelId).title, canvasWindows.getWindowTitle(panelId),
    `${panelId} should use the canonical canvas-window title in semantic panels`);
});
const channelTarget = {
  hasAttribute: (name) => name === 'data-starfall-command-channel',
  getAttribute: (name) => name === 'data-starfall-command-channel' ? 'ch2' : null
};
assert.deepStrictEqual(panels.getPanelShellDomAction(channelTarget), {
  handled: true,
  type: 'changeChannel',
  channelId: 'ch2'
}, 'semantic signal controls should reuse the command menu interaction router');
assert(input.DOM_CLICK_TARGET_ATTRIBUTES.includes('data-starfall-command-channel'),
  'signal controls should be discoverable when a nested label receives the click');

const stableAnnouncement = hud.getHudStatusAnnouncement({
  state: { player: { classId: 'vanguard', level: 4, hp: 80 } },
  stats: { maxHp: 100 },
  classData: { name: 'Vanguard' },
  onboarding: { nextStep: { id: 'reach_verge', title: 'Reach the Verge' } }
});
const stableAnnouncementAfterDamage = hud.getHudStatusAnnouncement({
  state: { player: { classId: 'vanguard', level: 4, hp: 55 } },
  stats: { maxHp: 100 },
  classData: { name: 'Vanguard' },
  onboarding: { nextStep: { id: 'reach_verge', title: 'Reach the Verge' } }
});
const criticalAnnouncement = hud.getHudStatusAnnouncement({
  state: { player: { classId: 'vanguard', level: 4, hp: 20 } },
  stats: { maxHp: 100 },
  classData: { name: 'Vanguard' },
  onboarding: { nextStep: { id: 'reach_verge', title: 'Reach the Verge' } }
});

assert.strictEqual(stableAnnouncement.signature, stableAnnouncementAfterDamage.signature,
  'routine health changes should not repeatedly announce the entire HUD');
assert.notStrictEqual(criticalAnnouncement.signature, stableAnnouncement.signature,
  'crossing the critical-health threshold should create a dedicated status update');
assert(criticalAnnouncement.message.includes('Health critical.'), 'critical HUD status should be announced clearly');

const riftAnnouncement = hud.getHudStatusAnnouncement({
  state: { player: { classId: 'vanguard', level: 4, hp: 80 } },
  stats: { maxHp: 100 },
  classData: { name: 'Vanguard' },
  mapModifiers: { rift: { counterplay: { active: [{ id: 'break_guard', label: 'Shatter the Guard' }] } } }
});
assert(riftAnnouncement.message.includes('Rift response active: Shatter the Guard.'),
  'the dedicated status region should announce live Rift counterplay');

const hudCss = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/hud.css'), 'utf8');
const responsiveCss = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/responsive.css'), 'utf8');
const starfallPage = fs.readFileSync(path.join(ROOT, 'pages/games/project-starfall.html'), 'utf8');
const uiCode = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-ui.js'), 'utf8');

assert(hudCss.includes('.project-starfall-hud-actions'), 'semantic mobile HUD should expose the same shortcut group as the canvas HUD');
assert(responsiveCss.includes('@media (max-width: 700px)') && responsiveCss.includes('clip-path: none'),
  'the semantic HUD should become visible and legible below the canvas on narrow screens');
assert(responsiveCss.includes('min-height: 44px'), 'narrow-screen HUD controls should meet the 44px touch target');
assert(hudCss.includes('visibility: hidden') && responsiveCss.includes('visibility: visible'),
  'the clipped desktop HUD should stay out of keyboard navigation while the narrow HUD remains interactive');
assert(!starfallPage.includes('data-starfall-hud aria-live="polite"'), 'the repeatedly replaced HUD should not be one large live region');
assert(starfallPage.includes('data-starfall-hud-status role="status" aria-live="polite"'),
  'HUD milestones should use a dedicated polite status region');
assert(uiCode.includes("this.questTrackerState = { x: 0, y: 0, compact: true, userPlaced: false }"),
  'the quest tracker should start compact so the field remains visually dominant');
assert(uiCode.includes('data-starfall-open-panel="${escapeHtml(action.panel)}"'),
  'the semantic HUD shortcut buttons should reuse the existing panel interaction contract');
assert(uiCode.includes('data-starfall-command-toggle aria-expanded='),
  'the semantic HUD Menu button should reuse the existing command toggle contract');
assert(uiCode.includes('renderDomCommandDeck()') &&
  uiCode.includes('data-starfall-command-menu') &&
  uiCode.includes('aria-controls="project-starfall-command-deck"') &&
  uiCode.includes("commandDeck.toggleAttribute('inert', !interactive)"),
  'narrow screens should receive a real focus-managed DOM command deck while desktop keeps it inert');
assert(hudCss.includes('.project-starfall-command-deck-item') &&
  responsiveCss.includes('.project-starfall-command-deck:not([hidden])'),
  'the semantic command deck should expose full-size themed controls only at the narrow breakpoint');
assert(uiCode.includes('project-starfall-rift-counterplay'),
  'active Rift counterplay should remain visible beyond its activation toast');
assert(uiCode.includes('RIFT RESPONSE'), 'canvas HUD should label the active Rift response');
assert(uiCode.includes('formatRiftOperationStatus') &&
  uiCode.includes('project-starfall-rift-operation') &&
  uiCode.includes('RIFT OP'),
  'timed Rift operations should expose one consistent status formatter across semantic and canvas HUDs');
assert(uiCode.includes('remaining - Tier') &&
  uiCode.includes('weeklyRewardClaimed') &&
  uiCode.includes('riftOperationStatus.record') &&
  uiCode.includes('riftOperationStatus.reward'),
  'Rift operation surfaces should communicate time, progression, personal records, and the weekly cache');
assert(hudCss.includes('.project-starfall-rift-operation') &&
  responsiveCss.includes('.project-starfall-rift-operation'),
  'Rift operation status should retain its observatory styling and mobile layout');
assert(uiCode.includes('drawCanvasCommandDeckFrame') && uiCode.includes('rgba(5,18,31,0.97)'),
  'the canvas command deck should use a dark observatory frame rather than the cream legacy menu skin');

console.log('Project Starfall professional HUD tests passed.');
