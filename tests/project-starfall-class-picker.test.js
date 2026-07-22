'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
const { ProjectStarfallUi } = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-ui.js'));

const starterCatalog = new Map((data.SHOP_ITEMS || []).map((item) => [item.id, item]));
Object.values(data.BASE_CLASSES || {}).forEach((classData) => {
  const starterId = data.STARTER_ITEMS && data.STARTER_ITEMS[classData.id] && data.STARTER_ITEMS[classData.id][0];
  const starterWeapon = starterCatalog.get(starterId);
  assert(classData.roleProfile && classData.roleProfile.primary && classData.roleProfile.specialty && classData.roleProfile.summary,
    `${classData.name} should expose its real role, specialty, and combat summary to the class picker`);
  assert(classData.resourceName && classData.weaponType && classData.description,
    `${classData.name} should expose its resource, combat type, and description to the class picker`);
  assert(starterWeapon && starterWeapon.slot === 'weapon' && starterWeapon.classId === classData.id,
    `${classData.name} should resolve a real class-compatible starter weapon`);
});
const baseClassRoles = Object.values(data.BASE_CLASSES || {}).map((classData) => classData.roleProfile.primary);
assert.strictEqual(new Set(baseClassRoles).size, baseClassRoles.length,
  'base classes should advertise distinct field roles in the creation decision');
assert(!baseClassRoles.includes('Hybrid'),
  'the creation decision should not flatten every base class into the same Hybrid role');

let renderCalls = 0;
let createCalls = 0;
let focusedClassId = '';
const ui = Object.create(ProjectStarfallUi.prototype);
Object.assign(ui, {
  characterCreateDraft: {
    active: true,
    slotId: 'slot_1',
    name: 'Nova',
    classId: '',
    lookId: data.CHARACTER_LOOKS[0].id,
    step: 'class',
    error: ''
  },
  renderClassSelect() {
    renderCalls += 1;
  },
  focusCharacterCreateClassChoice(classId) {
    focusedClassId = classId;
    return true;
  },
  createCharacterFromDraft() {
    createCalls += 1;
    return true;
  }
});

assert.strictEqual(ui.chooseCharacterCreateClass('mage'), true,
  'choosing a valid base class should update the review draft');
assert.strictEqual(ui.characterCreateDraft.classId, 'mage',
  'the selected base class should remain in the draft until confirmation');
assert.strictEqual(ui.characterCreateDraft.error, '',
  'a valid selection should clear stale class errors');
assert.strictEqual(renderCalls, 1,
  'choosing a class should rerender the review state once');
assert.strictEqual(focusedClassId, 'mage',
  'the selected class control should regain focus after the review rerender');
assert.strictEqual(createCalls, 0,
  'choosing a class must not create, save, or enter the character');

let forcedSerializeCalls = 0;
let forcedPersistCalls = 0;
const emptyDraftSlot = { slotId: 'slot_1', character: null };
const saveGuardUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(saveGuardUi, {
  engine: {
    state: { player: { classId: 'fighter' } },
    serialize() {
      forcedSerializeCalls += 1;
      return { state: { player: { classId: 'fighter' } } };
    }
  },
  snapshot: { state: { player: { classId: 'fighter' } } },
  selectedCharacterSlotId: 'slot_1',
  isCharacterSelectOpen: true,
  characterAutoSaveDirty: true,
  getCharacterSlot() {
    return emptyDraftSlot;
  },
  persistCharacterRoster() {
    forcedPersistCalls += 1;
    return true;
  }
});
assert.strictEqual(saveGuardUi.saveActiveCharacter({ force: true, silent: true }), false,
  'even a forced unload save should remain blocked while character selection is open');
assert.strictEqual(forcedSerializeCalls, 0,
  'a creation draft must not serialize the previously active engine character');
assert.strictEqual(forcedPersistCalls, 0,
  'a creation draft must not write an empty slot before explicit confirmation');
assert.strictEqual(emptyDraftSlot.character, null,
  'the selected empty slot should remain untouched throughout draft review');

assert.strictEqual(ui.confirmCharacterCreateStep(), true,
  'the class review confirmation should invoke final character creation');
assert.strictEqual(createCalls, 1,
  'final character creation should run exactly once after explicit confirmation');

ui.characterCreateDraft.classId = '';
assert.strictEqual(ui.chooseCharacterCreateClass('not_a_class'), false,
  'unknown class ids should be rejected');
assert.strictEqual(ui.characterCreateDraft.error, 'Choose a starting class.',
  'an invalid class selection should provide an accessible error');
assert.strictEqual(createCalls, 1,
  'invalid selections must not reach character creation');

let nameAdvanceCalls = 0;
ui.characterCreateDraft.step = 'name';
ui.advanceCharacterCreateName = () => {
  nameAdvanceCalls += 1;
  return true;
};
assert.strictEqual(ui.confirmCharacterCreateStep(), true,
  'the shared confirmation control should retain the name-step behavior');
assert.strictEqual(nameAdvanceCalls, 1,
  'name confirmation should advance to class selection without creating a character');

ui.characterCreateDraft = {
  active: true,
  slotId: 'slot_1',
  name: 'Nova',
  classId: 'fighter',
  lookId: data.CHARACTER_LOOKS[0].id,
  step: 'class',
  error: ''
};
const selectedMarkup = ui.renderCharacterCreateModal();
const fighter = data.BASE_CLASSES.fighter;
const fighterStarter = starterCatalog.get(data.STARTER_ITEMS.fighter[0]);
assert.strictEqual((selectedMarkup.match(/data-starfall-character-class=/g) || []).length, Object.keys(data.BASE_CLASSES).length,
  'the class rail should render every real base class exactly once');
assert(selectedMarkup.includes('aria-pressed="true"') && selectedMarkup.includes('project-starfall-character-choice-state') && selectedMarkup.includes('Selected'),
  'the chosen class should have an explicit visual and semantic selected state');
assert(selectedMarkup.includes('data-starfall-character-create-preview="fighter"') && selectedMarkup.includes('width="260" height="260"'),
  'the review should expose a substantially larger authored animation canvas');
assert(selectedMarkup.includes(fighter.roleProfile.primary) && selectedMarkup.includes(fighter.roleProfile.specialty) &&
  selectedMarkup.includes(fighter.resourceName) && selectedMarkup.includes(fighter.description) && selectedMarkup.includes(fighterStarter.name),
  'the dossier should render role, specialty, resource, description, and starter weapon from existing game data');
assert(selectedMarkup.includes('Create Nova as Fighter') && selectedMarkup.includes('data-starfall-character-create-confirm'),
  'the selected class should require a clearly labelled explicit confirmation');
assert(selectedMarkup.includes('Your character is not created until you confirm.'),
  'the class review should explain its non-persistent draft semantics');
assert(selectedMarkup.includes('data-starfall-character-create-modal') && selectedMarkup.includes('tabindex="-1"') && selectedMarkup.includes('aria-modal="true"'),
  'the class review should expose a programmatically focusable aria-modal boundary');

ui.characterCreateDraft.classId = '';
const emptyMarkup = ui.renderCharacterCreateModal();
assert(emptyMarkup.includes('Select a class to review') &&
  /data-starfall-character-create-confirm\s+disabled/.test(emptyMarkup),
  'the unselected review state should disable final creation and explain the next action');
ui.characterCreateDraft.step = 'name';
ui.characterRoster = { slots: [{ slotId: 'slot_1', index: 0, character: null }] };
const nameMarkup = ui.renderCharacterCreateModal();
assert(nameMarkup.includes('data-starfall-character-create-modal') && nameMarkup.includes('data-starfall-character-name'),
  'the name step should share the managed modal boundary and expose its primary field');
ui.characterCreateDraft.step = 'class';

const previewCanvas = {
  getAttribute(name) {
    if (name === 'data-starfall-character-create-preview') return 'archer';
    if (name === 'data-starfall-character-preview-state') return 'run';
    return '';
  },
  getContext() {
    return {};
  }
};
ui.isCharacterSelectOpen = true;
ui.elements = {
  classSelect: {
    querySelector() {
      return null;
    },
    querySelectorAll(selector) {
      assert(selector.includes('[data-starfall-character-create-preview]'),
        'the preview lifecycle should query the class-review animation canvas');
      return [previewCanvas];
    }
  }
};
const previewContexts = ui.getCharacterSelectPreviewContexts();
assert.strictEqual(previewContexts.length, 1,
  'the class review should join the existing character-select animation lifecycle');
assert.strictEqual(previewContexts[0].classData, data.BASE_CLASSES.archer,
  'the preview should resolve the selected class data');
assert.strictEqual(previewContexts[0].animation, data.BASE_CLASSES.archer.animation,
  'the preview should use the selected class authored runtime animation');
assert.strictEqual(previewContexts[0].state, 'run',
  'the large class review should use a visibly animated movement state');

let preloadCalls = 0;
let preloadedPaths = [];
let preloadLabel = '';
const preloadUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(preloadUi, {
  engine: {
    preloadAssetPaths(paths, options) {
      preloadCalls += 1;
      preloadedPaths = paths.slice();
      preloadLabel = options && options.label;
      return Promise.resolve({ complete: true, loaded: paths.length });
    }
  },
  isCharacterSelectOpen: false
});
const firstPreload = preloadUi.preloadCharacterSelectPreviewAssets();
const secondPreload = preloadUi.preloadCharacterSelectPreviewAssets();
const expectedPreviewSheets = Array.from(new Set(Object.values(data.BASE_CLASSES)
  .map((classData) => classData.animation && classData.animation.sheet)
  .filter(Boolean))).sort();
assert.strictEqual(preloadCalls, 1,
  'repeated picker renders should share one asset-manager preload request');
assert.strictEqual(firstPreload, secondPreload,
  'the picker should return the in-flight or completed preload promise for the same class sheets');
assert.deepStrictEqual(preloadedPaths, expectedPreviewSheets,
  'the picker should actively preload every distinct base-class animation sheet');
assert(preloadedPaths.length > 0 && preloadedPaths.every((assetPath) => /-sheet-v5\.png$/.test(assetPath)),
  'fresh-load previews should request the authored v5 animation sheets');
assert.strictEqual(preloadLabel, 'character-select:base-class-previews',
  'the asset-manager request should be identifiable in loading diagnostics');

function createFocusableControl(attributes) {
  return {
    attributes: Object.assign({}, attributes || {}),
    hidden: false,
    focusCalls: 0,
    focusOptions: null,
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
    },
    hasAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name);
    },
    focus(options) {
      this.focusCalls += 1;
      this.focusOptions = options || null;
    }
  };
}

const firstClassControl = createFocusableControl({ 'data-starfall-character-class': 'fighter' });
const selectedClassControl = createFocusableControl({ 'data-starfall-character-class': 'mage' });
const finalModalControl = createFocusableControl({ 'data-starfall-character-create-cancel': '' });
const nameControl = createFocusableControl({ 'data-starfall-character-name': '' });
const modalControls = [firstClassControl, selectedClassControl, finalModalControl];
const modal = {
  focusCalls: 0,
  querySelector(selector) {
    return selector === '[data-starfall-character-name]' ? nameControl : null;
  },
  querySelectorAll() {
    return modalControls;
  },
  contains(control) {
    return modalControls.includes(control) || control === nameControl;
  },
  focus() {
    this.focusCalls += 1;
  }
};
const keyboardUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(keyboardUi, {
  characterCreateDraft: { active: true, classId: '', step: 'name' },
  elements: {
    classSelect: {
      querySelector() {
        return modal;
      }
    }
  }
});
assert.strictEqual(keyboardUi.focusCharacterCreatePrimaryControl(), true,
  'a freshly rendered name step should focus its primary field');
assert.strictEqual(nameControl.focusCalls, 1,
  'the name field should receive initial modal focus');
keyboardUi.characterCreateDraft = { active: true, classId: 'mage', step: 'class' };
assert.strictEqual(keyboardUi.focusCharacterCreatePrimaryControl(), true,
  'a freshly rendered class step should focus the relevant class control');
assert.strictEqual(selectedClassControl.focusCalls, 1,
  'the selected class should receive focus after the review rerenders');

let tabPrevented = 0;
assert.strictEqual(keyboardUi.handleCharacterCreateModalKey({
  code: 'Tab',
  target: finalModalControl,
  shiftKey: false,
  preventDefault() { tabPrevented += 1; }
}, true), true, 'Tab from the last control should wrap inside the aria-modal dialog');
assert.strictEqual(firstClassControl.focusCalls, 1,
  'forward Tab should wrap to the first modal control');
assert.strictEqual(tabPrevented, 1,
  'wrapping Tab should prevent focus from escaping to the page');
assert.strictEqual(keyboardUi.handleCharacterCreateModalKey({
  code: 'Tab',
  target: firstClassControl,
  shiftKey: true,
  preventDefault() { tabPrevented += 1; }
}, true), true, 'Shift+Tab from the first control should wrap inside the aria-modal dialog');
assert.strictEqual(finalModalControl.focusCalls, 1,
  'reverse Tab should wrap to the last modal control');

let enterConfirmCalls = 0;
keyboardUi.characterCreateDraft = { active: true, classId: '', step: 'name' };
keyboardUi.confirmCharacterCreateStep = () => {
  enterConfirmCalls += 1;
  return true;
};
assert.strictEqual(keyboardUi.handleCharacterCreateModalKey({
  code: 'Enter',
  target: nameControl,
  repeat: false,
  preventDefault() {}
}, true), true, 'Enter in the name field should safely invoke the current step confirmation');
assert.strictEqual(enterConfirmCalls, 1,
  'name-field Enter should advance once without bypassing class review');
assert.strictEqual(keyboardUi.handleCharacterCreateModalKey({
  code: 'Enter',
  target: selectedClassControl,
  repeat: false,
  preventDefault() {}
}, true), false, 'Enter on a class button should retain its native select-only click behavior');
assert.strictEqual(enterConfirmCalls, 1,
  'class-button Enter must not create or confirm a character');

const detachedCreateTrigger = createFocusableControl({ 'data-starfall-character-create-open': 'slot_1' });
detachedCreateTrigger.isConnected = false;
const returnSlotControl = createFocusableControl({ 'data-starfall-character-slot': 'slot_1' });
returnSlotControl.isConnected = true;
const escapeModal = {
  contains() { return true; },
  querySelectorAll() { return [nameControl, finalModalControl]; }
};
const escapeUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(escapeUi, {
  characterCreateDraft: { active: true, slotId: 'slot_1', step: 'name', error: '' },
  characterCreateReturnFocus: { element: detachedCreateTrigger, slotId: 'slot_1' },
  characterSelectPopover: { active: false },
  elements: {
    classSelect: {
      querySelector() { return escapeModal; },
      querySelectorAll() { return [returnSlotControl]; }
    }
  },
  renderClassSelect() {},
  closeCharacterSelectPopover() { return false; }
});
let escapePrevented = 0;
assert.strictEqual(escapeUi.handleCharacterCreateModalKey({
  code: 'Escape',
  target: nameControl,
  preventDefault() { escapePrevented += 1; }
}, true), true, 'Escape should cancel the aria-modal creation flow');
assert.strictEqual(escapeUi.characterCreateDraft.active, false,
  'Escape cancellation should close the draft without creating a character');
assert.strictEqual(returnSlotControl.focusCalls, 1,
  'closing the modal should restore focus to the initiating empty slot when its create button was replaced');
assert.strictEqual(detachedCreateTrigger.focusCalls, 0,
  'focus restoration should not target a detached create trigger');
assert.strictEqual(escapePrevented, 1,
  'Escape cancellation should suppress the browser default');

const uiSource = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-ui.js'), 'utf8');
assert(uiSource.includes('this.preloadCharacterSelectPreviewAssets();') && uiSource.includes('this.focusCharacterCreatePrimaryControl();'),
  'each real picker render should start asset readiness and apply step-appropriate focus');
assert(uiSource.includes('openCharacterCreate(characterSelectAction.slotId, target)') && uiSource.includes('openCharacterCreate(createSlotId, target)'),
  'both roster click paths should preserve the initiating create trigger for focus restoration');

const css = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/character-select.css'), 'utf8');
assert(css.includes('.project-starfall-character-class-picker') &&
  css.includes('.project-starfall-character-class-review') &&
  css.includes('.project-starfall-character-create-preview'),
  'the class picker should have dedicated rail, dossier, and large-preview styling');
assert(css.includes('button.is-selected') && css.includes('inset 4px 0 0 var(--starfall-class-accent'),
  'the selected class should be materially clearer than an unselected option');
assert(css.includes('@media (max-width: 720px)') &&
  css.includes('.project-starfall-character-create-modal.is-class-step > .project-starfall-character-actions') &&
  css.includes('.project-starfall-canvas-wrap:has(.project-starfall-class-select:not([hidden]))'),
  'the two-step class review should expand the mobile game surface and adapt its confirmation controls');
assert(css.includes('button:focus-visible'),
  'class controls should retain a high-contrast keyboard focus treatment');

console.log('Project Starfall class picker tests passed.');
