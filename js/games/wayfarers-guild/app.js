(function (root) {
  'use strict';
  const core = root.WayfarersCore;
  const escapeHtml = value => String(value == null ? '' : value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const icon = (id, label) => root.WayfarersIcons ? root.WayfarersIcons.markup(id, { label: label || '' }) : '<span class="wg-icon-fallback" aria-hidden="true">◆</span>';
  const shortNames = { boots: 'Boots', preparation: 'Field kit', miners: 'Miners', forge: 'Workshop', foragers: 'Foragers', cooks: 'Cooks', scholars: 'Scholars', surveyors: 'Surveyors', mentors: 'Mentors', 'gear-tools': 'Tools', 'gear-boots': 'Boots', 'gear-instruments': 'Instruments', 'auto-work': 'Ledgers', 'auto-forge': 'Forge orders', 'efficient-smelting': 'Smelting', 'field-notes': 'Journals', 'balanced-meals': 'Provisions', 'ore-conversion': 'Alloys', 'map-survey': 'Surveying', 'auto-route': 'Dispatch', 'smart-reserve': 'Reserves', 'specialist-training': 'Training', 'frontier-compass': 'Compass', 'meal-none': 'Save supplies', 'meal-travel': 'Trail stew', 'meal-study': 'Scholar tea', 'meal-mining': 'Miner’s lunch', alloy: 'Alloy', survey: 'Survey', compass: 'Compass', artisan: 'Artisan', scholar: 'Scholar', 'banner-amber': 'Amber banner', 'banner-moon': 'Moon banner' };
  const itemName = item => item.id === 'build-forge' ? 'Forge' : shortNames[item.id] || item.label || item.name || item.id;
  const etaText = seconds => Number.isFinite(seconds) && seconds >= 0 ? seconds < 120 ? 'About ' + Math.max(1, Math.ceil(seconds)) + ' sec' : seconds < 3600 ? 'About ' + Math.ceil(seconds / 60) + ' min' : seconds < 86400 ? 'About ' + (seconds / 3600).toFixed(1) + ' hr' : 'About ' + (seconds / 86400).toFixed(1) + ' days' : '';
  function compactNumber(value) {
    const number = core.Numbers.from(value);
    if (!number.m || number.e < 3) return core.format(number);
    if (number.e >= 15) return Number(number.m.toFixed(1)) + 'e' + number.e;
    const group = Math.floor(number.e / 3);
    const scaled = number.m * Math.pow(10, number.e - group * 3);
    // Truncate rather than round a price down to a misleading next magnitude.
    return Number((Math.floor(scaled * (scaled < 10 ? 100 : scaled < 100 ? 10 : 1)) / (scaled < 10 ? 100 : scaled < 100 ? 10 : 1)).toFixed(2)) + ['', 'K', 'M', 'B', 'T'][group];
  }
  const SCENE_ART = {
    trail: '/img/wayfarers-guild/living-trail.webp?v=7a796037857e',
    room: '/img/wayfarers-guild/living-room.webp?v=d6a6aa91c6ce',
    mine: '/img/wayfarers-guild/living-mine.webp?v=bbd69124a375'
  };
  let active = null;

  function mount() {
    const element = document.querySelector('[data-wayfarers-guild]');
    if (!element || (active && active.element === element)) return;
    if (active) active.destroy();
    if (!core || !root.WayfarersStorage || !root.WayfarersScene) {
      const warning = element.querySelector('[data-storage-warning]');
      warning.hidden = false;
      warning.textContent = 'The game could not finish loading. Reload to try again. Existing saves have been kept.';
      return;
    }
    const controller = new AbortController();
    const signal = controller.signal;
    const q = selector => element.querySelector(selector);
    const qa = selector => Array.from(element.querySelectorAll(selector));
    const set = (selector, text) => {
      const node = q(selector);
      if (node && node.textContent !== String(text)) node.textContent = text;
    };
    const store = root.WayfarersStorage.createStore({ core });
    let awaitingPurchaseWallet = !!(root.WayfarersBilling && root.WayfarersPlayBilling && typeof root.WayfarersPlayBilling.postMessage === 'function');
    const loaded = store.load({ deferOffline: awaitingPurchaseWallet });
    let state = loaded.state || core.createState(Date.now());
    let view = core.getView(state);
    let expeditionUI = null;
    let tab = 'trail';
    let room = 'mine';
    let overview = false;
    let expandedDock = false;
    let journalPage = 'discoveries';
    let inspectedKey = null;
    let inspectedCurrency = 'coins';
    let highlightedKey = null;
    let teaching = null;
    let teachingTimer = null;
    let noticeTimer = null;
    let currentActions = [];
    let feedbackFind = null;
    let pendingFinds = [];
    let pendingSeenSeq = null;
    let interacted = false;
    let sound = false;
    let audio = null;
    let lastCollectedSeq = view.luck && view.luck.ledger ? view.luck.ledger.seen : 0;
    let caravanBusy = false;
    const receiptsInFlight = new Set();
    let lastReceiptRetry = 0;
    const overviewScenes = new Map();
    let dialogKind = null;
    let pendingChallenge = null;
    let pendingImport = null;
    let opener = null;
    const sheetStack = [];
    let sheetContentKey = '';
    const planChoices = new Map();
    let catalogContext = '';
    let lastSave = Date.now();
    let lastSuccessfulSave = loaded.savedAt || null;
    let saveFailure = loaded.canSave === false || loaded.ok === false ? loaded.message : null;
    let pendingCatchup = loaded.offline && loaded.offline.pendingSeconds || 0;
    let notices = [];
    let lastGoal = view.goal.title;
    let disposed = false;
    const billing = root.WayfarersBilling ? root.WayfarersBilling.createClient({ root }) : null;
    let billingSnapshot = billing ? billing.snapshot() : { native: false, available: false, configured: false, balance: 0, owned: [], products: [], message: 'Currency packs are available in the Google Play Android app.' };
    let billingBusy = false;
    const unsubscribeBilling = billing ? billing.subscribe(snapshot => {
      if (disposed) return;
      billingSnapshot = snapshot;
      advance();
      if (core.setPremiumEntitlements) core.setPremiumEntitlements(state, snapshot.owned);
      if (awaitingPurchaseWallet && (snapshot.hydrated || snapshot.freePlaySafe)) {
        awaitingPurchaseWallet = false;
        advance();
        save();
        announce(snapshot.hydrated ? 'Purchase wallet restored. Guild progress is ready.' : 'Guild progress is ready.');
      }
      if (snapshot.message) set('[data-shop-message]', snapshot.message);
      render();
    }) : () => {};
    const actionLookup = new Map();
    const descriptorLookup = new Map();
    const playPanel = q('[data-panel="play"]');
    // The world and its contextual dock share one screen. Rooms are selectors,
    // not dialogs; only deliberate inspection opens a sheet.
    q('.wg-scene-column').prepend(q('[data-overview]'));
    q('.wg-scene').appendChild(q('[data-activity]'));
    q('.wg-scene-column').appendChild(q('[data-discovery-dock]'));
    q('[data-open-finds]').setAttribute('aria-label', 'Recent finds');
    q('[data-open-caravan]').setAttribute('aria-label', 'Caravan visit and reward');
    const dockHeading = document.createElement('div');
    dockHeading.className = 'wg-dock-heading';
    dockHeading.innerHTML = '<strong data-dock-title>Trail</strong><span data-dock-hint>Coins arrive automatically</span>';
    dockHeading.appendChild(q('[data-more-upgrades]'));
    q('.wg-controls-column').prepend(dockHeading);
    const manageContent = document.createElement('div');
    manageContent.dataset.manageContent = '';
    manageContent.hidden = true;
    manageContent.innerHTML = '<div class="wg-action-list" data-all-upgrades></div>';
    ['[data-recipes-section]', '[data-routes-section]', '[data-supply-section]', '[data-development-section]', '[data-room-connections]'].forEach(selector => manageContent.appendChild(q(selector)));
    q('[data-sheet-scroll]').prepend(manageContent);
    const shopPanel = q('[data-panel="shop"]');
    const dialog = q('[data-dialog]');
    const contextHelp = [];
    // Explanations stay available beside their heading, outside the working view.
    qa('.wg-panel-intro, .wg-section > p').forEach(paragraph => {
      // Purchase status and wallet recovery instructions belong beside the transaction.
      if (paragraph.closest('[data-panel="shop"]')) return;
      const parent = paragraph.parentElement;
      if (!parent.querySelector('h2, h3')) return;
      let entry = contextHelp.find(item => item.parent === parent);
      if (!entry) {
        entry = { parent, paragraphs: [], title: parent.querySelector('h2, h3').textContent };
        contextHelp.push(entry);
        parent.classList.add('wg-has-help');
        const help = document.createElement('button');
        help.type = 'button';
        help.className = 'wg-context-help';
        help.dataset.contextHelp = String(contextHelp.length - 1);
        help.setAttribute('aria-label', 'About ' + entry.title);
        help.innerHTML = '<span aria-hidden="true">i</span>';
        parent.prepend(help);
      }
      entry.paragraphs.push(paragraph);
      paragraph.classList.add('wg-context-copy');
    });
    let activeHelp = null;
    const isRoomOpen = () => dialog.open && dialogKind === 'room';
    const destination = () => ['guild', 'crew', 'research', 'planning'].includes(tab) ? 'guild' : ['journey', 'finds'].includes(tab) ? 'journey' : 'trail';
    const presentation = () => view.presentation || { opening: view.unlocks.length === 1, primary: ['trail'], guildSections: [], journalSections: [], systems: [], introductions: [], nextUnlock: null, show: {} };
    const motionQuery = root.matchMedia('(prefers-reduced-motion: reduce)');
    let quiet = motionQuery.matches;
    try { sound = root.localStorage.getItem('wayfarers-guild-sound') === 'true'; } catch (error) {}
    try { quiet = root.localStorage.getItem('wayfarers-guild-quiet') === 'true' || quiet; } catch (error) { /* Saving the game remains available through export. */ }
    const scene = root.WayfarersScene.create(q('[data-scene]'), {
      layout: 'portrait',
      sceneArt: SCENE_ART,
      onStatus(status) {
        q('[data-scene-error]').hidden = status.state !== 'error';
      }
    });

    const rewarded = root.WayfarersRewarded ? root.WayfarersRewarded.createClient({ root }) : null;
    let rewardedSnapshot = rewarded ? rewarded.snapshot() : { native: false, available: false, configured: false, pending: false, state: 'unavailable', message: 'Rewarded ads are not configured. Your caravan will wait, or you can skip it.' };
    const unsubscribeRewarded = rewarded ? rewarded.subscribe(snapshot => {
      if (disposed) return;
      rewardedSnapshot = snapshot;
      caravanBusy = snapshot.state === 'showing';
      const receipts = snapshot.receipts || (snapshot.receipt ? [snapshot.receipt] : []);
      receipts.forEach(receipt => { applyAdReceipt(receipt).catch(() => announce('The verified caravan delivery will retry when storage is available.')); });
      if (snapshot.state === 'cancelled' && view.caravan && view.caravan.quote) { core.cancelCaravanReward(state, view.caravan.quote.offerId); save(); }
      render();
    }) : () => {};

    function announce(message) {
      if (!message) return;
      if (expeditionUI) expeditionUI.notify(message);
      set('[data-status]', message);
      q('[data-status]').dataset.visible = 'true';
      root.clearTimeout(noticeTimer);
      noticeTimer = root.setTimeout(() => { if (!disposed) q('[data-status]').dataset.visible = 'false'; }, 3600);
      notices = [String(message)].concat(notices).slice(0, 20);
    }
    function storageNotice(message) {
      const warning = q('[data-storage-warning]');
      warning.hidden = !message;
      warning.textContent = message || '';
    }
    function save() {
      if (awaitingPurchaseWallet) return { ok: false, message: 'Waiting for the purchase wallet before saving catch-up.' };
      const previousSeen = state.luck && state.luck.ledger ? state.luck.ledger.seen : null;
      if (pendingSeenSeq != null) core.act(state, { type: 'discovery-seen', seq: pendingSeenSeq });
      const result = store.save(state);
      if (!result.ok && previousSeen != null) state.luck.ledger.seen = previousSeen;
      if (result.ok) { pendingSeenSeq = null; lastSuccessfulSave = Date.now(); saveFailure = null; } else saveFailure = result.message;
      lastSave = Date.now();
      updateSaveStatus();
      storageNotice(result.ok ? '' : result.message);
      if (result.ok && pendingFinds.length) showFindFeedback();
      return result;
    }
    function advance() {
      if (awaitingPurchaseWallet) return;
      const summary = core.advanceTo(state, Date.now());
      pendingCatchup = summary && summary.pendingSeconds || 0;
      if (summary && summary.events && summary.events.length) announce(summary.events.slice(-2).join(' '));
      if (summary && summary.pendingSeconds > 1) announce('Catching up remaining guild activity. Progress earned during your absence is being processed.');
      if (summary && summary.seconds >= 60) renderReturnSummary(summary.summary, summary.seconds, summary.gained);
    }
    function openRoom(id) {
      if (!view.unlocks.includes(id)) return;
      if (room !== id) expandedDock = false;
      room = id;
      tab = 'guild';
      overview = true;
      if (dialog.open) closeDialog();
      render();
      const floor = q('[data-overview-rooms] [data-room="' + id + '"]');
      if (floor) floor.scrollIntoView({ block: 'nearest' });
    }
    function closeRoom() {
      closeDialog();
    }
    function navigate(action) {
      const target = action.tab || (action.room && action.room !== 'trail' ? 'guild' : 'trail');
      if (action.room) room = action.room;
      showTab(target);
      if (target === 'guild' && action.room) openRoom(action.room);
    }
    function perform(action) {
      if (!action) return { ok: false };
      if (action.type === 'ui' || action.type === 'navigate') { navigate(action); return { ok: true }; }
      if (awaitingPurchaseWallet) {
        const message = 'Restoring your purchase wallet before calculating offline progress. Refresh it in Settings if it has not responded.';
        announce(message);
        if (dialog.open) set('[data-dialog-notice]', message);
        return { ok: false, message };
      }
      if (caravanBusy && ['refit', 'charter', 'challenge'].includes(action.type)) { announce('Finish or close the ad before changing this expedition.'); return { ok: false }; }
      if (action.type === 'premium-paid') { billingAction('spend', action.id); return { ok: true, pending: true }; }
      if (action.type === 'challenge' && action.id && dialogKind !== 'challenge') {
        pendingChallenge = action;
        openDialog('challenge');
        return { ok: true };
      }
      const previousFocus = document.activeElement;
      advance();
      const result = core.act(state, action);
      if (result.ok && action.type === 'buy') {
        const card = q('[data-main-actions] [data-item="' + action.id + '"]');
        if (card) {
          card.classList.remove('wg-purchased');
          void card.offsetWidth;
          card.classList.add('wg-purchased');
        }
        announce((shortNames[action.id] || action.id) + ' upgraded');
      } else announce(result.message);
      if (result.ok) save();
      render();
      if (dialog.open && dialogKind === 'inspect') { if (descriptorLookup.has(inspectedKey)) renderInspect(); else backSheet(); }
      if (previousFocus && !previousFocus.isConnected) { const target = dialog.open ? q('.wg-dialog-heading [data-close-dialog]') : q('[data-goal-action]'); if (target && !target.hidden) target.focus({ preventScroll: true }); }
      return result;
    }
    function showTab(next) {
      if (dialog.open) closeDialog();
      q('[data-play-home]').after(playPanel);
      tab = next === 'journey' && ['discoveries', 'collections'].includes(journalPage) ? 'finds' : next;
      expandedDock = false;
      overview = next === 'guild';
      render();
      q('[data-stage]').scrollTop = 0;
    }
    function introductionAction(action) {
      if (!action) return;
      const section = action.journal || action.section;
      if (section && presentation().journalSections.includes(section)) journalPage = section;
      if (action.tab === 'shop') { openDialog('shop'); return; }
      if (action.tab === 'caravan') { openDialog('caravan'); return; }
      if (action.type === 'ui' || action.type === 'navigate') navigate(action);
    }
    function renderIntroduction() {
      const entries = presentation().introductions || [];
      if (expeditionUI) {
        if (entries.length) core.act(state, { type:'introduction-seen', ids:entries.map(entry => entry.id) });
        q('[data-teaching]').hidden = true;
        return;
      }
      if (!entries.length || awaitingPurchaseWallet || !q('[data-find-feedback]').hidden || pendingFinds.some(item => ['rare', 'epic', 'legendary'].includes(item.rarity))) return;
      const previous = state.introductions && JSON.parse(JSON.stringify(state.introductions));
      const result = core.act(state, { type: 'introduction-seen', ids: entries.map(item => item.id) });
      if (!result.ok) return;
      const saved = save();
      if (!saved.ok) { if (previous) state.introductions = previous; else delete state.introductions; return; }
      // A long absence earns one summary, never a queue of tutorial popups.
      const combined = entries.map(item => item.label).join(', ');
      const explanation = entries.map(item => '<section class="wg-section"><h3>' + escapeHtml(item.label) + '</h3><p>' + escapeHtml(item.requirement || '') + '</p><p>' + escapeHtml(item.effect) + '</p></section>').join('');
      if (!q('[data-return-summary]').hidden) {
        const details = q('[data-return-details]');
        let note = details.querySelector('[data-return-unlocks]');
        if (!note) { note = document.createElement('section'); note.dataset.returnUnlocks = ''; details.prepend(note); }
        setMarkup(note, '<h3>New features</h3>' + explanation);
        announce('New: ' + combined + '.');
        return;
      }
      teaching = entries[0];
      q('[data-teaching]').hidden = false;
      setMarkup(q('[data-teaching-icon]'), icon(teaching.icon || teaching.id));
      set('[data-teaching-title]', entries.length > 1 ? entries.length + ' new features' : teaching.label + ' unlocked');
      const shortEffect = {
        boots: 'Faster travel. More coins.', mine: 'Hire miners to earn ore.', forge: 'Turn ore into stronger equipment.',
        forage: 'Gather herbs for your guild.', kitchen: 'Cook supplies for your journey.', study: 'Research lasting improvements.',
        cartography: 'Choose routes and discover new realms.', hall: 'Recruit a specialist for your guild.'
      }[teaching.id];
      set('[data-teaching-copy]', shortEffect || teaching.effect);
      root.clearTimeout(teachingTimer);
      teachingTimer = root.setTimeout(() => { if (!disposed) q('[data-teaching]').hidden = true; }, 12000);
      q('[data-teaching-details]').hidden = entries.length < 2;
      setMarkup(q('[data-teaching-features]'), explanation);
      q('[data-open-teaching]').hidden = !teaching.action;
      set('[data-open-teaching]', 'Open');
      set('[data-status]', teaching.label + ' unlocked. ' + teaching.effect);
      q('[data-status]').dataset.visible = 'false';
    }
    async function billingAction(method, argument) {
      if (!billing || billingBusy) return;
      billingBusy = true;
      if (awaitingPurchaseWallet) announce('Restoring the purchase wallet before calculating guild progress.');
      set('[data-shop-message]', method === 'purchase' ? 'Opening Google Play…' : 'Checking your purchase wallet…');
      render();
      try {
        const result = await billing[method](argument);
        if (disposed) return;
        if (method !== 'refresh' && method !== 'purchase' && method !== 'backupWallet') await billing.refresh();
        if (disposed) return;
        set('[data-shop-message]', billingSnapshot.message || result && result.message || (method === 'purchase' ? 'Complete or cancel the purchase in Google Play. Pending payments award currency after verification.' : 'Purchase wallet checked.'));
      } catch (error) {
        if (!disposed) set('[data-shop-message]', error.message || 'The purchase wallet could not be checked. Try again when connected.');
      } finally {
        billingBusy = false;
        if (!disposed) render();
      }
    }
    function updateText(node, text) { if (node.textContent !== String(text)) node.textContent = text; }
    function descriptorButton(item, kind) {
      if (kind === 'premium') {
        if (item.action && item.action.type === 'premium-equip') return item.selected ? 'Equipped' : 'Equip banner';
        if (item.owned) return 'Owned';
        const cost = (item.cost || []).map(entry => core.format(entry.amount)).join(' + ');
        return 'Buy · ' + cost + ' ' + (item.paymentSource === 'paid' ? 'purchased' : 'earned') + ' Starshards';
      }
      if (item.action && item.action.type === 'relic-equip') return item.selected ? 'Equipped' : 'Equip relic';
      if (item.action && item.action.type === 'relic-hunt') return item.selected ? 'Active hunt' : 'Select hunt';
      if (item.action && item.action.type === 'kit-prepare') return 'Prepare kit';
      if (item.action && item.action.type === 'kit-use') return 'Use prepared kit';
      if (kind === 'specialists' && item.owned) return 'Recruited';
      if (kind === 'research' && item.owned) return 'Researched';
      if (item.selected) return 'Selected';
      const cost = (item.cost || []).map(entry => entry.text || core.format(entry.amount) + ' ' + entry.resource).join(' + ');
      let label = item.label || item.name || item.id;
      if (item.id === 'boots' && view.unlocks.length === 1) label = 'Better boots';
      if (item.id === 'miners' && item.level === 0) label = 'Hire miner';
      if (item.id === 'gear-tools') label = 'Mining tools';
      if (item.id === 'gear-boots') label = 'Travel boots';
      if (item.id === 'gear-instruments') label = 'Survey instruments';
      if (kind === 'research') label = 'Research';
      if (kind === 'recipes' && !cost) label = 'Choose';
      if (kind === 'routes') label = 'Travel here';
      if (kind === 'companions' && item.owned) label = 'Choose companion';
      if (kind === 'doctrines') label = 'Adopt doctrine';
      if (kind === 'automations') label = item.selected ? 'Enabled' : 'Enable';
      return label + (cost ? ' · ' + cost : '');
    }
    function costsMarkup(costs) {
      return (costs || []).map(entry => '<span class="wg-cost">' + icon(entry.resource) + '<span>' + escapeHtml(core.format(entry.amount)) + ' <span class="wg-cost-name">' + escapeHtml(entry.resource) + '</span></span></span>').join('');
    }
    function compactEffect(item, kind) {
      if (!['main', 'catalog'].includes(kind)) return item.effectText || item.effect || (item.description || '').split(/(?<=[.!?])\s/)[0];
      if (item.action && item.action.type === 'build-room') return 'Unlock ' + ((view.rooms.find(entry => entry.id === item.action.id) || {}).name || 'room');
      if (!item.action || item.action.type !== 'buy') return (item.effectText || item.description || '').split(';')[0];
      const target = { boots: 'travel', preparation: 'travel', miners: 'ore', forge: 'ore', foragers: 'herbs', cooks: 'provisions', scholars: 'knowledge', surveyors: 'maps', mentors: 'knowledge', 'gear-tools': 'ore', 'gear-boots': 'travel', 'gear-instruments': 'knowledge' }[item.id];
      if (!target) return (item.effectText || item.description || '').split(';')[0];
      // Calculate from the same engine as spending, rather than parsing rounded
      // display text or using the illustrative percentages in the concept.
      const preview = Object.assign({}, state, { upgrades: Object.assign({}, state.upgrades, { [item.id]: (state.upgrades[item.id] || 0) + 1 }) });
      if (core.setPremiumEntitlements) core.setPremiumEntitlements(preview, billingSnapshot.owned || []);
      const beforeRates = core.getRates(state), afterRates = core.getRates(preview);
      const before = target === 'travel' ? beforeRates.travel : core.Numbers.sub(beforeRates.gain[target], beforeRates.drain[target]);
      const after = target === 'travel' ? afterRates.travel : core.Numbers.sub(afterRates.gain[target], afterRates.drain[target]);
      const label = target === 'travel' ? 'travel' : ((view.resources.find(entry => entry.id === target) || {}).name || target).toLowerCase();
      if (core.Numbers.cmp(after, before) <= 0) return (item.effectText || item.description || '').split(';')[0];
      if (!before.m) return '+' + core.format(after) + ' ' + label + '/s';
      const percent = core.Numbers.toNumber(core.Numbers.mul(core.Numbers.sub(core.Numbers.div(after, before), 1), 100));
      if (!Number.isFinite(percent)) return (item.effectText || '').split(';')[0];
      const text = '+' + Number(percent.toFixed(1)) + '% ' + label;
      if (item.id !== 'boots') return text;
      const coinPercent = core.Numbers.toNumber(core.Numbers.mul(core.Numbers.sub(core.Numbers.div(afterRates.gain.coins, beforeRates.gain.coins), 1), 100));
      return text + ' · +' + Number(coinPercent.toFixed(1)) + '% coins';
    }
    function setMarkup(node, markup) {
      if (node && node.dataset.markup !== markup) { node.innerHTML = markup; node.dataset.markup = markup; }
    }
    // Keep purchase and detail controls stable while rates/costs update.
    function reconcile(selector, descriptors, kind) {
      const list = q(selector);
      if (!list) return;
      const visible = (descriptors || []).filter(item => item.visible !== false);
      const existing = new Map(Array.from(list.children).map(node => [node.dataset.item, node]));
      const keep = new Set();
      visible.forEach(item => {
        const key = kind + ':' + item.id;
        keep.add(item.id);
        let node = existing.get(item.id);
        if (!node) {
          node = document.createElement('article');
          node.className = 'wg-action';
          node.dataset.item = item.id;
          node.innerHTML = '<button type="button" class="wg-action-info"><span class="wg-item-icon"></span><span class="wg-action-heading"><strong></strong><span class="wg-action-level"></span></span><span class="wg-info-mark" aria-hidden="true">Details</span></button><p class="wg-action-effect"></p><div class="wg-action-footer"><div class="wg-action-cost"></div><button type="button" class="wg-button wg-buy"></button></div><p class="wg-action-reason"></p>';
          list.appendChild(node);
        }
        actionLookup.set(key, item.action);
        descriptorLookup.set(key, { item, kind });
        let detail = node.querySelector('.wg-action-info');
        const simple = false;
        if ((detail.tagName === 'DIV') !== simple) {
          const replacement = document.createElement(simple ? 'div' : 'button');
          replacement.className = 'wg-action-info';
          replacement.innerHTML = detail.innerHTML;
          detail.replaceWith(replacement);
          detail = replacement;
        }
        if (!simple) { detail.type = 'button'; detail.dataset.inspect = key; detail.setAttribute('aria-label', 'Details: ' + (item.label || item.name || item.id)); }
        detail.querySelector('.wg-info-mark').hidden = simple;
        updateText(detail.querySelector('.wg-info-mark'), kind === 'main' ? 'i' : 'Details');
        setMarkup(node.querySelector('.wg-item-icon'), icon(kind === 'specialists' && (item.specialistId || item.id) === 'scholar' ? 'crew-scholar' : item.icon || item.specialistId || item.id));
        updateText(node.querySelector('strong'), itemName(item));
        updateText(node.querySelector('.wg-action-level'), kind === 'caravan' ? item.selected ? 'Reward chosen' : 'Choose' : kind === 'relics' ? item.selected ? 'Equipped' : item.rarity : item.level != null ? 'Lv. ' + item.level : item.selected ? 'Active' : item.owned ? 'Owned' : item.rarity || '');
        if (kind === 'hunts') updateText(node.querySelector('.wg-action-level'), item.selected ? 'Active target' : 'Next searches');
        if (kind === 'kits') updateText(node.querySelector('.wg-action-level'), item.action && item.action.type === 'kit-use' ? 'Prepared' : '30 min boost');
        if (item.action && !['buy', 'refit-upgrade', 'legacy-upgrade', 'relic-equip', 'kit-prepare', 'kit-use', 'relic-hunt', 'caravan-select'].includes(item.action.type)) updateText(node.querySelector('.wg-action-level'), item.selected ? 'Active' : item.owned ? 'Owned' : '');
        updateText(node.querySelector('.wg-action-effect'), compactEffect(item, kind));
        setMarkup(node.querySelector('.wg-action-cost'), costsMarkup(item.cost) || (item.owned ? '<span>Owned</span>' : ''));
        const button = node.querySelector('.wg-buy');
        button.dataset.perform = key;
        button.dataset.actionType = item.action && item.action.type || '';
        button.disabled = awaitingPurchaseWallet || !!item.disabled || item.unlocked === false || item.affordable === false || (!!item.selected && kind !== 'automations');
        let label = item.selected || item.owned && !item.action ? '✓' : item.action && ['relic-equip','premium-equip'].includes(item.action.type) ? 'Equip' : kind === 'automations' ? item.selected ? 'On' : 'Off' : item.action && ['recipe','doctrine','route','companion','caravan-select'].includes(item.action.type) ? item.selected ? '✓' : 'Select' : '+';
        if (item.action && item.action.type === 'buy') label = item.id.startsWith('gear-') ? 'Craft' : 'Buy';
        if (kind === 'development' || item.action && item.action.type === 'build-room') label = 'Build';
        if (item.action && item.action.type === 'research') label = 'Learn';
        if (item.action && item.action.type === 'plan-goal') label = 'Save for';
        if (item.action && item.action.type === 'supply-plan') label = item.selected ? 'Active' : 'Select';
        if (kind === 'supply') label = item.selected ? 'Active' : 'Select';
        if (kind === 'route-preparation') label = item.selected ? 'Ready' : 'Prepare';
        if (item.action && item.action.type === 'relic-hunt') label = item.selected ? '✓' : 'Select';
        if (item.action && item.action.type === 'kit-prepare') label = 'Prepare';
        if (item.action && item.action.type === 'kit-use') label = 'Use';
        if (kind === 'automations') { button.disabled = awaitingPurchaseWallet || !!item.disabled || item.unlocked === false; button.setAttribute('aria-pressed', String(!!item.selected)); }
        if (kind === 'premium') label = item.owned ? item.action && item.action.type === 'premium-equip' ? item.selected ? '✓' : 'Equip' : '✓' : 'Buy';
        const dockAction = kind === 'main';
        if (dockAction) setMarkup(button, '<span class="wg-buy-costs">' + (item.cost || []).map(entry => '<span class="wg-cost">' + icon(entry.resource) + '<span>' + escapeHtml(compactNumber(entry.amount)) + '</span></span>').join('') + '</span><span class="wg-buy-label">' + escapeHtml(label) + '</span>');
        else updateText(button, label);
        node.querySelector('.wg-action-cost').hidden = dockAction;
        button.dataset.compactLabel = String(label.length > 1);
        button.setAttribute('aria-label', descriptorButton(item, kind) + ' — ' + (item.label || item.name || item.id));
        const reason = dockAction ? item.unlocked === false ? item.reason || 'Explore to unlock' : !item.affordable && Number.isFinite(item.etaSeconds) ? 'Ready in ' + (item.etaSeconds < 120 ? Math.max(1, Math.ceil(item.etaSeconds)) + 's' : Math.ceil(item.etaSeconds / 60) + 'm') : !item.affordable ? item.shortageText || item.reason || 'More materials needed' : '' : kind === 'premium' && !item.owned ? (item.paymentSource === 'paid' ? 'Uses purchased Starshards' : 'Uses earned Starshards') : item.shortageText ? item.shortageText + (etaText(item.etaSeconds) ? ' · ' + etaText(item.etaSeconds) + ' at current rates' : '') : item.unlocked === false || item.disabled ? item.reason || 'Not available yet' : '';
        updateText(node.querySelector('.wg-action-reason'), kind === 'catalog' ? item.affordable ? 'Ready' : item.unlocked === false ? item.reason || 'Not unlocked yet' : etaText(item.etaSeconds) || 'More materials needed' : reason || (dockAction ? 'Ready' : ''));
        node.dataset.ready = String(!button.disabled);
        const costProgress = (item.cost || []).map(entry => {
          const resource = view.resources.find(value => value.id === entry.resource);
          return resource ? Math.min(1, core.Numbers.toNumber(core.Numbers.div(resource.amount || state.resources[entry.resource] || 0, entry.amount))) : 0;
        });
        node.style.setProperty('--purchase-progress', Math.round(100 * (costProgress.length ? Math.min(...costProgress) : 1)) + '%');
        node.dataset.selected = String(!!item.selected);
        node.dataset.owned = String(!!item.owned);
        node.dataset.locked = String(item.unlocked === false);
        node.dataset.rarity = item.rarity || '';
        node.dataset.highlighted = String(highlightedKey === key);
      });
      existing.forEach((node, id) => { if (!keep.has(id)) { node.remove(); actionLookup.delete(kind + ':' + id); descriptorLookup.delete(kind + ':' + id); } });
      list.hidden = !visible.length;
    }
    function renderWallet() {
      const unlocked = new Set(view.unlocks);
      let preferred = ['coins'];
      if (tab === 'guild') {
        preferred = { mine: ['coins', 'ore'], forge: ['ore', 'herbs'], forage: ['coins', 'herbs'], kitchen: ['herbs', 'provisions'], study: ['coins', 'knowledge'], cartography: ['maps', 'knowledge'], hall: ['knowledge', 'notes'] }[room] || ['coins', 'ore'];
      } else if (tab === 'research') preferred = ['knowledge', 'herbs'];
      else if (tab === 'crew') preferred = ['coins', 'knowledge'];
      else if (tab === 'journey') preferred = ['notes', 'crests'];
      else if (tab === 'shop') preferred = ['starshards'];
      else if (unlocked.has('kitchen')) preferred.push('provisions');
      const resources = view.resources.filter(item => item.visible !== false && preferred.includes(item.id));
      const wallet = q('[data-wallet]');
      wallet.dataset.count = String(resources.length);
      const walletRates = core.getRates(state);
      const existing = new Map(Array.from(wallet.children).map(node => [node.dataset.resource, node]));
      resources.forEach(item => {
        let node = existing.get(item.id);
        if (!node) {
          node = document.createElement('button');
          node.type = 'button';
          node.dataset.walletResource = item.id;
          node.className = 'wg-resource';
          node.dataset.resource = item.id;
          node.innerHTML = '<span class="wg-resource-icon"></span><strong></strong><span class="wg-resource-name"></span><small class="wg-resource-rate"></small>';
          wallet.appendChild(node);
        }
        updateText(node.querySelector('strong'), compactNumber(state.resources[item.id]));
        setMarkup(node.querySelector('.wg-resource-icon'), icon(item.id));
        updateText(node.querySelector('.wg-resource-name'), item.id === 'starshards' ? 'earned Starshards' : item.name.toLowerCase());
        const gain = walletRates.gain[item.id] || 0, drain = walletRates.drain[item.id] || 0;
        const declining = core.Numbers.cmp(gain, drain) < 0;
        const net = declining ? core.Numbers.sub(drain, gain) : core.Numbers.sub(gain, drain);
        updateText(node.querySelector('.wg-resource-rate'), (declining ? '−' : '+') + compactNumber(net) + '/s');
        node.setAttribute('aria-label', item.formatted + ' ' + item.name + '. View resource balances');
        node.title = item.rateFormatted || core.format(item.rate || 0) + ' per second';
      });
      existing.forEach((node, id) => { if (!resources.some(item => item.id === id)) node.remove(); });
    }
    function renderRooms() {
      const selector = q('[data-room-selector]');
      const unlockedRooms = view.rooms.filter(item => item.unlocked && item.id !== 'trail');
      const existing = new Map(Array.from(selector.children).filter(node => node.dataset.room).map(node => [node.dataset.room, node]));
      if (!unlockedRooms.some(item => item.id === room)) room = unlockedRooms.length ? unlockedRooms[0].id : 'trail';
      unlockedRooms.forEach(item => {
        let button = existing.get(item.id);
        if (!button) {
          button = document.createElement('button');
          button.type = 'button';
          button.dataset.room = item.id;
          selector.appendChild(button);
        }
        setMarkup(button, icon(item.id) + '<span>' + escapeHtml(item.name) + '</span>');
        button.setAttribute('aria-pressed', String(room === item.id));
        button.title = item.description;
      });
      existing.forEach((node, id) => { if (!unlockedRooms.some(item => item.id === id)) node.remove(); });
      selector.hidden = true;
      let overviewButton = selector.querySelector('[data-show-overview]');
      if (!overviewButton && unlockedRooms.length >= 2) {
        overviewButton = document.createElement('button');
        overviewButton.type = 'button';
        overviewButton.dataset.showOverview = '';
        overviewButton.textContent = 'Overview';
        selector.prepend(overviewButton);
      }
      if (overviewButton) overviewButton.hidden = unlockedRooms.length < 2;
    }
    function renderOverview() {
      q('[data-overview]').hidden = tab !== 'guild';
      if (tab !== 'guild') return;
      const list = q('[data-overview-rooms]');
      Array.from(list.children).forEach(tile => {
        if (!view.unlocks.includes(tile.dataset.room)) {
          overviewScenes.get(tile.dataset.room).destroy();
          overviewScenes.delete(tile.dataset.room);
          tile.remove();
        }
      });
      view.rooms.filter(item => item.unlocked && item.id !== 'trail').slice().reverse().forEach((item, index) => {
        let tile = list.querySelector('[data-room="' + item.id + '"]');
        if (!tile) {
          tile = document.createElement('button');
          tile.type = 'button';
          tile.className = 'wg-room-tile wg-guild-floor';
          tile.dataset.room = item.id;
          tile.innerHTML = '<canvas width="288" height="112" aria-hidden="true"></canvas><strong></strong><small></small><span class="wg-floor-flow" aria-hidden="true"></span>';
          list.appendChild(tile);
          overviewScenes.set(item.id, root.WayfarersScene.create(tile.querySelector('canvas'), { layout: 'room', sceneArt: SCENE_ART }));
        }
        if (list.children[index] !== tile) list.insertBefore(tile, list.children[index] || null);
        tile.setAttribute('aria-pressed', String(room === item.id));
        tile.setAttribute('aria-label', 'Select ' + item.name + '. ' + item.description);
        updateText(tile.querySelector('strong'), item.name);
        const output = view.resources.find(resource => ({mine:'ore',forage:'herbs',kitchen:'provisions',study:'knowledge',cartography:'maps'}[item.id]) === resource.id);
        const outputText = output ? output.rateFormatted + ' ' + output.name.toLowerCase() : item.id === 'forge' ? 'Gear Lv ' + Math.max(state.upgrades['gear-tools'], state.upgrades['gear-boots'], state.upgrades['gear-instruments']) : 'Mastery ' + item.mastery;
        setMarkup(tile.querySelector('small'), icon(output ? output.id : item.id === 'forge' ? 'equipment' : item.id) + '<span>' + escapeHtml(outputText) + '</span>');
        setMarkup(tile.querySelector('.wg-floor-flow'), item.id === 'mine' && view.unlocks.includes('forge') ? '<span>↑</span>' + icon('ore') : '');
        overviewScenes.get(item.id).update({ room: item.id, realm: view.progression.realm, workers: 1, companion: null, progress: 0, reducedMotion: quiet || motionQuery.matches });
      });
      q('[data-overview-supplies]').hidden = !view.unlocks.includes('kitchen');
      q('[data-overview-study]').hidden = !view.unlocks.includes('study');
      q('[data-profession-flow]').hidden = true;
      q('[data-room-connections]').hidden = !view.unlocks.includes('forge');

    }
    function renderAssignments() {
      const list = q('[data-assignments]');
      const owned = (view.specialists || []).filter(item => item.owned && item.slot === 0).map(item => ({ id: item.specialistId || item.action.id, label: (core.Content.SPECIALISTS.find(def => def.id === item.action.id) || {}).name || item.label }));
      if (!owned.length) { list.hidden = true; return; }
      list.hidden = false;
      const slots = view.progression.specialists || [null, null];
      for (let slot = 0; slot < 2; slot += 1) {
        let select = list.querySelector('[data-specialist-slot="' + slot + '"]');
        if (!select) {
          const label = document.createElement('label');
          label.innerHTML = '<span class="wg-assignment-portrait" data-slot-portrait="' + slot + '"></span><span>Slot ' + (slot + 1) + '</span>';
          select = document.createElement('select');
          select.dataset.specialistSlot = String(slot);
          label.appendChild(select);
          list.appendChild(label);
        }
        const signature = owned.map(item => item.id).join(',');
        if (select.dataset.options !== signature) {
          select.innerHTML = '<option value="">No specialist</option>' + owned.map(item => '<option value="' + escapeHtml(item.id) + '">' + escapeHtml(item.label) + '</option>').join('');
          select.dataset.options = signature;
        }
        const assignment = Array.isArray(slots) ? slots[slot] : null;
        if (document.activeElement !== select) select.value = assignment || '';
        setMarkup(q('[data-slot-portrait="' + slot + '"]'), icon(assignment || 'crew'));
        select.disabled = awaitingPurchaseWallet;
      }
    }
    function renderRecord() {
      const p = view.progression;
      const rows = [['Farthest completed route', String(Math.max(0, p.highestRoute + 1))], ['Distance this expedition', view.stats.runDistance], ['Travel speed', view.stats.travelRate]];
      if (p.frontier > 0) rows.push(['Frontier chapter', String(p.frontier)]);
      if (p.refits > 0) rows.push(['Expedition refits', String(p.refits)]);
      if (p.charters > 0) rows.push(['Guild charters', String(p.charters)], ['Distance this charter', view.stats.chapterDistance]);
      if (p.collections && (Array.isArray(p.collections) ? p.collections.length : p.collections)) rows.push(['Landmark collections', Array.isArray(p.collections) ? p.collections.length : p.collections]);
      if (Array.isArray(p.collections)) p.collections.forEach(item => rows.push([item.name, item.description]));
      const mastery = p.mastery || {};
      Object.keys(mastery).forEach(id => {
        if (!view.unlocks.includes(id)) return;
        const definition = (core.Content.ROOMS || []).find(item => item.id === id);
        const value = mastery[id];
        rows.push([(definition ? definition.profession : id) + ' mastery', typeof value === 'object' ? core.format(value) : String(value)]);
      });
      const record = q('[data-record]');
      const signature = JSON.stringify(rows);
      if (record.dataset.signature !== signature) {
        record.innerHTML = rows.map(row => '<div class="wg-record-row"><span>' + escapeHtml(row[0]) + '</span><strong>' + escapeHtml(row[1]) + '</strong></div>').join('');
        record.dataset.signature = signature;
      }
    }
    function renderShop() {
      const premium = view.premium;
      const shopButton = q('[data-open="shop"]');
      shopButton.hidden = !(awaitingPurchaseWallet || presentation().show.shop || billingSnapshot.balance > 0 || billingSnapshot.owned.length || billingSnapshot.pending || billingSnapshot.debt > 0);
      if (!premium) return;
      if (billing && billing.setCatalogContext) {
        const context = { owned: state.premium.owned.slice(), earnedBalance: core.Numbers.toNumber(premium.balance) };
        const signature = JSON.stringify(context);
        if (catalogContext !== signature) { catalogContext = signature; billing.setCatalogContext(context); billingSnapshot = billing.snapshot(); }
      }
      shopButton.title = 'Starshard shop · ' + premium.balanceText + ' earned';
      q('[data-game]').dataset.banner = premium.equipped || '';
      set('[data-earned-shards]', premium.balanceText || core.format(premium.balance));
      set('[data-paid-shards]', billingSnapshot.balance);
      q('[data-wallet-debt]').hidden = !billingSnapshot.debt;
      set('[data-wallet-debt]', billingSnapshot.debt ? 'A refunded purchase left a ' + billingSnapshot.debt + '-Starshard deficit. Purchased keepsakes are inactive until the purchase wallet covers it. Earned keepsakes remain active.' : '');
      set('[data-drop-description]', premium.dropDescription);
      q('[data-paid-wallet]').hidden = !billingSnapshot.native;
      q('[data-billing-controls]').hidden = !billingSnapshot.native;
      q('[data-paid-save-note]').hidden = !billingSnapshot.native;
      set('[data-billing-status]', billingSnapshot.message);
      const items = premium.items.map(item => {
        const next = Object.assign({}, item);
        const amount = (item.cost || []).find(entry => entry.resource === 'starshards');
        const cost = amount ? core.Numbers.toNumber(amount.amount) : 0;
        if (!item.owned && !item.affordable && billingSnapshot.native && billingSnapshot.configured && billingSnapshot.balance >= cost && cost > 0) {
          next.paymentSource = 'paid';
          next.action = { type: 'premium-paid', id: item.id };
          next.affordable = true;
          next.disabled = billingBusy;
          next.reason = 'Uses purchased Starshards. This keepsake stays in your purchase wallet.';
        }
        if (billingBusy) next.disabled = true;
        return next;
      });
      reconcile('[data-premium-items]', items, 'premium');
      const packs = q('[data-premium-packs]');
      const budget = billingSnapshot.budget || (root.WayfarersBilling.getPurchaseBudget ? root.WayfarersBilling.getPurchaseBudget({ owned: Array.from(new Set(state.premium.owned.concat(billingSnapshot.catalogOwned || billingSnapshot.owned || []))), earnedBalance: core.Numbers.toNumber(premium.balance), paidBalance: billingSnapshot.balance, debt: billingSnapshot.debt || 0 }) : null);
      set('[data-shop-budget]', budget ? 'Remaining keepsake prices: ' + budget.remainingCost + ' Starshards. ' + (budget.reason || '') : 'Each keepsake is purchased wholly with earned or purchased Starshards.');
      Array.from(packs.children).forEach(node => { if (!billingSnapshot.products.some(item => item.productId === node.dataset.product)) node.remove(); });
      billingSnapshot.products.forEach(product => {
        let node = Array.from(packs.children).find(entry => entry.dataset.product === product.productId);
        if (!node) {
          node = document.createElement('article');
          node.className = 'wg-pack';
          node.dataset.product = product.productId;
          node.innerHTML = '<strong></strong><span></span><button type="button" class="wg-button"></button>';
          packs.appendChild(node);
        }
        updateText(node.querySelector('strong'), product.shards + ' Starshards');
        updateText(node.querySelector('span'), billingSnapshot.pending ? 'Waiting for a pending Google Play purchase before another checkout.' : product.blockedReason || product.title);
        const button = node.querySelector('button');
        button.dataset.buyPack = product.productId;
        updateText(button, product.price ? 'Buy · ' + product.price : billingSnapshot.native ? 'Unavailable from Google Play' : 'Google Play app');
        button.disabled = billingBusy || billingSnapshot.pending || !billingSnapshot.available || !billingSnapshot.configured || !product.available || product.budgetAvailable === false;
      });
      qa('[data-billing-controls] button').forEach(button => { button.disabled = billingBusy; });
    }
    function renderJournal() {
      const sections = [{ id: 'discoveries', name: 'Finds', icon: 'relic' }, { id: 'collections', name: 'Collections', icon: 'collection' }, { id: 'renewals', name: 'Renewal', icon: 'refit' }, { id: 'challenges', name: 'Contracts', icon: 'challenge' }, { id: 'record', name: 'Record', icon: 'journal' }].filter(item => presentation().journalSections.includes(item.id));
      if (!sections.some(item => item.id === journalPage)) journalPage = sections.length ? sections[0].id : 'discoveries';
      const list = q('[data-journal-nav]');
      const signature = sections.map(item => item.id).join(',');
      if (list.dataset.signature !== signature) { list.innerHTML = sections.map(item => '<button type="button" data-journal="' + item.id + '">' + icon(item.icon) + '<span>' + item.name + '</span></button>').join(''); list.dataset.signature = signature; }
      qa('[data-journal]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.journal === journalPage)));
      qa('[data-journal-page]').forEach(panel => { panel.hidden = panel.dataset.journalPage !== journalPage; });
      list.hidden = destination() !== 'journey' || sections.length < 2;
      q('[data-guild-nav]').hidden = destination() !== 'guild' || presentation().guildSections.length < 2;
      qa('[data-guild-view]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.guildView === tab)));
    }
    function updateSaveStatus() {
      set('[data-save-label]', saveFailure ? 'Autosave paused. ' + saveFailure : lastSuccessfulSave ? 'Last successful save: ' + new Date(lastSuccessfulSave).toLocaleString() : 'No successful local save yet. Export a backup if storage is unavailable.');
      set('[data-pending-status]', awaitingPurchaseWallet ? 'Offline progress is waiting for your purchase wallet.' : pendingCatchup > 1 ? Math.ceil(pendingCatchup / 60) + ' minutes of offline activity remain to process.' : rewardedSnapshot.pending || receiptsInFlight.size ? 'A caravan reward is waiting for verification or a successful local save. Ordinary play continues.' : 'Offline progress is up to date.');
    }
    function renderDevelopment() {
      const development = view.development;
      const projects = development ? (development.projects || []).concat(development.chapter && development.chapter.projects || []) : [];
      q('[data-development-section]').hidden = !projects.some(item => item.visible !== false);
      set('[data-development-title]', 'Guild projects');
      set('[data-development-description]', development ? (development.chapter && development.chapter.projects.some(item => item.visible !== false) ? development.chapter.condition : 'Build a new room with the materials your guild produces.') : '');
      reconcile('[data-development-projects]', projects, 'development');
    }
    function renderPlanning() {
      const planning = view.planning;
      if (!planning) return;
      q('[data-plan-editor]').hidden = !planning.unlocked;
      set('[data-plan-status]', planning.blockedReason || (planning.unlocked ? 'Your plan runs while you are away. Resource reserves protect the purchases you choose.' : 'Research guild planning to set protected goals and repeat purchases.'));
      const offeredCapabilities = (planning.capabilities || []).filter(item => item.visible !== false && item.unlocked !== false && !item.owned);
      reconcile('[data-plan-capabilities]', offeredCapabilities.slice(0, 1), 'plan-capabilities');
      q('[data-other-capabilities]').hidden = offeredCapabilities.length < 2;
      reconcile('[data-plan-capability-options]', offeredCapabilities.slice(1), 'plan-capability-options');
      if (!planning.unlocked) return;
      planChoices.clear();
      const descriptors = (view.actions || []).concat(view.research || [], view.development && view.development.projects || [], view.development && view.development.chapter && view.development.chapter.projects || []);
      descriptors.filter(item => item.visible !== false && item.action && ['buy','research','project'].includes(item.action.type) && !item.owned).forEach(item => planChoices.set(item.action.type + ':' + item.action.id, { type: item.action.type, id: item.action.id }));
      const options = '<option value="">Choose a purchase</option>' + Array.from(planChoices.keys()).map(key => {
        const item = descriptors.find(entry => entry.action && entry.action.type + ':' + entry.action.id === key);
        return '<option value="' + escapeHtml(key) + '">' + escapeHtml(item.label || item.name || item.id) + '</option>';
      }).join('');
      ['[data-plan-goal]', '[data-plan-queue-choice]'].forEach(selector => { const select = q(selector); if (select.dataset.options !== options) { const previous = select.value; select.innerHTML = options; select.dataset.options = options; if (planChoices.has(previous)) select.value = previous; } select.disabled = awaitingPurchaseWallet; });
      const capabilities = planning.capabilities || [];
      const enabled = id => capabilities.some(item => item.id === id && item.owned);
      q('[data-plan-queue-section]').hidden = !enabled('purchase-queue');
      q('[data-plan-kit-section]').hidden = !enabled('kit-plan');
      q('[data-plan-preparation-section]').hidden = !enabled('dispatch-preparation');
      q('[data-plan-loadouts-section]').hidden = !enabled('loadouts');
      q('[data-plan-equipment]').hidden = !(view.automations || []).some(item => item.id === 'equipment' && item.unlocked !== false);
      q('[data-plan-audit-section]').hidden = !(planning.audit || []).length;
      [['purchase-queue', 'queue'], ['kit-plan', 'kit'], ['dispatch-preparation', 'preparation'], ['loadouts', 'loadouts']].forEach(([id, section]) => {
        const holder = q('[data-plan-' + section + '-section]');
        const item = capabilities.find(entry => entry.id === id);
        let copy = holder.querySelector('[data-capability-help]');
        if (!copy) { copy = document.createElement('small'); copy.dataset.capabilityHelp = ''; holder.appendChild(copy); }
        copy.textContent = item && (item.effectText || item.description) || '';
      });
      q('[data-plan-queue-add]').disabled = awaitingPurchaseWallet || !enabled('purchase-queue');
      q('[data-plan-queue-choice]').disabled = awaitingPurchaseWallet || !enabled('purchase-queue');
      q('[data-plan-kit]').disabled = awaitingPurchaseWallet || !enabled('kit-plan');
      q('[data-plan-preparation]').disabled = awaitingPurchaseWallet || !enabled('dispatch-preparation');
      if (document.activeElement !== q('[data-plan-preparation]')) q('[data-plan-preparation]').value = planning.preparation || 'off';
      qa('[data-loadout-save]').forEach(button => { button.disabled = awaitingPurchaseWallet || !enabled('loadouts'); });
      const goal = q('[data-plan-goal]');
      if (document.activeElement !== goal) goal.value = planning.goal ? planning.goal.type + ':' + planning.goal.id : '';
      qa('[data-plan-priority]').forEach(select => { if (document.activeElement !== select) select.value = planning.priorities[select.dataset.planPriority]; select.disabled = awaitingPurchaseWallet; });
      if (document.activeElement !== q('[data-plan-kit]')) q('[data-plan-kit]').value = planning.kit;
      const reserves = q('[data-plan-reserves]');
      const introducedResources = new Set(view.resources.filter(item => item.visible !== false).map(item => item.id));
      Array.from(reserves.children).forEach(label => { const input = label.querySelector('input'); if (input && !introducedResources.has(input.dataset.planReserve)) label.remove(); });
      (planning.reserves || []).filter(item => introducedResources.has(item.id)).forEach(item => {
        let input = reserves.querySelector('[data-plan-reserve="' + item.id + '"]');
        if (!input) { const label = document.createElement('label'); label.innerHTML = '<span>' + escapeHtml(item.name) + '</span><input type="text" inputmode="decimal" data-plan-reserve="' + escapeHtml(item.id) + '" aria-label="Minimum ' + escapeHtml(item.name) + ' reserve">'; reserves.appendChild(label); input = label.querySelector('input'); }
        if (document.activeElement !== input) input.value = typeof item.value === 'object' ? item.value.e >= -6 && item.value.e < 15 ? String(core.Numbers.toNumber(item.value)) : item.value.m + 'e' + item.value.e : item.value == null ? '0' : String(item.value);
        input.disabled = awaitingPurchaseWallet;
      });
      setMarkup(q('[data-plan-queue]'), (planning.queue || []).map((item,index) => '<li><span>' + escapeHtml(item.name || item.id) + '</span><button type="button" class="wg-text-button" data-plan-remove="' + index + '" aria-label="Remove ' + escapeHtml(item.name || item.id) + '">Remove</button></li>').join('') || '<li>No queued purchases.</li>');
      const loadouts = q('[data-plan-loadouts]');
      for (let id = 0; id < 3; id += 1) {
        const saved = (planning.loadouts || []).find(item => Number(item.id) === id);
        let row = loadouts.querySelector('[data-loadout="' + id + '"]');
        if (!row) { row = document.createElement('div'); row.dataset.loadout = id; row.className = 'wg-loadout'; row.innerHTML = '<label>Plan ' + (id + 1) + '<input maxlength="40" data-loadout-name="' + id + '"></label><div><button class="wg-button" type="button" data-loadout-save="' + id + '">Save</button><button class="wg-text-button" type="button" data-loadout-use="' + id + '">Use</button><button class="wg-text-button" type="button" data-loadout-delete="' + id + '">Delete</button></div>'; loadouts.appendChild(row); }
        const input = row.querySelector('input'); if (document.activeElement !== input && row.dataset.saved !== (saved && saved.name || '')) input.value = saved && saved.name || '';
        row.dataset.saved = saved && saved.name || '';
        row.querySelector('[data-loadout-save]').disabled = !enabled('loadouts') || awaitingPurchaseWallet;
        row.querySelector('[data-loadout-use]').disabled = !saved || awaitingPurchaseWallet;
        row.querySelector('[data-loadout-delete]').disabled = !saved || awaitingPurchaseWallet;
      }
      setMarkup(q('[data-plan-audit]'), (planning.audit || []).slice(-8).reverse().map(item => '<li>' + escapeHtml(item.message) + '</li>').join('') || '<li>No automated purchases yet.</li>');
    }
    function renderReturnSummary(summary, seconds, gains) {
      const display = value => typeof value === 'string' ? value : value && (value.message || value.label || value.name || value.title) || '';
      const data = summary || {};
      const completed = (data.completed || []).map(display).filter(Boolean);
      const changes = (data.changes || []).map(display).filter(Boolean);
      const next = (data.nextChoices || []).map(display).filter(Boolean);
      const resources = Object.keys(gains || {}).filter(id => core.Numbers.cmp(gains[id], 0) > 0).slice(0, 4).map(id => '+' + core.format(gains[id]) + ' ' + id);
      set('[data-return-main]', (resources[0] || 'Your guild kept working') + ' · ' + (seconds >= 3600 ? (seconds / 3600).toFixed(1) + 'h' : Math.floor(seconds / 60) + 'm') + ' away');
      setMarkup(q('[data-return-details]'), ['<p>' + escapeHtml(resources.join(' · ')) + '</p>', completed.length ? '<h3>Completed</h3><ul>' + completed.slice(0, 4).map(text => '<li>' + escapeHtml(text) + '</li>').join('') + '</ul>' : '', changes.length ? '<h3>Changed</h3><ul>' + changes.slice(0, 4).map(text => '<li>' + escapeHtml(text) + '</li>').join('') + '</ul>' : '', data.blockedReason ? '<h3>What limited the plan</h3><p>' + escapeHtml(data.blockedReason) + '</p>' : '', next.length ? '<h3>Useful next choices</h3><ul>' + next.slice(0, 3).map(text => '<li>' + escapeHtml(text) + '</li>').join('') + '</ul>' : '<p>Review your next objective and the guild’s current plan.</p>'].join(''));
      q('[data-return-summary]').hidden = false;
    }
    function renderInspect() {
      const entry = descriptorLookup.get(inspectedKey);
      if (!entry) return;
      const item = entry.item;
      set('[data-dialog-title]', item.label || item.name || item.id);
      const body = q('[data-dialog-body]');
      if (body.dataset.inspect !== inspectedKey) {
        body.innerHTML = '<div class="wg-inspect-icon"></div><p class="wg-inspect-effect" data-inspect-effect></p><p data-inspect-level></p><div class="wg-inspect-cost" data-inspect-cost></div><p data-inspect-reason></p><p class="wg-inspect-comparison" data-inspect-comparison></p>';
        body.dataset.inspect = inspectedKey;
      }
      setMarkup(body.querySelector('.wg-inspect-icon'), icon(item.icon || item.specialistId || item.id));
      set('[data-inspect-effect]', item.description || item.effectText || item.effect || '');
      set('[data-inspect-level]', item.action && ['buy', 'refit-upgrade', 'legacy-upgrade'].includes(item.action.type) ? 'Current level: ' + item.level : item.rarity || '');
      setMarkup(q('[data-inspect-cost]'), costsMarkup(item.cost));
      set('[data-inspect-reason]', item.shortageText || item.reason || '');
      set('[data-inspect-comparison]', item.comparison && item.comparison.text || item.effect || '');
      setSheetFooter({ hook: 'data-perform', value: inspectedKey, label: descriptorButton(item, entry.kind), disabled: awaitingPurchaseWallet || item.disabled || item.unlocked === false || item.affordable === false || item.selected && entry.kind !== 'automations' }, { hook: sheetStack.length ? 'data-sheet-back' : 'data-close-dialog', label: sheetStack.length ? 'Back' : 'Keep exploring' });
    }
    function guideGoal() {
      const action = view.goal.action;
      if (!action) { announce(view.goal.description || 'Keep traveling. Your adventurer is working automatically.'); return; }
      let preferredKey = null;
      let node = null;
      if (action.type === 'ui' || action.type === 'navigate') {
        if (action.tab === 'journey') journalPage = 'renewals';
        navigate(action);
        if (action.room === 'mine' && !view.unlocks.includes('forge')) preferredKey = 'main:build-forge';
        else if (action.room === 'forge') preferredKey = 'main:' + (state.upgrades['gear-tools'] < 1 ? 'gear-tools' : 'gear-boots');
        else if (action.tab === 'research') preferredKey = 'research:auto-forge';
        else if (action.section === 'projects') node = qa('[data-development-projects] .wg-buy:not(:disabled),[data-development-projects] [data-inspect^="development:chapter-"]').find(button => button.getClientRects().length);
        else if (action.tab === 'journey') node = q('[data-open="' + (view.progression.refits > 0 ? 'charter' : 'refit') + '"]');
      }
      else if (action.type === 'refit' || action.type === 'charter') { openDialog(action.type); return; }
      else {
        const definition = (core.Content.UPGRADES || []).find(item => item.id === action.id);
        if (definition && definition.room !== 'trail') openRoom(definition.room);
        else if (action.type === 'research') showTab('research');
        else if (action.type === 'automation') showTab('planning');
        else if (action.type === 'plan-goal') { showTab('planning'); node = q('[data-plan-goal]'); }
        else if (action.type === 'project' || action.type === 'build-room') showTab('guild');
        else showTab('trail');
      }
      if (['route', 'recipe', 'supply-plan', 'route-preparation'].includes(action.type)) openDialog('manage');
      if (action.type === 'route') q('[data-routes-section]').open = true;
      const actionMatch = Array.from(actionLookup).find(entry => entry[1] && entry[1].type === action.type && entry[1].id === action.id);
      const key = preferredKey || actionMatch && actionMatch[0];
      if (key && !node) node = q('.wg-buy[data-perform="' + key + '"]');
      if (node && node.closest('[data-main-actions]') && node.closest('.wg-action').hidden) { openDialog('manage'); node = q('[data-perform="catalog:' + node.closest('.wg-action').dataset.item + '"]'); }
      if (node && node.disabled) node = q('[data-inspect="' + key + '"]');
      if (!node || !node.getClientRects().length) node = qa('.wg-buy:not(:disabled),.wg-action-info').find(button => button.getClientRects().length);
      qa('.wg-guided').forEach(button => button.classList.remove('wg-guided'));
      if (node) { highlightedKey = node.dataset.perform || node.dataset.inspect; render(); node.classList.add('wg-guided'); node.scrollIntoView({ block: 'center', behavior: quiet ? 'instant' : 'smooth' }); node.focus({ preventScroll: true }); }
      announce(view.goal.description);
    }
    function showFindFeedback() {
      if (!pendingFinds.length || disposed || dialog.open || expeditionUI && expeditionUI.isOpen() || document.hidden) return;
      const ranks = { common: 0, uncommon: 1, rare: 2, epic: 3, legendary: 4 };
      const finds = pendingFinds.splice(0);
      finds.sort((a,b) => (ranks[b.rarity] || 0) - (ranks[a.rarity] || 0));
      const best = finds[0];
      if ((ranks[best.rarity] || 0) < 2) {
        q('[data-wallet]').classList.remove('wg-resource-pulse');
        void q('[data-wallet]').offsetWidth;
        q('[data-wallet]').classList.add('wg-resource-pulse');
        announce(best.title + ': ' + best.reward + (finds.length > 1 ? ' · ' + (finds.length - 1) + ' other finds saved.' : ''));
        return;
      }
      feedbackFind = best;
      const box = q('[data-find-feedback]');
      box.hidden = false; box.dataset.rarity = best.rarity;
      setMarkup(q('[data-find-icon]'), icon(best.relicId || best.type || best.rarity));
      set('[data-find-rarity]', best.rarity);
      set('[data-find-title]', best.title);
      let effect = best.effect || '';
      const comparison = best.relicId && (view.luck.relics || []).find(item => item.id === best.relicId);
      if (comparison && comparison.comparison) effect += ' Equip: ' + comparison.comparison.text + '.';
      else if (best.relicId) {
        const before = core.getRates(state);
        const candidate = core.normalizeState(state, state.lastUpdate);
        core.setPremiumEntitlements(candidate, billingSnapshot.owned);
        if (core.act(candidate, { type: 'relic-equip', id: best.relicId }).ok) {
          const after = core.getRates(candidate);
          const id = best.relicId === 'golden-pickaxe' ? 'ore' : best.relicId === 'surveyors-lens' ? 'knowledge' : null;
          if (id) effect += ' Equip: ' + id + ' ' + core.format(before.gain[id]) + '/s → ' + core.format(after.gain[id]) + '/s.';
        }
      }
      set('[data-find-effect]', effect);
      announce(best.title + '. ' + best.reward + (effect ? '. ' + effect : '') + (finds.length > 1 ? ' ' + (finds.length - 1) + ' other finds saved.' : ''));
      set('[data-find-reward]', best.reward + (finds.length > 1 ? ' · +' + (finds.length - 1) + ' other finds saved' : ''));
      q('[data-equip-find]').hidden = !best.relicId || !(view.luck.owned || []).includes(best.relicId);
      if (sound && interacted && !quiet && !motionQuery.matches && !document.hidden) {
        try {
          audio = audio || new (root.AudioContext || root.webkitAudioContext)();
          const oscillator = audio.createOscillator(), gain = audio.createGain();
          oscillator.type = 'sine'; oscillator.frequency.setValueAtTime(660, audio.currentTime); oscillator.frequency.setValueAtTime(990, audio.currentTime + 0.1);
          gain.gain.setValueAtTime(0.035, audio.currentTime); gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.23);
          oscillator.connect(gain); gain.connect(audio.destination); oscillator.start(); oscillator.stop(audio.currentTime + 0.24);
        } catch (error) { /* Audio remains optional. */ }
      }
      if (rewarded && !quiet && !motionQuery.matches && !document.hidden) rewarded.feedback(best.rarity).catch(() => {});
    }
    function collectFinds() {
      const ledger = view.luck && view.luck.ledger;
      if (!ledger || awaitingPurchaseWallet || ledger.seq <= lastCollectedSeq) return;
      const fresh = ledger.recent.filter(item => item.seq > Math.max(lastCollectedSeq, ledger.seen));
      lastCollectedSeq = ledger.seq;
      if (!fresh.length) return;
      pendingFinds.push(...fresh);
      pendingSeenSeq = ledger.seq;
      // Advance already granted the result. Persist result and display cursor
      // before showing any animation or native feedback.
      save();
    }
    function renderDiscoveries() {
      const luck = view.luck;
      const hasFinds = presentation().journalSections.includes('discoveries');
      const hasCaravan = !!presentation().show.caravan;
      q('[data-discovery-dock]').hidden = !hasFinds && !hasCaravan;
      q('[data-open-finds]').hidden = !hasFinds;
      if (!luck) return;
      setMarkup(q('[data-finds-icon]'), icon('relic'));
      setMarkup(q('[data-caravan-icon]'), icon('caravan'));
      set('[data-luck-pity]', luck.pity && luck.pity.text || 'Finds arrive naturally as the guild works.');
      const activeRelic = (luck.relics || []).find(item => item.selected || item.id === luck.active);
      q('[data-active-relic]').hidden = !luck.owned.length;
      setMarkup(q('[data-active-relic]'), icon(activeRelic ? activeRelic.id : 'relic') + '<div><strong>' + escapeHtml(activeRelic ? activeRelic.label : 'One active relic') + '</strong><p>' + escapeHtml(activeRelic ? activeRelic.effect || activeRelic.description : 'Choose a discovered relic to support the guild.') + '</p></div>');
      const collection = q('[data-collection-banner]');
      collection.hidden = !luck.collectionBanner;
      if (luck.collectionBanner) setMarkup(collection, icon('collection') + '<span>Constellation pennant <small>Founders collection complete</small></span>');
      set('[data-panel="finds"] h2', journalPage === 'collections' ? 'Relic collections' : 'Discoveries');
      setMarkup(q('[data-collection-progress]'), (luck.collections || []).filter(item => item.unlocked).map(item => '<section class="wg-collection-summary"><h3>' + escapeHtml(item.name) + ' · ' + item.owned + ' / ' + item.total + '</h3><p>' + escapeHtml(item.reward) + (item.completed ? ' · Complete' : '') + '</p></section>').join(''));
      reconcile('[data-relics]', (luck.relics || []).filter(item => item.owned || item.chapter <= view.progression.charters), 'relics');
      reconcile('[data-luck-research]', luck.research, 'luck-research');
      q('[data-luck-research-section]').hidden = !(luck.research || []).some(item => item.visible !== false);
      reconcile('[data-hunts]', luck.hunts, 'hunts');
      q('[data-hunts-section]').hidden = !(luck.hunts || []).some(item => item.visible !== false);
      reconcile('[data-kits]', luck.kits, 'kits');
      q('[data-kits-section]').hidden = !(luck.kits || []).some(item => item.visible !== false);
      qa('[data-panel="finds"] [data-collection-only]').forEach(node => { node.hidden = journalPage !== 'collections' || node.hasAttribute('data-hunts-section') && !(luck.hunts || []).some(item => item.visible !== false); });
      qa('[data-panel="finds"] [data-find-only]').forEach(node => { node.hidden = journalPage === 'collections' || node.hasAttribute('data-kits-section') && !(luck.kits || []).some(item => item.visible !== false); });
      setMarkup(q('[data-discovery-ledger]'), (luck.ledger.recent || []).slice().reverse().map(item => '<article><span>' + icon(item.relicId || item.rarity) + '</span><div><strong>' + escapeHtml(item.title) + '</strong><p>' + escapeHtml(item.reward) + '</p><small>' + escapeHtml(item.effect) + '</small></div></article>').join(''));
      q('[data-open-caravan]').hidden = !hasCaravan;
      if (dialog.open && dialogKind === 'caravan') renderCaravan();
      collectFinds();
    }
    function caravanRewardMarkup(quote) {
      const reward = quote && quote.reward;
      if (!reward) return '<p>Choose an available delivery.</p>';
      let text = '<div class="wg-caravan-payout">' + Object.keys(reward.resources || {}).map(id => '<span>' + icon(id) + '<strong>+' + escapeHtml(core.format(reward.resources[id])) + '</strong><small>' + escapeHtml(id) + '</small></span>').join('') + '</div>';
      if (reward.surge) text += '<p class="wg-caravan-benefit">' + escapeHtml(reward.surge.multiplier + '× ' + reward.surge.resource + ' production for ' + Math.round(reward.surge.seconds / 60) + ' minutes') + '</p>';
      if (reward.relic) text += '<p class="wg-caravan-benefit">' + icon(reward.relic.id) + '+' + reward.relic.progress + '% relic discovery progress</p>';
      return text;
    }
    function renderCaravan() {
      set('[data-dialog-title]', view.caravan && view.caravan.offer && view.caravan.offer.golden ? 'Golden caravan' : 'Visiting caravan');
      const body = q('[data-dialog-body]');
      if (!body.querySelector('[data-caravan-choices]')) body.innerHTML = '<div class="wg-caravan-heading">' + icon('caravan') + '<p>Optional delivery. Review your guaranteed reward before watching. This visit does not expire.</p></div><div class="wg-action-list" data-caravan-choices></div><div class="wg-material-picker" data-caravan-materials></div><div data-caravan-payout></div><p class="wg-shop-note" data-ad-status></p><p class="wg-shop-note" data-ad-quota></p>';
      setSheetFooter({ hook: 'data-watch-caravan', label: 'Watch ad' }, { hook: 'data-skip-caravan', label: 'Skip visit' });
      const caravan = view.caravan || {};
      reconcile('[data-caravan-choices]', caravan.choices, 'caravan');
      const quote = core.getCaravanQuote ? core.getCaravanQuote(state) : caravan.quote;
      const choices = quote && quote.kind === 'relic' ? caravan.relicTargets || [] : (caravan.materials || []).filter(item => !quote || quote.kind !== 'shipment' || item.id !== 'coins');
      const chosen = quote && (quote.kind === 'relic' ? quote.reward.relic.id : quote.kind === 'surge' ? quote.reward.surge.resource : Object.keys(quote.reward.resources).find(id => id !== 'coins'));
      setMarkup(q('[data-caravan-materials]'), quote ? '<span>' + (quote.kind === 'relic' ? 'Relic target' : 'Profession') + '</span><div>' + choices.map(item => '<button type="button" data-caravan-material="' + escapeHtml(item.id) + '" aria-pressed="' + (chosen === item.id) + '"' + (caravan.offer && caravan.offer.locked || caravan.pending || caravanBusy ? ' disabled' : '') + '>' + icon(item.id) + '<span>' + escapeHtml(item.label) + '</span></button>').join('') + '</div>' + (caravan.offer && caravan.offer.locked ? '<small>Reward chosen for this visit. Retry the same delivery or skip.</small>' : '') : '');
      setMarkup(q('[data-caravan-payout]'), caravanRewardMarkup(quote));
      if (caravan.pending || rewardedSnapshot.pending) {
        const note = q('[data-caravan-materials] small');
        if (note) updateText(note, 'Awaiting verification. Your exact reward is saved; the guild keeps working.');
      }
      set('[data-ad-status]', rewardedSnapshot.message || 'Rewarded ads are available in the configured Android app.');
      set('[data-ad-quota]', caravan.quota ? caravan.quota.used + ' / ' + caravan.quota.limit + ' rewarded visits in the last 24 hours. No reward is granted for cancelling an ad.' : 'No reward is granted for cancelling an ad.');
      q('[data-watch-caravan]').disabled = awaitingPurchaseWallet || caravanBusy || !quote || !rewardedSnapshot.available || !rewardedSnapshot.configured || rewardedSnapshot.pending;
      q('[data-skip-caravan]').disabled = caravanBusy || !!rewardedSnapshot.pending;
      updateText(q('[data-watch-caravan]'), rewardedSnapshot.pending ? 'Verification pending' : caravanBusy ? 'Ad in progress…' : 'Watch ad');
    }
    async function watchCaravan() {
      if (!rewarded || caravanBusy || awaitingPurchaseWallet || rewardedSnapshot.pending) return;
      advance();
      const quote = core.getCaravanQuote(state);
      if (!quote) return;
      const started = core.beginCaravanReward(state, quote);
      if (!started.ok) { announce(started.message); return; }
      if (!save().ok) { core.cancelCaravanReward(state, quote.offerId); announce('The caravan needs a saved game before opening an ad.'); return; }
      caravanBusy = true; render();
      try {
        const result = await rewarded.watch(quote);
        if (result && (result.state === 'cancelled' || result.cancelled)) { core.cancelCaravanReward(state, quote.offerId); save(); }
      } catch (error) {
        if (!error.uncertain && !rewardedSnapshot.pending && rewardedSnapshot.state !== 'showing' && rewardedSnapshot.state !== 'verification_pending') { core.cancelCaravanReward(state, quote.offerId); save(); }
        announce(error.message || 'The ad could not start. Your caravan is still here.');
      }
      finally { caravanBusy = rewardedSnapshot.state === 'showing'; if (!disposed) render(); }
    }
    async function applyAdReceipt(receipt) {
      if (!receipt || receiptsInFlight.has(receipt.receiptId) || awaitingPurchaseWallet || disposed) return;
      receiptsInFlight.add(receipt.receiptId);
      try {
        advance();
        const result = core.grantCaravanReward(state, receipt);
        if (result.ok) {
          if (save().ok) { await rewarded.acknowledge(receipt.receiptId); announce(result.message || 'Caravan delivery saved.'); }
        } else announce(result.message);
      } finally { receiptsInFlight.delete(receipt.receiptId); if (!disposed) render(); }
    }
    function retryReceipts() {
      if (!rewarded || awaitingPurchaseWallet || Date.now() - lastReceiptRetry < 15000) return;
      lastReceiptRetry = Date.now();
      (rewardedSnapshot.receipts || []).forEach(receipt => {
        applyAdReceipt(receipt).catch(() => announce('Your verified caravan delivery is kept and will retry.'));
      });
    }
    function render() {
      if (disposed || document.hidden) return;
      element.dataset.quiet = String(quiet || motionQuery.matches);
      view = core.getView(state);
      const unlocked = new Set(view.unlocks);
      const shown = presentation();
      const isEarly = shown.opening;
      const guildSections = shown.guildSections.map(id => id === 'rooms' ? 'guild' : id);
      if (!shown.primary.includes(destination()) || ['guild', 'crew', 'research', 'planning'].includes(tab) && !guildSections.includes(tab)) tab = 'trail';
      if (!shown.journalSections.includes(journalPage)) journalPage = shown.journalSections[0] || 'discoveries';
      q('[data-game]').dataset.early = String(isEarly);
      q('[data-game]').dataset.screen = tab;
      q('[data-game]').dataset.expanded = String(unlocked.has('forge') || !['trail', 'guild'].includes(tab));
      q('[data-nav]').hidden = false;
      qa('[data-tab]').forEach(button => { button.hidden = !shown.primary.includes(button.dataset.tab); button.disabled = false; });
      q('[data-tab="guild"]').title = 'Your working guild rooms';
      qa('[data-guild-view]').forEach(button => { button.hidden = !guildSections.includes(button.dataset.guildView); });
      q('[data-opening-help]').hidden = true;
      qa('[data-tab]').forEach(button => {
        if (button.dataset.tab === destination()) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
      });
      if (destination() === 'journey') tab = ['discoveries', 'collections'].includes(journalPage) ? 'finds' : 'journey';
      qa('[data-panel]').forEach(panel => { panel.hidden = panel.dataset.panel === 'play' ? !['trail', 'guild'].includes(tab) : panel.dataset.panel === 'shop' ? dialogKind !== 'shop' : panel.dataset.panel !== tab; });
      q('.wg-scene').hidden = tab !== 'trail';
      renderWallet();
      renderShop();
      const walletWait = q('[data-purchase-wallet-wait]');
      if (walletWait) walletWait.hidden = !awaitingPurchaseWallet;
      renderRooms();
      renderOverview();
      const selectedRoom = view.rooms.find(item => item.id === (tab === 'guild' ? room : 'trail')) || view.rooms[0];
      q('[data-selected-room]').hidden = true;
      set('[data-dock-title]', tab === 'guild' ? selectedRoom.name : 'Trail');
      set('[data-dock-hint]', isEarly ? 'Coins arrive automatically' : '');
      set('[data-room-sheet-title]', selectedRoom.name);
      set('[data-room-name]', selectedRoom.name);
      set('[data-room-description]', selectedRoom.description);
      set('[data-room-mastery]', 'Level ' + selectedRoom.level + ' · Mastery ' + selectedRoom.mastery);
      const outputResource = view.resources.find(item => ({ mine: 'ore', forage: 'herbs', kitchen: 'provisions', study: 'knowledge', cartography: 'maps' }[room]) === item.id);
      set('[data-activity]', tab === 'guild' ? selectedRoom.name + (outputResource ? ' · ' + outputResource.rateFormatted + ' ' + outputResource.name.toLowerCase() : '') : isEarly ? 'Automatic travel' : view.progression.routeName + ' · ' + view.progression.realmName);
      const goal = view.goal;
      const nextUnlock = shown.nextUnlock;
      set('[data-goal-title]', isEarly ? 'Reach the Mine' : goal.title);
      const targetAction = goal.action && goal.action.type === 'plan-goal' ? goal.action.action : goal.action;
      const goalDescriptor = (view.actions || []).concat(view.research || [], view.development && view.development.projects || []).find(item => item.action && targetAction && item.action.type === targetAction.type && item.action.id === targetAction.id);
      const goalLabel = goal.actionLabel || (goal.action && goal.action.type === 'buy' ? 'Buy ' + (goalDescriptor ? itemName(goalDescriptor) : goal.action.id) : goal.action && goal.action.type === 'research' ? 'Research ' + (goalDescriptor ? itemName(goalDescriptor) : goal.action.id) : goalDescriptor ? goalDescriptor.label || goalDescriptor.name : goal.title);
      setMarkup(q('[data-goal-icon]'), icon(isEarly ? 'watchtower' : goal.action && goal.action.room || 'compass'));
      const goalHome = q('[data-goal-dock]');
      if (goalHome.nextElementSibling !== q('[data-goal]')) goalHome.after(q('[data-goal]'));
      q('[data-goal-progress]').value = Math.min(1, Math.max(0, isEarly ? view.progression.routePercent : goal.progress || 0));
      set('[data-goal-progress-text]', isEarly ? view.progression.routeProgress + ' / ' + view.progression.routeTarget : goal.ready ? goal.action && goal.action.type === 'plan-goal' ? 'Ready to plan' : 'Ready' : goal.progressText || goal.remainingText || 'Traveling');
      set('[data-goal-description]', goal.description);
      q('[data-goal-description]').hidden = true;
      q('[data-goal]').title = goal.description || '';
      set('[data-long-goal]', goal.longGoal || '');
      q('[data-long-goal]').hidden = true;
      q('[data-goal-options]').hidden = true;
      q('[data-goal-action]').hidden = false;
      const direct = q('[data-goal-direct]');
      direct.hidden = isEarly || !goal.action;
      const plannedGoal = goal.action && goal.action.type === 'plan-goal' && view.planning && view.planning.goal && view.planning.goal.type === goal.action.action.type && view.planning.goal.id === goal.action.action.id;
      direct.disabled = awaitingPurchaseWallet || !!plannedGoal;
      const directLabel = plannedGoal ? 'Saving for this' : goal.action && goal.action.type === 'route' ? 'Explore' : goal.action && goal.action.type === 'build-room' ? 'Build ' + ((view.rooms.find(item => item.id === goal.action.id) || {}).name || 'room') : goal.action && ['refit','charter'].includes(goal.action.type) ? 'Review ' + goal.action.type : goal.action && ['ui','navigate'].includes(goal.action.type) && goal.action.tab === 'journey' ? 'Review ' + (/charter/i.test(goal.actionLabel || goal.title) ? 'charter' : 'refit') : goalLabel;
      direct.setAttribute('aria-label', plannedGoal ? 'Saving for this' : goalLabel + (goalDescriptor && goalDescriptor.cost && goal.action.type !== 'plan-goal' ? ' · ' + goalDescriptor.cost.map(entry => entry.text || core.format(entry.amount) + ' ' + entry.resource).join(' + ') : ''));
      setMarkup(direct, '<span>' + escapeHtml(directLabel) + '</span>' + (goalDescriptor && goalDescriptor.cost && !plannedGoal && goal.action.type !== 'plan-goal' ? '<span class="wg-buy-costs">' + costsMarkup(goalDescriptor.cost) + '</span>' : ''));
      q('[data-next-unlock]').hidden = true;
      if (nextUnlock) {
        set('[data-next-unlock-title]', { mine: 'Mine', forge: 'Forge', forage: 'Foragers', kitchen: 'Kitchen', study: 'Study', hall: 'Guild Hall', cartography: 'Map Room' }[nextUnlock.id] || nextUnlock.label.replace(/^The /, ''));
        q('[data-open-goal-details]').setAttribute('aria-label', 'Next: ' + nextUnlock.label + '. View requirements and effects');
        set('[data-next-unlock-requirement]', nextUnlock.requirement);
        set('[data-next-unlock-effect]', nextUnlock.effect);
      }
      if (goal.title !== lastGoal) { lastGoal = goal.title; }
      q('[data-game]').dataset.room = tab === 'guild' ? room : 'trail';
      setMarkup(q('[data-goal-action]'), '<span class="wg-info-symbol" aria-hidden="true">i</span>');
      let actions = (tab === 'guild' ? selectedRoom.actions.filter(item => item.action.type !== 'recipe') : view.actions.filter(item => item.room === 'trail')) || [];
      if (tab === 'guild' && room === 'forge') actions = actions.slice().sort((a, b) => (a.id.startsWith('gear-') ? 0 : 1) - (b.id.startsWith('gear-') ? 0 : 1));
      currentActions = actions.filter(item => item.visible !== false);
      reconcile('[data-main-actions]', currentActions, 'main');
      const dockRows = Array.from(q('[data-main-actions]').children);
      dockRows.forEach((node, index) => { node.hidden = index >= 2; });
      q('[data-main-actions]').dataset.single = String(dockRows.length === 1);
      const more = q('[data-more-upgrades]');
      more.hidden = isEarly;
      more.setAttribute('aria-label', tab === 'guild' ? selectedRoom.name + ' upgrades and recipes' : 'All trail upgrades and supplies');
      more.setAttribute('aria-expanded', String(dialog.open && dialogKind === 'manage'));
      setMarkup(more, icon('equipment'));
      if (dialog.open && dialogKind === 'manage') reconcile('[data-all-upgrades]', currentActions, 'catalog');
      const recipes = (view.recipes || []).filter(item => item.room === room && tab === 'guild');
      q('[data-recipes-section]').hidden = !recipes.some(item => item.visible !== false);
      reconcile('[data-recipes]', recipes, 'recipes');
      q('[data-routes-section]').hidden = tab !== 'trail' || !unlocked.has('cartography');
      reconcile('[data-routes]', view.routes, 'routes');
      const expedition = view.expedition || {};
      set('[data-route-condition]', expedition.conditionText || '');
      reconcile('[data-route-preparation]', expedition.preparation, 'route-preparation');
      q('[data-supply-section]').hidden = !unlocked.has('kitchen') || tab !== 'trail';
      set('[data-supply-demand]', expedition.demandText || '');
      reconcile('[data-supply-choices]', expedition.supplyChoices, 'supply');
      const modes = q('[data-modes]');
      if (!modes.children.length) modes.innerHTML = core.Content.MODES.map(item => '<button type="button" data-mode="' + escapeHtml(item.id) + '" title="' + escapeHtml(item.description) + '">' + escapeHtml(item.name) + '</button>').join('');
      Array.from(modes.children).forEach(button => { button.setAttribute('aria-pressed', String(button.dataset.mode === view.progression.mode)); button.disabled = awaitingPurchaseWallet; });
      set('[data-mode-description]', (core.Content.MODES.find(item => item.id === view.progression.mode) || {}).description || '');
      reconcile('[data-research]', view.research, 'research');
      const specialists = view.specialists.filter(item => !item.owned || item.slot === 0).map(item => item.owned ? Object.assign({}, item, { label: (core.Content.SPECIALISTS.find(def => def.id === item.action.id) || {}).name || item.label, disabled: true, selected: false, reason: view.progression.specialists.includes(item.action.id) ? 'Assigned · choose a different slot above' : 'Available · choose an assignment above' }) : item);
      reconcile('[data-specialists]', specialists, 'specialists');
      reconcile('[data-companions]', view.companions, 'companions');
      q('[data-companions-section]').hidden = !(view.companions || []).some(item => item.visible !== false);
      renderAssignments();
      set('[data-journey-summary]', [view.progression.realmName, 'Route ' + (view.progression.highestRoute + 1), view.progression.refits ? view.progression.refits + ' refits' : '', view.progression.charters ? view.progression.charters + ' charters' : ''].filter(Boolean).join(' · '));
      set('[data-refit-summary]', view.refit.available ? 'Ready · ' + view.refit.rewardText : (view.refit.requirements.find(item => !item.met) || {}).label || 'Continue your expedition');
      const showCharter = (shown.unlockedIds || []).includes('charter');
      q('[data-charter-section]').hidden = !showCharter;
      set('[data-charter-summary]', view.charter.available ? 'Ready · ' + view.charter.rewardText : (view.charter.requirements.find(item => !item.met) || {}).label || 'Continue your guild');
      const legacy = (view.refitUpgrades || []).concat(view.legacyUpgrades || []);
      reconcile('[data-legacy]', legacy, 'legacy');
      [['doctrines', view.doctrines], ['challenges', view.challenges], ['automations', view.automations]].forEach(entry => {
        const available = (entry[1] || []).filter(item => item.id !== 'leave-challenge' && (entry[0] !== 'automations' || item.unlocked !== false));
        q('[data-' + entry[0] + '-section]').hidden = !available.some(item => item.visible !== false);
        reconcile('[data-' + entry[0] + ']', available, entry[0]);
      });
      q('[data-leave-challenge]').hidden = !view.progression.challenge;
      renderRecord();
      renderJournal();
      renderDiscoveries();
      const fraction = view.progression.routePercent || 0;
      if (!expeditionUI) scene.update({ room: tab === 'guild' ? room : 'trail', realm: view.progression.realm, progress: fraction > 1 ? fraction / 100 : fraction, workers: 1, companion: view.progression.companion || null, reducedMotion: quiet || motionQuery.matches });
      q('[data-scene]').setAttribute('aria-label', tab === 'guild' ? selectedRoom.name + ': ' + selectedRoom.description : 'An adventurer automatically traveling through ' + view.progression.realmName);
      if (dialog.open && ['refit', 'charter'].includes(dialogKind)) renderReset();
      if (dialog.open && dialogKind === 'inspect') renderInspect();
      if (dialog.open && dialogKind === 'goal') renderGoalDetails();
      if (dialog.open && dialogKind === 'resources') renderResourceDetails();
      renderPlanning();
      renderDevelopment();
      updateSaveStatus();
      renderIntroduction();
      if (expeditionUI) expeditionUI.update(view, { quiet:quiet || motionQuery.matches, saveFailure, awaitingWallet:awaitingPurchaseWallet });
    }

    function setSheetFooter(primary, secondary) {
      const hooks = ['data-perform', 'data-close-dialog', 'data-sheet-back', 'data-watch-caravan', 'data-skip-caravan', 'data-confirm-reset', 'data-confirm-challenge', 'data-confirm-import', 'data-cancel-import', 'data-goal-guide'];
      [['[data-sheet-primary]', primary], ['[data-sheet-secondary]', secondary]].forEach(([selector, action]) => {
        const button = q(selector);
        hooks.forEach(hook => button.removeAttribute(hook));
        button.hidden = !action;
        if (!action) return;
        button.setAttribute(action.hook, action.value || '');
        updateText(button, action.label);
        button.disabled = !!action.disabled;
      });
    }
    function openDialog(kind, restore) {
      if (expeditionUI) expeditionUI.close();
      if (!dialog.open) { opener = document.activeElement; sheetStack.length = 0; }
      else if (!restore && kind === 'inspect' && ['room', 'shop', 'manage'].includes(dialogKind)) sheetStack.push({ kind: dialogKind, room, focus: document.activeElement });
      else if (!restore && kind !== dialogKind) sheetStack.length = 0;
      dialogKind = kind;
      pendingImport = null;
      set('[data-dialog-notice]', '');
      dialog.dataset.kind = kind;
      q('[data-room-content]').hidden = kind !== 'room';
      q('[data-shop-content]').hidden = kind !== 'shop';
      q('[data-dialog-body]').hidden = ['room', 'shop', 'manage'].includes(kind);
      manageContent.hidden = kind !== 'manage';
      q('[data-sheet-back]').hidden = !sheetStack.length;
      if (sheetContentKey !== kind) { delete q('[data-dialog-body]').dataset.inspect; delete q('[data-dialog-body]').dataset.resetSignature; delete q('[data-dialog-body]').dataset.markup; sheetContentKey = kind; }
      setSheetFooter(null, { hook: 'data-close-dialog', label: 'Done' });
      if (kind === 'inspect') renderInspect();
      else if (kind === 'goal') renderGoalDetails();
      else if (kind === 'resources') renderResourceDetails();
      else if (kind === 'help' && activeHelp) { set('[data-dialog-title]', activeHelp.title); setMarkup(q('[data-dialog-body]'), activeHelp.paragraphs.map(node => '<p>' + escapeHtml(node.textContent) + '</p>').join('')); }
      else if (kind === 'manage') { set('[data-dialog-title]', (tab === 'guild' ? (view.rooms.find(item => item.id === room) || {}).name : 'Trail') + ' upgrades'); reconcile('[data-all-upgrades]', currentActions, 'catalog'); }
      else if (kind === 'caravan') renderCaravan();
      else if (kind === 'settings') { renderSettings(); updateSaveStatus(); }
      else if (kind === 'refit' || kind === 'charter') renderReset();
      else if (kind === 'challenge') renderChallenge();
      else if (kind === 'room') { q('[data-room-content]').appendChild(playPanel); set('[data-dialog-title]', (view.rooms.find(item => item.id === room) || {}).name || 'Guild room'); }
      else if (kind === 'shop') { q('[data-shop-content]').appendChild(shopPanel); shopPanel.hidden = false; set('[data-dialog-title]', 'Starshard shop'); renderShop(); }
      else return;
      if (!dialog.open) dialog.showModal();
      if (!restore) q('[data-sheet-scroll]').scrollTop = 0;
    }
    function closeDialog() {
      if (dialog.open) dialog.close();
      dialogKind = null;
      sheetStack.length = 0;
      pendingImport = null;
      pendingChallenge = null;
      q('[data-play-home]').after(playPanel);
      q('[data-shop-home]').after(shopPanel);
      render();
      if (opener && opener.isConnected && opener.getClientRects().length) opener.focus({ preventScroll: true });
      if (pendingFinds.length && pendingSeenSeq == null) showFindFeedback();
    }
    function backSheet() {
      const previous = sheetStack.pop();
      if (!previous) { closeDialog(); return; }
      room = previous.room;
      openDialog(previous.kind, true);
      render();
      if (previous.focus && previous.focus.isConnected && previous.focus.getClientRects().length) previous.focus.focus({ preventScroll: true });
    }
    function handleBack() {
      if (expeditionUI && expeditionUI.handleBack()) return true;
      if (dialog.open) { backSheet(); return true; }
      if (!q('[data-find-feedback]').hidden) { feedbackFind = null; q('[data-find-feedback]').hidden = true; return true; }
      if (['crew', 'research', 'planning'].includes(tab)) { showTab('guild'); return true; }
      if (destination() === 'journey' && journalPage !== 'discoveries') { journalPage = 'discoveries'; showTab('journey'); return true; }
      if (tab !== 'trail') { showTab('trail'); return true; }
      return false;
    }
    function renderSettings() {
      set('[data-dialog-title]', 'Settings & saves');
      const help = presentation().systems || [];
      q('[data-dialog-body]').innerHTML = '<p>Progress saves on this device. Export a backup to move your guild.</p>' +
        '<div data-purchase-wallet-wait' + (awaitingPurchaseWallet ? '' : ' hidden') + '><p>Waiting for the purchase wallet before calculating offline progress. Your saved guild is kept while it reconnects.</p><button type="button" class="wg-button" data-billing-refresh>Retry purchase wallet</button></div>' +
        (presentation().journalSections.includes('discoveries') ? '<label class="wg-toggle">Discovery chimes<input type="checkbox" data-sound' + (sound ? ' checked' : '') + '></label>' : '') +
        (rewardedSnapshot.privacyOptionsRequired ? '<button type="button" class="wg-button" data-ad-privacy>Ad privacy choices</button>' : '') +
        '<label class="wg-toggle">Quiet animation<input type="checkbox" data-quiet' + (quiet ? ' checked' : '') + '></label>' +
        '<p data-save-label></p><p data-pending-status></p>' +
        '<div class="wg-dialog-actions"><button type="button" class="wg-button" data-export="download">Download save</button><button type="button" class="wg-button" data-export="text">Show backup text</button></div>' +
        '<label class="wg-dialog-label" for="wg-save-text">Backup text</label><textarea id="wg-save-text" spellcheck="false" autocomplete="off" placeholder="Paste an exported Wayfarers save here"></textarea>' +
        '<div class="wg-dialog-actions"><button type="button" class="wg-text-button" data-copy-save>Copy backup</button><button type="button" class="wg-button" data-review-import>Review import</button></div>' +
        '<label class="wg-dialog-label" for="wg-save-file">Or choose a save file</label><input id="wg-save-file" type="file" accept=".json,application/json,text/plain">' +
        '<div data-import-preview hidden></div>' + (help.length ? '<details class="wg-details" data-unlocked-help><summary>Your unlocked features</summary>' + help.map(item => '<section class="wg-section"><h3>' + escapeHtml(item.label) + '</h3><p>' + escapeHtml(item.requirement || '') + '</p><p>' + escapeHtml(item.effect) + '</p></section>').join('') + '</details>' : '<p>Explore the trail and improve your boots. The next unlock explains what comes next.</p>');
    }
    function renderResourceDetails() {
      set('[data-dialog-title]', 'Resources');
      const description = { coins: 'Earn coins by exploring. Improve your boots to earn more.', ore: 'Miners gather ore. Spend it on guild equipment.', herbs: 'Foragers gather herbs for recipes and supplies.', provisions: 'The Kitchen turns ingredients into expedition supplies.', knowledge: 'The Study generates knowledge for research.', maps: 'Use maps to prepare routes and reach new frontiers.', notes: 'Expedition Refits award notes for permanent upgrades.', crests: 'Guild Charters award crests for lasting guild improvements.' }[inspectedCurrency] || 'Your earned resources stay with this guild save.';
      setMarkup(q('[data-dialog-body]'), '<p>' + escapeHtml(description) + '</p><div class="wg-record">' + view.resources.filter(item => item.visible !== false).map(item => '<div class="wg-record-row"><span>' + icon(item.id) + ' ' + escapeHtml(item.name) + '</span><strong>' + escapeHtml(item.formatted) + '<small>' + escapeHtml(item.rateFormatted || '') + '</small></strong></div>').join('') + '</div>');
    }
    function renderGoalDetails() {
      const goal = view.goal;
      const next = presentation().nextUnlock;
      set('[data-dialog-title]', 'Your next goal');
      const targetAction = goal.action && goal.action.type === 'plan-goal' ? goal.action.action : goal.action;
      const descriptor = (view.actions || []).concat(view.research || [], view.development && view.development.projects || []).find(item => item.action && targetAction && item.action.type === targetAction.type && item.action.id === targetAction.id);
      const body = q('[data-dialog-body]');
      if (!body.querySelector('[data-goal-detail-title]')) body.innerHTML = '<h3 data-goal-detail-title></h3><p data-goal-detail-description></p><p data-goal-detail-progress></p><div class="wg-inspect-cost" data-goal-detail-cost></div><p data-goal-detail-shortage></p><section data-goal-long-details><h3>After this</h3><p data-goal-detail-long></p></section><section class="wg-section" data-goal-unlock-details><h3 data-goal-detail-unlock></h3><p data-goal-detail-requirement></p><p data-goal-detail-effect></p></section><div class="wg-action-list" data-goal-choice-list></div>';
      set('[data-goal-detail-title]', goal.title);
      set('[data-goal-detail-description]', goal.description);
      set('[data-goal-detail-progress]', [goal.remainingText || goal.progressText, etaText(goal.etaSeconds)].filter(Boolean).join(' · '));
      setMarkup(q('[data-goal-detail-cost]'), descriptor ? costsMarkup(descriptor.cost) : '');
      set('[data-goal-detail-shortage]', descriptor ? descriptor.shortageText || descriptor.reason || '' : '');
      q('[data-goal-long-details]').hidden = !goal.longGoal;
      set('[data-goal-detail-long]', goal.longGoal || '');
      q('[data-goal-unlock-details]').hidden = !next;
      if (next) {
        set('[data-goal-detail-unlock]', next.label);
        set('[data-goal-detail-requirement]', next.requirement);
        set('[data-goal-detail-effect]', next.effect);
      }
      reconcile('[data-goal-choice-list]', (goal.options || []).slice(0, 2), 'goal-options');
      setSheetFooter(goal.action ? { hook: 'data-goal-guide', label: 'Go to action' } : null, { hook: 'data-close-dialog', label: 'Keep exploring' });
    }
    function renderChallenge() {
      const item = (view.challenges || []).find(entry => entry.action && entry.action.id === pendingChallenge.id);
      set('[data-dialog-title]', 'Begin a realm challenge');
      q('[data-dialog-body]').innerHTML = '<h3>' + escapeHtml(item ? item.label : 'Realm challenge') + '</h3><p>' + escapeHtml(item ? item.description : '') + '</p><h3>You rebuild</h3><p>Starting returns to the first route and clears ordinary coins, field preparation, and the active meal. It gives no Refit reward. Your rooms, equipment, mastery, research, and material stocks remain.</p>';
      setSheetFooter({ hook: 'data-confirm-challenge', label: 'Begin challenge', disabled: awaitingPurchaseWallet || caravanBusy }, { hook: 'data-close-dialog', label: 'Keep exploring' });
    }
    function renderReset() {
      const preview = dialogKind === 'charter' ? view.charter : view.refit;
      set('[data-dialog-title]', dialogKind === 'charter' ? 'New Guild Charter' : 'Expedition Refit');
      const signature = JSON.stringify([dialogKind, preview, awaitingPurchaseWallet]);
      const body = q('[data-dialog-body]');
      setSheetFooter({ hook: 'data-confirm-reset', label: 'Confirm ' + (dialogKind === 'charter' ? 'charter' : 'refit'), disabled: !preview.available || awaitingPurchaseWallet || caravanBusy }, { hook: 'data-close-dialog', label: 'Keep exploring' });
      if (body.dataset.resetSignature === signature) return;
      const hadConfirmFocus = document.activeElement && document.activeElement.hasAttribute('data-confirm-reset');
      body.dataset.resetSignature = signature;
      body.innerHTML = '<p class="wg-reward">' + escapeHtml(preview.rewardText) + '</p>' +
        '<ul class="wg-preview-requirements">' + preview.requirements.map(item => '<li data-met="' + item.met + '">' + (item.met ? 'Complete: ' : 'Required: ') + escapeHtml(item.label) + '</li>').join('') + '</ul>' +
        '<h3>You keep</h3><ul>' + preview.keeps.map(item => '<li>' + escapeHtml(item) + '</li>').join('') + '</ul>' +
        '<h3>You rebuild</h3><ul>' + preview.resets.map(item => '<li>' + escapeHtml(item) + '</li>').join('') + '</ul>' +
        '<h3>You gain</h3><ul>' + preview.gains.map(item => '<li>' + escapeHtml(item) + '</li>').join('') + '</ul>' +
        (preview.purchases && preview.purchases.length ? '<h3>Improvements you can afford</h3><ul>' + preview.purchases.map(item => '<li><strong>' + escapeHtml(item.name) + '</strong> · ' + escapeHtml(item.costText) + '<p>' + escapeHtml(item.effectText) + '</p></li>').join('') + '</ul>' : '') +
        (preview.planText ? '<h3>Your plan</h3><p>' + escapeHtml(preview.planText) + '</p>' : '') +
        (preview.recovery ? '<h3>Recovery estimate</h3><p>' + escapeHtml(preview.recovery.text) + '</p>' : '');
      if (hadConfirmFocus) q('[data-confirm-reset]').focus({ preventScroll: true });
    }
    function exportSave() {
      advance();
      const result = store.export(state);
      if (!result.ok) { set('[data-dialog-notice]', result.message); return null; }
      q('#wg-save-text').value = result.text;
      set('[data-dialog-notice]', 'Backup prepared. Keep a copy outside this browser.');
      return result;
    }
    function reviewImport(text) {
      pendingImport = null;
      setSheetFooter(null, { hook: 'data-close-dialog', label: 'Done' });
      q('[data-import-preview]').hidden = true;
      if (caravanBusy) { set('[data-dialog-notice]', 'Finish or close the ad before importing a guild.'); return; }
      if (awaitingPurchaseWallet) { set('[data-dialog-notice]', 'Wait for the purchase wallet to reconnect before reviewing an import. Your current guild has been kept.'); return; }
      const result = store.inspectImport(text, { premiumEntitlements: billingSnapshot.owned });
      if (!result.ok) { set('[data-dialog-notice]', result.message); return; }
      pendingImport = text;
      const candidate = core.getView(result.state);
      const summary = candidate.progression;
      const preview = q('[data-import-preview]');
      preview.hidden = false;
      preview.innerHTML = '<h3>Replace this guild?</h3><p>' + escapeHtml(summary.realmName) + ', route ' + (summary.highestRoute + 1) + ', ' + summary.refits + ' refits and ' + summary.charters + ' charters. Offline progress has been included in this preview. The current guild remains unchanged until you confirm.</p>';
      setSheetFooter({ hook: 'data-confirm-import', label: 'Replace with this save' }, { hook: 'data-cancel-import', label: 'Keep current guild' });
      preview.scrollIntoView({ block: 'nearest' });
      set('[data-dialog-notice]', 'Save validated. Review the guild above before replacing your progress.');
    }

    element.addEventListener('click', async event => {
      interacted = true;
      const button = event.target.closest('button');
      if (!button || button.disabled) return;
      if (button.dataset.inspect) { inspectedKey = button.dataset.inspect; openDialog('inspect'); }
      else if (button.dataset.perform) perform(actionLookup.get(button.dataset.perform));
      else if (button.hasAttribute('data-sheet-back')) backSheet();
      else if (button.hasAttribute('data-goal-direct')) guideGoal();
      else if (button.dataset.guildView) showTab(button.dataset.guildView);
      else if (button.hasAttribute('data-dismiss-return')) q('[data-return-summary]').hidden = true;
      else if (button.hasAttribute('data-plan-queue-add')) perform({ type: 'plan-queue', action: planChoices.get(q('[data-plan-queue-choice]').value) });
      else if (button.hasAttribute('data-plan-remove')) perform({ type: 'plan-remove', index: Number(button.dataset.planRemove) });
      else if (button.hasAttribute('data-loadout-save')) perform({ type: 'loadout-save', id: Number(button.dataset.loadoutSave), name: q('[data-loadout-name="' + button.dataset.loadoutSave + '"]').value });
      else if (button.hasAttribute('data-loadout-use')) perform({ type: 'loadout-use', id: Number(button.dataset.loadoutUse) });
      else if (button.hasAttribute('data-loadout-delete')) perform({ type: 'loadout-delete', id: Number(button.dataset.loadoutDelete) });
      else if (button.dataset.tab) showTab(button.dataset.tab);
      else if (button.dataset.room) openRoom(button.dataset.room);
      else if (button.hasAttribute('data-show-overview')) { closeRoom(); overview = true; render(); }
      else if (button.hasAttribute('data-close-overview')) openRoom(room);
      else if (button.hasAttribute('data-close-room')) closeRoom();
      else if (button.dataset.journal) { journalPage = button.dataset.journal; showTab('journey'); }
      else if (button.hasAttribute('data-open-finds')) { journalPage = 'discoveries'; showTab('journey'); }
      else if (button.hasAttribute('data-dismiss-teaching')) { teaching = null; q('[data-teaching]').hidden = true; }
      else if (button.hasAttribute('data-open-teaching')) { if (teaching) introductionAction(teaching.action); q('[data-teaching]').hidden = true; }
      else if (button.hasAttribute('data-open-caravan')) { openDialog('caravan'); if (rewarded) rewarded.refresh().catch(() => {}); }
      else if (button.hasAttribute('data-watch-caravan')) watchCaravan();
      else if (button.hasAttribute('data-ad-privacy') && rewarded) rewarded.privacyOptions().catch(error => set('[data-dialog-notice]', error.message));
      else if (button.dataset.caravanMaterial) { const quote = core.getCaravanQuote(state); if (quote) perform({ type: 'caravan-select', kind: quote.kind, [quote.kind === 'shipment' ? 'material' : quote.kind === 'surge' ? 'resource' : 'relicId']: button.dataset.caravanMaterial }); }
      else if (button.hasAttribute('data-skip-caravan')) { if (!rewardedSnapshot.pending && !caravanBusy && perform({ type: 'caravan-dismiss' }).ok) closeDialog(); }
      else if (button.hasAttribute('data-dismiss-find')) { feedbackFind = null; q('[data-find-feedback]').hidden = true; }
      else if (button.hasAttribute('data-equip-find')) { if (feedbackFind && feedbackFind.relicId && perform({ type: 'relic-equip', id: feedbackFind.relicId }).ok) { feedbackFind = null; q('[data-find-feedback]').hidden = true; } }
      else if (button.dataset.open === 'journey') showTab('journey');
      else if (button.dataset.open === 'shop') { advance(); openDialog('shop'); billingAction('refresh'); }
      else if (button.dataset.open) { advance(); render(); openDialog(button.dataset.open); }
      else if (button.hasAttribute('data-goal-action') || button.hasAttribute('data-open-goal-details')) openDialog('goal');
      else if (button.hasAttribute('data-goal-guide')) { closeDialog(); guideGoal(); }
      else if (button.hasAttribute('data-more-upgrades')) openDialog('manage');
      else if (button.hasAttribute('data-context-help')) { activeHelp = contextHelp[Number(button.dataset.contextHelp)]; openDialog('help'); }
      else if (button.hasAttribute('data-wallet-resource')) { inspectedCurrency = button.dataset.walletResource; openDialog('resources'); }
      else if (button.hasAttribute('data-back-trail')) showTab('trail');
      else if (button.dataset.mode) perform({ type: 'route', id: view.progression.routeId, mode: button.dataset.mode });
      else if (button.hasAttribute('data-leave-challenge')) perform({ type: 'challenge', id: null });
      else if (button.hasAttribute('data-close-dialog')) closeDialog();
      else if (button.dataset.buyPack) billingAction('purchase', button.dataset.buyPack);
      else if (button.hasAttribute('data-billing-refresh')) billingAction('refresh');
      else if (button.hasAttribute('data-billing-restore')) billingAction('restore');
      else if (button.hasAttribute('data-wallet-backup')) billingAction('backupWallet');
      else if (button.hasAttribute('data-wallet-restore')) billingAction('restoreWallet');
      else if (button.hasAttribute('data-confirm-reset')) {
        const kind = dialogKind;
        if (perform({ type: kind }).ok) { closeDialog(); tab = 'trail'; room = 'mine'; render(); }
      } else if (button.hasAttribute('data-confirm-challenge')) {
        const result = perform(pendingChallenge);
        if (result.ok) { closeDialog(); showTab('trail'); }
        else set('[data-dialog-notice]', result.message);
      } else if (button.dataset.export) {
        const exported = exportSave();
        if (exported && button.dataset.export === 'download') {
          const url = URL.createObjectURL(new Blob([exported.text], { type: 'application/json' }));
          const anchor = document.createElement('a');
          anchor.href = url;
          anchor.download = exported.filename;
          anchor.click();
          root.setTimeout(() => URL.revokeObjectURL(url), 1000);
        }
      } else if (button.hasAttribute('data-copy-save')) {
        if (!q('#wg-save-text').value) exportSave();
        try { await navigator.clipboard.writeText(q('#wg-save-text').value); set('[data-dialog-notice]', 'Backup copied. Paste it somewhere safe.'); }
        catch (error) { q('#wg-save-text').focus(); q('#wg-save-text').select(); set('[data-dialog-notice]', 'Backup selected. Use your device’s Copy command.'); }
      } else if (button.hasAttribute('data-review-import')) reviewImport(q('#wg-save-text').value);
      else if (button.hasAttribute('data-cancel-import')) { pendingImport = null; setSheetFooter(null, { hook: 'data-close-dialog', label: 'Done' }); q('[data-import-preview]').hidden = true; set('[data-dialog-notice]', 'Import cancelled. Your current guild is unchanged.'); }
      else if (button.hasAttribute('data-confirm-import')) {
        if (caravanBusy) { set('[data-dialog-notice]', 'Finish or close the ad before importing a guild.'); return; }
        if (awaitingPurchaseWallet) { set('[data-dialog-notice]', 'Reconnect the purchase wallet before importing. Your current guild has been kept.'); return; }
        const result = store.replaceImport(pendingImport, { premiumEntitlements: billingSnapshot.owned });
        if (!result.ok) { set('[data-dialog-notice]', result.message); return; }
        state = result.state;
        if (result.persisted) { lastSuccessfulSave = Date.now(); saveFailure = null; } else saveFailure = result.message;
        if (core.setPremiumEntitlements) core.setPremiumEntitlements(state, billingSnapshot.owned);
        view = core.getView(state);
        teaching = null;
        q('[data-teaching]').hidden = true;
        lastCollectedSeq = view.luck && view.luck.ledger ? view.luck.ledger.seen : 0;
        pendingFinds = [];
        pendingSeenSeq = null;
        feedbackFind = null;
        q('[data-find-feedback]').hidden = true;
        if (dialog.open) closeDialog();
        q('[data-play-home]').after(playPanel);
        tab = 'trail';
        overview = false;
        storageNotice(result.persisted ? '' : result.message);
        closeDialog();
        render();
        announce('Imported guild ready. ' + (result.persisted ? 'Saved on this device.' : result.message));
      }
    }, { signal });
    element.addEventListener('change', async event => {
      if (event.target.hasAttribute('data-sound')) { sound = event.target.checked; interacted = true; try { root.localStorage.setItem('wayfarers-guild-sound', String(sound)); } catch (error) {} }
      else if (event.target.hasAttribute('data-quiet')) {
        quiet = event.target.checked;
        try { root.localStorage.setItem('wayfarers-guild-quiet', String(quiet)); } catch (error) { set('[data-dialog-notice]', 'Quiet animation applies to this session. Browser storage is unavailable.'); }
        render();
      }
      else if (event.target.hasAttribute('data-plan-goal')) perform({ type: 'plan-goal', action: planChoices.get(event.target.value) || null });
      else if (event.target.hasAttribute('data-plan-priority')) perform({ type: 'plan-priority', group: event.target.dataset.planPriority, id: event.target.value });
      else if (event.target.hasAttribute('data-plan-reserve')) perform({ type: 'plan-reserve', id: event.target.dataset.planReserve, amount: event.target.value });
      else if (event.target.hasAttribute('data-plan-preparation')) perform({ type: 'plan-preparation', id: event.target.value });
      else if (event.target.hasAttribute('data-plan-kit')) perform({ type: 'plan-kit', id: event.target.value });
      else if (event.target.hasAttribute('data-specialist-slot')) perform({ type: 'specialist', slot: Number(event.target.dataset.specialistSlot), id: event.target.value || null });
      else if (event.target.id === 'wg-save-file') {
        pendingImport = null;
        setSheetFooter(null, { hook: 'data-close-dialog', label: 'Done' });
        q('[data-import-preview]').hidden = true;
        const file = event.target.files[0];
        if (!file) return;
        if (file.size >= root.WayfarersStorage.MAX_BYTES) { set('[data-dialog-notice]', 'Choose a save smaller than 1 MB. Your guild has not changed.'); return; }
        try { const text = await file.text(); if (!signal.aborted && dialogKind === 'settings') reviewImport(text); }
        catch (error) { set('[data-dialog-notice]', 'The file could not be read. Try pasting its text instead.'); }
      }
    }, { signal });
    element.addEventListener('input', event => {
      if (event.target.id === 'wg-save-text') {
        pendingImport = null;
        q('[data-import-preview]').hidden = true;
      }
    }, { signal });

    dialog.addEventListener('keydown', event => {
      if (event.key !== 'Tab') return;
      const focusable = Array.from(dialog.querySelectorAll('button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),textarea:not(:disabled),summary,[tabindex="0"]')).filter(node => !node.hidden && node.getClientRects().length);
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (!first) { event.preventDefault(); return; }
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }, { signal });
    dialog.addEventListener('cancel', event => { event.preventDefault(); backSheet(); }, { signal });
    dialog.addEventListener('close', () => { if (dialog.open) return; delete q('[data-dialog-body]').dataset.resetSignature; if (dialogKind) closeDialog(); }, { signal });
    dialog.addEventListener('click', event => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeDialog();
    }, { signal });
    document.addEventListener('visibilitychange', () => {
      if (disposed) return;
      advance();
      save();
      if (!document.hidden) {
        view = core.getView(state);
        render();
      }
    }, { signal });
    document.addEventListener('freeze', () => { advance(); save(); }, { signal });
    root.addEventListener('pagehide', () => { advance(); save(); }, { signal });
    root.addEventListener('storage', event => {
      if (event.key === root.WayfarersStorage.SAVE_KEY) save();
    }, { signal });
    motionQuery.addEventListener('change', render, { signal });
    const timer = root.setInterval(() => {
      if (document.hidden || disposed) return;
      advance();
      render();
      if (Date.now() - lastSave >= 15000) save();
      retryReceipts();
    }, 1000);
    qa('[data-tab]').forEach(button => { const id = button.dataset.tab; setMarkup(button, icon(id === 'journey' ? 'journal' : id) + '<span>' + (id === 'journey' ? 'Journal' : id[0].toUpperCase() + id.slice(1)) + '</span>'); });
    setMarkup(q('[data-open="shop"]'), icon('shop') + '<span>Shop</span>');

    setMarkup(q('[data-open="settings"]'), icon('settings'));
    setMarkup(q('[data-profession-flow]'), '<span>' + icon('ore') + 'Ore</span><b aria-hidden="true">→</b><span>' + icon('tools') + 'Equipment</span><b aria-hidden="true">→</b><span>' + icon('trail') + 'Routes</span>');
    if (root.WayfarersExpeditionUI && root.WayfarersExpeditionScene && view.expedition && view.expedition.local) {
      expeditionUI = root.WayfarersExpeditionUI.create({
        element:q('[data-game]'), perform, quiet,
        openLegacy:openDialog,
        overlayOpen:() => dialog.open || !q('[data-find-feedback]').hidden,
        nativeOptions:() => q('.wg-exit').click()
      });
      scene.destroy();
    }
    if (!expeditionUI && loaded.offline && loaded.offline.seconds >= 60) renderReturnSummary(loaded.offline.summary, loaded.offline.seconds, loaded.offline.gains);
    render();
    if (expeditionUI) expeditionUI.showReturn(loaded.offline);
    if (rewarded) rewarded.refresh().catch(() => announce('The caravan service could not connect. Saved deliveries will retry when it reconnects.'));
    if (billing && root.WayfarersPlayBilling) billingAction('refresh');
    if (loaded.status === 'new') save();
    if (loaded.message) { storageNotice(loaded.message); announce(loaded.message); }
    if (loaded.offline && loaded.offline.seconds >= 60) {
      const seconds = loaded.offline.seconds;
      const time = seconds >= 3600 ? (seconds / 3600).toFixed(1) + ' hours' : Math.floor(seconds / 60) + ' minutes';
      const gains = Object.keys(loaded.offline.gains || {}).map(id => '+' + core.format(loaded.offline.gains[id]) + ' ' + id).slice(0, 4).join(', ');
      announce('Welcome back. ' + time + ' of guild progress processed. Resource increases after any automatic spending: ' + (gains || 'none') + '.' + (loaded.offline.pendingSeconds > 1 ? ' Catching up remaining guild activity.' : ''));
    }
    root.WayfarersUI = { handleBack };
    active = {
      element,
      destroy() {
        if (disposed) return;
        advance();
        save();
        disposed = true;
        root.clearInterval(timer);
        root.clearTimeout(teachingTimer);
        root.clearTimeout(noticeTimer);
        controller.abort();
        unsubscribeBilling();
        unsubscribeRewarded();
        if (rewarded) rewarded.destroy();
        if (dialog.open) closeDialog();
        if (billing) billing.destroy();
        if (expeditionUI) expeditionUI.dispose();
        scene.destroy();
        if (audio) audio.close().catch(() => {});
        overviewScenes.forEach(renderer => renderer.destroy());
        if (dialog.open) dialog.close();
        if (root.WayfarersUI && root.WayfarersUI.handleBack === handleBack) delete root.WayfarersUI;
        active = null;
      }
    };
  }
  if (root.SiteRoutes && typeof root.SiteRoutes.register === 'function') root.SiteRoutes.register('games:wayfarers-guild', { mount, unmount() { if (active) active.destroy(); } });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true });
  else mount();
})(globalThis);
