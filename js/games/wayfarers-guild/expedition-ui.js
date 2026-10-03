(function (root) {
  'use strict';
  const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
  const iconNames = { porters:'backpack', picks:'tools', carts:'crate', furnace:'forge', lift:'equipment', beacon:'observatory', atlas:'maps' };
  const icon = name => root.WayfarersIcons.markup(iconNames[name] || name);
  const propCells = { boots:2, porters:3, picks:4, carts:5, furnace:6, lift:7, crew:1, beacon:9 };
  const areaNames = { greenway:'Trail', quarry:'Quarry', watchtower:'Tower', workshop:'Workshop', ruins:'Ruins', harbor:'Harbor', guild:'Guild' };
  const areaIcons = { greenway:'trail', quarry:'mine', watchtower:'observatory', workshop:'tools', ruins:'relic', harbor:'caravan', guild:'guild' };
  const groupNames = { local:'Area upgrades', area:'Area upgrades', development:'Developments', developments:'Developments', research:'Research', legacy:'Guild upgrades', equipment:'Equipment', production:'Production', blueprint:'Blueprints', blueprints:'Blueprints' };
  function localIcon(id, fallback) {
    const cell = propCells[id];
    return cell == null ? icon(fallback || id) : '<span class="wx-prop" aria-hidden="true" style="background-image:url(&quot;img/wayfarers-guild/expedition-props.png&quot;);background-position:' + ((cell % 4) * 100 / 3) + '% ' + (Math.floor(cell / 4) * 50) + '%"></span>';
  }
  function gainLabel(item) {
    const progression = item.catalogId && item.quantity;
    const track = item.trackId || item.id;
    const fields = (progression ? { boots:['travel','flow'],porters:['capacity','coins'],scouts:['research','maps'],caravans:['freight'],waystations:['capacity','herbs'],railways:['freight'],picks:['picks'],carts:['carts','capacity'],furnace:['furnace','ore'],geology:['ore'],recovery:['ore'],deepworks:['picks'],beacon:['research'],signals:['coordination','maps'],crew:['coordination','capacity','maps'],optics:['research'],forecasting:['travel','duration'], 'relay-grid':['coordination','knowledge'],assembly:['flow','assembly'],toolmaking:['picks','research'],metallurgy:['provisions'],mechanisms:['picks','assembly'],precision:['demand','research'],replication:['capacity','provisions'],delving:['delving'],archaeology:['interpretation','research'],'recovery-teams':['recovery','flow'],restoration:['knowledge'],attunement:['ore','coins','research'],resonance:['capacity','research'],shipbuilding:['cargo','capacity'],seamanship:['travel','duration'],stowage:['cargo'],contracts:['coins'],navigation:['travel','duration'],'fleet-command':['capacity','cargo'] } : { boots:['exploration','travel'],porters:['coins'],scouts:['exploration','maps'],picks:['extraction','picks'],carts:['haul','carts'],furnace:['smelting','furnace','ore'],crew:['knowledge','construction'],lift:['construction','repair'],beacon:['knowledge','protection'] })[track] || [];
    const changes = (item.impact || []).filter(change => change.current != null && change.next != null);
    const change = fields.map(field => changes.find(candidate => String(candidate.metric).split(':').pop() === field)).find(Boolean) || changes[0];
    if (change) {
      const current = root.WayfarersCore.Numbers.toNumber(change.currentValue ?? change.current);
      const next = root.WayfarersCore.Numbers.toNumber(change.nextValue ?? change.next);
      const key = String(change.metric).split(':').pop();
      const effect = ({ picks:'mining', carts:'haul', furnace:'smelt', flow:item.areaId === 'greenway' ? 'travel' : 'output',capacity:item.areaId === 'greenway' && track === 'porters' ? 'cargo' : 'capacity',demand:'input demand',travel:item.areaId === 'harbor' ? 'sailing speed' : 'travel',construction:item.description?.includes('Expansion') ? 'projects' : 'build' }[key] || String(change.label || '').split('·').pop().trim().toLowerCase().replace('exploration','travel').replace('extraction','mining').replace('hauling','haul').replace('smelting','smelt').replace('construction','build')).replace(' capacity','');
      if (current > 0 && next !== current) { const delta = (next / current - 1) * 100, magnitude = Math.abs(delta); return (delta < 0 ? '−' : '+') + (magnitude < .05 ? '<0.1' : magnitude < 1 ? magnitude.toFixed(1) : Math.round(magnitude)) + '% ' + effect; }
      if (next > 0) return change.next + ' ' + effect + (change.unit || '');
    }
    if (progression && !changes.length && !item.disabled) return 'Current plan unchanged';
    const rates = String(item.comparison || '').match(/^([\d.]+) → ([\d.]+)/);
    const label = item.effectText === 'Bridge repair capacity' ? 'repair' : { boots:'travel', porters:'coins', scouts:'travel', picks:'mining', carts:'haul', furnace:'smelt', crew:'build', lift:'build', beacon:'light' }[item.id] || 'work';
    return rates && Number(rates[1]) > 0 ? '+' + Math.round((Number(rates[2]) / Number(rates[1]) - 1) * 100) + '% ' + label : item.effectText;
  }
  const labels = { boots:'Boots', preparation:'Field kit', miners:'Miners', forge:'Workshop', foragers:'Foragers', cooks:'Cooks', scholars:'Scholars', mentors:'Mentors', surveyors:'Surveyors', 'gear-tools':'Tools', 'gear-boots':'Boots', 'gear-instruments':'Instruments' };
  const visible = items => (items || []).filter(item => item.visible !== false);
  const name = item => labels[item.id] || item.name || item.label || item.id;
  const percent = value => Math.max(0, Math.min(100, Number(value || 0) * 100));
  function compact(value) {
    const number = root.WayfarersCore.Numbers.toNumber(value);
    if (number < 1000) return String(Math.floor(number));
    if (!Number.isFinite(number) || number >= 1e15) return root.WayfarersCore.format(value);
    let power = Math.floor(Math.log10(number) / 3);
    let scaled = number / Math.pow(1000, power);
    if (scaled >= 999.95 && power < 4) { power += 1; scaled /= 1000; }
    return scaled.toFixed(1).replace(/\.0$/, '') + ['', 'K', 'M', 'B', 'T'][power];
  }

  function create(options) {
    const host = options.element;
    const controller = new AbortController();
    const commands = new Map();
    const descriptors = new Map();
    const stack = [];
    let view;
    let screen = 'expedition';
    let upgradeArea = 'all';
    let upgradeEffect = 'all';
    let upgradeSearch = '';
    let showOwnedUpgrades = false;
    let currentContext = {};
    let previousScreen = null;
    let sheet = null;
    let opener = null;
    let toastTimer;
    let previousRanks = {};
    let previousStage = null;
    let displayedArea = null;
    const areaScroll = new Map();
    const earnedTracks = new Map();
    const activePointers = new Set();
    let swipe = null;
    let suppressSceneTapUntil = 0;
    let knownAreaCount = null;
    let swipeTaught = false;
    const celebrated = new Set();
    const viewedStages = new Set();
    let seenSequence = null;
    let disposed = false;
    let detailFooter = '';
    const shell = document.createElement('section');
    shell.className = 'wx-game';
    shell.setAttribute('aria-label', 'Wayfarers Guild');
    shell.innerHTML = '<header class="wx-header"><button class="wx-wallet" data-wx-wallet aria-label="Resource stockpile"></button><div class="wx-local-count" data-wx-local-count hidden></div><div class="wx-utilities"><button data-wx-reward aria-label="Caravan reward" hidden>' + icon('caravan') + '</button><button data-wx-options aria-label="Settings and app updates">' + icon('settings') + '</button></div></header>' +
      '<div class="wx-objective"><button class="wx-area-step" data-wx-prev hidden aria-label="Previous area">‹</button><button data-wx-objective><span data-wx-stage-name></span><strong data-wx-objective-text></strong></button><button class="wx-area-step" data-wx-next hidden aria-label="Next area">›</button><progress data-wx-progress max="100" value="0" aria-label="Expedition completion"></progress></div>' +
      '<div class="wx-play"><div class="wx-world"><canvas data-wx-canvas role="img" aria-label="Your expedition"></canvas><button class="wx-world-label" data-wx-world-label aria-label="Current area objective"></button><button class="wx-network-link" data-wx-network hidden></button><button class="wx-focus-trigger" data-wx-focus hidden></button><div class="wx-hotspots" data-wx-hotspots></div><div class="wx-world-actions" data-wx-world-actions></div></div><section class="wx-dock" aria-label="Area upgrades"><div class="wx-dock-heading"><strong data-wx-upgrade-count>Upgrades</strong><button data-wx-batch hidden></button></div><div class="wx-tray" data-wx-tray tabindex="0" aria-label="Area upgrades; scroll for more"></div></section></div>' +
      '<section class="wx-destination" data-wx-destination hidden></section><nav class="wx-nav" aria-label="Game destinations"><button data-wx-nav="expedition">' + icon('maps') + '<span>Areas</span><i data-wx-attention hidden aria-label="New development"></i></button><button data-wx-nav="upgrades" hidden>' + icon('research') + '<span>Upgrades</span></button><button data-wx-nav="guild" hidden>' + icon('guild') + '<span>Guild</span></button></nav>' +
      '<div class="wx-toast" data-wx-toast role="status" aria-live="polite"></div><div class="wx-sr-status" data-wx-area-status role="status" aria-live="polite"></div><button class="wx-save-alert" data-wx-save-alert hidden>Save needs attention</button>';
    const dialog = document.createElement('dialog');
    dialog.className = 'wx-sheet';
    dialog.setAttribute('aria-labelledby', 'wx-sheet-title');
    dialog.innerHTML = '<header><button data-wx-back aria-label="Back">‹</button><h2 id="wx-sheet-title"></h2><button data-wx-close aria-label="Close">×</button></header><div class="wx-sheet-content" data-wx-sheet-content></div><footer class="wx-purchase-footer" data-wx-sheet-footer hidden></footer>';
    host.dataset.expeditionMode = 'true';
    host.append(shell);
    host.parentElement.append(dialog);
    const q = selector => shell.querySelector(selector);
    const scene = root.WayfarersExpeditionScene.create(q('[data-wx-canvas]'), {
      quiet: !!options.quiet,
      onSelect(target) { if (root.performance.now() < suppressSceneTapUntil || swipe || activePointers.size > 1 || dialog.open || options.overlayOpen()) return; if (target.kind === 'upgrade') inspectLocal(target.id); else if (target.kind === 'choice') open({ kind:'choice' }); }
    });
    function markup(node, html) {
      if (node.dataset.markup === html) return;
      const focus = node.contains(document.activeElement) && document.activeElement.dataset.wxDo;
      const searching = node.contains(document.activeElement) && document.activeElement.hasAttribute('data-wx-search');
      const caret = searching ? document.activeElement.selectionStart : null;
      node.innerHTML = html;
      node.dataset.markup = html;
      if (focus) Array.from(node.querySelectorAll('[data-wx-do]')).find(button => button.dataset.wxDo === focus)?.focus({ preventScroll:true });
      if (searching) { const input = node.querySelector('[data-wx-search]'); input?.focus({ preventScroll:true }); if (input && caret != null) input.setSelectionRange(caret, caret); }
    }
    function text(node, value) { if (node.textContent !== String(value)) node.textContent = value; }
    function command(action, key) {
      const id = key || JSON.stringify(action);
      commands.set(id, action);
      return ' data-wx-do="' + esc(id) + '"';
    }
    function button(label, action, settings) {
      const cfg = settings || {};
      return '<button type="button"' + command(action, cfg.key) + ' class="' + esc(cfg.className || 'wx-button') + '"' + (cfg.disabled ? ' disabled' : '') + (cfg.selected ? ' aria-pressed="true"' : '') + (cfg.aria ? ' aria-label="' + esc(cfg.aria) + '"' : '') + '>' + label + '</button>';
    }
    function costs(item) {
      return (item.cost || []).map(cost => '<span>' + icon(cost.resource) + esc(compact(cost.amount)) + '</span>').join('');
    }
    function focusInput(item) {
      return item.text || (item.resource ? item.resource[0].toUpperCase() + item.resource.slice(1) : 'Input') + ' ' + metric(item.current) + ' → ' + metric(item.next) + (item.unit || '');
    }
    function focusInputs(item) {
      const changed = (item.inputs || []).filter(input => Math.abs(Number(input.next) - Number(input.current)) > Math.max(1,Math.abs(Number(input.current))) * 1e-9);
      return changed.length ? 'Input rates: ' + changed.map(focusInput).join(' · ') : 'Input rates unchanged';
    }
    function purchaseLabel(item, compactRow) {
      if (item.quantity && !(item.cost || []).length) return '<span class="wx-batch-unavailable">×' + esc(item.quantity) + '<br>Unavailable</span>';
      return (!compactRow && item.quantity > 1 ? '<small class="wx-quantity">×' + esc(item.quantity) + '</small>' : '') + costs(item);
    }
    function batchAvailable() { return (view.expedition.batch?.options || []).some(item => item.unlocked && item.count > 1); }
    function batchControl() {
      return batchAvailable() ? button('Buy ×' + (view.expedition.batch.selected || 1) + ' ▾', () => open({ kind:'batch' }), { key:'batch', className:'wx-batch', aria:'Purchase quantity: ' + (view.expedition.batch.selected || 1) + '. Change exact quantity.' }) : '';
    }
    function descriptorKey(item) { return JSON.stringify(item.action || { id:item.id }); }
    function tiles(items) {
      return '<div class="wx-catalog">' + visible(items).map(item => {
        const id = descriptorKey(item);
        descriptors.set(id, item);
        const owned = item.owned && item.maxed;
        const caption = item.selected ? 'Active' : owned ? 'Complete' : item.owned && item.disabled && !item.quantity ? 'Owned' : (item.cost || []).length ? purchaseLabel(item) : 'Choose';
        return '<article class="wx-card" data-ready="' + (!item.disabled && !owned) + '">' +
          button(icon(item.icon || item.id) + '<strong>' + esc(name(item)) + '</strong>' + (item.level ? '<small>Rank ' + esc(item.level) + '</small>' : ''), () => open({ kind:'inspect', id }), { key:'inspect:' + id, className:'wx-card-info', aria:'Inspect ' + name(item) }) +
          '<span class="wx-card-effect">' + esc(item.effectText || item.description || '') + '</span>' +
          button(caption, item.action, { disabled: item.disabled || owned, className:'wx-price', aria:(item.selected ? 'Selected ' : item.quantity ? 'Buy ' + item.quantity + ' ranks of ' : 'Use ') + name(item) + ((item.cost || []).length ? ': ' + item.cost.map(c => c.text).join(', ') : '') }) + '</article>';
      }).join('') + '</div>';
    }
    function menu(label, detail, glyph, model) {
      return button(icon(glyph) + '<span><strong>' + esc(label) + '</strong><small>' + esc(detail || '') + '</small></span><b aria-hidden="true">›</b>', () => open(model), { key:'menu:' + JSON.stringify(model), className:'wx-menu' });
    }
    function section(title, html) { return html ? '<section class="wx-group"><h3>' + esc(title) + '</h3>' + html + '</section>' : ''; }
    function catalog(title, items) { const shown = visible(items); return shown.length ? section(title, tiles(shown)) : ''; }
    function metric(value) {
      if (typeof value === 'string') return value;
      if (value && typeof value === 'object') return root.WayfarersCore.format(value);
      if (!Number.isFinite(value)) return '—';
      return Math.abs(value) >= 10000 ? compact(value) : Number(value.toFixed(3)).toLocaleString();
    }
    function impacts(item) {
      return (item.impact || []).filter(change => change.current != null && change.next != null).map(change => { const values = impactValues(change); return '<div><span>' + esc(change.label || change.metric) + '</span><strong>' + esc(values[0]) + ' → ' + esc(values[1]) + (change.unit ? ' <small>' + esc(change.unit) + '</small>' : '') + '</strong></div>'; }).join('');
    }
    function impactValues(change) {
      let values = [metric(change.current),metric(change.next)];
      if (values[0] === values[1] && change.currentValue && change.nextValue && root.WayfarersCore.Numbers.cmp(change.currentValue,change.nextValue) !== 0) {
        for (let digits = 3; digits <= 8 && values[0] === values[1]; digits += 1) {
          values = [change.currentValue,change.nextValue].map(value => { const number = root.WayfarersCore.Numbers.toNumber(value); return Number.isFinite(number) ? String(Number(number.toPrecision(digits))) : Number(value.m).toPrecision(digits) + 'e' + value.e; });
        }
      }
      return values;
    }
    function areaLinks(item) {
      let source = [...new Set(item.sourceAreas || [])];
      const target = [...new Set(item.targetAreas || (item.areaId ? [item.areaId] : []))];
      if (source.length === 1 && target.length === 1 && source[0] === target[0]) source = [];
      if (!source.length && !target.length) return '';
      const side = ids => ids.map(id => '<span title="' + esc(areaNames[id] || id) + '">' + icon(areaIcons[id] || 'guild') + '<span class="wx-link-name">' + esc(areaNames[id] || id) + '</span></span>').join('');
      return '<span class="wx-area-links" aria-label="' + esc(source.map(id => areaNames[id] || id).join(' + ') + (source.length && target.length ? ' supports ' : '') + target.map(id => areaNames[id] || id).join(' + ')) + '">' + side(source) + (source.length && target.length ? '<b aria-hidden="true">→</b>' : '') + side(target) + '</span>';
    }
    function completedUpgrade(item) {
      const cap = item.maxLevel ?? item.maxRank;
      const level = item.level ?? item.rank ?? 0;
      return item.maxed === true || cap != null && Number(level) >= Number(cap) || !!item.owned && !['buy','expedition-buy','refit-upgrade','legacy-upgrade'].includes(item.action?.type) && !item.repeatable;
    }
    function upgradeRows(items) {
      return '<div class="wx-upgrade-list">' + items.map(item => {
        const complete = completedUpgrade(item);
        const running = view.expedition.commission?.id === item.action?.id && item.action?.type === 'expedition-development';
        const level = item.level == null ? item.rank : item.level;
        const label = name(item);
        const change = (item.impact || []).find(entry => entry.current != null && entry.next != null && !String(entry.metric).startsWith('behavior:'));
        const values = change ? impactValues(change) : [];
        const effect = change ? (change.label || change.metric) + ' ' + values[0] + ' → ' + values[1] + (change.unit || '') : item.effectText || item.description || '';
        const prerequisite = (item.dependencies || []).find(dependency => !dependency.met);
        return '<article class="wx-research-row" data-wx-upgrade="' + esc(item.id) + '" data-ready="' + (!item.disabled && !complete) + '">' +
          button(icon(item.icon || item.trackId || item.id) + '<span><strong>' + esc(label) + (level ? '<small> ' + level + '</small>' : '') + '</strong><span class="wx-research-effect">' + esc(effect) + '</span>' + areaLinks(item) + (prerequisite ? '<small class="wx-prerequisite">' + esc(prerequisite.label) + '</small>' : '') + '</span>', () => open({ kind:'upgrade', id:item.id }), { key:'upgrade-detail:' + item.id, className:'wx-research-info', aria:'Details: ' + label }) +
          button(complete ? '✓' : running ? Math.floor(percent(view.expedition.commission.progress)) + '%' : (item.cost || []).length ? purchaseLabel(item) : 'Unlock', item.action, { disabled:item.disabled || complete || currentContext.awaitingWallet, className:'wx-price', aria:(complete ? 'Completed ' : running ? 'Researching ' : 'Buy ' + (item.quantity || 1) + ' ranks of ') + label + (!running && (item.cost || []).length ? ', ' + item.cost.map(cost => cost.text).join(', ') : '') }) + '</article>';
      }).join('') + '</div>';
    }
    function upgradesBody() {
      const all = visible(view.globalUpgrades || []);
      const areas = (view.expedition.areas || []).filter(area => area.unlocked);
      if (upgradeArea !== 'all' && upgradeArea !== 'guild' && !areas.some(area => area.id === upgradeArea)) upgradeArea = 'all';
      const matchingArea = all.filter(item => upgradeArea === 'all' || item.areaId === upgradeArea || (item.targetAreas || []).includes(upgradeArea) || upgradeArea === 'guild' && !String(item.action?.type || '').startsWith('expedition-'));
      const effects = [...new Set(matchingArea.map(item => item.effectKind).filter(Boolean))];
      if (upgradeEffect !== 'all' && !effects.includes(upgradeEffect)) upgradeEffect = 'all';
      const search = upgradeSearch.trim().toLowerCase();
      const filtered = matchingArea.filter(item => (upgradeEffect === 'all' || item.effectKind === upgradeEffect) && (!search || [name(item),item.effectText,item.description,groupNames[item.group] || item.group,...(item.sourceAreas || []).map(id => areaNames[id] || id),...(item.targetAreas || []).map(id => areaNames[id] || id)].join(' ').toLowerCase().includes(search)));
      const filter = (label, id, type) => button(esc(label), () => { if (type === 'area') upgradeArea = id; else upgradeEffect = id; update(view); q('[data-wx-destination]').scrollTop = 0; }, { key:'filter:' + type + ':' + id, selected:(type === 'area' ? upgradeArea : upgradeEffect) === id, className:'wx-filter' });
      const human = value => ({ production:'Production', throughput:'Production', travel:'Travel', conversion:'Conversion', capacity:'Capacity', automation:'Automation', unlock:'Unlocks', research:'Research', synergy:'Connections', income:'Income', work:'Work', construction:'Building', protection:'Protection' }[value] || String(value).replace(/[-_]/g,' ').replace(/^./, c => c.toUpperCase()));
      let html = '<div class="wx-upgrade-tools"><label class="wx-upgrade-search">' + icon('research') + '<input data-wx-search type="search" aria-label="Find an upgrade" placeholder="Find an upgrade" value="' + esc(upgradeSearch) + '"></label>' + button('Filter' + (upgradeEffect !== 'all' ? ' ●' : ''), () => open({ kind:'upgrade-filters', effects }), { key:'upgrade-filters', className:'wx-filter-trigger', aria:'Filter by upgrade effect' }) + '<div class="wx-filter-strip wx-area-filters" aria-label="Affected area">' + filter('All','all','area') + areas.map(area => filter(areaNames[area.id] || area.label,area.id,'area')).join('') + filter('Guild','guild','area') + '</div>';
      html += button(esc(upgradeArea === 'all' ? 'All areas' : areaNames[upgradeArea] || upgradeArea) + ' ▾', () => open({kind:'upgrade-areas'}), {key:'upgrade-areas',className:'wx-area-filter-trigger',aria:'Filter upgrades by area'}) + batchControl() + '</div>';
      if (view.expedition.commission) {
        const commission = view.expedition.commission;
        html += menu(commission.name, 'Researching · ' + Math.floor(percent(commission.progress)) + '%', 'research', {kind:'upgrade',id:'development:' + commission.id});
      }
      const available = filtered.filter(item => !completedUpgrade(item));
      const groups = [...new Set(available.map(item => item.group || 'development'))];
      html += groups.map(group => section(groupNames[group] || human(group), upgradeRows(available.filter(item => (item.group || 'development') === group)))).join('');
      if (!available.length) html += '<p class="wx-empty">' + (search ? 'No matching upgrades.' : 'More developments open as your areas grow.') + '</p>';
      const owned = filtered.filter(completedUpgrade);
      if (owned.length) html += button((showOwnedUpgrades ? 'Hide completed' : 'Completed') + ' · ' + owned.length, () => { showOwnedUpgrades = !showOwnedUpgrades; update(view); }, { key:'owned-upgrades', className:'wx-owned-toggle', selected:showOwnedUpgrades }) + (showOwnedUpgrades ? upgradeRows(owned) : '');
      return html;
    }
    function legacy(kind) { close(); options.openLegacy(kind); }
    function open(model, replace) {
      cancelSwipe();
      if (!dialog.open) { opener = document.activeElement; stack.length = 0; }
      else if (!replace && sheet) stack.push(sheet);
      sheet = model;
      renderSheet();
      if (!dialog.open) dialog.showModal();
      dialog.querySelector('[data-wx-sheet-content]').scrollTop = 0;
    }
    function close() {
      sheet = null;
      stack.length = 0;
      if (dialog.open) dialog.close();
      if (opener?.isConnected && opener.getClientRects().length) opener.focus({ preventScroll:true });
    }
    function back() {
      if (stack.length) { sheet = stack.pop(); renderSheet(); return; }
      close();
    }
    function inspectLocal(id) { const item = view.expedition.cards.find(card => card.id === id); if (item) open({ kind:'local', id, areaId:item.action?.areaId || view.expedition.stage.kind, catalogId:item.catalogId }); }
    function showUpgrades(areaId, effect) {
      close(); upgradeArea = areaId || 'all'; upgradeEffect = effect === 'research' ? 'all' : effect || 'all'; upgradeSearch = effect === 'research' ? 'research' : ''; screen = 'upgrades'; update(view);
      q('[data-wx-destination]').scrollTop = 0;
    }
    function selectArea(areaId) {
      const area = (view.expedition.areas || []).find(item => item.id === areaId);
      if (!area || !area.unlocked) return;
      close(); screen = 'expedition'; execute(area.action || { type:'expedition-select', areaId }); update(view);
    }
    function stepArea(direction) {
      const areas = (view?.expedition.areas || []).filter(area => area.unlocked);
      const index = areas.findIndex(area => area.selected || area.id === displayedArea);
      if (areas[index + direction]) selectArea(areas[index + direction].id);
    }
    function cancelSwipe() {
      if (swipe?.moved) suppressSceneTapUntil = root.performance.now() + 500;
      swipe = null;
      q('.wx-world').style.removeProperty('--wx-swipe');
    }
    function startPointer(event) {
      activePointers.add(event.pointerId);
      if (activePointers.size > 1) { suppressSceneTapUntil = root.performance.now() + 1000; cancelSwipe(); return; }
      if (event.target !== q('[data-wx-canvas]') || event.button > 0 || screen !== 'expedition' || dialog.open || options.overlayOpen()) return;
      const bounds = q('.wx-world').getBoundingClientRect();
      const edge = 28;
      if (event.clientX < edge || event.clientX > root.innerWidth - edge || event.clientX < bounds.left + 12 || event.clientX > bounds.right - 12) { suppressSceneTapUntil = root.performance.now() + 500; return; }
      swipe = { id:event.pointerId, x:event.clientX, y:event.clientY, moved:false, cancelled:false, locked:false };
    }
    function movePointer(event) {
      if (!swipe || event.pointerId !== swipe.id) return;
      const dx = event.clientX - swipe.x, dy = event.clientY - swipe.y;
      if (Math.hypot(dx,dy) < 12) return;
      swipe.moved = true;
      if (!swipe.locked && (Math.abs(dy) > Math.abs(dx) * .55)) { swipe.cancelled = true; return; }
      if (swipe.cancelled) return;
      swipe.locked = true;
      if (event.cancelable) event.preventDefault();
    }
    function endPointer(event) {
      activePointers.delete(event.pointerId);
      if (!swipe || event.pointerId !== swipe.id) return;
      const gesture = swipe;
      const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y;
      const threshold = Math.max(48, Math.min(80, q('.wx-world').clientWidth * .16));
      const navigate = !gesture.cancelled && event.type === 'pointerup' && Math.abs(dx) >= threshold && Math.abs(dx) >= Math.abs(dy) * 1.8;
      if (gesture.moved || navigate || event.type !== 'pointerup') suppressSceneTapUntil = root.performance.now() + 500;
      swipe = null;
      if (navigate) { event.preventDefault(); stepArea(dx < 0 ? 1 : -1); }
    }
    function execute(action) {
      if (typeof action === 'function') { action(); return; }
      if (!action) return;
      if (['challenge'].includes(action.type)) close();
      const result = options.perform(action);
      if (result?.ok && action.type === 'expedition-choice') { notify('Plan selected'); back(); }
      if (result?.ok && ['expedition-config','expedition-specialize'].includes(action.type)) notify('Plan updated');
      if (result?.ok && action.type === 'expedition-batch') back();
      if (result?.ok && action.type === 'expedition-focus') close();
      if (result?.ok && ['expedition-next','expedition-expand','expedition-select'].includes(action.type)) { close(); screen = 'expedition'; update(view); }
      if (result?.ok && action.type === 'route') { close(); screen = 'expedition'; update(view); }
    }
    function choiceBody(choice) {
      if (!choice || !choice.visible) return '<p>This decision appears at the next checkpoint.</p>';
      const preferred = choice.id === 'smelting' ? ['guild:ore','quarry:furnace','guild:knowledge','guild:coins'] : choice.id === 'dispatch' ? ['guild:coins','quarry:carts','guild:maps'] : ['guild:coins','guild:ore','guild:knowledge'];
      const previews = item => {
        const changes = item.impact || [];
        const ordered = preferred.map(id => changes.find(change => change.metric === id)).filter(Boolean);
        const chosen = ordered.concat(changes.filter(change => !ordered.includes(change))).slice(0,3);
        return impacts({ impact:chosen.map(change => Object.assign({},change,{label:String(change.label || change.metric).replace('Guild · Total ','Guild ').replace('Quarry · Smelting capacity','Smelt capacity').replace('Quarry · Hauling capacity','Haul capacity')})) });
      };
      let html = '<div class="wx-options">' + choice.options.map(item => button(icon(item.icon || item.id) + '<strong>' + esc(item.label) + '</strong><span>' + esc(item.effectText) + '</span>' + ((item.impact || []).length ? '<span class="wx-impact wx-choice-impact">' + previews(item) + '</span>' : '') + (item.selected ? '<small>Selected</small>' : ''), item.action, { selected:item.selected, disabled:item.disabled, className:'wx-choice' })).join('') + '</div>';
      if (choice.options.some(item => (item.impact || []).length > 3)) html += button('Full rate comparison', () => open({ kind:'choice-metrics', id:choice.id }), { key:'choice-metrics:' + choice.id, className:'wx-compare' });
      return html;
    }
    function detailBody(item, local) {
      if (!item) return '<p>This upgrade is no longer available.</p>';
      const unmet = (item.dependencies || []).some(dependency => !dependency.met);
      const running = view.expedition.commission?.id === item.action?.id && item.action?.type === 'expedition-development';
      let html = '<div class="wx-detail-hero">' + icon(item.icon || item.trackId || item.id) + '<strong>' + esc(local ? 'Rank ' + (item.rank ?? item.level ?? 0) + ' / ' + (item.maxRank ?? item.maxLevel) : running ? 'Researching · ' + Math.floor(percent(view.expedition.commission.progress)) + '%' : item.selected ? 'Active' : completedUpgrade(item) ? 'Complete' : unmet ? 'Locked' : item.level ? 'Rank ' + item.level : item.disabled ? 'Save up' : 'Available') + '</strong></div><p class="wx-effect">' + esc(item.effectText || item.description) + '</p>';
      if (item.description && item.effectText && !item.description.startsWith(item.effectText)) html += '<p class="wx-muted">' + esc(item.description) + '</p>';
      html += areaLinks(item);
      if ((item.impact || []).length) html += '<div class="wx-impact">' + impacts(item) + '</div>';
      else if (item.quantity && item.group === 'area' && !item.disabled) html += '<p class="wx-muted">No immediate rate change in this quote.</p>' + button('Review area plans', () => { selectArea(item.areaId); open({kind:'choice'}); }, {key:'review-area-plan:' + item.areaId});
      if ((item.dependencies || []).length) html += '<ul class="wx-dependencies">' + item.dependencies.map(dependency => '<li data-met="' + !!dependency.met + '"><b aria-hidden="true">' + (dependency.met ? '✓' : '○') + '</b>' + esc(dependency.label) + '</li>').join('') + '</ul>';
      if (item.comparison && !(item.impact || []).length) html += '<p>' + esc(typeof item.comparison === 'string' ? item.comparison : item.comparison.text) + '</p>';
      if (item.chainOutput) html += '<p class="wx-muted">Whole chain: ' + item.chainOutput.current.toFixed(2) + ' → ' + item.chainOutput.next.toFixed(2) + ' ingots/s</p>';
      if (item.nextMilestone) html += '<div class="wx-milestone">' + icon('mastery') + '<span>' + esc(typeof item.nextMilestone === 'string' ? item.nextMilestone : 'Lv ' + item.nextMilestone.rank + ' · ' + (item.nextMilestone.label || item.nextMilestone.effectText)) + '</span></div>';
      if (item.reason || item.shortageText) html += '<p class="wx-muted">' + esc(item.shortageText || item.reason) + '</p>';
      const rankCap = item.maxRank ?? item.maxLevel;
      const rangeValid = item.rankAfter != null && (rankCap == null || item.rankAfter <= rankCap);
      const quantityLabel = item.quantity ? 'Exactly ' + item.quantity + ' rank' + (item.quantity === 1 ? '' : 's') + (rangeValid ? ' · ' + (item.rank ?? item.level ?? 0) + ' → ' + item.rankAfter : '') : '';
      const exactCost = (item.cost || []).map(cost => cost.text).join(' · ');
      detailFooter = '<div class="wx-purchase-summary">' + (quantityLabel ? '<strong>' + esc(quantityLabel) + '</strong>' : '') + (exactCost ? '<small>' + (running ? 'Funded: ' : 'Cost: ') + esc(exactCost) + '</small>' : '') + (item.quantity && !rangeValid && item.reason ? '<small>' + esc(item.reason) + '</small>' : '') + '</div><div class="wx-purchase-controls">' + (item.quantity && batchAvailable() ? batchControl() : '') + button(completedUpgrade(item) ? 'Complete' : running ? 'Researching' : item.quantity ? 'Buy ×' + item.quantity : exactCost ? 'Buy upgrade' : item.selected ? 'Selected' : 'Choose', item.action, { disabled:item.disabled || completedUpgrade(item) || currentContext.awaitingWallet, className:'wx-confirm',aria:(running ? 'Researching ' : item.quantity ? 'Buy exactly ' + item.quantity + ' ranks of ' : 'Buy ') + (item.label || item.name || item.id) + (!running && exactCost ? ', ' + exactCost : '') }) + '</div>';
      if (!local && !running && view.planning?.unlocked && ['buy','research','project','expedition-development'].includes(item.action?.type) && !item.owned) {
        html += section('Planning', button('Save for this', { type:'plan-goal', action:item.action }) + (view.planning.capabilities.some(c => c.id === 'purchase-queue' && c.owned) ? button('Add to queue', { type:'plan-queue', action:item.action }) : ''));
      }
      return html;
    }
    function crewBody() {
      const slots = view.progression.specialists || [null,null];
      return '<div class="wx-roster-slots">' + slots.map((id, slot) => button(icon(id || 'crew') + '<strong>' + esc(id ? (root.WayfarersCore.Content.SPECIALISTS.find(s => s.id === id)?.name || id) : 'Choose explorer') + '</strong><small>Explorer ' + (slot + 1) + '</small>', () => open({ kind:'roster', slot }), { key:'crew-slot:' + slot, className:'wx-portrait-slot' })).join('') + '</div>' +
        menu('Companion', view.progression.companion || 'Choose a travelling partner', view.progression.companion || 'fox', { kind:'companions' }) +
        catalog('Recruit', visible(view.specialists).filter(item => !item.owned)) +
        menu('Guild approach', (view.doctrines || []).find(item => item.selected)?.label || 'Open Roads', 'compass', { kind:'doctrines' });
    }
    function rosterBody(slot) {
      const owned = visible(view.specialists).filter(item => item.owned && (item.slot === 0 || item.slot == null));
      return '<div class="wx-options">' + owned.map(item => {
        const id = item.specialistId || item.action.id;
        const label = root.WayfarersCore.Content.SPECIALISTS.find(s => s.id === id)?.name || item.label;
        return button(icon(id) + '<strong>' + esc(label) + '</strong><span>' + esc(item.effectText) + '</span>', { type:'specialist', slot, id }, { className:'wx-choice', selected:view.progression.specialists[slot] === id });
      }).join('') + '</div>' + (!owned.length ? '<p>Recruit an explorer to fill this position.</p>' : button('Leave position empty', { type:'specialist', slot, id:null }));
    }
    function planningBody() {
      const planning = view.planning;
      let html = '';
      const local = view.expedition.automation;
      if (local?.unlocked) html += section('Area helper', button(local.enabled ? 'Helper on' : 'Turn helper on', { type:'expedition-automation', enabled:!local.enabled, priority:local.priority }, { selected:local.enabled }) + '<div class="wx-segments">' + (local.choices || ['balanced','progress','income'].map(id => ({ id, label:id[0].toUpperCase() + id.slice(1) }))).map(choice => button(esc(choice.label), { type:'expedition-automation', enabled:local.enabled, priority:choice.id }, { selected:local.priority === choice.id })).join('') + '</div><p class="wx-muted">Buys area ranks using shared coins and respects saved goals. Progress funds the current project; income funds production.</p>' + button(local.dispatch ? 'Auto-expand on' : 'Auto-expand off', { type:'expedition-automation', enabled:local.enabled, priority:local.priority, dispatch:!local.dispatch }, { selected:local.dispatch }));
      html += catalog('Guild automation', view.automations);
      if (planning?.unlocked) {
        html += menu('Purchase priorities', 'Choose what the guild buys first', 'compass', { kind:'priorities' });
        html += menu('Reserves & saved goal', planning.goal ? 'Saving for ' + (planning.goal.name || planning.goal.id) : 'Protect supplies from automatic spending', 'crate', { kind:'reserves' });
        if (planning.capabilities.some(item => item.id === 'purchase-queue' && item.owned)) html += menu('Purchase queue', planning.queue.length + ' planned upgrades', 'notes', { kind:'queue' });
        if (planning.capabilities.some(item => item.id === 'kit-plan' && item.owned)) html += menu('Automatic kits', planning.kit, 'backpack', { kind:'kit-plan' });
        if (planning.capabilities.some(item => item.id === 'dispatch-preparation' && item.owned)) html += menu('Route preparation', planning.preparation, 'maps', { kind:'prepare-plan' });
        if (planning.capabilities.some(item => item.id === 'loadouts' && item.owned)) html += menu('Saved playbooks', planning.loadouts.length + ' saved', 'quill', { kind:'loadouts' });
      }
      html += catalog('Planning licences', planning?.capabilities);
      return html || '<p>Expedition helpers unlock after restoring Watchtower.</p>';
    }
    function renderSheet() {
      if (!sheet || !view) return;
      const e = view.expedition;
      const luck = view.luck || {};
      let title = '';
      let html = '';
      const model = sheet;
      detailFooter = '';
      if (model.kind === 'areas') {
        title = 'Your areas';
        const areas = e.areas || [];
        const next = areas.find(area => !area.unlocked && area.visible !== false);
        html = '<div class="wx-area-grid">' + areas.filter(area => area.unlocked).map(area => '<button data-wx-area="' + esc(area.id) + '" class="wx-area-tile" aria-pressed="' + !!area.selected + '" aria-label="' + esc((areaNames[area.id] || area.label) + (area.selected ? ', current area' : '') + ', ' + (area.rateText || area.status || 'Working') + (area.attention ? ', new development' : '')) + '">' + icon(areaIcons[area.id]) + '<strong>' + esc(areaNames[area.id] || area.label) + '</strong><small>' + esc(area.rateText || area.status || 'Working') + '</small>' + (area.attention ? '<span class="wx-new">New</span>' : '') + '</button>').join('') + '</div>';
        if (next) html += '<div class="wx-next-discovery">' + icon(areaIcons[next.id]) + '<span><strong>Next · ' + esc(areaNames[next.id] || next.label) + '</strong><small>' + esc(next.requirement || next.reason || next.objective || 'Continue your current expedition') + '</small></span></div>';
      }
      else if (model.kind === 'batch') {
        title = 'Purchase quantity';
        const options = e.batch?.options || [{count:1,unlocked:true}];
        const next = options.find(item => !item.unlocked);
        html = '<div class="wx-batch-options">' + options.filter(item => item.unlocked).map(item => button('×' + item.count, item.action || { type:'expedition-batch', count:item.count }, { key:'batch:' + item.count, selected:(e.batch?.selected || 1) === item.count, aria:'Buy exactly ' + item.count + ' ranks per purchase' })).join('') + '</div><p class="wx-muted">Each purchase buys the full quantity. Prices and effects update together.</p>' + (next ? '<div class="wx-next-discovery"><span><strong>Next · ×' + next.count + '</strong><small>' + esc(next.requirement || 'Continue developing your guild') + '</small></span></div>' : '');
      }
      else if (model.kind === 'focus') {
        title = 'Guild Focus';
        const focus = e.focus || {};
        html = '<div class="wx-focus-summary"><strong>' + (focus.charges || 0) + ' / ' + (focus.max || 3) + '</strong><span>Shared across every area</span></div>' + (focus.active ? '<p class="wx-effect">' + esc(areaNames[focus.active] || focus.active) + ' focused · ' + Math.ceil(focus.remaining) + 's left</p>' : '') + (focus.charges < focus.max ? '<p class="wx-muted">Next charge in ' + esc(Math.ceil((focus.nextChargeSeconds || 0) / 60)) + ' min · Restores while away</p>' : '') + '<div class="wx-focus-choices">' + (focus.actions || []).map(item => button(icon(item.icon || areaIcons[item.areaId || e.stage.kind]) + '<span><strong>' + esc(item.label) + '</strong><small>' + esc(item.description || item.effectText) + '</small>' + (item.context ? '<small>' + esc(item.context) + '</small>' : '') + '<small>' + esc(focusInputs(item)) + '</small>' + ((item.impact || []).length ? '<span class="wx-impact">' + impacts(item) + '</span>' : '') + '</span><b>1 ◆</b>', item.action, { key:'focus:' + item.id, disabled:item.disabled || !focus.charges, className:'wx-focus-choice' })).join('') + '</div>';
      }
      else if (model.kind === 'local') { const item = (view.globalUpgrades || []).find(c => c.id === model.catalogId) || e.cards.find(c => c.id === model.id && (!model.areaId || c.action?.areaId === model.areaId || !c.action?.areaId)); title = item?.label || item?.name || 'Upgrade'; html = detailBody(item, true); }
      else if (model.kind === 'upgrade') { const item = (view.globalUpgrades || []).find(c => c.id === model.id); title = item ? name(item) : 'Upgrade'; html = detailBody(item); }
      else if (model.kind === 'inspect') { const item = descriptors.get(model.id); title = item ? name(item) : 'Upgrade'; html = detailBody(item); }
      else if (model.kind === 'choice') {
        title = 'Area plans';
        const groups = visible(e.choices || [e.choice]);
        const configurations = visible(e.configurations);
        if (configurations.length || (e.specializations || []).length > 1) {
          html = groups.map(choice => menu(choice.title, choice.options.find(item => item.selected)?.label || 'Choose a working plan', 'compass', {kind:'plan-choice',id:choice.id})).join('');
          html += configurations.map(group => menu(group.label, Array.isArray(group.selected) ? group.selected.filter(Boolean).length + ' / ' + group.slots + ' assigned' : group.options.find(item => item.id === group.selected)?.label || 'Choose', 'compass', {kind:'configuration',id:group.id,slot:0})).join('');
          if ((e.specializations || []).length > 1) html += menu('Specialization', e.specializations.find(item => item.selected)?.label || 'Balanced tracks', 'mastery', {kind:'specialization'});
        } else html = groups.map(choice => section(choice.title, choiceBody(choice))).join('');
      }
      else if (model.kind === 'plan-choice') { const choice = e.choices.find(item => item.id === model.id); title = choice?.title || 'Working plan'; html = choiceBody(choice); }
      else if (model.kind === 'configuration') {
        const group = (e.configurations || []).find(item => item.id === model.id);
        title = group?.label || 'Area plan';
        if (group) {
          const slot = Math.max(0,Math.min(model.slot || 0,group.slots - 1));
          if (group.slots > 1) html += '<div class="wx-segments">' + Array.from({length:group.slots},(_,index) => button('Slot ' + (index + 1), () => open({kind:'configuration',id:group.id,slot:index},true), {key:'config-slot:' + index,selected:slot === index})).join('') + '</div>';
          const options = group.slotOptions?.find(item => item.slot === slot)?.options || (slot === 0 ? group.options : []);
          const selected = Array.isArray(group.selected) ? group.selected[slot] : group.selected;
          html += choiceBody({visible:true,id:group.id,options:options.map(item => Object.assign({},item,{selected:item.id === selected,effectText:item.description || item.effectText}))});
        }
      }
      else if (model.kind === 'specialization') { title = 'Specialization'; html = choiceBody({visible:true,id:'specializations',options:(e.specializations || []).map(item => Object.assign({},item,{effectText:item.description || item.effect || 'No specialized track'}))}); }
      else if (model.kind === 'choice-metrics') {
        const choice = (e.choices || [e.choice]).find(item => item.id === model.id);
        title = 'Full rate comparison';
        html = (choice?.options || []).map(item => section(item.label + (item.selected ? ' · Selected' : ''), '<p class="wx-muted">' + esc(item.effectText) + '</p>' + ((item.impact || []).length ? '<div class="wx-impact">' + impacts(item) + '</div>' : '<p class="wx-muted">Current working plan.</p>'))).join('');
      }
      else if (model.kind === 'upgrade-filters') {
        title = 'Upgrade effects';
        html = '<div class="wx-options">' + ['all', ...model.effects].map(id => button(id === 'all' ? 'All effects' : id.replace(/[-_]/g,' ').replace(/^./, letter => letter.toUpperCase()), () => { upgradeEffect = id; close(); update(view); q('[data-wx-destination]').scrollTop = 0; }, { key:'filter:effect:' + id, selected:upgradeEffect === id })).join('') + '</div>';
      }
      else if (model.kind === 'upgrade-areas') {
        title = 'Affected area';
        html = '<div class="wx-options">' + ['all',...(e.areas || []).filter(area => area.unlocked).map(area => area.id),'guild'].map(id => button(id === 'all' ? 'All areas' : esc(areaNames[id] || id), () => { upgradeArea = id; close(); update(view); q('[data-wx-destination]').scrollTop = 0; }, {key:'filter:area:' + id,selected:upgradeArea === id})).join('') + '</div>';
      }
      else if (model.kind === 'objective') {
        title = e.stage.name;
        html = '<p class="wx-effect">' + esc(e.stage.objective) + '</p><ol class="wx-checkpoint-list">' + (e.checkpoints || []).map(c => '<li data-done="' + !!c.complete + '"><b>' + (c.complete ? '✓' : '○') + '</b><span>' + esc(c.label) + '</span></li>').join('') + '</ol>';
        if (e.scene?.kind === 'harbor' && e.scene?.ruleset === 'progression') {
          const voyage = e.scene.voyage || {};
          html += section('At sea', (voyage.manifests || []).map((manifest,index) => '<div class="wx-milestone"><span><strong>Ship ' + (index + 1) + ' · ' + Math.floor(percent(manifest.work / manifest.target)) + '%</strong><small>' + esc(manifest.port) + ' · On arrival: ' + esc(Object.entries(manifest.payout || {}).map(([id,value]) => '+' + metric(value) + ' ' + id).join(' · ')) + '</small></span></div>').join('') || '<p class="wx-muted">Ships launch automatically when supplies are available.</p>');
          html += '<p class="wx-muted">Next departure uses ' + esc(metric(e.scene.rates?.supply || 0)) + ' provisions. Arrival rewards are credited when a voyage finishes.</p>';
        }
        if (e.condition) html += '<p>' + esc(e.condition) + '</p>';
        if (e.stage.index > 2) html += '<ul class="wx-muted">' + (e.guildLinks || []).map(link => '<li>' + esc(link) + '</li>').join('') + '</ul>';
        if (e.choice?.visible) html += menu('Current approach', e.choice.options.find(c => c.selected)?.label, 'compass', { kind:'choice' });
      } else if (model.kind === 'finale') {
        title = e.stage.established || e.scene?.established ? 'Area developed' : 'Outpost established';
        const outpost = e.outposts?.find(item => item.id === e.stage.kind);
        const unlocks = e.stage.index === 0 ? '<div class="wx-milestone">' + icon('mine') + '<span>Mine unlocked · Ore for equipment</span></div>' : e.stage.index === 1 ? '<div class="wx-milestone">' + icon('forge') + '<span>Forge unlocked · Your first equipment rank</span></div>' : e.stage.index === 2 ? '<div class="wx-milestone">' + icon('crew') + '<span>Crew, blueprints & expedition helper unlocked</span></div>' : '';
        html = '<div class="wx-detail-hero wx-finale-art">' + icon(areaIcons[e.stage.kind] || 'guild') + '<strong>' + esc(e.stage.name) + '</strong></div><p class="wx-effect">' + esc(outpost?.effectText || outpost?.description || 'This area keeps working while you visit the others.') + '</p>' + unlocks + button(esc(e.next?.label || 'Continue developing') + ' →', e.next?.action, { className:'wx-confirm', disabled:e.next?.disabled || !e.next?.action });
        const newProject = e.stage.index > 0 && visible(view.globalUpgrades || []).find(item => !item.owned && item.action?.type === 'expedition-development' && (item.dependencies || []).every(dependency => dependency.met) && (item.sourceAreas || []).includes(e.stage.kind) && (item.targetAreas || []).some(id => id !== e.stage.kind));
        if (newProject) {
          const affected = newProject.targetAreas.find(id => id !== e.stage.kind);
          html += button(icon(areaIcons[affected]) + '<span><strong>' + esc(areaNames[affected] + ': ' + name(newProject)) + '</strong><small>New from ' + esc(areaNames[e.stage.kind]) + '</small></span><b>›</b>', () => { selectArea(affected); open({ kind:'upgrade', id:newProject.id }); }, { key:'unlock-project:' + newProject.id, className:'wx-menu' });
        }
        html += button('Visit the Guild', () => { close(); screen = 'guild'; update(view); }, { key:'finale-guild' });
        html += '<p class="wx-muted">' + esc(e.next?.description || 'Your upgrades, crews and production stay in every unlocked area.') + '</p>';
      } else if (model.kind === 'room') {
        const room = view.rooms.find(r => r.id === model.id);
        title = room?.name || 'Guild';
        html = catalog('Upgrades', (room?.actions || []).filter(item => item.action?.type === 'buy')) + catalog('Recipes', visible(view.recipes).filter(item => item.room === model.id));
        if (model.id === 'study') html += button('Open research upgrades', () => showUpgrades('all','research'), { key:'study-upgrades', className:'wx-confirm' });
        if (model.id === 'forge') html += catalog('Field kits', luck.kits);
        if (model.id === 'cartography') html += catalog('Preparations', e.preparation) + catalog('Route focus', view.modes);
      } else if (model.kind === 'crew') { title = 'Crew'; html = crewBody(); }
      else if (model.kind === 'roster') { title = 'Choose explorer ' + (model.slot + 1); html = rosterBody(model.slot); }
      else if (model.kind === 'companions') { title = 'Companions'; html = tiles(view.companions); }
      else if (model.kind === 'doctrines') { title = 'Guild approach'; html = tiles(view.doctrines); }
      else if (model.kind === 'research') { title = 'Research'; html = button('Open global upgrades', () => showUpgrades(), { key:'research-upgrades', className:'wx-confirm' }); }
      else if (model.kind === 'network') {
        title = 'Connected areas';
        html = (e.areas || []).filter(area => area.unlocked).map(area => button(icon(areaIcons[area.id]) + '<span><strong>' + esc(areaNames[area.id] || area.label) + '</strong><small>' + esc(area.rateText || area.status || '') + '</small></span><b>›</b>', () => selectArea(area.id), { key:'network-area:' + area.id, className:'wx-menu' })).join('');
        const connected = visible(view.globalUpgrades || []).filter(item => item.owned && (item.targetAreas || []).includes(e.stage.kind) && (item.sourceAreas || []).some(id => id !== e.stage.kind));
        html += section('Supporting this area', connected.map(item => '<article class="wx-connection">' + areaLinks(item) + '<strong>' + esc(name(item)) + '</strong><span>' + esc(item.effectText || item.description) + '</span>' + button('Details', () => open({ kind:'upgrade', id:item.id }), { key:'connection:' + item.id }) + '</article>').join(''));
        html += button('Find connected upgrades', () => showUpgrades(e.stage.kind), { key:'network-upgrades', className:'wx-confirm' });
      }
      else if (model.kind === 'projects') { title = 'Guild projects'; html = tiles([...(view.development?.projects || []), ...(view.development?.chapter?.projects || [])]); }
      else if (model.kind === 'planning') { title = 'Automation'; html = planningBody(); }
      else if (model.kind === 'priorities') {
        title = 'Purchase priorities';
        html = ['operations','equipment'].map(group => section(group === 'operations' ? 'Guild operations' : 'Equipment', '<div class="wx-options">' + (group === 'operations' ? ['balanced','travel','production'] : ['balanced','tools','boots','research']).map(id => button(esc(id[0].toUpperCase() + id.slice(1)), { type:'plan-priority', group, id }, { selected:view.planning.priorities[group] === id })).join('') + '</div>')).join('');
      } else if (model.kind === 'reserves') {
        title = 'Protected supplies';
        html = '<p>Automation leaves these amounts unspent.</p>' + visible(view.resources).filter(r => (view.planning.reserves || []).some(x => x.id === r.id)).map(r => { const item = view.planning.reserves.find(x => x.id === r.id); return menu(r.name, item.formatted + ' reserved', r.id, { kind:'reserve', id:r.id }); }).join('') + section('Saved purchase', view.planning.goal ? '<p>' + esc(view.planning.goal.name || view.planning.goal.id) + '</p>' + button('Clear saved goal', { type:'plan-goal', action:null }) : '<p>Inspect any guild upgrade, then choose “Save for this”.</p>');
      } else if (model.kind === 'reserve') {
        title = 'Reserve ' + model.id;
        const item = view.planning.reserves.find(r => r.id === model.id);
        html = '<label class="wx-field">Minimum to keep<input data-wx-reserve inputmode="decimal" value="' + esc(item.formatted.replace(/,/g,'')) + '" aria-label="Minimum reserve"></label>' + button('Save reserve', () => execute({ type:'plan-reserve', id:model.id, amount:dialog.querySelector('[data-wx-reserve]').value }), { key:'save-reserve', className:'wx-confirm' });
      } else if (model.kind === 'queue') {
        title = 'Purchase queue';
        html = (view.planning.queue || []).map((item,index) => '<div class="wx-row"><span>' + esc(item.name || item.id) + '</span>' + button('Remove', { type:'plan-remove', index }) + '</div>').join('') || '<p>Inspect an upgrade and add it to your queue.</p>';
      } else if (model.kind === 'loadouts') {
        title = 'Guild playbooks';
        html = [0,1,2].map(id => { const item = view.planning.loadouts.find(x => Number(x.id) === id); return section(item?.name || 'Plan ' + (id + 1), button('Save current', { type:'loadout-save', id, name:'Plan ' + (id + 1) }) + button('Use', { type:'loadout-use', id }, { disabled:!item }) + button('Delete', { type:'loadout-delete', id }, { disabled:!item })); }).join('');
      } else if (model.kind === 'kit-plan' || model.kind === 'prepare-plan') {
        const kit = model.kind === 'kit-plan';
        title = kit ? 'Automatic kit' : 'Automatic preparation';
        html = '<div class="wx-options">' + (kit ? ['off','mining','travel'] : ['off','scout','supply','survey']).map(id => button(esc(id), { type:kit ? 'plan-kit' : 'plan-preparation', id }, { selected:(kit ? view.planning.kit : view.planning.preparation) === id })).join('') + '</div>';
      } else if (model.kind === 'stockpile') {
        title = 'Guild stockpile';
        html = '<div class="wx-stockpile">' + visible(view.resources).map(r => '<div>' + icon(r.id) + '<span>' + esc(r.name) + '<small>' + esc(r.rateFormatted || '') + '</small></span><strong>' + esc(r.formatted) + '</strong></div>').join('') + '</div>';
      } else if (model.kind === 'blueprints') { title = 'Lasting blueprints'; html = '<p>' + (e.blueprints.some(item => !item.disabled) ? 'Choose a lasting advantage for this region.' : 'Complete another region to choose again.') + ' Kept through renewals.</p>' + tiles(e.blueprints); }
      else if (model.kind === 'outpost') {
        const item = e.outposts.find(p => p.id === model.id);
        title = item?.name || item?.label || 'Outpost';
        html = '<div class="wx-detail-hero">' + icon(item?.icon || 'guild') + '</div><p class="wx-effect">' + esc(item?.effectText || item?.description || item?.rateText || 'Working for the guild') + '</p>' + (item?.action ? button('Visit', item.action) : '');
      } else if (model.kind === 'route-confirm') {
        title = 'Travel to ' + (view.routes.find(item => item.action?.id === model.action.id)?.label || 'this route');
        html = '<p>Your areas keep their upgrades and continue working.</p>' + button('Visit area', model.action, { className:'wx-confirm' });
      } else if (model.kind === 'routes') { title = 'Expedition routes'; html = tiles(view.routes) + catalog('Reward focus', view.modes); }
      else if (model.kind === 'relics') { title = 'Relic collection'; html = tiles(luck.relics) + catalog('Discovery focus', luck.hunts) + catalog('Field kits', luck.kits); }
      else if (model.kind === 'journal') {
        title = 'Expedition journal';
        html = (luck.ledger?.recent || []).slice(-16).reverse().map(item => '<article class="wx-journal-entry"><strong>' + esc(item.title) + '</strong><span>' + esc(item.reward) + '</span></article>').join('') || '<p>Your discoveries will be recorded here.</p>';
        html += section('Guild record', '<ul>' + (view.events || []).slice(-10).reverse().map(item => '<li>' + esc(item) + '</li>').join('') + '</ul>');
      } else if (model.kind === 'renewals') {
        title = 'Guild renewals';
        html = menu('Expedition Refit', view.refit.rewardText, 'notes', { kind:'reset', reset:'refit' }) + menu('Guild Charter', view.charter.rewardText, 'crests', { kind:'reset', reset:'charter' }) + catalog('Field-note upgrades', view.refitUpgrades) + catalog('Legacy upgrades', view.legacyUpgrades) + catalog('Challenges', view.challenges);
      } else if (model.kind === 'reset') {
        title = model.reset === 'refit' ? 'Expedition Refit' : 'Guild Charter';
        html = '<p class="wx-effect">' + esc(view[model.reset].rewardText) + '</p><p>Review exactly what stays and restarts before confirming.</p>' + button('Review reset', () => legacy(model.reset), { key:'review:' + model.reset, className:'wx-confirm' });
      } else if (model.kind === 'supplies') { title = 'Expedition supplies'; html = catalog('Supply plan', e.supplyChoices) + catalog('Meals & recipes', view.recipes) + catalog('Preparation', e.preparation); }
      else if (model.kind === 'options') {
        title = 'Wayfarers’ Guild';
        html = button(icon('settings') + '<span>Settings & saves</span>', () => legacy('settings'), { key:'settings', className:'wx-menu' });
        if (root.WayfarersAndroid) html += button(icon('automation') + '<span>App updates</span>', () => { close(); options.nativeOptions(); }, { key:'native-options', className:'wx-menu' });
        if (view.premium?.unlocked) html += button(icon('starshards') + '<span>Starshard shop</span>', () => legacy('shop'), { key:'shop', className:'wx-menu' });
        html += menu('How this expedition works', e.stage.name, 'compass', { kind:'objective' });
      } else if (model.kind === 'return') {
        title = 'Welcome back';
        html = '<p>' + esc(model.time) + ' of guild work</p><div class="wx-stockpile">' + model.gains.map(item => '<div>' + icon(item.id) + '<span>' + esc(item.id) + '</span><strong>+' + esc(item.amount) + '</strong></div>').join('') + '</div>' + button('Continue expedition', close, { key:'return-close', className:'wx-confirm' });
      }
      text(dialog.querySelector('h2'), title);
      dialog.dataset.kind = model.kind;
      dialog.querySelector('[data-wx-back]').hidden = !stack.length;
      markup(dialog.querySelector('[data-wx-sheet-content]'), html);
      const footer = dialog.querySelector('[data-wx-sheet-footer]');
      footer.hidden = !detailFooter;
      markup(footer,detailFooter);
    }
    function destinationBody() {
      const e = view.expedition;
      if (screen === 'upgrades') return upgradesBody();
      if (screen === 'guild') {
        const rooms = view.rooms.filter(r => r.unlocked && r.id !== 'trail');
        let html = '<div class="wx-room-grid">' + rooms.map(room => button(icon(room.id) + '<strong>' + esc(room.name) + '</strong><small>' + esc(room.level ? 'Rank ' + room.level : 'Working') + '</small>', () => open({ kind:'room', id:room.id }), { key:'room:' + room.id, className:'wx-room' })).join('') + '</div>';
        if (visible(view.specialists).length) html += menu('Crew', 'Explorers & companions', 'crew', { kind:'crew' });
        html += button(icon('maps') + '<span><strong>Atlas & discoveries</strong><small>Outposts, relics and renewals</small></span><b>›</b>', () => { close(); screen = 'atlas'; update(view); }, { key:'guild-atlas', className:'wx-menu' });
        if (e.automation?.unlocked || view.planning?.unlocked) html += menu('Automation', e.automation?.enabled ? 'Expedition helper active' : 'Set your priorities', 'automation', { kind:'planning' });
        if (visible(view.globalUpgrades || []).length) html += button(icon('research') + '<span><strong>Global upgrades</strong><small>Developments, equipment & research</small></span><b>›</b>', () => showUpgrades(), { key:'guild-upgrades', className:'wx-menu' });
        if (visible(view.recipes).length) html += menu('Supplies', 'Meals, kits & preparation', 'provisions', { kind:'supplies' });
        html += menu('Stockpile', 'Production & resources', 'crate', { kind:'stockpile' });
        return html;
      }
      let html = '<div class="wx-atlas-path">' + (e.outposts || []).map((outpost,index) => button('<span class="wx-atlas-number">' + (index + 1) + '</span>' + icon(outpost.icon || (index % 3 === 1 ? 'mine' : index % 3 === 2 ? 'observatory' : 'guild')) + '<span><strong>' + esc(outpost.name || outpost.label || 'Outpost ' + (index + 1)) + '</strong><small>' + esc(outpost.rateText || outpost.effectText || 'Working automatically') + '</small></span><b>✓</b>', () => open({ kind:'outpost', id:outpost.id }), { key:'outpost:' + outpost.id, className:'wx-atlas-stop' })).join('') + '</div>';
      html += (e.areas || []).filter(area => area.unlocked).map(area => button(icon(areaIcons[area.id]) + '<span><strong>' + esc(areaNames[area.id] || area.label) + '</strong><small>' + esc(area.rateText || area.status || 'Working') + '</small></span><b>›</b>', () => selectArea(area.id), { key:'atlas-area:' + area.id, className:'wx-menu' })).join('');
      if (e.automation?.unlocked && visible(e.blueprints).length) html += menu('Blueprints', 'Keep a lasting advantage', 'quill', { kind:'blueprints' });
      if (view.luck?.relicsUnlocked) html += menu('Relics', view.luck.owned.length + ' collected', 'chest', { kind:'relics' });
      if (view.luck?.unlocked) html += menu('Journal', 'Finds & expedition record', 'journal', { kind:'journal' });
      if (view.presentation.journalSections.includes('renewals')) html += menu('Renewals', 'Field notes & guild crests', 'notes', { kind:'renewals' });
      if (view.routes.length > 1) html += menu('Route archive', 'Revisit discovered destinations', 'maps', { kind:'routes' });
      return html;
    }
    function update(nextView, context) {
      if (disposed || !nextView.expedition?.local) return;
      const incoming = nextView.expedition;
      const choiceGroups = (incoming.choices || []).some(choice => Array.isArray(choice.options)) ? incoming.choices : incoming.choices?.length ? [{id:'plans',title:'Working plan',visible:visible(incoming.choices).length > 1,options:visible(incoming.choices).map(item => Object.assign({effectText:item.description || item.effect},item))}] : [];
      const stageModel = Object.assign({ objective:incoming.stage.goal || incoming.stage.checkpoint?.label || 'Develop this area', established:incoming.scene?.established },incoming.stage);
      view = Object.assign({},nextView,{expedition:Object.assign({},incoming,{
        stage:stageModel,
        cards:(incoming.cards || []).map(item => item.trackId && item.id !== item.trackId ? Object.assign({},item,{id:item.trackId}) : item),
        choices:choiceGroups,
        choice:incoming.choice || choiceGroups.find(choice => choice.visible !== false) || {visible:false,options:[]},
        finale:incoming.finale || {ready:stageModel.phase === 'Capstone',completed:stageModel.completed,progress:stageModel.finaleTarget ? stageModel.finaleWork / stageModel.finaleTarget : 0},
        sequence:incoming.sequence ?? Math.max(0,...(incoming.events || []).map(event => event.sequence || 0)),
        checkpoints:incoming.checkpoints || [], outposts:incoming.outposts || []
      })});
      const e = view.expedition;
      const stage = e.stage;
      if (context) currentContext = context;
      const ctx = currentContext;
      const allDescriptors = ['actions','routes','modes','recipes','research','specialists','companions','doctrines','challenges','automations','refitUpgrades','legacyUpgrades'].flatMap(key => view[key] || []).concat(e.blueprints || [], e.preparation || [], e.supplyChoices || [], view.planning?.capabilities || [], view.development?.projects || [], view.development?.chapter?.projects || [], view.luck?.relics || [], view.luck?.research || [], view.luck?.hunts || [], view.luck?.kits || [], view.globalUpgrades || []);
      allDescriptors.forEach(item => descriptors.set(descriptorKey(item), item));
      const guildUnlocked = view.unlocks.some(id => id !== 'trail');
      const areas = e.areas || ['greenway','quarry','watchtower'].map((id,index) => ({ id, unlocked:stage.index >= index, selected:stage.kind === id }));
      const selectedArea = areas.find(area => area.selected)?.id || stage.kind;
      const unlockedAreas = areas.filter(area => area.unlocked);
      const areaIndex = unlockedAreas.findIndex(area => area.id === selectedArea);
      const canNavigateAreas = screen === 'expedition' && unlockedAreas.length > 1;
      const upgradesUnlocked = guildUnlocked || areas.filter(area => area.unlocked).length > 1;
      q('[data-wx-nav="guild"]').hidden = !guildUnlocked;
      q('[data-wx-nav="upgrades"]').hidden = !upgradesUnlocked;
      if ((screen === 'guild' || screen === 'atlas') && !guildUnlocked || screen === 'upgrades' && !upgradesUnlocked) screen = 'expedition';
      q('[data-wx-prev]').hidden = q('[data-wx-next]').hidden = !canNavigateAreas;
      q('[data-wx-prev]').disabled = areaIndex <= 0;
      q('[data-wx-next]').disabled = areaIndex >= unlockedAreas.length - 1;
      q('[data-wx-prev]').setAttribute('aria-label', 'Previous area' + (unlockedAreas[areaIndex - 1] ? ': ' + areaNames[unlockedAreas[areaIndex - 1].id] : ''));
      q('[data-wx-next]').setAttribute('aria-label', 'Next area' + (unlockedAreas[areaIndex + 1] ? ': ' + areaNames[unlockedAreas[areaIndex + 1].id] : ''));
      q('[data-wx-nav="expedition"] [data-wx-attention]').hidden = !areas.some(area => area.attention && !area.selected);
      q('[data-wx-nav="expedition"]').setAttribute('aria-label', 'Areas, current ' + (areaNames[selectedArea] || stage.name) + (areas.some(area => area.attention && !area.selected) ? ', new development' : ''));
      shell.dataset.screen = screen;
      shell.dataset.area = selectedArea;
      shell.querySelectorAll('[data-wx-nav]').forEach(node => node.setAttribute('aria-current', node.dataset.wxNav === screen || screen === 'atlas' && node.dataset.wxNav === 'guild' ? 'page' : 'false'));
      q('.wx-play').hidden = screen !== 'expedition';
      q('[data-wx-destination]').hidden = screen === 'expedition';
      const coins = view.resources.find(r => r.id === 'coins');
      markup(q('[data-wx-wallet]'), icon('coins') + '<span><strong>' + esc(compact(coins.value)) + '</strong><small>' + esc(coins.rateFormatted) + '</small></span>');
      const furnace = (e.stations || []).find(item => item.id === 'furnace');
      q('[data-wx-local-count]').hidden = screen !== 'expedition' || !furnace;
      if (furnace) {
        const repairing = e.finale.ready && !e.finale.completed;
        const output = stage.established ? e.scene?.rates?.materials : e.scene?.flows?.furnace;
        const value = repairing ? Math.floor(percent(e.finale.progress)) + '%' : output >= 1000 ? compact(output) : root.WayfarersCore.format(output || 0);
        markup(q('[data-wx-local-count]'), '<small>' + (repairing ? 'Lift repair' : stage.established ? 'Ore /s' : 'Smelt /s') + '</small><strong>' + esc(value) + '</strong>');
      }
      q('[data-wx-reward]').hidden = !view.caravan?.offer && !view.caravan?.pending;
      q('[data-wx-save-alert]').hidden = !ctx.saveFailure && !ctx.awaitingWallet;
      text(q('[data-wx-save-alert]'), ctx.awaitingWallet ? 'Restoring purchase wallet…' : 'Save needs attention');
      text(q('[data-wx-stage-name]'), screen === 'expedition' ? canNavigateAreas ? (areaIndex + 1) + ' OF ' + unlockedAreas.length + ' AREAS' : stage.name : screen === 'guild' ? 'YOUR GUILD' : screen === 'upgrades' ? 'CONNECTED GUILD' : 'THE ATLAS');
      text(q('[data-wx-objective-text]'), screen === 'expedition' ? canNavigateAreas ? (areaNames[selectedArea] || stage.name) + ' ▾' : stage.objective : screen === 'guild' ? 'Build your advantage' : screen === 'upgrades' ? 'Every improvement, one place' : 'Every journey leaves a mark');
      q('[data-wx-objective]').disabled = screen !== 'expedition';
      q('[data-wx-objective]').setAttribute('aria-label', canNavigateAreas ? 'Choose area. ' + (areaNames[selectedArea] || stage.name) + ', ' + (areaIndex + 1) + ' of ' + unlockedAreas.length : stage.name + ': ' + stage.objective);
      q('[data-wx-objective]').setAttribute('aria-haspopup', canNavigateAreas ? 'dialog' : 'false');
      q('[data-wx-progress]').hidden = screen !== 'expedition';
      q('[data-wx-progress]').value = percent(stage.progress);
      if (screen !== 'expedition') markup(q('[data-wx-destination]'), destinationBody());
      if (previousScreen !== screen) { q('[data-wx-destination]').scrollTop = 0; previousScreen = screen; }
      const shown = e.cards.filter(c => c.visible !== false);
      const tray = q('[data-wx-tray]');
      const switchingArea = displayedArea !== selectedArea;
      if (switchingArea && displayedArea) text(q('[data-wx-area-status]'),(areaNames[selectedArea] || stage.name) + ', area ' + (areaIndex + 1) + ' of ' + unlockedAreas.length);
      if (switchingArea && displayedArea) areaScroll.set(displayedArea,tray.scrollTop);
      tray.dataset.count = shown.length;
      q('.wx-dock').dataset.count = String(Math.min(3,shown.length));
      text(q('[data-wx-upgrade-count]'), 'Upgrades' + (shown.length > 1 ? ' · ' + shown.length : ''));
      const batch = q('[data-wx-batch]');
      batch.hidden = !batchAvailable();
      text(batch, 'Buy ×' + (e.batch?.selected || 1) + ' ▾');
      batch.setAttribute('aria-label', 'Purchase quantity: ' + (e.batch?.selected || 1) + '. Change exact quantity.');
      for (const node of Array.from(tray.children)) if (!shown.some(c => c.id === node.dataset.upgrade)) node.remove();
      shown.forEach(item => {
        let node = Array.from(tray.children).find(c => c.dataset.upgrade === item.id);
        if (!node) {
          node = document.createElement('article');
          node.className = 'wx-upgrade';
          node.dataset.upgrade = item.id;
          node.innerHTML = button(localIcon(item.id,item.icon) + '<span class="wx-track-copy"><strong>' + esc(item.label) + '</strong><small data-wx-rank></small><small class="wx-next-rate" data-wx-next-rate></small></span>', () => inspectLocal(item.id), { key:'local:' + item.id, className:'wx-upgrade-info', aria:'Inspect ' + item.label }) + '<button class="wx-price" data-wx-buy="' + esc(item.id) + '"></button>';
          tray.append(node);
        }
        text(node.querySelector('[data-wx-rank]'), item.rank + ' / ' + item.maxRank);
        text(node.querySelector('[data-wx-next-rate]'), item.rank >= item.maxRank ? 'Mastered' : gainLabel(item));
        const buy = node.querySelector('[data-wx-buy]');
        markup(buy, item.rank >= item.maxRank ? 'Max' : purchaseLabel(item, true));
        buy.disabled = !!item.disabled || !!ctx.awaitingWallet;
        buy.setAttribute('aria-label', 'Buy ' + (item.quantity || 1) + ' ranks of ' + item.label + ', rank ' + item.rank + ' of ' + item.maxRank + ', ' + (item.cost || []).map(c => c.text).join(', ') + (item.disabled && item.reason ? '. ' + item.reason : ''));
        node.dataset.ready = String(!buy.disabled);
        if (previousStage === stage.id && previousRanks[item.id] != null && previousRanks[item.id] < item.rank) { node.classList.remove('wx-purchased'); void node.offsetWidth; node.classList.add('wx-purchased'); }
      });
      if (switchingArea) tray.scrollTop = areaScroll.get(selectedArea) || 0;
      displayedArea = selectedArea;
      const previouslyEarned = earnedTracks.get(selectedArea);
      if (!switchingArea && previouslyEarned && shown.length > previouslyEarned.length) {
        const newTrack = shown.find(item => !previouslyEarned.includes(item.id));
        if (newTrack) { const node = Array.from(tray.children).find(item => item.dataset.upgrade === newTrack.id); node?.scrollIntoView({block:'nearest',inline:'nearest',behavior:'instant'}); notify((areaNames[selectedArea] || stage.name) + ': ' + newTrack.label + ' unlocked'); }
      }
      earnedTracks.set(selectedArea, shown.map(item => item.id));
      if (knownAreaCount !== null && knownAreaCount < unlockedAreas.length && !swipeTaught && unlockedAreas.length > 1) { swipeTaught = true; notify('Swipe the scenery to visit your other areas'); }
      knownAreaCount = unlockedAreas.length;
      previousRanks = Object.fromEntries(e.cards.map(c => [c.id,c.rank]));
      previousStage = stage.id;
      if (seenSequence == null) seenSequence = e.sequence;
      if (e.sequence > seenSequence) {
        const previousSequence = seenSequence;
        seenSequence = e.sequence;
        const event = (e.events || []).filter(item => item.sequence > previousSequence && item.kind !== 'stage').pop();
        if (event) { const affected = event.targetAreas?.find(id => id !== event.areaId) || event.areaId; notify((affected ? (areaNames[affected] || affected) + ': ' : '') + event.title); }
        root.queueMicrotask(() => { if (!disposed) options.perform({ type:'expedition-seen', sequence:seenSequence }); });
      }
      const bottleneck = (e.stations || []).find(s => s.bottleneck);
      const activeArea = areas.find(area => area.id === selectedArea);
      const voyageLabel = e.scene?.kind === 'harbor' && e.scene?.ruleset === 'progression' ? e.scene.voyage?.ships ? 'Voyage · ' + Math.floor(percent(e.scene.voyage.progress)) + '% · ' + e.scene.voyage.ships + ' at sea' : 'Ships waiting for supplies' : null;
      text(q('[data-wx-world-label]'), voyageLabel || (bottleneck ? bottleneck.label + ' · ' + (bottleneck.status || 'Bottleneck') : activeArea?.established ? activeArea.rateText || 'Area working' : canNavigateAreas ? stage.objective : stage.index === 0 && shown.length === 1 ? 'Coins arrive as you explore' : ''));
      q('[data-wx-world-label]').hidden = !q('[data-wx-world-label]').textContent;
      q('[data-wx-world-label]').setAttribute('aria-label','Area details: ' + q('[data-wx-world-label]').textContent);
      const focus = q('[data-wx-focus]');
      focus.hidden = !e.focus?.unlocked;
      if (e.focus?.unlocked) { markup(focus, '<span aria-hidden="true">◆</span><span>Focus<small>' + (e.focus.active ? Math.ceil(e.focus.remaining) + 's' : e.focus.charges + ' / ' + e.focus.max) + '</small></span>'); focus.setAttribute('aria-label', 'Guild Focus. ' + (e.focus.active ? (areaNames[e.focus.active] || e.focus.active) + ' active, ' + Math.ceil(e.focus.remaining) + ' seconds left. ' : '') + e.focus.charges + ' of ' + e.focus.max + ' shared charges. Review effects.'); }
      const support = visible(view.globalUpgrades || []).find(item => item.owned && (item.targetAreas || []).includes(selectedArea) && (item.sourceAreas || []).some(id => id !== selectedArea));
      const network = q('[data-wx-network]');
      network.hidden = !support;
      if (support) { markup(network, areaLinks(support) + '<span>' + esc(name(support)) + '</span>'); network.setAttribute('aria-label', 'Area connections: ' + name(support)); }
      const choiceCount = visible(e.choices || [e.choice]).length + visible(e.configurations).length + ((e.specializations || []).length > 1 ? 1 : 0);
      const selectedPlan = e.choice?.options?.find(option => option.selected)?.label;
      const planLabel = choiceCount > 1 ? 'Plans · ' + choiceCount : stage.kind === 'greenway' ? 'Path' : stage.kind === 'quarry' ? 'Processing' : 'Plans';
      let worldActions = choiceCount ? button(icon('compass') + '<span>' + esc(planLabel) + '</span><b>›</b>', () => open({ kind:'choice' }), { key:'world-choice', className:'wx-world-button', aria:'Area plans' + (selectedPlan ? ': ' + selectedPlan : '') }) : '';
      if (stage.completed && e.next?.action && !e.next.disabled) {
        const nextLabel = e.next.label.replace('Discover','Open').replace('Copper Quarry','Quarry').replace('Greenway','Trail').replace('Watchtower','Tower');
        worldActions += button(esc(nextLabel) + ' →', e.next.action, { className:'wx-world-button wx-gold', aria:e.next.label });
      }
      else if (activeArea?.established && upgradesUnlocked) worldActions += button(icon('tools') + '<span>Develop</span>', () => showUpgrades(selectedArea), { key:'area-develop:' + selectedArea, className:'wx-world-button wx-develop' });
      markup(q('[data-wx-world-actions]'), worldActions);
      scene.setQuiet(!!ctx.quiet);
      scene.update(Object.assign({}, e, { scene:Object.assign({}, e.scene, { banner:view.premium?.equipped, companion:view.progression.companion }) }));
      q('[data-wx-canvas]').setAttribute('aria-label', stage.name + '. ' + stage.objective + '. ' + Math.floor(percent(stage.progress)) + '% complete.' + (bottleneck ? ' ' + bottleneck.label + ' is the bottleneck.' : ''));
      if (sheet) renderSheet();
      if (!viewedStages.has(stage.id)) { viewedStages.add(stage.id); if (stage.completed) celebrated.add(stage.id); }
      if (stage.completed && !celebrated.has(stage.id) && !dialog.open && !options.overlayOpen()) {
        celebrated.add(stage.id);
        open({ kind:'finale' });
      }
    }
    function notify(message) {
      if (!message || disposed) return;
      // Complete transaction and help messages remain in their sheets/journal.
      const brief = String(message).split(/(?<=[.!?])\s/)[0];
      if (brief.length > 90) return;
      text(q('[data-wx-toast]'), brief);
      q('[data-wx-toast]').dataset.visible = 'true';
      root.clearTimeout(toastTimer);
      toastTimer = root.setTimeout(() => { q('[data-wx-toast]').dataset.visible = 'false'; }, 2500);
    }
    function click(event) {
      const target = event.target.closest('button');
      if (!target || target.disabled) return;
      event.stopPropagation();
      if (target.dataset.wxDo) execute(commands.get(target.dataset.wxDo));
      else if (target.dataset.wxBuy) execute(view.expedition.cards.find(c => c.id === target.dataset.wxBuy)?.action);
      else if (target.dataset.wxArea) selectArea(target.dataset.wxArea);
      else if (target.dataset.wxNav) { close(); screen = target.dataset.wxNav; update(view); }
      else if (target.hasAttribute('data-wx-network')) open({ kind:'network' });
      else if (target.hasAttribute('data-wx-close')) close();
      else if (target.hasAttribute('data-wx-back')) back();
      else if (target.hasAttribute('data-wx-wallet')) open({ kind:'stockpile' });
      else if (target.hasAttribute('data-wx-objective')) open({ kind:(view.expedition.areas || []).filter(area => area.unlocked).length > 1 ? 'areas' : 'objective' });
      else if (target.hasAttribute('data-wx-world-label')) open({kind:'objective'});
      else if (target.hasAttribute('data-wx-prev')) stepArea(-1);
      else if (target.hasAttribute('data-wx-next')) stepArea(1);
      else if (target.hasAttribute('data-wx-batch')) open({kind:'batch'});
      else if (target.hasAttribute('data-wx-focus')) open({kind:'focus'});
      else if (target.hasAttribute('data-wx-options')) open({ kind:'options' });
      else if (target.hasAttribute('data-wx-save-alert')) legacy('settings');
      else if (target.hasAttribute('data-wx-reward')) legacy('caravan');
    }
    shell.addEventListener('click', click, { signal:controller.signal });
    document.addEventListener('pointerdown', startPointer, { capture:true, signal:controller.signal });
    document.addEventListener('pointermove', movePointer, { capture:true, passive:false, signal:controller.signal });
    document.addEventListener('pointerup', endPointer, { capture:true, signal:controller.signal });
    document.addEventListener('pointercancel', endPointer, { capture:true, signal:controller.signal });
    root.addEventListener('blur', () => { cancelSwipe(); activePointers.clear(); }, {signal:controller.signal});
    root.addEventListener('resize', cancelSwipe, {signal:controller.signal});
    q('.wx-world').addEventListener('keydown', event => { if (event.target !== q('[data-wx-canvas]') || dialog.open) return; if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); stepArea(event.key === 'ArrowLeft' ? -1 : 1); } }, {signal:controller.signal});
    shell.addEventListener('input', event => { if (event.target.hasAttribute('data-wx-search')) { upgradeSearch = event.target.value; markup(q('[data-wx-destination]'), destinationBody()); } }, { signal:controller.signal });
    dialog.addEventListener('click', click, { signal:controller.signal });
    dialog.addEventListener('cancel', event => { event.preventDefault(); back(); }, { signal:controller.signal });
    return {
      update, notify, close,
      isOpen: () => dialog.open,
      handleBack() { if (dialog.open) { back(); return true; } if (screen !== 'expedition') { screen = 'expedition'; update(view); return true; } return false; },
      showReturn(offline) {
        if (!offline || offline.seconds < 60 || !view || dialog.open || options.overlayOpen()) return;
        const gains = Object.entries(offline.gains || {}).slice(0,4).map(([id, amount]) => ({ id, amount:root.WayfarersCore.format(amount) }));
        open({ kind:'return', time:offline.seconds >= 3600 ? (offline.seconds / 3600).toFixed(1) + ' hours' : Math.floor(offline.seconds / 60) + ' minutes', gains });
      },
      dispose() { disposed = true; controller.abort(); root.clearTimeout(toastTimer); scene.dispose(); dialog.remove(); shell.remove(); delete host.dataset.expeditionMode; }
    };
  }
  root.WayfarersExpeditionUI = { create };
})(typeof globalThis !== 'undefined' ? globalThis : this);
