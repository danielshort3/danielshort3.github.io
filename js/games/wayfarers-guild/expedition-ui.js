(function (root) {
  'use strict';
  const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
  const iconNames = { porters:'backpack', picks:'tools', carts:'crate', furnace:'forge', lift:'equipment', beacon:'observatory', atlas:'maps' };
  const icon = name => root.WayfarersIcons.markup(iconNames[name] || name);
  const propCells = { boots:2, porters:3, picks:4, carts:5, furnace:6, lift:7, crew:1, beacon:9 };
  const areaNames = { greenway:'Trail', quarry:'Quarry', watchtower:'Tower', guild:'Guild' };
  const areaIcons = { greenway:'trail', quarry:'mine', watchtower:'observatory', guild:'guild' };
  const groupNames = { local:'Area upgrades', area:'Area upgrades', development:'Developments', developments:'Developments', research:'Research', legacy:'Guild upgrades', equipment:'Equipment', production:'Production', blueprint:'Blueprints', blueprints:'Blueprints' };
  function localIcon(id) {
    const cell = propCells[id];
    return cell == null ? icon(id) : '<span class="wx-prop" aria-hidden="true" style="background-image:url(&quot;img/wayfarers-guild/expedition-props.png&quot;);background-position:' + ((cell % 4) * 100 / 3) + '% ' + (Math.floor(cell / 4) * 50) + '%"></span>';
  }
  function gainLabel(item) {
    const fields = { boots:['exploration','travel'], porters:['coins'], scouts:['exploration','maps'], picks:['extraction','picks'], carts:['haul','carts'], furnace:['smelting','furnace','ore'], crew:['knowledge','construction'], lift:['construction','repair'], beacon:['knowledge','protection'] }[item.trackId || item.id] || [];
    const changes = (item.impact || []).filter(change => change.current != null && change.next != null);
    const change = fields.map(field => changes.find(candidate => String(candidate.metric).split(':').pop() === field)).find(Boolean) || changes[0];
    if (change) {
      const current = root.WayfarersCore.Numbers.toNumber(change.currentValue ?? change.current);
      const next = root.WayfarersCore.Numbers.toNumber(change.nextValue ?? change.next);
      const key = String(change.metric).split(':').pop();
      const effect = ({ picks:'mining', carts:'haul', furnace:'smelt', construction:item.description?.includes('Expansion') ? 'projects' : 'build' }[key] || String(change.label || '').split('·').pop().trim().toLowerCase().replace('exploration','travel').replace('extraction','mining').replace('hauling','haul').replace('smelting','smelt').replace('construction','build')).replace(' capacity','');
      if (current > 0 && next > current) { const delta = (next / current - 1) * 100; return (delta < .05 ? '<0.1' : '+' + (delta < 1 ? delta.toFixed(1) : Math.round(delta))) + '% ' + effect; }
      if (next > 0) return change.next + ' ' + effect + (change.unit || '');
    }
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
    const celebrated = new Set();
    const viewedStages = new Set();
    let seenSequence = null;
    let disposed = false;
    const shell = document.createElement('section');
    shell.className = 'wx-game';
    shell.setAttribute('aria-label', 'Wayfarers Guild');
    shell.innerHTML = '<header class="wx-header"><button class="wx-wallet" data-wx-wallet aria-label="Resource stockpile"></button><div class="wx-local-count" data-wx-local-count hidden></div><div class="wx-utilities"><button data-wx-reward aria-label="Caravan reward" hidden>' + icon('caravan') + '</button><button data-wx-options aria-label="Settings and app updates">' + icon('settings') + '</button></div></header>' +
      '<div class="wx-objective"><button data-wx-objective><span data-wx-stage-name></span><strong data-wx-objective-text></strong></button><div class="wx-checkpoints" data-wx-checkpoints></div><progress data-wx-progress max="100" value="0" aria-label="Expedition completion"></progress></div>' +
      '<div class="wx-play"><div class="wx-world"><canvas data-wx-canvas role="img" aria-label="Your expedition"></canvas><div class="wx-world-label" data-wx-world-label></div><button class="wx-network-link" data-wx-network hidden></button><div class="wx-hotspots" data-wx-hotspots></div><div class="wx-world-actions" data-wx-world-actions></div></div><div class="wx-tray" data-wx-tray></div></div>' +
      '<section class="wx-destination" data-wx-destination hidden></section><nav class="wx-nav" aria-label="Game destinations">' + ['greenway','quarry','watchtower'].map(id => '<button data-wx-area="' + id + '"' + (id !== 'greenway' ? ' hidden' : '') + '>' + icon(areaIcons[id]) + '<span>' + areaNames[id] + '</span><i data-wx-attention hidden aria-label="New development"></i></button>').join('') + '<button data-wx-nav="upgrades" hidden>' + icon('research') + '<span>Upgrades</span></button><button data-wx-nav="guild" hidden>' + icon('guild') + '<span>Guild</span></button></nav>' +
      '<div class="wx-toast" data-wx-toast role="status" aria-live="polite"></div><button class="wx-save-alert" data-wx-save-alert hidden>Save needs attention</button>';
    const dialog = document.createElement('dialog');
    dialog.className = 'wx-sheet';
    dialog.setAttribute('aria-labelledby', 'wx-sheet-title');
    dialog.innerHTML = '<header><button data-wx-back aria-label="Back">‹</button><h2 id="wx-sheet-title"></h2><button data-wx-close aria-label="Close">×</button></header><div class="wx-sheet-content" data-wx-sheet-content></div>';
    host.dataset.expeditionMode = 'true';
    host.append(shell);
    host.parentElement.append(dialog);
    const q = selector => shell.querySelector(selector);
    const scene = root.WayfarersExpeditionScene.create(q('[data-wx-canvas]'), {
      quiet: !!options.quiet,
      onSelect(target) { if (target.kind === 'upgrade') inspectLocal(target.id); else if (target.kind === 'choice') open({ kind:'choice' }); }
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
    function descriptorKey(item) { return JSON.stringify(item.action || { id:item.id }); }
    function tiles(items) {
      return '<div class="wx-catalog">' + visible(items).map(item => {
        const id = descriptorKey(item);
        descriptors.set(id, item);
        const owned = item.owned && item.maxed;
        const caption = item.selected ? 'Active' : owned ? 'Complete' : item.owned && item.disabled ? 'Owned' : (item.cost || []).length ? costs(item) : 'Choose';
        return '<article class="wx-card" data-ready="' + (!item.disabled && !owned) + '">' +
          button(icon(item.icon || item.id) + '<strong>' + esc(name(item)) + '</strong>' + (item.level ? '<small>Rank ' + esc(item.level) + '</small>' : ''), () => open({ kind:'inspect', id }), { key:'inspect:' + id, className:'wx-card-info', aria:'Inspect ' + name(item) }) +
          '<span class="wx-card-effect">' + esc(item.effectText || item.description || '') + '</span>' +
          button(caption, item.action, { disabled: item.disabled || owned, className:'wx-price', aria:(item.selected ? 'Selected ' : 'Use ') + name(item) + ((item.cost || []).length ? ': ' + item.cost.map(c => c.text).join(', ') : '') }) + '</article>';
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
        const level = item.level == null ? item.rank : item.level;
        const label = name(item);
        const change = (item.impact || []).find(entry => entry.current != null && entry.next != null && !String(entry.metric).startsWith('behavior:'));
        const values = change ? impactValues(change) : [];
        const effect = change ? (change.label || change.metric) + ' ' + values[0] + ' → ' + values[1] + (change.unit || '') : item.effectText || item.description || '';
        const prerequisite = (item.dependencies || []).find(dependency => !dependency.met);
        return '<article class="wx-research-row" data-wx-upgrade="' + esc(item.id) + '" data-ready="' + (!item.disabled && !complete) + '">' +
          button(icon(item.icon || item.trackId || item.id) + '<span><strong>' + esc(label) + (level ? '<small> ' + level + '</small>' : '') + '</strong><span class="wx-research-effect">' + esc(effect) + '</span>' + areaLinks(item) + (prerequisite ? '<small class="wx-prerequisite">' + esc(prerequisite.label) + '</small>' : '') + '</span>', () => open({ kind:'upgrade', id:item.id }), { key:'upgrade-detail:' + item.id, className:'wx-research-info', aria:'Details: ' + label }) +
          button(complete ? '✓' : (item.cost || []).length ? costs(item) : 'Unlock', item.action, { disabled:item.disabled || complete || currentContext.awaitingWallet, className:'wx-price', aria:(complete ? 'Completed ' : 'Buy ') + label + ((item.cost || []).length ? ', ' + item.cost.map(cost => cost.text).join(', ') : '') }) + '</article>';
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
      html += '</div>';
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
    function execute(action) {
      if (typeof action === 'function') { action(); return; }
      if (!action) return;
      if (['challenge'].includes(action.type)) close();
      const result = options.perform(action);
      if (result?.ok && action.type === 'expedition-choice') { notify('Plan selected'); back(); }
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
      let html = '<div class="wx-detail-hero">' + icon(item.icon || item.trackId || item.id) + '<strong>' + esc(local ? 'Rank ' + (item.rank ?? item.level ?? 0) + ' / ' + (item.maxRank ?? item.maxLevel) : item.selected ? 'Active' : completedUpgrade(item) ? 'Complete' : unmet ? 'Locked' : item.level ? 'Rank ' + item.level : item.disabled ? 'Save up' : 'Available') + '</strong></div><p class="wx-effect">' + esc(item.effectText || item.description) + '</p>';
      if (item.description && item.effectText && item.description !== item.effectText) html += '<p class="wx-muted">' + esc(item.description) + '</p>';
      html += areaLinks(item);
      if ((item.impact || []).length) html += '<div class="wx-impact">' + impacts(item) + '</div>';
      if ((item.dependencies || []).length) html += '<ul class="wx-dependencies">' + item.dependencies.map(dependency => '<li data-met="' + !!dependency.met + '"><b aria-hidden="true">' + (dependency.met ? '✓' : '○') + '</b>' + esc(dependency.label) + '</li>').join('') + '</ul>';
      if (item.comparison) html += '<p>' + esc(typeof item.comparison === 'string' ? item.comparison : item.comparison.text) + '</p>';
      if (item.chainOutput) html += '<p class="wx-muted">Whole chain: ' + item.chainOutput.current.toFixed(2) + ' → ' + item.chainOutput.next.toFixed(2) + ' ingots/s</p>';
      if (item.nextMilestone) html += '<div class="wx-milestone">' + icon('mastery') + '<span>' + esc(typeof item.nextMilestone === 'string' ? item.nextMilestone : 'Lv ' + item.nextMilestone.rank + ' · ' + (item.nextMilestone.label || item.nextMilestone.effectText)) + '</span></div>';
      if (item.reason || item.shortageText) html += '<p class="wx-muted">' + esc(item.shortageText || item.reason) + '</p>';
      if ((item.cost || []).length) html += '<p class="wx-muted">Cost: ' + esc(item.cost.map(cost => cost.text).join(' · ')) + '</p>';
      html += button(completedUpgrade(item) ? 'Complete' : (item.cost || []).length ? costs(item) : item.selected ? 'Selected' : 'Choose', item.action, { disabled:item.disabled || completedUpgrade(item) || currentContext.awaitingWallet, className:'wx-confirm' });
      if (!local && view.planning?.unlocked && ['buy','research','project','expedition-development'].includes(item.action?.type) && !item.owned) {
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
      if (model.kind === 'local') { const item = (view.globalUpgrades || []).find(c => c.id === model.catalogId) || e.cards.find(c => c.id === model.id && (!model.areaId || c.action?.areaId === model.areaId || !c.action?.areaId)); title = item?.label || item?.name || 'Upgrade'; html = detailBody(item, true); }
      else if (model.kind === 'upgrade') { const item = (view.globalUpgrades || []).find(c => c.id === model.id); title = item ? name(item) : 'Upgrade'; html = detailBody(item); }
      else if (model.kind === 'inspect') { const item = descriptors.get(model.id); title = item ? name(item) : 'Upgrade'; html = detailBody(item); }
      else if (model.kind === 'choice') { title = 'Expedition choices'; html = visible(e.choices || [e.choice]).map(choice => section(choice.title, choiceBody(choice))).join(''); }
      else if (model.kind === 'choice-metrics') {
        const choice = (e.choices || [e.choice]).find(item => item.id === model.id);
        title = 'Full rate comparison';
        html = (choice?.options || []).map(item => section(item.label + (item.selected ? ' · Selected' : ''), '<p class="wx-muted">' + esc(item.effectText) + '</p>' + ((item.impact || []).length ? '<div class="wx-impact">' + impacts(item) + '</div>' : '<p class="wx-muted">Current working plan.</p>'))).join('');
      }
      else if (model.kind === 'upgrade-filters') {
        title = 'Upgrade effects';
        html = '<div class="wx-options">' + ['all', ...model.effects].map(id => button(id === 'all' ? 'All effects' : id.replace(/[-_]/g,' ').replace(/^./, letter => letter.toUpperCase()), () => { upgradeEffect = id; close(); update(view); q('[data-wx-destination]').scrollTop = 0; }, { key:'filter:effect:' + id, selected:upgradeEffect === id })).join('') + '</div>';
      }
      else if (model.kind === 'objective') {
        title = e.stage.name;
        html = '<p class="wx-effect">' + esc(e.stage.objective) + '</p><ol class="wx-checkpoint-list">' + (e.checkpoints || []).map(c => '<li data-done="' + !!c.complete + '"><b>' + (c.complete ? '✓' : '○') + '</b><span>' + esc(c.label) + '</span></li>').join('') + '</ol>';
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
      view = nextView;
      const e = view.expedition;
      const stage = e.stage;
      if (context) currentContext = context;
      const ctx = currentContext;
      const allDescriptors = ['actions','routes','modes','recipes','research','specialists','companions','doctrines','challenges','automations','refitUpgrades','legacyUpgrades'].flatMap(key => view[key] || []).concat(e.blueprints || [], e.preparation || [], e.supplyChoices || [], view.planning?.capabilities || [], view.development?.projects || [], view.development?.chapter?.projects || [], view.luck?.relics || [], view.luck?.research || [], view.luck?.hunts || [], view.luck?.kits || [], view.globalUpgrades || []);
      allDescriptors.forEach(item => descriptors.set(descriptorKey(item), item));
      const guildUnlocked = view.unlocks.some(id => id !== 'trail');
      const areas = e.areas || ['greenway','quarry','watchtower'].map((id,index) => ({ id, unlocked:stage.index >= index, selected:stage.kind === id }));
      const selectedArea = areas.find(area => area.selected)?.id || stage.kind;
      const upgradesUnlocked = guildUnlocked || areas.filter(area => area.unlocked).length > 1;
      q('[data-wx-nav="guild"]').hidden = !guildUnlocked;
      q('[data-wx-nav="upgrades"]').hidden = !upgradesUnlocked;
      if ((screen === 'guild' || screen === 'atlas') && !guildUnlocked || screen === 'upgrades' && !upgradesUnlocked) screen = 'expedition';
      shell.querySelectorAll('[data-wx-area]').forEach(node => {
        const area = areas.find(item => item.id === node.dataset.wxArea);
        node.hidden = !area?.unlocked;
        node.setAttribute('aria-current', screen === 'expedition' && selectedArea === area?.id ? 'page' : 'false');
        node.setAttribute('aria-label', (areaNames[node.dataset.wxArea] || area?.label) + (area?.rateText ? ', ' + area.rateText : '') + (area?.attention ? ', new development' : ''));
        node.querySelector('[data-wx-attention]').hidden = !area?.attention;
      });
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
      text(q('[data-wx-stage-name]'), screen === 'expedition' ? stage.name : screen === 'guild' ? 'YOUR GUILD' : screen === 'upgrades' ? 'CONNECTED GUILD' : 'THE ATLAS');
      text(q('[data-wx-objective-text]'), screen === 'expedition' ? stage.objective : screen === 'guild' ? 'Build your advantage' : screen === 'upgrades' ? 'Every improvement, one place' : 'Every journey leaves a mark');
      q('[data-wx-objective]').disabled = screen !== 'expedition';
      q('[data-wx-progress]').hidden = screen !== 'expedition';
      q('[data-wx-progress]').value = percent(stage.progress);
      markup(q('[data-wx-checkpoints]'), screen === 'expedition' ? (e.checkpoints || []).map(c => '<span data-complete="' + !!c.complete + '" title="' + esc(c.label) + '" aria-label="' + esc(c.label + (c.complete ? ': complete' : ': pending')) + '"></span>').join('') : '');
      if (screen !== 'expedition') markup(q('[data-wx-destination]'), destinationBody());
      if (previousScreen !== screen) { q('[data-wx-destination]').scrollTop = 0; previousScreen = screen; }
      const shown = e.cards.filter(c => c.visible !== false).slice(0,3);
      const tray = q('[data-wx-tray]');
      tray.dataset.count = shown.length;
      for (const node of Array.from(tray.children)) if (!shown.some(c => c.id === node.dataset.upgrade)) node.remove();
      shown.forEach(item => {
        let node = Array.from(tray.children).find(c => c.dataset.upgrade === item.id);
        if (!node) {
          node = document.createElement('article');
          node.className = 'wx-upgrade';
          node.dataset.upgrade = item.id;
          node.innerHTML = button('<strong>' + esc(item.label) + '</strong>' + localIcon(item.id) + '<small data-wx-rank></small><small class="wx-next-rate" data-wx-next-rate></small>', () => inspectLocal(item.id), { key:'local:' + item.id, className:'wx-upgrade-info', aria:'Inspect ' + item.label }) + '<button class="wx-price" data-wx-buy="' + esc(item.id) + '"></button>';
          tray.append(node);
        }
        text(node.querySelector('[data-wx-rank]'), 'Lv ' + item.rank);
        text(node.querySelector('[data-wx-next-rate]'), item.rank >= item.maxRank ? 'Mastered' : gainLabel(item));
        const buy = node.querySelector('[data-wx-buy]');
        markup(buy, item.rank >= item.maxRank ? 'Max' : costs(item));
        buy.disabled = !!item.disabled || !!ctx.awaitingWallet;
        buy.setAttribute('aria-label', 'Upgrade ' + item.label + ', level ' + item.rank + ', ' + (item.cost || []).map(c => c.text).join(', '));
        node.dataset.ready = String(!buy.disabled);
        if (previousStage === stage.id && previousRanks[item.id] != null && previousRanks[item.id] < item.rank) { node.classList.remove('wx-purchased'); void node.offsetWidth; node.classList.add('wx-purchased'); }
      });
      previousRanks = Object.fromEntries(e.cards.map(c => [c.id,c.rank]));
      previousStage = stage.id;
      if (seenSequence == null) seenSequence = e.sequence;
      if (e.sequence > seenSequence) {
        seenSequence = e.sequence;
        const event = (e.events || []).filter(item => item.kind !== 'stage').pop();
        if (event) notify(event.title + ': ' + event.text);
        root.queueMicrotask(() => { if (!disposed) options.perform({ type:'expedition-seen', sequence:seenSequence }); });
      }
      const bottleneck = (e.stations || []).find(s => s.bottleneck);
      const activeArea = areas.find(area => area.id === selectedArea);
      text(q('[data-wx-world-label]'), bottleneck ? bottleneck.label + ' · ' + (bottleneck.status || 'Bottleneck') : activeArea?.established ? activeArea.rateText || 'Area working' : stage.index === 0 && shown.length === 1 ? 'Coins arrive as you explore' : '');
      const support = visible(view.globalUpgrades || []).find(item => item.owned && (item.targetAreas || []).includes(selectedArea) && (item.sourceAreas || []).some(id => id !== selectedArea));
      const network = q('[data-wx-network]');
      network.hidden = !support;
      if (support) { markup(network, areaLinks(support) + '<span>' + esc(name(support)) + '</span>'); network.setAttribute('aria-label', 'Area connections: ' + name(support)); }
      const choiceCount = visible(e.choices || [e.choice]).length;
      const selectedPlan = e.choice?.options?.find(option => option.selected)?.label;
      const planLabel = choiceCount > 1 ? 'Plans · ' + choiceCount : stage.kind === 'greenway' ? 'Path' : stage.kind === 'quarry' ? 'Processing' : 'Crew plan';
      let worldActions = e.choice?.visible ? button(icon('compass') + '<span>' + esc(planLabel) + '</span><b>›</b>', () => open({ kind:'choice' }), { key:'world-choice', className:'wx-world-button', aria:e.choice.title + (selectedPlan ? ': ' + selectedPlan : '') }) : '';
      if (stage.completed && e.next?.action && !e.next.disabled) worldActions = button(esc(e.next.label) + ' →', e.next.action, { className:'wx-world-button wx-gold', disabled:e.next.disabled });
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
      else if (target.hasAttribute('data-wx-objective')) open({ kind:'objective' });
      else if (target.hasAttribute('data-wx-options')) open({ kind:'options' });
      else if (target.hasAttribute('data-wx-save-alert')) legacy('settings');
      else if (target.hasAttribute('data-wx-reward')) legacy('caravan');
    }
    shell.addEventListener('click', click, { signal:controller.signal });
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
