/* The frame, category links and panel survive route changes. Only their contents change. */
(() => {
  'use strict';
  if (typeof window === 'undefined' || !document.body) return;
  if (window.SiteFrame?.adopt) return;
  const compactQuery = window.matchMedia('(max-width: 959px), (max-height: 619px)');
  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const colors = { about: '#091f3b', projects: '#155dfc', tools: '#087f8c', games: '#c94b0a', resume: '#087f8c', contact: '#334155' };
  const personalOrder = ['about', 'projects', 'tools', 'games', 'contact'];
  const professionalOrder = ['about', 'projects', 'resume', 'contact'];
  const tabs = new Map();
  let frame;
  let stage;
  let panel;
  let slot;
  let canvas;
  let toolbar;
  let viewport;
  let loading;
  let current;
  let desiredTarget;
  let held;
  let flowReservation;
  let body;
  let geometry;
  let wipeMotion;
  let wipeClosed = false;
  let resizeFrame = 0;
  let localSequence = 0;
  let lastWidth = 0;
  let geometryCompletion = Promise.resolve(true);
  const duration = (token, fallback) => reducedQuery.matches ? 0 : (window.SiteMotion?.duration(frame, token, fallback) ?? fallback);
  const make = (tag, className, attribute) => {
    const node = document.createElement(tag);
    node.className = className;
    if (attribute) node.setAttribute(attribute, '');
    return node;
  };
  const rect = (node) => {
    const value = node.getBoundingClientRect();
    return { x: value.x, y: value.y, width: value.width, height: value.height };
  };

  function describe(scope, manifest = {}) {
    const home = scope.querySelector('[data-home-accordion]');
    const shell = scope.querySelector('[data-personal-accordion-shell]');
    if (!home && !shell) throw new Error('The destination has no shared frame.');
    const audience = scope.body?.dataset.audience || 'personal';
    const source = home || shell;
    const category = manifest.category || source.dataset.personalActiveCategory || source.dataset.activePanel || 'about';
    return {
      manifest,
      audience,
      category,
      view: home ? 'overview' : (manifest.view || 'detail'),
      fit: home ? 'viewport' : (scope.body?.dataset.personalFit || 'document'),
      home: Boolean(home),
      source,
      title: scope.title,
      canonical: scope.querySelector('link[rel="canonical"]')?.href || '',
      tabSources: [...source.querySelectorAll('[data-site-tab], [data-home-accordion-trigger]')],
      toolbar: shell?.querySelector('[data-site-route-toolbar]'),
      content: home || shell.querySelector('[data-personal-detail-content]')
    };
  }

  function ensureTab(category, source) {
    let link = tabs.get(category);
    if (!link) {
      link = make('a', 'site-frame__tab', 'data-site-tab');
      link.dataset.siteTab = category;
      link.dataset.siteTabCategory = category;
      link.id = `home-accordion-trigger-${category}`;
      const icon = make('span', 'site-frame__tab-icon');
      icon.setAttribute('aria-hidden', 'true');
      const label = make('span', 'site-frame__tab-label');
      label.textContent = category.charAt(0).toUpperCase() + category.slice(1);
      const notch = make('span', 'site-frame__tab-notch');
      notch.setAttribute('aria-hidden', 'true');
      link.append(icon, label, notch);
      stage.insertBefore(link, panel);
      tabs.set(category, link);
    }
    if (source) {
      const icon = source.querySelector('svg');
      if (icon) link.firstElementChild.replaceChildren(document.importNode(icon, true));
      const label = source.querySelector('[class$="rail-label"]');
      if (label) link.children[1].textContent = label.textContent;
      link.href = source.getAttribute('href') || `/#${category}`;
      link.setAttribute('aria-label', source.getAttribute('aria-label') || link.children[1].textContent);
    }
    link.style.setProperty('--rail-color', colors[category] || colors.about);
    return link;
  }

  function loadContent(description, original = false) {
    const source = original ? description.source : document.importNode(description.source, true);
    const result = { ...description, source, items: null };
    const nextBody = make(description.home ? 'main' : 'div', 'site-frame__body', 'data-site-route-content');
    nextBody.setAttribute('data-site-route-body', '');
    if (!description.home) {
      nextBody.classList.add('personal-accordion__content');
      nextBody.setAttribute('data-personal-detail-content', '');
    }
    if (description.home) {
      nextBody.id = 'main';
      const items = new Map();
      source.querySelectorAll('[data-home-accordion-item]').forEach((item) => {
        const id = item.dataset.homeAccordionItem;
        item.querySelector('.home-accordion__heading')?.remove();
        const contentPanel = item.querySelector('[data-home-accordion-panel]');
        if (contentPanel) {
          contentPanel.hidden = false;
          contentPanel.removeAttribute('inert');
        }
        item.classList.add('is-active');
        items.set(id, item);
      });
      result.items = items;
      nextBody.append(items.get(result.category) || items.values().next().value);
    } else {
      const content = source.querySelector('[data-personal-detail-content]');
      if (!content) throw new Error('The destination content is unavailable.');
      nextBody.append(...content.childNodes);
    }
    result.body = nextBody;
    return result;
  }

  function configure(description) {
    frame.dataset.frameAudience = description.audience;
    frame.dataset.frameView = description.view;
    frame.dataset.frameHome = String(description.home);
    frame.dataset.frameFit = description.fit;
    frame.dataset.frameCategory = description.category;
    frame.classList.toggle('home-accordion', description.home);
    frame.classList.toggle('is-library-mode', description.home && description.view === 'library');
    frame.classList.toggle('personal-accordion', !description.home);
    frame.toggleAttribute('data-home-accordion', description.home);
    frame.toggleAttribute('data-personal-accordion-shell', !description.home);
    frame.dataset.activePanel = description.category;
    frame.dataset.personalActiveCategory = description.category;
    frame.style.setProperty('--panel-color', colors[description.category] || colors.about);
    description.tabSources?.forEach((source) => ensureTab(source.dataset.siteTab || source.dataset.homeAccordionTrigger, source));
    const order = description.audience === 'personal' ? personalOrder : professionalOrder;
    const overview = description.home && description.view === 'overview';
    const visible = overview || description.audience !== 'personal' ? order : [description.category];
    visible.forEach((id) => ensureTab(id));
    tabs.forEach((link, id) => {
      const active = id === description.category;
      link.classList.toggle('is-active', active);
      link.dataset.siteTabActive = String(active);
      link.setAttribute('aria-current', active ? 'page' : 'false');
      link.hidden = !visible.includes(id);
      link.inert = link.hidden;
      link.tabIndex = link.hidden ? -1 : 0;
      if (description.home) {
        link.href = `/#${id}`;
        link.dataset.homeAccordionTrigger = id;
        link.setAttribute('aria-expanded', String(active));
        link.setAttribute('aria-controls', `home-accordion-panel-${id}`);
      } else {
        delete link.dataset.homeAccordionTrigger;
        link.removeAttribute('aria-expanded');
        link.removeAttribute('aria-controls');
      }
    });
    const compact = compactQuery.matches;
    frame.dataset.frameCompact = String(compact);
    const activeIndex = visible.indexOf(description.category);
    if (compact && overview) {
      stage.style.gridTemplateColumns = 'minmax(0, 1fr)';
      stage.style.gridTemplateRows = visible.flatMap((id) => id === description.category ? ['minmax(54px, auto)', 'auto'] : ['minmax(48px, auto)']).join(' ');
      visible.forEach((id, index) => {
        tabs.get(id).style.gridArea = `${index + 1 + (index > activeIndex ? 1 : 0)} / 1`;
      });
      slot.style.gridArea = `${activeIndex + 2} / 1`;
    } else if (compact) {
      stage.style.gridTemplateColumns = `repeat(${visible.length}, minmax(0, 1fr))`;
      stage.style.gridTemplateRows = `minmax(${description.audience !== 'personal' ? 56 : (description.home ? 78 : 48)}px, auto) auto`;
      visible.forEach((id, index) => { tabs.get(id).style.gridArea = `1 / ${index + 1}`; });
      slot.style.gridArea = `2 / 1 / 3 / ${visible.length + 1}`;
    } else {
      const columns = [];
      visible.forEach((id) => {
        columns.push(description.home && overview ? (id === description.category ? '68px' : '64px') : '68px');
        if (overview && id === description.category) columns.push('minmax(0, 1fr)');
      });
      if (!overview) columns.push('minmax(0, 1fr)');
      stage.style.gridTemplateColumns = columns.join(' ');
      stage.style.gridTemplateRows = 'minmax(0, 1fr)';
      visible.forEach((id, index) => {
        tabs.get(id).style.gridArea = `1 / ${index + 1 + (overview && index > activeIndex ? 1 : 0)}`;
      });
      slot.style.gridArea = `1 / ${overview ? activeIndex + 2 : visible.length + 1}`;
    }
    return visible;
  }

  function capture() {
    if (!frame) return null;
    const style = getComputedStyle(stage);
    const children = new Map();
    const tabRects = new Map();
    tabs.forEach((node, id) => {
      const bounds = rect(node);
      tabRects.set(id, bounds);
      children.set(id, [...node.querySelectorAll('.site-frame__tab-icon, .site-frame__tab-label')].map((child) => {
        const box = rect(child);
        const computed = getComputedStyle(child);
        let angle = 0;
        if (computed.transform !== 'none' && typeof DOMMatrix === 'function') {
          const matrix = new DOMMatrix(computed.transform);
          angle = Math.atan2(matrix.b, matrix.a) * 180 / Math.PI;
        }
        return { x: box.x + box.width / 2 - bounds.x, y: box.y + box.height / 2 - bounds.y,
          width: box.width, height: box.height, angle, fontSize: computed.fontSize };
      }));
    });
    const hostStyle = getComputedStyle(frame);
    return {
      category: frame.dataset.frameCategory,
      audience: frame.dataset.frameAudience,
      compact: frame.dataset.frameCompact === 'true',
      scroll: { x: window.scrollX, y: window.scrollY },
      host: { ...rect(frame), paddingTop: hostStyle.paddingTop, paddingRight: hostStyle.paddingRight,
        paddingBottom: hostStyle.paddingBottom, paddingLeft: hostStyle.paddingLeft },
      frame: { ...rect(stage), radius: style.borderRadius, borderWidth: style.borderTopWidth, borderColor: style.borderTopColor, shadow: style.boxShadow },
      panel: rect(panel), slot: rect(slot), tabs: tabRects, children
    };
  }

  function rememberStyle(records, node) {
    records.push({ node, saved: node.getAttribute('style'), hidden: node.hidden });
  }

  function restoreStyles(records) {
    records.forEach(({ node, saved, hidden }) => {
      if (saved == null) node.removeAttribute('style'); else node.setAttribute('style', saved);
      node.hidden = hidden;
    });
  }

  function pinBox(node, bounds, origin) {
    Object.assign(node.style, {
      position: 'absolute', gridArea: 'auto', margin: '0', minWidth: '0', minHeight: '0', maxWidth: 'none',
      left: `${bounds.x - origin.x}px`, top: `${bounds.y - origin.y}px`, right: 'auto', bottom: 'auto',
      width: `${bounds.width}px`, height: `${bounds.height}px`
    });
  }

  function pinSlot(bounds, origin, insets) {
    const style = insets || getComputedStyle(slot);
    const top = parseFloat(style.paddingTop || 0);
    const right = parseFloat(style.paddingRight || 0);
    const bottom = parseFloat(style.paddingBottom || 0);
    const left = parseFloat(style.paddingLeft || 0);
    const horizontal = left + right ? Math.min(1, bounds.width / (left + right)) : 1;
    const vertical = top + bottom ? Math.min(1, bounds.height / (top + bottom)) : 1;
    pinBox(slot, bounds, origin);
    // Padding must fit a closing slot too; an otherwise empty border box cannot
    // retain eight pixels of padding outside the moving panel boundary.
    slot.style.padding = `${top * vertical}px ${right * horizontal}px ${bottom * vertical}px ${left * horizontal}px`;
    slot.style.setProperty('--frame-slot-border-width', slot.style.padding);
  }

  function clearHold() {
    if (!held) return;
    restoreStyles(held.records);
    held = null;
    frame.classList.remove('site-frame--held');
  }

  function reserveFlow() {
    const root = document.documentElement;
    if (!flowReservation) {
      flowReservation = { value: root.style.getPropertyValue('min-height'), priority: root.style.getPropertyPriority('min-height') };
    }
    root.style.setProperty('min-height', `${Math.max(root.scrollHeight, window.scrollY + window.innerHeight)}px`, 'important');
    return { x: window.scrollX, y: window.scrollY };
  }

  function releaseFlow(scroll) {
    if (flowReservation) {
      const saved = flowReservation;
      flowReservation = null;
      if (saved.value) document.documentElement.style.setProperty('min-height', saved.value, saved.priority);
      else document.documentElement.style.removeProperty('min-height');
    }
    if (scroll && (window.scrollX !== scroll.x || window.scrollY !== scroll.y)) {
      window.scrollTo({ left: scroll.x, top: scroll.y, behavior: 'instant' });
    }
  }

  function guardLayout(operation) {
    let completed = false;
    try {
      const result = operation();
      completed = true;
      return result;
    } finally {
      if (!completed) {
        geometry?.finish(false);
        clearHold();
        releaseFlow();
      }
    }
  }

  // Keep the departing frame and its document-flow space intact while a mounted
  // destination is being prepared beneath the closed content wipe.
  function hold(options = {}) {
    return guardLayout(() => holdFrame(options));
  }

  function holdFrame(options) {
    if (!frame) return null;
    const from = options.from || capture();
    const scroll = reserveFlow();
    geometry?.finish(false);
    clearHold();
    const records = [];
    [frame, stage, panel, slot, canvas].forEach((node) => rememberStyle(records, node));
    Object.assign(frame.style, {
      height: `${from.host.height}px`, minHeight: '0', flex: '0 0 auto',
      paddingTop: from.host.paddingTop, paddingRight: from.host.paddingRight,
      paddingBottom: from.host.paddingBottom, paddingLeft: from.host.paddingLeft
    });
    Object.assign(stage.style, {
      height: `${from.frame.height}px`, width: `${from.frame.width}px`, minHeight: '0', maxWidth: 'none', marginInline: '0',
      borderWidth: from.frame.borderWidth, borderColor: from.frame.borderColor, borderRadius: from.frame.radius, boxShadow: from.frame.shadow
    });
    stage.style.marginLeft = `${from.frame.x + (from.scroll?.x || 0) - frame.getBoundingClientRect().x - window.scrollX - parseFloat(from.host.paddingLeft || 0)}px`;
    const stageBox = rect(stage);
    stage.style.transform = `translateY(${from.frame.y + (from.scroll?.y || 0) - stageBox.y - window.scrollY}px)`;
    Object.assign(panel.style, { position: 'absolute', inset: '0', display: 'block', gridArea: 'auto' });
    const origin = { x: from.frame.x + parseFloat(from.frame.borderWidth || 0), y: from.frame.y + parseFloat(from.frame.borderWidth || 0) };
    tabs.forEach((node, id) => {
      rememberStyle(records, node);
      const bounds = from.tabs.get(id);
      node.hidden = !bounds || !bounds.width || !bounds.height;
      if (node.hidden) return;
      pinBox(node, bounds, origin);
      [...node.querySelectorAll('.site-frame__tab-icon, .site-frame__tab-label')].forEach((child, index) => {
        const position = from.children?.get(id)?.[index];
        if (!position) return;
        rememberStyle(records, child);
        const icon = child.classList.contains('site-frame__tab-icon');
        Object.assign(child.style, {
          position: 'absolute', left: `${position.x}px`, top: `${position.y}px`, margin: '0', fontSize: position.fontSize,
          transform: `translate(-50%, -50%) rotate(${position.angle}deg)`,
          width: icon ? `${position.width}px` : 'max-content', height: icon ? `${position.height}px` : 'auto'
        });
      });
    });
    pinSlot(from.slot, origin);
    held = { from, records, scroll };
    frame.classList.add('site-frame--held');
    updateBoundary();
    updateLoadingPosition();
    return from;
  }

  function release(options = {}) {
    if (!frame || !desiredTarget) return Promise.resolve(false);
    const from = options.from || (held?.refreshing ? capture() : held?.from) || capture();
    clearHold();
    return transition(desiredTarget, { ...options, from });
  }

  function transition(description = current, options = {}) {
    return guardLayout(() => transitionFrame(description, options));
  }

  function collapsedTabBounds(layout, reference, id, active) {
    if (compactQuery.matches) return { x: active.x, y: active.y, width: active.width, height: 0 };
    const order = reference.audience === 'personal' ? personalOrder : professionalOrder;
    const tab = reference.tabs.get(id);
    // Professional rails are all on the left. Follow the visible placement
    // rather than assuming every category after the selection sits on the right.
    const right = !reference.compact && tab?.width > 0 && tab?.height > 0
      ? tab.x >= reference.slot.x + reference.slot.width - 1
      : order.indexOf(id) > order.indexOf(reference.category);
    const border = parseFloat(layout.frame.borderWidth || 0);
    return {
      x: right ? layout.frame.x + layout.frame.width - border : layout.frame.x + border,
      y: active.y, width: 0, height: active.height
    };
  }

  function transitionFrame(description, options) {
    if (!frame || !description) return Promise.resolve(false);
    const before = options.from || capture();
    const scroll = reserveFlow();
    geometry?.finish(false);
    clearHold();
    desiredTarget = description;
    const visible = configure(description);
    // A pending mount has not established the destination's natural height.
    // Resize its rails and width while retaining the departing block size.
    const measuring = [];
    if (options.holdHeight != null) {
      rememberStyle(measuring, frame);
      rememberStyle(measuring, stage);
      frame.style.paddingTop = before.host.paddingTop;
      frame.style.paddingBottom = before.host.paddingBottom;
      stage.style.height = `${options.holdHeight}px`;
      stage.style.minHeight = '0';
    }
    const after = capture();
    restoreStyles(measuring);
    const milliseconds = options.animate === false || reducedQuery.matches ? 0 :
      (typeof options.duration === 'number' ? options.duration : duration('--site-frame-geometry-duration', 320));
    if (!milliseconds || !stage.animate) {
      if (options.holdHeight != null) hold({ from: after });
      else releaseFlow();
      updateBoundary();
      return (geometryCompletion = Promise.resolve(true));
    }
    const records = [];
    const animations = [];
    const pin = (node, first, last) => {
      rememberStyle(records, node);
      node.hidden = false;
      pinBox(node, last, { x: after.frame.x + parseFloat(after.frame.borderWidth || 0), y: after.frame.y + parseFloat(after.frame.borderWidth || 0) });
      animations.push(node.animate([
        {
          transform: `translate(${first.x - before.frame.x - parseFloat(before.frame.borderWidth || 0) - last.x + after.frame.x + parseFloat(after.frame.borderWidth || 0)}px, ${first.y - before.frame.y - parseFloat(before.frame.borderWidth || 0) - last.y + after.frame.y + parseFloat(after.frame.borderWidth || 0)}px)`,
          width: `${first.width}px`, height: `${first.height}px`
        },
        { transform: 'translate(0, 0)', width: `${last.width}px`, height: `${last.height}px` }
      ], { duration: milliseconds, easing: 'cubic-bezier(.22, 1, .36, 1)', fill: 'both' }));
    };
    const active = after.tabs.get(description.category) || after.slot;
    const previousTab = before.tabs.get(before.category);
    const previousActive = previousTab?.width > 0 && previousTab?.height > 0 ? previousTab : before.slot;
    const retractingRightTabs = [];
    tabs.forEach((node, id) => {
      let first = before.tabs.get(id);
      if (!first || (!first.width && !first.height)) {
        first = collapsedTabBounds(before, after, id, previousActive);
      }
      const last = visible.includes(id) ? after.tabs.get(id) :
        collapsedTabBounds(after, before, id, active);
      if (!compactQuery.matches && !visible.includes(id) && first.width > 0 && first.x >= before.slot.x + before.slot.width - 1) {
        retractingRightTabs.push(node);
      }
      if (first.width || first.height || visible.includes(id)) pin(node, first, last);
      const firstChildren = before.children?.get(id);
      const lastChildren = after.children?.get(id);
      if (visible.includes(id) && firstChildren && before.tabs.get(id)?.width) {
        [...node.querySelectorAll('.site-frame__tab-icon, .site-frame__tab-label')].forEach((child, index) => {
          const start = firstChildren[index];
          const end = lastChildren[index];
          if (!start || !end) return;
          const icon = child.classList.contains('site-frame__tab-icon');
          rememberStyle(records, child);
          Object.assign(child.style, {
            position: 'absolute', left: `${end.x}px`, top: `${end.y}px`, margin: '0',
            transform: `translate(-50%, -50%) rotate(${end.angle}deg)`,
            width: icon ? `${end.width}px` : 'max-content', height: icon ? `${end.height}px` : 'auto'
          });
          animations.push(child.animate([
            { left: `${start.x}px`, top: `${start.y}px`, fontSize: start.fontSize, transform: `translate(-50%, -50%) rotate(${start.angle}deg)` },
            { left: `${end.x}px`, top: `${end.y}px`, fontSize: end.fontSize, transform: `translate(-50%, -50%) rotate(${end.angle}deg)` }
          ], { duration: milliseconds, easing: 'cubic-bezier(.22, 1, .36, 1)', fill: 'both' }));
        });
      }
    });
    [frame, stage, panel, slot, canvas].forEach((node) => rememberStyle(records, node));
    // The host owns document flow; its stage owns the visible border. Animate
    // both from the same endpoints so neither can temporarily cover the footer.
    Object.assign(frame.style, {
      height: `${before.host.height}px`, minHeight: '0', flex: '0 0 auto',
      paddingTop: after.host.paddingTop, paddingBottom: after.host.paddingBottom
    });
    stage.style.height = `${before.frame.height}px`;
    stage.style.width = `${after.frame.width}px`;
    stage.style.minHeight = '0';
    stage.style.maxWidth = 'none';
    stage.style.marginInline = '0';
    stage.style.marginLeft = `${after.frame.x - frame.getBoundingClientRect().x - parseFloat(getComputedStyle(frame).paddingLeft || 0)}px`;
    // The panel never retracts from the stage. Only its content slot changes,
    // leaving an uninterrupted background and border beneath crossing rails.
    Object.assign(panel.style, { position: 'absolute', inset: '0', display: 'block', gridArea: 'auto' });
    const slotStyle = getComputedStyle(slot);
    const slotInsets = { paddingTop: slotStyle.paddingTop, paddingRight: slotStyle.paddingRight,
      paddingBottom: slotStyle.paddingBottom, paddingLeft: slotStyle.paddingLeft };
    const insetLeft = parseFloat(slotStyle.paddingLeft || 0);
    const insetTop = parseFloat(slotStyle.paddingTop || 0);
    Object.assign(canvas.style, {
      position: 'absolute', left: `${insetLeft}px`, top: `${insetTop}px`,
      width: `${Math.max(0, after.slot.width - insetLeft - parseFloat(slotStyle.paddingRight || 0))}px`,
      height: `${Math.max(0, after.slot.height - insetTop - parseFloat(slotStyle.paddingBottom || 0))}px`
    });
    // Keep destination text at its measured width instead of reflowing it on
    // every frame. The outer slot clips to the space between the moving rails.
    const syncSlot = () => {
      const area = rect(panel);
      let left = area.x;
      let top = area.y;
      let right = area.x + area.width;
      let bottom = area.y + area.height;
      const boxes = visible.map((id) => rect(tabs.get(id)));
      const selected = visible.indexOf(description.category);
      const overview = description.home && description.view === 'overview';
      if (compactQuery.matches) {
        top = overview ? boxes[selected].y + boxes[selected].height : Math.max(...boxes.map((box) => box.y + box.height));
        if (overview && boxes[selected + 1]) bottom = boxes[selected + 1].y;
      } else {
        left = overview ? boxes[selected].x + boxes[selected].width : Math.max(...boxes.map((box) => box.x + box.width));
        if (overview && boxes[selected + 1]) right = boxes[selected + 1].x;
        // Let the content expand into the space the right-hand rails actually
        // release, while their left-hand counterparts retract behind the selection.
        retractingRightTabs.forEach((node) => {
          const bounds = rect(node);
          if (bounds.width > 0) right = Math.min(right, bounds.x);
        });
      }
      left = Math.max(area.x, Math.min(left, area.x + area.width));
      top = Math.max(area.y, Math.min(top, area.y + area.height));
      right = Math.max(left, Math.min(right, area.x + area.width));
      bottom = Math.max(top, Math.min(bottom, area.y + area.height));
      pinSlot({ x: left, y: top, width: right - left, height: bottom - top }, area, slotInsets);
      updateBoundary();
      updateLoadingPosition();
    };
    frame.classList.add('site-frame--moving');
    animations.push(frame.animate([
      { height: `${before.host.height}px` }, { height: `${after.host.height}px` }
    ], { duration: milliseconds, easing: 'cubic-bezier(.22, 1, .36, 1)', fill: 'both' }));
    animations.push(stage.animate([{
      transform: `translate(${before.frame.x + (before.scroll?.x || 0) - after.frame.x - (after.scroll?.x || 0)}px, ${before.frame.y + (before.scroll?.y || 0) - after.frame.y - (after.scroll?.y || 0)}px)`,
      width: `${before.frame.width}px`, height: `${before.frame.height}px`, borderRadius: before.frame.radius,
      borderWidth: before.frame.borderWidth, borderColor: before.frame.borderColor, boxShadow: before.frame.shadow
    }, {
      transform: 'translate(0, 0)', width: `${after.frame.width}px`, height: `${after.frame.height}px`,
      borderRadius: after.frame.radius, borderWidth: after.frame.borderWidth, borderColor: after.frame.borderColor, boxShadow: after.frame.shadow
    }], {
      duration: milliseconds, easing: 'cubic-bezier(.22, 1, .36, 1)', fill: 'both'
    }));
    releaseFlow(scroll);
    syncSlot();
    const pendingHold = options.holdHeight != null ? { from: before, records: [], refreshing: true } : null;
    if (pendingHold) {
      held = pendingHold;
      frame.classList.add('site-frame--held');
    }
    return (geometryCompletion = new Promise((resolve) => {
      let timer;
      let animationFrame;
      let settled = false;
      const finish = (completed) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        window.cancelAnimationFrame(animationFrame);
        options.signal?.removeEventListener('abort', abort);
        const holdAfter = completed && pendingHold && held === pendingHold ? capture() : null;
        animations.forEach((animation) => animation.cancel());
        restoreStyles(records);
        frame.classList.remove('site-frame--moving');
        if (geometry?.finish === finish) geometry = null;
        if (held === pendingHold && pendingHold) clearHold();
        if (holdAfter) hold({ from: holdAfter });
        updateBoundary();
        updateLoadingPosition();
        resolve(completed);
      };
      const tick = () => {
        if (settled) return;
        syncSlot();
        animationFrame = window.requestAnimationFrame(tick);
      };
      const abort = () => finish(false);
      geometry = { finish, target: description };
      animationFrame = window.requestAnimationFrame(tick);
      Promise.all(animations.map((animation) => animation.finished)).then(() => finish(true), () => finish(false));
      timer = window.setTimeout(() => finish(true), milliseconds + 80);
      options.signal?.addEventListener('abort', abort, { once: true });
    }));
  }

  function wipe(open, options = {}) {
    if (!viewport) return Promise.resolve(true);
    const first = getComputedStyle(viewport).clipPath;
    wipeMotion?.finish(false);
    wipeClosed = !open;
    const last = open ? 'inset(0% 0% 0% 0%)' : (compactQuery.matches ? 'inset(0% 0% 100% 0%)' : 'inset(0% 100% 0% 0%)');
    viewport.style.clipPath = last;
    viewport.inert = !open;
    const milliseconds = duration('--site-frame-wipe-duration', 160);
    if (!milliseconds || !viewport.animate) return Promise.resolve(true);
    const animation = viewport.animate([{ clipPath: first === 'none' ? 'inset(0% 0% 0% 0%)' : first }, { clipPath: last }], {
      duration: milliseconds, easing: 'cubic-bezier(.4, 0, .2, 1)', fill: 'both'
    });
    return new Promise((resolve) => {
      let timer;
      let settled = false;
      const finish = (completed) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        options.signal?.removeEventListener('abort', abort);
        const stopped = completed ? last : getComputedStyle(viewport).clipPath;
        animation.cancel();
        viewport.style.clipPath = stopped;
        if (wipeMotion?.finish === finish) wipeMotion = null;
        resolve(completed);
      };
      const abort = () => finish(false);
      wipeMotion = { finish };
      animation.finished.then(() => finish(true), () => finish(false));
      timer = window.setTimeout(() => finish(true), milliseconds + 80);
      options.signal?.addEventListener('abort', abort, { once: true });
    });
  }

  function setLoading(pending) {
    if (!loading) return;
    loading.hidden = !pending;
    panel.setAttribute('aria-busy', String(Boolean(pending)));
    updateLoadingPosition();
  }

  function updateLoadingPosition() {
    if (!loading || loading.hidden || !panel) return;
    // The moving content slot can briefly be below the viewport while rails
    // change orientation. The uninterrupted panel remains a visible anchor.
    const bounds = rect(panel);
    const header = document.querySelector('[data-site-shell-header]');
    const headerBottom = header ? Math.max(0, header.getBoundingClientRect().bottom) : 0;
    const screen = window.visualViewport;
    const screenLeft = screen?.offsetLeft || 0;
    const screenTop = screen?.offsetTop || 0;
    const screenRight = screenLeft + (screen?.width || window.innerWidth);
    const screenBottom = screenTop + (screen?.height || window.innerHeight);
    const top = Math.max(0, Math.max(headerBottom, screenTop) - bounds.y);
    const left = Math.max(0, screenLeft - bounds.x);
    const available = Math.max(0, Math.min(bounds.height - top, screenBottom - Math.max(bounds.y, headerBottom, screenTop) - 16));
    loading.style.top = `${top}px`;
    loading.style.left = `${left}px`;
    loading.style.width = `${Math.max(0, Math.min(bounds.width - left, screenRight - Math.max(bounds.x, screenLeft)))}px`;
    loading.style.height = `${available}px`;
  }

  function updateBoundary() {
    if (!frame || !stage) return;
    const host = rect(frame);
    const bounds = rect(stage);
    const content = rect(slot);
    const style = getComputedStyle(stage);
    const border = parseFloat(style.borderTopWidth || 0);
    const rightGap = Math.max(0, bounds.x + bounds.width - border - content.x - content.width);
    const topGap = Math.max(0, content.y - bounds.y - border);
    const bottomGap = Math.max(0, bounds.y + bounds.height - border - content.y - content.height);
    // The content border stays square beside another rail, then follows the
    // rounded stage corner as that rail retracts and releases the outer edge.
    frame.style.setProperty('--frame-slot-radius-top-right', `${Math.max(0, parseFloat(style.borderTopRightRadius || 0) - border - Math.max(rightGap, topGap))}px`);
    frame.style.setProperty('--frame-slot-radius-bottom-right', `${Math.max(0, parseFloat(style.borderBottomRightRadius || 0) - border - Math.max(rightGap, bottomGap))}px`);
    const left = Math.max(host.x, bounds.x);
    const top = Math.max(host.y, bounds.y);
    const right = Math.min(host.x + host.width, bounds.x + bounds.width);
    const bottom = Math.min(host.y + host.height, bounds.y + bounds.height);
    frame.style.setProperty('--frame-clip-left', `${Math.max(0, left - host.x)}px`);
    frame.style.setProperty('--frame-clip-top', `${Math.max(0, top - host.y)}px`);
    frame.style.setProperty('--frame-clip-right', `${Math.max(0, host.x + host.width - right)}px`);
    frame.style.setProperty('--frame-clip-bottom', `${Math.max(0, host.y + host.height - bottom)}px`);
    frame.style.setProperty('--frame-clip-border-left', bounds.x < host.x - .5 ? '4px' : '0px');
    frame.style.setProperty('--frame-clip-border-top', bounds.y < host.y - .5 ? '4px' : '0px');
    frame.style.setProperty('--frame-clip-border-right', bounds.x + bounds.width > host.x + host.width + .5 ? '4px' : '0px');
    frame.style.setProperty('--frame-clip-border-bottom', bounds.y + bounds.height > host.y + host.height + .5 ? '4px' : '0px');
  }

  function commit(description, options = {}) {
    const from = (held?.refreshing ? capture() : held?.from) || options.from || capture();
    const next = loadContent(description, Boolean(options.original));
    current = next;
    desiredTarget = next;
    body = next.body;
    viewport.replaceChildren(body);
    toolbar.replaceChildren(...(description.toolbar ? [...document.importNode(description.toolbar, true).childNodes] : []));
    toolbar.hidden = !description.toolbar;
    if (options.defer) prepareHeldTarget(next, from);
    else transition(next, { from, animate: options.animate !== false });
    return body;
  }

  function prepareHeldTarget(description, from) {
    return guardLayout(() => prepareHeldFrame(description, from));
  }

  function prepareHeldFrame(description, from) {
    const scroll = reserveFlow();
    geometry?.finish(false);
    clearHold();
    configure(description);
    const target = capture();
    hold({ from });
    const style = getComputedStyle(slot);
    Object.assign(canvas.style, {
      position: 'absolute', left: style.paddingLeft, top: style.paddingTop,
      width: `${Math.max(0, target.slot.width - parseFloat(style.paddingLeft || 0) - parseFloat(style.paddingRight || 0))}px`,
      height: `${Math.max(0, target.slot.height - parseFloat(style.paddingTop || 0) - parseFloat(style.paddingBottom || 0))}px`
    });
    if (window.scrollX !== scroll.x || window.scrollY !== scroll.y) {
      window.scrollTo({ left: scroll.x, top: scroll.y, behavior: 'instant' });
    }
  }

  function snapshot() {
    return current ? { description: current, body, toolbar: [...toolbar.childNodes] } : null;
  }

  function restore(saved, options = {}) {
    if (!saved) return;
    const from = (held?.refreshing ? capture() : held?.from) || options.from || capture();
    current = saved.description;
    desiredTarget = current;
    body = saved.body;
    viewport.replaceChildren(body);
    toolbar.replaceChildren(...saved.toolbar);
    toolbar.hidden = !saved.toolbar.length;
    if (options.defer) prepareHeldTarget(current, from);
    else transition(current, { from, animate: options.animate !== false });
    setLoading(false);
    return body;
  }

  function showHome(category, view, options = {}) {
    if (!current?.home || !current.items?.has(category)) return Promise.resolve(false);
    const sequence = ++localSequence;
    const next = { ...current, category, view, manifest: { ...current.manifest, category, view } };
    desiredTarget = next;
    const mounting = Boolean(held && options.animate === false);
    if (!mounting && options.animate !== false) hold();
    const closing = options.animate === false ? Promise.resolve(true) : wipe(false);
    return closing.then(async () => {
      if (sequence !== localSequence || !current?.home) return false;
      current = next;
      const item = next.items.get(category);
      const scroller = item.querySelector('[data-home-accordion-scroller]');
      if (scroller) [...scroller.children].forEach((child) => {
        const library = child.hasAttribute('data-home-library-view');
        child.hidden = view === 'library' ? !library : library;
        child.inert = child.hidden;
      });
      body.replaceChildren(item);
      if (mounting) { prepareHeldTarget(next, held.refreshing ? capture() : held.from); return true; }
      const moving = release({ animate: options.animate !== false });
      await Promise.all([moving, options.animate !== false ? wipe(true) : Promise.resolve(true)]);
      return sequence === localSequence;
    });
  }

  function hasFrameInteractionLayer() {
    if (document.fullscreenElement || document.pointerLockElement ||
        document.body.classList.contains('modal-open') || document.body.classList.contains('media-viewer-open') ||
        document.querySelector('dialog[open], .modal.active, .modal[data-motion-state="closing"], .nav-item.dropdown-open')) return true;
    return [...document.querySelectorAll('[aria-modal="true"], [popover], .nav-dropdown, [data-tools-account="disclosure"], [role="menu"], [role="listbox"]')]
      .some((node) => !node.closest('[hidden], [inert], [aria-hidden="true"]') &&
        node.getClientRects().length > 0 && getComputedStyle(node).visibility !== 'hidden');
  }

  function scrollFromFrameChrome(event) {
    if (event.defaultPrevented || !event.cancelable || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey ||
        !Number.isFinite(event.deltaY) || !event.deltaY || Math.abs(event.deltaX) >= Math.abs(event.deltaY) ||
        !frame?.isConnected || !viewport || current?.fit !== 'viewport' || compactQuery.matches ||
        geometry || held || wipeMotion || wipeClosed || viewport.closest('[inert], [aria-hidden="true"]')) return;
    const target = event.target?.nodeType === 1 ? event.target : event.target?.parentElement;
    // Content and embedded projects retain their native scrolling. Only the
    // fixed shell's surrounding chrome needs to forward wheel input.
    if (!target || viewport.contains(target) ||
        (target !== document.body && target !== document.documentElement && !frame.contains(target) &&
          !target.closest('[data-site-shell-header], [data-site-shell-footer]')) ||
        target.closest('[inert], input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="slider"], [role="spinbutton"]')) return;
    const maximum = viewport.scrollHeight - viewport.clientHeight;
    if (maximum <= 1 || !/^(auto|scroll)$/.test(getComputedStyle(viewport).overflowY) || hasFrameInteractionLayer()) return;
    // A toolbar or header can contain an independent scrolling area. Leave its
    // wheel behavior intact, including when it reaches its own boundary.
    for (let node = target; node && node !== document.body && node !== document.documentElement; node = node.parentElement) {
      const style = getComputedStyle(node);
      if ((node.scrollHeight > node.clientHeight + 1 && /^(auto|scroll)$/.test(style.overflowY)) ||
          (node.scrollWidth > node.clientWidth + 1 && /^(auto|scroll)$/.test(style.overflowX))) return;
    }
    const unit = event.deltaMode === 1 ? (parseFloat(getComputedStyle(viewport).lineHeight) || 16) :
      (event.deltaMode === 2 ? viewport.clientHeight : 1);
    const top = Math.max(0, Math.min(maximum, viewport.scrollTop + event.deltaY * unit));
    if (top === viewport.scrollTop) return;
    event.preventDefault();
    viewport.scrollTo({ top, behavior: 'instant' });
  }

  function adopt() {
    if (frame?.isConnected) return frame;
    const manifestNode = document.querySelector('[data-site-route-manifest]');
    let manifest = {};
    try { manifest = JSON.parse(manifestNode?.textContent || '{}'); } catch (_) {}
    if (manifest.navigation === 'hard') return null;
    let description;
    try { description = describe(document, manifest); } catch (_) { return null; }
    const outlet = document.querySelector('[data-site-route-content]');
    if (!outlet) return null;
    frame = make('section', 'site-frame', 'data-site-persistent-shell');
    stage = make('div', 'site-frame__stage', 'data-site-frame-stage');
    panel = make('div', 'site-frame__panel', 'data-site-route-panel');
    slot = make('div', 'site-frame__slot', 'data-site-frame-slot');
    canvas = make('div', 'site-frame__slot-content', 'data-site-frame-slot-content');
    toolbar = make('div', 'personal-accordion__toolbar site-frame__toolbar', 'data-site-route-toolbar');
    viewport = make('div', 'site-frame__viewport', 'data-site-frame-viewport');
    loading = make('div', 'site-frame__loading', 'data-site-frame-loading');
    loading.setAttribute('role', 'status');
    loading.setAttribute('aria-label', 'Loading page');
    loading.hidden = true;
    loading.append(make('span', 'site-frame__loading-bar'));
    canvas.append(toolbar, viewport);
    slot.append(canvas);
    panel.append(slot, loading);
    stage.append(panel);
    frame.append(stage);
    description.tabSources.forEach((source) => ensureTab(source.dataset.siteTab || source.dataset.homeAccordionTrigger, source));
    outlet.replaceWith(frame);
    commit(description, { original: true, animate: false });
    lastWidth = frame.clientWidth;
    return frame;
  }

  function refresh() {
    updateBoundary();
    updateLoadingPosition();
    if (!frame || !current || resizeFrame) return;
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = 0;
      const from = capture();
      if (held) {
        // Rotation can happen during an asynchronous mount. Retarget the held
        // frame without exposing its content, then keep holding its new shape.
        transition(desiredTarget || current, { from, holdHeight: from.frame.height });
        return;
      }
      transition(desiredTarget || current, { from });
    });
  }
  window.SiteFrame = Object.freeze({
    adopt, describe, capture, transition, wipe, commit, snapshot, restore, showHome, setLoading, refresh, hold, release,
    root: () => frame,
    outlet: () => body,
    viewport: () => viewport,
    homeState: () => current?.home ? current : null,
    current: () => current,
    whenSettled: () => geometryCompletion,
    tabs: () => tabs,
    cancel: () => { localSequence += 1; geometry?.finish(false); clearHold(); releaseFlow(); wipeMotion?.finish(false); },
    finish: () => { geometry?.finish(true); wipeMotion?.finish(true); }
  });
  adopt();
  document.addEventListener('wheel', scrollFromFrameChrome, { passive: false });
  window.addEventListener('resize', refresh, { passive: true });
  window.addEventListener('scroll', updateLoadingPosition, { passive: true });
  window.visualViewport?.addEventListener('resize', refresh, { passive: true });
  window.visualViewport?.addEventListener('scroll', updateLoadingPosition, { passive: true });
  compactQuery.addEventListener?.('change', refresh);
  reducedQuery.addEventListener?.('change', () => {
    geometry?.finish(true);
    wipeMotion?.finish(true);
    if (held && reducedQuery.matches) refresh();
  });
  if (typeof ResizeObserver === 'function' && frame) {
    new ResizeObserver(() => {
      if (frame.clientWidth !== lastWidth) { lastWidth = frame.clientWidth; refresh(); }
    }).observe(frame);
  }
})();
