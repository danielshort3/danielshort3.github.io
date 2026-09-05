const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {
  getHomeAccordionIconDefinitions,
  getWidgetDefinitions,
  renderVisualPageBody,
  resolveHomeAccordionIconId
} = require('../../api/_lib/cms-widgets');
const {
  GENERATED_HOME_LIBRARY_VISUALS,
  RETAINED_GAME_PREVIEW_IDS,
  RETAINED_PROJECT_PREVIEW_IDS,
  RETAINED_TOOL_PREVIEW_IDS,
  projectLibraryAsset
} = require('../../build/validate-home-library-visuals');

const ROOT = path.resolve(__dirname, '..', '..');

const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const readJson = (relativePath) => JSON.parse(read(relativePath));
const count = (value, pattern) => (String(value || '').match(pattern) || []).length;
const extractBlock = (source, marker) => {
  const markerIndex = source.indexOf(marker);
  if (markerIndex < 0) return '';
  const openIndex = source.indexOf('{', markerIndex);
  if (openIndex < 0) return '';
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') depth -= 1;
    if (depth === 0) return source.slice(markerIndex, index + 1);
  }
  return '';
};
const extractFunctionBlock = (source, marker) => {
  const markerIndex = source.indexOf(marker);
  if (markerIndex < 0) return '';
  const signatureEnd = source.indexOf(')', markerIndex + marker.length);
  if (signatureEnd < 0) return '';
  const openIndex = source.indexOf('{', signatureEnd + 1);
  if (openIndex < 0) return '';
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') depth -= 1;
    if (depth === 0) return source.slice(markerIndex, index + 1);
  }
  return '';
};
const readWebpDimensions = (relativePath) => {
  const diskPath = path.join(ROOT, String(relativePath || '').replace(/^[/\\]+/, ''));
  const buffer = fs.readFileSync(diskPath);
  if (buffer.length < 20 ||
    buffer.toString('ascii', 0, 4) !== 'RIFF' ||
    buffer.toString('ascii', 8, 12) !== 'WEBP') {
    throw new Error(`${relativePath} is not a valid WebP file`);
  }

  let offset = 12;
  while (offset + 8 <= buffer.length) {
    const chunkType = buffer.toString('ascii', offset, offset + 4);
    const chunkLength = buffer.readUInt32LE(offset + 4);
    const dataOffset = offset + 8;
    if (dataOffset + chunkLength > buffer.length) break;

    if (chunkType === 'VP8X' && chunkLength >= 10) {
      return {
        width: buffer.readUIntLE(dataOffset + 4, 3) + 1,
        height: buffer.readUIntLE(dataOffset + 7, 3) + 1
      };
    }
    if (chunkType === 'VP8 ' && chunkLength >= 10 &&
      buffer[dataOffset + 3] === 0x9d &&
      buffer[dataOffset + 4] === 0x01 &&
      buffer[dataOffset + 5] === 0x2a) {
      return {
        width: buffer.readUInt16LE(dataOffset + 6) & 0x3fff,
        height: buffer.readUInt16LE(dataOffset + 8) & 0x3fff
      };
    }
    if (chunkType === 'VP8L' && chunkLength >= 5 && buffer[dataOffset] === 0x2f) {
      const sizeBits = buffer.readUInt32LE(dataOffset + 1);
      return {
        width: (sizeBits & 0x3fff) + 1,
        height: ((sizeBits >>> 14) & 0x3fff) + 1
      };
    }

    offset = dataOffset + chunkLength + (chunkLength % 2);
  }
  throw new Error(`${relativePath} does not contain supported WebP dimensions`);
};

const createHomeClassList = (initial = []) => {
  const values = new Set(initial);
  return {
    add(...names) {
      names.forEach((name) => values.add(name));
    },
    contains(name) {
      return values.has(name);
    },
    remove(...names) {
      names.forEach((name) => values.delete(name));
    },
    toggle(name, force) {
      const enabled = typeof force === 'boolean' ? force : !values.has(name);
      if (enabled) values.add(name);
      else values.delete(name);
      return enabled;
    },
    values
  };
};

const createHomeElement = (dataset = {}) => {
  const attributes = new Map();
  const listeners = new Map();
  const element = {
    attributes,
    children: [],
    classList: createHomeClassList(),
    dataset: { ...dataset },
    hidden: false,
    scrollHeight: 100,
    scrollTop: 0,
    scrollWidth: 100,
    clientHeight: 200,
    clientWidth: 200,
    addEventListener(type, listener) {
      const entries = listeners.get(type) || [];
      entries.push(listener);
      listeners.set(type, entries);
    },
    append(...nodes) {
      this.children.push(...nodes);
    },
    appendChild(node) {
      this.children.push(node);
      return node;
    },
    cloneNode() {
      return createHomeElement({ ...this.dataset });
    },
    dispatchEvent(event) {
      this.events = this.events || [];
      this.events.push(event);
      (listeners.get(event.type) || []).forEach((listener) => listener(event));
      return true;
    },
    fire(type, event = {}) {
      const runtimeEvent = {
        altKey: false,
        button: 0,
        ctrlKey: false,
        defaultPrevented: false,
        metaKey: false,
        shiftKey: false,
        preventDefault() {
          this.defaultPrevented = true;
        },
        ...event,
        type
      };
      (listeners.get(type) || []).forEach((listener) => listener(runtimeEvent));
      return runtimeEvent;
    },
    focus(options) {
      this.focusedWith = options || null;
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    hasAttribute(name) {
      return attributes.has(name);
    },
    querySelector(selector) {
      return this.queryOne?.(selector) || null;
    },
    querySelectorAll(selector) {
      return this.queryMany?.(selector) || [];
    },
    removeAttribute(name) {
      attributes.delete(name);
    },
    replaceChildren(...nodes) {
      this.children = nodes;
    },
    scrollIntoView() {},
    setAttribute(name, value) {
      attributes.set(name, String(value));
    }
  };
  return element;
};

const runHomeTransitionRuntime = (source, options = {}) => {
  let now = 0;
  let nextTimerId = 1;
  const timers = [];
  const frames = [];
  const historyCalls = [];
  const nativeUpdates = [];
  const nativeFinishCallbacks = [];
  const reducedMotionListeners = [];
  const windowListeners = new Map();
  const root = createHomeElement({
    activePanel: 'tools',
    defaultPanel: 'tools',
    homeView: 'overview'
  });
  const item = createHomeElement({ homeAccordionItem: 'tools' });
  item.classList.add('is-active');
  const trigger = createHomeElement({ homeAccordionTrigger: 'tools' });
  trigger.setAttribute('aria-expanded', 'true');
  const panel = createHomeElement({ homeAccordionPanel: 'tools' });
  const scroller = createHomeElement();
  const libraryView = createHomeElement({
    homeLibraryRendered: 'true',
    homeLibraryView: 'tools'
  });
  libraryView.hidden = true;
  const libraryHeading = createHomeElement();
  const libraryOpen = createHomeElement({ homeLibraryOpen: 'tools' });
  const libraryClose = createHomeElement({ homeLibraryClose: 'tools' });
  const canonical = createHomeElement();
  canonical.setAttribute('href', 'https://www.danielshort.me/');
  item.queryOne = (selector) => {
    if (selector === '[data-home-accordion-trigger]') return trigger;
    if (selector === '[data-home-accordion-panel]') return panel;
    return null;
  };
  panel.queryOne = (selector) => {
    if (selector === '[data-home-accordion-scroller]') return scroller;
    return null;
  };
  libraryView.queryOne = (selector) => {
    if (selector === '[data-home-library-heading]') return libraryHeading;
    return null;
  };
  root.queryMany = (selector) => {
    if (selector === '[data-home-accordion-item]') return [item];
    if (selector === '[data-home-library-view]') return [libraryView];
    if (selector === '[data-home-library-open]') return [libraryOpen];
    if (selector === '[data-home-library-close]') return [libraryClose];
    return [];
  };
  root.queryOne = (selector) => {
    if (selector === '[data-home-library-open="tools"]') return libraryOpen;
    if (selector === '[data-home-accordion-trigger="tools"]') return trigger;
    return null;
  };

  function setTimeout(callback, delay = 0) {
    const id = nextTimerId;
    nextTimerId += 1;
    timers.push({ callback, cancelled: false, due: now + Math.max(0, Number(delay) || 0), id });
    return id;
  }

  function clearTimeout(timerId) {
    const timer = timers.find((entry) => entry.id === timerId);
    if (timer) timer.cancelled = true;
  }

  function advanceNextTimer() {
    const timer = timers
      .filter((entry) => !entry.cancelled)
      .sort((left, right) => left.due - right.due || left.id - right.id)[0];
    if (!timer) return false;
    timer.cancelled = true;
    now = timer.due;
    timer.callback();
    return true;
  }

  function requestAnimationFrame(callback) {
    const id = nextTimerId;
    nextTimerId += 1;
    frames.push({ callback, id });
    return id;
  }

  function flushFrames(limit = 40) {
    let remaining = limit;
    while (frames.length && remaining > 0) {
      const pending = frames.splice(0);
      pending.forEach(({ callback }) => callback(now));
      remaining -= 1;
    }
  }

  let currentUrl = new URL('https://www.danielshort.me/#tools');
  let runtimeWindow = null;
  const history = {
    state: null,
    pushState(state, title, url) {
      this.state = state;
      currentUrl = new URL(String(url), currentUrl.href);
      runtimeWindow.location = currentUrl;
      historyCalls.push({ method: 'pushState', state, url: currentUrl.href });
    },
    replaceState(state, title, url) {
      this.state = state;
      currentUrl = new URL(String(url), currentUrl.href);
      runtimeWindow.location = currentUrl;
      historyCalls.push({ method: 'replaceState', state, url: currentUrl.href });
    }
  };
  const reducedMotionQuery = {
    matches: Boolean(options.reducedMotion),
    addEventListener(type, listener) {
      if (type === 'change') reducedMotionListeners.push(listener);
    },
    addListener(listener) {
      reducedMotionListeners.push(listener);
    }
  };
  const railLayoutQuery = {
    matches: options.railLayout !== false,
    addEventListener() {},
    addListener() {}
  };
  const document = {
    title: 'Daniel Short',
    createDocumentFragment() {
      return createHomeElement();
    },
    createElement() {
      return createHomeElement();
    },
    querySelector(selector) {
      if (selector === '[data-home-accordion]') return root;
      if (selector === 'link[rel="canonical"]') return canonical;
      return null;
    }
  };
  if (options.nativeTransitions) {
    document.startViewTransition = (update) => {
      nativeUpdates.push(update);
      const finished = {
        catch() {
          return this;
        },
        finally(callback) {
          nativeFinishCallbacks.push(callback);
          return this;
        }
      };
      return { finished };
    };
  }
  runtimeWindow = {
    HOME_LIBRARY_DATA: { tools: { items: [] } },
    addEventListener(type, listener) {
      const entries = windowListeners.get(type) || [];
      entries.push(listener);
      windowListeners.set(type, entries);
    },
    cancelAnimationFrame() {},
    clearTimeout,
    history,
    location: currentUrl,
    matchMedia(query) {
      return query.includes('prefers-reduced-motion') ? reducedMotionQuery : railLayoutQuery;
    },
    requestAnimationFrame,
    scrollTo() {},
    scrollY: 0,
    setTimeout
  };
  class RuntimeCustomEvent {
    constructor(type, init) {
      this.bubbles = Boolean(init?.bubbles);
      this.detail = init?.detail;
      this.type = type;
    }
  }
  vm.runInNewContext(source, {
    CustomEvent: RuntimeCustomEvent,
    Map,
    Math,
    Number,
    Object,
    Set,
    String,
    URL,
    document,
    window: runtimeWindow
  });
  flushFrames();
  historyCalls.splice(0);
  root.events = [];

  return {
    advanceNextTimer,
    finishNativeTransition() {
      nativeFinishCallbacks.splice(0).forEach((callback) => callback());
    },
    flushFrames,
    historyCalls,
    libraryClose,
    libraryHeading,
    libraryOpen,
    libraryView,
    nativeUpdates,
    navigateHistory(href, state) {
      currentUrl = new URL(href, currentUrl.href);
      runtimeWindow.location = currentUrl;
      history.state = state;
      (windowListeners.get('popstate') || []).forEach((listener) => listener({ state }));
    },
    pendingTimerCount() {
      return timers.filter((entry) => !entry.cancelled).length;
    },
    root,
    runNativeUpdate() {
      const update = nativeUpdates.shift();
      if (update) update();
    },
    setReducedMotion(matches) {
      reducedMotionQuery.matches = Boolean(matches);
      reducedMotionListeners.forEach((listener) => listener({ matches: reducedMotionQuery.matches }));
    }
  };
};

module.exports = function runHomeCategoryAccordionTests({ assert }) {
  const personal = readJson('content/audiences/personal.json');
  const section = personal.page.sections.find((entry) => entry.type === 'home-accordion');
  const categories = section?.props?.categories || [];
  const ids = categories.map((category) => category.id);
  const html = renderVisualPageBody(personal.page);
  const indexHtml = read('index.html');
  const css = read('css/components/home-category-accordion.css');
  const libraryCss = read('css/components/home-library.css');
  const timelineCss = read('css/components/home-timeline.css');
  const js = read('js/home/category-accordion.js');
  const homeEntry = read('build/entries/site-home.entry.js');
  const homeStyles = read('css/styles-home.css');
  const sharedStyles = read('css/styles.css');
  const generator = read('build/generate-cms-artifacts.js');
  const visualValidator = read('build/validate-home-library-visuals.js');
  const buildSite = read('build/build-site.js');
  const copyToPublic = read('build/copy-to-public.js');
  const packageJson = readJson('package.json');
  const navigation = read('js/navigation/navigation.js');
  const activityEvents = read('js/analytics/activity-events.js');
  const homeLibraryData = require('../../js/home/home-library-data.js');
  const publishedProjects = fs.readdirSync(path.join(ROOT, 'content', 'projects'))
    .filter((fileName) => fileName.endsWith('.json'))
    .map((fileName) => readJson(path.join('content', 'projects', fileName)))
    .filter((project) => project && project.id && project.published !== false);
  const toolsDirectoryContext = { window: {} };
  vm.runInNewContext(read('js/portfolio/tools-directory-data.js'), toolsDirectoryContext);
  const toolsDirectoryItems = toolsDirectoryContext.window.DIRECTORY_WORKBENCH?.items || [];
  const homeAccordionWidget = getWidgetDefinitions()
    .find((widget) => widget.type === 'home-accordion');
  const iconDefinitions = getHomeAccordionIconDefinitions();

  assert(section && section.variant === 'shallow-wedge',
    'personal homepage should use the selected shallow-wedge accordion variant');
  assert(JSON.stringify(ids) === JSON.stringify(['about', 'projects', 'tools', 'games', 'contact']),
    'homepage categories should stay in the approved About, Projects, Tools, Games, Contact order');
  assert(section.props.defaultPanel === 'about',
    'homepage should open the About panel by default');
  assert(homeAccordionWidget?.defaultProps?.defaultPanel === 'about',
    'new CMS homepage accordion widgets should also default to About');
  assert(count(html, /data-site-tab="(?:about|projects|tools|games|contact)"/g) === 5 &&
    count(html, /data-site-tab-active="true"/g) === 1 &&
    html.includes('data-site-tab-rail data-site-tab-rail-mode="overview"') &&
    /<main\b[^>]*\bdata-site-route-content\b/i.test(html) &&
    count(html, /data-site-route-toolbar/g) === 1,
  'generated homepage markup should expose five stable tab slots, default About, and one complete route scene');
  assert(count(indexHtml, /data-site-shell-header/g) === 1 &&
    count(indexHtml, /data-site-shell-footer/g) === 1 &&
    count(indexHtml, /data-site-route-toolbar/g) === 1 &&
    count(indexHtml, /data-site-route-progress/g) === 1 &&
    count(indexHtml, /data-site-route-announcer/g) === 1 &&
    count(indexHtml, /data-site-route-content/g) === 1 &&
    count(indexHtml, /id="site-route-manifest"/g) === 1,
  'built homepage should expose one persistent chrome, route surface, progress line, announcer, and manifest');
  assert(
    JSON.stringify(categories.map((category) => category.color)) === JSON.stringify([
      '#091f3b', '#155dfc', '#087f8c', '#c94b0a', '#334155'
    ]),
    'homepage rails should use the approved site-native category colors'
  );
  const categoryIconIds = ['about', 'projects', 'tools', 'games', 'contact'];
  const uniqueCardIconIds = [
    'stormbreak', 'stellar-dogfight', 'probability', 'message', 'email', 'github'
  ];
  const specificIconIds = [...categoryIconIds, ...uniqueCardIconIds];
  assert(specificIconIds.every((id) => iconDefinitions[id]) &&
    new Set(specificIconIds.map((id) => iconDefinitions[id])).size === specificIconIds.length,
  'category and glyph-backed card icons should resolve to distinct on-brand SVG definitions');
  const authoredIconIds = categories.flatMap((category) => (category.items || [])
    .map((item) => item.icon)
    .filter(Boolean));
  assert(authoredIconIds.every((id) => iconDefinitions[id]) &&
    resolveHomeAccordionIconId('unknown-home-icon') === 'spark',
  'every authored homepage icon should resolve explicitly and unknown keys should use a neutral fallback');
  const expectedItemIcons = {
    stormbreak: 'stormbreak',
    'stellar-dogfight': 'stellar-dogfight',
    'probability-engine': 'probability',
    'contact-form': 'message',
    email: 'email',
    github: 'github'
  };
  const actualItemIcons = Object.fromEntries(categories
    .flatMap((category) => category.items || [])
    .filter((item) => expectedItemIcons[item.id])
    .map((item) => [item.id, item.icon]));
  assert(JSON.stringify(actualItemIcons) === JSON.stringify(expectedItemIcons),
    'icon-backed homepage cards should use specific semantic icon assignments');
  const expectedItemIds = {
    about: [],
    projects: ['babynames', 'handwritingRating', 'sheetMusicUpscale', 'ufoDashboard'],
    tools: ['text-compare', 'image-optimizer', 'qr-code-generator', 'screen-recorder'],
    games: ['stormbreak', 'stellar-dogfight', 'probability-engine'],
    contact: ['contact-form', 'email', 'github']
  };
  categories.forEach((category) => {
    assert(JSON.stringify((category.items || []).map((item) => item.id)) === JSON.stringify(expectedItemIds[category.id]),
      `${category.id} should remain a concise, curated homepage preview`);
  });
  const about = categories.find((category) => category.id === 'about');
  const games = categories.find((category) => category.id === 'games');
  const contact = categories.find((category) => category.id === 'contact');
  assert(!about.meta && !about.cta && /Based in Grand Junction/.test(about.context || '') &&
    about.profile?.image === 'img/hero/head-avatar-384.jpg' &&
    about.profile?.imageAlt === 'Daniel Short' &&
    about.profile?.imageWidth === 384 &&
    about.profile?.imageHeight === 384,
  'About should use personal prose and the deliberately sized personal portrait instead of cards or a directory CTA');
  const timelineItems = about.timeline?.items || [];
  const expectedTimelineIds = [
    'target',
    'google-data-analytics',
    'ibm-data-analyst',
    'ibm-machine-learning',
    'purdue-bs-data-analytics',
    'google-advanced-data-analytics',
    'randall-reilly',
    'visit-grand-junction',
    'google-analytics',
    'eastern-ms-data-science'
  ];
  assert(!about.timeline?.title && !about.timeline?.lead &&
    JSON.stringify(timelineItems.map((item) => item.id)) === JSON.stringify(expectedTimelineIds),
  'About should carry the approved 10-event personal timeline in chronological narrative order');
  const purdueTimelineItem = timelineItems.find((item) => item.id === 'purdue-bs-data-analytics');
  assert(purdueTimelineItem?.image === 'img/cert_logos/purdue_global.png' &&
    purdueTimelineItem?.imageWidth === 137 &&
    purdueTimelineItem?.imageHeight === 136 &&
    purdueTimelineItem?.imageTone === 'dark',
  'Purdue should use its full-resolution logo and an authored high-contrast plaque treatment');
  assert(count(html, /class="home-timeline__axis"/g) === timelineItems.length &&
    count(html, /class="home-timeline__dot"/g) === timelineItems.length &&
    html.includes('<ol class="home-timeline__list" data-home-timeline-scroller>'),
  'each timeline event should render one shared axis and explicit dot aligned with the dedicated timeline scroll region');
  const expectedCertificateDates = {
    'google-data-analytics': '2023-01-03',
    'ibm-data-analyst': '2023-01-11',
    'ibm-machine-learning': '2023-02-12',
    'google-advanced-data-analytics': '2023-05-20',
    'google-analytics': '2024-04-18'
  };
  const certificateDates = Object.fromEntries(timelineItems
    .filter((item) => item.type === 'certification')
    .map((item) => [item.id, item.date]));
  assert(JSON.stringify(certificateDates) === JSON.stringify(expectedCertificateDates),
    'About certifications should retain their exact issue dates');
  assert(timelineItems.every((item) => !Object.keys(item)
    .some((key) => /expir|validUntil/i.test(key))),
  'homepage timeline events should not invent certificate expiration or validity dates');
  assert(games.items.every((item) => item.icon && !item.image) &&
    !JSON.stringify(games).includes('project-starfall'),
  'Games should use consistent semantic glyph tiles and exclude inactive Project Starfall content');
  assert(!contact.meta && !contact.cta && contact.items.at(-1).presentation === 'tertiary',
    'Contact should avoid repeated guidance and keep GitHub as a quieter tertiary option');

  assert(count(html, /data-home-accordion-item=/g) === 5 &&
    count(html, /data-home-accordion-trigger=/g) === 5 &&
    count(html, /data-home-accordion-panel=/g) === 5,
  'generated homepage should render one static item, trigger, and attached panel per category');
  assert(count(html, /aria-expanded="true"/g) === 1 &&
    count(html, /aria-expanded="false"/g) >= 4 &&
    count(html, /aria-disabled="true"/g) === 1 &&
    count(html, /data-home-accordion-panel="[^"]+" hidden inert/g) === 4 &&
    !html.includes('aria-current="page"'),
  'homepage should author only About as expanded and noncollapsible, without treating same-page accordion buttons as page links');
  const getItemHtml = (id) => {
    const marker = html.indexOf(`data-home-accordion-item="${id}"`);
    const itemMarker = '<article class="home-accordion__item';
    const start = marker >= 0 ? html.lastIndexOf(itemMarker, marker) : -1;
    const next = start >= 0 ? html.indexOf(itemMarker, marker + 1) : -1;
    return start >= 0 ? html.slice(start, next >= 0 ? next : html.length) : '';
  };
  ids.forEach((id) => {
    const itemHtml = getItemHtml(id);
    const panelTag = itemHtml.match(new RegExp(`<section[^>]+data-home-accordion-panel="${id}"[^>]*>`))?.[0] || '';
    if (id === 'about') {
      assert(itemHtml.includes('home-accordion__item--about is-active') &&
        itemHtml.includes('aria-expanded="true"') &&
        itemHtml.includes('aria-disabled="true"') &&
        panelTag && !/\bhidden\b/.test(panelTag) && !/\binert\b/.test(panelTag),
      'About should be the specific expanded and interactive panel in authored homepage markup');
    } else {
      assert(!itemHtml.includes(`home-accordion__item--${id} is-active`) &&
        itemHtml.includes('aria-expanded="false"') &&
        /\bhidden\b/.test(panelTag) && /\binert\b/.test(panelTag),
      `${id} should be specifically authored as collapsed and non-interactive`);
    }
    assert(count(itemHtml, new RegExp(`data-home-icon="${id}"`, 'g')) === 2,
      `${id} rail and panel title should share one deliberate category icon identity`);
    assert(itemHtml.indexOf('home-accordion__rail-icon') >= 0 &&
      itemHtml.indexOf('home-accordion__rail-icon') < itemHtml.indexOf('home-accordion__rail-label'),
    `${id} rail should keep its icon before its label for the desktop column layout`);
    assert(itemHtml.includes(`aria-labelledby="home-accordion-trigger-${id}"`) &&
      new RegExp(`<h2 class="home-accordion__heading">[\\s\\S]*?<button[^>]+id="home-accordion-trigger-${id}"`).test(itemHtml),
    `${id} item should be labelled by a semantic accordion heading button`);
  });
  const aboutHtml = getItemHtml('about');
  assert(aboutHtml.includes('home-accordion__panel-head--profile') &&
    /<img src="img\/hero\/head-avatar-384\.jpg" alt="Daniel Short"[^>]+width="384" height="384">/.test(aboutHtml),
  'rendered About content should pair its heading with a meaningful, intrinsically sized profile image');
  assert(aboutHtml.includes('<section class="home-timeline" data-home-timeline aria-label="Timeline">') &&
    !aboutHtml.includes('home-timeline__head') &&
    !aboutHtml.includes('My path so far') &&
    aboutHtml.includes('<ol class="home-timeline__list" data-home-timeline-scroller>') &&
    count(aboutHtml, /<li class="home-timeline__item[^>]+data-home-timeline-item=/g) === 10 &&
    !aboutHtml.includes('role="list"') &&
    !aboutHtml.includes('role="listitem"'),
  'rendered About timeline should be a labelled section with a native ordered list of 10 semantic events');
  assert(!timelineCss.includes('.home-timeline__head') &&
    !aboutHtml.includes('A timeline of real milestones in learning and work.'),
  'timeline should remove its heading, subtext, and reserved heading styles at every viewport');
  assert(/data-home-timeline-item="purdue-bs-data-analytics"[\s\S]*?data-home-timeline-media-tone="dark"><img src="img\/cert_logos\/purdue_global\.png"[^>]+width="137" height="136"/.test(aboutHtml),
    'rendered Purdue milestone should retain the explicit dark plaque flag and undistorted intrinsic logo dimensions');
  Object.entries(expectedCertificateDates).forEach(([id, issueDate]) => {
    const marker = `data-home-timeline-item="${id}"`;
    const markerIndex = aboutHtml.indexOf(marker);
    const itemMarker = '<li class="home-timeline__item';
    const start = markerIndex >= 0 ? aboutHtml.lastIndexOf(itemMarker, markerIndex) : -1;
    const next = start >= 0 ? aboutHtml.indexOf(itemMarker, markerIndex + marker.length) : -1;
    const itemHtml = start >= 0 ? aboutHtml.slice(start, next >= 0 ? next : aboutHtml.length) : '';
    assert(itemHtml.includes('home-timeline__item--certification') &&
      itemHtml.includes(`<time datetime="${issueDate}">`) &&
      itemHtml.includes(`id="home-timeline-about-${id}-date"`) &&
      itemHtml.includes(`aria-describedby="home-timeline-about-${id}-date"`) &&
      /<a class="home-timeline__entry"[^>]+target="_blank" rel="noopener noreferrer"/.test(itemHtml),
    `${id} should render its exact issue date in a semantic time element and expose a safe external credential link`);
    const fullTitle = timelineItems.find((item) => item.id === id).title;
    const compactTitle = fullTitle.replace(/\s+(?:Professional\s+Certificate|Certification|Certificate)$/i, '');
    assert(itemHtml.includes(`<span class="home-timeline__title-full">${fullTitle}</span>`) &&
      itemHtml.includes(`<span class="home-timeline__title-compact" aria-hidden="true">${compactTitle}</span>`),
    `${id} should shorten only its visual mobile label while retaining the full accessible credential title`);
  });
  uniqueCardIconIds.forEach((id) => {
    assert(count(html, new RegExp(`data-home-icon="${id}"`, 'g')) === 1,
      `${id} should render exactly once as a unique homepage card glyph`);
  });
  assert(count(getItemHtml('contact'), /data-home-icon="external-arrow"/g) === 1,
    'the external GitHub card should use one dedicated external-link arrow');
  assert(count(html, /<h1\b/g) === 1 && html.includes('id="home-accordion-title"'),
    'homepage should expose one accessible H1');
  [
    ['projects', '/portfolio', 'View all projects'],
    ['tools', '/tools', 'View all tools'],
    ['games', '/games', 'View all games']
  ].forEach(([id, href, label]) => {
    const itemHtml = getItemHtml(id);
    const ctaMarker = `data-home-library-open="${id}"`;
    assert(itemHtml.includes(`href="${href}"`) &&
      itemHtml.includes(ctaMarker) &&
      itemHtml.includes(`aria-controls="home-library-view-${id}"`) &&
      itemHtml.indexOf(ctaMarker) > itemHtml.indexOf('<ul class="home-accordion__cards">') &&
      itemHtml.includes(`data-home-library-view="${id}"`) &&
      itemHtml.includes('hidden inert') &&
      itemHtml.includes(`data-home-library-close="${id}"`),
    `${id} View all link should keep its canonical fallback while exposing an authored in-page library and return control`);
  });
  assert(count(html, /data-home-library-view=/g) === 3 &&
    count(html, /data-home-library-open=/g) === 3 &&
    count(html, /data-home-library-close=/g) === 3 &&
    html.includes('data-home-view="overview"') &&
    !html.includes('Back to categories') &&
    html.includes('Back to homepage') &&
    html.includes('data-personal-tool-account="true"') &&
    !html.includes('>All tools<'),
  'homepage markup should author three progressively enhanced libraries while retaining the five-category overview');
  assert(html.includes('<ul class="home-accordion__cards">') &&
    html.includes('<li class="home-accordion__card-item">') &&
    /<a class="home-accordion__card" href="\/tools\/text-compare"/.test(html) &&
    !/<a class="home-accordion__card" role="listitem"/.test(html),
  'homepage cards should use semantic list wrappers without masking native link roles');

  ids.forEach((id) => {
    const pairPattern = new RegExp(
      `<h2[^>]+home-accordion__heading[^>]*>\\s*<button[^>]+id="home-accordion-trigger-${id}"[^>]+type="button"[^>]+aria-controls="home-accordion-panel-${id}"[\\s\\S]*?<\\/button>\\s*<\\/h2>\\s*<section[^>]+id="home-accordion-panel-${id}"[^>]+role="region"[^>]+aria-labelledby="home-accordion-trigger-${id}"`
    );
    assert(pairPattern.test(html), `${id} rail should be a heading-wrapped native button followed by its labeled region`);
  });

  const requiredRoutes = [
    '/portfolio/handwritingRating',
    '/portfolio/babynames',
    '/portfolio/sheetMusicUpscale',
    '/portfolio/ufoDashboard',
    '/tools/text-compare',
    '/tools/image-optimizer',
    '/tools/qr-code-generator',
    '/tools/screen-recorder',
    '/games/stormbreak',
    '/games/stellar-dogfight',
    '/games/probability-engine',
    '/contact#contact-modal',
    'mailto:daniel@danielshort.me'
  ];
  requiredRoutes.forEach((route) => {
    assert(html.includes(`href="${route}"`), `generated homepage missing approved route ${route}`);
  });
  assert(/href="\/contact#contact-modal"[^>]*data-contact-modal-link/.test(getItemHtml('contact')),
    'homepage Send a message card should opt into the shared in-page contact modal');
  ['/tools/word-frequency', '/games/roulette', '/games/ocean-wave-simulation'].forEach((route) => {
    assert(!html.includes(`href="${route}"`), `homepage preview should leave ${route} to its full directory`);
  });
  ['campaign-creative-tracker', 'short-links', 'ga4-utm-performance', 'job-application-tracker', 'transcribe'].forEach((toolId) => {
    assert(!JSON.stringify(categories).includes(`\"id\":\"${toolId}\"`),
      `homepage should not expose hidden tool ${toolId}`);
  });
  const expectedLibraryCounts = {
    projects: 16,
    tools: 10,
    games: 5
  };
  assert(JSON.stringify(Object.fromEntries(Object.entries(homeLibraryData)
    .map(([id, library]) => [id, library.items?.length || 0]))) === JSON.stringify(expectedLibraryCounts),
  'generated HOME_LIBRARY_DATA should expose all 16 projects, 10 public tools, and 5 games');
  const publishedProjectIds = new Set(publishedProjects.map((project) => String(project.id)));
  assert(publishedProjects.length === expectedLibraryCounts.projects &&
    publishedProjects.every((project) => homeLibraryData.projects.items
      .some((item) => item.id === String(project.id))) &&
    homeLibraryData.projects.items.every((item) => publishedProjectIds.has(item.id)),
  'homepage project library should remain a complete projection of published project content');

  const publishedProjectsById = new Map(publishedProjects.map((project) => [String(project.id), project]));
  const projectPreviewPaths = homeLibraryData.projects.items.map((item) => item.image);
  assert(homeLibraryData.projects.items.every((item) => {
    const project = publishedProjectsById.get(item.id);
    return project &&
      item.image === projectLibraryAsset(project.image) &&
      item.image === `/img/projects/${item.id}-640.webp` &&
      item.imageAlt === '';
  }),
  'all 16 project library cards should derive their original optimized preview from the canonical project image');
  const allManifestMotifs = Object.values(GENERATED_HOME_LIBRARY_VISUALS)
    .flatMap((visuals) => Object.values(visuals));
  assert(JSON.stringify(Object.keys(GENERATED_HOME_LIBRARY_VISUALS).sort()) === JSON.stringify(['games']) &&
    allManifestMotifs.length === expectedLibraryCounts.games &&
    new Set(allManifestMotifs).size === allManifestMotifs.length &&
    Object.entries(GENERATED_HOME_LIBRARY_VISUALS).every(([category, visuals]) =>
      JSON.stringify(Object.keys(visuals).sort()) ===
      JSON.stringify(homeLibraryData[category].items.map((item) => item.id).sort())),
  'every generated game preview should retain one unique semantic concept');
  const allTools = fs.readdirSync(path.join(ROOT, 'content', 'tools'))
    .filter((fileName) => fileName.endsWith('.json'))
    .map((fileName) => readJson(`content/tools/${fileName}`));
  const toolsById = new Map(allTools.map((tool) => [tool.slug, tool]));
  const toolIconPaths = homeLibraryData.tools.items.map((item) => item.image);
  assert(homeLibraryData.tools.items.every((item) =>
    item.image === `/${toolsById.get(item.id)?.iconImage}` && item.imageAlt === ''),
  'every public tool library card should reuse its original canonical PNG icon');
  assert(allTools.length === 15 && allTools.every((tool) => {
    if (tool.iconImage !== `img/tools/icons/${tool.slug}.png`) return false;
    const buffer = fs.readFileSync(path.join(ROOT, tool.iconImage));
    return buffer.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) &&
      buffer.readUInt32BE(16) > 0 && buffer.readUInt32BE(20) > 0;
  }),
  'all 15 catalog tools should retain valid original PNG assets without changing public visibility');
  const featuredTools = categories.find((category) => category.id === 'tools').items;
  assert(featuredTools.every((tool) => homeLibraryData.tools.items.some((item) =>
    item.id === tool.id && item.image === `/${tool.image}`)),
  'View all tools should use the same original icon set as the homepage featured tools');
  assert(!fs.existsSync(path.join(ROOT, 'build', 'generate-home-library-visuals.js')) &&
    !fs.existsSync(path.join(ROOT, 'img', 'home-previews', 'sources')),
  'generated preview WebPs should be authoritative static assets without an old screenshot or code-art regeneration path');

  const generatedPreviewPaths = [];
  const previewRoot = path.join(ROOT, 'img', 'home-previews');
  const actualPreviewCategories = fs.readdirSync(previewRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  assert(JSON.stringify(actualPreviewCategories) ===
    JSON.stringify(Object.keys(expectedLibraryCounts).sort()),
  'homepage preview root should contain only the exact lowercase projects, tools, and games directories');
  const retainedProjectPreviewNames = fs.readdirSync(path.join(previewRoot, 'projects')).sort();
  assert(JSON.stringify(retainedProjectPreviewNames) === JSON.stringify(
    RETAINED_PROJECT_PREVIEW_IDS.map((id) => `${id}.webp`).sort()) &&
    homeLibraryData.projects.items.every((item) =>
      !item.image.startsWith('/img/home-previews/projects/')),
  'legacy AI project previews should remain available but unused by the project library');
  assert(JSON.stringify(fs.readdirSync(path.join(previewRoot, 'tools')).sort()) === JSON.stringify(
    RETAINED_TOOL_PREVIEW_IDS.map((id) => `${id}.webp`).sort()) &&
    homeLibraryData.tools.items.every((item) => !item.image.startsWith('/img/home-previews/tools/')),
  'legacy generated tool previews should remain available but unused by the tools library');
  Object.entries(GENERATED_HOME_LIBRARY_VISUALS).forEach(([category, visuals]) => {
    const expectedCount = Object.keys(visuals).length;
    const items = homeLibraryData[category]?.items || [];
    const retainedFileNames = category === 'games'
      ? RETAINED_GAME_PREVIEW_IDS.map((id) => `${id}.webp`)
      : [];
    const expectedFileNames = [
      ...items.map((item) => `${item.id}.webp`),
      ...retainedFileNames
    ].sort();
    const categoryDir = path.join(previewRoot, category);
    const actualEntries = fs.readdirSync(categoryDir, { withFileTypes: true });
    const actualFileNames = actualEntries
      .map((entry) => entry.name)
      .sort();
    items.forEach((item) => generatedPreviewPaths.push(item.image));
    assert(items.length === expectedCount &&
      actualEntries.length === expectedCount + retainedFileNames.length &&
      actualEntries.every((entry) => entry.isFile() && entry.name.endsWith('.webp')) &&
      items.every((item) => item.image === `/img/home-previews/${category}/${item.id}.webp` &&
        item.imageAlt === '') &&
      JSON.stringify(actualFileNames) === JSON.stringify(expectedFileNames),
    `${category} should map every item to one exact-case, decorative generated WebP asset`);
  });
  assert(projectPreviewPaths.length === expectedLibraryCounts.projects &&
    toolIconPaths.length === expectedLibraryCounts.tools &&
    generatedPreviewPaths.length === expectedLibraryCounts.games &&
    new Set([...projectPreviewPaths, ...toolIconPaths, ...generatedPreviewPaths]).size === 31,
  'all 31 public library preview paths should remain unique');
  assert(projectPreviewPaths.every((previewPath) => {
    const dimensions = readWebpDimensions(previewPath);
    return dimensions.width === 640 && dimensions.height > 0;
  }),
  'every original project library preview should be a valid 640px-wide WebP');
  assert(generatedPreviewPaths.every((previewPath) => {
    const dimensions = readWebpDimensions(previewPath);
    return dimensions.width === 640 && dimensions.height === 360;
  }),
  'every generated game preview should carry exact 640 by 360 WebP metadata');
  const previewHashes = generatedPreviewPaths.map((previewPath) => crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(ROOT, previewPath.replace(/^\/+/, ''))))
    .digest('hex'));
  assert(new Set(previewHashes).size === generatedPreviewPaths.length,
  'all five public game preview files should have unique visual content');

  const cmsPreviewMappings = [
    'image: projectLibraryPreviewAsset(project.image)',
    'image: tool.iconImage',
    "image: homeLibraryPreviewAsset('games', game.id)"
  ];
  const visualBuildIndex = buildSite.indexOf(
    "runNodeScript(path.join('build', 'validate-home-library-visuals.js')"
  );
  const publicCopyIndex = buildSite.indexOf("runNodeScript(path.join('build', 'copy-to-public.js')");
  const publicVisualBuildIndex = buildSite.indexOf("{ verbose, args: ['--public'] }");
  assert(generator.includes('function homeLibraryPreviewAsset(category, id)') &&
    generator.includes('function projectLibraryPreviewAsset(image)') &&
    cmsPreviewMappings.every((mapping) => generator.includes(mapping)) &&
    count(generator, /imageAlt: '',/g) >= 3 &&
    visualValidator.includes("const sharp = require('sharp');") &&
    visualValidator.includes('const GENERATED_HOME_LIBRARY_VISUALS = {') &&
    visualValidator.includes("const RETAINED_GAME_PREVIEW_IDS = ['project-starfall'];") &&
    visualValidator.includes('const RETAINED_PROJECT_PREVIEW_IDS = [') &&
    visualValidator.includes('const RETAINED_TOOL_PREVIEW_IDS = [') &&
    visualValidator.includes('function projectLibraryAsset(image)') &&
    visualValidator.includes('function validateCatalogMappings()') &&
    visualValidator.includes('async function validateProjectAssets(baseDir, projects)') &&
    visualValidator.includes('async function validateToolIconAssets(baseDir)') &&
    visualValidator.includes('validateMatchingHashes(toolSourceHashes, toolDeployedHashes)') &&
    visualValidator.includes('function listPreviewTree(baseDir)') &&
    visualValidator.includes("validatePreviewTree(publicPreviewRoot, 'public/img/home-previews')") &&
    visualValidator.includes('validateMatchingHashes(sourceHashes, deployedHashes)') &&
    visualValidator.includes("metadata.width !== 640 || metadata.height !== 360") &&
    visualValidator.includes("new Set(hashes.values()).size !== hashes.size") &&
    visualValidator.includes('Validated ${projectSourceHashes.size} original project previews') &&
    packageJson.scripts?.['validate:home-library-visuals'] === 'node build/validate-home-library-visuals.js' &&
    Boolean(packageJson.devDependencies?.sharp) &&
    buildSite.includes('const scriptArgs = Array.isArray(options.args) ? options.args : [];') &&
    visualBuildIndex >= 0 && publicCopyIndex > visualBuildIndex && publicVisualBuildIndex > publicCopyIndex &&
    /const dirs = \[[^\]]*'img'/.test(copyToPublic),
  'CMS generation, project-original validation, generated-preview validation, main build, and hash-identical public copy should stay integrated');

  assert(js.includes('HOME_LIBRARY_DATA') &&
    js.includes('createLibraryMedia') &&
    js.includes('createLibraryCard') &&
    js.includes('renderLibrary') &&
    js.includes("link.dataset.personalTransition = 'detail'"),
  'homepage controller should render the complete library data in place and mark item links for continuous detail navigation');
  const libraryToolIds = new Set(homeLibraryData.tools.items.map((tool) => tool.id));
  const excludedTools = toolsDirectoryItems.filter((tool) =>
    String(tool.visibility || 'public').toLowerCase() !== 'public' || tool.hidden || tool.noindex);
  const publicTools = toolsDirectoryItems.filter((tool) =>
    String(tool.visibility || 'public').toLowerCase() === 'public' && !tool.hidden && !tool.noindex);
  assert(excludedTools.length === 5 &&
    excludedTools.every((tool) => !libraryToolIds.has(tool.id)) &&
    publicTools.every((tool) => libraryToolIds.has(tool.id)) &&
    homeLibraryData.tools.items.every((tool) => publicTools.some((source) => source.id === tool.id)),
  'homepage tool library should include every public tool while excluding every hidden, admin, authenticated, or noindex tool');
  assert(generator.includes("String(tool.visibility || 'public').trim().toLowerCase() === 'public'") &&
    generator.includes('!tool.hidden && !tool.noindex'),
  'the HOME_LIBRARY_DATA generator should keep its explicit public, visible, and indexable tool filter');
  assert(!/recruiter|available for hire|resume|case study|kpi|professional analytics profile/i.test(JSON.stringify(categories)),
    'personal homepage copy should avoid job-seeking and dashboard framing');

  assert(!html.includes('data-home-graph') &&
    !html.includes('data-graph-') &&
    !html.includes('home-graph__inspector') &&
    !html.includes('home-graph__mobile-deck'),
  'generated homepage should not retain the detached graph, inspector, or duplicate mobile deck');
  assert(!fs.existsSync(path.join(ROOT, 'js/home/project-graph.js')) &&
    !fs.existsSync(path.join(ROOT, 'css/components/home-project-graph.css')),
  'retired graph source files should be removed');

  const libraryDataImportIndex = homeEntry.indexOf("import '../../js/home/home-library-data.js';");
  const accordionImportIndex = homeEntry.indexOf("import '../../js/home/category-accordion.js';");
  const toolsLoaderImportIndex = homeEntry.indexOf("import '../../js/accounts/tools-page-loader.js';");
  assert(libraryDataImportIndex >= 0 && accordionImportIndex > libraryDataImportIndex &&
    toolsLoaderImportIndex > accordionImportIndex &&
    homeStyles.includes('@import url("components/home-category-accordion.css");') &&
    homeStyles.includes('@import url("components/home-library.css");') &&
    sharedStyles.includes('@import url("components/home-timeline.css");'),
  'homepage bundles should load library data before the controller, attach the lazy tools account loader, and include accordion, library, and shared timeline styles');
  assert(personal.page.bottomScripts.some((script) => script.src === 'dist/site-home.js') &&
    !JSON.stringify(personal.page.bottomScripts).includes('project-graph'),
  'managed homepage source should use the stable home bundle without raw graph scripts');

  const fallbackViewRuntime = runHomeTransitionRuntime(js);
  const firstOpen = fallbackViewRuntime.libraryOpen.fire('click');
  assert(firstOpen.defaultPrevented &&
    fallbackViewRuntime.root.dataset.homeView === 'overview' &&
    fallbackViewRuntime.libraryView.hidden &&
    fallbackViewRuntime.root.classList.contains('is-view-leaving') &&
    !fallbackViewRuntime.root.classList.contains('is-view-entering') &&
    fallbackViewRuntime.historyCalls.length === 0 &&
    fallbackViewRuntime.root.events.length === 0,
  'fallback homepage expansion should preserve the old overview throughout its leaving phase');
  const pendingTimersAfterOpen = fallbackViewRuntime.pendingTimerCount();
  fallbackViewRuntime.libraryOpen.fire('click');
  assert(fallbackViewRuntime.pendingTimerCount() === pendingTimersAfterOpen &&
    fallbackViewRuntime.historyCalls.length === 0 &&
    fallbackViewRuntime.root.events.length === 0,
  'rapid homepage expansion clicks should not queue duplicate swaps, history entries, or events');
  assert(fallbackViewRuntime.advanceNextTimer() &&
    fallbackViewRuntime.root.dataset.homeView === 'library' &&
    !fallbackViewRuntime.libraryView.hidden &&
    !fallbackViewRuntime.root.classList.contains('is-view-leaving') &&
    fallbackViewRuntime.root.classList.contains('is-view-entering') &&
    fallbackViewRuntime.historyCalls.filter((entry) => entry.method === 'pushState').length === 1 &&
    fallbackViewRuntime.root.events.filter((event) => (
      event.type === 'home:library-change' && event.detail?.expanded === true
    )).length === 1,
  'fallback homepage expansion should swap state, URL, and events together under cover before entering');
  fallbackViewRuntime.flushFrames();
  assert(fallbackViewRuntime.advanceNextTimer() &&
    !fallbackViewRuntime.root.classList.contains('is-view-changing') &&
    !fallbackViewRuntime.root.classList.contains('is-view-entering'),
  'fallback homepage expansion should settle and release its transition lock after the entry phase');

  fallbackViewRuntime.libraryClose.fire('click');
  assert(fallbackViewRuntime.root.dataset.homeView === 'library' &&
    !fallbackViewRuntime.libraryView.hidden &&
    fallbackViewRuntime.root.classList.contains('is-view-leaving') &&
    fallbackViewRuntime.historyCalls.length === 1,
  'fallback homepage collapse should preserve the old library throughout its leaving phase');
  assert(fallbackViewRuntime.advanceNextTimer() &&
    fallbackViewRuntime.root.dataset.homeView === 'overview' &&
    fallbackViewRuntime.libraryView.hidden &&
    fallbackViewRuntime.root.classList.contains('is-view-entering') &&
    fallbackViewRuntime.historyCalls.filter((entry) => entry.method === 'pushState').length === 2 &&
    fallbackViewRuntime.root.events.filter((event) => (
      event.type === 'home:library-change' && event.detail?.expanded === false
    )).length === 1,
  'fallback homepage collapse should swap state, URL, and events together under cover before entering');
  fallbackViewRuntime.flushFrames();
  fallbackViewRuntime.advanceNextTimer();

  const historyDuringEntry = runHomeTransitionRuntime(js);
  historyDuringEntry.libraryOpen.fire('click');
  historyDuringEntry.advanceNextTimer();
  historyDuringEntry.navigateHistory('https://www.danielshort.me/#tools', {
    homePanel: 'tools',
    homeView: 'overview',
    personalCategory: 'tools',
    personalView: 'overview'
  });
  assert(historyDuringEntry.root.dataset.homeView === 'library' &&
    historyDuringEntry.root.classList.contains('is-view-entering'),
  'history traversal during entry should wait for the active visual handoff instead of desynchronizing the panel');
  historyDuringEntry.advanceNextTimer();
  historyDuringEntry.flushFrames();
  assert(historyDuringEntry.root.dataset.homeView === 'library' &&
    historyDuringEntry.root.classList.contains('is-view-leaving'),
  'queued history state should begin reconciliation as soon as the active handoff settles');
  historyDuringEntry.advanceNextTimer();
  assert(historyDuringEntry.root.dataset.homeView === 'overview' &&
    historyDuringEntry.libraryView.hidden,
  'queued history reconciliation should ultimately align the homepage view with the browser URL');

  const nativeViewRuntime = runHomeTransitionRuntime(js, { nativeTransitions: true });
  nativeViewRuntime.libraryOpen.fire('click');
  assert(nativeViewRuntime.nativeUpdates.length === 0 &&
    nativeViewRuntime.root.dataset.homeView === 'overview' &&
    nativeViewRuntime.historyCalls.length === 0 &&
    nativeViewRuntime.root.classList.contains('is-view-changing') &&
    nativeViewRuntime.root.classList.contains('is-view-leaving'),
  'homepage transitions should avoid browser snapshot transitions and animate only live content');
  nativeViewRuntime.advanceNextTimer();
  assert(nativeViewRuntime.root.dataset.homeView === 'library' &&
    !nativeViewRuntime.libraryView.hidden &&
    nativeViewRuntime.historyCalls.filter((entry) => entry.method === 'pushState').length === 1,
  'the content-only transition should update visual state and history after its short exit phase');
  nativeViewRuntime.advanceNextTimer();
  assert(!nativeViewRuntime.root.classList.contains('is-view-changing'),
    'the content-only transition should release its interaction lock when finished');

  const reducedViewRuntime = runHomeTransitionRuntime(js, { reducedMotion: true });
  reducedViewRuntime.libraryOpen.fire('click');
  assert(reducedViewRuntime.root.dataset.homeView === 'library' &&
    !reducedViewRuntime.libraryView.hidden &&
    reducedViewRuntime.historyCalls.filter((entry) => entry.method === 'pushState').length === 1 &&
    reducedViewRuntime.pendingTimerCount() === 0 &&
    !reducedViewRuntime.root.classList.contains('is-view-changing') &&
    !reducedViewRuntime.root.classList.contains('is-view-leaving') &&
    !reducedViewRuntime.root.classList.contains('is-view-entering'),
  'reduced-motion homepage expansion should update synchronously without transition phases');

  const reduceDuringExitRuntime = runHomeTransitionRuntime(js);
  reduceDuringExitRuntime.libraryOpen.fire('click');
  reduceDuringExitRuntime.setReducedMotion(true);
  assert(reduceDuringExitRuntime.root.dataset.homeView === 'library' &&
    !reduceDuringExitRuntime.libraryView.hidden &&
    reduceDuringExitRuntime.historyCalls.filter((entry) => entry.method === 'pushState').length === 1 &&
    reduceDuringExitRuntime.pendingTimerCount() === 0 &&
    !reduceDuringExitRuntime.root.classList.contains('is-view-changing'),
  'enabling reduced motion during the leaving phase should complete the requested state change without animation');

  const overviewPanelCss = extractBlock(css, '.home-accordion__panel {');
  const overviewScrollerCss = extractBlock(css, '.home-accordion__scroller {');
  const overviewItemCss = extractBlock(css, '.home-accordion__item {');
  const overviewShellCss = extractBlock(css, '.home-accordion__shell {');
  const overviewRailCss = extractBlock(css, '.home-accordion__rail {');
  const profileImageCss = extractBlock(css, '.home-accordion__profile-portrait img {');
  const mobileAccordionCss = extractBlock(css, '@media (max-width: 959px), (max-height: 619px)');
  const timelineRootCss = extractBlock(timelineCss, '.home-timeline {');
  const timelineAxisCss = extractBlock(timelineCss, '.home-timeline__axis {');
  const timelineEntryCss = extractBlock(timelineCss, '.home-timeline__entry {');
  const timelineMediaCss = extractBlock(timelineCss, '.home-timeline__media {');
  const desktopTimelineCss = extractBlock(timelineCss, '@media (min-width: 960px) and (min-height: 620px)');
  const desktopTimelineListCss = extractBlock(
    desktopTimelineCss,
    '.home-accordion__item--about .home-timeline__list {'
  );
  const desktopTimelineOverlapCss = extractBlock(
    desktopTimelineCss,
    '.home-accordion__item--about .home-timeline__item + .home-timeline__item {'
  );
  const mobileTimelineCss = extractBlock(timelineCss, '@media (max-width: 959px), (max-height: 619px)');
  const mobileTimelineRootCss = extractBlock(mobileTimelineCss, '.home-timeline {');
  const mobileTimelineListCss = extractBlock(mobileTimelineCss, '.home-timeline__list {');
  const mobileTimelineItemCss = extractBlock(mobileTimelineCss, '.home-timeline__item {');
  const mobileTimelineEntryCss = extractBlock(mobileTimelineCss, '.home-timeline__entry,');
  const mobileTimelineAxisCss = extractBlock(mobileTimelineCss, '.home-timeline__axis {');
  const mobileTimelineDotCss = extractBlock(mobileTimelineCss, '.home-timeline__dot {');
  const mobileTimelineMediaCss = extractBlock(mobileTimelineCss, '.home-timeline__media {');
  const mobileTimelineFullTitleCss = extractBlock(mobileTimelineCss, '.home-timeline__title-full {');
  const evenTimelineAxisCss = extractBlock(timelineCss, '.home-timeline__item:nth-child(even) .home-timeline__axis::after');
  assert(css.includes('--home-rail-width: 64px;') &&
    css.includes('--home-active-rail-width: 68px;') &&
    css.includes('--home-collapsed-rails-width: 256px;') &&
    css.includes('--home-panel-motion: var(--motion-slow);') &&
    css.includes('flex-basis: calc(100% - var(--home-collapsed-rails-width));') &&
    css.includes('@keyframes homeAccordionContentIn') &&
    css.includes('gap: 0;') &&
    overviewRailCss.includes('background: var(--panel-color);') &&
    !overviewRailCss.includes('linear-gradient') &&
    overviewPanelCss.includes('border: 4px solid var(--panel-color);') &&
    overviewPanelCss.includes('border-left: 0;') &&
    css.includes('right: -11px;') &&
    css.includes('width: 12px;') &&
    css.includes('height: 22px;') &&
    css.includes('clip-path: polygon(0 0, 100% 50%, 0 100%);'),
  'desktop overview should use compact solid 64px rails, a 68px active rail, a 4px frame, and a smaller right-facing notch');
  assert(overviewShellCss.includes('container: home-accordion-shell / inline-size;') &&
    overviewShellCss.includes('--home-overview-panel-inline-size: calc(100cqi - var(--home-collapsed-rails-width) - var(--home-active-rail-width));') &&
    overviewItemCss.includes('overflow: hidden;') &&
    overviewPanelCss.includes('flex: 0 0 var(--home-overview-panel-inline-size);') &&
    overviewPanelCss.includes('width: var(--home-overview-panel-inline-size);') &&
    overviewPanelCss.includes('min-width: var(--home-overview-panel-inline-size);') &&
    !css.includes('.home-accordion__item.is-closing .home-accordion__scroller') &&
    mobileAccordionCss.includes('overflow: visible;') &&
    mobileAccordionCss.includes('min-width: 0;'),
  'desktop overview panels should reveal a settled container-sized surface without content reflow while mobile panels remain in normal flow');
  assert(overviewPanelCss.includes('background: #ffffff;') &&
    overviewPanelCss.includes('border-radius: 0;') &&
    overviewScrollerCss.includes('overflow-y: auto;') &&
    overviewScrollerCss.includes('overscroll-behavior: contain;') &&
    css.includes('position: sticky;') &&
    overviewScrollerCss.includes('scrollbar-color: var(--home-scrollbar-thumb) var(--home-scrollbar-track);') &&
    css.includes('.home-accordion__item:last-child.is-active .home-accordion__panel') &&
    css.includes('border-radius: 0 12px 12px 0;'),
  'selected overview panels should stay white and square, use conditional themed scrolling, and round both outer-right corners only for Contact');
  assert(css.includes('.home-accordion__panel-head {\n    box-sizing: border-box;') &&
    css.includes('.home-accordion__context {\n    box-sizing: border-box;') &&
    css.includes('.home-accordion__meta {\n    box-sizing: border-box;') &&
    css.includes('.home-accordion__cards {\n    box-sizing: border-box;') &&
    css.includes('max-inline-size: 100%;') &&
    profileImageCss.includes('object-fit: contain;') &&
    profileImageCss.includes('object-position: center;'),
  'all padded panel content should remain inside the scroller width and keep the profile portrait fully visible');
  assert(css.includes('.home-accordion__card-glyph {\n    box-sizing: border-box;'),
    'homepage card glyphs should keep their padding inside the media tile instead of clipping');
  assert(mobileAccordionCss.includes('display: block;') &&
    mobileAccordionCss.includes('max-height: none;') &&
    mobileAccordionCss.includes('overflow: visible;') &&
    mobileAccordionCss.includes('height: 48px;') &&
    mobileAccordionCss.includes('min-height: 48px;') &&
    mobileAccordionCss.includes('height: 54px;') &&
    mobileAccordionCss.includes('transition: height var(--motion-slow) var(--easing-standard)') &&
    !mobileAccordionCss.includes('min-height: 54px;') &&
    mobileAccordionCss.includes('.home-accordion:not(.is-library-mode) .home-accordion__scroller') &&
    mobileAccordionCss.includes('border: 4px solid var(--panel-color);') &&
    mobileAccordionCss.includes('width: 20px;') &&
    mobileAccordionCss.includes('height: 10px;') &&
    mobileAccordionCss.includes('clip-path: polygon(0 0, 100% 0, 50% 100%);'),
  'narrow or short accordion layouts should stack compact 48px and 54px bars, use document scrolling, and retain the 4px framed panel');
  assert(mobileTimelineCss.includes('grid-auto-flow: column;') &&
    mobileTimelineCss.includes('grid-auto-columns: min(220px, 100%);') &&
    mobileTimelineCss.includes('overflow-x: auto;') &&
    mobileTimelineCss.includes('overflow-y: hidden;') &&
    mobileTimelineCss.includes('scroll-snap-type: inline proximity;') &&
    mobileTimelineCss.includes('overscroll-behavior-inline: contain;') &&
    mobileTimelineCss.includes('scroll-snap-align: start;') &&
    !mobileTimelineCss.includes('scroll-snap-stop: always;') &&
    !mobileTimelineCss.includes('touch-action: pan-x') &&
    mobileTimelineCss.includes('scrollbar-width: none;') &&
    mobileTimelineCss.includes('display: none;') &&
    mobileTimelineCss.includes('.home-timeline__dot') &&
    mobileTimelineCss.includes('width: calc(100% + var(--home-timeline-column-gap));') &&
    mobileTimelineCss.includes('height: 2px;') &&
    mobileTimelineDotCss.includes('width: 8px;') &&
    mobileTimelineDotCss.includes('height: 8px;') &&
    mobileTimelineDotCss.includes('top: 50%;'),
  'mobile timeline should offer native proximity snapping without blocking vertical gestures and connect compact nodes with a continuous 2px spine');
  assert(timelineRootCss.includes('--home-timeline-gap: 10px;') &&
    timelineRootCss.includes('padding: 14px var(--home-timeline-gutter) 24px;') &&
    timelineAxisCss.includes('min-height: 76px;') &&
    timelineEntryCss.includes('min-height: 76px;') &&
    timelineEntryCss.includes('padding: 10px;') &&
    timelineMediaCss.includes('width: 40px;') &&
    timelineMediaCss.includes('height: 40px;') &&
    desktopTimelineListCss.includes('--home-timeline-stagger-overlap: 44px;') &&
    desktopTimelineListCss.includes('padding-bottom: 24px;') &&
    desktopTimelineOverlapCss.includes('margin-top: calc(0px - var(--home-timeline-stagger-overlap));') &&
    !mobileTimelineCss.includes('--home-timeline-stagger-overlap') &&
    !/\n\s*(?:height|max-height)\s*:/.test(timelineEntryCss),
  'desktop timeline should retain its compact alternating cards and interleaved milestones');
  assert(mobileTimelineRootCss.includes('padding: 10px 0 20px;') &&
    mobileTimelineListCss.includes('gap: 6px var(--home-timeline-column-gap);') &&
    mobileTimelineListCss.includes('padding: 4px 14px 10px;') &&
    mobileTimelineListCss.includes('grid-template-rows: auto 12px auto;') &&
    mobileTimelineItemCss.includes('grid-column: auto;') &&
    mobileTimelineItemCss.includes('grid-row: 1 / span 3;') &&
    mobileTimelineItemCss.includes('grid-template-rows: subgrid;') &&
    mobileTimelineAxisCss.includes('position: relative;') &&
    mobileTimelineAxisCss.includes('grid-column: 1;') &&
    mobileTimelineAxisCss.includes('grid-row: 2;') &&
    mobileTimelineEntryCss.includes('grid-row: 3;') &&
    mobileTimelineEntryCss.includes('min-height: 44px;') &&
    mobileTimelineEntryCss.includes('padding: 10px;') &&
    mobileTimelineMediaCss.includes('width: 28px;') &&
    mobileTimelineMediaCss.includes('height: 28px;') &&
    !/\n\s*(?:height|max-height)\s*:/.test(mobileTimelineEntryCss),
  'mobile dates, axes, and cards should share auto-sized rows without inherited desktop placement or fixed-height text clipping');
  assert(mobileTimelineFullTitleCss.includes('position: absolute;') &&
    mobileTimelineFullTitleCss.includes('clip-path: inset(50%);') &&
    !mobileTimelineFullTitleCss.includes('display: none') &&
    extractBlock(mobileTimelineCss, '.home-timeline__title-compact {').includes('display: inline;') &&
    extractBlock(timelineCss, '.home-timeline__title-compact {').includes('display: none;'),
  'compact certificate labels should preserve their full accessible titles while desktop keeps the full visual wording');
  assert(timelineCss.includes('.home-timeline__axis {') &&
    timelineCss.includes('grid-column: 2;') &&
    timelineCss.includes('grid-row: 2;') &&
    timelineCss.includes('.home-timeline__item::before') &&
    timelineCss.includes('.home-timeline__item::after') &&
    timelineCss.includes('height: var(--home-timeline-gap);') &&
    timelineCss.includes('.home-timeline__item:first-child .home-timeline__axis::before') &&
    timelineCss.includes('.home-timeline__item:last-child .home-timeline__axis::before') &&
    timelineCss.includes('.home-timeline__dot {') &&
    timelineCss.includes('transform: translate(-50%, -50%);') &&
    evenTimelineAxisCss.includes('right: 0;') &&
    evenTimelineAxisCss.includes('left: 50%;') &&
    desktopTimelineCss.includes('overflow-y: hidden;') &&
    desktopTimelineCss.includes('width: 100%;') &&
    desktopTimelineCss.includes('padding-right: 0;') &&
    desktopTimelineCss.includes('width: min(100%, var(--home-timeline-readable-width));') &&
    desktopTimelineCss.includes('grid-template-rows: minmax(0, 1fr);') &&
    !desktopTimelineCss.includes('.home-accordion__item--about .home-timeline__head') &&
    desktopTimelineCss.includes('flex: 0 0 auto;') &&
    desktopTimelineCss.includes('.home-accordion__item--about .home-timeline__list') &&
    desktopTimelineCss.includes('overflow-y: auto;'),
  'desktop timeline should connect exact milestone centers while its scrollport keeps readable cards below the profile without reserving a removed heading row');
  assert(timelineCss.includes('.home-timeline__media[data-home-timeline-media-tone="dark"]') &&
    timelineCss.includes('object-fit: contain;') &&
    timelineCss.includes('object-position: center;') &&
    timelineCss.includes('background: #091f3b;'),
  'timeline logo plaques should preserve aspect ratios and support an explicit high-contrast treatment');
  const homepageCardCss = extractBlock(css, '.home-accordion__card {');
  const homepageCardInteractiveCss = extractBlock(css, '.home-accordion__card:is(a):is(:hover, :focus-visible)');
  const reducedMotionCss = extractBlock(css, '@media (prefers-reduced-motion: reduce)');
  assert(homepageCardCss.includes('padding-inline-start .18s ease') &&
    homepageCardInteractiveCss.includes('padding-inline-start: 8px;') &&
    reducedMotionCss.includes('.home-accordion__card') &&
    reducedMotionCss.includes('transition: none;'),
  'main-tab preview cards should gain a subtle smooth left inset on hover and keyboard focus without animating for reduced motion');
  assert(css.includes('@media (pointer: coarse)') &&
    css.includes('min-height: 44px;') &&
    css.includes('@media (prefers-reduced-motion: reduce)') &&
    css.includes('.home-accordion__rail:focus-visible') &&
    libraryCss.includes('@media (prefers-reduced-motion: reduce)') &&
    timelineCss.includes('@media (prefers-reduced-motion: reduce)') &&
    timelineCss.includes('scroll-behavior: auto;'),
  'accordion should preserve touch targets, reduced motion, and visible keyboard focus');
  assert(!/prefers-color-scheme\s*:\s*dark/i.test(css) &&
    !/prefers-color-scheme\s*:\s*dark/i.test(libraryCss) &&
    !/prefers-color-scheme\s*:\s*dark/i.test(timelineCss) &&
    !/color-scheme\s*:\s*dark/i.test(css) &&
    !/data-theme/i.test(css),
  'homepage accordion should remain intentionally light-only');

  const canonicalPanelHashJs = extractFunctionBlock(js, 'function canonicalPanelHash');
  const updateLocationJs = extractFunctionBlock(js, 'function updateLocation');
  const updateTriggerStateJs = extractFunctionBlock(js, 'function updateTriggerState');
  const libraryLocationJs = extractFunctionBlock(js, 'function libraryCategoryFromLocation');
  const updateLibraryVisibilityJs = extractFunctionBlock(js, 'function updateLibraryViewVisibility');
  const applyLibraryModeJs = extractFunctionBlock(js, 'function applyLibraryMode');
  const resolveTriggerTargetJs = extractFunctionBlock(js, 'function resolveTriggerTarget');
  const activatePanelTriggerJs = extractFunctionBlock(js, 'function activatePanelTrigger');
  const syncDocumentMetadataJs = extractFunctionBlock(js, 'function syncDocumentMetadata');
  const focusLibraryHeadingJs = extractFunctionBlock(js, 'function focusLibraryHeading');
  const focusOverviewReturnJs = extractFunctionBlock(js, 'function focusOverviewReturn');
  const openLibraryJs = extractFunctionBlock(js, 'function openLibrary');
  const closeLibraryJs = extractFunctionBlock(js, 'function closeLibrary');
  const handleLocationChangeJs = extractFunctionBlock(js, 'function handleLocationChange');
  assert(js.includes('const closeTimers = new Map();') &&
    /const PANEL_TRANSITION_MS = \d+;/.test(js) &&
    js.includes("item.classList.add('is-closing')") &&
    js.includes('finishClosingPanel(itemId)') &&
    js.includes("panel.setAttribute('inert', '')") &&
    js.includes("panel.removeAttribute('inert')") &&
    js.includes('if (!ids.includes(id)) return false;') &&
    js.includes('if (id === activeId) {'),
  'accordion controller should animate the outgoing desktop panel, enforce one interactive panel, and keep its low-level selection operation idempotent');
  const triggerResolutionContext = {};
  vm.runInNewContext(
    `${resolveTriggerTargetJs}\nthis.resolveTriggerTarget = resolveTriggerTarget;`,
    triggerResolutionContext
  );
  const triggerResolutionCases = [
    ['about', 'about', 'about'],
    ['projects', 'projects', 'about'],
    ['tools', 'tools', 'about'],
    ['games', 'games', 'about'],
    ['contact', 'contact', 'about'],
    ['projects', 'tools', 'tools']
  ];
  assert(triggerResolutionCases.every(([currentId, requestedId, expectedId]) =>
    triggerResolutionContext.resolveTriggerTarget(currentId, requestedId, 'about') === expectedId) &&
    activatePanelTriggerJs.includes('const nextId = resolveTriggerTarget(activeId, id, defaultPanel);') &&
    activatePanelTriggerJs.includes('triggerById.get(defaultPanel)?.focus({ preventScroll: true })') &&
    count(js, /activatePanelTrigger\(String\(trigger\.dataset\.homeAccordionTrigger \|\| ''\)\)/g) === 2,
  'overview tabs should retain their five-category selection and collapse-to-About behavior');
  assert(js.includes("event.key === 'ArrowDown'") &&
    js.includes("event.key === 'ArrowRight'") &&
    js.includes("event.key === 'ArrowUp'") &&
    js.includes("event.key === 'ArrowLeft'") &&
    js.includes("event.key === 'Home'") &&
    js.includes("event.key === 'End'") &&
    js.includes("event.key === 'Enter'") &&
    js.includes("event.key === ' '"),
  'accordion rails should support directional movement plus Enter and Space activation');
  const homeKeyIndex = js.indexOf("} else if (event.key === 'Home')");
  const endKeyIndex = js.indexOf("} else if (event.key === 'End')", homeKeyIndex);
  const activationKeyIndex = js.indexOf("} else if (event.key === 'Enter'", endKeyIndex);
  const homeKeyJs = homeKeyIndex >= 0 && endKeyIndex > homeKeyIndex
    ? js.slice(homeKeyIndex, endKeyIndex)
    : '';
  const endKeyJs = endKeyIndex >= 0 && activationKeyIndex > endKeyIndex
    ? js.slice(endKeyIndex, activationKeyIndex)
    : '';
  assert(homeKeyJs.includes('isLibraryMode ? triggerById.get(activeId) : triggers[0]') &&
    endKeyJs.includes('isLibraryMode ? triggerById.get(activeId) : triggers[triggers.length - 1]'),
  'Home and End should keep focus on the sole visible active rail trigger while a library is expanded');
  assert(js.includes('scrollPositions') &&
    js.includes('scroller.scrollTop') &&
    js.includes('timelineScrollerById') &&
    js.includes("panel?.querySelector('[data-home-timeline-scroller]')") &&
    js.includes("window.matchMedia('(min-width: 960px) and (min-height: 620px)')") &&
    js.includes('getVisibleHeaderBottom()') && js.includes('window.scrollTo({'),
  'accordion should preserve spacious-rail panel scroll positions and reveal newly opened stacked sections');
  assert(js.includes('decodeURIComponent(rawId)') &&
    js.includes('catch (error)') &&
    libraryLocationJs.includes("url.searchParams.get('view') === 'library'") &&
    js.includes("projects: '/portfolio'") &&
    js.includes("tools: '/tools'") &&
    js.includes("games: '/games'") &&
    handleLocationChangeJs.includes("currentPath !== '/' && !Object.values(LIBRARY_ROUTES).includes(currentPath)") &&
    handleLocationChangeJs.includes('const nextLibraryCategory = libraryCategoryFromLocation();') &&
    handleLocationChangeJs.includes('const nextPanel = nextLibraryCategory || hashPanel || defaultPanel;') &&
    handleLocationChangeJs.includes('selectPanel(nextPanel, { updateHistory: false, reveal: !nextLibraryMode })') &&
    handleLocationChangeJs.includes('applyLibraryMode(nextLibraryMode') &&
    js.includes("window.addEventListener('hashchange', handleLocationChange)") &&
    js.includes("window.addEventListener('popstate', handleLocationChange)"),
  'accordion should preserve hash history, recognize canonical library routes, normalize legacy library links, and restore popstate in place');
  const duplicateLocationGuardIndex = handleLocationChangeJs.indexOf(
    'if (currentHref === lastHandledLocationHref) return;'
  );
  const locationStateGuardIndex = handleLocationChangeJs.indexOf('if (modeChanged || panelChanged) {');
  const locationScrollSaveIndex = handleLocationChangeJs.indexOf(
    'saveScrollPosition(activeId, isLibraryMode);'
  );
  const locationDocumentSaveIndex = handleLocationChangeJs.indexOf(
    'saveDocumentScrollPosition(activeId, isLibraryMode);'
  );
  const locationStateGuardJs = extractBlock(
    handleLocationChangeJs,
    'if (modeChanged || panelChanged)'
  );
  assert(js.includes("let lastHandledLocationHref = '';") &&
    handleLocationChangeJs.includes('const currentHref = window.location.href;') &&
    handleLocationChangeJs.includes('lastHandledLocationHref = currentHref;') &&
    handleLocationChangeJs.includes('const modeChanged = nextLibraryMode !== isLibraryMode;') &&
    handleLocationChangeJs.includes('const panelChanged = nextPanel !== activeId;') &&
    duplicateLocationGuardIndex >= 0 &&
    locationStateGuardIndex > duplicateLocationGuardIndex &&
    locationScrollSaveIndex > locationStateGuardIndex &&
    locationDocumentSaveIndex > locationStateGuardIndex &&
    locationStateGuardJs.includes('saveScrollPosition(activeId, isLibraryMode);') &&
    locationStateGuardJs.includes('saveDocumentScrollPosition(activeId, isLibraryMode);'),
  'popstate and hashchange should share a location guard and save scroll state only when the active panel or view mode changes');
  assert(focusLibraryHeadingJs.includes("querySelector('[data-home-library-heading]')") &&
    focusLibraryHeadingJs.includes('focus({ preventScroll: true })') &&
    focusOverviewReturnJs.includes('`[data-home-library-open="${id}"]`') &&
    focusOverviewReturnJs.includes('|| triggerById.get(id)') &&
    focusOverviewReturnJs.includes('focus({ preventScroll: true })') &&
    handleLocationChangeJs.includes('afterApply: modeChanged ? restoreLocationState : null') &&
    handleLocationChangeJs.includes('if (nextLibraryMode) focusLibraryHeading(nextPanel);') &&
    handleLocationChangeJs.includes('else focusOverviewReturn(nextPanel);') &&
    applyLibraryModeJs.includes("typeof options.afterApply === 'function'") &&
    applyLibraryModeJs.includes('options.afterApply();'),
  'history-driven mode changes should move focus to the visible library heading or the matching overview return control after the view updates');
  const canonicalHashContext = {};
  vm.runInNewContext(
    `${canonicalPanelHashJs}\nthis.canonicalPanelHash = canonicalPanelHash;`,
    canonicalHashContext
  );
  const expectedTabHashes = {
    about: '#about',
    projects: '#projects',
    tools: '#tools',
    games: '#games',
    contact: '#contact'
  };
  assert(ids.every((id) => canonicalHashContext.canonicalPanelHash(id) === expectedTabHashes[id]) &&
    updateLocationJs.includes('url.hash = id;'),
  'every homepage tab should have a stable, canonical hash URL');
  assert(updateLocationJs.includes('url.pathname = LIBRARY_ROUTES[id]') &&
    updateLocationJs.includes("url.pathname = '/'") &&
    updateLocationJs.includes("url.searchParams.delete('view')") &&
    updateLocationJs.includes("mode === 'replace' ? 'replaceState' : 'pushState'") &&
    updateLocationJs.includes("const view = libraryMode ? 'library' : 'overview';") &&
    updateLocationJs.includes('personalCategory: id') &&
    updateLocationJs.includes('personalView: view') &&
    updateLocationJs.includes('if (nextLocation === currentLocation)') &&
    updateLocationJs.includes('window.history.replaceState(nextState') &&
    openLibraryJs.includes("updateLocation(id, 'push', true)") &&
    closeLibraryJs.includes("updateLocation(closingId, options.historyMode || 'push', false)"),
  'library expansion should push canonical dedicated paths, preserve semantic history state on reload, and collapse to the matching homepage hash');
  assert(js.includes('const LIBRARY_DOCUMENT_TITLES = Object.freeze({') &&
    js.includes("const canonicalLink = document.querySelector('link[rel=\"canonical\"]');") &&
    js.includes('const overviewDocumentMetadata = Object.freeze({') &&
    js.includes('title: document.title') &&
    js.includes("canonicalHref: canonicalLink?.getAttribute('href') || ''") &&
    syncDocumentMetadataJs.includes('document.title = libraryMode && LIBRARY_DOCUMENT_TITLES[id]') &&
    syncDocumentMetadataJs.includes(': overviewDocumentMetadata.title;') &&
    syncDocumentMetadataJs.includes("canonicalLink.setAttribute('href', libraryMode && LIBRARY_ROUTES[id]") &&
    syncDocumentMetadataJs.includes(': overviewDocumentMetadata.canonicalHref);') &&
    count(applyLibraryModeJs, /syncDocumentMetadata\(activeId, next\);/g) === 2,
  'inline library URLs should publish their dedicated title and canonical URL, then restore the homepage metadata on overview return');
  assert(updateLibraryVisibilityJs.includes('visible = isLibraryMode && id === activeId') &&
    applyLibraryModeJs.includes("root.classList.toggle('is-library-mode', next)") &&
    activatePanelTriggerJs.includes('if (isLibraryMode && id === activeId)') &&
    activatePanelTriggerJs.includes('return closeLibrary({ restoreFocus: false });') &&
    updateTriggerStateJs.includes('selected && triggerId === defaultPanel && !isLibraryMode') &&
    updateTriggerStateJs.includes("trigger.setAttribute('aria-disabled', 'true')") &&
    updateTriggerStateJs.includes("trigger.removeAttribute('aria-disabled')") &&
    updateTriggerStateJs.includes("trigger.removeAttribute('aria-current')") &&
    js.includes("scrollTarget.setAttribute('tabindex', '0')") &&
    js.includes("region.removeAttribute('tabindex')") &&
    js.includes('scrollTarget.scrollHeight > scrollTarget.clientHeight + 1') &&
    js.includes('scrollTarget.scrollWidth > scrollTarget.clientWidth + 1'),
  'expanded libraries should remain collapsible from their single rail, animate accessibly, and expose tab stops only for real overflow');
  assert(js.includes('const initialLibraryCategory = libraryCategoryFromLocation();') &&
    js.includes('const initialPanel = initialLibraryCategory || initialHashPanel || defaultPanel;') &&
    js.includes('applyLibraryMode(initialLibraryMode, { animate: false, force: true })') &&
    js.includes("updateLocation(initialPanel, 'replace', true)") &&
    js.includes("updateLocation(initialPanel, 'replace', false)") &&
    js.includes('revealPanelTrigger(initialHashPanel);'),
  'initial deep links should restore canonical library or overview state without animating first paint');
  assert(!html.includes('data-home-accordion-scroller tabindex="0"') &&
    html.includes('<h3>Hi, I’m Daniel.</h3>'),
  'authored panels should avoid generic scroller tab stops and keep panel titles beneath accordion headings');

  assert(navigation.includes("const nextExpanded = enhanced && Boolean(expanded);") &&
    navigation.includes("if (!form.classList.contains('is-expanded'))") &&
    !navigation.includes('const isHomeSearch ='),
  'homepage search should use the same compact, explicitly expandable desktop behavior as the rest of the site');

  assert(!navigation.includes('setupMobileSiteDock') &&
    css.includes('.mobile-site-dock {') &&
    css.includes('display: none !important;'),
  'homepage should share the dock-free navigation runtime and retain a CSS safety fallback');
  assert(count(css, /var\(--personal-footer-block-size, 0px\)/g) === 3 &&
    css.includes('min-height: min(540px, calc(100svh') &&
    !css.includes('.footer.footer-classic'),
  'desktop homepage height should reserve the compact personal footer without hiding it');
  assert(activityEvents.includes("target.closest('[data-home-accordion]')") &&
    activityEvents.includes('category.dataset.homeAccordionTrigger'),
  'homepage category changes should remain analytics-visible');
};
