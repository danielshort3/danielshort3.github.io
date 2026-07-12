'use strict';

(function initSiteChatbot() {
  const allowedPages = new Set([
    'home',
    'contact',
    'portfolio',
    'project',
    'tools',
    'games'
  ]);

  const body = document.body;
  const pageId = body && body.dataset ? body.dataset.page : '';
  if (!allowedPages.has(pageId) || document.querySelector('[data-site-chatbot]')) return;

  const STORAGE_KEY = 'daniel-short-chatbot-conversation';
  const SESSION_STATE_KEY = 'daniel-short-chatbot-session-state';
  const NUDGE_STORAGE_KEY = 'daniel-short-chatbot-nudge-seen';
  const NUDGE_DESKTOP_DELAY_MS = 6000;
  const NUDGE_MOBILE_DELAY_MS = 10000;
  const NUDGE_AUTO_DISMISS_MS = 6000;
  const NUDGE_MOBILE_SCROLL_RATIO = 0.35;
  const NAVIGATION_RESTORE_MS = 3 * 60 * 1000;
  const TRANSCRIPT_MAX_TURNS = 8;
  const DISPLAY_TRANSCRIPT_MAX_CHARS = 6000;
  const HISTORY_TRANSCRIPT_MAX_CHARS = 700;
  const UI_VERSION = 'public-materials-2026-05-29';
  const API_PATH = '/api/chatbot';
  const TURNSTILE_SRC = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
  const AUDIENCE_CONFIG = {};
  const SHARED_FALLBACK_LINKS = [
    { title: 'Home', url: '/', reason: 'Start from the personal site homepage.' },
    { title: 'Projects', url: '/portfolio', reason: 'Browse projects, demos, and case studies.' },
    { title: 'Tools', url: '/tools', reason: 'Open the public tool directory.' },
    { title: 'Games', url: '/games', reason: 'Browse browser games and simulations.' },
    { title: 'Contact', url: '/contact', reason: 'Send a note or find public contact links.' }
  ];
  const storedSessionState = readSessionState();
  const initialAudience = pageAudience() || normalizeAudience(storedSessionState.audience);
  const storedTranscript = normalizeTranscript(storedSessionState.transcript);
  const initialStarterPromptsHidden = storedSessionState.starterPromptsHidden === true ||
    storedSessionState.chipsHidden === true ||
    storedTranscript.some((turn) => turn.role === 'user');

  const state = {
    open: false,
    expanded: false,
    ready: false,
    sending: false,
    locked: false,
    config: null,
    controller: null,
    conversationId: getConversationId(),
    challengeToken: '',
    audience: initialAudience,
    starterPromptsHidden: initialStarterPromptsHidden,
    transcript: storedTranscript,
    pendingFollowupContext: null,
    retryTimer: 0,
    nudgeTimer: 0,
    nudgeDismissTimer: 0,
    nudgeDelayReady: false,
    nudgeShown: false,
    turnstileWidgetId: null
  };

  const root = document.createElement('div');
  root.className = 'site-chatbot';
  root.dataset.siteChatbot = '';
  root.dataset.chatbotVersion = UI_VERSION;
  root.dataset.state = 'closed';
  root.dataset.enabled = 'pending';
  root.dataset.expanded = 'false';
  root.dataset.keyboard = 'false';
  root.dataset.nudge = 'false';
  root.dataset.audience = state.audience || 'general';
  root.dataset.chips = state.starterPromptsHidden ? 'hidden' : 'visible';
  root.innerHTML = `
    <button class="site-chatbot__launcher" type="button" aria-label="Open website chatbot" aria-haspopup="dialog" aria-expanded="false">
      <span class="site-chatbot__launcher-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" focusable="false">
          <path d="M5.6 5.5h12.8a2.6 2.6 0 0 1 2.6 2.6v6.4a2.6 2.6 0 0 1-2.6 2.6h-6.1l-4.1 3.2v-3.2H5.6A2.6 2.6 0 0 1 3 14.5V8.1a2.6 2.6 0 0 1 2.6-2.6Z"/>
          <path d="M8 10.1h8M8 13.2h5.2"/>
        </svg>
      </span>
      <span class="site-chatbot__launcher-text" aria-hidden="true">Open website chatbot</span>
    </button>
    <div class="site-chatbot__nudge" aria-hidden="true">
      <button class="site-chatbot__nudge-action" type="button" tabindex="-1">Need help? Ask the site assistant.</button>
      <button class="site-chatbot__nudge-close" type="button" aria-label="Dismiss chatbot suggestion" tabindex="-1">
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M6 6l12 12M18 6 6 18"/>
        </svg>
      </button>
    </div>
    <section class="site-chatbot__panel" id="site-chatbot-panel" role="dialog" aria-modal="false" aria-label="Ask Daniel's site assistant" aria-hidden="true" inert>
      <header class="site-chatbot__header">
        <button class="site-chatbot__header-expand" type="button" aria-controls="site-chatbot-panel" aria-label="Expand chat panel">
          <span class="site-chatbot__header-copy">
            <span class="site-chatbot__eyebrow">Navigation assistant</span>
            <span class="site-chatbot__title">Ask Daniel's site</span>
          </span>
        </button>
        <button class="site-chatbot__header-toggle" type="button" aria-controls="site-chatbot-panel" aria-expanded="false" aria-label="Expand chat panel">
          <span class="site-chatbot__expand-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false">
              <path d="m7 10 5 5 5-5"/>
            </svg>
          </span>
        </button>
        <button class="site-chatbot__reset" type="button" aria-label="Reset chat">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M4 5v5h5"/>
            <path d="M20 19v-5h-5"/>
            <path d="M6.1 15.5A7 7 0 0 0 18 18"/>
            <path d="M17.9 8.5A7 7 0 0 0 6 6"/>
          </svg>
        </button>
        <button class="site-chatbot__close" type="button" aria-label="Close assistant">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="M6 6l12 12M18 6 6 18"/>
          </svg>
        </button>
      </header>
      <div class="site-chatbot__messages" role="log" aria-live="polite" aria-relevant="additions text"></div>
      <div class="site-chatbot__quick-prompts" aria-label="Suggested questions">
        ${renderQuickPromptButtons()}
      </div>
      <div class="site-chatbot__challenge" hidden></div>
      <form class="site-chatbot__form">
        <div class="site-chatbot__input-shell">
          <label class="visually-hidden" for="site-chatbot-message">Ask a question</label>
          <textarea id="site-chatbot-message" class="site-chatbot__input" name="message" rows="2" maxlength="1000" placeholder="Ask about projects, tools, games, or contact details"></textarea>
          <button class="site-chatbot__send" type="submit" aria-label="Send question">
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="m4 12 16-8-4.8 16-3.1-6.1L4 12Z"/>
              <path d="m12.1 13.9 7.9-9.9"/>
            </svg>
          </button>
        </div>
        <input type="text" name="website" tabindex="-1" autocomplete="off" class="site-chatbot__website" aria-hidden="true">
      </form>
      <p class="site-chatbot__status" aria-live="polite"></p>
    </section>
  `;
  document.body.appendChild(root);

  const launcher = root.querySelector('.site-chatbot__launcher');
  const nudge = root.querySelector('.site-chatbot__nudge');
  const nudgeAction = root.querySelector('.site-chatbot__nudge-action');
  const nudgeClose = root.querySelector('.site-chatbot__nudge-close');
  const panel = root.querySelector('.site-chatbot__panel');
  const header = root.querySelector('.site-chatbot__header');
  const headerExpand = root.querySelector('.site-chatbot__header-expand');
  const headerToggle = root.querySelector('.site-chatbot__header-toggle');
  const resetButton = root.querySelector('.site-chatbot__reset');
  const closeButton = root.querySelector('.site-chatbot__close');
  const messages = root.querySelector('.site-chatbot__messages');
  const quickPrompts = root.querySelector('.site-chatbot__quick-prompts');
  const challenge = root.querySelector('.site-chatbot__challenge');
  const form = root.querySelector('.site-chatbot__form');
  const input = root.querySelector('.site-chatbot__input');
  const sendButton = root.querySelector('.site-chatbot__send');
  const status = root.querySelector('.site-chatbot__status');
  let chromeOffsetFrame = 0;
  let nudgeCheckFrame = 0;
  syncQuickPromptVisibility();
  restoreTranscriptMessages();
  setupChromeOffsetTracking();
  scheduleInitialNudge();
  restoreOpenAfterChatbotNavigation();
  if (!state.ready) loadConfig();

  launcher.addEventListener('click', () => {
    const nextOpen = !state.open;
    dismissNudge('launcher');
    setOpen(nextOpen);
    if (nextOpen) trackChatbotEvent('chatbot_launcher_opened', { source: 'launcher' });
  });
  nudgeAction.addEventListener('click', () => {
    dismissNudge('opened');
    setOpen(true);
  });
  nudgeClose.addEventListener('click', (event) => {
    event.stopPropagation();
    dismissNudge('dismissed');
  });
  headerExpand.addEventListener('click', () => requestExpanded('header'));
  headerToggle.addEventListener('click', () => toggleExpanded('toggle'));
  header.addEventListener('click', (event) => {
    if (event.target.closest('.site-chatbot__close, .site-chatbot__reset, .site-chatbot__header-toggle, .site-chatbot__header-expand')) return;
    requestExpanded('header');
  });
  resetButton.addEventListener('click', resetChat);
  closeButton.addEventListener('click', () => setOpen(false));
  root.addEventListener('click', handleChatbotLinkClick);
  form.addEventListener('submit', handleSubmit);
  sendButton.addEventListener('click', (event) => {
    if (!state.sending) return;
    event.preventDefault();
    abortActiveRequest();
  });
  quickPrompts.addEventListener('click', (event) => {
    const button = event.target.closest('[data-chatbot-prompt]');
    if (!button || button.disabled) return;
    hideStarterPrompts();
    clearFollowups();
    submitPromptText(button.dataset.chatbotPrompt || button.textContent || '');
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  input.addEventListener('focus', requestChromeOffsetUpdate);
  input.addEventListener('blur', () => window.setTimeout(requestChromeOffsetUpdate, 160));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && state.open) setOpen(false);
  });
  document.addEventListener('pointerdown', handleOutsidePointerDown, true);

  function normalizeAudience(value) {
    const normalized = String(value || '').trim().toLowerCase().replace(/_/g, '-');
    if (normalized === 'data science' || normalized === 'datascience') return 'data-science';
    return Object.prototype.hasOwnProperty.call(AUDIENCE_CONFIG, normalized) ? normalized : '';
  }

  function pageAudience() {
    const bodyAudience = normalizeAudience(body && body.dataset ? body.dataset.audience : '');
    if (bodyAudience) return bodyAudience;
    try {
      const params = new URL(window.location.href).searchParams;
      return normalizeAudience(params.get('audience'));
    } catch {
      return '';
    }
  }

  function readSessionState() {
    try {
      const raw = sessionStorage.getItem(SESSION_STATE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      if (!parsed || typeof parsed !== 'object') return {};
      if (parsed.uiVersion !== UI_VERSION) return {};
      return parsed;
    } catch {
      return {};
    }
  }

  function persistSessionState(extra = {}) {
    try {
      const prior = readSessionState();
      sessionStorage.setItem(SESSION_STATE_KEY, JSON.stringify({
        ...prior,
        uiVersion: UI_VERSION,
        audience: state.audience || '',
        starterPromptsHidden: state.starterPromptsHidden === true,
        chipsHidden: state.starterPromptsHidden === true,
        transcript: state.transcript.slice(-TRANSCRIPT_MAX_TURNS),
        open: state.open === true,
        expanded: state.expanded === true,
        updatedAt: Date.now(),
        ...extra
      }));
    } catch {}
  }

  function normalizeTranscript(value) {
    return (Array.isArray(value) ? value : [])
      .map((turn) => {
        const safeRole = turn && turn.role === 'assistant' ? 'assistant' : 'user';
        const normalized = {
          role: safeRole,
          text: normalizeTranscriptMarkdown(turn && turn.text, DISPLAY_TRANSCRIPT_MAX_CHARS)
        };
        if (safeRole === 'assistant') {
          normalized.sources = normalizeStoredLinks(turn && turn.sources, 5);
          normalized.followups = normalizeFollowups(turn && turn.followups);
          normalized.previousQuestion = normalizeTranscriptMarkdown(turn && turn.previousQuestion, 300);
        }
        return normalized;
      })
      .filter((turn) => turn.text)
      .slice(-TRANSCRIPT_MAX_TURNS);
  }

  function normalizeStoredLinks(value, maxItems = 5) {
    const seen = new Set();
    return (Array.isArray(value) ? value : [])
      .map((source) => {
        const rawUrl = source && (source.url || source.href) ? String(source.url || source.href) : '';
        if (!rawUrl) return null;
        let url = '';
        try {
          const parsed = new URL(rawUrl, window.location.origin);
          if (!/^https?:$/.test(parsed.protocol)) return null;
          url = parsed.pathname === rawUrl ? rawUrl : `${parsed.pathname}${parsed.search || ''}${parsed.hash || ''}`;
          if (/^https?:\/\//i.test(rawUrl)) url = parsed.href;
        } catch {
          return null;
        }
        const title = String(source.title || rawUrl).replace(/\s+/g, ' ').trim().slice(0, 80);
        const reason = String(source.reason || '').replace(/\s+/g, ' ').trim().slice(0, 180);
        return { title, url, reason };
      })
      .filter((source) => {
        if (!source || seen.has(source.url)) return false;
        seen.add(source.url);
        return true;
      })
      .slice(0, maxItems);
  }

  function normalizeTranscriptMarkdown(value, maxChars = DISPLAY_TRANSCRIPT_MAX_CHARS) {
    return String(value || '')
      .replace(/\r\n?/g, '\n')
      .replace(/\n{4,}/g, '\n\n\n')
      .trim()
      .slice(0, maxChars);
  }

  function apiTranscript(value) {
    return (Array.isArray(value) ? value : [])
      .map((turn) => ({
        role: turn && turn.role === 'assistant' ? 'assistant' : 'user',
        text: normalizeTranscriptMarkdown(turn && turn.text, HISTORY_TRANSCRIPT_MAX_CHARS)
      }))
      .filter((turn) => turn.text)
      .slice(-TRANSCRIPT_MAX_TURNS);
  }

  function audienceConfig(audience = state.audience) {
    return AUDIENCE_CONFIG[normalizeAudience(audience)] || null;
  }

  function fallbackLinksForAudience(audience = state.audience) {
    const config = audienceConfig(audience);
    if (!config) return SHARED_FALLBACK_LINKS;
    return SHARED_FALLBACK_LINKS;
  }

  function getConversationId() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (/^[a-zA-Z0-9_-]{12,80}$/.test(stored || '')) return stored;
      const generated = makeConversationId();
      localStorage.setItem(STORAGE_KEY, generated);
      return generated;
    } catch {
      return makeConversationId();
    }
  }

  function makeConversationId() {
    return window.crypto && window.crypto.randomUUID
      ? window.crypto.randomUUID().replace(/-/g, '')
      : `${Date.now()}${Math.random().toString(36).slice(2)}`;
  }

  function resetConversationId() {
    const generated = makeConversationId();
    try {
      localStorage.setItem(STORAGE_KEY, generated);
    } catch {}
    return generated;
  }

  function getQuickPrompts() {
    const config = audienceConfig();
    if (pageId === 'project') {
      if (config) {
        return [
          ['Project context', `How does this project connect to Daniel's ${config.roleLabel} work?`],
          ['Skill proof', `What ${config.roleLabel} skills does Daniel demonstrate in this project?`],
          ['Team impact', "How would Daniel's work help a team?"]
        ];
      }
      return [
        ['Project context', "How does this project connect to Daniel's other work?"],
        ['Skill proof', 'What skills does Daniel demonstrate in this project?'],
        ['Implementation', 'What implementation choices matter here?']
      ];
    }
    if (config && pageId === 'portfolio') {
      return config.prompts;
    }
    if (pageId === 'portfolio') {
      return [
        ['Project themes', 'Which projects should I look at first?'],
        ['AI work', 'Which projects use AI or machine learning?'],
        ['Contact', 'How do I contact Daniel?']
      ];
    }
    if (pageId === 'tools') {
      return [
        ['Writing tools', 'Which tools help with writing or text cleanup?'],
        ['Media tools', 'Which tools work with images or media?'],
        ['Contact', 'How do I contact Daniel?']
      ];
    }
    if (pageId === 'games') {
      return [
        ['Game guide', 'Which game should I try first?'],
        ['Simulations', 'Which games are simulations?'],
        ['Tools', 'Which tools has Daniel built?']
      ];
    }
    return [
      ['Projects', 'Which projects should I review first?'],
      ['Tools', 'Which tools has Daniel built?'],
      ['Games', 'Which games has Daniel built?']
    ];
  }

  function renderQuickPromptButtons() {
    return getQuickPrompts().map(([label, prompt]) => (
      `<button type="button" data-chatbot-prompt="${escapeHtml(prompt)}" title="${escapeHtml(label)}">${escapeHtml(prompt)}</button>`
    )).join('\n        ');
  }

  function escapeHtml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function syncQuickPromptVisibility() {
    quickPrompts.hidden = state.starterPromptsHidden;
    root.dataset.chips = state.starterPromptsHidden ? 'hidden' : 'visible';
  }

  function hideStarterPrompts() {
    if (state.starterPromptsHidden) return;
    state.starterPromptsHidden = true;
    syncQuickPromptVisibility();
    persistSessionState();
    trackChatbotEvent('chatbot_starter_prompts_hidden', { audience: state.audience || 'general' });
  }

  function restoreTranscriptMessages() {
    state.transcript.forEach((turn) => {
      if (turn.role === 'assistant') {
        appendMessage('assistant', turn.text, turn.sources || [], [], {
          followups: turn.followups || [],
          previousQuestion: turn.previousQuestion || ''
        });
        return;
      }
      appendMessage('user', turn.text);
    });
  }

  function restoreOpenAfterChatbotNavigation() {
    const shouldRestore = storedSessionState.restoreOpen === true &&
      Number(storedSessionState.restoreUntil || 0) >= Date.now();
    if (!shouldRestore) {
      if (storedSessionState.restoreOpen) persistSessionState({ restoreOpen: false, restoreUntil: 0 });
      return;
    }
    setOpen(true);
    setExpanded(storedSessionState.expanded === true);
    persistSessionState({ restoreOpen: false, restoreUntil: 0 });
  }

  function resetChat() {
    if (state.controller) {
      try {
        state.controller.abort();
      } catch {}
      state.controller = null;
    }
    state.conversationId = resetConversationId();
    state.transcript = [];
    state.pendingFollowupContext = null;
    state.challengeToken = '';
    state.starterPromptsHidden = false;
    input.value = '';
    messages.replaceChildren();
    clearChallenge();
    setSending(false);
    if (root.dataset.enabled === 'false') {
      setDisabled(true);
      appendAssistant('The assistant is not enabled right now. Use these shortcuts instead.', [], fallbackLinksForAudience());
    } else {
      setDisabled(false);
      if (!state.ready) loadConfig();
    }
    syncQuickPromptVisibility();
    setStatus('');
    persistSessionState({
      transcript: [],
      starterPromptsHidden: false,
      chipsHidden: false,
      restoreOpen: false,
      restoreUntil: 0
    });
    trackChatbotEvent('chatbot_reset');
    if (state.open && !state.locked) input.focus({ preventScroll: true });
  }

  function sameSiteNavigationUrl(rawHref) {
    try {
      const parsed = new URL(rawHref, window.location.href);
      const publicHost = parsed.hostname === 'www.danielshort.me' || parsed.hostname === 'danielshort.me';
      if (parsed.origin !== window.location.origin && !publicHost) return null;
      return new URL(`${parsed.pathname}${parsed.search}${parsed.hash}`, window.location.origin);
    } catch {
      return null;
    }
  }

  function linkAllowedForActiveAudience(rawHref) {
    const config = audienceConfig();
    if (!config) return true;
    const url = sameSiteNavigationUrl(rawHref);
    if (!url) return true;
    const path = (url.pathname || '/').replace(/\/+$/, '') || '/';
    const fullPath = `${path}${url.search || ''}`;
    if (path === '/contact') return true;
    if (path === config.homeUrl) return true;
    if (fullPath.startsWith('/portfolio?audience=')) return fullPath === config.portfolioUrl;
    if (path === '/portfolio' || path.startsWith('/portfolio/')) return true;
    return true;
  }

  function handleChatbotLinkClick(event) {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target && event.target.closest ? event.target.closest('a[href]') : null;
    if (!link || !root.contains(link)) return;
    const destination = sameSiteNavigationUrl(link.href);
    const linkType = link.closest('.site-chatbot__sources')
      ? 'source'
      : link.closest('.site-chatbot__nav-links') ? 'suggested' : 'answer';
    let destinationSection = 'external';
    if (destination) {
      destinationSection = String(destination.pathname || '/')
        .split('/')
        .filter(Boolean)[0] || 'home';
    }
    trackChatbotEvent('chatbot_link_click', {
      link_type: linkType,
      destination_kind: destination ? 'internal' : 'external',
      destination_section: safeEventId(destinationSection)
    });
    if (!destination) return;
    event.preventDefault();
    persistSessionState({
      restoreOpen: true,
      restoreUntil: Date.now() + NAVIGATION_RESTORE_MS,
      open: true,
      expanded: state.expanded === true
    });
    window.location.href = destination.href;
  }

  function handleOutsidePointerDown(event) {
    if (!state.open || !event || !event.target) return;
    if (root.contains(event.target)) return;
    setOpen(false);
  }

  function setOpen(nextOpen) {
    state.open = Boolean(nextOpen);
    if (state.open) dismissNudge();
    if (!state.open) setExpanded(false);
    root.dataset.state = state.open ? 'open' : 'closed';
    launcher.setAttribute('aria-expanded', String(state.open));
    panel.setAttribute('aria-hidden', String(!state.open));
    if (state.open) {
      panel.removeAttribute('inert');
    } else {
      panel.setAttribute('inert', '');
    }
    if (state.open) {
      if (!state.ready) loadConfig();
      updateChromeOffset();
      window.setTimeout(() => {
        const focusTarget = state.locked ? closeButton : input;
        focusTarget.focus({ preventScroll: true });
        updateChromeOffset();
      }, 40);
    }
    persistSessionState({ restoreOpen: false, restoreUntil: 0 });
  }

  function setExpanded(nextExpanded) {
    state.expanded = Boolean(nextExpanded);
    root.dataset.expanded = state.expanded ? 'true' : 'false';
    headerToggle.setAttribute('aria-expanded', String(state.expanded));
    headerToggle.setAttribute('aria-label', state.expanded ? 'Collapse chat panel' : 'Expand chat panel');
    headerExpand.setAttribute('aria-disabled', String(state.expanded));
    headerExpand.setAttribute('aria-label', state.expanded ? 'Chat panel expanded' : 'Expand chat panel');
    headerExpand.tabIndex = state.expanded ? -1 : 0;
    updateChromeOffset();
    persistSessionState();
  }

  function requestExpanded(_source = 'header') {
    if (state.expanded) return;
    setExpanded(true);
  }

  function requestCollapsed(_source = 'toggle') {
    if (!state.expanded) return;
    setExpanded(false);
  }

  function toggleExpanded(source = 'toggle') {
    if (state.expanded) {
      requestCollapsed(source);
    } else {
      requestExpanded(source);
    }
  }

  function scheduleInitialNudge() {
    if (!shouldShowNudge()) return;
    window.addEventListener('scroll', requestNudgeCheck, { passive: true });
    state.nudgeTimer = window.setTimeout(() => {
      state.nudgeTimer = 0;
      state.nudgeDelayReady = true;
      requestNudgeCheck();
    }, nudgeDelayMs());
  }

  function shouldShowNudge() {
    try {
      return localStorage.getItem(NUDGE_STORAGE_KEY) !== 'true';
    } catch {
      return true;
    }
  }

  function requestNudgeCheck() {
    if (nudgeCheckFrame) return;
    const schedule = window.requestAnimationFrame || ((callback) => window.setTimeout(callback, 16));
    nudgeCheckFrame = schedule(() => {
      nudgeCheckFrame = 0;
      tryShowNudge();
    });
  }

  function tryShowNudge() {
    if (!state.nudgeDelayReady || state.nudgeShown || state.open || !root.isConnected || root.dataset.enabled !== 'true' || !shouldShowNudge()) return;
    if (isConsentBannerOpen()) return;
    if (isMobileViewport() && !hasScrolledEnoughForNudge()) return;
    state.nudgeShown = true;
    root.dataset.nudge = 'true';
    nudge.setAttribute('aria-hidden', 'false');
    nudgeAction.tabIndex = 0;
    nudgeClose.tabIndex = 0;
    trackChatbotEvent('chatbot_nudge_shown');
    state.nudgeDismissTimer = window.setTimeout(() => dismissNudge('timeout'), NUDGE_AUTO_DISMISS_MS);
  }

  function nudgeDelayMs() {
    return isMobileViewport() ? NUDGE_MOBILE_DELAY_MS : NUDGE_DESKTOP_DELAY_MS;
  }

  function hasScrolledEnoughForNudge() {
    const doc = document.documentElement;
    const scrollable = Math.max(0, (doc && doc.scrollHeight ? doc.scrollHeight : 0) - (window.innerHeight || 0));
    if (scrollable <= 0) return true;
    const scrollY = window.scrollY || window.pageYOffset || 0;
    return scrollY / scrollable >= NUDGE_MOBILE_SCROLL_RATIO;
  }

  function isMobileViewport() {
    return window.matchMedia
      ? window.matchMedia('(max-width: 640px)').matches
      : (window.innerWidth || document.documentElement.clientWidth || 0) <= 640;
  }

  function isConsentBannerOpen() {
    const banner = document.getElementById('pcz-banner');
    return document.body.dataset.consentBanner === 'open' && (!banner || banner.hidden !== true);
  }

  function dismissNudge(reason = 'dismissed') {
    const wasVisible = root.dataset.nudge === 'true';
    if (state.nudgeTimer) {
      window.clearTimeout(state.nudgeTimer);
      state.nudgeTimer = 0;
    }
    if (state.nudgeDismissTimer) {
      window.clearTimeout(state.nudgeDismissTimer);
      state.nudgeDismissTimer = 0;
    }
    window.removeEventListener('scroll', requestNudgeCheck);
    root.dataset.nudge = 'false';
    nudge.setAttribute('aria-hidden', 'true');
    nudgeAction.tabIndex = -1;
    nudgeClose.tabIndex = -1;
    try {
      localStorage.setItem(NUDGE_STORAGE_KEY, 'true');
    } catch {}
    if (wasVisible) {
      trackChatbotEvent(reason === 'opened' ? 'chatbot_nudge_opened' : 'chatbot_nudge_dismissed', { reason });
    }
  }

  async function loadConfig() {
    state.ready = true;
    setStatus('Connecting...');
    try {
      const res = await fetch(API_PATH, {
        headers: { accept: 'application/json' },
        credentials: 'same-origin'
      });
      const data = await res.json().catch(() => null);
      if (!res.ok || !data) throw new Error('Assistant unavailable.');
      state.config = data;
      setStatus('');
      if (!data.enabled) {
        setAvailability(false);
        appendAssistant('The assistant is not enabled right now. Use these shortcuts instead.', [], fallbackLinksForAudience());
        setDisabled(true);
        return;
      }
      setAvailability(true);
    } catch {
      setAvailability(false);
      appendAssistant('The assistant API is not available from this view. Use these shortcuts instead.', [], fallbackLinksForAudience());
      setDisabled(true);
      setStatus('');
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (state.sending) {
      abortActiveRequest();
      return;
    }

    const formData = new FormData(form);
    const website = String(formData.get('website') || '').trim();
    const message = String(formData.get('message') || '').trim();
    if (website || !message) return;

    const followupContext = state.pendingFollowupContext;
    const priorTranscript = apiTranscript(state.transcript);
    trackChatbotEvent('chatbot_question_submit', {
      question_length_bucket: eventLengthBucket(message),
      question_token_bucket: eventTokenBucket(message),
      transcript_size_bucket: eventCountBucket(priorTranscript.length),
      is_followup: Boolean(followupContext)
    });
    state.pendingFollowupContext = null;
    hideStarterPrompts();
    clearFollowups();
    appendUser(message);
    rememberTurn('user', message);
    input.value = '';
    setSending(true);
    setStatus('Thinking...');
    clearChallenge();
    const assistant = appendAssistant('', [], [], { pending: true });

    try {
      const payload = {
        message,
        stream: true,
        conversationId: state.conversationId,
        challengeToken: state.challengeToken,
        website: '',
        history: priorTranscript,
        followupContext,
        pageContext: {
          url: window.location.href,
          title: document.title,
          audience: state.audience || ''
        }
      };
      const data = await submitStreamingRequest(payload, assistant, message);
      if (!data) return;
      state.challengeToken = '';
      if (data.answer) {
        rememberTurn('assistant', data.answer, {
          sources: data.sources || [],
          followups: data.followups || [],
          previousQuestion: message
        });
      }
      trackChatbotEvent('chatbot_response_success', {
        response_length_bucket: eventLengthBucket(data.answer),
        source_count: Array.isArray(data.sources) ? data.sources.length : 0,
        followup_count: Array.isArray(data.followups) ? data.followups.length : 0,
        response_mode: data.responseMode === 'json' ? 'json' : 'stream'
      });
      setStatus('');
    } catch (err) {
      if (err && err.name === 'AbortError') {
        const partial = String(assistant.bubble.dataset.partialAnswer || '').trim();
        updateAssistantMessage(assistant, partial ? `${partial}\n\nStopped before final answer.` : 'Stopped before final answer.');
        trackChatbotEvent('chatbot_response_stopped', {
          had_partial_response: Boolean(partial)
        });
      } else {
        updateAssistantMessage(assistant, err && err.message ? err.message : 'The assistant could not answer right now.');
        trackChatbotEvent('chatbot_response_error', {
          error_type: err instanceof TypeError ? 'network' : 'api'
        });
      }
      setStatus('');
    } finally {
      setSending(false);
    }
  }

  async function submitStreamingRequest(payload, assistant, lastMessage) {
    state.controller = new AbortController();
    const res = await fetch(API_PATH, {
      method: 'POST',
      headers: {
        accept: 'application/x-ndjson, application/json',
        'content-type': 'application/json',
        'x-chatbot-session': state.conversationId
      },
      credentials: 'same-origin',
      signal: state.controller.signal,
      body: JSON.stringify(payload)
    });

    const contentType = String(res.headers.get('content-type') || '');
    if (!res.body || contentType.includes('application/json')) {
      const data = await res.json().catch(() => null);
      if (res.status === 429 && data) {
        handleRateLimit(data, lastMessage, assistant);
        return null;
      }
      if (!res.ok || !data || data.ok === false) {
        throw new Error(data && data.error ? data.error : 'The assistant could not answer right now.');
      }
      updateAssistantMessage(assistant, data.answer || 'I could not find a supported answer.', data.sources || [], data.suggestedLinks || [], data.followups || [], lastMessage);
      return { ...data, responseMode: 'json' };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    const meta = { sources: [], suggestedLinks: [] };
    let buffer = '';
    let streamedText = '';
    let finalData = null;

    const consumeEvent = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      const event = JSON.parse(trimmed);
      if (event.type === 'meta') {
        meta.sources = event.sources || meta.sources;
        meta.suggestedLinks = event.suggestedLinks || meta.suggestedLinks;
        return;
      }
      if (event.type === 'token') {
        streamedText += String(event.text || '');
        assistant.bubble.dataset.partialAnswer = streamedText;
        renderAssistantBubble(assistant.bubble, streamedText, meta.sources);
        messages.scrollTop = messages.scrollHeight;
        return;
      }
      if (event.type === 'error') {
        setStatus(event.error || 'The streamed answer stopped early.');
        return;
      }
      if (event.type === 'done') {
        finalData = event.data || {};
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split(/\n/);
      buffer = lines.pop() || '';
      lines.forEach(consumeEvent);
      if (done) break;
    }
    if (buffer.trim()) consumeEvent(buffer);

    const answer = finalData && finalData.answer ? finalData.answer : streamedText;
    updateAssistantMessage(
      assistant,
      answer || 'I could not find a supported answer.',
      (finalData && finalData.sources) || meta.sources,
      (finalData && finalData.suggestedLinks) || meta.suggestedLinks,
      (finalData && finalData.followups) || [],
      lastMessage
    );
    return {
      ...(finalData || {}),
      answer,
      sources: (finalData && finalData.sources) || meta.sources,
      suggestedLinks: (finalData && finalData.suggestedLinks) || meta.suggestedLinks,
      responseMode: 'stream'
    };
  }

  function abortActiveRequest() {
    if (state.controller) {
      state.controller.abort();
      setStatus('Stopping...');
    }
  }

  function handleRateLimit(data, lastMessage, target = null) {
    const seconds = Math.max(1, Number(data.retryAfter) || 20);
    const message = limitMessage(data, seconds);
    const limitType = data.challengeRequired
      ? 'challenge'
      : String(data.error || '').includes('Daily') ? 'daily' : 'throttle';
    trackChatbotEvent('chatbot_rate_limited', {
      limit_type: limitType,
      retry_bucket: eventRetryBucket(seconds),
      challenge_required: Boolean(data.challengeRequired)
    });
    if (target) {
      updateAssistantMessage(target, message);
    } else {
      appendAssistant(message);
    }
    if (data.challengeRequired) {
      setStatus('Verification needed.');
      renderChallenge(lastMessage, data);
    } else {
      startRetryTimer(seconds);
    }
    setSending(false);
  }

  function limitMessage(data, seconds) {
    if (data.challengeRequired) {
      return 'Please complete the quick verification before sending another question.';
    }
    if (String(data.error || '').includes('Daily')) {
      return 'The daily question limit has been reached for this browser/session.';
    }
    return `Please wait ${seconds} seconds before sending another question.`;
  }

  function startRetryTimer(seconds) {
    window.clearInterval(state.retryTimer);
    let remaining = seconds;
    setDisabled(true);
    setStatus(`Try again in ${remaining}s`);
    state.retryTimer = window.setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        window.clearInterval(state.retryTimer);
        setDisabled(false);
        setStatus('');
      } else {
        setStatus(`Try again in ${remaining}s`);
      }
    }, 1000);
  }

  function renderChallenge(lastMessage, limitData = {}) {
    const siteKey = state.config && state.config.turnstileSiteKey;
    setDisabled(true);
    challenge.hidden = false;
    challenge.innerHTML = '';
    const note = document.createElement('p');
    note.textContent = siteKey
      ? 'Verification helps keep chatbot costs controlled.'
      : 'Verification is configured on the server, but no public site key is available in this environment.';
    challenge.appendChild(note);

    if (!siteKey) {
      startRetryTimer(Math.max(1, Number(limitData.retryAfter) || 30));
      return;
    }

    const container = document.createElement('div');
    container.className = 'site-chatbot__turnstile';
    challenge.appendChild(container);
    loadTurnstile()
      .then(() => {
        if (!window.turnstile || !container.isConnected) return;
        if (state.turnstileWidgetId !== null) {
          try {
            window.turnstile.remove(state.turnstileWidgetId);
          } catch {}
        }
        state.turnstileWidgetId = window.turnstile.render(container, {
          sitekey: siteKey,
          callback: (token) => {
            state.challengeToken = token;
            setDisabled(false);
            setStatus('Verified. Send your question again.');
            input.value = lastMessage || input.value;
            input.focus({ preventScroll: true });
          },
          'expired-callback': () => {
            state.challengeToken = '';
            setStatus('Verification expired.');
          }
        });
      })
      .catch(() => {
        note.textContent = 'Verification could not load. Please try again in a moment.';
      });
  }

  function loadTurnstile() {
    if (window.turnstile) return Promise.resolve();
    const existing = document.querySelector(`script[src="${TURNSTILE_SRC}"]`);
    if (existing) {
      return new Promise((resolve, reject) => {
        existing.addEventListener('load', resolve, { once: true });
        existing.addEventListener('error', reject, { once: true });
      });
    }
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = TURNSTILE_SRC;
      script.async = true;
      script.defer = true;
      script.addEventListener('load', resolve, { once: true });
      script.addEventListener('error', reject, { once: true });
      document.head.appendChild(script);
    });
  }

  function clearChallenge() {
    challenge.hidden = true;
    challenge.innerHTML = '';
    if (state.turnstileWidgetId !== null && window.turnstile) {
      try {
        window.turnstile.remove(state.turnstileWidgetId);
      } catch {}
      state.turnstileWidgetId = null;
    }
  }

  function appendUser(text) {
    return appendMessage('user', text);
  }

  function appendAssistant(text, sources = [], suggestedLinks = [], options = {}) {
    return appendMessage('assistant', text, sources, suggestedLinks, options);
  }

  function appendMessage(role, text, sources = [], suggestedLinks = [], options = {}) {
    const item = document.createElement('article');
    item.className = `site-chatbot__message site-chatbot__message--${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'site-chatbot__bubble';
    if (role === 'assistant') {
      renderAssistantBubble(bubble, text, sources, options.pending);
    } else {
      renderPlainBubble(bubble, text);
    }
    item.appendChild(bubble);

    if (role === 'assistant') renderAssistantExtras(item, sources, suggestedLinks, options.followups || [], options.previousQuestion || '', text);

    messages.appendChild(item);
    messages.scrollTop = messages.scrollHeight;
    return { item, bubble };
  }

  function updateAssistantMessage(target, text, sources = [], suggestedLinks = [], followups = [], previousQuestion = '') {
    if (!target || !target.bubble || !target.item) return;
    renderAssistantBubble(target.bubble, text, sources);
    renderAssistantExtras(target.item, sources, suggestedLinks, followups, previousQuestion, text);
    messages.scrollTop = messages.scrollHeight;
  }

  function renderPlainBubble(bubble, text) {
    bubble.replaceChildren();
    textToParagraphs(text).forEach((paragraph) => {
      const p = document.createElement('p');
      p.textContent = paragraph;
      bubble.appendChild(p);
    });
  }

  function renderAssistantBubble(bubble, text, sources = [], pending = false) {
    bubble.replaceChildren();
    const normalized = String(text || '').trim();
    if (!normalized && pending) {
      const p = document.createElement('p');
      p.textContent = 'Thinking...';
      bubble.appendChild(p);
      return;
    }
    renderMarkdown(bubble, normalized || 'I could not find a supported answer.', normalizeLinks(sources, 8));
  }

  function renderAssistantExtras(item, sources = [], suggestedLinks = [], followups = [], previousQuestion = '', answer = '') {
    item.querySelectorAll('.site-chatbot__nav-links, .site-chatbot__sources, .site-chatbot__followups').forEach((node) => node.remove());

    const sourceLinks = normalizeLinks(sources, 5);

    if (sourceLinks.length) {
      const sourceList = document.createElement('details');
      sourceList.className = 'site-chatbot__sources';
      const summary = document.createElement('summary');
      summary.textContent = `Sources (${sourceLinks.length})`;
      const sourcePills = document.createElement('div');
      sourcePills.className = 'site-chatbot__source-pills';
      sourceLinks.forEach((source) => sourcePills.appendChild(makeLink(source)));
      sourceList.append(summary, sourcePills);
      item.appendChild(sourceList);
    }

    const cleanFollowups = normalizeFollowups(followups);
    if (cleanFollowups.length) {
      const wrap = document.createElement('div');
      wrap.className = 'site-chatbot__followups';
      cleanFollowups.forEach((text) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = text;
        button.disabled = state.sending || state.locked;
        button.addEventListener('click', () => {
          hideStarterPrompts();
          clearFollowups();
          submitPromptText(text, {
            source: 'recommended_followup',
            prompt: text,
            previous_question: previousQuestion,
            previous_answer: String(answer || '').slice(0, 650),
            source_labels: sourceLinks.map((source) => source.title).slice(0, 8),
            source_urls: sourceLinks.map((source) => source.href).slice(0, 8)
          });
        });
        wrap.appendChild(button);
      });
      item.appendChild(wrap);
    }
  }

  function textToParagraphs(value) {
    const text = String(value || '').trim();
    if (!text) return ['I could not find a supported answer.'];
    return text.split(/\n{2,}/).map((item) => item.trim()).filter(Boolean).slice(0, 5);
  }

  function normalizeLinks(sources, maxItems = 5) {
    const seen = new Set();
    return (Array.isArray(sources) ? sources : [])
      .map((source) => {
        const rawUrl = source && source.url ? String(source.url) : '';
        if (!rawUrl) return null;
        let href = '';
        try {
          const parsed = new URL(rawUrl, window.location.origin);
          if (!/^https?:$/.test(parsed.protocol)) return null;
          href = parsed.href;
        } catch {
          return null;
        }
        const title = String(source.title || rawUrl).trim();
        const reason = String(source.reason || '').trim();
        return { href, title: title.slice(0, 80), reason };
      })
      .filter((source) => {
        if (!source || seen.has(source.href)) return false;
        if (!linkAllowedForActiveAudience(source.href)) return false;
        seen.add(source.href);
        return true;
      })
      .slice(0, maxItems);
  }

  function makeLink(source) {
    const link = document.createElement('a');
    link.href = source.href;
    link.textContent = source.title;
    link.title = source.reason || source.title;
    link.rel = 'noopener noreferrer';
    return link;
  }

  function normalizeFollowups(value) {
    const seen = new Set();
    return (Array.isArray(value) ? value : [])
      .map((item) => String(item || '').replace(/\s+/g, ' ').trim())
      .filter((item) => {
        const key = item.toLowerCase();
        if (!item || seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .slice(0, 3);
  }

  function renderMarkdown(container, text, sources = []) {
    const state = { usedUrls: new Set(), inlineCount: 0, maxLinks: 8 };
    String(text || '').replace(/\r\n?/g, '\n').split(/\n{2,}/).filter(Boolean).slice(0, 8).forEach((block) => {
      const lines = block.split('\n').map((line) => line.trim()).filter(Boolean);
      const bulletLines = lines.filter((line) => /^[-*]\s+/.test(line));
      if (bulletLines.length && bulletLines.length === lines.length) {
        const list = document.createElement('ul');
        bulletLines.forEach((line) => {
          const item = document.createElement('li');
          appendInlineMarkdown(item, line.replace(/^[-*]\s+/, ''), state);
          list.appendChild(item);
        });
        container.appendChild(list);
        return;
      }
      const p = document.createElement('p');
      lines.forEach((line, index) => {
        if (index > 0) p.appendChild(document.createElement('br'));
        appendInlineMarkdown(p, line, state);
      });
      container.appendChild(p);
    });
    autoLinkSourcePhrases(container, sources, state);
  }

  function appendInlineMarkdown(container, text, state, options = {}) {
    const allowLinks = options.allowLinks !== false;
    let index = 0;
    const source = String(text || '');
    while (index < source.length) {
      const markdownLink = nextRegexMatch(/\[([^\]]+)\]\(([^)\s]+)\)/g, source, index);
      const bareUrl = nextRegexMatch(/https?:\/\/[^\s<>)"']+/g, source, index);
      const bold = source.indexOf('**', index);
      const italic = source.indexOf('*', index);
      const candidates = [
        markdownLink ? { type: 'link', index: markdownLink.index, match: markdownLink } : null,
        bareUrl ? { type: 'url', index: bareUrl.index, match: bareUrl } : null,
        bold >= 0 ? { type: 'bold', index: bold } : null,
        italic >= 0 ? { type: 'italic', index: italic } : null
      ].filter(Boolean).sort((a, b) => a.index - b.index);
      const next = candidates[0];
      if (!next) {
        container.append(document.createTextNode(source.slice(index)));
        return;
      }
      if (next.index > index) container.append(document.createTextNode(source.slice(index, next.index)));
      if (next.type === 'link') {
        const [, label, rawUrl] = next.match;
        if (allowLinks) {
          appendSafeLink(container, label, rawUrl, state);
        } else {
          container.append(document.createTextNode(label));
        }
        index = next.index + next.match[0].length;
        continue;
      }
      if (next.type === 'url') {
        const rawUrl = cleanLinkedUrl(next.match[0]);
        if (allowLinks) {
          appendSafeLink(container, rawUrl, rawUrl, state);
        } else {
          container.append(document.createTextNode(rawUrl));
        }
        index = next.index + next.match[0].length;
        continue;
      }
      const marker = next.type === 'bold' ? '**' : '*';
      const end = source.indexOf(marker, next.index + marker.length);
      if (end > next.index + marker.length) {
        const node = document.createElement(next.type === 'bold' ? 'strong' : 'em');
        appendInlineMarkdown(node, source.slice(next.index + marker.length, end), state, { allowLinks });
        container.appendChild(node);
        index = end + marker.length;
        continue;
      }
      container.append(document.createTextNode(source.charAt(next.index)));
      index = next.index + 1;
    }
  }

  function nextRegexMatch(pattern, text, index) {
    pattern.lastIndex = index;
    return pattern.exec(text);
  }

  function cleanLinkedUrl(rawUrl) {
    return String(rawUrl || '').replace(/[.,;:!?]+$/, '');
  }

  function appendSafeLink(container, label, rawUrl, state) {
    let href = '';
    try {
      const parsed = new URL(cleanLinkedUrl(rawUrl), window.location.origin);
      if (!/^https?:$/.test(parsed.protocol)) throw new Error('Unsupported protocol');
      href = parsed.href;
    } catch {
      container.append(document.createTextNode(label));
      return null;
    }
    const key = href.replace(/\/+$/, '').toLowerCase();
    if (state && (state.usedUrls.has(key) || state.inlineCount >= state.maxLinks)) {
      container.append(document.createTextNode(label));
      return null;
    }
    if (!linkAllowedForActiveAudience(href)) {
      container.append(document.createTextNode(label));
      return null;
    }
    if (state) {
      state.usedUrls.add(key);
      state.inlineCount += 1;
    }
    const link = document.createElement('a');
    link.href = href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = label;
    container.appendChild(link);
    return link;
  }

  function autoLinkSourcePhrases(container, sources, state) {
    if (!sources.length || state.inlineCount >= state.maxLinks) return;
    const candidates = sources.flatMap((source) => sourcePhrases(source).map((phrase) => ({ phrase, href: source.href })));
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (state.inlineCount >= state.maxLinks || (node.parentElement && node.parentElement.closest('a'))) return;
      const text = node.nodeValue || '';
      const lower = text.toLowerCase();
      const match = candidates
        .map((candidate) => ({ ...candidate, index: lower.indexOf(candidate.phrase.toLowerCase()) }))
        .filter((candidate) => candidate.index >= 0 && !state.usedUrls.has(candidate.href.replace(/\/+$/, '').toLowerCase()))
        .sort((a, b) => b.phrase.length - a.phrase.length)[0];
      if (!match) return;
      const after = node.splitText(match.index);
      const rest = after.splitText(match.phrase.length);
      const link = document.createElement('a');
      link.href = match.href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = after.nodeValue;
      after.parentNode.replaceChild(link, after);
      state.usedUrls.add(match.href.replace(/\/+$/, '').toLowerCase());
      state.inlineCount += 1;
      rest.normalize();
    });
  }

  function sourcePhrases(source) {
    const phrases = [];
    const add = (value) => {
      const text = String(value || '').replace(/\s+/g, ' ').trim();
      if (text.length > 4 && !phrases.some((item) => item.toLowerCase() === text.toLowerCase())) phrases.push(text);
    };
    add(source.title);
    try {
      const parsed = new URL(source.href);
      const slug = parsed.pathname.split('/').filter(Boolean).pop() || '';
      add(slug.replace(/[-_]+/g, ' '));
    } catch {}
    return phrases.slice(0, 4);
  }

  function clearFollowups() {
    messages.querySelectorAll('.site-chatbot__followups').forEach((node) => node.remove());
  }

  function rememberTurn(role, text, metadata = {}) {
    const safeRole = role === 'assistant' ? 'assistant' : 'user';
    const safeText = normalizeTranscriptMarkdown(text);
    if (!safeText) return;
    const turn = { role: safeRole, text: safeText.slice(0, DISPLAY_TRANSCRIPT_MAX_CHARS) };
    if (safeRole === 'assistant') {
      turn.sources = normalizeStoredLinks(metadata.sources, 5);
      turn.followups = normalizeFollowups(metadata.followups);
      turn.previousQuestion = normalizeTranscriptMarkdown(metadata.previousQuestion, 300);
    }
    state.transcript.push(turn);
    state.transcript = state.transcript.slice(-TRANSCRIPT_MAX_TURNS);
    persistSessionState();
  }

  function setSending(nextSending) {
    state.sending = Boolean(nextSending);
    root.dataset.busy = state.sending ? 'true' : 'false';
    if (!state.sending) state.controller = null;
    syncControls();
  }

  function setDisabled(disabled) {
    state.locked = Boolean(disabled);
    syncControls();
  }

  function setAvailability(available) {
    root.dataset.enabled = available ? 'true' : 'false';
    if (available) requestNudgeCheck();
  }

  function syncControls() {
    const disabled = state.locked;
    sendButton.disabled = disabled;
    input.disabled = state.sending || disabled;
    sendButton.setAttribute('aria-label', state.sending ? 'Stop response' : 'Send question');
    sendButton.title = state.sending ? 'Stop response' : '';
    quickPrompts.querySelectorAll('button').forEach((button) => {
      button.disabled = state.sending || disabled;
    });
    messages.querySelectorAll('.site-chatbot__followups button').forEach((button) => {
      button.disabled = state.sending || disabled;
    });
  }

  function setStatus(text) {
    status.textContent = text || '';
  }

  function submitPromptText(text, followupContext = null) {
    const prompt = String(text || '').trim();
    if (!prompt || state.sending || state.locked) return;
    state.pendingFollowupContext = followupContext;
    input.value = prompt;
    form.requestSubmit();
  }

  function trackChatbotEvent(name, params = {}) {
    if (typeof window.gaEvent !== 'function') return;
    window.gaEvent(name, {
      page_id: pageId || 'unknown',
      audience: state.audience || 'general',
      ...params
    });
  }

  function safeEventId(value) {
    return String(value || '')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 48) || 'unknown';
  }

  function eventLengthBucket(value) {
    const length = String(value || '').trim().length;
    if (length === 0) return '0';
    if (length <= 50) return '1-50';
    if (length <= 150) return '51-150';
    if (length <= 400) return '151-400';
    if (length <= 800) return '401-800';
    return '801-plus';
  }

  function eventTokenBucket(value) {
    const count = String(value || '').trim().split(/\s+/).filter(Boolean).length;
    if (count <= 3) return '1-3';
    if (count <= 8) return '4-8';
    if (count <= 20) return '9-20';
    return '21-plus';
  }

  function eventCountBucket(value) {
    const count = Math.max(0, Number(value) || 0);
    if (count === 0) return '0';
    if (count <= 2) return '1-2';
    if (count <= 5) return '3-5';
    return '6-plus';
  }

  function eventRetryBucket(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    if (value <= 10) return '1-10';
    if (value <= 30) return '11-30';
    if (value <= 60) return '31-60';
    return '61-plus';
  }

  function setupChromeOffsetTracking() {
    if (!document.body) {
      document.addEventListener('DOMContentLoaded', setupChromeOffsetTracking, { once: true });
      return;
    }
    updateChromeOffset();
    window.addEventListener('resize', requestChromeOffsetUpdate, { passive: true });
    window.addEventListener('orientationchange', requestChromeOffsetUpdate, { passive: true });
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', requestChromeOffsetUpdate, { passive: true });
      window.visualViewport.addEventListener('scroll', requestChromeOffsetUpdate, { passive: true });
    }

    const mutationObserver = new MutationObserver(() => {
      observeConsentBanner();
      updateChromeOffset();
      requestNudgeCheck();
    });
    try {
      mutationObserver.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-consent-banner'],
        childList: true
      });
    } catch {
      mutationObserver.disconnect();
    }

    let resizeObserver = null;
    function observeConsentBanner() {
      if (!('ResizeObserver' in window)) return;
      const banner = document.getElementById('pcz-banner');
      if (!banner || banner.dataset.siteChatbotObserved === 'true') return;
      banner.dataset.siteChatbotObserved = 'true';
      if (!resizeObserver) resizeObserver = new ResizeObserver(updateChromeOffset);
      resizeObserver.observe(banner);
    }

    observeConsentBanner();
    window.setTimeout(requestChromeOffsetUpdate, 80);
    window.setTimeout(requestChromeOffsetUpdate, 480);
    window.setTimeout(requestNudgeCheck, 80);
  }

  function requestChromeOffsetUpdate() {
    if (chromeOffsetFrame) return;
    const schedule = window.requestAnimationFrame || ((callback) => window.setTimeout(callback, 16));
    chromeOffsetFrame = schedule(() => {
      chromeOffsetFrame = 0;
      updateChromeOffset();
    });
  }

  function updateChromeOffset() {
    const banner = document.getElementById('pcz-banner');
    const bannerIsOpen = document.body.dataset.consentBanner === 'open';
    const bannerRect = banner && bannerIsOpen ? banner.getBoundingClientRect() : null;
    const offset = bannerRect && bannerRect.height > 0 ? Math.ceil(bannerRect.height + 10) : 0;
    root.style.setProperty('--site-chatbot-consent-offset', `${offset}px`);

    const viewport = window.visualViewport || null;
    const layoutHeight = window.innerHeight || document.documentElement.clientHeight || 0;
    const visualHeight = viewport && viewport.height ? viewport.height : layoutHeight;
    const visualOffsetTop = viewport && Number.isFinite(viewport.offsetTop) ? viewport.offsetTop : 0;
    const heightLoss = Math.max(0, layoutHeight - visualHeight);
    const keyboardOffset = viewport ? Math.max(0, Math.ceil(layoutHeight - visualHeight - visualOffsetTop)) : 0;
    const mobileViewport = window.matchMedia
      ? window.matchMedia('(max-width: 640px)').matches
      : (window.innerWidth || document.documentElement.clientWidth || 0) <= 640;
    const activeElement = document.activeElement;
    const focusWithinChatbot = Boolean(activeElement && root.contains(activeElement));
    const keyboardLikely = mobileViewport && state.open && focusWithinChatbot && (
      activeElement === input || keyboardOffset > 24 || heightLoss > 80
    );

    root.style.setProperty('--site-chatbot-viewport-height', `${Math.max(240, Math.floor(visualHeight || layoutHeight || 0))}px`);
    root.style.setProperty('--site-chatbot-keyboard-offset', `${keyboardLikely ? keyboardOffset : 0}px`);
    root.dataset.keyboard = keyboardLikely ? 'true' : 'false';
    if (keyboardLikely) messages.scrollTop = messages.scrollHeight;
  }
})();
