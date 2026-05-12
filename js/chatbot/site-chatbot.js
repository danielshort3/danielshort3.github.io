'use strict';

(function initSiteChatbot() {
  const allowedPages = new Set([
    'analytics',
    'data-science',
    'tourism',
    'contact',
    'portfolio',
    'project',
    'resume-analytics',
    'resume-data-science',
    'resume-tourism'
  ]);

  const body = document.body;
  const pageId = body && body.dataset ? body.dataset.page : '';
  if (!allowedPages.has(pageId) || document.querySelector('[data-site-chatbot]')) return;
  body.dataset.siteChatbotActive = 'true';

  const STORAGE_KEY = 'daniel-short-chatbot-conversation';
  const NUDGE_STORAGE_KEY = 'daniel-short-chatbot-nudge-seen';
  const NUDGE_DESKTOP_DELAY_MS = 6000;
  const NUDGE_MOBILE_DELAY_MS = 10000;
  const NUDGE_AUTO_DISMISS_MS = 6000;
  const NUDGE_MOBILE_SCROLL_RATIO = 0.35;
  const API_PATH = '/api/chatbot';
  const TURNSTILE_SRC = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
  const FALLBACK_LINKS = [
    { title: 'Analytics resume', url: '/resume-analytics', reason: 'Open the analytics-focused resume.' },
    { title: 'Portfolio', url: '/portfolio?audience=analytics', reason: 'Browse project examples.' },
    { title: 'Contact', url: '/contact', reason: 'Find email, LinkedIn, and message options.' }
  ];

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
    transcript: [],
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
  root.dataset.state = 'closed';
  root.dataset.enabled = 'pending';
  root.dataset.expanded = 'false';
  root.dataset.keyboard = 'false';
  root.dataset.nudge = 'false';
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
        <label class="visually-hidden" for="site-chatbot-message">Ask a question</label>
        <textarea id="site-chatbot-message" class="site-chatbot__input" name="message" rows="2" maxlength="1000" placeholder="Ask about portfolio work, resume, analytics, tourism, or contact details"></textarea>
        <input type="text" name="website" tabindex="-1" autocomplete="off" class="site-chatbot__website" aria-hidden="true">
        <button class="site-chatbot__send" type="submit" aria-label="Send question">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path d="m4 12 16-8-4.8 16-3.1-6.1L4 12Z"/>
            <path d="m12.1 13.9 7.9-9.9"/>
          </svg>
        </button>
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
  const closeButton = root.querySelector('.site-chatbot__close');
  const messages = root.querySelector('.site-chatbot__messages');
  const quickPrompts = root.querySelector('.site-chatbot__quick-prompts');
  const challenge = root.querySelector('.site-chatbot__challenge');
  const form = root.querySelector('.site-chatbot__form');
  const input = root.querySelector('.site-chatbot__input');
  const sendButton = root.querySelector('.site-chatbot__send');
  const status = root.querySelector('.site-chatbot__status');
  const footerOpeners = Array.from(document.querySelectorAll('[data-site-chatbot-open]'));
  let chromeOffsetFrame = 0;
  let nudgeCheckFrame = 0;
  setupChromeOffsetTracking();
  setupFooterOpeners();
  scheduleInitialNudge();

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
    if (event.target.closest('.site-chatbot__close, .site-chatbot__header-toggle, .site-chatbot__header-expand')) return;
    requestExpanded('header');
  });
  closeButton.addEventListener('click', () => setOpen(false));
  form.addEventListener('submit', handleSubmit);
  sendButton.addEventListener('click', (event) => {
    if (!state.sending) return;
    event.preventDefault();
    abortActiveRequest();
  });
  quickPrompts.addEventListener('click', (event) => {
    const button = event.target.closest('[data-chatbot-prompt]');
    if (!button || button.disabled) return;
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

  function getConversationId() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (/^[a-zA-Z0-9_-]{12,80}$/.test(stored || '')) return stored;
      const generated = window.crypto && window.crypto.randomUUID
        ? window.crypto.randomUUID().replace(/-/g, '')
        : `${Date.now()}${Math.random().toString(36).slice(2)}`;
      localStorage.setItem(STORAGE_KEY, generated);
      return generated;
    } catch {
      return `${Date.now()}${Math.random().toString(36).slice(2)}`;
    }
  }

  function getQuickPrompts() {
    if (pageId === 'project') {
      return [
        ['Summarize this project', 'Summarize this project'],
        ['Show similar projects', 'Show similar projects'],
        ['Where is the resume?', 'Where is the best resume?']
      ];
    }
    if (pageId === 'portfolio') {
      return [
        ['Relevant projects', 'Show relevant projects'],
        ['Best resume', 'Which resume fits this role?'],
        ['Contact', 'How do I contact Daniel?']
      ];
    }
    if (pageId === 'resume-analytics' || pageId === 'resume-data-science' || pageId === 'resume-tourism') {
      return [
        ['Summarize experience', "Summarize Daniel Short's experience"],
        ['Portfolio proof', 'Show portfolio proof for this resume'],
        ['Contact Daniel', 'How do I contact Daniel?']
      ];
    }
    if (pageId === 'analytics' || pageId === 'data-science' || pageId === 'tourism') {
      return [
        ['Relevant projects', `Show relevant ${pageId.replace('-', ' ')} projects`],
        ['Best resume', 'Which resume fits this role?'],
        ['Contact', 'How do I contact Daniel?']
      ];
    }
    return [
      ['Analytics projects', 'Show me analytics projects'],
      ['Resume', 'Where is the best resume?'],
      ['Contact', 'How do I contact Daniel?']
    ];
  }

  function renderQuickPromptButtons() {
    return getQuickPrompts().map(([label, prompt]) => (
      `<button type="button" data-chatbot-prompt="${escapeHtml(prompt)}">${escapeHtml(label)}</button>`
    )).join('\n        ');
  }

  function escapeHtml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
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
        appendAssistant('The assistant is not enabled right now. Use these shortcuts instead.', [], FALLBACK_LINKS);
        setDisabled(true);
        return;
      }
      setAvailability(true);
      appendAssistant('Ask me about Daniel Short, his projects, resumes, contact options, or where to go next on this site.');
    } catch {
      setAvailability(false);
      appendAssistant('The assistant API is not available from this view. Use these shortcuts instead.', [], FALLBACK_LINKS);
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
    const priorTranscript = state.transcript.slice(-8);
    state.pendingFollowupContext = null;
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
          title: document.title
        }
      };
      const data = await submitStreamingRequest(payload, assistant, message);
      if (!data) return;
      state.challengeToken = '';
      if (data.answer) rememberTurn('assistant', data.answer);
      setStatus('');
    } catch (err) {
      if (err && err.name === 'AbortError') {
        const partial = String(assistant.bubble.dataset.partialAnswer || '').trim();
        updateAssistantMessage(assistant, partial ? `${partial}\n\nStopped before final answer.` : 'Stopped before final answer.');
      } else {
        updateAssistantMessage(assistant, err && err.message ? err.message : 'The assistant could not answer right now.');
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
      return data;
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
    return finalData || { answer };
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

    if (role === 'assistant') renderAssistantExtras(item, sources, suggestedLinks, options.followups || [], options.previousQuestion || '');

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

    const navLinks = normalizeLinks(suggestedLinks, 5);
    const sourceLinks = normalizeLinks(sources, 5).filter((source) => !navLinks.some((link) => link.href === source.href));
    if (navLinks.length) {
      const navList = document.createElement('div');
      navList.className = 'site-chatbot__nav-links';
      navLinks.forEach((source) => navList.appendChild(makeLink(source)));
      item.appendChild(navList);
    }

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
          submitPromptText(text, {
            source: 'recommended_followup',
            prompt: text,
            previous_question: previousQuestion,
            previous_answer: String(answer || '').slice(0, 650),
            source_labels: [...navLinks, ...sourceLinks].map((source) => source.title).slice(0, 8),
            source_urls: [...navLinks, ...sourceLinks].map((source) => source.href).slice(0, 8)
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

  function rememberTurn(role, text) {
    const safeRole = role === 'assistant' ? 'assistant' : 'user';
    const safeText = String(text || '').replace(/\s+/g, ' ').trim();
    if (!safeText) return;
    state.transcript.push({ role: safeRole, text: safeText.slice(0, 700) });
    state.transcript = state.transcript.slice(-8);
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
    footerOpeners.forEach((button) => {
      button.hidden = !available;
    });
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

  function setupFooterOpeners() {
    footerOpeners.forEach((button) => {
      button.hidden = root.dataset.enabled !== 'true';
      button.addEventListener('click', () => {
        dismissNudge('footer');
        setOpen(true);
        trackChatbotEvent('chatbot_launcher_opened', { source: 'footer' });
      });
    });
  }

  function trackChatbotEvent(name, params = {}) {
    if (typeof window.gaEvent !== 'function') return;
    window.gaEvent(name, {
      page_id: pageId || 'unknown',
      ...params
    });
  }

  function setupChromeOffsetTracking() {
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
    mutationObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-consent-banner'],
      childList: true
    });

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
