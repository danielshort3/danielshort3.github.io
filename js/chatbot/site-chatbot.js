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
    conversationId: getConversationId(),
    challengeToken: '',
    retryTimer: 0,
    turnstileWidgetId: null
  };

  const root = document.createElement('div');
  root.className = 'site-chatbot';
  root.dataset.siteChatbot = '';
  root.dataset.state = 'closed';
  root.dataset.enabled = 'pending';
  root.dataset.expanded = 'false';
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
    <section class="site-chatbot__panel" id="site-chatbot-panel" role="dialog" aria-modal="false" aria-label="Ask Daniel's site assistant" aria-hidden="true" inert>
      <header class="site-chatbot__header">
        <button class="site-chatbot__header-toggle" type="button" aria-controls="site-chatbot-panel" aria-expanded="false" aria-label="Expand chat panel">
          <span class="site-chatbot__header-copy">
            <span class="site-chatbot__eyebrow">Navigation assistant</span>
            <span class="site-chatbot__title">Ask Daniel's site</span>
          </span>
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
        <button type="button" data-chatbot-prompt="Show me analytics projects">Analytics projects</button>
        <button type="button" data-chatbot-prompt="Where is the best resume?">Resume</button>
        <button type="button" data-chatbot-prompt="How do I contact Daniel?">Contact</button>
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
  const panel = root.querySelector('.site-chatbot__panel');
  const header = root.querySelector('.site-chatbot__header');
  const headerToggle = root.querySelector('.site-chatbot__header-toggle');
  const closeButton = root.querySelector('.site-chatbot__close');
  const messages = root.querySelector('.site-chatbot__messages');
  const quickPrompts = root.querySelector('.site-chatbot__quick-prompts');
  const challenge = root.querySelector('.site-chatbot__challenge');
  const form = root.querySelector('.site-chatbot__form');
  const input = root.querySelector('.site-chatbot__input');
  const sendButton = root.querySelector('.site-chatbot__send');
  const status = root.querySelector('.site-chatbot__status');
  setupChromeOffsetTracking();

  launcher.addEventListener('click', () => setOpen(!state.open));
  headerToggle.addEventListener('click', () => setExpanded(!state.expanded));
  header.addEventListener('click', (event) => {
    if (event.target.closest('.site-chatbot__close, .site-chatbot__header-toggle')) return;
    setExpanded(!state.expanded);
  });
  closeButton.addEventListener('click', () => setOpen(false));
  form.addEventListener('submit', handleSubmit);
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

  function setOpen(nextOpen) {
    state.open = Boolean(nextOpen);
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
      }, 40);
    }
  }

  function setExpanded(nextExpanded) {
    state.expanded = Boolean(nextExpanded);
    root.dataset.expanded = state.expanded ? 'true' : 'false';
    headerToggle.setAttribute('aria-expanded', String(state.expanded));
    headerToggle.setAttribute('aria-label', state.expanded ? 'Collapse chat panel' : 'Expand chat panel');
    updateChromeOffset();
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
    if (state.sending) return;

    const formData = new FormData(form);
    const website = String(formData.get('website') || '').trim();
    const message = String(formData.get('message') || '').trim();
    if (website || !message) return;

    appendUser(message);
    input.value = '';
    setSending(true);
    setStatus('Thinking...');
    clearChallenge();

    try {
      const payload = {
        message,
        conversationId: state.conversationId,
        challengeToken: state.challengeToken,
        website: '',
        pageContext: {
          url: window.location.href,
          title: document.title
        }
      };
      const res = await fetch(API_PATH, {
        method: 'POST',
        headers: {
          accept: 'application/json',
          'content-type': 'application/json',
          'x-chatbot-session': state.conversationId
        },
        credentials: 'same-origin',
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => null);
      if (res.status === 429 && data) {
        handleRateLimit(data, message);
        return;
      }
      if (!res.ok || !data || data.ok === false) {
        throw new Error(data && data.error ? data.error : 'The assistant could not answer right now.');
      }
      state.challengeToken = '';
      appendAssistant(data.answer || 'I could not find a supported answer.', data.sources || [], data.suggestedLinks || []);
      setStatus('');
    } catch (err) {
      appendAssistant(err && err.message ? err.message : 'The assistant could not answer right now.');
      setStatus('');
    } finally {
      setSending(false);
    }
  }

  function handleRateLimit(data, lastMessage) {
    const seconds = Math.max(1, Number(data.retryAfter) || 20);
    appendAssistant(limitMessage(data, seconds));
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
    appendMessage('user', text);
  }

  function appendAssistant(text, sources = [], suggestedLinks = []) {
    appendMessage('assistant', text, sources, suggestedLinks);
  }

  function appendMessage(role, text, sources = [], suggestedLinks = []) {
    const item = document.createElement('article');
    item.className = `site-chatbot__message site-chatbot__message--${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'site-chatbot__bubble';
    textToParagraphs(text).forEach((paragraph) => {
      const p = document.createElement('p');
      p.textContent = paragraph;
      bubble.appendChild(p);
    });
    item.appendChild(bubble);

    const navLinks = normalizeLinks(suggestedLinks);
    if (navLinks.length) {
      const navList = document.createElement('div');
      navList.className = 'site-chatbot__nav-links';
      navLinks.forEach((source) => {
        const link = document.createElement('a');
        link.href = source.href;
        link.textContent = source.title;
        link.title = source.reason || source.title;
        link.rel = 'noopener noreferrer';
        navList.appendChild(link);
      });
      item.appendChild(navList);
    }

    const links = normalizeLinks(sources);
    if (links.length) {
      const sourceList = document.createElement('details');
      sourceList.className = 'site-chatbot__sources';
      const summary = document.createElement('summary');
      summary.textContent = `Sources (${links.length})`;
      const sourcePills = document.createElement('div');
      sourcePills.className = 'site-chatbot__source-pills';
      links.forEach((source) => {
        const link = document.createElement('a');
        link.href = source.href;
        link.textContent = source.title;
        link.title = source.reason || source.title;
        link.rel = 'noopener noreferrer';
        sourcePills.appendChild(link);
      });
      sourceList.append(summary, sourcePills);
      item.appendChild(sourceList);
    }

    messages.appendChild(item);
    messages.scrollTop = messages.scrollHeight;
  }

  function textToParagraphs(value) {
    const text = String(value || '').trim();
    if (!text) return ['I could not find a supported answer.'];
    return text.split(/\n{2,}/).map((item) => item.trim()).filter(Boolean).slice(0, 5);
  }

  function normalizeLinks(sources) {
    const seen = new Set();
    return (Array.isArray(sources) ? sources : [])
      .map((source) => {
        const rawUrl = source && source.url ? String(source.url) : '';
        if (!rawUrl) return null;
        const href = new URL(rawUrl, window.location.origin).href;
        const title = String(source.title || rawUrl).trim();
        const reason = String(source.reason || '').trim();
        return { href, title: title.slice(0, 80), reason };
      })
      .filter((source) => {
        if (!source || seen.has(source.href)) return false;
        seen.add(source.href);
        return true;
      })
      .slice(0, 4);
  }

  function setSending(nextSending) {
    state.sending = Boolean(nextSending);
    root.dataset.busy = state.sending ? 'true' : 'false';
    syncControls();
  }

  function setDisabled(disabled) {
    state.locked = Boolean(disabled);
    syncControls();
  }

  function setAvailability(available) {
    root.dataset.enabled = available ? 'true' : 'false';
  }

  function syncControls() {
    const disabled = state.sending || state.locked;
    sendButton.disabled = disabled;
    input.disabled = disabled;
    quickPrompts.querySelectorAll('button').forEach((button) => {
      button.disabled = disabled;
    });
  }

  function setStatus(text) {
    status.textContent = text || '';
  }

  function submitPromptText(text) {
    const prompt = String(text || '').trim();
    if (!prompt || state.sending || state.locked) return;
    input.value = prompt;
    form.requestSubmit();
  }

  function setupChromeOffsetTracking() {
    updateChromeOffset();
    window.addEventListener('resize', updateChromeOffset, { passive: true });
    window.addEventListener('orientationchange', updateChromeOffset, { passive: true });

    const mutationObserver = new MutationObserver(() => {
      observeConsentBanner();
      updateChromeOffset();
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
    window.setTimeout(updateChromeOffset, 80);
    window.setTimeout(updateChromeOffset, 480);
  }

  function updateChromeOffset() {
    const banner = document.getElementById('pcz-banner');
    const bannerIsOpen = document.body.dataset.consentBanner === 'open';
    const bannerRect = banner && bannerIsOpen ? banner.getBoundingClientRect() : null;
    const offset = bannerRect && bannerRect.height > 0 ? Math.ceil(bannerRect.height + 10) : 0;
    root.style.setProperty('--site-chatbot-consent-offset', `${offset}px`);
  }
})();
