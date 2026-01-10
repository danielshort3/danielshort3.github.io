(() => {
  'use strict';

  const ACTIVE_SESSION_PREFIX = 'toolsActiveSession:';
  const AUTO_SAVE_MS = 20 * 1000;

  const TOOL_CATALOG = {
    'word-frequency': { name: 'Word Frequency Analyzer', href: '/tools/word-frequency' },
    'text-compare': { name: 'Text Compare', href: '/tools/text-compare' },
    'point-of-view-checker': { name: 'Point of View Checker', href: '/tools/point-of-view-checker' },
    'oxford-comma-checker': { name: 'Oxford Comma Checker', href: '/tools/oxford-comma-checker' },
    'background-remover': { name: 'Background Remover', href: '/tools/background-remover' },
    'nbsp-cleaner': { name: 'Non-breaking Space Cleaner', href: '/tools/nbsp-cleaner' },
    'ocean-wave-simulation': { name: 'Ocean Wave Simulation', href: '/tools/ocean-wave-simulation' },
    'qr-code-generator': { name: 'QR Code Generator', href: '/tools/qr-code-generator' },
    'image-optimizer': { name: 'Image Optimizer', href: '/tools/image-optimizer' },
    'screen-recorder': { name: 'Screen Recorder', href: '/tools/screen-recorder' },
    'job-application-tracker': { name: 'Job Application Tracker', href: '/tools/job-application-tracker' },
    'short-links': { name: 'Short Links', href: '/tools/short-links' },
    'utm-batch-builder': { name: 'UTM Batch Builder', href: '/tools/utm-batch-builder' },
    'whisper-transcribe-monitor': { name: 'Whisper Capacity Monitor', href: '/tools/whisper-transcribe-monitor' }
  };

  const $ = (sel, root = document) => root.querySelector(sel);

  const getToolInfo = (toolId) => {
    const known = TOOL_CATALOG[toolId];
    if (known) return known;
    return { name: toolId || 'Tool', href: toolId ? `/tools/${toolId}` : '/tools' };
  };

  const formatTime = (ts) => {
    const numeric = Number(ts) || 0;
    if (!numeric) return '';
    try {
      return new Date(numeric).toLocaleString();
    } catch {
      return '';
    }
  };

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const getSessionParam = () => {
    try {
      const url = new URL(window.location.href);
      return url.searchParams.get('session') || '';
    } catch {
      return '';
    }
  };

  const setSessionParam = (sessionId) => {
    try {
      const url = new URL(window.location.href);
      if (sessionId) {
        url.searchParams.set('session', sessionId);
      } else {
        url.searchParams.delete('session');
      }
      const query = url.searchParams.toString();
      const next = query ? `${url.pathname}?${query}${url.hash}` : `${url.pathname}${url.hash}`;
      window.history.replaceState({}, document.title, next);
    } catch {}
  };

  const getActiveSessionId = (toolId) => {
    if (!toolId) return '';
    try {
      return (localStorage.getItem(`${ACTIVE_SESSION_PREFIX}${toolId}`) || '').trim();
    } catch {
      return '';
    }
  };

  const setActiveSessionId = (toolId, sessionId) => {
    if (!toolId) return;
    try {
      if (sessionId) {
        localStorage.setItem(`${ACTIVE_SESSION_PREFIX}${toolId}`, sessionId);
      } else {
        localStorage.removeItem(`${ACTIVE_SESSION_PREFIX}${toolId}`);
      }
    } catch {}
  };

  const dispatchValueEvents = (el) => {
    if (!el) return;
    try { el.dispatchEvent(new Event('input', { bubbles: true })); } catch {}
    try { el.dispatchEvent(new Event('change', { bubbles: true })); } catch {}
  };

  const serializeToolFields = (root) => {
    const fields = {};
    if (!root) return fields;

    const elements = [...root.querySelectorAll('input, textarea, select')];
    const seenRadioNames = new Set();

    elements.forEach((el) => {
      if (!el || el.disabled) return;
      const tag = String(el.tagName || '').toLowerCase();
      const type = tag === 'input' ? String(el.type || '').toLowerCase() : '';

      if (tag === 'input') {
        if (['file', 'password', 'button', 'submit', 'reset', 'image', 'hidden'].includes(type)) return;
      }

      const key = (type === 'radio')
        ? String(el.name || '').trim()
        : String(el.id || el.name || '').trim();
      if (!key) return;

      if (type === 'radio') {
        if (seenRadioNames.has(key)) return;
        seenRadioNames.add(key);
        const checked = root.querySelector(`input[type="radio"][name="${CSS.escape(key)}"]:checked`);
        if (!checked) return;
        fields[key] = { kind: 'radio', value: String(checked.value || '').trim() };
        return;
      }

      if (type === 'checkbox') {
        fields[key] = { kind: 'checkbox', checked: !!el.checked, value: String(el.value || '') };
        return;
      }

      if (tag === 'select' && el.multiple) {
        const values = [...el.options].filter(opt => opt.selected).map(opt => String(opt.value || ''));
        fields[key] = { kind: 'multi', values };
        return;
      }

      fields[key] = { kind: 'value', value: String(el.value || '') };
    });

    return fields;
  };

  const applyToolFields = (root, fields) => {
    if (!root || !fields || typeof fields !== 'object') return;
    Object.entries(fields).forEach(([key, payload]) => {
      if (!key || !payload || typeof payload !== 'object') return;
      const kind = String(payload.kind || '').trim();

      if (kind === 'radio') {
        const value = String(payload.value || '');
        const radios = [...root.querySelectorAll(`input[type="radio"][name="${CSS.escape(key)}"]`)];
        radios.forEach((radio) => {
          radio.checked = String(radio.value || '') === value;
          dispatchValueEvents(radio);
        });
        return;
      }

      const byId = root.querySelector(`#${CSS.escape(key)}`);
      const target = byId || root.querySelector(`[name="${CSS.escape(key)}"]`);
      if (!target) return;

      const type = String(target.type || '').toLowerCase();
      if (kind === 'checkbox' && type === 'checkbox') {
        target.checked = !!payload.checked;
        dispatchValueEvents(target);
        return;
      }

      if (kind === 'multi' && String(target.tagName || '').toLowerCase() === 'select' && target.multiple) {
        const values = Array.isArray(payload.values) ? payload.values.map(v => String(v)) : [];
        [...target.options].forEach((opt) => {
          opt.selected = values.includes(String(opt.value || ''));
        });
        dispatchValueEvents(target);
        return;
      }

      if (kind === 'value' && !['file', 'password'].includes(type)) {
        target.value = String(payload.value || '');
        dispatchValueEvents(target);
      }
    });
  };

  const buildSnapshot = ({ toolId, root }) => ({
    version: 1,
    toolId,
    capturedAt: Date.now(),
    fields: serializeToolFields(root)
  });

  const renderAccountBar = ({ barEl, toolId, sessionId, statusText, toolActionsEnabled } = {}) => {
    if (!barEl) return;
    const auth = window.ToolsAuth.getAuth();
    const authed = window.ToolsAuth.authIsValid(auth);
    const user = authed ? window.ToolsAuth.getUser(auth) : { email: '', name: '', sub: '' };
    const label = authed
      ? `Signed in${user.email ? ` as ${user.email}` : ''}`
      : 'Not signed in';
    const pillClass = authed ? 'tools-account-pill is-authed' : 'tools-account-pill';
    const toolInfo = toolId ? getToolInfo(toolId) : null;

    const dashboardLink = `<a class="btn-secondary" href="tools/dashboard">Dashboard</a>`;
    const accountButton = `<button type="button" class="btn-secondary" data-tools-action="open-account">Account</button>`;
    const signInButton = `<button type="button" class="btn-primary" data-tools-action="sign-in">Sign in</button>`;
    const signOutButton = `<button type="button" class="btn-ghost" data-tools-action="sign-out">Sign out</button>`;
    const allowToolActions = !!(toolId && toolActionsEnabled !== false);
    const saveButton = allowToolActions ? `<button type="button" class="btn-secondary" data-tools-action="save-session">Save session</button>` : '';
    const newButton = allowToolActions ? `<button type="button" class="btn-ghost" data-tools-action="new-session">New session</button>` : '';

    const sessionLine = allowToolActions
      ? `<span class="tools-account-status">${escapeHtml(toolInfo?.name || toolId)}${sessionId ? ` · Session ${escapeHtml(sessionId.slice(0, 10))}…` : ''}${statusText ? ` · ${escapeHtml(statusText)}` : ''}</span>`
      : (statusText ? `<span class="tools-account-status">${escapeHtml(statusText)}</span>` : '');

    barEl.innerHTML = `
      <span class="${pillClass}" aria-label="${escapeHtml(label)}">${escapeHtml(label)}</span>
      ${accountButton}
      ${authed ? dashboardLink : ''}
      ${allowToolActions && authed ? saveButton : ''}
      ${allowToolActions && authed ? newButton : ''}
      ${authed ? signOutButton : signInButton}
      ${sessionLine}
    `.trim();
  };

  const initAccountModal = () => {
    const modalEl = document.createElement('div');
    modalEl.className = 'modal tools-account-modal';
    modalEl.id = 'tools-account-modal';
    modalEl.setAttribute('data-tools-account', 'modal');
    modalEl.setAttribute('aria-hidden', 'true');
    modalEl.innerHTML = `
      <div class="modal-content modal-wide" role="dialog" aria-modal="true" tabindex="-1" aria-labelledby="tools-account-modal-title">
        <button type="button" class="modal-close" aria-label="Close dialog" data-tools-account-action="close">&times;</button>
        <div class="modal-title-strip">
          <h3 class="modal-title" id="tools-account-modal-title">Account</h3>
          <p class="modal-subtitle" data-tools-account="modal-subtitle">View your signed-in history and saved sessions across tools.</p>
        </div>
        <div class="modal-body stacked">
          <div class="tools-account-modal-actions" data-tools-account="modal-actions"></div>
          <div class="tools-account-modal-status" data-tools-account="modal-status" role="status" aria-live="polite"></div>
          <div class="tools-dashboard-grid tools-account-modal-grid" data-tools-account="modal-grid"></div>
        </div>
      </div>
    `.trim();
    document.body.appendChild(modalEl);

    const contentEl = modalEl.querySelector('.modal-content');
    const actionsEl = modalEl.querySelector('[data-tools-account="modal-actions"]');
    const statusEl = modalEl.querySelector('[data-tools-account="modal-status"]');
    const gridEl = modalEl.querySelector('[data-tools-account="modal-grid"]');

    let handlers = {
      signIn: () => {},
      signOut: () => {},
    };

    const setStatus = (message) => {
      if (!statusEl) return;
      statusEl.textContent = String(message || '').trim();
    };

    const renderSignedOut = () => {
      if (actionsEl) {
        actionsEl.innerHTML = `
          <button type="button" class="btn-primary" data-tools-account-action="sign-in">Sign in</button>
          <a class="btn-secondary" href="tools/dashboard">Open dashboard</a>
        `.trim();
      }
      if (gridEl) {
        gridEl.innerHTML = `
          <section class="tools-dashboard-card" aria-labelledby="tools-account-modal-signed-out">
            <header class="tools-dashboard-card-head">
              <h2 id="tools-account-modal-signed-out">Sign in to view your history</h2>
              <p class="tools-dashboard-subtitle">Once signed in, your sessions and tool activity appear here and on the dashboard.</p>
            </header>
            <p class="tools-dashboard-empty">You can still use most tools without signing in; history and saved sessions require an account.</p>
          </section>
        `.trim();
      }
    };

    const renderLoading = ({ user }) => {
      if (!actionsEl) return;
      actionsEl.innerHTML = `
        <button type="button" class="btn-secondary" data-tools-account-action="refresh">Refresh</button>
        <a class="btn-secondary" href="tools/dashboard">Open dashboard</a>
        <button type="button" class="btn-ghost" data-tools-account-action="sign-out">Sign out</button>
      `.trim();

      if (gridEl) {
        const email = escapeHtml(user?.email || '');
        gridEl.innerHTML = `
          <section class="tools-dashboard-card" aria-labelledby="tools-account-modal-account">
            <header class="tools-dashboard-card-head">
              <h2 id="tools-account-modal-account">Account</h2>
              <p class="tools-dashboard-subtitle">${email ? `Signed in as ${email}.` : 'Signed in.'}</p>
            </header>
            <p class="tools-dashboard-empty">Loading history…</p>
          </section>
        `.trim();
      }
    };

    const renderDashboardData = ({ user, data }) => {
      const email = escapeHtml(user?.email || '');
      const name = escapeHtml(user?.name || '');
      const sub = escapeHtml(user?.sub || '');

      const tools = Array.isArray(data?.tools) ? data.tools : [];
      const recentSessions = Array.isArray(data?.recentSessions) ? data.recentSessions : [];
      const recentActivity = Array.isArray(data?.recentActivity) ? data.recentActivity : [];

      const toolsMarkup = (() => {
        if (!tools.length) return '<p class="tools-dashboard-empty">No signed-in tool activity yet.</p>';
        return tools.slice(0, 12).map((entry) => {
          const toolId = String(entry?.toolId || '').trim();
          const meta = entry?.meta || {};
          const info = getToolInfo(toolId);
          const lastUsed = meta?.lastUsedAt ? formatTime(meta.lastUsedAt) : '';
          const sessionCount = Number(meta?.sessionCount) || 0;
          const activityCount = Number(meta?.activityCount) || 0;
          return `
            <div class="tools-dashboard-item">
              <div>
                <p class="tools-dashboard-item-title"><a href="${escapeHtml(info.href)}">${escapeHtml(info.name)}</a></p>
                <p class="tools-dashboard-item-meta">${lastUsed ? `Last used ${escapeHtml(lastUsed)} · ` : ''}${sessionCount} sessions · ${activityCount} events</p>
              </div>
              <div class="tools-dashboard-item-actions">
                <a class="btn-secondary" href="${escapeHtml(info.href)}">Open</a>
              </div>
            </div>
          `.trim();
        }).join('');
      })();

      const sessionsMarkup = (() => {
        if (!recentSessions.length) return '<p class="tools-dashboard-empty">No saved sessions yet.</p>';
        return recentSessions.slice(0, 10).map((session) => {
          const toolId = String(session?.toolId || '').trim();
          const sessionId = String(session?.sessionId || '').trim();
          const info = getToolInfo(toolId);
          const updated = session?.updatedAt ? formatTime(session.updatedAt) : '';
          const summary = String(session?.outputSummary || '').trim();
          const href = `${info.href}?session=${encodeURIComponent(sessionId)}`;
          return `
            <div class="tools-dashboard-item">
              <div>
                <p class="tools-dashboard-item-title"><a href="${escapeHtml(href)}">${escapeHtml(info.name)}</a></p>
                <p class="tools-dashboard-item-meta">${updated ? `Updated ${escapeHtml(updated)} · ` : ''}${summary ? escapeHtml(summary) : `Session ${escapeHtml(sessionId.slice(0, 10))}…`}</p>
              </div>
              <div class="tools-dashboard-item-actions">
                <a class="btn-secondary" href="${escapeHtml(href)}">Reopen</a>
              </div>
            </div>
          `.trim();
        }).join('');
      })();

      const activityMarkup = (() => {
        if (!recentActivity.length) return '<p class="tools-dashboard-empty">No activity yet.</p>';
        return recentActivity.slice(0, 16).map((event) => {
          const toolId = String(event?.toolId || '').trim();
          const info = getToolInfo(toolId);
          const ts = event?.ts ? formatTime(event.ts) : '';
          const type = String(event?.type || '').trim();
          const summary = String(event?.summary || '').trim();
          return `
            <div class="tools-dashboard-item">
              <div>
                <p class="tools-dashboard-item-title">${escapeHtml(info.name)}${type ? ` · ${escapeHtml(type)}` : ''}</p>
                <p class="tools-dashboard-item-meta">${ts ? `${escapeHtml(ts)} · ` : ''}${summary ? escapeHtml(summary) : ''}</p>
              </div>
              <div class="tools-dashboard-item-actions">
                <a class="btn-secondary" href="tools/dashboard">View</a>
              </div>
            </div>
          `.trim();
        }).join('');
      })();

      if (actionsEl) {
        actionsEl.innerHTML = `
          <button type="button" class="btn-secondary" data-tools-account-action="refresh">Refresh</button>
          <a class="btn-secondary" href="tools/dashboard">Open dashboard</a>
          <button type="button" class="btn-ghost" data-tools-account-action="sign-out">Sign out</button>
        `.trim();
      }

      if (gridEl) {
        gridEl.innerHTML = `
          <section class="tools-dashboard-card" aria-labelledby="tools-account-modal-account">
            <header class="tools-dashboard-card-head">
              <h2 id="tools-account-modal-account">Account</h2>
              <p class="tools-dashboard-subtitle">Your tools history is private to this signed-in account.</p>
            </header>
            <dl class="tools-account-modal-meta">
              ${email ? `<div class="tools-account-modal-meta-row"><dt>Email</dt><dd>${email}</dd></div>` : ''}
              ${name ? `<div class="tools-account-modal-meta-row"><dt>Name</dt><dd>${name}</dd></div>` : ''}
              ${sub ? `<div class="tools-account-modal-meta-row"><dt>User ID</dt><dd><code>${sub}</code></dd></div>` : ''}
            </dl>
          </section>
          <section class="tools-dashboard-card" aria-labelledby="tools-account-modal-tools">
            <header class="tools-dashboard-card-head">
              <h2 id="tools-account-modal-tools">Tools used</h2>
              <p class="tools-dashboard-subtitle">Tools you have opened while signed in.</p>
            </header>
            <div class="tools-dashboard-list">${toolsMarkup}</div>
          </section>
          <section class="tools-dashboard-card" aria-labelledby="tools-account-modal-sessions">
            <header class="tools-dashboard-card-head">
              <h2 id="tools-account-modal-sessions">Recent sessions</h2>
              <p class="tools-dashboard-subtitle">Pick up where you left off.</p>
            </header>
            <div class="tools-dashboard-list">${sessionsMarkup}</div>
          </section>
          <section class="tools-dashboard-card" aria-labelledby="tools-account-modal-activity">
            <header class="tools-dashboard-card-head">
              <h2 id="tools-account-modal-activity">Recent activity</h2>
              <p class="tools-dashboard-subtitle">Timestamped events across tools.</p>
            </header>
            <div class="tools-dashboard-list">${activityMarkup}</div>
          </section>
        `.trim();
      }
    };

    const refresh = async () => {
      if (!window.ToolsAuth || !window.ToolsAuth.getAuth || !window.ToolsAuth.authIsValid) {
        setStatus('Account system is unavailable on this page.');
        renderSignedOut();
        return;
      }

      const auth = window.ToolsAuth.getAuth();
      const authed = window.ToolsAuth.authIsValid(auth);
      if (!authed) {
        setStatus('');
        renderSignedOut();
        return;
      }

      const user = window.ToolsAuth.getUser(auth);
      setStatus('');
      renderLoading({ user });

      if (!window.ToolsState || !window.ToolsState.getDashboard) {
        setStatus('Dashboard API is unavailable on this page.');
        return;
      }

      setStatus('Loading history...');
      let data;
      try {
        data = await window.ToolsState.getDashboard();
      } catch (err) {
        setStatus(err?.message || 'Unable to load history.');
        return;
      }
      setStatus('');
      renderDashboardData({ user, data });
    };

    const open = () => {
      modalEl.classList.add('active');
      modalEl.setAttribute('aria-hidden', 'false');
      document.body.classList.add('modal-open');
      try {
        contentEl?.focus();
      } catch {}
      refresh().catch(() => {});
    };

    const close = () => {
      modalEl.classList.remove('active');
      modalEl.setAttribute('aria-hidden', 'true');
      document.body.classList.remove('modal-open');
      setStatus('');
    };

    modalEl.addEventListener('click', (event) => {
      if (event.target === modalEl) close();
      const actionEl = event.target.closest('[data-tools-account-action]');
      if (!actionEl) return;
      const action = String(actionEl.dataset.toolsAccountAction || '').trim();
      if (action === 'close') {
        close();
      } else if (action === 'refresh') {
        refresh().catch(() => {});
      } else if (action === 'sign-in') {
        handlers.signIn();
      } else if (action === 'sign-out') {
        handlers.signOut();
      }
    });

    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (modalEl.classList.contains('active')) close();
    });

    return {
      open,
      close,
      refresh,
      setHandlers: (nextHandlers = {}) => {
        handlers = { ...handlers, ...nextHandlers };
      }
    };
  };

  const initAccountBar = ({ toolId, root, toolActionsEnabled, onOpenAccount } = {}) => {
    const barEl = (() => {
      const existing = $('[data-tools-account="bar"]');
      if (!existing) return null;
      const inDock = existing.closest('[data-tools-account="dock"]');
      if (inDock) return existing;

      const headerHost = document.querySelector('#combined-header-nav');
      const main = document.querySelector('main') || document.querySelector('#main');
      let dock = document.querySelector('[data-tools-account="dock"]');
      if (!dock) {
        dock = document.createElement('div');
        dock.className = 'tools-account-dock';
        dock.setAttribute('data-tools-account', 'dock');
        dock.innerHTML = '<div class="wrapper tools-account-dock-inner" data-tools-account="dock-inner"></div>';
        const hero = document.querySelector('.tools-hero');
        if (hero && hero.insertAdjacentElement) {
          hero.insertAdjacentElement('afterend', dock);
        } else if (headerHost && headerHost.insertAdjacentElement) {
          headerHost.insertAdjacentElement('afterend', dock);
        } else if (main && main.parentNode) {
          main.parentNode.insertBefore(dock, main);
        } else {
          document.body.insertBefore(dock, document.body.firstChild);
        }
      }

      const dockInner = dock.querySelector('[data-tools-account="dock-inner"]');
      if (dockInner) {
        dockInner.appendChild(existing);
      }

      return existing;
    })();
    if (!barEl) return { barEl: null, setStatus: () => {} };

    let sessionId = '';
    let statusText = '';

    const setStatus = (nextStatus, nextSessionId) => {
      if (typeof nextSessionId === 'string') sessionId = nextSessionId;
      statusText = String(nextStatus || '').trim();
      renderAccountBar({ barEl, toolId, sessionId, statusText, toolActionsEnabled });
    };

    renderAccountBar({ barEl, toolId, sessionId, statusText, toolActionsEnabled });

    barEl.addEventListener('click', (event) => {
      const button = event.target.closest('[data-tools-action]');
      if (!button) return;
      const action = button.dataset.toolsAction;
      if (action === 'sign-in') {
        window.ToolsAuth.signIn({ returnTo: `${window.location.pathname}${window.location.search}${window.location.hash}` })
          .catch((err) => setStatus(err?.message || 'Unable to start sign-in.'));
      } else if (action === 'sign-out') {
        window.ToolsAuth.signOut();
        setStatus('Signed out.');
        try {
          document.dispatchEvent(new CustomEvent('tools:auth-changed', { detail: { source: 'tools-account-ui' } }));
        } catch {}
      } else if (action === 'new-session') {
        setActiveSessionId(toolId, '');
        setSessionParam('');
        setStatus('New session (not saved yet).', '');
      } else if (action === 'save-session') {
        document.dispatchEvent(new CustomEvent('tools:save-session', { detail: { toolId } }));
      } else if (action === 'open-account') {
        if (typeof onOpenAccount === 'function') onOpenAccount();
      }
    });

    document.addEventListener('tools:auth-changed', (event) => {
      const source = String(event?.detail?.source || '').trim();
      if (source && source.startsWith('tools-account')) return;
      statusText = '';
      renderAccountBar({ barEl, toolId, sessionId, statusText, toolActionsEnabled });
    });

    return { barEl, setStatus };
  };

  const initDashboard = async ({ setStatus }) => {
    const statusEl = $('[data-tools-dashboard="status"]');
    const toolsEl = $('[data-tools-dashboard="tools"]');
    const sessionsEl = $('[data-tools-dashboard="sessions"]');
    const activityEl = $('[data-tools-dashboard="activity"]');

    const setDashboardStatus = (message) => {
      if (statusEl) statusEl.textContent = message || '';
      if (setStatus) setStatus(message || '');
    };

    const clearLists = () => {
      if (toolsEl) toolsEl.innerHTML = '';
      if (sessionsEl) sessionsEl.innerHTML = '';
      if (activityEl) activityEl.innerHTML = '';
    };

    const auth = window.ToolsAuth.getAuth();
    if (!window.ToolsAuth.authIsValid(auth)) {
      clearLists();
      setDashboardStatus('Sign in to see your saved sessions and activity.');
      return;
    }

    setDashboardStatus('Loading dashboard...');
    let data;
    try {
      data = await window.ToolsState.getDashboard();
    } catch (err) {
      clearLists();
      setDashboardStatus(err?.message || 'Unable to load dashboard.');
      return;
    }

    setDashboardStatus('');

    const tools = Array.isArray(data?.tools) ? data.tools : [];
    if (toolsEl) {
      if (!tools.length) {
        toolsEl.innerHTML = '<p class="tools-dashboard-empty">No signed-in tool activity yet.</p>';
      } else {
        toolsEl.innerHTML = tools.map((entry) => {
          const toolId = String(entry?.toolId || '').trim();
          const meta = entry?.meta || {};
          const info = getToolInfo(toolId);
          const lastUsed = meta?.lastUsedAt ? formatTime(meta.lastUsedAt) : '';
          const sessionCount = Number(meta?.sessionCount) || 0;
          const activityCount = Number(meta?.activityCount) || 0;
          return `
            <div class="tools-dashboard-item">
              <div>
                <p class="tools-dashboard-item-title"><a href="${escapeHtml(info.href)}">${escapeHtml(info.name)}</a></p>
                <p class="tools-dashboard-item-meta">${lastUsed ? `Last used ${escapeHtml(lastUsed)} · ` : ''}${sessionCount} sessions · ${activityCount} events</p>
              </div>
              <div class="tools-dashboard-item-actions">
                <a class="btn-secondary" href="${escapeHtml(info.href)}">Open</a>
              </div>
            </div>
          `.trim();
        }).join('');
      }
    }

    const sessions = Array.isArray(data?.recentSessions) ? data.recentSessions : [];
    if (sessionsEl) {
      if (!sessions.length) {
        sessionsEl.innerHTML = '<p class="tools-dashboard-empty">No saved sessions yet.</p>';
      } else {
        sessionsEl.innerHTML = sessions.map((session) => {
          const toolId = String(session?.toolId || '').trim();
          const sessionId = String(session?.sessionId || '').trim();
          const info = getToolInfo(toolId);
          const updated = session?.updatedAt ? formatTime(session.updatedAt) : '';
          const summary = String(session?.outputSummary || '').trim();
          const href = `${info.href}?session=${encodeURIComponent(sessionId)}`;
          return `
            <div class="tools-dashboard-item" data-session-tool="${escapeHtml(toolId)}" data-session-id="${escapeHtml(sessionId)}">
              <div>
                <p class="tools-dashboard-item-title"><a href="${escapeHtml(href)}">${escapeHtml(info.name)}</a></p>
                <p class="tools-dashboard-item-meta">${updated ? `Updated ${escapeHtml(updated)} · ` : ''}${summary ? escapeHtml(summary) : `Session ${escapeHtml(sessionId.slice(0, 10))}…`}</p>
              </div>
              <div class="tools-dashboard-item-actions">
                <a class="btn-secondary" href="${escapeHtml(href)}">Reopen</a>
                <button type="button" class="btn-ghost" data-tools-action="delete-session">Delete</button>
              </div>
            </div>
          `.trim();
        }).join('');
      }
    }

    const events = Array.isArray(data?.recentActivity) ? data.recentActivity : [];
    if (activityEl) {
      if (!events.length) {
        activityEl.innerHTML = '<p class="tools-dashboard-empty">No activity yet.</p>';
      } else {
        activityEl.innerHTML = events.slice(0, 50).map((event) => {
          const toolId = String(event?.toolId || '').trim();
          const info = getToolInfo(toolId);
          const ts = event?.ts ? formatTime(event.ts) : '';
          const type = String(event?.type || '').trim();
          const summary = String(event?.summary || '').trim();
          return `
            <div class="tools-dashboard-item">
              <div>
                <p class="tools-dashboard-item-title">${escapeHtml(info.name)}${type ? ` · ${escapeHtml(type)}` : ''}</p>
                <p class="tools-dashboard-item-meta">${ts ? `${escapeHtml(ts)} · ` : ''}${summary ? escapeHtml(summary) : ''}</p>
              </div>
              <div class="tools-dashboard-item-actions">
                <a class="btn-secondary" href="${escapeHtml(info.href)}">Open</a>
              </div>
            </div>
          `.trim();
        }).join('');
      }
    }

    if (sessionsEl) {
      sessionsEl.addEventListener('click', async (event) => {
        const button = event.target.closest('[data-tools-action="delete-session"]');
        if (!button) return;
        const row = button.closest('[data-session-tool][data-session-id]');
        if (!row) return;
        const toolId = String(row.dataset.sessionTool || '').trim();
        const sessionId = String(row.dataset.sessionId || '').trim();
        if (!toolId || !sessionId) return;

        const ok = window.confirm('Delete this saved session? This cannot be undone.');
        if (!ok) return;

        try {
          await window.ToolsState.deleteSession({ toolId, sessionId });
          row.remove();
          setDashboardStatus('Session deleted.');
          setTimeout(() => setDashboardStatus(''), 1600);
        } catch (err) {
          setDashboardStatus(err?.message || 'Unable to delete session.');
        }
      });
    }
  };

  const initToolAutoSave = ({ toolId, root, setStatus }) => {
    if (!toolId || !root) return;

    const sessionIdFromUrl = getSessionParam() || '';
    const authed = window.ToolsAuth.authIsValid(window.ToolsAuth.getAuth());
    let sessionId = sessionIdFromUrl || (authed ? getActiveSessionId(toolId) : '') || '';
    let dirty = false;
    let saveInFlight = false;

    const updateStatus = (message) => {
      if (setStatus) setStatus(message, sessionId);
    };

    if (!authed) {
      if (sessionIdFromUrl) {
        updateStatus('Sign in to load this saved session.');
      }
      return;
    }

    const applySession = async () => {
      const auth = window.ToolsAuth.getAuth();
      if (!window.ToolsAuth.authIsValid(auth) || !sessionId) return;
      updateStatus('Loading session...');
      try {
        const data = await window.ToolsState.getSession({ toolId, sessionId });
        const snapshot = data?.session?.snapshot;
        if (snapshot && typeof snapshot === 'object') {
          applyToolFields(root, snapshot.fields || {});
          updateStatus('Session loaded.');
          setTimeout(() => updateStatus(''), 1500);
          try {
            await window.ToolsState.logActivity({ toolId, type: 'session_load', summary: `Loaded session ${sessionId}` });
          } catch {}
        }
      } catch {
        updateStatus('Unable to load session.');
      }
    };

    const saveSession = async ({ keepalive } = {}) => {
      const auth = window.ToolsAuth.getAuth();
      if (!window.ToolsAuth.authIsValid(auth)) return;
      if (saveInFlight) return;
      saveInFlight = true;
      updateStatus('Saving...');

      const snapshot = buildSnapshot({ toolId, root });
      try {
        const res = await window.ToolsState.saveSession({ toolId, sessionId: sessionId || undefined, snapshot, keepalive: !!keepalive });
        const nextSessionId = res?.session?.sessionId ? String(res.session.sessionId) : sessionId;
        if (nextSessionId && nextSessionId !== sessionId) {
          sessionId = nextSessionId;
          setActiveSessionId(toolId, sessionId);
          setSessionParam(sessionId);
        }
        dirty = false;
        updateStatus('Saved.');
        setTimeout(() => updateStatus(''), 1200);
        try {
          await window.ToolsState.logActivity({ toolId, type: 'session_save', summary: `Saved session ${sessionId}`, keepalive: !!keepalive });
        } catch {}
      } catch (err) {
        updateStatus(err?.message || 'Save failed.');
      } finally {
        saveInFlight = false;
      }
    };

    document.addEventListener('tools:save-session', (event) => {
      if (event?.detail?.toolId && event.detail.toolId !== toolId) return;
      saveSession().catch(() => {});
    });

    root.addEventListener('input', () => {
      dirty = true;
    });
    root.addEventListener('change', () => {
      dirty = true;
    });

    setActiveSessionId(toolId, sessionId);
    updateStatus('');

    applySession().catch(() => {});

    const tick = () => {
      if (dirty) saveSession().catch(() => {});
    };
    const timer = window.setInterval(tick, AUTO_SAVE_MS);

    const flush = () => {
      if (!dirty) return;
      saveSession({ keepalive: true }).catch(() => {});
    };

    window.addEventListener('beforeunload', flush);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') flush();
    });

    window.ToolsState.logActivity({ toolId, type: 'tool_open', summary: 'Opened tool' }).catch(() => {});

    return () => {
      window.clearInterval(timer);
      window.removeEventListener('beforeunload', flush);
    };
  };

  document.addEventListener('DOMContentLoaded', async () => {
    let redirectHandled = false;
    try {
      const result = await window.ToolsAuth.handleRedirect();
      redirectHandled = !!result?.redirected;
    } catch (err) {
      const statusEl = $('[data-tools-dashboard="status"]');
      if (statusEl) statusEl.textContent = err?.message || 'Sign-in failed.';
    }
    if (redirectHandled) return;

    const page = String(document.body?.dataset?.page || '').trim();
    const toolId = page && page !== 'tools' ? page : '';
    const root = document.getElementById('main');

    const autosaveMode = String(document.body?.dataset?.toolsAutosave || '').trim().toLowerCase();
    const toolActionsEnabled = autosaveMode !== 'false' && autosaveMode !== 'off' && autosaveMode !== '0';

    const accountModal = initAccountModal();
    const { setStatus } = initAccountBar({
      toolId: page === 'tools-dashboard' ? '' : toolId,
      root,
      toolActionsEnabled,
      onOpenAccount: accountModal.open
    });

    accountModal.setHandlers({
      signIn: () => {
        window.ToolsAuth.signIn({ returnTo: `${window.location.pathname}${window.location.search}${window.location.hash}` })
          .catch((err) => setStatus(err?.message || 'Unable to start sign-in.'));
      },
      signOut: () => {
        window.ToolsAuth.signOut();
        setStatus('Signed out.');
        accountModal.refresh().catch(() => {});
        try {
          document.dispatchEvent(new CustomEvent('tools:auth-changed', { detail: { source: 'tools-account-ui' } }));
        } catch {}
      }
    });

    if (page === 'tools-dashboard') {
      initDashboard({ setStatus }).catch(() => {});
      return;
    }

    if (toolId && root && toolActionsEnabled) {
      initToolAutoSave({ toolId, root, setStatus });
    }
  });
})();
