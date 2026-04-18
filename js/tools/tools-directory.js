(() => {
  'use strict';

  const body = document.body;
  if (!body || body.dataset.toolsLayout !== 'directory') return;

  const resumePanel = document.querySelector('[data-tools-resume="panel"]');
  const resumeStatusEl = document.querySelector('[data-tools-resume="status"]');
  const resumeContentEl = document.querySelector('[data-tools-resume="content"]');
  const cards = Array.from(document.querySelectorAll('.tool-card'));

  if (!cards.length) return;

  const cleanText = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const formatTime = (timestamp) => {
    const numeric = Number(timestamp);
    if (!numeric) return '';
    try {
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short'
      }).format(new Date(numeric));
    } catch {
      return '';
    }
  };

  const toTitleCase = (value) => cleanText(String(value || '').replace(/[-_]+/g, ' '))
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ') || 'Tool';

  const normalizeHref = (href) => String(href || '')
    .trim()
    .replace(/^\.\//, '')
    .replace(/^\//, '')
    .replace(/\.html$/i, '');

  const toolInfoById = new Map();
  cards.forEach((card) => {
    const link = card.querySelector('a[href]');
    if (!link) return;
    const href = normalizeHref(link.getAttribute('href'));
    const toolId = cleanText(href.split('/').pop());
    if (!toolId) return;
    const name = cleanText(card.querySelector('h3')?.textContent) || toTitleCase(toolId);
    toolInfoById.set(toolId, { href, name });
  });

  const getToolInfo = (toolId) => toolInfoById.get(toolId) || {
    href: `tools/${toolId}`,
    name: toTitleCase(toolId)
  };

  const setResumeStatus = (message) => {
    if (!resumeStatusEl) return;
    resumeStatusEl.textContent = cleanText(message);
  };

  const renderResumeShell = (html, statusMessage = '') => {
    if (!resumePanel || !resumeContentEl) return;
    resumePanel.hidden = false;
    setResumeStatus(statusMessage);
    resumeContentEl.innerHTML = html;
  };

  const renderResumeSignedOut = () => {
    renderResumeShell(`
      <div class="tools-resume-prompt">
        <div class="tools-resume-prompt-copy">
          <p class="tools-resume-group-title">Sign in to reopen saved sessions.</p>
          <p class="tools-resume-group-note">Pinned work and recent tool runs will appear here for quick return access.</p>
        </div>
        <div class="tools-resume-prompt-actions">
          <button type="button" class="btn-primary" data-tools-resume-action="sign-in">Sign in</button>
        </div>
      </div>
    `, '');
  };

  const renderResumeLoading = () => {
    renderResumeShell(`
      <div class="tools-resume-empty">
        <p>Loading your saved sessions…</p>
      </div>
    `, '');
  };

  const renderResumeEmpty = () => {
    renderResumeShell(`
      <div class="tools-resume-empty">
        <p>No saved sessions yet. Start a tool and your recent work will show up here.</p>
      </div>
    `, '');
  };

  const renderResumeFailure = (message) => {
    renderResumeShell(`
      <div class="tools-resume-empty">
        <p>${escapeHtml(message || 'Unable to load saved sessions right now.')}</p>
      </div>
    `, '');
  };

  const renderResumeCard = (session) => {
    const info = getToolInfo(session.toolId);
    const href = `${info.href}?session=${encodeURIComponent(session.sessionId)}`;
    const updatedAt = formatTime(session.updatedAt || session.createdAt);
    const metaParts = [info.name];
    if (updatedAt) metaParts.push(`Updated ${updatedAt}`);
    if (session.outputSummary) metaParts.push(session.outputSummary);

    return `
      <article class="tools-resume-card">
        <div class="tools-resume-card-main">
          <h3 class="tools-resume-card-title"><a href="${escapeHtml(href)}">${escapeHtml(session.title || info.name)}</a></h3>
          <p class="tools-resume-card-meta">${escapeHtml(metaParts.join(' · '))}</p>
        </div>
        <div class="tools-resume-card-actions">
          <a class="btn-secondary" href="${escapeHtml(href)}">Reopen</a>
        </div>
      </article>
    `.trim();
  };

  const renderResumeData = (sessions) => {
    const pinned = sessions.filter((session) => session.pinned).slice(0, 4);
    const recent = sessions.filter((session) => !session.pinned).slice(0, 6);

    if (!pinned.length && !recent.length) {
      renderResumeEmpty();
      return;
    }

    const groups = [];
    if (pinned.length) {
      groups.push(`
        <section class="tools-resume-group" aria-labelledby="tools-resume-pinned">
          <div class="tools-resume-group-head">
            <h3 class="tools-resume-group-title" id="tools-resume-pinned">Pinned</h3>
            <p class="tools-resume-group-note">Quick launchers for the sessions you revisit most.</p>
          </div>
          <div class="tools-resume-list">
            ${pinned.map(renderResumeCard).join('')}
          </div>
        </section>
      `);
    }

    if (recent.length) {
      groups.push(`
        <section class="tools-resume-group" aria-labelledby="tools-resume-recent">
          <div class="tools-resume-group-head">
            <h3 class="tools-resume-group-title" id="tools-resume-recent">Recent</h3>
            <p class="tools-resume-group-note">Return to the latest saved work across tools.</p>
          </div>
          <div class="tools-resume-list">
            ${recent.map(renderResumeCard).join('')}
          </div>
        </section>
      `);
    }

    renderResumeShell(`<div class="tools-resume-groups">${groups.join('')}</div>`, '');
  };

  let resumeRefreshTimer = 0;
  let resumeAttempts = 0;
  const maxResumeAttempts = 24;

  const canLoadResume = () => Boolean(
    window.ToolsAuth
    && typeof window.ToolsAuth.getAuth === 'function'
    && typeof window.ToolsAuth.authIsValid === 'function'
    && typeof window.ToolsAuth.signIn === 'function'
    && window.ToolsState
    && typeof window.ToolsState.getDashboard === 'function'
  );

  const scheduleResumeRefresh = (delay = 0) => {
    if (!resumePanel) return;
    window.clearTimeout(resumeRefreshTimer);
    resumeRefreshTimer = window.setTimeout(() => {
      void loadResume();
    }, delay);
  };

  const loadResume = async () => {
    if (!resumePanel) return;
    resumePanel.hidden = false;

    if (!canLoadResume()) {
      renderResumeLoading();
      if (resumeAttempts < maxResumeAttempts) {
        resumeAttempts += 1;
        scheduleResumeRefresh(250);
      } else {
        renderResumeFailure('Account tools are still loading. Open the dashboard once they finish initializing.');
      }
      return;
    }

    resumeAttempts = 0;
    const auth = window.ToolsAuth.getAuth();
    if (!window.ToolsAuth.authIsValid(auth)) {
      renderResumeSignedOut();
      return;
    }

    renderResumeLoading();
    try {
      const data = await window.ToolsState.getDashboard({ sessionsLimit: 12, activityLimit: 0 });
      const sessions = Array.isArray(data?.recentSessions)
        ? data.recentSessions
          .map((session) => ({
            toolId: cleanText(session?.toolId),
            sessionId: cleanText(session?.sessionId),
            title: cleanText(session?.title),
            outputSummary: cleanText(session?.outputSummary || session?.note),
            createdAt: Number(session?.createdAt) || 0,
            updatedAt: Number(session?.updatedAt) || 0,
            pinned: Boolean(session?.pinned)
          }))
          .filter((session) => session.toolId && session.sessionId)
        : [];

      renderResumeData(sessions);
    } catch (error) {
      renderResumeFailure(error?.message || 'Unable to load saved sessions right now.');
    }
  };

  if (resumeContentEl) {
    resumeContentEl.addEventListener('click', (event) => {
      const actionEl = event.target.closest('[data-tools-resume-action]');
      if (!actionEl) return;
      const action = cleanText(actionEl.dataset.toolsResumeAction);
      if (action !== 'sign-in' || !canLoadResume()) return;
      window.ToolsAuth.signIn({ returnTo: `${window.location.pathname}${window.location.search}${window.location.hash}` });
    });
  }

  document.addEventListener('tools:auth-changed', () => {
    scheduleResumeRefresh(0);
  });
  document.addEventListener('tools:session-meta-updated', () => {
    scheduleResumeRefresh(0);
  });
  document.addEventListener('tools:session-deleted', () => {
    scheduleResumeRefresh(0);
  });
  window.addEventListener('load', () => {
    scheduleResumeRefresh(0);
  });

  if (resumePanel) scheduleResumeRefresh(0);
})();
