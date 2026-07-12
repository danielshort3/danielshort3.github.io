(() => {
  'use strict';

  const API = {
    me: '/api/tools/me',
    dashboard: '/api/tools/dashboard',
    state: '/api/tools/state',
    activity: '/api/tools/activity'
  };
  const MAX_KEEPALIVE_BODY_BYTES = 60 * 1024;

  const utf8Bytes = (value) => {
    const text = String(value || '');
    if (typeof TextEncoder === 'function') return new TextEncoder().encode(text).byteLength;
    return unescape(encodeURIComponent(text)).length;
  };

  const readJson = async (res) => {
    let data;
    try {
      data = await res.json();
    } catch {
      data = null;
    }
    if (!res.ok) {
      const message = (data && (data.error || data.message)) ? String(data.error || data.message) : `Request failed (${res.status})`;
      const err = new Error(message);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  };

  const getToolStateUrl = (params) => {
    const search = new URLSearchParams();
    if (params?.toolId) search.set('tool', params.toolId);
    if (params?.sessionId) search.set('session', params.sessionId);
    if (params?.limit) search.set('limit', String(params.limit));
    if (params?.cursor) search.set('cursor', params.cursor);
    if (typeof params?.expectedVersion !== 'undefined') search.set('version', String(params.expectedVersion));
    const query = search.toString();
    return query ? `${API.state}?${query}` : API.state;
  };

  const getActivityUrl = (params) => {
    const search = new URLSearchParams();
    if (params?.toolId) search.set('tool', params.toolId);
    if (params?.limit) search.set('limit', String(params.limit));
    if (params?.cursor) search.set('cursor', params.cursor);
    const query = search.toString();
    return query ? `${API.activity}?${query}` : API.activity;
  };

  const getMe = async () => {
    const res = await window.ToolsAuth.fetchWithAuth(API.me, { method: 'GET' });
    return readJson(res);
  };

  const getDashboardUrl = (params) => {
    const search = new URLSearchParams();
    if (params?.sessionsLimit) search.set('sessionsLimit', String(params.sessionsLimit));
    if (params?.activityLimit) search.set('activityLimit', String(params.activityLimit));
    const query = search.toString();
    return query ? `${API.dashboard}?${query}` : API.dashboard;
  };

  const getDashboard = async ({ sessionsLimit, activityLimit } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(getDashboardUrl({ sessionsLimit, activityLimit }), { method: 'GET' });
    return readJson(res);
  };

  const logActivity = async ({ toolId, type, summary, data, keepalive } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(API.activity, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ toolId, type, summary, data }),
      keepalive: !!keepalive
    });
    return readJson(res);
  };

  const listActivity = async ({ toolId, limit, cursor } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(getActivityUrl({ toolId, limit, cursor }), { method: 'GET' });
    return readJson(res);
  };

  const saveSession = async ({ toolId, sessionId, snapshot, outputSummary, expectedVersion, keepalive } = {}) => {
    const body = JSON.stringify({ toolId, sessionId, snapshot, outputSummary, expectedVersion });
    const res = await window.ToolsAuth.fetchWithAuth(API.state, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive: !!keepalive && utf8Bytes(body) <= MAX_KEEPALIVE_BODY_BYTES
    });
    return readJson(res);
  };

  const updateSessionMeta = async ({ toolId, sessionId, title, note, tags, pinned, expectedVersion } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(API.state, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ toolId, sessionId, title, note, tags, pinned, expectedVersion })
    });
    return readJson(res);
  };

  const listSessions = async ({ toolId, limit, cursor } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(getToolStateUrl({ toolId, limit, cursor }), { method: 'GET' });
    return readJson(res);
  };

  const getSession = async ({ toolId, sessionId }) => {
    const res = await window.ToolsAuth.fetchWithAuth(getToolStateUrl({ toolId, sessionId }), { method: 'GET' });
    return readJson(res);
  };

  const deleteSession = async ({ toolId, sessionId, expectedVersion }) => {
    const res = await window.ToolsAuth.fetchWithAuth(getToolStateUrl({ toolId, sessionId, expectedVersion }), { method: 'DELETE' });
    return readJson(res);
  };

  const deleteAllAccountData = async () => {
    const res = await window.ToolsAuth.fetchWithAuth(`${API.state}?all=true`, {
      method: 'DELETE',
      headers: { 'X-Tools-Confirm': 'DELETE-ALL' }
    });
    return readJson(res);
  };

  window.ToolsState = {
    getMe,
    getDashboard,
    logActivity,
    listActivity,
    saveSession,
    updateSessionMeta,
    listSessions,
    getSession,
    deleteSession,
    deleteAllAccountData
  };
})();
