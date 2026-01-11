(() => {
  'use strict';

  const API = {
    me: '/api/tools/me',
    dashboard: '/api/tools/dashboard',
    state: '/api/tools/state',
    activity: '/api/tools/activity'
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
    const query = search.toString();
    return query ? `${API.state}?${query}` : API.state;
  };

  const getActivityUrl = (params) => {
    const search = new URLSearchParams();
    if (params?.toolId) search.set('tool', params.toolId);
    if (params?.limit) search.set('limit', String(params.limit));
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

  const listActivity = async ({ toolId, limit } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(getActivityUrl({ toolId, limit }), { method: 'GET' });
    return readJson(res);
  };

  const saveSession = async ({ toolId, sessionId, snapshot, outputSummary, keepalive } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(API.state, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ toolId, sessionId, snapshot, outputSummary }),
      keepalive: !!keepalive
    });
    return readJson(res);
  };

  const listSessions = async ({ toolId, limit } = {}) => {
    const res = await window.ToolsAuth.fetchWithAuth(getToolStateUrl({ toolId, limit }), { method: 'GET' });
    return readJson(res);
  };

  const getSession = async ({ toolId, sessionId }) => {
    const res = await window.ToolsAuth.fetchWithAuth(getToolStateUrl({ toolId, sessionId }), { method: 'GET' });
    return readJson(res);
  };

  const deleteSession = async ({ toolId, sessionId }) => {
    const res = await window.ToolsAuth.fetchWithAuth(getToolStateUrl({ toolId, sessionId }), { method: 'DELETE' });
    return readJson(res);
  };

  window.ToolsState = {
    getMe,
    getDashboard,
    logActivity,
    listActivity,
    saveSession,
    listSessions,
    getSession,
    deleteSession
  };
})();
