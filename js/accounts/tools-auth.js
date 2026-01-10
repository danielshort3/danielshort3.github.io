(() => {
  'use strict';

  const STORAGE_KEY = 'toolsAuth';
  const LEGACY_STORAGE_KEYS = ['jobTrackerAuth'];
  const STATE_KEY = 'toolsAuthState';
  const VERIFIER_KEY = 'toolsAuthCodeVerifier';
  const RETURN_TO_KEY = 'toolsAuthReturnTo';

  const getConfig = () => {
    const source = document.body || document.documentElement || {};
    const fromDataset = {
      cognitoDomain: (source.dataset.cognitoDomain || '').trim(),
      cognitoClientId: (source.dataset.cognitoClientId || '').trim(),
      cognitoRedirect: (source.dataset.cognitoRedirect || '').trim(),
      cognitoScopes: (source.dataset.cognitoScopes || '').trim()
    };

    const globalConfig = window.TOOLS_AUTH_CONFIG || {};
    const fallbackRedirect = (() => {
      try {
        return `${window.location.origin}/tools/dashboard`;
      } catch {
        return '';
      }
    })();

    return {
      cognitoDomain: fromDataset.cognitoDomain || String(globalConfig.cognitoDomain || '').trim(),
      cognitoClientId: fromDataset.cognitoClientId || String(globalConfig.cognitoClientId || '').trim(),
      cognitoRedirect: fromDataset.cognitoRedirect || String(globalConfig.cognitoRedirect || '').trim() || fallbackRedirect,
      cognitoScopes: fromDataset.cognitoScopes || String(globalConfig.cognitoScopes || '').trim() || 'openid email profile'
    };
  };

  const parseJwt = (token) => {
    try {
      const payload = String(token || '').split('.')[1];
      if (!payload) return null;
      const normalized = payload.replace(/-/g, '+').replace(/_/g, '/');
      const decoded = atob(normalized.padEnd(normalized.length + (4 - normalized.length % 4) % 4, '='));
      return JSON.parse(decoded);
    } catch {
      return null;
    }
  };

  const getAuthClaims = (auth) => {
    if (!auth?.idToken) return {};
    return auth.claims || parseJwt(auth.idToken) || {};
  };

  const getAuthExpiresAt = (auth) => {
    const numeric = Number(auth?.expiresAt) || 0;
    if (numeric) return numeric;
    const claims = getAuthClaims(auth);
    if (claims?.exp) return claims.exp * 1000;
    return 0;
  };

  const normalizeAuth = (auth) => {
    if (!auth?.idToken) return null;
    const claims = getAuthClaims(auth);
    const expiresAt = getAuthExpiresAt({ ...auth, claims });
    return {
      ...auth,
      claims,
      expiresAt
    };
  };

  const authIsValid = (auth) => {
    if (!auth || !auth.idToken) return false;
    const expiresAt = getAuthExpiresAt(auth);
    if (!expiresAt) return false;
    if (Date.now() > expiresAt - 60 * 1000) return false;
    return true;
  };

  const loadAuthFromKey = (key) => {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || !parsed.idToken) return null;
      return parsed;
    } catch {
      return null;
    }
  };

  const loadAuth = () => {
    const primary = normalizeAuth(loadAuthFromKey(STORAGE_KEY));
    if (authIsValid(primary)) return primary;

    for (const legacyKey of LEGACY_STORAGE_KEYS) {
      const legacy = normalizeAuth(loadAuthFromKey(legacyKey));
      if (authIsValid(legacy)) {
        saveAuth(legacy);
        return legacy;
      }
    }

    return null;
  };

  const saveAuth = (auth) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(auth));
    } catch {}
  };

  const clearAuth = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
  };

  const clearAllAuth = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {}
    LEGACY_STORAGE_KEYS.forEach((key) => {
      try {
        localStorage.removeItem(key);
      } catch {}
    });
    try {
      sessionStorage.removeItem(STATE_KEY);
      sessionStorage.removeItem(VERIFIER_KEY);
      sessionStorage.removeItem(RETURN_TO_KEY);
    } catch {}
  };

  const randomBase64Url = (size = 32) => {
    const buffer = new Uint8Array(size);
    crypto.getRandomValues(buffer);
    const binary = String.fromCharCode(...buffer);
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  };

  const sha256 = async (plain) => {
    const encoder = new TextEncoder();
    const data = encoder.encode(plain);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return new Uint8Array(digest);
  };

  const buildAuthorizeUrl = async (config) => {
    const verifier = randomBase64Url(48);
    const challengeBytes = await sha256(verifier);
    const challenge = btoa(String.fromCharCode(...challengeBytes))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
    const authState = randomBase64Url(16);
    sessionStorage.setItem(STATE_KEY, authState);
    sessionStorage.setItem(VERIFIER_KEY, verifier);

    const params = new URLSearchParams({
      response_type: 'code',
      client_id: config.cognitoClientId,
      redirect_uri: config.cognitoRedirect,
      scope: config.cognitoScopes,
      code_challenge_method: 'S256',
      code_challenge: challenge,
      state: authState
    });
    return `https://${config.cognitoDomain}/oauth2/authorize?${params.toString()}`;
  };

  const exchangeCodeForTokens = async (config, code) => {
    const verifier = sessionStorage.getItem(VERIFIER_KEY) || '';
    sessionStorage.removeItem(VERIFIER_KEY);
    sessionStorage.removeItem(STATE_KEY);
    if (!verifier) throw new Error('Missing PKCE verifier.');

    const params = new URLSearchParams({
      grant_type: 'authorization_code',
      client_id: config.cognitoClientId,
      redirect_uri: config.cognitoRedirect,
      code,
      code_verifier: verifier
    });

    const res = await fetch(`https://${config.cognitoDomain}/oauth2/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params.toString()
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Unable to exchange auth code.');
    }
    const data = await res.json();
    if (!data.id_token) throw new Error('Missing id_token from auth response.');
    const claims = parseJwt(data.id_token) || {};
    const expiresAt = claims.exp ? claims.exp * 1000 : Date.now() + (data.expires_in || 3600) * 1000;
    const auth = normalizeAuth({
      idToken: data.id_token,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt,
      claims
    });
    if (!auth) throw new Error('Unable to save auth.');
    saveAuth(auth);
    return auth;
  };

  const refreshTokens = async (config, auth) => {
    const refreshToken = String(auth?.refreshToken || '').trim();
    if (!refreshToken) throw new Error('Missing refresh token.');

    const params = new URLSearchParams({
      grant_type: 'refresh_token',
      client_id: config.cognitoClientId,
      refresh_token: refreshToken
    });

    const res = await fetch(`https://${config.cognitoDomain}/oauth2/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params.toString()
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Unable to refresh session.');
    }
    const data = await res.json();
    if (!data.id_token) throw new Error('Missing id_token from refresh response.');
    const claims = parseJwt(data.id_token) || {};
    const expiresAt = claims.exp ? claims.exp * 1000 : Date.now() + (data.expires_in || 3600) * 1000;
    const next = normalizeAuth({
      ...auth,
      idToken: data.id_token,
      accessToken: data.access_token || auth.accessToken,
      expiresAt,
      claims
    });
    if (!next) throw new Error('Unable to normalize auth.');
    saveAuth(next);
    return next;
  };

  const normalizeReturnTo = (value) => {
    const raw = String(value || '').trim();
    if (!raw) return '';
    try {
      const url = new URL(raw, window.location.origin);
      if (url.origin !== window.location.origin) return '';
      return `${url.pathname}${url.search}${url.hash}`;
    } catch {
      return '';
    }
  };

  const signIn = async (options = {}) => {
    const config = getConfig();
    if (!config.cognitoDomain || !config.cognitoClientId || !config.cognitoRedirect) {
      throw new Error('Cognito settings are missing.');
    }

    const returnTo = normalizeReturnTo(options.returnTo || `${window.location.pathname}${window.location.search}${window.location.hash}`);
    if (returnTo) {
      try {
        sessionStorage.setItem(RETURN_TO_KEY, returnTo);
      } catch {}
    }

    const url = await buildAuthorizeUrl(config);
    window.location.assign(url);
  };

  const handleRedirect = async () => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const returnedState = params.get('state') || '';
    const storedState = sessionStorage.getItem(STATE_KEY) || '';
    if (!code) return { handled: false };
    if (storedState && returnedState && storedState !== returnedState) {
      throw new Error('Auth state mismatch.');
    }

    const config = getConfig();
    await exchangeCodeForTokens(config, code);

    params.delete('code');
    params.delete('state');
    const nextQuery = params.toString();
    const nextUrl = nextQuery ? `${window.location.pathname}?${nextQuery}` : window.location.pathname;
    window.history.replaceState({}, document.title, nextUrl);

    const returnTo = normalizeReturnTo(sessionStorage.getItem(RETURN_TO_KEY) || '');
    try {
      sessionStorage.removeItem(RETURN_TO_KEY);
    } catch {}

    if (returnTo && returnTo !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.location.replace(returnTo);
      return { handled: true, redirected: true };
    }

    return { handled: true, redirected: false };
  };

  const getUser = (auth) => {
    const claims = getAuthClaims(auth);
    return {
      sub: String(claims.sub || '').trim(),
      email: String(claims.email || '').trim(),
      name: String(claims.name || claims['cognito:username'] || '').trim()
    };
  };

  const getAuth = () => loadAuth();

  const ensureFreshAuth = async () => {
    const config = getConfig();
    const current = loadAuth();
    if (authIsValid(current)) return current;
    if (!current) return null;
    try {
      const refreshed = await refreshTokens(config, current);
      return authIsValid(refreshed) ? refreshed : null;
    } catch {
      clearAuth();
      return null;
    }
  };

  const fetchWithAuth = async (url, options = {}) => {
    const auth = await ensureFreshAuth();
    if (!auth) throw new Error('Not authenticated.');
    const headers = new Headers(options.headers || {});
    if (!headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${auth.idToken}`);
    }
    const res = await fetch(url, { ...options, headers });
    return res;
  };

  window.ToolsAuth = {
    getConfig,
    getAuth,
    getUser,
    authIsValid,
    signIn,
    signOut: clearAllAuth,
    handleRedirect,
    ensureFreshAuth,
    fetchWithAuth
  };
})();
