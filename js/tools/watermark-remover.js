(() => {
  'use strict';

  const LAUNCH_API = '/api/tools/watermark-remover';
  const LAUNCH_PATH = '/home-tools/watermark/session';
  const FUNNEL_ORIGIN = 'https://daniel-pc.tail9686c5.ts.net:8443';
  const LOOPBACK_HOSTS = new Set(['127.0.0.1', '[::1]', 'localhost']);

  function launchDestination(value) {
    const url = new URL(value);
    const localDevelopment = LOOPBACK_HOSTS.has(window.location.hostname) &&
      LOOPBACK_HOSTS.has(url.hostname) && url.protocol === 'http:';
    if ((url.origin !== FUNNEL_ORIGIN && !localDevelopment) || url.pathname !== LAUNCH_PATH ||
        url.search || url.hash || url.username || url.password) {
      throw new Error('Invalid launch destination.');
    }
    return url.href;
  }

  function init() {
    const root = document.querySelector('[data-watermark-launch]');
    if (!root) return;
    const action = root.querySelector('[data-watermark-action]');
    const button = root.querySelector('[data-watermark-open]');
    const status = root.querySelector('[data-watermark-status]');
    let pending = null;
    let form = null;

    function access() {
      const api = window.ToolsAuth;
      const auth = api?.getAuth?.();
      const signedIn = Boolean(api?.authIsValid?.(auth));
      const owner = signedIn && api.isAdmin(auth) ? api.getUser(auth)?.sub : '';
      return { signedIn, owner };
    }

    function updateAccess() {
      const { signedIn, owner } = access();
      if (pending && pending.owner !== owner) {
        pending.controller.abort();
        pending = null;
      }
      action.hidden = !owner;
      button.disabled = !owner || Boolean(pending);
      status.textContent = owner ? (pending ? 'Opening Watermark Remover…' : '') : signedIn
        ? 'This account does not have administrator access.'
        : 'Sign in with an administrator account to open Watermark Remover.';
    }

    button.addEventListener('click', async () => {
      const { owner } = access();
      if (!owner || pending) return;

      const request = { owner, controller: new AbortController() };
      pending = request;
      updateAccess();
      const timer = setTimeout(() => request.controller.abort(), 12_000);
      let opened = false;
      try {
        const response = await window.ToolsAuth.fetchWithAuth(LAUNCH_API, {
          method: 'POST',
          cache: 'no-store',
          redirect: 'error',
          signal: request.controller.signal,
          headers: { Accept: 'application/json' }
        });
        const data = await response.json();
        if (request !== pending || owner !== access().owner) return;
        if (request.controller.signal.aborted) throw new Error('Launch expired.');
        if (!response.ok || !data.ok) {
          status.textContent = response.status === 401
            ? 'Sign in again to open Watermark Remover.'
            : response.status === 403
              ? 'This account does not have administrator access.'
              : 'Watermark Remover is unavailable. Try again when the PC is online.';
          return;
        }
        if (typeof data.ticket !== 'string' || !/^[\w-]+\.[\w-]+$/.test(data.ticket)) {
          throw new Error('Invalid launch ticket.');
        }
        const destination = launchDestination(data.launchUrl);
        form = document.createElement('form');
        form.method = 'POST';
        form.target = '_top';
        form.action = destination;
        form.autocomplete = 'off';
        form.hidden = true;
        const ticket = document.createElement('input');
        ticket.type = 'hidden';
        ticket.name = 'ticket';
        ticket.value = data.ticket;
        form.append(ticket);
        root.append(form);
        form.submit();
        opened = true;
      } catch {
        if (request === pending) status.textContent = 'Could not open Watermark Remover. Try again.';
      } finally {
        clearTimeout(timer);
        if (!opened && request === pending) {
          form?.remove();
          form = null;
          pending = null;
          button.disabled = false;
        }
      }
    });

    document.addEventListener('tools:auth-changed', updateAccess);
    window.addEventListener('pagehide', () => {
      pending?.controller.abort();
      pending = null;
      form?.remove();
      form = null;
    });
    window.addEventListener('pageshow', updateAccess);
    updateAccess();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
