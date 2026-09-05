/* ===================================================================
   File: site-route-runtime.js
   Purpose: Shared lifecycle management for persistent-shell routes.
=================================================================== */
(() => {
  'use strict';

  if (typeof window === 'undefined' || typeof document === 'undefined') return;
  if (window.SiteRoutes && window.SiteRoutes.version) return;

  const ROUTE_CONTENT_SELECTOR = '[data-site-route-content], [data-personal-detail-content], #main';
  const ROUTE_MANIFEST_SELECTOR = 'script#site-route-manifest[data-site-route-manifest]';
  const LEGACY_LIFECYCLE = Symbol('legacy-route-lifecycle');
  const registry = new Map();
  const legacyScopes = new Map();
  const loadingScopes = new Set();
  let currentRecord = null;
  let activeScope = null;
  let operationId = 0;
  let directLoadStarted = false;

  const normalizeId = (value) => String(value || '').trim();

  const emit = (name, detail = {}, { cancelable = false } = {}) => {
    let event;
    try {
      event = new CustomEvent(name, { bubbles: false, cancelable, detail });
    } catch (_) {
      event = document.createEvent('CustomEvent');
      event.initCustomEvent(name, false, cancelable, detail);
    }
    document.dispatchEvent(event);
    return event;
  };

  const reportError = (phase, id, error) => {
    emit('site:route-error', { phase, id, error });
  };

  const callSafely = async (fn, context, phase, id) => {
    if (typeof fn !== 'function') return undefined;
    try {
      return await fn(context);
    } catch (error) {
      reportError(phase, id, error);
      throw error;
    }
  };

  const makeCleanupBag = () => {
    const callbacks = [];
    let cleaned = false;

    const add = (callback) => {
      if (typeof callback !== 'function') {
        throw new TypeError('SiteRoutes cleanup expects a function.');
      }
      if (cleaned) {
        Promise.resolve().then(callback).catch(() => {});
        return callback;
      }
      callbacks.push(callback);
      return callback;
    };

    const run = async () => {
      if (cleaned) return;
      cleaned = true;
      const errors = [];
      while (callbacks.length) {
        const callback = callbacks.pop();
        try {
          await callback();
        } catch (error) {
          errors.push(error);
        }
      }
      if (errors.length) {
        const error = new Error(`Site route cleanup failed (${errors.length}).`);
        error.causes = errors;
        throw error;
      }
    };

    return Object.freeze({ add, run });
  };

  const normalizedScriptUrl = (value) => {
    const source = typeof value === 'string' ? value : value?.src;
    if (!source) return '';
    try {
      return new URL(source, window.location.href).href;
    } catch (_) {
      return String(source);
    }
  };

  const isPersistentScript = (value) => {
    const src = normalizedScriptUrl(value);
    if (!src) return false;
    let pathname = src.toLowerCase();
    try { pathname = new URL(src).pathname.toLowerCase(); } catch (_) {}
    return /\/(?:dist\/)?site-(?:shell|consent|tools-account)(?:[.-]|$)/.test(pathname) ||
      /\/js\/(?:common\/(?:no-js|common|audience-config|site-realm|modal-accessibility|certifications-modal)|navigation\/navigation|animations\/animations)(?:[.-]|$)/.test(pathname) ||
      /\/js\/(?:analytics|privacy|vendor)\//.test(pathname);
  };

  const normalizeScripts = (scripts) => [...new Set((Array.isArray(scripts) ? scripts : [])
    .map(normalizedScriptUrl)
    .filter((src) => src && !isPersistentScript(src)))];

  const createLegacyScope = (routeId, scripts = []) => {
    const id = normalizeId(routeId);
    const existing = legacyScopes.get(id);
    if (existing && !existing.closed) {
      normalizeScripts(scripts).forEach((src) => existing.scripts.add(src));
      return existing;
    }
    const cleanup = makeCleanupBag();
    const scope = {
      id,
      scripts: new Set(normalizeScripts(scripts)),
      cleanup,
      fetchControllers: new Set(),
      mediaRecorders: new Set(),
      beforeUnloadListeners: new Set(),
      closed: false
    };
    cleanup.add(() => {
      scope.fetchControllers.forEach((controller) => controller.abort());
      scope.fetchControllers.clear();
    });
    legacyScopes.set(id, scope);
    return scope;
  };

  const scriptBelongsToScope = (scope, script) => {
    if (!scope || scope.closed || !script) return false;
    const src = normalizedScriptUrl(script.src || script.getAttribute?.('src'));
    if (!src) return false;
    return scope.scripts.size === 0 || scope.scripts.has(src);
  };

  const scopeForCurrentExecution = () => {
    if (activeScope && !activeScope.closed) return activeScope;
    const script = document.currentScript;
    if (!script) return null;
    const scopes = [...loadingScopes];
    for (let index = scopes.length - 1; index >= 0; index -= 1) {
      if (scriptBelongsToScope(scopes[index], script)) return scopes[index];
    }
    return null;
  };

  const callInScope = (scope, callback, thisArg, args = []) => {
    if (!scope || scope.closed) return callback.apply(thisArg, args);
    const previous = activeScope;
    activeScope = scope;
    try {
      return callback.apply(thisArg, args);
    } finally {
      activeScope = previous;
    }
  };

  const cleanupLegacyScope = async (scope) => {
    if (!scope || scope.closed) return;
    scope.closed = true;
    loadingScopes.delete(scope);
    if (legacyScopes.get(scope.id) === scope) legacyScopes.delete(scope.id);
    await scope.cleanup.run();
  };

  const installLegacyResourceTracking = () => {
    const EventTargetConstructor = window.EventTarget;
    if (EventTargetConstructor?.prototype) {
      const prototype = EventTargetConstructor.prototype;
      const originalAdd = prototype.addEventListener;
      const originalRemove = prototype.removeEventListener;
      const listenerEntries = new WeakMap();
      const captureValue = (options) => typeof options === 'boolean' ? options : Boolean(options?.capture);

      prototype.addEventListener = function addScopedEventListener(type, listener, options) {
        const scope = scopeForCurrentExecution();
        if (!scope || !listener) return originalAdd.call(this, type, listener, options);

        const readyEvent = (this === document && type === 'DOMContentLoaded' && document.readyState !== 'loading') ||
          (this === window && type === 'load' && document.readyState === 'complete');
        const callback = typeof listener === 'function'
          ? listener
          : (...args) => listener.handleEvent?.(...args);
        const wrapped = (...args) => callInScope(scope, callback, this, args);
        if (readyEvent) {
          Promise.resolve().then(() => {
            if (!scope.closed) wrapped(new Event(type));
          });
          return;
        }

        let entries = listenerEntries.get(this);
        if (!entries) {
          entries = [];
          listenerEntries.set(this, entries);
        }
        const entry = { type, listener, wrapped, capture: captureValue(options), options, scope };
        entries.push(entry);
        if (this === window && type === 'beforeunload') scope.beforeUnloadListeners.add(entry);
        originalAdd.call(this, type, wrapped, options);
        scope.cleanup.add(() => {
          originalRemove.call(this, type, wrapped, options);
          scope.beforeUnloadListeners.delete(entry);
          const index = entries.indexOf(entry);
          if (index >= 0) entries.splice(index, 1);
        });
      };

      prototype.removeEventListener = function removeScopedEventListener(type, listener, options) {
        const capture = captureValue(options);
        const entries = listenerEntries.get(this) || [];
        const entry = entries.find((candidate) => (
          candidate.type === type && candidate.listener === listener && candidate.capture === capture
        ));
        if (!entry) return originalRemove.call(this, type, listener, options);
        originalRemove.call(this, type, entry.wrapped, entry.options);
        entries.splice(entries.indexOf(entry), 1);
      };
    }

    const trackScheduledCallback = (setName, clearName, repeating = false) => {
      const originalSet = window[setName];
      const originalClear = window[clearName];
      if (typeof originalSet !== 'function' || typeof originalClear !== 'function') return;
      const scheduledScopes = new Map();
      const registeredScopes = new WeakSet();
      const registerScopeCleanup = (scope) => {
        if (registeredScopes.has(scope)) return;
        registeredScopes.add(scope);
        scope.cleanup.add(() => {
          scheduledScopes.forEach((scheduledScope, handle) => {
            if (scheduledScope !== scope) return;
            scheduledScopes.delete(handle);
            originalClear.call(window, handle);
          });
        });
      };
      window[setName] = function setScopedCallback(callback, ...args) {
        const scope = scopeForCurrentExecution();
        if (!scope || typeof callback !== 'function') return originalSet.call(window, callback, ...args);
        registerScopeCleanup(scope);
        let handle;
        const wrapped = (...callbackArgs) => {
          if (!repeating) scheduledScopes.delete(handle);
          return callInScope(scope, callback, window, callbackArgs);
        };
        handle = originalSet.call(window, wrapped, ...args);
        scheduledScopes.set(handle, scope);
        return handle;
      };
      window[clearName] = function clearScopedCallback(handle) {
        scheduledScopes.delete(handle);
        return originalClear.call(window, handle);
      };
    };

    trackScheduledCallback('setTimeout', 'clearTimeout');
    trackScheduledCallback('setInterval', 'clearInterval', true);
    trackScheduledCallback('requestAnimationFrame', 'cancelAnimationFrame');
    trackScheduledCallback('requestIdleCallback', 'cancelIdleCallback');

    ['MutationObserver', 'ResizeObserver', 'IntersectionObserver', 'PerformanceObserver'].forEach((name) => {
      const Original = window[name];
      if (typeof Original !== 'function' || typeof Proxy !== 'function') return;
      try {
        window[name] = new Proxy(Original, {
          construct(Target, args, NewTarget) {
            const scope = scopeForCurrentExecution();
            if (scope && typeof args[0] === 'function') {
              const callback = args[0];
              args[0] = (...callbackArgs) => callInScope(scope, callback, null, callbackArgs);
            }
            const observer = Reflect.construct(Target, args, NewTarget);
            if (scope && typeof observer.disconnect === 'function') {
              scope.cleanup.add(() => observer.disconnect());
            }
            return observer;
          }
        });
      } catch (_) {}
    });

    const OriginalWorker = window.Worker;
    if (typeof OriginalWorker === 'function' && typeof Proxy === 'function') {
      try {
        window.Worker = new Proxy(OriginalWorker, {
          construct(Target, args, NewTarget) {
            const scope = scopeForCurrentExecution();
            const worker = Reflect.construct(Target, args, NewTarget);
            if (scope && typeof worker.terminate === 'function') {
              scope.cleanup.add(() => worker.terminate());
            }
            return worker;
          }
        });
      } catch (_) {}
    }

    const OriginalMediaRecorder = window.MediaRecorder;
    if (typeof OriginalMediaRecorder === 'function' && typeof Proxy === 'function') {
      try {
        window.MediaRecorder = new Proxy(OriginalMediaRecorder, {
          construct(Target, args, NewTarget) {
            const scope = scopeForCurrentExecution();
            const recorder = Reflect.construct(Target, args, NewTarget);
            if (scope) {
              scope.mediaRecorders.add(recorder);
              scope.cleanup.add(() => {
                scope.mediaRecorders.delete(recorder);
                if (recorder.state && recorder.state !== 'inactive') recorder.stop();
              });
            }
            return recorder;
          }
        });
      } catch (_) {}
    }

    ['AudioContext', 'webkitAudioContext'].forEach((name) => {
      const Original = window[name];
      if (typeof Original !== 'function' || typeof Proxy !== 'function') return;
      try {
        window[name] = new Proxy(Original, {
          construct(Target, args, NewTarget) {
            const scope = scopeForCurrentExecution();
            const context = Reflect.construct(Target, args, NewTarget);
            if (scope && typeof context.close === 'function') {
              scope.cleanup.add(() => context.state === 'closed' ? undefined : context.close());
            }
            return context;
          }
        });
      } catch (_) {}
    });

    const urlApi = window.URL;
    if (urlApi && typeof urlApi.createObjectURL === 'function' && typeof urlApi.revokeObjectURL === 'function') {
      try {
        const originalCreate = urlApi.createObjectURL.bind(urlApi);
        const originalRevoke = urlApi.revokeObjectURL.bind(urlApi);
        const objectUrlScopes = new Map();
        urlApi.createObjectURL = (value) => {
          const objectUrl = originalCreate(value);
          const scope = scopeForCurrentExecution();
          if (scope) {
            objectUrlScopes.set(objectUrl, scope);
            scope.cleanup.add(() => {
              if (objectUrlScopes.delete(objectUrl)) originalRevoke(objectUrl);
            });
          }
          return objectUrl;
        };
        urlApi.revokeObjectURL = (objectUrl) => {
          objectUrlScopes.delete(objectUrl);
          return originalRevoke(objectUrl);
        };
      } catch (_) {}
    }

    const mediaDevices = window.navigator?.mediaDevices;
    ['getUserMedia', 'getDisplayMedia'].forEach((name) => {
      const original = mediaDevices?.[name];
      if (typeof original !== 'function') return;
      try {
        mediaDevices[name] = function getScopedMedia(...args) {
          const scope = scopeForCurrentExecution();
          const result = original.apply(this, args);
          if (!scope || !result || typeof result.then !== 'function') return result;
          return result.then((stream) => {
            if (stream && typeof stream.getTracks === 'function') {
              scope.cleanup.add(() => stream.getTracks().forEach((track) => track.stop()));
            }
            return stream;
          });
        };
      } catch (_) {}
    });

    const originalFetch = window.fetch;
    if (typeof originalFetch === 'function') {
      window.fetch = function scopedFetch(input, init) {
        const scope = scopeForCurrentExecution();
        if (!scope || typeof AbortController !== 'function') {
          return originalFetch.call(this, input, init);
        }
        const controller = new AbortController();
        const callerSignal = init?.signal || (typeof Request === 'function' && input instanceof Request ? input.signal : null);
        let signal = controller.signal;
        if (callerSignal) {
          if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.any === 'function') {
            signal = AbortSignal.any([callerSignal, controller.signal]);
          } else {
            return originalFetch.call(this, input, init);
          }
        }
        scope.fetchControllers.add(controller);
        let request;
        try {
          request = originalFetch.call(this, input, { ...(init || {}), signal });
        } catch (error) {
          scope.fetchControllers.delete(controller);
          throw error;
        }
        return Promise.resolve(request).finally(() => scope.fetchControllers.delete(controller));
      };
    }
  };

  installLegacyResourceTracking();

  const readManifest = (scope = document) => {
    const manifestNode = scope.querySelector?.(ROUTE_MANIFEST_SELECTOR);
    if (!manifestNode) return null;
    try {
      const value = JSON.parse(manifestNode.textContent || '{}');
      return value && typeof value === 'object' ? value : null;
    } catch (error) {
      reportError('manifest', '', error);
      return null;
    }
  };

  const routeIdForDocument = () => {
    const manifest = readManifest();
    const bodyId = normalizeId(document.body?.dataset?.siteRouteId);
    const manifestId = normalizeId(manifest?.id);
    const moduleId = normalizeId(manifest?.module);
    if (moduleId && registry.has(moduleId)) return moduleId;
    if (manifestId && registry.has(manifestId)) return manifestId;
    return moduleId || manifestId || bodyId;
  };

  const contextWith = (record, additions = {}) => Object.freeze({
    ...record.context,
    ...additions,
    signal: record.controller.signal,
    cleanup: record.cleanup.add
  });

  const unregister = (id, lifecycle) => {
    if (registry.get(id) !== lifecycle) return false;
    registry.delete(id);
    emit('site:route-unregistered', { id });
    return true;
  };

  const register = (routeId, lifecycle = {}) => {
    const id = normalizeId(routeId);
    if (!id) throw new TypeError('SiteRoutes.register requires a route id.');
    if (!lifecycle || typeof lifecycle !== 'object') {
      throw new TypeError(`SiteRoutes.register(${id}) requires a lifecycle object.`);
    }
    ['mount', 'beforeLeave', 'unmount'].forEach((name) => {
      if (lifecycle[name] != null && typeof lifecycle[name] !== 'function') {
        throw new TypeError(`SiteRoutes.register(${id}) ${name} must be a function.`);
      }
    });

    registry.set(id, lifecycle);
    if (currentRecord?.id === id && currentRecord.lifecycle?.[LEGACY_LIFECYCLE]) {
      currentRecord.lifecycle = lifecycle;
    }
    emit('site:route-registered', { id });
    return () => unregister(id, lifecycle);
  };

  const get = (routeId) => registry.get(normalizeId(routeId)) || null;

  const addCleanup = (callback, routeId = '') => {
    if (typeof callback !== 'function') throw new TypeError('SiteRoutes.addCleanup expects a function.');
    const id = normalizeId(routeId);
    const scope = id ? legacyScopes.get(id) : scopeForCurrentExecution();
    if (scope && !scope.closed) return scope.cleanup.add(callback);
    if (currentRecord && (!id || currentRecord.id === id || currentRecord.routeId === id)) {
      return currentRecord.cleanup.add(callback);
    }
    return callback;
  };

  const runInScope = (routeId, callback) => {
    const id = normalizeId(routeId);
    if (!id) throw new TypeError('SiteRoutes.runInScope requires a route id.');
    if (typeof callback !== 'function') throw new TypeError('SiteRoutes.runInScope requires a callback.');
    const scope = createLegacyScope(id);
    loadingScopes.add(scope);
    let result;
    try {
      result = callInScope(scope, callback, null);
    } catch (error) {
      loadingScopes.delete(scope);
      throw error;
    }
    if (!result || typeof result.then !== 'function') {
      loadingScopes.delete(scope);
      return result;
    }
    return Promise.resolve(result).finally(() => loadingScopes.delete(scope));
  };

  const loadLegacyScript = (scope, src) => runInScope(scope.id, () => new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = false;
    script.dataset.siteRouteOwnedScript = scope.id;
    script.addEventListener('load', () => {
      script.remove();
      resolve();
    }, { once: true });
    script.addEventListener('error', () => {
      script.remove();
      reject(new Error(`Failed to load route script: ${src}`));
    }, { once: true });
    document.head.appendChild(script);
  }));

  const ensureLegacyRoute = (routeId, options = {}) => {
    const id = normalizeId(routeId);
    if (!id) throw new TypeError('SiteRoutes.ensureLegacyRoute requires a route id.');
    const scripts = normalizeScripts(options.scripts);
    const existing = registry.get(id);
    if (existing && !existing[LEGACY_LIFECYCLE]) return existing;

    const scope = createLegacyScope(id, scripts);
    if (existing?.[LEGACY_LIFECYCLE]) return existing;

    const lifecycle = {
      [LEGACY_LIFECYCLE]: true,
      beforeLeave() {
        const active = legacyScopes.get(id);
        if (!active) return true;
        let vetoed = false;
        active.beforeUnloadListeners.forEach((entry) => {
          try {
            const event = new Event('beforeunload', { cancelable: true });
            const result = entry.wrapped(event);
            if (result === false || event.defaultPrevented) vetoed = true;
          } catch (_) {}
        });
        const recording = [...active.mediaRecorders].some((recorder) => (
          recorder?.state === 'recording' || recorder?.state === 'paused'
        ));
        return !vetoed && !recording;
      },
      async mount(context) {
        const active = createLegacyScope(id, scripts);
        context.cleanup(() => cleanupLegacyScope(active));
        if (context.navigationType === 'load') return;
        // A retried route may come from a newer build than its registration.
        // Keep its cleanup scope, but execute only this document's script plan.
        active.scripts = new Set(Array.isArray(context.manifest?.scripts)
          ? normalizeScripts(context.manifest.scripts)
          : scripts);
        for (const src of active.scripts) {
          if (context.signal.aborted) return;
          await loadLegacyScript(active, src);
        }
      },
      async unmount() {
        await cleanupLegacyScope(legacyScopes.get(id));
      }
    };
    registry.set(id, lifecycle);
    emit('site:route-registered', { id, legacy: true, scripts: [...scope.scripts] });
    return lifecycle;
  };

  const beforeLeave = async (options = {}) => {
    const record = currentRecord;
    if (!record) return true;
    const context = contextWith(record, options);
    const pending = [];
    const event = emit('site:route-before-leave', {
      id: record.routeId,
      root: record.context.root,
      url: context.url,
      navigationType: context.navigationType,
      waitUntil: (operation) => pending.push(Promise.resolve(operation))
    }, { cancelable: true });
    const lifecycleResult = callSafely(record.lifecycle?.beforeLeave, context, 'before-leave', record.routeId);
    const results = await Promise.all([...pending, lifecycleResult]);
    return currentRecord === record && !event.defaultPrevented && results.every((result) => result !== false);
  };

  const unmountRecord = async (record, options = {}) => {
    if (!record || record.unmounted) return;
    record.unmounted = true;
    record.controller.abort();
    const context = contextWith(record, options);
    let hookError = null;
    let cleanupError = null;

    try {
      await callSafely(record.lifecycle?.unmount, context, 'unmount', record.id);
    } catch (error) {
      hookError = error;
    }
    try {
      await record.cleanup.run();
    } catch (error) {
      cleanupError = error;
      reportError('cleanup', record.id, error);
    }

    emit('site:route-unmounted', {
      id: record.routeId,
      root: record.context.root,
      url: context.url,
      navigationType: context.navigationType
    });
    await cleanupLegacyScope(legacyScopes.get(record.id)).catch((error) => {
      cleanupError = cleanupError || error;
      reportError('legacy-cleanup', record.id, error);
    });
    if (record.routeId !== record.id) {
      await cleanupLegacyScope(legacyScopes.get(record.routeId)).catch((error) => {
        cleanupError = cleanupError || error;
        reportError('legacy-cleanup', record.routeId, error);
      });
    }
    if (hookError) throw hookError;
    if (cleanupError) throw cleanupError;
  };

  const unmount = async (options = {}) => {
    const record = currentRecord;
    if (!record) return;
    if (currentRecord === record) currentRecord = null;
    await unmountRecord(record, options);
  };

  const mount = async (routeId, options = {}) => {
    const id = normalizeId(routeId);
    if (!id) return null;
    const mountOperation = ++operationId;
    if (currentRecord) await unmount({ reason: 'replace' });
    if (mountOperation !== operationId) return null;

    const cleanup = makeCleanupBag();
    const controller = new AbortController();
    const externalSignal = options.signal;
    if (externalSignal) {
      if (externalSignal.aborted) {
        controller.abort(externalSignal.reason);
      } else {
        const abortFromExternal = () => controller.abort(externalSignal.reason);
        externalSignal.addEventListener('abort', abortFromExternal, { once: true });
        cleanup.add(() => externalSignal.removeEventListener('abort', abortFromExternal));
      }
    }

    const root = options.root || document.querySelector(ROUTE_CONTENT_SELECTOR) || document.body;
    const url = options.url instanceof URL
      ? options.url
      : new URL(String(options.url || window.location.href), window.location.href);
    const navigationType = normalizeId(options.navigationType) || 'load';
    const lifecycle = get(id);
    const instanceId = normalizeId(options.manifest?.id) || id;
    const record = {
      id,
      routeId: instanceId,
      lifecycle,
      cleanup,
      controller,
      unmounted: false,
      context: Object.freeze({
        ...options,
        id: instanceId,
        module: id,
        root,
        url,
        navigationType,
        signal: controller.signal,
        cleanup: cleanup.add
      })
    };
    currentRecord = record;

    emit('site:route-before-mount', { id: instanceId, root, url, navigationType });
    try {
      const result = await callSafely(lifecycle?.mount, record.context, 'mount', id);
      if (typeof result === 'function') cleanup.add(result);
      if (mountOperation !== operationId || currentRecord !== record || controller.signal.aborted) {
        if (currentRecord === record) currentRecord = null;
        await unmountRecord(record, { reason: 'superseded' });
        return null;
      }
      emit('site:route-mounted', { id: instanceId, root, url, navigationType });
      return record.context;
    } catch (error) {
      if (currentRecord === record) currentRecord = null;
      await unmountRecord(record, { reason: 'mount-error' }).catch(() => {});
      throw error;
    }
  };

  const current = () => currentRecord ? currentRecord.context : null;

  const mountDirectLoad = () => {
    if (directLoadStarted) return;
    directLoadStarted = true;
    const id = routeIdForDocument();
    if (!id) return;
    mount(id, {
      root: document.querySelector(ROUTE_CONTENT_SELECTOR) || document.body,
      url: window.location.href,
      navigationType: 'load',
      manifest: readManifest()
    }).catch((error) => {
      console.error(`Failed to mount route ${id}.`, error);
    });
  };

  window.SiteRoutes = Object.freeze({
    version: 1,
    register,
    unregister,
    get,
    addCleanup,
    ensureLegacyRoute,
    runInScope,
    mount,
    beforeLeave,
    unmount,
    current,
    readManifest
  });

  const prepareDirectLegacyScope = () => {
    const manifest = readManifest();
    const id = normalizeId(manifest?.module || manifest?.id || document.body?.dataset?.siteRouteId);
    if (!id || registry.has(id)) return;
    const scripts = normalizeScripts(manifest?.scripts);
    if (!scripts.length) return;
    const scope = createLegacyScope(id, scripts);
    ensureLegacyRoute(id, { scripts });
    loadingScopes.add(scope);
    const stopTracking = () => loadingScopes.delete(scope);
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', stopTracking, { once: true });
    } else {
      stopTracking();
    }
  };

  prepareDirectLegacyScope();

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountDirectLoad, { once: true });
  } else {
    Promise.resolve().then(mountDirectLoad);
  }
})();
