import { isCurrentPrivacyConsent, PRIVACY_CONSENT_KEY } from '../shared/privacy-consent.js';

const PAGE_RUNTIME_FILE = 'content/page-runtime.js';

const isScriptablePageUrl = (value) => {
  if (typeof value !== 'string' || !value.trim()) return false;
  try {
    return ['http:', 'https:'].includes(new URL(value).protocol);
  } catch {
    return false;
  }
};

export class ToolbarActionLauncher {
  constructor({
    sidePanel = globalThis.chrome?.sidePanel,
    scripting = globalThis.chrome?.scripting,
    storageLocal = globalThis.chrome?.storage?.local
  } = {}) {
    if (!sidePanel?.open || !scripting?.executeScript || !storageLocal?.get) {
      throw new Error('Toolbar launch requires side panel, scripting, and local storage APIs.');
    }
    this.sidePanel = sidePanel;
    this.scripting = scripting;
    this.storageLocal = storageLocal;
  }

  launch(tab) {
    if (!Number.isSafeInteger(tab?.id)) {
      return Promise.resolve({ panelOpened: false, injected: false, reason: 'missing_tab' });
    }

    let panelPromise;
    try {
      panelPromise = Promise.resolve(this.sidePanel.open({ tabId: tab.id }));
    } catch (error) {
      panelPromise = Promise.reject(error);
    }
    const preparationPromise = this.#preparePage(tab);

    return Promise.allSettled([panelPromise, preparationPromise]).then(([panelResult, preparationResult]) => ({
      panelOpened: panelResult.status === 'fulfilled',
      injected: preparationResult.status === 'fulfilled' && preparationResult.value.injected,
      reason: preparationResult.status === 'fulfilled'
        ? preparationResult.value.reason
        : 'preparation_failed'
    }));
  }

  async #preparePage(tab) {
    if (!isScriptablePageUrl(tab.url)) return { injected: false, reason: 'unsupported_page' };
    const stored = await this.storageLocal.get([PRIVACY_CONSENT_KEY]);
    if (!isCurrentPrivacyConsent(stored?.[PRIVACY_CONSENT_KEY])) {
      return { injected: false, reason: 'privacy_consent_required' };
    }
    await this.scripting.executeScript({
      target: { tabId: tab.id, frameIds: [0] },
      files: [PAGE_RUNTIME_FILE]
    });
    return { injected: true, reason: 'ready' };
  }
}
