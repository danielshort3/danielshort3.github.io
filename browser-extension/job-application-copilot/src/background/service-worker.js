import { TrackerCaptureRouter } from './tracker-capture-router.js';
import { FieldSelectionRouter } from './field-selection-router.js';
import { ToolbarActionLauncher } from './action-launcher.js';

const FOUNDATION_VERSION = 1;
const trackerCaptureRouter = new TrackerCaptureRouter();
const fieldSelectionRouter = new FieldSelectionRouter();
const toolbarActionLauncher = new ToolbarActionLauncher();
const foundationReady = trackerCaptureRouter.initialize();
const actionBehaviorReady = chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: false })
  .catch(() => false);

chrome.runtime.onInstalled.addListener(async () => {
  await Promise.all([foundationReady, actionBehaviorReady]);
  await chrome.storage.local.set({ foundationVersion: FOUNDATION_VERSION });
});

chrome.action.onClicked.addListener((tab) => {
  void toolbarActionLauncher.launch(tab);
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (fieldSelectionRouter.ownsMessage(message)) {
    fieldSelectionRouter.handle(message, sender)
      .then(sendResponse)
      .catch(error => sendResponse({
        ok: false,
        error: {
          code: 'FIELD_SELECTION_REJECTED',
          message: error instanceof Error ? error.message : 'Field selection failed.'
        }
      }));
    return true;
  }
  if (!trackerCaptureRouter.ownsMessage(message)) return false;
  foundationReady.then(() => trackerCaptureRouter.handle(message, sender))
    .then(sendResponse)
    .catch(error => sendResponse({
      ok: false,
      error: {
        code: 'TRACKER_CAPTURE_REJECTED',
        message: error instanceof Error ? error.message : 'Tracker capture request failed.'
      }
    }));
  return true;
});
