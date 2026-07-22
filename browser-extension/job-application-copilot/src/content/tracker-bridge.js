import {
  TRACKER_EXTENSION_SOURCE,
  TRACKER_MESSAGE_TYPES,
  TRACKER_PAGE_SOURCE,
  isTrackerPageUrl,
  validateTrackerWindowMessage
} from '../shared/tracker-protocol.js';

const PAGE_TO_EXTENSION_TYPES = new Set([
  TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST,
  TRACKER_MESSAGE_TYPES.FILE_CHUNK_REQUEST,
  TRACKER_MESSAGE_TYPES.COMPLETE,
  TRACKER_MESSAGE_TYPES.DISMISSED
]);

const EXTENSION_TO_PAGE_TYPES = new Set([
  TRACKER_MESSAGE_TYPES.MANIFEST,
  TRACKER_MESSAGE_TYPES.FILE_CHUNK,
  TRACKER_MESSAGE_TYPES.COMPLETE_ACK
]);

export const installTrackerBridge = ({ windowObject, runtime }) => {
  if (!windowObject || !runtime?.sendMessage || !isTrackerPageUrl(windowObject.location?.href)) return () => {};

  const onWindowMessage = async (event) => {
    if (event.source !== windowObject || event.origin !== windowObject.location.origin) return;
    if (event.data?.source !== TRACKER_PAGE_SOURCE) return;
    try {
      const request = validateTrackerWindowMessage(event.data, { expectedSource: TRACKER_PAGE_SOURCE });
      if (!PAGE_TO_EXTENSION_TYPES.has(request.type)) throw new Error('Tracker page cannot send this message type.');
      const response = await runtime.sendMessage(request);
      if (!response || !EXTENSION_TO_PAGE_TYPES.has(response.type)) return;
      const validatedResponse = validateTrackerWindowMessage(response, {
        expectedSource: TRACKER_EXTENSION_SOURCE
      });
      if (validatedResponse.captureId !== request.captureId
        || validatedResponse.requestId !== request.requestId
        || validatedResponse.channelNonce !== request.channelNonce) {
        throw new Error('Extension response does not match the tracker request.');
      }
      windowObject.postMessage(validatedResponse, windowObject.location.origin);
    } catch (error) {
      console.warn('Job Application Copilot rejected a tracker transfer message.', error);
    }
  };

  windowObject.addEventListener('message', onWindowMessage);
  return () => windowObject.removeEventListener('message', onWindowMessage);
};

if (typeof window !== 'undefined' && typeof chrome !== 'undefined') {
  installTrackerBridge({ windowObject: window, runtime: chrome.runtime });
}
