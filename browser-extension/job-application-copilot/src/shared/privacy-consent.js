export const PRIVACY_CONSENT_KEY = 'jobCopilotPrivacyConsentV1';
export const PRIVACY_NOTICE_VERSION = 4;

export const isCurrentPrivacyConsent = value => Boolean(
  value
  && value.version === PRIVACY_NOTICE_VERSION
  && typeof value.acceptedAt === 'string'
  && Number.isFinite(Date.parse(value.acceptedAt))
);
