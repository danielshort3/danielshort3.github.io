import { applicantProAdapter } from './applicantpro.js';
import { ashbyAdapter } from './ashby.js';
import { genericAdapter } from './generic.js';
import { greenhouseAdapter } from './greenhouse.js';
import { leverAdapter } from './lever.js';
import { PLATFORM_PROFILE_ADAPTERS } from './platform-profiles.js';
import { smartRecruitersAdapter } from './smartrecruiters.js';
import { workdayAdapter } from './workday.js';

export const ATS_ADAPTERS = Object.freeze([
  applicantProAdapter,
  greenhouseAdapter,
  leverAdapter,
  ashbyAdapter,
  smartRecruitersAdapter,
  workdayAdapter,
  ...PLATFORM_PROFILE_ADAPTERS,
  genericAdapter
]);

export const selectAdapter = ({ doc, url } = {}) => ATS_ADAPTERS.find((adapter) => {
  try {
    return adapter.matches({ doc, url });
  } catch {
    return false;
  }
}) || genericAdapter;

export {
  applicantProAdapter,
  ashbyAdapter,
  genericAdapter,
  greenhouseAdapter,
  leverAdapter,
  smartRecruitersAdapter,
  workdayAdapter
};

export { PLATFORM_PROFILE_ADAPTERS };
export * from './platform-profiles.js';
