# Google Tag Manager activity container

`GTM-MX6DNH8L-activity.json` is the importable source of truth for the site's GTM web container.

It contains the base GA4 Google tag plus 17 event tags covering discovery, navigation, directories, portfolio depth, career intent, tools, games, contact intent, lead generation, chatbot usage, site search, content engagement, reliability, and virtual page views.

Regenerate the import file after changing `generate-activity-container.js`:

```powershell
node build/gtm/generate-activity-container.js
node tests/site/gtm-container.test.js
```

Import it into the Default Workspace with **Merge** and **Overwrite conflicting tags, triggers, and variables**. Preview and validate consent before publishing.

Privacy rules:

- Do not send contact-form contents, names, email addresses, chatbot prompts or answers, search terms, uploaded filenames, saved-session contents, or free-form error messages.
- Keep event values to curated IDs, enums, booleans, counts, and low-cardinality buckets.
- Let the initial Google tag own page-load views, and Enhanced Measurement own outbound clicks, file downloads, form submissions, and engagement time. The custom container owns completed client-side route changes, semantic directory depth, project/resource selection, resume intent, and tool outcomes.
- `virtual_page_view` routes through **Virtual Page View** as the GA4 event `page_view`. Its separate event settings use `page_location`, `page_title`, `page_referrer`, `page_id`, and `audience`, without carrying previous activity fields into the page view. The site supplies the completed route's URL/title and preceding page URL together in the same data-layer event; it must not emit this event for initial page load, same-page changes, or failed navigation.
- The initial Google tag also reads `page_id` and `audience` from the site's synchronous context push so the first page view has the same reporting dimensions.
- `directory_depth_reached` routes through **Directory Behavior**, `tool_run_error` through **Tool Activation**, and `game_milestone` through **Game Activation**.

QA traffic settings:

- The site sets `analytics_debug: true` and `traffic_type: 'internal'` before GTM loads for opted-in local testing or explicitly marked QA traffic (`utm_source=qa` or `analytics_debug=1`). The initial Google tag and every custom event map these values to GA4 `debug_mode` and `traffic_type`.
- Ordinary traffic explicitly clears both data-layer keys with JavaScript `undefined`; the data-layer variables have no default. Do not send `false` or the string `'undefined'` for `debug_mode`: [Google requires omitting this parameter to disable debug mode](https://support.google.com/analytics/answer/7201382?hl=en).
- The site must provide complete page context on every `virtual_page_view` and clear absent per-event fields. GTM's data model persists values between pushes, so omitting a previously populated key would retain its old value.
- Configure GA4 data filters separately to exclude developer/internal traffic from ordinary reports. Marking traffic does not activate a reporting filter by itself. Consent is still required before collection.

GA4 web-stream settings are not included in a GTM container import. An Analytics Admin or Editor must configure them separately:

1. In **Enhanced Measurement**, disable **Site Search** so raw `q` values are not collected as `view_search_results` terms.
2. In **Page Views → Show advanced settings**, keep browser page-load tracking enabled and disable page changes based on browser history events. The site emits `virtual_page_view` after a successful client-side route change, so history tracking would duplicate those views and count incidental `replaceState` changes. See [Google's single-page application guidance](https://developers.google.com/analytics/devguides/collection/ga4/single-page-applications).
3. Enable email redaction and redact the query parameters `q`, `code`, `state`, `session`, `cfg`, and `povcfg`. Keep standard UTM attribution parameters available.
4. After collecting enough validation traffic, review which business outcomes should be marked as key events and register only the low-cardinality custom dimensions needed for reporting.

## Web Vitals (local configuration; publish separately)

The consent-gated web_vital event carries only metric_name (LCP, INP, CLS), numeric metric_value, metric_rating, page_path, page_id and audience. LCP/INP values use milliseconds; CLS is unitless. Context is frozen at the document entry route; tab changes do not start new page-load measurements. No DOM attribution, queries, account identifiers or tool contents are collected. Withdrawal suppresses further document metrics; a new document is required after re-consenting.

The dedicated Web Vitals tag/settings prevent stale activity fields from entering these hits. Create event-scoped dimensions for metric_name, metric_rating and page_path, and a numeric metric for metric_value. In BigQuery or a reporting tool, group by metric_name and entry page_path and calculate the 75th percentile from the latest report per page visit/metric. Do not sum values or combine CLS with millisecond metrics. GA4 collection must have sufficient consenting visits before making real-user claims; the local Lighthouse report measures lab TBT, not INP.

Validate locally with test:recovery and test:analytics before importing. Run Tag Assistant preview and check consent denial, grant, withdrawal and the dedicated event parameters before any separately approved publication. This repository update does not publish GTM or configure GA4 dimensions.
