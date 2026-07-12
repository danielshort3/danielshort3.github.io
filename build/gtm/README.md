# Google Tag Manager activity container

`GTM-MX6DNH8L-activity.json` is the importable source of truth for the site's GTM web container.

It contains the base GA4 Google tag plus 15 activity tags covering discovery, navigation, directories, portfolio depth, career intent, tools, games, contact intent, lead generation, chatbot usage, site search, content engagement, and reliability.

Regenerate the import file after changing `generate-activity-container.js`:

```powershell
node build/gtm/generate-activity-container.js
```

Import it into the Default Workspace with **Merge** and **Overwrite conflicting tags, triggers, and variables**. Preview and validate consent before publishing.

Privacy rules:

- Do not send contact-form contents, names, email addresses, chatbot prompts or answers, search terms, uploaded filenames, saved-session contents, or free-form error messages.
- Keep event values to curated IDs, enums, booleans, counts, and low-cardinality buckets.
- Let GA4 Enhanced Measurement own page-load views, outbound clicks, file downloads, form submissions, and engagement time. The custom container owns semantic directory depth, project/resource selection, resume intent, and tool outcomes.
- `directory_depth_reached` routes through **Directory Behavior** and `tool_run_error` routes through **Tool Activation**.

GA4 web-stream settings are not included in a GTM container import. An Analytics Admin or Editor must configure them separately:

1. In **Enhanced Measurement**, disable **Site Search** so raw `q` values are not collected as `view_search_results` terms.
2. In **Page Views → Show advanced settings**, keep browser page-load tracking enabled and disable page changes based on browser history events. The site emits its own semantic interaction events and does not need `replaceState`-driven page views.
3. Enable email redaction and redact the query parameters `q`, `code`, `state`, `session`, `cfg`, and `povcfg`. Keep standard UTM attribution parameters available.
4. After collecting enough validation traffic, review which business outcomes should be marked as key events and register only the low-cardinality custom dimensions needed for reporting.
