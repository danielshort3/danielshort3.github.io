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
- Let GA4 Enhanced Measurement own page views, outbound clicks, file downloads, form submissions, and engagement time.
