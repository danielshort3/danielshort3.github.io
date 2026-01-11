# Templates (Scaffolding)

These templates exist to keep new pages/tools/projects consistent with the repo conventions and `npm test` contracts.

## Quick start

- Scaffold a new tool page + JS file:
  - `node build/scaffold.js tool <tool-id> "<Tool Name>" --eyebrow "<Tool Name> - One-line summary."`
- Scaffold a new non-tool page under `pages/`:
  - `node build/scaffold.js page <slug> "<Page Title>" --description "One-line meta description."`
- Print a project entry snippet (paste into `js/portfolio/projects-data.js`):
  - `node build/scaffold.js project <project-id> "<Project Title>"`

## After scaffolding (manual checklist)

- Add a clean-URL rewrite for the new page in `vercel.json`.
- If it’s a tool:
  - Add it to the tools index in `pages/tools.html`.
  - Add it to the dashboard catalog in `js/accounts/tools-account-ui.js` (`TOOL_CATALOG`).
- If the tool calls external APIs:
  - Update the `connect-src` allowlist in `vercel.json` (CSP).
- Run `npm test` and `npm run build`.

