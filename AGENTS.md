# Repository Guidelines

## Project Structure & Module Organization
Source HTML lives at the root (`index.html`, `portfolio.html`, `contact.html`), while assets stay under `css/`, `js/` (grouped by area like `js/navigation/`), `img/`, and `documents/`. Build helpers sit in `build/` (`build-css.js`, `copy-to-public.js`, `resize_logo.py`). The CSS bundle lands in `dist/styles.[hash].css`, and deployable artifacts are mirrored into `public/`. Add new asset folders only after updating `build/copy-to-public.js` so they are included in the deployment.

## Templates & Scaffolding
Use the no-deps templates under `build/templates/` to keep new work consistent:
- Tools: `build/templates/tool-page.template.html` + `build/templates/tool-script.template.js`
- Site pages: `build/templates/page.template.html`
- Portfolio projects: `build/templates/project-entry.template.js`

Scaffold helpers:
- `npm run scaffold -- tool <tool-id> "<Tool Name>" --eyebrow "..." --description "..."`
- `npm run scaffold -- page <slug> "<Page Title>" --description "..."`
- `npm run scaffold -- project <project-id> "<Project Title>"` (prints a snippet to paste)

After scaffolding:
- Add clean-URL rewrites in `vercel.json` (new pages/tools will 404 without them).
- If it’s a tool:
  - Add a card in `pages/tools.html`.
  - Add to `TOOL_CATALOG` in `js/accounts/tools-account-ui.js` so session history shows a friendly name/link.

## Adding Pages (Vercel Rewrites + CSP)
This repo relies on `vercel.json` rewrites for clean URLs (examples: `/tools/text-compare` → `/pages/text-compare`).
- New tool route: add both `/tools/<toolId>` and `/tools/<toolId>.html` rewrites.
- New site route under `pages/`: add both `/<slug>` and `/<slug>.html` rewrites.

If a tool calls new external APIs, update the CSP `connect-src` allowlist in `vercel.json` or requests will fail in production.

## Tools Account System (AWS-backed)
All tools share the same Cognito login + per-tool session storage:
- Frontend auth/config: `js/accounts/tools-config.js`, `js/accounts/tools-auth.js`
- Frontend storage API wrapper: `js/accounts/tools-state.js`
- Shared UI shell (hero + account bar + modals): `js/accounts/tools-account-ui.js`
- Backend endpoints: `api/tools/me.js`, `api/tools/state.js`, `api/tools/dashboard.js`, `api/tools/activity.js`

Tool integration requirements:
- Tool pages should include `.skip-link`, `<main id="main">`, the `.tools-hero` section, and the `[data-tools-account="dock"]` container (see existing tools or the templates).
- Tool pages should include the tools account scripts (and keep the tool script loaded before `js/accounts/tools-account-ui.js` so capture/restore hooks are registered in time).
- Session capture: listen for `tools:session-capture` on `#main` and populate `detail.payload.outputSummary`, `detail.snapshot.output`, and `detail.snapshot.inputs` for “Recent sessions” previews.
- Mark “dirty” when output changes without a form input event by dispatching `tools:session-dirty` (useful for button-driven tools).
- Secrets: session serialization skips `input[type="password"]`, `file`, `hidden`, etc. Use `type="password"` for any sensitive fields you never want stored.

Session metadata:
- Users can now set session `title`, `tags`, `note`, and `pinned` via the Session modal.
- Backend support is `PATCH /api/tools/state` (see `api/tools/state.js` and `js/accounts/tools-state.js`).

## Build, Test, and Development Commands
Run `npm run build:css` to compose all `css/styles.css` imports into the hashed bundle. `npm run build:icons` regenerates favicons from `img/ui/logo.png`. `npm run build` runs the CSS build and copies site assets into `public/`. Lightweight project checks live in `npm test`. After any build, open `public/index.html` (or the root files) in a browser or serve `public/` through any static server for validation.

## Coding Style & Naming Conventions
Use 2-space indentation everywhere; HTML attributes stay in double quotes. JavaScript is ES2015+, with camelCase variables/functions, kebab-case filenames, and uppercase constants. End every JS statement with semicolons and keep concise semantic selectors; behavior hooks should rely on data attributes. CSS honors the existing `@layer` order (`tokens` → `base` → `layout` → `components` → `utilities` → `overrides`), so add new component styles under `css/components/` and import them from `css/styles.css`.

## Testing Guidelines
`npm test` runs `test.js`, which enforces markup contracts (e.g., `.skip-link`, `<main id="main">`), checks data arrays, and timing expectations such as `START_TIMEOUT_SEC ≥ 600` in `chatbot-demo.html`. Keep additions fast, dependency-free, and expand the tests whenever new pages or requirements arise.

## Commit & Pull Request Guidelines
Follow clear imperative subjects (e.g., “Update hero copy”); prefixes like `feat:` or `fix:` are welcome but optional. Before pushing, run `npm run build && npm test` and describe any visual changes or build-script updates. Pull requests should link issues, summarize scope, and attach screenshots for UI tweaks. Keep diffs focused and mention if build tooling or deployment assets changed.

## Security & Configuration Tips
Never commit secrets or API keys; analytics helpers belong in `js/analytics/`. Deployment reads from `public/` via `vercel.json`, so ensure `robots.txt` and `sitemap.xml` remain current, especially when adding pages or blocking crawlers.

## AWS Notes & Debugging Checklist
The tools account backend is designed to run on AWS (Cognito + DynamoDB), with optional KV fallback.

Tools accounts env vars (serverless / Vercel):
- Auth verification: `TOOLS_COGNITO_ISSUER`, `TOOLS_COGNITO_CLIENT_ID` (see `api/_lib/cognito-jwt.js`)
- DynamoDB storage: `TOOLS_DDB_TABLE` (or `TOOLS_DDB_TABLE_NAME`), plus `AWS_REGION` / `AWS_DEFAULT_REGION`
- Credentials: prefer `TOOLS_AWS_ACCESS_KEY_ID` + `TOOLS_AWS_SECRET_ACCESS_KEY` (optional `TOOLS_AWS_SESSION_TOKEN`)

Short links env vars:
- `SHORTLINKS_ADMIN_TOKEN` (dashboard + health check)
- DynamoDB table config lives in `api/_lib/short-links-store.js`

Warm-up / health-check methodology:
- Tools with cold-starting backends should expose a cheap “is it alive?” check and show user-facing status during warm-up (see `js/tools/whisper-transcribe-monitor.js` and `/api/short-links/health` + `js/admin/short-links.js`).
- When debugging AWS connectivity from a terminal, start with `aws sts get-caller-identity`, then `aws dynamodb describe-table --table-name <name> --region <region>`.
