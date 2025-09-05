# Repository Guidelines

## Project Structure & Module Organization
- Root HTML: `index.html`, `portfolio.html`, `contact.html`, etc.
- Assets: `css/` (modular layers and components), `js/` (by area: `common/`, `navigation/`, `portfolio/`, etc.), `img/`, `documents/`.
- Build output: `dist/styles.css` (bundled from `css/styles.css`), and deployable site in `public/` (generated).
- Build scripts: `build/build-css.js`, `build/copy-to-public.js`, `build/resize_logo.py`.
- Adding new asset folders? Update `build/copy-to-public.js` so they are copied to `public/`.

## Build, Test, and Development Commands
- Node ≥ 18 required. Python 3 + Pillow for icon tasks.
- `npm run build:css`: Bundle CSS imports into `dist/styles.css`.
- `npm run build:icons`: Generate favicon/UI icons from `img/ui/logo.png`.
- `npm run build`: Build CSS and copy site + assets into `public/`.
- `npm test`: Run lightweight assertions in `test.js` (HTML snippets, script sanity, data arrays, chatbot demo timer).
- Local preview: open `index.html` (or `public/index.html` after build) in a browser, or serve `public/` with a static server.

## Coding Style & Naming Conventions
- Indent with 2 spaces; keep HTML attributes in double quotes.
- JavaScript: ES2015+, no frameworks, end statements with semicolons; functions/vars `camelCase`, files `kebab-case`; constants `UPPER_SNAKE`.
- CSS: use the existing `@layer` order (`tokens, base, layout, components, utilities, overrides`). New component styles go in `css/components/*.css` and are imported from `css/styles.css`.
- Keep selectors small and classes semantic; prefer data attributes for behavior hooks.

## Testing Guidelines
- `npm test` validates shared markup (e.g., `.skip-link`, `<main id="main">`, `og:image`) and evaluates core scripts.
- If you add pages or change requirements, update `test.js` accordingly (e.g., keep `START_TIMEOUT_SEC ≥ 600` in `chatbot-demo.html`).
- Keep tests fast and dependency-free.

## Commit & Pull Request Guidelines
- History is informal; use clear, imperative subjects (e.g., "Update hero copy"). Optional prefixes like `feat:`/`fix:` are welcome.
- PRs: include a concise description, linked issues, and screenshots for visual changes. Run `npm run build && npm test` before submitting. Keep diffs focused and note any build-script updates.

## Security & Configuration Tips
- Do not embed secrets or API keys in client code. Analytics helpers live under `js/analytics/`.
- Deployments read from `public/` (see `vercel.json`). Ensure `robots.txt` and `sitemap.xml` remain accurate.

