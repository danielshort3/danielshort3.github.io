# Repository Guidelines

## Project Structure & Module Organization
Source HTML lives at the root (`index.html`, `portfolio.html`, `contact.html`), while assets stay under `css/`, `js/` (grouped by area like `js/navigation/`), `img/`, and `documents/`. Build helpers sit in `build/` (`build-css.js`, `copy-to-public.js`, `resize_logo.py`). The CSS bundle lands in `dist/styles.[hash].css`, and deployable artifacts are mirrored into `public/`. Add new asset folders only after updating `build/copy-to-public.js` so they are included in the deployment.

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