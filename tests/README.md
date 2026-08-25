# Tests

Project tests, grouped by domain. Run the full chain from the repo root with
`npm test` (see `package.json` for per-suite entry points such as
`test:starfall:systems`, `test:chatbot-stream`, etc.).

| Directory | Area | Files |
|---|---|---|
| `project-starfall/` | Starfall game: balance, combat, maps, classes, skill FX, boss/loop behavior | `project-starfall-*.js`, `balance-harness.js` |
| `tools/` | Site tooling & accounts: chatbot proxy, campaign tracker, transcribe, text-compare, UTM, QR, tools auth, job-app bridge/attachments | `*.test.js` |
| `site/` | Cross-site contracts: resume/portfolio recommendations, responsive density | `*.test.js` |
| `infra/` | AWS: credentials, data migrations | `*.test.js` |

Notes:
- Files that read repo sources anchor the root via `path.resolve(__dirname, '..', '..')` (two levels up from their subdirectory). Keep that depth when moving a file.
- `project-starfall-balance-harness.js` is shared by the Starfall map/loop tests and `build/analyze-project-starfall-balance.js`.
