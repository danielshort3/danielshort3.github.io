# PixiJS CSP support

Unmodified official PixiJS 8.18.1 distribution, matching `pixi.min.js`.

- Source: https://github.com/pixijs/pixijs/releases/download/v8.18.1/unsafe-eval.min.js
- License: MIT (retained in the distributed header).
- SHA-256: `4bbae0dceca43ad8f2e456ee37d39f87f5afd71c5287e4abc2bc558cd373edd8`

Despite its package name, this module replaces dynamic shader/uniform code generation with interpreted synchronization functions. Load it after Pixi and before the Starfall bundle; no CSP relaxation is required.
