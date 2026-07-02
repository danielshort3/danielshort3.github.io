# Slot Machine Demo Archive

This folder preserves the old slot-machine prototype as local-only reference material.

It is not part of the public site, is not linked from the games page, and is not run by the default `npm test` suite. The public probability game is `probability-engine.html`.

Archived layout:

- `demo.html`: original standalone demo page
- `assets/`: original slot art
- `config/`: original client-side machine and drop configuration
- `aws-function/`: original AWS Lambda source for the server-backed prototype
- `slot-machine-demo.test.js`: original local test harness for the archived demo

Generated deployment archives such as `aws/slot-machine-function.zip` are intentionally not kept. Rebuild a deployment bundle from `aws-function/` if this prototype is ever revived.
