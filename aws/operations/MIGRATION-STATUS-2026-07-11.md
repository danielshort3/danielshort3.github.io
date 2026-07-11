# Website AWS migration status — 2026-07-11

This file is the secret-free handoff for the July 11 website migration. It records what is live, the rollback anchors, and the actions that must wait for observation or user verification. Do not add credentials, tokens, exported environment values, or Cognito passwords.

## Recovery anchors

- Production baseline commit: `a49c159b7a80b47fb3c648f66ea4fff2c58eca12`
- Pushed production tag: `pre-aws-migration-2026-07-11`
- Snapshot branch and commit: `codex/pre-migration-snapshot` at `e6f7ddaa`
- Local recovery directory: `C:\Users\clopt\Documents\coding\Backups\danielshort3.github.io\2026-07-11-pre-aws-migration`
- The recovery directory contains the repository bundle, binary patch, Lambda packages, secret-free AWS/Vercel manifests, DynamoDB backup and restore evidence, and a verified 33-file SHA-256 manifest.
- Six DynamoDB on-demand backups completed. The restore drill reproduced all 610 source items with the same full primary-key hash.

## Live application state

- Current Production deployment: `BBuzvA8HnoyBZ9KHna9fRX5MjVgN` (`website-4g6x2y19n-daniel-shorts-projects.vercel.app`), aliased to `www.danielshort.me` and `dshort.me`.
- Previous corrected OIDC-capable Production deployment: `E3GWNK3n12ECrc6bkLEakpB9PE6Q` (`website-1ui3ql31b-daniel-shorts-projects.vercel.app`).
- Stable Preview alias: `website-aws-migration-preview.vercel.app`.
- Production and Preview use exact Vercel OIDC subjects and separate workload roles. CloudTrail has recorded Production assumptions for Tools, Short Links, Transcribe, Chatbot DynamoDB, Chatbot Bedrock, and Demo Invoke.
- Contact delivery remains in `legacy` mode because the SES account is still sandboxed. The OIDC Contact role and direct SES code are deployed but are not the active delivery path.
- Static AWS keys remain present only as the staged rollback path. Do not remove or deactivate them before the observation gates below.

## Acceptance evidence

- Repository: `npm run build`, `npm test` (5,715 checks), `npm run test:all` (42,034 checks), and `npm audit` all pass.
- AWS templates: all nine templates pass AWS CloudFormation validation; the deployed template revisions also passed `cfn-lint 1.53.0`.
- Lambda inference: all ten Preview aliases and all ten Production aliases pass real inference/data canaries.
- Direct exposure: all twelve retained Function URLs use `AWS_IAM`; anonymous requests return `403`.
- IAM boundaries: positive simulations allow each workload's intended resource, while cross-workload and Preview-to-Production simulations are implicitly denied.
- Authenticated canaries: Tools state CRUD/isolation, Job Tracker CRUD/attachments/cross-user denial, Short Links CRUD/redirect/click history/admin enforcement, Transcribe job cleanup, and Chatbot streaming/logging/rate limiting pass.
- SEO: all 38 sitemap URLs return `200` directly, are self-canonical, and are indexable. Documents and audience query variants return `X-Robots-Tag: noindex, nofollow`; `/admin` remains a noindex `404`.
- Search Console accepted priority crawl requests for the homepage, Portfolio, Tools, Games, Contact, and the five priority project pages.
- The SSA Baby Names source loaded successfully through a real browser GET; its prior HEAD-only `403` was not treated as an outage.

## Required follow-ups

### User verification

1. Complete the AWS IAM Identity Center invitation/setup email and enroll MFA.
2. Confirm the SNS security/operations email subscription, which is currently `PendingConfirmation`.
3. Verify the SES recipient identity and request SES Production access. After approval, run a Preview contact canary, set `CONTACT_DELIVERY_MODE=ses`, and verify Production before removing the legacy contact path.
4. Sign in with `website-admin` and verify it before deleting the temporary `CodexIdentityCenterBootstrapRole`.

### Observation gates

1. After at least 72 healthy hours, verify CloudTrail role attribution, Vercel runtime errors, alarms, DynamoDB counts, and authenticated canaries. Only then deactivate shared/workload IAM keys and keep `AWS_AUTH_MODE=legacy` available for rollback.
2. After seven healthy days, remove static Vercel credential variables and the legacy credential code path.
3. Retain the disabled IAM keys through day 14, then delete the keys. Leave disabled IAM users for a later explicit retirement decision.
4. Keep obsolete VGJ APIs, the route-less duplicate API, archived Slot Machine, and legacy Whisper resources quarantined for at least 30 days. Do not permanently delete them in this rollout.
5. The `website-ecr-backup-retirement-20260711` stack schedules removal of exactly the 80 `backup-20260711-*` tags on `2026-08-11T00:00:00Z`. Verify CloudTrail and tag inventories before deleting that stack.
6. Review GuardDuty's measured cost and findings by trial day 25. The `$5` security-services budget alerts at 50%, 80%, and 100%.

### Capacity and image remediation

- The account's current Lambda regional concurrency quota is 10. Do not set reserved concurrency at this level; it could starve other functions. Re-request a quota of 200 through the account's supported quota path before applying the planned reservations.
- ECR scan-on-push, immutable release tags, digest-pinned deployments, and lifecycle policies are active. Current scans contain no critical findings, but several images have high-severity findings and must be rebuilt before they are considered remediated. The Shape image currently has the largest high-severity count.

## Rollback

If an OIDC workload regresses during the observation window:

1. Select the previous corrected OIDC deployment above, or restore the tagged baseline deployment if the regression is unrelated to OIDC.
2. For a credential-path failure only, set the affected deployment to `AWS_AUTH_MODE=legacy` while the retained key remains active.
3. For an AWS Lambda regression, move that function's `live` alias back to the recorded immutable version; do not mutate or delete the prior version.
4. Restore a DynamoDB table only from the recorded on-demand backup after comparing counts and sample/full key hashes.
5. Record the incident and new evidence before attempting the cutover again.
