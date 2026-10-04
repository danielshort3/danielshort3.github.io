# Contact sender

This directory defines the strict contact sender. It requires a matching signing proxy at `/api/contact`; publishing website source does not activate this Lambda. The proxy must validate the form and same-origin request, then sign the exact body, a pseudonymous client address, origin, timestamp and nonce using the shared protection helper. The strict sender accepts only valid, fresh signed requests from `ALLOWED_ORIGINS`. DynamoDB atomically reserves a replay nonce and per-client/global quotas before SES is called. Protection failures block delivery; the website keeps the draft.

Required Lambda environment variables:

- `CONTACT_PROXY_SECRET`: the same fresh random secret used by Vercel, at least 32 UTF-8 bytes. Never store it in source or deployment logs.
- `CONTACT_RATE_LIMIT_TABLE`: an on-demand DynamoDB table with string partition key `pk`, string sort key `sk`, and TTL enabled on `ttl`.
- `SENDER_EMAIL`, `RECIPIENT_EMAIL`: retain the established SES identities and recipient.
- `ALLOWED_ORIGINS`: explicit comma-separated website origins; no wildcard. The default is the two canonical HTTPS site origins.

Optional quotas default to `CONTACT_PER_MINUTE_LIMIT=2`, `CONTACT_PER_HOUR_LIMIT=5`, `CONTACT_PER_DAY_LIMIT=10`, and `CONTACT_GLOBAL_DAILY_LIMIT=100`. Quota/replay rows expire after two days and contain no message, email or raw client address. Set appropriate separate tables/secrets/origins if adding a preview sender.

The sending Lambda role needs `dynamodb:PutItem` and `dynamodb:UpdateItem` on that table, plus `ses:SendEmail` restricted to the verified sender identity and `ses:FromAddress`. Use `ses-policy.template.json` with the exact region, account and sender email; this code does not require `ses:SendRawEmail`. `rate-limit-policy.template.json` is the additional table policy; replace its placeholder with the exact table ARN. DynamoDB transaction elements use their underlying action permissions, not a standalone `dynamodb:TransactWriteItems` IAM action.

Build and tests:

```powershell
npm.cmd --prefix aws/contact-function ci --ignore-scripts
node tests/infra/contact-protection.test.js
node tests/site/contact-delivery.test.js
node build/package-contact-lambda.cjs --output=C:/outside-the-repository/contact-lambda.zip
```

The build bundles the actual source, shared protection helper and pinned AWS SDK v3 clients into `index.js`; the ZIP retains the `index.handler` entry point on Node 24. It does not deploy anything. The package command uses the repository's existing esbuild dependency and a temporary staging directory outside the checkout.

Rollout order: provision the table, IAM and matching secret first; deploy Vercel's signing proxy while the old Lambda still accepts normal submissions; verify the proxy deployment and its rejection of invalid form submissions; then deploy the strict Lambda. The offline proxy test verifies the signature of the actual forwarded request. After strict deployment, a correctly signed invalid payload should return 400 before reserving quota or calling SES. Verify direct unsigned, foreign-origin and expired/tampered requests return 403. Automated verification must never send a valid real contact message.
