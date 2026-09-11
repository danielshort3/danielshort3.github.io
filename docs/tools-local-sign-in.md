# Tools sign-in on localhost

Run the repository server on a registered port, for example:

```powershell
npm run dev -- --port 4181
```

Tools sign-in and sign-out return to the current origin plus `/tools/dashboard`. Cognito must allow that exact URL in both `CallbackURLs` and `LogoutURLs`, including scheme, hostname, port, and path. `localhost` and `127.0.0.1` are separate origins; use the same one throughout sign-in. If the server chooses another port because the requested port is busy, register the actual port printed by the server.

## Register local callbacks and sign-out URLs

Use the installed AWS CLI with a profile permitted to describe and update the existing Cognito app client. Preview the additions first:

```powershell
node scripts/setup-tools-local-auth.js
```

The defaults come from this site's public configuration: account `886623862678`, region `us-east-2`, pool `us-east-2_y80EG3pKd`, and client `78oo663obb0t28u63u9bqn00o9`. The helper adds `/tools/dashboard` callback and sign-out URLs for both `http://localhost` and `http://127.0.0.1` on ports `3000`, `4173`, and `4181`. Run it again if you used the older callback-only helper; it will add the missing sign-out URLs.

Apply the reviewed additions:

```powershell
node scripts/setup-tools-local-auth.js --apply
```

For a named profile or another local port:

```powershell
node scripts/setup-tools-local-auth.js --profile your-profile --ports 4182
node scripts/setup-tools-local-auth.js --profile your-profile --ports 4182 --apply
```

`--ports` controls which local URLs to add; existing callbacks and sign-out URLs remain in their respective merged lists. The helper checks the AWS account before writing, reads the full current client, checks its mutable fields against the installed CLI's input schema, and changes only `CallbackURLs` and `LogoutURLs`. It checks again for configuration drift before the update, verifies the result afterward, and never prints the client secret or full AWS response. Repeating an already-applied command performs no update.

When updating the CloudFormation stack, preserve the full live callback and sign-out lists in its `CallbackUrls` and `LogoutUrls` parameters, including any local URLs you still use. The template accepts both lists explicitly; its defaults do not register local origins.

Do not replace this with a partial `update-user-pool-client` command containing only callback URLs. Cognito resets omitted settings to defaults, including settings needed for OAuth. The update must carry the existing client configuration. [AWS UpdateUserPoolClient documentation](https://docs.aws.amazon.com/cognito-user-identity-pools/latest/APIReference/API_UpdateUserPoolClient.html)

## Configure the local backend

The callback allowlist and local API configuration are separate requirements. Set these in the ignored repository-root `.env.local`, using the same Cognito client as the browser:

```dotenv
TOOLS_COGNITO_ISSUER=https://cognito-idp.us-east-2.amazonaws.com/us-east-2_y80EG3pKd
TOOLS_COGNITO_CLIENT_ID=78oo663obb0t28u63u9bqn00o9
TOOLS_SESSION_SECRETS=<your-local-32-byte-session-key>
TOOLS_AUTH_BEARER_FALLBACK=true
```

Generate a local session key with `node -e "console.log(require('node:crypto').randomBytes(32).toString('hex'))"` and store it only in `.env.local`. Saved tool sessions also need `TOOLS_DDB_TABLE`; Links & QR Codes needs its short-link and click-table settings from `.env.example` and local AWS credentials permitted to access those tables. Use local AWS credentials locally; leave Vercel OIDC role settings out of the local environment. Restart the dev server after configuration changes.

For this site's existing account history and Links & QR Codes library, configure their regions separately:

```dotenv
TOOLS_AWS_REGION=us-east-2
TOOLS_DDB_TABLE=danielshort-tools-accounts
SHORTLINKS_AWS_REGION=us-east-2
SHORTLINKS_DDB_TABLE=danielshort-short-links
SHORTLINKS_DDB_CLICKS_TABLE=danielshort-short-links-clicks
```

These are the shared live account and link tables; local edits affect that same library. Each service-specific region takes precedence over `AWS_REGION`, then `AWS_DEFAULT_REGION`. This matters because the local environment loader preserves inherited shell settings: a shell using `AWS_REGION=us-east-1` would otherwise send storage requests to the wrong region, even when `.env.local` specifies `us-east-2`.

The Job Application Tracker uses the repository development server's `/api/job-tracker` proxy on loopback hosts. This sends only the tracker's supported API requests to its fixed AWS endpoint and preserves the signed-in user's bearer token. No cloud CORS changes or extra AWS keys are needed for that connection. The deployed site keeps its configured API endpoint. If only a local cookie session was restored, sign in again to obtain the ID token required by the tracker. Tracker data is the same account data used on the live site; attachment transfers continue to use their signed S3 URLs.

The tracker attachment bucket separately permits `http://localhost` and `http://127.0.0.1` on ports `4173` and `4181`. These four exact origins are included in the bucket CORS configuration in `aws/job-application-tracker/template.yaml`; the API Gateway origin allowlist stays separate. Other local ports can display tracker data through the proxy, but signed attachment transfers require a matching bucket origin. The bucket remains private and each transfer still needs its signed URL.

## Check the result

Open `/tools/dashboard` on the same local origin, choose Sign in, and complete account selection. The browser should return to that local dashboard, then restore the requested tool. A `redirect_mismatch` page indicates that the exact callback is still missing. An API configuration error after returning indicates missing backend variables. An access-denied response from link management requires an authorized account; registering callbacks does not grant administrator access.

Then sign out and sign in again. Sign-out must pass through Cognito's `/logout` endpoint to clear its hosted UI session cookie, and the next sign-in should show the Google/email choices. Cognito validates `logout_uri` against `LogoutURLs` independently of the callback list. [AWS logout documentation](https://docs.aws.amazon.com/cognito/latest/developerguide/logout-endpoint.html)

## Transcription on localhost

Amazon Transcribe uses the existing preview bucket and ledger for local work. Set these service-specific values in ignored `.env.local`; the explicit table prevents transcription runs from using the live account-history table:

```dotenv
TRANSCRIBE_AWS_REGION=us-east-2
TRANSCRIBE_UPLOAD_BUCKET=danielshort-transcribe-tool-preview-886623862678-us-east-2
TRANSCRIBE_DDB_TABLE=danielshort-tools-accounts-preview
TRANSCRIBE_SIGNING_SECRET=<your-local-random-32-byte-key>
TRANSCRIBE_UPLOAD_PREFIX=tools-transcribe/
TRANSCRIBE_LEDGER_MODE=required
TRANSCRIBE_MAX_TOTAL_COST_USD=10
TRANSCRIBE_DAILY_COST_LIMIT_USD=10
TRANSCRIBE_GLOBAL_DAILY_COST_LIMIT_USD=10
TRANSCRIBE_DAILY_FILE_LIMIT=10
TRANSCRIBE_GLOBAL_DAILY_FILE_LIMIT=20
TRANSCRIBE_MAX_CONCURRENT=2
```

Generate the signing secret using the same cryptographic method as the local session key, with a different value. Keep the existing local AWS credential chain; do not copy deployed Vercel OIDC role settings. The preview bucket already permits uploads from localhost and `127.0.0.1`. These local settings require the ledger and cap both per-user and global daily reservations at $10. Starting an AWS transcription still incurs normal AWS charges. After restarting, `GET /api/tools/transcribe/config` should return `configured: true` and region `us-east-2`; this check does not start a job.

Home GPU processing uses its separate existing `LOCAL_TRANSCRIBE_SHARED_SECRET` and defaults to `http://127.0.0.1:8765` during local development. `GET /v1/health` on that worker reports readiness without creating a job. A missing `LOCAL_TRANSCRIBE_WORKER_ORIGIN` is valid when the worker runs on this computer.

## GA4 reports on localhost

The local server handles `/api/ga4/report`, but report access still uses the tool's existing admin-token form. Tools account sign-in does not replace that token. Configure only these GA4 settings in ignored `.env.local`:

```dotenv
GA4_ADMIN_TOKEN=<existing-authorized-GA4-admin-token>
GA4_PROPERTY_ID=<authorized-numeric-property-id>
GA4_SERVICE_ACCOUNT_JSON_B64=<existing-authorized-service-account-JSON-as-base64>
```

An authorized, ignored Vercel environment snapshot can supply those three existing values. Copy only the scoped GA4 entries, preserve other local settings, and never print credential values or import the full deployment environment. The configured property ID is automatically allowlisted; `GA4_ALLOWED_PROPERTY_IDS` is needed only for additional authorized properties.

Restart the server, enter the existing admin token in the GA4 tool, and choose its access check. This performs a bounded, read-only GA4 report. Verify success without logging report contents, tokens, or service-account credentials. A successful check confirms reporting access; it does not modify analytics settings or events.
