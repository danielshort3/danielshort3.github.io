# Amazon Transcribe Tool Setup

The website file transcription tool uses Vercel API routes to presign private S3 uploads and start Amazon Transcribe batch jobs. Users must be signed in with the existing tools account before paid jobs can start.

## Environment variables

Set these in Vercel:

```bash
AWS_REGION=us-east-2
AWS_AUTH_MODE=oidc
AWS_OIDC_AUDIENCE=sts.amazonaws.com
TRANSCRIBE_AWS_REGION=us-east-2
TRANSCRIBE_AWS_ROLE_ARN=<environment-specific-vercel-oidc-role-arn>
TRANSCRIBE_UPLOAD_BUCKET=<private-upload-bucket>
TRANSCRIBE_UPLOAD_PREFIX=tools-transcribe/
TRANSCRIBE_SIGNING_SECRET=<long-random-secret>
TRANSCRIBE_PRICE_PER_SECOND=0.0001
TRANSCRIBE_MAX_FILES_PER_RUN=10
TRANSCRIBE_MAX_FILE_BYTES=524288000
TRANSCRIBE_MAX_TOTAL_COST_USD=100
TRANSCRIBE_LANGUAGE_CODE=en-US
TRANSCRIBE_RUN_TOKEN_TTL_SECONDS=86400
TRANSCRIBE_DDB_TABLE=<pk-sk-dynamodb-table>
TRANSCRIBE_DDB_TTL_ATTRIBUTE=ttl
TRANSCRIBE_LEDGER_MODE=required
TRANSCRIBE_DAILY_COST_LIMIT_USD=100
TRANSCRIBE_TRUSTED_COST_SUBS=<optional-comma-separated-owner-cognito-subs>
TRANSCRIBE_DAILY_FILE_LIMIT=50
TRANSCRIBE_GLOBAL_DAILY_COST_LIMIT_USD=100
TRANSCRIBE_GLOBAL_DAILY_FILE_LIMIT=200
TRANSCRIBE_MAX_CONCURRENT=2
TRANSCRIBE_LEDGER_START_LEASE_SECONDS=300
TRANSCRIBE_LEDGER_LEASE_SECONDS=43200
TRANSCRIBE_LEDGER_RENEWAL_SECONDS=60
TRANSCRIBE_LEDGER_RUN_RETENTION_DAYS=45
TRANSCRIBE_LEDGER_DAY_RETENTION_DAYS=45
TRANSCRIBE_HISTORY_RETENTION_DAYS=90
TRANSCRIBE_TRANSCRIPT_FETCH_TIMEOUT_MS=12000
TRANSCRIBE_MAX_TRANSCRIPT_FETCH_BYTES=26214400
TRANSCRIBE_MAX_TRANSCRIPT_BYTES=3145728
```

Use an exact production or preview trust subject on each OIDC role. Static `TRANSCRIBE_AWS_ACCESS_KEY_ID` and `TRANSCRIBE_AWS_SECRET_ACCESS_KEY` values remain available only for local development or a temporary rollback; remove them from Vercel after OIDC canaries pass. `TRANSCRIBE_AWS_SESSION_TOKEN` is used only with temporary `ASIA...` access keys.

`TRANSCRIBE_DDB_TABLE` may be omitted when `TOOLS_DDB_TABLE` (or the legacy `TOOLS_DDB_TABLE_NAME`) points at the existing tools table. The table must have string partition and sort keys named `pk` and `sk`. The ledger is required by default and fails closed before a paid job starts if it cannot reserve usage. `TRANSCRIBE_LEDGER_MODE=disabled` is accepted only outside production for explicit local testing; production ignores that opt-out.

The ledger atomically reserves each deterministic job against all three controls:

- the cost of AWS's hard 28,800-second media maximum against `TRANSCRIBE_DAILY_COST_LIMIT_USD`;
- one file against `TRANSCRIBE_DAILY_FILE_LIMIT`;
- one leased per-user slot out of `TRANSCRIBE_MAX_CONCURRENT`.

The same transaction also reserves global UTC-day cost and file counters. These use `TRANSCRIBE_GLOBAL_DAILY_COST_LIMIT_USD` and `TRANSCRIBE_GLOBAL_DAILY_FILE_LIMIT`, so creating additional Cognito accounts cannot multiply the deployment's daily spend ceiling.

Replaying the same signed quote reuses its run record and does not charge the counters twice. A short start-attempt lease prevents concurrent retries from racing a refund against a successful AWS start. Active status checks renew the longer concurrency lease, while `TRANSCRIBE_LEDGER_RENEWAL_SECONDS` limits how often the API persists repeated last-seen updates (60 seconds by default). If a process disappears, the leases eventually expire so capacity can recover. A confirmed start failure with no matching AWS job atomically refunds the daily cost and file count.

## S3 bucket

Create a private bucket in the same region as Transcribe and apply CORS similar to:

```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["https://www.danielshort.me", "https://danielshort.me"],
      "AllowedMethods": ["POST", "PUT"],
      "AllowedHeaders": ["content-type", "x-amz-*"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 300
    }
  ]
}
```

Every presigned POST policy enforces the quoted file byte count and content type, and requires a `tagging` form field containing S3 Tagging XML for `tool=amazon-transcribe` plus `retention=temporary`. Add a lifecycle rule filtered to that tag pair (and optionally the `tools-transcribe/` prefix) that expires objects after a few days. The app deletes uploaded media when jobs complete, fail, or are confirmed missing, but lifecycle expiration is the cleanup backstop.

Completed transcript text is copied to a deterministic private object under `tools-transcribe/<cognito-sub>/history/` and tagged `tool=amazon-transcribe` plus `retention=history`. It contains only UTF-8 transcript text; original media is never retained as history. Add a second lifecycle rule for the `retention=history` tag that expires objects after 90 days, matching `TRANSCRIBE_HISTORY_RETENTION_DAYS`. Keep the bucket private, enable default encryption, and block all public access. A signed-in user may also delete an individual saved transcript immediately through the history API.

The completed transcript response is streamed through the API with both a timeout and a byte ceiling. Only HTTPS transcript URLs on AWS domains are accepted. Increase the two transcript fetch limits deliberately if legitimate transcripts exceed the defaults.

## DynamoDB TTL and retention

Enable DynamoDB Time to Live on the ledger table using the numeric attribute configured by `TRANSCRIBE_DDB_TTL_ATTRIBUTE`. It defaults to `TOOLS_DDB_TTL_ATTRIBUTE` when present, then to `ttl`, so a shared tools table uses one consistent TTL attribute. Run records and UTC-day usage records default to 45 days of retention. When a transcript is saved, its run metadata and user-scoped history index are extended to `TRANSCRIBE_HISTORY_RETENTION_DAYS` (90 days by default) so the private S3 text object and its metadata expire together. Slot records also carry TTL, but correctness does not depend on prompt TTL deletion: every slot claim checks `leaseExpiresAt`, so delayed DynamoDB cleanup cannot permanently block a user.

History list cursors are signed and scoped to the authenticated Cognito subject. The API derives every DynamoDB partition and S3 object path from that verified subject; it never accepts a user-supplied partition or object key. List responses contain only the filename and bounded metadata. Transcript text is loaded only by the authenticated detail endpoint and all API responses use `Cache-Control: no-store`.

## One-time history backfill

Jobs completed before private history storage was deployed can be migrated only while their AWS transcript result still exists. Use the guarded migration rather than scanning Transcribe globally. It queries one exact `TRANSCRIBE#<cognito-sub>` ledger partition with consistent reads, requires a narrow completion window plus an exact expected count, revalidates every job's S3 owner path, and applies the same AWS-only redirect, timeout, and byte limits as the production endpoint. It never logs the subject, filename, transcript text, or signed result URL.

Set the subject in an environment variable so it is not exposed in the process command line, then run a complete dry run first:

```powershell
$env:TRANSCRIBE_BACKFILL_SUB = '<exact-cognito-sub>'
npm run migrate:transcribe-history -- --environment production --table '<tools-table>' --bucket '<private-upload-bucket>' --region us-east-2 --expected-count <count> --completed-after '<ISO-start>' --completed-before '<ISO-end>'
```

After verifying the dry-run counts and a current DynamoDB backup or point-in-time recovery window, repeat the same command with `--apply --backup-confirmed`. The guarded window cannot exceed seven days or reach more than 14 days into the past. Writes are per-job, deterministic, and resumable: successful histories remain committed if a later job fails, existing matching objects are verified rather than overwritten, and a newly created object is removed if its metadata transaction fails. Rerun the dry run after applying; it should report every guarded job as already backfilled and zero planned writes.

Daily counters normally record pessimistically reserved spend, not the browser-probed duration. At the default Ohio rate, an untrusted account reserves $2.88 per accepted job (the cost of AWS's non-adjustable 28,800-second maximum), so a caller cannot evade the daily guard by understating duration. Owner-controlled Cognito subjects explicitly listed in `TRANSCRIBE_TRUSTED_COST_SUBS` instead reserve the signed estimate shown in the browser. Only use that allowlist for accounts you control: the original duration comes from browser media metadata and is not a hard security boundary against a malicious authenticated caller. The default per-user and global daily spend ceilings are both $100; an additional reservation is rejected when it would take the UTC-day total over that amount. `TRANSCRIBE_MAX_TOTAL_COST_USD` is also $100 by default and remains a user-facing aggregate run-estimate check. Completed and failed AWS jobs remain charged. Only a failure for which the API confirms that no AWS transcription job exists is refunded.

## IAM permissions

The Vercel AWS principal needs least-privilege permissions like:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:PutObjectTagging", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::<private-upload-bucket>/tools-transcribe/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "transcribe:StartTranscriptionJob",
        "transcribe:GetTranscriptionJob",
        "transcribe:DeleteTranscriptionJob",
        "transcribe:TagResource"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:UpdateItem",
        "dynamodb:PutItem",
        "dynamodb:DeleteItem",
        "dynamodb:TransactWriteItems"
      ],
      "Resource": "arn:aws:dynamodb:<region>:<account-id>:table/<ledger-table>"
    }
  ]
}
```

Amazon Transcribe must also be able to read the uploaded S3 objects. If jobs fail with an S3 access error, add a bucket policy that permits the `transcribe.amazonaws.com` service principal to `s3:GetObject` on the upload prefix, scoped with `aws:SourceAccount` where possible.

## Deployment checklist

1. Deploy the DynamoDB table configuration and IAM permissions before deploying the API code; the production route intentionally fails closed without the ledger.
2. Enable TTL on the configured TTL attribute (normally `ttl`) and add both tag-filtered S3 lifecycle rules: short-lived `retention=temporary` media and 90-day `retention=history` transcript text.
3. Grant the Transcribe role `dynamodb:Query`; usage reads need `GetItem`, while paginated user history needs `Query`.
4. Set the daily cost, daily file, concurrency, and transcript-history retention limits for the expected traffic and AWS budget.
5. Verify a start failure refunds its reservation, an exact quote replay is idempotent, a completed/partial transcript is saved once across repeated polls, history cannot cross accounts, deletion removes both S3 text and DynamoDB metadata, and a completed/failed poll releases its slot.
6. Existing jobs completed before history persistence was deployed require a one-time user-scoped backfill if they should appear in history; no transcript bodies existed in the ledger previously.
7. Keep deterministic Transcribe jobs long enough for run-token retries. If a separate scheduled task deletes old Transcribe jobs, do so only after the corresponding run retention window.
