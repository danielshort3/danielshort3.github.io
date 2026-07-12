# Amazon Transcribe Tool Setup

The website file transcription tool uses Vercel API routes to presign private S3 uploads and start Amazon Transcribe batch jobs. Users must be signed in with the existing tools account before paid jobs can start.

## Environment variables

Set these in Vercel:

```bash
AWS_REGION=us-east-2
TRANSCRIBE_AWS_REGION=us-east-2
TRANSCRIBE_UPLOAD_BUCKET=<private-upload-bucket>
TRANSCRIBE_UPLOAD_PREFIX=tools-transcribe/
TRANSCRIBE_SIGNING_SECRET=<long-random-secret>
TRANSCRIBE_PRICE_PER_SECOND=0.0004
TRANSCRIBE_MAX_FILES_PER_RUN=10
TRANSCRIBE_MAX_FILE_BYTES=524288000
TRANSCRIBE_MAX_TOTAL_COST_USD=10
TRANSCRIBE_LANGUAGE_CODE=en-US
TRANSCRIBE_RUN_TOKEN_TTL_SECONDS=86400
TRANSCRIBE_DDB_TABLE=<pk-sk-dynamodb-table>
TRANSCRIBE_DDB_TTL_ATTRIBUTE=ttl
TRANSCRIBE_LEDGER_MODE=required
TRANSCRIBE_DAILY_COST_LIMIT_USD=25
TRANSCRIBE_DAILY_FILE_LIMIT=50
TRANSCRIBE_GLOBAL_DAILY_COST_LIMIT_USD=100
TRANSCRIBE_GLOBAL_DAILY_FILE_LIMIT=200
TRANSCRIBE_MAX_CONCURRENT=2
TRANSCRIBE_LEDGER_START_LEASE_SECONDS=300
TRANSCRIBE_LEDGER_LEASE_SECONDS=43200
TRANSCRIBE_LEDGER_RENEWAL_SECONDS=60
TRANSCRIBE_LEDGER_RUN_RETENTION_DAYS=45
TRANSCRIBE_LEDGER_DAY_RETENTION_DAYS=45
TRANSCRIBE_TRANSCRIPT_FETCH_TIMEOUT_MS=12000
TRANSCRIBE_MAX_TRANSCRIPT_FETCH_BYTES=26214400
TRANSCRIBE_MAX_TRANSCRIPT_BYTES=3145728
TRANSCRIBE_AWS_ACCESS_KEY_ID=<access-key>
TRANSCRIBE_AWS_SECRET_ACCESS_KEY=<secret-key>
```

`TRANSCRIBE_AWS_SESSION_TOKEN` is optional for temporary credentials.

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
      "AllowedMethods": ["POST"],
      "AllowedHeaders": ["content-type", "x-amz-*"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 300
    }
  ]
}
```

Every presigned POST policy enforces the quoted file byte count and content type, and requires a `tagging` form field containing S3 Tagging XML for `tool=amazon-transcribe` plus `retention=temporary`. Add a lifecycle rule filtered to that tag pair (and optionally the `tools-transcribe/` prefix) that expires objects after a few days. The app deletes uploaded media when jobs complete, fail, or are confirmed missing, but lifecycle expiration is the cleanup backstop. Keep the bucket private and block public access.

The completed transcript response is streamed through the API with both a timeout and a byte ceiling. Only HTTPS transcript URLs on AWS domains are accepted. Increase the two transcript fetch limits deliberately if legitimate transcripts exceed the defaults.

## DynamoDB TTL and retention

Enable DynamoDB Time to Live on the ledger table using the numeric attribute configured by `TRANSCRIBE_DDB_TTL_ATTRIBUTE`. It defaults to `TOOLS_DDB_TTL_ATTRIBUTE` when present, then to `ttl`, so a shared tools table uses one consistent TTL attribute. Run records and UTC-day usage records default to 45 days of retention. Slot records also carry TTL, but correctness does not depend on prompt TTL deletion: every slot claim checks `leaseExpiresAt`, so delayed DynamoDB cleanup cannot permanently block a user.

Daily counters record pessimistically reserved spend, not the browser-probed duration. At the default rate, each accepted job reserves $11.52 (the cost of AWS's non-adjustable 28,800-second maximum), so a caller cannot evade the daily guard by understating duration. `TRANSCRIBE_MAX_TOTAL_COST_USD` remains a user-facing estimate check; the durable daily ledger is the server-side spend boundary. Completed and failed AWS jobs remain charged. Only a failure for which the API confirms that no AWS transcription job exists is refunded.

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
2. Enable TTL on the configured TTL attribute (normally `ttl`) and add the tag-filtered S3 lifecycle rule.
3. Set the daily cost, daily file, and concurrency limits for the expected traffic and AWS budget.
4. Verify a start failure refunds its reservation, an exact quote replay is idempotent, and a completed/failed poll releases its slot.
5. Keep deterministic Transcribe jobs long enough for run-token retries. If a separate scheduled task deletes old Transcribe jobs, do so only after the corresponding run retention window.
