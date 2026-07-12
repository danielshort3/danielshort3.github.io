# Job Application Tracker (Lambda + HTTP API)

Private tracker API with Cognito auth, DynamoDB storage, S3 attachments, and analytics endpoints for dashboards.

## Deploy

1) Install Lambda dependencies (only needed after dependency changes):

```bash
cd aws/job-application-tracker
npm install
```

2) Zip the Lambda code:

```bash
python3 - <<'PY'
import zipfile, pathlib
root = pathlib.Path('aws/job-application-tracker')
zip_path = root / 'job-application-tracker.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in root.rglob('*'):
        if path.is_file() and path.name != 'job-application-tracker.zip':
            zf.write(path, path.relative_to(root))
PY
```

3) Upload to S3:

```bash
aws s3 cp aws/job-application-tracker/job-application-tracker.zip s3://YOUR_BUCKET/job-application-tracker/job-application-tracker.zip
```

4) Deploy CloudFormation:

```bash
aws cloudformation deploy \
  --stack-name job-application-tracker \
  --template-file aws/job-application-tracker/template.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    LambdaCodeBucket=YOUR_BUCKET \
    LambdaCodeKey=job-application-tracker/job-application-tracker.zip \
    AllowedOrigins="https://www.danielshort.me,https://www.danielshort.me" \
    CognitoDomainPrefix=job-tracker-auth \
    CallbackUrls="https://www.danielshort.me/tools/job-application-tracker" \
    LogoutUrls="https://www.danielshort.me/tools/job-application-tracker"
```

5) Update `pages/job-application-tracker.html` with:

- `data-api-base` (from stack output `ApiEndpoint`)
- `data-cognito-domain` (from stack output `UserPoolDomain`)
- `data-cognito-client-id` (from stack output `UserPoolClientId`)
- `data-cognito-redirect` (your hosted UI callback URL)

If the API or user pool is in a different region, update the CSP entries in `vercel.json` (`connect-src`) to match the new region.

When rolling out the presigned-POST upload change, deploy the dual-compatible frontend first, then deploy this Lambda/template. The updated frontend accepts both the legacy PUT response and the new POST fields; the legacy frontend cannot use a POST-only response.

## Environment variables

Provided via CloudFormation:

- `APPLICATIONS_TABLE` - DynamoDB table name.
- `ATTACHMENTS_BUCKET` - S3 bucket for uploads.
- `ALLOWED_ORIGINS` - comma-separated CORS allowlist.
- `PRESIGN_TTL_SECONDS` - presign TTL (defaults to 900).
- `MAX_ATTACHMENT_BYTES` - max bytes per attachment (defaults to 10 MB).
- `MAX_ATTACHMENT_COUNT` - max attachments stored per entry (defaults to 12).
- `MAX_TAGS` - max tags per entry (defaults to 12).
- `MAX_CUSTOM_FIELDS` - max custom fields per entry (defaults to 12).
- `MAX_EXPORT_APPLICATIONS` - synchronous export cap (hard-capped at 1,000).
- `MAX_EXPORT_ATTACHMENTS` - synchronous attachment cap (hard-capped at 50).
- `MAX_EXPORT_BYTES` - maximum total metadata plus attachment input (hard-capped at 50 MiB).
- `MAX_EXPORT_METADATA_BYTES` - maximum JSON/CSV metadata input (hard-capped at 8 MiB).

Uploads are signed browser POSTs with an exact content-length policy and a one-day `purpose=staging` lifecycle tag. Saving the attachment verifies its S3 metadata, persists the actual size/ETag, and changes the tag to `purpose=attachment`.

ZIP exports remain synchronous and are intentionally bounded below the API Gateway HTTP API timeout. The next architecture step for larger exports is a `202 Accepted` job endpoint backed by a queue/worker plus a status/download endpoint.

Application deletion removes the DynamoDB record first, then deletes S3 objects. If immediate S3 cleanup fails, the response includes `cleanupPending: true` and the function attempts to restore the one-day staging lifecycle tag. This is a best-effort cleanup path; production-scale cleanup should use a durable queue.

## Auth

API Gateway uses a Cognito JWT authorizer. The Lambda reads the `sub` claim from `event.requestContext.authorizer.jwt.claims` and enforces per-user row security by partition key. The front-end uses the Cognito Hosted UI with PKCE and sends the ID token as `Authorization: Bearer <token>`.

## Example API responses

`GET /api/analytics/summary?start=2025-01-01&end=2025-03-31`

```json
{
  "totalApplications": 18,
  "interviews": 5,
  "offers": 1,
  "rejections": 6,
  "start": "2025-01-01",
  "end": "2025-03-31"
}
```

`GET /api/analytics/applications-over-time?start=2025-01-01&end=2025-01-07`

```json
{
  "series": [
    { "date": "2025-01-01", "count": 1 },
    { "date": "2025-01-02", "count": 0 },
    { "date": "2025-01-03", "count": 2 },
    { "date": "2025-01-04", "count": 0 },
    { "date": "2025-01-05", "count": 1 },
    { "date": "2025-01-06", "count": 0 },
    { "date": "2025-01-07", "count": 1 }
  ],
  "start": "2025-01-01",
  "end": "2025-01-07"
}
```

`GET /api/analytics/status-breakdown?start=2025-01-01&end=2025-03-31`

```json
{
  "statuses": [
    { "status": "Applied", "count": 8 },
    { "status": "Interview", "count": 5 },
    { "status": "Rejected", "count": 4 },
    { "status": "Offer", "count": 1 }
  ],
  "start": "2025-01-01",
  "end": "2025-03-31"
}
```

`GET /api/analytics/calendar?start=2025-01-01&end=2025-03-31`

```json
{
  "days": [
    { "date": "2025-01-01", "count": 1 },
    { "date": "2025-01-02", "count": 0 }
  ],
  "start": "2025-01-01",
  "end": "2025-03-31"
}
```

`POST /api/applications`

```json
{
  "company": "Acme Corp",
  "title": "Data Analyst",
  "appliedDate": "2025-01-15",
  "postingDate": "2025-01-10",
  "captureDate": "2025-01-12",
  "status": "Applied",
  "notes": "Reached out to recruiter.",
  "jobUrl": "https://acme.com/jobs/123",
  "location": "Remote",
  "source": "LinkedIn",
  "batch": "Spring outreach 2025",
  "tags": ["referral", "remote"],
  "followUpDate": "2025-01-20",
  "followUpNote": "Nudge recruiter after screening.",
  "customFields": { "salary": "120k", "priority": "High" }
}
```

Response:

```json
{
  "userId": "USER_SUB",
  "applicationId": "APP#1737061085123#...",
  "company": "Acme Corp",
  "title": "Data Analyst",
  "appliedDate": "2025-01-15",
  "postingDate": "2025-01-10",
  "captureDate": "2025-01-12",
  "status": "Applied",
  "notes": "Reached out to recruiter.",
  "jobUrl": "https://acme.com/jobs/123",
  "location": "Remote",
  "source": "LinkedIn",
  "batch": "Spring outreach 2025",
  "statusHistory": [{ "status": "Applied", "date": "2025-01-20T02:40:03.123Z" }],
  "createdAt": "2025-01-20T02:40:03.123Z",
  "updatedAt": "2025-01-20T02:40:03.123Z"
}
```

`GET /api/prospects`

```json
{
  "items": [
    {
      "applicationId": "PROSPECT#1737061085123#...",
      "company": "Acme Corp",
      "title": "Senior Analyst",
      "jobUrl": "https://company.com/jobs/123",
      "status": "Active",
      "location": "Remote",
      "source": "LinkedIn",
      "postingDate": "2025-01-10",
      "captureDate": "2025-01-15",
      "notes": "Follow up next week.",
      "batch": "Remote searches - January"
    }
  ]
}
```

`POST /api/prospects`

```json
{
  "company": "Acme Corp",
  "title": "Senior Analyst",
  "jobUrl": "https://company.com/jobs/123",
  "status": "Active",
  "location": "Remote",
  "source": "LinkedIn",
  "postingDate": "2025-01-10",
  "captureDate": "2025-01-15",
  "notes": "Follow up next week."
}
```

`PATCH /api/prospects/{id}`

```json
{
  "status": "Inactive"
}
```

`DELETE /api/prospects/{id}` removes a prospect without routing it through the application deletion path.

`POST /api/exports`

```json
{
  "start": "2025-01-01",
  "end": "2025-03-31"
}
```

Response:

```json
{
  "downloadUrl": "https://bucket.s3.amazonaws.com/user-id/exports/job-applications-2025-01-01-to-2025-03-31-1737061090000.zip",
  "key": "USER_SUB/exports/job-applications-2025-01-01-to-2025-03-31-1737061090000.zip",
  "expiresIn": 900,
  "start": "2025-01-01",
  "end": "2025-03-31",
  "totalApplications": 12,
  "attachmentsExported": 8,
  "attachmentsMissing": 1
}
```

`POST /api/attachments/presign`

```json
{
  "applicationId": "APP#1737061085123#...",
  "filename": "resume.pdf",
  "contentType": "application/pdf",
  "size": 245760
}
```

Response:

```json
{
  "uploadUrl": "https://bucket.s3.amazonaws.com/",
  "uploadMethod": "POST",
  "fields": {
    "key": "USER_SUB/staging/APP#1737061085123#/...-resume.pdf",
    "policy": "...",
    "x-amz-signature": "..."
  },
  "key": "USER_SUB/staging/APP#1737061085123#/...-resume.pdf",
  "bucket": "job-application-tracker-attachments-123456789012",
  "expiresIn": 900
}
```

`POST /api/attachments/download`

```json
{
  "key": "USER_SUB/APP#1737061085123#/1737061090000-resume.pdf"
}
```

Response:

```json
{
  "downloadUrl": "https://bucket.s3.amazonaws.com/user-id/APP...",
  "key": "USER_SUB/APP#1737061085123#/1737061090000-resume.pdf",
  "expiresIn": 900
}
```
