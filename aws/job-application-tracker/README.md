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
    AllowedOrigins="https://danielshort.me,https://www.danielshort.me" \
    CognitoDomainPrefix=job-tracker-auth \
    CallbackUrls="https://danielshort.me/tools/job-application-tracker" \
    LogoutUrls="https://danielshort.me/tools/job-application-tracker"
```

5) Update `pages/job-application-tracker.html` with:

- `data-api-base` (from stack output `ApiEndpoint`)
- `data-cognito-domain` (from stack output `UserPoolDomain`)
- `data-cognito-client-id` (from stack output `UserPoolClientId`)
- `data-cognito-redirect` (your hosted UI callback URL)

If the API or user pool is in a different region, update the CSP entries in `vercel.json` (`connect-src`) to match the new region.

## Environment variables

Provided via CloudFormation:

- `APPLICATIONS_TABLE` - DynamoDB table name.
- `ATTACHMENTS_BUCKET` - S3 bucket for uploads.
- `ALLOWED_ORIGINS` - comma-separated CORS allowlist.
- `PRESIGN_TTL_SECONDS` - presign TTL (defaults to 900).

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
  "status": "Applied",
  "notes": "Reached out to recruiter."
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
  "status": "Applied",
  "notes": "Reached out to recruiter.",
  "statusHistory": [{ "status": "Applied", "date": "2025-01-20T02:40:03.123Z" }],
  "createdAt": "2025-01-20T02:40:03.123Z",
  "updatedAt": "2025-01-20T02:40:03.123Z"
}
```

`POST /api/attachments/presign`

```json
{
  "applicationId": "APP#1737061085123#...",
  "filename": "resume.pdf",
  "contentType": "application/pdf"
}
```

Response:

```json
{
  "uploadUrl": "https://bucket.s3.amazonaws.com/user-id/APP...",
  "key": "USER_SUB/APP#1737061085123#/1737061090000-resume.pdf",
  "bucket": "job-application-tracker-attachments-123456789012",
  "expiresIn": 900
}
```
