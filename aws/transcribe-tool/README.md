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
TRANSCRIBE_AWS_ACCESS_KEY_ID=<access-key>
TRANSCRIBE_AWS_SECRET_ACCESS_KEY=<secret-key>
```

`TRANSCRIBE_AWS_SESSION_TOKEN` is optional for temporary credentials.

## S3 bucket

Create a private bucket in the same region as Transcribe and apply CORS similar to:

```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["https://www.danielshort.me", "https://danielshort.me"],
      "AllowedMethods": ["PUT"],
      "AllowedHeaders": ["content-type", "x-amz-*"],
      "ExposeHeaders": ["ETag"],
      "MaxAgeSeconds": 300
    }
  ]
}
```

Add a lifecycle rule that expires `tools-transcribe/` objects after a few days. The app deletes uploaded media when jobs complete or fail, but lifecycle expiration is the cleanup backstop.

## IAM permissions

The Vercel AWS principal needs least-privilege permissions like:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
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
    }
  ]
}
```

Amazon Transcribe must also be able to read the uploaded S3 objects. If jobs fail with an S3 access error, add a bucket policy that permits the `transcribe.amazonaws.com` service principal to `s3:GetObject` on the upload prefix, scoped with `aws:SourceAccount` where possible.
