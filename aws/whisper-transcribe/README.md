# Whisper Transcribe Lambda (Function URL)

HTTPS transcription endpoint for the website using Hugging Face `openai/whisper-tiny.en`.

## Build + push (Docker)

```bash
aws ecr create-repository --repository-name whisper-transcribe --region us-east-2
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 886623862678.dkr.ecr.us-east-2.amazonaws.com

docker build -t whisper-transcribe aws/whisper-transcribe
docker tag whisper-transcribe:latest 886623862678.dkr.ecr.us-east-2.amazonaws.com/whisper-transcribe:latest
docker push 886623862678.dkr.ecr.us-east-2.amazonaws.com/whisper-transcribe:latest
```

If Lambda rejects the image with `image manifest ... media type ... not supported`, rebuild/push with:

```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t 886623862678.dkr.ecr.us-east-2.amazonaws.com/whisper-transcribe:latest \
  --push \
  aws/whisper-transcribe
```

## Deploy Lambda

```bash
aws iam create-role \
  --role-name whisperTranscribeLambdaRole \
  --assume-role-policy-document file://aws/whisper-transcribe/trust-policy.json

aws iam attach-role-policy \
  --role-name whisperTranscribeLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws lambda create-function \
  --function-name whisper-transcribe \
  --package-type Image \
  --code ImageUri=886623862678.dkr.ecr.us-east-2.amazonaws.com/whisper-transcribe:latest \
  --role arn:aws:iam::886623862678:role/whisperTranscribeLambdaRole \
  --memory-size 3008 \
  --timeout 900

aws lambda update-function-configuration \
  --function-name whisper-transcribe \
  --ephemeral-storage Size=1024 \
  --environment "Variables={MODEL_ID=openai/whisper-tiny.en,TARGET_SAMPLE_RATE=16000,MAX_AUDIO_SECONDS=0,MAX_AUDIO_BYTES=5242880,MAX_DIRECT_UPLOAD_BYTES=5242880,MAX_UPLOAD_BYTES=104857600,UPLOAD_BUCKET=danielshort-whisper-transcribe-uploads-886623862678-us-east-2,UPLOAD_PREFIX=whisper-uploads/,UPLOAD_URL_TTL_SEC=300,DELETE_UPLOAD_AFTER_TRANSCRIBE=true,MAX_PART_MINUTES=30,DEFAULT_PART_MINUTES=30,PROMPT_MAX_TOKENS=224,FFMPEG_TIMEOUT_SEC=300,NUM_BEAMS=1,MAX_NEW_TOKENS=256}"
```

## Optional: enable larger uploads via S3

Function URLs have strict request payload limits. To support larger uploads (audio/video), the Lambda can issue a presigned S3 POST upload and then transcribe from S3.

### Create bucket + CORS + lifecycle

```bash
BUCKET="danielshort-whisper-transcribe-uploads-886623862678-us-east-2"
aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region us-east-2 \
  --create-bucket-configuration LocationConstraint=us-east-2

aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-cors \
  --bucket "$BUCKET" \
  --cors-configuration file://aws/whisper-transcribe/s3-cors.json

aws s3api put-bucket-lifecycle-configuration \
  --bucket "$BUCKET" \
  --lifecycle-configuration file://aws/whisper-transcribe/s3-lifecycle.json
```

### Allow Lambda role to use the bucket

```bash
aws iam put-role-policy \
  --role-name whisperTranscribeLambdaRole \
  --policy-name whisperTranscribeUploads \
  --policy-document file://aws/whisper-transcribe/s3-iam-policy.json
```

## Function URL

```bash
aws lambda create-function-url-config \
  --function-name whisper-transcribe \
  --auth-type NONE \
  --cors file://aws/whisper-transcribe/cors.json

aws lambda add-permission \
  --function-name whisper-transcribe \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE
```

Function URL:

```
https://coxbbervgzwhm5tu53dutxwfca0vxdkg.lambda-url.us-east-2.on.aws/
```

After creating the Function URL, add it to `vercel.json` `connect-src` if the site is served with that CSP.

## Request format (audio/video)

### Option A: Raw media upload (recommended)

POST the file bytes directly and set `Content-Type` to the file's MIME type (examples: `audio/mpeg`, `audio/wav`, `video/mp4`).

```bash
curl -X POST \
  -H "Content-Type: audio/mpeg" \
  --data-binary @sample.mp3 \
  "https://coxbbervgzwhm5tu53dutxwfca0vxdkg.lambda-url.us-east-2.on.aws/transcribe?part_minutes=30"
```

### Option B: JSON base64 payload (compat)

```json
{ "audio_b64": "<base64 media>", "mime_type": "audio/mpeg" }
```

Notes:
- Audio/video inputs are converted to 16kHz mono WAV (PCM) via `ffmpeg` before transcription.
- The handler accepts `/` and `/transcribe`.
- Long media can be split into parts with `part_minutes` (1..`MAX_PART_MINUTES`, default `DEFAULT_PART_MINUTES`).
- Responses include:
  - `transcript`: combined transcript
  - `parts`: per-part transcripts with `{ index, start_sec, end_sec, transcript }`
  - `part_seconds`: the part length used (seconds)
- Optional S3 upload flow:
  - `POST /presign` (JSON) → returns `{ upload_url, fields, key }`
  - browser uploads to S3
  - `POST /transcribe-s3?part_minutes=30` (JSON `{ key }`) → returns transcript
- Duration limits are disabled by default (`MAX_AUDIO_SECONDS=0`). Size limits are enforced with `MAX_DIRECT_UPLOAD_BYTES` and `MAX_UPLOAD_BYTES` (100 MB in the config above).
