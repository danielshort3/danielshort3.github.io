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
  --memory-size 2048 \
  --timeout 120

aws lambda update-function-configuration \
  --function-name whisper-transcribe \
  --environment "Variables={MODEL_ID=openai/whisper-tiny.en,TARGET_SAMPLE_RATE=16000,MAX_AUDIO_SECONDS=30,MAX_AUDIO_BYTES=5242880,NUM_BEAMS=1,MAX_NEW_TOKENS=256}"
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
  "https://coxbbervgzwhm5tu53dutxwfca0vxdkg.lambda-url.us-east-2.on.aws/transcribe"
```

### Option B: JSON base64 payload (compat)

```json
{ "audio_b64": "<base64 media>", "mime_type": "audio/mpeg" }
```

Notes:
- Audio/video inputs are converted to 16kHz mono WAV (PCM) via `ffmpeg` before transcription.
- The handler accepts `/` and `/transcribe`.
- Limits are enforced with `MAX_AUDIO_SECONDS` and `MAX_AUDIO_BYTES` (defaults: 30 seconds, 5 MB).
