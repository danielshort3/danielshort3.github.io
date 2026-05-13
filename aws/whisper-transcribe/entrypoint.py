import importlib
import json
import os

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-tiny.en")
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "0"))
DEFAULT_MAX_DIRECT_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(DEFAULT_MAX_DIRECT_BYTES)))
MAX_DIRECT_UPLOAD_BYTES = int(os.getenv("MAX_DIRECT_UPLOAD_BYTES", str(MAX_AUDIO_BYTES)))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(MAX_DIRECT_UPLOAD_BYTES)))
MAX_PART_MINUTES = int(os.getenv("MAX_PART_MINUTES", "30"))
DEFAULT_PART_MINUTES = int(os.getenv("DEFAULT_PART_MINUTES", str(min(30, MAX_PART_MINUTES))))
UPLOAD_BUCKET = (os.getenv("UPLOAD_BUCKET") or "").strip()
RESPONSE_HEADERS = {
  "Content-Type": "application/json",
  "Cache-Control": "no-store"
}

_APP = None


def response(status_code, body=None):
  return {
    "statusCode": status_code,
    "headers": RESPONSE_HEADERS,
    "body": json.dumps(body or {})
  }


def request_path(event):
  raw = (
    event.get("rawPath")
    or event.get("path")
    or event.get("requestContext", {}).get("http", {}).get("path")
    or "/"
  )
  path = "/" + str(raw).strip().lstrip("/")
  return path.rstrip("/") or "/"


def request_method(event):
  return (
    event.get("requestContext", {}).get("http", {}).get("method")
    or event.get("httpMethod")
    or "GET"
  )


def load_app():
  global _APP
  if _APP is None:
    _APP = importlib.import_module("app")
  return _APP


def model_loaded():
  if _APP is None:
    return False
  return getattr(_APP, "_MODEL", None) is not None


def health_response():
  return response(200, {
    "status": "ok",
    "model": MODEL_ID,
    "model_loaded": model_loaded(),
    "target_sample_rate": TARGET_SAMPLE_RATE,
    "max_audio_seconds": MAX_AUDIO_SECONDS,
    "max_part_minutes": MAX_PART_MINUTES,
    "default_part_minutes": DEFAULT_PART_MINUTES,
    "max_audio_bytes": MAX_AUDIO_BYTES,
    "max_direct_upload_bytes": MAX_DIRECT_UPLOAD_BYTES,
    "max_upload_bytes": MAX_UPLOAD_BYTES,
    "uploads_enabled": bool(UPLOAD_BUCKET)
  })


def handler(event, context):
  method = request_method(event)
  path = request_path(event)

  if method == "OPTIONS":
    return { "statusCode": 204 }

  if method == "GET" and path not in ("/warmup",):
    return health_response()

  return load_app().handler(event, context)
