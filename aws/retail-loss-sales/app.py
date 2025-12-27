import json
import os
from urllib.parse import parse_qs

import boto3

S3_CLIENT = boto3.client("s3")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")
DATA_KEY = os.environ.get("DATA_KEY", "").lstrip("/")

CACHE = {
  "data": None
}


def json_response(status_code, payload, extra_headers=None):
  headers = {
    "Content-Type": "application/json"
  }
  if extra_headers:
    headers.update(extra_headers)
  body = "" if payload is None else json.dumps(payload, separators=(",", ":"))
  return {
    "statusCode": status_code,
    "headers": headers,
    "body": body
  }


def parse_query(event):
  query = event.get("queryStringParameters") or {}
  if query:
    return query
  raw = event.get("rawQueryString") or ""
  if not raw:
    return {}
  parsed = parse_qs(raw)
  return {k: v[0] if isinstance(v, list) and v else v for k, v in parsed.items()}


def maybe_refresh_cache(query):
  raw = str(query.get("refresh", "")).lower()
  if raw in ("1", "true", "yes"):
    CACHE["data"] = None


def load_data():
  if CACHE["data"] is not None:
    return CACHE["data"]
  if not DATA_BUCKET or not DATA_KEY:
    return None
  resp = S3_CLIENT.get_object(Bucket=DATA_BUCKET, Key=DATA_KEY)
  raw = resp["Body"].read()
  data = json.loads(raw)
  CACHE["data"] = data
  return data


def lambda_handler(event, context):
  method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod", "GET")
  path = event.get("rawPath") or event.get("path") or "/"
  path = path.rstrip("/") if path != "/" else path
  query = parse_query(event)

  if method == "OPTIONS":
    return json_response(204, None)

  if path in ("/health", "/healthz"):
    return json_response(200, {"status": "ok"})

  if not DATA_BUCKET or not DATA_KEY:
    return json_response(500, {"error": "DATA_BUCKET or DATA_KEY not configured"})

  try:
    if path in ("/", "/data"):
      maybe_refresh_cache(query)
      data = load_data()
      if data is None:
        return json_response(500, {"error": "Data not available"})
      return json_response(200, data)
    if path == "/meta":
      maybe_refresh_cache(query)
      data = load_data()
      if data is None:
        return json_response(500, {"error": "Data not available"})
      return json_response(200, data.get("meta", {}))
  except Exception as exc:
    return json_response(500, {"error": "Server error", "detail": str(exc)})

  return json_response(404, {"error": "Not found"})
