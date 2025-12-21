import json
import os
from datetime import datetime
from urllib.parse import parse_qs

import boto3
from botocore.exceptions import ClientError

S3_CLIENT = boto3.client("s3")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "")
DATA_PREFIX = os.environ.get("DATA_PREFIX", "").strip("/")

CACHE = {
  "meta": None,
  "dates": {},
  "states": {}
}


def build_key(*parts):
  clean_parts = [p.strip("/") for p in parts if p]
  key = "/".join(clean_parts)
  if DATA_PREFIX:
    return f"{DATA_PREFIX}/{key}"
  return key


def json_response(status_code, payload, extra_headers=None):
  headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
  }
  if extra_headers:
    headers.update(extra_headers)
  body = "" if payload is None else json.dumps(payload, separators=(",", ":"))
  return {
    "statusCode": status_code,
    "headers": headers,
    "body": body
  }


def load_json(key):
  resp = S3_CLIENT.get_object(Bucket=DATA_BUCKET, Key=key)
  raw = resp["Body"].read()
  return json.loads(raw)


def load_meta():
  if CACHE["meta"] is None:
    CACHE["meta"] = load_json(build_key("meta.json"))
  return CACHE["meta"]


def load_date(date_str):
  if date_str in CACHE["dates"]:
    return CACHE["dates"][date_str]
  data = load_json(build_key("by-date", f"{date_str}.json"))
  CACHE["dates"][date_str] = data
  return data


def load_state(state_id):
  state_id = state_id.upper()
  if state_id in CACHE["states"]:
    return CACHE["states"][state_id]
  data = load_json(build_key("state", f"{state_id}.json"))
  CACHE["states"][state_id] = data
  return data


def parse_query(event):
  query = event.get("queryStringParameters") or {}
  if query:
    return query
  raw = event.get("rawQueryString") or ""
  if not raw:
    return {}
  parsed = parse_qs(raw)
  return {k: v[0] if isinstance(v, list) and v else v for k, v in parsed.items()}


def parse_body(event):
  body = event.get("body")
  if not body:
    return {}
  if event.get("isBase64Encoded"):
    return {}
  try:
    return json.loads(body)
  except json.JSONDecodeError:
    return {}


def maybe_refresh_cache(query):
  raw = str(query.get("refresh", "")).lower()
  if raw in ("1", "true", "yes"):
    CACHE["meta"] = None
    CACHE["dates"].clear()
    CACHE["states"].clear()


def handle_states(query):
  meta = load_meta()
  date_str = query.get("date") or meta.get("latest")
  if not date_str:
    return json_response(400, {"error": "Missing date"})
  try:
    datetime.strptime(date_str, "%Y-%m-%d")
  except ValueError:
    return json_response(400, {"error": "Invalid date format"})
  try:
    payload = load_date(date_str)
  except ClientError as exc:
    if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
      return json_response(404, {"error": "Date not found"})
    raise
  return json_response(200, payload)


def handle_state(state_id, query):
  meta = load_meta()
  date_str = query.get("date") or meta.get("latest")
  if not state_id:
    return json_response(400, {"error": "Missing state"})
  try:
    payload = load_state(state_id)
  except ClientError as exc:
    if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
      return json_response(404, {"error": "State not found"})
    raise
  if date_str:
    summary = None
    try:
      by_date = load_date(date_str)
      summary = next((row for row in by_date.get("states", []) if row.get("id") == state_id.upper()), None)
    except ClientError as exc:
      if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
        summary = None
      else:
        raise
    if summary:
      payload = {**payload, "date": date_str, "summary": summary}
  return json_response(200, payload)


def handle_query(body):
  query = body if isinstance(body, dict) else {}
  if query.get("state"):
    return handle_state(query.get("state"), query)
  return handle_states(query)


def lambda_handler(event, context):
  method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod", "GET")
  path = event.get("rawPath") or event.get("path") or "/"
  path = path.rstrip("/") if path != "/" else path
  query = parse_query(event)

  if method == "OPTIONS":
    return json_response(204, None)

  if path in ("/health", "/healthz"):
    return json_response(200, {"status": "ok"})

  if not DATA_BUCKET:
    return json_response(500, {"error": "DATA_BUCKET not configured"})

  try:
    if path == "/meta":
      maybe_refresh_cache(query)
      return json_response(200, load_meta())
    if path == "/states":
      maybe_refresh_cache(query)
      return handle_states(query)
    if path.startswith("/state"):
      parts = [p for p in path.split("/") if p]
      state_id = parts[1] if len(parts) > 1 else query.get("state")
      maybe_refresh_cache(query)
      return handle_state(state_id, query)
    if path == "/query" and method in ("POST", "GET"):
      body = parse_body(event) if method == "POST" else query
      return handle_query(body)
  except Exception as exc:
    return json_response(500, {"error": "Server error", "detail": str(exc)})

  return json_response(404, {"error": "Not found"})
