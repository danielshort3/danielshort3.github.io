import importlib
import json
import os

DEFAULT_GRID_SIZE = int(os.getenv("GRID_SIZE", "9"))
DEFAULT_MINE_COUNT = int(os.getenv("MINE_COUNT", "0"))
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
  return getattr(_APP, "_AGENT", None) is not None


def health_response():
  mine_count = DEFAULT_MINE_COUNT
  if mine_count <= 0:
    mine_count = max(1, round(DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE * 0.15))
  return response(200, {
    "status": "ok",
    "service": "minesweeper-solver",
    "model_loaded": model_loaded(),
    "grid": DEFAULT_GRID_SIZE,
    "mines": mine_count
  })


def handler(event, context):
  method = request_method(event)
  path = request_path(event)

  if method == "OPTIONS":
    return { "statusCode": 204 }

  if method == "GET" and path not in ("/warmup",):
    return health_response()

  return load_app().handler(event, context)
