import base64
import io
import json
import os
import re
import secrets
import subprocess
import time
import wave
from urllib.parse import parse_qs

import audioop
import numpy as np

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-tiny.en")
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "0"))
DEFAULT_MAX_DIRECT_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(DEFAULT_MAX_DIRECT_BYTES)))
MAX_DIRECT_UPLOAD_BYTES = int(os.getenv("MAX_DIRECT_UPLOAD_BYTES", str(MAX_AUDIO_BYTES)))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(MAX_DIRECT_UPLOAD_BYTES)))
MAX_PART_MINUTES = int(os.getenv("MAX_PART_MINUTES", "30"))
DEFAULT_PART_MINUTES = int(os.getenv("DEFAULT_PART_MINUTES", str(min(30, MAX_PART_MINUTES))))
PROMPT_MAX_TOKENS = int(os.getenv("PROMPT_MAX_TOKENS", "224"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

RESPONSE_HEADERS = {
  "Content-Type": "application/json"
}

_MODEL = None
_PROCESSOR = None

UPLOAD_BUCKET = (os.getenv("UPLOAD_BUCKET") or "").strip()
UPLOAD_PREFIX = (os.getenv("UPLOAD_PREFIX") or "whisper-uploads/").strip()
UPLOAD_URL_TTL_SEC = int(os.getenv("UPLOAD_URL_TTL_SEC", "300"))
DELETE_UPLOAD_AFTER_TRANSCRIBE = os.getenv("DELETE_UPLOAD_AFTER_TRANSCRIBE", "true").strip().lower() in ("1", "true", "yes")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_TIMEOUT_SEC = float(os.getenv("FFMPEG_TIMEOUT_SEC", "20"))


def json_response(status_code, payload):
  body = "" if payload is None else json.dumps(payload)
  return {
    "statusCode": status_code,
    "headers": RESPONSE_HEADERS,
    "body": body
  }


def parse_body(event):
  if not event:
    return {}
  body = event.get("body")
  if not body:
    return {}
  if event.get("isBase64Encoded"):
    try:
      body = base64.b64decode(body).decode("utf-8")
    except Exception:
      return {}
  if isinstance(body, (bytes, bytearray)):
    try:
      body = body.decode("utf-8")
    except Exception:
      return {}
  if isinstance(body, str):
    try:
      return json.loads(body)
    except json.JSONDecodeError:
      return {}
  if isinstance(body, dict):
    return body
  return {}


def header_value(event, name):
  headers = event.get("headers") if event else None
  if not isinstance(headers, dict):
    return ""
  needle = (name or "").strip().lower()
  if not needle:
    return ""
  for key, value in headers.items():
    if str(key or "").strip().lower() == needle:
      return "" if value is None else str(value)
  return ""


def query_value(event, name):
  needle = (name or "").strip().lower()
  if not needle:
    return ""

  params = event.get("queryStringParameters") if event else None
  if isinstance(params, dict):
    for key, value in params.items():
      if str(key or "").strip().lower() == needle:
        return "" if value is None else str(value)

  raw = event.get("rawQueryString") if event else None
  if isinstance(raw, str) and raw.strip():
    try:
      parsed = parse_qs(raw, keep_blank_values=True)
      for key, values in parsed.items():
        if str(key or "").strip().lower() == needle and values:
          return "" if values[0] is None else str(values[0])
    except Exception:
      return ""
  return ""


def clamp_int(value, min_value, max_value, fallback):
  try:
    parsed = int(value)
  except Exception:
    return fallback
  return max(min_value, min(max_value, parsed))


def decode_body_bytes(event):
  if not event:
    return b""
  body = event.get("body")
  if body is None:
    return b""
  if event.get("isBase64Encoded"):
    try:
      return base64.b64decode(body)
    except Exception:
      return b""
  if isinstance(body, (bytes, bytearray)):
    return bytes(body)
  if isinstance(body, str):
    return body.encode("utf-8")
  return b""


def parse_content_type(value):
  if not value:
    return ""
  raw = str(value).strip()
  if not raw:
    return ""
  return raw.split(";", 1)[0].strip().lower()


def load_whisper():
  global _MODEL, _PROCESSOR
  if _MODEL is not None and _PROCESSOR is not None:
    return _MODEL, _PROCESSOR

  import warnings
  import torch
  from transformers import WhisperForConditionalGeneration, WhisperProcessor
  try:
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
  except Exception:
    pass
  warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

  torch.set_grad_enabled(False)
  torch.set_num_threads(1)
  os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
  cache_dir = os.getenv("HF_HOME") or None
  processor = WhisperProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=cache_dir,
    local_files_only=bool(cache_dir)
  )
  model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    cache_dir=cache_dir,
    local_files_only=bool(cache_dir)
  )
  model.eval()

  _MODEL = model
  _PROCESSOR = processor
  return _MODEL, _PROCESSOR


def strip_data_url(value):
  if not isinstance(value, str):
    return ""
  raw = value.strip()
  if not raw:
    return ""
  if raw.startswith("data:"):
    parts = raw.split(",", 1)
    if len(parts) == 2:
      return parts[1].strip()
  return raw


def decode_audio_b64(payload):
  b64 = (
    payload.get("audio_b64")
    or payload.get("audioB64")
    or payload.get("audio")
    or payload.get("media_b64")
    or payload.get("mediaB64")
    or payload.get("file_b64")
    or payload.get("fileB64")
    or payload.get("b64")
    or payload.get("data")
  )
  b64 = strip_data_url(b64)
  if not b64:
    return None, "Missing base64 audio payload."
  try:
    raw = base64.b64decode(b64)
  except Exception:
    return None, "Invalid base64 audio payload."
  if len(raw) > MAX_DIRECT_UPLOAD_BYTES:
    return None, f"Audio payload exceeds {MAX_DIRECT_UPLOAD_BYTES} bytes."
  return raw, None


def probe_duration_seconds(path):
  if MAX_AUDIO_SECONDS <= 0:
    return None
  try:
    res = subprocess.run(
      [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        path
      ],
      capture_output=True,
      text=True,
      check=True,
      timeout=max(1, int(FFMPEG_TIMEOUT_SEC))
    )
  except Exception:
    return None

  out = (res.stdout or "").strip()
  if not out or out == "N/A":
    return None
  try:
    return float(out)
  except Exception:
    return None


def sanitize_filename(value):
  raw = str(value or "").strip()
  if not raw:
    return "media"
  raw = raw.replace("\\", "/").split("/")[-1]
  raw = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._")
  raw = raw[:80] if len(raw) > 80 else raw
  return raw or "media"


def ensure_prefix(prefix):
  raw = str(prefix or "").strip()
  if not raw:
    return "whisper-uploads/"
  return raw if raw.endswith("/") else f"{raw}/"


def get_s3_client():
  import boto3
  return boto3.client("s3")


def presign_upload_post(filename, content_type, bytes_len):
  if not UPLOAD_BUCKET:
    raise ValueError("Uploads are not enabled.")
  if bytes_len <= 0:
    raise ValueError("Missing file bytes.")
  if bytes_len > MAX_UPLOAD_BYTES:
    raise ValueError(f"File exceeds {MAX_UPLOAD_BYTES} bytes.")

  prefix = ensure_prefix(UPLOAD_PREFIX)
  token = secrets.token_hex(16)
  safe_name = sanitize_filename(filename)
  key = f"{prefix}{int(time.time())}-{token}-{safe_name}"

  fields = {
    "Content-Type": content_type or "application/octet-stream",
    "success_action_status": "201"
  }
  conditions = [
    ["content-length-range", 1, int(MAX_UPLOAD_BYTES)],
    {"key": key},
    {"success_action_status": "201"},
    {"Content-Type": fields["Content-Type"]}
  ]

  s3 = get_s3_client()
  post = s3.generate_presigned_post(
    Bucket=UPLOAD_BUCKET,
    Key=key,
    Fields=fields,
    Conditions=conditions,
    ExpiresIn=max(60, int(UPLOAD_URL_TTL_SEC))
  )
  return {
    "upload_url": post.get("url"),
    "fields": post.get("fields") or {},
    "key": key,
    "expires_in": max(60, int(UPLOAD_URL_TTL_SEC))
  }


def download_s3_object(bucket, key, dest_path):
  s3 = get_s3_client()
  head = s3.head_object(Bucket=bucket, Key=key)
  size = int(head.get("ContentLength") or 0)
  if size <= 0:
    raise ValueError("Empty upload.")
  if size > MAX_UPLOAD_BYTES:
    raise ValueError(f"File exceeds {MAX_UPLOAD_BYTES} bytes.")
  content_type = str(head.get("ContentType") or "").strip().lower()

  obj = s3.get_object(Bucket=bucket, Key=key)
  body = obj.get("Body")
  written = 0
  with open(dest_path, "wb") as fp:
    while True:
      chunk = body.read(1024 * 1024)
      if not chunk:
        break
      fp.write(chunk)
      written += len(chunk)
      if written > MAX_UPLOAD_BYTES:
        raise ValueError(f"File exceeds {MAX_UPLOAD_BYTES} bytes.")
  return content_type or "application/octet-stream", written


def delete_s3_object(bucket, key):
  try:
    s3 = get_s3_client()
    s3.delete_object(Bucket=bucket, Key=key)
  except Exception:
    return


def ffmpeg_to_wav_16k_mono(input_bytes, mime_type):
  token = str(time.time_ns())
  input_path = f"/tmp/whisper-input-{token}"
  output_path = f"/tmp/whisper-output-{token}.wav"
  try:
    with open(input_path, "wb") as fp:
      fp.write(input_bytes)

    duration = probe_duration_seconds(input_path)
    if duration is not None and duration > MAX_AUDIO_SECONDS:
      raise ValueError(f"Audio duration exceeds {MAX_AUDIO_SECONDS} seconds.")

    cmd = [
      FFMPEG_BIN,
      "-hide_banner",
      "-nostdin",
      "-loglevel",
      "error"
    ]
    if MAX_AUDIO_SECONDS > 0:
      cmd.extend(["-t", str(MAX_AUDIO_SECONDS)])
    cmd.extend(
      [
        "-i",
        input_path,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-y",
        output_path
      ]
    )

    try:
      subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        timeout=max(1, int(FFMPEG_TIMEOUT_SEC))
      )
    except subprocess.CalledProcessError as exc:
      stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
      message = (stderr or "").strip() or "Unsupported media format."
      if "matches no streams" in message or "does not contain any stream" in message:
        raise ValueError("No audio track found in media file.")
      raise ValueError(message.splitlines()[-1] if "\n" in message else message)
    except FileNotFoundError:
      raise RuntimeError("ffmpeg not available in runtime environment.")

    with open(output_path, "rb") as fp:
      return fp.read()
  finally:
    try:
      if os.path.exists(input_path):
        os.remove(input_path)
    except Exception:
      pass
    try:
      if os.path.exists(output_path):
        os.remove(output_path)
    except Exception:
      pass


def ffmpeg_file_to_wav_16k_mono(input_path):
  token = str(time.time_ns())
  output_path = f"/tmp/whisper-output-{token}.wav"
  try:
    duration = probe_duration_seconds(input_path)
    if duration is not None and duration > MAX_AUDIO_SECONDS:
      raise ValueError(f"Audio duration exceeds {MAX_AUDIO_SECONDS} seconds.")

    cmd = [
      FFMPEG_BIN,
      "-hide_banner",
      "-nostdin",
      "-loglevel",
      "error"
    ]
    if MAX_AUDIO_SECONDS > 0:
      cmd.extend(["-t", str(MAX_AUDIO_SECONDS)])
    cmd.extend(
      [
        "-i",
        input_path,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-y",
        output_path
      ]
    )

    try:
      subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        timeout=max(1, int(FFMPEG_TIMEOUT_SEC))
      )
    except subprocess.CalledProcessError as exc:
      stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
      message = (stderr or "").strip() or "Unsupported media format."
      if "matches no streams" in message or "does not contain any stream" in message:
        raise ValueError("No audio track found in media file.")
      raise ValueError(message.splitlines()[-1] if "\n" in message else message)
    except FileNotFoundError:
      raise RuntimeError("ffmpeg not available in runtime environment.")

    with open(output_path, "rb") as fp:
      return fp.read()
  finally:
    try:
      if os.path.exists(output_path):
        os.remove(output_path)
    except Exception:
      pass


def wav_to_float32_mono_16k(wav_bytes):
  try:
    wf = wave.open(io.BytesIO(wav_bytes), "rb")
  except Exception:
    raise ValueError("Unsupported audio format. Send WAV (PCM) audio.")

  with wf:
    channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    sample_rate = wf.getframerate()
    frames = wf.getnframes()
    if not sample_rate or sample_rate <= 0:
      raise ValueError("Invalid WAV sample rate.")

    duration_seconds = frames / float(sample_rate)
    if MAX_AUDIO_SECONDS > 0 and duration_seconds > MAX_AUDIO_SECONDS:
      raise ValueError(f"Audio duration exceeds {MAX_AUDIO_SECONDS} seconds.")

    pcm = wf.readframes(frames)

  if channels == 2:
    pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
    channels = 1
  elif channels != 1:
    raise ValueError("Only mono or stereo WAV audio is supported.")

  if sample_width != 2:
    pcm = audioop.lin2lin(pcm, sample_width, 2)
    sample_width = 2

  if sample_rate != TARGET_SAMPLE_RATE:
    pcm, _ = audioop.ratecv(pcm, sample_width, channels, sample_rate, TARGET_SAMPLE_RATE, None)
    sample_rate = TARGET_SAMPLE_RATE

  audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
  return audio, sample_rate, duration_seconds


def build_prompt_ids(processor, text):
  raw = (text or "").strip()
  if not raw:
    return None
  try:
    ids = processor.tokenizer(raw, add_special_tokens=False).input_ids
  except Exception:
    return None
  if not ids:
    return None
  ids = ids[-max(1, int(PROMPT_MAX_TOKENS)):]
  import torch
  return torch.tensor(ids, dtype=torch.long)


def transcribe_long_form(audio, sample_rate, part_minutes):
  import torch

  model, processor = load_whisper()
  try:
    short_samples = int(getattr(processor.feature_extractor, "n_samples"))
  except Exception:
    short_samples = int(sample_rate * 30)
  minutes = clamp_int(part_minutes, 1, max(1, int(MAX_PART_MINUTES)), int(DEFAULT_PART_MINUTES))
  part_seconds = max(60, minutes * 60)
  part_samples = max(1, int(part_seconds * sample_rate))
  try:
    max_target_positions = int(getattr(model.config, "max_target_positions"))
  except Exception:
    max_target_positions = 448
  safe_prompt_limit = max(0, max_target_positions - max(1, int(MAX_NEW_TOKENS)) - 1)

  prompt_ids = None
  parts = []
  full_text_pieces = []
  total_samples = len(audio) if audio is not None else 0
  offset = 0
  index = 0

  while offset < total_samples:
    end = min(total_samples, offset + part_samples)
    chunk = audio[offset:end]
    is_short = len(chunk) <= short_samples
    feats = processor.feature_extractor(
      chunk,
      sampling_rate=sample_rate,
      return_tensors="pt",
      truncation=bool(is_short),
      padding=("max_length" if is_short else "longest"),
      return_attention_mask=True
    )

    kwargs = {
      "attention_mask": feats.attention_mask,
      "return_timestamps": True,
      "return_segments": True,
      "condition_on_prev_tokens": True,
      "return_dict_in_generate": True,
      "num_beams": max(1, NUM_BEAMS),
      "max_new_tokens": max(1, MAX_NEW_TOKENS)
    }
    if prompt_ids is not None:
      if safe_prompt_limit > 0 and len(prompt_ids) > safe_prompt_limit:
        kwargs["prompt_ids"] = prompt_ids[-safe_prompt_limit:]
      else:
        kwargs["prompt_ids"] = prompt_ids

    with torch.no_grad():
      out = model.generate(feats.input_features, **kwargs)

    segs = out.get("segments") if isinstance(out, dict) else None
    seg_list = segs[0] if isinstance(segs, list) and segs else None
    chunk_text = ""

    if isinstance(seg_list, list) and seg_list:
      chunk_text_pieces = []
      for seg in seg_list:
        ids = seg.get("tokens")
        if ids is None:
          result = seg.get("result")
          if isinstance(result, dict):
            ids = result.get("sequences")
          else:
            ids = result
        if ids is None:
          continue
        try:
          text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
          text = ""
        if text:
          chunk_text_pieces.append(text)
      chunk_text = ("".join(chunk_text_pieces)).strip()
    else:
      sequences = out.get("sequences") if isinstance(out, dict) else out
      if isinstance(sequences, torch.Tensor) and sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)
      try:
        texts = processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
      except Exception:
        texts = []
      chunk_text = (texts[0] if texts else "").strip()
    part_start_sec = float(offset) / float(sample_rate)
    part_end_sec = float(end) / float(sample_rate)
    parts.append({
      "index": index + 1,
      "start_sec": round(part_start_sec, 4),
      "end_sec": round(part_end_sec, 4),
      "transcript": chunk_text
    })
    full_text_pieces.append(chunk_text)

    prompt_ids = build_prompt_ids(processor, chunk_text)
    offset = end
    index += 1

  full_text = "\n\n".join([p for p in full_text_pieces if p]).strip()
  return full_text, parts, part_seconds


def handler(event, context):
  start = time.time()
  method = (
    (event.get("requestContext", {}).get("http", {}).get("method"))
    or event.get("httpMethod")
    or "GET"
  )
  path = event.get("rawPath") or event.get("path") or "/"
  path = path.rstrip("/") or "/"
  content_type = parse_content_type(header_value(event, "content-type"))

  if method == "OPTIONS":
    return {"statusCode": 204}

  if method == "GET":
    return json_response(200, {
      "status": "ok",
      "model": MODEL_ID,
      "model_loaded": _MODEL is not None,
      "target_sample_rate": TARGET_SAMPLE_RATE,
      "max_audio_seconds": MAX_AUDIO_SECONDS,
      "max_part_minutes": MAX_PART_MINUTES,
      "default_part_minutes": DEFAULT_PART_MINUTES,
      "max_audio_bytes": MAX_DIRECT_UPLOAD_BYTES,
      "max_direct_upload_bytes": MAX_DIRECT_UPLOAD_BYTES,
      "max_upload_bytes": MAX_UPLOAD_BYTES,
      "uploads_enabled": bool(UPLOAD_BUCKET)
    })

  if method != "POST":
    return json_response(405, {"error": "Method not allowed."})

  if path == "/presign":
    payload = parse_body(event)
    if not isinstance(payload, dict):
      return json_response(400, {"error": "Invalid JSON payload."})
    filename = payload.get("filename") or payload.get("name") or "media"
    content_type = parse_content_type(payload.get("content_type") or payload.get("contentType") or "application/octet-stream")
    bytes_len = payload.get("bytes") or payload.get("size") or 0
    try:
      bytes_len = int(bytes_len)
    except Exception:
      bytes_len = 0
    try:
      post = presign_upload_post(filename, content_type, bytes_len)
      return json_response(200, post)
    except ValueError as exc:
      message = str(exc) or "Invalid request."
      status = 413 if "exceeds" in message else 400
      return json_response(status, {"error": message})
    except Exception:
      return json_response(500, {"error": "Server error"})

  if path == "/transcribe-s3":
    if not UPLOAD_BUCKET:
      return json_response(400, {"error": "Uploads are not enabled."})
    payload = parse_body(event)
    if not isinstance(payload, dict):
      return json_response(400, {"error": "Invalid JSON payload."})
    key = str(payload.get("key") or "").strip()
    prefix = ensure_prefix(UPLOAD_PREFIX)
    if not key or not key.startswith(prefix):
      return json_response(400, {"error": "Invalid upload key."})

    token = str(time.time_ns())
    input_path = f"/tmp/whisper-s3-{token}"
    try:
      part_minutes = query_value(event, "part_minutes") or query_value(event, "partMinutes") or DEFAULT_PART_MINUTES
      mime_type, _ = download_s3_object(UPLOAD_BUCKET, key, input_path)
      wav_bytes = ffmpeg_file_to_wav_16k_mono(input_path)
      audio, sample_rate, duration_seconds = wav_to_float32_mono_16k(wav_bytes)
      transcript, parts, part_seconds = transcribe_long_form(audio, sample_rate, part_minutes)
      duration_ms = int((time.time() - start) * 1000)
      return json_response(200, {
        "transcript": transcript,
        "parts": parts,
        "part_seconds": part_seconds,
        "model": MODEL_ID,
        "language": "en",
        "audio_seconds": round(float(duration_seconds), 4),
        "duration_ms": duration_ms,
        "input_mime_type": mime_type,
        "converted_to_wav": True
      })
    except ValueError as exc:
      message = str(exc)
      if message.startswith("Audio duration exceeds"):
        return json_response(413, {"error": message})
      if message.startswith("No audio track found") or message.startswith("Unsupported audio format"):
        return json_response(415, {"error": message})
      if "exceeds" in message:
        return json_response(413, {"error": message})
      return json_response(400, {"error": message or "Invalid request"})
    except RuntimeError as exc:
      return json_response(500, {"error": str(exc) or "Server error"})
    except Exception as exc:
      print("Transcribe error", exc)
      return json_response(500, {"error": "Server error"})
    finally:
      try:
        if os.path.exists(input_path):
          os.remove(input_path)
      except Exception:
        pass
      if DELETE_UPLOAD_AFTER_TRANSCRIBE:
        delete_s3_object(UPLOAD_BUCKET, key)

  if path not in ("/", "/transcribe"):
    return json_response(404, {"error": "Not found."})

  raw = None
  mime_type = content_type or "application/octet-stream"
  is_json = content_type == "application/json" or content_type.endswith("+json")
  if is_json:
    payload = parse_body(event)
    if not isinstance(payload, dict):
      return json_response(400, {"error": "Invalid JSON payload."})
    raw, error = decode_audio_b64(payload)
    if error:
      status = 413 if "exceeds" in error else 400
      return json_response(status, {"error": error})
    mime_type = parse_content_type(
      payload.get("mime_type")
      or payload.get("mimeType")
      or payload.get("content_type")
      or payload.get("contentType")
      or mime_type
    ) or "application/octet-stream"
  else:
    raw = decode_body_bytes(event)
    if not raw:
      return json_response(400, {"error": "Missing request body."})
    if len(raw) > MAX_DIRECT_UPLOAD_BYTES:
      return json_response(413, {"error": f"Audio payload exceeds {MAX_DIRECT_UPLOAD_BYTES} bytes."})
    mime_type = mime_type or "application/octet-stream"

  converted = False
  try:
    probably_wav = (
      mime_type in ("audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave")
      or raw[:4] == b"RIFF"
    )
    if probably_wav:
      audio, sample_rate, duration_seconds = wav_to_float32_mono_16k(raw)
    else:
      raise ValueError("Non-WAV input.")
  except ValueError as exc:
    message = str(exc)
    if message.startswith("Audio duration exceeds"):
      return json_response(413, {"error": message})
    try:
      wav_bytes = ffmpeg_to_wav_16k_mono(raw, mime_type)
      converted = True
      audio, sample_rate, duration_seconds = wav_to_float32_mono_16k(wav_bytes)
    except ValueError as exc2:
      message = str(exc2)
      if message.startswith("Audio duration exceeds"):
        return json_response(413, {"error": message})
      if message.startswith("No audio track found") or message.startswith("Unsupported audio format"):
        return json_response(415, {"error": message})
      return json_response(415, {"error": message})
    except RuntimeError as exc2:
      return json_response(500, {"error": str(exc2) or "Server error"})
  except RuntimeError as exc:
    return json_response(500, {"error": str(exc) or "Server error"})
  except Exception:
    return json_response(400, {"error": "Invalid audio payload."})

  try:
    part_minutes = query_value(event, "part_minutes") or query_value(event, "partMinutes") or DEFAULT_PART_MINUTES
    transcript, parts, part_seconds = transcribe_long_form(audio, sample_rate, part_minutes)
    duration_ms = int((time.time() - start) * 1000)
    return json_response(200, {
      "transcript": transcript,
      "parts": parts,
      "part_seconds": part_seconds,
      "model": MODEL_ID,
      "language": "en",
      "audio_seconds": round(float(duration_seconds), 4),
      "duration_ms": duration_ms,
      "input_mime_type": mime_type,
      "converted_to_wav": converted
    })
  except Exception as exc:
    print("Transcribe error", exc)
    return json_response(500, {"error": "Server error"})
