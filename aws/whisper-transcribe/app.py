import base64
import io
import json
import os
import subprocess
import time
import wave

import audioop
import numpy as np

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-tiny.en")
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "30"))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(5 * 1024 * 1024)))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

RESPONSE_HEADERS = {
  "Content-Type": "application/json"
}

_MODEL = None
_PROCESSOR = None

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

  import torch
  from transformers import WhisperForConditionalGeneration, WhisperProcessor

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
  if len(raw) > MAX_AUDIO_BYTES:
    return None, f"Audio payload exceeds {MAX_AUDIO_BYTES} bytes."
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


def transcribe(audio, sample_rate):
  import torch

  model, processor = load_whisper()
  inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
  forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
  predicted_ids = model.generate(
    inputs.input_features,
    forced_decoder_ids=forced_decoder_ids,
    num_beams=max(1, NUM_BEAMS),
    max_new_tokens=max(1, MAX_NEW_TOKENS)
  )
  text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
  return (text or "").strip()


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
      "max_audio_bytes": MAX_AUDIO_BYTES
    })

  if method != "POST":
    return json_response(405, {"error": "Method not allowed."})

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
    if len(raw) > MAX_AUDIO_BYTES:
      return json_response(413, {"error": f"Audio payload exceeds {MAX_AUDIO_BYTES} bytes."})
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
    transcript = transcribe(audio, sample_rate)
    duration_ms = int((time.time() - start) * 1000)
    return json_response(200, {
      "transcript": transcript,
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
