FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=${LAMBDA_TASK_ROOT}/hf
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false

# Install static ffmpeg/ffprobe for audio/video conversion.
RUN python - <<'PY'
import os
import shutil
import stat
import tarfile
import urllib.request

url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
archive_path = "/tmp/ffmpeg.tar.xz"
extract_root = "/tmp/ffmpeg-extract"

os.makedirs(extract_root, exist_ok=True)
urllib.request.urlretrieve(url, archive_path)

with tarfile.open(archive_path, mode="r:*") as tar:
  root_abs = os.path.abspath(extract_root)
  for member in tar.getmembers():
    member_path = os.path.abspath(os.path.join(extract_root, member.name))
    if not (member_path == root_abs or member_path.startswith(root_abs + os.sep)):
      raise RuntimeError(f"Blocked path traversal in tar member: {member.name}")
  tar.extractall(extract_root)

entries = sorted([e for e in os.listdir(extract_root) if e.startswith("ffmpeg") and e.endswith("static")])
if not entries:
  raise RuntimeError("Unable to locate extracted ffmpeg directory.")

src_dir = os.path.join(extract_root, entries[0])
for name in ("ffmpeg", "ffprobe"):
  src = os.path.join(src_dir, name)
  dst = f"/usr/local/bin/{name}"
  shutil.copy2(src, dst)
  os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
  print("Installed", dst)

try:
  os.remove(archive_path)
except OSError:
  pass
shutil.rmtree(extract_root, ignore_errors=True)
PY

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image to avoid first-request downloads.
RUN python - <<'PY'
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = "openai/whisper-tiny.en"
WhisperProcessor.from_pretrained(model_id, cache_dir="/var/task/hf")
WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir="/var/task/hf")
print("Downloaded", model_id)
PY

COPY app.py ${LAMBDA_TASK_ROOT}/app.py

CMD ["app.handler"]
