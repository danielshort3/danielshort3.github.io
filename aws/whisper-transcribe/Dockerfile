FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=${LAMBDA_TASK_ROOT}/hf
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false

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
