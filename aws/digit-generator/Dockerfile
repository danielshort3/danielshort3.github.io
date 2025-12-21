FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ${LAMBDA_TASK_ROOT}/app.py
COPY latent_stats.json ${LAMBDA_TASK_ROOT}/latent_stats.json
COPY model ${LAMBDA_TASK_ROOT}/model

CMD ["app.handler"]
