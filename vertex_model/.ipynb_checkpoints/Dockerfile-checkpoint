# Dockerfile (corrigido)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY model/ /app/model/

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
