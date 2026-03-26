FROM python:3.11-slim

WORKDIR /plant_detect

RUN pip install --upgrade pip setuptools wheel
COPY backend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY backend/ backend/

ENV PYTHONPATH=/plant_detect/backend/app/src


CMD uvicorn backend.app.api.main:api --host 0.0.0.0 --port ${PORT:-8080}