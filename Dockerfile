FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY config.py /app/config.py
COPY db.py /app/db.py
COPY translations.py /app/translations.py
COPY repositories /app/repositories
COPY services /app/services
COPY src /app/src
COPY templates /app/templates
COPY static /app/static
COPY config /app/config
COPY models /app/models
COPY data /app/data

RUN mkdir -p /app/uploads /app/output /app/temp /app/logs /app/staging /app/src/DBNet/weights /mnt/data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
