# La inferencia corre sobre series ya agregadas: la CPU es suficiente.
FROM python:3.11-slim

WORKDIR /service
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY parking_ml ./parking_ml
COPY app ./app
COPY artifacts ./artifacts

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
