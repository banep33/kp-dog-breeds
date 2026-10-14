# Produkcioni servis: ONNX Runtime, bez PyTorch-a. Izmereno: 605 MB
# (python:3.12-slim + onnxruntime + 37,7 MB modela). Sa torch-om bi bilo visestruko vise.
# Build trazi da `artifacts/` vec postoji (model.onnx + model.json [+ notdog.onnx]).
# Ti fajlovi nisu u gitu, pa se pre build-a preuzimaju ili generisu:
#     python -m dogbreeds.export --checkpoint runs/<run>/best.pt --out artifacts
#     python scripts/export_notdog.py
#
#     docker build -t kp-dog-breeds .
#     docker run --rm -p 8000:8000 kp-dog-breeds
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOG_DB_PATH=/data/service.db

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY dogbreeds/__init__.py dogbreeds/inference.py dogbreeds/notdog.py ./dogbreeds/
COPY service/ ./service/
COPY artifacts/ ./artifacts/
COPY reports/threshold.json ./reports/threshold.json

# dnevnik predikcija ide u volumen, da prezivi restart kontejnera
RUN mkdir -p /data && useradd -m -u 10001 app && chown -R app:app /app /data
USER app
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
