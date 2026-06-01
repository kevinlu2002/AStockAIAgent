FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASHARE_HOST=0.0.0.0 \
    ASHARE_PORT=7860

WORKDIR /app

RUN python -m pip install --upgrade pip
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install -r requirements.txt

COPY configs ./configs
COPY data ./data
COPY models ./models
COPY reports ./reports
COPY scripts ./scripts
COPY web ./web
COPY README.md ./

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3).read()"

CMD ["python", "scripts/serve_production.py"]
