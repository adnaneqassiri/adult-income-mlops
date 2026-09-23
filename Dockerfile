FROM python:3.10-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml .

RUN uv sync --no-dev --no-install-project

COPY src ./src

COPY logs ./logs

RUN uv sync --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"
ENV MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
