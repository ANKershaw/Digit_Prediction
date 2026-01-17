FROM python:3.13.1-slim-bookworm
LABEL authors="amandakershaw"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY  ".python-version" ".python-version"
COPY "docker.uv.lock" "uv.lock"
COPY "docker.pyproject.toml" "pyproject.toml"

RUN uv sync --locked

COPY "predict.py" "digit_classifier_scratch.onnx" ./

EXPOSE 8080

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]