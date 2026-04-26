FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --extra dev --frozen

COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "kolb_loop.ingress.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
