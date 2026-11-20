# Single-container demo for a Hugging Face Docker Space (CPU basic tier).
# Build from the repository root:
#   docker build -f deploy/huggingface/Dockerfile -t lacunaframe-space .
FROM node:20-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json tsconfig.json vite.config.ts index.html ./
COPY src src
COPY public public
RUN npm ci && npm run build

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app
COPY backend/requirements-lock.txt backend/requirements-lock.txt
RUN pip install --no-cache-dir -r backend/requirements-lock.txt
COPY backend backend
COPY assets assets
COPY public/reports/natural-study.json examples/natural-v02/summary.json
COPY --from=frontend /build/dist dist
USER user
ENV LACUNA_DATA=/tmp/lacunaframe \
    LACUNA_STATIC=/home/user/app/dist \
    LACUNA_CPU_THREADS=2
EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
