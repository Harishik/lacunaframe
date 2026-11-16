FROM python:3.12-slim
WORKDIR /app
# Copy small sources before dependencies: avoids duplicating large installed
# packages in storage drivers that materialize each filesystem layer.
COPY backend backend
COPY assets assets
COPY public/reports/natural-study.json examples/natural-v02/summary.json
RUN --mount=type=secret,id=pip_ca,target=/run/secrets/pip_ca \
    useradd --create-home researcher && \
    if [ -f /run/secrets/pip_ca ]; then \
      PIP_CERT=/run/secrets/pip_ca pip install --no-cache-dir -r backend/requirements-lock.txt; \
    else pip install --no-cache-dir -r backend/requirements-lock.txt; fi
USER researcher
ENV LACUNA_DATA=/tmp/lacunaframe
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
