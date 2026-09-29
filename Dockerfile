FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY asl ./asl
RUN pip install ".[serve]"
# mediapipe lists these as dependencies but the hands solution never imports them
RUN pip uninstall -y jax jaxlib scipy sounddevice ml_dtypes opt_einsum

FROM python:3.12-slim
# mediapipe pulls in a non-headless opencv, which needs these
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000
RUN useradd --create-home app
USER app
WORKDIR /home/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["asl-serve"]
