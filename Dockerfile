## Stage 1: Builder
# Keep the builder interpreter identical to the runtime interpreter.
FROM python:3.12-slim AS builder

# Copy uv binary from official image (no apt install needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install required build tools and dependencies for TA-Lib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and build TA-Lib from source
WORKDIR /tmp
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd /tmp/ta-lib-0.6.4 && \
    ./configure --prefix=/usr && \
    make && make install && \
    rm -rf /tmp/ta-lib-0.6.4*

# Create virtual environment outside /app so host volume mounts (-v ${PWD}:/app) do not overwrite it
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

# Set the working directory
WORKDIR /app

# Copy only dependency files first (maximises Docker layer cache)
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev deps, no project itself)
RUN uv sync --frozen --no-dev --no-install-project

# Install TA-Lib Python wrapper inside the virtual environment
RUN uv pip install --python /opt/venv/bin/python --no-cache ta-lib

## Stage 2: Final runtime environment (Python Slim)
FROM python:3.12-slim

# Copy only the application runtime dependencies from the builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/lib/libta_lib.so* /usr/lib/

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
