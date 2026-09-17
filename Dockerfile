## Stage 1: Builder
# Keep the builder interpreter identical to the runtime interpreter.  In
# particular, NumPy 1.26.x supports Python 3.9 through 3.12, while
# ubuntu:latest can provide a newer Python release.
FROM python:3.12-slim AS builder

# Install required build tools and dependencies for TA-Lib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment using the pinned Python 3.12 interpreter.
RUN python3 -m venv /venv

# Install pip for Python 3.12 within the virtual environment.
RUN /venv/bin/python -m ensurepip --upgrade

# Install required Python packages from requirements.txt
COPY requirements.txt .
RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt

# Download and build TA-Lib from source
WORKDIR /tmp
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd /tmp/ta-lib-0.6.4 && \
    ./configure --prefix=/usr && \
    make && make install && \
    rm -rf /tmp/ta-lib-0.6.4*

# Install TA-Lib Python wrapper inside the virtual environment
RUN /venv/bin/pip install --no-cache-dir ta-lib

# Stage 2: Final runtime environment (Python Slim)
FROM python:3.12-slim

# Copy only the application runtime dependencies from the builder.
COPY --from=builder /venv /venv
COPY --from=builder /usr/lib/libta_lib.so* /usr/lib/

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Start an interactive shell for debugging
#CMD ["/bin/bash"]

# Use the virtual environment Python for the command
CMD ["/venv/bin/python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]

# Command used for Kubernetes
#CMD ["/venv/bin/python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
