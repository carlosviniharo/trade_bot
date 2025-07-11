## Stage 1: Builder
FROM ubuntu:latest AS builder

# Install required build tools and dependencies for TA-Lib and Python
#Install required build tools and dependencies for TA-Lib and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    gcc \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment using Python 3.12
RUN python3 -m venv /venv

# Install pip for Python 3.12 within the virtual environment
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

# Copy everything from the builder stage (this includes the virtual environment and TA-Lib)
COPY --from=builder / /

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Start an interactive shell for debugging
#CMD ["/bin/bash"]

# Use the virtual environment Python for the command
CMD ["/venv/bin/python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# Command used for Kubernetes
#CMD ["/venv/bin/python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
