# Stage 1: Build TA-Lib and Python dependencies in an Ubuntu environment
FROM ubuntu:latest AS builder

# Install required build tools and dependencies for TA-Lib and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    gcc \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and install Python dependencies
RUN python3 -m venv /venv
COPY requirements.txt .
RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt

# Download and build TA-Lib from source
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && \
    make && make install && \
    rm -rf /tmp/ta-lib*

# Install TA-Lib Python wrapper inside the virtual environment
RUN /venv/bin/pip install --no-cache-dir ta-lib

# Stage 2: Final runtime environment (Python Slim)
FROM python:3.12-slim

# Install TA-Lib runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy everything from the builder stage (this includes the virtual environment and TA-Lib)
COPY --from=builder / /

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Use the virtual environment Python for the command
CMD ["/venv/bin/python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
# Command used for Kubernetes
#CMD ["/venv/bin/python", "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "80"]
