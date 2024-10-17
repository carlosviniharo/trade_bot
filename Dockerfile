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

# Copy and install Python requirements
# Create a virtual environment
RUN python3 -m venv /venv
# Activate virtual environment and install Python dependencies
COPY requirements.txt .
RUN /bin/bash -c "source venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt"
# Download and build TA-Lib from source
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr \
    && make && make install \
    && rm -rf /tmp/ta-lib*

RUN /venv/bin/pip install --no-cache-dir ta-lib

# Stage 2: Final runtime environment (Python Slim)
FROM python:3.12-slim

# Copy the virtual environment from the builder stage
COPY --from=builder / /

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Set the default command to run the app
CMD ["bash", "-c", "source /venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"]