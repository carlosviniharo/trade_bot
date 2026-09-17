# trade_bot

This is a microservice to extract high changes in the volume of the coins listed on Binance Futures.

## Prerequisites

- [Python 3.12+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- [Docker](https://docs.docker.com/get-docker/) (for containerised deployment)

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/carlosviniharo/trade_bot.git
cd trade_bot
uv sync            # installs all deps (prod + dev) and creates .venv/
```

### 2. Run the app locally

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> **Note:** `uv run` automatically activates the `.venv` — no need to run `source .venv/bin/activate`.

### 3. Run tests

```bash
uv run pytest                          # run all tests
uv run pytest --cov=app --cov-report=term-missing   # with coverage
```

### 4. Lint and format

```bash
# Check for lint errors
uv run ruff check app/ tests/

# Auto-fix what ruff can fix
uv run ruff check app/ tests/ --fix

# Check formatting (dry-run)
uv run ruff format app/ tests/ --check

# Apply formatting
uv run ruff format app/ tests/
```

## Docker

### How to create the MongoDB container

```bash
docker run -d -p 27017:27017 --name mongo mongo
```

### Build and run with Docker

```bash
docker build -t trade_bot .
docker run -v $(pwd):/app -p 8000:8000 trade_bot      # Linux/macOS
docker run -v ${PWD}:/app -p 8000:8000 trade_bot       # PowerShell
```

> **Note:** TA-Lib is compiled from source inside the Dockerfile. Do **not** add it to `pyproject.toml`.

### Debug the container

```bash
docker run -it --entrypoint /bin/bash trade_bot
```

### Run with Docker Compose

```bash
docker compose up --build
```

## Kubernetes Deployment

### Navigate to the k8s directory

```bash
cd k8s
```

### Registry for the Docker image

```
us-central1-docker.pkg.dev/inspired-oath-441023-v1/docker-repo/
```

### Connect to the cluster

```bash
gcloud container clusters get-credentials my-first-cluster-2 --zone northamerica-northeast2-a --project inspired-oath-441023-v1
```

### Create secrets from `.env`

```bash
kubectl create secret generic tradebot-env --from-env-file=.env
```

### Deploy from the k8s directory

```bash
kubectl apply -f .
```
