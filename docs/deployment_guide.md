# Trade Bot — Automated DigitalOcean Droplet Deployment Guide

## 1. Architecture & Deployment Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Developer
    participant GH as GitHub (Actions)
    participant Hub as Docker Hub
    participant Droplet as DigitalOcean Droplet
    participant Mongo as MongoDB Atlas
    participant TG as Telegram

    Dev->>GH: git push origin main
    Note over GH: 1. Lint (Ruff) & Format<br/>2. Pytest (TA-Lib)<br/>3. Build & Push Docker Images
    GH->>Hub: Push trade-bot & trade-bot-nginx (:sha, :latest)
    Note over GH: 4. Deploy via SSH (appleboy/ssh-action)
    GH->>Droplet: SSH connection with GitHub Secrets
    Droplet->>Droplet: Generate .env from Secrets (chmod 600)
    Droplet->>Droplet: Update git config (compose, nginx)
    Droplet->>Hub: docker compose pull
    Droplet->>Droplet: docker compose up -d (fastapi + nginx)
    Droplet->>Droplet: curl health check (:8000/docs)
    Droplet->>Mongo: Connects to MongoDB Atlas
    Droplet->>TG: Scheduled market spike alerts
```

---

## 2. Prerequisites Checklist

| Component | Description |
|-----------|-------------|
| **DigitalOcean Account** | Active account with billing enabled |
| **SSH Key Pair** | `ed25519` or `rsa` key for droplet & GitHub Actions access |
| **Docker Hub Account** | Free Docker Hub account + Personal Access Token |
| **MongoDB Atlas** | Database URI with network access permission |
| **Telegram Bot** | Bot token and chat ID for notifications |

---

## 3. Step 1 — Create the Droplet with the Startup Script

The repository includes [`scripts/setup_droplet.sh`](file:///c:/Users/carlo/OneDrive/Documents/Repositories/trade_bot/scripts/setup_droplet.sh). This script automates 100% of droplet preparation:
- Sets up a **2GB swapfile** (vital to prevent Out-Of-Memory kills on Python/Pandas tasks)
- Updates Ubuntu & installs essential tools (`curl`, `git`, `htop`, `jq`, `ufw`, `fail2ban`)
- Installs the official **Docker Engine** & **Docker Compose plugin**
- Creates a dedicated `deploy` user with passwordless `sudo` and `docker` permissions
- Copies your SSH key to `deploy`
- Configures **UFW firewall** (opens ports 22, 80, 443)
- Prepares `/home/deploy/trade_bot`

### Option A: Using DigitalOcean User Data (Recommended — Zero-Touch)

1. Go to **DigitalOcean Console → Create → Droplets**.
2. Select options:
   - **Distribution**: Ubuntu 24.04 LTS x64
   - **Plan**: Basic Droplet
     - *Minimum*: Regular with SSD — **$6/mo** (1 GB RAM, 1 vCPU, 25 GB SSD) — *Swap ensures it won't crash!*
     - *Recommended*: Premium Intel or AMD — **$12/mo** (2 GB RAM, 1 vCPU, 50 GB SSD)
   - **Datacenter Region**: Choose closest to you or Binance servers (e.g. Frankfurt, New York, Singapore)
   - **Authentication**: **SSH Key** (select your public SSH key)
   - **Advanced Options**: Check **Add Initialization scripts (User Data)**
3. Copy the entire contents of [`scripts/setup_droplet.sh`](file:///c:/Users/carlo/OneDrive/Documents/Repositories/trade_bot/scripts/setup_droplet.sh) and paste it into the **User Data** text area.
4. Set **Hostname**: `trade-bot`
5. Click **Create Droplet**.

Within 2-3 minutes, your droplet will be fully provisioned, secured, and ready for CI/CD.

---

### Option B: Run Manually on an Existing Droplet

If your droplet is already created, SSH into it as `root` and run:

```bash
curl -sSL https://raw.githubusercontent.com/carlosviniharo/trade_bot/main/scripts/setup_droplet.sh | bash
```

Or copy the script to the droplet:

```bash
scp scripts/setup_droplet.sh root@YOUR_DROPLET_IP:/tmp/
ssh root@YOUR_DROPLET_IP "bash /tmp/setup_droplet.sh"
```

---

## 4. Step 2 — Configure GitHub Secrets

Go to your GitHub repository: **Settings → Secrets and variables → Actions → New repository secret**.

Add the following secrets:

### Droplet Connection Secrets

| Secret Name | Value / Description |
|-------------|---------------------|
| `DROPLET_HOST` | Your droplet's public IP address (e.g. `143.198.xxx.xxx`) |
| `DROPLET_USER` | `deploy` |
| `DROPLET_SSH_KEY` | Your **Private** SSH Key (e.g. content of `~/.ssh/id_ed25519` or `~/.ssh/id_rsa`). Must match the public key registered on the droplet. |
| `DROPLET_SSH_PASSPHRASE` | *(Optional)* Passphrase for your private key, if protected. |

> [!TIP]
> To generate a dedicated SSH key specifically for GitHub Actions deployment:
> ```bash
> ssh-keygen -t ed25519 -C "github-actions-tradebot" -f ~/.ssh/tradebot_deploy
> ```
> Add `tradebot_deploy.pub` to `/home/deploy/.ssh/authorized_keys` on your droplet, and paste the contents of `tradebot_deploy` into the GitHub Secret `DROPLET_SSH_KEY`.

---

### Docker Hub Registry Secrets

| Secret Name | Value / Description |
|-------------|---------------------|
| `REGISTRY_USERNAME` | Your Docker Hub username (e.g. `carlosviniharo`) |
| `REGISTRY_PASSWORD` | Your Docker Hub **Personal Access Token (PAT)** *(Account Settings → Security → New Access Token)* |

---

### Application `.env` Secrets

These values are written directly to `/home/deploy/trade_bot/.env` during the deployment step:

| Secret Name | Example Value / Description |
|-------------|-----------------------------|
| `MONGODB_URI` | `mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority` |
| `MONGODB_NAME` | `my_database` |
| `WHATSAPP_TOKEN` | Meta WhatsApp API token *(optional)* |
| `PHONE_NUMBER_ID` | WhatsApp Phone Number ID *(optional)* |
| `TELEGRAM_BOT_TOKEN` | `7235266896:AAGVvFqActKb...` |
| `TELEGRAM_CHAT_ID` | `6402992359` |
| `ENV` | `production` |

*(Optional Alternative: If you prefer managing your `.env` as a single multi-line block, you can add an `ENV_FILE` secret with the full `.env` text; the pipeline prioritizes `ENV_FILE` if present).*

---

## 5. Step 3 — Whitelist Droplet IP in MongoDB Atlas

1. Copy your droplet's public IP address.
2. Open [MongoDB Atlas Dashboard](https://cloud.mongodb.com/).
3. Navigate to **Security → Network Access**.
4. Click **Add IP Address**.
5. Enter your droplet's IP (e.g. `143.198.xxx.xxx/32`) and add comment `DigitalOcean Droplet`.
6. Click **Confirm**.

---

## 6. Step 4 — How the CI/CD Pipeline Deploys

The pipeline in [`.github/workflows/ci-cd.yml`](file:///c:/Users/carlo/OneDrive/Documents/Repositories/trade_bot/.github/workflows/ci-cd.yml) executes automatically on every `git push` to `main`:

1. **Job 1: Lint & Format** — Runs `uv run ruff check` and `uv run ruff format --check`.
2. **Job 2: Test** — Installs TA-Lib C library, installs dependencies with `uv sync --frozen`, and runs `pytest` with coverage.
3. **Job 3: Build & Push** — Builds multi-stage Docker images for both `trade-bot` and `trade-bot-nginx` using GitHub Actions layer caching, and pushes tagged versions (`:sha` and `:latest`) to Docker Hub.
4. **Job 4: Deploy to Droplet**:
   - Connects securely via SSH as `deploy` user.
   - Syncs the repository configs (`docker-compose.yml`, `nginx/nginx.conf`).
   - Writes the `.env` file securely from GitHub Secrets (`chmod 600`).
   - Logs into Docker Hub on the droplet.
   - Runs `docker compose pull` to fetch the pre-built images.
   - Runs `docker compose up -d --remove-orphans` to recreate containers with zero downtime.
   - Runs `docker image prune -f` to clean up dangling layers and protect disk space.
   - Runs a loop healthcheck against `http://localhost:8000/docs` and `http://localhost/docs` to verify application health.

To trigger your first automated deployment:
```bash
git add .
git commit -m "feat: setup automated DigitalOcean deployment and droplet startup script"
git push origin main
```

Watch the pipeline progress under the **Actions** tab on GitHub.

---

## 7. Step 5 — (Optional) Configure Custom Domain & HTTPS

If you have a domain name (e.g. `tradebot.yourdomain.com`):

### 7.1 Point DNS A Record
In your domain DNS manager (Cloudflare, Namecheap, Route53, etc.):
```
A record: tradebot.yourdomain.com -> YOUR_DROPLET_IP
```

### 7.2 Obtain Let's Encrypt Certificate
SSH into the droplet:
```bash
ssh deploy@YOUR_DROPLET_IP
sudo apt install -y certbot
sudo docker compose -f /home/deploy/trade_bot/docker-compose.yml stop nginx
sudo certbot certonly --standalone -d tradebot.yourdomain.com
```

### 7.3 Mount Certificates in `docker-compose.yml`
Update the `nginx` service in `/home/deploy/trade_bot/docker-compose.yml`:
```yaml
  nginx:
    image: ${REGISTRY_USERNAME:-carlosviniharo}/trade-bot-nginx:${IMAGE_TAG:-latest}
    build: ./nginx
    container_name: nginx_proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - fastapi_app
    networks:
      - web
    restart: always
```

### 7.4 Update `nginx/nginx.conf`
Configure SSL termination in `nginx/nginx.conf`:
```nginx
events { worker_connections 4096; }

http {
    server {
        listen 80;
        server_name tradebot.yourdomain.com;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name tradebot.yourdomain.com;

        ssl_certificate /etc/letsencrypt/live/tradebot.yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/tradebot.yourdomain.com/privkey.pem;

        location / {
            proxy_pass http://fastapi_app:8000;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### 7.5 Setup Auto-Renewal Cron
```bash
sudo crontab -e
```
Add:
```
0 3 * * * certbot renew --quiet && docker compose -f /home/deploy/trade_bot/docker-compose.yml restart nginx
```

---

## 8. Server Operations & Maintenance Cheatsheet

Connect to droplet:
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/trade_bot
```

| Task | Command |
|------|---------|
| **Live App Logs** | `docker compose logs -f fastapi_app` |
| **Live Nginx Logs** | `docker compose logs -f nginx` |
| **Container Status** | `docker compose ps` |
| **Resource Usage (RAM/CPU)** | `docker stats` or `htop` |
| **Free Memory & Swap** | `free -h` |
| **Disk Space** | `df -h` |
| **Manual Restart** | `docker compose restart` |
| **Rebuild Locally on Droplet** | `docker compose up -d --build` |
| **View Setup Log (Cloud-Init)** | `cat /var/log/tradebot_setup.log` |
