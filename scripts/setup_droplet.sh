#!/usr/bin/env bash
# ==============================================================================
# Trade Bot — DigitalOcean Droplet Startup & Provisioning Script
# ==============================================================================
# This script configures a clean Ubuntu 24.04 / 22.04 LTS Droplet for Trade Bot.
#
# USAGE OPTIONS:
# 1. DigitalOcean User Data (Recommended):
#    Paste the entire content of this script into the "User Data" field when
#    creating your droplet under "Advanced Options".
#
# 2. Run manually on a fresh droplet via SSH as root:
#    curl -sSL https://raw.githubusercontent.com/carlosviniharo/trade_bot/main/scripts/setup_droplet.sh | bash
#    (or copy this file to the droplet and run: sudo bash setup_droplet.sh)
# ==============================================================================

set -euo pipefail

# Ensure script runs as root
if [ "$(id -u)" -ne 0 ]; then
    echo "[-] Error: This script must be run as root." >&2
    exit 1
fi

LOG_FILE="/var/log/tradebot_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "==> Starting Trade Bot Droplet Provisioning: $(date -u)"
echo "============================================================"

export DEBIAN_FRONTEND=noninteractive
DEPLOY_USER="deploy"
APP_DIR="/home/${DEPLOY_USER}/trade_bot"

# ------------------------------------------------------------------------------
# 1. Create 2GB Swap Space (Protects against OOM crashes on small droplets)
# ------------------------------------------------------------------------------
if ! swapon --show | grep -q "/swapfile"; then
    echo "==> [1/7] Creating 2GB swapfile..."
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab

    # Reduce swappiness to 10 for better server performance
    sysctl vm.swappiness=10
    echo 'vm.swappiness=10' >> /etc/sysctl.d/99-swappiness.conf
else
    echo "==> [1/7] Swap already exists, skipping."
fi

# ------------------------------------------------------------------------------
# 2. System Update & Essential Packages
# ------------------------------------------------------------------------------
echo "==> [2/7] Updating apt repositories and installing prerequisites..."
apt-get update
apt-get upgrade -y
apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    htop \
    jq \
    ufw \
    fail2ban \
    unattended-upgrades

# ------------------------------------------------------------------------------
# 3. Create Deploy User with Sudo and Docker privileges
# ------------------------------------------------------------------------------
echo "==> [3/7] Setting up non-root deploy user (${DEPLOY_USER})..."
if ! id -u "$DEPLOY_USER" >/dev/null 2>&1; then
    useradd -m -s /bin/bash "$DEPLOY_USER"
fi

usermod -aG sudo "$DEPLOY_USER"

# Configure passwordless sudo for automation
echo "${DEPLOY_USER} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-${DEPLOY_USER}"
chmod 0440 "/etc/sudoers.d/90-${DEPLOY_USER}"

# Copy authorized_keys from root if present
if [ -f /root/.ssh/authorized_keys ]; then
    mkdir -p "/home/${DEPLOY_USER}/.ssh"
    cp /root/.ssh/authorized_keys "/home/${DEPLOY_USER}/.ssh/authorized_keys"
    chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "/home/${DEPLOY_USER}/.ssh"
    chmod 700 "/home/${DEPLOY_USER}/.ssh"
    chmod 600 "/home/${DEPLOY_USER}/.ssh/authorized_keys"
    echo "    SSH authorized_keys copied to ${DEPLOY_USER}."
fi

# ------------------------------------------------------------------------------
# 4. Install Official Docker Engine & Docker Compose Plugin
# ------------------------------------------------------------------------------
echo "==> [4/7] Installing Docker CE and Compose plugin..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable & start Docker service
systemctl enable docker
systemctl start docker

# Add deploy user to docker group
usermod -aG docker "$DEPLOY_USER"

# ------------------------------------------------------------------------------
# 5. Configure Firewall (UFW) & Fail2ban
# ------------------------------------------------------------------------------
echo "==> [5/7] Configuring UFW Firewall and Fail2ban..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment "SSH"
ufw allow 80/tcp comment "HTTP (Nginx)"
ufw allow 443/tcp comment "HTTPS (Nginx SSL)"
ufw --force enable

systemctl enable fail2ban
systemctl start fail2ban

# ------------------------------------------------------------------------------
# 6. Prepare Application Directory
# ------------------------------------------------------------------------------
echo "==> [6/7] Preparing application directory at ${APP_DIR}..."
mkdir -p "$APP_DIR"
chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "$APP_DIR"

# Clone repository if not already cloned
if [ ! -d "${APP_DIR}/.git" ]; then
    echo "    Cloning trade_bot repository into ${APP_DIR}..."
    sudo -u "$DEPLOY_USER" git clone https://github.com/carlosviniharo/trade_bot.git "$APP_DIR" || {
        echo "    Note: Git clone skipped or failed (will be handled by GitHub Actions CI/CD)."
    }
fi

# ------------------------------------------------------------------------------
# 7. Verification & Summary
# ------------------------------------------------------------------------------
echo "==> [7/7] Verifying installation..."
echo "------------------------------------------------------------"
docker --version
docker compose version
echo "UFW Status:"
ufw status verbose
echo "Swap Status:"
free -h
echo "------------------------------------------------------------"
echo "✅ Trade Bot Droplet provisioning complete!"
echo "Droplet IP: $(curl -s https://ipinfo.io/ip || hostname -I | awk '{print $1}')"
echo "Deploy User: ${DEPLOY_USER}"
echo "Project Directory: ${APP_DIR}"
echo "============================================================"
