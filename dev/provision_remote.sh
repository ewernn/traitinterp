#!/bin/bash
# Provision a new Vast.ai instance from local machine.
# Does everything: copies .env, creates dev user, installs deps, clones repo, configures R2.
#
# Usage:
#   ./dev/provision_remote.sh "ssh -p 45396 root@207.180.148.74 -L 8080:localhost:8080"
#   ./dev/provision_remote.sh -p 45396 -h 207.180.148.74

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
if [[ "$1" == ssh* ]]; then
    PORT=$(echo "$1" | sed -n 's/.*-p \([0-9]*\).*/\1/p')
    HOST=$(echo "$1" | sed -n 's/.*root@\([0-9.]*\).*/\1/p')
else
    while getopts "p:h:" opt; do
        case $opt in
            p) PORT="$OPTARG" ;;
            h) HOST="$OPTARG" ;;
        esac
    done
fi

if [[ -z "$PORT" || -z "$HOST" ]]; then
    echo "Usage: $0 \"ssh -p PORT root@HOST ...\""
    echo "   or: $0 -p PORT -h HOST"
    exit 1
fi

SSH="ssh -p $PORT root@$HOST"
SCP="scp -P $PORT"

echo "Provisioning $HOST:$PORT"
echo ""

# --- Step 1: Copy .env to remote ---
ENV_FILE="$REPO_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "[1/5] Uploading .env..."
    $SCP "$ENV_FILE" root@"$HOST":/tmp/.env
else
    echo "[1/5] ERROR: .env not found at $ENV_FILE"
    exit 1
fi

# --- Step 2: Copy Claude commands ---
CLAUDE_COMMANDS="$HOME/.claude/commands"
if [[ -d "$CLAUDE_COMMANDS" ]]; then
    echo "[2/5] Uploading Claude commands..."
    $SCP -r "$CLAUDE_COMMANDS" root@"$HOST":/tmp/claude-commands
else
    echo "[2/5] No Claude commands found, skipping"
fi

# --- Step 3: Run provisioning on remote ---
echo "[3/5] Provisioning remote (install deps, create user, clone repo)..."

# Source .env locally to get tokens for the remote script
source "$ENV_FILE"

$SSH bash -s << REMOTEEOF
set -e

# Disable tmux
touch /root/.no_auto_tmux

# System packages
apt-get update -qq
apt-get install -y -qq git curl rclone > /dev/null 2>&1

# Create dev user
useradd -m -s /bin/bash dev 2>/dev/null || true
echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers 2>/dev/null || true
touch /home/dev/.no_auto_tmux
chown dev:dev /home/dev/.no_auto_tmux

# Env vars for all users
cat > /etc/environment << EOF
GITHUB_TOKEN=${GITHUB_TOKEN}
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}
R2_ENDPOINT=${R2_ENDPOINT}
R2_BUCKET_NAME=${R2_BUCKET_NAME:-trait-interp-bucket}
HF_TOKEN=${HF_TOKEN}
OPENAI_API_KEY=${OPENAI_API_KEY}
HF_HOME=/home/dev/.cache/huggingface
EOF

# Git config
su - dev -c 'git config --global user.name "ewernn"'
su - dev -c 'git config --global user.email "ewernn@users.noreply.github.com"'

# Clone repo
if [ ! -d /home/dev/trait-interp ]; then
    su - dev -c "git clone https://${GITHUB_TOKEN}@github.com/ewernn/traitinterp.git /home/dev/trait-interp && cd /home/dev/trait-interp && git checkout dev"
else
    su - dev -c "cd /home/dev/trait-interp && git pull"
fi

# Move .env into repo
mv /tmp/.env /home/dev/trait-interp/.env 2>/dev/null || true
chown dev:dev /home/dev/trait-interp/.env

# Move Claude commands
if [ -d /tmp/claude-commands ]; then
    mkdir -p /home/dev/.claude/commands
    cp -r /tmp/claude-commands/* /home/dev/.claude/commands/
    rm -rf /tmp/claude-commands
    chown -R dev:dev /home/dev/.claude
fi

# Bashrc
grep -q "trait-interp" /home/dev/.bashrc 2>/dev/null || cat >> /home/dev/.bashrc << 'BASHEOF'
export PATH="\$HOME/.local/bin:\$PATH"
source ~/trait-interp/.venv/bin/activate 2>/dev/null || true
alias cla='claude --dangerously-skip-permissions'
alias clar='claude --dangerously-skip-permissions --resume'
alias cl='clear'
cd ~/trait-interp
BASHEOF
chown dev:dev /home/dev/.bashrc

# Configure rclone for R2
mkdir -p /home/dev/.config/rclone
cat > /home/dev/.config/rclone/rclone.conf << EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY_ID}
secret_access_key = ${R2_SECRET_ACCESS_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF
chown -R dev:dev /home/dev/.config/rclone

echo "Remote provisioning done."
REMOTEEOF

# --- Step 4: Install Python deps ---
echo "[4/5] Installing Python deps..."
$SSH 'su - dev -c "cd ~/trait-interp && pip3 install --break-system-packages uv 2>/dev/null; uv venv 2>/dev/null; uv pip install -r requirements.txt 2>&1 | tail -3"'

# --- Step 5: Install Claude Code ---
echo "[5/5] Installing Claude Code..."
$SSH 'su - dev -c "curl -fsSL https://claude.ai/install.sh | bash"' 2>/dev/null || echo "Claude Code install may need manual setup"

echo ""
echo "Done! SSH in as:"
echo "  ssh -p $PORT dev@$HOST"
echo ""
echo "Then log into Claude Code (interactive, one-time):"
echo "  claude"
echo "  # Follow the browser auth link, paste code back"
echo ""
echo "Remember: R2 push before killing the instance!"
echo "  cd ~/trait-interp && ./dev/r2_push.sh --only starter"
