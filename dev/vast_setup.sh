#!/bin/bash
# Transfer .env and Claude commands to a Vast.ai instance.
# Run AFTER vast_onstart.sh has provisioned the instance.
#
# What it does:
#   - Uploads .env to ~/traitinterp/.env (credentials for HF, OpenAI, R2, etc.)
#   - Uploads ~/.claude/commands/ for Claude Code slash commands
#   - Sources .env values into /etc/environment so all shells see them
#
# Usage:
#   ./dev/vast_setup.sh "ssh -p 45396 root@207.180.148.74 -L 8080:localhost:8080"
#   ./dev/vast_setup.sh -p 45396 -h 207.180.148.74

set -e

# Parse arguments
if [[ "$1" == ssh* ]]; then
    # Parse from SSH command string
    PORT=$(echo "$1" | sed -n 's/.*-p \([0-9]*\).*/\1/p')
    HOST=$(echo "$1" | sed -n 's/.*root@\([0-9.]*\).*/\1/p')
else
    # Parse from flags
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

echo "Setting up Vast.ai instance at $HOST:$PORT"

# Files to transfer
ENV_FILE=".env"
CLAUDE_COMMANDS="$HOME/.claude/commands"

# Transfer .env
if [[ -f "$ENV_FILE" ]]; then
    echo "Uploading .env..."
    scp -P "$PORT" "$ENV_FILE" root@"$HOST":/tmp/.env
else
    echo "Warning: .env not found at $ENV_FILE"
fi

# Transfer Claude commands
if [[ -d "$CLAUDE_COMMANDS" ]]; then
    echo "Uploading Claude commands..."
    scp -P "$PORT" -r "$CLAUDE_COMMANDS" root@"$HOST":/tmp/claude-commands
else
    echo "Warning: Claude commands not found at $CLAUDE_COMMANDS"
fi

# Move files to dev user locations (run after dev user is created)
echo "Moving files to dev user locations..."
ssh -p "$PORT" root@"$HOST" bash -s << 'EOF'
# Wait for dev user to exist (in case script is run before setup)
if id "dev" &>/dev/null; then
    mkdir -p /home/dev/.claude/commands

    # Move .env if it exists
    [[ -f /tmp/.env ]] && mv /tmp/.env /home/dev/traitinterp/.env 2>/dev/null || true

    # Move Claude commands if they exist
    if [[ -d /tmp/claude-commands ]]; then
        cp -r /tmp/claude-commands/* /home/dev/.claude/commands/
        rm -rf /tmp/claude-commands
    fi

    chown -R dev:dev /home/dev/.claude
    [[ -f /home/dev/traitinterp/.env ]] && chown dev:dev /home/dev/traitinterp/.env

    # Source .env values into /etc/environment so all shells + subprocesses see them
    if [[ -f /home/dev/traitinterp/.env ]]; then
        echo "Updating /etc/environment with .env values..."
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ -z "$key" || "$key" == \#* ]] && continue
            # Remove surrounding quotes from value
            value="${value%\"}"
            value="${value#\"}"
            # Update or append to /etc/environment
            if grep -q "^${key}=" /etc/environment 2>/dev/null; then
                sed -i "s|^${key}=.*|${key}=${value}|" /etc/environment
            else
                echo "${key}=${value}" >> /etc/environment
            fi
        done < /home/dev/traitinterp/.env
        echo "  /etc/environment updated"
    fi

    echo "Done! Files moved to /home/dev/"
else
    echo "Dev user doesn't exist yet. Files staged in /tmp/"
    echo "Run the dev user setup first, then re-run this script."
fi
EOF

echo "Setup complete for $HOST:$PORT"
