#!/usr/bin/env bash

CURRENT_USER=$(whoami)

# Fix permissions if needed
if [ ! -w "." ]; then
    echo "⚠️  Fixing permissions..."
    sudo chown -R "$CURRENT_USER:$CURRENT_USER" . 2>/dev/null
fi

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Git safe directory
git config --global --add safe.directory "$(pwd)" 2>/dev/null

echo "✅ Ready! Python: $(python --version 2>&1 | cut -d' ' -f2)"

if [ -d ".venv" ]; then
    echo "💡 Activate venv: source .venv/bin/activate"
elif [ -f "pyproject.toml" ]; then
    echo "💡 Create venv: poetry install"
fi