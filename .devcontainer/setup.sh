#!/usr/bin/env bash
set +e

echo "========================================"
echo "🚀 DevContainer Setup"
echo "========================================"

CURRENT_USER=$(whoami)
USER_HOME="$HOME"

echo "User: $CURRENT_USER"
echo ""

# Install Poetry
echo "📦 Installing Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 - || {
        echo "❌ Poetry installation failed"
        exit 1
    }
    export PATH="$USER_HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$USER_HOME/.bashrc"
    echo "✅ Poetry installed"
else
    echo "✅ Poetry already installed"
fi

# Configure Poetry
echo ""
echo "⚙️  Configuring Poetry..."
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true
echo "✅ Poetry configured"

# Git config
echo ""
echo "🔧 Configuring Git..."
git config --global --add safe.directory "*"
git config --global commit.gpgsign false
echo "✅ Git configured"

# Fix permissions
echo ""
echo "🔐 Fixing workspace permissions..."
WORKSPACE_DIR="$(pwd)"
sudo chown -R "$CURRENT_USER:$CURRENT_USER" "$WORKSPACE_DIR" 2>&1 || {
    echo "⚠️  Could not fix permissions automatically"
    echo "You may need to run: sudo chown -R $CURRENT_USER:$CURRENT_USER ."
}

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    poetry install 2>&1 || {
        echo "⚠️  Poetry install had issues. You can run 'poetry install' manually."
    }
    
    if [ -d ".venv" ]; then
        echo "✅ Virtual environment created"
    else
        echo "⚠️  No .venv created - check permissions"
    fi
else
    echo "ℹ️  No pyproject.toml found"
fi

# Shell aliases
echo ""
echo "🎨 Adding aliases..."
cat >> "$USER_HOME/.bashrc" << 'EOF'

# Python
alias py='python'
alias venv='source .venv/bin/activate'

# Poetry
alias pi='poetry install'
alias pa='poetry add'
alias pr='poetry run'

# Git
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'

# Directory
alias ll='ls -alh'
alias ..='cd ..'
EOF

echo "✅ Aliases added"

echo ""
echo "========================================"
echo "✨ Setup complete!"
echo "========================================"
echo ""
echo "🎯 Quick commands:"
echo "  poetry install    - Install dependencies"
echo "  poetry run pytest - Run tests"
echo "  source .venv/bin/activate - Activate venv"
echo ""

# Always exit 0 so container doesn't fail
exit 0