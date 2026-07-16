#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VAP_HOME="${VAP_HOME:-$HOME/.vap}"
BIN_DIR="$VAP_HOME/bin"
PERFETTO_HOME="$VAP_HOME/perfetto-home"
VENV_DIR="$VAP_HOME/venv"
UV_CACHE_DIR="$VAP_HOME/cache/uv"
UV_PYTHON_INSTALL_DIR="$VAP_HOME/uv-python"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
mkdir -p "$BIN_DIR" "$PERFETTO_HOME" "$VAP_HOME/logs" "$VAP_HOME/tmp/configs" "$VAP_HOME/cache" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
else
    UV_BIN="$BIN_DIR/uv"
    if [[ ! -x "$UV_BIN" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$BIN_DIR" sh
    fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$UV_BIN" venv "$VENV_DIR" --python 3.12
fi

"$UV_BIN" pip install --python "$VENV_DIR/bin/python" -e .
if [[ ! -f "$VAP_HOME/config.json" ]]; then
    cp example-config.json "$VAP_HOME/config.json"
fi
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/vap" <<EOF
#!/usr/bin/env bash
VAP_HOME="\${VAP_HOME:-$VAP_HOME}" exec "$VENV_DIR/bin/vap" "\$@"
EOF
chmod +x "$HOME/.local/bin/vap"
mkdir -p "$PERFETTO_HOME"

if [[ ! -x "$BIN_DIR/trace_processor" ]]; then
    curl -fL https://get.perfetto.dev/trace_processor -o "$BIN_DIR/trace_processor"
    chmod +x "$BIN_DIR/trace_processor"
fi

HOME="$PERFETTO_HOME" "$BIN_DIR/trace_processor" --help >/dev/null

echo "VAP installed. Runtime files are in $VAP_HOME."
echo "Command installed at $HOME/.local/bin/vap. Add $HOME/.local/bin to PATH if needed."