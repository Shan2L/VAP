#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BIN_DIR="$PWD/bin"
PERFETTO_HOME="$BIN_DIR/perfetto-home"
mkdir -p "$BIN_DIR"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
else
    UV_BIN="$BIN_DIR/uv"
    if [[ ! -x "$UV_BIN" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$BIN_DIR" sh
    fi
fi

if [[ ! -x .venv/bin/python ]]; then
    "$UV_BIN" venv .venv --python 3.12
fi

"$UV_BIN" pip install --python .venv/bin/python -e .
mkdir -p "$PERFETTO_HOME"

if [[ ! -x "$BIN_DIR/trace_processor" ]]; then
    curl -fL https://get.perfetto.dev/trace_processor -o "$BIN_DIR/trace_processor"
    chmod +x "$BIN_DIR/trace_processor"
fi

HOME="$PERFETTO_HOME" "$BIN_DIR/trace_processor" --help >/dev/null

echo "VAP installed. Local installer binaries are in $BIN_DIR."