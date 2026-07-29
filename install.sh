#!/usr/bin/env bash
set -euo pipefail
umask 077

cd "$(dirname "$0")"

VAP_HOME="${VAP_HOME:-$HOME/.vap}"
BIN_DIR="$VAP_HOME/bin"
PERFETTO_HOME="$VAP_HOME/perfetto-home"
VENV_DIR="$VAP_HOME/venv"
UV_CACHE_DIR="$VAP_HOME/cache/uv"
UV_PYTHON_INSTALL_DIR="$VAP_HOME/uv-python"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ "$OS" != "Linux" || "$ARCH" != "x86_64" ]]; then
    echo "Unsupported installer platform: $OS/$ARCH" >&2
    echo "Install uv and Perfetto trace_processor manually, then re-run with supported artifacts." >&2
    exit 1
fi

for command in curl install mktemp sha256sum tar; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command is not available: $command" >&2
        exit 1
    fi
done

for project_file in pyproject.toml example-config.json; do
    if [[ ! -f "$project_file" ]]; then
        echo "Run install.sh from a complete VAP source checkout; missing $project_file" >&2
        exit 1
    fi
done

mkdir -p "$BIN_DIR" "$PERFETTO_HOME" "$VAP_HOME/logs" "$VAP_HOME/tmp/configs" "$VAP_HOME/cache" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

UV_VERSION="0.12.0"
UV_ARCHIVE="uv-x86_64-unknown-linux-gnu.tar.gz"
UV_URL="https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/${UV_ARCHIVE}"
UV_SHA256="eaf842262aa1c418d8ecc5605f02ee1ebfd369124fa48548e85f9481a47831a9"
PERFETTO_VERSION="v57.2"
PERFETTO_URL="https://commondatastorage.googleapis.com/perfetto-luci-artifacts/${PERFETTO_VERSION}/linux-amd64/trace_processor_shell"
PERFETTO_SHA256="55ba613fc6d4f71df81eee2dbfc293020063655c241b3e314bff75345b802684"

TMP_DIR="$(mktemp -d)"
UV_STAGE="$BIN_DIR/.uv.$$"
PERFETTO_STAGE="$BIN_DIR/.trace_processor.$$"
PERFETTO_VERSION_STAGE="$BIN_DIR/.trace_processor.version.$$"
trap 'rm -rf "$TMP_DIR"; rm -f "$UV_STAGE" "$PERFETTO_STAGE" "$PERFETTO_VERSION_STAGE"' EXIT

download_verified() {
    local url="$1"
    local sha256="$2"
    local output="$3"
    curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error "$url" -o "$output"
    printf '%s  %s\n' "$sha256" "$output" | sha256sum --check --status
}

UV_BIN="$BIN_DIR/uv"
if [[ ! -x "$UV_BIN" ]] || [[ "$("$UV_BIN" --version 2>/dev/null || true)" != "uv ${UV_VERSION}" ]]; then
    download_verified "$UV_URL" "$UV_SHA256" "$TMP_DIR/$UV_ARCHIVE"
    tar -xzf "$TMP_DIR/$UV_ARCHIVE" -C "$TMP_DIR"
    install -m 0755 "$TMP_DIR/uv-x86_64-unknown-linux-gnu/uv" "$UV_STAGE"
    mv -f "$UV_STAGE" "$UV_BIN"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
    rm -rf "$VENV_DIR"
    "$UV_BIN" venv "$VENV_DIR" --python 3.12
fi

"$UV_BIN" pip install --python "$VENV_DIR/bin/python" -e .
"$VENV_DIR/bin/vap" --help >/dev/null

if [[ ! -f "$VAP_HOME/config.json" ]]; then
    install -m 0600 example-config.json "$VAP_HOME/config.json"
else
    chmod 0600 "$VAP_HOME/config.json"
fi

if [[ ! -x "$BIN_DIR/trace_processor" ]] || [[ ! -f "$BIN_DIR/trace_processor.version" ]] || [[ "$(<"$BIN_DIR/trace_processor.version")" != "$PERFETTO_VERSION" ]]; then
    download_verified "$PERFETTO_URL" "$PERFETTO_SHA256" "$TMP_DIR/trace_processor"
    install -m 0755 "$TMP_DIR/trace_processor" "$PERFETTO_STAGE"
    mv -f "$PERFETTO_STAGE" "$BIN_DIR/trace_processor"
    printf '%s\n' "$PERFETTO_VERSION" > "$PERFETTO_VERSION_STAGE"
    mv -f "$PERFETTO_VERSION_STAGE" "$BIN_DIR/trace_processor.version"
fi

HOME="$PERFETTO_HOME" "$BIN_DIR/trace_processor" --help >/dev/null

USER_BIN_DIR="$HOME/.local/bin"
VAP_WRAPPER="$USER_BIN_DIR/vap"
mkdir -p "$USER_BIN_DIR"
if [[ -e "$VAP_WRAPPER" ]] && ! grep -Fq "$VENV_DIR/bin/vap" "$VAP_WRAPPER"; then
    echo "Refusing to overwrite an unmanaged command at $VAP_WRAPPER" >&2
    exit 1
fi
cat > "$VAP_WRAPPER" <<EOF
#!/usr/bin/env bash
VAP_HOME="\${VAP_HOME:-$VAP_HOME}" exec "$VENV_DIR/bin/vap" "\$@"
EOF
chmod +x "$VAP_WRAPPER"

# Marker used by uninstall.sh to distinguish a managed VAP runtime directory
# from an unrelated user directory.
printf 'installed_from=%s\n' "$(pwd -P)" > "$VAP_HOME/.vap-installed"
chmod 0600 "$VAP_HOME/.vap-installed"

echo "VAP installed successfully."
echo "Runtime files: $VAP_HOME"
echo "Command: $VAP_WRAPPER"
if [[ ":$PATH:" == *":$USER_BIN_DIR:"* ]]; then
    echo "Run: vap start"
else
    echo "Add $USER_BIN_DIR to PATH:"
    echo "  export PATH=\"$USER_BIN_DIR:\$PATH\""
    echo "Then run: vap start"
fi