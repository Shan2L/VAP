#!/usr/bin/env bash
set -euo pipefail
umask 077

PURGE=0
REMOVE_SOURCE=0
ASSUME_YES=0

usage() {
    cat <<'EOF'
Usage: bash uninstall.sh [options]

Options:
  --purge          Also remove config, logs, and all files under VAP_HOME.
  --remove-source  Remove the managed bootstrap source checkout.
  --yes            Do not ask for interactive confirmation.
  -h, --help       Show this help message.

By default, VAP executables and caches are removed while config.json and logs/
are preserved.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --purge)
            PURGE=1
            ;;
        --remove-source)
            REMOVE_SOURCE=1
            ;;
        --yes)
            ASSUME_YES=1
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

for command in realpath rm grep; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command is not available: $command" >&2
        exit 1
    fi
done
if [[ "$REMOVE_SOURCE" -eq 1 ]] && ! command -v git >/dev/null 2>&1; then
    echo "Git is required when using --remove-source." >&2
    exit 1
fi

VAP_HOME="$(realpath -m "${VAP_HOME:-$HOME/.vap}")"
HOME_PATH="$(realpath -m "$HOME")"
VENV_DIR="$VAP_HOME/venv"
USER_BIN_DIR="$HOME/.local/bin"
VAP_WRAPPER="$USER_BIN_DIR/vap"
INSTALL_MARKER="$VAP_HOME/.vap-installed"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
SOURCE_DIR="$(realpath -m "${VAP_SOURCE_DIR:-$DATA_HOME/vap/source}")"
EXPECTED_REPO_URL="https://github.com/Shan2L/VAP.git"

wrapper_is_managed() {
    if [[ -L "$VAP_WRAPPER" ]]; then
        [[ "$(realpath -m "$VAP_WRAPPER")" == "$VENV_DIR/bin/vap" ]]
        return
    fi
    if [[ ! -f "$VAP_WRAPPER" ]]; then
        return 1
    fi
    grep -Fq "# VAP_MANAGED_WRAPPER=1" "$VAP_WRAPPER" \
        || grep -Fq "$VENV_DIR/bin/vap" "$VAP_WRAPPER"
}

# Never allow an empty value, /, or the user's entire home directory to become
# an rm -rf target because of an environment-variable mistake.
case "$VAP_HOME" in
    "" | "/" | "$HOME_PATH")
        echo "Refusing unsafe VAP_HOME: $VAP_HOME" >&2
        exit 1
        ;;
esac
if [[ "$REMOVE_SOURCE" -eq 1 ]]; then
    case "$SOURCE_DIR" in
        "" | "/" | "$HOME_PATH")
            echo "Refusing unsafe VAP_SOURCE_DIR: $SOURCE_DIR" >&2
            exit 1
            ;;
    esac
fi

# New installations have a marker. The executable check keeps the uninstaller
# compatible with installations made before the marker was introduced. A
# managed wrapper check also lets a repeated uninstall clean up a wrapper left
# by an interrupted older uninstall.
if [[ ! -f "$INSTALL_MARKER" ]] \
    && [[ ! -x "$VENV_DIR/bin/vap" ]] \
    && ! wrapper_is_managed; then
    echo "No managed VAP installation found at $VAP_HOME" >&2
    exit 1
fi

if command -v pgrep >/dev/null 2>&1 \
    && pgrep -f "$VENV_DIR/bin/vap" >/dev/null 2>&1; then
    echo "VAP is still running. Stop it before uninstalling." >&2
    exit 1
fi

echo "VAP runtime: $VAP_HOME"
if [[ "$PURGE" -eq 1 ]]; then
    echo "Config and logs will also be deleted."
else
    echo "config.json and logs/ will be preserved."
fi
if [[ "$REMOVE_SOURCE" -eq 1 ]]; then
    echo "Managed source checkout: $SOURCE_DIR"
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
    if [[ ! -t 0 ]]; then
        echo "Interactive confirmation is unavailable; re-run with --yes." >&2
        exit 1
    fi
    read -r -p "Continue uninstalling VAP? [y/N] " reply
    case "$reply" in
        y | Y | yes | YES)
            ;;
        *)
            echo "Uninstall cancelled."
            exit 0
            ;;
    esac
fi

# Delete the command only when it is the wrapper managed by install.sh. Include
# broken symlinks so older installations can still be cleaned up.
if [[ -e "$VAP_WRAPPER" || -L "$VAP_WRAPPER" ]]; then
    if wrapper_is_managed; then
        rm -f "$VAP_WRAPPER"
        echo "Removed command: $VAP_WRAPPER"
    else
        echo "Keeping unmanaged command: $VAP_WRAPPER" >&2
    fi
else
    echo "Command already absent: $VAP_WRAPPER"
fi

if [[ "$PURGE" -eq 1 ]]; then
    rm -rf "$VAP_HOME"
    echo "Removed runtime: $VAP_HOME"
else
    rm -rf \
        "$VAP_HOME/bin" \
        "$VAP_HOME/cache" \
        "$VAP_HOME/perfetto-home" \
        "$VAP_HOME/tmp" \
        "$VAP_HOME/uv-python" \
        "$VAP_HOME/venv"
    rm -f "$INSTALL_MARKER"
    echo "Preserved: $VAP_HOME/config.json"
    echo "Preserved: $VAP_HOME/logs/"
fi

if [[ "$REMOVE_SOURCE" -eq 1 ]]; then
    if [[ ! -d "$SOURCE_DIR/.git" ]]; then
        echo "Managed source checkout not found; nothing to remove."
    elif [[ "$(git -C "$SOURCE_DIR" remote get-url origin 2>/dev/null || true)" != "$EXPECTED_REPO_URL" ]]; then
        echo "Refusing to remove source with an unexpected Git origin: $SOURCE_DIR" >&2
        exit 1
    elif [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
        echo "Refusing to remove source checkout with local changes: $SOURCE_DIR" >&2
        exit 1
    else
        rm -rf "$SOURCE_DIR"
        echo "Removed managed source: $SOURCE_DIR"
    fi
fi

echo "VAP uninstalled successfully."
