#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_URL="https://github.com/Shan2L/VAP.git"
VAP_REF="${VAP_REF:-main}"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
SOURCE_DIR="${VAP_SOURCE_DIR:-$DATA_HOME/vap/source}"

if ! command -v git >/dev/null 2>&1; then
    echo "Git is required for the bootstrap installer." >&2
    exit 1
fi

case "$VAP_REF" in
    "" | -* | /* | *..* | *[!A-Za-z0-9._/-]*)
        echo "Invalid VAP_REF: $VAP_REF" >&2
        exit 1
        ;;
esac

if [[ -e "$SOURCE_DIR" && ! -d "$SOURCE_DIR/.git" ]]; then
    echo "Refusing to replace a non-Git path at $SOURCE_DIR" >&2
    exit 1
fi

if [[ -d "$SOURCE_DIR/.git" ]]; then
    current_remote="$(git -C "$SOURCE_DIR" remote get-url origin)"
    if [[ "$current_remote" != "$REPO_URL" ]]; then
        echo "Refusing to update $SOURCE_DIR because its origin is $current_remote" >&2
        exit 1
    fi
    if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
        echo "Refusing to update a VAP source checkout with local changes: $SOURCE_DIR" >&2
        exit 1
    fi
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$VAP_REF"
    git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
else
    source_parent="$(dirname "$SOURCE_DIR")"
    mkdir -p "$source_parent"
    stage_dir="$(mktemp -d "$source_parent/.vap-bootstrap.XXXXXX")"
    trap 'rm -rf "$stage_dir"' EXIT
    git clone --depth 1 --branch "$VAP_REF" "$REPO_URL" "$stage_dir/source"
    mv "$stage_dir/source" "$SOURCE_DIR"
    rmdir "$stage_dir"
    trap - EXIT
fi

exec bash "$SOURCE_DIR/install.sh"
