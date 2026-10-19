#!/bin/sh
# Kartrix installer for macOS and Linux. Usage:
#   curl -LsSf https://raw.githubusercontent.com/RoshanKarthik/Kartrix/main/scripts/install.sh | sh
# Installs uv (the free Python tool manager) if it's missing, then Kartrix with its own Python. No root, no Docker,
# no database setup. Set KARTRIX_SOURCE to install from elsewhere (a folder, a .whl file, a git URL).
# Dependencies get the tested versions from constraints.txt (uv.lock exported): the published one, a source folder's
# own, or KARTRIX_CONSTRAINTS (a path or URL; "none" for the newest versions pyproject.toml allows).
set -eu

DEFAULT_SOURCE="git+https://github.com/RoshanKarthik/Kartrix"
DEFAULT_CONSTRAINTS="https://raw.githubusercontent.com/RoshanKarthik/Kartrix/main/constraints.txt"

SOURCE="${KARTRIX_SOURCE:-$DEFAULT_SOURCE}"

CONSTRAINTS="${KARTRIX_CONSTRAINTS:-}"
if [ -z "$CONSTRAINTS" ]; then
  if [ "$SOURCE" = "$DEFAULT_SOURCE" ]; then
    CONSTRAINTS="$DEFAULT_CONSTRAINTS"
  elif [ -f "$SOURCE/constraints.txt" ]; then
    CONSTRAINTS="$SOURCE/constraints.txt"
  fi
fi
if [ "$CONSTRAINTS" = "none" ]; then
  CONSTRAINTS=""
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (free Python tool manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi

echo "Installing Kartrix from $SOURCE ..."
if [ -n "$CONSTRAINTS" ]; then
  uv tool install --python 3.12 --force --constraints "$CONSTRAINTS" "$SOURCE"
else
  echo "No constraints file: dependencies get the newest versions allowed, not the tested ones."
  uv tool install --python 3.12 --force "$SOURCE"
fi
uv tool update-shell >/dev/null 2>&1 || true

echo
echo "Kartrix is installed."
echo "Open a NEW terminal, go to one of your repositories and run:  kartrix"
echo "The first run asks for a free NVIDIA key (https://build.nvidia.com). Check everything with:  kartrix doctor"
if [ "$(uname)" = "Linux" ] && ! command -v bwrap >/dev/null 2>&1; then
  echo "Tip: install bubblewrap (e.g. sudo apt install bubblewrap) so Kartrix can sandbox the commands it runs."
fi
