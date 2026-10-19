#!/bin/sh
# Kartrix installer for macOS and Linux. Usage:
#   curl -LsSf https://raw.githubusercontent.com/RoshanKarthik/Kartrix/main/scripts/install.sh | sh
# Installs uv (the free Python tool manager) if it's missing, then Kartrix with its own Python. No root, no Docker,
# no database setup. Set KARTRIX_SOURCE to install from elsewhere (a folder, a .whl file, a git URL).
set -eu

DEFAULT_SOURCE="git+https://github.com/RoshanKarthik/Kartrix"

SOURCE="${KARTRIX_SOURCE:-$DEFAULT_SOURCE}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (free Python tool manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi

echo "Installing Kartrix from $SOURCE ..."
uv tool install --python 3.12 --force "$SOURCE"
uv tool update-shell >/dev/null 2>&1 || true

echo
echo "Kartrix is installed."
echo "Open a NEW terminal, go to one of your repositories and run:  kartrix"
echo "The first run asks for a free NVIDIA key (https://build.nvidia.com). Check everything with:  kartrix doctor"
if [ "$(uname)" = "Linux" ] && ! command -v bwrap >/dev/null 2>&1; then
  echo "Tip: install bubblewrap (e.g. sudo apt install bubblewrap) so Kartrix can sandbox the commands it runs."
fi
