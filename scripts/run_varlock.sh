#!/usr/bin/env bash
set -euo pipefail

# If varlock is installed, use it to load/validate environment variables before running the command
if command -v varlock &>/dev/null; then
  exec varlock run -- "$@"
else
  exec "$@"
fi
