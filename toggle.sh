#!/usr/bin/env bash
# Global-shortcut entry (Ctrl+Alt+V): start if stopped, stop if running.
# All the logic lives in run.sh - this just forwards the verb.
set -uo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/run.sh" toggle "$@"
