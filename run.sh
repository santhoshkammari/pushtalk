#!/usr/bin/env bash
# pushtalk launcher - docker-style verbs, so one file drives everything.
#
#   ./run.sh start   [opts]   launch overlay + engine in the background
#   ./run.sh stop             stop both
#   ./run.sh restart [opts]   stop then start (re-reads the LIVE_ASR_* env)
#   ./run.sh status           whether it is running, and the pids
#   ./run.sh toggle  [opts]   start if stopped, stop if running (keybinding)
#   ./run.sh run     [opts]   foreground; ctrl-c stops (old run.sh behaviour)
#   ./run.sh dictate [start|stop|status|run]
#                             Phonon-2 speech-to-text service (phonon2/): hold
#                             Shift+Space anywhere to type what you say. The
#                             F9 engine below uses this same service, and starts it
#                             on demand, so the model is only ever loaded once.
#
# [opts] are passed straight to the engine:
#   --tts          use the speaking engine (ptt_tts.py), else ptt.py
#   --no-agent     transcribe only, print text
#   ...            anything else goes to the engine's own argparse
#
# Env (all read at launch, so `restart` picks up any change):
#   LIVE_ASR_VENV      venv whose bin/python runs overlay + engine (default ~/main)
#   LIVE_ASR_PROVIDER  OpenChamber provider id (default vllm)
#   LIVE_ASR_MODEL     OpenChamber model id    (default laguna)
#   LIVE_ASR_DISPATCH  "bonsai" -> plain chat via dispatch_bonsai.py (Ollama,
#                       no tools/sessions), anything else -> OpenChamber
#   LIVE_ASR_LOG       log file (default /tmp/pushtalk.log)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${LIVE_ASR_VENV:-$HOME/main}"
PY="$VENV/bin/python"
LOG="${LIVE_ASR_LOG:-/tmp/pushtalk.log}"
STATE="${XDG_CACHE_HOME:-$HOME/.cache}/pushtalk"
ENGINE_PID="$STATE/engine.pid"
OVERLAY_PID="$STATE/overlay.pid"
DICTATE_PID="$STATE/dictate.pid"
DICTATE_LOG="$HOME/.cache/phonon-asr/service.log"

# First arg is the verb; everything after it is engine options.
VERB="${1:-start}"
shift || true

# Split --tts (launcher concern) out of the engine options.
ENGINE=ptt.py
ARGS=()
for a in "$@"; do
    if [ "$a" = "--tts" ]; then
        ENGINE=ptt_tts.py
    else
        ARGS+=("$a")
    fi
done

notify() { command -v notify-send >/dev/null 2>&1 && notify-send -a "pushtalk" "$1" "${2:-}" 2>/dev/null || true; }

running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

kill_pidfile() {
    local f="$1"
    [ -f "$f" ] || return 0
    local p
    p="$(cat "$f")"
    kill -0 "$p" 2>/dev/null && kill "$p" 2>/dev/null || true
    rm -f "$f"
}

do_start() {
    mkdir -p "$STATE"
    if running "$ENGINE_PID"; then
        echo "already running (engine pid $(cat "$ENGINE_PID"))"
        return 0
    fi
    # A stale overlay from a previous run would fight for the socket.
    kill_pidfile "$OVERLAY_PID"

    : > "$LOG"
    nohup "$PY" "$HERE/overlay_qt.py" >> "$LOG" 2>&1 &
    echo $! > "$OVERLAY_PID"
    disown

    sleep 1  # let the bus socket bind before the engine sends anything

    nohup "$PY" "$HERE/$ENGINE" ${ARGS[@]+"${ARGS[@]}"} >> "$LOG" 2>&1 &
    echo $! > "$ENGINE_PID"
    disown

    echo "started ($ENGINE): overlay pid $(cat "$OVERLAY_PID"), engine pid $(cat "$ENGINE_PID")"
    echo "log: $LOG"
}

do_stop() {
    kill_pidfile "$ENGINE_PID"
    kill_pidfile "$OVERLAY_PID"
    sleep 0.4
    # Fallback sweep in case a pidfile went missing.
    pkill -f "$HERE/overlay_qt.py" 2>/dev/null || true
    pkill -f "$HERE/ptt" 2>/dev/null || true
    echo "stopped"
}

do_status() {
    if running "$ENGINE_PID"; then
        echo "running: overlay pid $(cat "$OVERLAY_PID" 2>/dev/null || echo '?'), engine pid $(cat "$ENGINE_PID")"
    else
        echo "stopped"
        return 1
    fi
}

do_run() {
    "$PY" "$HERE/overlay_qt.py" &
    local ov=$!
    trap "kill $ov 2>/dev/null || true" EXIT INT TERM
    sleep 1
    "$PY" "$HERE/$ENGINE" ${ARGS[@]+"${ARGS[@]}"}
}

do_dictate() {
    case "${1:-start}" in
        run)  # foreground; what start and the autostart entry launch
            mkdir -p "$STATE"
            echo $$ > "$DICTATE_PID"
            cd "$HERE" && exec "$PY" -m phonon2.service ;;
        start)
            if running "$DICTATE_PID"; then echo "dictate already running (pid $(cat "$DICTATE_PID"))"; return 0; fi
            mkdir -p "$(dirname "$DICTATE_LOG")"
            nohup "$HERE/run.sh" dictate run >> "$DICTATE_LOG" 2>&1 &
            echo "dictate started (model loads in ~25s), log: $DICTATE_LOG" ;;
        stop)   kill_pidfile "$DICTATE_PID"; echo "dictate stopped" ;;
        status) if running "$DICTATE_PID"; then echo "dictate running (pid $(cat "$DICTATE_PID"))"; else echo "dictate stopped"; return 1; fi ;;
        *) echo "usage: $0 dictate {start|stop|status|run}" >&2; exit 2 ;;
    esac
}

case "$VERB" in
    dictate) do_dictate ${ARGS[@]+"${ARGS[@]}"} ;;
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    toggle)
        if running "$ENGINE_PID"; then
            do_stop
            notify "voice service stopped"
        else
            do_start
            notify "voice service started" "hold F9 to speak"
        fi
        ;;
    run)     do_run ;;
    *)
        echo "usage: $0 {start|stop|restart|status|toggle|run|dictate} [--tts] [engine opts]" >&2
        exit 2
        ;;
esac
