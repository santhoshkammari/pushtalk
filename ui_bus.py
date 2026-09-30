"""Unix-socket IPC between the voice engine and the GTK overlay.

The engine runs in ~/main (sherpa/parakeet) and the overlay runs under the
system python (that is the only interpreter here with gi/GTK), so they can
never share a process. One newline-delimited JSON message per event.

Sending is deliberately best-effort: if the overlay is not running, the
voice engine must keep working regardless.
"""
from __future__ import annotations

import json
import os
import socket

SOCKET_PATH = os.path.expanduser("~/.cache/pushtalk-ui.sock")


def send(state: str, text: str = "") -> bool:
    """Fire an event at the overlay. False if it isn't listening."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.sendto(json.dumps({"state": state, "text": text}).encode(), SOCKET_PATH)
        s.close()
        return True
    except (OSError, socket.timeout):
        return False


def serve(callback):
    """Listen forever, calling callback(state, text) per message."""
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(SOCKET_PATH)
    while True:
        try:
            data, _ = srv.recvfrom(65536)
            msg = json.loads(data.decode())
            callback(msg.get("state", ""), msg.get("text", ""))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
