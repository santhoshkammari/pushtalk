"""Client for the Phonon-2 service (service.py): send audio, get text. No model is loaded here.

    from phonon2.client import transcribe
    text = transcribe(audio_float32_16k_mono)
"""
import os, socket, struct, subprocess, time
import numpy as np

RUN_SH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run.sh")
SOCK = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "phonon-asr", "asr.sock")


def _connect():
    s = socket.socket(socket.AF_UNIX)
    s.connect(SOCK)
    return s


def ensure_running(timeout=90):
    """Connect to the service, starting it (once) if it is not up."""
    try:
        return _connect().close()
    except OSError:
        pass
    os.makedirs(os.path.dirname(SOCK), exist_ok=True)
    subprocess.Popen(["nohup", RUN_SH, "dictate", "run"],
                     stdout=open(os.path.join(os.path.dirname(SOCK), "service.log"), "ab"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    end = time.time() + timeout
    while time.time() < end:
        try:
            return _connect().close()
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("phonon-asr service did not come up; see ~/.cache/phonon-asr/service.log")


def _recv(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("service closed the connection")
        buf += chunk
    return bytes(buf)


def transcribe(audio: np.ndarray) -> str:
    audio = np.ascontiguousarray(audio, dtype="<f4")
    with _connect() as s:
        s.sendall(struct.pack("<I", len(audio)) + audio.tobytes())
        (n,) = struct.unpack("<I", _recv(s, 4))
        return _recv(s, n).decode()
