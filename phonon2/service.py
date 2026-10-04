"""Phonon-2 speech-to-text service. One process, one loaded model.

  * hold Shift+Space anywhere: records; release: transcribes and types at the cursor
  * unix socket (SOCK): other programs send 16 kHz mono float32 audio, get text back
    (see client.py) so they never load a model of their own
"""
import os, socket, struct, subprocess, tempfile, threading, time
import numpy as np, sounddevice as sd, soundfile as sf
from Xlib import X, XK, display as xdisplay
from Xlib.error import ConnectionClosedError
from fermion._speech import backends, fetch
from fermion.transcribe import _resolve

SR = 16000
MIN_SECONDS = 0.3
SOCK = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "phonon-asr", "asr.sock")


def load():
    repo, key, pin, local_dir = _resolve("phonon-2")
    model_dir = local_dir if local_dir is not None else fetch.ensure(repo, key, pin)
    kind = backends.resolve("ptt")
    return backends.load(kind, model_dir, profile=key, backend=pin["backend"], quiet=True)


print("loading model (~25s)...", flush=True)
speech = load()
busy = threading.Lock()  # the engine decodes one clip at a time


def transcribe(audio):
    with busy, tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, audio, SR)
        return speech.transcribe_detailed(f.name).triple()[0].strip()


# ---------------------------------------------------------------- socket API
def recv_exact(c, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = c.recv(n - len(buf))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return bytes(buf)


def serve_client(c):
    try:
        (n,) = struct.unpack("<I", recv_exact(c, 4))
        audio = np.frombuffer(recv_exact(c, n * 4), dtype="<f4")
        text = transcribe(audio) if n >= SR * MIN_SECONDS else ""
        out = text.encode()
        c.sendall(struct.pack("<I", len(out)) + out)
    except Exception as e:
        print("client error:", e, flush=True)
    finally:
        c.close()


def serve_socket():
    os.makedirs(os.path.dirname(SOCK), exist_ok=True)
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    s = socket.socket(socket.AF_UNIX)
    s.bind(SOCK)
    os.chmod(SOCK, 0o600)
    s.listen(4)
    while True:
        c, _ = s.accept()
        threading.Thread(target=serve_client, args=(c,), daemon=True).start()


# ---------------------------------------------------------- hold-to-talk hotkey
# Nothing in here may kill the service: every failure is logged, the recording
# state is reset, and the next Shift+Space works again.
class Recording:
    """One hold of the key. Owns its own frames, so a quick re-press cannot mix audio."""
    def __init__(self):
        self.frames = []
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                     callback=lambda indata, n, t, st: self.frames.append(indata.copy()))
        self.stream.start()

    def finish(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception as e:
            print("stream close error:", e, flush=True)
        return np.concatenate(self.frames)[:, 0] if self.frames else np.zeros(0, "float32")


current = None


def notify(msg):
    subprocess.run(["notify-send", "-a", "phonon", "-t", "3000", "Dictation", msg],
                   stderr=subprocess.DEVNULL, check=False)


def start():
    global current
    try:
        current = Recording()
    except Exception as e:  # device busy / unplugged / PortAudio stale: rescan devices and retry once
        print("mic open failed, rescanning devices:", e, flush=True)
        try:
            sd._terminate(); sd._initialize()
            current = Recording()
        except Exception as e2:
            current = None
            print("mic open failed again:", e2, flush=True)
            notify("microphone unavailable")


def stop_and_type(rec):
    try:
        audio = rec.finish()
        if len(audio) < SR * MIN_SECONDS:
            return  # a plain Shift+Space tap, not a dictation
        text = transcribe(audio)
        print(text, flush=True)
        if text:
            subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text + " "],
                           timeout=60)
    except Exception as e:
        print("dictation error:", repr(e), flush=True)
        notify("dictation failed, try again")


def grab_keys(d):
    keycode = d.keysym_to_keycode(XK.XK_space)
    errors = []
    # grab with and without CapsLock (Lock) / NumLock (Mod2) so the combo works either way
    for extra in (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask):
        d.screen().root.grab_key(keycode, X.ShiftMask | extra, True, X.GrabModeAsync, X.GrabModeAsync,
                                 onerror=lambda e, *a: errors.append(e))
    d.sync()
    return keycode, errors


def handle(d, keycode, ev):
    global current
    if ev.type not in (X.KeyPress, X.KeyRelease) or ev.detail != keycode:
        return
    if ev.type == X.KeyPress:
        if current is None:  # auto-repeat presses arrive while recording: ignore
            start()
    elif current is not None:
        # Holding the key makes X send fake release+press pairs (auto-repeat).
        # Ask the server whether the key is really still down; only a real
        # release ends the recording.
        if d.query_keymap()[keycode // 8] & (1 << (keycode % 8)):
            return
        rec, current = current, None
        threading.Thread(target=stop_and_type, args=(rec,), daemon=True).start()


def hotkey_loop():
    """Run forever. If the X connection drops (session reset, display restart) reconnect."""
    global current
    first = True
    while True:
        try:
            d = xdisplay.Display()
            keycode, errors = grab_keys(d)
            if errors:
                print("cannot grab Shift+Space: another app already owns it", flush=True)
                if first:
                    os._exit(1)
                time.sleep(5)
                continue
            first = False
            print("ready: hold Shift+Space to talk, release to type", flush=True)
            while True:
                try:
                    handle(d, keycode, d.next_event())
                except (ConnectionClosedError, OSError, EOFError):
                    raise
                except Exception as e:  # a bad event must not stop the loop
                    print("event error:", repr(e), flush=True)
                    if current is not None:  # state may be inconsistent: drop the recording
                        rec, current = current, None
                        rec.finish()
        except Exception as e:
            print("X connection lost, reconnecting:", repr(e), flush=True)
            if current is not None:
                rec, current = current, None
                rec.finish()
            time.sleep(2)


threading.Thread(target=serve_socket, daemon=True).start()
hotkey_loop()
