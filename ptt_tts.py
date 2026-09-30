#!/usr/bin/env python3
"""Push-to-talk, speech in and speech out: F9 -> agent -> spoken answer.

Same engine as ptt.py (Phonon-2 decode, OpenChamber dispatch, Qt overlay) with
the agent's reply spoken aloud as it streams. ptt.py is left untouched; run
whichever you want.

The agent's answer is already streamed token by token to drive the overlay, so
the same delta events feed the speaker. Sentences are cut out of the stream and
synthesized as soon as they close, which means speech starts after the agent's
first sentence rather than after its whole turn.

Barge-in: pressing F9 aborts the agent's turn (as before) and silences playback
immediately. There is no echo cancellation here, so the mic does hear Kokoro -
that is exactly why interruption is bound to the key and not to voice activity.

Run:
    python ptt_tts.py                     # talk to the agent, hear it answer
    python ptt_tts.py --voice am_michael  # different voice
    python ptt_tts.py --no-speak          # behaves like ptt.py
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard

SAMPLE_RATE = 16000
from phonon2 import client as asr_client  # Phonon-2 service; the model lives there, not in this process

p = argparse.ArgumentParser()
p.add_argument("--key", default="f9", help="hotkey name, e.g. f9, f8, ctrl_r")
p.add_argument("--tap-sec", type=float, default=0.4,
               help="press shorter than this = tap (toggle); longer = hold")
p.add_argument("--device", type=int, default=None, help="input device index")
p.add_argument("--no-agent", action="store_true", help="transcribe only, do not dispatch")
p.add_argument("--dir", default=None,
               help="working directory the agent operates in (default: current directory)")
p.add_argument("--model", default=os.environ.get("LIVE_ASR_MODEL", "deepseek-v4.1-flash"))
p.add_argument("--provider", default=os.environ.get("LIVE_ASR_PROVIDER", "opencode-go"))
p.add_argument("--agent-name", default=None, help="OpenChamber agent (default: build)")
p.add_argument("--max-sec", type=float, default=120.0, help="safety cap on one recording")
p.add_argument("--no-ui", action="store_true", help="do not drive the overlay")
p.add_argument("--no-speak", action="store_true", help="do not speak the reply")
p.add_argument("--voice", default="af_heart", help="kokoro voice, e.g. am_michael")
p.add_argument("--tts-speed", type=float, default=1.0, help="0.5 - 2.0")
p.add_argument("--out-device", type=int, default=None, help="output device index")
args = p.parse_args()

import ui_bus

# ---------------------------------------------------------------------- tts
speaker = None
cutter = None
if not args.no_speak:
    from tts_worker import SentenceCutter, Speaker

    speaker = Speaker(voice=args.voice, speed=args.tts_speed,
                      device=args.out_device)
    cutter = SentenceCutter()

# ---------------------------------------------------------------- model load
print("connecting to phonon-2 service...", file=sys.stderr, flush=True)
asr_client.ensure_running()
print("phonon-2 ready", file=sys.stderr, flush=True)

if speaker is not None:
    speaker.load()

# ------------------------------------------------------------------ dispatch
def on_event(state: str, text: str = "") -> None:
    """Drive the overlay, and speak the answer as it streams in."""
    ui(state, text)
    if speaker is None:
        return
    if state == "delta":
        for sentence in cutter.feed(text):
            speaker.say(sentence)
    elif state == "reply":
        # The last sentence usually has no trailing whitespace to close it.
        tail = cutter.flush()
        if tail:
            speaker.say(tail)
        cutter.reset()
    elif state in ("interrupted", "error"):
        speaker.stop()
        cutter.reset()


dispatcher = None
if not args.no_agent:
    if os.environ.get("LIVE_ASR_DISPATCH") == "bonsai":
        from dispatch_bonsai import Dispatcher
    else:
        from dispatch import Dispatcher

    work_dir = args.dir if args.dir else os.getcwd()
    dispatcher = Dispatcher(directory=work_dir, provider_id=args.provider,
                            model_id=args.model, agent=args.agent_name,
                            on_event=on_event)

# ------------------------------------------------------------------ recording
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
recording = threading.Event()
_rec_start = 0.0


_last_level = 0.0


def on_audio(indata, frames, time_info, status):
    global _last_level
    if status:
        print(f"[audio {status}]", file=sys.stderr, flush=True)
    if recording.is_set():
        chunk = indata[:, 0].copy()
        audio_q.put(chunk)
        now = time.time()
        if now - _last_level > 0.05:
            _last_level = now
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            ui("level", f"{min(1.0, rms * 8.0):.3f}")


def ui(state: str, text: str = "") -> None:
    """Best-effort overlay update; the engine runs fine without the UI."""
    if not args.no_ui:
        ui_bus.send(state, text)


def drain() -> np.ndarray:
    """Pull everything buffered so far into one array."""
    out = []
    while True:
        try:
            out.append(audio_q.get_nowait())
        except queue.Empty:
            break
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def start_recording():
    global _rec_start
    # Talking over the agent: stop its voice first so you are not competing
    # with it, then kill the turn server-side.
    if speaker is not None:
        speaker.stop()
        cutter.reset()
    if dispatcher is not None:
        dispatcher.interrupt()
    while not audio_q.empty():          # drop anything stale
        audio_q.get_nowait()
    _rec_start = time.time()
    recording.set()
    ui("listening")
    print("\n\033[91m● REC\033[0m  speak...", file=sys.stderr, flush=True)


def stop_and_process():
    """Stop recording, transcribe the whole utterance, dispatch it."""
    recording.clear()
    time.sleep(0.15)                    # let the last callback land
    audio = drain()
    if len(audio) > args.max_sec * SAMPLE_RATE:
        audio = audio[-int(args.max_sec * SAMPLE_RATE):]
        print(f"\033[93m(capped to last {args.max_sec:.0f}s)\033[0m",
              file=sys.stderr, flush=True)
    secs = len(audio) / SAMPLE_RATE
    if secs < 0.3:
        print("\033[93m…too short, ignored\033[0m", file=sys.stderr, flush=True)
        ui("idle")
        return
    print(f"\033[94m■ {secs:.1f}s\033[0m  transcribing...", file=sys.stderr, flush=True)
    ui("thinking")

    t0 = time.time()
    text = asr_client.transcribe(audio).strip()
    dt = time.time() - t0

    if not text:
        print("\033[93m…nothing heard\033[0m", file=sys.stderr, flush=True)
        ui("idle")
        return

    print(f"\033[92m>\033[0m {text}", flush=True)
    print(f"  (decoded {secs:.1f}s in {dt:.2f}s)", file=sys.stderr, flush=True)

    if dispatcher is not None:
        ui("thinking", text)          # show what was heard while the agent works
        dispatcher.send(text)
    else:
        ui("reply", text)
        if speaker is not None:
            speaker.say(text)


# ------------------------------------------------------------------ hotkey
HOTKEY = getattr(keyboard.Key, args.key, None)
if HOTKEY is None:
    sys.exit(f"unknown key: {args.key}")

_press_t = 0.0
_toggle_on = False
_held = False


def on_escape():
    """Esc: same interrupt() as talking over the agent, then close the
    overlay - Esc must abort a running turn (and its voice), not just hide
    the window."""
    if speaker is not None:
        speaker.stop()
        cutter.reset()
    if dispatcher is not None:
        dispatcher.interrupt()
    ui("idle")


def on_press(key):
    """F9 down: begin recording (works for both hold and toggle)."""
    global _press_t, _held, _toggle_on
    if key == keyboard.Key.esc:
        on_escape()
        return
    if key != HOTKEY or _held:
        return                          # ignore key-repeat while held
    _held = True
    _press_t = time.time()
    if _toggle_on:
        return                          # toggle is running; release decides
    start_recording()


def on_release(key):
    """F9 up: a hold ends the recording, a tap flips toggle state."""
    global _held, _toggle_on
    if key != HOTKEY:
        return
    _held = False
    held_for = time.time() - _press_t

    if _toggle_on:                      # second tap -> stop and send
        _toggle_on = False
        stop_and_process()
        return

    if held_for >= args.tap_sec:        # hold -> stop and send
        stop_and_process()
    else:                               # tap -> stay recording until next tap
        _toggle_on = True
        print("  (toggle on - tap again to send)", file=sys.stderr, flush=True)


key_name = args.key.upper()
if dispatcher:
    dir_label = args.dir or os.getcwd()
else:
    dir_label = "transcribe only (no agent)"
print(f"""
  \033[1mpush-to-talk ready (speech in, speech out)\033[0m
    hold {key_name}          speak, release to send
    tap  {key_name}          start, tap again to send
    {"agent -> " + dir_label if dispatcher else dir_label}
    {"voice  -> " + args.voice if speaker else "speech output off"}
  ctrl-c to quit
""", file=sys.stderr, flush=True)

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=1600, device=args.device, callback=on_audio):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as kl:
            kl.join()
except KeyboardInterrupt:
    print("\nstopped.", file=sys.stderr, flush=True)
