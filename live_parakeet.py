#!/usr/bin/env python3
"""Live streaming transcription with Parakeet TDT (sherpa-onnx, ONNX int8, CPU).

Parakeet is a full-context OFFLINE model — it has no cache-aware streaming
mode. To get live text out of it we re-decode an overlapping rolling window
every --chunk-sec and commit words with LocalAgreement-2: a word is printed
only once two successive overlapping decodes agree on it, so the unstable
trailing edge never reaches the screen. The audio buffer is then trimmed at
the last committed word (using the token timestamps sherpa returns), which
bounds both latency and how much audio gets re-decoded each step.

Latency floor is ~chunk-sec. Nothing is ever un-printed.

Run:
    python live_parakeet.py                # defaults
    python live_parakeet.py --chunk-sec 0.4 --debug
"""
import argparse
import os
import queue
import sys
import time

import numpy as np
import sounddevice as sd
import sherpa_onnx

SAMPLE_RATE = 16000
MODEL_DIR = os.path.expanduser("~/.config/openchamber/speech-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")

p = argparse.ArgumentParser()
p.add_argument("--model-dir", default=MODEL_DIR)
p.add_argument("--chunk-sec", type=float, default=0.5,
               help="new audio per step / re-decode cadence. lower = snappier, more compute")
p.add_argument("--max-context-sec", type=float, default=6.0,
               help="hard cap on the rolling window. decode cost grows with window "
                    "size, so this is what keeps you ahead of realtime on CPU")
p.add_argument("--trim-margin", type=float, default=0.2,
               help="audio kept before the last committed word")
p.add_argument("--silence-rms", type=float, default=0.005,
               help="whole-window RMS below this = silence; window is reset")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--device", type=int, default=None, help="input device index")
p.add_argument("--debug", action="store_true", help="show per-step decode timing")
p.add_argument("--no-fillers", action="store_true",
               help="hide um/uh/hmm etc. parakeet transcribes them because you "
                    "said them; this only filters the printed output")
p.add_argument("--agent", action="store_true",
               help="dispatch to OpenChamber when a trigger phrase is heard: "
                    "'agent send' / 'send agent' / 'do this' / 'do that'")
p.add_argument("--dir", default=os.path.expanduser("~"),
               help="working directory the agent operates in")
p.add_argument("--model", default=os.environ.get("LIVE_ASR_MODEL", "deepseek-v4.1-flash"),
               help="model id")
p.add_argument("--provider", default=os.environ.get("LIVE_ASR_PROVIDER", "opencode-go"),
               help="provider id")
p.add_argument("--agent-name", default=None,
               help="OpenChamber agent (default: build)")
p.add_argument("--end-pause", type=float, default=1.5,
               help="silence that marks the end of a spoken command")
args = p.parse_args()

CHUNK_SAMPLES = int(args.chunk_sec * SAMPLE_RATE)
MAX_SAMPLES = int(args.max_context_sec * SAMPLE_RATE)

# ---------------------------------------------------------------- model load
print("loading parakeet int8 (this takes a few seconds)...", file=sys.stderr, flush=True)
_t0 = time.time()
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{args.model_dir}/encoder.int8.onnx",
    decoder=f"{args.model_dir}/decoder.int8.onnx",
    joiner=f"{args.model_dir}/joiner.int8.onnx",
    tokens=f"{args.model_dir}/tokens.txt",
    num_threads=args.threads,
    provider="cpu",
    decoding_method="greedy_search",
    model_type="nemo_transducer",
)
print(f"model loaded in {time.time() - _t0:.1f}s", file=sys.stderr, flush=True)

# warm up so the first real decode isn't paying allocation costs
_warm = recognizer.create_stream()
_warm.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE, dtype=np.float32))
recognizer.decode_stream(_warm)
print("warmed up", file=sys.stderr, flush=True)


def decode(buf: np.ndarray):
    """Decode a float32 window -> [(word_lower, word_orig, start_s, end_s), ...].

    sherpa gives per-token timestamps; parakeet's tokens are sentencepiece
    pieces where a leading space marks a new word, so we merge pieces into
    words and carry the start time of the first piece / end of the last.
    """
    s = recognizer.create_stream()
    s.accept_waveform(SAMPLE_RATE, buf)
    recognizer.decode_stream(s)
    res = s.result
    toks, stamps = res.tokens, res.timestamps
    if not toks:
        return []
    words, cur, cur_start, cur_end = [], "", None, None
    for tok, ts in zip(toks, stamps):
        if tok.startswith(" ") and cur:
            words.append((cur.strip().lower(), cur.strip(), cur_start, cur_end))
            cur, cur_start = tok, ts
        else:
            if not cur:
                cur_start = ts
            cur += tok
        cur_end = ts
    if cur.strip():
        words.append((cur.strip().lower(), cur.strip(), cur_start, cur_end))
    return words


def common_prefix(a, b):
    """How many leading words of a and b are the same (LocalAgreement-2)."""
    n = 0
    for x, y in zip(a, b):
        if x[0] != y[0]:
            break
        n += 1
    return n


# Parakeet transcribes disfluencies verbatim — it is not misfiring, it is
# reporting what you actually said. Strip them only when asked.
FILLERS = {
    "um", "uh", "umm", "uhh", "uhm", "hmm", "hm", "mmm", "mm",
    "er", "erm", "ah", "aah", "eh", "huh", "mhm", "uh-huh",
}


def is_filler(word_lower: str) -> bool:
    return word_lower.strip(".,!?;:").lower() in FILLERS


# Last few words printed, lowercased — used to catch repeats that slip past
# the timestamp filter when word boundaries shift between decodes.
_tail: list[str] = []
TAIL_KEEP = 8


def drop_repeat_of_tail(new):
    """Drop a leading run of words that merely replays what we just printed.

    The trim margin deliberately keeps a little already-committed audio, so
    the next decode often re-emits the last word or two. Timestamps shift
    slightly between decodes, so comparing text is the reliable check.
    """
    if not new or not _tail:
        return new
    # longest overlap between the tail and the head of `new` (bounded by tail)
    max_ov = min(len(_tail), len(new))
    for ov in range(max_ov, 0, -1):
        if _tail[-ov:] == [w[0] for w in new[:ov]]:
            return new[ov:]
    return new


# ------------------------------------------------------------------ mic feed
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def on_audio(indata, frames, time_info, status):
    if status:
        print(f"[audio status] {status}", file=sys.stderr, flush=True)
    audio_q.put(indata[:, 0].copy())


audio_buf = np.zeros(0, dtype=np.float32)
buf_start_t = 0.0        # absolute time of audio_buf[0]
total_consumed = 0       # samples pulled from the mic so far
committed_end_t = 0.0    # abs end time of the last printed word
prev_unc = []            # previous step's uncommitted words
at_bol = True
step = 0


def emit(words):
    """Append finalized words to the current line. Append-only, never rewritten."""
    global at_bol, _tail
    if not words:
        return
    # Track what was actually said (fillers included) so repeat detection
    # still lines up when they are being hidden from the output.
    _tail = (_tail + [w[0] for w in words])[-TAIL_KEEP:]

    # Feed the trigger watcher. Fillers are skipped so an "um" inside a
    # command never reaches the agent.
    if cmd_buf is not None:
        for w in words:
            if not is_filler(w[0]):
                cmd_buf.add(w[1])

    if args.no_fillers:
        words = [w for w in words if not is_filler(w[0])]
        if not words:
            return
    txt = " ".join(w[1] for w in words)
    sys.stdout.write(txt if at_bol else " " + txt)
    sys.stdout.flush()
    at_bol = False


def newline():
    global at_bol
    if not at_bol:
        sys.stdout.write("\n")
        sys.stdout.flush()
        at_bol = True


# ------------------------------------------------------------ agent dispatch
cmd_buf = None
dispatcher = None
if args.agent:
    from dispatch import TRIGGERS, CommandBuffer, Dispatcher

    cmd_buf = CommandBuffer(end_pause=args.end_pause)
    dispatcher = Dispatcher(directory=args.dir, provider_id=args.provider,
                            model_id=args.model, agent=args.agent_name)
    print(f"  agent dispatch ON -> {args.dir}",
          file=sys.stderr, flush=True)
    print(f"  triggers: {', '.join(repr(t) for t in TRIGGERS)}",
          file=sys.stderr, flush=True)


def maybe_dispatch():
    """Send the buffered command once the trailing pause has elapsed."""
    if cmd_buf is None:
        return
    text = cmd_buf.check_timeout()
    if text:
        newline()
        dispatcher.send(text)


print(f"\n  chunk-sec={args.chunk_sec}  max-context={args.max_context_sec}s  threads={args.threads}",
      file=sys.stderr, flush=True)
print("  listening — speak now. ctrl-c to stop.\n", file=sys.stderr, flush=True)

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK_SAMPLES, device=args.device,
                        callback=on_audio):
        while True:
            chunk = audio_q.get()
            total_consumed += len(chunk)
            audio_buf = np.concatenate([audio_buf, chunk])
            window_end_t = total_consumed / SAMPLE_RATE

            # dead air must not grow the window
            rms = float(np.sqrt(np.mean(audio_buf ** 2))) if audio_buf.size else 0.0
            if rms < args.silence_rms:
                if audio_buf.size:
                    newline()
                audio_buf = np.zeros(0, dtype=np.float32)
                buf_start_t = window_end_t
                committed_end_t = max(committed_end_t, window_end_t)
                prev_unc = []
                _tail = []  # new utterance: nothing to repeat-check against
                # A command ends *in* silence, so this is where it usually fires.
                maybe_dispatch()
                continue

            step += 1
            t0 = time.time()
            words = decode(audio_buf)
            dt = time.time() - t0

            # window-relative -> absolute time
            words = [(wl, wo, s + buf_start_t, e + buf_start_t)
                     for (wl, wo, s, e) in words]
            # A word is uncommitted only if it *starts* after what we already
            # printed. Filtering on end-time alone lets a word that straddles
            # the trim boundary through, and it prints a second time.
            unc = [w for w in words if w[2] > committed_end_t - 1e-3]

            if args.debug:
                win = len(audio_buf) / SAMPLE_RATE
                print(f"\n[{step:4d}] win={win:5.2f}s decode={dt:5.3f}s "
                      f"rtf={dt / win:5.3f} budget={dt / args.chunk_sec * 100:5.1f}% "
                      f"q={audio_q.qsize()} unc={len(unc)}",
                      file=sys.stderr, flush=True)

            # decode cost grows with window size; if a step costs more than the
            # audio it covers we are falling behind and the mic queue backs up.
            if dt > args.chunk_sec and audio_q.qsize() > 2:
                print(f"\n[LAGGING] decode {dt:.2f}s > chunk {args.chunk_sec}s, "
                      f"{audio_q.qsize()} chunks queued — raise --chunk-sec or "
                      f"lower --max-context-sec", file=sys.stderr, flush=True)

            # LocalAgreement-2: commit the prefix the last two decodes agree on
            k = common_prefix(unc, prev_unc)
            if k:
                new = unc[:k]
                # Second guard: timestamps drift between decodes, so a repeat
                # can survive the time filter. Drop any leading run that just
                # replays the tail we already printed.
                new = drop_repeat_of_tail(new)
                if new:
                    emit(new)
                committed_end_t = unc[k - 1][3]
            prev_unc = unc

            # trim at the last committed word, and hard-cap the window
            target_start = max(buf_start_t, committed_end_t - args.trim_margin)
            drop = int((target_start - buf_start_t) * SAMPLE_RATE)
            if len(audio_buf) - drop > MAX_SAMPLES:
                drop = len(audio_buf) - MAX_SAMPLES
            if drop > 0:
                audio_buf = audio_buf[drop:]
                buf_start_t += drop / SAMPLE_RATE

            # also check while speech continues, for a pause too short to
            # trip the silence branch above
            maybe_dispatch()

except KeyboardInterrupt:
    newline()
    print("\nstopped.", file=sys.stderr, flush=True)
