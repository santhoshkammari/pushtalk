#!/usr/bin/env python3
"""Standalone Kokoro-82M text-to-speech.

Runs hexgrad/Kokoro-82M through the official `kokoro` package (PyTorch). This
laptop has no GPU, so it synthesizes on CPU; 82M params is small enough for
that to be usable. NOT wired into pushtalk yet - this is a standalone test
rig for checking quality and latency before deciding how (or whether) to
integrate it as the voice output of the overlay.

The first run downloads the model (~330 MB) and the chosen voice from the HF
hub into the normal HF cache; later runs reuse them.

Run:
    python kokoro_tts.py "hello, this is a test"
    python kokoro_tts.py --voice af_bella --speed 1.1 --play "hi there"
    echo "read me out loud" | python kokoro_tts.py --out /tmp/a.wav
    python kokoro_tts.py --list-voices
"""
import argparse
import sys
import time

import numpy as np

SAMPLE_RATE = 24000

# Voice names by lang_code, straight from the model card's VOICES.md. The
# pipeline downloads each voice lazily from the HF repo on first use.
VOICES = {
    "a": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "b": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "z": [
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "i": ["if_sara", "im_nicola"],
    "p": ["pf_dora", "pm_alex", "pm_santa"],
}

p = argparse.ArgumentParser(description="Kokoro-82M text-to-speech (standalone)")
p.add_argument("text", nargs="?", default=None,
               help="text to speak; omit to read --file or stdin")
p.add_argument("--file", default=None, help="read text from this file")
p.add_argument("--voice", default="af_heart",
               help="voice name, e.g. af_heart (A), am_michael, bf_emma (B-)")
p.add_argument("--speed", type=float, default=1.0, help="0.5 - 2.0")
p.add_argument("--lang", default="a",
               help="lang_code: a=US, b=UK, j=ja, z=zh, e=es, f=fr, h=hi, i=it, p=pt-br")
p.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
p.add_argument("--device", default=None, choices=["cpu", "cuda"],
               help="default: cuda if available else cpu")
p.add_argument("--out", default="/tmp/kokoro_tts.wav", help="output wav path")
p.add_argument("--gap", type=float, default=0.15,
               help="silence inserted between chunk joins, seconds")
p.add_argument("--play", action="store_true", help="play the audio when done")
p.add_argument("--debug", action="store_true",
               help="print graphemes/phonemes for every chunk")
p.add_argument("--list-voices", action="store_true",
               help="list known voices for --lang and exit")
args = p.parse_args()

if args.list_voices:
    print(f"voices for lang_code={args.lang!r}:", file=sys.stderr)
    for v in VOICES.get(args.lang, []):
        print(f"  {v}")
    sys.exit(0)

# -------------------------------------------------------------------- text in
if args.text is not None:
    text = args.text
elif args.file:
    with open(args.file, encoding="utf-8") as f:
        text = f.read()
elif not sys.stdin.isatty():
    text = sys.stdin.read()
else:
    sys.exit("no text: pass it as an argument, --file, or on stdin")

text = text.strip()
if not text:
    sys.exit("empty text")

# ---------------------------------------------------------------- model load
print("loading kokoro (downloads model + voice on first use)...",
      file=sys.stderr, flush=True)
_t0 = time.time()
from kokoro import KPipeline  # noqa: E402  (deliberately after arg parsing)

pipeline = KPipeline(lang_code=args.lang, repo_id=args.repo_id,
                     device=args.device)
print(f"pipeline ready in {time.time() - _t0:.1f}s  "
      f"(device={next(pipeline.model.parameters()).device})",
      file=sys.stderr, flush=True)

# ---------------------------------------------------------------- synthesize
print(f"  voice={args.voice}  speed={args.speed}  lang={args.lang}",
      file=sys.stderr, flush=True)

t0 = time.time()
chunks: list[np.ndarray] = []
for i, (gs, ps, audio) in enumerate(
        pipeline(text, voice=args.voice, speed=args.speed)):
    if audio is None:
        continue
    if args.debug:
        print(f"\n[{i}] text: {gs}", file=sys.stderr, flush=True)
        print(f"[{i}] phonemes: {ps}", file=sys.stderr, flush=True)
    if chunks and args.gap > 0:
        chunks.append(np.zeros(int(args.gap * SAMPLE_RATE), dtype=np.float32))
    chunks.append(audio.detach().cpu().numpy())
dt = time.time() - t0

if not chunks:
    sys.exit("no audio produced (empty text after G2P?)")

audio = np.concatenate(chunks).astype(np.float32)
dur = len(audio) / SAMPLE_RATE
print(f"  synthesized {len(chunks)} chunk(s): {dur:.2f}s audio in {dt:.2f}s "
      f"(RTF {dt / dur:.2f})", file=sys.stderr, flush=True)

# ------------------------------------------------------------------ save/play
import soundfile as sf  # noqa: E402

sf.write(args.out, audio, SAMPLE_RATE)
print(f"  wrote {args.out}", file=sys.stderr, flush=True)

if args.play:
    import sounddevice as sd  # noqa: E402

    print("  playing...", file=sys.stderr, flush=True)
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
