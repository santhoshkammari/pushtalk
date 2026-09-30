"""Speak an LLM's token stream as it arrives, one sentence at a time.

The agent emits text token by token. Waiting for the whole turn before
synthesizing costs seconds of silence, so this cuts complete sentences out of
the growing stream and hands each one to Kokoro the moment it is closed.

Three threads, two queues:

    feed(delta)  ->  [sentence queue]  ->  synth  ->  [audio queue]  ->  play

Kokoro on this CPU runs at RTF ~0.55, so synthesis outruns playback: after the
first sentence (~1.4s) the audio queue never empties and speech is gapless.
That headroom is the whole reason this design works - measurements in
the Kokoro notes in the README.

Sentence splitting is not cosmetic. KPipeline defaults to split_pattern=r'\\n+'
and only chunks again at 510 phonemes, so a paragraph of prose is ONE chunk:
time-to-first-audio 5.65s instead of 1.39s. We always pass single sentences.
"""
from __future__ import annotations

import queue
import re
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 24000

# Silence prepended to the first chunk of an utterance. Kokoro runs only just
# faster than realtime (~3.1s of audio in ~2.2s), so playback would otherwise
# catch up with synthesis on the first handoff and leave an audible hole. This
# buys that margin for the cost of a barely perceptible delay, and the lead
# only grows from there.
LEAD_SILENCE = 1.2

# A sentence is closed by .!? followed by whitespace. The lookbehind keeps the
# punctuation with the sentence it ends. Decimals and abbreviations ("v1.1",
# "e.g.") would split wrongly on a bare [.], so require the next character to
# be whitespace and the one before not to be a digit.
_SENTENCE_END = re.compile(r'(?<=[.!?])(?<!\d\.)\s+')

# Speech-hostile spans. Reading a fenced diff or an absolute path aloud is
# unbearable, so they are dropped before synthesis rather than mangled.
_CODE_FENCE = re.compile(r'```.*?```', re.S)
_INLINE_CODE = re.compile(r'`[^`]*`')
_URL = re.compile(r'https?://\S+')
_ABS_PATH = re.compile(r'(?<![\w/])[~/][\w./-]{6,}')
_MD_EMPHASIS = re.compile(r'[*_#>]+')


def speakable(text: str) -> str:
    """Strip what should never be read aloud, keep the prose."""
    text = _CODE_FENCE.sub(' code block. ', text)
    text = _INLINE_CODE.sub(' ', text)
    text = _URL.sub(' link. ', text)
    text = _ABS_PATH.sub(' path. ', text)
    text = _MD_EMPHASIS.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip()


class SentenceCutter:
    """Turns a growing text stream into complete sentences.

    The dispatcher sends the whole answer so far on every delta, not just the
    new characters, so feed() takes the cumulative text and tracks how much of
    it has already been emitted.
    """

    def __init__(self, min_chars: int = 12):
        # Very short fragments ("OK.") synthesize into clipped-sounding blips
        # and cost a whole inference, so they wait to join the next sentence.
        self.min_chars = min_chars
        self._emitted = 0
        self._pending = ''

    def feed(self, full_text: str) -> list[str]:
        """Return sentences newly completed by this update."""
        self._pending += full_text[self._emitted:]
        self._emitted = len(full_text)

        parts = _SENTENCE_END.split(self._pending)
        self._pending = parts.pop()          # trailing fragment stays open

        out, buf = [], ''
        for part in parts:
            buf = f'{buf} {part}'.strip() if buf else part
            if len(buf) >= self.min_chars:
                out.append(buf)
                buf = ''
        if buf:
            self._pending = f'{buf} {self._pending}'.strip()
        return out

    def flush(self) -> str:
        """The final fragment, which has no closing punctuation."""
        tail, self._pending = self._pending.strip(), ''
        return tail

    def reset(self) -> None:
        self._emitted = 0
        self._pending = ''


class Speaker:
    """Synthesizes and plays sentences in the background.

    say() returns immediately. stop() cuts everything already queued - the
    point of barge-in is that pressing the talk key silences the agent now,
    not after the current sentence finishes.
    """

    def __init__(self, voice: str = 'af_heart', speed: float = 1.0,
                 device: int | None = None):
        self.voice = voice
        self.speed = speed
        self.device = device
        self._sentences: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()
        # Bumped by stop(); work tagged with an older generation is discarded
        # instead of being cancelled, so neither thread can block the other.
        self._gen = 0
        self._lock = threading.Lock()
        self._pipeline = None
        self._out = None
        # Audio waiting to reach the sound card. The output callback drains it
        # and stop() empties it - that is what makes silencing instant.
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_lock = threading.Lock()
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def load(self) -> None:
        """Import and warm Kokoro. Slow (~3s), so call it before the first use."""
        t0 = time.time()
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M',
                                   device='cpu')
        list(self._pipeline('Ready.', voice=self.voice))
        print(f'  tts ready in {time.time() - t0:.1f}s (voice={self.voice})',
              file=sys.stderr, flush=True)

    def say(self, text: str) -> None:
        text = speakable(text)
        if text:
            self._sentences.put((self._gen, text))

    def stop(self) -> None:
        """Silence immediately and drop everything pending.

        The output stream is deliberately left running. Calling abort() on it
        wedges the ALSA device ("File descriptor in bad state", -77) and every
        later write fails, so silence is produced by emptying the buffer the
        callback reads from instead.
        """
        with self._lock:
            self._gen += 1
        _drain(self._sentences)
        _drain(self._audio)
        with self._pending_lock:
            self._pending = np.zeros(0, dtype=np.float32)

    def _synth_loop(self) -> None:
        while True:
            gen, text = self._sentences.get()
            if gen != self._gen:
                continue
            try:
                for _, _, audio in self._pipeline(text, voice=self.voice,
                                                  speed=self.speed):
                    if audio is None or gen != self._gen:
                        continue
                    self._audio.put((gen, audio.detach().cpu().numpy()))
            except Exception as e:                      # noqa: BLE001
                print(f'  [TTS FAILED] {type(e).__name__}: {e}',
                      file=sys.stderr, flush=True)

    def _on_output(self, out, frames, _time, _status) -> None:
        """Feed the sound card from the pending buffer, padding with silence."""
        with self._pending_lock:
            n = min(frames, len(self._pending))
            out[:n, 0] = self._pending[:n]
            self._pending = self._pending[n:]
        if n < frames:
            out[n:, 0] = 0

    def _play_loop(self) -> None:
        import sounddevice as sd

        # One long-lived callback stream. A blocking write() cannot be
        # interrupted without aborting the device, and opening a stream per
        # sentence adds startup latency and clicks between chunks.
        self._out = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype='float32', device=self.device,
                                    blocksize=1024, callback=self._on_output)
        self._out.start()
        while True:
            gen, chunk = self._audio.get()
            if gen != self._gen:
                continue
            with self._pending_lock:
                if len(self._pending) == 0:
                    chunk = np.concatenate([_lead(), chunk])
                self._pending = np.concatenate([self._pending, chunk])

    def close(self) -> None:
        """Stop playback and release the device."""
        self.stop()
        if self._out is not None:
            self._out.stop()
            self._out.close()
            self._out = None


def _lead() -> np.ndarray:
    return np.zeros(int(LEAD_SILENCE * SAMPLE_RATE), dtype=np.float32)


def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
