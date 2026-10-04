"""Trigger-phrase detection and non-blocking dispatch to OpenChamber.

Speech has no "enter" key, so a command is delimited by a trigger phrase at
the front and a pause at the back:

    "... agent send fix the login bug in inresearch" <pause>
         ^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         trigger    command

Anything spoken before a trigger is ignored, so ordinary talking near the mic
costs nothing. Dispatch runs on a worker thread: OpenChamber's chat route
blocks until the agent finishes, and the mic must not stall behind it.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time

# --------------------------------------------------------------- model routing
# No model is baked in: the caller picks one and passes it via the environment
# before launching (the `asr-*` aliases in ~/.bashrc do this).
#   LIVE_ASR_PROVIDER  provider id (default: opencode-go)
#   LIVE_ASR_MODEL     model id    (default: deepseek-v4.1-flash)
# The provider/model is used exactly as given - there is no silent fallback,
# so a dead vLLM box surfaces as a dispatch error instead of answering from
# some other model behind your back.
DEFAULT_PROVIDER = os.environ.get("LIVE_ASR_PROVIDER", "opencode-go")
DEFAULT_MODEL = os.environ.get("LIVE_ASR_MODEL", "deepseek-v4.1-flash")

# Spoken triggers. Matched on the committed word stream, so they must be
# written exactly as parakeet would transcribe them (lowercase, no
# punctuation - punctuation is stripped before matching).
TRIGGERS = ("agent send", "send agent", "do this", "do that")

# A command is considered finished after this much silence.
DEFAULT_END_PAUSE = 1.5


def _norm(word: str) -> str:
    """Lowercase and strip punctuation so 'send,' matches 'send'."""
    return re.sub(r"[^\w'-]", "", word.lower())


class CommandBuffer:
    """Watches committed words for a trigger, then collects the command.

    Fed one committed word at a time (never revised, so nothing needs to be
    taken back). Once armed by a trigger, subsequent words accumulate until
    `check_timeout` sees enough silence.
    """

    def __init__(self, triggers=TRIGGERS, end_pause: float = DEFAULT_END_PAUSE):
        # Longest trigger first so "send agent" wins over a bare "send".
        self.triggers = sorted((t.lower().split() for t in triggers),
                               key=len, reverse=True)
        self.max_trigger_len = max(len(t) for t in self.triggers)
        self.end_pause = end_pause
        self.recent: list[str] = []   # tail of normalized words, for matching
        self.armed = False
        self.command: list[str] = []  # original-case words after the trigger
        self.last_word_t = 0.0

    def _match_trigger(self) -> int:
        """Return the trigger length if `recent` ends with a trigger, else 0."""
        for t in self.triggers:
            if len(self.recent) >= len(t) and self.recent[-len(t):] == t:
                return len(t)
        return 0

    def add(self, word_orig: str) -> None:
        """Feed one committed word."""
        self.last_word_t = time.time()
        n = _norm(word_orig)
        if not n:
            return

        if self.armed:
            self.command.append(word_orig)
            return

        self.recent.append(n)
        if len(self.recent) > self.max_trigger_len:
            self.recent.pop(0)

        if self._match_trigger():
            self.armed = True
            self.command = []
            self.recent.clear()
            print("\n  [TRIGGER] listening for command...", file=sys.stderr, flush=True)

    def check_timeout(self) -> str | None:
        """Return the finished command if the trailing pause has elapsed."""
        if not self.armed or not self.command:
            return None
        if time.time() - self.last_word_t < self.end_pause:
            return None
        text = " ".join(self.command).strip()
        self.reset()
        return text or None

    def reset(self) -> None:
        self.armed = False
        self.command = []
        self.recent.clear()


class Dispatcher:
    """Sends commands to OpenChamber on a worker thread.

    One session is reused across commands so the agent keeps context between
    them, the same way a chat thread would.

    Only one turn runs at a time. If a new command arrives while the agent is
    still answering, the running turn is aborted (server-side + stream stops).
    """

    def __init__(self, directory: str, provider_id: str = DEFAULT_PROVIDER,
                 model_id: str = DEFAULT_MODEL, agent: str | None = None,
                 timeout: float = 600, on_event=None):
        self.directory = directory
        self.provider_id = provider_id
        self.model_id = model_id
        self.agent = agent
        self.timeout = timeout
        # on_event(state, text) - lets a UI follow the dispatch without this
        # module needing to know anything about the UI.
        self.on_event = on_event or (lambda state, text="": None)
        self._oc = None
        self._session_id: str | None = None
        # (provider_id, model_id) the current session was created with, so a
        # later model switch can force a fresh session.
        self._model: tuple[str, str] | None = None
        self._lock = threading.Lock()
        # Bumped on every send()/interrupt(); a worker whose gen != _gen has
        # been superseded and must stop touching the stream and the UI.
        self._gen = 0
        # gen of the turn currently generating, or None when idle. Set by the
        # worker, cleared by the worker or by interrupt().
        self._active_gen: int | None = None

    def _client(self):
        if self._oc is None:
            if os.environ.get("OPENCHAMBER_AI_PATH"):
                sys.path.insert(0, os.environ["OPENCHAMBER_AI_PATH"])
            from openchamber_ai import OpenChamber
            self._oc = OpenChamber(directory=self.directory, timeout=self.timeout)
        return self._oc

    def interrupt(self) -> bool:
        """Abort the turn in flight, if any. Safe to call when idle.

        Called the instant the talk key goes down so the model stops
        generating immediately, not only once the new utterance is decoded.
        """
        with self._lock:
            if self._active_gen is None:
                return False
            self._gen += 1              # supersede the running worker
            self._active_gen = None
            sid = self._session_id
        if sid and self._oc is not None:
            ok = self._oc.abort(sid)
            print(f"\n  [INTERRUPT] abort -> {ok}\n", file=sys.stderr, flush=True)
        self.on_event("interrupted", "")
        return True

    def send(self, text: str) -> None:
        """Fire and forget - returns immediately so the mic keeps running.

        _active_gen is claimed HERE, synchronously, under the lock - not at
        the top of _run(). A thread that has only just been started() is not
        guaranteed to be scheduled before the caller's next line runs, so if
        two send() calls land close together (mic release racing a fresh
        press) and _active_gen were set inside _run(), the second call could
        still see _active_gen is None and skip the abort - both turns would
        then hit the same session/prompt route concurrently and the model
        would concatenate them into one garbled answer.
        """
        with self._lock:
            self._gen += 1
            gen = self._gen
            still_running = self._active_gen is not None
            sid = self._session_id
            self._active_gen = gen
        if still_running:
            # New command landed mid-answer without a prior interrupt()
            # (e.g. toggle mode) - abort now.
            if sid and self._oc is not None:
                self._oc.abort(sid)
            self.on_event("interrupted", "")
        threading.Thread(target=self._run, args=(text, gen), daemon=True).start()

    def _run(self, text: str, gen: int) -> None:
        try:
            if gen != self._gen:
                # Superseded before we even got to run (a newer send()/
                # interrupt() landed while this thread was still being
                # scheduled) - do not touch the session or the network.
                return
            oc = self._client()

            # Model comes from the env at launch (LIVE_ASR_PROVIDER /
            # LIVE_ASR_MODEL), with --provider/--model able to override it.
            # It MUST be set at create_session time - the per-message
            # providerID/modelID is ignored when the session has no model
            # pinned (it falls back to the global default). So if the choice
            # changed since the session was made, drop it and make a fresh one.
            provider_id, model_id = self.provider_id, self.model_id
            with self._lock:
                if gen != self._gen:
                    # Someone else claimed _active_gen while we were waiting
                    # on the lock (or resolving the model) - stand down
                    # rather than create a session or send a stale prompt.
                    return
                if self._session_id is not None and self._model != (provider_id, model_id):
                    print(f"  [MODEL] switched {self._model} -> {(provider_id, model_id)}, new session",
                          file=sys.stderr, flush=True)
                    self._session_id = None
                if self._session_id is None:
                    s = oc.create_session(agent=self.agent,
                                          provider_id=provider_id, model_id=model_id)
                    self._session_id = s["id"]
                    self._model = (provider_id, model_id)
                    print(f"  [SESSION] {self._session_id}", file=sys.stderr, flush=True)
                session_id = self._session_id

            print(f"  [SENT] ({provider_id}/{model_id}) {text!r}", file=sys.stderr, flush=True)
            t0 = time.time()

            # Stream the answer so the overlay fills in as the agent talks
            # instead of sitting blank until the whole turn is done.
            buf: list[str] = []
            tool_seen = None
            # Only session.tool.input.started carries the tool's `name`;
            # called/progress/success/failed carry just the `id` it was
            # started with, so remember the name here to re-attach it.
            tool_names: dict[str, str] = {}
            for ev in oc.chat(session_id, text, stream=True):
                if gen != self._gen:
                    # Superseded by a newer command - stop consuming; the
                    # server turn was already aborted by interrupt()/send().
                    print("  [DISPATCH] stream abandoned (superseded)",
                          file=sys.stderr, flush=True)
                    return
                delta = ev.text_delta
                if delta:
                    buf.append(delta)
                    self.on_event("delta", "".join(buf))
                    continue
                tool = ev.tool
                if not tool:
                    continue
                if tool["name"]:
                    tool_names[tool["id"]] = tool["name"]
                name = tool["name"] or tool_names.get(tool["id"], "")
                if name and tool["status"] != tool_seen:
                    tool_seen = tool["status"]
                    # v2 status comes from the event-type suffix: input.started
                    # (name known, not yet invoked) / called (invoked,
                    # running) / progress / success / failed - "pending"/
                    # "running" were the old v1 vocabulary and never matched
                    # here, so the hint never fired after the v2 migration.
                    if tool["status"] in ("input.started", "called", "progress"):
                        self.on_event("tool", name)

            if gen != self._gen:
                return
            out = "".join(buf).strip()
            dt = time.time() - t0
            print(f"\n  [AGENT {dt:.1f}s] {out}\n", file=sys.stderr, flush=True)
            self.on_event("reply", out or "(empty reply)")
        except Exception as e:
            if gen == self._gen:
                print(f"\n  [DISPATCH FAILED] {type(e).__name__}: {e}\n",
                      file=sys.stderr, flush=True)
                self.on_event("error", f"{type(e).__name__}: {e}")
        finally:
            with self._lock:
                if self._active_gen == gen:
                    self._active_gen = None
