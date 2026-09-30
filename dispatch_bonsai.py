"""Dispatch to bonsai (eslider/bonsai-1.7b) on Ollama, with read_file/bash
tools, plain chat (no OpenChamber sessions).

Same on_event/interrupt/send contract as dispatch.Dispatcher, so ptt.py and
ptt_tts.py can use this in place of the OpenChamber dispatcher unmodified -
just swap the import. Kept separate from dispatch.py because that one carries
OpenChamber sessions/agent/abort semantics this doesn't need or want.

Chat history is kept in-process (a plain list of messages) so the model has
conversational context across commands, the same way OpenChamber's session
does for the agent path.

Tool loop: a "delta" event only fires for assistant text, never for tool
call/result plumbing - the overlay (and ptt_tts's speaker) should stay quiet
while a tool runs. A "tool" event fires instead, same shape as dispatch.py's,
so the overlay can show what's running.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("LIVE_ASR_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = "eslider/bonsai-1.7b:latest"

INTERRUPT_NOTE = (
    "<interrupted>The user pressed the talk key and started speaking again "
    "before you finished the previous response, so that response was cut off. "
    "Do not resume it unless asked - just handle what they say next.</interrupted>\n\n"
)

MAX_TOOL_ROUNDS = 6         # hard stop against a runaway tool-call loop
BASH_TIMEOUT = 30
READ_FILE_MAX_BYTES = 20_000  # spoken/streamed back, so keep it sane

# Kept short on purpose - this is a 1.7b model, a long/dense prompt eats its
# attention budget and it starts ignoring parts of it. The failure mode this
# guards against is *describing* the tool call instead of making it (e.g.
# "you could run `date`, want me to?") - so the rule against that is stated
# on its own line, first, imperative, not buried in a longer sentence.
SYSTEM_PROMPT = (
    "You are an agent with tools: read_file(path), bash(command). "
    "You have no built-in knowledge of the current date, time, files, or "
    "system state - the only way to know any of that is to call a tool. "
    "Example: user asks the date -> you call bash(\"date\"), you never answer "
    "from memory and never say you can't access it. "
    "Always call the tool via the real function-call mechanism, never write "
    "the call as text or JSON in your reply. "
    "If a tool fails, call it again with a fixed command. "
    "Keep replies short, this is voice."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or ~-relative file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return its combined stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
]


def _tool_read_file(path: str) -> str:
    import os
    p = os.path.expanduser(path)
    try:
        with open(p, "r", errors="replace") as f:
            data = f.read(READ_FILE_MAX_BYTES + 1)
    except OSError as e:
        return f"ERROR: {e}"
    if len(data) > READ_FILE_MAX_BYTES:
        data = data[:READ_FILE_MAX_BYTES] + f"\n...(truncated at {READ_FILE_MAX_BYTES} bytes)"
    return data


def _tool_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {BASH_TIMEOUT}s"
    out = (result.stdout + result.stderr).strip()
    return f"exit={result.returncode}\n{out}"


TOOL_IMPLS = {"read_file": _tool_read_file, "bash": _tool_bash}


class Dispatcher:
    """Same shape as dispatch.Dispatcher: interrupt(), send(text), on_event.

    No sessions, no directory/agent concept - just a running chat history,
    seeded with SYSTEM_PROMPT, sent straight to Ollama's OpenAI-compatible
    endpoint.
    """

    def __init__(self, directory: str | None = None, provider_id: str | None = None,
                 model_id: str | None = None, agent: str | None = None,
                 timeout: float = 600, on_event=None,
                 base_url: str = DEFAULT_BASE_URL,
                 system_prompt: str = SYSTEM_PROMPT):
        # directory/provider_id/agent accepted-and-ignored so callers built
        # for dispatch.Dispatcher's signature drop in without edits.
        self.model_id = model_id or DEFAULT_MODEL
        self.timeout = timeout
        self.on_event = on_event or (lambda state, text="": None)
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        self._history: list[dict] = []
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})
        self._lock = threading.Lock()
        self._gen = 0
        self._active_gen: int | None = None
        self._interrupted_pending = False
        self._stream = None  # current openai stream, for abort-by-close

    def interrupt(self) -> bool:
        with self._lock:
            if self._active_gen is None:
                return False
            self._gen += 1
            self._active_gen = None
            self._interrupted_pending = True
            stream = self._stream
            self._stream = None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        self.on_event("interrupted", "")
        return True

    def send(self, text: str) -> None:
        with self._lock:
            self._gen += 1
            gen = self._gen
            tag = self._interrupted_pending
            self._interrupted_pending = False
            still_running = self._active_gen is not None
            stream = self._stream
            self._stream = None
            self._active_gen = gen
        if still_running:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self.on_event("interrupted", "")
            tag = True
        if tag:
            text = INTERRUPT_NOTE + text
        threading.Thread(target=self._run, args=(text, gen), daemon=True).start()

    def _run(self, text: str, gen: int) -> None:
        try:
            if gen != self._gen:
                return
            with self._lock:
                if gen != self._gen:
                    return
                self._history.append({"role": "user", "content": text})

            print(f"  [SENT] ({self.model_id}) {text!r}", file=sys.stderr, flush=True)
            t0 = time.time()

            for round_i in range(MAX_TOOL_ROUNDS):
                with self._lock:
                    if gen != self._gen:
                        return
                    messages = list(self._history)

                stream = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    tools=TOOLS,
                    stream=True,
                    timeout=self.timeout,
                )
                with self._lock:
                    if gen != self._gen:
                        stream.close()
                        return
                    self._stream = stream

                buf: list[str] = []
                tool_calls: dict[int, dict] = {}  # index -> {id, name, args}
                for chunk in stream:
                    if gen != self._gen:
                        print("  [DISPATCH] stream abandoned (superseded)",
                              file=sys.stderr, flush=True)
                        return
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        buf.append(delta.content)
                        self.on_event("delta", "".join(buf))
                    for tc in (delta.tool_calls or []):
                        slot = tool_calls.setdefault(
                            tc.index, {"id": None, "name": None, "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments

                if gen != self._gen:
                    return

                if not tool_calls:
                    # Plain text turn - done.
                    out = "".join(buf).strip()
                    dt = time.time() - t0
                    print(f"\n  [AGENT {dt:.1f}s] {out}\n", file=sys.stderr, flush=True)
                    with self._lock:
                        if gen == self._gen:
                            self._history.append({"role": "assistant", "content": out})
                    self.on_event("reply", out or "(empty reply)")
                    return

                # Tool call(s): record the assistant turn, run each tool,
                # append its result, then loop for the model's next move.
                calls_sorted = [tool_calls[i] for i in sorted(tool_calls)]
                assistant_msg = {
                    "role": "assistant",
                    "content": "".join(buf) or None,
                    "tool_calls": [
                        {
                            "id": c["id"] or f"call_{i}",
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"]},
                        }
                        for i, c in enumerate(calls_sorted)
                    ],
                }
                with self._lock:
                    if gen != self._gen:
                        return
                    self._history.append(assistant_msg)

                for c in calls_sorted:
                    name = c["name"]
                    call_id = c["id"] or "call_0"
                    print(f"  [TOOL] {name}({c['args']})", file=sys.stderr, flush=True)
                    self.on_event("tool", name or "")
                    impl = TOOL_IMPLS.get(name)
                    if impl is None:
                        result = f"ERROR: unknown tool {name!r}"
                    else:
                        try:
                            kwargs = json.loads(c["args"] or "{}")
                        except json.JSONDecodeError as e:
                            result = f"ERROR: bad arguments json: {e}"
                        else:
                            try:
                                result = impl(**kwargs)
                            except Exception as e:
                                result = f"ERROR: {type(e).__name__}: {e}"
                    print(f"  [TOOL RESULT] {name} -> {result[:300]!r}",
                          file=sys.stderr, flush=True)
                    with self._lock:
                        if gen != self._gen:
                            return
                        self._history.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result,
                        })

            # Exhausted MAX_TOOL_ROUNDS without a final text answer.
            self.on_event("reply", "(stopped after too many tool calls)")
        except Exception as e:
            if gen == self._gen:
                print(f"\n  [DISPATCH FAILED] {type(e).__name__}: {e}\n",
                      file=sys.stderr, flush=True)
                self.on_event("error", f"{type(e).__name__}: {e}")
        finally:
            with self._lock:
                if self._active_gen == gen:
                    self._active_gen = None
                if self._stream is not None:
                    self._stream = None
