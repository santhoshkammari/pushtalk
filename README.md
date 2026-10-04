# pushtalk

[![Stars](https://img.shields.io/github/stars/santhoshkammari/pushtalk?style=social)](https://github.com/santhoshkammari/pushtalk/stargazers)
[![Watchers](https://img.shields.io/github/watchers/santhoshkammari/pushtalk?style=social)](https://github.com/santhoshkammari/pushtalk/watchers)
[![Forks](https://img.shields.io/github/forks/santhoshkammari/pushtalk?style=social)](https://github.com/santhoshkammari/pushtalk/network/members)
[![License: MIT](https://img.shields.io/github/license/santhoshkammari/pushtalk)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/santhoshkammari/pushtalk)](https://github.com/santhoshkammari/pushtalk/commits/main)
[![Issues](https://img.shields.io/github/issues/santhoshkammari/pushtalk)](https://github.com/santhoshkammari/pushtalk/issues)

**Push-to-talk voice control for AI coding agents.** Hold a key, speak, release. Your speech is transcribed offline on CPU, sent to an LLM agent, and the answer streams into a translucent always-on-top HUD, optionally spoken back with Kokoro TTS.

No cloud speech API. No chat window. Works with any OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama) or an [OpenChamber](https://github.com/openchamber/openchamber) agent session.

Keywords: voice assistant, voice control for LLM agents, push-to-talk, offline speech-to-text, local voice agent, Linux, PyQt5 overlay, Kokoro TTS, vLLM, llama.cpp, Ollama.

## How it works

```
 F9 held ──► mic ──► Phonon-2 STT (local, CPU) ──► agent / LLM ──► HUD (streamed)
                                                                └─► Kokoro TTS (optional)
```

- Hold **F9**, speak, release. The utterance is transcribed offline with Phonon-2 (164 MB, English, ~10x realtime on a laptop CPU).
- The text is dispatched to an OpenChamber session, or, in plain-chat mode, straight to an OpenAI-compatible server.
- The answer streams into a glass panel at the top of the screen and stays until you press **Esc**. The panel is a real window, so it shows up in alt-tab.
- Press F9 while the agent is answering and its turn is aborted server-side; your next command is then sent normally, with no interruption note.
- **Shift+Space** anywhere is system-wide dictation: it types what you say at the cursor.

## Requirements

- Linux with X11, a working mic, `xdotool`
- Python 3 venv with PyQt5, torch, sounddevice, soundfile, python-xlib, numpy (`LIVE_ASR_VENV`, default `~/main`)
- `transformers==5.17.0` and `fermion-research` for Phonon-2, plus `kokoro` for speech output

```bash
uv pip install --python ~/main/bin/python fermion-research huggingface_hub "transformers==5.17.0"
```

The Phonon-2 model auto-downloads on first run.

## Quick start

```bash
git clone https://github.com/santhoshkammari/pushtalk && cd pushtalk
./run.sh start           # overlay + engine in the background
./run.sh status
./run.sh stop
./run.sh restart         # re-reads env
./run.sh run             # foreground, ctrl-c stops
./run.sh run --no-agent  # transcribe only, print text
```

Logs go to `/tmp/pushtalk.log`, pids to `~/.cache/pushtalk/`.

## Pick a model

Everything is driven by environment variables, read on every `start` / `restart`:

| var | meaning | default |
|-----|---------|---------|
| `LIVE_ASR_VENV` | venv whose `bin/python` runs overlay + engine | `~/main` |
| `LIVE_ASR_PROVIDER` | OpenChamber provider id | `opencode-go` |
| `LIVE_ASR_MODEL` | model id | `deepseek-v4.1-flash` |
| `LIVE_ASR_DISPATCH` | `bonsai` = plain chat to an OpenAI-compatible server, no tools/sessions | unset (OpenChamber) |
| `LIVE_ASR_BASE_URL` | base URL for plain-chat mode | `http://localhost:11434/v1` (Ollama) |
| `OPENCHAMBER_AI_PATH` | directory containing `openchamber_ai.py` if it is not installed | unset |

```bash
# agent session through OpenChamber, model served by vLLM
LIVE_ASR_PROVIDER=vllm LIVE_ASR_MODEL=my-model ./run.sh start

# plain chat against any OpenAI-compatible server (llama.cpp, vLLM, Ollama)
LIVE_ASR_DISPATCH=bonsai LIVE_ASR_BASE_URL=http://my-gpu-box:8080/v1 \
LIVE_ASR_MODEL=my-model ./run.sh start
```

The provider/model is used exactly as given. There is no silent fallback: if the server is down, the dispatch fails loudly instead of answering from some other model.

A plain shell alias cannot take an argument, so put the model in the alias name:

```bash
alias asr='LIVE_ASR_PROVIDER=opencode-go LIVE_ASR_MODEL=deepseek-v4.1-flash /path/to/pushtalk/run.sh start'
alias asr-local='LIVE_ASR_DISPATCH=bonsai LIVE_ASR_BASE_URL=http://localhost:8080/v1 LIVE_ASR_MODEL=my-model /path/to/pushtalk/run.sh start'
alias asr-stop='/path/to/pushtalk/run.sh stop'
```

## Speech in, speech out

Add `--tts` and the answer is spoken aloud.

```bash
./run.sh start --tts
./run.sh start --tts --voice am_michael --tts-speed 1.1
./run.sh start --tts --no-speak
```

The reply is already streamed token by token to drive the HUD, so the same deltas feed the voice. Complete sentences are cut out of the stream and synthesized as soon as they close, so speech starts after the agent's **first sentence**, not after its whole turn. Kokoro's default splitter treats a paragraph as one chunk, which cost 5.65 s to first audio against 1.39 s with sentence splitting.

Kokoro runs at RTF ~0.55 on CPU, so synthesis outruns playback. The first chunk is padded with 1.2 s of silence to build a cushion. Code blocks, URLs and absolute paths are read as "code block", "link", "path". F9 silences playback immediately; there is no echo cancellation, which is why interruption is bound to the key and not to voice activity.

## Global shortcut

`toggle.sh` wraps `run.sh toggle`: first press starts overlay + engine, next press stops both. Bind it to a key, for example on GNOME:

```bash
P=/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom2/
S=org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$P
gsettings set "$S" name 'pushtalk voice toggle'
gsettings set "$S" command "$PWD/toggle.sh"
gsettings set "$S" binding '<Control><Alt>v'
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "['$P']"
```

## HUD states

| state | shows |
|-------|-------|
| `listening` | live waveform driven by mic level |
| `thinking` | animated dots |
| `delta` | the answer, streamed in place |
| `reply` | answer settles, stays until **Esc** |
| `interrupted` | red flash, answer dropped |
| `error` | red cross + message |

## Layout

| file | role |
|------|------|
| `ptt.py` | engine: F9 hotkey, mic capture, STT client call, dispatch |
| `ptt_tts.py` | same engine, reply spoken back |
| `tts_worker.py` | sentence cutter + Kokoro synth thread + streaming playback |
| `dispatch.py` | trigger-phrase detection + non-blocking send to OpenChamber, interrupt handling |
| `dispatch_bonsai.py` | plain-chat dispatcher for any OpenAI-compatible server |
| `overlay_qt.py` | HUD: PyQt5 frameless glass panel + system tray, driven over a unix socket |
| `ui_bus.py` | one-line JSON IPC between engine and HUD |
| `phonon2/` | the Phonon-2 STT service + client (Shift+Space dictation, shared by F9). See `phonon2/README.md` |
| `live_parakeet.py` | standalone live rolling-window transcription (sherpa-onnx Parakeet) |
| `kokoro_tts.py` | standalone batch TTS test rig |
| `run.sh` / `toggle.sh` | launcher and global-shortcut wrapper |

## Support

If pushtalk is useful to you, a **star** on GitHub helps other people find it. **Watch** the repo to get release news, and open an issue if something breaks.

## License

MIT
