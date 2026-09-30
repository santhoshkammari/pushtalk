# phonon2

Speech-to-text service. Model: Phonon-2 (164 MB, English, CPU only, ~10x realtime on this laptop).

One process hold model. Everything else talk to it. Model load once, never twice.

```
phonon2/service.py   the service: model + Shift+Space hotkey + unix socket
phonon2/client.py    tiny client: send audio, get text. no model inside
../run.sh dictate    start/stop/status the service
```

## What it do

- Hold **Shift+Space** anywhere -> record mic. Let go -> text typed at cursor.
- Tap shorter than 0.3s = ignored.
- Socket `~/.cache/phonon-asr/asr.sock`: any program send audio, get text back.
- pushtalk F9 engine (`../ptt.py`, `../ptt_tts.py`) use this socket. No own model.

## Setup (once)

Use `~/main` env. Never make new venv.

```bash
uv pip install --python ~/main/bin/python fermion-research huggingface_hub "transformers==5.17.0"
```

- Needs torch, safetensors, sounddevice, soundfile, python-xlib, numpy. `~/main` already have.
- `transformers==5.17.0` needed (has `ParakeetForTDT`). Old 5.5.3 fail. Rollback: `transformers==5.5.3`.
- Model (164 MB) auto-download first run to `~/.cache/fermion/speech/FermionResearch__Phonon-2/`.
- Needs: X11 session, `xdotool`, working mic.

Autostart at login: `~/.config/autostart/phonon-asr.desktop`
```
Exec=/path/to/pushtalk/run.sh dictate run
```

## Run

```bash
./run.sh dictate start     # background, model load ~25s
./run.sh dictate status
./run.sh dictate stop
```

See `USAGE.md` for details, socket protocol, trouble.

## Limits

- X11 only. Wayland no work (hotkey grab).
- Shift+Space grabbed system-wide: normal Shift+Space no longer reach any app.
- English. 16 kHz mono. CPU.
- Notification: none. Silent on purpose.
