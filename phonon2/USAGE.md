# phonon2 usage (for agents)

Rule: never load Phonon-2 yourself. Call service. Service already hold model.

## Is it up?

```bash
./run.sh dictate status   # exit 0 = running
ls ~/.cache/phonon-asr/asr.sock                                 # socket exist = ready
tail ~/.cache/phonon-asr/service.log                            # "ready: hold Shift+Space..." = model loaded
```

Not up -> `run.sh dictate start`. Wait ~25s for model. Or just call client: it start service for you.

## Transcribe from python

Run with `~/main/bin/python`, cwd or PYTHONPATH = `pushtalk/`.

```python
import soundfile as sf
from phonon2.client import transcribe, ensure_running

ensure_running()                       # start service if down, wait till socket up (max 90s)
audio, sr = sf.read("clip.wav", dtype="float32")   # must be 16000 Hz mono
print(transcribe(audio))               # -> "He could wait no longer."
```

- Input: numpy float32, 16 kHz, mono, values -1..1. Not 16 kHz? resample first.
- Shorter than 0.3s -> return `""`.
- Service decode one clip at a time. Parallel calls queue up. Fine.
- Speed: ~0.2s for 2s clip.

## Transcribe from shell (no python)

```bash
~/main/bin/fermion transcribe file.wav --model phonon-2
```
This load own copy of model (~25s, 2 GB RAM). Use only one-off, not in loops. Prefer socket.

## Socket protocol (other languages)

Unix socket `~/.cache/phonon-asr/asr.sock` (file mode 0600).

1. Send: `uint32 LE  n_samples` then `n_samples * float32 LE` (16 kHz mono).
2. Recv: `uint32 LE  n_bytes` then UTF-8 text.
3. One request per connection. Close after.

## User dictation

Human hold Shift+Space, talk, let go. Text typed into focused window via `xdotool type`. Agent no need do anything.

## Live-asr F9 engine

```bash
run.sh start            # overlay + F9 engine (ptt.py) -> agent
run.sh start --tts      # same, agent answer spoken (ptt_tts.py)
```
Engine call `phonon2.client`. Service auto-start if down. Old Parakeet versions in `../backup/`.

## Trouble

| symptom | cause -> fix |
|---|---|
| `cannot grab Shift+Space` in log, service exit | other app own key -> close it (Handy gone). Then `dictate start` |
| socket missing, status stopped | `run.sh dictate start`, wait 25s |
| `ImportError ParakeetForTDT` | wrong transformers -> `uv pip install --python ~/main/bin/python "transformers==5.17.0"` |
| nothing typed after release | clip <0.3s, or silence, or no mic -> check `arecord -l`, log |
| stale socket, connect refused | service dead -> `dictate stop; dictate start` |
| two services fight | only one may run. `run.sh dictate stop`, `pgrep -af phonon2.service` |

## Files / paths

- Service log: `~/.cache/phonon-asr/service.log`
- Pid: `~/.cache/pushtalk/dictate.pid`
- Model: `~/.cache/fermion/speech/FermionResearch__Phonon-2/`
- Autostart: `~/.config/autostart/phonon-asr.desktop`
