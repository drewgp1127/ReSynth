# ReSynth

Prototype that turns a short clip of a synth sound into a loadable [Vital](https://vital.audio)
preset by render-and-compare search. See `docs/plan.md` for the full design and
`docs/vita-notes.md` for facts about the `vita` Python bindings this relies on.

Built so far: the synth wrapper, loss, CMA-ES matcher, starting templates, and the isolate and
analyse stages. The Claude stages (research, plan, report), web UI and deployment are not built yet.

## Setup

Linux or macOS, Python 3.11-3.13 (developed on 3.13).

```bash
python3 -m venv .venv
# CPU-only torch (for stem separation); the PyPI default pulls several GB of CUDA libraries
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev,analysis,separation]"
# basic-pitch without its tensorflow requirement (uses the ONNX model bundled in the wheel)
.venv/bin/pip install --no-deps basic-pitch==0.4.0
```

Minimal install (synth, loss, match only): `.venv/bin/pip install -e ".[dev]"`. Without
basic-pitch the analyse stage falls back to librosa pYIN (monophonic notes only); without
audio-separator the isolate stage degrades to the raw slice.

Stem-separation weights (htdemucs_6s, ~55 MB, from dl.fbaipublicfiles.com / github.com) are
downloaded on first use into `models/` (override with `RESYNTH_MODELS`).

## Tests

```bash
.venv/bin/python -m pytest              # default suite (slow tests skipped)
.venv/bin/python -m pytest -m slow      # long searches only
.venv/bin/python -m pytest -m ""        # everything
```

Other scripts:

```bash
.venv/bin/python scripts/spike_vita.py  # milestone 0 checks of the vita API
.venv/bin/python scripts/bench.py       # render / loss throughput
.venv/bin/python scripts/make_templates.py   # regenerate app/templates/*.vital (deterministic)
.venv/bin/python -m tests.e2e_pipeline [out_dir] [--budget 180]   # end-to-end check -> scratch/e2e
```

## Stages

Every stage is a function that takes a job directory and reads/writes plain files there, and
writes progress to `<job>/status/<stage>.json`.

| Stage | Function / CLI | Reads | Writes |
|---|---|---|---|
| isolate | `app.stages.isolate.run_isolate(job_dir, upload, start_s, end_s)` then `choose_target(job_dir, stem)` / `python -m app.stages.isolate <job_dir> <upload> <start> <end>` | uploaded clip | `slice.wav`, `stems/*.wav`, `isolate.json`; then `target.wav`, `target_mono.wav` |
| analyse | `app.stages.analyse.run_analyse(job_dir)` / `python -m app.stages.analyse <job_dir>` | `target.wav` | `analysis.json` |
| match | `app.stages.match.run_match(job_dir)` / `python -m app.stages.match <job_dir>` | `target.wav`, `analysis.json`, `plan.json` | `match.json`, `match_{1,2,3}.vital`, `match_1.wav` |

`app.stages.common.make_ab(target, match)` builds the loudness-matched A/B audio for the report stage.

Templates live in `app/templates/` with `manifest.json` (description + default search parameters
per template). All templates share one modulation layout (see `app/synth/builder.py`), so the
parameter names in `app/synth/params.py` mean the same thing in every template.

`plan.json` (normally written by the plan stage) looks like:

```json
{"template": "pitch_blip_keys",
 "params": ["filter_cutoff", "amp_attack", "osc_detune_cents"],
 "ranges": {"filter_cutoff": [300, 8000]},
 "initial": {"amp_attack": 0.01},
 "fixed": {},
 "weights": {"stft": 0.5, "ltas": 3.0, "mfcc": 0.5, "env": 1.0, "attack": 2.0, "pitch": 0.0}}
```

Parameter names and units are listed in `app/synth/params.py`.
