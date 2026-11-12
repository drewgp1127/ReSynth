# Patch Finder

Prototype that turns a short clip of a synth sound into a loadable [Vital](https://vital.audio)
preset by render-and-compare search. See `docs/plan.md` for the full design and
`docs/vita-notes.md` for facts about the `vita` Python bindings this relies on.

Built so far: the synth wrapper, loss, CMA-ES matcher, starting templates, and the isolate and
analyse stages. The Claude stages (research, plan, report), web UI and deployment are not built yet.

## Setup

Linux or macOS, Python 3.11-3.13 (developed on 3.13).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

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
```

## Stages

Every stage is a function that takes a job directory and reads/writes plain files there, and
writes progress to `<job>/status/<stage>.json`.

| Stage | Function / CLI | Reads | Writes |
|---|---|---|---|
| match | `app.stages.match.run_match(job_dir)` / `python -m app.stages.match <job_dir>` | `target.wav`, `analysis.json`, `plan.json` | `match.json`, `match_{1,2,3}.vital`, `match_1.wav` |

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
