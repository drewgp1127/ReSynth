# ReSynth

ReSynth turns a short clip of a synth sound from a song into a loadable [Vital](https://vital.audio)
preset, plus a report on the gear that was likely used. It separates the sound from the mix,
measures it, and searches Vital's parameters until a render sounds like the clip
(render-and-compare). Claude (optional) researches the gear on the web, helps plan the search and
writes the "on your gear" notes.

See `docs/plan.md` for the design and `docs/vita-notes.md` for facts about the `vita` Python
bindings this relies on. Prototype: personal use, one job at a time.

## Windows setup (plain steps)

ReSynth has been developed and tested on Linux. **It has not been tested on Windows yet**; the
setup and start scripts below are written for Windows 10/11 but have never been run there. If a
step fails, copy the message from the window and send it to whoever maintains this.

1. **Install Python 3.13** from <https://www.python.org/downloads/>. In the installer, tick
   **"Add python.exe to PATH"**.
2. **(Only for M4A/AAC files) install ffmpeg.** WAV, FLAC and MP3 work without it. To install
   it, open *Command Prompt* and run `winget install Gyan.FFmpeg`, then close and reopen any open
   windows. (Or convert your clip to WAV first.)
3. **Get the ReSynth folder** onto your computer (download and unzip it, e.g. to `Documents\ReSynth`).
4. **(Optional) set your Claude API key** so ReSynth can research the gear. Open the Start menu,
   search **"Edit environment variables for your account"**, click **New...**, and enter
   - Variable name: `RESYNTH_ANTHROPIC_KEY`
   - Variable value: your API key from <https://console.anthropic.com>

   Click OK. Without a key everything still works except the gear research, and the plan and
   report are made by simple rules instead of Claude.
5. **Double-click `scripts\run_windows.bat`.** The first time it installs everything (about
   1 GB of downloads, 10-20 minutes). Then it opens ReSynth in your browser at
   <http://127.0.0.1:8000>. Keep the black window open while you use it; close it to stop.

Also installed on first use: the stem-separation model (~55 MB, downloaded on the first job).

**Use it from your phone or another computer on the same Wi-Fi:** start it with
`scripts\run_windows.bat --lan` and open `http://<this computer's IP>:8000`. Set a password first
(a `RESYNTH_PASSWORD` variable, the same way as the API key): anyone who can reach the app can use
it and spend your API credit. Do the same before putting it behind a tunnel.

What is most likely to need fixing on Windows (untested): the PowerShell execution policy (the
.bat passes `-ExecutionPolicy Bypass`), the `py -3.13` / `python` lookup in
`scripts/setup_windows.ps1`, audio-separator / torch install on Windows, and antivirus software
slowing down the first run.

## Using the app

1. Upload a clip (up to 60 s; WAV, MP3, M4A or FLAC), give the song, artist, a description of the
   sound, and optionally your gear (e.g. "Serum, Logic Alchemy").
2. Drag the highlighted box over the 1-3 seconds where the sound is clearest and press
   **Separate stems**.
3. Listen to each stem (and the original selection) and pick the one where the sound is clearest
   (the suggested one is preselected). Press **Find the patch**.
4. Wait for the search (about 3 minutes by default). You get an A/B player (your target, then the
   match at the same loudness), a similarity label (close / same family / weak), `patch.vital`
   plus two runner-ups, and the report: gear research with confidence labels and links, the
   recipe in generic synth terms (from the matched preset), notes for your gear, recommended
   effects and caveats.

Gear research runs alongside stem separation. If there is no API key or the research fails, the
job still finishes and the page says the research was skipped and why.

## Setup on Linux / macOS (developers)

Python 3.11-3.13 (developed on 3.13).

```bash
python3 -m venv .venv
# CPU-only torch (for stem separation); the PyPI default pulls several GB of CUDA libraries
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev,analysis,separation,web,claude]"
# basic-pitch without its tensorflow requirement (uses the ONNX model bundled in the wheel)
.venv/bin/pip install --no-deps basic-pitch==0.4.0
```

Start the app:

```bash
.venv/bin/python -m app.main            # http://127.0.0.1:8000 (this computer only)
.venv/bin/python -m app.main --lan      # all interfaces (set RESYNTH_PASSWORD first)
```

Settings are environment variables: `RESYNTH_ANTHROPIC_KEY` (the only place the API key is read
from), `RESYNTH_PORT`, `RESYNTH_HOST`, `RESYNTH_PASSWORD`, `RESYNTH_MATCH_BUDGET` (search seconds,
default 180), `RESYNTH_JOBS` (job folder, default `./jobs`), `RESYNTH_MODELS` (separation weights,
default `./models`), `RESYNTH_FFMPEG` (path to ffmpeg if it is not on PATH).

Audio decoding: soundfile (libsndfile) reads WAV, FLAC, OGG and MP3. M4A/AAC needs the `ffmpeg`
program; the upload says so if it is missing.

Minimal install (synth, loss, match only): `.venv/bin/pip install -e ".[dev]"`. Without
basic-pitch the analyse stage falls back to librosa pYIN (monophonic notes only); without
audio-separator the isolate stage degrades to the raw slice.

Stem-separation weights (htdemucs_6s, ~55 MB, from dl.fbaipublicfiles.com / github.com) are
downloaded on first use into `models/`.

## Tests

```bash
.venv/bin/python -m pytest              # default suite (slow tests skipped)
.venv/bin/python -m pytest -m slow      # long searches only
.venv/bin/python -m pytest -m ""        # everything
```

The Claude stages are unit-tested with a stubbed client (`tests/fake_claude.py`); no test calls
the real API.

Browser test (needs Node with the `playwright` package and its Chromium; starts nothing itself):

```bash
.venv/bin/python -m tests.browser.make_clip scratch/browser          # synthetic clip, no song audio
RESYNTH_MATCH_BUDGET=40 RESYNTH_JOBS=scratch/browser/jobs .venv/bin/python -m app.main &
node tests/browser/flow.mjs http://127.0.0.1:8000 scratch/browser/song.wav 2.0 4.6 scratch/browser docs/screenshots
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
writes progress to `<job>/status/<stage>.json`. The web app runs them through `app/jobs.py`, one
child process per phase.

| Stage | Function / CLI | Reads | Writes |
|---|---|---|---|
| research | `app.stages.research.run_research(job_dir, song, artist, description)` / `python -m app.stages.research <job_dir> --song .. --artist ..` | inputs | `research.json` |
| isolate | `app.stages.isolate.run_isolate(job_dir, upload, start_s, end_s)` then `choose_target(job_dir, stem)` / `python -m app.stages.isolate <job_dir> <upload> <start> <end>` | uploaded clip | `slice.wav`, `stems/*.wav`, `isolate.json`; then `target.wav`, `target_mono.wav` |
| analyse | `app.stages.analyse.run_analyse(job_dir)` / `python -m app.stages.analyse <job_dir>` | `target.wav` | `analysis.json` |
| plan | `app.stages.plan.run_plan(job_dir, description)` / `python -m app.stages.plan <job_dir> [--offline]` | `analysis.json`, `research.json` | `plan.json` |
| match | `app.stages.match.run_match(job_dir)` / `python -m app.stages.match <job_dir>` | `target.wav`, `analysis.json`, `plan.json` | `match.json`, `match_{1,2,3}.vital`, `match_1.wav` |
| report | `app.stages.report.run_report(job_dir, gear)` / `python -m app.stages.report <job_dir> [--gear ..] [--offline]` | all of the above | `patch.vital`, `patch_alt{1,2}.vital`, `ab.wav`, `report.{json,md,html}` |

Claude stages (`claude-opus-5-5` via the official `anthropic` SDK): research uses the server-side
web search tool, then structured outputs to turn its notes into `research.json`; every cited URL
is checked against the URLs the search actually returned, a "confirmed" claim on an unverified URL
is downgraded, recreations are always marked as not the original gear, and "model unknown" is
stated when no confirmed source names a model. The plan stage validates Claude's plan against
`app/synth/params.py` and the manifest and falls back to the rule-based plan
(`app/stages/plan_rules.py`) on any problem or without a key. The report's recipe is generated
from the final matched values (`app/stages/recipe.py`); Claude only writes the gear notes,
effect suggestions and tips around it.

Templates live in `app/templates/` with `manifest.json` (description + default search parameters
per template). All templates share one modulation layout (see `app/synth/builder.py`), so the
parameter names in `app/synth/params.py` mean the same thing in every template.

`plan.json` looks like:

```json
{"template": "pitch_blip_keys",
 "params": ["filter_cutoff", "amp_attack", "osc_detune_cents"],
 "ranges": {"filter_cutoff": [300, 8000]},
 "initial": {"amp_attack": 0.01},
 "fixed": {},
 "weights": {"stft": 0.5, "ltas": 3.0, "mfcc": 0.5, "env": 1.0, "attack": 2.0, "pitch": 0.0}}
```

### Match search

`app/stages/match.py` searches in phases within one time budget: CMA-ES on a short window of
whole chords (about 1.3 s, roughly twice the renders per second of a 2.7 s region) when the region
is longer, then the best window candidates are re-scored on the whole region and refined there
for the last 15% of the budget. Reported losses are always whole-region losses.

Starting values and ranges come from the analysis (`app/stages/plan_rules.py`): with a detected
pitch blip the pitch-envelope amount is searched from the measured peak (a lower bound) upwards,
fast attacks are capped at 80 ms, and confirmed beating seeds the oscillator detune. The
rule-based plan searches up to 10 parameters.

Measured on the end-to-end case (pitch-blip chords under drums + bass, separated; one fixed
target so runs are comparable; loss of the match against the clean synth, lower is better; the
true values score 1.105; 180 s budget, 4 cores). Rows marked "sweeps" used coordinate sweeps
before CMA-ES, which were later dropped (see below).

| Search | Runs | Loss vs clean | Pitch env (true +7 st) | Cutoff (true 5 kHz) |
|---|---|---|---|---|
| Before (CMA-ES on the whole region, 12 params from the template) | 3 | 1.44, 1.44, 1.29 | +2.4, -2.1, +2.6 | 5.1k, 4.6k, 9.0k |
| Window + refine, same 12 params | 1 | 1.34 | +5.9 | 1.9k |
| Window + sweeps + refine, same 12 params | 4 | 1.14, 1.39, 1.36, 1.33 | +4.1, +6.6, +5.0, +8.4 | 4.0k, 3.0k, 2.7k, 15.9k |
| + analysis priors, 12 params, sweeps | 3 | 1.12, 1.32, 1.30 | +6.3, +8.4, +4.9 | 3.3k, 3.0k, 2.4k |
| + rule plan, 8 params, sweeps | 3 | 1.38, 1.29, 1.22 | +4.2, +5.8, +14.7 | 2.4k, 4.1k, 2.4k |
| + rule plan, 10 params, sweeps | 4 | 1.10, 1.17, 1.13, 1.16 | +7.0, +6.9, +6.9, +7.3 | 3.2k, 4.1k, 4.0k, 6.1k |
| **Final**: window + refine, rule plan, 10 params, no sweeps | 2 | **1.31, 1.14** | **+6.1, +6.0** | **2.9k, 5.6k** |

The full `tests/e2e_pipeline.py` run (separation and analysis redone each time, so its target and
the true-values score change from run to run), 180 s budget:

| | Loss vs clean (true values on that run) | Pitch env | Cutoff |
|---|---|---|---|
| Before, 180 s / 300 s | 1.50 (1.09) / 1.32 (0.78) | -2.5 / +2.0 | 15.9k / 11.3k |
| Final, 180 s, two runs | 1.11 (1.19) / 1.19 (0.87) | +1.8 / +2.5 | 7.5k / 4.5k |

The loss against the clean synth now lands at or near the true values' loss, but the pitch-envelope
amount is not reliably recovered: amount and decay trade off (a smaller, longer blip scores about
the same), and on these two runs the search stayed near the measured lower bound.

"Before": the first run is the previous code; the other two are the new code with the window
switched off. Run-to-run spread is large (the search is time-budgeted), so single runs mean little.

What helped: the analysis priors and the 10-parameter rule plan (pitch envelope found every time
instead of 2 st or the wrong sign) and the window (more renders per second). Tried and dropped:
- coordinate sweeps before CMA-ES: maybe slightly better on the mixed case (within the noise) but
  on clean targets they locked onto wrong ridges of coupled parameters (pitch-envelope amount vs
  decay, cutoff vs filter-envelope amount, detune basins) and broke the recovery tests that plain
  CMA-ES passes (e.g. pluck cutoff 385 Hz instead of 700, loss 0.04 vs 0.001);
- a release-time prior: the analyser's fall time depends on sustain and clip length;
- a spectral-centroid "blip contour" loss term: under a mix it did not separate the true pitch
  envelope from wrong ones.
