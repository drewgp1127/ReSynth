# ReSynth — prototype plan

## Context

A producer hears a synth in a song and wants that sound. Today there is no way to get from "that sound in that song" to a playable patch. This tool takes a song clip plus a description, researches what was likely used, and returns a game plan and a loadable preset.

**Is it possible? Yes, as a strong starting point.** Research findings that shape the design:

- Commercial audio-to-patch tools exist (Synplant 2, Audial, MicroMusic). All need a clean isolated one-shot. None works from a full mix, and none does gear research. Combining research + mix input + patch is the gap.
- No state-of-the-art matching model has public weights. The buildable route is render-and-compare search on a synth that runs headless from Python.
- Vital is the right target: free, runs on Mac, presets are plain JSON, and `pip install vita` renders it in-process and writes `.vital` files.
- Claude has no audio input. All listening is done by signal analysis; Claude reasons over numbers and research.
- Test case reality check: no source names the Lemon Glow synth. Reverb Machine published a recreation (two detuned tri-saw oscillators, a fast pitch-envelope blip on each chord, treble boost and grit). That is our ground truth for the type, not the model.

**Decisions made with the user**

| Topic | Decision |
|---|---|
| Input | Song clip + song name + text description. User marks the cleanest 1–3 s. |
| Output | Gear report, written recipe, Vital preset, A/B audio with similarity score. |
| Approach | Research agent + listen-and-match search loop. |
| Form | Web app. |
| Success bar | Right family, 70–80% there; he finishes by ear. |
| Licensing | Personal tool. Non-commercial models allowed. |
| His gear | Logic Pro, Serum, stock synths, hardware; will install Vital. |

## Architecture

One Python service, one job at a time, six stages. Each stage reads and writes plain files in a job folder, so each can be run and tested alone.

```
upload → 1 research ─────────────┐
       → 2 isolate → 3 analyse ──┼→ 4 plan → 5 match → 6 report
```

Project root: `/home/user/patch-finder/` (new git repo).

```
patch-finder/
  app/
    main.py            FastAPI: upload, job status, results, static UI
    jobs.py            job folder + stage runner (background thread, status.json)
    stages/
      research.py      1. gear research (Claude + web search)
      isolate.py       2. stem separation + crop to marked region
      analyse.py       3. notes + timbre descriptors
      plan.py          4. template choice + search ranges (Claude)
      match.py         5. CMA-ES render-and-compare (vita)
      report.py        6. recipe text, A/B audio, final preset
    synth/
      vital.py         thin wrapper over vita: load template, set params, render notes, save
      params.py        the searchable parameter list with ranges and scaling
      loss.py          weighted audio distance
    templates/         ~12 hand-built .vital starting patches
    web/               index.html, app.js (wavesurfer.js regions), style.css
  tests/
  Dockerfile
  render.yaml
```

### Stage details

**1. Research** (`research.py`). Claude API with the web search tool. Input: song, artist, user description. Output `research.json`: instruments split into *confirmed* / *community-sourced* / *speculated*, each with source URL; the synthesis type hypothesis; any published recreation with its settings. Rule in the prompt: never present a recreation as the original gear. Use the `claude-api` skill for current model IDs and web-search tool usage. Runs in parallel with stages 2–3.

**2. Isolate** (`isolate.py`). `audio-separator` (htdemucs_6s first; try a RoFormer model if results are weak). Produces stems for the marked region only, to keep it fast on CPU. The UI plays each stem plus the raw slice and the user picks the one where the sound is clearest. Output `target.wav` (mono + stereo copies, 44.1 kHz).

**3. Analyse** (`analyse.py`). librosa + basic-pitch. Output `analysis.json`:
- notes (MIDI pitch, onset, duration) — polyphonic via basic-pitch
- amplitude envelope and rough attack / release times
- spectral centroid over time, rolloff, flatness, odd/even harmonic ratio
- pitch movement per note (catches the Lemon Glow blip), vibrato rate
- tremolo / filter-LFO rate from the centroid and RMS tracks
- beating rate (detune hint), stereo width

**4. Plan** (`plan.py`). Claude gets `research.json`, `analysis.json` and the user description. It returns: one template name, initial values, and which parameters to search with what ranges. Claude edits a template; it never writes a preset from scratch.

**5. Match** (`match.py`). `pycma` CMA-ES over 15–40 parameters from `params.py`: oscillator wave/level/detune/unison, filter cutoff/resonance/env amount, amp and filter ADSR, pitch-envelope amount/time, LFO rate/depth, drive. Each candidate is rendered with `vita` using the detected notes, then scored by `loss.py`:
- multi-resolution STFT distance (`auraloss`)
- MFCC distance
- amplitude-envelope distance
- pitch-track distance (only when stage 3 found pitch movement)

Renders run in a process pool. Time budget 3–5 minutes, stop early on plateau. Keep the best 3 candidates. Effects stay off during the search except drive; reverb/delay/chorus are recommended in the recipe, not searched.

**6. Report** (`report.py`). Outputs:
- `patch.vital` (plus 2 runner-up variations)
- `ab.wav`: target then match, loudness-matched
- similarity score with a plain label (close / same family / weak)
- gear report from stage 1 with confidence labels and links
- recipe: Claude turns the final parameter values into generic synth terms, then adds a "on your gear" section from a free-text gear field (Serum, Logic Alchemy/ES2, hardware). Text only for non-Vital gear.

### Web UI

One page. Upload clip (≤60 s), song + artist, description, gear field. Waveform with a draggable region (wavesurfer.js). Stem picker with play buttons. Progress per stage. Results: A/B player, downloads, report.

## Build order

Each milestone ends with something runnable.

0. **Spike (go/no-go).** Install `vita` on Linux. Render a note, change parameters, save a `.vital`, confirm the JSON is well-formed. Confirm how to render chords (if `render` is single-note only, render each note and sum before drive). If vita fails, fall back to Surge XT `surgepy`.
1. **Synth wrapper + loss + match**, proven on synthetic targets: render a template with hidden parameter values, run the search, check it recovers the sound.
2. **Templates**: ~12 Vital starting patches (saw lead, supersaw, square, pluck, pad, organ-ish, FM bell, bass, pitch-blip keys, noise sweep, brass, strings).
3. **Analyse + isolate** on real clips.
4. **Research + plan + report** (Claude stages).
5. **Web UI + job runner.**
6. **Deploy** to Render with a Dockerfile.

## Verification

- `pytest`: synthetic recovery — for each template, hide random parameter values, run match with a short budget, assert final loss is well below the starting loss and key parameters (cutoff, attack, detune) land near truth.
- `pytest`: each stage runs alone from fixture files and writes valid JSON.
- Stage 1 on Lemon Glow must report "model unknown", cite the Scally "ugliest keyboard sound" quote and the Reverb Machine recreation, and label each correctly.
- End-to-end on a rendered Reverb-Machine-style patch mixed under drums and bass: analysis must detect the pitch blip and two detuned saw-like oscillators.
- Browser test with Playwright: upload, mark region, pick stem, wait, download `.vital`.
- Human test: the brother loads `patch.vital` in Vital inside Logic for 5 songs and rates each (unusable / right family / close).

## ⚠️ Needs you

- **Anthropic API key** for the research, plan and report stages.
- **GitHub repo**: this session has none attached. Create an empty repo (or tell me the name) so the build can be pushed.
- **Hosting cost**: stem separation needs about 2 GB RAM, so a paid Render instance (roughly $25/month; confirm at deploy time). Milestones 0–5 run without it.
- **Song clips**: your brother supplies them. The tool does not download songs.

## ⚠️ Risks

- **Dense mixes.** Separation smears pads and layered sounds, and the search then matches the smear. Marking a clean moment is the main defence. Some songs will fail.
- **Effects-heavy sounds.** Reverb, delay and tape character are described in the recipe but not matched by the search.
- **One slice ≠ a full patch.** The preset matches the marked notes. Behaviour across the keyboard and velocity is his to finish.
- **Gear model ID is often impossible.** For most non-famous songs the honest answer is a type plus "model unknown". The report says so.
- **Non-commercial models.** Fine for now; they must be swapped before selling this.

## Out of scope for the prototype

DAW plugin, Serum/hardware preset files, text-only mode, accounts, training our own matching model.
