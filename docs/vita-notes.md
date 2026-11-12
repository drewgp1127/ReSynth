# vita API notes (vita 0.1.0, Python 3.13 manylinux wheel)

Facts verified by `scripts/spike_vita.py` and ad-hoc checks on Linux x86_64.

## Install
- `pip install vita` installs a 1.4 MB wheel for cp313; no system deps. Import: `import vita`.
- The package has no numpy dependency declared but returns numpy arrays; install numpy yourself.

## Core API
- `vita.Synth()` starts on the init preset (one saw oscillator, filters off, env_1 = amp env).
- `set_sample_rate(44100)`, `set_bpm(bpm)`.
- `render(midi_note: int, velocity: float 0-1, note_dur_s, render_dur_s) -> float32 ndarray (2, n)`.
  **One note per call.** There is no multi-note / MIDI-sequence render. Chords and sequences:
  render each note separately and sum at its onset. This is exact for everything per-voice
  (oscillators, filters, envelopes, voice modulation); it is NOT exact for global effects
  (distortion, chorus, reverb, compressor...), which are nonlinear or stateful. So "drive" in the
  search uses `filter_1_drive` (per-voice), and global effects stay off during the search.
- `to_json() -> str` (same format as a `.vital` file), `load_json(str) -> bool`,
  `load_preset(path) -> bool`, `load_init_preset()` (also clears modulations).
- `get_controls() -> dict[str, ControlValue]` (772 controls). `ControlValue.set(v)` takes the
  **raw stored value**, not a display value; `set_normalized(0..1)` / `get_normalized()` also exist.
- `get_control_details(name)` -> min, max, default_value, scale (ValueScale), display_units,
  options (discrete names). `get_control_text(name)` -> display string.
- `connect_modulation(source, dest) -> bool` fills the next free slot `modulation_N`; set its
  strength with `controls["modulation_N_amount"]` (-1..1). `disconnect_modulation`, `clear_modulations`.
  Sources/destinations: `vita.get_modulation_sources()` (32), `vita.get_modulation_destinations()` (428).
- Synth releases the GIL in render/load/to_json: use one Synth per thread in a ThreadPoolExecutor.
  `Synth` is also picklable (`__getstate__` = JSON).

## Value scaling (raw value -> what Vital shows)
- Envelope times (`env_N_attack/decay/release`): raw 0..2.378, **seconds = raw ** 4** (quartic).
- `filter_N_cutoff`: raw 8..136 = MIDI note number (display shows semitones offset by -60).
  Hz = 440 * 2 ** ((raw - 69) / 12).
- `osc_N_level`: quadratic, shown value = raw ** 2. `osc_N_unison_detune`: 0..10, shown % = raw ** 2.
- `osc_N_unison_voices`: 1..16 integer. `osc_N_transpose`: -48..48 semitones, `osc_N_tune`: -1..1 (x100 cents).
- `lfo_N_frequency`: raw -7..9, Hz = 2 ** raw when `lfo_N_sync` = 0 (Seconds); default sync is Tempo.
- `filter_N_drive`: 0..20 dB. `volume`: square-root scale; default 5473 = -6 dB.
- Modulation amount: fraction of the destination's full range. env -> `voice_transpose` (range 96
  semitones) with amount 12/96 = +12 semitones peak. Use unipolar (bipolar shifts the base pitch).
- `voice_transpose` is a modulation destination, so one envelope modulation moves every oscillator's pitch.
- Velocity has no effect on the init patch (`velocity_track` = 0).

## Determinism gotchas (important for search)
1. `osc_N_random_phase` defaults to 1.0: unison voices start at random phases and two renders of the
   same patch differ. Set it to 0 for search renders (we set it to 0 in templates).
2. **Control smoothing**: after `load_json`/`load_preset` or after setting controls, Vital glides
   smoothed controls (e.g. filter cutoff) from their previous/default value over ~0.2 s of processed audio.
   The first render after a change therefore has a transient. Fix: one warm-up render
   `render(60, 0.0, 0.02, 0.3)` (discard output) after any change; then renders are bit-stable
   (diff < 1e-6) and a saved/reloaded preset renders identically to the live synth.
3. Global effects (e.g. reverb) keep their state across render calls; keep them off during search.

## Wavetables
- Each oscillator's wave lives in `settings.wavetables[i].groups[0].components[0].keyframes[k].wave_data`:
  base64 of 2048 little-endian float32 samples (one cycle). The init wave is a saw.
- Writing our own single-cycle wave (e.g. a square) and calling `load_json` works: harmonics measured
  as expected (odd only for a square). This is how templates get square/triangle/sine waves without
  any factory content.
- `settings.sample` holds the sampler's audio (init = 44100 samples of "White Noise"), usable for noise.

## Speed (one core)
- load_json ~11 ms, warm-up ~7 ms; a 1.5 s render with 2 oscillators + filter ~36-43 ms
  (barely depends on unison voice count).
