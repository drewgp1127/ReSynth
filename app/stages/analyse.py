"""Stage 3: describe the target sound with numbers the plan and match stages can use.

Input:  <job>/target.wav
Output: <job>/analysis.json, <job>/status/analyse.json

Run: python -m app.stages.analyse <job_dir>

Notes come from basic-pitch (ONNX model bundled in the wheel) when it is importable, otherwise
from librosa pYIN (monophonic). Pitch movement is measured per onset group by correlating each
early frame's log-frequency spectrum against the group's settled spectrum, which works for
chords because a pitch envelope moves every note of the chord together.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from app.stages.common import SR, read_audio, write_json, write_status

HOP = 220  # 5 ms
N_FFT = 2048
LOGF_BINS_PER_ST = 10
LOGF_FMIN, LOGF_FMAX = 300.0, 8000.0


# ---- notes --------------------------------------------------------------------------
@dataclass
class DetectedNote:
    midi: int
    onset: float
    duration: float
    amplitude: float

    def as_dict(self) -> dict[str, Any]:
        return {"midi": self.midi, "onset": round(self.onset, 4), "duration": round(self.duration, 4),
                "velocity": 0.8, "amplitude": round(self.amplitude, 3)}


def _basic_pitch_notes(path: Path) -> list[DetectedNote]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model, predict

        model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
        _, _, events = predict(str(path), model, minimum_note_length=60.0)
    return [DetectedNote(int(p), float(s), float(e - s), float(a)) for s, e, p, a, *_ in events]


def _pyin_notes(y: np.ndarray, sr: int) -> list[DetectedNote]:
    f0, voiced, _ = librosa.pyin(y, fmin=45, fmax=2000, sr=sr, frame_length=2048, hop_length=HOP)
    midi = np.where(voiced, np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    notes, start = [], None
    for i in range(len(midi) + 1):
        cur = midi[i] if i < len(midi) else -1
        if start is not None and (cur != midi[start]):
            dur = (i - start) * HOP / sr
            if dur >= 0.05:
                notes.append(DetectedNote(int(midi[start]), start * HOP / sr, dur, float(rms[start:i].max())))
            start = None
        if start is None and cur >= 0:
            start = i
    peak = max((n.amplitude for n in notes), default=1.0)
    return [DetectedNote(n.midi, n.onset, n.duration, n.amplitude / peak) for n in notes]


def clean_notes(notes: list[DetectedNote], onsets: np.ndarray) -> list[DetectedNote]:
    """Merge same-pitch fragments, drop weak harmonic ghosts, snap onsets to detected onsets."""
    notes = sorted(notes, key=lambda n: (n.midi, n.onset))
    merged: list[DetectedNote] = []
    for n in notes:
        prev = merged[-1] if merged else None
        if prev and prev.midi == n.midi and n.onset - (prev.onset + prev.duration) < 0.05:
            end = max(prev.onset + prev.duration, n.onset + n.duration)
            merged[-1] = DetectedNote(prev.midi, prev.onset, end - prev.onset, max(prev.amplitude, n.amplitude))
        else:
            merged.append(n)
    kept = []
    for n in merged:
        overlapping = [m for m in merged if m is not n and m.onset < n.onset + n.duration and n.onset < m.onset + m.duration]
        ghost = any((n.midi - m.midi) in (12, 19, 24, 28, 31) and n.amplitude < 0.75 * m.amplitude for m in overlapping)
        if not ghost:
            kept.append(n)
    snapped = []
    for n in kept:
        near = onsets[(onsets >= n.onset - 0.1) & (onsets <= n.onset + 0.03)] if len(onsets) else onsets
        onset = float(near[np.argmin(np.abs(near - n.onset))]) if len(near) else n.onset
        snapped.append(DetectedNote(n.midi, onset, n.duration + (n.onset - onset), n.amplitude))
    return harmonise_chords(sorted(snapped, key=lambda n: (n.onset, n.midi)))


def harmonise_chords(notes: list[DetectedNote], tol: float = 0.08) -> list[DetectedNote]:
    """Chord notes that end within `tol` of the group's median end get exactly that end (one
    shared note-off; also lets the matcher reuse renders of repeated notes)."""
    out = []
    for g in onset_groups(notes):
        ends = np.array([n.onset + n.duration for n in g])
        med = float(np.median(ends))
        for n, e in zip(g, ends):
            end = med if abs(e - med) <= tol else e
            out.append(DetectedNote(n.midi, g[0].onset, round(end - g[0].onset, 3), n.amplitude))
    return out


def onset_groups(notes: list[DetectedNote], tol: float = 0.04) -> list[list[DetectedNote]]:
    groups: list[list[DetectedNote]] = []
    for n in sorted(notes, key=lambda n: n.onset):
        if groups and n.onset - groups[-1][0].onset <= tol:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


# ---- envelope -----------------------------------------------------------------------
def envelope_db(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP, center=True)[0]
    return 20 * np.log10(np.maximum(rms, 1e-6))


def attack_time(env_db: np.ndarray, onset: float, window: float = 0.6) -> float | None:
    """10% -> 90% rise time (linear amplitude) after an onset."""
    i0 = max(int(onset * SR / HOP) - 2, 0)
    seg = 10 ** (env_db[i0:i0 + int(window * SR / HOP)] / 20)
    if len(seg) < 3:
        return None
    peak_i = int(np.argmax(seg))
    lo, hi = 0.1 * seg[peak_i], 0.9 * seg[peak_i]
    rise = seg[:peak_i + 1]
    t10 = int(np.argmax(rise >= lo)) if (rise >= lo).any() else 0
    t90 = int(np.argmax(rise >= hi))
    return max(t90 - t10, 0) * HOP / SR


def release_time(env_db: np.ndarray, drop_db: float = 40.0) -> float | None:
    """Estimated time for the final note(s) to fall `drop_db` once they start releasing.

    The release starts at the last frame within 9 dB of the final segment's peak. If the clip
    ends before the level has fallen `drop_db`, the observed dB-per-second slope is extrapolated
    (needs at least 50 ms of tail). None when there is no usable tail."""
    floor = env_db.max() - 70
    active = np.nonzero(env_db > floor)[0]
    if len(active) < 3:
        return None
    end = active[-1]
    tail_peak_i = max(int(end - 1.0 * SR / HOP), 0) + int(np.argmax(env_db[max(int(end - 1.0 * SR / HOP), 0):end + 1]))
    peak = env_db[tail_peak_i]
    near = np.nonzero(env_db[tail_peak_i:] >= peak - 9)[0]
    start = tail_peak_i + int(near[-1])
    ref = env_db[start]
    seg = env_db[start:]
    below = np.nonzero(seg < ref - drop_db)[0]
    if len(below):
        return float(below[0] * HOP / SR)
    if len(seg) < int(0.05 * SR / HOP):
        return None
    t = np.arange(len(seg)) * HOP / SR
    slope = float(np.polyfit(t, seg, 1)[0])  # dB per second (negative when falling)
    return float(drop_db / -slope) if slope < -1.0 else None


# ---- periodic modulation ------------------------------------------------------------
def modulation_rate(track: np.ndarray, frame_rate: float, fmin: float, fmax: float,
                    min_strength: float = 0.4) -> tuple[float | None, float, float]:
    """Dominant periodicity of a detrended track via autocorrelation.

    Returns (rate_hz or None, strength 0..1, peak-to-peak depth of the detrended track)."""
    x = np.asarray(track, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < frame_rate / fmax * 4:
        return None, 0.0, 0.0
    fmin = max(fmin, 1.5 * frame_rate / len(x))  # need ~1.5 periods of the slowest rate
    trend_len = max(int(frame_rate / fmin), 3)
    trend = np.convolve(np.pad(x, trend_len, mode="edge"), np.ones(trend_len) / trend_len, "same")[trend_len:-trend_len]
    d = x - trend
    d = d - d.mean()
    if np.std(d) < 1e-9:
        return None, 0.0, 0.0
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac /= ac[0]
    lo, hi = max(int(frame_rate / fmax), 1), min(int(frame_rate / fmin), len(ac) - 2)
    if hi <= lo + 1:
        return None, 0.0, 0.0
    seg = ac[lo:hi + 1]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0, 0.0
    i = max(peaks, key=lambda k: seg[k])
    strength = float(seg[i])
    a, b, c = seg[i - 1], seg[i], seg[i + 1]
    off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    lag = lo + i + off
    depth = float(np.percentile(d, 95) - np.percentile(d, 5))
    return (frame_rate / lag if strength >= min_strength else None), strength, depth


# ---- pitch movement -----------------------------------------------------------------
def log_freq_spec(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-magnitude spectrogram resampled onto a log-frequency axis (10 bins per semitone)."""
    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    n_st = 12 * np.log2(LOGF_FMAX / LOGF_FMIN)
    grid = LOGF_FMIN * 2 ** (np.arange(int(n_st * LOGF_BINS_PER_ST)) / (12 * LOGF_BINS_PER_ST))
    logmag = np.log(mag + 1e-6)
    out = np.empty((len(grid), mag.shape[1]), dtype=np.float32)
    for t in range(mag.shape[1]):
        out[:, t] = np.interp(grid, freqs, logmag[:, t])
    return out, grid


def _best_shift(frame: np.ndarray, ref: np.ndarray, max_shift: int) -> tuple[float, float]:
    """Shift (in log-freq bins) of `frame` relative to `ref` maximising correlation."""
    f = frame - frame.mean()
    r = ref - ref.mean()
    scores = []
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a, b = f[s:], r[:len(r) - s]
        else:
            a, b = f[:s], r[-s:]
        scores.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    scores = np.array(scores)
    i = int(np.argmax(scores))
    off = 0.0
    if 0 < i < len(scores) - 1:
        a, b, c = scores[i - 1], scores[i], scores[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
    return (i - max_shift + off), float(scores[i])


def pitch_offset_track(spec: np.ndarray, start: float, end: float, max_semitones: float = 12.0) -> dict[str, Any] | None:
    """Pitch offset (semitones) over time for one onset group, relative to its settled pitch."""
    i0, i1 = int(start * SR / HOP), min(int(end * SR / HOP), spec.shape[1])
    if i1 - i0 < 12:
        return None
    settle = slice(i0 + min(int(0.15 * SR / HOP), (i1 - i0) // 2), i1)
    ref = np.median(spec[:, settle], axis=1)
    max_shift = int(max_semitones * LOGF_BINS_PER_ST)
    offs, conf = [], []
    for t in range(i0, i1):
        s, c = _best_shift(spec[:, t], ref, max_shift)
        offs.append(s / LOGF_BINS_PER_ST)
        conf.append(c)
    return {"offsets": np.array(offs), "confidence": np.array(conf)}


def describe_pitch_movement(track: dict[str, Any] | None) -> dict[str, Any]:
    if track is None:
        return {"blip": False}
    offs, conf = track["offsets"], track["confidence"]
    early = slice(0, min(int(0.15 * SR / HOP), len(offs)))
    e_off, e_conf = offs[early], conf[early]
    good = e_conf > 0.5
    if not good.any():
        return {"blip": False}
    k = int(np.argmax(np.abs(np.where(good, e_off, 0))))
    peak = float(e_off[k])
    after = np.abs(offs[k:]) < 0.5
    settle_t = float(np.argmax(after) * HOP / SR) if after.any() else None
    blip = abs(peak) >= 1.0 and settle_t is not None and settle_t <= 0.4
    return {"blip": bool(blip), "peak_semitones": round(peak, 2), "peak_time_s": round(k * HOP / SR, 4),
            "settle_time_s": round(settle_t, 4) if settle_t is not None else None,
            "contour_semitones": [round(float(x), 2) for x in offs[:int(0.2 * SR / HOP)]]}


# ---- timbre -------------------------------------------------------------------------
def odd_even_ratio(y: np.ndarray, notes: list[DetectedNote]) -> float | None:
    """Power ratio of odd (3,5,7,9) to even (2,4,6,8) harmonics, using partials that don't
    collide with other simultaneous notes' harmonics. Median over notes."""
    ratios = []
    for n in notes:
        a, b = int((n.onset + 0.05) * SR), int((n.onset + n.duration) * SR)
        if b - a < 4096:
            continue
        seg = y[a:b] * np.hanning(b - a)
        mag = np.abs(np.fft.rfft(seg, n=max(16384, 1 << int(np.ceil(np.log2(b - a))))))
        freqs = np.fft.rfftfreq(len(mag) * 2 - 2, 1 / SR)
        f0 = librosa.midi_to_hz(n.midi)
        others = [librosa.midi_to_hz(m.midi) for m in notes
                  if m is not n and m.onset < n.onset + n.duration and n.onset < m.onset + m.duration]
        odd = even = 0.0
        for k in range(2, 10):
            fk = k * f0
            if fk > SR / 2 - 1000:
                break
            if any(abs(1200 * np.log2(fk / (j * fo))) < 60 for fo in others for j in range(1, 20)):
                continue
            band = (freqs > fk * 2 ** (-0.4 / 12)) & (freqs < fk * 2 ** (0.4 / 12))
            p = float(mag[band].max() ** 2) if band.any() else 0.0
            if k % 2:
                odd += p
            else:
                even += p
        if odd > 0 and even > 0:
            ratios.append(odd / even)
    return float(np.median(ratios)) if ratios else None


def beating(y: np.ndarray, notes: list[DetectedNote]) -> dict[str, Any]:
    """Beat rate of each note's low harmonics (amplitude modulation of their STFT bins).

    Two oscillators detuned by d cents beat at k * f0 * (2 ** (d / 1200) - 1) on harmonic k, so
    each beat rate gives a detune estimate. Harmonics close to another simultaneous note's
    harmonics are skipped (those beat for other reasons). When two or more harmonics show a rate,
    true beating has rates proportional to k (equal detune estimates); equal rates mean a common
    amplitude/filter modulation instead, and the note is skipped. `confirmed` is True when at
    least one note showed the harmonic scaling."""
    stft = np.abs(librosa.stft(y, n_fft=8192, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=8192)
    rates, cents, confirmed = [], [], 0
    for n in notes:
        i0, i1 = int((n.onset + 0.1) * SR / HOP), int((n.onset + n.duration - 0.15) * SR / HOP)
        if i1 - i0 < int(0.5 * SR / HOP):
            continue
        f0 = librosa.midi_to_hz(n.midi)
        others = [librosa.midi_to_hz(m.midi) for m in notes
                  if m is not n and m.onset < n.onset + n.duration and n.onset < m.onset + m.duration]
        found = []
        for k in (1, 2, 3):
            fk = k * f0
            if any(abs(1200 * np.log2(fk / (j * fo))) < 100 for fo in others for j in range(1, 12)):
                continue
            b = int(np.argmin(np.abs(freqs - fk)))
            track = 20 * np.log10(stft[b - 1:b + 2, i0:i1].sum(axis=0) + 1e-9)
            rate, strength, _ = modulation_rate(track, SR / HOP, 0.3, 20.0, min_strength=0.25)
            if rate:
                found.append((rate / k, strength, 1200 * np.log2(1 + rate / fk)))
        if len(found) >= 2:
            det = [f[2] for f in found]
            if max(det) / min(det) > 1.35:
                continue  # rates don't scale with k: not oscillator beating
            confirmed += 1
        if found:
            best = max(found, key=lambda f: f[1])
            rates.append(best[0])
            cents.append(best[2])
    if not rates:
        return {"rate_hz": None, "detune_cents_estimate": None, "confirmed": False}
    return {"rate_hz": round(float(np.median(rates)), 3), "detune_cents_estimate": round(float(np.median(cents)), 2),
            "confirmed": confirmed > 0}


def stereo_width(audio: np.ndarray) -> dict[str, float]:
    if audio.shape[0] < 2:
        return {"width": 0.0, "correlation": 1.0}
    left, right = audio[0].astype(float), audio[1].astype(float)
    mid, side = (left + right) / 2, (left - right) / 2
    width = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12))
    corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 1.0
    return {"width": round(width, 4), "correlation": round(corr, 4)}


def _summary(x: np.ndarray, step: int) -> list[float]:
    return [round(float(v), 1) for v in x[::step]]


# ---- stage --------------------------------------------------------------------------
def analyse_audio(audio: np.ndarray, wav_path: Path | None = None, note_method: str = "auto") -> dict[str, Any]:
    y = audio.mean(axis=0)
    duration = len(y) / SR
    onsets = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP, units="time", backtrack=True)

    method = note_method
    raw: list[DetectedNote] = []
    if note_method in ("auto", "basic-pitch") and wav_path is not None:
        try:
            raw = _basic_pitch_notes(wav_path)
            method = "basic-pitch"
        except ImportError:
            if note_method == "basic-pitch":
                raise
    if not raw and method != "basic-pitch":
        raw, method = _pyin_notes(y, SR), "pyin"
    notes = clean_notes(raw, onsets)

    env = envelope_db(y)
    groups = onset_groups(notes)
    attacks = [a for a in (attack_time(env, g[0].onset) for g in groups) if a is not None]
    release = release_time(env)

    spec, _ = log_freq_spec(y)
    movements, vib_tracks = [], []
    for i, g in enumerate(groups):
        start = g[0].onset
        end = min(groups[i + 1][0].onset if i + 1 < len(groups) else duration, max(n.onset + n.duration for n in g))
        track = pitch_offset_track(spec, start, end)
        movements.append({"onset": round(start, 4), "notes": [n.midi for n in g], **describe_pitch_movement(track)})
        if track is not None:
            sustain = track["offsets"][int(0.15 * SR / HOP):]
            if len(sustain) > int(0.5 * SR / HOP):
                vib_tracks.append(sustain * 100)  # cents
    blips = [m for m in movements if m["blip"]]

    vib_rate, vib_strength, vib_depth = (None, 0.0, 0.0)
    if vib_tracks:
        longest = max(vib_tracks, key=len)
        vib_rate, vib_strength, vib_depth = modulation_rate(longest, SR / HOP, 2.0, 12.0)
        if vib_depth < 4.0:  # cents peak-to-peak; below this it is filter/beating leakage
            vib_rate = None

    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)[0]
    loud = env > env.max() - 30
    sustain_mask = loud.copy()
    for g in groups:  # exclude attacks from LFO estimates
        i = int(g[0].onset * SR / HOP)
        sustain_mask[i:i + int(0.15 * SR / HOP)] = False
    trem_rate, trem_strength, trem_depth = modulation_rate(env[sustain_mask], SR / HOP, 0.5, 15.0)
    lfo_rate, lfo_strength, lfo_depth = modulation_rate(12 * np.log2(np.maximum(centroid[sustain_mask], 1.0)),
                                                        SR / HOP, 0.3, 15.0)

    beat = beating(y, notes)
    if lfo_rate and beat["confirmed"] and abs(lfo_rate / beat["rate_hz"] - 1) < 0.2:
        lfo_rate = None  # centroid wobble explained by oscillator beating, not a filter LFO
    blip_peak = float(np.median([m["peak_semitones"] for m in blips])) if blips else 0.0
    blip_settle = float(np.median([m["settle_time_s"] for m in blips])) if blips else None
    return {
        "sr": SR,
        "duration": round(duration, 4),
        "channels": int(audio.shape[0]),
        "note_method": method,
        "notes": [n.as_dict() for n in notes],
        "onsets_s": [round(float(o), 4) for o in onsets],
        "envelope": {
            "hop_s": HOP / SR, "rms_db_10ms": _summary(env, 2),
            "attack_s": round(float(np.median(attacks)), 4) if attacks else None,
            "release_s": round(release, 4) if release is not None else None,
        },
        "spectral": {
            "hop_s": HOP / SR,
            "centroid_hz_10ms": _summary(centroid, 2),
            "centroid_hz_mean": round(float(centroid[loud].mean()), 1),
            "rolloff_hz_mean": round(float(rolloff[loud].mean()), 1),
            "flatness_mean": round(float(flatness[loud].mean()), 4),
            "odd_even_ratio": (lambda r: round(r, 3) if r is not None else None)(odd_even_ratio(y, notes)),
        },
        "pitch_movement": {
            "detected": bool(blips) and len(blips) >= max(1, len(groups) // 2),
            "blip_count": len(blips), "group_count": len(groups),
            "peak_semitones": round(blip_peak, 2), "settle_time_s": blip_settle,
            "per_group": movements,
        },
        "vibrato": {"rate_hz": round(vib_rate, 3) if vib_rate else None, "strength": round(vib_strength, 3),
                    "depth_cents_pp": round(vib_depth, 1)},
        "tremolo": {"rate_hz": round(trem_rate, 3) if trem_rate else None, "strength": round(trem_strength, 3),
                    "depth_db_pp": round(trem_depth, 2)},
        "filter_lfo": {"rate_hz": round(lfo_rate, 3) if lfo_rate else None, "strength": round(lfo_strength, 3),
                       "depth_semitones_pp": round(lfo_depth, 2)},
        "beating": beat,
        "stereo": stereo_width(audio),
    }


def run_analyse(job_dir: str | Path, note_method: str = "auto") -> dict[str, Any]:
    job = Path(job_dir)
    write_status(job, "analyse", "running", 0.1, "reading target.wav")
    try:
        audio = read_audio(job / "target.wav", SR)
        write_status(job, "analyse", "running", 0.3, "detecting notes and measuring")
        result = analyse_audio(audio, job / "target.wav", note_method)
        write_json(job / "analysis.json", result)
        write_status(job, "analyse", "done", 1.0,
                     f"{len(result['notes'])} notes via {result['note_method']}; "
                     f"pitch blip {'detected' if result['pitch_movement']['detected'] else 'not detected'}")
        return result
    except Exception as e:
        write_status(job, "analyse", "failed", 0.0, f"{type(e).__name__}: {e}")
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch Finder stage 3: analyse target.wav")
    ap.add_argument("job_dir")
    ap.add_argument("--notes", default="auto", choices=["auto", "basic-pitch", "pyin"])
    a = ap.parse_args()
    r = run_analyse(a.job_dir, a.notes)
    print(f"{len(r['notes'])} notes ({r['note_method']}), blip={r['pitch_movement']['detected']}, "
          f"vibrato={r['vibrato']['rate_hz']}, attack={r['envelope']['attack_s']}, release={r['envelope']['release_s']}")


if __name__ == "__main__":
    main()
