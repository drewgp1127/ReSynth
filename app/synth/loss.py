"""Weighted audio distance between a target clip and a candidate render.

Pure numpy/scipy (no torch):
- stft:  multi-resolution STFT on mel-binned magnitudes (spectral convergence + log L1, as in
         auraloss but mel-smoothed, which is less sensitive to exact partial phases/beating)
- ltas:  long-term average spectrum at high frequency resolution (8192 and 32768-point), in dB
         with a floor 50 dB under the peak. Gives a smooth basin for detune/unison spread,
         which the frame-wise terms see only as phase noise.
- mfcc:  MFCC distance (c1..c19)
- env:   amplitude envelope (5 ms hop RMS, dB) distance over the whole clip
- attack: the same envelope distance restricted to the 250 ms after each note onset (when onsets
         are known), so attack shape is not drowned out by the sustain
- pitch: optional pitch-track distance (YIN), for clips with pitch movement
Both signals are mono-mixed and RMS-normalised to the same loudness before comparing.
Target features are computed once; `LossFn` is picklable so it can be shipped to worker processes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dct, rfft

SR = 44100
FFT_SIZES = (2048, 1024, 512, 256)
N_MELS = {2048: 128, 1024: 96, 512: 64, 256: 32}
LTAS_SIZES = (8192, 32768)
LTAS_FLOOR = 1e-5  # -50 dB below the peak
ENV_HOP = 220  # 5 ms
ATTACK_WINDOW_S = 0.25
TARGET_RMS = 0.1
EPS = 1e-5


@dataclass
class LossWeights:
    stft: float = 0.5
    ltas: float = 3.0
    mfcc: float = 0.5
    env: float = 1.0
    attack: float = 2.0
    pitch: float = 0.0  # enable when analysis found pitch movement


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def normalise(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return x * (TARGET_RMS / rms) if rms > 1e-9 else x


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    n = 1 + (len(x) - size) // hop
    return np.lib.stride_tricks.sliding_window_view(x, size)[::hop][:n]


def stft_mag(x: np.ndarray, size: int) -> np.ndarray:
    hop = size // 4
    win = np.hanning(size).astype(np.float32)
    return np.abs(rfft(_frames(x, size, hop) * win, axis=1)).astype(np.float32)


def rms_env_db(x: np.ndarray) -> np.ndarray:
    fr = _frames(x, 512, ENV_HOP)
    rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-4))  # -80 dB floor (signals normalised to 0.1 rms)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(30.0), hz_to_mel(min(16000.0, sr / 2)), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
    return fb


def mfcc_from_mag(mag: np.ndarray, fb: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    mel = (mag.astype(np.float64) ** 2) @ fb.T
    return dct(np.log(mel + 1e-6), type=2, norm="ortho", axis=1)[:, 1:n_mfcc]  # drop c0 (loudness)


def pitch_track(x: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0) -> np.ndarray:
    import librosa  # lazy: only needed when the pitch term is on

    return librosa.yin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=441)


def ltas_db(x: np.ndarray) -> np.ndarray:
    out = []
    for n in LTAS_SIZES:
        p = np.mean(stft_mag(x, n).astype(np.float64) ** 2, axis=0)
        out.append(10 * np.log10(np.maximum(p, p.max() * LTAS_FLOOR + 1e-20)))
    return np.concatenate(out)


@dataclass
class Features:
    mels: dict[int, np.ndarray]
    ltas: np.ndarray
    mfcc: np.ndarray
    env: np.ndarray
    pitch: np.ndarray | None = None


@dataclass
class LossFn:
    target: np.ndarray  # mono, normalised
    sr: int = SR
    weights: LossWeights = field(default_factory=LossWeights)
    onsets: list[float] | None = None  # note onsets in seconds, for the attack term

    def __post_init__(self) -> None:
        self.target = normalise(to_mono(self.target))
        self.attack_mask = None
        self.fbs = {n: mel_filterbank(self.sr, n, N_MELS[n]) for n in FFT_SIZES}
        self.fb = mel_filterbank(self.sr, FFT_SIZES[0])
        self.tf = self.features(self.target, raw=True)
        if self.onsets:
            times = np.arange(len(self.tf.env)) * ENV_HOP / self.sr
            mask = np.zeros(len(times), dtype=bool)
            for o in self.onsets:
                mask |= (times >= o - 0.01) & (times < o + ATTACK_WINDOW_S)
            self.attack_mask = mask if mask.any() else None
        self.voiced = None
        if self.weights.pitch > 0:
            f0 = self.tf.pitch
            self.voiced = np.isfinite(f0) & (f0 > 0)

    def features(self, x: np.ndarray, raw: bool = False) -> Features:
        if not raw:
            x = normalise(to_mono(x))
        mags = {n: stft_mag(x, n) for n in FFT_SIZES}
        return Features(
            mels={n: np.sqrt((mags[n].astype(np.float64) ** 2) @ self.fbs[n].T) for n in FFT_SIZES},
            ltas=ltas_db(x),
            mfcc=mfcc_from_mag(mags[FFT_SIZES[0]], self.fb),
            env=rms_env_db(x),
            pitch=pitch_track(x, self.sr) if self.weights.pitch > 0 else None,
        )

    def _fit(self, audio: np.ndarray) -> np.ndarray:
        x = to_mono(audio)
        n = len(self.target)
        return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))

    def parts(self, audio: np.ndarray) -> dict[str, float]:
        x = self._fit(audio)
        if float(np.sqrt(np.mean(x ** 2))) < 1e-7:
            return {"stft": 10.0, "ltas": 10.0, "mfcc": 10.0, "env": 10.0, "attack": 10.0, "pitch": 10.0, "silent": 1.0}
        cf = self.features(x)
        tf = self.tf
        stft = 0.0
        for n in FFT_SIZES:
            t, c = tf.mels[n], cf.mels[n]
            sc = np.linalg.norm(t - c) / (np.linalg.norm(t) + EPS)
            lm = np.mean(np.abs(np.log(t + EPS) - np.log(c + EPS)))
            stft += sc + 0.25 * lm
        out = {
            "stft": float(stft / len(FFT_SIZES)),
            "ltas": float(np.mean(np.abs(tf.ltas - cf.ltas)) / 10.0),
            "mfcc": float(np.mean(np.abs(tf.mfcc - cf.mfcc)) / 10.0),
            "env": float(np.mean(np.abs(tf.env - cf.env)) / 20.0),
        }
        if self.attack_mask is not None:
            out["attack"] = float(np.mean(np.abs(tf.env - cf.env)[self.attack_mask]) / 20.0)
        if self.weights.pitch > 0 and self.voiced is not None and self.voiced.any():
            t = 12 * np.log2(tf.pitch[self.voiced])
            c = 12 * np.log2(np.maximum(cf.pitch[self.voiced], 1.0))
            out["pitch"] = float(np.mean(np.minimum(np.abs(t - c), 12.0)) / 12.0)
        return out

    def total(self, parts: dict[str, float]) -> float:
        w = self.weights
        return (w.stft * parts["stft"] + w.ltas * parts["ltas"] + w.mfcc * parts["mfcc"] + w.env * parts["env"] + w.attack * parts.get("attack", 0.0)
                + w.pitch * parts.get("pitch", 0.0))

    def __call__(self, audio: np.ndarray) -> tuple[float, dict[str, float]]:
        p = self.parts(audio)
        return self.total(p), p
