"""Stage 5: CMA-ES render-and-compare search.

Inputs in the job folder:
  target.wav     the sound to match (mono or stereo, any rate)
  analysis.json  needs "notes": [{midi, onset, duration, velocity?}, ...] and "duration"
  plan.json      optional: {"template": name or path, "params": [names],
                  "ranges": {name: [lo, hi]}, "initial": {name: value}, "fixed": {name: value},
                  "weights": {stft, mfcc, env, pitch}}
Outputs:
  match.json             best 3 candidates (values in natural units + loss parts), initial loss, stats
  match_1.vital ..3      presets for the best 3 candidates
  match_1.wav            render of the best candidate
  status/match.json      progress

Run: python -m app.stages.match <job_dir> [--budget SECONDS] [--workers N]
"""
from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cma
import numpy as np

from app.stages.common import SR, read_audio, read_json, write_audio, write_json, write_status
from app.synth.loss import LossFn, LossWeights
from app.synth.params import PARAMS, SearchSpace
from app.synth.vital import Note, VitalRenderer, load_template, save_vital

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
DEFAULT_PARAMS = [
    "osc1_wave", "osc1_unison_detune", "osc_detune_cents", "osc2_level",
    "filter_cutoff", "filter_resonance", "filter_env_amount", "filter_drive",
    "amp_attack", "amp_decay", "amp_sustain", "amp_release", "filt_decay",
]


@dataclass
class MatchConfig:
    template: str
    params: list[str] = field(default_factory=lambda: list(DEFAULT_PARAMS))
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    initial: dict[str, float] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)
    weights: LossWeights = field(default_factory=LossWeights)
    budget_s: float = 180.0
    max_evals: int | None = None  # also stop after this many renders (fixed work, for tests); refine gets its share
    plateau_gens: int = 30
    plateau_tol: float = 1e-3  # relative improvement that counts as progress
    restarts: int = 2  # on plateau, restart around the best point with 2x population (IPOP), this many times
    workers: int = max(1, (os.cpu_count() or 2) - 1)  # vita uses ~1.3 cores per render
    popsize: int | None = 12
    sigma0: float = 0.25
    seed: int = 1
    keep: int = 3
    keep_search: int = 8  # distinct candidates kept per phase (re-scored on the whole clip)
    window_s: float | None = 1.3  # search on a span of whole chords this long; None = whole clip
    verify_top: int = 8
    refine_frac: float = 0.15  # share of the budget for refining on the whole clip
    sigma_refine: float = 0.08

    @classmethod
    def from_plan(cls, plan: dict[str, Any], **overrides: Any) -> "MatchConfig":
        kw: dict[str, Any] = {"template": plan["template"]}
        for k in ("params", "initial", "fixed"):
            if k in plan:
                kw[k] = plan[k]
        if "ranges" in plan:
            kw["ranges"] = {k: tuple(v) for k, v in plan["ranges"].items()}
        if "weights" in plan:
            kw["weights"] = LossWeights(**plan["weights"])
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)


def template_path(name: str) -> Path:
    p = Path(name)
    if p.suffix == ".vital" and p.exists():
        return p
    return TEMPLATES_DIR / f"{name}.vital"


def template_values(patch: dict[str, Any], names: list[str]) -> dict[str, float]:
    """Natural-unit values of params as currently set in the template."""
    from app.synth.vital import modulation_slots

    st, slots, out = patch["settings"], modulation_slots(patch), {}
    for n in names:
        ctl = PARAMS[n].targets[0].control
        key = f"modulation_{slots[ctl]}_amount" if "->" in ctl else ctl
        out[n] = PARAMS[n].clip(PARAMS[n].from_raw(float(st[key])))
    return out


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


@contextlib.contextmanager
def single_threaded_children():
    """Workers are one process per core; stop BLAS/OpenMP in each from spawning more threads."""
    old = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: "1" for k in _THREAD_VARS})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---- worker process state -------------------------------------------------------------
_W: dict[str, Any] = {}


@dataclass
class EvalContext:
    """What one evaluation renders and scores against: a note list over `duration` seconds."""
    notes: list[Note]
    duration: float
    loss: LossFn


def _init_worker(patch: dict[str, Any], space: SearchSpace, contexts: dict[str, EvalContext]) -> None:
    _W.update(renderer=VitalRenderer(patch), space=space, contexts=contexts)


def _evaluate(job: tuple[str, list[float]]) -> tuple[float, dict[str, float]]:
    ctx_name, u = job
    ctx: EvalContext = _W["contexts"][ctx_name]
    values = _W["space"].values_from_unit(u)
    r: VitalRenderer = _W["renderer"]
    r.set_params(_W["space"].raw(values))
    return ctx.loss(r.render_notes(ctx.notes, ctx.duration))


# ---- search window --------------------------------------------------------------------
def note_groups(notes: list[Note], tol: float = 0.04) -> list[list[Note]]:
    groups: list[list[Note]] = []
    for n in sorted(notes, key=lambda n: n.onset):
        if groups and n.onset - groups[-1][0].onset <= tol:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def select_window(notes: list[Note], duration: float, max_s: float) -> tuple[float, float] | None:
    """A short span of whole onset groups to search on (renders cost time per note-second).

    Picks the earliest run of consecutive onset groups that fits in `max_s` and has the most
    notes, preferring runs that start with a group as large as the clip's typical chord (a chord
    whose notes were all detected). The span runs from just before its first onset to the next
    group's onset (so the note-off and some release are inside) or the clip end. None when the
    whole clip is already short enough."""
    if duration <= max_s * 1.15:
        return None
    groups = note_groups(notes)
    typical = int(np.median([len(g) for g in groups]))
    best: tuple[tuple[int, int], tuple[float, float]] | None = None
    for i in range(len(groups)):
        start = max(groups[i][0].onset - 0.03, 0.0)
        count, end = 0, None
        for j in range(i, len(groups)):
            nxt = groups[j + 1][0].onset if j + 1 < len(groups) else duration
            if nxt - start > max_s and count:
                break
            count += len(groups[j])
            end = min(nxt, start + max_s) if j == i else nxt
        score = (int(len(groups[i]) >= typical), count)
        if end is not None and (best is None or score > best[0]):
            best = (score, (start, end))
    return best[1] if best else None


def crop_context(target: np.ndarray, notes: list[Note], span: tuple[float, float], weights: LossWeights) -> EvalContext:
    a, b = span
    seg = target[:, int(round(a * SR)):int(round(b * SR))]
    inside = [Note(n.midi, n.onset - a, n.duration, n.velocity) for n in notes if a - 1e-6 <= n.onset < b]
    return EvalContext(inside, seg.shape[1] / SR, LossFn(seg, SR, weights, onsets=sorted({n.onset for n in inside})))


# ---- search --------------------------------------------------------------------------
@dataclass
class Candidate:
    loss: float
    parts: dict[str, float]
    unit: list[float]
    values: dict[str, float]


def _keep_best(best: list[Candidate], cand: Candidate, keep: int, min_dist: float = 0.05) -> list[Candidate]:
    """Best `keep` candidates that differ from each other by at least min_dist (unit-cube RMS)."""
    u = np.asarray(cand.unit)
    for i, b in enumerate(best):
        if np.sqrt(np.mean((np.asarray(b.unit) - u) ** 2)) < min_dist:
            if cand.loss < b.loss:
                best[i] = cand
            return sorted(best, key=lambda c: c.loss)
    return sorted(best + [cand], key=lambda c: c.loss)[:keep]


class _Search:
    """State shared by the search phases: the pool, the evaluation count, the best candidates."""

    def __init__(self, pool: ProcessPoolExecutor, space: SearchSpace, cfg: MatchConfig, t0: float, progress) -> None:
        self.pool, self.space, self.cfg, self.t0, self.progress = pool, space, cfg, t0, progress
        self.evals, self.gen = 0, 0
        self.history: list[dict[str, Any]] = []
        self.pools: dict[str, list[Candidate]] = {}

    def elapsed(self) -> float:
        return time.time() - self.t0

    def run(self, ctx: str, xs: list[list[float]], keep: int) -> list[tuple[float, dict[str, float]]]:
        results = list(self.pool.map(_evaluate, [(ctx, [float(v) for v in x]) for x in xs]))
        self.evals += len(xs)
        best = self.pools.setdefault(ctx, [])
        for x, (l, parts) in zip(xs, results):
            best = _keep_best(best, Candidate(l, parts, [float(v) for v in x], self.space.values_from_unit(x)), keep)
        self.pools[ctx] = best
        return results

    def log(self, phase: str, ctx: str, restart: int = 0) -> None:
        self.gen += 1
        best = self.pools[ctx][0].loss
        self.history.append({"gen": self.gen, "evals": self.evals, "best": best, "phase": phase, "restart": restart,
                             "elapsed": round(self.elapsed(), 2)})
        if self.progress:
            self.progress(min(self.elapsed() / self.cfg.budget_s, 1.0), best, self.gen, self.evals)

    def cma(self, ctx: str, x0: list[float], sigma: float, until_s: float, phase: str,
            until_evals: float | None = None) -> str:
        cfg, restart = self.cfg, 0
        ref = self.pools[ctx][0].loss if self.pools.get(ctx) else float("inf")
        start = list(x0)

        def new_es() -> cma.CMAEvolutionStrategy:
            opts = {"bounds": [0.0, 1.0], "seed": cfg.seed + restart, "verbose": -9}
            if cfg.popsize:
                opts["popsize"] = cfg.popsize * 2 ** restart
            return cma.CMAEvolutionStrategy([min(max(u, 0.03), 0.97) for u in start], sigma, opts)

        es, last_improve = new_es(), self.gen
        while True:
            xs = es.ask()
            es.tell(xs, [r[0] for r in self.run(ctx, [list(x) for x in xs], cfg.keep_search)])
            if self.pools[ctx][0].loss < ref * (1 - cfg.plateau_tol):
                ref, last_improve = self.pools[ctx][0].loss, self.gen + 1
            self.log(phase, ctx, restart)
            if self.elapsed() >= until_s or (until_evals is not None and self.evals >= until_evals):
                return "budget"
            stalled = self.gen - last_improve >= cfg.plateau_gens
            if stalled or es.stop():
                if restart >= cfg.restarts:
                    return "plateau" if stalled else "cma:" + ",".join(es.stop().keys())
                restart += 1
                start = list(self.pools[ctx][0].unit)  # restart around the best so far, wider population
                es, last_improve = new_es(), self.gen


def search(patch: dict[str, Any], target: np.ndarray, notes: list[Note], duration: float, cfg: MatchConfig,
           progress=None) -> dict[str, Any]:
    """Find parameter values whose render of `notes` sounds like `target`.

    Phases: (1) CMA-ES on a short window of whole chords when the clip is long (`window_s`),
    (2) the best window candidates re-scored on the whole clip, (3) a short CMA-ES refine on the
    whole clip.
    Returned losses are always whole-clip losses."""
    space = SearchSpace.build(cfg.params, cfg.ranges, cfg.fixed)
    start_vals = {**template_values(patch, cfg.params), **{k: v for k, v in cfg.initial.items() if k in cfg.params}}
    # CMA-ES with box bounds tends to stick to a bound it starts on, so start slightly inside
    x0 = [min(max(u, 0.05), 0.95) for u in space.unit_from_values(start_vals)]
    contexts = {"full": EvalContext(notes, duration, LossFn(target, SR, cfg.weights,
                                                            onsets=sorted({n.onset for n in notes})))}
    span = select_window(notes, duration, cfg.window_s) if cfg.window_s else None
    if span:
        contexts["win"] = crop_context(target, notes, span, cfg.weights)
    work = "win" if span else "full"

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with single_threaded_children():
        pool = ProcessPoolExecutor(cfg.workers, mp_context=ctx, initializer=_init_worker,
                                   initargs=(patch, space, contexts))
        pool.submit(int, 0).result()  # start one worker while the env vars are set
        for _ in range(cfg.workers - 1):
            pool.submit(int, 0)
    with pool:
        s = _Search(pool, space, cfg, t0, progress)
        init_loss, init_parts = s.run("full", [x0], cfg.keep)[0]
        t_search = time.time()
        refine_s = cfg.budget_s * cfg.refine_frac if span else 0.0
        refine_n = cfg.max_evals * cfg.refine_frac if (cfg.max_evals and span) else 0.0
        stop_reason = s.cma(work, x0, cfg.sigma0, cfg.budget_s - refine_s, "cma",
                            cfg.max_evals - refine_n if cfg.max_evals else None)
        if span:
            # re-score the best distinct window candidates on the whole clip, then refine there
            s.run("full", [c.unit for c in s.pools["win"][:cfg.verify_top]], cfg.keep)
            s.log("verify", "full")
            if refine_s > 0 and s.elapsed() < cfg.budget_s and (not cfg.max_evals or s.evals < cfg.max_evals):
                s.cma("full", s.pools["full"][0].unit, cfg.sigma_refine, cfg.budget_s, "refine", cfg.max_evals)
        best = s.pools["full"][:cfg.keep]
        search_time = time.time() - t_search
    return {
        "template": patch.get("preset_name", ""),
        "params": cfg.params,
        "initial_loss": init_loss,
        "initial_parts": init_parts,
        "initial_values": start_vals,
        "best": [asdict(c) for c in best],
        "window_s": [round(v, 4) for v in span] if span else None,
        "window_notes": len(contexts[work].notes),
        "evals": s.evals,
        "generations": s.gen,
        "stop_reason": stop_reason,
        "elapsed_s": round(time.time() - t0, 2),
        "evals_per_s": round((s.evals - 1) / max(search_time, 1e-9), 2),
        "history": s.history,
        "fixed": cfg.fixed,
        "weights": asdict(cfg.weights),
    }


def load_notes(analysis: dict[str, Any]) -> tuple[list[Note], float]:
    notes = [Note.from_dict(n) for n in analysis["notes"]]
    if not notes:
        raise ValueError("analysis.json has no notes; match needs at least one")
    duration = float(analysis.get("duration") or max(n.onset + n.duration for n in notes) + 0.5)
    return notes, duration


def run_match(job_dir: str | Path, config: MatchConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """Stage entry point. Uses `config` if given, else job_dir/plan.json."""
    job = Path(job_dir)
    write_status(job, "match", "running", 0.0, "loading")
    try:
        if config is None:
            plan_path = job / "plan.json"
            if not plan_path.exists():
                raise FileNotFoundError("match needs plan.json or an explicit MatchConfig")
            config = MatchConfig.from_plan(read_json(plan_path), **overrides)
        patch = load_template(template_path(config.template))
        notes, duration = load_notes(read_json(job / "analysis.json"))
        target = read_audio(job / "target.wav", SR)
        target = target[:, : int(duration * SR)]
        duration = target.shape[1] / SR

        def progress(frac: float, best: float, gen: int, evals: int) -> None:
            write_status(job, "match", "running", frac, f"generation {gen}, {evals} renders, best loss {best:.4f}",
                         best_loss=best, evals=evals)

        result = search(patch, target, notes, duration, config, progress)
        space = SearchSpace.build(config.params, config.ranges, config.fixed)
        for i, c in enumerate(result["best"], 1):
            save_vital(patch, job / f"match_{i}.vital", space.raw(c["values"]), name=f"ReSynth match {i}")
        best_raw = space.raw(result["best"][0]["values"])
        r = VitalRenderer(patch)
        r.set_params(best_raw)
        write_audio(job / "match_1.wav", r.render_notes(notes, duration))
        result["files"] = [f"match_{i}.vital" for i in range(1, len(result["best"]) + 1)] + ["match_1.wav"]
        write_json(job / "match.json", result)
        write_status(job, "match", "done", 1.0,
                     f"best loss {result['best'][0]['loss']:.4f} (start {result['initial_loss']:.4f}), "
                     f"{result['evals']} renders, stopped: {result['stop_reason']}",
                     best_loss=result["best"][0]["loss"], initial_loss=result["initial_loss"])
        return result
    except Exception as e:
        write_status(job, "match", "failed", 0.0, f"{type(e).__name__}: {e}")
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="ReSynth stage 5: CMA-ES match")
    ap.add_argument("job_dir")
    ap.add_argument("--budget", type=float, default=None, help="time budget in seconds")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    res = run_match(a.job_dir, budget_s=a.budget, workers=a.workers)
    print(f"initial loss {res['initial_loss']:.4f} -> best {res['best'][0]['loss']:.4f} "
          f"({res['evals']} renders, {res['evals_per_s']}/s, {res['stop_reason']})")


if __name__ == "__main__":
    main()
